"""CPU OCR with spatial line reconstruction.

Tesseract's plain `image_to_string` returns text in its own reading order,
which on a two-column invoice header interleaves the columns — the label
"Subtotal" ends up in a different line from its amount, and any downstream
extraction keyed on adjacency breaks.

So we take word-level boxes instead and rebuild lines by vertical overlap.
That restores "label ... value" adjacency, which is what both the rule-based
fast path and the language model need in order to associate the two.

Measured at roughly 0.8-1.0s per 300 DPI page on four cores, so OCR is not
the bottleneck in a CPU pipeline — generation is.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pytesseract
from PIL import Image

log = logging.getLogger(__name__)

# Words below this Tesseract confidence are usually noise from scan edges.
MIN_WORD_CONFIDENCE = 30


@dataclass
class Word:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y(self) -> int:
        return min(w.y for w in self.words)


def _words_from_image(image: Image.Image, lang: str) -> list[Word]:
    data = pytesseract.image_to_data(
        image, lang=lang, output_type=pytesseract.Output.DICT
    )
    words: list[Word] = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if confidence < MIN_WORD_CONFIDENCE:
            continue
        words.append(
            Word(
                text=text.strip(),
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=confidence,
            )
        )
    return words


def _group_into_lines(words: list[Word]) -> list[Line]:
    """Group words into visual lines by vertical overlap.

    Tolerance is a fraction of the word's own height rather than a fixed pixel
    count, so it holds across DPIs and across the large font-size range an
    invoice uses between its total and its footer.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.y, w.x))
    buckets: list[list[Word]] = []
    centres: list[float] = []

    for word in ordered:
        centre = word.y + word.height / 2
        for idx, existing in enumerate(centres):
            if abs(existing - centre) < word.height * 0.6:
                buckets[idx].append(word)
                # Running mean keeps the bucket centred as it grows.
                centres[idx] = (existing * (len(buckets[idx]) - 1) + centre) / len(
                    buckets[idx]
                )
                break
        else:
            buckets.append([word])
            centres.append(centre)

    lines = [Line(words=sorted(b, key=lambda w: w.x)) for b in buckets]
    lines.sort(key=lambda line: line.y)
    return lines


def read_page(png_bytes: bytes, lang: str = "eng") -> list[Line]:
    image = Image.open(io.BytesIO(png_bytes))
    return _group_into_lines(_words_from_image(image, lang))


def page_to_text(lines: list[Line], column_gap_px: int = 60) -> str:
    """Render reconstructed lines as text for a language model.

    Wide horizontal gaps become tab-like separators so the model can still see
    that a page has columns — collapsing everything to single spaces would
    make "Subtotal 1,000.00 VAT 200.00" ambiguous about what pairs with what.
    """
    out: list[str] = []
    for line in lines:
        parts: list[str] = []
        previous: Word | None = None
        for word in line.words:
            if previous is not None and word.x - previous.right > column_gap_px:
                parts.append("   |   ")
            parts.append(word.text)
            previous = word
        out.append(" ".join(parts).replace(" |  ", " | "))
    return "\n".join(out)


def document_to_text(pages: list[bytes], lang: str = "eng") -> str:
    """OCR every page of a document into one model-ready text block."""
    chunks: list[str] = []
    for index, png in enumerate(pages):
        lines = read_page(png, lang=lang)
        header = f"--- page {index + 1} ---" if len(pages) > 1 else ""
        chunks.append(f"{header}\n{page_to_text(lines)}" if header else page_to_text(lines))
    return "\n".join(chunks).strip()
