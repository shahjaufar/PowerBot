"""CPU extraction backend: OCR the page, then reason over the *text*.

The reason this is a separate backend rather than the GPU one with a smaller
model: on CPU the cost is dominated by the vision encoder and by prefilling
one to two thousand image tokens per page. Running OCR first and handing the
model plain text removes both. A reconstructed invoice page measures around
110 tokens, so prefill stops mattering and only the ~200 tokens of JSON output
are actually generated.

The trade is real and worth stating plainly: OCR discards the visual layout
that makes a VLM good at this, so expect lower accuracy than the GPU path,
particularly on poor-quality photos where OCR itself is the limiting factor.
The validation layer is what keeps that usable — arithmetic cross-checks catch
a misread total regardless of which backend produced it.

Serve a model with llama.cpp's OpenAI-compatible server:

    python -m llama_cpp.server \\
        --model qwen2.5-7b-instruct-q4_k_m.gguf \\
        --n_ctx 8192 --n_threads 8 --port 8000
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .extract import ExtractionError
from .ocr import document_to_text
from .schema import Field, json_schema, prompt_fields_block

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an invoice data extraction engine. You are given the \
OCR text of a single invoice and must return the requested fields as JSON.

The text preserves the page's visual lines. A "|" marks a wide horizontal gap, \
so "Subtotal   |   1,000.00" means the label and value sit in separate columns \
of the same line and belong together.

Rules:
- Transcribe values exactly as they appear. Do not calculate or correct anything.
- If a field is not present, return an empty string. Never guess, and never \
carry a value over from a different field.
- The vendor issues the invoice and is paid; the customer is billed under \
"Bill To". Position on the page is not reliable evidence — use the labels.
- OCR makes character errors. Prefer a reading consistent with the rest of the \
document, but never invent a value to make the arithmetic work.
- Amounts: digits and at most one decimal point. No symbols or thousands \
separators. Use a decimal point even where the invoice uses a comma.
- A parenthesised or trailing-minus amount is negative: prefix it with a minus.
- Dates: ISO 8601 (YYYY-MM-DD)."""


def _build_payload(
    text: str, fields: list[Field], cfg: dict[str, Any]
) -> dict[str, Any]:
    user = (
        "Extract the following fields from this invoice:\n\n"
        + prompt_fields_block(fields)
        + "\n\n--- INVOICE TEXT ---\n"
        + text
    )
    return {
        "model": cfg.get("name", "local"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens": cfg.get("max_tokens", 1024),
        # llama.cpp compiles this schema to a GBNF grammar, giving the same
        # structural guarantee xgrammar provides on the GPU path.
        "response_format": {
            "type": "json_object",
            "schema": json_schema(fields),
        },
    }


async def extract_cpu(
    client: httpx.AsyncClient,
    page_images: list[bytes],
    fields: list[Field],
    cfg: dict[str, Any],
    loop: Any = None,
) -> dict[str, str]:
    """OCR then extract. Signature matches extract() so the pipeline can swap."""
    if not page_images:
        raise ExtractionError("no page images to extract from")

    lang = cfg.get("ocr_lang", "eng")

    # Tesseract is a blocking C call that releases the GIL, so pushing it to a
    # thread lets OCR of one document overlap generation for another.
    if loop is not None:
        text = await loop.run_in_executor(None, document_to_text, page_images, lang)
    else:
        text = document_to_text(page_images, lang)

    if not text.strip():
        raise ExtractionError(
            "OCR produced no text — the page is likely blank, inverted, or too "
            "low-resolution. Try raising preprocess.dpi."
        )

    try:
        response = await client.post(
            "/chat/completions", json=_build_payload(text, fields, cfg)
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ExtractionError(
            f"inference server returned {exc.response.status_code}: "
            f"{exc.response.text[:400]}"
        ) from exc
    except httpx.RequestError as exc:
        raise ExtractionError(f"could not reach inference server: {exc}") from exc

    body = response.json()
    try:
        raw = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ExtractionError(f"unexpected response shape: {body!r}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"model returned non-JSON despite a schema being supplied — check "
            f"the server supports response_format.schema. Output: {raw[:400]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ExtractionError(f"expected a JSON object, got {type(parsed).__name__}")

    return {
        f.name: ("" if parsed.get(f.name) is None else str(parsed.get(f.name)).strip())
        for f in fields
    }
