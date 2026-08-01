"""Document -> clean page images.

On phone photos this stage moves accuracy more than any model choice will.
Skew, perspective, shadow and low effective DPI each degrade extraction on
their own, and they usually arrive together.

Every step is individually reversible via config, because preprocessing that
is wrong for your inputs is worse than none at all — an autocrop that latches
onto a desk edge will happily crop away half the invoice.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


def rasterise(path: Path, cfg: dict[str, Any]) -> list[np.ndarray]:
    """Load a PDF or image file as a list of BGR page arrays."""
    suffix = path.suffix.lower()
    max_pages = int(cfg.get("max_pages", 10))

    if suffix in PDF_SUFFIXES:
        return _rasterise_pdf(path, dpi=int(cfg.get("dpi", 300)), max_pages=max_pages)

    if suffix in IMAGE_SUFFIXES:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        page = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if page is None:
            raise ValueError(f"could not decode image: {path}")
        return [page]

    raise ValueError(f"unsupported file type {suffix!r}: {path}")


def _rasterise_pdf(path: Path, dpi: int, max_pages: int) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    with fitz.open(path) as doc:
        if doc.page_count > max_pages:
            log.warning(
                "%s has %d pages; processing first %d only",
                path.name, doc.page_count, max_pages,
            )
        for page in doc.pages(0, min(doc.page_count, max_pages)):
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # PyMuPDF gives RGB(A); OpenCV wants BGR.
            code = cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
            pages.append(cv2.cvtColor(img, code))
    return pages


def _detect_skew(gray: np.ndarray, max_skew_deg: float) -> float:
    """Estimate page skew from the dominant near-horizontal text lines.

    minAreaRect over all foreground pixels is the common recipe, but it keys on
    the bounding box of the whole page and gets dragged around by marginal
    content. Hough lines over dilated text rows track the baselines themselves,
    which is what we actually want to level.
    """
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Smear characters into continuous lines so Hough sees baselines, not glyphs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    smeared = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        smeared, rho=1, theta=np.pi / 720, threshold=120,
        minLineLength=max(60, gray.shape[1] // 6), maxLineGap=25,
    )
    if lines is None:
        return 0.0

    angles = []
    # HoughLinesP returns (N, 1, 4) on most builds but (N, 4) on some;
    # reshaping makes the iteration independent of which one we got.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines; vertical rules and table borders
        # would otherwise pull the median to 90 degrees.
        if abs(angle) <= max_skew_deg:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def _deskew(img: np.ndarray, max_skew_deg: float) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = _detect_skew(gray, max_skew_deg)

    # Sub-degree skew is invisible to the model and rotation costs a resample.
    if abs(angle) < 0.15:
        return img

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    # Expand the canvas so rotation doesn't guillotine the corners.
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2) - (w / 2)
    matrix[1, 2] += (new_h / 2) - (h / 2)

    return cv2.warpAffine(
        img, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _autocrop(img: np.ndarray) -> np.ndarray:
    """Crop to the page when it sits on a contrasting background.

    Deliberately conservative: a crop that removes real content is far more
    damaging than a crop that never happened, so anything suspicious is
    declined and the original returned.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    page_area = img.shape[0] * img.shape[1]
    area_ratio = cv2.contourArea(largest) / page_area

    # Below 35% we probably found a logo or a text block, not the page.
    # Above 98% there was nothing to crop anyway.
    if not (0.35 < area_ratio < 0.98):
        return img

    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    return img[y : y + h, x : x + w]


def _enhance(img: np.ndarray) -> np.ndarray:
    """Local contrast equalisation on the luminance channel.

    CLAHE rather than a global stretch, because the characteristic phone-photo
    failure is a shadow gradient across the page: a global curve that fixes the
    lit half blows out the dark half.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _resize(img: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def prepare(path: Path, cfg: dict[str, Any]) -> list[bytes]:
    """Full pipeline: file -> cleaned page images as PNG bytes."""
    pages = rasterise(path, cfg)
    out: list[bytes] = []

    for i, page in enumerate(pages):
        try:
            if cfg.get("autocrop", True):
                page = _autocrop(page)
            if cfg.get("deskew", True):
                page = _deskew(page, float(cfg.get("max_skew_deg", 15)))
            if cfg.get("enhance_contrast", True):
                page = _enhance(page)
            page = _resize(page, int(cfg.get("max_edge_px", 2200)))
        except cv2.error as exc:
            # A preprocessing failure must not lose the page — the model can
            # usually still read the raw scan.
            log.warning("preprocessing failed on %s page %d (%s); using raw page",
                        path.name, i + 1, exc)

        rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
        out.append(buf.getvalue())

    return out
