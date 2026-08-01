"""Calls the local VLM through the OpenAI-compatible endpoint vLLM/SGLang expose.

The one thing worth understanding here is `guided_json`. Without it, a model
that has been asked for JSON will produce JSON *nearly* always — and the
residual failure rate shows up as a handful of unparseable documents per
thousand that then need hand-handling. Constrained decoding masks invalid
tokens at each step, so malformed output stops being a failure mode rather
than becoming a rarer one.

It does not make the *content* correct. That is validate.py's job.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from .schema import Field, json_schema, prompt_fields_block

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an invoice data extraction engine. You are given the \
page images of a single invoice and must return the requested fields as JSON.

Rules:
- Transcribe values exactly as printed. Do not calculate, normalise or correct \
anything except where a field explicitly asks for a specific format.
- If a field is not present on the invoice, return an empty string. Never guess \
and never carry a value over from a different field.
- The vendor is the party issuing the invoice and being paid; the customer is \
the party being billed. Letterhead position is not reliable evidence — use the \
"Bill To" / "Remit To" labelling.
- Amounts: digits and at most one decimal point. Strip currency symbols and \
thousands separators. Use a decimal point even if the invoice uses a comma.
- A parenthesised or trailing-minus amount is negative: return it with a \
leading minus sign.
- Dates: ISO 8601 (YYYY-MM-DD). If the day/month order is genuinely ambiguous, \
prefer the interpretation consistent with other dates on the same document."""


class ExtractionError(RuntimeError):
    """Raised when the model could not be reached or returned unusable output."""


def _build_messages(page_images: list[bytes], fields: list[Field]) -> list[dict]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Extract the following fields from this invoice:\n\n"
                + prompt_fields_block(fields)
            ),
        }
    ]
    for png in page_images:
        b64 = base64.b64encode(png).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def extract(
    client: httpx.AsyncClient,
    page_images: list[bytes],
    fields: list[Field],
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Extract one invoice. Returns raw string values keyed by field name."""
    if not page_images:
        raise ExtractionError("no page images to extract from")

    payload = {
        "model": cfg["name"],
        "messages": _build_messages(page_images, fields),
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens": cfg.get("max_tokens", 4096),
        # vLLM and SGLang both accept this; it is the whole point of the setup.
        "guided_json": json_schema(fields),
        "guided_decoding_backend": cfg.get("guided_backend", "xgrammar"),
    }

    try:
        response = await client.post("/chat/completions", json=payload)
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
        # Should be unreachable with guided decoding active. If this fires,
        # guided_json was silently ignored — worth surfacing loudly rather than
        # limping on, since it means every document is unconstrained.
        raise ExtractionError(
            f"model returned non-JSON despite guided decoding — check that the "
            f"server supports guided_json. Raw output: {raw[:400]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ExtractionError(f"expected a JSON object, got {type(parsed).__name__}")

    # Normalise to plain stripped strings so validate.py has one input shape.
    return {
        f.name: ("" if parsed.get(f.name) is None else str(parsed.get(f.name)).strip())
        for f in fields
    }
