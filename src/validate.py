"""Coercion, arithmetic checks and confidence scoring.

This is the stage that decides whether the pipeline is usable. A 95%-accurate
extractor with no validation gives you a register where 1 row in 20 is quietly
wrong and you cannot tell which. The same extractor with arithmetic
cross-checks gives you a register where the doubtful rows are labelled — and
invoices happen to be unusually well suited to this, because subtotal + tax =
total is a constraint the document carries with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as date_parser

from .schema import Field

# Currency symbols and codes that may cling to an amount despite the prompt.
_CURRENCY_NOISE = re.compile(r"[^\d.,\-()]")
_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")

_SYMBOL_TO_ISO = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
    "₹": "INR", "₽": "RUB", "R$": "BRL", "CHF": "CHF",
}


@dataclass
class ValidatedInvoice:
    values: dict[str, Any]
    flags: list[str] = dc_field(default_factory=list)
    confidence: float = 1.0

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


def _parse_amount(raw: str) -> Decimal | None:
    """Parse an amount, surviving the formats invoices actually use."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    negative = text.startswith("(") and text.endswith(")")
    text = _CURRENCY_NOISE.sub("", text).replace("(", "").replace(")", "")
    if text.endswith("-"):      # trailing-minus notation
        negative, text = True, text[:-1]
    if not text:
        return None

    # Disambiguate European 1.234,56 from US 1,234.56 by which separator is
    # rightmost — the rightmost one is the decimal point.
    last_comma, last_dot = text.rfind(","), text.rfind(".")
    if last_comma > last_dot:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return -value if negative and value > 0 else value


def _parse_date(raw: str, min_year: int, max_year: int) -> date | None:
    if not raw or not raw.strip():
        return None
    try:
        # dayfirst=False matches the ISO output the prompt asks for; genuinely
        # ambiguous vendor formats are resolved model-side, where the rest of
        # the document is visible as context.
        parsed = date_parser.parse(raw.strip(), fuzzy=True).date()
    except (ValueError, OverflowError, TypeError):
        return None
    if not (min_year <= parsed.year <= max_year):
        return None
    return parsed


def _normalise_currency(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip().upper()
    if _ISO_CURRENCY.match(text):
        return text
    for symbol, iso in _SYMBOL_TO_ISO.items():
        if symbol in raw:
            return iso
    return None


def validate(
    raw: dict[str, str],
    fields: list[Field],
    cfg: dict[str, Any],
) -> ValidatedInvoice:
    """Coerce raw model output into typed values and score our confidence in it."""
    tolerance = Decimal(str(cfg.get("arithmetic_tolerance", 0.02)))
    min_year = int(cfg.get("min_year", 2000))
    max_year = int(cfg.get("max_year", 2100))

    values: dict[str, Any] = {}
    flags: list[str] = []

    for f in fields:
        text = raw.get(f.name, "")

        if f.type in ("currency", "number"):
            parsed = _parse_amount(text)
            if text and parsed is None:
                flags.append(f"{f.name}: could not parse amount {text!r}")
            values[f.name] = parsed

        elif f.type == "date":
            parsed = _parse_date(text, min_year, max_year)
            if text and parsed is None:
                flags.append(f"{f.name}: could not parse date {text!r}")
            values[f.name] = parsed

        elif f.type == "integer":
            try:
                values[f.name] = int(text) if text else None
            except ValueError:
                flags.append(f"{f.name}: could not parse integer {text!r}")
                values[f.name] = None

        elif f.name == "currency":
            iso = _normalise_currency(text)
            if text and iso is None:
                flags.append(f"currency: unrecognised code {text!r}")
            values[f.name] = iso or text or None

        else:
            values[f.name] = text or None

        if f.required and values.get(f.name) in (None, ""):
            flags.append(f"{f.name}: required field is missing")

    _check_arithmetic(values, tolerance, flags)
    _check_dates(values, flags)

    # Each independent problem halves confidence. The absolute number matters
    # less than the ordering — it exists to sort a review queue, not to be
    # interpreted as a probability.
    confidence = round(0.5 ** len(flags), 4)

    return ValidatedInvoice(values=values, flags=flags, confidence=confidence)


def _check_arithmetic(
    values: dict[str, Any], tolerance: Decimal, flags: list[str]
) -> None:
    """subtotal + tax == total.

    The highest-value check available: it uses two fields to verify a third,
    so a misread in any one of them surfaces. Only meaningful when all three
    were found.
    """
    subtotal = values.get("subtotal")
    tax = values.get("tax_amount")
    total = values.get("total_amount")

    if subtotal is None or total is None:
        return

    expected = subtotal + (tax or Decimal(0))
    delta = abs(expected - total)
    if delta > tolerance:
        flags.append(
            f"arithmetic: subtotal {subtotal} + tax {tax or 0} = {expected}, "
            f"but total reads {total} (off by {delta})"
        )

    if total is not None and total < 0:
        flags.append(f"total_amount: negative total {total} — credit note?")


def _check_dates(values: dict[str, Any], flags: list[str]) -> None:
    invoice_date = values.get("invoice_date")
    due_date = values.get("due_date")

    if isinstance(invoice_date, date) and isinstance(due_date, date):
        if due_date < invoice_date:
            flags.append(
                f"dates: due date {due_date} precedes invoice date {invoice_date} "
                f"— fields may be swapped"
            )
