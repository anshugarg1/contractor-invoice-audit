"""Parsing and formatting of money, quantities and dates.

Money is always Decimal, never float: in binary floating point 0.1 + 0.2 != 0.3,
and an invoice check that is off by a cent is a wrong invoice check.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

CENT = Decimal("0.01")
_NOISE = re.compile(r"(?i)eur|€|%|\s")
_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d.%m.%y")


def to_cents(value: Decimal) -> Decimal:
    """Round to cents the way invoices do (half up, not banker's rounding)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def parse_amount(raw: str | int | Decimal) -> Decimal:
    """Parse an amount or quantity as printed: '1.234,50 €', '12,5', '1234.50', '19 %'.

    A value with only a dot is read as a decimal point ('1.234' -> 1.234). German
    invoices always print cents ('1.234,00'), so this ambiguity does not arise for
    money in practice.
    """
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise ValueError(f"expected a string amount, got {type(raw).__name__}")
    s = _NOISE.sub("", str(raw))
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):  # German: 1.234,50
            s = s.replace(".", "").replace(",", ".")
        else:  # English: 1,234.50
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        value = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"not an amount: {raw!r}") from None
    if not value.is_finite():
        raise ValueError(f"not an amount: {raw!r}")
    return value


def parse_date(raw: str) -> date:
    """Parse '14.07.2026' or '2026-07-14'."""
    s = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"not a date: {raw!r}")


def format_eur(value: Decimal) -> str:
    """1234.5 -> '1.234,50 €'"""
    return f"{to_cents(value):,.2f}".translate(str.maketrans(",.", ".,")) + " €"


def format_qty(value: Decimal) -> str:
    """12.5 -> '12,5'; 40 -> '40'"""
    return format(value.normalize(), "f").replace(".", ",")


def format_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")
