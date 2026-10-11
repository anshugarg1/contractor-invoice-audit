from datetime import date
from decimal import Decimal

import pytest

from invoice_checker.money import format_eur, format_qty, parse_amount, parse_date, to_cents


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234,50 €", "1234.50"),
        ("1234,50", "1234.50"),
        ("1234.50", "1234.50"),
        ("1,234.50", "1234.50"),
        ("12,5", "12.5"),
        ("0,40 €", "0.40"),
        ("19 %", "19"),
        ("2.345.678,90 EUR", "2345678.90"),
        (" 45,00 € ", "45.00"),
        (7, "7"),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["", "abc", "12,50,00x", "NaN"])
def test_parse_amount_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_amount(raw)


def test_parse_amount_refuses_float():
    with pytest.raises(ValueError):
        parse_amount(0.1)


def test_to_cents_rounds_half_up_not_bankers():
    assert to_cents(Decimal("0.125")) == Decimal("0.13")
    assert to_cents(Decimal("2.675")) == Decimal("2.68")  # float would give 2.67


def test_formatting_round_trips():
    assert format_eur(Decimal("1234.5")) == "1.234,50 €"
    assert parse_amount(format_eur(Decimal("1234.5"))) == Decimal("1234.50")
    assert format_qty(Decimal("12.5")) == "12,5"
    assert format_qty(Decimal("40")) == "40"


def test_parse_date():
    assert parse_date("14.07.2026") == date(2026, 7, 14)
    assert parse_date("2026-07-14") == date(2026, 7, 14)
    with pytest.raises(ValueError):
        parse_date("July 14")
