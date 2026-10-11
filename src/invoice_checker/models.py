"""Data models.

Two layers on purpose:
- Extracted*: what the LLM returns. Every value is a string copied as printed.
- Document / LineItem: typed values (Decimal, date) that the checks run on.

The LLM copies, the code computes. `to_document` is the only bridge, and it
raises ValueError when something does not parse, which sends the invoice to a
human instead of guessing.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from .money import parse_amount, parse_date

Trade = Literal["Sanitär", "Trocknung", "Maler", "Bodenleger", "Dachdecker", "Elektro", "Sonstiges"]
Severity = Literal["info", "warning", "critical"]


# --- Domain -----------------------------------------------------------------


class Claim(BaseModel):
    claim_number: str
    insurer: str
    damage_type: str
    damage_date: date
    location: str
    description: str


class LineItem(BaseModel):
    position: str
    code: str | None = None
    description: str
    quantity: Decimal
    unit: str
    unit_price: Decimal
    line_total: Decimal


class Document(BaseModel):
    doc_type: Literal["invoice", "quote"]
    number: str
    date: date
    contractor: str
    claim_number: str | None = None
    line_items: list[LineItem]
    net_total: Decimal
    vat_rate: Decimal  # percent, e.g. 19
    vat_amount: Decimal
    gross_total: Decimal


class Finding(BaseModel):
    """One reason why a document may need an expert. warning/critical block auto-approval."""

    code: str
    severity: Severity
    message: str
    source: Literal["rule", "tool", "llm"]
    position: str | None = None


# --- LLM output schemas -----------------------------------------------------


class Classification(BaseModel):
    """Type and trade of a document submitted in a property-insurance claim."""

    doc_type: Literal["invoice", "quote", "other"] = Field(
        description="invoice = Rechnung; quote = Angebot or Kostenvoranschlag; other = anything else"
    )
    trade: Trade = Field(description="The contractor's trade (Gewerk)")


class ExtractedLineItem(BaseModel):
    position: str = Field(description="Position number exactly as printed")
    code: str | None = Field(description="Catalogue code exactly as printed, e.g. 'SAN-004'; null if none")
    description: str = Field(description="Description of the work exactly as printed")
    quantity: str = Field(description="Quantity exactly as printed, e.g. '12,5'")
    unit: str = Field(description="Unit exactly as printed, e.g. 'm²', 'Std', 'pauschal'")
    unit_price: str = Field(description="Unit price exactly as printed, e.g. '38,00 €'")
    line_total: str = Field(description="Line total exactly as printed")


class ExtractedDocument(BaseModel):
    """Fields of a contractor invoice or quote, copied exactly as printed. Nothing is calculated or corrected."""

    number: str = Field(description="Invoice or quote number")
    date: str = Field(description="Invoice or quote date exactly as printed")
    contractor: str = Field(description="Company name of the contractor who issued the document")
    claim_number: str | None = Field(description="Claim number (Schadennummer) if printed, else null")
    line_items: list[ExtractedLineItem]
    net_total: str = Field(description="Net total (Nettobetrag) exactly as printed")
    vat_rate: str = Field(description="VAT rate in percent exactly as printed, e.g. '19'")
    vat_amount: str = Field(description="VAT amount (MwSt.) exactly as printed")
    gross_total: str = Field(description="Gross total (brutto) exactly as printed")


class ItemJudgment(BaseModel):
    position: str = Field(description="Position number of the invoice line being judged")
    verdict: Literal["plausible", "questionable", "implausible"]
    reason: str = Field(description="One sentence that refers to the claim description")


class JudgeResult(BaseModel):
    """One judgment per invoice position."""

    items: list[ItemJudgment]


# --- Bridge -----------------------------------------------------------------

_NONE_MARKERS = {"", "-", "–", "—", "null", "none", "n/a"}


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value.casefold() in _NONE_MARKERS else value


def to_document(extracted: ExtractedDocument, doc_type: Literal["invoice", "quote"]) -> Document:
    """Turn strings copied by the LLM into typed values. Raises ValueError if anything does not parse."""
    if not extracted.line_items:
        raise ValueError("no line items extracted")
    items = []
    for item in extracted.line_items:
        code = _optional(item.code)
        items.append(
            LineItem(
                position=item.position.strip().rstrip("."),
                code=code.upper() if code else None,
                description=item.description.strip(),
                quantity=parse_amount(item.quantity),
                unit=item.unit.strip(),
                unit_price=parse_amount(item.unit_price),
                line_total=parse_amount(item.line_total),
            )
        )
    return Document(
        doc_type=doc_type,
        number=extracted.number.strip(),
        date=parse_date(extracted.date),
        contractor=extracted.contractor.strip(),
        claim_number=_optional(extracted.claim_number),
        line_items=items,
        net_total=parse_amount(extracted.net_total),
        vat_rate=parse_amount(extracted.vat_rate),
        vat_amount=parse_amount(extracted.vat_amount),
        gross_total=parse_amount(extracted.gross_total),
    )
