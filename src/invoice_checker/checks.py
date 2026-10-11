"""Deterministic checks. Pure functions: typed document in, findings out. No LLM here.

Rule of thumb for the whole project: use code where correctness is needed
(arithmetic, dates, lookups) and the LLM only where judgment is needed.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from decimal import Decimal

from .catalogue import CatalogueItem, PaidRegister
from .models import Claim, Document, Finding, JudgeResult
from .money import to_cents

TOLERANCE = Decimal("0.01")  # one cent, for rounding on printed invoices
STANDARD_VAT = Decimal("19")
TYPICAL_MARGIN = Decimal("1.15")  # above typical +15 % -> info only
QUOTE_MARGIN = Decimal("1.10")  # invoice may exceed the quote by 10 %


def _finding(code, severity, message, source="rule", position=None) -> Finding:
    return Finding(code=code, severity=severity, message=message, source=source, position=position)


def check_arithmetic(doc: Document) -> list[Finding]:
    out = []
    for item in doc.line_items:
        expected = to_cents(item.quantity * item.unit_price)
        if abs(expected - item.line_total) > TOLERANCE:
            out.append(_finding(
                "ARITH_LINE_TOTAL", "critical",
                f"Pos. {item.position}: {item.quantity} x {item.unit_price} = {expected}, but {item.line_total} is printed",
                position=item.position,
            ))
    lines_sum = sum((i.line_total for i in doc.line_items), Decimal("0"))
    if abs(lines_sum - doc.net_total) > TOLERANCE:
        out.append(_finding("ARITH_NET_TOTAL", "critical", f"Lines add up to {lines_sum}, but net total is {doc.net_total}"))
    if doc.vat_rate != STANDARD_VAT:
        out.append(_finding("VAT_RATE_UNUSUAL", "warning", f"VAT rate {doc.vat_rate} %, expected {STANDARD_VAT} %"))
    expected_vat = to_cents(doc.net_total * doc.vat_rate / 100)
    if abs(expected_vat - doc.vat_amount) > TOLERANCE:
        out.append(_finding("ARITH_VAT", "critical", f"{doc.vat_rate} % of {doc.net_total} = {expected_vat}, but VAT is {doc.vat_amount}"))
    if abs(doc.net_total + doc.vat_amount - doc.gross_total) > TOLERANCE:
        out.append(_finding("ARITH_GROSS_TOTAL", "critical", f"Net {doc.net_total} + VAT {doc.vat_amount} != gross {doc.gross_total}"))
    return out


def check_dates(doc: Document, claim: Claim, today: date) -> list[Finding]:
    out = []
    if doc.date < claim.damage_date:
        out.append(_finding("DATE_BEFORE_DAMAGE", "critical", f"Document dated {doc.date}, damage happened on {claim.damage_date}"))
    if doc.date > today:
        out.append(_finding("DATE_IN_FUTURE", "warning", f"Document dated {doc.date}, today is {today}"))
    return out


def check_claim_reference(doc: Document, claim: Claim) -> list[Finding]:
    if doc.claim_number is None:
        return [_finding("CLAIM_NUMBER_MISSING", "info", "No claim number printed")]
    if "".join(doc.claim_number.split()).upper() != "".join(claim.claim_number.split()).upper():
        return [_finding("CLAIM_NUMBER_MISMATCH", "warning", f"Document refers to claim {doc.claim_number}, not {claim.claim_number}")]
    return []


def check_duplicates(doc: Document, paid: PaidRegister) -> list[Finding]:
    out = []
    if paid.contains(doc.contractor, doc.number):
        out.append(_finding("DUPLICATE_INVOICE", "critical", f"Invoice {doc.number} from {doc.contractor} was already paid", source="tool"))
    counts = Counter((i.code, i.description.casefold(), i.quantity, i.unit_price) for i in doc.line_items)
    for (code, description, _, _), n in counts.items():
        if n > 1:
            out.append(_finding("DUPLICATE_LINE", "warning", f"'{description}' ({code or 'no code'}) is billed {n} times"))
    return out


_UNIT_ALIASES = {
    "stück": "stk", "st": "stk", "stunde": "std", "stunden": "std", "h": "std",
    "qm": "m2", "psch": "pauschal", "pauschale": "pauschal", "tage": "tag",
}


def normalise_unit(unit: str) -> str:
    u = unit.strip().casefold().replace("²", "2").replace(".", "")
    return _UNIT_ALIASES.get(u, u)


def check_prices(doc: Document, catalogue: dict[str, CatalogueItem]) -> list[Finding]:
    out = []
    for item in doc.line_items:
        ref = catalogue.get(item.code.upper()) if item.code else None
        if ref is None:
            out.append(_finding(
                "UNKNOWN_ITEM", "warning",
                f"Pos. {item.position} '{item.description}' has no catalogue price; an expert must assess it",
                source="tool", position=item.position,
            ))
            continue
        if normalise_unit(item.unit) != normalise_unit(ref.unit):
            out.append(_finding(
                "UNIT_MISMATCH", "warning",
                f"Pos. {item.position}: billed per '{item.unit}', catalogue prices per '{ref.unit}'",
                source="tool", position=item.position,
            ))
            continue
        above_typical = (item.unit_price / ref.typical_price - 1) * 100
        if item.unit_price > ref.max_price:
            out.append(_finding(
                "PRICE_ABOVE_MAX", "warning",
                f"Pos. {item.position}: {item.unit_price} €/{ref.unit} is above the maximum {ref.max_price} "
                f"({above_typical:+.0f} % vs typical {ref.typical_price})",
                source="tool", position=item.position,
            ))
        elif item.unit_price > ref.typical_price * TYPICAL_MARGIN:
            out.append(_finding(
                "PRICE_ABOVE_TYPICAL", "info",
                f"Pos. {item.position}: {above_typical:+.0f} % vs typical price",
                source="tool", position=item.position,
            ))
    return out


def check_quote(invoice: Document, quote: Document) -> list[Finding]:
    out = []
    if invoice.gross_total > quote.gross_total * QUOTE_MARGIN:
        pct = (invoice.gross_total / quote.gross_total - 1) * 100
        out.append(_finding("EXCEEDS_QUOTE", "warning", f"Invoice {invoice.gross_total} is {pct:.0f} % above quote {quote.gross_total}"))
    quoted = {i.code: i for i in quote.line_items if i.code}
    for item in invoice.line_items:
        ref = quoted.get(item.code) if item.code else None
        if ref is None:
            out.append(_finding("NOT_IN_QUOTE", "warning", f"Pos. {item.position} '{item.description}' was not quoted", position=item.position))
            continue
        if item.quantity > ref.quantity * QUOTE_MARGIN:
            out.append(_finding(
                "QTY_ABOVE_QUOTE", "warning",
                f"Pos. {item.position}: {item.quantity} {item.unit} billed, {ref.quantity} quoted",
                position=item.position,
            ))
        if item.unit_price > ref.unit_price + TOLERANCE:
            out.append(_finding(
                "PRICE_ABOVE_QUOTE", "warning",
                f"Pos. {item.position}: {item.unit_price} billed, {ref.unit_price} quoted",
                position=item.position,
            ))
    return out


# A tripwire, not the defence. The real defence is architectural: document text
# is data, the judge never sees the raw text, and arithmetic cannot be talked round.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignor\w*\s+(all\w*\s+)?(previous|prior|above|vorherige\w*|bisherige\w*|obige\w*)\s+(instructions|anweisungen)",
        r"\b(ki|ai|llm)\b[-\s]*(system|prüf\w*|assistent\w*|model\w*)",
        r"automatische\w*\s+prüf\w*",
        r"system\s*prompt",
        r"(als|as)\s+(plausibel|plausible|approved|freigegeben)\s+(bewerten|bewerte|markieren|markiere|mark|rate)",
        r"bereits\s+(vom\s+versicherer\s+)?(geprüft|freigegeben)",
    )
]


def detect_injection(text: str) -> list[Finding]:
    hits = [m.group(0) for p in _INJECTION_PATTERNS if (m := p.search(text))]
    if not hits:
        return []
    return [_finding("PROMPT_INJECTION_SUSPECTED", "critical", f"Text addressed to an automated checker: {hits[0]!r}")]


def judge_findings(result: JudgeResult, doc: Document) -> list[Finding]:
    """Turn the LLM judge's verdicts into findings, and catch a judge that is 'pretending'.

    A judge that skips positions or cites positions that do not exist is not doing
    the task, however fluent its reasons sound. That alone sends the invoice to a human.
    """
    out = []
    expected = {i.position for i in doc.line_items}
    cited = [j.position.strip().rstrip(".") for j in result.items]
    missing, unknown = expected - set(cited), set(cited) - expected
    if missing or unknown:
        out.append(_finding(
            "JUDGE_INCOMPLETE", "warning",
            f"Judge skipped positions {sorted(missing)} and cited unknown positions {sorted(unknown)}",
            source="llm",
        ))
    for judgment, position in zip(result.items, cited):
        if judgment.verdict != "plausible" and position in expected:
            out.append(_finding(
                "JUDGE_FLAGGED", "warning",
                f"Pos. {position} {judgment.verdict}: {judgment.reason}",
                source="llm", position=position,
            ))
    return out
