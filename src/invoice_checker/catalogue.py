"""Reference data: a toy price catalogue (Leistungsverzeichnis) and the register of paid invoices."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class CatalogueItem:
    code: str
    trade: str
    description: str
    unit: str
    typical_price: Decimal
    max_price: Decimal


def load_catalogue(path: Path) -> dict[str, CatalogueItem]:
    with open(path, encoding="utf-8", newline="") as f:
        return {
            row["code"].upper(): CatalogueItem(
                code=row["code"].upper(),
                trade=row["trade"],
                description=row["description"],
                unit=row["unit"],
                typical_price=Decimal(row["typical_price"]),
                max_price=Decimal(row["max_price"]),
            )
            for row in csv.DictReader(f)
        }


def _key(contractor: str, invoice_number: str) -> tuple[str, str]:
    return " ".join(contractor.casefold().split()), "".join(invoice_number.upper().split())


class PaidRegister:
    """Invoices that were already paid. Matched on contractor name + invoice number.

    A real system would match on tax ID or IBAN as well, because names are spelled
    inconsistently.
    """

    def __init__(self, entries: set[tuple[str, str]]) -> None:
        self._entries = entries

    def contains(self, contractor: str, invoice_number: str) -> bool:
        return _key(contractor, invoice_number) in self._entries


def load_paid_register(path: Path) -> PaidRegister:
    with open(path, encoding="utf-8", newline="") as f:
        return PaidRegister({_key(row["contractor"], row["invoice_number"]) for row in csv.DictReader(f)})
