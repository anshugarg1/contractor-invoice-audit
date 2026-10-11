"""Loading the evaluation cases in data/cases/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
CATALOGUE_PATH = DATA_DIR / "price_catalogue.csv"
PAID_REGISTER_PATH = DATA_DIR / "paid_invoices.csv"


def load_case(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cases(directory: Path = CASES_DIR, pattern: str = "*.json") -> list[dict]:
    return [load_case(p) for p in sorted(Path(directory).glob(pattern))]


def initial_state(case: dict) -> dict:
    """The graph input for one case. Ground truth and expectations stay out of the graph."""
    return {
        "case_id": case["case_id"],
        "claim": case["claim"],
        "document_text": case["document_text"],
        "quote_text": case.get("quote_text"),
        "findings": [],
    }
