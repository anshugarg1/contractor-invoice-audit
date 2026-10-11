from datetime import date

import pytest

from invoice_checker.cases import CASES_DIR, CATALOGUE_PATH, PAID_REGISTER_PATH, load_cases
from invoice_checker.catalogue import load_catalogue, load_paid_register

TODAY = date(2026, 9, 13)


@pytest.fixture(scope="session")
def catalogue():
    return load_catalogue(CATALOGUE_PATH)


@pytest.fixture(scope="session")
def paid_register():
    return load_paid_register(PAID_REGISTER_PATH)


@pytest.fixture(scope="session")
def cases():
    loaded = load_cases(CASES_DIR)
    assert loaded, "no cases found; run: python scripts/build_cases.py"
    return {c["case_id"]: c for c in loaded}
