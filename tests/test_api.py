
"""The HTTP API end to end with the baseline LLM: submit, pause for review, resume."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from invoice_checker.api import create_app
from invoice_checker.baseline import OracleLLM
from invoice_checker.cases import CASES_DIR, load_case, load_cases


def client() -> TestClient:
    return TestClient(create_app(OracleLLM(load_cases(CASES_DIR)), InMemorySaver()))


def test_overpriced_invoice_waits_for_review_then_finishes():
    api = client()
    case = load_case(CASES_DIR / "C06_overpriced_labour.json")
    thread_id = api.post("/invoices", json=case).json()["id"]
    assert api.get(f"/invoices/{thread_id}").json()["status"] == "waiting_for_review"

    assert api.post(f"/invoices/{thread_id}/review", json={"decision": "reject"}).status_code == 202
    body = api.get(f"/invoices/{thread_id}").json()
    assert body["status"] == "done"
    assert body["audit"]["decision"] == "rejected_by_expert"


def test_review_is_refused_when_nothing_is_waiting():
    assert client().post("/invoices/nope/review", json={"decision": "approve"}).status_code == 409

