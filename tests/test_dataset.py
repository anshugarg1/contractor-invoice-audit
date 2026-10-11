"""The eval set is only useful if its labels are right. These tests lock that down.

For every case, running the deterministic checks on the ground-truth document must
produce exactly the rule findings the case expects. Findings that only the LLM can
produce (judge, classification) are excluded here.
"""

from invoice_checker import checks
from invoice_checker.money import parse_amount
from invoice_checker.models import Claim, Document, ExtractedDocument, to_document

from conftest import TODAY

LLM_ONLY_CODES = {"JUDGE_FLAGGED", "WRONG_DOC_TYPE"}


def truth_document(case, key="document", doc_type="invoice") -> Document:
    return to_document(ExtractedDocument.model_validate(case["truth"][key]), doc_type)


def rule_codes(case, catalogue, paid_register) -> set[str]:
    invoice = truth_document(case)
    claim = Claim.model_validate(case["claim"])
    found = (
        checks.detect_injection(case["document_text"])
        + checks.check_arithmetic(invoice)
        + checks.check_dates(invoice, claim, TODAY)
        + checks.check_claim_reference(invoice, claim)
        + checks.check_duplicates(invoice, paid_register)
        + checks.check_prices(invoice, catalogue)
    )
    if case["truth"]["quote"]:
        found += checks.check_quote(invoice, truth_document(case, "quote", "quote"))
    return {f.code for f in found if f.severity != "info"}


def test_there_are_clean_and_bad_cases(cases):
    routes = [c["expected"]["route"] for c in cases.values()]
    assert routes.count("auto_approve") >= 5
    assert routes.count("human_review") >= 10


def test_rule_findings_match_labels(cases, catalogue, paid_register):
    for case_id, case in cases.items():
        if case["truth"]["doc_type"] != "invoice":
            continue
        expected = set(case["expected"]["codes"]) - LLM_ONLY_CODES
        assert rule_codes(case, catalogue, paid_register) == expected, case_id


def test_judgment_cases_pass_every_rule(cases, catalogue, paid_register):
    # These cases exist to show what rules cannot catch.
    for case in cases.values():
        if case["category"] == "judgment":
            assert rule_codes(case, catalogue, paid_register) == set(), case["case_id"]


def test_printed_gross_matches_ground_truth(cases):
    for case in cases.values():
        gross = case["truth"]["document"]["gross_total"]
        line = next(line for line in case["document_text"].splitlines() if "(brutto)" in line)
        assert parse_amount(line.split(":", 1)[1]) == parse_amount(gross), case["case_id"]
