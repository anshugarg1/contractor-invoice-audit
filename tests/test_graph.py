"""Graph behaviour with fake LLMs: routing, fan-out/fan-in, interrupt/resume, audit, fail-safe."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from invoice_checker.baseline import OracleLLM
from invoice_checker.cases import initial_state
from invoice_checker.graph import build_graph
from invoice_checker.models import ItemJudgment, JudgeResult

from conftest import TODAY


def make_graph(llm, catalogue, paid_register):
    return build_graph(llm, catalogue, paid_register, checkpointer=InMemorySaver(), today=TODAY)


def run(graph, case, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    graph.invoke(initial_state(case), config)
    return config, graph.get_state(config)


def test_clean_invoice_is_auto_approved(cases, catalogue, paid_register):
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C01_clean_plumbing"])
    assert snapshot.next == ()
    assert snapshot.values["route"] == "auto_approve"
    assert snapshot.values["decision"] == "auto_approved"
    audit = snapshot.values["audit"]
    assert audit["prompt_version"] and audit["graph_version"] and len(audit["document_sha256"]) == 64


def test_parallel_checks_all_contribute_findings(cases, catalogue, paid_register):
    # C13 has a rule finding (injection, from screen) and a tool finding (price, from check_prices).
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C13_prompt_injection"])
    found = {f["code"] for f in snapshot.values["findings"]}
    assert {"PROMPT_INJECTION_SUSPECTED", "PRICE_ABOVE_MAX"} <= found


def test_review_pauses_then_resumes_with_expert_decision(cases, catalogue, paid_register):
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    config, snapshot = run(graph, cases["C06_overpriced_labour"])
    assert snapshot.next == ("human_review",)
    payload = snapshot.interrupts[0].value
    assert any(f["code"] == "PRICE_ABOVE_MAX" for f in payload["findings"])

    graph.invoke(Command(resume={"decision": "reject", "note": "Rate too high"}), config)
    final = graph.get_state(config)
    assert final.next == ()
    assert final.values["decision"] == "rejected_by_expert"
    assert final.values["audit"]["reviewer_note"] == "Rate too high"


def test_threads_are_isolated(cases, catalogue, paid_register):
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    _, first = run(graph, cases["C06_overpriced_labour"], thread="a")
    _, second = run(graph, cases["C01_clean_plumbing"], thread="b")
    assert first.next == ("human_review",) and second.values["route"] == "auto_approve"


def test_non_invoice_skips_extraction_and_goes_to_review(cases, catalogue, paid_register):
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C17_quote_not_invoice"])
    assert snapshot.next == ("human_review",)
    assert "invoice" not in snapshot.values
    assert {f["code"] for f in snapshot.values["findings"]} == {"WRONG_DOC_TYPE"}


class PretendingJudge(OracleLLM):
    """Sounds confident, but judges a position that does not exist and skips the real ones."""

    def judge(self, claim, invoice):
        return JudgeResult(items=[ItemJudgment(position="99", verdict="plausible", reason="Looks fine.")])


def test_pretending_judge_blocks_auto_approval(cases, catalogue, paid_register):
    graph = make_graph(PretendingJudge(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C01_clean_plumbing"])
    assert snapshot.values["route"] == "human_review"
    assert "JUDGE_INCOMPLETE" in {f["code"] for f in snapshot.values["findings"]}


class BrokenExtractor(OracleLLM):
    def extract(self, text):
        raise ValueError("model returned malformed JSON")


def test_extraction_failure_fails_safe(cases, catalogue, paid_register):
    graph = make_graph(BrokenExtractor(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C01_clean_plumbing"])
    assert snapshot.values["route"] == "human_review"
    assert {f["code"] for f in snapshot.values["findings"]} == {"EXTRACTION_FAILED"}


class SpyJudge(OracleLLM):
    def __init__(self, cases):
        super().__init__(cases)
        self.seen = []

    def judge(self, claim, invoice):
        self.seen.append(claim.model_dump_json() + invoice.model_dump_json())
        return super().judge(claim, invoice)


def test_injected_text_never_reaches_the_judge(cases, catalogue, paid_register):
    spy = SpyJudge(list(cases.values()))
    run(make_graph(spy, catalogue, paid_register), cases["C13_prompt_injection"])
    assert spy.seen and all("Ignoriere" not in s and "Prüfsysteme" not in s for s in spy.seen)


def test_resume_rejects_invalid_answer(cases, catalogue, paid_register):
    graph = make_graph(OracleLLM(list(cases.values())), catalogue, paid_register)
    config, _ = run(graph, cases["C06_overpriced_labour"])
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={"decision": "maybe"}), config)


class TimingOutExtractor(OracleLLM):
    def extract(self, text):
        raise TimeoutError()


def test_provider_error_fails_safe(cases, catalogue, paid_register):
    graph = make_graph(TimingOutExtractor(list(cases.values())), catalogue, paid_register)
    _, snapshot = run(graph, cases["C01_clean_plumbing"])
    assert snapshot.values["route"] == "human_review"
    finding = snapshot.values["findings"][0]
    assert finding["code"] == "EXTRACTION_FAILED" and finding["message"].startswith("TimeoutError")

