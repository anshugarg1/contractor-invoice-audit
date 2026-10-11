"""The LangGraph workflow.

    START -> screen -> classify -> extract -> [check_rules | check_prices | judge]   (parallel)
                                  -> route -> finalize -> END                         (clean)
                                  -> route -> human_review (interrupt) -> finalize    (findings)
    classify / extract jump straight to route when the document is not an invoice
    or cannot be parsed, so a human decides.

A fixed workflow, not a free-roaming agent: the path through an invoice check is
known, so the LLM only gets freedom where judgment is needed (classify, extract,
judge). Everything that must be correct (sums, VAT, dates, prices, duplicates) is code.
"""

from __future__ import annotations

import hashlib
import operator
from datetime import date, datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import checks
from .catalogue import CatalogueItem, PaidRegister
from .llm import InvoiceLLM
from .models import Claim, Document, Finding, to_document
from .prompts import PROMPT_VERSION

GRAPH_VERSION = "1.0"
BLOCKING_SEVERITIES = {"warning", "critical"}


class CheckState(TypedDict, total=False):
    # Input
    case_id: str
    claim: dict[str, Any]
    document_text: str
    quote_text: str | None
    # Written by nodes. Plain JSON only, so any checkpointer (memory, SQLite, Postgres) can store it.
    doc_type: str
    trade: str | None
    invoice: dict[str, Any] | None
    quote: dict[str, Any] | None
    # Three nodes add findings in parallel; the operator.add reducer concatenates their updates.
    findings: Annotated[list[dict[str, Any]], operator.add]
    judgments: list[dict[str, Any]]
    route: str
    decision: str
    reviewer_note: str | None
    audit: dict[str, Any]


def _dump(findings: list[Finding]) -> list[dict[str, Any]]:
    return [f.model_dump() for f in findings]


def _failure(code: str, severity: str, message: str, source: str) -> list[dict[str, Any]]:
    return _dump([Finding(code=code, severity=severity, message=message, source=source)])


def build_graph(
    llm: InvoiceLLM,
    catalogue: dict[str, CatalogueItem],
    paid_register: PaidRegister,
    checkpointer: BaseCheckpointSaver | None = None,
    today: date | None = None,
):
    """Compile the graph. A checkpointer is required for human review (interrupt/resume)."""

    def screen(state: CheckState) -> dict:
        found = checks.detect_injection(state["document_text"])
        if state.get("quote_text"):
            found += checks.detect_injection(state["quote_text"])
        return {"findings": _dump(found)}

    def classify(state: CheckState) -> dict:
        try:
            result = llm.classify(state["document_text"])
        except ValueError as exc:  # unparseable model output: fail safe, a human decides
            return {"doc_type": "unknown", "findings": _failure("CLASSIFICATION_FAILED", "critical", str(exc), "llm")}
        update: dict[str, Any] = {"doc_type": result.doc_type, "trade": result.trade}
        if result.doc_type != "invoice":
            update["findings"] = _failure(
                "WRONG_DOC_TYPE", "critical", f"Document was classified as '{result.doc_type}', not an invoice", "llm"
            )
        return update

    def extract(state: CheckState) -> dict:
        try:
            invoice = to_document(llm.extract(state["document_text"]), "invoice")
        except ValueError as exc:
            return {"invoice": None, "findings": _failure("EXTRACTION_FAILED", "critical", str(exc), "llm")}
        update: dict[str, Any] = {"invoice": invoice.model_dump(mode="json")}
        if state.get("quote_text"):
            try:
                update["quote"] = to_document(llm.extract(state["quote_text"]), "quote").model_dump(mode="json")
            except ValueError as exc:
                update["findings"] = _failure("QUOTE_EXTRACTION_FAILED", "warning", str(exc), "llm")
        return update

    def check_rules(state: CheckState) -> dict:
        invoice = Document.model_validate(state["invoice"])
        claim = Claim.model_validate(state["claim"])
        found = (
            checks.check_arithmetic(invoice)
            + checks.check_dates(invoice, claim, today or date.today())
            + checks.check_claim_reference(invoice, claim)
            + checks.check_duplicates(invoice, paid_register)
        )
        if state.get("quote"):
            found += checks.check_quote(invoice, Document.model_validate(state["quote"]))
        return {"findings": _dump(found)}

    def check_prices(state: CheckState) -> dict:
        invoice = Document.model_validate(state["invoice"])
        return {"findings": _dump(checks.check_prices(invoice, catalogue))}

    def judge(state: CheckState) -> dict:
        invoice = Document.model_validate(state["invoice"])
        claim = Claim.model_validate(state["claim"])
        try:
            result = llm.judge(claim, invoice)
        except ValueError as exc:
            return {"judgments": [], "findings": _failure("JUDGE_FAILED", "warning", str(exc), "llm")}
        return {
            "judgments": [j.model_dump() for j in result.items],
            "findings": _dump(checks.judge_findings(result, invoice)),
        }

    def route(state: CheckState) -> dict:
        blocking = [f for f in state.get("findings", []) if f["severity"] in BLOCKING_SEVERITIES]
        return {"route": "human_review" if blocking else "auto_approve"}

    def human_review(state: CheckState) -> dict:
        # interrupt() pauses the run and saves it in the checkpointer. On resume with
        # Command(resume=...), this node runs again from the top and interrupt()
        # returns the reviewer's answer, so nothing before it may have side effects.
        answer = interrupt({
            "case_id": state.get("case_id"),
            "question": "Expert review required: approve or reject?",
            "findings": [f for f in state.get("findings", []) if f["severity"] in BLOCKING_SEVERITIES],
        })
        decision = (answer or {}).get("decision")
        if decision not in ("approve", "reject"):
            raise ValueError(f"resume with {{'decision': 'approve' | 'reject'}}, got {answer!r}")
        return {
            "decision": "approved_by_expert" if decision == "approve" else "rejected_by_expert",
            "reviewer_note": answer.get("note") or None,
        }

    def finalize(state: CheckState, config: RunnableConfig) -> dict:
        decision = state.get("decision") or "auto_approved"
        audit = {
            "case_id": state.get("case_id"),
            "thread_id": config.get("configurable", {}).get("thread_id"),
            "document_sha256": hashlib.sha256(state["document_text"].encode("utf-8")).hexdigest(),
            "decision": decision,
            "route": state.get("route"),
            "reviewer_note": state.get("reviewer_note"),
            "findings": state.get("findings", []),
            "judgments": state.get("judgments", []),
            "model": llm.model_name,
            "prompt_version": PROMPT_VERSION,
            "graph_version": GRAPH_VERSION,
            "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return {"decision": decision, "audit": audit}

    def after_classify(state: CheckState) -> str:
        return "extract" if state.get("doc_type") == "invoice" else "route"

    def after_extract(state: CheckState) -> str | list[str]:
        if state.get("invoice") is None:
            return "route"
        return ["check_rules", "check_prices", "judge"]  # fan out: the three run in parallel

    def after_route(state: CheckState) -> str:
        return "human_review" if state["route"] == "human_review" else "finalize"

    builder = StateGraph(CheckState)
    for name, node in [
        ("screen", screen), ("classify", classify), ("extract", extract),
        ("check_rules", check_rules), ("check_prices", check_prices), ("judge", judge),
        ("route", route), ("human_review", human_review), ("finalize", finalize),
    ]:
        builder.add_node(name, node)

    builder.add_edge(START, "screen")
    builder.add_edge("screen", "classify")
    builder.add_conditional_edges("classify", after_classify, ["extract", "route"])
    builder.add_conditional_edges("extract", after_extract, ["check_rules", "check_prices", "judge", "route"])
    builder.add_edge(["check_rules", "check_prices", "judge"], "route")  # fan in: waits for all three
    builder.add_conditional_edges("route", after_route, ["human_review", "finalize"])
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
