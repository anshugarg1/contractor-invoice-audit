"""Rules-only baseline: a stand-in for the LLM that reads documents perfectly and never judges.

It returns the ground-truth classification and extraction from the case files, and
marks every position as plausible. Running the eval with it answers the question
"what do rules plus a price lookup achieve on their own?" (baseline first), and it
needs no API key. It only works on the eval cases, because it looks up their ground truth.
"""

from __future__ import annotations

from .models import Claim, Classification, Document, ExtractedDocument, ItemJudgment, JudgeResult


class OracleLLM:
    model_name = "baseline: ground-truth extraction + rules only (no LLM judgment)"

    def __init__(self, cases: list[dict]) -> None:
        self._types: dict[str, tuple[str, str]] = {}
        self._docs: dict[str, dict] = {}
        for case in cases:
            truth = case["truth"]
            self._types[case["document_text"]] = (truth["doc_type"], truth["trade"])
            self._docs[case["document_text"]] = truth["document"]
            if case.get("quote_text"):
                self._docs[case["quote_text"]] = truth["quote"]

    def classify(self, text: str) -> Classification:
        doc_type, trade = self._types[text]
        return Classification(doc_type=doc_type, trade=trade)

    def extract(self, text: str) -> ExtractedDocument:
        return ExtractedDocument.model_validate(self._docs[text])

    def judge(self, claim: Claim, invoice: Document) -> JudgeResult:
        return JudgeResult(items=[
            ItemJudgment(position=i.position, verdict="plausible", reason="Baseline: no judgment.")
            for i in invoice.line_items
        ])
