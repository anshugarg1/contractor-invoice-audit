"""Optional Langfuse tracing. Off unless LANGFUSE_PUBLIC_KEY and
  LANGFUSE_SECRET_KEY are set."""

from __future__ import annotations

import os


def _enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and
os.environ.get("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks() -> list:
    """Callbacks to put in a LangGraph config. Empty list when tracing is
off."""
    if not _enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def flush() -> None:
    """Send buffered traces before a short script exits."""
    if _enabled():
        from langfuse import get_client

        get_client().flush()

def last_trace_id(callbacks: list) -> str | None:
    """Id of the trace the Langfuse handler in `callbacks` just
recorded, or None."""
    return next((cb.last_trace_id for cb in callbacks if
getattr(cb, "last_trace_id", None)), None)


def score_trace(callbacks: list, scores: dict[str, float]) ->None:
    """Attach scores to the trace the Langfuse handler in `callbacks` just recorded."""
    # trace_id = next((cb.last_trace_id for cb in callbacks if getattr(cb, "last_trace_id", None)), None)
    trace_id = last_trace_id(callbacks)
    if trace_id is None:  # tracing off, or nothing was recorded
        return
    from langfuse import get_client

    client = get_client()
    for name, value in scores.items():
        client.create_score(trace_id=trace_id, name=name,value=value)


DATASET_NAME = "invoice-eval"

def sync_dataset(cases: list[dict]) -> None:
    """Upload the eval cases as a Langfuse dataset. Safe to
re-run: each case has a fixed item id."""
    if not _enabled():
        return
    from langfuse import get_client

    client = get_client()
    client.create_dataset(name=DATASET_NAME,description="Synthetic contractor invoices with expected route and findings")
    for case in cases:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"{DATASET_NAME}-{case['case_id']}",
            input={"case_id": case["case_id"], "claim":
case["claim"], "document_text": case["document_text"]},
            expected_output=case["expected"],
            metadata={"category": case["category"],
"what_it_tests": case["what_it_tests"]},
        )


def link_dataset_run(run_name: str, traces: list[tuple[str, str]],
metadata: dict) -> None:
    """Link each (case_id, trace_id) to its dataset item, as one
dataset run (experiment)."""
    if not _enabled():
        return
    from langfuse import get_client
    from langfuse.model import CreateDatasetRunItemRequest

    client = get_client()
    for case_id, trace_id in traces:
        client.api.dataset_run_items.create(
            request=CreateDatasetRunItemRequest(
                runName=run_name,
                datasetItemId=f"{DATASET_NAME}-{case_id}",
                traceId=trace_id,
                metadata=metadata,
            )
        )

