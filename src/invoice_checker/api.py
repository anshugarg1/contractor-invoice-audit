
"""HTTP API: submit an invoice, check its status, answer the expert review.

    uvicorn invoice_checker.api:create_app --factory --reload

Checks run in the background. The graph state lives in a SQLite checkpointer,
so an invoice waiting for review survives a server restart.
"""


from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from time import time
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Request, Response
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .baseline import OracleLLM
from .cases import CASES_DIR, CATALOGUE_PATH, PAID_REGISTER_PATH, ROOT, initial_state, load_cases
from .catalogue import load_catalogue, load_paid_register
from .graph import build_graph
from .llm import LangChainLLM
from .observability import CHECK_SECONDS, CHECKS, HTTP_REQUESTS, HTTP_SECONDS, record_run, setup_logging
from .tracing import flush, langfuse_callbacks


log = logging.getLogger(__name__)
CHECKPOINT_DB = ROOT / "runs" / "checkpoints.sqlite"


class InvoiceIn(BaseModel):
    """The same fields as a case file, so data/cases/*.json can be posted as is."""

    case_id: str
    claim: dict[str, Any]
    document_text: str
    quote_text: str | None = None



class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = ""


def _llm():
    if os.environ.get("API_MODE", "llm") == "baseline":  # no API key needed: ground truth + rules only
        return OracleLLM(load_cases(CASES_DIR))
    return LangChainLLM()


def _checkpointer() -> SqliteSaver:
    path = Path(os.environ.get("CHECKPOINT_DB", CHECKPOINT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Background tasks run in other threads; SqliteSaver locks the connection itself.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def create_app(llm=None, checkpointer=None) -> FastAPI:
    if llm is None:
        load_dotenv(ROOT / ".env")
        llm = _llm()
    graph = build_graph(
        llm,
        load_catalogue(CATALOGUE_PATH),
        load_paid_register(PAID_REGISTER_PATH),
        checkpointer=checkpointer if checkpointer is not None else _checkpointer(),
    )
    failed: dict[str, str] = {}  # thread_id -> error; in memory only, lost on restart
    app = FastAPI(title="Invoice checker")

    @app.middleware("http")
    async def measure(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        # The route template (/invoices/{thread_id}), not the real URL: one time series per endpoint, not per invoice.
        path = getattr(request.scope.get("route"), "path", "unmatched")
        HTTP_SECONDS.labels(request.method, path).observe(time.perf_counter() - start)
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        return response


    def thread(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def run(graph_input, thread_id: str) -> None:
        usage = UsageMetadataCallbackHandler()  # adds up tokens per model across every LLM call in this run
        config = thread(thread_id) | {
            "callbacks": langfuse_callbacks(),
            "metadata": {"langfuse_session_id": thread_id, "langfuse_tags": ["api"]},
        }
        start = time.perf_counter()
        try:
            graph.invoke(graph_input, config)
        except Exception as exc:  # LLM errors already go to review inside the graph; this catches bugs
            log.exception("check %s failed", thread_id)
            failed[thread_id] = f"{type(exc).__name__}: {exc}"
            CHECKS.labels("failed").inc()
        else:
            record_run(thread_id, graph.get_state(thread(thread_id)), usage.usage_metadata, time.perf_counter() - start)
        finally:
            flush()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": llm.model_name}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    @app.post("/invoices", status_code=202)
    def submit(invoice: InvoiceIn, tasks: BackgroundTasks) -> dict:
        thread_id = f"{invoice.case_id}-{uuid4().hex[:8]}"
        tasks.add_task(run, initial_state(invoice.model_dump()), thread_id)
        log.info("invoice submitted", extra={"fields": {"thread_id": thread_id}})
        return {"id": thread_id, "status": "queued"}

    @app.get("/invoices/{thread_id}")
    def status(thread_id: str) -> dict:
        if thread_id in failed:
            return {"id": thread_id, "status": "failed", "error": failed[thread_id]}
        snapshot = graph.get_state(thread(thread_id))
        if not snapshot.values:
            raise HTTPException(404, "unknown id, or the check hasn't started yet")
        if snapshot.interrupts:
            return {"id": thread_id, "status": "waiting_for_review", "review": snapshot.interrupts[0].value}
        if snapshot.next:
            return {"id": thread_id, "status": "running", "next": list(snapshot.next)}
        return {"id": thread_id, "status": "done", "audit": snapshot.values.get("audit")}

    @app.post("/invoices/{thread_id}/review", status_code=202)
    def review(thread_id: str, answer: ReviewIn, tasks: BackgroundTasks) -> dict:
        if not graph.get_state(thread(thread_id)).interrupts:
            raise HTTPException(409, "this invoice is not waiting for a review")
        tasks.add_task(run, Command(resume=answer.model_dump()), thread_id)
        log.info("review received", extra={"fields": {"thread_id": thread_id, "decision": answer.decision}})
        return {"id": thread_id, "status": "resuming"}

    return app
