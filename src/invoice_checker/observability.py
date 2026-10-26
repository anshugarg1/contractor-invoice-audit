"""Logs and metrics for the API.
Logs answer "what happened to invoice X?" (one JSON line per event, searchable by thread_id).
Metrics answer "how is the system doing?" (counts and latencies for dashboards and alerts).
Traces of single LLM calls stay in Langfuse.
"""

from __future__ import annotations

import json
import logging

from prometheus_client import Counter, Histogram

from .costs import cost_usd

log = logging.getLogger(__name__)

# Defined once at import: Prometheus refuses to register the same metric name twice.
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_SECONDS = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])
CHECKS = Counter("invoice_checks_total", "Finished graph runs by outcome", ["outcome"])
CHECK_SECONDS = Histogram(
    "invoice_check_duration_seconds", "Time for one graph run", ["phase"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),  # LLM checks take ~20 s, local models minutes
)
FINDINGS = Counter("invoice_findings_total", "Findings by code and severity", ["code", "severity"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used", ["model", "kind"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM cost in US dollars")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so a log platform (Loki, Elastic, Azure Monitor) can filter by field."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))  # passed as extra={"fields": {...}}
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


def record_run(thread_id: str, snapshot, usage: dict[str, dict], seconds: float) -> None:
    """Count one finished graph run in the metrics and log it as one line."""
    values = snapshot.values
    outcome = "waiting_for_review" if snapshot.interrupts else values.get("decision", "unknown")
    findings = values.get("findings", [])
    if outcome in ("waiting_for_review", "auto_approved"):  # first run only; a resume has the same findings
        for f in findings:
            FINDINGS.labels(f["code"], f["severity"]).inc()
    CHECKS.labels(outcome).inc()
    for model, u in usage.items():
        LLM_TOKENS.labels(model, "input").inc(u.get("input_tokens", 0))
        LLM_TOKENS.labels(model, "output").inc(u.get("output_tokens", 0))
    if cost := cost_usd(usage):
        LLM_COST.inc(cost)
    log.info("check finished", extra={"fields": {
        "thread_id": thread_id,
        "outcome": outcome,
        "seconds": round(seconds, 2),
        "findings": [f["code"] for f in findings],
        "tokens": sum(u.get("total_tokens", 0) for u in usage.values()),
    }})

