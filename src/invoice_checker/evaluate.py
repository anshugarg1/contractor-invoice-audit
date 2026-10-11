"""Run the graph over the evaluation cases and report metrics.

    python -m invoice_checker.evaluate --mode baseline
    python -m invoice_checker.evaluate --mode llm --repeats 5

Invoices that reach human review are left paused: the eval measures the routing
decision, which is what decides whether money is paid without a human.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from dotenv import load_dotenv

from .baseline import OracleLLM
from .cases import CASES_DIR, CATALOGUE_PATH, PAID_REGISTER_PATH, ROOT, initial_state, load_cases
from .catalogue import load_catalogue, load_paid_register
from .graph import build_graph
from .llm import LangChainLLM
from .tracing import flush, langfuse_callbacks, last_trace_id, link_dataset_run, score_trace, sync_dataset

RESULTS_DIR = ROOT / "results"
_AMOUNT_FIELDS = ("net_total", "vat_rate", "vat_amount", "gross_total")
_LINE_FIELDS = ("quantity", "unit_price", "line_total")


def _same_amount(a, b) -> bool:
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except Exception:
        return False


def _scores(case: dict, run: dict) -> dict[str, float]:
    """Per-case scores sent to Langfuse. 1.0 = yes, 0.0 = no."""
    expected = case["expected"]["route"]
    scores = {
        "route_correct": float(run["route"] == expected),
        "false_approval": float(expected == "human_review" and run["route"] == "auto_approve"),
    }
    if "extraction" in run:
        correct, total = run["extraction"]
        scores["extraction_accuracy"] = correct / total
    return scores


def compare_extraction(truth: dict, got: dict | None) -> tuple[int, int]:
    """(correct, total) over the fields the checks depend on. Amounts compare as numbers."""
    checks = [
        (truth["number"].strip().upper(), (got or {}).get("number", "").strip().upper()),
        (truth["date"], (got or {}).get("date")),
        (truth["claim_number"], (got or {}).get("claim_number")),
        (len(truth["line_items"]), len((got or {}).get("line_items", []))),
    ]
    correct = sum(a == b for a, b in checks)
    total = len(checks) + len(_AMOUNT_FIELDS) + len(truth["line_items"]) * (len(_LINE_FIELDS) + 1)
    if got is None:
        return correct, total
    correct += sum(_same_amount(truth[f], got[f]) for f in _AMOUNT_FIELDS)
    for t_line, g_line in zip(truth["line_items"], got["line_items"]):
        correct += (t_line["code"] or None) == (g_line["code"] or None)
        correct += sum(_same_amount(t_line[f], g_line[f]) for f in _LINE_FIELDS)
    return correct, total


def run_case(graph, case: dict, repeat: int, session_id: str) -> dict:
    # config = {"configurable": {"thread_id": f"eval-{case['case_id']}-{repeat}-{uuid4().hex[:6]}"}}
    callbacks = langfuse_callbacks()
    config = {
        "configurable": {"thread_id": f"eval-{case['case_id']}-{repeat}-{uuid4().hex[:6]}"},
        "callbacks": callbacks,
        "run_name": f"eval {case['case_id']}",
        "metadata": {"langfuse_session_id": session_id, "langfuse_tags": ["eval", case["case_id"]]},
    }

    started = time.perf_counter()
    try:
        graph.invoke(initial_state(case), config)
    except Exception as exc:  # one failing call must not kill the whole eval; it is reported
        run = {"route": "error", "error":f"{type(exc).__name__}: {exc}", "codes": [], "latency_s": 0.0}
        # return {"route": "error", "error": f"{type(exc).__name__}: {exc}", "codes": [], "latency_s": 0.0}
    else:
        latency = time.perf_counter() - started
        values = graph.get_state(config).values
        run = {
            "route": values.get("route"),
            "codes": sorted({f["code"] for f in values.get("findings", []) if f["severity"] != "info"}),
            "latency_s": round(latency, 2),
            "findings": values.get("findings", []),
            "judgments": values.get("judgments", []),
        }
        if case["truth"]["doc_type"] == "invoice":
            run["extraction"] = compare_extraction(case["truth"]["document"], values.get("invoice"))
    run["trace_id"] = last_trace_id(callbacks)
    score_trace(callbacks, _scores(case, run))
    return run


def _ratio(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "n/a"


def summarise(rows: list[dict], repeats: int) -> dict:
    runs = [(r["case"], run) for r in rows for run in r["runs"]]
    bad = [(c, run) for c, run in runs if c["expected"]["route"] == "human_review"]
    clean = [(c, run) for c, run in runs if c["expected"]["route"] == "auto_approve"]
    expected_codes = sum(len(c["expected"]["codes"]) for c, _ in runs)
    found_codes = sum(len(set(c["expected"]["codes"]) & set(run["codes"])) for c, run in runs)
    extraction = [run["extraction"] for _, run in runs if "extraction" in run]
    fully_correct = sum(k == n for k, n in extraction)
    latencies = [run["latency_s"] for _, run in runs if run["route"] != "error"]
    consistent = sum(len({run["route"] for run in r["runs"]}) == 1 for r in rows)
    same_codes = sum(len({tuple(run["codes"]) for run in r["runs"]}) == 1 for r in rows)
    return {
        "Correct route": _ratio(sum(run["route"] == c["expected"]["route"] for c, run in runs), len(runs)),
        "False approvals (bad invoice auto-approved)": _ratio(sum(run["route"] == "auto_approve" for _, run in bad), len(bad)),
        "Dark-processing rate (clean invoice auto-approved)": _ratio(sum(run["route"] == "auto_approve" for _, run in clean), len(clean)),
        "Expected findings detected": _ratio(found_codes, expected_codes),
        "Extraction: fields exactly right": _ratio(sum(k for k, _ in extraction), sum(n for _, n in extraction)),
        "Extraction: documents fully right": _ratio(fully_correct, len(extraction)),
        f"Same route in all {repeats} runs": _ratio(consistent, len(rows)) if repeats > 1 else "n/a (1 run)",
        f"Same findings in all {repeats} runs": _ratio(same_codes, len(rows)) if repeats > 1 else "n/a (1 run)",
        "Errors": str(sum(run["route"] == "error" for _, run in runs)),
        "Latency per case (mean / max)": (
            f"{statistics.mean(latencies):.1f} s / {max(latencies):.1f} s" if latencies else "n/a"
        ),
    }


def to_markdown(meta: dict, summary: dict, rows: list[dict]) -> str:
    lines = [
        f"# Eval: {meta['mode']}",
        "",
        f"- Model: {meta['model']}",
        f"- Cases: {meta['cases']}, runs per case: {meta['repeats']}",
        f"- Prompt version: {meta['prompt_version']}, graph version: {meta['graph_version']}",
        f"- Run at: {meta['run_at']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| Case | Category | Expected | Got | Missing findings | Extra findings |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        case, runs = r["case"], r["runs"]
        routes = [run["route"] for run in runs]
        got = ", ".join(f"{route} x{routes.count(route)}" for route in dict.fromkeys(routes))
        found = set().union(*(run["codes"] for run in runs))
        missing = sorted(set(case["expected"]["codes"]) - found)
        extra = sorted(found - set(case["expected"]["codes"]))
        mark = "" if all(route == case["expected"]["route"] for route in routes) else " **(wrong)**"
        lines.append(
            f"| {case['case_id']} | {case['category']} | {case['expected']['route']} | {got}{mark} | "
            f"{', '.join(missing) or '-'} | {', '.join(extra) or '-'} |"
        )
    errors = [(r["case"]["case_id"], run["error"]) for r in rows for run in r["runs"] if run["route"] == "error"]
    if errors:
        lines += ["", "## Errors", "", *[f"- {cid}: {err}" for cid, err in errors]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["llm", "baseline"], default="llm")
    parser.add_argument("--repeats", type=int, default=1, help="runs per case, to measure consistency")
    parser.add_argument("--cases", default="*.json", help="glob inside data/cases, e.g. 'C1*.json'")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cases = load_cases(CASES_DIR, args.cases)
    llm = OracleLLM(cases) if args.mode == "baseline" else LangChainLLM()
    graph = build_graph(llm, load_catalogue(CATALOGUE_PATH), load_paid_register(PAID_REGISTER_PATH))
    session_id = f"eval-{args.mode}-{datetime.now():%Y%m%d-%H%M%S}"
    sync_dataset(cases)
    rows = []
    for case in cases:
        runs = [run_case(graph, case, i, session_id) for i in range(args.repeats)]
        rows.append({"case": case, "runs": runs})
        print(f"{case['case_id']:<28} expected {case['expected']['route']:<13} got {[run['route'] for run in runs]}")

    from .graph import GRAPH_VERSION
    from .prompts import PROMPT_VERSION

    meta = {
        "mode": args.mode, "model": llm.model_name, "cases": len(cases), "repeats": args.repeats,
        "prompt_version": PROMPT_VERSION, "graph_version": GRAPH_VERSION,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    link_dataset_run(
          f"{session_id} {llm.model_name}",
          [(r["case"]["case_id"], run["trace_id"]) for r in rows for
  run in r["runs"] if run.get("trace_id")],
          meta,
      )
    summary = summarise(rows, args.repeats)
    report = to_markdown(meta, summary, rows)
    print("\n" + report)

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"eval_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
    raw = {"meta": meta, "summary": summary, "cases": [
        {"case_id": r["case"]["case_id"], "expected": r["case"]["expected"], "runs": r["runs"]} for r in rows
    ]}
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results/{stem}.md and .json")
    flush()  # send traces to Langfuse if enabled
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
