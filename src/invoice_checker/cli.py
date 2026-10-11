"""Check one case end to end, streaming progress, with a real human-review pause.

    python -m invoice_checker.cli data/cases/C06_overpriced_labour.json
    python -m invoice_checker.cli data/cases/C06_overpriced_labour.json --mode baseline --decision reject
"""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.types import Command

from .baseline import OracleLLM
from .cases import CASES_DIR, CATALOGUE_PATH, PAID_REGISTER_PATH, ROOT, initial_state, load_case, load_cases
from .catalogue import load_catalogue, load_paid_register
from .graph import build_graph
from .llm import LangChainLLM
from .tracing import flush, langfuse_callbacks

RUNS_DIR = ROOT / "runs"


def _stream(graph, graph_input, config) -> None:
    # stream_mode="updates" yields one {node: update} dict per finished node.
    for update in graph.stream(graph_input, config, stream_mode="updates"):
        for node, value in update.items():
            if node == "__interrupt__":
                print("  -> paused: waiting for an expert")
            else:
                n = len((value or {}).get("findings", []))
                print(f"  -> {node}" + (f" ({n} finding{'s' * (n != 1)})" if n else ""))


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="path to a case JSON file")
    parser.add_argument("--mode", choices=["llm", "baseline"], default="llm")
    parser.add_argument("--decision", choices=["approve", "reject"], help="answer the review without prompting")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    case = load_case(args.case)
    llm = LangChainLLM() if args.mode == "llm" else OracleLLM(load_cases(CASES_DIR))
    graph = build_graph(llm, load_catalogue(CATALOGUE_PATH), load_paid_register(PAID_REGISTER_PATH))
    # config = {"configurable": {"thread_id": f"{case['case_id']}-{uuid4().hex[:8]}"}}

    thread_id = f"{case['case_id']}-{uuid4().hex[:8]}"
    config = {
          "configurable": {"thread_id": thread_id},
          "callbacks": langfuse_callbacks(),
          "run_name": f"check {case['case_id']}",
          "metadata": {
              "langfuse_session_id": thread_id, 
              "langfuse_tags":["cli", args.mode, case["case_id"]]},
      }
    
    print(f"Case {case['case_id']}: {case['what_it_tests']}")
    print(f"Model: {llm.model_name}\n")
    _stream(graph, initial_state(case), config)

    snapshot = graph.get_state(config)
    if snapshot.next:  # paused inside human_review
        payload = snapshot.interrupts[0].value
        print(f"\n{payload['question']}")
        for f in payload["findings"]:
            print(f"  [{f['severity']}] {f['code']}: {f['message']}")
        decision = args.decision or input("\nDecision [approve/reject]: ").strip().lower()
        note = "" if args.decision else input("Note (optional): ").strip()
        print()
        _stream(graph, Command(resume={"decision": decision, "note": note}), config)

    audit = graph.get_state(config).values["audit"]
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{audit['thread_id']}.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDecision: {audit['decision']}   (expected route: {case['expected']['route']})")
    print(f"Audit record: {out.relative_to(ROOT)}")
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
