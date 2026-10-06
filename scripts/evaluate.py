"""Evaluate the assistant on the golden dataset and write a report.

    python -m scripts.evaluate                 # all cases, about 20-30 minutes
    python -m scripts.evaluate --only P01,O03  # selected cases, for debugging

Every run uses a fresh, temporary setup (seeded orders, newly ingested policies, empty tickets, in-memory
conversations), so results don't depend on, or change, the local databases.
"""

import argparse
import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.agent import current_turn
from app.config import PROJECT_ROOT, get_settings
from app.graph import build_graph
from app.ingest import ingest
from app.memory import make_checkpointer
from app.orders import seed_db
from app.retriever import retrieve
from app.vector_store import get_vector_store

EVAL_DIR = PROJECT_ROOT / "eval"
Route = Literal["rag", "agent", "escalate", "respond"]


class EvalCase(BaseModel):
    """One golden case. Only the expectations that are set are checked."""

    id: str
    category: str
    turns: list[str]  # the customer's messages; the last turn's result is checked
    route: Route | None = None
    escalation_reason: str | None = None
    source: str | None = None  # a policy document that must be cited
    tools: list[str] = []  # tools that must be called in the last turn
    no_tools: bool = False  # no tool may be called in the last turn
    must_contain: list[str] = []  # each item must appear in the reply; "a|b" means a or b
    must_not_contain: list[str] = []  # none of these may appear (known hallucinations)


def load_cases(path: Path = EVAL_DIR / "golden_set.json") -> list[EvalCase]:
    return [EvalCase(**case) for case in json.loads(path.read_text(encoding="utf-8"))]


def route_of(result: dict) -> Route:
    if result.get("escalation_reason"):
        return "escalate"
    intent = result["classification"].intent
    return {"policy_question": "rag", "order_issue": "agent"}.get(intent, "respond")


def tools_called(result: dict) -> list[str]:
    return [call["name"] for message in current_turn(result["messages"]) for call in getattr(message, "tool_calls", [])]


def check_case(case: EvalCase, result: dict) -> dict[str, bool]:
    """Run every expectation the case defines. Text checks ignore upper/lower case."""
    reply = (result.get("response") or "").lower()
    checks = {}
    if case.route:
        checks["route"] = route_of(result) == case.route
    if case.escalation_reason:
        checks["escalation_reason"] = result.get("escalation_reason") == case.escalation_reason
    if case.source:
        policy_answer = result.get("policy_answer")
        checks["source"] = policy_answer is not None and case.source in policy_answer.sources
    if case.tools:
        checks["tools"] = all(tool in tools_called(result) for tool in case.tools)
    if case.no_tools:
        checks["no_tools"] = not tools_called(result)
    if case.must_contain:
        checks["facts"] = all(
            any(option.lower() in reply for option in group.split("|")) for group in case.must_contain
        )
    if case.must_not_contain:
        checks["no_hallucination"] = not any(phrase.lower() in reply for phrase in case.must_not_contain)
    return checks


def build_eval_graph(tmp: Path):
    """The real graph and models, on fresh temporary data."""
    seed_db(tmp / "orders.db")
    store = get_vector_store(persist_directory=tmp / "chroma")
    ingest(store)
    return build_graph(
        retriever=lambda question: retrieve(question, vector_store=store),
        db_path=tmp / "orders.db", tickets_db_path=tmp / "tickets.db", checkpointer=make_checkpointer(),
    )


def run_case(graph, case: EvalCase) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    start = time.perf_counter()
    try:
        for message in case.turns:
            result = graph.invoke({"message": message}, config)
    except Exception as error:  # a crash is recorded as a failed case, not a crashed evaluation
        return {"id": case.id, "category": case.category, "passed": False, "checks": {"no_crash": False},
                "reply": f"{type(error).__name__}: {error}", "seconds": round(time.perf_counter() - start, 1)}
    checks = check_case(case, result)
    return {
        "id": case.id,
        "category": case.category,
        "passed": all(checks.values()),
        "checks": checks,
        "route": route_of(result),
        "escalation_reason": result.get("escalation_reason"),
        "tools": tools_called(result),
        "reply": result["response"],
        "seconds": round(time.perf_counter() - start, 1),
    }


def rate(results: list[dict]) -> str:
    passed = sum(r["passed"] for r in results)
    return f"{passed}/{len(results)} ({passed / len(results):.0%})"


def write_report(results: list[dict], minutes: float) -> str:
    settings = get_settings()
    lines = [
        "# Evaluation report",
        "",
        f"Generated by `python -m scripts.evaluate` on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Chat model `{settings.ollama_model}`, embeddings `{settings.ollama_embedding_model}`. "
        f"{len(results)} cases in {minutes:.0f} minutes.",
        "",
        f"**Overall: {rate(results)} cases passed every check.**",
        "",
        "## By category",
        "",
        "| Category | Passed |",
        "|---|---|",
    ]
    for category in dict.fromkeys(r["category"] for r in results):
        lines.append(f"| {category} | {rate([r for r in results if r['category'] == category])} |")

    lines += ["", "## By check", "", "| Check | Passed | Cases where it applies |", "|---|---|---|"]
    for check in ["route", "escalation_reason", "source", "tools", "no_tools", "facts", "no_hallucination", "no_crash"]:
        applicable = [r for r in results if check in r["checks"]]
        if applicable:
            passed = sum(r["checks"][check] for r in applicable)
            lines.append(f"| {check} | {passed}/{len(applicable)} ({passed / len(applicable):.0%}) | {len(applicable)} |")

    failures = [r for r in results if not r["passed"]]
    lines += ["", f"## Failures ({len(failures)})", "", "| Case | Failed checks | Route | Tools | Reply |", "|---|---|---|---|---|"]
    for r in failures:
        failed = ", ".join(name for name, ok in r["checks"].items() if not ok)
        reply = r["reply"].replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {r['id']} | {failed} | {r.get('route', '-')} | {', '.join(r.get('tools', [])) or '-'} | {reply} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated case ids, e.g. P01,O03 (no report is written)")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        cases = [case for case in cases if case.id in args.only.split(",")]

    start = time.perf_counter()
    results = []
    # ignore_cleanup_errors: on Windows Chroma keeps its index file open, so deleting the folder failed, and
    # it crashed the run at the very end, before the report was written.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        graph = build_eval_graph(Path(tmp))
        for number, case in enumerate(cases, 1):
            result = run_case(graph, case)
            results.append(result)
            print(f"[{number}/{len(cases)}] {case.id} {'PASS' if result['passed'] else 'FAIL'} "
                  f"{result['checks']} ({result['seconds']}s)", flush=True)

    minutes = (time.perf_counter() - start) / 60
    print(f"\nOverall: {rate(results)} in {minutes:.0f} minutes")
    if not args.only:
        (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        (EVAL_DIR / "report.md").write_text(write_report(results, minutes), encoding="utf-8")
        print(f"Wrote {EVAL_DIR / 'report.md'} and {EVAL_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
