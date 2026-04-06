"""E2E test runner for Pile — runs questions sequentially through the workflow.

Usage:
    cd /Users/tantran/works/gg
    uv run python tests/e2e/runner.py [--start N] [--end N] [--timeout 120] [--no-cache]
    uv run python tests/e2e/runner.py --sample 20 --seed 42

Outputs:
    tests/e2e/results/run_<timestamp>.jsonl   — one JSON line per question
    tests/e2e/results/run_<timestamp>.summary — human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

RESULTS_DIR = Path(__file__).parent / "results"
QUESTIONS_FILE = Path(__file__).parent / "questions.json"


@dataclass
class TestResult:
    id: int
    question: str
    expected_route: str
    actual_route: str
    category: str
    lang: str
    status: str  # "ok", "error", "timeout", "empty"
    response_text: str
    response_length: int
    route_correct: bool
    duration_s: float  # total wall-clock seconds
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""
    cached: bool = False


def load_questions(start: int = 1, end: int = 100) -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        all_q = json.load(f)
    return [q for q in all_q if start <= q["id"] <= end]


async def run_single_query(
    workflow, tracker, question: dict, timeout: float
) -> TestResult:
    """Run a single question through the workflow, capture output + timing."""
    from pile.cache import get_cached

    qid = question["id"]
    query = question["q"]
    expected = question["route"]

    cached_hit = get_cached(query) is not None

    # Run the workflow with timeout
    full_text = ""
    status = "ok"
    error_msg = ""
    actual_route = ""
    start_time = time.monotonic()

    try:
        async with asyncio.timeout(timeout):
            async for event in workflow.run(query, stream=True):
                if event.type == "output":
                    if hasattr(event.data, "text") and event.data.text:
                        full_text += event.data.text
                elif event.type == "executor_invoked":
                    # Capture the first agent name as actual route
                    if not actual_route:
                        name = event.data if isinstance(event.data, str) else str(event.data)
                        actual_route = name.replace("Agent", "").lower().strip()
    except TimeoutError:
        status = "timeout"
        error_msg = f"Timed out after {timeout}s"
        if hasattr(workflow, "_reset_running_flag"):
            workflow._reset_running_flag()
    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        if hasattr(workflow, "_reset_running_flag"):
            workflow._reset_running_flag()

    elapsed = round(time.monotonic() - start_time, 2)

    if status == "ok" and not full_text.strip():
        status = "empty"

    # Map agent names back to route keys for comparison
    _NAME_TO_KEY = {
        "triage": "triage", "jiraquery": "jira_query", "jirawrite": "jira_write",
        "board": "board", "sprint": "sprint", "epic": "epic",
        "scrum": "scrum", "git": "git",
    }
    route_key = _NAME_TO_KEY.get(actual_route, actual_route)

    # Collect tool calls — get from tracker (what hasn't been drained by workflow)
    calls = tracker.drain()
    tool_call_records = [
        {"name": c.name, "args": c.arguments, "duration_ms": c.duration_ms}
        for c in calls
    ]

    return TestResult(
        id=qid,
        question=query,
        expected_route=expected,
        actual_route=route_key,
        category=question["cat"],
        lang=question["lang"],
        status=status,
        response_text=full_text[:2000],
        response_length=len(full_text),
        route_correct=(route_key == expected),
        duration_s=elapsed,
        tool_calls=tool_call_records,
        error=error_msg,
        cached=cached_hit,
    )


def print_result_line(r: TestResult):
    """Print a compact one-line result."""
    route_mark = "✓" if r.route_correct else "✗"
    status_mark = {"ok": "✓", "error": "✗", "timeout": "⏱", "empty": "∅"}[r.status]
    tools = ",".join(c["name"] for c in r.tool_calls) or "-"
    print(
        f"  [{r.id:3d}] {status_mark} route:{route_mark} "
        f"{r.actual_route:12s} {r.duration_s:6.1f}s "
        f"len={r.response_length:5d} tools=[{tools}] "
        f"{'CACHED' if r.cached else ''}"
    )


def write_summary(results: list[TestResult], filepath: Path, elapsed_total: float):
    """Write human-readable summary."""
    total = len(results)
    ok = sum(1 for r in results if r.status == "ok")
    errors = sum(1 for r in results if r.status == "error")
    timeouts = sum(1 for r in results if r.status == "timeout")
    empty = sum(1 for r in results if r.status == "empty")
    route_ok = sum(1 for r in results if r.route_correct)
    cached = sum(1 for r in results if r.cached)

    durations = [r.duration_s for r in results if r.status == "ok"]
    avg_dur = sum(durations) / len(durations) if durations else 0
    max_dur = max(durations) if durations else 0
    min_dur = min(durations) if durations else 0

    routes: dict[str, list[TestResult]] = {}
    for r in results:
        routes.setdefault(r.actual_route, []).append(r)

    cats: dict[str, list[TestResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    lines = [
        "=" * 70,
        f"PILE E2E TEST REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Total questions: {total}",
        f"Total time:      {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)",
        "",
        "--- Status ---",
        f"  OK:      {ok:3d} ({ok/total*100:.0f}%)",
        f"  Error:   {errors:3d} ({errors/total*100:.0f}%)",
        f"  Timeout: {timeouts:3d} ({timeouts/total*100:.0f}%)",
        f"  Empty:   {empty:3d} ({empty/total*100:.0f}%)",
        f"  Cached:  {cached:3d}",
        "",
        "--- Routing ---",
        f"  Correct: {route_ok:3d}/{total} ({route_ok/total*100:.0f}%)",
        "",
        "--- Timing (OK only) ---",
        f"  Avg: {avg_dur:.1f}s | Min: {min_dur:.1f}s | Max: {max_dur:.1f}s",
        "",
        "--- Per Agent ---",
    ]
    for agent, rs in sorted(routes.items()):
        ok_rs = [r for r in rs if r.status == "ok"]
        avg = sum(r.duration_s for r in ok_rs) / len(ok_rs) if ok_rs else 0
        lines.append(
            f"  {agent:14s} total={len(rs):2d} ok={len(ok_rs):2d} "
            f"avg={avg:.1f}s"
        )

    lines += ["", "--- Per Category ---"]
    for cat, rs in sorted(cats.items()):
        ok_rs = [r for r in rs if r.status == "ok"]
        avg = sum(r.duration_s for r in ok_rs) / len(ok_rs) if ok_rs else 0
        lines.append(
            f"  {cat:18s} total={len(rs):2d} ok={len(ok_rs):2d} "
            f"avg={avg:.1f}s"
        )

    failures = [r for r in results if r.status != "ok"]
    if failures:
        lines += ["", "--- Failures ---"]
        for r in failures:
            lines.append(
                f"  [{r.id:3d}] {r.status:7s} route={r.actual_route} "
                f"dur={r.duration_s:.1f}s err={r.error[:80]}"
            )
            lines.append(f"        Q: {r.question}")

    slow = [r for r in results if r.status == "ok" and r.duration_s > 30]
    if slow:
        lines += ["", "--- Slow Queries (>30s) ---"]
        for r in sorted(slow, key=lambda x: -x.duration_s):
            lines.append(
                f"  [{r.id:3d}] {r.duration_s:.1f}s route={r.actual_route} "
                f"Q: {r.question[:60]}"
            )

    misroutes = [r for r in results if not r.route_correct]
    if misroutes:
        lines += ["", "--- Route Mismatches ---"]
        for r in misroutes:
            lines.append(
                f"  [{r.id:3d}] expected={r.expected_route} "
                f"actual={r.actual_route} Q: {r.question[:60]}"
            )

    lines.append("")
    text = "\n".join(lines)
    filepath.write_text(text)
    print(text)


async def main():
    parser = argparse.ArgumentParser(description="Pile E2E Test Runner")
    parser.add_argument("--start", type=int, default=1, help="Start question ID")
    parser.add_argument("--end", type=int, default=100, help="End question ID")
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per question (seconds)")
    parser.add_argument("--no-cache", action="store_true", help="Clear cache before each question")
    parser.add_argument("--sample", type=int, default=0, help="Random sample N questions (0=all in range)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"run_{ts}.log"
    jsonl_file = RESULTS_DIR / f"run_{ts}.jsonl"
    summary_file = RESULTS_DIR / f"run_{ts}.summary"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    print(f"Loading questions {args.start}-{args.end}...")
    questions = load_questions(args.start, args.end)
    if args.sample and args.sample < len(questions):
        import random
        random.seed(args.seed)
        questions = random.sample(questions, args.sample)
        print(f"Sampled {len(questions)} questions (seed={args.seed})")
    else:
        print(f"Loaded {len(questions)} questions")

    print("Initializing workflow...")
    from pile.workflows.interactive import create_workflow
    from pile.cache import clear_cache

    workflow, tracker = create_workflow()
    print(f"Workflow ready. Timeout={args.timeout}s per question.\n")
    print(f"Results: {jsonl_file}")
    print(f"Logs:    {log_file}")
    print("-" * 70)

    results: list[TestResult] = []
    total_start = time.monotonic()

    for i, q in enumerate(questions):
        if args.no_cache:
            clear_cache()

        print(f"\n[{i+1}/{len(questions)}] Q{q['id']}: {q['q'][:60]}...")

        result = await run_single_query(workflow, tracker, q, args.timeout)
        results.append(result)
        print_result_line(result)

        # Write JSONL incrementally (crash-safe)
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    total_elapsed = time.monotonic() - total_start

    print("\n" + "=" * 70)
    write_summary(results, summary_file, total_elapsed)
    print("\nFiles saved:")
    print(f"  JSONL:   {jsonl_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Log:     {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
