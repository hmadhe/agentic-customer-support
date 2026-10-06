"""Run the real-model tests several times and report each test's pass rate.

A model can pass a test 4 times out of 5. A single run turns that into a misleading "pass" or "fail";
repeated runs show it as a rate.

    python -m scripts.pass_rates --runs 3                      # all real-model tests
    python -m scripts.pass_rates --runs 5 tests/test_graph_llm.py
"""

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def outcome(testcase: ET.Element) -> str:
    skipped = testcase.find("skipped")
    if skipped is not None:
        return "xfail" if "xfail" in (skipped.get("type", "") + skipped.get("message", "")) else "skipped"
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return "failed"
    return "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("paths", nargs="*", help="test files or folders (default: all tests)")
    args = parser.parse_args()

    results: dict[str, list[str]] = defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        for run in range(1, args.runs + 1):
            report = Path(tmp) / f"run{run}.xml"
            print(f"run {run}/{args.runs}...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pytest", "-m", "llm", "-q", "-p", "no:warnings", f"--junitxml={report}", *args.paths],
                capture_output=True,
            )
            for testcase in ET.parse(report).getroot().iter("testcase"):
                results[f"{testcase.get('classname')}::{testcase.get('name')}"].append(outcome(testcase))

    flaky, failing = 0, 0
    print(f"\n{'pass rate':>9}  test")
    for name, outcomes in sorted(results.items(), key=lambda item: item[1].count("passed") / len(item[1])):
        if all(o == "xfail" for o in outcomes):
            continue  # known limitations, reported by pytest itself
        passed = outcomes.count("passed")
        if passed < len(outcomes):
            flaky += passed > 0
            failing += passed == 0
            print(f"{passed:>4}/{len(outcomes):<4}  {name}")
    always = sum(1 for o in results.values() if o.count("passed") == len(o))
    print(f"\n{always} tests passed every run, {flaky} passed only sometimes, {failing} never passed "
          f"({args.runs} runs, known xfails not listed).")


if __name__ == "__main__":
    main()
