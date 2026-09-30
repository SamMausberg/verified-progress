#!/usr/bin/env python3
"""Rerun supplied CPU evidence; record but never mask unavailable Lean checking."""
from __future__ import annotations
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def invoke(command: list[str], log_name: str) -> dict:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, timeout=45, check=False)
        output = result.stdout + "\n--- stderr ---\n" + result.stderr
        code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        code = -1
    (EVIDENCE / log_name).write_text(output)
    return {"command": command, "exit_code": code, "log": f"evidence/{log_name}"}


def main() -> int:
    EVIDENCE.mkdir(exist_ok=True)
    jobs = [
        ([sys.executable, "tests/test_decisions.py"], "decision_tests.log"),
        ([sys.executable, "tests/test_races.py"], "race_tests.log"),
        ([sys.executable, "tests/test_benchmark_client.py"], "benchmark_client_tests.log"),
        ([sys.executable, "tests/test_v2.py"], "v2_tests.log"),
        ([sys.executable, "src/v1_reference.py", "--output", "evidence/v1_tests.json"], "v1_tests.log"),
        ([sys.executable, "experiments/synthetic_drift.py"], "synthetic_drift.log"),
    ]
    results = [invoke(command, log) for command, log in jobs]
    formal = invoke(["bash", "scripts/check_lean.sh"], "lean_attempt.log")
    ok = all(r["exit_code"] == 0 for r in results)
    report = {"python": platform.python_version(), "cpu_checks_pass": ok,
              "cpu_runs": results, "lean_check": formal,
              "lean_status": "compiled" if formal["exit_code"] == 0 else "not_validated",
              "gpu_status": "not_run", "real_model_status": "not_run"}
    (EVIDENCE / "validation_run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
