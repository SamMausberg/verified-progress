"""The admission preflight counts only the measured phase: a warm-up at 128 must not pass it."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/moonshot/check_admission.py'
START = 1_790_812_810  # profiling phase 2026-10-01 00:00:10 to 00:00:40 UTC
END = START + 30


def line(t: int, running: int) -> str:
    stamp = datetime.fromtimestamp(t, UTC).strftime('%Y-%m-%d %H:%M:%S')
    return f'[{stamp}] Decode batch, #running-req: {running}, #token: 1, gen throughput (token/s): 1.0\n'


def make(root: Path, warmup_peak: int, phase_peak: int) -> Path:
    run = root / 'arm' / '20261001-000000'
    (run / 'server').mkdir(parents=True)
    (run / 'sweep.json').write_text('{}')
    log = ''.join(line(t, warmup_peak) for t in range(START - 30, START - 5, 2))  # warm-up
    log += ''.join(line(t, phase_peak) for t in range(START, END, 2))
    (run / 'server/server.log').write_text(log)
    aiperf = run / 'r0/c128/aiperf'
    aiperf.mkdir(parents=True)
    records = [
        {'metadata': {'benchmark_phase': 'profiling', 'request_start_ns': START * 10**9,
                      'request_end_ns': END * 10**9}}
    ] * 128  # fmt: skip
    with gzip.open(aiperf / 'profile_export_raw.jsonl.gz', 'wt') as handle:
        handle.writelines(json.dumps(r) + '\n' for r in records)
    return root


def run(root: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), str(root), '--arms', 'arm']
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_phase_at_128_passes(tmp_path: Path) -> None:
    assert run(make(tmp_path, warmup_peak=128, phase_peak=128)).returncode == 0


def test_warmup_at_128_with_phase_at_127_fails(tmp_path: Path) -> None:
    result = run(make(tmp_path, warmup_peak=128, phase_peak=127))
    assert result.returncode == 1
    assert 'peak #running-req in the profiling phase 127' in result.stdout


def test_missing_arm_fails(tmp_path: Path) -> None:
    command = [sys.executable, str(SCRIPT), str(tmp_path), '--arms', 'absent']
    assert subprocess.run(command, capture_output=True, check=False).returncode == 1
