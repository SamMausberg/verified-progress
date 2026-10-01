"""validate_p4_ab.py must fail closed: partial points, failed arms and a wrong order are FAILED."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/moonshot/validate_p4_ab.py'
ORDER = [
    ('plain+no_radix', 'r1'),
    ('plain+no_radix+exact_replay', 'r1'),
    ('plain+no_radix+exact_replay', 'r2'),
    ('plain+no_radix', 'r2'),
    ('plain+no_radix', 'r3'),
    ('plain+no_radix+exact_replay', 'r3'),
    ('plain+no_radix+exact_replay', 'r4'),
    ('plain+no_radix', 'r4'),
]


def point(rate: float, **overrides: Any) -> dict[str, Any]:
    base = {
        'concurrency': 128,
        'requests': 256,
        'completed': 256,
        'failed': 0,
        'osl_mismatch': 0,
        'aiperf_exit_code': 0,
        'prompts_as_expected': True,
        'y': rate * 0.6,
        'server_log': {'logged_gen_tps_full_batch': rate},
    }
    base.update(overrides)
    return base


def make_run(root: Path, order: list[tuple[str, str]] = ORDER, **overrides: Any) -> Path:
    """A synthetic A/B directory; `overrides` apply to the first exact-replay arm's point."""
    records = []
    first_exact = True
    for arm, pair in order:
        config = f'{arm}#{pair}'
        records.append({'config': config, 'status': 'exit 0', 'seconds': 1})
        run = root / config.replace('#', '_') / '20261001-000000'
        (run / 'server').mkdir(parents=True)
        exact = arm.endswith('exact_replay')
        extra = overrides if exact and first_exact else {}
        first_exact = first_exact and not exact
        rate = 10_000.0 * (1.2 if exact else 1.0) * (1.0 + 0.001 * int(pair[1]))
        (run / 'sweep.json').write_text(json.dumps({'points': [point(rate, **extra)]}))
        log = 'GDN decode: exact replay kernel, ring length 4\n' if exact else 'decode\n'
        (run / 'server/server.log').write_text(log)
    (root / 'lever_sweep_log.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return root


def run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True, check=False
    )


def test_complete_run_is_decided(tmp_path: Path) -> None:
    result = run(make_run(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['decision'] == 'supported'


def test_partial_point_fails(tmp_path: Path) -> None:
    # AIPerf stopped early: 128 exported rows, all successful.
    result = run(make_run(tmp_path, requests=128, completed=128))
    assert result.returncode == 1
    assert 'FAILED' in result.stdout


def test_aiperf_error_fails(tmp_path: Path) -> None:
    result = run(make_run(tmp_path, aiperf_exit_code=1))
    assert result.returncode == 1


def test_unexpected_prompts_fail(tmp_path: Path) -> None:
    result = run(make_run(tmp_path, prompts_as_expected=False))
    assert result.returncode == 1


def test_wrong_order_fails(tmp_path: Path) -> None:
    swapped = [ORDER[1], ORDER[0], *ORDER[2:]]
    result = run(make_run(tmp_path, order=swapped))
    assert result.returncode == 1
    assert 'order' in result.stdout


def test_failed_arm_fails(tmp_path: Path) -> None:
    root = make_run(tmp_path)
    log = root / 'lever_sweep_log.jsonl'
    log.write_text(log.read_text().replace('"exit 0"', '"exit 1"', 1))
    assert run(root).returncode == 1
