"""Tests for experiments/moonshot/admission_plateaus.py on synthetic scheduler logs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments' / 'moonshot' / 'admission_plateaus.py'
sys.path.insert(0, str(SCRIPT.parent))

from admission_plateaus import plateaus

HEADER = '[2026-10-01 11:52:36] max_total_num_tokens=655360, max_running_requests=128\n'


def prefill(n: int, tokens: int, running: int, queued: int) -> str:
    return (
        f'[2026-10-01 11:53:30] Prefill batch, #new-seq: {n}, #new-token: {tokens}, '
        f'#cached-token: 0, #running-req: {running}, #queue-req: {queued}, #pending-token: 0\n'
    )


def decode(running: int, queued: int) -> str:
    return f'[2026-10-01 11:53:31] Decode batch, #running-req: {running}, #queue-req: {queued}\n'


def test_continuation_in_the_last_pass_predicts_one_below_the_limit() -> None:
    # The pass at R = 119 starts a chunked request (5 in the pass, running rises by 4); the
    # last pass carries its tail plus three new requests and stops at 127 with one queued.
    log = HEADER + prefill(5, 8192, 119, 1) + prefill(4, 8179, 123, 1) + decode(127, 1)
    (row,) = plateaus(log)
    assert row['continuation'] == 1
    assert row['plateau_predicted'] == row['plateau_observed'] == 127


def test_no_continuation_predicts_the_limit() -> None:
    log = HEADER + prefill(4, 8192, 116, 8) + prefill(4, 8192, 120, 4) + decode(124, 4)
    (row,) = plateaus(log)
    assert row['continuation'] == 0
    assert row['plateau_predicted'] == 128
    assert row['plateau_observed'] == 124


def test_finishes_between_passes_are_not_inferable() -> None:
    # Running fell by more than one between passes: requests finished, so C is unknown.
    log = HEADER + prefill(2, 4096, 120, 3) + prefill(3, 6144, 110, 1) + decode(113, 1)
    (row,) = plateaus(log)
    assert row['continuation'] == ''
    assert row['plateau_predicted'] == ''


def test_decode_without_a_queue_is_not_a_plateau() -> None:
    log = HEADER + prefill(4, 8192, 120, 0) + decode(124, 0)
    assert plateaus(log) == []


def run_cli(tmp_path: Path, logs: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    """Lay the logs out as lever_sweep does (<run>/<arm>/<stamp>/server/server.log)."""
    run = tmp_path / 'run'
    for arm, text in logs.items():
        log = run / arm / '20261001-000000' / 'server' / 'server.log'
        log.parent.mkdir(parents=True)
        log.write_text(text)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run), *args], capture_output=True, text=True, check=False
    )


PLATEAU_127 = HEADER + prefill(5, 8192, 119, 1) + prefill(4, 8179, 123, 1) + decode(127, 1)


def test_cli_passes_when_every_log_plateaus_as_predicted(tmp_path: Path) -> None:
    result = run_cli(tmp_path, {'dense': PLATEAU_127, 'exact': PLATEAU_127})
    assert result.returncode == 0, result.stdout


def test_cli_fails_on_a_log_without_plateaus(tmp_path: Path) -> None:
    filled = HEADER + prefill(4, 8192, 120, 0) + decode(124, 0)
    result = run_cli(tmp_path, {'dense': PLATEAU_127, 'exact': filled})
    assert result.returncode == 1
    assert 'exact: 0 plateaus' in result.stdout


def test_cli_fails_below_the_required_count(tmp_path: Path) -> None:
    result = run_cli(tmp_path, {'dense': PLATEAU_127}, '--min-plateaus', '2')
    assert result.returncode == 1


def test_cli_fails_on_a_plateau_outside_its_scope(tmp_path: Path) -> None:
    finished = HEADER + prefill(2, 4096, 120, 3) + prefill(3, 6144, 110, 1) + decode(113, 1)
    result = run_cli(tmp_path, {'dense': PLATEAU_127, 'exact': finished})
    assert result.returncode == 1
    assert '1 with C not inferable' in result.stdout
