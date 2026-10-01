"""Tests for experiments/moonshot/admission_plateaus.py on synthetic scheduler logs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'moonshot'))

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
