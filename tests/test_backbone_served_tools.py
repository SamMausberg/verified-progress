"""Tests of the backbone workstream's served-evidence tools (CPU only)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'backbone'))

from bitwise_runs import compare
from insitu_gemm import complete_replays


def test_complete_replays_drops_cut_and_boundary_replays() -> None:
    # Replay 1 was cut by the window's start, replay 6 by its end.
    order = [1, 2, 3, 4, 5, 6]
    sizes = {1: 3, 2: 10, 3: 10, 4: 10, 5: 10, 6: 7}
    kept, full = complete_replays(order, sizes)
    assert full == 10
    # Full-count replays are 2-5; the first and last of them are dropped too.
    assert kept == [3, 4]


def test_complete_replays_counts_a_missing_replay_as_empty() -> None:
    kept, full = complete_replays([1, 2, 3, 4], {2: 5, 3: 5, 4: 5})
    assert full == 5
    assert kept == [3]


def rec(ids, tops):
    return {'output_ids': ids, 'top_logprobs': tops}


def test_bitwise_counts_tokens_and_full_logprob_arrays() -> None:
    a = {
        'p0': rec([1, 2], [[[-0.1, 1], [-9.0, 7]], [[-0.2, 2], [-8.0, 5]]]),
        'p1': rec([3], [[[-0.3, 3], [-9.5, 4]]]),
        'p2': rec([4], [[[-0.4, 4]]]),
    }
    b = {
        'p0': rec([1, 2], [[[-0.1, 1], [-9.0, 7]], [[-0.2, 2], [-8.0, 5]]]),
        # Same tokens; a tail entry below -4 nats differs (drift ignores it).
        'p1': rec([3], [[[-0.3, 3], [-9.25, 4]]]),
        'p2': rec([5], [[[-0.4, 5]]]),
    }
    s = compare(a, b)
    assert s['prompts'] == 3
    assert s['tokens_equal'] == 2
    assert s['top_logprobs_equal'] == 1
    assert s['bitwise_equal'] == 1
    assert s['differing_ids_first_10'] == ['p1', 'p2']


def test_bitwise_refuses_different_prompt_sets() -> None:
    with pytest.raises(ValueError, match='prompt sets differ'):
        compare({'p0': rec([1], [[]])}, {'p1': rec([1], [[]])})


def test_bitwise_refuses_runs_without_logprobs() -> None:
    with pytest.raises(ValueError, match='no top_logprobs'):
        compare({'p0': {'output_ids': [1]}}, {'p0': rec([1], [[]])})


def test_bitwise_cli_fails_on_a_missing_run(tmp_path: Path) -> None:
    import subprocess

    pairs = tmp_path / 'pairs.json'
    pairs.write_text(json.dumps([['x', 'a/c1', 'b/c1']]))
    script = Path(__file__).resolve().parents[1] / 'experiments' / 'backbone' / 'bitwise_runs.py'
    run = subprocess.run(
        [sys.executable, str(script), '--runs', str(tmp_path), '--pairs', str(pairs),
         '--out', str(tmp_path / 'out.json')],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert run.returncode != 0
    assert 'missing run' in run.stderr
    assert not (tmp_path / 'out.json').exists()
