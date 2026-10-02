"""CPU tests of the lossy study's partial GSM8K comparison (experiments/lossy/gsm8k_partial.py)."""

from __future__ import annotations

import pytest

from experiments.lossy import gsm8k_partial


def rows(correct: list[bool], truncated: tuple[int, ...] = ()) -> dict[str, dict[str, object]]:
    return {
        f'p{i}': {
            'id': f'p{i}',
            'correct': str(c),
            'finish_reason': 'length' if i in truncated else 'stop',
        }
        for i, c in enumerate(correct)
    }


def test_against_pairs_finished_problems_and_bounds_the_rest() -> None:
    # Reference: 10 problems, 8 correct (p8 and p9 wrong). Run: p0-p7 finished, p0 wrong.
    reference = rows([True] * 8 + [False, False], truncated=(9,))
    finished = rows([False] + [True] * 7)
    result = gsm8k_partial.against(finished, reference)
    assert result['finished_problems'] == 8
    assert result['correct_only_reference'] == 1
    assert result['correct_only_run'] == 0
    assert result['delta_pt_on_finished'] == pytest.approx(-12.5)
    assert result['reference_on_unfinished'] == {'correct': 0, 'truncated': 1}
    # 7 correct of 10 at worst, 9 of 10 at best, against the reference's 8 of 10.
    assert result['full_split_delta_bounds_pt'] == pytest.approx([-10.0, 10.0])
    assert result['declared_rule_on_bounds'].startswith('undetermined')


def test_bounds_verdict() -> None:
    assert gsm8k_partial.bounds_verdict(-0.5, 0.3) == 'met at both bounds'
    assert gsm8k_partial.bounds_verdict(-1.8, -1.2) == 'missed at both bounds'
    assert gsm8k_partial.bounds_verdict(-1.5, -0.4).startswith('undetermined')


def test_against_refuses_unknown_problems() -> None:
    with pytest.raises(ValueError, match='does not have'):
        gsm8k_partial.against(rows([True, True, True]), rows([True, True]))
