"""CPU tests of the BF16-path readouts (experiments/bf16_paths): the per-target summary and
the selection-free rate comparison with its declared decision rule."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.bf16_paths import rates, summary


def entry(lps: dict[int, float], token: int) -> dict[str, Any]:
    top = max(lps, key=lambda t: lps[t])
    return {
        'top1': top,
        'top1_logprob': lps[top],
        'tracked': {str(t): v for t, v in lps.items()},
        'token': token,
        'token_logprob': lps[token],
    }


def test_readout_measures_against_fp32_and_scores_the_path_top1() -> None:
    fp32 = {'9': entry({1: -0.01, 2: -5.0}, 1), '10': entry({1: -0.5, 2: -1.5}, 1)}
    path = {'9': entry({1: -0.03, 2: -4.0}, 1), '10': entry({1: -6.0, 2: -0.01}, 1)}
    r = summary.readout(path, fp32, 10, [1, 2])
    assert r['top1'] == 2
    assert r['max_abs_diff_from_fp32_tracked'] == pytest.approx(5.5)
    # FP32's top logprob (-0.5) minus FP32's logprob of the path's top-1 (token 2, -1.5).
    assert r['fp32_top_minus_fp32_of_path_top1'] == pytest.approx(1.0)
    assert r['max_abs_diff_from_fp32_recorded_before'] == pytest.approx(0.02)
    assert r['positions_before'] == 1


def missed(count: int, start: int = 0) -> set[tuple[str, int]]:
    return {('p', start + i) for i in range(count)}


def events(sglang: int = 0, **runs: int) -> dict[str, set[tuple[str, int]]]:
    """SGLang misses positions 0..sglang-1; each comparator misses its own, disjoint ones."""
    found = {'sglang': missed(sglang), **{name: set() for name in rates.COMPARATORS}}
    for name, count in runs.items():
        found[name] = missed(count, start=1000)
    return found


@pytest.mark.parametrize(
    ('sglang', 'hf', 'verdict'),
    [
        (6, 0, 'sglang-specific'),
        (10, 2, 'sglang-specific'),
        (9, 3, 'inconclusive'),  # ratio 3 but one-sided p = 0.073
        (4, 0, 'inconclusive'),  # too few SGLang events
        (8, 4, 'inconclusive'),  # ratio 2
        (6, 6, 'not specific'),
        (3, 3, 'inconclusive'),  # too few events in all
    ],
)
def test_decision_rule(sglang: int, hf: int, verdict: str) -> None:
    assert rates.decide(events(sglang, hf_bf16_float32state=hf))['verdict'] == verdict


def test_decision_ignores_the_bf16_state_run() -> None:
    assert rates.decide(events(6, hf_bf16_modelstate=50))['verdict'] == 'sglang-specific'


def test_decision_pairs_positions_both_miss() -> None:
    # Six SGLang misses, one of them shared with the comparator: five discordant, p = 1/32.
    shared = events(6)
    shared['hf_bf16_float32state'] = {('p', 0)}
    decision = rates.decide(shared)
    assert (decision['sglang_only'], decision['comparator_only']) == (5, 0)
    assert decision['paired_exact_p_one_sided'] == pytest.approx(1 / 32)
    assert decision['verdict'] == 'sglang-specific'
    # One shared miss and nothing else: no discordant position, p = 1.
    one = events(1)
    one['hf_bf16_float32state'] = {('p', 0)}
    assert rates.decide(one)['paired_exact_p_one_sided'] == 1.0


def test_decision_compares_against_the_worse_transformers_run() -> None:
    decision = rates.decide(events(9, hf_bf16_fla_float32state=6))
    assert decision['comparator'] == 'hf_bf16_fla_float32state'
    assert decision['verdict'] == 'not specific'


def write_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def test_rates_summary_counts_regret_by_region(tmp_path: Path) -> None:
    output = [5, rates.EOT[0], 6]
    good = [[-0.1, 7], [-3.0, 8]]
    bad = [[-0.1, 8], [-3.0, 7]]
    write_gz(
        tmp_path / 'sglang.jsonl.gz',
        [
            {
                'prompt': 'a',
                'prompt_ids': [1],
                'output_ids': output,
                'decode': [good, good, bad],
                'prefill': [good] * 3,
            }
        ],
    )
    for name in rates.HF_RUNS:
        write_gz(
            tmp_path / f'{name}.jsonl.gz',
            [{'prompt': 'a', 'decode': [good] * 3, 'prefill': [good] * 3}],
        )
    fp = {'top': [[-0.05, 7]], 'lp': {'7': -0.05, '8': -2.6}}
    write_gz(tmp_path / 'fp32.jsonl.gz', [{'prompt': 'a', 'positions': [fp, fp, fp]}])
    result = rates.summary(tmp_path)
    assert result['positions'] == {'before_eot': 2, 'after_eot': 1}
    decode = result['counts']['sglang/decode']
    after = decode['regret_above']['after_eot']
    assert (after['0.5'], after['1.0'], after['2.0'], after['5.0']) == (1, 1, 1, 0)
    assert decode['regret_above']['before_eot']['0.05'] == 0  # the path's top-1 is FP32's there
    # FP32's top-1 (7) at -0.05 against the path's -0.1 (two positions) and -3.0 (one).
    assert decode['fp32_top1_logprob_abs_diff']['max'] == pytest.approx(2.95)
    assert decode['top1_differs'] == {'before_eot': 0, 'after_eot': 1}
    assert result['worst']['sglang/decode'][0]['regret'] == pytest.approx(2.55)
    assert result['decision']['sglang_events'] == 1
    assert result['positions_missed_above_2_nats']['sglang'] == [['a', 2]]


def test_rates_summary_counts_a_position_missed_on_both_paths_once(tmp_path: Path) -> None:
    output = [5, 6]
    good = [[-0.1, 7], [-3.0, 8]]
    bad = [[-0.1, 8], [-3.0, 7]]
    write_gz(
        tmp_path / 'sglang.jsonl.gz',
        [
            {
                'prompt': 'a',
                'prompt_ids': [1],
                'output_ids': output,
                'decode': [good, bad],
                'prefill': [good, bad],
            }
        ],
    )
    for name in rates.HF_RUNS:
        write_gz(
            tmp_path / f'{name}.jsonl.gz',
            [{'prompt': 'a', 'decode': [good] * 2, 'prefill': [good] * 2}],
        )
    fp = {'top': [[-0.05, 7]], 'lp': {'7': -0.05, '8': -2.6}}
    write_gz(tmp_path / 'fp32.jsonl.gz', [{'prompt': 'a', 'positions': [fp, fp]}])
    result = rates.summary(tmp_path)
    assert result['counts']['sglang/decode']['regret_above']['before_eot']['2.0'] == 1
    assert result['counts']['sglang/prefill']['regret_above']['before_eot']['2.0'] == 1
    assert result['decision']['sglang_events'] == 1


def test_rates_summary_requires_every_source(tmp_path: Path) -> None:
    write_gz(tmp_path / 'sglang.jsonl.gz', [{'prompt': 'a'}])
    with pytest.raises(SystemExit, match='missing'):
        rates.summary(tmp_path)


def test_perturbed_reports_seed_range_and_control() -> None:
    def run(target_lp: float, before_lp: float) -> dict[str, Any]:
        return {
            '4': entry({1: before_lp, 2: -9.0}, 1),
            '5': entry({1: target_lp, 2: -1.0}, 1),
        }

    data = {
        'meta': {'site': 'gdn'},
        'results': [
            {
                'id': 't',
                'position': 5,
                'track': [1, 2],
                'runs': {
                    'none': run(-0.5, -0.01),
                    'seed_0': run(-0.4, -0.02),
                    'seed_1': run(-3.0, -0.01),
                },
            }
        ],
    }
    out = summary.perturbed(data)['targets']['t']
    assert out['range_over_seeds']['1'] == [-3.0, -0.4]
    assert out['top1_over_seeds'] == [1, 2]
    assert out['max_abs_change_recorded_before'] == pytest.approx(0.01)
