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


def counts_with(events: dict[str, int]) -> dict[str, Any]:
    paths = ['sglang/decode', 'sglang/prefill']
    paths += [f'{n}/{k}' for n in rates.HF_RUNS for k in ('decode', 'prefill')]
    out = {}
    for path in paths:
        table = {
            region: dict.fromkeys(map(str, rates.THRESHOLDS), 0)
            for region in ('before_eot', 'after_eot')
        }
        table['after_eot']['2.0'] = events.get(path, 0)
        out[path] = {'regret_above': table}
    return out


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
    counts = counts_with({'sglang/decode': sglang, 'hf_bf16_float32state/prefill': hf})
    assert rates.decide(counts)['verdict'] == verdict


def test_decision_ignores_the_bf16_state_run() -> None:
    counts = counts_with({'sglang/decode': 6, 'hf_bf16_modelstate/decode': 50})
    assert rates.decide(counts)['verdict'] == 'sglang-specific'


def test_decision_compares_against_the_worse_transformers_run() -> None:
    counts = counts_with({'sglang/decode': 9, 'hf_bf16_fla_float32state/prefill': 6})
    decision = rates.decide(counts)
    assert decision['comparator'] == 'hf_bf16_fla_float32state'
    assert decision['verdict'] == 'not specific'


def write_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def test_rates_summary_counts_regret_by_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rates, 'OUTPUT_LEN', 3)
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
    assert decode['regret_above']['after_eot'] == {'0.5': 1, '1.0': 1, '2.0': 1, '5.0': 0}
    assert decode['regret_above']['before_eot']['0.5'] == 0
    assert decode['top1_differs'] == {'before_eot': 0, 'after_eot': 1}
    assert result['worst']['sglang/decode'][0]['regret'] == pytest.approx(2.55)
    assert result['decision']['sglang_events'] == 1


def test_rates_summary_requires_every_source(tmp_path: Path) -> None:
    write_gz(tmp_path / 'sglang.jsonl.gz', [{'prompt': 'a'}])
    with pytest.raises(SystemExit, match='missing'):
        rates.summary(tmp_path)
