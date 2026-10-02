"""CPU tests of the teacher-forced score summary and the gross-event report (benchcert)."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from bench.results import prompt_hash
from experiments.benchcert import plan, score_report
from experiments.benchcert.drain import FAMILY, TARGET


def test_point_names_give_group_family_variant_and_level() -> None:
    d = score_report.describe('h6/cert1/r1/c064')
    assert (d['group'], d['family'], d['variant'], d['repeat'], d['concurrency']) == (
        'h6',
        'mtp',
        'cert',
        'r1',
        64,
    )
    assert score_report.describe('h7/cert0a/r0/c008')['variant'] == 'cert0'
    s = score_report.describe(f's2/{FAMILY.stock_label}/c016')
    assert (s['group'], s['family'], s['variant'], s['concurrency']) == ('s2', 'mtp', 'stock', 16)
    plain = plan.FAMILIES['plain']
    assert score_report.describe(f's1/{plain.cert_label}/c128')['family'] == 'plain'
    assert score_report.describe(f'check2/{plain.check_label}/c001')['variant'] == 'check'


def disagree(position: int, token: int, top1: int, gap: float) -> dict[str, Any]:
    return {
        'position': position,
        'token': token,
        'logprob': -1.0 - gap,
        'top1': top1,
        'top1_logprob': -1.0,
    }


def record(phase: str, prompt: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [e['top1_logprob'] - e['logprob'] for e in entries]
    return {
        'phase': phase,
        'prompt': prompt,
        'positions': 512,
        'disagree': entries,
        'max_gap': max(gaps, default=0.0),
        'near': sum(1 for g in gaps if 0.5 < g < 2),
        'gross': sum(1 for g in gaps if g >= 2),
    }


def test_point_summary_counts_all_and_measured_requests() -> None:
    target = TARGET[0] + 'ffffffff'
    records = [
        record('warmup', 'a' * 16, [disagree(3, 1, 2, 0.8)]),
        record('profiling', 'b' * 16, [disagree(5, 1, 2, 2.5), disagree(9, 1, 2, 0.125)]),
        record('profiling', target, [disagree(TARGET[1], TARGET[2], 8078, 3.8125)]),
        {'phase': 'profiling', 'prompt': 'c' * 16, 'error': 'boom'},
    ]
    row = score_report.summarize_point('h6/cert1/r1/c064', records)
    assert (row['requests'], row['errors'], row['positions']) == (4, 1, 3 * 512)
    assert (row['near'], row['gross']) == (1, 2)
    assert (row['requests_measured'], row['near_measured'], row['gross_measured']) == (3, 0, 2)
    assert row['token_439_if_not_top1'] == TARGET[2] and row['gap_439'] == 3.8125
    assert row['max_gap'] == 3.8125


def test_clopper_pearson_matches_closed_forms() -> None:
    lo, hi = score_report.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-6)
    lo, hi = score_report.clopper_pearson(10, 10)
    assert lo == pytest.approx(0.025 ** (1 / 10), abs=1e-6) and hi == 1.0
    lo, hi = score_report.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-5) and hi == pytest.approx(0.812914, abs=1e-5)


def test_rate_ratio_interval_contains_the_ratio_and_handles_zero_counts() -> None:
    even = score_report.rate_ratio(20, 1000, 20, 1000)
    assert even['ratio'] == 1.0 and even['ci95'][0] < 1 < even['ci95'][1]
    double = score_report.rate_ratio(40, 1000, 10, 500)
    assert double['ratio'] == 2.0 and double['ci95'][0] < 2 < double['ci95'][1]
    none_in_second = score_report.rate_ratio(2, 100, 0, 100)
    assert none_in_second['ratio'] is None and none_in_second['ci95'][1] is None
    assert 0 < none_in_second['ci95'][0] < 1
    nothing = score_report.rate_ratio(0, 100, 0, 100)
    assert nothing['ratio'] is None and nothing['ci95'] == [0.0, None]
    assert math.isclose(score_report.binom_cdf(3, 3, 0.4), 1.0)


def write_export(path: Path, requests: list[dict[str, Any]]) -> None:
    """A minimal aiperf raw export: streamed chunks with cumulative completion tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as handle:
        for req in requests:
            responses = []
            for t, tokens in req['chunks']:
                chunk = {'usage': {'completion_tokens': tokens}}
                if tokens == len(req['output']):
                    chunk['sglext'] = {'output_ids': [req['output']], 'input_ids': req['input']}
                responses.append({'perf_ns': t, 'packets': [{'value': json.dumps(chunk)}]})
            responses.append(
                {'perf_ns': req['chunks'][-1][0] + 1, 'packets': [{'value': '[DONE]'}]}
            )
            handle.write(
                json.dumps(
                    {
                        'metadata': {
                            'benchmark_phase': req['phase'],
                            'conversation_id': req['cid'],
                        },
                        'start_perf_ns': req['start'],
                        'responses': responses,
                        'payload': {'messages': [{'content': req['text']}]},
                    }
                )
                + '\n'
            )


MS = 1_000_000


def test_report_times_gross_events_and_classes_co_batched_disagreements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = ['victim prompt', 'neighbour prompt', 'late prompt']
    requests = [
        # The event: position 6 streams at 1,000 ms.
        {
            'text': texts[0],
            'phase': 'profiling',
            'cid': 's0',
            'start': 0,
            'input': [7, 8],
            'output': list(range(10)),
            'chunks': [(500 * MS, 4), (1000 * MS, 8), (1500 * MS, 10)],
        },
        # Co-batched: positions 4-7 stream at 1,100 ms (in the window), 8-9 at 2,000 ms.
        {
            'text': texts[1],
            'phase': 'profiling',
            'cid': 's1',
            'start': 0,
            'input': [9],
            'output': list(range(10)),
            'chunks': [(400 * MS, 4), (1100 * MS, 8), (2000 * MS, 10)],
        },
        # Not in flight then.
        {
            'text': texts[2],
            'phase': 'profiling',
            'cid': 's2',
            'start': 1600 * MS,
            'input': [9],
            'output': list(range(10)),
            'chunks': [(1700 * MS, 10)],
        },
    ]
    point = tmp_path / 'runs' / 'point'
    write_export(point / 'aiperf/profile_export_raw.jsonl.gz', requests)
    name = 'h6/cert1/r0/c064'
    records = [
        record('profiling', prompt_hash(texts[0]), [disagree(6, 11, 12, 3.0)]),
        record(
            'profiling',
            prompt_hash(texts[1]),
            [disagree(5, 1, 2, 0.75), disagree(7, 1, 2, 0.0), disagree(9, 1, 2, 1.5)],
        ),
        record('profiling', prompt_hash(texts[2]), [disagree(2, 1, 2, 0.25)]),
    ]
    out = tmp_path / 'drain'
    path = score_report.score_file(out, name)
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    monkeypatch.setattr(score_report, 'point_dirs', lambda o, r: [(name, point)])
    contexts = tmp_path / 'contexts.jsonl'
    result = score_report.report(out, tmp_path / 'runs', contexts)
    (event,) = result['gross_events']
    assert (event['position'], event['token'], event['top1'], event['gap']) == (6, 11, 12, 3.0)
    assert event['in_flight'] == 2 and event['ms_before_point_end'] == pytest.approx(
        1000.0, abs=0.1
    )
    assert event['co_batched_requests_in_window'] == 1
    assert event['co_batched_disagreements_by_class'] == {
        'tie': 1,
        'rounding': 0,
        'near': 1,
        'gross': 0,
    }
    (near,) = event['co_batched_events_above_near']
    assert near['position'] == 5 and near['dt_ms'] == 100.0
    (ctx,) = [json.loads(line) for line in contexts.read_text().splitlines()]
    assert ctx['input_ids'] == [7, 8, 0, 1, 2, 3, 4, 5] and ctx['tokens'] == [11, 12]
    assert result['positive_control'] is None
    assert result['gross_by_variant'] == [
        {'group': 'h6', 'family': 'mtp', 'variant': 'cert', 'gross_events': 1}
    ]


def test_point_files_with_errors_are_scored_again(tmp_path: Path) -> None:
    from experiments.benchcert.drain import scored

    path = tmp_path / 'p.jsonl'
    assert not scored(path)
    path.write_text(json.dumps({'phase': 'warmup', 'positions': 512}) + '\n')
    assert scored(path)
    path.write_text(path.read_text() + json.dumps({'phase': 'profiling', 'error': 'x'}) + '\n')
    assert not scored(path)
