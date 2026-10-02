"""CPU tests for experiments/hostgap/summarize.py: where the derived idle's GPU busy time comes from.

The host-trace analysis is replaced by synthetic traces, so these tests check only the
per-concurrency choice between a label's own trace and its patched trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/hostgap'))

import summarize

TRACES: dict[str, dict[str, Any]] = {}


def write_label(tag: Path, label: str, cycles: dict[int, float], traces: dict[int, Any]) -> None:
    """One label directory: an unprofiled window per concurrency in `cycles`, and a
    collected window per concurrency in `traces` (value: the synthetic trace)."""
    d = tag / label
    d.mkdir(parents=True)
    rows = []
    for c, cycle in cycles.items():
        rows.append(
            {
                'concurrency': c,
                'window_kind': 'unprofiled',
                'repeat': 0,
                'cycle_ms': cycle,
                'tokens_per_s_per_user': 100.0,
                'tokens_per_s': 100.0 * c,
                'tokens_per_request_per_cycle': 4.0,
                'hostload': {'foreign_cores_mean': 0.1, 'contended': False},
            }
        )
    for c, trace in traces.items():
        report = d / f'c{c}.nsys-rep'
        report.write_text('')
        TRACES[report.name + label] = trace
        rows.append({'concurrency': c, 'window_kind': 'collected', 'report': str(report)})
    (d / 'windows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


def trace(busy: float, eager: bool = True) -> dict[str, Any]:
    return {
        'gpu_busy_ms_per_cycle': busy,
        'cycle_ms': busy + 2.0,
        'gpu_idle_ms_per_cycle': 2.0,
        'idle_fraction': 2.0 / (busy + 2.0),
        'eager_kernel_records': eager,
    }


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict[str, Any]:
    monkeypatch.setattr(
        summarize, 'analyze', lambda report: TRACES[report.name + report.parent.name]
    )
    out = tmp_path / 'summary.json'
    monkeypatch.setattr(
        sys, 'argv', ['summarize.py', str(tmp_path / 'tag'), '--out', str(out), *extra]
    )
    assert summarize.main() == 0
    return json.loads(out.read_text())


def test_stock_borrows_patched_busy_only_where_its_own_trace_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = tmp_path / 'tag'
    write_label(tag, 'x-none', {1: 7.0, 8: 9.0, 32: 16.0}, {})
    # Own trace: usable at c = 1, missing at c = 8, without eager kernels at c = 32.
    write_label(tag, 'x-host', {}, {1: trace(5.0), 32: trace(11.0, eager=False)})
    write_label(tag, 'x-patched-none', {1: 6.9, 8: 8.9, 32: 15.8}, {})
    write_label(tag, 'x-patched-host', {}, {1: trace(4.9), 8: trace(6.0), 32: trace(14.0)})
    rows = run(tmp_path, monkeypatch)['derived_idle']['x']
    assert rows['1']['gpu_busy_source'] == 'x-host'
    assert rows['1']['traced_gpu_busy_ms_per_cycle'] == 5.0
    assert 'profiled_idle_ms_per_cycle' in rows['1']
    assert rows['8']['traced_gpu_busy_ms_per_cycle'] == 6.0
    assert 'no usable trace at c = 8' in rows['8']['gpu_busy_source']
    assert 'profiled_idle_ms_per_cycle' not in rows['8']
    assert rows['32']['traced_gpu_busy_ms_per_cycle'] == 14.0
    assert 'no eager kernel records at c = 32' in rows['32']['gpu_busy_source']
    assert rows['32']['derived_unprofiled_idle_ms_per_cycle'] == pytest.approx(2.0)


def test_patched_label_drops_only_the_concurrency_whose_trace_lost_its_kernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = tmp_path / 'tag'
    write_label(tag, 'x-patched-none', {1: 6.9, 8: 8.9}, {})
    write_label(tag, 'x-patched-host', {}, {1: trace(4.9), 8: trace(6.0, eager=False)})
    rows = run(tmp_path, monkeypatch)['derived_idle']['x-patched']
    assert set(rows) == {'1'}
    assert rows['1']['gpu_busy_source'] == 'x-patched-host'


def test_label_pattern_selects_labels_and_rejects_an_empty_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = tmp_path / 'tag'
    write_label(tag, 'a-none', {1: 7.0}, {})
    write_label(tag, 'b-none', {1: 7.0}, {})
    assert set(run(tmp_path, monkeypatch, '--labels', 'a-*')['labels']) == {'a-none'}
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, '--labels', 'c-*')
