"""CPU tests of the certified verify head's stress test (benchcert, fallback_stress.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from experiments.benchcert import fallback_stress as fs


def test_batches_pad_to_the_captured_verify_graphs() -> None:
    assert fs.padded_bs(1) == 1 and fs.padded_bs(9) == 10 and fs.padded_bs(15) == 16
    assert fs.padded_bs(33) == 40 and fs.padded_bs(61) == 64 and fs.padded_bs(65) == 72
    with pytest.raises(ValueError):
        fs.padded_bs(129)
    assert fs.Step(15, True, 'drain').m == 64 and fs.Step(15, True, 'drain').rows == 60
    assert fs.Step(9, True, 'drain').m == 40


def test_gate_follows_the_engine_row_limit() -> None:
    assert fs.engine_gate(16) and fs.engine_gate(1)
    assert not fs.engine_gate(17)  # 68 rows
    assert not fs.engine_gate(16, eligible=False)
    assert fs.engine_gate(64, max_rows=1000)  # 256 rows fit the head's graphs; 288 do not
    assert fs.engine_gate(64, max_rows=256) and not fs.engine_gate(72, max_rows=1000)


def replay(n: int, t: float, bs: int) -> dict[str, Any]:
    return {
        'kind': 'replay',
        'path': 'verify',
        'n': n,
        't': t,
        'batch_size': bs,
        'rows': 4 * bs,
        'gates': {'verify': fs.engine_gate(bs)},
    }


def test_drain_sequence_is_the_first_full_point_from_its_steady_tail() -> None:
    records: list[dict[str, Any]] = [replay(0, 0.0, 1), {'kind': 'sample', 'n': 0}]
    sizes = [64] * 10 + [60, 40, 20, 16, 9, 3, 1]
    t = 10.0
    for i, bs in enumerate(sizes):
        records.append(replay(i + 1, t, bs))
        t += 0.01
    t += 5  # the next point
    records += [replay(100 + i, t + 0.01 * i, bs) for i, bs in enumerate([64, 64, 2])]
    steps = fs.drain_steps(records, tail_from=60, lead=3)
    assert [s.bs for s in steps] == [64, 64, 64, 60, 40, 20, 16, 9, 3, 1]
    assert [s.gate for s in steps] == [False] * 6 + [True] * 4
    assert fs.gate_toggles(steps) == 1
    bad = [dict(r, gates={'verify': True}) if r.get('batch_size') == 40 else r for r in records]
    with pytest.raises(ValueError, match='differs from the engine rule'):
        fs.drain_steps(bad, tail_from=60, lead=3)


def test_synthetic_sizes_toggle_the_gate_at_each_size() -> None:
    steps = fs.synthetic_steps()
    assert [s.rows for s in steps] == [r for r in fs.SYNTHETIC_ROWS for _ in (0, 1)]
    assert [s.gate for s in steps] == [True, False] * len(fs.SYNTHETIC_ROWS)
    assert fs.gate_toggles(steps) == len(steps) - 1


def pools(with_context: bool = True) -> fs.Pools:
    return fs.Pools(
        clear=np.arange(0, 100),
        column=np.arange(100, 120),
        dense=np.arange(120, 125),
        context=np.arange(200, 712) if with_context else np.arange(0),
        draft=np.arange(800, 900),
    )


def test_batches_mix_row_classes_and_place_the_target_request() -> None:
    rng = np.random.default_rng(0)
    p = pools()
    target = p.target_rows()
    assert target is not None and list(target) == [200 + 438, 200 + 439, 200 + 440, 200 + 441]
    seen_target = seen_dense = 0
    column_rows = total = 0
    for _ in range(400):
        idx, cls = fs.compose(rng, 64, p)
        assert len(idx) == len(cls) == 64
        assert np.all(np.isin(idx[cls == 0], p.clear)) and np.all(np.isin(idx[cls == 1], p.column))
        assert np.all(np.isin(idx[cls == 2], p.dense))
        if (cls == 3).any():
            seen_target += 1
            (at,) = np.nonzero(idx == 200 + 439)
            assert len(at) == 1 and at[0] % 4 == 1  # the request's row for position 439
            assert list(idx[at[0] - 1 : at[0] + 3]) == list(target)
        seen_dense += int((cls == 2).any())
        column_rows += int((cls == 1).sum())
        total += 64
    assert 150 < seen_target < 250 and 90 < seen_dense < 180
    assert 0.04 < column_rows / total < 0.1
    idx, cls = fs.compose(rng, 8, pools(with_context=False))
    assert not (cls == 3).any() and pools(False).target_rows() is None


def test_bf16_rounding_is_round_to_nearest_even() -> None:
    x = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8, -2.5, 1.0 + 2**-8 + 2**-12], np.float32)
    bits = fs.bf16_bits(x)
    assert list(fs.bf16_value(bits)) == [1.0, 1.0, 1.0 + 2**-6, -2.5, 1.0 + 2**-7]
    assert fs.bf16_ulp(1.0) == 2**-7 and fs.bf16_ulp(24.0) == 0.125 and fs.bf16_ulp(-0.3) == 2**-9


def test_nudge_sets_the_requested_gaps_up_to_bf16_rounding() -> None:
    rng = np.random.default_rng(1)
    k = 2560
    w = (rng.normal(0, 0.02, (3, k))).astype(np.float32)
    w = fs.bf16_value(fs.bf16_bits(w)).astype(np.float64)
    h = fs.bf16_value(fs.bf16_bits(rng.normal(0, 1, k).astype(np.float32))).astype(np.float64)
    gaps = np.array([0.01, -0.03])
    out = fs.bf16_value(fs.nudge(h, w, gaps)).astype(np.float64)
    z = w @ out
    assert z[0] - z[1] == pytest.approx(0.01, abs=0.01)
    assert z[0] - z[2] == pytest.approx(-0.03, abs=0.01)
    assert np.abs(out - h).max() < 0.5  # a small move, not a new vector


def test_context_rows_start_one_position_before_the_output() -> None:
    prompt, output = 5, 3
    chunks = [[[float(i)] * 4 for i in range(4)], [[float(i)] * 4 for i in range(4, 7)], [99.0] * 4]
    rows = fs.context_rows(chunks, prompt, output)
    assert rows.shape == (3, 4) and list(rows[:, 0]) == [4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        fs.context_rows(chunks[:1], prompt, output)
