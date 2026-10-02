"""CPU tests of the h7 ring readings (benchcert, ring_report.py)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.benchcert import ring_report as rr


def test_target_row_is_tracked_on_the_device_from_the_first_verify() -> None:
    ref = list(range(1000, 1512))
    steps = 128  # each verify accepts all three drafts and the bonus: 4 positions
    records, predict, accept = [], np.zeros((steps, 256), np.int32), np.zeros((steps, 128), np.int32)
    for k in range(steps):
        pos = 1 + 4 * k
        # Request 0: another prompt length; request 1: the target, in slot 9, whose host
        # output length lags by one verify.
        records.append(
            {'slots': [3, 9], 'prompt_lens': [70, 75], 'output_lens': [pos, max(0, pos - 4)]}
        )
        accept[k, :2] = 4
        predict[k, 4:8] = ref[pos : pos + 4]
    hits = rr.find_target(records, predict, accept, ref, 75)
    (hit,) = [h for h in hits if h['offset'] == 1]
    # Position 439 = 1 + 4 * 109 + 2: verify 109, the request's third row.
    assert (hit['index'], hit['request'], hit['slot'], hit['row']) == (109, 1, 9, 4 + 2)
    assert hit['output_len'] == 437 and hit['host_output_len'] == 433
    assert all(h['offset'] == 1 for h in hits)  # offsets 0 and 2 fail the token check
    # A different prefix (a token before 439 differs from the reference) is not located.
    predict[50, 5] += 1
    assert rr.find_target(records, predict, accept, ref, 75) == []


def test_readings_name_each_failure_mode() -> None:
    ok = {'head_id': 68189, 'status': 2}
    assert rr.readings(True, True, 40, 5, ok) == []
    assert rr.readings(True, False, 40, 5, ok) == [1]
    assert rr.readings(False, False, 40, 5, None) == [5]
    assert rr.readings(True, True, 5, 5, ok) == [4]
    assert rr.readings(True, True, 40, 5, {'head_id': 1756, 'status': 0}) == [2]
    assert rr.readings(True, True, 40, 5, {'head_id': 1756, 'status': 2}) == [3]
    assert rr.readings(True, True, 40, 5, {'head_id': 1756, 'status': 1}) == [3]


def arrays(steps: int) -> dict[str, np.ndarray]:
    return {
        'gate': np.ones(steps, bool),
        'valid': np.full(steps, 8, np.int32),
        'ids': np.zeros((steps, 64), np.int64),
        'status': np.zeros((steps, 64), np.int32),
        'count': np.ones((steps, 64), np.int32),
        'cand': np.zeros((steps, 64, 64), np.int32),
        'rlo': np.zeros((steps, 64, 64), np.float32),
        'rhi': np.zeros((steps, 64, 64), np.float32),
        'predict': np.zeros((steps, 256), np.int32),
        'accept': np.zeros((steps, 128), np.int32),
    }


def test_consistency_flags_impossible_ids_and_emission_mismatches() -> None:
    a = arrays(1)
    records: list[dict[str, Any]] = [{'k': 7, 'rows': 8, 'batch_size': 2, 'host_gate': True}]
    # Row 0: certified (status 0), id has the largest lower bound: fine.
    a['count'][0, 0] = 2
    a['cand'][0, 0, :2] = [10, 11]
    a['rlo'][0, 0, :2] = [5.0, 3.0]
    a['rhi'][0, 0, :2] = [5.5, 3.5]
    a['ids'][0, 0] = 10
    # Row 1: column fallback chose a candidate whose upper bound is below another's lower.
    a['status'][0, 1] = rr.STATUS_AMBIGUOUS
    a['count'][0, 1] = 3
    a['cand'][0, 1, :3] = [20, 21, 1756]
    a['rlo'][0, 1, :3] = [7.0, 6.9, 3.0]
    a['rhi'][0, 1, :3] = [7.1, 7.0, 3.2]
    a['ids'][0, 1] = 1756
    # Row 2: id not among the candidates.
    a['count'][0, 2] = 1
    a['cand'][0, 2, 0] = 30
    a['ids'][0, 2] = 31
    # Emission: request 0 accepted two rows; row 1's predicted id differs from the head's.
    a['accept'][0, 0] = 2
    a['predict'][0, :2] = [10, 99]
    a['ids'][0, 3:8] = 0
    result = rr.consistency(a, records)
    t = result['totals']
    assert t['certified_steps'] == 1 and t['certified_rows'] == 8
    assert t['id_rhi_below_max_rlo'] == 1 and t['id_not_in_cand'] == 1
    assert t['status0_id_not_max_rlo'] == 0
    assert t['emission_rows'] == 2 and t['emission_mismatch'] == 1
    kinds = sorted(e['kind'] for e in result['examples'])
    assert kinds == ['emission', 'id_not_in_cand', 'id_rhi_below_max_rlo']
    bad = next(e for e in result['examples'] if e['kind'] == 'id_rhi_below_max_rlo')
    assert bad['wrong_in_cand'] and bad['id_rhi_below_max_rlo'] and bad['near_tie_in_cand'] == []


def test_gate_off_steps_are_not_read_and_gate_mismatch_is_counted() -> None:
    a = arrays(2)
    a['gate'][:] = [False, True]
    a['ids'][1, 0] = 5  # not in its one-candidate list (candidate 0)
    records = [
        {'k': 1, 'rows': 8, 'batch_size': 2, 'host_gate': False},
        {'k': 2, 'rows': 4, 'batch_size': 1, 'host_gate': False},
    ]
    t = rr.consistency(a, records)['totals']
    assert t['certified_steps'] == 1 and t['gate_differs_from_host'] == 1
    assert t['valid_differs_from_rows'] == 1 and t['id_not_in_cand'] == 1


def test_context_waves_are_fixed_small_and_split_between_blocks() -> None:
    from experiments.benchcert import context_waves as cw

    keys = [f'{i:016x}' for i in range(511)] + [cw.TARGET + '0' * 8]
    partner = 7
    waves = cw.plan_waves(keys, partner)
    assert waves == cw.plan_waves(keys, partner) and len(waves) == cw.WAVES
    target = len(keys) - 1
    for wave in waves:
        assert cw.SIZES[0] <= wave['size'] <= cw.SIZES[1] == 16
        assert wave['members'][0] == target and len(set(wave['members'])) == wave['size']
        assert all(0 <= d <= cw.STAGGER_S for d in wave['delays'])
        if wave['partner']:
            assert wave['members'][1] == partner
            lead = wave['delays'][0] - wave['delays'][1]
            assert cw.PARTNER_LEAD_S[0] - 1e-3 <= lead <= cw.PARTNER_LEAD_S[1] + 1e-3
        else:
            assert partner not in wave['members']
    assert sum(w['partner'] for w in waves) == cw.WAVES // 2
    halves = cw.block_waves(waves, 0) + cw.block_waves(waves, 1)
    assert halves == waves
    ref = list(range(600))
    assert cw.classify(None, ref) == 'incomplete'
    assert cw.classify(ref[:400], ref) == 'incomplete'
    assert cw.classify([*ref[:439], cw.WRONG, *ref[440:512]], ref) == 'wrong'
    assert cw.classify([*ref[:439], 68189], ref) == '68189'
    assert cw.classify([7, *ref[1:512]], ref) == 'other_prefix'
