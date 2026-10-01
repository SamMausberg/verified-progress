"""The declared first-cycle analysis on synthetic runs (CPU only)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

import numpy as np
from first_cycle import RUNS, analyse, counts, log_odds, void_reasons
from server import POOL_PIN

FRAGILE = [[-0.6, 1], [-0.7, 2]]  # top-2 gap 0.1 nats
SURE = [[-0.01, 1], [-5.0, 2]]


def rec(ids, top, chunks=None):
    r = {'output_ids': ids, 'top_logprobs': [top] * len(ids)}
    if chunks is not None:
        r['chunks'] = [[n, None] for n in chunks]
        r['spec_verify_ct'] = len(chunks) - 1
    return r


def test_counts_split_first_and_later_cycles_at_fragile_positions():
    ref = rec([1] * 8, FRAGILE)
    # Diverges at index 2, which is inside the first verify cycle (chunk 1: 1-3).
    comp = rec([1, 1, 2, 1, 1, 1, 1, 1], FRAGILE, chunks=[1, 3, 4])
    assert counts(ref, comp, comp) == [1, 2, 0, 0]
    # No divergence: indices 1-3 are the first cycle, 4-7 later; index 0 is prefill.
    same = rec([1] * 8, FRAGILE, chunks=[1, 3, 4])
    assert counts(ref, same, same) == [0, 3, 0, 4]
    # Positions where the reference is not near a tie are not counted.
    assert counts(rec([1] * 8, SURE), same, same) == [0, 0, 0, 0]


def test_log_odds_adds_half_to_every_cell():
    t = np.array([0.0, 10.0, 0.0, 100.0])
    assert abs(float(log_odds(t)) - np.log(0.5 * 100.5 / (10.5 * 0.5))) < 1e-12


def test_decision_supported_only_when_both_bounds_hold():
    prompts = [f'p{i}' for i in range(200)]

    def runs(first_cycle_diverges_in_primary):
        out = {r: {} for r in RUNS}
        for k, p in enumerate(prompts):
            plain = rec([1] * 10, FRAGILE)
            out['plain/c1'][p] = plain
            for cfg in ('mtp_s5', 'mtp_tree'):
                # Primary pairs diverge in the first cycle for half the prompts and
                # otherwise late; control pairs diverge late for a fifth of them.
                ids = [1] * 10
                if first_cycle_diverges_in_primary and k % 2 == 0:
                    ids[2] = 2
                elif k % 5 == 0:
                    ids[8] = 2
                out[f'{cfg}/c1'][p] = rec(ids, FRAGILE, chunks=[1, 3, 3, 3])
                c32 = list(out[f'{cfg}/c1'][p]['output_ids'])
                if k % 5 == 1:
                    c32[8] = 3
                out[f'{cfg}/c32'][p] = rec(c32, FRAGILE, chunks=[1, 3, 3, 3])
        return out

    res = analyse(runs(True))
    assert res['prompts_included'] == 200
    assert res['a_holds'] and res['b_holds'] and res['decision'] == 'supported'
    res = analyse(runs(False))
    assert not res['a_holds'] and res['decision'] == 'inconclusive'


def test_missing_or_unpinned_runs_make_the_result_void(tmp_path):
    assert any('missing' in r for r in void_reasons(tmp_path))
    for run in RUNS:
        path = tmp_path / f'{run}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
        meta = {
            'pool_pin': POOL_PIN,
            'server_info': {'resolved_pools': POOL_PIN},
            'max_new_tokens': 256,
            'top_logprobs_num': 5,
        }
        (tmp_path / f'{run}.meta.json').write_text(json.dumps(meta))
    assert void_reasons(tmp_path) == []
    meta['server_info'] = {'resolved_pools': {**POOL_PIN, 'max_total_tokens': 1}}
    (tmp_path / 'plain/c1.meta.json').write_text(json.dumps(meta))
    assert void_reasons(tmp_path) == ['plain/c1: pools not pinned as declared']
