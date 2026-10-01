"""The declared first-cycle analysis on synthetic runs (CPU only)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

import numpy as np
from first_cycle import (
    DECLARATION,
    DECLARED,
    RUNS,
    SGLANG_PIN,
    analyse,
    counts,
    log_odds,
    void_reasons,
)
from server import CONFIGS, MODEL_REVISION, POOL_PIN, pool_flags

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

    res = analyse(runs(True), fisher=False)
    assert res['prompts_included'] == 200
    assert res['a_holds'] and res['b_holds'] and res['decision'] == 'supported'
    res = analyse(runs(False), fisher=False)
    assert not res['a_holds'] and res['decision'] == 'inconclusive'


def _write_declared(tmp_path, ids):
    """Five declared runs over ids, plus the matching prompt file and manifest."""
    import hashlib

    tmp_path.mkdir(parents=True, exist_ok=True)
    prompts = tmp_path / 'prompts.jsonl'
    items = [{'id': i, 'input_ids': [k]} for k, i in enumerate(ids)]
    prompts.write_text(''.join(json.dumps(it) + '\n' for it in items))
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps([it['id'], it['input_ids']]).encode())
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'input_ids_sha256': h.hexdigest(), 'num_prompts': len(ids)}))
    runs = tmp_path / 'runs'
    for run in RUNS:
        path = runs / f'{run}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {'id': i, 'output_ids': [1, 2], 'top_logprobs': [[[-0.1, 1], [-0.2, 2]]] * 2}
            for i in ids
        ]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        algo, steps, topk, conc = DECLARED[run]
        config = run.split('/')[0]
        meta = {
            'config': config,
            'flags': CONFIGS[config] + pool_flags(),
            'model_revision': MODEL_REVISION,
            'warm': False,
            'sglang_sha': SGLANG_PIN,
            'sglang_dirty': False,
            'repo_sha': DECLARATION,
            'concurrency': conc,
            'pool_pin': POOL_PIN,
            'max_new_tokens': 256,
            'top_logprobs_num': 5,
            'server_info': {
                'resolved_pools': POOL_PIN,
                'speculative_algorithm': algo,
                'speculative_num_steps': steps,
                'speculative_eagle_topk': topk,
                'disable_radix_cache': False,
                'disable_overlap_schedule': False,
                'attention_backend': 'flashinfer',
            },
        }
        (runs / f'{run}.meta.json').write_text(json.dumps(meta))
    return runs, prompts, manifest


def _edit_meta(runs, run, **info):
    path = runs / f'{run}.meta.json'
    meta = json.loads(path.read_text())
    for k, v in info.items():
        if k in meta:
            meta[k] = v
        else:
            meta['server_info'][k] = v
    path.write_text(json.dumps(meta))


def test_declared_runs_are_not_void(tmp_path, monkeypatch):
    import first_cycle

    monkeypatch.setattr(first_cycle, 'DECLARED_PROMPTS', 3)
    runs, prompts, manifest = _write_declared(tmp_path, ['a', 'b', 'c'])
    assert void_reasons(runs, prompts, manifest) == []


def test_each_departure_from_the_declaration_makes_the_result_void(tmp_path, monkeypatch):
    import first_cycle

    monkeypatch.setattr(first_cycle, 'DECLARED_PROMPTS', 3)

    def reasons(edit):
        runs, prompts, manifest = _write_declared(tmp_path / edit.__name__, ['a', 'b', 'c'])
        edit(runs, prompts, manifest)
        return void_reasons(runs, prompts, manifest)

    def missing_run(runs, prompts, manifest):
        (runs / 'mtp_tree/c32.jsonl').unlink()

    def unpinned(runs, prompts, manifest):
        _edit_meta(runs, 'plain/c1', resolved_pools={**POOL_PIN, 'max_total_tokens': 1})

    def plain_is_speculative(runs, prompts, manifest):
        _edit_meta(runs, 'plain/c1', speculative_algorithm='EAGLE')

    def wrong_steps(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_s5/c1', speculative_num_steps=3)

    def wrong_tree_topk(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_tree/c32', speculative_eagle_topk=1)

    def wrong_concurrency(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_s5/c32', concurrency=8)

    def radix_off(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_tree/c1', disable_radix_cache=True)

    def extra_flag(runs, prompts, manifest):
        meta = json.loads((runs / 'mtp_s5/c1.meta.json').read_text())
        _edit_meta(runs, 'mtp_s5/c1', flags=meta['flags'] + ['--disable-radix-cache'])

    def warm_pass(runs, prompts, manifest):
        _edit_meta(runs, 'plain/c1', warm=True)

    def other_revision(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_tree/c1', model_revision='0' * 40)

    def other_engine(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_s5/c32', sglang_sha='0' * 40)

    def dirty_engine(runs, prompts, manifest):
        _edit_meta(runs, 'plain/c1', sglang_dirty=True)

    def other_runner_code(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_tree/c32', repo_sha='0' * 40)

    def short_logprobs(runs, prompts, manifest):
        path = runs / 'mtp_s5/c1.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['top_logprobs'] = rows[0]['top_logprobs'][:1]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def one_candidate(runs, prompts, manifest):
        path = runs / 'plain/c1.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[1]['top_logprobs'][1] = [[-0.1, 1]]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def missing_prompt(runs, prompts, manifest):
        path = runs / 'mtp_s5/c1.jsonl'
        path.write_text(''.join(path.read_text().splitlines(keepends=True)[:2]))

    def prompts_not_frozen(runs, prompts, manifest):
        prompts.write_text(prompts.read_text().replace('"input_ids": [0]', '"input_ids": [9]'))

    expected = {
        missing_run: 'mtp_tree/c32: missing',
        unpinned: 'plain/c1: pools not pinned',
        plain_is_speculative: 'plain/c1: configuration',
        wrong_steps: 'mtp_s5/c1: configuration',
        wrong_tree_topk: 'mtp_tree/c32: configuration',
        wrong_concurrency: 'mtp_s5/c32: configuration',
        radix_off: 'mtp_tree/c1: radix cache or overlap',
        extra_flag: 'mtp_s5/c1: configuration or flags differ',
        warm_pass: 'plain/c1: not a cold pass',
        other_revision: 'mtp_tree/c1: model revision',
        other_engine: 'mtp_s5/c32: engine is not the clean pin',
        dirty_engine: 'plain/c1: engine is not the clean pin',
        other_runner_code: 'mtp_tree/c32: runner code differs',
        short_logprobs: 'mtp_s5/c1: 1 records lack top-5 logprobs',
        one_candidate: 'plain/c1: 1 records lack top-5 logprobs',
        missing_prompt: 'mtp_s5/c1: prompt IDs differ',
        prompts_not_frozen: 'does not match the frozen manifest',
    }
    for edit, text in expected.items():
        got = reasons(edit)
        assert any(text in r for r in got), (edit.__name__, got)
