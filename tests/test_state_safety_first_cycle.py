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
    chunk_counters_ok,
    counts,
    declaration_hashes,
    log_odds,
    void_reasons,
)
from server import CONFIGS, MODEL_REVISION, POOL_PIN, pool_flags

FRAGILE = [[-0.6, 1], [-0.7, 2]]  # top-2 gap 0.1 nats
SURE = [[-0.01, 1], [-5.0, 2]]


def rec(ids, top, chunks=None):
    r = {'output_ids': ids, 'top_logprobs': [top] * len(ids)}
    if chunks is not None:
        # As the server streams them: the cumulative verify count only in the last chunk.
        r['chunks'] = [[n, 0] for n in chunks]
        r['chunks'][-1][1] = len(chunks) - 1
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
            {
                'id': i,
                'output_ids': [1, 2],
                'top_logprobs': [[[-0.1, 1], [-0.2, 2]]] * 2,
                'finish_reason': {'type': 'length'},
                'completion_tokens': 2,
                'aborted_by_client': False,
            }
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
                'server_id': f'30058:{config}:1',
            },
        }
        (runs / f'{run}.meta.json').write_text(json.dumps(meta))
    files = declaration_hashes()
    (runs / 'attest').mkdir()
    for hold in ('plain', 'mtp'):
        for when in ('before', 'after'):
            rec = {
                'head': DECLARATION,
                'porcelain': '',
                'files': files,
                'pid': {'plain': 100, 'mtp': 200}[hold],
                'process_cwd': '/w/experiments/state_safety',
                'runner_dir': '/w/experiments/state_safety',
            }
            (runs / 'attest' / f'{hold}-{when}.json').write_text(json.dumps(rec))
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

    def other_server_for_c32(runs, prompts, manifest):
        _edit_meta(runs, 'mtp_tree/c32', server_id='30058:other:2')

    def no_after_attestation(runs, prompts, manifest):
        (runs / 'attest' / 'mtp-after.json').unlink()

    def dirty_checkout(runs, prompts, manifest):
        path = runs / 'attest' / 'plain-before.json'
        rec = json.loads(path.read_text())
        rec['porcelain'] = ' M experiments/state_safety/client.py\n'
        path.write_text(json.dumps(rec))

    def edited_runner_file(runs, prompts, manifest):
        path = runs / 'attest' / 'mtp-after.json'
        rec = json.loads(path.read_text())
        rec['files'] = {**rec['files'], 'server.py': '0' * 64}
        path.write_text(json.dumps(rec))

    def other_process_after(runs, prompts, manifest):
        path = runs / 'attest' / 'mtp-after.json'
        rec = json.loads(path.read_text())
        rec['pid'] = 999
        path.write_text(json.dumps(rec))

    def process_in_other_checkout(runs, prompts, manifest):
        path = runs / 'attest' / 'plain-before.json'
        rec = json.loads(path.read_text())
        rec['process_cwd'] = '/dirty/experiments/state_safety'
        path.write_text(json.dumps(rec))

    def unfinished_record(runs, prompts, manifest):
        path = runs / 'mtp_tree/c1.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['finish_reason'] = None
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def completion_count_differs(runs, prompts, manifest):
        path = runs / 'plain/c1.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[2]['completion_tokens'] = 5
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def aborted_by_client(runs, prompts, manifest):
        path = runs / 'mtp_s5/c32.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[1]['aborted_by_client'] = True
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
        other_runner_code: 'mtp_tree/c32: repo_sha is not the declaration commit',
        other_server_for_c32: 'mtp_tree: c1 and c32 were not served by the same server',
        other_process_after: 'hold mtp: attestations not bound to one run_matrix process',
        process_in_other_checkout: 'hold plain: run_matrix process not in the attested checkout',
        unfinished_record: 'mtp_tree/c1: 1 incomplete records',
        completion_count_differs: 'plain/c1: 1 incomplete records',
        aborted_by_client: 'mtp_s5/c32: 1 incomplete records',
        no_after_attestation: 'hold mtp: no after attestation',
        dirty_checkout: 'hold plain: checkout not clean (before)',
        edited_runner_file: 'hold mtp: runner files differ',
        short_logprobs: 'mtp_s5/c1: 1 incomplete records',
        one_candidate: 'plain/c1: 1 incomplete records',
        missing_prompt: 'mtp_s5/c1: prompt IDs differ',
        prompts_not_frozen: 'does not match the frozen manifest',
    }
    for edit, text in expected.items():
        got = reasons(edit)
        assert any(text in r for r in got), (edit.__name__, got)


def test_chunks_must_fit_one_verify_cycle_each():
    ok = {'chunks': [[1, 0], [6, 0], [2, 2]], 'spec_verify_ct': 2, 'output_ids': [1] * 9}
    assert chunk_counters_ok(ok, max_commit=6)
    # A chunk longer than one cycle can commit.
    assert not chunk_counters_ok({**ok, 'chunks': [[1, 0], [7, 0], [2, 2]]}, max_commit=6)
    # A counter in a middle chunk, or a final count that is not the chunk count.
    assert not chunk_counters_ok({**ok, 'chunks': [[1, 0], [6, 1], [2, 2]]}, max_commit=6)
    assert not chunk_counters_ok({**ok, 'chunks': [[1, 0], [6, 0], [2, 3]]}, max_commit=6)
    assert not chunk_counters_ok({**ok, 'spec_verify_ct': 3}, max_commit=6)
    # The first chunk must be the single prefill token, and the chunks must cover
    # every output token.
    assert not chunk_counters_ok({**ok, 'chunks': [[2, 0], [5, 0], [2, 2]]}, max_commit=6)
    assert not chunk_counters_ok({**ok, 'output_ids': [1] * 10}, max_commit=6)


def test_attest_runner_records_and_detects_only_the_python_process(tmp_path, monkeypatch):
    import subprocess

    import attest_runner

    repo = tmp_path / 'repo'
    base = repo / 'experiments' / 'state_safety'
    base.mkdir(parents=True)
    for name in attest_runner.RUNNER_FILES:
        (base / name).write_text(name)
    git = ['git', '-C', str(repo)]
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run(
        [*git, '-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '-qm', 'x'],
        check=True,
    )
    rec = attest_runner.attest(repo)
    assert rec['porcelain'] == '' and len(rec['head']) == 40
    (base / 'server.py').write_text('edited')
    assert attest_runner.attest(repo)['porcelain'] != ''

    runs = tmp_path / 'runs'
    waiting = (
        f' 7 bash gpu_lock.sh -x bash -c python run_matrix.py --configs plain --out-dir {runs}'
    )
    started = f'42 python run_matrix.py --configs mtp_s5,mtp_tree --passes c1,c32 --out-dir {runs}'

    class Out:
        def __init__(self, text):
            self.stdout = text

    monkeypatch.setattr(attest_runner.subprocess, 'run', lambda *a, **k: Out(waiting + '\n'))
    assert attest_runner.running_hold(runs) is None
    monkeypatch.setattr(attest_runner.subprocess, 'run', lambda *a, **k: Out(started + '\n'))
    assert attest_runner.running_hold(runs) == ('mtp', 42)
    # The record names the observed process and its directory.
    rec = attest_runner.attest(repo, 42, '/somewhere')
    assert rec['pid'] == 42 and rec['process_cwd'] == '/somewhere'
    assert rec['runner_dir'] == str(base.resolve())


def test_watch_attests_back_to_back_holds(monkeypatch, tmp_path):
    import attest_runner

    # The plain hold ends and the MTP hold starts within one poll.
    seen = iter([None, ('plain', 11), ('plain', 11), ('mtp', 22), ('mtp', 22), None])
    monkeypatch.setattr(attest_runner, 'running_hold', lambda runs: next(seen))
    monkeypatch.setattr(attest_runner, 'process_cwd', lambda pid: f'/proc/{pid}')
    written = []
    monkeypatch.setattr(
        attest_runner,
        'write',
        lambda runs, hold, when, checkout, pid, cwd: written.append((hold, when, pid, cwd)),
    )
    attest_runner.watch(tmp_path, tmp_path, poll=0)
    assert written == [
        ('plain', 'before', 11, '/proc/11'),
        ('plain', 'after', 11, '/proc/11'),
        ('mtp', 'before', 22, '/proc/22'),
        ('mtp', 'after', 22, '/proc/22'),
    ]
