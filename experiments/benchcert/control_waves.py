"""Exploratory controls, not declared: certified against stock in synchronized waves.

    python -m experiments.benchcert.control_waves start --family F --variant V --out DIR
    python -m experiments.benchcert.control_waves waves --family F --variant V --out DIR
    python -m experiments.benchcert.control_waves stop --family F --variant V --out DIR
    python -m experiments.benchcert.control_waves compare --family F --out DIR

(`start` runs inside gpu_startup_lock.sh.) F is `mtp` (variants stock, cert) or
`dflash16` (stock, cert, cert0).

The timed MTP pairs diverge from each other above concurrency 1 more often than two
stock runs do. This control fixes the batch evolution: `mtp-tuned-triton` on the
benchmark's engine (a small shared-lane server: --mem-fraction-static 0.25, 100,000
KV tokens, 8 running requests and mamba slots), once stock and once with the timed
runs' certified environment, each serving the confirmation split's first 64 prompts
(their prompt token ids as recorded by session 1's stock MTP run at c = 8) as 8
waves of 8. Each wave is one batched /generate call (512 greedy tokens, ignore_eos),
so its 8 requests are prefilled together and the batch then evolves as a function of
the tokens alone.

Reading rule (set before the run): 64/64 identical outputs mean the certified head
gives identical tokens under identical batch evolution, the claim check mode also
makes; it does not show the cause of every timed divergence. Any difference sends
its first divergence to the same re-score classes (`compare` writes the contexts).

The DFlash block-16 control (added after session 3, approved by main) tests why the
timed block-16 pairs at c = 8, a gated-off point, diverged on the same three prompts
in every session. Same waves (prompts recorded by session 1's stock block-16 run at
c = 8), three arms: stock, certified as timed, and certified with
`SGLANG_CERTIFIED_HEAD_MAX_ROWS=0`, so the head never runs and only the stock head and
draft sampler inside the conditional nodes remain. Reading rule (main's): cert0
differing from stock points to the gated-off path (an integration exactness bug);
cert0 equal to stock while cert differs points to the head's certified ramp-down,
which check mode should then have caught at that shape; all three equal points to
closed-loop timing in the timed runs.

The seeded MTP control `mtpsmall` (added after h8's reference step, approved by main with
the red team's design) is a mechanism probe at the most sensitive row known, not a
reproduction of the closed-loop event. h8's reference step found that at 579ae7ce's output
position 439 stock plain decoding at batch 1, started at position 400, puts 1756 on top
(-0.32 nats), while prefills put it near -8. Here 579ae7ce's input is its prompt plus
session 1's output through position 399, so MTP decodes positions 400 onward and the state
at 439 comes from a prefill to 400 plus 39 decoded tokens. `mtp-tuned-triton` runs at the
timed pools (exclusive for memory, untimed) on four fresh servers in turn: stock, cert0
(the timed certified environment with `SGLANG_CERTIFIED_HEAD_MAX_ROWS=0`, where h7 saw the
event twice), cert, and stock again (`stock2`). Each serves the same 10 synchronized waves
twice in a row: 579ae7ce alone, and with 7, 11 or 15 companions (three fixed sets per
size, drawn once from a fixed seed from the c = 64 point's other measured prompts, never
session_000527). Each wave is one batched /generate call (512 greedy tokens, ignore_eos),
so the batch evolves as a function of the tokens alone; where SGLang allows it the call
also returns logprobs (top 5, and 1756, 68189, 8078, 5715 and 9471), kept for 579ae7ce's
positions 400-450.

Reading rule (set before the run), over every request's tokens:
- (i) Stock MTP's batch-1 token and logprob at 439 are the MTP path's own answer at this
  row. If it is 1756, stock MTP produces the event's token itself, and the closed loop's
  0 of 19 stock draws is batch-evolution luck.
- Baseline: on each server the two passes of a wave are identical, and stock and stock2
  are identical. If either fails, stock is not deterministic under identical batch
  evolution here, and the arms are read only as divergence rates beside stock against
  stock.
- (ii) With the baseline intact, any request on which cert0 or cert differs from both
  stock servers, at batch 1 or within a size: the certified graphs change the numerics
  (located by its first divergence). All identical: they do not, under identical batch
  evolution at the most sensitive row known, so differences between arms in the closed
  loop need a different batch evolution (timing) or a race.

The seeded control's run wrote no counters (the timed environment writes them every 20,000
glue calls), so it could not show that the certified head decided the cert arm's verifies;
a silent fallback to the stock head would also match stock. `mtpstats` serves the same 10
waves twice on the cert arm only (`certstats`: the timed certified environment with the
counters written on every glue call) and compares the outputs with the seeded control's
stock run.

Counter rerun (set before it ran): certified verify rows above zero, with every request
identical to stock on both passes, means the certified head decided and matched; zero
certified verify rows means a silent fallback, reported as such; certified rows with any
output differing from stock is a certified-head mismatch under identical batch evolution.
Uncounted calls (host-gated certified steps without a device count) must be zero.

The counter rerun counted 5 certified verify calls over the server's life, all at most 4
rows (the warm-up): requests that ask for logprobs keep the verify on the stock head
(SGLang's certified_head.py `_adjusts_logits`), so the seeded waves' verifies were stock in
cert as in cert0, while the draft and draft-extend heads ran certified. `mtptokens` serves
the same waves twice without logprobs on two servers in turn, stock (`stocktokens`) and
certified with the counters on every glue call (`certtokens`), and reads the counters
before and after every wave.

Token-only rerun (set before it ran): certified verify steps during the waves above zero,
with every token equal to stocktokens on both passes, means the certified verify decided
and matched; zero means it did not run; any token difference is a certified-verify
mismatch under identical batch evolution. Each server's two passes must be identical, and
stocktokens is compared with the seeded control's logprob stock run as a side check
(logprob requests should not change stock tokens).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.arms import ArgValue, resolve_arm
from bench.results import iter_jsonl, prompt_hash
from bench.server import Server
from experiments.benchcert import plan
from experiments.benchcert.analyze import compare, context_id, request_ids
from experiments.benchcert.rescore import stop as stop_server

PORT = 30083
WAVES, OSL = 8, 512
# The prompt of the one large divergence (MTP, c = 64, session 1, certified run).
TARGET_PROMPT = '579ae7ce'
HOOK_DIR = Path(__file__).resolve().parent / 'replay_hook'


@dataclass(frozen=True)
class Control:
    family: str  # plan family whose arm runs
    point: int  # the timed concurrency whose recorded prompts are used
    wave_size: int
    variants: tuple[str, ...]
    kv_tokens: int | None  # explicit KV cap of a small shared-lane server; None: timed pools


CONTROLS = {
    'mtp': Control('mtp', 8, 8, ('stock', 'cert'), 100000),
    'dflash16': Control('dflash16', 8, 8, ('stock', 'cert', 'cert0'), 30000),
    # The settling hold (exclusive, untimed): waves of 64 at the timed pools, and a
    # logged check-mode replay of the large event's wave (certlog).
    'mtp64': Control('mtp', 64, 64, ('stock', 'cert', 'certlog'), None),
    # Seeded MTP control (exclusive, untimed): stock, cert0, cert, stock again, at the
    # timed pools; waves of SMALL_SIZES (wave_size is the largest).
    'mtpsmall': Control('mtp', 64, 16, ('stock', 'cert0', 'cert', 'stock2'), None),
    # Its cert arm again with the counters written on every glue call (exclusive, untimed).
    'mtpstats': Control('mtp', 64, 16, ('certstats',), None),
    # The same waves without logprobs (which keep the verify on the stock head), stock then
    # certified, with counters on every glue call (exclusive, untimed).
    'mtptokens': Control('mtp', 64, 16, ('stocktokens', 'certtokens'), None),
}
VARIANTS = {name: control.variants for name, control in CONTROLS.items()}
SMALL_SIZES = (1, 8, 12, 16)
SMALL_SETS = 3  # companion sets per size above 1
SMALL_REPS = 2  # passes of every wave on each server
SMALL_SEED = 20261003
SMALL_TARGET, SMALL_PARTNER = '579ae7ce', 'session_000527'
SEED_AT = 400  # 579ae7ce's input carries session 1's output through position 399
SMALL_POSITION = 439
SMALL_TRACK = (1756, 68189, 8078, 5715, 9471)
SMALL_LOGPROB_SPAN = (400, 451)  # 579ae7ce's positions whose logprobs are kept


def small_plan(keys: list[str], partner: int, seed: int = SMALL_SEED) -> list[dict[str, Any]]:
    """The seeded control's waves: 579ae7ce first in each, companions drawn once from the
    other prompts (never session_000527): one wave of 1, SMALL_SETS of each larger size."""
    target = next(i for i, k in enumerate(keys) if k.startswith(SMALL_TARGET))
    others = [i for i in range(len(keys)) if i not in (target, partner)]
    rng = random.Random(seed)
    waves: list[dict[str, Any]] = []
    for size in SMALL_SIZES:
        for _ in range(1 if size == 1 else SMALL_SETS):
            members = [target, *rng.sample(others, size - 1)]
            waves.append({'wave': len(waves), 'size': size, 'members': members})
    return waves


def overrides(name: str) -> dict[str, ArgValue]:
    control = CONTROLS[name]
    if control.kv_tokens is None:
        return {}  # the arm's own pools (128 requests, 1,000,000 KV tokens, 128 slots)
    return {
        'mem-fraction-static': 0.25,
        'max-total-tokens': control.kv_tokens,
        'max-running-requests': control.wave_size,
        'max-mamba-cache-size': control.wave_size,
    }


WORKLOAD = plan.REPO / 'bench/workloads/mixed-v2/confirm.jsonl'


def source_point(runs: Path, name: str) -> Path:
    """Session 1's stock point at the control's concurrency (its measured prompts)."""
    control = CONTROLS[name]
    label = plan.FAMILIES[control.family].stock_label
    found = sorted((runs / 's1' / label).glob(f'2026*/r0/c{control.point:03d}'))
    if len(found) != 1:
        raise SystemExit(f'expected one s1 {label} c={control.point} point, found {found}')
    return found[0]


def prompts(runs: Path, name: str = 'mtp') -> list[tuple[str, list[int]]]:
    """(prompt hash, prompt token ids) of the control's prompts, in workload order."""
    count = WAVES * CONTROLS[name].wave_size
    recorded = request_ids(source_point(runs, name))
    wanted = [prompt_hash(item['text']) for item in iter_jsonl(WORKLOAD)][:count]
    missing = [key for key in wanted if 'input' not in recorded.get(key, {})]
    if missing:
        raise SystemExit(f'{len(missing)} of the first {count} prompts have no recorded token ids')
    return [(key, recorded[key]['input']) for key in wanted]


def variant_env(name: str, variant: str, stats: Path) -> dict[str, str]:
    """The timed certified environment; cert0 sets MAX_ROWS=0 so the head never runs;
    certlog adds check mode and the per-replay log (replay_hook/sitecustomize.py)."""
    if variant in ('stock', 'stock2', 'stocktokens'):
        return {}
    env = plan.certified_env(plan.FAMILIES[CONTROLS[name].family], 'cert', plan.REPO / 'src', stats)
    if variant in ('certstats', 'certtokens'):
        env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = '1'
    if variant == 'cert0':
        env['SGLANG_CERTIFIED_HEAD_MAX_ROWS'] = '0'
    if variant == 'certlog':
        env['SGLANG_CERTIFIED_HEAD_CHECK'] = '1'
        env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = '1'
        env['PYTHONPATH'] = str(HOOK_DIR)
        env['BENCHCERT_REPLAY_LOG'] = str(stats.parent / 'replay.jsonl')
    return env


def start(out: Path, name: str, variant: str) -> int:
    control = CONTROLS[name]
    env = variant_env(name, variant, out / variant / 'certified_stats.json')
    arm = resolve_arm(plan.FAMILIES[control.family].arm, overrides(name), env)
    if control.kv_tokens is not None:
        arm = type(arm)(**{**arm.to_json(), 'max_concurrency': control.wave_size})
    server = Server(arm, out / variant / 'server', PORT, sglang_worktree=plan.ENGINE_WORKTREE)
    server.start()
    try:
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (out / variant / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def waves(out: Path, name: str, variant: str, runs: Path) -> int:
    size = CONTROLS[name].wave_size
    items = prompts(runs, name)
    selected = range(WAVES)
    if variant == 'certlog':
        # Only the wave that holds the large event's prompt.
        index = next((i for i, (key, _) in enumerate(items) if key.startswith(TARGET_PROMPT)), None)
        if index is None:
            raise SystemExit(f'prompt {TARGET_PROMPT} is not among the control prompts')
        selected = range(index // size, index // size + 1)
    target = out / variant / 'outputs.jsonl'
    tmp = target.with_suffix('.jsonl.tmp')
    with tmp.open('w') as handle:
        for wave in selected:
            batch = items[wave * size : (wave + 1) * size]
            for (key, ids), output in zip(batch, generate_wave(batch, wave), strict=True):
                record = {'prompt': key, 'wave': wave, 'input_ids': ids, 'output_ids': output}
                handle.write(json.dumps(record) + '\n')
    tmp.replace(target)
    return 0


def generate_wave(batch: list[tuple[str, list[int]]], wave: int) -> list[list[int]]:
    """One synchronized wave: a single batched /generate call; OSL tokens per prompt."""
    return [list(r['output_ids']) for r in generate_results(batch, wave, {})]


def generate_results(
    batch: list[tuple[str, list[int]]], wave: int, extra: dict[str, Any]
) -> list[dict[str, Any]]:
    """The batched /generate call's per-request results (with `extra` request fields)."""
    body = {
        'input_ids': [ids for _, ids in batch],
        'sampling_params': {'max_new_tokens': OSL, 'temperature': 0.0, 'ignore_eos': True},
        **extra,
    }
    request = urllib.request.Request(
        f'http://127.0.0.1:{PORT}/generate',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        results: list[dict[str, Any]] = json.loads(response.read())
    if len(results) != len(batch):
        raise SystemExit(f'wave {wave}: {len(results)} results for {len(batch)} prompts')
    for result in results:
        if len(result['output_ids']) != OSL:
            raise SystemExit(
                f'wave {wave}: {len(result["output_ids"])} output tokens, wanted {OSL}'
            )
    return results


def seeded_logprobs(meta: dict[str, Any], first: int) -> dict[str, Any]:
    """579ae7ce's logprobs at positions SMALL_LOGPROB_SPAN from a seeded request's
    meta_info (generated index i is output position first + i)."""
    tops = meta.get('output_top_logprobs') or []
    ids = meta.get('output_token_ids_logprobs') or []
    kept = {}
    for position in range(*SMALL_LOGPROB_SPAN):
        i = position - first
        if 0 <= i < len(tops):
            kept[str(position)] = {
                'top5': [[float(e[0]), int(e[1])] for e in tops[i] or [] if e[0] is not None],
                'tracked': {
                    str(int(e[1])): float(e[0])
                    for e in (ids[i] if i < len(ids) else None) or []
                    if e[0] is not None
                },
            }
    return kept


def small_waves(out: Path, variant: str, runs: Path) -> int:
    """The seeded control on one server: every planned wave, SMALL_REPS passes each, with
    579ae7ce's input seeded with session 1's output through SEED_AT - 1."""
    import urllib.error

    from experiments.benchcert.context_waves import partner_index
    from experiments.benchcert.drain import point_dirs, requests

    items = prompts(runs, 'mtp64')
    keys = [key for key, _ in items]
    planned = small_plan(keys, partner_index(runs, keys))
    s1 = point_dirs(runs / 'drain', runs)[0][1]  # session 1's certified MTP c = 64 point
    recorded = next(
        r
        for r in requests(s1)
        if r['phase'] == 'profiling' and r['prompt'].startswith(SMALL_TARGET)
    )
    seeded = recorded['input'] + recorded['output'][:SEED_AT]
    track = {'return_logprob': True, 'top_logprobs_num': 5, 'token_ids_logprob': list(SMALL_TRACK)}
    logprobs_ok = not variant.endswith('tokens')  # logprob requests keep the verify on stock
    stats = out / variant / 'certified_stats.json'
    counted = variant in ('certstats', 'certtokens')
    counters = (out / variant / 'counters.jsonl').open('w') if counted else None
    target = out / variant / 'outputs.jsonl'
    tmp = target.with_suffix('.jsonl.tmp')
    with tmp.open('w') as handle:
        for wave in planned:
            batch = [items[i] for i in wave['members']]
            batch[0] = (batch[0][0], seeded)
            for rep in range(SMALL_REPS):
                before = read_stats(stats) if counted else {}
                results = None
                if logprobs_ok:
                    try:
                        results = generate_results(batch, wave['wave'], track)
                    except urllib.error.HTTPError as exc:  # logprobs refused with MTP
                        print(f'logprobs refused ({exc}); continuing without', flush=True)
                        logprobs_ok = False
                if results is None:
                    results = generate_results(batch, wave['wave'], {})
                if counters is not None:
                    time.sleep(STATS_SETTLE_S)  # the last step's counters land after the reply
                    delta = stats_delta(before, read_stats(stats))
                    row = {'wave': wave['wave'], 'size': wave['size'], 'rep': rep, 'paths': delta}
                    counters.write(json.dumps(row) + '\n')
                    counters.flush()
                for index, ((key, ids), result) in enumerate(zip(batch, results, strict=True)):
                    record: dict[str, Any] = {
                        'prompt': key,
                        'wave': wave['wave'],
                        'size': wave['size'],
                        'rep': rep,
                        'input_ids': ids,
                        'output_ids': list(result['output_ids']),
                    }
                    if index == 0:
                        meta = result.get('meta_info') or {}
                        record['seeded_at'] = SEED_AT
                        record['token_at_position'] = record['output_ids'][SMALL_POSITION - SEED_AT]
                        record['logprobs'] = seeded_logprobs(meta, SEED_AT) if logprobs_ok else None
                        record['spec_verify_ct'] = meta.get('spec_verify_ct')
                    handle.write(json.dumps(record) + '\n')
    tmp.replace(target)
    if counters is not None:
        counters.close()
    return 0


STATS_SETTLE_S = 1.0


def read_stats(path: Path) -> dict[str, Any]:
    """The certified head's cumulative counters per path ({} before the first write)."""
    for _ in range(5):
        if not path.exists():
            return {}
        try:
            paths: dict[str, Any] = json.loads(path.read_text()).get('paths', {})
            return paths
        except json.JSONDecodeError:  # caught mid-write
            time.sleep(0.2)
    raise SystemExit(f'{path}: unreadable counters')


def stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per path, the counters added between two snapshots; max_rows is the largest
    certified batch (rows) whose histogram count grew."""
    out = {}
    for path, a in after.items():
        b = before.get(path, {})
        hist_a = a.get('certified_rows_histogram') or {}
        hist_b = b.get('certified_rows_histogram') or {}
        steps_a, steps_b = a.get('host_steps') or {}, b.get('host_steps') or {}
        out[path] = {
            'calls': a.get('calls', 0) - b.get('calls', 0),
            'rows': a.get('rows', 0) - b.get('rows', 0),
            'certified_steps': steps_a.get('certified', 0) - steps_b.get('certified', 0),
            'stock_graph_steps': steps_a.get('stock_graph', 0) - steps_b.get('stock_graph', 0),
            'max_rows': max((int(k) for k, v in hist_a.items() if v > hist_b.get(k, 0)), default=0),
        }
    return out


def tokens_compare(out: Path, reference: Path) -> dict[str, Any]:
    """The token-only rerun: certtokens against stocktokens pass by pass, each server's two
    passes, stocktokens against the seeded control's (logprob) stock run, and the certified
    verify's counters over the waves only, per wave size."""
    stock = outputs_by_rep(out / 'stocktokens' / 'outputs.jsonl')
    cert = outputs_by_rep(out / 'certtokens' / 'outputs.jsonl')
    logprob_stock = outputs_by_rep(reference / 'stock' / 'outputs.jsonl')
    within = {
        'stocktokens': pair_outputs(stock[0], stock[1]),
        'certtokens': pair_outputs(cert[0], cert[1]),
    }
    against = {
        f'certtokens_vs_stocktokens/rep{rep}': pair_outputs(stock[rep], cert[rep])
        for rep in range(SMALL_REPS)
    }
    side = {
        f'stocktokens_vs_logprob_stock/rep{rep}': pair_outputs(logprob_stock[rep], stock[rep])
        for rep in range(SMALL_REPS)
    }
    rows = list(iter_jsonl(out / 'certtokens' / 'counters.jsonl'))
    by_size: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        for path, c in row['paths'].items():
            entry = by_size.setdefault(path, {}).setdefault(
                str(row['size']),
                {
                    'certified_steps': 0,
                    'calls': 0,
                    'stock_graph_steps': 0,
                    'rows': 0,
                    'max_rows': 0,
                },
            )
            entry['certified_steps'] += c['certified_steps']
            entry['calls'] += c['calls']
            entry['stock_graph_steps'] += c['stock_graph_steps']
            entry['rows'] += c['rows']
            entry['max_rows'] = max(entry['max_rows'], c['max_rows'])
    # Device coverage: every host-gated certified step must have its device calls
    # (calls_per_step per path), checked for each wave pass and path on its own, as
    # check_counters requires of the check launches (a signed sum could cancel).
    per_step = plan.FAMILIES[CONTROLS['mtptokens'].family].calls_per_step
    uncovered = []
    for row in rows:
        for path, c in row['paths'].items():
            missing = c['certified_steps'] * per_step.get(path, 1) - c['calls']
            entry = by_size[path][str(row['size'])]
            entry['uncounted_calls'] = entry.get('uncounted_calls', 0) + abs(missing)
            if missing != 0:
                uncovered.append(
                    {
                        'wave': row['wave'],
                        'rep': row['rep'],
                        'path': path,
                        'uncounted_calls': missing,
                    }
                )
    verify_steps = sum(v['certified_steps'] for v in by_size.get('verify', {}).values())
    # Baselines: each server's two passes, and the token-only stock run against the logprob
    # stock run (another server); without them the arms are not compared.
    baseline = all(r['identical'] == r['prompts'] for r in within.values())
    stock_stable = all(r['identical'] == r['prompts'] for r in side.values())
    identical = all(r['identical'] == r['prompts'] for r in against.values())
    if uncovered:
        reading = 'incomplete: host-gated certified steps without device calls'
    elif not (baseline and stock_stable):
        reading = 'baseline failed: stock is not deterministic here; no match or mismatch reading'
    elif verify_steps == 0:
        reading = 'the certified verify did not run in the waves'
    elif identical:
        reading = "the certified verify decided the waves' verifies and matched stock"
    else:
        reading = 'certified verify mismatch under identical batch evolution'
    return {
        'declared': False,
        'reading_rule': (__doc__ or '').split('Token-only rerun (set before it ran):')[1].strip(),
        'reference': str(reference),
        'within_server': within,
        'certtokens_vs_stocktokens': against,
        'stocktokens_vs_logprob_stock': side,
        'baseline_identical': baseline,
        'waves_counters_by_size': by_size,
        'waves_certified_verify_steps': verify_steps,
        'stock_runs_identical': stock_stable,
        'waves_uncovered_entries': uncovered,
        'reading': reading,
    }


def pair_outputs(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, Any]:
    """analyze.compare of two runs keyed by wave and prompt, with its divergences listed."""
    result = compare(a, b)
    result['divergences'] = [
        {'request': key, 'position': d, 'tokens': [ta, tb]}
        for key, d, ta, tb in result.pop('events')
    ]
    del result['positions']
    return result


def outputs_by_rep(path: Path) -> dict[int, dict[str, list[int]]]:
    """A small-control outputs file as {pass: {"wave:prompt": output ids}}."""
    by_rep: dict[int, dict[str, list[int]]] = {}
    for r in iter_jsonl(path):
        by_rep.setdefault(r['rep'], {})[f'{r["wave"]}:{r["prompt"]}'] = r['output_ids']
    return by_rep


def stats_compare(out: Path, reference: Path) -> dict[str, Any]:
    """The cert arm rerun with counters on every glue call: its counters per path and its
    outputs against the seeded control's stock outputs, pass by pass."""
    cert = outputs_by_rep(out / 'certstats' / 'outputs.jsonl')
    stock = outputs_by_rep(reference / 'stock' / 'outputs.jsonl')
    pairs = {
        f'certstats_vs_stock/rep{rep}': pair_outputs(stock[rep], cert[rep])
        for rep in range(SMALL_REPS)
    }
    stats = json.loads((out / 'certstats' / 'certified_stats.json').read_text())
    family = plan.FAMILIES[CONTROLS['mtpstats'].family]
    paths = {}
    for path, c in stats['paths'].items():
        host = (c.get('host_steps') or {}).get('certified', 0)
        paths[path] = {
            'calls': c.get('calls', 0),
            'rows': c.get('rows', 0),
            'fallback_rows': c.get('fallback_rows', 0),
            'mismatch_rows': c.get('mismatch_rows', 0),
            'max_certified_rows': c.get('max_certified_rows'),
            'host_certified_steps': host,
            'uncounted_calls': host * family.calls_per_step.get(path, 1) - c.get('calls', 0),
        }
    verify_rows = paths.get('verify', {}).get('rows', 0)
    identical = all(r['identical'] == r['prompts'] for r in pairs.values())
    # The reading needs every host-gated step counted on every path, and each run's two
    # passes identical (the cert arm's, and the reference stock run's).
    covered = all(p['uncounted_calls'] == 0 for p in paths.values())
    stable = all(
        pair_outputs(runs[0], runs[1])['identical'] == len(runs[0]) for runs in (cert, stock)
    )
    # Counted over the server's life, warm-up included: the waves' own verifies are not
    # separable here (tokens_compare counts them wave by wave).
    if not covered:
        reading = 'incomplete: host-gated certified steps without device calls'
    elif not stable:
        reading = 'baseline failed: a run differs between its two passes; no reading'
    elif verify_rows == 0:
        reading = 'no certified verify row: a silent fallback to the stock head'
    elif identical:
        reading = (
            'certified verify rows over the server life (warm-up included); outputs equal stock'
        )
    else:
        reading = 'certified verify rows over the server life (warm-up included); outputs differ'
    return {
        'declared': False,
        'reading_rule': (__doc__ or '').split('Counter rerun (set before it ran):')[1].strip(),
        'reference': str(reference),
        'paths': paths,
        'uncounted_calls_zero': covered,
        'passes_identical': stable,
        'against_stock': pairs,
        'reading': reading,
    }


def small_compare(out: Path) -> dict[str, Any]:
    """The seeded control's comparisons: the two passes on each server, each pair of
    servers on every pass (keyed by wave and prompt), and 579ae7ce's token at 439 per wave."""
    runs: dict[str, dict[int, dict[str, list[int]]]] = {}
    seeded: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS['mtpsmall']:
        by_rep: dict[int, dict[str, list[int]]] = {}
        for r in iter_jsonl(out / variant / 'outputs.jsonl'):
            by_rep.setdefault(r['rep'], {})[f'{r["wave"]}:{r["prompt"]}'] = r['output_ids']
            if 'seeded_at' in r:
                at = (r.get('logprobs') or {}).get(str(SMALL_POSITION)) or {}
                seeded.setdefault(variant, []).append(
                    {
                        'wave': r['wave'],
                        'size': r['size'],
                        'rep': r['rep'],
                        'token_at_439': r['token_at_position'],
                        'top5_at_439': at.get('top5'),
                        'tracked_at_439': at.get('tracked'),
                    }
                )
        runs[variant] = by_rep

    def diverged(name: str) -> set[str]:
        return {
            d['request'] for k, r in across.items() if k.startswith(name) for d in r['divergences']
        }

    within = {v: pair_outputs(reps[0], reps[1]) for v, reps in runs.items()}
    across: dict[str, dict[str, Any]] = {}
    names = list(VARIANTS['mtpsmall'])
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for rep in range(SMALL_REPS):
                across[f'{right}_vs_{left}/rep{rep}'] = pair_outputs(
                    runs[left][rep], runs[right][rep]
                )
    baseline = all(r['identical'] == r['prompts'] for r in within.values()) and all(
        r['identical'] == r['prompts']
        for k, r in across.items()
        if k.startswith('stock2_vs_stock/')
    )
    # A request on which an arm differs from both stock servers.
    arm_only = {
        arm: sorted(diverged(f'{arm}_vs_stock/') & diverged(f'stock2_vs_{arm}/'))
        for arm in ('cert0', 'cert')
    }
    batch1 = {v: [s for s in rows if s['size'] == 1] for v, rows in seeded.items()}
    stock1 = {s['token_at_439'] for s in batch1.get('stock', [])}
    if not baseline:
        reading_ii = 'baseline failed: stock is not deterministic here; arms read as rates only'
    elif any(arm_only.values()):
        reading_ii = (
            'an arm differs from both stock servers: the certified graphs change the numerics'
        )
    else:
        reading_ii = 'all identical: the certified graphs do not change the numerics here'
    return {
        'declared': False,
        'reading_rule': (__doc__ or '').split('Reading rule (set before the run), over')[1].strip(),
        'sizes': SMALL_SIZES,
        'sets_per_size': SMALL_SETS,
        'reps': SMALL_REPS,
        'seeded_at': SEED_AT,
        'osl': OSL,
        'reading_i_stock_batch1_token_at_439': sorted(stock1),
        'seeded_target': seeded,
        'within_server': within,
        'across_servers': across,
        'baseline_identical': baseline,
        'differs_from_both_stock': arm_only,
        'reading_ii': reading_ii,
    }


def compare_variants(out: Path, family: str = 'mtp', reference: Path | None = None) -> int:
    """Every pair of the control's variants: identical outputs and first divergences
    (over the prompts both served: certlog serves one wave)."""
    if family == 'mtptokens':
        tokens = tokens_compare(out, reference or out.parent / 'seeded')
        (out / 'compare.json').write_text(json.dumps(tokens, indent=1) + '\n')
        print(
            json.dumps(
                {k: tokens[k] for k in ('waves_counters_by_size', 'baseline_identical', 'reading')}
            )
        )
        for name, result in {
            **tokens['certtokens_vs_stocktokens'],
            **tokens['stocktokens_vs_logprob_stock'],
        }.items():
            print(name, {k: v for k, v in result.items() if k != 'divergences'})
        return 0
    if family == 'mtpstats':
        stats = stats_compare(out, reference or out.parent / 'seeded')
        (out / 'compare.json').write_text(json.dumps(stats, indent=1) + '\n')
        print(json.dumps({k: stats[k] for k in ('paths', 'uncounted_calls_zero', 'reading')}))
        return 0
    if family == 'mtpsmall':
        summary = small_compare(out)
        (out / 'compare.json').write_text(json.dumps(summary, indent=1) + '\n')
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        'reading_i_stock_batch1_token_at_439',
                        'baseline_identical',
                        'reading_ii',
                    )
                }
            )
        )
        for name, result in {**summary['within_server'], **summary['across_servers']}.items():
            print(name, {k: v for k, v in result.items() if k != 'divergences'})
        return 0
    runs = {
        variant: {r['prompt']: r for r in iter_jsonl(out / variant / 'outputs.jsonl')}
        for variant in VARIANTS[family]
    }
    pairs = {}
    contexts: dict[str, dict[str, Any]] = {}
    names = list(VARIANTS[family])
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(runs[left]) & set(runs[right])
            a = {k: v for k, v in runs[left].items() if k in shared}
            b = {k: v for k, v in runs[right].items() if k in shared}
            result = compare(
                {k: v['output_ids'] for k, v in a.items()},
                {k: v['output_ids'] for k, v in b.items()},
            )
            events = result.pop('events')
            result['divergences'] = [
                {'prompt': key, 'position': d, 'tokens': [ta, tb]} for key, d, ta, tb in events
            ]
            for key, d, ta, tb in events:
                ids, prefix = a[key]['input_ids'], a[key]['output_ids'][:d]
                cid = context_id(ids, prefix, (ta, tb))
                contexts[cid] = {'id': cid, 'input_ids': ids + prefix, 'tokens': sorted((ta, tb))}
            pairs[f'{right}_vs_{left}'] = result
    with (out / 'contexts.jsonl').open('w') as handle:
        for record in contexts.values():
            handle.write(json.dumps(record) + '\n')
    rule = (
        '64/64 identical: the certified head gives identical tokens under identical batch'
        ' evolution (the claim check mode makes), not the cause of every timed divergence;'
        ' any difference: its first divergence is re-scored into the same classes'
    )
    if family == 'dflash16':
        rule += (
            '; cert0 (MAX_ROWS=0) differing from stock points to the gated-off path inside the'
            ' conditional nodes; cert0 equal to stock while cert differs points to the'
            ' certified ramp-down; all equal points to closed-loop timing in the timed runs'
        )
    summary = {
        'declared': False,
        'family': family,
        'reading_rule': rule,
        'pairs': pairs,
        'waves': WAVES,
        'wave_size': CONTROLS[family].wave_size,
        'osl': OSL,
    }
    (out / 'compare.json').write_text(json.dumps(summary, indent=1) + '\n')
    for name, result in pairs.items():
        print(name, {k: v for k, v in result.items() if k not in ('positions', 'divergences')})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'waves', 'stop'):
        p = sub.add_parser(name)
        p.add_argument('--family', choices=sorted(VARIANTS), required=True)
        p.add_argument(
            '--variant',
            choices=(
                'stock',
                'cert',
                'cert0',
                'certlog',
                'stock2',
                'certstats',
                'stocktokens',
                'certtokens',
            ),
            required=True,
        )
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    c = sub.add_parser('compare')
    c.add_argument('--family', choices=sorted(VARIANTS), required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--reference', type=Path, help="mtpstats: the seeded control's output directory")
    args = parser.parse_args(argv)
    if args.command == 'compare':
        return compare_variants(args.out, args.family, args.reference)
    if args.variant not in VARIANTS[args.family]:
        parser.error(f'{args.family} has variants {VARIANTS[args.family]}')
    (args.out / args.variant).mkdir(parents=True, exist_ok=True)
    if args.command == 'start':
        return start(args.out, args.family, args.variant)
    if args.command == 'waves':
        if args.family in ('mtpsmall', 'mtpstats', 'mtptokens'):
            return small_waves(args.out, args.variant, args.runs)
        return waves(args.out, args.family, args.variant, args.runs)
    return stop_server(args.out / args.variant)


if __name__ == '__main__':
    sys.exit(main())
