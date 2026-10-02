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
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from bench.arms import ArgValue, resolve_arm
from bench.results import iter_jsonl, prompt_hash
from bench.server import Server
from experiments.benchcert import plan
from experiments.benchcert.analyze import compare, context_id, request_ids
from experiments.benchcert.rescore import stop as stop_server

PORT = 30083
WAVES, WAVE_SIZE, OSL = 8, 8, 512
VARIANTS = {'mtp': ('stock', 'cert'), 'dflash16': ('stock', 'cert', 'cert0')}
# Small shared-lane servers with explicit KV caps.
KV_TOKENS = {'mtp': 100000, 'dflash16': 30000}


def overrides(family: str) -> dict[str, ArgValue]:
    return {
        'mem-fraction-static': 0.25,
        'max-total-tokens': KV_TOKENS[family],
        'max-running-requests': WAVE_SIZE,
        'max-mamba-cache-size': WAVE_SIZE,
    }
WORKLOAD = plan.REPO / 'bench/workloads/mixed-v2/confirm.jsonl'


def source_point(runs: Path, family: str) -> Path:
    """Session 1's stock point at c = 8 (64 measured requests, prompts 0-63)."""
    label = plan.FAMILIES[family].stock_label
    found = sorted((runs / 's1' / label).glob('2026*/r0/c008'))
    if len(found) != 1:
        raise SystemExit(f'expected one s1 {label} c=8 point, found {found}')
    return found[0]


def prompts(runs: Path, family: str = 'mtp') -> list[tuple[str, list[int]]]:
    """(prompt hash, prompt token ids) of the first 64 confirmation prompts, in order."""
    recorded = request_ids(source_point(runs, family))
    wanted = [prompt_hash(item['text']) for item in iter_jsonl(WORKLOAD)][: WAVES * WAVE_SIZE]
    missing = [key for key in wanted if 'input' not in recorded.get(key, {})]
    if missing:
        raise SystemExit(f'{len(missing)} of the first 64 prompts have no recorded token ids')
    return [(key, recorded[key]['input']) for key in wanted]


def variant_env(family: str, variant: str, stats: Path) -> dict[str, str]:
    """The timed certified environment; cert0 sets MAX_ROWS=0 so the head never runs."""
    if variant == 'stock':
        return {}
    env = plan.certified_env(plan.FAMILIES[family], 'cert', plan.REPO / 'src', stats)
    if variant == 'cert0':
        env['SGLANG_CERTIFIED_HEAD_MAX_ROWS'] = '0'
    return env


def start(out: Path, family: str, variant: str) -> int:
    fam = plan.FAMILIES[family]
    env = variant_env(family, variant, out / variant / 'certified_stats.json')
    arm = resolve_arm(fam.arm, overrides(family), env)
    arm = type(arm)(**{**arm.to_json(), 'max_concurrency': WAVE_SIZE})
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


def waves(out: Path, family: str, variant: str, runs: Path) -> int:
    items = prompts(runs, family)
    target = out / variant / 'outputs.jsonl'
    tmp = target.with_suffix('.jsonl.tmp')
    with tmp.open('w') as handle:
        for wave in range(WAVES):
            batch = items[wave * WAVE_SIZE : (wave + 1) * WAVE_SIZE]
            body = {
                'input_ids': [ids for _, ids in batch],
                'sampling_params': {'max_new_tokens': OSL, 'temperature': 0.0, 'ignore_eos': True},
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
            for (key, ids), result in zip(batch, results, strict=True):
                output = list(result['output_ids'])
                if len(output) != OSL:
                    raise SystemExit(f'wave {wave}: {len(output)} output tokens, wanted {OSL}')
                record = {'prompt': key, 'wave': wave, 'input_ids': ids, 'output_ids': output}
                handle.write(json.dumps(record) + '\n')
    tmp.replace(target)
    return 0


def compare_variants(out: Path, family: str = 'mtp') -> int:
    """Every pair of the family's variants: identical outputs and first divergences."""
    runs = {
        variant: {r['prompt']: r for r in iter_jsonl(out / variant / 'outputs.jsonl')}
        for variant in VARIANTS[family]
    }
    pairs = {}
    contexts: dict[str, dict[str, Any]] = {}
    names = list(VARIANTS[family])
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = runs[left], runs[right]
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
            " certified ramp-down; all equal points to closed-loop timing in the timed runs"
        )
    summary = {
        'declared': False,
        'family': family,
        'reading_rule': rule,
        'pairs': pairs,
        'waves': WAVES,
        'wave_size': WAVE_SIZE,
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
        p.add_argument('--variant', choices=('stock', 'cert', 'cert0'), required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    c = sub.add_parser('compare')
    c.add_argument('--family', choices=sorted(VARIANTS), required=True)
    c.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        return compare_variants(args.out, args.family)
    if args.variant not in VARIANTS[args.family]:
        parser.error(f'{args.family} has variants {VARIANTS[args.family]}')
    (args.out / args.variant).mkdir(parents=True, exist_ok=True)
    if args.command == 'start':
        return start(args.out, args.family, args.variant)
    if args.command == 'waves':
        return waves(args.out, args.family, args.variant, args.runs)
    return stop_server(args.out / args.variant)


if __name__ == '__main__':
    sys.exit(main())
