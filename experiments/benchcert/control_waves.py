"""Exploratory control, not declared: MTP stock against certified in synchronized waves.

    python -m experiments.benchcert.control_waves start --variant stock --out DIR  # in gpu_startup_lock.sh
    python -m experiments.benchcert.control_waves waves --variant stock --out DIR
    python -m experiments.benchcert.control_waves stop --variant stock --out DIR
    python -m experiments.benchcert.control_waves compare --out DIR

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
FAMILY = plan.FAMILIES['mtp']
WAVES, WAVE_SIZE, OSL = 8, 8, 512
OVERRIDES: dict[str, ArgValue] = {
    'mem-fraction-static': 0.25,
    'max-total-tokens': 100000,
    'max-running-requests': WAVE_SIZE,
    'max-mamba-cache-size': WAVE_SIZE,
}
WORKLOAD = plan.REPO / 'bench/workloads/mixed-v2/confirm.jsonl'


def source_point(runs: Path) -> Path:
    """Session 1's stock MTP point at c = 8 (64 measured requests, prompts 0-63)."""
    found = sorted((runs / 's1' / FAMILY.stock_label).glob('2026*/r0/c008'))
    if len(found) != 1:
        raise SystemExit(f'expected one s1 {FAMILY.stock_label} c=8 point, found {found}')
    return found[0]


def prompts(runs: Path) -> list[tuple[str, list[int]]]:
    """(prompt hash, prompt token ids) of the first 64 confirmation prompts, in order."""
    recorded = request_ids(source_point(runs))
    wanted = [prompt_hash(item['text']) for item in iter_jsonl(WORKLOAD)][: WAVES * WAVE_SIZE]
    missing = [key for key in wanted if 'input' not in recorded.get(key, {})]
    if missing:
        raise SystemExit(f'{len(missing)} of the first 64 prompts have no recorded token ids')
    return [(key, recorded[key]['input']) for key in wanted]


def start(out: Path, variant: str) -> int:
    env: dict[str, str] = {}
    if variant == 'cert':
        stats = out / 'certified_stats.json'
        env = plan.certified_env(FAMILY, 'cert', plan.REPO / 'src', stats)
    arm = resolve_arm(FAMILY.arm, OVERRIDES, env)
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


def waves(out: Path, variant: str, runs: Path) -> int:
    items = prompts(runs)
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


def compare_variants(out: Path) -> int:
    runs = {
        variant: {r['prompt']: r for r in iter_jsonl(out / variant / 'outputs.jsonl')}
        for variant in ('stock', 'cert')
    }
    stock, cert = runs['stock'], runs['cert']
    result = compare(
        {k: v['output_ids'] for k, v in stock.items()},
        {k: v['output_ids'] for k, v in cert.items()},
    )
    events = result.pop('events')
    contexts = []
    for key, d, tok_stock, tok_cert in events:
        ids, prefix = stock[key]['input_ids'], stock[key]['output_ids'][:d]
        contexts.append(
            {
                'id': context_id(ids, prefix, (tok_stock, tok_cert)),
                'input_ids': ids + prefix,
                'tokens': sorted((tok_stock, tok_cert)),
            }
        )
    with (out / 'contexts.jsonl').open('w') as handle:
        for record in contexts:
            handle.write(json.dumps(record) + '\n')
    summary = {
        'declared': False,
        'reading_rule': (
            '64/64 identical: the certified head gives identical tokens under identical batch'
            ' evolution (the claim check mode makes), not the cause of every timed divergence;'
            ' any difference: its first divergence is re-scored into the same classes'
        ),
        **result,
        'waves': WAVES,
        'wave_size': WAVE_SIZE,
        'osl': OSL,
    }
    (out / 'compare.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'positions'}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'waves', 'stop'):
        p = sub.add_parser(name)
        p.add_argument('--variant', choices=('stock', 'cert'), required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    sub.add_parser('compare').add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        return compare_variants(args.out)
    (args.out / args.variant).mkdir(parents=True, exist_ok=True)
    if args.command == 'start':
        return start(args.out, args.variant)
    if args.command == 'waves':
        return waves(args.out, args.variant, args.runs)
    return stop_server(args.out / args.variant)


if __name__ == '__main__':
    sys.exit(main())
