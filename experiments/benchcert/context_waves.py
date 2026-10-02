"""Small-batch waves at 579ae7ce's context, stock against certified (exploratory, not declared).

    python -m experiments.benchcert.context_waves start --variant V --block B --out DIR  # startup lock
    python -m experiments.benchcert.context_waves waves --variant V --block B --out DIR
    python -m experiments.benchcert.context_waves stop --variant V --block B --out DIR
    python -m experiments.benchcert.context_waves compare --out DIR

The certified engine committed 1756 at 579ae7ce's position 439 in two closed-loop timed
runs (session 1 and one h6a rerun), each time in the c = 64 point's final drain, where the
verify batch is 64 rows or fewer and the certified verify head runs. This control asks
whether that context alone, served in small batches, produces the token. Each wave
serves 579ae7ce with B - 1 neighbours (B drawn from 8 to 16; neighbours drawn from the
other 511 measured prompts of the c = 64 point, by their prompt token ids as session 1's
stock run recorded them, the chat-templated prompt ids from its sglext), each as its own
/generate request (512 greedy tokens, ignore_eos) started after a random delay of up to
STAGGER_S seconds. The batch never exceeds 16 requests, so every verify, including
579ae7ce's at position 439, has at most 64 rows and runs certified in the certified arm.
Every other wave also holds session_000527 (the request that emits 1756 legitimately and
finished about 0.85 s before 579ae7ce's position 439 in the drain), started 0.5-1.5 s
before 579ae7ce; the other waves exclude it, and the two strata are reported apart.
Servers run `mtp-tuned-triton` at the timed pools (128 running requests, 128 mamba slots,
1,000,000 KV tokens; the arm runs with the radix cache off), stock or in the timed runs'
certified environment, and the cache is flushed before every wave, as the timed points
flush it. The waves are drawn once from a fixed seed and served by three arms in six blocks: stock,
cert0 (the certified graphs and conditional nodes with MAX_ROWS=0, so the head never
runs), certified, certified with the device ring (replay_hook/benchcert_ring.py), cert0,
stock; each block is a fresh server serving half of the waves, so every arm serves the
same waves.

Reading rule (set before the run; main's, with the red team's counts), over the waves
whose 579ae7ce output reaches position 439 with session 1's prefix. Each of cert0 and
cert is compared with stock: any 1756 in the stock arm means the token is shared with the
stock engine (fragile numerics at a near tie); 5 or more events in a certified-graph arm,
with a one-sided conditional binomial p < 0.05 against an equal split with stock, mean a
fault of that configuration at this context (in cert0, of the certified integration
without the head's decision); anything else is inconclusive and reported as counts.
Power: at the timed drains' rate (2 events in 14 draws that reached the context) an arm
would expect about 11 events in 100 waves; at 3-5% per draw, 3-4, so a null is likely
and decides little.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from bench.results import iter_jsonl
from experiments.benchcert import control_waves
from experiments.benchcert.rescore import stop as stop_server

PORT = 30083
SEED = 20261002
WAVES = 100  # per arm
SIZES = (8, 16)  # wave size range, inclusive
STAGGER_S = 2.0
OSL = 512
TARGET, POSITION, WRONG = '579ae7ce', 439, 1756
PARTNER = 'session_000527'  # emits 1756 legitimately; finished just before the event
PARTNER_LEAD_S = (0.5, 1.5)
PAUSE_S = 1.5  # after each wave: the ring writes after a second without a verify
# (variant, half of the waves); arms: stock, cert0 (the certified graphs with MAX_ROWS=0:
# the head never runs), cert (one half plain, one half with the device ring).
BLOCKS = (('stock', 0), ('cert0', 0), ('cert', 0), ('certring', 1), ('cert0', 1), ('stock', 1))
ARMS = {
    'stock': (('stock', 0), ('stock', 1)),
    'cert0': (('cert0', 0), ('cert0', 1)),
    'cert': (('cert', 0), ('certring', 1)),
}
HOOK_DIR = Path(__file__).resolve().parent / 'replay_hook'


def plan_waves(
    keys: list[str], partner: int, seed: int = SEED, n: int = WAVES
) -> list[dict[str, Any]]:
    """The waves: size, members (target first; the partner second in even waves) and start
    delays. Random neighbours never include the partner, so the strata stay clean."""
    target = next(i for i, k in enumerate(keys) if k.startswith(TARGET))
    others = [i for i in range(len(keys)) if i not in (target, partner)]
    rng = random.Random(seed)
    waves = []
    for w in range(n):
        size = rng.randint(*SIZES)
        with_partner = w % 2 == 0
        fixed = [target, partner] if with_partner else [target]
        members = [*fixed, *rng.sample(others, size - len(fixed))]
        delays = [round(rng.uniform(0, STAGGER_S), 3) for _ in members]
        if with_partner:
            delays[0] = round(rng.uniform(PARTNER_LEAD_S[1], STAGGER_S), 3)
            delays[1] = round(delays[0] - rng.uniform(*PARTNER_LEAD_S), 3)
        waves.append(
            {'wave': w, 'size': size, 'partner': with_partner, 'members': members, 'delays': delays}
        )
    return waves


def partner_index(runs: Path, keys: list[str]) -> int:
    """session_000527's prompt, by its conversation id in session 1's stock c = 64 point."""
    import gzip

    from bench.results import prompt_hash

    point = control_waves.source_point(runs, 'mtp64')
    with gzip.open(point / 'aiperf/profile_export_raw.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('metadata', {}).get('conversation_id') == PARTNER:
                messages = record.get('payload', {}).get('messages') or [{}]
                return keys.index(prompt_hash(messages[-1].get('content', '')))
    raise SystemExit(f'{PARTNER} not in {point}')


def block_waves(waves: list[dict[str, Any]], half: int) -> list[dict[str, Any]]:
    mid = len(waves) // 2
    return waves[:mid] if half == 0 else waves[mid:]


def generate(ids: list[int]) -> list[int]:
    body = {
        'input_ids': ids,
        'sampling_params': {'max_new_tokens': OSL, 'temperature': 0.0, 'ignore_eos': True},
    }
    request = urllib.request.Request(
        f'http://127.0.0.1:{PORT}/generate',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return list(json.loads(response.read())['output_ids'])


def flush() -> None:
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/flush_cache', data=b'')
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def run_wave(wave: dict[str, Any], items: list[tuple[str, list[int]]]) -> dict[str, Any]:
    """Flush the cache, start each member after its delay in its own thread, and return
    the target's output."""
    flush()
    outputs: dict[int, list[int] | str] = {}

    def one(index: int, delay: float) -> None:
        time.sleep(delay)
        try:
            outputs[index] = generate(items[index][1])
        except Exception as exc:  # recorded; the wave is then incomplete
            outputs[index] = repr(exc)

    threads = [
        threading.Thread(target=one, args=(i, d), daemon=True)
        for i, d in zip(wave['members'], wave['delays'], strict=True)
    ]
    t0 = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    target = wave['members'][0]
    out = outputs.get(target)
    errors = [i for i, o in outputs.items() if not isinstance(o, list)]
    return {
        **wave,
        'seconds': round(time.time() - t0, 2),
        'errors': errors,
        'target_output': out if isinstance(out, list) else None,
    }


def block_dir(out: Path, variant: str, half: int) -> Path:
    """One block's directory: its server record and its waves."""
    return out / f'block{half}' / variant


def waves_block(out: Path, variant: str, half: int, runs: Path) -> int:
    items = control_waves.prompts(runs, 'mtp64')
    keys = [key for key, _ in items]
    planned = plan_waves(keys, partner_index(runs, keys))
    target = block_dir(out, variant, half) / 'waves.jsonl'
    with target.open('w') as handle:
        for wave in block_waves(planned, half):
            handle.write(json.dumps(run_wave(wave, items)) + '\n')
            handle.flush()
            time.sleep(PAUSE_S)
    return 0


def start(block: Path, variant: str) -> int:
    """A fresh server for one block, as control_waves starts its mtp64 servers; certring
    adds the device ring."""
    from bench.arms import resolve_arm
    from bench.server import Server
    from experiments.benchcert import plan

    base = {'stock': 'stock', 'cert0': 'cert0'}.get(variant, 'cert')
    env = control_waves.variant_env('mtp64', base, block / 'certified_stats.json')
    if variant == 'certring':
        env['PYTHONPATH'] = str(HOOK_DIR)
        env['BENCHCERT_RING'] = str(block / 'ring')
    arm = resolve_arm(plan.FAMILIES['mtp'].arm, control_waves.overrides('mtp64'), env)
    server = Server(arm, block / 'server', PORT, sglang_worktree=plan.ENGINE_WORKTREE)
    server.start()
    try:
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (block / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def classify(output: list[int] | None, ref: list[int]) -> str:
    if output is None or len(output) <= POSITION:
        return 'incomplete'
    if output[:POSITION] != ref[:POSITION]:
        return 'other_prefix'
    return 'wrong' if output[POSITION] == WRONG else str(output[POSITION])


def first_difference(output: list[int] | None, ref: list[int]) -> int | None:
    """The first output position before POSITION where the wave left session 1's text."""
    if output is None:
        return None
    return next((i for i in range(min(len(output), POSITION)) if output[i] != ref[i]), None)


def compare(out: Path, runs: Path) -> int:
    """Per arm: waves reaching the context, and the token each committed at 439."""
    from experiments.benchcert.drain import point_dirs, target_record

    control_point = next(p for n, p in point_dirs(runs / 'drain', runs) if n.startswith('s1/'))
    record = target_record(control_point)
    assert record is not None
    ref = record['output']
    rule = (__doc__ or '').split('Reading rule')[1].strip()
    summary: dict[str, Any] = {'declared': False, 'reading_rule': rule}
    for arm, blocks in ARMS.items():
        strata: dict[str, dict[str, int]] = {'with_partner': {}, 'without_partner': {}}
        left: dict[str, dict[str, int]] = {'with_partner': {}, 'without_partner': {}}
        for variant, half in blocks:
            path = block_dir(out, variant, half) / 'waves.jsonl'
            if not path.exists():
                continue
            for wave in iter_jsonl(path):
                counts = strata['with_partner' if wave.get('partner') else 'without_partner']
                key = classify(wave.get('target_output'), ref)
                counts[key] = counts.get(key, 0) + 1
                where = first_difference(wave.get('target_output'), ref)
                if where is not None:
                    side = left['with_partner' if wave.get('partner') else 'without_partner']
                    side[str(where)] = side.get(str(where), 0) + 1
        summary[arm] = {
            name: {
                'tokens_at_439': counts,
                'reached_context': sum(
                    v for k, v in counts.items() if k not in ('incomplete', 'other_prefix')
                ),
                'wrong': counts.get('wrong', 0),
                'left_session_1_text_at': left[name],
            }
            for name, counts in strata.items()
        }
    (out / 'compare.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('stock', 'cert') if k in summary}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'waves', 'stop'):
        p = sub.add_parser(name)
        p.add_argument('--variant', choices=('stock', 'cert', 'cert0', 'certring'), required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
        p.add_argument('--block', type=int, choices=(0, 1), required=True)
    c = sub.add_parser('compare')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    args = parser.parse_args(argv)
    if args.command == 'compare':
        return compare(args.out, args.runs)
    block = block_dir(args.out, args.variant, args.block)
    block.mkdir(parents=True, exist_ok=True)
    if args.command == 'start':
        return start(block, args.variant)
    if args.command == 'waves':
        return waves_block(args.out, args.variant, args.block, args.runs)
    return stop_server(block)


if __name__ == '__main__':
    sys.exit(main())
