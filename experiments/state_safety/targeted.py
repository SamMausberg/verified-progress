"""Targeted state tests for speculative decoding on the hybrid GDN model.

Each test starts one server configuration (see server.CONFIGS), runs its cases
at concurrency 1 unless stated, and writes one JSON file with every case and a
summary. Tests:

  truncation  max_new_tokens ending at every offset inside a verify cycle;
              the output must equal the prefix of an untruncated run on the
              same server (same batch shape, so equality is exact).
  stops       stop_token_ids hit at every offset inside a verify cycle (drafts
              after the stop were accepted and committed to the GDN state);
              same exactness check, then a multi-turn extension of the stopped
              conversation served warm (radix cache) and cold (after a flush).
  prefix      long generations cross the GDN checkpoint boundaries (every
              mamba_track_interval tokens) during decode. Prefixes that end
              just past a boundary are re-served warm (restoring the tracked
              GDN state) and cold (full prefill), including checkpoints taken
              in a cycle that also finished the request (truncated inside it).
  abort       requests aborted mid-stream while the batch and the GDN pool are
              full, so the next request reuses the freed slot; the probe
              outputs are compared with the same probes served alone.
  prefill     per-position input logprobs of long prompts, for comparing
              chunked-prefill sizes offline (compare_prefill.py).

Run under the shared GPU lock, e.g.

    scripts/gpu_lock.sh -s python experiments/state_safety/targeted.py \
        truncation --config mtp_s3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import generate, load_prompts
from compare import classify, infer_ulp, margin
from server import CONFIGS, MODEL, MODEL_REVISION, flush_cache, git_sha, launch_with_retry
from server import sglang_source_dir as sglang_dir

REPO = Path(__file__).resolve().parents[2]
TRACK_INTERVAL = 256  # SGLang default --mamba-track-interval (page size 1)
TIMEOUT = aiohttp.ClientTimeout(total=None, sock_read=600)


def diff(
    ref: dict[str, Any], test: dict[str, Any], expect_len: int | None = None
) -> dict[str, Any]:
    """First divergence of test against ref, with both runs' margins there."""
    a, b = ref['output_ids'], test['output_ids']
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    out: dict[str, Any] = {'identical': d is None and len(a) == len(b), 'len_ref': len(a)}
    out['len_test'] = len(b)
    if expect_len is not None:
        out['identical'] = d is None and len(b) == expect_len
    if d is None:
        return out
    out['pos'] = d
    ta, tb = ref.get('top_logprobs') or [], test.get('top_logprobs') or []
    if d < len(ta) and d < len(tb):
        ma, lba = margin(ta[d], a[d], b[d])
        mb, lbb = margin(tb[d], b[d], a[d])
        ulp = infer_ulp(ta[d], tb[d])
        out.update(margin_ref=ma, margin_test=mb, ulp=ulp, cls=classify(ma, mb, ulp, lba or lbb))
    return out


def cycles_of(rec: dict[str, Any]) -> list[tuple[int, int]]:
    """(first output position, length) of every commit (prefill token, then cycles)."""
    out, pos = [], 0
    for n, _ in rec['chunks']:
        out.append((pos, n))
        pos += n
    return out


# ---------------------------------------------------------------- truncation / stops


async def run_truncation(url: str, prompts: list[dict[str, Any]], args: argparse.Namespace):
    rng = random.Random(0)
    cases = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
        for p in prompts[: args.num_prompts]:
            flush_cache(url)
            full = await generate(s, url, p['input_ids'], args.full_len)
            by_offset: dict[int, list[int]] = {}
            for start, n in cycles_of(full)[1:]:
                for j in range(1, n + 1):
                    by_offset.setdefault(j, []).append(start + j)
            for j, ms in sorted(by_offset.items()):
                m = rng.choice(ms)
                if m >= len(full['output_ids']):
                    continue
                flush_cache(url)
                rec = await generate(s, url, p['input_ids'], m)
                d = diff(full, rec, expect_len=m)
                # Ref prefix check: the truncated run must be the first m tokens.
                d['identical'] = rec['output_ids'] == full['output_ids'][:m]
                cases.append(
                    {
                        'id': p['id'],
                        'max_new_tokens': m,
                        'kept_in_cycle': j,
                        'finish_reason': rec['finish_reason'],
                        **d,
                    }
                )
    ok = sum(c['identical'] for c in cases)
    by_j: dict[int, list[int]] = {}
    for c in cases:
        by_j.setdefault(c['kept_in_cycle'], [0, 0])
        by_j[c['kept_in_cycle']][0] += 1
        by_j[c['kept_in_cycle']][1] += c['identical']
    summary = {
        'cases': len(cases),
        'identical': ok,
        'by_tokens_kept_in_final_cycle': {
            str(k): {'cases': v[0], 'identical': v[1]} for k, v in by_j.items()
        },
        'finish_not_length': sum(
            1 for c in cases if (c['finish_reason'] or {}).get('type') != 'length'
        ),
    }
    return {'cases': cases, 'summary': summary}


def _extension_ids(tok_ids: list[int], turn2: list[int]) -> list[int]:
    return tok_ids + turn2


async def run_stops(url: str, prompts: list[dict[str, Any]], args: argparse.Namespace):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    # Close the stopped assistant turn and ask for more, as a chat client would.
    turn2 = tok.encode(
        '<|im_end|>\n<|im_start|>user\nContinue, and add one more detail.<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n',
        add_special_tokens=False,
    )
    rng = random.Random(1)
    cases = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
        for p in prompts[: args.num_prompts]:
            flush_cache(url)
            full = await generate(s, url, p['input_ids'], args.full_len)
            out = full['output_ids']
            first_seen: dict[int, int] = {}
            for i, t in enumerate(out):
                first_seen.setdefault(t, i)
            by_offset: dict[int, list[int]] = {}
            for start, n in cycles_of(full)[1:]:
                for j in range(1, n + 1):
                    pos = start + j - 1
                    if pos < len(out) and first_seen[out[pos]] == pos:
                        by_offset.setdefault(j, []).append(pos)
            for j, positions in sorted(by_offset.items()):
                pos = rng.choice(positions)
                stop_tok = out[pos]
                flush_cache(url)
                rec = await generate(
                    s, url, p['input_ids'], args.full_len, stop_token_ids=[stop_tok]
                )
                visible = full['output_ids'][: pos + 1]
                ok = rec['output_ids'] == visible
                ext_ids = _extension_ids(p['input_ids'] + rec['output_ids'], turn2)
                warm = await generate(s, url, ext_ids, args.ext_len)
                flush_cache(url)
                cold = await generate(s, url, ext_ids, args.ext_len)
                cases.append(
                    {
                        'id': p['id'],
                        'stop_pos': pos,
                        'stop_token': stop_tok,
                        'stop_index_in_cycle': j,
                        'finish_reason': rec['finish_reason'],
                        'stop_output_identical': ok,
                        'stop_len': len(rec['output_ids']),
                        'warm_cached_tokens': warm['cached_tokens'],
                        'extension_warm_vs_cold': diff(cold, warm),
                    }
                )
    summary = {
        'cases': len(cases),
        'stop_output_identical': sum(c['stop_output_identical'] for c in cases),
        'stop_inside_block': sum(1 for c in cases if c['stop_index_in_cycle'] < args.block),
        'extension_identical': sum(c['extension_warm_vs_cold']['identical'] for c in cases),
        'extension_divergence_classes': _count(
            c['extension_warm_vs_cold'].get('cls')
            for c in cases
            if not c['extension_warm_vs_cold']['identical']
        ),
        'by_stop_index_in_cycle': _group(cases, 'stop_index_in_cycle', 'stop_output_identical'),
    }
    return {'cases': cases, 'summary': summary}


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def _group(cases: list[dict[str, Any]], key: str, flag: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for c in cases:
        g = out.setdefault(str(c[key]), {'cases': 0, 'ok': 0})
        g['cases'] += 1
        g['ok'] += bool(c[flag])
    return out


# ---------------------------------------------------------------- prefix reuse


async def run_prefix(url: str, prompts: list[dict[str, Any]], args: argparse.Namespace):
    """Checkpointed GDN states from decode, restored for new requests."""
    chosen = [p for p in prompts if p['thinking']][: args.num_prompts]
    cases = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
        for p in chosen:
            P = len(p['input_ids'])
            flush_cache(url)
            a = await generate(s, url, p['input_ids'], args.long_len, ignore_eos=True)
            seq = p['input_ids'] + a['output_ids']
            bounds = [
                t
                for t in range(TRACK_INTERVAL, len(seq) - args.ext_len, TRACK_INTERVAL)
                if t > P + 1
            ]
            lengths = sorted({b + d for b in bounds for d in (0, 1, 3, 37)})
            warm = {}
            for L in lengths:
                warm[L] = await generate(s, url, seq[:L], args.ext_len)
            flush_cache(url)
            for L in lengths:
                cold = await generate(s, url, seq[:L], args.ext_len)
                own = {'output_ids': seq[L : L + args.ext_len], 'top_logprobs': []}
                cases.append(
                    {
                        'id': p['id'],
                        'kind': 'decode_checkpoint',
                        'prefix_len': L,
                        'boundary': L - (L % TRACK_INTERVAL),
                        'warm_cached_tokens': warm[L]['cached_tokens'],
                        'cold_cached_tokens': cold['cached_tokens'],
                        'warm_vs_cold': diff(cold, warm[L]),
                        'warm_vs_original': diff(own, warm[L]),
                    }
                )
            # A checkpoint taken in the cycle that also finished the request:
            # truncate inside the boundary-crossing cycle, then reuse.
            for b in bounds:
                cyc = _crossing_cycle(a, P, b)
                if cyc is None:
                    continue
                start, n = cyc
                for m in range(start + 1, start + n):
                    if P + m - 1 >= b:
                        break
                    flush_cache(url)
                    trunc = await generate(s, url, p['input_ids'], m)
                    L = b + 1
                    w = await generate(s, url, seq[:L], args.ext_len)
                    flush_cache(url)
                    c = await generate(s, url, seq[:L], args.ext_len)
                    cases.append(
                        {
                            'id': p['id'],
                            'kind': 'checkpoint_in_finishing_cycle',
                            'boundary': b,
                            'max_new_tokens': m,
                            'truncated_identical': trunc['output_ids'] == a['output_ids'][:m],
                            'prefix_len': L,
                            'warm_cached_tokens': w['cached_tokens'],
                            'warm_vs_cold': diff(c, w),
                        }
                    )
    summary: dict[str, Any] = {}
    for kind in ('decode_checkpoint', 'checkpoint_in_finishing_cycle'):
        ks = [c for c in cases if c['kind'] == kind]
        summary[kind] = {
            'cases': len(ks),
            'warm_vs_cold_identical': sum(c['warm_vs_cold']['identical'] for c in ks),
            'divergence_classes': _count(
                c['warm_vs_cold'].get('cls') for c in ks if not c['warm_vs_cold']['identical']
            ),
            'warm_cache_hit_at_boundary': sum(
                1
                for c in ks
                if c['warm_cached_tokens'] is not None and c['warm_cached_tokens'] >= c['boundary']
            ),
        }
    return {'cases': cases, 'summary': summary}


def _crossing_cycle(rec: dict[str, Any], prompt_len: int, boundary: int) -> tuple[int, int] | None:
    """The cycle whose GDN state advance crosses `boundary` (materialized length)."""
    for start, n in cycles_of(rec)[1:]:
        # Before the cycle the model has consumed prompt_len + start - 1 tokens.
        before = prompt_len + start - 1
        if before < boundary <= before + n:
            return start, n
    return None


# ---------------------------------------------------------------- abort and slot reuse


async def run_abort(url: str, prompts: list[dict[str, Any]], args: argparse.Namespace):
    rng = random.Random(2)
    probes = [p for p in prompts if not p['thinking']][: args.num_prompts]
    victims = [p for p in prompts if p['thinking']]
    ref = {}
    async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
        flush_cache(url)
        for p in probes:
            ref[p['id']] = await generate(s, url, p['input_ids'], args.full_len)
        flush_cache(url)

        results: dict[str, Any] = {}
        aborts: list[dict[str, Any]] = []
        queue = list(probes)

        async def lane(k: int) -> None:
            while queue:
                v = victims[rng.randrange(len(victims))]
                after = rng.randint(1, args.abort_max_tokens)
                rid = f'victim-{k}-{time.monotonic_ns()}'
                vrec = await generate(
                    s,
                    url,
                    v['input_ids'],
                    1024,
                    rid=rid,
                    ignore_eos=True,
                    abort_after_tokens=after,
                )
                async with s.post(f'{url}/abort_request', json={'rid': rid}) as r:
                    await r.read()
                aborts.append({'victim': v['id'], 'after_tokens': len(vrec['output_ids'])})
                if not queue:
                    break
                p = queue.pop(0)
                results[p['id']] = await generate(s, url, p['input_ids'], args.full_len)

        await asyncio.gather(*(lane(k) for k in range(args.lanes)))
    cases = []
    for pid, rec in results.items():
        d = diff(ref[pid], rec)
        cases.append({'id': pid, **d})
    summary = {
        'probes': len(cases),
        'aborts': len(aborts),
        'identical': sum(c['identical'] for c in cases),
        'divergence_classes': _count(c.get('cls') for c in cases if not c['identical']),
        'abort_after_tokens_hist': _count(a['after_tokens'] for a in aborts),
    }
    return {'cases': cases, 'aborts': aborts, 'summary': summary}


# ---------------------------------------------------------------- prefill logprobs


async def run_prefill(url: str, prompts: list[dict[str, Any]], args: argparse.Namespace):
    long_prompts = sorted(prompts, key=lambda p: -len(p['input_ids']))[: args.num_prompts]
    records = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
        for p in long_prompts:
            flush_cache(url)
            payload = {
                'input_ids': p['input_ids'],
                'sampling_params': {'temperature': 0.0, 'max_new_tokens': args.full_len},
                'return_logprob': True,
                'logprob_start_len': 0,
                'top_logprobs_num': 5,
            }
            async with s.post(f'{url}/generate', json=payload) as r:
                out = await r.json()
            m = out['meta_info']
            records.append(
                {
                    'id': p['id'],
                    'input_len': len(p['input_ids']),
                    'input_top_logprobs': [
                        [[lp, t] for lp, t, *_ in (e or [])] for e in m['input_top_logprobs']
                    ],
                    'output_ids': out['output_ids'],
                    'top_logprobs': [
                        [[lp, t] for lp, t, *_ in (e or [])] for e in m['output_top_logprobs']
                    ],
                }
            )
    return {'records': records, 'summary': {'prompts': len(records)}}


TESTS = {
    'truncation': run_truncation,
    'stops': run_stops,
    'prefix': run_prefix,
    'abort': run_abort,
    'prefill': run_prefill,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('test', choices=sorted(TESTS))
    ap.add_argument('--config', required=True, help='name from server.CONFIGS')
    ap.add_argument('--extra-flags', default='', help='appended to the configuration flags')
    ap.add_argument('--tag', default='', help='suffix for the output name')
    ap.add_argument('--prompts', default=str(Path.home() / 'vp-data/state/prompts/prompts.jsonl'))
    ap.add_argument('--out-dir', default=str(Path.home() / 'vp-data/state/targeted'))
    ap.add_argument('--port', type=int, default=30052)
    ap.add_argument('--num-prompts', type=int, default=40)
    ap.add_argument('--full-len', type=int, default=160)
    ap.add_argument('--ext-len', type=int, default=48)
    ap.add_argument('--long-len', type=int, default=700)
    ap.add_argument('--lanes', type=int, default=4)
    ap.add_argument('--abort-max-tokens', type=int, default=60)
    ap.add_argument('--block', type=int, default=4, help='tokens per full verify cycle')
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    flags = CONFIGS[args.config] + args.extra_flags.split()
    name = f'{args.test}__{args.config}' + (f'__{args.tag}' if args.tag else '')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with launch_with_retry(flags, args.port, out_dir / f'{name}.server.log') as srv:
        result = asyncio.run(TESTS[args.test](srv['base_url'], prompts, args))
    result['meta'] = {
        'test': args.test,
        'config': args.config,
        'flags': flags,
        'args': {k: v for k, v in vars(args).items() if k != 'prompts'},
        'server_info': srv['server_info'],
        'model_revision': MODEL_REVISION,
        'repo_sha': git_sha(REPO),
        'sglang_dir': str(sglang_dir()),
        'sglang_sha': git_sha(sglang_dir()),
        'wall_s': round(time.time() - t0, 1),
    }
    (out_dir / f'{name}.json').write_text(json.dumps(result) + '\n')
    print(name, json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
