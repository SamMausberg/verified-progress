"""A position's stock distribution along batch-1 processing paths, and in FP32 (after h8).

    python -m experiments.benchcert.paths targets --out DIR [--runs RUNS]   # CPU
    python -m experiments.benchcert.paths stock --out DIR --url URL         # stock server up
    python -m experiments.benchcert.paths fp32 --out DIR                    # no server; CPU

h8's reference step (`fallback_stress refs`) read 579ae7ce's distribution at output position
439 on a stock plain-decoding server, one request at a time: three prefills put 1756 at
-8.1 to -8.9 nats, while decoding from position 400 put it on top at -0.32. These commands
repeat and extend that reading, and add a reference that does not depend on the engine's
BF16 kernels.

`targets` writes the positions to read: 579ae7ce's 439 in session 1's certified MTP c = 64
point, and every gross event of the h6s scores (score_report's gross events, which include
a4db11ff's 333 in the plain c = 128 points), each with the prompt ids, the output ids up to
and including the position, and the tokens to track (the committed token, the scorer's
top-1, and for 579ae7ce the near-tied trio and the planted 9471). Identical contexts are
merged.

`stock` reads each target on the running server (drain.py `start`: `plain-tuned`, radix
cache off), one request at a time, flushing the cache before each:
- prefill paths: prompt + output[:end] for the ends in `prefill_ends` (for 579ae7ce,
  439, 501 and 512 output tokens: absolute 514, 576 and 587, as h8 read them), with the
  logprobs of every position from TRACE positions before the target to the target;
- the decode path: prompt + output[:position - TRACE], then greedy decoding to the target,
  with each step's logprobs and whether the decode stayed on the recorded text.

`fp32` loads the model in FP32 on the CPU with Hugging Face transformers (eager attention; with no
flash-linear-attention or causal-conv1d installed, the GDN layers use transformers' torch
implementations; TF32 off) and reads the same positions twice: one full forward over
prompt + output[:position], and a forward over prompt + output[:position - TRACE]
followed by one token at a time through the recurrent cache. FP32 at about 1e-7 relative
precision decides which BF16 path is accurate where they differ by nats.

Readings (set before the run): if FP32's full and recurrent paths agree and side with the
stock prefills (1756 far below the trio), the stock GDN decode path carries the error at
this row, a stock-engine numerics fault independent of the certified head; if they side
with the stock decode path (1756 on top), the prefill and scorer references are the
inaccurate ones and the h6s "gross" label at this row inverts; if FP32's own two paths
disagree by more than 0.1 nats, the row is ill-conditioned even in FP32 and neither BF16
path can be called accurate. The trace shows where the paths part: a gap that grows after
the request's first end of text points to state instability, a jump at one position to a
kernel or chunk boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

TRACE = 39  # positions read before each target (h8's decode started 39 before 439)
WRONG, TRIO, PLANTED = 1756, (68189, 8078, 5715), 9471
MODEL, REVISION = 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def prefill_ends(position: int, output_len: int) -> list[int]:
    """The prefill lengths (in output tokens) that h8 read for 579ae7ce: the target, 62
    more, and the whole output less its last token (absolute 514, 576 and 587)."""
    return sorted({position, min(position + 62, output_len - 1), output_len - 1})


def targets(out: Path, runs: Path) -> list[dict[str, Any]]:
    """579ae7ce/439 and every h6s gross event, merged by identical context."""
    from experiments.benchcert.drain import TARGET, point_dirs, requests
    from experiments.benchcert.score_report import gross_events, read_jsonl, score_file

    merged: dict[str, dict[str, Any]] = {}

    def add(name: str, item: dict[str, Any], position: int, tokens: list[int]) -> None:
        prompt, output = item['input'], item['output']
        key = json.dumps([prompt, output[:position]])
        entry = merged.setdefault(
            key,
            {
                'id': f'{name}|{item["prompt"]}|{position}',
                'prompt_ids': prompt,
                'output_ids': output,
                'position': position,
                'track': [],
                'points': [],
            },
        )
        entry['points'].append(name)
        entry['track'] = sorted({*entry['track'], *tokens})

    drain_out = out
    points = point_dirs(drain_out, runs)
    s1_name, s1_point = points[0]
    item = next(
        r for r in requests(s1_point) if r['phase'] == 'profiling' and r['prompt'].startswith(TARGET[0])
    )
    add(s1_name, item, TARGET[1], [item['output'][TARGET[1]], WRONG, *TRIO, PLANTED])
    for name, point in points:
        path = score_file(drain_out, name)
        if not path.exists():
            continue
        records = read_jsonl(path)
        events = gross_events(name, records)
        if not events:
            continue
        items = requests(point)
        for event in events:
            found = items[event['request_index']]
            if found.get('prompt') != event['prompt'] or 'output' not in found:
                continue
            add(name, found, event['position'], [event['token'], event['top1']])
    return list(merged.values())


def post(url: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.loads(response.read())
        return result


def flush(url: str) -> None:
    request = urllib.request.Request(f'{url}/flush_cache', data=b'')
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def position_entry(top: Any, tracked: Any, token: int | None) -> dict[str, Any]:
    """One position: top-5, the tracked tokens' logprobs, the recorded token's logprob."""
    top5 = [[float(e[0]), int(e[1])] for e in top or [] if e and e[0] is not None]
    lps = {int(e[1]): float(e[0]) for e in tracked or [] if e and e[0] is not None}
    return {
        'top1': top5[0][1] if top5 else None,
        'top1_logprob': top5[0][0] if top5 else None,
        'top5': top5,
        'tracked': {str(k): v for k, v in sorted(lps.items())},
        'token': token,
        'token_logprob': lps.get(token) if token is not None else None,
    }


def stock_target(url: str, target: dict[str, Any]) -> dict[str, Any]:
    prompt, output, pos = target['prompt_ids'], target['output_ids'], target['position']
    n, first = len(prompt), max(0, pos - TRACE)
    track = sorted({*target['track'], *output[first : pos + 1]})
    common = {'return_logprob': True, 'top_logprobs_num': 5, 'token_ids_logprob': track}
    paths: dict[str, Any] = {}
    full = output
    for end in prefill_ends(pos, len(output)):
        flush(url)
        body = {
            'input_ids': prompt + full[:end],
            'sampling_params': {'max_new_tokens': 1, 'temperature': 0.0},
            'logprob_start_len': n + first - 1,
            **common,
        }
        meta = post(f'{url}/generate', body)['meta_info']
        tops, ids = meta.get('input_top_logprobs') or [], meta.get('input_token_ids_logprobs') or []
        trace = {}
        for p in range(first, min(end, pos + 1)):
            j = p - first + 1  # entry 0 is position n + first - 1 (left empty)
            if j < len(tops):
                trace[str(p)] = position_entry(tops[j], ids[j] if j < len(ids) else None, full[p])
        if end == pos:
            trace[str(pos)] = position_entry(
                (meta.get('output_top_logprobs') or [None])[0],
                (meta.get('output_token_ids_logprobs') or [None])[0],
                full[pos],
            )
        paths[f'prefill_{n + end}'] = trace
    flush(url)
    body = {
        'input_ids': prompt + output[:first],
        'sampling_params': {'max_new_tokens': pos - first + 1, 'temperature': 0.0},
        **common,
    }
    response = post(f'{url}/generate', body)
    meta = response['meta_info']
    generated = list(response.get('output_ids') or [])
    off_text = next((first + i for i, t in enumerate(generated[: pos - first]) if t != output[first + i]), None)
    tops = meta.get('output_top_logprobs') or []
    ids = meta.get('output_token_ids_logprobs') or []
    trace = {
        str(first + i): position_entry(tops[i], ids[i] if i < len(ids) else None, output[first + i])
        for i in range(min(len(tops), pos - first + 1))
    }
    paths[f'decode_from_{first}'] = trace
    return {
        'id': target['id'],
        'points': target['points'],
        'position': pos,
        'decode_left_text_at': off_text,
        'decode_token_at_target': generated[pos - first] if len(generated) > pos - first else None,
        'paths': paths,
    }


def stock(out: Path, url: str) -> int:
    found = [json.loads(line) for line in (out / 'targets.jsonl').read_text().splitlines() if line]
    results = [stock_target(url, t) for t in found]
    (out / 'stock.json').write_text(json.dumps(results, indent=1) + '\n')
    for r in results:
        pos = str(r['position'])
        summary = {k: (v.get(pos) or {}).get('tracked') for k, v in r['paths'].items()}
        print(r['id'], 'decode left text at', r['decode_left_text_at'], json.dumps(summary))
    return 0


def fp32(out: Path, device: str = 'cpu', threads: int = 8) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(threads)
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation='eager',
        output_loading_info=True,
    )
    missing = [k for k in info.get('missing_keys', []) if not k.startswith('lm_head')]
    if missing:
        raise SystemExit(f'{len(missing)} weights missing from the FP32 model: {missing[:5]}')
    model = model.to(device)
    model.eval()
    found = [json.loads(line) for line in (out / 'targets.jsonl').read_text().splitlines() if line]
    results = []
    for target in found:
        prompt, output, pos = target['prompt_ids'], target['output_ids'], target['position']
        n, first = len(prompt), max(0, pos - TRACE)
        ids = torch.tensor([prompt + output[:pos]], device=device)
        track = sorted({*target['track'], *output[first : pos + 1]})

        def entry(logits: Any, p: int) -> dict[str, Any]:
            lp = torch.log_softmax(logits.double(), dim=-1)
            top = torch.topk(lp, 5)
            top5 = [[float(v), int(i)] for v, i in zip(top.values, top.indices, strict=True)]
            return {
                'top1': top5[0][1],
                'top1_logprob': top5[0][0],
                'top5': top5,
                'tracked': {str(t): float(lp[t]) for t in track},
                'token': output[p],
                'token_logprob': float(lp[output[p]]),
            }

        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
            full = {str(p): entry(logits[n + p - 1], p) for p in range(first, pos + 1)}
            step = model(input_ids=ids[:, : n + first], use_cache=True)
            cache = step.past_key_values
            recurrent = {str(first): entry(step.logits[0, -1], first)}
            for p in range(first + 1, pos + 1):
                step = model(input_ids=ids[:, n + p - 1 : n + p], past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                recurrent[str(p)] = entry(step.logits[0, -1], p)
        # A sanity check of the load: the full forward's top-1 should follow the recorded
        # text at most traced positions before the target.
        agree = sum(full[str(p)]['top1'] == output[p] for p in range(first, pos))
        results.append(
            {
                'id': target['id'],
                'position': pos,
                'top1_follows_text': [agree, pos - first],
                'paths': {'fp32_full': full, f'fp32_recurrent_from_{first}': recurrent},
            }
        )
        print(target['id'], json.dumps({k: v[str(pos)]['tracked'] for k, v in results[-1]['paths'].items()}))
    meta = {'model': MODEL, 'revision': REVISION, 'dtype': 'float32', 'torch': torch.__version__}
    (out / 'fp32.json').write_text(json.dumps({'meta': meta, 'results': results}, indent=1) + '\n')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('targets')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--drain', type=Path, default=Path.home() / 'vp-data/benchcert/drain')
    t.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    s = sub.add_parser('stock')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--url', required=True)
    f = sub.add_parser('fp32')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--device', default='cpu')
    f.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'targets':
        found = targets(args.drain, args.runs)
        (args.out / 'targets.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in found))
        print(f'{len(found)} targets: ' + ', '.join(t['id'] for t in found))
        return 0
    if args.command == 'stock':
        return stock(args.out, args.url)
    return fp32(args.out, args.device, args.threads)


if __name__ == '__main__':
    sys.exit(main())
