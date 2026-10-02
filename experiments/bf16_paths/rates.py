"""How often each BF16 path misses FP32's top token, over whole outputs (selection-free).

    python -m experiments.bf16_paths.rates prompts --out DIR [--count 16]          # CPU
    python -m experiments.bf16_paths.rates sglang --out DIR --url URL              # server up
    python -m experiments.bf16_paths.rates hf --out DIR [--state-dtype float32]    # GPU, BF16
    python -m experiments.bf16_paths.rates fp32 --out DIR [--threads 8]            # CPU
    python -m experiments.bf16_paths.rates summary --out DIR --json FILE           # CPU

579ae7ce/439 and a4db11ff/333 were found because an SGLang path erred there, so a reading
at those two positions cannot say whether SGLang's BF16 arithmetic errs more often than
another BF16 implementation: every implementation may have its own rare positions. This
module compares the paths on positions nobody selected:

- `prompts`: the first `--count` distinct prompts, by prompt hash, of session 1's plain
  c = 128 point (the workload a4db11ff came from); with `--eot-before N`, only prompts whose
  recorded output there ends its text before position N, so that a long enough decode
  (`--output-len`) reaches text written after the end-of-text token, where both events lie;
- `sglang`: on a stock server, each prompt decoded greedily (`--output-len` tokens, 512 by
  default) at batch 1 with ignore_eos, as the benchmark did (the decode path, top-20
  logprobs per step), then the prompt and that output prefilled in one request (the
  prefill path, top-20 per position);
- `hf`: transformers' BF16 model (eager attention) over the same text, one forward
  (prefill path) and token by token through the cache (decode path): with its torch GDN
  implementation (FP32 inside the recurrence) and the cached state in FP32 or BF16, and
  with flash-linear-attention's Triton kernels (run with `fla` on PYTHONPATH) and an FP32
  state;
- `fp32`: transformers in FP32 on the CPU, one forward over the same text: per position its
  top-20 and its logprob of every token in any path's top-20;
- `summary`: per path, the positions where FP32's logprob of the path's top-1 falls short
  of FP32's top logprob by more than 2 nats (the decision) and by 0.05-5 nats (secondary),
  before and after the output's first end-of-text token; the path's absolute logprob error
  on FP32's top-1 token (mean, 99th percentile, maximum; secondary); the worst positions.

Readings (set before the run; `decide`): events are positions where FP32's logprob of a
path's top-1 falls more than 2 nats short of FP32's top, pooled over decode and prefill (a
position missed on both paths counted once, a correction made after both runs; it changes
neither verdict).
The comparator is the transformers configuration with more events among the two with an
FP32 state (torch GDN, fla GDN). SGLang-specific: SGLang at least 5 events, at least 3 times
the comparator's, and a one-sided exact binomial p below 0.05 for SGLang's share of the two
counts under equal rates; then SGLang's BF16 arithmetic is less accurate than either
transformers implementation at this model, and 579ae7ce/439 is an instance of that. Not
specific: at least 10 events in the two counts and SGLang at most 1.5 times the
comparator. Anything else is inconclusive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

from experiments.benchcert.paths import MODEL, REVISION, flush, post

OUTPUT_LEN = 512
TOP = 20
# The decision uses 2 nats; the finer thresholds and the logprob differences are secondary.
THRESHOLDS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
DECISION_NATS = 2.0
EOT = (248044, 248046)  # <|endoftext|>, <|im_end|>
POINT_GLOB = 's1/plain-tuned/2026*/r0/c128'
HF_RUNS = ('hf_bf16_float32state', 'hf_bf16_modelstate', 'hf_bf16_fla_float32state')
# The comparators of the decision: both transformers GDN implementations, FP32 cached state.
COMPARATORS = ('hf_bf16_float32state', 'hf_bf16_fla_float32state')


def sources(out: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Every BF16 source by prompt: SGLang's paths and the transformers runs (the two
    comparators required, the BF16-state run when present)."""
    found = {'sglang': read_jsonl(out / 'sglang.jsonl.gz')}
    for name in HF_RUNS:
        file = out / f'{name}.jsonl.gz'
        if not file.exists():
            if name in COMPARATORS:
                raise SystemExit(f'{file} missing')
            continue
        found[name] = read_jsonl(file)
    prompts = [r['prompt'] for r in found['sglang']]
    for name, rows in found.items():
        if [r['prompt'] for r in rows] != prompts:
            raise SystemExit(f'{name}: prompts differ from sglang.jsonl.gz')
    return {name: {r['prompt']: r for r in rows} for name, rows in found.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt') if path.suffix == '.gz' else tmp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompts(
    out: Path, runs: Path, count: int, output_len: int, eot_before: int | None = None
) -> int:
    """The first `count` distinct prompts by hash; with `eot_before`, only prompts whose
    recorded output in that point has an end of text before that position (a property of
    the benchmark's own output, not of any reading here)."""
    from experiments.benchcert.drain import requests

    points = sorted(runs.glob(POINT_GLOB))
    if len(points) != 1:
        raise SystemExit(f'expected one point at {runs}/{POINT_GLOB}, found {len(points)}')
    found: dict[str, list[int]] = {}
    for item in requests(points[0]):
        if item['phase'] != 'profiling' or 'input' not in item:
            continue
        if eot_before is not None and not any(
            t in EOT for t in item.get('output', [])[:eot_before]
        ):
            continue
        found.setdefault(item['prompt'], item['input'])
    chosen = sorted(found)[:count]
    if len(chosen) < count:
        raise SystemExit(f'only {len(chosen)} distinct prompts qualify in {points[0]}')
    write_jsonl(
        out / 'prompts.jsonl',
        [{'prompt': h, 'prompt_ids': found[h], 'output_len': output_len} for h in chosen],
    )
    print(f'{len(chosen)} prompts from {points[0]} (end of text before {eot_before})')
    return 0


def tops(entries: Any) -> list[list[float | int]]:
    return [[float(e[0]), int(e[1])] for e in entries or [] if e and e[0] is not None]


def sglang(out: Path, url: str) -> int:
    rows = []
    for item in read_jsonl(out / 'prompts.jsonl'):
        prompt = item['prompt_ids']
        common = {'return_logprob': True, 'top_logprobs_num': TOP}
        flush(url)
        response = post(
            f'{url}/generate',
            {
                'input_ids': prompt,
                'sampling_params': {
                    'max_new_tokens': item['output_len'],
                    'temperature': 0.0,
                    'ignore_eos': True,
                },
                **common,
            },
        )
        output = list(response['output_ids'])
        decode = [tops(e) for e in response['meta_info']['output_top_logprobs']]
        if len(output) != item['output_len'] or len(decode) != item['output_len']:
            raise SystemExit(f'{item["prompt"]}: {len(output)} tokens, {len(decode)} logprob steps')
        flush(url)
        meta = post(
            f'{url}/generate',
            {
                'input_ids': prompt + output[:-1],
                'sampling_params': {'max_new_tokens': 1, 'temperature': 0.0},
                'logprob_start_len': len(prompt) - 1,
                **common,
            },
        )['meta_info']
        # Entry j of the input logprobs predicts output position j - 1 (entry 0 is the
        # prompt's last token); the last position comes from the one generated token.
        inputs = meta['input_top_logprobs']
        prefill = [tops(inputs[p + 1]) for p in range(len(output) - 1)]
        prefill.append(tops(meta['output_top_logprobs'][0]))
        rows.append({**item, 'output_ids': output, 'decode': decode, 'prefill': prefill})
        print(
            item['prompt'],
            'decode/prefill top-1 agree at',
            sum(d[0][1] == q[0][1] for d, q in zip(decode, prefill, strict=True)),
            'of',
            len(output),
            flush=True,
        )
    write_jsonl(out / 'sglang.jsonl.gz', rows)
    return 0


def top_entries(logits: Any) -> list[list[float | int]]:
    import torch

    lp = torch.log_softmax(logits.double(), dim=-1)
    top = torch.topk(lp, TOP)
    return [[float(v), int(i)] for v, i in zip(top.values, top.indices, strict=True)]


def load_model(dtype: Any, device: str, state_dtype: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    from experiments.bf16_paths.hf_paths import keep_state_in_fp32

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if state_dtype == 'float32':
        keep_state_in_fp32()
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, attn_implementation='eager', output_loading_info=True
    )
    missing = [k for k in info.get('missing_keys', []) if not k.startswith('lm_head')]
    if missing:
        raise SystemExit(f'{len(missing)} weights missing from the model: {missing[:5]}')
    return model.to(device).eval()


def hf(out: Path, state_dtype: str) -> int:
    import torch

    from experiments.bf16_paths.hf_paths import gdn_kernels

    model = load_model(torch.bfloat16, 'cuda', state_dtype)
    kernels = gdn_kernels(model)
    fla = kernels['chunk'].startswith('fla.')
    print('GDN kernels', kernels, flush=True)
    rows = []
    for item in read_jsonl(out / 'sglang.jsonl.gz'):
        prompt, output = item['prompt_ids'], item['output_ids']
        n = len(prompt)
        ids = torch.tensor([prompt + output[:-1]], device='cuda')
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0]
            prefill = [top_entries(logits[n + p - 1]) for p in range(len(output))]
            step = model(input_ids=ids[:, :n], use_cache=True)
            decode = [top_entries(step.logits[0, -1])]
            for p in range(1, len(output)):
                step = model(
                    input_ids=ids[:, n + p - 1 : n + p],
                    past_key_values=step.past_key_values,
                    use_cache=True,
                )
                decode.append(top_entries(step.logits[0, -1]))
        rows.append({'prompt': item['prompt'], 'decode': decode, 'prefill': prefill})
        print(item['prompt'], 'hf done', flush=True)
    write_jsonl(out / f'hf_bf16_{"fla_" if fla else ""}{state_dtype}state.jsonl.gz', rows)
    return 0


def fp32(out: Path, threads: int) -> int:
    import torch

    torch.set_num_threads(threads)
    model = load_model(torch.float32, 'cpu', 'model')
    paths = sources(out)
    rows = []
    for item in paths['sglang'].values():
        prompt, output = item['prompt_ids'], item['output_ids']
        n = len(prompt)
        with torch.no_grad():
            lp = torch.log_softmax(
                model(input_ids=torch.tensor([prompt + output[:-1]]), use_cache=False)
                .logits[0, n - 1 :]
                .double(),
                dim=-1,
            )
        positions = []
        for p in range(len(output)):
            seen = {
                int(e[1])
                for by_prompt in paths.values()
                for kind in ('decode', 'prefill')
                for e in by_prompt[item['prompt']][kind][p]
            }
            top = torch.topk(lp[p], TOP)
            positions.append(
                {
                    'top': [
                        [float(v), int(t)] for v, t in zip(top.values, top.indices, strict=True)
                    ],
                    'lp': {str(t): float(lp[p, t]) for t in sorted(seen)},
                }
            )
        rows.append({'prompt': item['prompt'], 'positions': positions})
        print(item['prompt'], 'fp32 done', flush=True)
    write_jsonl(out / 'fp32.jsonl.gz', rows)
    return 0


def decide(event_positions: dict[str, int]) -> dict[str, Any]:
    """The rule declared before the run (module docstring, "Readings"), on the number of
    positions where a source's decode or prefill path (or both) misses by more than 2 nats.
    A position missed on both paths counts once: the two paths share the prompt, the
    position and most of the arithmetic, so they are not independent events."""
    sglang_events = event_positions['sglang']
    by_comparator = {name: event_positions[name] for name in COMPARATORS}
    comparator = max(by_comparator, key=lambda name: by_comparator[name])
    hf_events = by_comparator[comparator]
    total = sglang_events + hf_events
    p_value = sum(math.comb(total, k) for k in range(sglang_events, total + 1)) / 2**total
    if sglang_events >= 5 and sglang_events >= 3 * hf_events and p_value < 0.05:
        verdict = 'sglang-specific'
    elif total >= 10 and sglang_events <= 1.5 * hf_events:
        verdict = 'not specific'
    else:
        verdict = 'inconclusive'
    return {
        'threshold_nats': DECISION_NATS,
        'counted': 'positions missed on either path, each position once',
        'sglang_events': sglang_events,
        'transformers_events': by_comparator,
        'comparator': comparator,
        'binomial_p_one_sided': round(p_value, 6),
        'verdict': verdict,
    }


def summary(out: Path) -> dict[str, Any]:
    by_source = sources(out)
    sg = list(by_source['sglang'].values())
    ref = {r['prompt']: r['positions'] for r in read_jsonl(out / 'fp32.jsonl.gz')}
    if sorted(ref) != sorted(by_source['sglang']):
        raise SystemExit('fp32.jsonl.gz does not cover the same prompts')
    regions = ('before_eot', 'after_eot')
    counts: dict[str, Any] = {}
    worst: dict[str, list[dict[str, Any]]] = {}
    event_positions: dict[str, set[tuple[str, int]]] = {}
    totals = dict.fromkeys(regions, 0)
    for item in sg:
        output = item['output_ids']
        eot = next((i for i, t in enumerate(output) if t in EOT), len(output))
        for p in range(len(output)):
            totals[regions[p > eot]] += 1
    for name, by_prompt in by_source.items():
        for kind in ('decode', 'prefill'):
            path = f'{name}/{kind}'
            table = {r: dict.fromkeys(map(str, THRESHOLDS), 0) for r in regions}
            disagree = dict.fromkeys(regions, 0)
            differences: list[float] = []
            missed = event_positions.setdefault(name, set())
            events = []
            for item in sg:
                output = item['output_ids']
                eot = next((i for i, t in enumerate(output) if t in EOT), len(output))
                for p, entries in enumerate(by_prompt[item['prompt']][kind]):
                    fp = ref[item['prompt']][p]
                    top1 = int(entries[0][1])
                    regret = fp['top'][0][0] - fp['lp'][str(top1)]
                    region = regions[p > eot]
                    disagree[region] += top1 != fp['top'][0][1]
                    for t in THRESHOLDS:
                        table[region][str(t)] += regret > t
                    if regret > DECISION_NATS:
                        missed.add((item['prompt'], p))
                    own = {int(e[1]): float(e[0]) for e in entries}
                    if fp['top'][0][1] in own:
                        differences.append(abs(own[fp['top'][0][1]] - fp['top'][0][0]))
                    if regret > THRESHOLDS[0]:
                        events.append(
                            {
                                'prompt': item['prompt'],
                                'position': p,
                                'region': region,
                                'path_top1': top1,
                                'fp32_top1': fp['top'][0][1],
                                'regret': round(regret, 3),
                            }
                        )
            ranked = sorted(differences)
            counts[path] = {
                'regret_above': table,
                'top1_differs': disagree,
                'fp32_top1_logprob_abs_diff': {
                    'positions': len(ranked),
                    'mean': round(sum(ranked) / len(ranked), 5) if ranked else None,
                    'p99': round(ranked[int(0.99 * (len(ranked) - 1))], 5) if ranked else None,
                    'max': round(ranked[-1], 5) if ranked else None,
                },
            }
            worst[path] = sorted(events, key=lambda e: -e['regret'])[:10]
    return {
        'model': MODEL,
        'revision': REVISION,
        'prompts': [r['prompt'] for r in sg],
        'positions': totals,
        'thresholds_nats': list(THRESHOLDS),
        'counts': counts,
        'positions_missed_above_2_nats': {
            name: sorted([prompt, position] for prompt, position in found)
            for name, found in event_positions.items()
        },
        'decision': decide({name: len(found) for name, found in event_positions.items()}),
        'worst': worst,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prompts')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--output-len', type=int, default=OUTPUT_LEN)
    p.add_argument(
        '--eot-before',
        type=int,
        help='only prompts whose recorded output ends its text before this position',
    )
    s = sub.add_parser('sglang')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--url', required=True)
    h = sub.add_parser('hf')
    h.add_argument('--out', type=Path, required=True)
    h.add_argument('--state-dtype', choices=('model', 'float32'), default='float32')
    f = sub.add_parser('fp32')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--threads', type=int, default=8)
    m = sub.add_parser('summary')
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--json', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'prompts':
        if args.count < 1:
            raise SystemExit('--count must be positive')
        if args.output_len < 2 or (args.eot_before is not None and args.eot_before < 1):
            raise SystemExit('--output-len must be at least 2 and --eot-before positive')
        return prompts(args.out, args.runs, args.count, args.output_len, args.eot_before)
    if args.command == 'sglang':
        return sglang(args.out, args.url)
    if args.command == 'hf':
        return hf(args.out, args.state_dtype)
    if args.command == 'fp32':
        return fp32(args.out, args.threads)
    result = summary(args.out)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + '\n')
    for path, c in result['counts'].items():
        print(path, json.dumps(c))
    return 0


if __name__ == '__main__':
    sys.exit(main())
