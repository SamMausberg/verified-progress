"""How far BF16-sized rounding moves the gross positions in FP32 (CPU).

    python -m experiments.bf16_paths.perturb --targets TARGETS --out FILE [--site residual|gdn] \
        [--seeds 8] [--threads 8]

The transformers and SGLang readings show which BF16 implementations miss FP32 at
579ae7ce/439 and a4db11ff/333, but not how sensitive the positions themselves are. This
script measures that sensitivity in FP32, where the arithmetic adds almost no error of its
own: one forward over the prompt and output[:position] (the `fp32_full` path of
`paths.py fp32`), repeated with every decoder layer's output multiplied elementwise by
(1 + u), u drawn uniformly from [-2^-8, 2^-8] with a fixed seed per run. 2^-8 bounds the
relative error of rounding to BF16 (round to nearest, 8 significand bits; values just above
a power of two come close to it), so each run perturbs the residual stream by relative errors
of the same order as storing it in BF16 after every layer, at every position of the prefix
(`--site residual`). It is not an upper or lower bound on any particular rounding: each draw
is uniform within the bound, while a real rounding error is fixed by the value rounded. The
first runs used 2^-9, half that bound (superseded). Projections average such elementwise
noise down, so it understates the effect of rounding a quantity the GDN recurrence uses
directly; `--site gdn` instead multiplies the GDN core's inputs (query, key, value and beta,
all BF16 in SGLang and in transformers' BF16 model) by (1 + u) in every GDN layer. The
unperturbed forward is repeated first and must match `paths.py fp32`'s within 0.001 nats on
the tracked tokens.

Readings (set before the run): the spread of the target's tracked logprobs over the seeds
says how much of a BF16 implementation's miss these positions explain by themselves. A
spread of several nats (1756 at 579ae7ce/439 or 18299 at a4db11ff/333 moving by more than
2 nats between seeds) means the positions are ill-conditioned at BF16 rounding scale, so a
BF16 path missing FP32 there by a comparable amount says nothing specific about its
kernels; a spread well under 1 nat means they are not, and a 9-14 nat miss needs an
explanation beyond rounding. The positions before the target are the control. Random draws
can show sensitivity but do not bound it: a small spread over eight draws is evidence of good
conditioning in the sampled directions, not a worst case over every in-bound rounding pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.benchcert.paths import MODEL, REVISION, TRACE
from experiments.bf16_paths.hf_paths import entry

EPS = 2.0**-8  # the relative rounding-error bound of BF16 (8 significand bits)


def perturb_hook(generator: Any) -> Any:
    import torch

    def hook(module: Any, args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        noise = torch.rand(hidden.shape, generator=generator, dtype=hidden.dtype)
        perturbed = hidden * (1 + (2 * noise - 1) * EPS)
        return (perturbed, *output[1:]) if isinstance(output, tuple) else perturbed

    return hook


def perturb_gdn_inputs(rule: Any, generator: Any) -> Any:
    """Wrap a GDN layer's `chunk_gated_delta_rule` so its query, key, value and beta are
    multiplied by (1 + u) elementwise before the recurrence."""
    import torch

    def jitter(x: Any) -> Any:
        noise = torch.rand(x.shape, generator=generator, dtype=torch.float32)
        return (x.float() * (1 + (2 * noise - 1) * EPS)).to(x.dtype)

    def wrapped(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs['beta'] = jitter(kwargs['beta'])
        return rule(jitter(query), jitter(key), jitter(value), *args, **kwargs)

    return wrapped


def install(model: Any, site: str, generator: Any) -> list[Any]:
    """Install the perturbation; returns the callables that undo it."""
    undo: list[Any] = []
    for layer in model.model.layers:
        if site == 'residual':
            undo.append(layer.register_forward_hook(perturb_hook(generator)).remove)
            continue
        attn = getattr(layer, 'linear_attn', None)
        if attn is None:
            continue
        original = attn.chunk_gated_delta_rule
        attn.chunk_gated_delta_rule = perturb_gdn_inputs(original, generator)
        undo.append(
            lambda attn=attn, original=original: setattr(attn, 'chunk_gated_delta_rule', original)
        )
    if not undo:
        raise SystemExit(f'no layer to perturb at site {site}')
    return undo


def run(
    targets: Path, out: Path, site: str, seeds: int, threads: int, fp32_file: Path | None
) -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision('highest')
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation='eager',
        output_loading_info=True,
    )
    missing = [k for k in info.get('missing_keys', []) if not k.startswith('lm_head')]
    if missing:
        raise SystemExit(f'{len(missing)} weights missing from the model: {missing[:5]}')
    model.eval()
    found = [json.loads(line) for line in targets.read_text().splitlines() if line]
    if not found:
        raise SystemExit(f'no targets in {targets}')
    reference = (
        {r['id']: r['paths']['fp32_full'] for r in json.loads(fp32_file.read_text())['results']}
        if fp32_file
        else {}
    )
    results = []
    for target in found:
        prompt, output, pos = target['prompt_ids'], target['output_ids'], target['position']
        n, first = len(prompt), max(0, pos - TRACE)
        track = sorted({*target['track'], *output[first : pos + 1]})
        ids = torch.tensor([prompt + output[:pos]])
        runs: dict[str, Any] = {}
        for seed in [None, *range(seeds)]:
            undo = []
            if seed is not None:
                undo = install(model, site, torch.Generator().manual_seed(seed))
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, use_cache=False).logits[0]
            finally:
                for step in undo:
                    step()
            runs['none' if seed is None else f'seed_{seed}'] = {
                str(p): entry(logits[n + p - 1], output[p], track) for p in range(first, pos + 1)
            }
            at = runs['none' if seed is None else f'seed_{seed}'][str(pos)]
            print(
                target['id'],
                seed,
                at['top1'],
                {t: round(at['tracked'][str(t)], 3) for t in target['track']},
                flush=True,
            )
        if target['id'] in reference:
            ref = reference[target['id']][str(pos)]['tracked']
            gap = max(
                abs(runs['none'][str(pos)]['tracked'][str(t)] - ref[str(t)])
                for t in target['track']
            )
            if gap > 1e-3:
                raise SystemExit(
                    f'{target["id"]}: unperturbed forward differs from fp32.json by {gap}'
                )
        results.append(
            {'id': target['id'], 'position': pos, 'track': target['track'], 'runs': runs}
        )
    meta = {
        'model': MODEL,
        'revision': REVISION,
        'dtype': 'float32',
        'device': 'cpu',
        'attn_implementation': 'eager',
        'transformers': transformers.__version__,
        'torch': torch.__version__,
        'site': site,
        'perturbation': {
            'residual': 'each decoder layer output times (1 + u), u ~ U[-2^-8, 2^-8], per element',
            'gdn': 'each GDN layer core input (query, key, value, beta) times (1 + u), per element',
        }[site],
        'eps': EPS,
        'seeds': seeds,
        'trace': TRACE,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps({'meta': meta, 'results': results}, indent=1) + '\n')
    tmp.replace(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--targets', type=Path, required=True, help="paths.py's targets.jsonl")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--site', choices=('residual', 'gdn'), default='residual')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument(
        '--fp32', type=Path, help="paths.py's fp32.json, to check the unperturbed run"
    )
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.threads < 1:
        raise SystemExit('--seeds and --threads must be positive')
    if args.out.exists():
        raise SystemExit(f'{args.out} exists')
    return run(args.targets, args.out, args.site, args.seeds, args.threads, args.fp32)


if __name__ == '__main__':
    sys.exit(main())
