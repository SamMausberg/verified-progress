"""The gross positions along the prefill and decode paths in Hugging Face transformers.

    python -m experiments.bf16_paths.hf_paths --targets TARGETS --out FILE \
        [--dtype bfloat16|float32] [--device cuda|cpu] [--state-dtype model|float32]

`experiments/benchcert/paths.py` read two positions (579ae7ce/439 and a4db11ff/333) on
SGLang's stock BF16 server and in FP32 with transformers on the CPU. There SGLang's BF16
decode path and its BF16 prefill each miss FP32 by 9-14 nats, at different positions. This
script asks whether an independent BF16 implementation does the same: it loads the same
checkpoint in transformers (`Qwen3_5ForCausalLM` through `AutoModelForCausalLM`, eager
attention, TF32 off) and reads every target along the paths SGLang took:

- `prefill_<n>`: one forward over the first n tokens (prompt plus output), for the same
  ends as the SGLang prefills (`paths.prefill_ends`), reading positions TRACE before the
  target through the target;
- `decode_from_<k>`: one forward over the prompt and output[:k], then one token at a time
  through the model's recurrent cache to the target, on the recorded text.

The GDN layers use flash-linear-attention's Triton kernels when `fla` can be imported and
transformers' torch implementations otherwise; the output records which ran
(`meta.gdn_kernels`). transformers keeps the recurrent GDN state in the cache in the
model's dtype, so in BF16 it rounds the state to BF16 after every decode step, while
SGLang keeps it in FP32 for this model (`mamba_ssm_dtype` in its config).
`--state-dtype float32` keeps the cached state in FP32 instead, to match SGLang.

The target list is `paths.py targets`' `targets.jsonl`; each entry's tracked tokens, plus
the recorded tokens at the traced positions, are the tokens whose logprobs are written.

Readings (set before the run), on the target's tracked tokens and top-1 against FP32
(`paths.py fp32`): if any transformers BF16 run puts 1756 within 2 nats of its top-1 along
the decode path at 579ae7ce/439, or puts 18299 more than 5 nats below its top-1 along a
prefill at a4db11ff/333, an independent BF16 implementation reproduces SGLang's error, the
positions are ill-conditioned in BF16 and the error is not specific to SGLang. If every run
stays within 1 nat of FP32 on those tokens on both paths at both targets, while SGLang is
9-14 nats off, the error is specific to SGLang's kernels. Anything between is reported as
a partial BF16 sensitivity, not as an SGLang fault.

Qualified after the run (review of #222): agreement of transformers with FP32 at positions
selected because SGLang erred there shows that SGLang misses where transformers does not, not
that SGLang misses more often; `rates.py` compares the rates on unselected positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.benchcert.paths import MODEL, REVISION, TRACE, prefill_ends


def entry(logits: Any, token: int, track: list[int]) -> dict[str, Any]:
    """One position: top-10 and the tracked and recorded tokens' logprobs (log-softmax in
    FP64 over the model's logits, whatever their dtype)."""
    import torch

    lp = torch.log_softmax(logits.double(), dim=-1)
    top = torch.topk(lp, 10)
    top10 = [[float(v), int(i)] for v, i in zip(top.values, top.indices, strict=True)]
    return {
        'top1': top10[0][1],
        'top1_logprob': top10[0][0],
        'top10': top10,
        'tracked': {str(t): float(lp[t]) for t in track},
        'token': token,
        'token_logprob': float(lp[token]),
    }


def keep_state_in_fp32() -> None:
    """Make transformers' linear-attention cache layers hold the recurrent state in FP32.

    `LinearAttentionLayer.lazy_initialization` allocates the recurrent state in the dtype
    of the conv state (the model's dtype); this re-allocates it in FP32 right after, so
    `update_recurrent_state`'s copy keeps the kernel's FP32 state unrounded."""
    import torch
    from transformers import cache_utils

    layer_cls = cache_utils.LinearAttentionLayer
    original = layer_cls.lazy_initialization

    def lazy_initialization(
        self: Any, conv_states: Any = None, recurrent_states: Any = None
    ) -> None:
        original(self, conv_states=conv_states, recurrent_states=recurrent_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(
                recurrent_states, dtype=torch.float32, device=recurrent_states.device
            )

    layer_cls.lazy_initialization = lazy_initialization


def gdn_kernels(model: Any) -> dict[str, str]:
    """Which implementation each GDN entry point resolved to (the first GDN layer's)."""
    for layer in model.model.layers:
        attn = getattr(layer, 'linear_attn', None)
        if attn is None:
            continue
        return {
            'chunk': f'{attn.chunk_gated_delta_rule.__module__}.{attn.chunk_gated_delta_rule.__name__}',
            'recurrent': f'{attn.recurrent_gated_delta_rule.__module__}.'
            f'{attn.recurrent_gated_delta_rule.__name__}',
            'conv_prefill': 'causal_conv1d'
            if attn.causal_conv1d_fn is not None
            else 'torch.nn.Conv1d',
            'conv_update': f'{attn.causal_conv1d_update.__module__}.'
            f'{attn.causal_conv1d_update.__name__}',
            'gated_norm': type(attn.norm).__module__ + '.' + type(attn.norm).__name__,
        }
    raise SystemExit('no GDN layer found')


def recurrent_state_dtypes(cache: Any) -> list[str]:
    return sorted(
        {
            str(layer.recurrent_states.dtype)
            for layer in cache.layers
            if getattr(layer, 'is_recurrent_states_initialized', False)
        }
    )


def read_target(model: Any, target: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    prompt, output, pos = target['prompt_ids'], target['output_ids'], target['position']
    n, first = len(prompt), max(0, pos - TRACE)
    track = sorted({*target['track'], *output[first : pos + 1]})
    paths: dict[str, Any] = {}
    with torch.no_grad():
        for end in prefill_ends(pos, len(output)):
            ids = torch.tensor([prompt + output[:end]], device=device)
            logits = model(input_ids=ids, use_cache=False).logits[0]
            paths[f'prefill_{n + end}'] = {
                str(p): entry(logits[n + p - 1], output[p], track) for p in range(first, pos + 1)
            }
        ids = torch.tensor([prompt + output[:pos]], device=device)
        step = model(input_ids=ids[:, : n + first], use_cache=True)
        cache = step.past_key_values
        decode = {str(first): entry(step.logits[0, -1], output[first], track)}
        for p in range(first + 1, pos + 1):
            step = model(input_ids=ids[:, n + p - 1 : n + p], past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            decode[str(p)] = entry(step.logits[0, -1], output[p], track)
        paths[f'decode_from_{first}'] = decode
    return {
        'id': target['id'],
        'position': pos,
        'recurrent_state_dtype': recurrent_state_dtypes(cache),
        'paths': paths,
    }


def run(targets: Path, out: Path, dtype: str, device: str, state_dtype: str) -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if state_dtype == 'float32':
        keep_state_in_fp32()
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=getattr(torch, dtype),
        attn_implementation='eager',
        output_loading_info=True,
    )
    missing = [k for k in info.get('missing_keys', []) if not k.startswith('lm_head')]
    if missing:
        raise SystemExit(f'{len(missing)} weights missing from the model: {missing[:5]}')
    model = model.to(device)
    model.eval()
    found = [json.loads(line) for line in targets.read_text().splitlines() if line]
    if not found:
        raise SystemExit(f'no targets in {targets}')
    results = []
    for target in found:
        result = read_target(model, target, device)
        expected = [str(torch.float32 if state_dtype == 'float32' else getattr(torch, dtype))]
        if result['recurrent_state_dtype'] != expected:
            raise SystemExit(
                f'cached recurrent state is {result["recurrent_state_dtype"]}, not {expected}'
            )
        results.append(result)
        pos = str(result['position'])
        print(
            result['id'],
            result['recurrent_state_dtype'],
            json.dumps(
                {
                    name: {k: round(v, 3) for k, v in trace[pos]['tracked'].items()}
                    | {'top1': trace[pos]['top1']}
                    for name, trace in result['paths'].items()
                }
            ),
        )
    try:
        import fla

        fla_version = getattr(fla, '__version__', 'unknown')
    except ImportError:
        fla_version = None
    meta = {
        'model': MODEL,
        'revision': REVISION,
        'implementation': 'transformers',
        'transformers': transformers.__version__,
        'torch': torch.__version__,
        'fla': fla_version,
        'dtype': dtype,
        'device': device,
        'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
        'attn_implementation': 'eager',
        'tf32': False,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'state_dtype_requested': state_dtype,
        'gdn_kernels': gdn_kernels(model),
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
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--state-dtype', choices=('model', 'float32'), default='model')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists')
    return run(args.targets, args.out, args.dtype, args.device, args.state_dtype)


if __name__ == '__main__':
    sys.exit(main())
