"""Explain divergences between two tapped runs at the tensor level.

Input: two directories written by tap_runs.py (A = reference, B = other), each
holding client.jsonl and tap/<rid>/<forward>_<mode>.npz with per-module hashes
of every row of a tapped request. For each tapped prompt:

1. Committed rows. Every absolute sequence position q is processed by exactly
   one row on the committed path (the prefill row, the decode step, or the
   verify row whose input token equals the committed token and whose earlier
   rows in the cycle were accepted). Rejected-draft rows are ignored.
2. First difference. Walking positions in order, the first q where any module
   output hash differs between A and B, and at that q the first module in
   execution order. Before that point every output bit agrees, including all
   cached state, so this module's kernel is where the runs part.
3. At the token divergence (output index d; the logits come from row P+d-1):
   whether the logits fed to argmax are bitwise equal, whether the head input
   (final norm output) is bitwise equal, and the exact (float64) logits of the
   two competing tokens recomputed from each run's head input and the BF16
   head weights. Classes:

   tie_rule        logits bitwise equal, different token (tie handling differs)
   head_gemm       head input equal, logits differ (LM-head GEMM accumulation)
   order_flip      head inputs differ and the exact logit order of the two
                   tokens differs between the runs
   rounding_flip   head inputs differ, the exact order is the same in both
                   runs, and BF16 rounding of the head output creates the tie
                   or reversal in one of them
   order_unresolved  an exact gap is within twice the FP32 accumulation bound,
                   so the float64 recomputation cannot stand in for the kernel
   upstream_order_unknown  head inputs differ but a run did not save the head
                   input of that row (first tap version, MTP verify rows)

    python experiments/state_safety/mechanism.py --a tap/plain_c1 --b tap/mtp_s3_c1 \
        --out evidence/state_safety/mechanism_plain_c1_vs_mtp_s3_c1.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = (
    Path.home()
    / '.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots'
    / '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)
HEAD_WEIGHT = 'model.language_model.embed_tokens.weight'


class Head:
    """Rows of the tied BF16 LM head, loaded lazily from the checkpoint."""

    def __init__(self) -> None:
        from safetensors import safe_open

        index = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text())
        self.f = safe_open(str(MODEL_DIR / index['weight_map'][HEAD_WEIGHT]), 'pt')
        self.w = self.f.get_slice(HEAD_WEIGHT)
        self.cache: dict[int, np.ndarray] = {}

    def row(self, tok: int) -> np.ndarray:
        if tok not in self.cache:
            self.cache[tok] = self.w[tok : tok + 1].float().numpy()[0].astype(np.float64)
        return self.cache[tok]

    def exact(self, h: np.ndarray, tok: int) -> float:
        return float(np.dot(self.row(tok), h.astype(np.float64)))


def load_client(d: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in (d / 'client.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('tapped'):
            out[r['id']] = r
    return out


def committed_rows(tap_dir: Path, seq: list[int]) -> dict[int, dict[str, Any]]:
    """Map absolute position -> the committed row's data."""
    rows: dict[int, dict[str, Any]] = {}
    for f in sorted(tap_dir.glob('*.npz')):
        z = np.load(f)
        ids, pos = z['input_ids'], z['positions']
        mode = f.stem.split('_', 1)[1]
        n = len(ids)
        ok_prefix = True
        for r in range(n):
            q = int(pos[r])
            match = q < len(seq) and int(ids[r]) == seq[q]
            if mode == 'TARGET_VERIFY':
                ok_prefix = ok_prefix and match
                if not ok_prefix:
                    break
            elif not match:
                continue
            rec: dict[str, Any] = {
                'file': f.name,
                'mode': mode,
                'batch_size': int(z['batch_size']),
                'num_tokens': int(z['num_tokens']),
                'hash': z['hashes_tok'][:, r],
                'hash_rows': z['hash_rows'],
            }
            # Logits for this row: extend keeps only the last row of the request.
            lrow = None if mode == 'EXTEND' and r != n - 1 else (0 if mode == 'EXTEND' else r)
            if lrow is not None and 'top16_ids' in z:
                rec['top16_ids'] = z['top16_ids'][lrow]
                rec['top16_values'] = z['top16_values'][lrow]
                rec['argmax'] = int(z['argmax'][lrow])
                full_rows = list(z['full_logits_rows']) if 'full_logits_rows' in z else []
                if lrow in full_rows:
                    rec['full_logits'] = z['full_logits'][full_rows.index(lrow)]
                if 'head_in' in z:
                    hi = z['head_in']
                    if len(hi) == n:  # decode and verify: one row per token
                        rec['head_in'] = hi[r]
                    elif r == n - 1:  # prefill, or a verify from the first tap
                        rec['head_in'] = hi[-1]  # version, which kept the last row only
            rows[q] = rec
    return rows


# Execution order inside a decoder layer (children before the parent module).
_LAYER_ORDER = [
    'input_layernorm',
    'input_layernorm#1',
    'linear_attn.in_proj_qkvz',
    'linear_attn.in_proj_ba',
    'linear_attn.gdn_conv',
    'linear_attn.gdn_core',
    'linear_attn.attn',
    'linear_attn.attn#1',
    'linear_attn.norm',
    'linear_attn.out_proj',
    'linear_attn',
    'qkv_proj',
    'attn',
    'attn#1',
    'o_proj',
    'post_attention_layernorm',
    'post_attention_layernorm#1',
    'mlp.gate_up_proj',
    'mlp.act_fn',
    'mlp.down_proj',
    'mlp',
    '',
]


def exec_key(name: str) -> tuple[int, int]:
    """Sort key that follows the model's execution order."""
    if name.startswith('model.embed_tokens'):
        return (-1, 0)
    if name.startswith('model.layers.'):
        rest = name[len('model.layers.') :]
        idx, _, suffix = rest.partition('.')
        if '#' in idx:  # the layer module's own extra outputs
            idx, extra = idx.split('#', 1)
            suffix = '#' + extra
        rank = _LAYER_ORDER.index(suffix) if suffix in _LAYER_ORDER else 50
        return (int(idx), rank)
    tail = ['model.norm', 'model.norm#1', 'model', 'logits_processor']
    return (10_000, tail.index(name) if name in tail else 50)


def first_hash_difference(
    ra: dict[int, dict[str, Any]],
    rb: dict[int, dict[str, Any]],
    names_a: list[str],
    names_b: list[str],
    upto: int,
) -> dict[str, Any] | None:
    """First (position, module) whose output bits differ; modules matched by name."""
    common = sorted(set(names_a) & set(names_b), key=exec_key)
    ia = np.array([names_a.index(n) for n in common])
    ib = np.array([names_b.index(n) for n in common])
    for q in range(upto + 1):
        if q not in ra or q not in rb:
            continue
        a, b = ra[q], rb[q]
        ha = np.array([a['hash'][i] if i < len(a['hash']) else 0 for i in ia])
        hb = np.array([b['hash'][i] if i < len(b['hash']) else 0 for i in ib])
        # Token-indexed slots written in both forwards.
        wa = np.array(
            [
                a['hash_rows'][i] >= max(1, a['num_tokens']) if i < len(a['hash_rows']) else False
                for i in ia
            ]
        )
        wb = np.array(
            [
                b['hash_rows'][i] >= max(1, b['num_tokens']) if i < len(b['hash_rows']) else False
                for i in ib
            ]
        )
        diff = np.nonzero((ha != hb) & wa & wb)[0]
        if len(diff):
            k = int(diff[0])  # `common` is already in execution order
            return {
                'position': q,
                'module': common[k],
                'modules_differing_at_position': len(diff),
                'mode_a': a['mode'],
                'mode_b': b['mode'],
                'batch_a': a['batch_size'],
                'batch_b': b['batch_size'],
                'tokens_a': a['num_tokens'],
                'tokens_b': b['num_tokens'],
            }
    return None


def analyse_prompt(
    pid: str,
    ca,
    cb,
    dir_a: Path,
    dir_b: Path,
    names: tuple[list[str], list[str]],
    head: Head,
    prompt,
):
    P = len(prompt)
    oa, ob = ca['output_ids'], cb['output_ids']
    n = min(len(oa), len(ob))
    d = next((i for i in range(n) if oa[i] != ob[i]), None)
    ra = committed_rows(dir_a / 'tap' / f'tap-{pid}', prompt + oa)
    rb = committed_rows(dir_b / 'tap' / f'tap-{pid}', prompt + ob)
    upto = P + (d if d is not None else n) - 1
    out: dict[str, Any] = {'id': pid, 'prompt_len': P, 'diverged_at': d}
    out['first_difference'] = first_hash_difference(ra, rb, names[0], names[1], upto)
    if out['first_difference'] is not None:
        fd = out['first_difference']
        fd['output_index'] = fd['position'] - P + 1 if fd['position'] >= P else None
    if d is None:
        out['cls'] = 'no_divergence'
        return out
    q = P + d - 1
    ta, tb = oa[d], ob[d]
    a, b = ra.get(q), rb.get(q)
    if a is None or b is None or 'top16_ids' not in a or 'top16_ids' not in b:
        out['cls'] = 'missing_rows'
        return out
    out['argmax_a'], out['argmax_b'] = a['argmax'], b['argmax']

    def logit(rec, tok):
        if 'full_logits' in rec:
            return float(rec['full_logits'][tok])
        ids = list(rec['top16_ids'])
        return float(rec['top16_values'][ids.index(tok)]) if tok in ids else None

    la = (logit(a, ta), logit(a, tb))
    lb = (logit(b, ta), logit(b, tb))
    out['bf16_logits_a'] = la
    out['bf16_logits_b'] = lb
    logits_equal = (
        bool(np.array_equal(a['full_logits'], b['full_logits']))
        if 'full_logits' in a and 'full_logits' in b
        else None
    )
    out['logits_bitwise_equal'] = logits_equal
    na, nb = names
    head_equal = bool(a['hash'][na.index('model.norm')] == b['hash'][nb.index('model.norm')])
    out['head_input_bitwise_equal'] = head_equal
    if logits_equal:
        out['cls'] = 'tie_rule'
        return out
    if head_equal:
        out['cls'] = 'head_gemm'
        return out
    if a.get('head_in') is None or b.get('head_in') is None:
        out['cls'] = 'upstream_order_unknown'
        return out
    ea = (head.exact(a['head_in'], ta), head.exact(a['head_in'], tb))
    eb = (head.exact(b['head_in'], ta), head.exact(b['head_in'], tb))
    out['exact_logits_a'] = ea
    out['exact_logits_b'] = eb
    out['head_input_max_abs_diff'] = float(np.max(np.abs(a['head_in'] - b['head_in'])))
    # The float64 recomputation stands in for cuBLAS's FP32 accumulator, which
    # can differ from it by about 2*D*2**-24 relative to sum |w*h|. Treat exact
    # gaps inside that bound as unresolved rather than as an order.
    bound = max(
        accumulation_bound(head, a['head_in'], (ta, tb)),
        accumulation_bound(head, b['head_in'], (ta, tb)),
    )
    out['accumulation_bound'] = bound
    ga, gb = ea[0] - ea[1], eb[0] - eb[1]
    if min(abs(ga), abs(gb)) <= 2 * bound:
        out['cls'] = 'order_unresolved'
    elif np.sign(ga) != np.sign(gb):
        out['cls'] = 'order_flip'
    else:
        out['cls'] = 'rounding_flip'
    return out


def accumulation_bound(head: Head, h: np.ndarray, toks: tuple[int, int]) -> float:
    d = h.shape[0]
    return max(2 * d * 2.0**-24 * float(np.sum(np.abs(head.row(t) * h))) for t in toks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--prompts', default=str(Path.home() / 'vp-data/state/prompts/prompts.jsonl'))
    ap.add_argument('--out', required=True)
    ap.add_argument(
        '--untapped-a',
        help='matrix run of configuration A at the same concurrency (jsonl); checks that '
        'the tap leaves tokens and logprobs bitwise unchanged',
    )
    args = ap.parse_args()
    dir_a, dir_b = Path(args.a), Path(args.b)
    names = json.loads((dir_a / 'tap' / 'slots.json').read_text())['names']
    names_b = json.loads((dir_b / 'tap' / 'slots.json').read_text())['names']
    prompts = {
        json.loads(line)['id']: json.loads(line)['input_ids']
        for line in Path(args.prompts).read_text().splitlines()
    }
    ca, cb = load_client(dir_a), load_client(dir_b)
    head = Head()
    cases = []
    for pid in sorted(ca.keys() & cb.keys()):
        cases.append(
            analyse_prompt(
                pid, ca[pid], cb[pid], dir_a, dir_b, (names, names_b), head, prompts[pid]
            )
        )
    fd = [c['first_difference'] for c in cases if c.get('first_difference')]
    summary = {
        'a': args.a,
        'b': args.b,
        'prompts': len(cases),
        'classes': dict(Counter(c['cls'] for c in cases)),
        'first_difference_module': dict(Counter(f['module'] for f in fd).most_common(20)),
        'first_difference_modes': dict(Counter(f'{f["mode_a"]} vs {f["mode_b"]}' for f in fd)),
        'first_difference_output_index': dict(
            Counter(
                'prompt' if f['output_index'] is None else str(f['output_index']) for f in fd
            ).most_common(10)
        ),
        'no_hash_difference_before_divergence': sum(
            1 for c in cases if c['diverged_at'] is not None and not c.get('first_difference')
        ),
    }
    if args.untapped_a:
        ref = {
            json.loads(line)['id']: json.loads(line)
            for line in Path(args.untapped_a).read_text().splitlines()
        }
        same = [
            pid
            for pid, r in ca.items()
            if r['output_ids'] == ref[pid]['output_ids'][: len(r['output_ids'])]
            and r['top_logprobs'] == ref[pid]['top_logprobs'][: len(r['top_logprobs'])]
        ]
        summary['tap_check'] = {
            'untapped_run': args.untapped_a,
            'prompts': len(ca),
            'tokens_and_logprobs_bitwise_equal': len(same),
        }
    Path(args.out).write_text(json.dumps({'summary': summary, 'cases': cases}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
