"""Target-anchored residual evaluation on real DFlash blocks (P3 Stage B, decision level).

The target is the Hugging Face Qwen3.5-4B text model (BF16, pinned revision),
run block by block from an exact prefix cache. Every linear operator z = W x
(GDN in_proj_qkv/z/a/b and out_proj, attention q/k/v/o_proj, MLP gate/up/down,
and the tied LM head) is patched so that one forward can run in three modes:

- plain: z = W x (the exact target; optionally records operator inputs);
- anchor: z = W x, and the pass over the anchor block y0 keeps x-bar and
  z-bar = W x-bar for every operator and block position;
- repair: z~ = z-bar + (W U) U^T (x - x-bar), with a fixed orthonormal basis
  U (d_in x r) per operator input and precomputed W U; every nonlinearity,
  normalization, the causal convolution, the GDN recurrence (from the exact
  committed state) and attention (over the exact prefix KV) run exactly on
  the repaired values. Rank 0 reuses z-bar unchanged.

Blocks and anchors are real: y0 is a DFlash draft recorded by serve_probe.py
--mode probe (the engine's own verify pass decided its acceptance). The
cascading replays follow Sam's critique: y1 = F(y0) and y2 = F(y1) are the
exact Jacobi iterates (y_{k+1}[0] = y0[0], y_{k+1}[j] = argmax F(y_k)[j - 1]),
and the evaluator repairs each against the ORIGINAL anchor y0.

Development and held-out splits are by request (even and odd request index).
Bases come from the development split only: the leading principal components
of the exact input changes dx = x(y_k) - x(y0) of each operator, over both
replays and every changed position.

Held-out metrics per rank (the decision-level headline):

- replay 1 and replay 2: per position, whether argmax F~ equals argmax F, the
  certificate ratio R_i = 2 ||z~_i - z_i||_inf / m_i (m_i the exact top-1 minus
  top-2 logit gap; R_i < 1 guarantees the same decision), and the first
  disagreement after the first changed position;
- free-running repair: y_{k+1} = shift(argmax F~(y_k)) for k = 1..K, each
  iterate audited by an exact pass; accepted drafts after every sweep, and the
  per-position conditional failure hazards h_i of the audited candidates;
- the exact-Jacobi ceiling (the same loop with F instead of F~);
- operator-local accuracy on held-out exact changes,
  ||W (I - U U^T) dx|| / ||W dx|| (diagnostic only).

    python experiments/repair/residual_eval.py --run ~/vp-data/repair/runs/probe_b32 \\
        --requests ~/vp-data/repair/panel/checkpoints.jsonl --out ~/vp-data/repair/residual/b32
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAX_ROWS_PER_VARIANT = 32


def layer_groups(model) -> dict[tuple[int, str], list[torch.nn.Linear]]:
    groups: dict[tuple[int, str], list[torch.nn.Linear]] = {}
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, 'linear_attn'):
            m = layer.linear_attn
            groups[(li, 'mix')] = [m.in_proj_qkv, m.in_proj_z, m.in_proj_a, m.in_proj_b]
            groups[(li, 'mixout')] = [m.out_proj]
        else:
            m = layer.self_attn
            groups[(li, 'mix')] = [m.q_proj, m.k_proj, m.v_proj]
            groups[(li, 'mixout')] = [m.o_proj]
        groups[(li, 'mlp')] = [layer.mlp.gate_proj, layer.mlp.up_proj]
        groups[(li, 'down')] = [layer.mlp.down_proj]
    groups[(-1, 'head')] = [model.lm_head]
    return groups


class Anchored:
    """Mode switch and state for the patched linear operators."""

    def __init__(self, model):
        self.model = model
        self.groups = layer_groups(model)
        self.group_of: dict[int, tuple[int, str]] = {}
        self.mode = 'plain'
        self.capture = False
        self.rank = 0
        self.captured: dict[tuple[int, str], torch.Tensor] = {}
        self.ax: dict[tuple[int, str], torch.Tensor] = {}
        self.az: dict[int, torch.Tensor] = {}
        self.U: dict[tuple[int, str], torch.Tensor] = {}
        self.WU: dict[int, torch.Tensor] = {}
        for key, mods in self.groups.items():
            for mod in mods:
                self.group_of[id(mod)] = key
                mod.forward = self._make_forward(mod)

    def _make_forward(self, mod: torch.nn.Linear):
        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.linear(mod, x)

        return forward

    def linear(self, mod: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
        key = self.group_of[id(mod)]
        if self.mode == 'repair':
            dx = x[0].float() - self.ax[key]
            z = self.az[id(mod)]
            if self.rank > 0:
                z = z + (dx @ self.U[key][:, : self.rank]) @ self.WU[id(mod)][:, : self.rank].T
            if key[1] == 'head':
                return z[None]
            return z.to(x.dtype)[None]
        out = F.linear(x, mod.weight, mod.bias)
        if self.mode == 'anchor':
            self.ax[key] = x[0].float()
            self.az[id(mod)] = out[0].float()
        elif self.capture:
            self.captured[key] = x[0].float()
        return out

    def set_bases(self, bases: dict[tuple[int, str], torch.Tensor]) -> None:
        self.U = bases
        self.WU = {}
        for key, mods in self.groups.items():
            if key not in bases:
                continue
            u = bases[key]
            for mod in mods:
                w = mod.weight
                parts = [w[i : i + 32768].float() @ u for i in range(0, w.shape[0], 32768)]
                self.WU[id(mod)] = torch.cat(parts, dim=0)


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def top_basis(dx: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading principal directions of the rows of dx and all eigenvalues, largest first."""
    n, d = dx.shape
    r = min(rank, d, n)
    if n >= d:
        evals, evecs = torch.linalg.eigh(dx.T @ dx)
        return evecs[:, -r:].flip(-1).contiguous(), evals.flip(-1).clamp_min(0)
    evals, w = torch.linalg.eigh(dx @ dx.T)
    evals = evals.flip(-1).clamp_min(0)
    w = w.flip(-1)[:, :r]
    u = dx.T @ w / evals[:r].clamp_min(1e-30).sqrt()
    u, _ = torch.linalg.qr(u)
    return u.contiguous(), evals


def shift(y: list[int], logits: torch.Tensor) -> list[int]:
    t = logits.argmax(-1).tolist()
    return [y[0], *t[:-1]]


def accepted(y: list[int], logits: torch.Tensor) -> int:
    t = logits.argmax(-1).tolist()
    a = 0
    while a < len(y) - 1 and y[a + 1] == t[a]:
        a += 1
    return a


def first_change(y: list[int], y0: list[int]) -> int:
    for i, (a, b) in enumerate(zip(y, y0, strict=True)):
        if a != b:
            return i
    return len(y)


def build_cases(run: Path, requests: Path) -> list[dict[str, Any]]:
    reqs = [json.loads(line) for line in requests.read_text().splitlines() if line.strip()]
    results = [
        json.loads(line)
        for line in (run / 'results.jsonl').read_text().splitlines()
        if line.strip()
    ]
    by_id = {r['id']: r for r in reqs}
    by_rid: dict[str, list[dict[str, Any]]] = collections.OrderedDict()
    for line in (run / 'trace.jsonl').read_text().splitlines():
        rec = json.loads(line)
        by_rid.setdefault(rec['rid'], []).append(rec)
    rids = list(by_rid)[: len(results)]
    block = int(json.loads((run / 'run.json').read_text())['block'])
    cases = []
    for ri, (rid, res) in enumerate(zip(rids, results, strict=False)):
        prompt = by_id[res['id']]['input_ids']
        full = prompt + res['output_ids']
        for rec in sorted(by_rid[rid], key=lambda r: r['prefix']):
            p, a0 = rec['prefix'], rec['accept']
            if a0 >= block - 2 or p + block > len(full) or full[p] != rec['cand'][0]:
                continue
            cases.append(
                {
                    'request': ri,
                    'id': res['id'],
                    'prefix': p,
                    'a0_engine': a0,
                    'block': block,
                    'full': full,
                    'y0': rec['cand'],
                    'tpred_engine': rec['tpred'],
                }
            )
    return cases


def build_cases_drafter(trace_dir: Path, requests: Path) -> list[dict[str, Any]]:
    """Cases from the drafter workstream's DFlash trace (cycles*.jsonl: rid, prefix_len, draft,
    target, accept) joined to requests (id, input_ids, output_ids) by matching committed streams."""
    reqs = [json.loads(line) for line in requests.read_text().splitlines() if line.strip()]
    by_rid: dict[str, list[dict[str, Any]]] = collections.OrderedDict()
    for path in sorted(trace_dir.glob('cycles*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_rid.setdefault(rec['rid'], []).append(rec)
    streams = {}
    for rid, recs in by_rid.items():
        recs.sort(key=lambda r: r['prefix_len'])
        stream = {}
        for rec in recs:
            p, a = rec['prefix_len'], rec['accept']
            stream[p] = rec['draft'][0]
            for j in range(1, a + 1):
                stream[p + j] = rec['draft'][j]
            stream[p + a + 1] = rec['target'][a]
        streams[rid] = (recs[0]['prefix_len'], stream)
    cases = []
    for ri, req in enumerate(reqs):
        full = req['input_ids'] + req['output_ids']
        n_prompt = len(req['input_ids'])
        match = None
        for rid, (start, stream) in streams.items():
            if start != n_prompt:
                continue
            pos = [q for q in sorted(stream) if q < len(full)][:64]
            if pos and all(stream[q] == full[q] for q in pos):
                match = rid
                break
        if match is None:
            continue
        block = len(by_rid[match][0]['draft'])
        for rec in by_rid[match]:
            p, a0 = rec['prefix_len'], rec['accept']
            if a0 >= block - 2 or p + block > len(full) or full[p] != rec['draft'][0]:
                continue
            cases.append(
                {
                    'request': ri,
                    'id': req['id'],
                    'prefix': p,
                    'a0_engine': a0,
                    'block': block,
                    'full': full,
                    'y0': rec['draft'],
                    'tpred_engine': rec['target'],
                }
            )
    return cases


def spread(cases: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if len(cases) <= n:
        return cases
    stride = len(cases) / n
    return [cases[int(i * stride)] for i in range(n)]


class Runner:
    def __init__(self, model, ctl: Anchored, device: str):
        self.model = model
        self.ctl = ctl
        self.device = device
        self.cache = None
        self.cache_len = 0
        self.cache_request = None

    def advance(self, case: dict[str, Any]) -> None:
        """Bring the prefix cache to the case's block start (incrementally within a request)."""
        from transformers import DynamicCache

        if self.cache_request != case['request'] or self.cache_len > case['prefix']:
            self.cache = DynamicCache(config=self.model.config)
            self.cache_len = 0
            self.cache_request = case['request']
        seg = case['full'][self.cache_len : case['prefix']]
        if seg:
            self.ctl.mode = 'plain'
            self.ctl.capture = False
            ids = torch.tensor([seg], device=self.device)
            with torch.no_grad():
                self.model(
                    input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1
                )
            self.cache_len = case['prefix']

    def block(
        self, y: list[int], mode: str, capture: bool = False, rank: int = 0
    ) -> tuple[torch.Tensor, Any]:
        self.ctl.mode = mode
        self.ctl.capture = capture
        self.ctl.rank = rank
        self.ctl.captured = {}
        cache = copy.deepcopy(self.cache)
        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([y], device=self.device),
                past_key_values=cache,
                use_cache=True,
            )
        self.ctl.mode = 'plain'
        return out.logits[0].float(), cache


def gdn_state_change(c0, c1) -> dict[int, float]:
    out = {}
    for li, (l0, l1) in enumerate(zip(c0.layers, c1.layers, strict=True)):
        s0 = getattr(l0, 'recurrent_states', None)
        s1 = getattr(l1, 'recurrent_states', None)
        if s0 is None or s1 is None:
            continue
        out[li] = float((s1.float() - s0.float()).norm() / s0.float().norm().clamp_min(1e-30))
    return out


def decision_stats(approx: torch.Tensor, exact: torch.Tensor, start: int) -> dict[str, Any]:
    top2 = exact.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0)
    err = (approx - exact).abs().amax(dim=-1)
    ratio = (2 * err / margin.clamp_min(1e-30)).tolist()
    agree = (approx.argmax(-1) == exact.argmax(-1)).tolist()
    first = next((i - start for i in range(start, len(agree)) if not agree[i]), None)
    return {
        'start': start,
        'agree': [int(a) for a in agree[start:]],
        'ratio': [min(r, 1e9) for r in ratio[start:]],
        'tie': [int(m == 0) for m in margin.tolist()[start:]],
        'first_disagreement': first,
        'agree_before_start': all(agree[:start]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--run', type=Path, default=None, help='serve_probe.py probe run directory')
    ap.add_argument(
        '--drafter-trace', type=Path, default=None, help='drafter DFlash trace directory'
    )
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--dev-cases', type=int, default=120)
    ap.add_argument('--held-cases', type=int, default=80)
    ap.add_argument('--ranks', type=int, nargs='+', default=[0, 16, 32, 64, 128])
    ap.add_argument('--sweeps', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16
    )
    model = model.to(args.device).eval()
    ctl = Anchored(model)
    runner = Runner(model, ctl, args.device)
    max_rank = max(args.ranks)

    if args.drafter_trace is not None:
        cases = build_cases_drafter(args.drafter_trace, args.requests)
    else:
        cases = build_cases(args.run, args.requests)
    dev = spread([c for c in cases if c['request'] % 2 == 0], args.dev_cases)
    held = spread([c for c in cases if c['request'] % 2 == 1], args.held_cases)
    meta = {
        'run': str(args.run or args.drafter_trace),
        'requests': str(args.requests),
        'cases_total': len(cases),
        'dev': len(dev),
        'held': len(held),
        'ranks': args.ranks,
        'sweeps': args.sweeps,
        'model_revision': MODEL_REVISION,
        'basis': 'PCA of exact dx over dev replays 1 and 2',
        'max_rows_per_variant': MAX_ROWS_PER_VARIANT,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    for name in ('diagnostics.jsonl', 'held_out.jsonl'):
        (out / name).unlink(missing_ok=True)

    # ---- development split: exact replays, input changes for the bases ----
    rows: dict[tuple[int, str], list[torch.Tensor]] = collections.defaultdict(list)
    t_start = time.time()
    for ci, case in enumerate(dev):
        runner.advance(case)
        y0 = case['y0']
        l0, c0 = runner.block(y0, 'anchor')
        y1 = shift(y0, l0)
        l1, c1 = runner.block(y1, 'plain', capture=True)
        x1 = dict(ctl.captured)
        y2 = shift(y1, l1)
        _, c2 = runner.block(y2, 'plain', capture=True)
        x2 = dict(ctl.captured)
        rec: dict[str, Any] = {
            'split': 'dev',
            'case': ci,
            'id': case['id'],
            'prefix': case['prefix'],
            'a0_engine': case['a0_engine'],
            'a0_hf': accepted(y0, l0),
            'engine_hf_argmax_agree': sum(
                int(a == b)
                for a, b in zip(l0.argmax(-1).tolist(), case['tpred_engine'], strict=True)
            )
            / len(y0),
            'gdn_state_change': {'y1': gdn_state_change(c0, c1), 'y2': gdn_state_change(c0, c2)},
            'rel_change': {},
        }
        for name, y, xs in (('y1', y1, x1), ('y2', y2, x2)):
            f = first_change(y, y0)
            rel = {}
            for key, x in xs.items():
                d = x[f:] - ctl.ax[key][f:]
                rel[f'{key[0]}:{key[1]}'] = (
                    d.norm(dim=-1) / ctl.ax[key][f:].norm(dim=-1).clamp_min(1e-12)
                ).tolist()
                if d.shape[0]:
                    rows[key].append(d[:MAX_ROWS_PER_VARIANT].to(torch.float16).cpu())
            rec['rel_change'][name] = {'first_changed': f, 'by_op': rel}
        append_jsonl(out / 'diagnostics.jsonl', rec)
        if ci % 20 == 0:
            print(f'dev {ci + 1}/{len(dev)} {time.time() - t_start:.0f}s', flush=True)

    # ---- fixed bases from the development split ----
    bases = {}
    spectrum = {}
    for key, chunks in rows.items():
        dx = torch.cat(chunks).to(args.device).float()
        bases[key], ev = top_basis(dx, max_rank)
        total = float(ev.sum())
        spectrum[f'{key[0]}:{key[1]}'] = {
            'd_in': dx.shape[1],
            'rows': dx.shape[0],
            'energy_frac': {
                str(k): float(ev[:k].sum() / total) for k in (16, 32, 64, 128) if k <= ev.numel()
            },
        }
    del rows
    ctl.set_bases(bases)
    (out / 'bases_spectrum.json').write_text(json.dumps(spectrum, indent=1))
    print(f'bases fitted for {len(bases)} operator inputs', flush=True)

    # ---- held-out split: decision-level evaluation ----
    for ci, case in enumerate(held):
        runner.advance(case)
        y0 = case['y0']
        l0, c0 = runner.block(y0, 'anchor')
        exact_iter = [y0]
        exact_logits = [l0]
        exact_x = []
        for _ in range(args.sweeps + 1):
            y = shift(exact_iter[-1], exact_logits[-1])
            logits, _ = runner.block(y, 'plain', capture=True)
            exact_iter.append(y)
            exact_logits.append(logits)
            exact_x.append(dict(ctl.captured))
        rec = {
            'split': 'held',
            'case': ci,
            'id': case['id'],
            'prefix': case['prefix'],
            'a0_engine': case['a0_engine'],
            'a0_hf': accepted(y0, l0),
            'exact_jacobi_accept': [
                accepted(y, lg) for y, lg in zip(exact_iter, exact_logits, strict=True)
            ],
            'exact_margin': {},
            'ranks': {},
        }
        # operator-local accuracy of the fixed bases on exact held-out changes (replay 1)
        local: dict[str, dict[str, float]] = {}
        f1 = first_change(exact_iter[1], y0)
        for key, x in exact_x[0].items():
            dx = x[f1:] - ctl.ax[key][f1:]
            if dx.shape[0] == 0:
                continue
            w = ctl.groups[key][0].weight
            wdx = (dx.to(w.dtype) @ w.T).float()
            for r in args.ranks:
                if r == 0:
                    resid = wdx
                else:
                    u = ctl.U[key][:, :r]
                    resid = ((dx - (dx @ u) @ u.T).to(w.dtype) @ w.T).float()
                local.setdefault(str(r), {})[f'{key[0]}:{key[1]}'] = float(
                    resid.norm() / wdx.norm().clamp_min(1e-30)
                )
        rec['local_rel_error_replay1'] = local
        for r in args.ranks:
            entry: dict[str, Any] = {}
            for name, idx in (('replay1', 1), ('replay2', 2)):
                y = exact_iter[idx]
                approx, _ = runner.block(y, 'repair', rank=r)
                entry[name] = decision_stats(approx, exact_logits[idx], first_change(y, y0))
            # free-running repair against the original anchor, audited exactly after every sweep
            y = exact_iter[1]
            accepts = [accepted(y, exact_logits[1])]
            audited = [y]
            for _ in range(args.sweeps):
                approx, _ = runner.block(y, 'repair', rank=r)
                y = shift(y, approx)
                audit, _ = runner.block(y, 'plain')
                accepts.append(accepted(y, audit))
                audited.append(y)
            entry['free_running_accept'] = accepts
            rec['ranks'][str(r)] = entry
        append_jsonl(out / 'held_out.jsonl', rec)
        if ci % 10 == 0:
            print(f'held {ci + 1}/{len(held)} {time.time() - t_start:.0f}s', flush=True)
    meta['elapsed_s'] = time.time() - t_start
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
