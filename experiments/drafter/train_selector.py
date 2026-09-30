"""Train a DFlash 2 candidate selector over a frozen DFlash drafter (proposal P6).

The public DFlash backbone and the target stay frozen. For each block the
backbone's unary logits give K = 16 candidates per drafted position (top-K
through the target's tied head); the selector adds a low-rank,
predecessor-conditioned transition score (DFlash 2's parameterization, SpecForge's
`CandidateSelector`, the parameter names SGLang's `DFlash2DraftModel` loads):

    score_i(b | a) = l_i(b) + < A[a] * P h_i , B[b] >,   b in C_i

With the target's greedy continuation g as teacher-forced predecessors,
q_i = softmax over C_i of score_i( . | g_{i-1}), and q_i(g_i) = 0 when g_i is not a
candidate. Objectives (`--objective`):

Several objectives can be trained side by side (`--objectives prefix,vat`): each
arm has its own selector and optimizer, and all arms see the same sequences,
anchors and candidates.

  prefix  expected accepted length of a draft sampled from q:
          E[L] = sum_k prod_{i<=k} q_i(g_i | g_{i-1}); the loss is -E[L] per block.
          At a fixed block size the cycle cost does not depend on the selector,
          so this is the rate objective R - lambda C of P6 up to a constant.
  ce      position-weighted cross-entropy on covered positions, weights
          exp(-(i-1)/gamma) (SpecForge's DFlash 2 selector loss; the control).
  dpace   D-PACE weights (SpecForge's formulation: cumulative products of
          (1-alpha) q + alpha, suffix-summed, detached) on the cross-entropy.
  vat     VAT weights: 1 up to the current greedy walk's first rejection, then
          the exponential decay restarted there.

Metrics per position: unary greedy acceptance, the selector's greedy lattice
walk (argmax at each slot with the chosen predecessor; with teacher forcing it
stays on the truth while it is correct), and the top-K support bound U.

    python experiments/drafter/train_selector.py --run ~/vp-data/drafter/ckpt/sel-prefix \
        --init z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
        --data ~/vp-data/drafter/data/targets-v1.jsonl --objective prefix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rho_pairs import block_hidden
from train_dflash import (
    append_csv,
    build_draft,
    epoch_order,
    load_sequences,
    load_target,
    resolve_init,
    target_features,
)


def sample_anchors(prompt_len: int, length: int, block: int, count: int) -> list[int]:
    """Anchors in the response whose whole block has a known continuation."""
    candidates = list(range(prompt_len, length - block))
    if len(candidates) <= count:
        return candidates
    return sorted(random.sample(candidates, count))


class SelectorTrainer:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from specforge.modeling.draft.dflash2 import CandidateSelector

        self.args = args
        self.device = device
        self.target = load_target(device)
        self.draft, self.config = build_draft(
            resolve_init(args.init),
            argparse.Namespace(dflash2=False, attention_backend=args.attention_backend),
        )
        self.draft.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        dflash = self.config['dflash_config']
        self.block = int(dflash.get('block_size', self.config.get('block_size', 16)))
        self.mask_token = int(dflash['mask_token_id'])
        self.top_k = args.top_k
        # One selector and optimizer per objective. The arms see the same sequences,
        # anchors and candidates (computed once per sequence), so the comparison
        # between objectives is matched by construction.
        self.objectives = args.objectives
        self.selectors: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for objective in self.objectives:
            torch.manual_seed(args.seed)  # identical initialization in every arm
            selector = CandidateSelector(
                hidden_size=int(self.config['hidden_size']),
                vocab_size=int(self.config['vocab_size']),
                state_rank=args.rank,
                top_k=args.top_k,
                initializer_range=float(self.config.get('initializer_range', 0.02)),
            ).to(device)
            self.selectors[objective] = selector
            self.optimizers[objective] = torch.optim.AdamW(
                selector.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
            )
        self.step = self.epoch = self.cursor = 0

    @torch.no_grad()
    def candidates(self, row: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        ids_list = row['ids']
        anchors_list = sample_anchors(
            row['prompt_len'], len(ids_list), self.block, self.args.anchors
        )
        if not anchors_list:
            return None
        ids = torch.tensor([ids_list], device=self.device)
        features, _ = target_features(self.target, ids, list(self.draft.target_layer_ids))
        anchors = torch.tensor(anchors_list, device=self.device)
        hidden = block_hidden(
            self.draft,
            self.target.model.embed_tokens,
            features,
            ids,
            anchors,
            self.block,
            self.mask_token,
        )[:, 1:]
        head = self.target.lm_head.weight
        values, cand = [], []
        for chunk in hidden.split(64):
            top = (chunk.to(head.dtype) @ head.T).float().topk(self.top_k, dim=-1)
            values.append(top.values)
            cand.append(top.indices)
        truth = ids[0, anchors[:, None] + torch.arange(1, self.block, device=self.device)]
        return {
            'hidden': hidden.float(),
            'unary': torch.cat(values),
            'cand': torch.cat(cand),
            'truth': truth,
            'anchor_tok': ids[0, anchors],
        }

    def scores(self, batch: dict[str, torch.Tensor], objective: str) -> torch.Tensor:
        predecessors = torch.cat([batch['anchor_tok'][:, None], batch['truth'][:, :-1]], dim=1)
        return self.selectors[objective].score_candidates(
            candidate_ids=batch['cand'],
            unary_logits=batch['unary'],
            hidden_states=batch['hidden'],
            predecessor_ids=predecessors,
        )

    def loss(
        self, batch: dict[str, torch.Tensor], objective: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scores = self.scores(batch, objective)  # [n, B-1, K]
        match = batch['cand'] == batch['truth'][..., None]
        covered = match.any(-1)
        log_q = torch.log_softmax(scores.float(), dim=-1)
        log_q_truth = torch.where(
            covered, (log_q * match).sum(-1), torch.full_like(covered, -1e4, dtype=log_q.dtype)
        )
        positions = torch.arange(self.block - 1, device=self.device)
        if objective == 'prefix':
            survival = torch.exp(torch.cumsum(log_q_truth, dim=-1))
            loss = -survival.sum(-1).mean()
        else:
            ce = -log_q_truth * covered
            if objective == 'ce':
                weights = torch.exp(-positions.float() / self.args.gamma)[None]
            elif objective == 'dpace':
                with torch.no_grad():
                    q = torch.exp(log_q_truth)
                    smooth = (1 - self.args.dpace_alpha) * q + self.args.dpace_alpha
                    prefix = torch.cumprod(smooth, dim=-1) * covered
                    weights = torch.flip(torch.cumsum(torch.flip(prefix, [-1]), -1), [-1])
            elif objective == 'vat':
                # VAT (arXiv 2608.30135, Eq. 6): full weight up to the current greedy
                # walk's first rejection k*, then the base decay restarted at k*.
                with torch.no_grad():
                    hit = (scores.argmax(-1) == match.float().argmax(-1)) & covered
                    first_miss = (hit.long().cumprod(-1).sum(-1, keepdim=True)).float()
                    after = (positions.float()[None] - first_miss).clamp_min(0)
                    weights = torch.exp(-after / self.args.gamma)
            else:
                raise ValueError(objective)
            weights = weights * covered
            loss = (ce * weights).sum() / weights.sum().clamp_min(1.0)
        with torch.no_grad():
            unary_hit = batch['cand'][..., 0] == batch['truth']
            walk_hit = scores.argmax(-1) == match.float().argmax(-1)
            walk_hit &= covered
            terms = {
                'unary_survival': unary_hit.long().cumprod(-1).float().sum(0),
                'walk_survival': walk_hit.long().cumprod(-1).float().sum(0),
                'support_survival': covered.long().cumprod(-1).float().sum(0),
                'blocks': torch.tensor(float(unary_hit.shape[0]), device=self.device),
            }
        return loss, terms


def summarize(terms: dict[str, torch.Tensor], prefix: str) -> dict[str, float]:
    blocks = float(terms['blocks'])
    out: dict[str, float] = {}
    for name in ('unary', 'walk', 'support'):
        survival = (terms[f'{name}_survival'] / blocks).tolist()
        out[f'{prefix}{name}_tau'] = 1 + sum(survival)
        for k, value in enumerate(survival, start=1):
            out[f'{prefix}{name}_S{k}'] = value
    return out


def add_terms(total: dict[str, torch.Tensor], terms: dict[str, torch.Tensor]) -> None:
    for key, value in terms.items():
        total[key] = total[key] + value if key in total else value.clone()


def evaluate(trainer: SelectorTrainer, rows: list[dict[str, Any]]) -> dict[str, float]:
    state = random.getstate()
    random.seed(1234)  # the same anchors at every evaluation
    totals: dict[str, dict[str, torch.Tensor]] = {obj: {} for obj in trainer.objectives}
    losses = dict.fromkeys(trainer.objectives, 0.0)
    with torch.no_grad():
        for row in rows:
            batch = trainer.candidates(row)
            if batch is None:
                continue
            for objective in trainer.objectives:
                loss, terms = trainer.loss(batch, objective)
                losses[objective] += float(loss)
                add_terms(totals[objective], terms)
    random.setstate(state)
    out: dict[str, float] = {}
    for objective in trainer.objectives:
        out[f'eval/{objective}/loss'] = losses[objective] / max(1, len(rows))
        out.update(summarize(totals[objective], f'eval/{objective}/'))
    return out


def export(trainer: SelectorTrainer, objective: str, init_dir: Path, out: Path) -> str:
    """DFlash 2 checkpoint: the frozen backbone's tensors plus one arm's selector."""
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)
    state = load_file(str(init_dir / 'model.safetensors'))
    for key, value in trainer.selectors[objective].state_dict().items():
        state[f'candidate_selector.{key}'] = value.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(state, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    config = dict(trainer.config)
    config['architectures'] = ['DFlash2DraftModel']
    config.pop('auto_map', None)
    config['dflash_config'] = dict(
        config['dflash_config'], selector_rank=trainer.args.rank, selector_top_k=trainer.top_k
    )
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    digest = hashlib.sha256((out / 'model.safetensors').read_bytes()).hexdigest()
    (out / 'SHA256').write_text(f'{digest}  model.safetensors\n')
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--init', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument(
        '--objectives',
        type=lambda text: text.split(','),
        default=['prefix'],
        help='comma-separated arms trained side by side: prefix, ce, dpace, vat',
    )
    parser.add_argument('--gamma', type=float, default=7.0)
    parser.add_argument('--dpace-alpha', type=float, default=0.5)
    parser.add_argument('--rank', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=16)
    parser.add_argument('--anchors', type=int, default=256, help='blocks per sequence')
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--heldout-modulus', type=int, default=50)
    parser.add_argument('--eval-sequences', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--segment-minutes', type=float, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--attention-backend', choices=['flex_attention', 'sdpa'], default='flex_attention'
    )
    args = parser.parse_args()
    for objective in args.objectives:
        if objective not in ('prefix', 'ce', 'dpace', 'vat'):
            parser.error(f'unknown objective {objective}')

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run = args.run.expanduser()
    run.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train_rows = load_sequences(
        args.data, heldout=False, modulus=args.heldout_modulus, max_len=args.max_len
    )
    eval_rows = load_sequences(
        args.data, heldout=True, modulus=args.heldout_modulus, max_len=args.max_len
    )
    eval_rows = eval_rows[: args.eval_sequences]
    print(f'{len(train_rows)} training and {len(eval_rows)} held-out sequences', flush=True)
    trainer = SelectorTrainer(args, device)
    state_path = run / 'state.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        for objective in trainer.objectives:
            trainer.selectors[objective].load_state_dict(state['arms'][objective]['selector'])
            trainer.optimizers[objective].load_state_dict(state['arms'][objective]['optimizer'])
        trainer.step, trainer.epoch, trainer.cursor = state['step'], state['epoch'], state['cursor']
        random.setstate(state['python_rng'])
    else:
        (run / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')
        if eval_rows:
            append_csv(run / 'eval.csv', {'step': 0, **evaluate(trainer, eval_rows)})
    order = epoch_order(len(train_rows), args.seed, trainer.epoch)
    window: dict[str, dict[str, torch.Tensor]] = {}
    window_loss: dict[str, float] = {}
    window_items, window_start = 0, time.monotonic()
    while (
        trainer.step < args.total_steps and time.monotonic() - started < args.segment_minutes * 60
    ):
        warm = min(1.0, (trainer.step + 1) / args.warmup_steps)
        progress = min(1.0, trainer.step / max(1, args.total_steps))
        cosine = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
        for optimizer in trainer.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = args.lr * warm * cosine
        for _ in range(args.accumulate):
            if trainer.cursor >= len(order):
                trainer.epoch, trainer.cursor = trainer.epoch + 1, 0
                order = epoch_order(len(train_rows), args.seed, trainer.epoch)
            row = train_rows[order[trainer.cursor]]
            trainer.cursor += 1
            batch = trainer.candidates(row)
            if batch is None:
                continue
            window_items += 1
            for objective in trainer.objectives:
                loss, terms = trainer.loss(batch, objective)
                (loss / args.accumulate).backward()
                window_loss[objective] = window_loss.get(objective, 0.0) + float(loss.detach())
                add_terms(window.setdefault(objective, {}), terms)
        for objective in trainer.objectives:
            torch.nn.utils.clip_grad_norm_(trainer.selectors[objective].parameters(), 1.0)
            trainer.optimizers[objective].step()
        trainer.step += 1
        if trainer.step % args.log_every == 0 and window:
            elapsed = time.monotonic() - window_start
            row_out: dict[str, float] = {
                'step': trainer.step,
                'epoch': trainer.epoch,
                'lr': args.lr * warm * cosine,
                'seq_per_s': window_items / elapsed,
            }
            for objective in trainer.objectives:
                row_out[f'train/{objective}/loss'] = window_loss[objective] / max(1, window_items)
                row_out.update(summarize(window[objective], f'train/{objective}/'))
            append_csv(run / 'train.csv', row_out)
            print(
                json.dumps(
                    {k: round(v, 4) for k, v in row_out.items() if '_S' not in k.split('/')[-1]}
                ),
                flush=True,
            )
            window, window_loss, window_items, window_start = {}, {}, 0, time.monotonic()
        if trainer.step % args.eval_every == 0 and eval_rows:
            append_csv(run / 'eval.csv', {'step': trainer.step, **evaluate(trainer, eval_rows)})
    torch.save(
        {
            'arms': {
                objective: {
                    'selector': trainer.selectors[objective].state_dict(),
                    'optimizer': trainer.optimizers[objective].state_dict(),
                }
                for objective in trainer.objectives
            },
            'step': trainer.step,
            'epoch': trainer.epoch,
            'cursor': trainer.cursor,
            'python_rng': random.getstate(),
        },
        run / 'state.pt.tmp',
    )
    os.replace(run / 'state.pt.tmp', state_path)
    digests = {}
    for objective in trainer.objectives:
        out = run / f'export-{objective}'
        digests[objective] = export(trainer, objective, resolve_init(args.init), out)
        (out / 'STEP').write_text(f'{trainer.step}\n')
    digest = ' '.join(f'{k}={v}' for k, v in digests.items())
    if trainer.step >= args.total_steps:
        print('final step reached', flush=True)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0
    print(
        f'segment done: step={trainer.step} export sha256={digest} peak_gpu_gib={peak_gib:.1f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
