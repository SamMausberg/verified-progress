"""Fine-tune a DFlash / DFlash 2 drafter for Qwen3.5-4B against the frozen target.

The objective, anchor sampling, block masks and per-position metrics are
SpecForge's (`OnlineDFlashModel`, sgl-project/SpecForge at 3cb0510, MIT); this
script supplies what SpecForge's launcher would: the frozen Hugging Face target
(hidden states after the draft's `target_layer_ids`, computed on the fly), the
target-generated training sequences from `gen_targets.py`, a single-GPU
optimizer loop that runs in resumable time-boxed segments, CSV logging, and an
export that SGLang's `DFlashDraftModel` / `DFlash2DraftModel` loads unchanged.

Optional additions, both off by default:
  --dflash2               add DFlash 2's grouped convolutions and candidate
                          selector (both initialise as exact no-ops, so step 0
                          reproduces the initial DFlash drafter) and train them
                          with SpecForge's selector objective;
  --rho-weight L          auxiliary loss L * ||h_t - h_d||^2 / ||h_t||^2 between
                          the draft's head input at block slot k and the
                          target's head input that predicts the same token
                          (the geometry workstream's drift ratio rho).

Environment: the SGLang venv (torch 2.13, transformers 5.12.1) with
PYTHONPATH=<fla-core 0.5.2 + flash-linear-attention 0.5.2>:<SpecForge checkout>;
see experiments/drafter/README.md. One segment:

    python experiments/drafter/train_dflash.py --run ~/vp-data/drafter/ckpt/ft-a \
        --init z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
        --data ~/vp-data/drafter/data/targets-v1.jsonl --segment-minutes 25
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

TARGET = 'Qwen/Qwen3.5-4B'
TARGET_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BUCKETS = (1024, 2048, 3072, 4096)


# --------------------------------------------------------------------------- target


def load_target(device: torch.device) -> torch.nn.Module:
    """Qwen3.5-4B text model (no vision tower, no MTP layer), BF16, frozen."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    root = Path(
        snapshot_download(
            TARGET, revision=TARGET_REVISION, allow_patterns=['*.json', '*.safetensors']
        )
    )
    text_config = AutoConfig.from_pretrained(root).text_config
    text_config._attn_implementation = 'sdpa'
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Qwen3_5ForCausalLM(text_config)
    finally:
        torch.set_default_dtype(default_dtype)
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(root.glob('*.safetensors')):
        with safe_open(shard, 'pt', device=str(device)) as handle:
            for key in handle.keys():  # noqa: SIM118 (safe_open is not a mapping)
                if key.startswith('model.language_model.'):
                    state['model.' + key[len('model.language_model.') :]] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key != 'lm_head.weight']
    if missing or unexpected:
        raise RuntimeError(f'target load: missing={missing} unexpected={unexpected}')
    model.lm_head.weight = model.model.embed_tokens.weight  # tied head
    model.requires_grad_(False).eval()
    return model


@torch.no_grad()
def target_features(
    target: torch.nn.Module, input_ids: torch.Tensor, layer_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenated hidden states after `layer_ids` and the final head input."""
    out = target.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states  # embeddings, then one entry per layer
    features = torch.cat([hidden[i + 1] for i in layer_ids], dim=-1)
    return features, out.last_hidden_state


# --------------------------------------------------------------------------- draft


def resolve_init(spec: str) -> Path:
    path = Path(spec).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    repo, _, revision = spec.partition('@')
    return Path(snapshot_download(repo, revision=revision or None))


def build_draft(init_dir: Path, args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    from safetensors.torch import load_file
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from transformers import Qwen3Config

    config_dict = json.loads((init_dir / 'config.json').read_text())
    if args.dflash2 and 'selector_rank' not in config_dict.get('dflash_config', {}):
        config_dict['architectures'] = ['DFlash2DraftModel']
        config_dict['dflash_config'].update(
            conv_kernel_size=args.conv_kernel_size,
            conv_group_size=args.conv_group_size,
            selector_rank=args.selector_rank,
            selector_top_k=args.selector_top_k,
        )
        config_dict.pop('auto_map', None)
    config = Qwen3Config(**config_dict)
    config._attn_implementation = 'flex_attention'
    is_dflash2 = 'selector_rank' in config.dflash_config
    cls = DFlash2DraftModel if is_dflash2 else DFlashDraftModel
    model = cls(config)
    state = load_file(str(init_dir / 'model.safetensors'))
    missing, unexpected = model.load_state_dict(state, strict=False)
    new_params = [key for key in missing if '_conv.' in key or 'candidate_selector' in key]
    if set(missing) - set(new_params) or unexpected:
        raise RuntimeError(f'draft load: missing={missing} unexpected={unexpected}')
    if new_params:
        print(f'initialised {len(new_params)} new DFlash 2 tensors as no-ops', flush=True)
    return model, config_dict


def export_draft(model: torch.nn.Module, config_dict: dict[str, Any], out_dir: Path) -> str:
    """Write config.json + BF16 model.safetensors that SGLang loads; return sha256."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().to(torch.bfloat16).contiguous().cpu()
        for key, value in model.state_dict().items()
        if 'rotary_emb' not in key
    }
    save_file(state, str(out_dir / 'model.safetensors'), metadata={'format': 'pt'})
    (out_dir / 'config.json').write_text(json.dumps(config_dict, indent=2) + '\n')
    digest = hashlib.sha256((out_dir / 'model.safetensors').read_bytes()).hexdigest()
    (out_dir / 'SHA256').write_text(f'{digest}  model.safetensors\n')
    return digest


# --------------------------------------------------------------------------- data


def is_heldout(row_id: str, modulus: int) -> bool:
    return int(hashlib.md5(row_id.encode()).hexdigest(), 16) % modulus == 0


def loop_onset(ids: list[int], start: int, ngram: int = 32, repeats: int = 3) -> int:
    """First position where an `ngram`-token window occurs for the `repeats`-th time.

    Greedy thinking on this target falls into verbatim repetition loops (the bench
    workstream found a quarter of natural-stopping greedy outputs end in one); the
    looped tail is trivially predictable and would inflate acceptance, so training
    sequences are cut where the loop starts. Returns len(ids) if there is none.
    """
    counts: dict[tuple[int, ...], int] = {}
    for end in range(start + ngram, len(ids) + 1):
        window = tuple(ids[end - ngram : end])
        counts[window] = counts.get(window, 0) + 1
        if counts[window] >= repeats:
            return end
    return len(ids)


def load_sequences(path: Path, heldout: bool, modulus: int, max_len: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_heldout(row['id'], modulus) != heldout or len(row['output_ids']) < 2:
            continue
        ids = row['prompt_ids'] + row['output_ids']
        ids = ids[: min(max_len, loop_onset(ids, len(row['prompt_ids'])))]
        if len(ids) - len(row['prompt_ids']) < 16:
            continue
        rows.append(
            {
                'id': row['id'],
                'domain': row['domain'],
                'ids': ids,
                'prompt_len': len(row['prompt_ids']),
            }
        )
    return rows


def bucket_length(n: int) -> int:
    for bucket in BUCKETS:
        if n <= bucket:
            return bucket
    return BUCKETS[-1]


def make_batch(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence padded to its length bucket; loss on response tokens only."""
    length = bucket_length(len(row['ids']))
    ids = torch.zeros(1, length, dtype=torch.long)
    mask = torch.zeros(1, length)
    ids[0, : len(row['ids'])] = torch.tensor(row['ids'])
    mask[0, row['prompt_len'] : len(row['ids'])] = 1.0
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


# --------------------------------------------------------------------------- model


class Trainer:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        self.args = args
        self.device = device
        self.target = load_target(device)
        init_dir = resolve_init(args.init)
        self.draft, self.config_dict = build_draft(init_dir, args)
        self.draft.to(device=device, dtype=torch.float32).train()
        self.layer_ids = list(self.draft.target_layer_ids)
        dflash_config = self.config_dict['dflash_config']
        self.block_size = int(
            dflash_config.get('block_size', self.config_dict.get('block_size', 16))
        )
        self.model = OnlineDFlashModel(
            draft_model=self.draft,
            target_lm_head=self.target.lm_head,
            target_embed_tokens=self.target.model.embed_tokens,
            mask_token_id=int(dflash_config['mask_token_id']),
            block_size=self.block_size,
            attention_backend='flex_attention',
            num_anchors=args.num_anchors,
            loss_decay_gamma=args.gamma,
            selector_loss_alpha=args.selector_alpha,
        )
        self._stash: dict[str, torch.Tensor] = {}
        original = self.model._forward_draft_blocks

        def forward_and_stash(*a: Any, **k: Any):
            anchors, keep, hidden = original(*a, **k)
            self._stash.update(anchors=anchors, keep=keep, hidden=hidden)
            return anchors, keep, hidden

        self.model._forward_draft_blocks = forward_and_stash

        new = [
            p
            for n, p in self.draft.named_parameters()
            if '_conv.' in n or 'candidate_selector' in n
        ]
        new_ids = {id(p) for p in new}
        old = [p for p in self.draft.parameters() if id(p) not in new_ids]
        groups = [{'params': old, 'lr': args.lr}]
        if new:
            groups.append({'params': new, 'lr': args.lr_new})
        if args.freeze_backbone:
            for p in old:
                p.requires_grad_(False)
            groups = groups[1:]
        self.base_lrs = [group['lr'] for group in groups]
        self.optimizer = torch.optim.AdamW(
            groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True
        )
        self.step = 0
        self.epoch = 0
        self.cursor = 0

    def lr_scale(self) -> float:
        warmup, total = self.args.warmup_steps, self.args.total_steps
        if self.step < warmup:
            return (self.step + 1) / warmup
        progress = min(1.0, (self.step - warmup) / max(1, total - warmup))
        floor = self.args.min_lr_ratio
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    def rho_terms(
        self, last_hidden: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-slot sum of rho^2 and count over valid positions (slots 1..B-1)."""
        anchors, keep, hidden = self._stash['anchors'], self._stash['keep'], self._stash['hidden']
        batch, blocks = anchors.shape
        hidden = hidden.view(batch, blocks, self.block_size, -1)[:, :, 1:]
        offsets = torch.arange(1, self.block_size, device=anchors.device)
        label = anchors.unsqueeze(-1) + offsets  # token position each slot predicts
        valid = keep.unsqueeze(-1) & (label < seq_len)
        source = (label - 1).clamp(max=seq_len - 1)  # target position predicting it
        target = torch.gather(
            last_hidden, 1, source.reshape(batch, -1, 1).expand(-1, -1, last_hidden.shape[-1])
        ).view_as(hidden)
        target = target.float()
        diff = hidden.float() - target
        rho2 = diff.pow(2).sum(-1) / target.pow(2).sum(-1).clamp_min(1e-6)
        rho2 = rho2 * valid
        return rho2.sum(dim=(0, 1)), valid.sum(dim=(0, 1)).float(), rho2.sqrt().sum(dim=(0, 1))

    def forward(
        self, row: dict[str, Any], collect: bool
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        ids, mask = make_batch(row, self.device)
        features, last_hidden = target_features(self.target, ids, self.layer_ids)
        valid = int(((mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)).sum())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _, metrics = self.model(
                input_ids=ids,
                hidden_states=features,
                loss_mask=mask,
                target_last_hidden_states=last_hidden if collect else None,
                max_valid_anchors=valid,
                collect_detailed_metrics=collect,
            )
        rho2_sum, rho_count, rho_sum = self.rho_terms(last_hidden, ids.shape[1])
        extra = {'rho2_sum': rho2_sum.detach(), 'rho_sum': rho_sum.detach(), 'rho_count': rho_count}
        if self.args.rho_weight > 0:
            loss = loss + self.args.rho_weight * rho2_sum.sum() / rho_count.sum().clamp_min(1)
        return loss, metrics, extra


# --------------------------------------------------------------------------- logging


class MetricSum:
    """Accumulates SpecForge ratio metrics (numerator, denominator) and rho terms."""

    def __init__(self) -> None:
        self.num: dict[str, float] = {}
        self.den: dict[str, float] = {}
        self.rho: dict[str, torch.Tensor] = {}
        self.loss = 0.0
        self.items = 0

    def add(self, loss: float, metrics: dict[str, Any], extra: dict[str, torch.Tensor]) -> None:
        self.loss += loss
        self.items += 1
        for name, (num, den) in metrics.get('ratio_metrics', {}).items():
            self.num[name] = self.num.get(name, 0.0) + float(num)
            self.den[name] = self.den.get(name, 0.0) + float(den)
        for key, value in extra.items():
            value = value.float().cpu()
            self.rho[key] = self.rho[key] + value if key in self.rho else value

    def row(self, prefix: str, block_size: int) -> dict[str, float]:
        out = {f'{prefix}loss': self.loss / max(1, self.items)}
        for name in self.num:
            if self.den[name] > 0:
                out[f'{prefix}{name}'] = self.num[name] / self.den[name]
        if 'rho_count' in self.rho:
            count, rho2, rho = self.rho['rho_count'], self.rho['rho2_sum'], self.rho['rho_sum']
            total = float(count.sum())
            if total > 0:
                out[f'{prefix}rho_mean'] = float(rho.sum()) / total
            for k in range(1, block_size):
                n = float(count[k - 1])
                if n > 0:
                    out[f'{prefix}rho_rms/pos{k}'] = math.sqrt(float(rho2[k - 1]) / n)
                    out[f'{prefix}rho_mean/pos{k}'] = float(rho[k - 1]) / n
        return out


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append a row; the header is the union of keys seen (file rewritten if it grows)."""
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    if path.exists():
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    new_fields = [key for key in row if key not in fields]
    if new_fields or not path.exists():
        fields += new_fields
        rows.append(row)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open('a', newline='') as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)


# --------------------------------------------------------------------------- loop


def save_state(trainer: Trainer, run: Path) -> None:
    state = {
        'draft': trainer.draft.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'step': trainer.step,
        'epoch': trainer.epoch,
        'cursor': trainer.cursor,
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'python': random.getstate(),
        },
        'args': vars(trainer.args),
    }
    tmp = run / 'state.pt.tmp'
    torch.save(state, tmp)
    os.replace(tmp, run / 'state.pt')


def load_state(trainer: Trainer, run: Path) -> bool:
    path = run / 'state.pt'
    if not path.exists():
        return False
    state = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.draft.load_state_dict(state['draft'])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.step, trainer.epoch, trainer.cursor = state['step'], state['epoch'], state['cursor']
    torch.set_rng_state(state['rng']['torch'].cpu())
    torch.cuda.set_rng_state(state['rng']['cuda'].cpu())
    random.setstate(state['rng']['python'])
    return True


@torch.no_grad()
def evaluate(trainer: Trainer, rows: list[dict[str, Any]]) -> dict[str, float]:
    trainer.draft.eval()
    fork = torch.random.fork_rng(devices=[trainer.device])
    with fork:
        torch.manual_seed(1234)  # same anchors at every evaluation
        total = MetricSum()
        for row in rows:
            loss, metrics, extra = trainer.forward(row, collect=True)
            total.add(float(loss), metrics, extra)
    trainer.draft.train()
    return total.row('eval/', trainer.block_size)


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run = args.run.expanduser()
    run.mkdir(parents=True, exist_ok=True)
    budget = args.segment_minutes * 60
    started = time.monotonic()

    train_rows = load_sequences(
        args.data, heldout=False, modulus=args.heldout_modulus, max_len=args.max_len
    )
    eval_rows = load_sequences(
        args.data, heldout=True, modulus=args.heldout_modulus, max_len=args.max_len
    )
    eval_rows = eval_rows[: args.eval_sequences]
    print(f'{len(train_rows)} training and {len(eval_rows)} held-out sequences', flush=True)
    trainer = Trainer(args, device)
    resumed = load_state(trainer, run)
    if not resumed:
        (run / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')
        if eval_rows and args.eval_every:
            row = {'step': 0, 'epoch': 0, 'tokens': 0, **evaluate(trainer, eval_rows)}
            append_csv(run / 'eval.csv', row)
            print(
                json.dumps(
                    {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in row.items()
                        if 'pos' not in k
                    }
                ),
                flush=True,
            )
    print(
        f'resumed={resumed} step={trainer.step} epoch={trainer.epoch} cursor={trainer.cursor}',
        flush=True,
    )

    order = epoch_order(len(train_rows), args.seed, trainer.epoch)
    window = MetricSum()
    window_tokens = 0
    window_start = time.monotonic()
    tokens_file = run / 'tokens.txt'
    tokens_seen = int(tokens_file.read_text()) if tokens_file.exists() else 0
    while trainer.step < args.total_steps:
        if time.monotonic() - started > budget:
            break
        for group, base in zip(trainer.optimizer.param_groups, trainer.base_lrs, strict=True):
            group['lr'] = base * trainer.lr_scale()
        trainer.optimizer.zero_grad(set_to_none=True)
        collect = (trainer.step + 1) % args.log_every == 0
        for _ in range(args.accumulate):
            if trainer.cursor >= len(order):
                trainer.epoch += 1
                trainer.cursor = 0
                order = epoch_order(len(train_rows), args.seed, trainer.epoch)
            row = train_rows[order[trainer.cursor]]
            trainer.cursor += 1
            loss, metrics, extra = trainer.forward(row, collect=collect)
            (loss / args.accumulate).backward()
            window.add(float(loss.detach()), metrics, extra)
            window_tokens += len(row['ids'])
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in trainer.draft.parameters() if p.requires_grad], args.max_grad_norm
        )
        trainer.optimizer.step()
        trainer.step += 1
        tokens_seen += window_tokens if collect else 0
        if collect:
            elapsed = time.monotonic() - window_start
            row = {
                'step': trainer.step,
                'epoch': trainer.epoch,
                'lr': trainer.optimizer.param_groups[0]['lr'],
                'grad_norm': float(grad_norm),
                'tokens_per_s': window_tokens / elapsed,
                'tokens': tokens_seen,
                **window.row('train/', trainer.block_size),
            }
            append_csv(run / 'train.csv', row)
            brief = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in row.items()
                if '/' not in k
                or k in ('train/loss', 'train/acc', 'train/dflash/hard_label/walk_accepted_length')
            }
            print(json.dumps(brief), flush=True)
            window, window_tokens, window_start = MetricSum(), 0, time.monotonic()
            tokens_file.write_text(str(tokens_seen))
        if args.eval_every and trainer.step % args.eval_every == 0 and eval_rows:
            row = {
                'step': trainer.step,
                'epoch': trainer.epoch,
                'tokens': tokens_seen,
                **evaluate(trainer, eval_rows),
            }
            append_csv(run / 'eval.csv', row)
            print(
                json.dumps(
                    {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in row.items()
                        if 'pos' not in k
                    }
                ),
                flush=True,
            )
    save_state(trainer, run)
    digest = export_draft(trainer.draft, trainer.config_dict, run / 'export')
    (run / 'export' / 'STEP').write_text(f'{trainer.step}\n')
    print(f'segment done: step={trainer.step} export sha256={digest}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', type=Path, required=True, help='run directory (resumes)')
    parser.add_argument('--init', required=True, help='draft dir or repo@revision')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--segment-minutes', type=float, default=25)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--accumulate', type=int, default=8, help='sequences per step')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr-new', type=float, default=5e-4, help='lr for new DFlash 2 tensors')
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--num-anchors', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=7.0, help='loss decay over positions')
    parser.add_argument('--dflash2', action='store_true')
    parser.add_argument('--conv-kernel-size', type=int, default=2)
    parser.add_argument('--conv-group-size', type=int, default=16)
    parser.add_argument('--selector-rank', type=int, default=256)
    parser.add_argument('--selector-top-k', type=int, default=16)
    parser.add_argument('--selector-alpha', type=float, default=1.0)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--rho-weight', type=float, default=0.0)
    parser.add_argument('--heldout-modulus', type=int, default=50)
    parser.add_argument('--eval-sequences', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
