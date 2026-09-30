"""Teacher-forced head-input pairs (draft vs target) for a DFlash drafter.

For target-generated greedy sequences, runs the frozen Hugging Face target once
(features after the drafter's target layers, and the final-norm head input h_t)
and the drafter on blocks anchored at fixed positions of the response. For block
slot k >= 1 anchored at position a, the drafter's head input h_d (after its final
norm) is paired with the target's head input at position a + k - 1, the row
whose argmax the drafted token is verified against. Records

    rho = ||h_t - h_d||_2 / ||h_t||_2

with `accepted` (the two head inputs have the same argmax through the tied head)
and `prefix_accepted` (slots 1..k all accepted, i.e. the engine would reach and
accept slot k). Writes pairs as BF16 tensors plus a per-slot summary.

Sequences come from a JSONL with prompt token ids and greedy `output_ids`
(gen_targets.py format), or from the geometry workstream's capture
(`--geometry-prompts` + `--geometry-outputs`, filtered by split).

    python experiments/drafter/rho_pairs.py --draft z-lab/Qwen3.5-4B-DFlash@9a1996cc... \
        --geometry-prompts ~/vp-data/geometry/prompts.jsonl \
        --geometry-outputs ~/vp-data/geometry/plain4b/outputs.jsonl --split heldout \
        --stride 16 --out ~/vp-data/drafter/rho/zlab_b16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dflash import (
    TARGET,
    TARGET_REVISION,
    build_draft,
    load_target,
    resolve_init,
    target_features,
)


def geometry_sequences(prompts: Path, outputs: Path, split: str) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)
    by_id = {row['prompt_id']: row for row in map(json.loads, prompts.read_text().splitlines())}
    rows = []
    for out in map(json.loads, outputs.read_text().splitlines()):
        prompt = by_id[out['prompt_id']]
        if prompt['split'] != split:
            continue
        text = tokenizer.apply_chat_template(
            prompt['messages'],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(prompt['thinking']),
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        rows.append(
            {
                'id': out['rid'],
                'domain': out['domain'],
                'prompt_ids': ids,
                'output_ids': out['output_ids'],
            }
        )
    return rows


@torch.no_grad()
def block_hidden(
    draft: torch.nn.Module,
    embed: torch.nn.Module,
    features: torch.Tensor,
    ids: torch.Tensor,
    anchors: torch.Tensor,
    block: int,
    mask_token: int,
) -> torch.Tensor:
    """Draft head inputs for blocks at `anchors`: [n_anchors, block, hidden]."""
    from specforge.algorithms.common.dflash_family_model import (
        create_dflash_block_mask,
        create_dflash_sdpa_mask,
    )

    seq_len = ids.shape[1]
    n = anchors.shape[0]
    noise = torch.full((1, n * block), mask_token, dtype=torch.long, device=ids.device)
    noise[0, torch.arange(n, device=ids.device) * block] = ids[0, anchors]
    positions = torch.cat(
        [
            torch.arange(seq_len, device=ids.device),
            (anchors[:, None] + torch.arange(block, device=ids.device)).reshape(-1),
        ]
    )[None]
    keep = torch.ones(1, n, dtype=torch.bool, device=ids.device)
    flex = draft.config._attn_implementation == 'flex_attention'
    builder = create_dflash_block_mask if flex else create_dflash_sdpa_mask
    masks = {}
    layer_types = set(getattr(draft, 'layer_types', ['full_attention']))
    if 'full_attention' in layer_types:
        masks['full_attention'] = builder(anchors[None], keep, seq_len, block, ids.device)
    if 'sliding_attention' in layer_types:
        masks['sliding_attention'] = builder(
            anchors[None], keep, seq_len, block, ids.device, sliding_window=draft.sliding_window
        )
    kwargs = {'kernel_options': {'BACKEND': 'TRITON'}} if flex else {}
    with torch.autocast(ids.device.type, dtype=torch.bfloat16):
        hidden = draft(
            position_ids=positions,
            noise_embedding=embed(noise),
            target_hidden=features,
            attention_mask=masks,
            **kwargs,
        )
    return hidden.view(n, block, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--draft', required=True, help='draft dir or repo@revision')
    parser.add_argument('--data', type=Path, help='gen_targets.py JSONL')
    parser.add_argument('--geometry-prompts', type=Path)
    parser.add_argument('--geometry-outputs', type=Path)
    parser.add_argument('--split', default='heldout')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--stride', type=int, default=16, help='anchor spacing in the response')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    if args.geometry_prompts:
        rows = geometry_sequences(args.geometry_prompts, args.geometry_outputs, args.split)
    else:
        rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    target = load_target(device)
    build_args = argparse.Namespace(dflash2=False)
    draft, config = build_draft(resolve_init(args.draft), build_args)
    draft.to(device=device, dtype=torch.bfloat16).eval()
    block = int(config['dflash_config'].get('block_size', config.get('block_size', 16)))
    mask_token = int(config['dflash_config']['mask_token_id'])
    head = target.lm_head.weight

    keys = ('h_d', 'h_t', 'slot', 'accepted', 'prefix_accepted', 'anchor', 'row')
    pairs: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    rids: list[str] = []
    domains: list[str] = []
    for index, row in enumerate(rows):
        ids_list = row['prompt_ids'] + row['output_ids']
        start = len(row['prompt_ids'])
        anchors_list = list(range(start, len(ids_list) - block, args.stride))
        if not anchors_list:
            continue
        ids = torch.tensor([ids_list], device=device)
        features, last_hidden = target_features(target, ids, list(draft.target_layer_ids))
        anchors = torch.tensor(anchors_list, device=device)
        hidden = block_hidden(
            draft, target.model.embed_tokens, features, ids, anchors, block, mask_token
        )
        h_d = hidden[:, 1:]  # slots 1..block-1
        source = anchors[:, None] + torch.arange(block - 1, device=device)  # target rows a+k-1
        h_t = last_hidden[0, source]
        draft_arg = (h_d.to(head.dtype) @ head.T).argmax(-1)
        target_arg = (h_t.to(head.dtype) @ head.T).argmax(-1)
        accepted = draft_arg == target_arg
        prefix = accepted.long().cumprod(-1).bool()
        n = anchors.shape[0]
        pairs['h_d'].append(h_d.reshape(-1, h_d.shape[-1]).to(torch.bfloat16).cpu())
        pairs['h_t'].append(h_t.reshape(-1, h_t.shape[-1]).to(torch.bfloat16).cpu())
        pairs['slot'].append(torch.arange(1, block).repeat(n))
        pairs['accepted'].append(accepted.reshape(-1).cpu())
        pairs['prefix_accepted'].append(prefix.reshape(-1).cpu())
        pairs['anchor'].append(anchors[:, None].expand(-1, block - 1).reshape(-1).cpu())
        pairs['row'].append(source.reshape(-1).cpu())
        rids += [row['id']] * (n * (block - 1))
        domains += [row['domain']] * (n * (block - 1))
        if index % 20 == 0:
            print(f'{index + 1}/{len(rows)} sequences', flush=True)
    data = {key: torch.cat(value) for key, value in pairs.items()}
    data['rid'] = rids
    data['domain'] = domains
    data['meta'] = {
        'draft': args.draft,
        'target': f'{TARGET}@{TARGET_REVISION}',
        'definition': 'rho = ||h_t - h_d|| / ||h_t|| at head inputs; slot k vs target row a+k-1',
        'stride': args.stride,
        'source': str(args.data or args.geometry_outputs),
        'split': args.split,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out / 'pairs.pt')

    h_d, h_t = data['h_d'].float(), data['h_t'].float()
    rho = (h_t - h_d).norm(dim=-1) / h_t.norm(dim=-1)
    summary: dict[str, Any] = {'pairs': len(rho), 'by_slot': []}
    for slot in range(1, block):
        for label, mask in (
            ('all', data['slot'] == slot),
            ('accepted', (data['slot'] == slot) & data['accepted']),
            ('rejected', (data['slot'] == slot) & ~data['accepted']),
        ):
            values = sorted(rho[mask].tolist())
            if not values:
                continue
            q = statistics.quantiles(values, n=10) if len(values) > 1 else [values[0]] * 9
            summary['by_slot'].append(
                {
                    'slot': slot,
                    'subset': label,
                    'n': len(values),
                    'p10': q[0],
                    'median': statistics.median(values),
                    'p90': q[8],
                    'max': values[-1],
                    'accept_rate': float(data['accepted'][data['slot'] == slot].float().mean()),
                }
            )
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for entry in summary['by_slot']:
        if entry['subset'] == 'all':
            print(
                f'slot {entry["slot"]:2d} n={entry["n"]} accept={entry["accept_rate"]:.3f} '
                f'rho p10/med/p90/max = {entry["p10"]:.3f}/{entry["median"]:.3f}/'
                f'{entry["p90"]:.3f}/{entry["max"]:.3f}'
            )


if __name__ == '__main__':
    main()
