"""Zero-training support screen for a selector over the drafter's candidates (P6).

For every verify cycle of a greedy DFlash trace (engine hook, `run_trace.sh`), the
drafter's block at that cycle's anchor is recomputed offline with the Hugging Face
target's features over the committed sequence (the same context the engine had),
and its top-K candidate sets C_k (k = 1..B-1) are taken through the tied head.
With g_k the realized greedy continuation (the request's own output tokens),

    U_K = max{k : g_i in C_i^K for all i <= k}

is an upper bound on the accepted prefix of ANY selector that picks one path from
the frozen top-K candidates at that state (DFlash 2's selector, a rate-trained
selector, ...). It is compared with the engine's accepted length L and with the
offline unary argmax L_hf, whose agreement with the engine's drafted tokens checks
the offline pipeline. Tokens per cycle are reported as E[L]+1 and E[U_K]+1, pooled
per domain; survival S(k) per position is written as CSV.

With `--selector RUN:OBJECTIVE` (a train_selector.py run's state.pt arm) it also
walks the selector greedily over the same frozen candidates from each cycle's
anchor (each chosen token is the next predecessor) and reports its accepted
prefix L_sel, and for P9 writes selector_rewalk.pt: for cycles the engine rejected
early (L < B-1) whose correction g_{L+1} is among the top-K at slot L+1, the greedy
re-walk of the later slots with that correction as the predecessor.

    python experiments/drafter/support_screen.py --trace ~/vp-data/drafter/trace/b16 \
        --panel experiments/drafter/panel-v1.jsonl \
        --draft z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
        --out ~/vp-data/drafter/support/zlab_b16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rho_pairs import block_hidden
from train_dflash import (
    TARGET,
    TARGET_REVISION,
    build_draft,
    load_target,
    resolve_init,
    target_features,
)

KS = (1, 2, 4, 8, 16)


def prefix_len(hits: torch.Tensor) -> torch.Tensor:
    """Number of leading True values per row."""
    return hits.long().cumprod(-1).sum(-1)


def greedy_walk(
    selector: torch.nn.Module,
    candidates: torch.Tensor,
    unary: torch.Tensor,
    hidden: torch.Tensor,
    first_predecessor: torch.Tensor,
    start: torch.Tensor,
    corrections: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy walk of the selector over frozen candidate sets.

    candidates, unary: [n, S, K]; hidden: [n, S, H] (the drafter's hidden state at
    each slot). Row r walks slots start[r]..S-1; the predecessor of its first walked
    slot is first_predecessor[r] and each chosen token is the next predecessor.
    Returns the chosen tokens [n, S], -1 at slots before start[r].
    """
    n, slots, _ = candidates.shape
    walk = torch.full((n, slots), -1, dtype=candidates.dtype, device=candidates.device)
    predecessor = first_predecessor.clone()
    for slot in range(slots):
        active = start <= slot
        if not bool(active.any()):
            continue
        scores = selector.score_candidates(
            candidate_ids=candidates[active, slot],
            unary_logits=unary[active, slot].float(),
            hidden_states=hidden[active, slot].float(),
            predecessor_ids=predecessor[active],
        )
        choice = candidates[active, slot].gather(1, scores.argmax(-1, keepdim=True))[:, 0]
        walk[active, slot] = choice
        predecessor[active] = choice
    return walk


def load_selector(
    spec: str, config: dict, top_k: int, device: torch.device
) -> tuple[torch.nn.Module, int, int]:
    """CandidateSelector from a train_selector.py run ('RUN_DIR:OBJECTIVE'), its rank
    and the training step it was saved at."""
    from specforge.modeling.draft.dflash2 import CandidateSelector

    run, objective = spec.rsplit(':', 1)
    state = torch.load(Path(run) / 'state.pt', map_location='cpu', weights_only=False)
    weights = state['arms'][objective]['selector']
    rank = int(weights['predecessor_codebook'].shape[-1])
    selector = CandidateSelector(
        hidden_size=int(config['hidden_size']),
        vocab_size=int(config['vocab_size']),
        state_rank=rank,
        top_k=top_k,
        initializer_range=float(config.get('initializer_range', 0.02)),
    )
    selector.load_state_dict(weights)
    return selector.to(device).eval(), rank, int(state.get('step', -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trace', type=Path, required=True, help='run dir with cycles-panel.jsonl')
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--save-cycles',
        action='store_true',
        help='also write cycles.pt: per-cycle accepted lengths, U_K, the realized '
        'continuation, the engine draft and the top-16 candidate ids and logits',
    )
    parser.add_argument(
        '--selector',
        help="RUN_DIR:OBJECTIVE of a train_selector.py run: also report the selector's "
        'greedy walk (L_sel) and write selector_rewalk.pt (P9)',
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)
    panel = {row['id']: row for row in map(json.loads, args.panel.read_text().splitlines())}
    requests = {
        row['id']: row
        for row in map(json.loads, (args.trace / 'requests.jsonl').read_text().splitlines())
    }
    cycles: dict[str, list[dict]] = defaultdict(list)
    for line in (args.trace / 'cycles-panel.jsonl').read_text().splitlines():
        cycle = json.loads(line)
        cycles[cycle['rid']].append(cycle)

    device = torch.device('cuda')
    target = load_target(device)
    draft, config = build_draft(resolve_init(args.draft), argparse.Namespace(dflash2=False))
    draft.to(device=device, dtype=torch.bfloat16).eval()
    block = len(next(iter(cycles.values()))[0]['draft'])
    mask_token = int(config['dflash_config']['mask_token_id'])
    head = target.lm_head.weight
    selector, rank, selector_step = None, 0, -1
    if args.selector:
        selector, rank, selector_step = load_selector(args.selector, config, max(KS), device)
    rewalk: dict[str, list] = defaultdict(list)

    # Per domain: sums of L (engine), L_hf, U_K; counts; per-position survival counts.
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    survival: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * block)
    agree = [0, 0]
    saved: dict[str, list] = defaultdict(list)
    for index, (rid, rows) in enumerate(sorted(cycles.items())):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': panel[rid]['text']}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = tokenizer.encode(text, add_special_tokens=False)
        sequence = prompt + requests[rid]['output_ids']
        rows = [r for r in rows if r['prefix_len'] + block - 1 < len(sequence)]
        if not rows:
            continue
        ids = torch.tensor([sequence], device=device)
        anchors = torch.tensor([r['prefix_len'] for r in rows], device=device)
        anchor_ok = ids[0, anchors] == torch.tensor([r['draft'][0] for r in rows], device=device)
        if not bool(anchor_ok.all()):
            raise RuntimeError(f'{rid}: trace anchors do not match the output sequence')
        features, _ = target_features(target, ids, list(draft.target_layer_ids))
        hidden = block_hidden(
            draft, target.model.embed_tokens, features, ids, anchors, block, mask_token
        )
        # Top-K through the tied head in chunks ([n, B-1, vocab] logits would be GBs).
        tops = [
            (chunk.to(head.dtype) @ head.T).topk(max(KS), dim=-1)
            for chunk in hidden[:, 1:].split(64)
        ]
        top = torch.cat([t.indices for t in tops])  # sorted, top[..., 0] = argmax
        top_logits = torch.cat([t.values for t in tops])
        truth = ids[0, anchors[:, None] + torch.arange(1, block, device=device)]  # g_1..g_{B-1}
        engine_draft = torch.tensor([r['draft'][1:] for r in rows], device=device)
        agree[0] += int((top[..., 0] == engine_draft).sum())
        agree[1] += engine_draft.numel()
        engine_accept = torch.tensor([r['accept'] for r in rows], device=device)
        values = {'L_engine': engine_accept, 'L_hf': prefix_len(top[..., 0] == truth)}
        for k in KS:
            values[f'U_{k}'] = prefix_len((top[..., :k] == truth[..., None]).any(-1))
        if selector is not None:
            with torch.no_grad():
                slot_hidden = hidden[:, 1:]
                zero = torch.zeros_like(engine_accept)
                walk = greedy_walk(selector, top, top_logits, slot_hidden, ids[0, anchors], zero)
                values['L_sel'] = prefix_len(walk == truth)
                # P9: correction g_{L+1} (index L) as the predecessor of slot L+2.
                last = block - 2
                index = engine_accept.clamp(max=last)
                correction = truth.gather(1, index[:, None])[:, 0]
                supported = (
                    top.gather(1, index[:, None, None].expand(-1, 1, top.shape[-1]))[:, 0]
                    == correction[:, None]
                ).any(-1)
                eligible = (engine_accept < block - 1) & supported
                start = torch.where(eligible, engine_accept + 1, torch.full_like(index, block))
                corrected = greedy_walk(selector, top, top_logits, slot_hidden, correction, start)
            rewalk['rid'] += [rid] * len(rows)
            for name, tensor in (
                ('prefix_len', anchors),
                ('L_engine', engine_accept),
                ('eligible', eligible),
                ('rewalk', corrected),
                ('walk', walk),
                ('truth', truth),
            ):
                rewalk[name].append(tensor.cpu())
        domain = panel[rid]['domain']
        if args.save_cycles:
            saved['rid'] += [rid] * len(rows)
            saved['domain'] += [domain] * len(rows)
            for name, tensor in (
                ('prefix_len', anchors),
                ('candidates', top),
                ('candidate_logits', top_logits.to(torch.bfloat16)),
                ('truth', truth),
                ('engine_draft', engine_draft),
                *values.items(),
            ):
                saved[name].append(tensor.cpu())
        for group in (domain, 'all'):
            totals[group]['cycles'] += len(rows)
            for name, value in values.items():
                totals[group][name] += float(value.sum())
                for k in range(1, block):
                    survival[(group, name)][k] += float((value >= k).sum())
        if index % 10 == 0:
            print(f'{index + 1}/{len(cycles)} requests', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_cycles:
        torch.save(
            {
                **{
                    name: torch.cat(parts) if name not in ('rid', 'domain') else parts
                    for name, parts in saved.items()
                },
                'meta': {
                    'draft': args.draft,
                    'trace': str(args.trace),
                    'note': 'one row per kept verify cycle; slot k = 1..B-1 in the last '
                    'dimension; candidates sorted by drafter logit (top-16 through the '
                    'tied head, bf16 logits); truth = realized greedy continuation; '
                    'cycles whose block runs past the output end are excluded',
                },
            },
            args.out / 'cycles.pt',
        )
    if selector is not None:
        torch.save(
            {
                **{
                    name: torch.cat(parts) if name != 'rid' else parts
                    for name, parts in rewalk.items()
                },
                'meta': {
                    'selector': args.selector,
                    'selector_step': selector_step,
                    'rank': rank,
                    'flops_per_slot': 2 * rank * int(config['hidden_size'])
                    + 2 * max(KS) * rank
                    + rank,
                    'note': 'rows in cycles.pt order; slot index j = 0..B-2 is block '
                    'position j+1; walk = greedy selector walk from the anchor; rewalk = '
                    'for eligible rows (L_engine < B-1 and truth[L] in the top-K at index '
                    'L) the greedy walk of indices L+1..B-2 with truth[L] as the first '
                    'predecessor, -1 elsewhere',
                },
            },
            args.out / 'selector_rewalk.pt',
        )
    summary = {
        'draft': args.draft,
        'selector': args.selector,
        'trace': str(args.trace),
        'offline_argmax_agreement_with_engine': agree[0] / max(1, agree[1]),
        'domains': {
            group: {
                'cycles': int(t['cycles']),
                **{f'tau_{name}': 1 + t[name] / t['cycles'] for name in t if name != 'cycles'},
            }
            for group, t in totals.items()
        },
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.out / 'survival.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['domain', 'quantity', 'position', 'survival'])
        for (group, name), counts in sorted(survival.items()):
            for k in range(1, block):
                writer.writerow([group, name, k, counts[k] / totals[group]['cycles']])
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
