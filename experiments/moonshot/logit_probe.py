"""Quality proxy for engine levers: greedy divergence and top-k logit KL against a reference.

Two measurements against a running SGLang server, on a fixed prompt set:

* ``generate`` (decode path): greedy, fixed-length continuations with the top-k
  log-probabilities of every output position. Compared with a reference run, it
  gives the first position where the two token streams differ and, on the shared
  prefix (identical context, so the distributions are comparable), the KL
  divergence KL(ref || cand) over the reference's top-k tokens. This is the path
  that exercises per-step state precision (the recurrent GDN state is read and
  written once per decode step).
* ``score`` (teacher-forced prefill path): the reference continuation is fed back
  as input and the server returns the top-k log-probabilities at every position,
  so every position has an identical context. It gives the argmax agreement with
  the reference token and the same KL at all positions, not just the shared
  prefix. It exercises weights and KV cache but runs GDN in the chunked prefill
  kernel, so it does not test decode-state precision.

Top-k KL is a proxy: a reference token missing from the candidate's top-k is
assigned the candidate's k-th log-probability (an upper bound on its probability,
so the missing term is underestimated). With k = 20 the reference's top-20 mass is
above 0.99 at most positions.

    python experiments/moonshot/logit_probe.py run --url http://127.0.0.1:30070 \
        --mode generate --out ~/vp-data/moonshot/quality/plain.generate.json
    python experiments/moonshot/logit_probe.py compare REF.json CAND.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
REPO_DIR = Path(__file__).resolve().parents[2]
WORKLOAD_CANDIDATES = (
    REPO_DIR / 'bench/workloads/mixed-v1/tune.jsonl',
    Path.home() / 'vp-wt/bench/bench/workloads/mixed-v1/tune.jsonl',
)


def default_workload() -> Path:
    for path in WORKLOAD_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit('mixed-v1 tune split not found; pass --workload')


def select_prompts(workload: Path, per_domain: int) -> list[dict[str, Any]]:
    """The first `per_domain` prompts of each domain, in file order."""
    rows = [json.loads(line) for line in workload.read_text().splitlines() if line]
    taken: dict[str, int] = {}
    chosen = []
    for row in rows:
        if taken.get(row['domain'], 0) < per_domain:
            taken[row['domain']] = taken.get(row['domain'], 0) + 1
            chosen.append(row)
    return chosen


def tokenize(prompts: list[dict[str, Any]], thinking: bool) -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    out = []
    for row in prompts:
        ids = tok.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=True,
        )
        if isinstance(ids, dict):
            ids = ids['input_ids']
        out.append(list(ids))
    return out


def post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.loads(response.read())
        return result


def top_list(entries: list[Any] | None) -> list[list[float]]:
    """[[logprob, token_id], ...] from SGLang's [logprob, token_id, text] triples."""
    if not entries:
        return []
    return [[float(e[0]), int(e[1])] for e in entries if e is not None and e[0] is not None]


def run_generate(url: str, ids: list[int], max_new: int, topk: int) -> dict[str, Any]:
    payload = {
        'input_ids': ids,
        'sampling_params': {'temperature': 0.0, 'max_new_tokens': max_new, 'ignore_eos': True},
        'return_logprob': True,
        'top_logprobs_num': topk,
        'logprob_start_len': -1,
    }
    result = post(f'{url}/generate', payload)
    meta = result['meta_info']
    return {
        'tokens': [int(e[1]) for e in meta['output_token_logprobs']],
        'top': [top_list(t) for t in meta['output_top_logprobs']],
        'spec_accept': meta.get('spec_accept_length') or meta.get('spec_verify_ct'),
    }


def run_score(url: str, ids: list[int], continuation: list[int], topk: int) -> dict[str, Any]:
    payload = {
        'input_ids': ids + continuation,
        'sampling_params': {'temperature': 0.0, 'max_new_tokens': 0},
        'return_logprob': True,
        'top_logprobs_num': topk,
        'logprob_start_len': len(ids),
    }
    result = post(f'{url}/generate', payload)
    meta = result['meta_info']
    # input_top_logprobs[j] is the distribution that predicts input position
    # logprob_start_len + j; the first entry predicts continuation[0]... but SGLang
    # reports None for the very first input position of the sequence only.
    tops = [top_list(t) for t in meta['input_top_logprobs']]
    tokens = [int(e[1]) for e in meta['input_token_logprobs']]
    return {'tokens': tokens, 'top': tops}


def cmd_run(args: argparse.Namespace) -> None:
    workload = Path(args.workload) if args.workload else default_workload()
    prompts = select_prompts(workload, args.per_domain)
    input_ids = tokenize(prompts, args.thinking)
    reference = None
    if args.mode == 'score':
        if not args.reference:
            raise SystemExit('--mode score needs --reference (a generate run)')
        reference = json.loads(Path(args.reference).read_text())
    started = time.time()

    def one(i: int) -> dict[str, Any]:
        if args.mode == 'generate':
            return run_generate(args.url, input_ids[i], args.max_new_tokens, args.topk)
        assert reference is not None
        return run_score(args.url, input_ids[i], reference['sequences'][i]['tokens'], args.topk)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        sequences = list(pool.map(one, range(len(prompts))))
    record = {
        'mode': args.mode,
        'url': args.url,
        'label': args.label,
        'workload': str(workload),
        'workload_sha256': hashlib.sha256(workload.read_bytes()).hexdigest(),
        'prompt_ids': [p['id'] for p in prompts],
        'thinking': args.thinking,
        'max_new_tokens': args.max_new_tokens,
        'topk': args.topk,
        'concurrency': args.concurrency,
        'seconds': round(time.time() - started, 1),
        'sequences': sequences,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    print(f'wrote {out} ({len(sequences)} sequences, {record["seconds"]} s)')


def kl_topk(ref: list[list[float]], cand: list[list[float]]) -> float:
    """KL(ref || cand) over the reference's top-k support, both renormalised to it."""
    if not ref or not cand:
        return math.nan
    cand_map = {tok: lp for lp, tok in cand}
    floor = min(lp for lp, _ in cand)
    ref_lp = [lp for lp, _ in ref]
    ref_z = math.log(sum(math.exp(lp) for lp in ref_lp))
    cand_lp = [cand_map.get(tok, floor) for _, tok in ref]
    cand_z = math.log(sum(math.exp(lp) for lp in cand_lp))
    return sum(
        math.exp(r - ref_z) * ((r - ref_z) - (c - cand_z))
        for r, c in zip(ref_lp, cand_lp, strict=True)
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def compare_runs(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    if ref['prompt_ids'] != cand['prompt_ids']:
        raise SystemExit('runs use different prompt sets')
    kls: list[float] = []
    first_div: list[int] = []
    agree = total = 0
    top1_logprob_delta: list[float] = []
    own_total = own_non_argmax = 0
    own_gap: list[float] = []
    for r, c in zip(ref['sequences'], cand['sequences'], strict=True):
        n = min(len(r['tokens']), len(c['tokens']))
        if cand['mode'] == 'score':
            # Teacher forced: every position has the same context. The score
            # run's argmax is its top-1 entry; compare it with the ref token.
            for j in range(n):
                if not c['top'][j]:
                    continue
                total += 1
                agree += int(c['top'][j][0][1] == r['tokens'][j])
                kls.append(kl_topk(r['top'][j], c['top'][j]))
            continue
        # Emitted tokens that are not the candidate's own argmax (relaxed acceptance
        # or sampling), and their log-probability gap to the argmax.
        for j, token in enumerate(c['tokens']):
            if not c['top'][j]:
                continue
            own_total += 1
            if token != c['top'][j][0][1]:
                own_non_argmax += 1
                chosen = next((lp for lp, t in c['top'][j] if t == token), c['top'][j][-1][0])
                own_gap.append(c['top'][j][0][0] - chosen)
        div = next((j for j in range(n) if r['tokens'][j] != c['tokens'][j]), n)
        first_div.append(div)
        for j in range(div):
            kls.append(kl_topk(r['top'][j], c['top'][j]))
            top1_logprob_delta.append(abs(r['top'][j][0][0] - c['top'][j][0][0]))
    kls = [k for k in kls if not math.isnan(k)]
    summary: dict[str, Any] = {
        'reference': ref.get('label'),
        'candidate': cand.get('label'),
        'mode': cand['mode'],
        'positions': len(kls),
        'kl_mean': statistics.fmean(kls) if kls else math.nan,
        'kl_p50': percentile(kls, 0.5),
        'kl_p99': percentile(kls, 0.99),
        'kl_max': max(kls) if kls else math.nan,
    }
    if cand['mode'] == 'score':
        summary['argmax_agreement'] = agree / total if total else math.nan
        summary['argmax_disagreements'] = total - agree
    else:
        length = len(ref['sequences'][0]['tokens'])
        diverged = [d for d in first_div if d < length]
        summary.update(
            {
                'sequences': len(first_div),
                'sequences_identical': len(first_div) - len(diverged),
                'first_divergence_median': statistics.median(first_div),
                'divergences_per_1k_shared_tokens': 1000 * len(diverged) / max(1, sum(first_div)),
                'top1_logprob_abs_delta_mean': statistics.fmean(top1_logprob_delta)
                if top1_logprob_delta
                else math.nan,
                'emitted_non_argmax_rate': own_non_argmax / own_total if own_total else math.nan,
                'emitted_non_argmax_logprob_gap_mean': statistics.fmean(own_gap)
                if own_gap
                else 0.0,
            }
        )
    return summary


def cmd_compare(args: argparse.Namespace) -> None:
    ref = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    print(json.dumps(compare_runs(ref, cand), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--url', required=True)
    run.add_argument('--mode', choices=['generate', 'score'], default='generate')
    run.add_argument('--reference', help='generate run whose tokens a score run feeds back')
    run.add_argument('--out', required=True)
    run.add_argument('--label', default=None)
    run.add_argument('--workload', default=None)
    run.add_argument('--per-domain', type=int, default=16)
    run.add_argument('--max-new-tokens', type=int, default=256)
    run.add_argument('--topk', type=int, default=20)
    run.add_argument('--concurrency', type=int, default=16)
    run.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=cmd_run)
    comp = sub.add_parser('compare')
    comp.add_argument('reference')
    comp.add_argument('candidate')
    comp.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
