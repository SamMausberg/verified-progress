"""Smoke test: does `dflash-tuned-b16` start and decode with FA4 attention on sm_90?

For each configuration (stock, FA4 for the drafter only, FA4 for target and
drafter) this launches the arm through bench's Server (launch checks recorded,
not enforced), sends a few greedy requests from the confirm workload one at a
time, records the generated token ids, the verify-step counts and the server's
resolved attention backends, and stops the server. Token agreement with the
stock configuration is reported as information only: this is a correctness
smoke, not an equality classification (that needs bench's equality runner).

    scripts/gpu_lock.sh -x python experiments/speed_lowc/fa4_smoke.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.arms import parse_overrides, resolve_arm
from bench.server import Server, http_post

CONFIGS = {
    'stock': [],
    'fa4_draft': ['speculative-draft-attention-backend=fa4'],
    'fa4_both': ['attention-backend=fa4', 'speculative-draft-attention-backend=fa4'],
}


def prompts(n: int) -> list[str]:
    path = REPO / 'bench/workloads/mixed-v2/confirm.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()[:n]]
    if len(rows) < n:
        raise SystemExit(f'{path} has fewer than {n} prompts')
    return [r['text'] for r in rows]


def run_config(
    name: str, sets: list[str], out: Path, port: int, texts: list[str], osl: int
) -> dict:
    arm = resolve_arm('dflash-tuned-b16', parse_overrides(sets, []), {})
    record: dict = {'config': name, 'sets': sets, 'started': time.time()}
    server = Server(arm, out / name, port, strict=False, startup_timeout=300.0)
    try:
        with server:
            record['ready_after_s'] = server.launch_record.get('ready_after_s')
            record['checks'] = [
                {'name': c.name, 'ok': c.ok, 'required': c.required, 'detail': c.detail}
                for c in server.checks
            ]
            info = server.server_info()
            record['attention_backend'] = info.get('attention_backend')
            record['draft_attention_backend'] = info.get('speculative_draft_attention_backend')
            outputs = []
            for text in texts:
                payload = {
                    'text': text,
                    'sampling_params': {'temperature': 0, 'max_new_tokens': osl,
                                        'ignore_eos': True},
                }  # fmt: skip
                t0 = time.monotonic()
                resp = json.loads(http_post(f'{server.base_url}/generate', payload, timeout=300))
                meta = resp.get('meta_info', {})
                outputs.append({
                    'output_ids': resp.get('output_ids'),
                    'text_head': resp.get('text', '')[:160],
                    'completion_tokens': meta.get('completion_tokens'),
                    'spec_verify_ct': meta.get('spec_verify_ct'),
                    'seconds': time.monotonic() - t0,
                })  # fmt: skip
            record['outputs'] = outputs
            record['ok'] = all(o['completion_tokens'] == osl for o in outputs)
    except Exception as exc:  # a failed start is the result this smoke is for
        record['ok'] = False
        record['error'] = repr(exc)[:4000]
        record['traceback'] = traceback.format_exc()[-4000:]
        record['log_tail'] = server.log_tail(80)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=30211)
    parser.add_argument('--prompts', type=int, default=6)
    parser.add_argument('--osl', type=int, default=128)
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    args = parser.parse_args()
    if args.prompts < 1 or args.osl < 1:
        parser.error('--prompts and --osl must be positive')
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    texts = prompts(args.prompts)
    records = [run_config(c, CONFIGS[c], out, args.port, texts, args.osl) for c in args.configs]
    ref = next((r for r in records if r['config'] == 'stock' and r.get('ok')), None)
    for r in records:
        if ref is not None and r is not ref and r.get('ok'):
            same = [a['output_ids'] == b['output_ids']
                    for a, b in zip(ref['outputs'], r['outputs'], strict=True)]  # fmt: skip
            r['identical_to_stock'] = f'{sum(same)}/{len(same)}'
        tokens = sum(o['completion_tokens'] or 0 for o in r.get('outputs', []))
        verifies = sum(o['spec_verify_ct'] or 0 for o in r.get('outputs', []))
        r['tokens_per_verify'] = tokens / verifies if verifies else None
        print(r['config'], 'ok' if r.get('ok') else 'FAILED', r.get('error', '')[:300],
              r.get('identical_to_stock'), r.get('tokens_per_verify'), flush=True)  # fmt: skip
    (out / 'smoke.json').write_text(json.dumps(records, indent=1, default=str) + '\n')
    sys.exit(0 if all(r.get('ok') for r in records) else 3)


if __name__ == '__main__':
    main()
