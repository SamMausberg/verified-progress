"""Calibration client for SGLANG_FP8_DENSE_ACT=calibrate: reset the server's maxima, send the tune split, save them.

Greedy chat requests with thinking on, as bench's workload sends them, max_tokens 256, concurrency 64. The final
file holds the per-layer activation maxima plus how they were collected; every layer must have seen input.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def read(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--dump', required=True, help="the server's SGLANG_FP8_DENSE_CALIB_OUT")
    ap.add_argument('--workload', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument(
        '--exclude-probe-prompts',
        action='store_true',
        help="leave out the logit probe's 48 prompts (held-out calibration)",
    )
    args = ap.parse_args()
    for _ in range(120):
        if os.path.exists(args.dump):
            break
        time.sleep(1)
    before = read(args.dump)['resets']
    Path(args.dump + '.reset').touch()
    while read(args.dump)['resets'] <= before:
        time.sleep(0.5)
    rows = [json.loads(line) for line in Path(args.workload).read_text().splitlines() if line]
    if any('confirm' in args.workload or r.get('split') == 'confirm' for r in rows):
        raise SystemExit('calibration must not use the confirm split')
    excluded = []
    if args.exclude_probe_prompts:
        # The logit probe's prompts: the first --per-domain (default 16,
        # experiments/moonshot/logit_probe.py:318) of each domain of the same workload, chosen by
        # its own select_prompts, so the probe is out of sample for this calibration.
        from experiments.moonshot.logit_probe import select_prompts

        excluded = [r['id'] for r in select_prompts(Path(args.workload), 16)]
        rows = [r for r in rows if r['id'] not in set(excluded)]
    t0 = time.time()

    def one(r):
        body = {
            'model': 'default',
            'messages': [{'role': 'user', 'content': r['text']}],
            'max_tokens': args.max_tokens,
            'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': True},
        }
        return post(args.url + '/v1/chat/completions', body)['usage']['completion_tokens']

    with ThreadPoolExecutor(args.concurrency) as ex:
        tokens = list(ex.map(one, rows))
    done = time.time()
    while read(args.dump)['time'] < done + 2.5:
        time.sleep(0.5)
    rec = read(args.dump)
    bad = [n for n in rec['amax'] if not (rec['amax'][n] > 0 and rec['calls'][n] > 0)]
    if bad:
        raise SystemExit(f'layers without calibration data: {bad[:5]} ({len(bad)})')
    out = {
        'amax': rec['amax'],
        'calls': rec['calls'],
        'prompts': len(rows),
        'excluded_prompt_ids': excluded,
        'completion_tokens': sum(tokens),
        'max_tokens': args.max_tokens,
        'workload': args.workload,
        'workload_sha256': hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
        'seconds': round(done - t0, 1),
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(
        f'calibrated {len(rec["amax"])} layers on {len(rows)} prompts, {sum(tokens)} tokens -> {args.out}'
    )


if __name__ == '__main__':
    main()
