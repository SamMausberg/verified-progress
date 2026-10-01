"""Collect the targeted-test outputs into one evidence file.

Keeps each test's summary, its run metadata and every case that was not
identical (with its divergence position, margins and class), lists every pair
the history test served, and adds the
chunked-prefill comparisons against the unchunked run of the same config.

    python experiments/state_safety/summarize_targeted.py \
        --out evidence/state_safety/targeted.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server import public_server_info


def not_identical(test: str, case: dict[str, Any]) -> bool:
    if test == 'truncation':
        return not case['identical'] or not case.get('logprobs_identical', True)
    if test == 'stops':
        return not (
            case['stop_output_identical']
            and case.get('stop_logprobs_identical', True)
            and case['extension_warm_vs_cold']['identical']
        )
    if test == 'prefix':
        return not case['warm_vs_cold']['identical'] or not case.get('truncated_identical', True)
    if test == 'abort':
        return not case['identical']
    if test == 'history':
        return not case['logprobs_identical']
    if test == 'repeat':
        return case['logprobs_bitwise_identical'] < case['repeats'] - 1
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', default=str(Path.home() / 'vp-data/state/targeted'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.dir)
    out: dict[str, Any] = {}
    for path in sorted(root.glob('*.json')):
        if path.name.startswith('compare_prefill'):
            continue
        data = json.loads(path.read_text())
        meta = data['meta']
        test = meta['test']
        entry: dict[str, Any] = {
            'summary': data['summary'],
            'flags': meta['flags'],
            'server_info': public_server_info(meta['server_info']),
            'repo_sha': meta['repo_sha'],
            'sglang_sha': meta['sglang_sha'],
            'wall_s': meta['wall_s'],
        }
        if test != 'prefill':
            entry['non_identical_cases'] = [c for c in data['cases'] if not_identical(test, c)]
        if test == 'history':
            # Every pair tested, identical or not: which prompt followed which.
            entry['pairs'] = [
                {
                    k: c[k]
                    for k in (
                        'id',
                        'predecessor',
                        'shared_prefix_tokens',
                        'tokens_identical',
                        'first_logprob_difference',
                    )
                }
                for c in data['cases']
            ]
            # Prefills that reused a cached prefix, from the server log: with none, the
            # only sharing between requests is the repoint after the prefill.
            log = path.with_suffix('.server.log')
            if log.exists():
                cached = re.findall(r'#cached-token: (\d+)', log.read_text())
                entry['prefill_cache_hits'] = {
                    'prefills': len(cached),
                    'with_cached_tokens': sum(int(c) > 0 for c in cached),
                }
        out[path.stem] = entry

    # Chunked prefill: compare each chunked run with the unchunked run.
    for chunked in sorted(root.glob('prefill__*__chunk*.json')):
        base = chunked.with_name(chunked.name.split('__chunk')[0] + '.json')
        chunk = int(chunked.stem.split('__chunk')[1])
        cmp_path = root / f'compare_{chunked.stem}.json'
        subprocess.run(
            [
                sys.executable,
                str(HERE / 'compare_prefill.py'),
                '--a',
                str(base),
                '--b',
                str(chunked),
                '--chunk',
                str(chunk),
                '--out',
                str(cmp_path),
            ],
            check=True,
            capture_output=True,
        )
        res = json.loads(cmp_path.read_text())
        res['a'], res['b'] = base.name, chunked.name
        out[f'compare_{chunked.stem}'] = res

    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    for name, entry in out.items():
        print(name, json.dumps(entry.get('summary', entry.get('generation')))[:300])


if __name__ == '__main__':
    main()
