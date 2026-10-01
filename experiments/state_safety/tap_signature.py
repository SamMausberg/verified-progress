"""Where two tapped sessions of the same configuration first part ways.

The v1 and v3 taps both served the same 167 prompts, in the same order, at the
same configuration and batch shape. For each prompt whose client output
(tokens or logprobs, over the length both sessions generated) differs between
the two sessions, this finds the first position at which a module output
hashed the same way by both tap versions differs, and lists the modules that
differ there in execution order. It also records whether both sessions served
the prompts in the same order and how many prefills reused a cached prefix,
from the server logs. The GDN
core, its gated norm and the logits are left out, because the v1 tap did not
hash them per token (README, "v1 withdrawal").

    python experiments/state_safety/tap_signature.py \
        --pair ~/vp-data/state/tap/plain_c1 ~/vp-data/state/tap/v3_plain_c1 \
        --out evidence/state_safety/tap_signature.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mechanism import committed_rows, exec_key, load_client

_NOT_COMPARABLE = ('linear_attn.attn', 'linear_attn.norm', 'gdn_', 'logits')
_PREFILL = re.compile(r'Prefill batch, #new-seq: \d+, #new-token: \d+, #cached-token: (\d+)')


def prefill_cache_hits(run_dir: Path) -> dict[str, int]:
    """Prefills in the server log, and how many reused a cached prefix."""
    text = (run_dir / 'server.log').read_text().splitlines()
    cached = [int(m.group(1)) for m in map(_PREFILL.search, text) if m]
    return {'prefills': len(cached), 'with_cached_tokens': sum(c > 0 for c in cached)}


def first_split(dir_a: Path, dir_b: Path, pid: str, prompt: list[int], ca, cb) -> dict[str, Any]:
    na = json.loads((dir_a / 'tap' / 'slots.json').read_text())['names']
    nb = json.loads((dir_b / 'tap' / 'slots.json').read_text())['names']
    common = sorted(
        (n for n in set(na) & set(nb) if not any(k in n for k in _NOT_COMPARABLE)),
        key=exec_key,
    )
    ra = committed_rows(dir_a / 'tap' / f'tap-{pid}', prompt + ca['output_ids'])
    rb = committed_rows(dir_b / 'tap' / f'tap-{pid}', prompt + cb['output_ids'])
    P = len(prompt)
    for q in sorted(ra.keys() & rb.keys()):
        a, b = ra[q], rb[q]
        diff = [
            n
            for n in common
            if a['hash_rows'][na.index(n)] >= a['num_tokens']
            and b['hash_rows'][nb.index(n)] >= b['num_tokens']
            and a['hash'][na.index(n)] != b['hash'][nb.index(n)]
        ]
        if diff:
            return {
                'position': q,
                'output_index': q - P + 1 if q >= P else None,
                'mode_a': a['mode'],
                'mode_b': b['mode'],
                'first_modules': diff[:4],
                'differing_modules': len(diff),
                'compared_modules': len(common),
            }
    return {'position': None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('A', 'B'))
    ap.add_argument('--prompts', default=str(Path.home() / 'vp-data/state/prompts/prompts.jsonl'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    prompts = {
        json.loads(line)['id']: json.loads(line)['input_ids']
        for line in Path(args.prompts).read_text().splitlines()
    }
    out: list[dict[str, Any]] = []
    for a, b in args.pair:
        dir_a, dir_b = Path(a), Path(b)
        ca, cb = load_client(dir_a), load_client(dir_b)
        cases = {}
        for pid in sorted(ca.keys() & cb.keys()):
            # The sessions may cap output lengths differently: compare the common part.
            n = min(len(ca[pid]['output_ids']), len(cb[pid]['output_ids']))
            if (ca[pid]['output_ids'][:n], ca[pid]['top_logprobs'][:n]) == (
                cb[pid]['output_ids'][:n],
                cb[pid]['top_logprobs'][:n],
            ):
                continue
            cases[pid] = first_split(dir_a, dir_b, pid, prompts[pid], ca[pid], cb[pid])
        out.append(
            {
                'a': dir_a.name,
                'b': dir_b.name,
                'prompts': len(ca.keys() & cb.keys()),
                'same_order': [r['id'] for r in ca.values()] == [r['id'] for r in cb.values()],
                'prefill_cache_hits_a': prefill_cache_hits(dir_a),
                'prefill_cache_hits_b': prefill_cache_hits(dir_b),
                'differing': cases,
            }
        )
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    for entry in out:
        print(entry['a'], entry['b'], len(entry['differing']), 'differ')


if __name__ == '__main__':
    main()
