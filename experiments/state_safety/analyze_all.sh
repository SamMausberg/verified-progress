#!/usr/bin/env bash
# Regenerate the committed state-safety evidence from the raw runs in
# ~/vp-data/state (matrix runs, targeted tests). Single-threaded and light; the
# tensor-tap analyses (mechanism.py over many prompts) run inside the GPU holds
# that collect them (see README.md), because they read tens of thousands of files.
#
#   experiments/state_safety/analyze_all.sh
#
# Two run roots: runs_pinned/ (pinned pools, the current matrix; *_pinned outputs)
# and runs/ (the first, unpinned matrix; the unsuffixed outputs). Once runs_pinned/
# exists, every pair in pairs_pinned.json must have both runs, or the script fails;
# STATE_ALLOW_MISSING=1 regenerates from what has landed so far.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
evidence="$repo/evidence/state_safety"
runs="$HOME/vp-data/state/runs"
pinned="$HOME/vp-data/state/runs_pinned"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
cd "$here"

nice -n 19 python compare.py --runs "$runs" --pairs pairs.json \
  --out-json "$evidence/noise_floor.json" --out-table "$evidence/noise_floor.csv" \
  --out-csv "$evidence/divergences.csv" --out-meta "$evidence/run_meta.json"
if [ -d "$pinned" ]; then
  require=(--require-all)
  if [ "${STATE_ALLOW_MISSING:-0}" = 1 ]; then require=(); fi
  nice -n 19 python compare.py --runs "$pinned" --pairs pairs_pinned.json "${require[@]}" \
    --out-json "$evidence/noise_floor_pinned.json" \
    --out-table "$evidence/noise_floor_pinned.csv" \
    --out-csv "$evidence/divergences_pinned.csv" --out-meta "$evidence/run_meta_pinned.json"
fi

# One deliberate cross-regime pair: does the pool regime alone change batch-1 output?
if [ -f "$pinned/plain/c1.jsonl" ] && [ -f "$runs/plain/c1.jsonl" ]; then
  nice -n 19 python compare.py --runs "$HOME/vp-data/state" --pairs pairs_cross_regime.json \
    --allow-mixed-pins --all-logprob-differences --out-json "$evidence/cross_regime.json" \
    --out-csv "$evidence/divergences_cross_regime.csv" > /dev/null
fi

# Top-2 BF16 gap statistics of plain decode at batch 1, kept with the noise floor.
for root_floor in "$runs:noise_floor.json" "$pinned:noise_floor_pinned.json"; do
  root="${root_floor%%:*}"
  floor="$evidence/${root_floor##*:}"
  [ -f "$root/plain/c1.jsonl" ] && [ -f "$floor" ] || continue
  nice -n 19 python - "$root/plain/c1.jsonl" "$floor" <<'PY'
import json
import sys

n = tie = small = frag = 0
for line in open(sys.argv[1]):
    for top in json.loads(line)['top_logprobs']:
        v = sorted((x[0] for x in top), reverse=True)
        if len(v) < 2:
            continue
        n += 1
        g = v[0] - v[1]
        tie += g < 1e-6
        small += 1e-6 <= g <= 0.1251
        frag += g <= 0.25
d = json.load(open(sys.argv[2]))
d['top2_gap_plain_c1'] = {
    'run': 'plain/c1',
    'positions': n,
    'exact_tie': tie,
    'exact_tie_per_1k': round(1000 * tie / n, 2),
    'gap_nonzero_le_0.125': small,
    'gap_nonzero_le_0.125_per_1k': round(1000 * small / n, 2),
    'gap_le_0.25_per_1k': round(1000 * frag / n, 1),
}
open(sys.argv[2], 'w').write(json.dumps(d, indent=2) + '\n')
PY
done

# First differing module per comparison, one row per module, for the paper's figure.
nice -n 19 python - "$evidence" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

KINDS = {
    'linear_attn.gdn_core': 'gdn_core',
    'linear_attn.gdn_conv': 'gdn_conv',
    'linear_attn.norm': 'gated_norm',
    'linear_attn.attn': 'gdn_block',
    'attn': 'attn',
    'mlp.down_proj': 'mlp_down_proj',
}
evidence = Path(sys.argv[1])
rows = []
for path in sorted(evidence.glob('mechanism_*.json')):
    pair = path.stem[len('mechanism_'):]
    summary = json.loads(path.read_text())['summary']
    for module, count in summary['first_difference_module'].items():
        m = re.match(r'model\.layers\.(\d+)\.(.+)$', module)
        layer, suffix = (int(m.group(1)), m.group(2)) if m else ('', module)
        kind = KINDS.get(suffix, suffix.replace('.', '_'))
        rows.append([pair, module, layer, kind, count])
with open(evidence / 'first_difference_by_module.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pair', 'module', 'layer', 'kind', 'count'])
    w.writerows(rows)
PY

# Tap neutrality: which tapped prompts differ from the untapped matrix run, and where.
nice -n 19 python - "$HOME/vp-data/state" "$evidence/tap_check.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = {}
for session, ref in (('v3_plain_c1', 'plain/c1'), ('plain_c1', 'plain/c1')):
    path = root / 'tap' / session / 'client.jsonl'
    if not path.exists():
        continue
    tap = {json.loads(x)['id']: json.loads(x) for x in path.read_text().splitlines()}
    base = {
        json.loads(x)['id']: json.loads(x)
        for x in (root / 'runs' / f'{ref}.jsonl').read_text().splitlines()
    }
    mismatched = {}
    for pid, r in sorted(tap.items()):
        if not r.get('tapped', True):
            continue
        n = len(r['output_ids'])
        b = base[pid]
        if r['output_ids'] == b['output_ids'][:n] and r['top_logprobs'] == b['top_logprobs'][:n]:
            continue
        first = next(
            (i for i, (x, y) in enumerate(zip(r['top_logprobs'], b['top_logprobs'])) if x != y),
            None,
        )
        tok = next(
            (i for i, (x, y) in enumerate(zip(r['output_ids'], b['output_ids'])) if x != y),
            None,
        )
        entry = {
            'first_logprob_difference': first,
            'tokens_equal': tok is None,
        }
        if tok is not None:
            # Top-2 logprob gap at the first changed token, in each run (a gap of 0 is
            # an exact tie; 0.125 is one BF16 step for logits in [16, 32)).
            entry['first_token_difference'] = tok
            entry['top2_gap_tapped'] = r['top_logprobs'][tok][0][0] - r['top_logprobs'][tok][1][0]
            entry['top2_gap_untapped'] = b['top_logprobs'][tok][0][0] - b['top_logprobs'][tok][1][0]
        mismatched[pid] = entry
    tag = 'tap v3 (module rows per token)' if session.startswith('v3') else 'tap v1'
    out[session] = {
        'tap_version': tag,
        'untapped_run': ref,
        'tapped_prompts': sum(1 for r in tap.values() if r.get('tapped', True)),
        'mismatched': mismatched,
    }
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
PY

# Where the v1 and v3 tapped sessions (same prompts, same order) first part ways.
tap="$HOME/vp-data/state/tap"
pairs=()
for cfg in plain_c1 mtp_s3_c1; do
  if [ -f "$tap/$cfg/client.jsonl" ] && [ -f "$tap/v3_$cfg/client.jsonl" ]; then
    pairs+=(--pair "$tap/$cfg" "$tap/v3_$cfg")
  fi
done
if [ ${#pairs[@]} -gt 0 ]; then
  nice -n 19 python tap_signature.py "${pairs[@]}" --out "$evidence/tap_signature.json" > /dev/null
fi

# Pools of both servers for every comparison in the evidence (team rule: equality
# comparisons pin the pools; the earlier runs did not, so record what each had).
nice -n 19 python pools.py --root "$HOME/vp-data/state" --evidence "$evidence"

# Drift and divergences by rejection position, for every speculative config.
for spec in mtp_s1 mtp_s3 mtp_s5 mtp_tree; do
  if [ -f "$runs/$spec/c1.jsonl" ]; then
    nice -n 19 python cycles.py --runs "$runs" --ref plain/c1 --spec "$spec/c1" \
      --out "$evidence/cycles_$spec.json" > /dev/null
  fi
  if [ -f "$pinned/$spec/c1.jsonl" ] && [ -f "$pinned/plain/c1.jsonl" ]; then
    nice -n 19 python cycles.py --runs "$pinned" --ref plain/c1 --spec "$spec/c1" \
      --out "$evidence/cycles_${spec}_pinned.json" > /dev/null
  fi
done

if compgen -G "$HOME/vp-data/state/targeted/*.json" > /dev/null; then
  nice -n 19 python summarize_targeted.py --out "$evidence/targeted.json" > /dev/null
fi
echo "evidence regenerated in $evidence"
