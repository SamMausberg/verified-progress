#!/usr/bin/env bash
# Regenerate the committed state-safety evidence from the raw runs in
# ~/vp-data/state (matrix runs, targeted tests). Single-threaded and light; the
# tensor-tap analyses (mechanism.py over many prompts) run inside the GPU holds
# that collect them (see README.md), because they read tens of thousands of files.
#
#   experiments/state_safety/analyze_all.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
evidence="$here/../../evidence/state_safety"
runs="$HOME/vp-data/state/runs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$here"

nice -n 19 python compare.py --pairs pairs.json \
  --out-json "$evidence/noise_floor.json" --out-table "$evidence/noise_floor.csv" \
  --out-csv "$evidence/divergences.csv" --out-meta "$evidence/run_meta.json"

# Top-2 BF16 gap statistics of plain decode at batch 1, kept with the noise floor.
nice -n 19 python - "$runs/plain/c1.jsonl" "$evidence/noise_floor.json" <<'PY'
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

# Drift and divergences by rejection position, for every speculative config.
for spec in mtp_s1 mtp_s3 mtp_s5 mtp_tree; do
  if [ -f "$runs/$spec/c1.jsonl" ]; then
    nice -n 19 python cycles.py --ref plain/c1 --spec "$spec/c1" \
      --out "$evidence/cycles_$spec.json" > /dev/null
  fi
done

if compgen -G "$HOME/vp-data/state/targeted/*.json" > /dev/null; then
  nice -n 19 python summarize_targeted.py --out "$evidence/targeted.json" > /dev/null
fi
echo "evidence regenerated in $evidence"
