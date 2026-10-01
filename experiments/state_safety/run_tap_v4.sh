#!/usr/bin/env bash
# Tap v4: module outputs and the caches each forward reads, for plain vs MTP steps 3,
# radix on vs off, humaneval-0044 repeats and the two history pairs. Needs the state
# tap engine patch (engine/sglang/patches/state/0001-state-tap.patch) applied in
# ~/sglang-wt/state. Run in one shared hold:
#
#   scripts/gpu_lock.sh -s experiments/state_safety/run_tap_v4.sh
#
# The summaries it writes under ~/vp-data/state/tap are copied to evidence/state_safety
# as cachecheck_v4_*.json, history_v4_*.json and repeats_v4_h44_*.json.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="$HOME/sglang-wt/state"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
cd "$here"
T="$HOME/vp-data/state/tap"
U="$HOME/vp-data/state/runs/plain/c1.jsonl"
# The tapped prompt lists and per-prompt token limits, committed next to this script.
I="$here/tap_v4_inputs"
mkdir -p "$T"
rm -rf "$T"/v4_*

echo "v4 smoke $(date +%T)"
python tap_runs.py --config plain --concurrency 1 --tap-ids "$I/ids_smoke.txt" \
  --out-dir "$T/v4_smoke" --max-new-tokens 16 --port 30056
python - "$T/v4_smoke/client.jsonl" "$U" <<'PY'
import json
import sys

tap = {json.loads(x)['id']: json.loads(x) for x in open(sys.argv[1])}
ref = {json.loads(x)['id']: json.loads(x) for x in open(sys.argv[2])}
ok = all(
    r['output_ids'] == ref[i]['output_ids'][:16] and r['top_logprobs'] == ref[i]['top_logprobs'][:16]
    for i, r in tap.items()
)
print('v4 smoke: bitwise equal to untapped run:', ok)
sys.exit(0 if ok else 1)
PY

echo "v4 sessions $(date +%T)"
for config in plain plain_noradix mtp_s3; do
  python tap_runs.py --config "$config" --concurrency 1 --tap-ids "$I/ids_v4.txt" \
    --limits "$I/limits_c1.json" --out-dir "$T/v4_${config}_c1" --port 30056
done

echo "h44 $(date +%T)"
python tap_runs.py --config plain --concurrency 1 --repeats 5 --tap-ids "$I/ids_h44.txt" \
  --out-dir "$T/v4_h44_noflush" --max-new-tokens 4 --port 30056
python tap_runs.py --config plain --concurrency 1 --repeats 5 --flush-each \
  --tap-ids "$I/ids_h44.txt" --out-dir "$T/v4_h44_flush" --max-new-tokens 4 --port 30056

echo "history pairs $(date +%T)"
# Each prompt served alone and right after its predecessor, each on a fresh server.
for pair in mt_bench-0054:mt_bench-0056 humaneval-0000:humaneval-0008; do
  x="${pair%%:*}"
  y="${pair##*:}"
  printf '%s\n%s\n' "$x" "$y" > "$T/ids_hist_$y.txt"
  printf '%s\n' "$y" > "$T/ids_alone_$y.txt"
  python tap_runs.py --config plain --concurrency 1 --tap-ids "$T/ids_hist_$y.txt" \
    --out-dir "$T/v4_hist_after_$y" --max-new-tokens 64 --port 30056
  python tap_runs.py --config plain --concurrency 1 --tap-ids "$T/ids_alone_$y.txt" \
    --out-dir "$T/v4_hist_alone_$y" --max-new-tokens 64 --port 30056
done

echo "analysis $(date +%T)"
python mechanism.py --a "$T/v4_plain_c1" --b "$T/v4_mtp_s3_c1" --untapped-a "$U" \
  --out "$T/cachecheck_v4_plain_c1_vs_mtp_s3_c1.json" > /dev/null
python mechanism.py --a "$T/v4_plain_c1" --b "$T/v4_plain_noradix_c1" --untapped-a "$U" \
  --out "$T/cachecheck_v4_plain_c1_radix_vs_noradix.json" > /dev/null
for flush in noflush flush; do
  python mechanism.py --a "$T/v4_h44_$flush" --repeat-of humaneval-0044 \
    --out "$T/repeats_v4_h44_$flush.json" > /dev/null
done
for y in mt_bench-0056 humaneval-0008; do
  python mechanism.py --a "$T/v4_hist_alone_$y" --b "$T/v4_hist_after_$y" \
    --out "$T/history_v4_$y.json" > /dev/null
done
echo "done $(date +%T)"
