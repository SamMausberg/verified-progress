#!/usr/bin/env bash
# Classify the greedy outputs of run_admission_logprob.sh (CPU only):
#   experiments/admission/classify_logprob.sh [--unrecorded-repo-state] [OUT]
# OUT (default ~/vp-data/speed_highc/logprob) holds runs/ from the GPU hold. Every
# comparison of the delayed run with an undelayed one (both undelayed launches, at c = 64 and
# 128) is its own entry in arms.json, so bench.divergence classifies each of them; the third
# element of each entry is the launch-to-launch control at the same concurrency, reported
# beside the class. The script fails unless all four comparisons are exact-up-to-rounding.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
unrecorded_repo_state=0
if [ "${1:-}" = --unrecorded-repo-state ]; then
  unrecorded_repo_state=1
  shift
fi
case "${1:-}" in
  -*) echo "unknown option $1; usage: $0 [--unrecorded-repo-state] [OUT]" >&2; exit 2 ;;
esac
out="${1:-$HOME/vp-data/speed_highc/logprob}"
runs="$out/runs"
prompts="$HOME/vp-data/state/prompts/prompts_fresh.jsonl"
manifest="evidence/state_safety/prompt_manifest_fresh.json"
# The runs used stock SGLang at the paper's pin (engine/sglang/README.md:4, SETUP.md:50).
pin=bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824
cat > "$out/pairs.json" <<'PAIRS'
[
  ["mtp n0 vs n0 repeat c128", "mtp_s3_replayssm__adm_n0/c128", "mtp_s3_replayssm__adm_n0b/c128"],
  ["mtp pd vs n0 c128", "mtp_s3_replayssm__adm_n0/c128", "mtp_s3_replayssm__adm_pd/c128"],
  ["mtp pd vs n0 repeat c128", "mtp_s3_replayssm__adm_n0b/c128", "mtp_s3_replayssm__adm_pd/c128"],
  ["mtp n0 vs n0 repeat c64", "mtp_s3_replayssm__adm_n0/c64", "mtp_s3_replayssm__adm_n0b/c64"],
  ["mtp pd vs n0 c64", "mtp_s3_replayssm__adm_n0/c64", "mtp_s3_replayssm__adm_pd/c64"],
  ["mtp pd vs n0 repeat c64", "mtp_s3_replayssm__adm_n0b/c64", "mtp_s3_replayssm__adm_pd/c64"]
]
PAIRS
cat > "$out/arms.json" <<'ARMS'
[
  ["prefill delayer vs n0, c128", "mtp pd vs n0 c128", "mtp n0 vs n0 repeat c128"],
  ["prefill delayer vs n0 repeat, c128", "mtp pd vs n0 repeat c128", "mtp n0 vs n0 repeat c128"],
  ["prefill delayer vs n0, c64", "mtp pd vs n0 c64", "mtp n0 vs n0 repeat c64"],
  ["prefill delayer vs n0 repeat, c64", "mtp pd vs n0 repeat c64", "mtp n0 vs n0 repeat c64"]
]
ARMS
rm -f "$out/summary.json" "$out/report.json" "$out/classes.json" "$out/divergences.csv" \
  "$out/table.csv"
status=0
# The six passes of run_admission_logprob.sh (its three runs at c = 64 and 128) must each have
# their outputs and their metadata. Every pass must record the same repository and SGLang
# revisions with a clean SGLang tree (sglang_dirty false), and the same model revision, prompt
# count, token budget, top-k and pool pin; the runs differ only in the delay flags. A missing
# file, a pass without revisions, a dirty or unrecorded tree or any difference fails the check.
metas=()
passes=()
for run in mtp_s3_replayssm__adm_n0 mtp_s3_replayssm__adm_n0b mtp_s3_replayssm__adm_pd; do
  for pass in c64 c128; do
    for file in "$runs/$run/$pass.jsonl" "$runs/$run/$pass.meta.json"; do
      [ -s "$file" ] || { echo "$file is missing or empty" >&2; status=1; }
    done
    metas+=("$runs/$run/$pass.meta.json")
    passes+=("$runs/$run/$pass.jsonl")
  done
done
# One jq call per file: jq 1.6 given several files takes its exit status from the last one.
revisions=""
for meta in "${metas[@]}"; do
  if ! line="$(jq -er --arg pin "$pin" 'if (.repo_sha // "") == "" or (.sglang_sha // "") == ""
      then error("no revisions")
      elif .sglang_sha != $pin then error("SGLang \(.sglang_sha), not the pin")
      elif .sglang_dirty != false then error("SGLang tree dirty or not recorded")
      else "\(.repo_sha) \(.sglang_sha) \(.model_revision) \(.num_prompts) \(.max_new_tokens)"
        + " \(.top_logprobs_num) \(.pool_pin | tojson)" end' "$meta")"; then
    echo "$meta: no revisions, not the pinned SGLang, or a dirty SGLang tree" >&2
    status=1
  fi
  revisions+="$line"$'\n'
done
revisions="$(sort -u <<< "$revisions" | grep .)"
if [ "$(grep -c . <<< "$revisions")" != 1 ]; then
  echo "the passes differ in revisions or settings: $revisions" >&2
  status=1
fi
# run_admission_logprob.sh records the repository's state at launch (repo_state.json): it must be
# a clean tree at the passes' commit. A run from before it recorded one (the committed run) needs
# --unrecorded-repo-state, which says so.
state="$out/repo_state.json"
if [ -e "$state" ]; then
  jq -e --arg sha "${revisions%% *}" '.head == $sha and .dirty_files == []' "$state" > /dev/null \
    || { echo "$state: not a clean tree at the passes' commit" >&2; status=1; }
elif [ "$unrecorded_repo_state" = 1 ]; then
  echo "no repository state recorded at launch (--unrecorded-repo-state)" >&2
else
  echo "$state is missing; a run from before it was recorded needs --unrecorded-repo-state" >&2
  status=1
fi
# The prompt file must be the frozen fresh set (its count and input_ids_sha256 in the committed
# manifest), and every pass must have run exactly those prompts, each at its frozen length.
python - "$prompts" "$manifest" "${passes[@]}" <<'PROMPTS' || status=1
import json, sys
sys.path.insert(0, 'experiments/state_safety')
from prompts import ids_digest

prompts, manifest, *passes = sys.argv[1:]
items = [json.loads(line) for line in open(prompts) if line.strip()]
frozen = json.load(open(manifest))
if len(items) != frozen['num_prompts'] or ids_digest(items) != frozen['input_ids_sha256']:
    sys.exit(f'{prompts} is not the frozen set of {manifest}')
lengths = {item['id']: len(item['input_ids']) for item in items}
for path in passes:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    ran = {row['id']: row['prompt_tokens'] for row in rows}
    if len(rows) != len(lengths) or ran != lengths:
        sys.exit(f'{path}: its prompts are not the frozen set')
PROMPTS
python experiments/state_safety/compare.py --runs "$runs" --pairs "$out/pairs.json" \
  --out-json "$out/summary.json" --out-csv "$out/divergences.csv" \
  --out-table "$out/table.csv" > "$out/compare.log" 2>&1 || status=1
if grep -q '^skip' "$out/compare.log"; then
  echo "compare.py skipped a pair (missing run files)" >&2
  status=1
fi
python -m bench.divergence "$out/summary.json" --floor "mtp n0 vs n0 repeat c128" \
  --out "$out/report.json" --arms "$out/arms.json" --classes-out "$out/classes.json" \
  --expect-prompts "$(grep -c . "$prompts")" || status=1
if [ "$status" = 0 ]; then
  python - "$out/classes.json" "$out/summary.json" <<'CHECK' || status=1
import json, sys
classes = json.load(open(sys.argv[1]))
pairs = json.load(open(sys.argv[2]))['pairs']
# compare.py files a divergence it could not classify (no top-k logprobs at that position) as
# 'unknown', which bench.divergence's class does not see: every first divergence of every pair
# must carry one of the five known classes.
known = {'tie', 'one_ulp', 'near', 'large', 'not_argmax'}
unclassified = [
    name
    for name, p in pairs.items()
    if set(p['classes']) - known or sum(p['classes'].values()) != p['diverged']
]
if len(pairs) != 6 or unclassified:
    sys.exit(f'pairs with unclassified divergences: {unclassified or len(pairs)}')
# The class assumes equal pinned pools on both sides of every pair (as resolved by each server).
unpinned = [
    name
    for name, p in pairs.items()
    if not (p['pools_identical'] is True and p['pinned_a'] is True and p['pinned_b'] is True)
]
if unpinned:
    sys.exit(f'pairs without equal pinned pools: {unpinned}')
bad = [c['arm'] for c in classes if c.get('exactness') != 'exact-up-to-rounding']
if len(classes) != 4 or bad:
    sys.exit(f'not every delayed comparison is exact-up-to-rounding: {bad or len(classes)}')
print('all four delayed comparisons exact-up-to-rounding')
CHECK
fi
exit "$status"
