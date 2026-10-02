#!/usr/bin/env bash
# The stack campaign's analysis, end to end, into a fresh directory (no GPU):
#
#   experiments/stack/analyze_campaign.sh [<root>]      (default root ~/vp-data/stack/analysis)
#
# Every analysis gets its own directory <root>/<UTC time>/ and none is ever overwritten. It
# holds the outputs, the exact command lines (commands.sh) and run.json (the time and this
# repository's commit, which must have no uncommitted changes, so the code behind every
# output is committed). Steps, all on the campaign the pin names:
#   1. bench.pareto --points-only over every sweep run of the campaign (points.csv,
#      launches.csv; a launch whose server never started has no sweep.json and no points);
#   2. analyze.py, the declared statistics and decision (composition.json, .csv);
#   3. provenance.py, the checks amendment 1 moved to analysis time (provenance.json);
#   4. startup_memory.py, each server's start-up memory from its log (startup_memory.csv);
#   5. phases.py on the equality hold's phase diagnostic (B0 and FG at c = 1 and 8);
#   6. figures.py, the figure tables and figures (frontier, ratios, gap, last lever);
#   7. a copy of the equality hold's records (equality/: gate, comparison summary, pairs,
#      first divergences, check-mode statistics, plan, routing table, logs), whose gate and
#      summary must hash to the campaign pin's values.
# Steps 1-5 run with the repository venv; step 6 needs matplotlib (SGLang venv). Any
# failing step stops the script with its exit status, except step 3, whose failure is
# recorded and returned at the end.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
root=${1:-$HOME/vp-data/stack/analysis}
[ -z "$(git status --porcelain)" ] || { echo "uncommitted changes: commit the code first" >&2; exit 1; }
pin=$HOME/vp-data/stack/campaign_gate.json
campaign=$(sha256sum "$pin" | cut -c1-12)
runs=$HOME/vp-data/stack/runs/$campaign
cert_src=$(cat "$HOME/vp-data/stack/holds/cert_src")
repo_py=$HOME/verified-progress/.venv/bin/python  # the repository venv (AGENTS.md)
sgl_py=$HOME/sglang/.venv/bin/python
eq_run=$("$repo_py" -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' "$pin")
utc=$(date -u +%Y%m%dT%H%M%SZ)
out=$root/$utc
mkdir -p "$root"
mkdir "$out"  # fails if it exists: no analysis is overwritten
run() {
  printf '%q ' "$@" >>"$out/commands.sh"
  printf '\n' >>"$out/commands.sh"
  "$@"
}
printf '%s\n' "{\"utc\": \"$utc\", \"repo_commit\": \"$(git rev-parse HEAD)\", \"campaign\": \"$campaign\"}" \
  >"$out/run.json"
shopt -s nullglob
sweeps=("$runs"/stack-*/2026*)
(( ${#sweeps[@]} )) || { echo "no sweep runs under $runs" >&2; exit 1; }
run "$repo_py" -m bench.pareto "${sweeps[@]}" --out "$out" --points-only --status stack
run "$repo_py" experiments/stack/analyze.py --points "$out/points.csv" --campaign "$pin" \
  --runs-root "$runs" --out "$out/composition.json" --csv "$out/composition.csv" |
  tee "$out/analyze.log"
# A failed provenance check is a result (provenance.json records it); the remaining steps
# still run and the script exits non-zero at the end.
prov=0
run "$repo_py" experiments/stack/provenance.py --campaign "$pin" --cert-src "$cert_src" \
  --cert-commit 01502cc --prompts "$HOME/vp-data/state/prompts/prompts.jsonl" \
  --out "$out/provenance.json" || prov=$?
run "$repo_py" experiments/stack/startup_memory.py --runs-root "$runs" \
  --session-logs "$HOME"/vp-data/stack/session_s*.log \
  --equality "$eq_run" \
  --out "$out/startup_memory.csv"
for arm in B0 FG; do
  run "$repo_py" experiments/stack/phases.py "$eq_run/phases_$arm.jsonl" --out "$out/phases_$arm.json"
done
run "$sgl_py" experiments/stack/figures.py --points "$out/points.csv" \
  --composition "$out/composition.json" --expected evidence/stack/expected.json \
  --ceiling evidence/stack/ceiling.json --frame evidence/frontier/frame.json \
  --bench-frontier evidence/bench/confirm/frontier.csv --out-dir "$out"
mkdir "$out/equality"
for f in gate.json summary.json pairs.json table.csv divergences.csv plan.jsonl identity.json \
  certified_stats_H.json certified_stats_FGH.json backbone_table_v1.json compare.log hold.log; do
  cp "$eq_run/$f" "$out/equality/$f"
done
for f in gate summary; do
  want=$("$repo_py" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$pin" "${f}_sha256")
  [ "$(sha256sum "$out/equality/$f.json" | cut -d' ' -f1)" = "$want" ] ||
    { echo "copied $f.json does not match the campaign pin" >&2; exit 1; }
done
echo "analysis in $out"
(( prov == 0 )) || { echo "provenance checks failed (provenance.json)" >&2; exit "$prov"; }
