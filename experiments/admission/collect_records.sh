#!/usr/bin/env bash
# Build the record files of evidence/admission/ from the raw runs (CPU only, jq):
#   experiments/admission/collect_records.sh [DATA] [OUT]
# DATA defaults to ~/vp-data/speed_highc, OUT to evidence/admission. launches.csv has one row
# per server of probes 1-3 (bench's server/launch.json and the run's sweep.json);
# prefill_requests.json condenses the two prefill-probe servers' client.json; logprob_runs.csv
# lists the logprob servers' resolved pools, prefill batches and commits.
set -euo pipefail
data="${1:-$HOME/vp-data/speed_highc}"
out="${2:-evidence/admission}"
probes=(probe1:admission probe2:queue-delay probe3:natural-20261002T195727Z)
prefill="$data/prefill-20261002T175303Z"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

{
  echo "probe,label,session,repo_head,sglang_head,sglang_branch,env,command"
  for spec in "${probes[@]}"; do
    probe=${spec%%:*}
    dir="$data/${spec#*:}"
    found=0
    for launch in "$dir"/*/[0-9]*-[0-9]*/server/launch.json; do
      [ -e "$launch" ] || continue
      found=1
      run="$(dirname "$(dirname "$launch")")"
      label="$(basename "$(dirname "$run")")"
      session="$(jq -r '.session' "$run/sweep.json")"
      jq -r --arg p "$probe" --arg l "$label" --arg s "$session" \
        '[$p, $l, $s, .repo.head, .sglang_source.head, .sglang_source.branch,
          (.env_overrides | to_entries | map("\(.key)=\(.value)") | join(" ")),
          (.command[3:] | join(" "))] | @csv' "$launch"
    done
    [ "$found" = 1 ] || { echo "no launch records under $dir" >&2; exit 1; }
  done
} > "$tmp/launches.csv"

jq -n --slurpfile s "$prefill/stock/client.json" --slurpfile f "$prefill/fi-prefill/client.json" '
def summ(c): {arm: c.arm, server_under_nsys_launch: (c.command[0]=="nsys"), linear_attn_prefill_backend_flag: (c.args["linear-attn-prefill-backend"] // null),
  untraced: {single_median_ms: c.untraced.sequential_median_ms, concurrent8_wall_median_ms: c.untraced.concurrent8_wall_median_ms,
    single_ms_by_isl: [c.untraced.sequential[] | {isl, ms}]},
  traced: (if c.traced then {single_median_ms: c.traced.sequential_median_ms, concurrent8_wall_median_ms: c.traced.concurrent8_wall_median_ms} else null end)};
{run: "prefill-20261002T175303Z", requests: "30 confirm-split prompts one at a time (max_tokens 1, non-streaming, thinking on), then 5 rounds of 8 concurrent",
 resolved_gdn_backends: "decode=triton, prefill=flashinfer, verify=triton in both servers (server logs)",
 stock: summ($s[0]), explicit_flashinfer_prefill: summ($f[0])}' > "$tmp/prefill_requests.json"

jq '{gap_ms_between_windows: 20, windows: .windows, first_30_median: .first_30_median, rows: .rows}' \
  "$prefill/stock/trace_summary.json" > "$tmp/prefill_trace.json"
cp "$prefill/gdn_prefill_bench.json" "$tmp/gdn_prefill_bench.json"

{
  echo "run,prefill_batches,max_total_num_tokens,max_mamba_cache_size,max_running_requests,sglang_sha,repo_sha"
  for run in "$data"/logprob/runs/*; do
    log="$run/server.log"
    echo "$(basename "$run"),$(grep -c 'Prefill batch' "$log"),$(grep -o 'max_total_num_tokens=[0-9]*' "$log" | tail -1 | cut -d= -f2),$(grep -o 'max_mamba_cache_size: [0-9]*' "$log" | tail -1 | cut -d' ' -f2),$(grep -o 'max_running_requests=[0-9]*' "$log" | tail -1 | cut -d= -f2),$(jq -r .sglang_sha "$run/c128.meta.json"),$(jq -r .repo_sha "$run/c128.meta.json")"
  done
} > "$tmp/logprob_runs.csv"

mkdir -p "$out"
for file in launches.csv prefill_requests.json prefill_trace.json gdn_prefill_bench.json \
  logprob_runs.csv; do
  mv "$tmp/$file" "$out/$file"
done
