#!/usr/bin/env bash
# Build the record files of evidence/admission/ from the raw runs (CPU only, jq):
#   experiments/admission/collect_records.sh [DATA] [OUT]
# DATA defaults to ~/vp-data/speed_highc, OUT to evidence/admission. launches.csv has one row
# per server of probes 1-3 and confirmation sessions 0-2 (bench's server/launch.json and the
# run's sweep.json); prefill_requests.json condenses the two prefill-probe servers'
# client.json; logprob_runs.csv lists the logprob servers' resolved pools, prefill batches and
# commits.
set -euo pipefail
data="${1:-$HOME/vp-data/speed_highc}"
out="${2:-evidence/admission}"
# The engine each group ran on: probes 1-3 on the pin plus drafter 0001-0004 committed in
# ~/sglang-wt/speed_highc (evidence/admission/README.md, "Provenance"), the confirmation on the
# stock pin (engine/sglang/README.md:4).
patched=8f4225186e13d4245c3b9bfadbc7f62aea694463
pin=bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824
# probe:directory:engine:every label its hold script launched (run_admission_probe.sh:35-45,
# run_queue_delay_probe.sh:34-38, run_natural_probe.sh:24-26 for the length generator and
# 50-55, run_admission_confirm.sh:52-62); each must have exactly one launch record, on that
# engine, and no other may appear.
probes=(
  "probe1:admission:$patched:dflash-fold-n16 dflash-fold-n4 dflash-n16 dflash-n4 mtp-n0 mtp-n32 mtp-n8 plain-tuned replayssm"
  "probe2:queue-delay:$patched:mtp-n0 mtp-pd plain-pd plain-tuned replayssm"
  "probe3:natural-20261002T195727Z/gen:$patched:natural-gen"
  "probe3:natural-20261002T195727Z:$patched:dflash-fold-pd mtp-n0 mtp-pd plain-pd plain-tuned replayssm"
  "confirm-s0:confirm/s0:$pin:dflash dflash-delay mtp-delay mtp-n0 plain-delay plain-tuned replayssm"
  "confirm-s1:confirm/s1:$pin:dflash dflash-delay mtp-delay mtp-n0 plain-delay plain-tuned replayssm"
  "confirm-s2:confirm/s2:$pin:dflash dflash-delay mtp-delay mtp-n0 plain-delay plain-tuned replayssm"
)
# The three servers of run_admission_logprob.sh:28.
logprob_runs=(mtp_s3_replayssm__adm_n0 mtp_s3_replayssm__adm_n0b mtp_s3_replayssm__adm_pd)
prefill="$data/prefill-20261002T175303Z"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

{
  echo "probe,label,session,repo_head,sglang_head,sglang_branch,env,command"
  for spec in "${probes[@]}"; do
    IFS=: read -r probe sub engine expected <<< "$spec"
    dir="$data/$sub"
    found=()
    for launch in "$dir"/*/[0-9]*-[0-9]*/server/launch.json; do
      [ -e "$launch" ] || continue
      run="$(dirname "$(dirname "$launch")")"
      label="$(basename "$(dirname "$run")")"
      found+=("$label")
      session="$(jq -er '.session' "$run/sweep.json")"
      # A record without its repository and engine revisions, or whose repository or engine
      # tree had tracked modifications (dirty_files, absent counting as unknown), cannot
      # identify the code that ran: stop.
      jq -er --arg p "$probe" --arg l "$label" --arg s "$session" --arg e "$engine" \
        'if [.repo.head, .sglang_source.head, .sglang_source.branch]
            | any(. == null or . == "") then error("\(input_filename): no commit metadata")
         elif .sglang_source.head != $e
           then error("\(input_filename): engine \(.sglang_source.head), declared \($e)")
         elif .repo.dirty_files != [] or .sglang_source.dirty_files != []
           then error("\(input_filename): dirty or unrecorded tree")
         else [$p, $l, $s, .repo.head, .sglang_source.head, .sglang_source.branch,
           (.env_overrides | to_entries | map("\(.key)=\(.value)") | join(" ")),
           (.command[3:] | join(" "))] | @csv end' "$launch"
    done
    got="$(printf '%s\n' "${found[@]}" | sort | tr '\n' ' ')"
    want="$(tr ' ' '\n' <<< "$expected" | sort | tr '\n' ' ')"
    if [ "$got" != "$want" ]; then
      echo "$dir: launch records for [$got], expected one each for [$want]" >&2
      exit 1
    fi
  done
} > "$tmp/launches.csv"

# Revisions of the prefill probe: client.json holds them for runs since prefill_probe.py recorded
# them; for the committed run, which predates that, the hold's log (prefill_hold.log, if kept)
# gives the repository head. Each server's log gives the SGLang tree it imported (with $HOME as
# ~). A server with no repository head from either source, or no import path, stops the script.
repo_head="$(grep -m1 -oE "^repo [0-9a-f]{40} out .*/${prefill##*/} " "$data/prefill_hold.log" \
  2>/dev/null | cut -d' ' -f2 || true)"
imported() {
  local path
  path="$(grep -m1 -oE '/[^ :]*/python/sglang/launch_server\.py' "$1" || true)"
  [ -n "$path" ] || { echo "$1: no SGLang import path" >&2; return 1; }
  echo "~${path#"$HOME"}"
}
stock_from="$(imported "$prefill/stock/server.log")"
fi_from="$(imported "$prefill/fi-prefill/server.log")"
# Where client.json records the two trees (runs since prefill_probe.py recorded them), both must
# be clean and SGLang at the pin, as for the serving launches; the committed run predates that.
jq -n --slurpfile s "$prefill/stock/client.json" --slurpfile f "$prefill/fi-prefill/client.json" \
  --arg repo "$repo_head" --arg sp "$stock_from" --arg fp "$fi_from" --arg pin "$pin" '
def check(c): if (c.repo != null and c.repo.dirty_files != [])
    or (c.sglang_source != null and c.sglang_source.dirty_files != [])
  then error("\(c.arm): dirty or unrecorded tree in client.json")
  elif c.sglang_source != null and c.sglang_source.head != $pin
  then error("\(c.arm): SGLang \(c.sglang_source.head), not the pin")
  else c end;
def summ(c; p): {arm: c.arm, server_under_nsys_launch: (c.command[0]=="nsys"), linear_attn_prefill_backend_flag: (c.args["linear-attn-prefill-backend"] // null),
  repo_head: (c.repo.head // (if $repo == "" then error("\(c.arm): no repository head") else $repo end)), sglang_head: (c.sglang_source.head // null), sglang_imported_from: p,
  untraced: {single_median_ms: c.untraced.sequential_median_ms, concurrent8_wall_median_ms: c.untraced.concurrent8_wall_median_ms,
    single_ms_by_isl: [c.untraced.sequential[] | {isl, ms}]},
  traced: (if c.traced then {single_median_ms: c.traced.sequential_median_ms, concurrent8_wall_median_ms: c.traced.concurrent8_wall_median_ms} else null end)};
{run: "prefill-20261002T175303Z", requests: "30 confirm-split prompts one at a time (max_tokens 1, non-streaming, thinking on), then 5 rounds of 8 concurrent",
 resolved_gdn_backends: "decode=triton, prefill=flashinfer, verify=triton in both servers (server logs)",
 revisions: "repo_head from client.json, else the hold log; sglang_head from client.json (null where the probe did not record it); sglang_imported_from from the server log",
 stock: summ(check($s[0]); $sp), explicit_flashinfer_prefill: summ(check($f[0]); $fp)}' > "$tmp/prefill_requests.json"

jq '{gap_ms_between_windows: 20, windows: .windows, first_30_median: .first_30_median, rows: .rows}' \
  "$prefill/stock/trace_summary.json" > "$tmp/prefill_trace.json"
# Each point holds tokens, requests and eager and queued times for four kernels: 10 numbers.
# A failed measurement is a string there, so a point with fewer numbers stops the script.
jq -e '(.points | length) > 0 and all(.points[]; ([.[] | numbers] | length) == 10)' \
  "$prefill/gdn_prefill_bench.json" > /dev/null \
  || { echo "$prefill/gdn_prefill_bench.json: a failed or missing measurement" >&2; exit 1; }
cp "$prefill/gdn_prefill_bench.json" "$tmp/gdn_prefill_bench.json"

{
  echo "run,prefill_batches,max_total_num_tokens,max_mamba_cache_size,max_running_requests,sglang_sha,repo_sha"
  for name in "${logprob_runs[@]}"; do
    run="$data/logprob/runs/$name"
    log="$run/server.log"
    # Each value must be present: a failed grep or a null field stops the script.
    batches="$(grep -c 'Prefill batch' "$log")"
    kv="$(grep -o 'max_total_num_tokens=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
    mamba="$(grep -o 'max_mamba_cache_size: [0-9]*' "$log" | tail -1 | cut -d' ' -f2)"
    running="$(grep -o 'max_running_requests=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
    sglang="$(jq -er .sglang_sha "$run/c128.meta.json")"
    repo="$(jq -er .repo_sha "$run/c128.meta.json")"
    echo "$name,$batches,$kv,$mamba,$running,$sglang,$repo"
  done
} > "$tmp/logprob_runs.csv"

mkdir -p "$out"
for file in launches.csv prefill_requests.json prefill_trace.json gdn_prefill_bench.json \
  logprob_runs.csv; do
  mv "$tmp/$file" "$out/$file"
done
