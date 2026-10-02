#!/usr/bin/env bash
# Exactness class of admission batching (speed_highc lever 1): greedy outputs with top-5
# logprobs, untimed, for tuned MTP (s3, buffered verify, radix off) at client concurrency
# 64 and 128 at capacity 128, without the delay (twice: the launch-to-launch control) and with
# SGLang's queue-based prefill delayer (as run_queue_delay_probe.sh). Pools pinned identically in
# all three (running 128, KV 120,000
# tokens, 128 GDN slots; run_matrix restarts a server until it resolves exactly these).
# 960 fresh prompts, 256 new tokens, natural stopping, so completions desynchronize and
# the delay fires. Classified with experiments/state_safety/compare.py and
# bench.divergence (event classes tie / one_ulp / near / large).
# Correctness only, untimed, about 12 minutes:
#   GPU_STARTUP_MIN_FREE_GB=88 scripts/gpu_lock.sh -x experiments/admission/run_admission_logprob.sh
# Exclusive only for GPU memory (128 FP32 GDN slots at --mem-fraction-static 0.25 need
# about 88 GB free at start-up, which a shared lane cannot promise); listed in
# ~/vp-coord/untimed_holds.txt.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
out="${1:-$HOME/vp-data/speed_highc/logprob}"
runs="$out/runs"
prompts="$HOME/vp-data/state/prompts/prompts_fresh.jsonl"
mkdir -p "$runs"
pin="--disable-radix-cache --max-running-requests 128 --max-total-tokens 120000 --max-mamba-cache-size 128"
status=0
for tag in adm_n0 adm_n0b adm_pd; do
  flags=$pin
  [ "$tag" = adm_pd ] && flags="$pin --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.125 --prefill-max-requests 16"
  python experiments/state_safety/run_matrix.py --passes c64,c128 --port 30236 --out-dir "$runs" \
    --prompts "$prompts" --configs mtp_s3_replayssm --tag "$tag" --allow-mixed-pins \
    "--extra-flags=$flags" || { echo "run $tag failed"; status=1; }
done
cat > "$out/pairs.json" <<'PAIRS'
[
  ["mtp n0 vs n0 repeat c128", "mtp_s3_replayssm__adm_n0/c128", "mtp_s3_replayssm__adm_n0b/c128"],
  ["mtp pd vs n0 c128", "mtp_s3_replayssm__adm_n0/c128", "mtp_s3_replayssm__adm_pd/c128"],
  ["mtp pd vs n0 repeat c128", "mtp_s3_replayssm__adm_n0b/c128", "mtp_s3_replayssm__adm_pd/c128"],
  ["mtp n0 vs n0 repeat c64", "mtp_s3_replayssm__adm_n0/c64", "mtp_s3_replayssm__adm_n0b/c64"],
  ["mtp pd vs n0 c64", "mtp_s3_replayssm__adm_n0/c64", "mtp_s3_replayssm__adm_pd/c64"]
]
PAIRS
cat > "$out/arms.json" <<'ARMS'
[
  ["mtp-tuned + prefill delayer, c128", "mtp pd vs n0 c128", "mtp pd vs n0 repeat c128"],
  ["mtp-tuned + prefill delayer, c64", "mtp pd vs n0 c64", "mtp n0 vs n0 repeat c64"]
]
ARMS
rm -f "$out/summary.json" "$out/report.json" "$out/classes.json" "$out/divergences.csv"
python experiments/state_safety/compare.py --runs "$runs" --pairs "$out/pairs.json" \
  --out-json "$out/summary.json" --out-csv "$out/divergences.csv" \
  --out-table "$out/table.csv" > "$out/compare.log" 2>&1 || status=1
python -m bench.divergence "$out/summary.json" --floor "mtp n0 vs n0 repeat c128" \
  --out "$out/report.json" --arms "$out/arms.json" --classes-out "$out/classes.json" \
  --expect-prompts "$(grep -c . "$prompts")" || status=1
exit "$status"
