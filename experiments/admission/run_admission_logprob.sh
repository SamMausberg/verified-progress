#!/usr/bin/env bash
# Exactness class of admission batching (speed_highc lever 1): greedy outputs with top-5
# logprobs, untimed, for tuned MTP (s3, buffered verify, radix off) at client concurrency
# 64 and 128 at capacity 128, without the delay (twice: the launch-to-launch control) and with
# SGLang's queue-based prefill delayer (as run_queue_delay_probe.sh). Pools pinned identically in
# all three (running 128, KV 120,000
# tokens, 128 GDN slots; run_matrix restarts a server until it resolves exactly these).
# 960 fresh prompts, 256 new tokens, natural stopping, so completions desynchronize and
# the delay fires. classify_logprob.sh then classifies every delayed-vs-undelayed comparison
# with experiments/state_safety/compare.py and bench.divergence (tie / one_ulp / near / large).
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
"$here/classify_logprob.sh" "$out" || status=1
exit "$status"
