#!/usr/bin/env bash
# Exactness check of DFlash with buffered GDN verify against stock DFlash verify:
# panel-v2, block 16, greedy, top-5 logprobs, concurrency 1 and 8, same engine
# build (patches drafter/0001-0003). Arms: stock (off), the compact circular replay
# (--enable-linear-replayssm-spec, patch 0002) and fold-every-commit
# (+ SGLANG_GDN_REPLAYSSM_FOLD=1, patch 0003). At concurrency 8 a second stock
# run (off2, last) measures how often stock differs from itself when batch
# composition changes between runs.
# It starts with the kernel-level parity check (gdn_verify_parity.py) and ends
# with the MTP s3 arms (run_replay_check_mtp.sh, output OUT-mtp, by default
# replay-check-mtp).
# Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_replay_check.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/replay-check}"
# Kernel-level parity first (seconds): stock vs fold vs circular GDN verify on
# random inputs at the Qwen3.5-4B shape, with the launch tiles each path selects.
# A failed check leaves no JSON behind (an older one would pass for this run's).
rm -f "$out/gdn_verify_parity.json"
python "$here/gdn_verify_parity.py" --out "$out/gdn_verify_parity.json" ||
  echo "[replay-check] kernel parity failed (no gdn_verify_parity.json); continuing with the served check"
for conc in 1 8; do
  arms=(off circular fold)
  if [ "$conc" = 8 ]; then arms+=(off2); fi
  for arm in "${arms[@]}"; do
    # Both buffered arms and the stock arm use the Triton GDN decode/verify
    # backend (bench's shared default; buffered verify refuses
    # --linear-attn-decode-backend flashinfer).
    extra="--linear-attn-decode-backend triton"
    env=()
    case "$arm" in circular | fold) extra="$extra --enable-linear-replayssm-spec" ;; esac
    if [ "$arm" = fold ]; then env=(--env SGLANG_GDN_REPLAYSSM_FOLD=1); fi
    python "$here/serve_run.py" --arm dflash --block 16 --port 30086 --out "$out/c$conc-$arm" \
      --mem 0.25 --max-running 8 --extra="$extra" "${env[@]}" \
      --client "python $here/accept_probe.py --port {port} --workload $here/panel-v2.jsonl \
        --per-domain 32 --max-new-tokens 2048 --concurrency $conc --logprobs \
        --label c$conc-$arm --out {out}"
  done
  for arm in "${arms[@]:1}"; do
    python "$here/compare_outputs.py" --ref "$out/c$conc-off/requests.jsonl" \
      --test "$out/c$conc-$arm/requests.jsonl" --out "$out/c$conc-$arm-equality.json"
  done
done
# The MTP s3 arms (stock, fold, stock repeat at c=8; radix off) in the same hold,
# written next to OUT as OUT-mtp (default replay-check-mtp).
"$here/run_replay_check_mtp.sh" "$out-mtp"
