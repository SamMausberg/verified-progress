#!/usr/bin/env bash
# Same-session control for the verify decomposition (gpu_lock.sh -x): forced full acceptance at
# B = 16 and 256 with the default FlashInfer GDN verify kernel, with per-position states dropped,
# and with SGLang's Triton GDN verify kernel, one after another in one hold.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/control1}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
run() { python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 --port 30097 --mode force --max-new-tokens 2048 --ignore-eos "$@" 2>&1 | tail -1; }
for B in 16 256; do
  run --block "$B" --out "$R/force_b$B"
  run --block "$B" --env SGLANG_REPAIR_DROP_VERIFY_STATES=1 --out "$R/force_nostate_b$B"
  run --block "$B" --arg linear-attn-verify-backend=triton --out "$R/force_tritonverify_b$B"
done
