#!/usr/bin/env bash
# Verify-pass decomposition (gpu_lock.sh -x, ~35 min), most decisive runs first:
# forced full acceptance without per-position GDN states (SGLANG_REPAIR_DROP_VERIFY_STATES=1,
# engine patch 0002, timing only); the same forced acceptance with SGLang's Triton GDN verify
# kernel instead of FlashInfer's (--linear-attn-verify-backend triton), and stock DFlash-16 with
# that verifier (the baseline gets the same verifier); Nsight Systems kernel traces of
# forced-acceptance verify at B = 256 and 64.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/nsys1}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
run() { python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 --port 30097 "$@" 2>&1 | tail -1; }
nostate() { run --mode force --block "$1" --max-new-tokens 2048 --ignore-eos --env SGLANG_REPAIR_DROP_VERIFY_STATES=1 --out "$R/force_nostate_b$1"; }
triton() { run --mode force --block "$1" --max-new-tokens 2048 --ignore-eos --arg linear-attn-verify-backend=triton --out "$R/force_tritonverify_b$1"; }
profile() { run --mode force --block "$1" --max-new-tokens 2048 --ignore-eos --limit 2 --nsys "$R/force_b$1" --out "$R/force_nsys_b$1"; }
nostate 256
triton 256
triton 64
run --mode fresh --block 16 --max-new-tokens 2048 --arg linear-attn-verify-backend=triton --out "$R/fresh_tritonverify_b16"
profile 256
nostate 64
nostate 16
triton 16
profile 64
