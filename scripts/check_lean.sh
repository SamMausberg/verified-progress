#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../formal"
command -v lean >/dev/null || { echo 'Lean is not installed; formal validation NOT performed.' >&2; exit 2; }
lean --version
lean DecisionGuards.lean
