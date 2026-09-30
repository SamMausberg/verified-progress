# shellcheck shell=bash
# Source this before running SGLang, the benchmarks or GPU tools on this machine:
#
#   source scripts/sglang_env.sh
#
# The Lambda Stack driver (570) supports CUDA 12.8, while SGLang's wheels need
# CUDA 13. Everything here is user-level: CUDA 13 forward-compatibility libraries
# and a CUDA 13.0.3 toolkit under ~/.local. The system driver and /usr/local/cuda
# are untouched.

SGLANG_DIR="${SGLANG_DIR:-$HOME/sglang}"
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-$HOME/.local/cuda-compat-13.0}"

export CUDA_HOME="${CUDA_HOME_13:-$HOME/.local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck source=/dev/null
source "$SGLANG_DIR/.venv/bin/activate"
