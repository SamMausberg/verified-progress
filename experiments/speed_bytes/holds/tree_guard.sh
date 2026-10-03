# shellcheck shell=bash
# Sourced by the holds (checkout and runtime records). dirty_tree <checkout> <import root> prints
# what makes the checkout differ from its commit as Python sees it: tracked edits, untracked
# files, and ignored .py files under the directory Python imports from (an untracked or ignored
# sitecustomize.py there would load into every process). Ignored build products (compiled
# extensions, egg-info, caches, the python/sglang/_version.py that setuptools-scm writes at
# install) and a virtualenv in .venv/ are allowed. Empty output means clean.
dirty_tree() {
  git -C "$1" status --porcelain --untracked-files=all || echo "git status failed in $1"
  git -C "$1" status --porcelain --ignored --untracked-files=all -- "$2" \
    | grep -E '^!! .*\.py$' | grep -v -E '^!! (\.venv/|python/sglang/_version\.py$)' || true
}

# log_runtime prints the runtime of the hold's Python (the virtualenv scripts/sglang_env.sh
# activated, which bench launches the servers with) as "runtime <JSON>" (runtime_record.py: the
# interpreter, torch and its CUDA, the serving packages' versions and the hash of sgl-kernel's
# libraries) for summarize.py to check.
log_runtime() {
  python "$(dirname "${BASH_SOURCE[0]}")/../runtime_record.py"
}

# kill_own_servers <port> stops the SGLang servers on that port that this hold started: the
# launch_server processes whose environment carries SB_HOLD=<this hold's output directory>, which
# every hold that starts servers exports first (bench and run_profiles.py pass their environment
# on). A server another hold runs on the port is left alone.
kill_own_servers() {
  local sig p
  for sig in TERM KILL; do
    for p in $(pgrep -f -- "sglang.launch_server.* --port $1( |\$)"); do
      if tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | grep -qxF "SB_HOLD=$SB_HOLD"; then
        kill "-$sig" "$p" 2>/dev/null || true
      fi
    done
    if [ "$sig" = TERM ]; then sleep 5; fi
  done
}
