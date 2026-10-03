# shellcheck shell=bash
# Sourced by the holds (checkout and runtime records). dirty_tree <checkout> <import root> prints what makes the checkout differ
# from its commit as Python sees it: tracked edits, untracked files, and ignored .py files under the
# directory Python imports from (an untracked or ignored sitecustomize.py there would load into
# every process). Ignored build products (compiled extensions, egg-info, caches, the
# python/sglang/_version.py that setuptools-scm writes at install) and a virtualenv in .venv/ are
# allowed. Empty output means clean.
dirty_tree() {
  git -C "$1" status --porcelain --untracked-files=all || echo "git status failed in $1"
  git -C "$1" status --porcelain --ignored --untracked-files=all -- "$2" \
    | grep -E '^!! .*\.py$' | grep -v -E '^!! (\.venv/|python/sglang/_version\.py$)' || true
}

# log_runtime prints the interpreter, torch and torch's CUDA that the hold's Python (the virtualenv
# scripts/sglang_env.sh activated, which bench launches the servers with) runs, as
# "runtime <python> <torch> <cuda>" for summarize.py to check.
log_runtime() {
  python -c 'import sys, torch; print("runtime", sys.executable, torch.__version__, torch.version.cuda)'
}
