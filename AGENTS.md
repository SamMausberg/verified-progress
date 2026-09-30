# AGENTS.md

Follow the writing guidance in `writing.md`.

## Environment

`SETUP.md` describes the machine: CUDA 13 via user-level compat libraries,
the SGLang clone in `~/sglang`, pinned models and the aarch64 attention
backend caveat. Run `source scripts/sglang_env.sh` before any GPU or SGLang
command. Never install or upgrade the NVIDIA driver or CUDA through apt on
this Lambda box.

## Formatting and linting

When you finish a change, run the pre-commit hooks on the files you touched and
fix what they report:

```sh
. .venv/bin/activate
pre-commit run --files <changed files>
```

The hooks run ruff (lint and format), mypy, codespell and shellcheck, and they
also run automatically on `git commit`. Do not mass-reformat files you did not
change: the bundle's evidence is tied to the current source (`sources/bundle-v3.sha256` records the
imported bundle, not the current tree),
so a cleanup of existing code needs its own commit and a rerun of
`scripts/verify_artifact.py`.

## Commits and pull requests

Do not credit yourself (Claude or any other AI agent) as a co-author. Leave out
`Co-Authored-By` trailers and "Generated with ..." lines in commit messages and
PR descriptions.
