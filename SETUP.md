# Machine setup

How this Lambda GH200 box is set up for the paper's GPU work (set up
2026-09-30). Everything below is user-level unless marked as apt. The NVIDIA
driver, `/usr/local/cuda` and Lambda Stack packages are untouched; do not
upgrade them.

## Quick start

```sh
source scripts/sglang_env.sh
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --attention-backend flashinfer \
  --mm-attention-backend triton_attn \
  --host 127.0.0.1 --port 30000
```

CUDA graphs (prefill and decode) and the overlap scheduler are on by default;
the startup log shows both graph captures.

## Hardware and driver

| Item | Value |
|---|---|
| GPU | NVIDIA GH200 480GB (96 GB HBM3, sm_90) |
| CPU / RAM | 64 vCPU aarch64 (Grace), 525 GB |
| OS | Ubuntu 22.04.5, Lambda Stack |
| Driver | 570.195.03 (supports CUDA 12.8 natively) |

## CUDA 13 without a driver upgrade

SGLang at the pinned commit needs CUDA 13 (torch 2.13.0+cu130). The 570 driver
only supports 12.8, so `scripts/sglang_env.sh` puts these first on the paths:

- `~/.local/cuda-compat-13.0`: NVIDIA forward-compatibility libraries
  (`cuda-compat-13-0` 580.178.04, unpacked from the .deb, not apt-installed).
  Without them torch reports that the driver is too old.
- `~/.local/cuda-13.0`: CUDA 13.0.3 toolkit (nvcc 13.0.88, compute-sanitizer,
  ncu, cuobjdump), the version SGLang's Docker image uses. It serves as
  `CUDA_HOME` for JIT kernels. The system `/usr/local/cuda` (12.8) is unused.

apt on this box pins NVIDIA-repo packages to priority -1 (Lambda Stack owns
the driver and CUDA libs), so never `apt install` CUDA or driver packages.

## SGLang

- Clone: `~/sglang`, branch `verified-progress` at
  `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` (the paper's source pin).
- Venv: `~/sglang/.venv` (Python 3.12), editable install `-e "python[test]"`
  plus `flashinfer-cubin` and `flashinfer-jit-cache` 0.6.18 (cu130).
- Rust extensions (radix tree, gRPC, renderer) build with rustup's toolchain
  in `~/.cargo` (1.92 from `rust/rust-toolchain.toml`).
- SGLang's own pre-commit hooks are installed in the clone.

Key versions: torch 2.13.0+cu130, triton 3.7.1, flashinfer 0.6.18,
sglang-kernel 0.4.7, flash-attn-4 4.0.0b19, transformers 5.12.1,
xgrammar 0.2.7, lm-eval 0.5.0.dev1, sgl-eval 0.1.2.

### aarch64 caveat: no FA3

The aarch64 `sglang-kernel` wheel ships without FA3 (`flash_ops`), and
`kernels-community/sgl-flash-attn3` publishes x86_64 builds only. The default
`fa3` backend therefore fails here, both for text attention and for Qwen3.5's
vision encoder during warmup. Use `--attention-backend flashinfer` and
`--mm-attention-backend triton_attn`. Reference results that used FA3 (the
DFlash2 card's H200 numbers) are not like-for-like with this machine.

Other startup notes:

- `max_running_requests` is capped at 133 by the Mamba/GDN state cache at
  default memory settings (`evidence/moonshot/README.md`, section 1.3, derives
  the number); `bench/arms.toml` shows the flags the serving arms use to admit
  128 requests with speculation.
- The `torchcodec` import warning is harmless (no aarch64 wheel).

## Models

Downloaded to the Hugging Face cache (`~/.cache/huggingface/hub`) at these
revisions; pass `--revision` so a moved `main` cannot change a run.

| Model | Revision | Size |
|---|---|---|
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9.3 GB |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 55.6 GB |
| `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` | 3.8 GB |
| `z-lab/Qwen3.5-4B-DFlash` | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` | 1.2 GB |

The 27B pair is the paper's separate transfer lane. No public 4B DFlash2
draft exists; the public DFlash drafter for Qwen3.5-4B is the 4B lane's block
drafter.

## Tools

| Tool | Version | Location |
|---|---|---|
| AIPerf | 0.13.0 | `~/.local/bin/aiperf` (isolated uv tool) |
| Nsight Systems | 2026.3.2 | `/usr/local/bin/nsys` (preinstalled) |
| Nsight Compute | 2025.3.1 (CUDA 13.0.3) | `~/.local/cuda-13.0/bin/ncu` |
| Hugging Face CLI | 2.0.0 | `~/.local/bin/hf` |
| Lean | 4.19.0 | `~/.elan` (from `formal/lean-toolchain`) |
| TeX Live | 2022 + latexmk 4.76 | apt, Ubuntu archive only |
| pre-commit | 4.6.2 | `~/.local/bin/pre-commit` and repo `.venv` |

## This repository

- `.venv` (Python 3.13): `numpy==2.3.5` from `requirements-cpu.txt` plus the
  pinned dev tools in `requirements-dev.txt`.
- `python scripts/verify_artifact.py` passes all six CPU jobs, but it
  rewrites `evidence/`. Run it on a copy unless you mean to regenerate the
  evidence.
- The paper builds with `latexmk` (pdfLaTeX and BibTeX). The TeX Gyre fonts it needs
  come from the `tex-gyre` apt package.
- Every Lean file in `formal/`, the three at its top level and the three in
  `formal/drafting/`, elaborates without errors under Lean 4.19.0
  (`bash scripts/check_lean.sh`; `formal/STATUS.md`).

## Verified so far

- A Qwen3.5-4B server starts with CUDA graphs and overlap scheduling and
  answers chat requests correctly.
- AIPerf streams against it (16/16 requests, 0 errors).
- Native MTP speculation and the DFlash-4B drafter launch and serve with CUDA
  graphs; `evidence/bench/README.md` and `evidence/drafter/README.md` record
  each launch and the configurations that failed or were rejected.
- No committed run has served the 27B DFlash2 pair on this machine.
