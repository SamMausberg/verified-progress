# Backbone GEMMs and RMSNorm launches

Every Qwen3.5-4B decode step runs 129 backbone weight GEMMs (24 GDN layers with an input and an
output projection, 8 attention layers with qkv and o projections, 32 MLPs with gate/up and down)
and 65 RMSNorm launches. The profile ranks the GEMMs' time above the head GEMM's bandwidth at
13-16% of a plain step and the norms at about 5%
([`evidence/profiles/README.md`](../../evidence/profiles/README.md)). These scripts measure what
SGLang runs for each projection and what reads the same BF16 weights faster, and build the routing
table of the engine patch. Results: [`evidence/backbone/`](../../evidence/backbone/README.md).

The Triton kernel is part of the engine patch (`sglang/srt/layers/backbone_gemm.py`, see the
backbone section of [`engine/sglang/README.md`](../../engine/sglang/README.md)), so everything here
runs with the backbone engine worktree on the path.

| File | Role |
|---|---|
| `gemm_bench.py` | `gemm`: every projection x M = 1..128 under CUDA graphs: cuBLAS (SGLang's `F.linear`), cuBLASLt (PyTorch's pick and every heuristic algorithm), SGLang's Hopper GEMV, the Triton sweep (configurations compiled and checked in worker processes first). `norm`: RMSNorm kernels and bitwise agreement of the GEMM prologues with the stock norm and SiLU. `merge`: the packed GDN input projection against the separate pair. `chain`: MLP and GDN layer skeletons with PDL and folded prologues |
| `lt_algos.cpp` | cuBLASLt heuristic enumeration and execution of any returned algorithm (built by `gemm_bench.py` with `torch.utils.cpp_extension`) |
| `make_table.py` | The engine's routing table (JSON) from a `gemm` result |
| `summarize.py` | Markdown tables from the results |

GPU tests of the kernel: `tests/test_backbone_gemm.py` (skip without CUDA or the patch).

## Commands

Exclusive GPU lock for anything timed; raw outputs stay in `~/vp-data/backbone/`.

```sh
scripts/sglang_worktree.sh backbone
git -C ~/sglang-wt/backbone am "$PWD"/engine/sglang/patches/backbone/000[1-3]-*.patch  # the hold-1 tree
SGLANG_WORKTREE=~/sglang-wt/backbone source scripts/sglang_env.sh
R=~/vp-data/backbone/runs/h1
scripts/gpu_lock.sh -x python -m pytest -q tests/test_backbone_gemm.py
scripts/gpu_lock.sh -x python experiments/backbone/gemm_bench.py gemm \
    --out $R/gemm_full.json --evidence evidence/backbone/gemm_microbench.json
scripts/gpu_lock.sh -x python experiments/backbone/gemm_bench.py norm --out evidence/backbone/norm_microbench.json
scripts/gpu_lock.sh -x python experiments/backbone/gemm_bench.py merge --out evidence/backbone/merge_microbench.json
scripts/gpu_lock.sh -x python experiments/backbone/gemm_bench.py chain \
    --gemm-json evidence/backbone/gemm_microbench.json --out evidence/backbone/chain_microbench.json
python experiments/backbone/summarize.py --gemm evidence/backbone/gemm_microbench.json \
    --norm evidence/backbone/norm_microbench.json --merge evidence/backbone/merge_microbench.json \
    --chain evidence/backbone/chain_microbench.json > evidence/backbone/tables.md
python experiments/backbone/make_table.py --gemm-json evidence/backbone/gemm_microbench.json \
    --gemv-m1 --pdl --max-m 16 --out <table.json>
```

`gemm --evidence` writes the committed form (screened configurations reduced to counts and
timings); `--out` keeps everything. Both carry the same `summary`, which is all `chain` and
`make_table.py` read, so they rebuild from the committed file (the hold-1 chain run read the full
file; the routing table built from either is byte-identical).
