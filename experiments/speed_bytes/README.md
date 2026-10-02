# speed_bytes: reading fewer weight bytes per token with FP8

Code behind `evidence/speed_bytes/`. The question: on this GH200 (aarch64), does FP8 for the
dense linear layers of Qwen3.5-4B make decoding faster, and what does it cost in quality? The
engine side is `engine/sglang/patches/speed-bytes/` (switches `SGLANG_FP8_DENSE*` and
`SGLANG_FP8_DRAFT_HEAD`, documented in `engine/sglang/README.md`).

| File | What it does |
|---|---|
| `fp8_gemm_probe.py` | GEMM microbenchmark at the model's shapes, M = 1-256: BF16 `F.linear` against FP8 through `torch._scaled_mm` (per-tensor scales, cuBLASLt; per-row scales), SGLang's Triton W8A8 kernel, Marlin W8A16 and `torch._int_mm`, plus the per-token activation quantization kernel. CUDA-graph replay cycling enough weight copies to keep the weights out of L2 |
| `fp8_dense_unit.py` | GPU check of the engine switch outside the server: error against an FP32 product, a row computed alone against the same row inside a batch (per-row and per-tensor activation scales), CUDA-graph replay against eager |
| `step_budget.py` | Per-step kernel budget of a plain-decode Nsight Systems window by kernel class (time, launch gaps, PDL overlap) |
| `summarize.py` | Builds the evidence files from the raw outputs under `~/vp-data/speed-bytes/` |
| `holds/` | The GPU holds: `fp8_probe.sh` (microbenchmark), `kill1.sh`, `kill2b.sh` and `kill3.sh` (exclusive served kill tests on the bench harness), `probe1.sh` (shared logit probe) |

The holds ran on 2026-10-02 from copies of these scripts in a scratch directory, before this
commit. The committed hold scripts differ from those copies only in how they locate the
repository and the helper scripts; `fp8_gemm_probe.py` differs only by formatting and a lint
directive; `step_budget.py` is a cleaned version of the analysis that was run, and the committed
evidence was regenerated with it. `kill3.sh` also ran a first cuBLASLt outer-vector-scale
microbenchmark after its sweeps; that attempt was invalid (see `evidence/speed_bytes/README.md`)
and is left out of the committed script.
