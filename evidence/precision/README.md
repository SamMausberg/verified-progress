# Evidence: certified low-precision head (theory workstream)

CPU only; no GPU, no model inference. Machine: GH200 host, 64-core Grace (aarch64).
Repository `.venv`: Python 3.13.5, NumPy 2.3.5. SGLang venv (for the weight-only calculation):
Python 3.12, torch 2.13.0+cu130 on CPU, safetensors 0.9.0-rc.1.

## `precision_tests.json`, `precision_tests.log`

Exact tests of `src/precision_reference.py`. Every BF16, FP32 and FP64 rounding is emulated on
Fractions, so each bound and decision is checked against real arithmetic. Produced at commit
`9df1e0c7` (recorded in the JSON as `repo_commit`), seed 20260930:

```sh
. .venv/bin/activate
python tests/test_precision.py 2>&1 | tee evidence/precision/precision_tests.log
```

Result: 19 test methods passed in 35 s; 39,760 generated checks (per-kind counts in the JSON).
Highlights:

- 5,600 accumulation-bound checks across eight summation models (sequential, pairwise, blocked,
  split-K, round toward zero, flush-to-zero, and a fused multi-term adder model of tensor cores);
  the largest observed error was 17-83% of the bound, depending on the model.
- 1,767 approximate-pass rows and 2,767 integer-pass (W8A8) rows enclosed; witnesses that the
  accumulation term (575 cases), the outward rounding and the norm inflation (58 cases) are each
  necessary.
- 420 certified argmax cases against exact argmax, 72 of them exact ties; the certificate ended at
  the approximate pass in 183 cases, after FP32 re-scoring in 191, FP64 in 24 and exact
  arithmetic in 22. Approximate-only certification refused 186 of 300 near-tie heads and was
  right in the other 114.
- 360 races under a fixed noise field (62 with a forced tie at the top), 200 top-k sets, 300
  acceptance decisions (148 with the uniform placed next to the threshold), 300 residual races,
  300 SGLang-style sibling verifications, 200 unresolved-probability measures and 200 tail-bound
  checks.
- Stock-head emulation: on adversarial near-tie heads, the BF16-tie contract (R-bf16) matched the
  emulated stock head in 900 of 900 comparisons, the real-argmax contract (R-real) in 582 of 900.
  A constructed witness shows reduced-precision split-K reductions reversing a token that the
  FP32-model gap condition certifies.
- A stream of noise consumed in evaluation order changed the race winner in 247 of 300 cases; a
  counter-based field never can.

Scope: synthetic heads with small V and D. The tensor-core adder is a model, not hardware. Tests
are not proofs.

## `lean_check.log`

```sh
PATH=$HOME/.elan/bin:$PATH bash scripts/check_lean.sh > evidence/precision/lean_check.log 2>&1
```

Lean 4.19.0 elaborates `formal/CertifiedArgmax.lean` and `formal/DecisionGuards.lean` with no
errors. See `formal/STATUS.md` for what is and is not formalized.

## `head_constants.json`

Weight-only constants of the Qwen3.5-4B tied head
(`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, tensor
`model.language_model.embed_tokens.weight`, 248,320 x 2,560 BF16), computed in float64 on CPU at
commit `9df1e0c7`:

```sh
~/sglang/.venv/bin/python experiments/precision_head_constants/head_constants.py \
    --output evidence/precision/head_constants.json
```

Per-row round-to-nearest quantization errors (int8 and int4 per row, int8 with 128-column groups),
64-row tile radii and diameters in token-id order and for random tiles, and the relative
hidden-state drift below which transport's bound is narrower than self-evidence's. Quantiles are
over a one-million-row sample. This is a derived calculation from the weights; no hidden states
are involved, so it does not by itself decide whether either certificate is useful.
