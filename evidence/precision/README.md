# Evidence: exact floating-point reference, Lean check and head constants

CPU only; no GPU, no model inference. Machine: GH200 host, 64-core Grace (aarch64).
Repository `.venv`: Python 3.13.15, NumPy 2.3.5. SGLang venv (for the weight-only calculation):
Python 3.12, torch 2.13.0+cu130 on CPU, safetensors 0.9.0-rc.1.

## `precision_tests.json`, `precision_tests.log`

Exact tests of `src/precision_reference.py`. Every BF16, FP32 and FP64 rounding is emulated on
Fractions, so each bound and decision is checked against real arithmetic. Produced at commit
`960aa802` (recorded in the JSON as `repo_commit`), seed 20260930:

```sh
. .venv/bin/activate
python tests/test_precision.py 2>&1 | tee evidence/precision/precision_tests.log
```

Result: 21 test methods passed in 37 s; 40,015 generated checks (per-kind counts in the JSON).
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
- Stock-head emulation. The adopted contract is R-stock: the token the stock head kernel returns
  at this batch shape, certified only through the gap condition and otherwise computed by the
  stock head itself. R-real and R-bf16 are batch-invariant references, and neither is stock
  equality. In all 900 comparisons on adversarial near-tie heads (D <= 8), the emulated stock
  rounding coincided with exact rounding, so R-bf16's 900/900 agreement there exercises only the
  tie rule (R-real agreed in 582 of 900). A constructed witness (`test_bf16_contract_is_not_stock`)
  has R-bf16 choose a different token from the emulated stock head, and the gap condition refuses
  to certify it. Another witness shows reduced-precision split-K reductions reversing a token that
  the FP32-model gap condition certifies.
- A stream of noise consumed in evaluation order changed the race winner in 247 of 300 cases; a
  counter-based field never can.
- Subnormal operands: 160 heads with subnormal BF16 weights and hidden states, with and without
  flush to zero; every FP32 rescoring interval enclosed the exact logit, and the 71 exact ties
  between identical rows resolved to the smallest index. Underflowing products stay exact until
  the modelled accumulator rounds them (for example `2^-149 + 2^-150` rounds to even, `2^-148`).

Scope: synthetic heads with small V and D. The tensor-core adder is a model, not hardware. Tests
are not proofs.

## `lean_check.log`

```sh
PATH=$HOME/.elan/bin:$PATH bash scripts/check_lean.sh > evidence/precision/lean_check.log 2>&1
```

Lean 4.19.0 elaborates `formal/CertifiedArgmax.lean`, `formal/DecisionGuards.lean` and the three
drafting-proposal files in `formal/drafting/` with no errors; the log also lists the axioms of
every theorem in `formal/drafting/` (only `propext`, `Quot.sound` and `Classical.choice`). See
`formal/STATUS.md` for what is and is not formalized. Regenerated at commit `4533e5e`.

## `head_constants.json`

Weight-only constants of the Qwen3.5-4B tied head
(`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, tensor
`model.language_model.embed_tokens.weight`, 248,320 x 2,560 BF16), computed in float64 on CPU at
the commit recorded in the JSON (`repo_commit`; first computed at `9df1e0c7`, regenerated with
identical values after the quantile code stopped subsampling):

```sh
~/sglang/.venv/bin/python experiments/precision_head_constants/head_constants.py \
    --output evidence/precision/head_constants.json
```

Per-row round-to-nearest quantization errors (int8 and int4 per row, int8 with 128-column groups),
64-row tile radii and diameters in token-id order and for random tiles, and the relative
hidden-state drift below which transport's bound is narrower than self-evidence's. That drift
threshold is defined as follows: rho = `||h_t - h_d||_2 / ||h_t||_2` at the head input (after the
final norm); for row i, transport's l2 half-width `r_c(i) ||Delta||_2` is narrower than the int8
per-row round-to-nearest half-width `||e_i||_2 ||h_t||_2` if and only if rho < `||e_i||_2 / r_c(i)`,
with `r_c(i)` the l2 radius about the mean of row i's 64-row tile of contiguous token ids. Its
median over rows is 0.85% (p10 0.67%, p90 1.15%); in the paper's l_inf/l_1 family the median
is 0.30%. It compares envelope width only, not certification rate or runtime. Quantiles are
over all 248,320 rows. This is a derived calculation from the weights; no hidden states
are involved, so it does not by itself decide whether either certificate is useful.
