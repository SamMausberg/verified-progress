# Evidence: witnesses that separate the exactness contracts (P7)

P7 is the exactness-contract proposal listed in `TASKS.md`. CPU only. Repository `.venv`:
Python 3.13.15, NumPy 2.3.5. Produced at the commit recorded in the JSON (`repo_commit`):

```sh
. .venv/bin/activate
python tests/test_contracts.py > evidence/contracts/contract_witnesses.log 2>&1
```

The script writes `contract_witnesses.json` (7 test methods, all passing) and the log. Exact
cases use Fractions. Floating-point cases use NumPy float32 scalar and elementwise arithmetic
(IEEE binary32, round to nearest, ties to even, no FMA contraction), with every reduction
(including the key normalization, summed in float64) written as an explicit sequential loop so
the evaluation order is the stated one; a kernel that contracts, reorders or accumulates
differently can round differently.

| Test | What it shows |
|---|---|
| `test_reassociation_flips_a_greedy_winner` | Row 1 sums `1 + 2^-24 + 2^-24`. Sequential FP32 accumulation returns `1` and ties row 0 (`1`), so the first-index argmax picks token 0; the regrouped sum returns `1 + 2^-23` and picks token 1, as real arithmetic does. Identical greedy output depends on the reduction order. |
| `test_distinct_fp32_values_share_a_bf16_cell` | `1` and `1 + 2^-10` differ in FP32 but both round to `1` in BF16, so a BF16-output head ties them and the index rule, not the values, decides. |
| `test_same_argmax_different_law` | In the base-2 mass model, logits `(1, 0)` and `(2, 0)` give laws `(2/3, 1/3)` and `(4/5, 1/5)`: the same greedy token, different distributions. |
| `test_same_law_different_seeded_tokens` | Two exact inverse-CDF samplers for `p = (1/2, 1/2)` that list the tokens in different orders have the same law, but the uniform `1/4` gives token 0 from one and token 1 from the other. |
| `test_regrouped_gdn_recurrence_changes_outputs` | The gated-delta recurrence `S_t = S_{t-1} A_t + B_t` evaluated sequentially and in the split form `S_t = S_0 P_t + W_t` (equal in exact arithmetic), 50 steps at width 32, seed 7: 1,481 of 1,600 FP32 outputs `o_t = S_t q_t` differ, 2 after rounding to BF16. |
| `test_equal_quality_different_outputs` | Two greedy decoders answer four problems; each is right on two (accuracy 1/2 for both, so their scores are equivalent within any margin), yet their answers differ on all four (two discordant pairs each way). Empirical quality equivalence implies none of the stronger contracts. |
| `test_transition_norm_and_error_growth` | `(I - beta k k^T) k = (1 - beta ||k||^2) k` and vectors orthogonal to `k` are fixed, so the transition's spectral norm is at most 1 exactly when `0 <= beta ||k||^2 <= 2` (norm `3/2` at `beta ||k||^2 = 5/2`). Under that condition a state error obeys `||E_t|| <= |alpha_t| ||E_{t-1}|| + ||eta_t||`. |

The query-invisible witness (equal readout now, different later) is in
`evidence/state_structure/`. These witnesses show that the contracts are distinct in
general; they are not measurements of Qwen3.5-4B and say nothing about how often the
differences occur in the engine.
