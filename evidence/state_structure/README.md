# Evidence: recurrent-state structure and cross-request sharing (P4, P5 witnesses)

P4 and P5 are the proposals listed in `TASKS.md` (minimal recurrent state; sharing
computation across unrelated requests).

CPU only. Repository `.venv`: Python 3.13.15, NumPy 2.3.5. Produced at the commit recorded
in the JSON (`repo_commit`):

```sh
. .venv/bin/activate
python tests/test_state_structure.py > evidence/state_structure/state_structure_tests.log 2>&1
```

The script writes `state_structure_tests.json` (9 test methods, all passing) and the log. Each
test is one small counterexample from the P4 (minimal recurrent state) and P5 (sharing across
requests) analyses, checked in exact rational arithmetic, or with NumPy float32 where the point
is rounding. The Gated DeltaNet step is `S' = S alpha (I - beta k k^T) + beta v k^T` with
`S` of shape (d_v, d_k).

| Test | What it shows |
|---|---|
| `test_query_invisible_is_not_unobservable` | With `alpha = beta = 1/2`, `k = (3/5, 4/5)`, `q = e1` and a state difference along `e2`, `q . ds = 0` but `q^T A ds = -3/25`: a component the current query cannot see is visible after one step. |
| `test_coordinate_key_writes_reach_any_matrix` | With unit coordinate keys and gates `alpha = beta = 1/2`, three writes reach an arbitrary 3 x 3 target from an arbitrary start, so the reachable set is not confined to a subspace without structure in the keys. |
| `test_transition_is_contraction_not_erasure` | `det(alpha (I - beta k k^T)) = alpha^d (1 - beta ||k||^2)`, here `243/1000 > 0` for a unit key and `beta = 2/3`: a step contracts the state but erases no direction. |
| `test_nonorthogonal_keys_do_not_commute` | Transitions for keys `e1` and `(3/5, 4/5)` do not commute, so there is no common diagonal basis. |
| `test_silu_raises_rank` | `G = log 2 [[1, 2], [2, 4]]` has rank one; `SiLU(G) = log 2 [[2/3, 8/5], [8/5, 64/17]]` has determinant `-64/1275 log^2 2` and rank two. |
| `test_one_gdn_step_separates_identical_states` | Four requests with the same 2 x 2 state and different keys, values and gates have states spanning all four dimensions after one step (batch rank 1 to 4). |
| `test_rmsnorm_hides_no_null_space` | `o1 = (1, 1)` and `o2 = (1, 7)` differ only in the null space of `W_o = [1, 0]`, but after RMSNorm (`x / sqrt(mean(x^2))`, no epsilon and no weight) the projections are `1` and `1/5`. |
| `test_fp32_lazy_decay_is_one_ulp_off` | In IEEE binary32 with round to nearest, ties to even (NumPy scalar arithmetic, no FMA contraction), `fl(fl(a1 x) a2)` and `fl(fl(a1 a2) x)` differ by one ulp for `a1 = a2 = 0.9`, `x = 1.3` (binary32 values), so deferring a decay is not bit-identical. A kernel that contracts to FMA or reorders could round differently. |
| `test_bf16_rounding_breaks_rank_one` | An outer product of 32 BF16 values with 32 BF16 values is exact in float32 and has rank one; rounded to BF16 it has exact rank 28 (NumPy `default_rng(0)`). |

Scope: these witnesses show that particular shortcuts are not valid in general. They are not
measurements of Qwen3.5-4B states and say nothing about how often a shortcut would fail on them.
