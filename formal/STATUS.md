# Formal evidence status

Checked on 2026-09-30 with Lean 4.19.0 (elan toolchain from `lean-toolchain`,
aarch64): `bash scripts/check_lean.sh` elaborates `DecisionGuards.lean` and
`CertifiedArgmax.lean` with no errors (it also checks the files in `drafting/`,
below). `StockDecision.lean` was added and checked the same way on 2026-10-03.
None of these files contains `sorry`, `admit` or a new `axiom`. `#print axioms`
on the main theorems reports only Lean's standard axioms: `propext`, `Quot.sound`
and, for `envelope_compose`, `widen` and `shift_encloses` (through `omega`),
`Classical.choice`. No axiom is added. A deliberately false
variant of `sequential_sum_bound` is rejected, so the check is not vacuous.

## `DecisionGuards.lean` (from the supplied bundle, unchanged)

Nine declarations: scaled-integer acceptance and rejection guards, monotone
refinement, chronological function composition, deterministic replay, and
publication guards. It now elaborates. Its header comment still says "NOT
COMPILED"; that comment is stale but the file is left byte-identical to the
bundle (`sources/bundle-v3.sha256`).

## `CertifiedArgmax.lean` (theory workstream)

Scores are integers after clearing a common positive denominator.

- `envelope_compose`, `widen`: a quantization bound and an accumulation bound
  add; outward widening keeps an enclosure valid.
- `separation`, `certified_argmax`, `certificate_unique`: disjoint enclosures
  decide comparisons; a certified winner is the unique maximizer, and any two
  certificates from valid enclosures name the same token.
- `maximizer_survives`, `dropped_row_loses`: the candidate filter
  `hi_i >= max_j lo_j` keeps every maximizer and drops only beaten rows.
- `shift_encloses`, `scale_encloses`: races (fixed noise added) and positive
  temperature scaling preserve enclosures.
- `accept_guard`, `reject_guard`, `guards_exclusive`: acceptance guards with
  an enclosed numerator as well as an enclosed partition; they never both fire.
- `residual_interval`: residual weights `(w - Z q)_+` (Nat truncated
  subtraction) are enclosed by the numerator and partition enclosures.
- `holder`: `|sum e_k h_k| <= E sum |h_k|` when every `|e_k| <= E`, the
  l_inf/l_1 bound behind both the Hoelder quantization envelope and the
  manuscript's transport bound.
- `sequential_sum_bound`: a running sum with per-step error at most `1/K` of
  the magnitudes it combines satisfies
  `K^n (|computed - exact| + sum|t|) <= (K+1)^n sum|t|`, i.e. the gamma_n bound
  `|error| <= ((1 + 1/K)^n - 1) sum|t|`.

## `StockDecision.lean` (the stock token under a rounding model; added 2026-10-03)

Values are integers after clearing a common positive denominator. Rounding is a
function `rn` with a spacing `u` on magnitudes, assumed to satisfy `RoundModel`:
`rn` is monotone, `2 |rn y - y| <= u |y|`, and `u` does not decrease with the
magnitude. BF16 round-to-nearest-even has these properties in its finite range;
that is argued on paper, not checked. The stock token is the first maximal index
(`IsFirstMax`, unique by `firstMax_unique`).

- `separation`: Lemma 2.2 (BF16 separation).
- `pair_gap`, `gap_condition`: Theorem 2.3 (gap condition); `a` need not be
  given as the real winner, since the hypothesis implies it.
- `screen_drop`: the screen margin of Section 2.2. Given valid intervals and
  bounds `Gmax >= G_i` and `zbar >=` every interval end's magnitude, an entry whose
  upper end is more than `2 Gmax + u (zbar + Gmax)` below another's lower end rounds
  strictly below it and is not the stock token.
- `interval_condition`, `interval_condition_rounded`: Proposition 2.4, on rounded
  interval ends and with the ends rounded from accumulator intervals by a monotone
  `rn`. The proof uses only that `k` attains the largest lower end in `C`.
- `roundModel_grid`: a concrete rounding satisfies the model, so it is not vacuous.

`#print axioms` reports only `propext`, `Quot.sound` and `Classical.choice`.
Weakening the strict hypothesis of `separation` (`>` to `>=`), or the condition on
earlier entries in `interval_condition` (`<` to `<=`), makes the file fail to
elaborate. Both weakened statements are false: a tie rounded to even, or an
earlier index tying `k`.

## `drafting/`: three drafting proposals (supplied by the author, fixed and checked here)

Lean sources for three drafting proposals (innovation-clock drafting, prefix-isolated sparse
planning, causal defect drafting), supplied by the author and first committed as supplied.
Checked on 2026-10-01 with Lean 4.19.0 after two mechanical fixes, in their own commit:
`Nat.strong_induction_on` (a Mathlib name, absent from core and Std) became core's
`Nat.strongRecOn` with its case name `ind`, and a binder named `prefix` (a keyword) became
`«prefix»`. No theorem statement changed apart from that binder's spelling. All three files
then elaborate with no errors and no `sorry`; `#print axioms` on all 33 theorems reports only
`propext`, `Quot.sound` and `Classical.choice`. The only warnings are two unused
`[DecidableEq α]` section variables in `InnovationDrafting.lean`. The experiments in
`experiments/frontier/` rely on two of these results: `teacher_forced_prefix_iff` (a
deterministic causal proposal's free-running accepted prefix equals its teacher-forced one,
so acceptance can be computed offline exactly) and `oracle_clipping`.

- `InnovationDrafting.lean`. Commands KEEP (take the interpreter's token) and PUT(x)
  (override): the canonical code reconstructs any continuation (`decode_encode`) with
  exactly as many overrides as there are innovations, positions where the interpreter is
  wrong on the true prefix (`writes_encode`); no command program producing that
  continuation uses fewer (`minimum_override_cost`); a prefix is reachable within a budget
  if and only if its innovation count fits (`realizable_iff`, `prefix_realizable_iff`).
  Greedy auditing of any proposed string publishes only the reference continuation
  (`audit_exact`, `interpreted_program_exact`). `execution_potential_bound` and
  `terminal_cost_lower_bound` are a generic Bellman-potential lower bound on execution
  cost; they construct no oracle.
- `PrefixIsolatedPlanning.lean`. If an expander is isolated (a planning token affects only
  its own and later positions) and clamped (it emits each planning token at its anchor),
  then its first mismatch with planning tokens z is the minimum of its first mismatch with
  the true anchors and the first wrong anchor (`oracle_clipping`, `committed_clipping`). A
  causal deterministic planner's first teacher-forced error equals its first free-running
  error (`teacher_forced_first_error`). Greedy verification keeps the target's prefix and
  bonus (`verified_prefix_sound`, `verified_bonus_sound`). The expected-progress formula,
  sampled proposals and the uniform-target obstruction are not formalized.
- `CausalDefect.lean`. `weighted_depth` and `causal_defect_converges`: in an abstract
  triangular sweep whose cheap part reads only earlier new-sweep values and whose defect
  part reads only old-sweep values, a position of weighted depth d is settled after d + 1
  sweeps. Caveat: the premises require exact locality. For a real target the defect F - G
  depends on every earlier position, so the depth of position i is i and the bound is one
  sweep per position, the bound of plain Jacobi iteration; it gives no one-sweep result for
  a real model. `guarded_unique_winner` (with `margin_pair`, `bias_cancels`,
  `anchored_error_identity`): an additive error whose oscillation is below the target's
  margin keeps the winner, in integer scores. `verified_prefix`, `first_correction` and
  `teacher_forced_prefix_iff`: audit soundness and the exact offline acceptance of a
  deterministic causal proposal map.

## What is not formalized

IEEE-754 rounding itself (that BF16 round-to-nearest-even satisfies
`RoundModel`) and the bridge from floating-point values to these integers; the
stock kernel's error bound `G_i` (an assumed error model); the tree (pairwise, blocked, fused) version of the accumulation
bound; Cauchy-Schwarz; real exponentials and the transport mass bounds; the
Gumbel-max law and every probability statement; the tensor-core adder model;
compiler lowering, CUDA race freedom and engine state restoration. The exact
CPU reference (`src/precision_reference.py`) tests several of these
numerically; tests are not proofs.
