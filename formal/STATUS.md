# Formal evidence status

Checked on 2026-09-30 with Lean 4.19.0 (elan toolchain from `lean-toolchain`,
aarch64): `bash scripts/check_lean.sh` elaborates every file below with no
errors. Neither file contains `sorry`, `admit` or a new `axiom`. `#print axioms`
on the main theorems reports only Lean's standard axioms: `propext`, `Quot.sound`
and, for `envelope_compose`, `widen` and `shift_encloses` (through `omega`),
`Classical.choice`. No axiom is added. A deliberately false
variant of `sequential_sum_bound` is rejected, so the check is not vacuous.

## `DecisionGuards.lean` (from the supplied bundle, unchanged)

Nine declarations: scaled-integer acceptance and rejection guards, monotone
refinement, chronological function composition, deterministic replay, and
publication guards. It now elaborates. Its header comment still says "NOT
COMPILED"; that comment is stale but the file is left byte-identical to the
bundle (`SHA256SUMS`).

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

## What is not formalized

IEEE-754 rounding itself and the bridge from floating-point values to these
integers; the tree (pairwise, blocked, fused) version of the accumulation
bound; Cauchy-Schwarz; real exponentials and the transport mass bounds; the
Gumbel-max law and every probability statement; the tensor-core adder model;
compiler lowering, CUDA race freedom and engine state restoration. The exact
CPU reference (`src/precision_reference.py`) tests several of these
numerically; tests are not proofs.
