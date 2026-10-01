import Std

/-
Prefix-isolated sparse planning: deterministic logical core.

STATUS: NOT COMPILED IN THIS SESSION. Intended toolchain: Lean 4.19.0.
There are no placeholder proof terms or added axioms in this source. It must
be elaborated locally before being described as machine-checked.

Formalized below:
* first-mismatch uniqueness;
* the exact oracle-clipping law from isolation and hard clamping;
* preservation of a causal deterministic target's accepted prefix and bonus;
* equality of the first teacher-forced and free-running planner errors;
* finite-sample oracle dominance.

Not formalized: probability/integration, the neural architecture's compliance
with isolation, floating-point arithmetic, compiled GPU kernels, state-cache
restoration, model accuracy, or measured performance.

Indices in this file are ZERO BASED. The first mismatch k equals the number
of matching draft tokens. A cycle commits k+1 including the correction/bonus.
-/
namespace PrefixIsolatedPlanning

universe u v

/-- A finite prefix agrees and, unless at the end, fails at exactly `k`. -/
structure FirstMismatch {α : Type u}
    (n : Nat) (truth draft : Nat → α) (k : Nat) : Prop where
  bound : k ≤ n
  before : ∀ i, i < k → draft i = truth i
  fails : k < n → draft k ≠ truth k

/-- A first-mismatch certificate identifies a unique length. -/
theorem firstMismatch_unique {α : Type u}
    {n k l : Nat} {truth draft : Nat → α}
    (hk : FirstMismatch n truth draft k)
    (hl : FirstMismatch n truth draft l) : k = l := by
  by_cases hkl : k < l
  · exact False.elim ((hk.fails (by have := hl.bound; omega)) (hl.before k hkl))
  · by_cases hlk : l < k
    · exact False.elim ((hl.fails (by have := hk.bound; omega)) (hk.before l hlk))
    · omega

/-- A message at anchor a(j) has no influence on earlier output positions. -/
def Isolated {α : Type u} {κ : Type v}
    (a : κ → Nat) (E : (κ → α) → Nat → α) : Prop :=
  ∀ z z' i, (∀ j, a j ≤ i → z j = z' j) → E z i = E z' i

/-- The expander emits each supplied message verbatim at its anchor. -/
def Clamped {α : Type u} {κ : Type v}
    (a : κ → Nat) (E : (κ → α) → Nat → α) : Prop :=
  ∀ z j, E z (a j) = z j

/-- All outputs before the first wrong anchor equal their oracle-code outputs. -/
theorem expander_eq_before {α : Type u} {κ : Type v}
    {a : κ → Nat} {E : (κ → α) → Nat → α}
    (hIso : Isolated a E) {z zstar : κ → α} {q i : Nat}
    (hBefore : ∀ j, a j < q → z j = zstar j)
    (hi : i < q) : E z i = E zstar i := by
  apply hIso z zstar i
  intro j hj
  exact hBefore j (by omega)

/-- A wrong clamped anchor forces an actual token error at that position. -/
theorem wrong_anchor_fails {α : Type u} {κ : Type v}
    {a : κ → Nat} {E : (κ → α) → Nat → α}
    (hClamp : Clamped a E) {z zstar : κ → α} {truth : Nat → α}
    (hTruth : ∀ j, zstar j = truth (a j)) {q : Nat}
    (hWrong : ∃ j, a j = q ∧ z j ≠ zstar j) : E z q ≠ truth q := by
  obtain ⟨j, hPos, hNe⟩ := hWrong
  intro hEq
  apply hNe
  calc
    z j = E z (a j) := (hClamp z j).symm
    _ = E z q := by rw [hPos]
    _ = truth q := hEq
    _ = truth (a j) := by rw [hPos]
    _ = zstar j := (hTruth j).symm

/--
Exact clipping law: learned-code prefix length = min(oracle prefix, first bad anchor).
`q = n` represents no bad anchor. At genuine anchors the supplied oracle
message is the true token; no distributional or independence assumption is used.
-/
theorem oracle_clipping {α : Type u} {κ : Type v}
    {n ell q : Nat} {truth : Nat → α} {a : κ → Nat}
    {E : (κ → α) → Nat → α} {z zstar : κ → α}
    (hIso : Isolated a E) (hClamp : Clamped a E)
    (hTruth : ∀ j, zstar j = truth (a j))
    (hOracle : FirstMismatch n truth (E zstar) ell)
    (hq : q ≤ n)
    (hBefore : ∀ j, a j < q → z j = zstar j)
    (hWrong : q < n → ∃ j, a j = q ∧ z j ≠ zstar j) :
    FirstMismatch n truth (E z) (min ell q) := by
  refine ⟨?_, ?_, ?_⟩
  · have := hOracle.bound
    omega
  · intro i hi
    have hiq : i < q := by omega
    have hiel : i < ell := by omega
    exact (expander_eq_before hIso hBefore hiq).trans (hOracle.before i hiel)
  · intro hEnd
    by_cases hqell : q ≤ ell
    · have hMin : min ell q = q := Nat.min_eq_right hqell
      rw [hMin]
      apply wrong_anchor_fails hClamp hTruth
      apply hWrong
      omega
    · have hellq : ell < q := by omega
      have hMin : min ell q = ell := Nat.min_eq_left (by omega)
      rw [hMin]
      have heq : E z ell = E zstar ell := expander_eq_before hIso hBefore hellq
      intro hCorrect
      have hOracleCorrect : E zstar ell = truth ell := heq.symm.trans hCorrect
      exact hOracle.fails (by omega) hOracleCorrect

/-- Conversion of the draft-prefix law to committed tokens, including bonus. -/
theorem committed_clipping (ell q : Nat) :
    min ell q + 1 = min (ell + 1) (q + 1) := by
  omega

/-- A deterministic step-token map reads only preceding token decisions. -/
def Causal {α : Type u} (step : Nat → (Nat → α) → α) : Prop :=
  ∀ i x y, (∀ j, j < i → x j = y j) → step i x = step i y

/-- Greedy verification preserves the target prefix for an arbitrary drafter. -/
theorem verified_prefix_sound {α : Type u}
    {step : Nat → (Nat → α) → α}
    (hCausal : Causal step) {truth draft : Nat → α} {k : Nat}
    (hTarget : ∀ i, truth i = step i truth)
    (hMatch : ∀ i, i < k → draft i = step i draft) :
    ∀ i, i < k → draft i = truth i := by
  intro i
  induction i using Nat.strong_induction_on with
  | h i ih =>
    intro hik
    calc
      draft i = step i draft := hMatch i hik
      _ = step i truth := hCausal i draft truth (by
        intro j hji
        exact ih j hji (by omega))
      _ = truth i := (hTarget i).symm

/-- The target token at the first mismatch, or the all-accepted bonus, is correct. -/
theorem verified_bonus_sound {α : Type u}
    {step : Nat → (Nat → α) → α}
    (hCausal : Causal step) {truth draft : Nat → α} {k : Nat}
    (hTarget : ∀ i, truth i = step i truth)
    (hMatch : ∀ i, i < k → draft i = step i draft) :
    step k draft = truth k := by
  calc
    step k draft = step k truth :=
      hCausal k draft truth (verified_prefix_sound hCausal hTarget hMatch)
    _ = truth k := (hTarget k).symm

/--
Teacher-forced and free-running planners have the same FIRST message error.
This does not assert that their later predictions agree. A sampled planner can
use the lemma after fixing its per-message random values in `planner`.
-/
theorem teacher_forced_first_error {α : Type u}
    {planner : Nat → (Nat → α) → α}
    (hCausal : Causal planner)
    {zstar run : Nat → α} {r e : Nat}
    (hRun : ∀ j, run j = planner j run)
    (hTF : FirstMismatch r zstar (fun j => planner j zstar) e) :
    FirstMismatch r zstar run e := by
  have hBefore : ∀ j, j < e → run j = zstar j := by
    intro j
    induction j using Nat.strong_induction_on with
    | h j ih =>
      intro hje
      calc
        run j = planner j run := hRun j
        _ = planner j zstar := hCausal j run zstar (by
          intro k hkj
          exact ih k hkj (by omega))
        _ = zstar j := hTF.before j hje
  refine ⟨hTF.bound, hBefore, ?_⟩
  intro her
  have hAt : run e = planner e zstar :=
    (hRun e).trans (hCausal e run zstar hBefore)
  intro hCorrect
  exact hTF.fails her (hAt.symm.trans hCorrect)

/-- Finite empirical totals, requiring neither real analysis nor probability. -/
def total : List Nat → Nat
  | [] => 0
  | x :: xs => x + total xs

/-- Pointwise oracle dominance remains valid after pooling any fixed trace. -/
theorem pooled_oracle_dominance (xs : List (Nat × Nat)) :
    total (xs.map (fun p => min p.1 p.2 + 1)) ≤
    total (xs.map (fun p => p.1 + 1)) := by
  induction xs with
  | nil => simp [total]
  | cons p ps ih =>
    simp only [List.map_cons, total]
    have hPoint : min p.1 p.2 + 1 ≤ p.1 + 1 := by omega
    omega

#print axioms oracle_clipping
#print axioms verified_prefix_sound
#print axioms verified_bonus_sound
#print axioms teacher_forced_first_error
#print axioms pooled_oracle_dominance

end PrefixIsolatedPlanning
