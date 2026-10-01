/-
Causal defect drafting.
Proposal source supplied by the author, fixed and checked here with Lean 4.19.0
(formal/STATUS.md: the two mechanical fixes, what is proved and the caveats).

No additional axioms or admitted proofs are introduced. This file formalizes:
  * weighted causal-depth convergence for an abstract triangular sweep;
  * a locality bridge from its score equation to that theorem;
  * exact integer bias cancellation and strict margin guards;
  * soundness of audited prefixes and the first correction.
It does not formalize Qwen, IEEE rounding, GPU costs, sampling laws, learned
response accuracy, or the existence of a cheap response model. The sweep's
triangular score equation is an explicit premise; existence/uniqueness
follows by forward recursion and is not formalized here.
Caveat: `weighted_depth` assumes exact locality. For a real target the defect
F - G depends on every earlier position, so every position has depth equal to
its index and the bound is one sweep per position, the bound of plain Jacobi
iteration; the one-sweep claim rests on the margin lemma instead.
-/
import Std

namespace CausalDefect

universe u v
variable {Token : Type u} {Score : Type v}

abbrev Stream (Token : Type u) := Nat → Token

/-- Repeated corrected sweeps; a target pass produces the old-prefix defects. -/
def iterate (step : Stream Token → Stream Token) (initial : Stream Token) :
    Nat → Stream Token
  | 0 => initial
  | n + 1 => step (iterate step initial n)

/-- Exact dependence on a specified set of parent positions. -/
def Local (parents : Nat → Nat → Prop)
    (value : Nat → Stream Token → Score) : Prop :=
  ∀ i x y, (∀ j, parents i j → x j = y j) → value i x = value i y

/-- Zero-cost edges carry new-sweep values; unit-cost edges carry old-sweep
values. A coordinate of weighted depth d has settled after d+1 sweeps. -/
theorem weighted_depth
    (step : Stream Token → Stream Token)
    (initial target : Stream Token)
    (cheap residual : Nat → Nat → Prop)
    (depth : Nat → Nat)
    (cheap_causal : ∀ i j, cheap i j → j < i)
    (cheap_depth : ∀ i j, cheap i j → depth j ≤ depth i)
    (residual_depth : ∀ i j, residual i j → depth j < depth i)
    (settles : ∀ old i,
      (∀ j, cheap i j → step old j = target j) →
      (∀ j, residual i j → old j = target j) →
      step old i = target i) :
    ∀ n i, depth i < n → iterate step initial n i = target i := by
  intro n
  induction n with
  | zero =>
      intro i hi
      omega
  | succ n ih =>
      intro i
      induction i using Nat.strongRecOn with
      | ind i inner =>
          intro hi
          change step (iterate step initial n) i = target i
          apply settles
          · intro j hj
            have hd : depth j < n + 1 :=
              Nat.lt_of_le_of_lt (cheap_depth i j hj) hi
            exact inner j (cheap_causal i j hj) hd
          · intro j hj
            have hji := residual_depth i j hj
            have hd : depth j < n := by omega
            exact ih j hd

/-- Derive the settling premise from the actual score recurrence.
`combine` is vector addition in the logit application, and `r` is F-G.
The target equation states selection of F = G + R. -/
theorem settles_from_locality
    (step : Stream Token → Stream Token) (target : Stream Token)
    (cheap residual : Nat → Nat → Prop)
    (g r : Nat → Stream Token → Score)
    (combine : Score → Score → Score)
    (select : Nat → Score → Token)
    (hg : Local cheap g) (hr : Local residual r)
    (step_eq : ∀ old i,
      step old i = select i (combine (g i (step old)) (r i old)))
    (target_eq : ∀ i,
      target i = select i (combine (g i target) (r i target)))
    (old : Stream Token) (i : Nat)
    (new_parents : ∀ j, cheap i j → step old j = target j)
    (old_parents : ∀ j, residual i j → old j = target j) :
    step old i = target i := by
  have eg : g i (step old) = g i target := hg i (step old) target new_parents
  have er : r i old = r i target := hr i old target old_parents
  calc
    step old i = select i (combine (g i (step old)) (r i old)) := step_eq old i
    _ = select i (combine (g i target) (r i target)) := by rw [eg, er]
    _ = target i := (target_eq i).symm

/-- The full convergence theorem for any supplied causal score decomposition. -/
theorem causal_defect_converges
    (step : Stream Token → Stream Token)
    (initial target : Stream Token)
    (cheap residual : Nat → Nat → Prop) (depth : Nat → Nat)
    (g r : Nat → Stream Token → Score)
    (combine : Score → Score → Score) (select : Nat → Score → Token)
    (cheap_causal : ∀ i j, cheap i j → j < i)
    (cheap_depth : ∀ i j, cheap i j → depth j ≤ depth i)
    (residual_depth : ∀ i j, residual i j → depth j < depth i)
    (hg : Local cheap g) (hr : Local residual r)
    (step_eq : ∀ old i,
      step old i = select i (combine (g i (step old)) (r i old)))
    (target_eq : ∀ i,
      target i = select i (combine (g i target) (r i target))) :
    ∀ n i, depth i < n → iterate step initial n i = target i := by
  apply weighted_depth step initial target cheap residual depth
    cheap_causal cheap_depth residual_depth
  intro old i hn ho
  exact settles_from_locality step target cheap residual g r combine select
    hg hr step_eq target_eq old i hn ho

/-- Arbitrary prefix-independent, token-specific bias in the cheap model cancels.
This is stronger than invariance under a common logit shift. -/
theorem bias_cancels (fOld gOld gNew bias : Int) :
    (gNew + bias) + (fOld - (gOld + bias)) = gNew + (fOld - gOld) := by
  omega

/-- The proposal error is the change in the defect, not the absolute model error. -/
theorem anchored_error_identity (fOld gOld fNew gNew : Int) :
    (gNew + (fOld - gOld)) - fNew =
      (fOld - gOld) - (fNew - gNew) := by
  omega

/-- Oscillation upper bound hi-lo is sufficient; no absolute error bound is needed. -/
theorem margin_pair {fa fb ea eb lo hi : Int}
    (ha : lo ≤ ea) (hb : eb ≤ hi) (margin : hi - lo < fa - fb) :
    fb + eb < fa + ea := by
  omega

/-- A strict target margin larger than an error oscillation bound keeps the winner. -/
theorem guarded_unique_winner {n : Nat}
    (f err : Fin n → Int) (a : Fin n) (lo hi : Int)
    (enclosure : ∀ v, lo ≤ err v ∧ err v ≤ hi)
    (margin : ∀ b, b ≠ a → hi - lo < f a - f b) :
    ∀ b, b ≠ a → f b + err b < f a + err a := by
  intro b hb
  exact margin_pair (enclosure a).1 (enclosure b).2 (margin b hb)

/-- A target audit certifies a prefix for an arbitrary proposal, regardless of
how its tokens were computed. A fixed numerical target must satisfy causality. -/
theorem verified_prefix
    (decide : Nat → Stream Token → Token)
    (target proposal : Stream Token)
    (causal : ∀ i x y,
      (∀ j, j < i → x j = y j) → decide i x = decide i y)
    (target_eq : ∀ i, target i = decide i target)
    (m : Nat) (verified : ∀ i, i < m → proposal i = decide i proposal) :
    ∀ i, i < m → proposal i = target i := by
  intro i
  induction i using Nat.strongRecOn with
  | ind i ih =>
      intro him
      calc
        proposal i = decide i proposal := verified i him
        _ = decide i target := by
          apply causal
          intro j hj
          exact ih j hj (Nat.lt_trans hj him)
        _ = target i := (target_eq i).symm

/-- The first mismatching position's target token is safe to emit as a correction. -/
theorem first_correction
    (decide : Nat → Stream Token → Token)
    (target proposal : Stream Token)
    (causal : ∀ i x y,
      (∀ j, j < i → x j = y j) → decide i x = decide i y)
    (target_eq : ∀ i, target i = decide i target)
    (m : Nat) (verified : ∀ i, i < m → proposal i = decide i proposal) :
    decide m proposal = target m := by
  calc
    decide m proposal = decide m target := by
      apply causal
      intro j hj
      exact verified_prefix decide target proposal causal target_eq m verified j hj
    _ = target m := (target_eq m).symm

/-- Teacher forcing exactly predicts the free-running proposal's accepted
prefix. This permits testing a response model without building its GPU decoder.
`decide` here is the frozen-anchor cheap proposal map, NOT the target model. -/
theorem teacher_forced_prefix_iff
    (decide : Nat → Stream Token → Token)
    (target proposal : Stream Token)
    (causal : ∀ i x y,
      (∀ j, j < i → x j = y j) → decide i x = decide i y)
    (proposal_eq : ∀ i, proposal i = decide i proposal)
    (m : Nat) :
    (∀ i, i < m → proposal i = target i) ↔
    (∀ i, i < m → decide i target = target i) := by
  constructor
  · intro hp i hi
    calc
      decide i target = decide i proposal := by
        apply causal
        intro j hj
        exact (hp j (Nat.lt_trans hj hi)).symm
      _ = proposal i := (proposal_eq i).symm
      _ = target i := hp i hi
  · intro ht i
    induction i using Nat.strongRecOn with
    | ind i ih =>
        intro hi
        calc
          proposal i = decide i proposal := proposal_eq i
          _ = decide i target := by
            apply causal
            intro j hj
            exact ih j hj (Nat.lt_trans hj hi)
          _ = target i := ht i hi

#print axioms weighted_depth
#print axioms causal_defect_converges
#print axioms guarded_unique_winner
#print axioms verified_prefix
#print axioms first_correction
#print axioms teacher_forced_prefix_iff

-- Axioms of every theorem (formal/STATUS.md).
#print axioms CausalDefect.weighted_depth
#print axioms CausalDefect.settles_from_locality
#print axioms CausalDefect.causal_defect_converges
#print axioms CausalDefect.bias_cancels
#print axioms CausalDefect.anchored_error_identity
#print axioms CausalDefect.margin_pair
#print axioms CausalDefect.guarded_unique_winner
#print axioms CausalDefect.verified_prefix
#print axioms CausalDefect.first_correction
#print axioms CausalDefect.teacher_forced_prefix_iff

end CausalDefect
