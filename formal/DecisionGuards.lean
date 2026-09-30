import Std

/-
Source-level formalisation for The Work a Verifier Needs.
STATUS: NOT COMPILED in the authoring environment. See STATUS.md.
No placeholder proofs or added axioms. These are small logical kernels, not a
formalisation of GPU floating-point enclosures or the sampling distribution.
Natural-number guards represent rational comparisons after clearing a common
positive denominator. The representation theorem is a separate obligation.
-/
namespace DecisionGuards

theorem acceptance_guard {uQ actual upper weight : Nat}
    (inside : actual ≤ upper) (cert : uQ * upper < weight) :
    uQ * actual < weight :=
  Nat.lt_of_le_of_lt (Nat.mul_le_mul_left uQ inside) cert

theorem rejection_guard {uQ lower actual weight : Nat}
    (inside : lower ≤ actual) (cert : weight ≤ uQ * lower) :
    weight ≤ uQ * actual :=
  Nat.le_trans cert (Nat.mul_le_mul_left uQ inside)

theorem acceptance_survives_refinement {uQ upper upper' weight : Nat}
    (refined : upper' ≤ upper) (cert : uQ * upper < weight) :
    uQ * upper' < weight :=
  acceptance_guard refined cert

theorem rejection_survives_refinement {uQ lower lower' weight : Nat}
    (refined : lower ≤ lower') (cert : weight ≤ uQ * lower) :
    weight ≤ uQ * lower' :=
  rejection_guard refined cert

/-- Chronological composition: first f, then g. -/
def chronological {α : Type} (f g : α → α) : α → α := fun x => g (f x)

theorem chronological_associative {α : Type} (f g h : α → α) :
    chronological (chronological f g) h =
    chronological f (chronological g h) := rfl

def replay {State Record : Type} (step : State → Record → State) :
    State → List Record → State
  | s, [] => s
  | s, x :: xs => replay step (step s x) xs

theorem replay_append {State Record : Type}
    (step : State → Record → State) (xs ys : List Record) (s : State) :
    replay step s (xs ++ ys) = replay step (replay step s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih =>
    simpa [replay] using ih (step s x)

theorem replay_equal_start {State Record : Type}
    (step : State → Record → State) (xs : List Record) (a b : State)
    (same : a = b) : replay step a xs = replay step b xs := by
  cases same
  rfl

def admissible (leaseEpoch currentEpoch leaseBase currentBase : Nat) : Prop :=
  leaseEpoch = currentEpoch ∧ leaseBase = currentBase

theorem stale_epoch_cannot_publish {e e' p p' : Nat} (different : e ≠ e') :
    ¬ admissible e e' p p' := by
  intro h
  exact different h.1

theorem stale_prefix_cannot_publish {e e' p p' : Nat} (different : p ≠ p') :
    ¬ admissible e e' p p' := by
  intro h
  exact different h.2

end DecisionGuards
