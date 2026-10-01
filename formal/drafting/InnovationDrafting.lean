import Std

/-!
# Causal innovation coding and exact publication

Intended toolchain: Lean 4.19.0, matching verified-progress.
STATUS: source written here, but NOT compiler-checked: this environment has no
Lean executable and the attempted toolchain download was unavailable.

This file supplies proof terms, not placeholder declarations. It does not model
IEEE-754, GPU kernels, learned-model accuracy, wall-clock speed, real probability,
or the finite-alphabet counting theorem. Those scopes are kept separate in the
research note. In particular, `next` must denote an already specified,
prefix-consistent reference; this file does not establish that property of SGLang.

History is stored MOST RECENT TOKEN FIRST, to make its update `x :: h`.
Generated continuations and command streams remain chronological.
-/

namespace InnovationDrafting

universe u
variable {α : Type u} [DecidableEq α]

inductive Command (α : Type u) where
  | keep
  | put (value : α)
  deriving DecidableEq, Repr

/-- Evaluate one default transition or explicit override. -/
def emit (base : List α → α) (h : List α) : Command α → α
  | .keep => base h
  | .put x => x

/-- Interpret a command stream, feeding each realized token back into the base. -/
def decode (base : List α → α) (h : List α) : List (Command α) → List α
  | [] => []
  | c :: cs =>
      let x := emit base h c
      x :: decode base (x :: h) cs

/-- Canonical code uses an override exactly when the TRUE prefix needs one. -/
def encode (base : List α → α) (h : List α) : List α → List (Command α)
  | [] => []
  | x :: xs =>
      (if x = base h then Command.keep else Command.put x) ::
        encode base (x :: h) xs

/-- Number of positions at which the base, on the true prefix, is wrong. -/
def innovations (base : List α → α) (h : List α) : List α → Nat
  | [] => 0
  | x :: xs =>
      (if x = base h then 0 else 1) + innovations base (x :: h) xs

/-- Cost counts explicit token overrides, not default interpreter transitions. -/
def writes : List (Command α) → Nat
  | [] => 0
  | .keep :: cs => writes cs
  | .put _ :: cs => 1 + writes cs

/-- Lossless reconstruction, including arbitrarily long causal cascades. -/
theorem decode_encode (base : List α → α) (h : List α) (xs : List α) :
    decode base h (encode base h xs) = xs := by
  induction xs generalizing h with
  | nil => rfl
  | cons x xs ih =>
      by_cases hx : x = base h
      · simp [encode, decode, emit, hx, ih]
      · simp [encode, decode, emit, hx, ih]

/-- A canonical code uses exactly the causal innovation count. -/
theorem writes_encode (base : List α → α) (h : List α) (xs : List α) :
    writes (encode base h xs) = innovations base h xs := by
  induction xs generalizing h with
  | nil => rfl
  | cons x xs ih =>
      by_cases hx : x = base h
      · simp [encode, innovations, writes, hx, ih]
      · simp [encode, innovations, writes, hx, ih]

/-- Every interpreted program pays for all innovations in its output.
    Redundant overrides may pay MORE, never less. -/
theorem innovations_decode_le (base : List α → α) (h : List α)
    (cs : List (Command α)) :
    innovations base h (decode base h cs) ≤ writes cs := by
  induction cs generalizing h with
  | nil => simp [decode, innovations, writes]
  | cons c cs ih =>
      cases c with
      | keep =>
          simpa [decode, emit, innovations, writes] using ih (base h :: h)
      | put x =>
          have hi := ih (x :: h)
          by_cases hx : x = base h
          · have hi' : innovations base (x :: h) (decode base (x :: h) cs) ≤
                1 + writes cs := by omega
            simpa [decode, emit, innovations, writes, hx] using hi'
          · simpa [decode, emit, innovations, writes, hx] using
              Nat.add_le_add_left hi 1

/-- Global minimality among ALL command programs yielding this continuation. -/
theorem minimum_override_cost (base : List α → α) (h : List α)
    (xs : List α) (cs : List (Command α))
    (hdecode : decode base h cs = xs) :
    innovations base h xs ≤ writes cs := by
  have hc := innovations_decode_le base h cs
  rw [hdecode] at hc
  exact hc

/-- Exact necessary-and-sufficient budget characterization. -/
theorem realizable_iff (base : List α → α) (h : List α)
    (xs : List α) (budget : Nat) :
    (∃ cs : List (Command α),
      decode base h cs = xs ∧ writes cs ≤ budget) ↔
      innovations base h xs ≤ budget := by
  constructor
  · rintro ⟨cs, hd, hw⟩
    exact Nat.le_trans (minimum_override_cost base h xs cs hd) hw
  · intro hi
    refine ⟨encode base h xs, decode_encode base h xs, ?_⟩
    rw [writes_encode]
    exact hi

/-- The sharp oracle applies to every desired prefix, not only full blocks. -/
theorem prefix_realizable_iff (base : List α → α) (h : List α)
    (xs : List α) (n budget : Nat) :
    (∃ cs : List (Command α),
      decode base h cs = xs.take n ∧ writes cs ≤ budget) ↔
      innovations base h (xs.take n) ≤ budget := by
  exact realizable_iff base h (xs.take n) budget

/-- Equal command prefixes have equal decoded token prefixes. -/
theorem decode_append_prefix (base : List α → α) (h : List α)
    (cs ds : List (Command α)) :
    (decode base h (cs ++ ds)).take cs.length = decode base h cs := by
  induction cs generalizing h with
  | nil => simp [decode]
  | cons c cs ih =>
      cases c <;> simp [decode, emit, ih]

/-- Command-prefix agreement suffices; later commands may be arbitrary. -/
theorem common_program_prefix (base : List α → α) (h : List α)
    (prefix left right : List (Command α)) :
    (decode base h (prefix ++ left)).take prefix.length =
      (decode base h (prefix ++ right)).take prefix.length := by
  rw [decode_append_prefix, decode_append_prefix]

/-- A causal reference can include a fixed request/position-keyed random field.
    Its numerical and RNG semantics must already be fixed in `next`. -/
def rollout (next : List α → α) (h : List α) : Nat → List α
  | 0 => []
  | n + 1 => next h :: rollout next (next h :: h) n

/-- A chronological continuation agrees with the reference at every position. -/
inductive Follows (next : List α → α) : List α → List α → Prop where
  | nil (h : List α) : Follows next h []
  | cons {h : List α} {tail : List α} :
      Follows next (next h :: h) tail →
      Follows next h (next h :: tail)

/-- Prefix acceptance, then the exact correction/bonus token. -/
def audit (next : List α → α) (h : List α) : List α → List α
  | [] => [next h]
  | x :: xs =>
      if x = next h then x :: audit next (x :: h) xs else [next h]

/-- The verifier never publishes an unverified program decision. -/
theorem audit_follows (next : List α → α) (h : List α) (draft : List α) :
    Follows next h (audit next h draft) := by
  induction draft generalizing h with
  | nil =>
      exact Follows.cons (Follows.nil (next h :: h))
  | cons x xs ih =>
      by_cases hx : x = next h
      · simpa [audit, hx] using Follows.cons (ih (next h :: h))
      · simpa [audit, hx] using
          (Follows.cons (Follows.nil (next h :: h)))

/-- Agreement at every position identifies the exact reference continuation. -/
theorem follows_eq_rollout (next : List α → α) (h : List α)
    (xs : List α) (hf : Follows next h xs) :
    xs = rollout next h xs.length := by
  induction xs generalizing h with
  | nil => rfl
  | cons x xs ih =>
      cases hf with
      | cons htail =>
          simpa [rollout] using
            congrArg (List.cons (next h)) (ih (next h :: h) htail)

/-- Pathwise exactness for arbitrary proposed token strings. -/
theorem audit_exact (next : List α → α) (h : List α) (draft : List α) :
    audit next h draft = rollout next h (audit next h draft).length := by
  exact follows_eq_rollout next h (audit next h draft) (audit_follows next h draft)

/-- Exactness is independent of the base, program, and their prediction errors. -/
theorem interpreted_program_exact (base next : List α → α)
    (h : List α) (cs : List (Command α)) :
    audit next h (decode base h cs) =
      rollout next h (audit next h (decode base h cs)).length := by
  exact audit_exact next h (decode base h cs)


/-! ## Whole-trace cost certificates
A Bellman potential turns local lower bounds into a global bound, even when
changing the proposal changes all future cycle boundaries. `edge` contains
only the transitions admitted by the chosen oracle program class.
-/

inductive Execution {State : Type u} (edge : State → State → Nat → Prop) :
    State → State → Nat → Prop where
  | done (s : State) : Execution edge s s 0
  | step {s t z : State} {c rest : Nat} :
      edge s t c → Execution edge t z rest →
      Execution edge s z (c + rest)

/-- Any potential satisfying every local Bellman inequality lower-bounds every
    complete execution. Integer costs can represent conservative time ticks. -/
theorem execution_potential_bound {State : Type u}
    (edge : State → State → Nat → Prop) (potential : State → Nat)
    (local_bound : ∀ s t c, edge s t c → potential s ≤ c + potential t)
    {s z : State} {cost : Nat} (run : Execution edge s z cost) :
    potential s ≤ cost + potential z := by
  induction run with
  | done s => simp
  | @step s t z c rest hedge hrun ih =>
      have hl := local_bound s t c hedge
      omega

/-- A terminal potential of zero gives a lower bound on total execution cost. -/
theorem terminal_cost_lower_bound {State : Type u}
    (edge : State → State → Nat → Prop) (potential : State → Nat)
    (local_bound : ∀ s t c, edge s t c → potential s ≤ c + potential t)
    {s z : State} {cost : Nat} (run : Execution edge s z cost)
    (terminal : potential z = 0) :
    potential s ≤ cost := by
  have hb := execution_potential_bound edge potential local_bound run
  simpa [terminal] using hb

#print axioms terminal_cost_lower_bound

#print axioms decode_encode
#print axioms minimum_override_cost
#print axioms prefix_realizable_iff
#print axioms common_program_prefix
#print axioms interpreted_program_exact

end InnovationDrafting
