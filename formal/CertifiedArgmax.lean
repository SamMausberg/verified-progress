/-
Certified decisions from enclosures, in scaled-integer form.

Scores are integers after clearing a common positive denominator (rational
bounds become integer bounds). These lemmas are the logical core of the
self-evidence certificates: an envelope `lo ≤ z ≤ hi` per row, the
certified argmax and its candidate filter, races as shifted and scaled
scores, acceptance guards with an enclosed numerator, residual intervals,
the Hoelder bound behind the l_inf/l_1 envelope, and the sequential
accumulation bound (the gamma_n lemma) for any rounding whose relative
error per addition is at most 1/K.

They do not formalise IEEE-754 rounding itself, the bridge from floating
values to these integers, exponentials or sampling laws.
-/
namespace CertifiedArgmax

/-- `lo ≤ z ≤ hi`: a two-sided enclosure of a score. -/
def Encloses (lo z hi : Int) : Prop := lo ≤ z ∧ z ≤ hi

/-- A quantization bound and an accumulation bound add (triangle inequality). -/
theorem envelope_compose {z x zt q a : Int}
    (hq : x - q ≤ z ∧ z ≤ x + q) (ha : zt - a ≤ x ∧ x ≤ zt + a) :
    Encloses (zt - (q + a)) z (zt + (q + a)) := by
  unfold Encloses; omega

/-- Widening an enclosure keeps it valid (outward rounding). -/
theorem widen {lo z hi lo' hi' : Int} (h : Encloses lo z hi) (hl : lo' ≤ lo) (hh : hi ≤ hi') :
    Encloses lo' z hi' := by
  unfold Encloses at *; omega

/-- Disjoint enclosures decide a comparison. -/
theorem separation {loA zA hiA loB zB hiB : Int}
    (hA : Encloses loA zA hiA) (hB : Encloses loB zB hiB) (sep : hiB < loA) : zB < zA := by
  unfold Encloses at hA hB; omega

/-- Certified argmax: if every other upper bound is below the winner's lower
bound, the winner is the unique maximizer of the enclosed scores. -/
theorem certified_argmax {n : Nat} (lo z hi : Fin n → Int)
    (enc : ∀ i, Encloses (lo i) (z i) (hi i)) (a : Fin n)
    (cert : ∀ b, b ≠ a → hi b < lo a) : ∀ b, b ≠ a → z b < z a :=
  fun b hb => separation (enc a) (enc b) (cert b hb)

/-- Certificates issued from any two valid enclosures (for example two
levels of the refinement ladder) name the same winner. -/
theorem certificate_unique {n : Nat} (z lo hi lo' hi' : Fin n → Int)
    (enc : ∀ i, Encloses (lo i) (z i) (hi i)) (enc' : ∀ i, Encloses (lo' i) (z i) (hi' i))
    (a a' : Fin n) (cert : ∀ b, b ≠ a → hi b < lo a) (cert' : ∀ b, b ≠ a' → hi' b < lo' a') :
    a = a' := by
  by_cases h : a = a'
  · exact h
  · have h1 := certified_argmax lo z hi enc a cert a' (fun e => h e.symm)
    have h2 := certified_argmax lo' z hi' enc' a' cert' a h
    omega

/-- Candidate filter, completeness: every maximizer survives `hi ≥ τ` when
`τ` is some row's lower bound. -/
theorem maximizer_survives {n : Nat} (lo z hi : Fin n → Int)
    (enc : ∀ i, Encloses (lo i) (z i) (hi i)) (m j : Fin n) (τ : Int)
    (hmax : ∀ i, z i ≤ z m) (hτ : τ ≤ lo j) : τ ≤ hi m := by
  have hj := enc j
  have hm := enc m
  have hjm := hmax j
  unfold Encloses at hj hm
  omega

/-- Candidate filter, soundness: a dropped row is strictly beaten. -/
theorem dropped_row_loses {n : Nat} (lo z hi : Fin n → Int)
    (enc : ∀ i, Encloses (lo i) (z i) (hi i)) (i j : Fin n) (τ : Int)
    (hi_lt : hi i < τ) (hτ : τ ≤ lo j) : z i < z j := by
  have hi' := enc i
  have hj := enc j
  unfold Encloses at hi' hj
  omega

/-- Races: adding a fixed noise value to a row's score shifts its enclosure. -/
theorem shift_encloses {lo z hi : Int} (g : Int) (h : Encloses lo z hi) :
    Encloses (lo + g) (z + g) (hi + g) := by
  unfold Encloses at *; omega

/-- Temperature (as a positive scale after clearing denominators) preserves enclosures. -/
theorem scale_encloses {lo z hi : Int} (c : Int) (hc : 0 ≤ c) (h : Encloses lo z hi) :
    Encloses (c * lo) (c * z) (c * hi) :=
  ⟨Int.mul_le_mul_of_nonneg_left h.1 hc, Int.mul_le_mul_of_nonneg_left h.2 hc⟩

/-- Acceptance `U q Z < w` from an upper partition bound and a lower numerator bound. -/
theorem accept_guard {uq Z Zhi w wlo : Nat} (hZ : Z ≤ Zhi) (hw : wlo ≤ w)
    (cert : uq * Zhi < wlo) : uq * Z < w :=
  Nat.lt_of_le_of_lt (Nat.mul_le_mul_left uq hZ) (Nat.lt_of_lt_of_le cert hw)

/-- Rejection `U q Z ≥ w` from a lower partition bound and an upper numerator bound. -/
theorem reject_guard {uq Z Zlo w whi : Nat} (hZ : Zlo ≤ Z) (hw : w ≤ whi)
    (cert : whi ≤ uq * Zlo) : w ≤ uq * Z :=
  Nat.le_trans hw (Nat.le_trans cert (Nat.mul_le_mul_left uq hZ))

/-- The two guards never both fire on valid enclosures. -/
theorem guards_exclusive {uq Zlo Zhi wlo whi : Nat} (hZ : Zlo ≤ Zhi) (hw : wlo ≤ whi)
    (acc : uq * Zhi < wlo) (rej : whi ≤ uq * Zlo) : False := by
  have := Nat.mul_le_mul_left uq hZ
  omega

/-- Residual weights `(w - Z q)_+` (truncated subtraction is the positive
part) are enclosed by the numerator and partition enclosures. -/
theorem residual_interval {w wlo whi Z Zlo Zhi q : Nat} (hw : wlo ≤ w ∧ w ≤ whi)
    (hZ : Zlo ≤ Z ∧ Z ≤ Zhi) :
    wlo - Zhi * q ≤ w - Z * q ∧ w - Z * q ≤ whi - Zlo * q :=
  ⟨Nat.le_trans (Nat.sub_le_sub_right hw.1 _) (Nat.sub_le_sub_left (Nat.mul_le_mul_right q hZ.2) _),
    Nat.le_trans (Nat.sub_le_sub_right hw.2 _) (Nat.sub_le_sub_left (Nat.mul_le_mul_right q hZ.1) _)⟩

/-- Hoelder (l_inf / l_1): `|Σ e_k h_k| ≤ E Σ |h_k|` when every `|e_k| ≤ E`.
With `e = w - dequantized w` this is the Hoelder quantization envelope, and
with `e = w - centre` and `h = Δ` it is the manuscript's transport bound. -/
theorem holder (E : Nat) :
    ∀ xs : List (Int × Int), (∀ p : Int × Int, p ∈ xs → p.1.natAbs ≤ E) →
      ((xs.map (fun (p : Int × Int) => p.1 * p.2)).sum).natAbs ≤ E * (xs.map (fun (p : Int × Int) => p.2.natAbs)).sum
  | [], _ => by simp
  | (e, h) :: rest, hb => by
    have ih := holder E rest (fun p hp => hb p (List.mem_cons_of_mem _ hp))
    have he : e.natAbs ≤ E := hb (e, h) List.mem_cons_self
    simp only [List.map_cons, List.sum_cons]
    calc (e * h + (rest.map (fun (p : Int × Int) => p.1 * p.2)).sum).natAbs
        ≤ (e * h).natAbs + ((rest.map (fun (p : Int × Int) => p.1 * p.2)).sum).natAbs := Int.natAbs_add_le _ _
      _ = e.natAbs * h.natAbs + ((rest.map (fun (p : Int × Int) => p.1 * p.2)).sum).natAbs := by
          rw [Int.natAbs_mul]
      _ ≤ E * h.natAbs + E * (rest.map (fun (p : Int × Int) => p.2.natAbs)).sum :=
          Nat.add_le_add (Nat.mul_le_mul_right _ he) ih
      _ = E * (h.natAbs + (rest.map (fun (p : Int × Int) => p.2.natAbs)).sum) := by rw [Nat.mul_add]

/-! ### The sequential accumulation bound (gamma_n), scaled

A computed running sum starts at `s` and, at each step, adds an exact term
`t` and a rounding error `d`. `Valid K` says every error is at most `1/K` of
the magnitudes it combines, which the IEEE standard model implies with
`K = 2^p` for round-to-nearest (`|d| ≤ u |s + t| ≤ u (|s| + |t|)`) and
`K = 2^(p-1)` for directed rounding. The theorem is
`|computed - exact| ≤ ((1 + 1/K)^n - 1) Σ|t|`, stated without subtraction. -/

/-- Computed running sum. -/
def runSum : Int → List (Int × Int) → Int
  | s, [] => s
  | s, (t, d) :: rest => runSum (s + t + d) rest

/-- Every step's error is at most `1/K` of the magnitudes it combines. -/
def Valid (K : Nat) : Int → List (Int × Int) → Prop
  | _, [] => True
  | s, (t, d) :: rest => K * d.natAbs ≤ s.natAbs + t.natAbs ∧ Valid K (s + t + d) rest

/-- Exact sum of the terms. -/
def terms (steps : List (Int × Int)) : Int := (steps.map Prod.fst).sum

/-- Sum of the terms' magnitudes. -/
def mass (steps : List (Int × Int)) : Nat := (steps.map (fun p => p.1.natAbs)).sum

theorem step_alg (K P Qm a b c e T0 a' b' : Nat) (hPQ : P ≤ Qm)
    (h1 : P * (a + T0) ≤ Qm * T0) (h2 : P * b ≤ Qm * T0) (he : K * e ≤ b + c)
    (ha' : a' ≤ a + e) (hb' : b' ≤ b + c + e) :
    K * P * (a' + (T0 + c)) ≤ (K + 1) * Qm * (T0 + c) ∧
      K * P * b' ≤ (K + 1) * Qm * (T0 + c) := by
  have k1 := Nat.mul_le_mul_left K h1
  have k2 := Nat.mul_le_mul_left K h2
  have k3 := Nat.mul_le_mul_left P he
  have k4 := Nat.mul_le_mul_left (K * P) ha'
  have k5 := Nat.mul_le_mul_left (K * P) hb'
  have k6 := Nat.mul_le_mul_right c hPQ
  have k7 := Nat.mul_le_mul_left K k6
  simp only [Nat.mul_add, Nat.add_mul, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm,
    Nat.one_mul, Nat.mul_one] at *
  constructor <;> omega

theorem chain (K : Nat) : ∀ (steps : List (Int × Int)) (s S0 : Int) (T0 m : Nat),
    K ^ m * ((s - S0).natAbs + T0) ≤ (K + 1) ^ m * T0 →
    K ^ m * s.natAbs ≤ (K + 1) ^ m * T0 →
    Valid K s steps →
    K ^ (m + steps.length) * ((runSum s steps - (S0 + terms steps)).natAbs + (T0 + mass steps))
        ≤ (K + 1) ^ (m + steps.length) * (T0 + mass steps) ∧
      K ^ (m + steps.length) * (runSum s steps).natAbs
        ≤ (K + 1) ^ (m + steps.length) * (T0 + mass steps)
  | [], s, S0, T0, m, h1, h2, _ => by
    simp only [runSum, terms, mass, List.map_nil, List.sum_nil, List.length_nil, Nat.add_zero,
      Int.add_zero]
    exact ⟨h1, h2⟩
  | (t, d) :: rest, s, S0, T0, m, h1, h2, hv => by
    obtain ⟨hd, hrest⟩ := hv
    have hPQ : K ^ m ≤ (K + 1) ^ m := Nat.pow_le_pow_left (Nat.le_succ K) m
    have ha' : (s + t + d - (S0 + t)).natAbs ≤ (s - S0).natAbs + d.natAbs := by
      have e : s + t + d - (S0 + t) = (s - S0) + d := by omega
      rw [e]
      exact Int.natAbs_add_le _ _
    have hb' : (s + t + d).natAbs ≤ s.natAbs + t.natAbs + d.natAbs :=
      Nat.le_trans (Int.natAbs_add_le _ _) (Nat.add_le_add_right (Int.natAbs_add_le _ _) _)
    obtain ⟨n1, n2⟩ := step_alg K (K ^ m) ((K + 1) ^ m) (s - S0).natAbs s.natAbs t.natAbs
      d.natAbs T0 _ _ hPQ h1 h2 hd ha' hb'
    have e1 : K * K ^ m = K ^ (m + 1) := by rw [Nat.pow_succ, Nat.mul_comm]
    have e2 : (K + 1) * (K + 1) ^ m = (K + 1) ^ (m + 1) := by rw [Nat.pow_succ, Nat.mul_comm]
    rw [e1, e2] at n1 n2
    have ih := chain K rest (s + t + d) (S0 + t) (T0 + t.natAbs) (m + 1) n1 n2 hrest
    simp only [runSum, terms, mass, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    have el : m + (rest.length + 1) = m + 1 + rest.length := by omega
    have eS : S0 + (t + (rest.map Prod.fst).sum) = S0 + t + (rest.map Prod.fst).sum := by omega
    have eT : T0 + (t.natAbs + (rest.map (fun p => p.1.natAbs)).sum)
        = T0 + t.natAbs + (rest.map (fun p => p.1.natAbs)).sum := by omega
    rw [el, eS, eT]
    exact ih

/-- The gamma_n bound: a running sum from zero with per-step relative error
at most `1/K` satisfies `K^n (|computed - exact| + Σ|t|) ≤ (K+1)^n Σ|t|`. -/
theorem sequential_sum_bound (K : Nat) (steps : List (Int × Int)) (hv : Valid K 0 steps) :
    K ^ steps.length * ((runSum 0 steps - terms steps).natAbs + mass steps)
      ≤ (K + 1) ^ steps.length * mass steps := by
  have h := (chain K steps 0 0 0 0 (by simp) (by simp) hv).1
  simpa using h

end CertifiedArgmax
