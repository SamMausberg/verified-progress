/-
The stock head's token under a rounding model, in scaled-integer form.

Real logits, FP32 accumulators and their BF16 roundings are dyadic rationals, so
they are integers after clearing a common positive denominator. The stock head
returns the first index among the maxima of its rounded logits (`IsFirstMax`,
the rule of `torch.argmax`). Rounding is a function `rn : Int → Int` with a
spacing `u : Int → Int` on magnitudes, assumed to have three properties of
IEEE round-to-nearest in the finite range (`RoundModel`): `rn` is monotone, it
moves a value by at most half the spacing at that value's magnitude, and the
spacing does not decrease with the magnitude.

From these hypotheses the file checks the paper's Lemma 2.2 (BF16
separation), Theorem 2.3 (gap condition), the screen margin of Section 2.2
(an entry dropped by the margin cannot be the stock token), and
Proposition 2.4 (interval condition), with its rounding of interval ends.

Not formalised: that BF16 round-to-nearest-even satisfies `RoundModel` (a
property of the format, argued on paper), the bound `G_i` on the stock
kernel's accumulation error (an assumed error model, tested on replayed
positions), and the bridge from floating-point values to these integers.
-/
namespace StockDecision

/-- Absolute value on `Int`, in a form `omega` can unfold. -/
def iabs (x : Int) : Int := if 0 ≤ x then x else -x

theorem iabs_nonneg (x : Int) : 0 ≤ iabs x := by
  unfold iabs; split <;> omega

theorem le_iabs (x : Int) : x ≤ iabs x ∧ -x ≤ iabs x := by
  unfold iabs; split <;> omega

theorem iabs_le {x c : Int} (h : iabs x ≤ c) : x ≤ c ∧ -x ≤ c := by
  have := le_iabs x; omega

theorem iabs_le_of {x c : Int} (h1 : x ≤ c) (h2 : -x ≤ c) : iabs x ≤ c := by
  unfold iabs; split <;> omega

/-- Three properties of round-to-nearest that the decision needs. `u m` is the
spacing of the format at magnitude `m`. -/
structure RoundModel (rn u : Int → Int) : Prop where
  /-- Rounding is monotone. -/
  mono : ∀ a b, a ≤ b → rn a ≤ rn b
  /-- Rounding moves `y` by at most half the spacing at `|y|`. -/
  near : ∀ y, 2 * iabs (rn y - y) ≤ u (iabs y)
  /-- The spacing does not decrease with the magnitude. -/
  spacing_mono : ∀ a b, 0 ≤ a → a ≤ b → u a ≤ u b

/-- `k` is the first index among the maxima of `L`: the token the stock head
returns from rounded logits `L`. -/
def IsFirstMax {n : Nat} (L : Fin n → Int) (k : Fin n) : Prop :=
  (∀ j, L j ≤ L k) ∧ ∀ j, L j = L k → k ≤ j

/-- The first maximizer is unique, so `IsFirstMax` names the stock token. -/
theorem firstMax_unique {n : Nat} (L : Fin n → Int) (k k' : Fin n)
    (h : IsFirstMax L k) (h' : IsFirstMax L k') : k = k' := by
  have e : L k = L k' := by
    have := h.1 k'; have := h'.1 k; omega
  have := h.2 k' e.symm; have := h'.2 k e; omega

/-- A strict maximizer is the first maximizer. -/
theorem firstMax_of_strict {n : Nat} (L : Fin n → Int) (a : Fin n)
    (h : ∀ b, b ≠ a → L b < L a) : IsFirstMax L a := by
  refine ⟨fun j => ?_, fun j hj => ?_⟩
  · by_cases e : j = a
    · subst e; omega
    · have := h j e; omega
  · by_cases e : j = a
    · subst e; omega
    · have := h j e; omega

/-- Lemma 2.2 (BF16 separation): if `ya - yb` exceeds the spacing at
`max (|ya|, |yb|)`, rounding keeps `ya` strictly above `yb`. -/
theorem separation {rn u : Int → Int} (R : RoundModel rn u) {ya yb : Int}
    (h : ya - yb > u (max (iabs ya) (iabs yb))) : rn yb < rn ya := by
  have na := R.near ya
  have nb := R.near yb
  have sa := R.spacing_mono (iabs ya) (max (iabs ya) (iabs yb)) (iabs_nonneg ya) (by omega)
  have sb := R.spacing_mono (iabs yb) (max (iabs ya) (iabs yb)) (iabs_nonneg yb) (by omega)
  have ea := le_iabs (rn ya - ya)
  have eb := le_iabs (rn yb - yb)
  omega

/-- The pairwise core of the gap condition: with accumulators within `G` of
the real logits, a real gap above `G a + G b` plus the spacing at the larger
magnitude, widened by the larger `G`, keeps `a`'s rounded logit above `b`'s. -/
theorem pair_gap {rn u : Int → Int} (R : RoundModel rn u) {za zb ya yb Ga Gb : Int}
    (ha : iabs (ya - za) ≤ Ga) (hb : iabs (yb - zb) ≤ Gb)
    (gap : za - zb > Ga + Gb + u (max (iabs za) (iabs zb) + max Ga Gb)) :
    rn yb < rn ya := by
  apply separation R
  have da := iabs_le ha
  have db := iabs_le hb
  have za' := le_iabs za
  have zb' := le_iabs zb
  have ya' : iabs ya ≤ iabs za + Ga := iabs_le_of (by omega) (by omega)
  have yb' : iabs yb ≤ iabs zb + Gb := iabs_le_of (by omega) (by omega)
  have m : max (iabs ya) (iabs yb) ≤ max (iabs za) (iabs zb) + max Ga Gb := by omega
  have s := R.spacing_mono _ _ (by have := iabs_nonneg ya; omega) m
  omega

/-- Theorem 2.3 (gap condition). If every accumulator lies within `G i` of its
real logit and `a` beats every other entry by more than
`δ a b = G a + G b + u (max (|z a|, |z b|) + max (G a, G b))`, the stock head
returns `a`. (`a` is then necessarily the real-arithmetic winner.) -/
theorem gap_condition {n : Nat} {rn u : Int → Int} (R : RoundModel rn u)
    (z y G : Fin n → Int) (hG : ∀ i, iabs (y i - z i) ≤ G i) (a : Fin n)
    (gap : ∀ b, b ≠ a →
      z a - z b > G a + G b + u (max (iabs (z a)) (iabs (z b)) + max (G a) (G b))) :
    IsFirstMax (fun i => rn (y i)) a :=
  firstMax_of_strict _ a (fun b hb => pair_gap R (hG a) (hG b) (gap b hb))

/-- The screen margin of Section 2.2. Given valid intervals `lo ≤ z ≤ hi`, a
bound `Gmax` on every `G i` and a bound `zbar` on every interval end's
magnitude, an entry `j` whose upper end lies more than
`δmax = 2 Gmax + u (zbar + Gmax)` below some entry's lower end rounds strictly
below that entry, so it cannot be the stock token. -/
theorem screen_drop {n : Nat} {rn u : Int → Int} (R : RoundModel rn u)
    (z y G lo hi : Fin n → Int) (hG : ∀ i, iabs (y i - z i) ≤ G i)
    (enc : ∀ i, lo i ≤ z i ∧ z i ≤ hi i) (Gmax zbar : Int)
    (hGmax : ∀ i, G i ≤ Gmax) (hzbar : ∀ i, iabs (lo i) ≤ zbar ∧ iabs (hi i) ≤ zbar)
    (i j : Fin n) (drop : hi j < lo i - (2 * Gmax + u (zbar + Gmax))) :
    rn (y j) < rn (y i) ∧ ¬ IsFirstMax (fun k => rn (y k)) j := by
  have zb : ∀ k, iabs (z k) ≤ zbar := by
    intro k
    have e := enc k; have b := hzbar k
    have l := le_iabs (lo k); have h := le_iabs (hi k)
    exact iabs_le_of (by omega) (by omega)
  have g0 : ∀ k, 0 ≤ G k := fun k => by
    have := hG k; have := iabs_nonneg (y k - z k); omega
  have m : max (iabs (z i)) (iabs (z j)) + max (G i) (G j) ≤ zbar + Gmax := by
    have := zb i; have := zb j; have := hGmax i; have := hGmax j; omega
  have s := R.spacing_mono _ _
    (by have := iabs_nonneg (z i); have := g0 i; omega) m
  have ei := enc i; have ej := enc j
  have hi' := hGmax i; have hj' := hGmax j
  have lt : rn (y j) < rn (y i) := pair_gap R (hG i) (hG j) (by omega)
  refine ⟨lt, fun h => ?_⟩
  have h1 : rn (y i) ≤ rn (y j) := h.1 i
  omega

/-- Proposition 2.4 (interval condition), on rounded values. Let `lo i ≤ L i ≤ hi i`
for every entry, let `C` be a set of entries and `k ∈ C` attain the largest
lower end in `C`. If every entry outside `C` has its upper end below some lower
end in `C`, every earlier entry of `C` has its upper end below `lo k`, and every
later entry of `C` has its upper end at most `lo k`, then the stock head
returns `k`. (Choosing `k` as the first entry of `C` attaining that largest
lower end is what makes the second condition checkable; the proof needs only
that `k` attains it.) -/
theorem interval_condition {n : Nat} (L lo hi : Fin n → Int)
    (enc : ∀ i, lo i ≤ L i ∧ L i ≤ hi i) (C : Fin n → Prop) (k : Fin n) (hk : C k)
    (kmax : ∀ i, C i → lo i ≤ lo k)
    (outside : ∀ j, ¬ C j → ∃ i, C i ∧ hi j < lo i)
    (before : ∀ j, C j → j < k → hi j < lo k)
    (after : ∀ j, C j → k < j → hi j ≤ lo k) :
    IsFirstMax L k := by
  have below : ∀ j, j ≠ k → L j < L k ∨ (k < j ∧ L j ≤ L k) := by
    intro j hj
    have ej := enc j; have ek := enc k
    by_cases c : C j
    · by_cases o : j < k
      · have := before j c o; left; omega
      · have o' : k < j := by
          have : j ≠ k := hj
          have : j.val ≠ k.val := fun e => this (Fin.ext e)
          rw [Fin.lt_def] at o ⊢; omega
        have := after j c o'; right; exact ⟨o', by omega⟩
    · obtain ⟨i, ci, hlt⟩ := outside j c
      have := kmax i ci; left; omega
  refine ⟨fun j => ?_, fun j hj => ?_⟩
  · by_cases e : j = k
    · subst e; omega
    · rcases below j e with h | h <;> omega
  · by_cases e : j = k
    · subst e; omega
    · rcases below j e with h | h
      · omega
      · exact Fin.le_of_lt h.1

/-- Proposition 2.4 with the rounding of Section 2.3: intervals around the
stock kernel's accumulators, rounded at both ends, contain the rounded
logits because rounding is monotone, so the interval condition on the rounded
ends fixes the stock token. -/
theorem interval_condition_rounded {n : Nat} {rn u : Int → Int} (R : RoundModel rn u)
    (y ylo yhi : Fin n → Int) (enc : ∀ i, ylo i ≤ y i ∧ y i ≤ yhi i)
    (C : Fin n → Prop) (k : Fin n) (hk : C k)
    (kmax : ∀ i, C i → rn (ylo i) ≤ rn (ylo k))
    (outside : ∀ j, ¬ C j → ∃ i, C i ∧ rn (yhi j) < rn (ylo i))
    (before : ∀ j, C j → j < k → rn (yhi j) < rn (ylo k))
    (after : ∀ j, C j → k < j → rn (yhi j) ≤ rn (ylo k)) :
    IsFirstMax (fun i => rn (y i)) k :=
  interval_condition (fun i => rn (y i)) (fun i => rn (ylo i)) (fun i => rn (yhi i))
    (fun i => ⟨R.mono _ _ (enc i).1, R.mono _ _ (enc i).2⟩) C k hk kmax outside before after

/-- The model is not vacuous: rounding down to an even grid, with spacing 2,
satisfies it (round-to-nearest on a grid of spacing `s` does the same with
`u = s`). -/
theorem roundModel_grid : RoundModel (fun y => 2 * (y / 2)) (fun _ => 2) where
  mono := fun a b h => by omega
  near := fun y => by
    have : iabs (2 * (y / 2) - y) ≤ 1 := iabs_le_of (by omega) (by omega)
    omega
  spacing_mono := fun _ _ _ _ => by omega

/-- Round-to-nearest on the even integers with ties to multiples of 4 (ties to
even): the kind of rounding the model describes, with spacing 2. -/
def rnTie (y : Int) : Int := if y % 4 = 1 then y - 1 else if y % 4 = 3 then y + 1 else y

theorem roundModel_tie : RoundModel rnTie (fun _ => 2) where
  mono := fun a b h => by unfold rnTie; split <;> split <;> (try split) <;> (try split) <;> omega
  near := fun y => by
    have : iabs (rnTie y - y) ≤ 1 := by
      unfold rnTie; split
      · exact iabs_le_of (by omega) (by omega)
      · split
        · exact iabs_le_of (by omega) (by omega)
        · exact iabs_le_of (by omega) (by omega)
    omega
  spacing_mono := fun _ _ _ _ => by omega

/-- Lemma 2.2 needs its strict gap: with a gap equal to the spacing, a tie
rounded to even can make the two values round to the same number. -/
theorem separation_nonstrict_false :
    ¬ ∀ (rn u : Int → Int), RoundModel rn u → ∀ ya yb : Int,
      ya - yb ≥ u (max (iabs ya) (iabs yb)) → rn yb < rn ya := by
  intro h
  have := h rnTie (fun _ => 2) roundModel_tie 1 (-1) (by show (1:Int) - -1 ≥ 2; omega)
  simp [rnTie] at this

/-- Theorem 2.3 needs its strict gap: with equality the stock token can be an
earlier index that ties after rounding. -/
theorem gap_condition_nonstrict_false :
    ¬ ∀ (rn u : Int → Int), RoundModel rn u → ∀ (z y G : Fin 2 → Int),
      (∀ i, iabs (y i - z i) ≤ G i) → ∀ a : Fin 2,
      (∀ b, b ≠ a → z a - z b ≥ G a + G b + u (max (iabs (z a)) (iabs (z b)) + max (G a) (G b))) →
      IsFirstMax (fun i => rn (y i)) a := by
  intro h
  let z : Fin 2 → Int := fun i => if i.val = 0 then -1 else 1
  have hz : ∀ i, iabs (z i - z i) ≤ 0 := fun i => by simp [iabs]
  have gap : ∀ b : Fin 2, b ≠ 1 → z 1 - z b ≥ 0 + 0 + 2 := by
    intro b hb
    have : b.val = 0 := by have := b.isLt; have : b.val ≠ 1 := fun e => hb (Fin.ext e); omega
    simp [z, this]
  have := h rnTie (fun _ => 2) roundModel_tie z z (fun _ => 0) hz 1 (fun b hb => by
    have := gap b hb; simpa using this)
  have h0 := this.2 0 (by simp [z, rnTie])
  exact absurd h0 (by decide)

/-- Proposition 2.4 needs an earlier entry of `C` to lie strictly below `lo k`:
if it may tie, the first-index rule returns the earlier entry. The statement keeps
the paper's choice of `k` as the first entry of `C` with the largest lower end
(`kfirst`), so the counterexample refutes the paper's weakened proposition, not a
looser one. -/
theorem interval_condition_tie_false :
    ¬ ∀ (L lo hi : Fin 2 → Int), (∀ i, lo i ≤ L i ∧ L i ≤ hi i) → ∀ (C : Fin 2 → Prop) (k : Fin 2), C k →
      (∀ i, C i → lo i ≤ lo k) → (∀ i, C i → lo i = lo k → k ≤ i) →
      (∀ j, ¬ C j → ∃ i, C i ∧ hi j < lo i) →
      (∀ j, C j → j < k → hi j ≤ lo k) → (∀ j, C j → k < j → hi j ≤ lo k) →
      IsFirstMax L k := by
  intro h
  -- lower ends (-1, 0), upper ends and values (0, 0): k = 1 is the first entry with the
  -- largest lower end, entry 0 ties it in value and wins the first-index rule.
  let lo : Fin 2 → Int := fun i => if i.val = 0 then -1 else 0
  have enc : ∀ i, lo i ≤ (fun _ => (0 : Int)) i ∧ (fun _ => (0 : Int)) i ≤ (fun _ => (0 : Int)) i := by
    intro i; simp only [lo]; split <;> omega
  have lo1 : lo 1 = 0 := by simp [lo]
  have kmax : ∀ i, (fun _ => True) i → lo i ≤ lo 1 := by
    intro i _; rw [lo1]; simp only [lo]; split <;> omega
  have kfirst : ∀ i : Fin 2, (fun _ => True) i → lo i = lo 1 → (1 : Fin 2) ≤ i := by
    intro i _ e
    have hi2 := i.isLt
    by_cases h0 : i.val = 0
    · simp [lo, h0] at e
    · show (1 : Fin 2).val ≤ i.val; simp; omega
  have := h (fun _ => 0) lo (fun _ => 0) enc (fun _ => True) 1 trivial kmax kfirst
    (fun _ c => absurd trivial c)
    (fun _ _ _ => by show (0:Int) ≤ lo 1; rw [lo1]; exact Int.le_refl 0)
    (fun _ _ _ => by show (0:Int) ≤ lo 1; rw [lo1]; exact Int.le_refl 0)
  have h0 := this.2 0 rfl
  exact absurd h0 (by decide)

end StockDecision

#print axioms StockDecision.separation
#print axioms StockDecision.gap_condition
#print axioms StockDecision.screen_drop
#print axioms StockDecision.interval_condition
#print axioms StockDecision.interval_condition_rounded
#print axioms StockDecision.firstMax_unique
#print axioms StockDecision.roundModel_grid
#print axioms StockDecision.separation_nonstrict_false
#print axioms StockDecision.gap_condition_nonstrict_false
#print axioms StockDecision.interval_condition_tie_false
