#!/usr/bin/env bash
# Check that the Lean lemmas are not vacuous: each deliberately false variant below must fail to
# elaborate. Every variant weakens one hypothesis of a lemma in formal/ (a strict inequality made
# non-strict, or a margin halved), on a copy in a temporary directory; formal/ itself is never
# modified, because scripts/check_lean.sh elaborates every file there and must pass.
#
#   PATH=$HOME/.elan/bin:$PATH bash scripts/check_lean_variants.sh
#
# Exit status: 0 if every variant is rejected, 1 if any variant elaborates or an edit did not
# apply, 2 if Lean is not installed.
set -euo pipefail
formal="$(cd "$(dirname "$0")/../formal" && pwd)"
command -v lean >/dev/null || { echo 'Lean is not installed; variants NOT checked.' >&2; exit 2; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$formal/lean-toolchain" "$work/"
cd "$work"
lean --version

status=0
# check_variant NAME FILE SED-SCRIPT: apply the edit to a copy of FILE and expect Lean to reject it.
check_variant() {
  local name="$1" file="$2" edit="$3"
  sed -e "$edit" "$formal/$file" > "$work/$file"
  if cmp -s "$formal/$file" "$work/$file"; then
    echo "FAIL $name: the edit did not apply"
    status=1
  elif lean "$work/$file" > "$work/$name.log" 2>&1; then
    echo "FAIL $name: the false variant elaborates"
    status=1
  else
    # Report each error's line, so a reader can see that it lies in the weakened lemma.
    echo "rejected $name: $(grep -o "$file:[0-9]*:[0-9]*: error: [^:]*" "$work/$name.log" | sed "s/^$file://" | paste -sd ';' -)"
  fi
}

# Lemma 2.2: a gap equal to the spacing does not separate (a tie can round to even either way).
check_variant separation_nonstrict StockDecision.lean \
  's/(h : ya - yb > u (max (iabs ya) (iabs yb)))/(h : ya - yb >= u (max (iabs ya) (iabs yb)))/'
# Theorem 2.3: the pairwise gap condition with a non-strict gap.
check_variant gap_condition_nonstrict StockDecision.lean \
  's/(gap : za - zb > Ga + Gb/(gap : za - zb >= Ga + Gb/; s/z a - z b > G a + G b/z a - z b >= G a + G b/'
# Section 2.2: the screen margin with delta_max halved to Gmax + u(zbar + Gmax).
check_variant screen_drop_half_margin StockDecision.lean \
  's/(drop : hi j < lo i - (2 \* Gmax + u (zbar + Gmax)))/(drop : hi j < lo i - (Gmax + u (zbar + Gmax)))/'
# Proposition 2.4: an earlier entry of C allowed to tie k (both forms of the condition).
check_variant interval_condition_earlier_tie StockDecision.lean \
  's/(before : ∀ j, C j → j < k → hi j < lo k)/(before : ∀ j, C j → j < k → hi j ≤ lo k)/; s/(before : ∀ j, C j → j < k → rn (yhi j) < rn (ylo k))/(before : ∀ j, C j → j < k → rn (yhi j) ≤ rn (ylo k))/'
# The gamma_n bound with the accumulation error dropped: K^n (|error| + mass) <= K^n mass.
check_variant sequential_sum_bound_exact CertifiedArgmax.lean \
  's/      ≤ (K + 1) ^ steps.length \* mass steps := by/      ≤ K ^ steps.length * mass steps := by/'

exit "$status"
