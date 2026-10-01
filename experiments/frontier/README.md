# Drafting-frontier oracles (pre-registration)

Three external proposals aim at the 5x target over optimized DFlash with blocks of 64 to
256 tokens without near-perfect per-position drafting accuracy: innovation-clock drafting
(P-A), prefix-isolated sparse planning (P-B) and causal defect drafting (P-C). Each names a
first decisive measurement. This directory runs the cheapest version of each, against
rules written here before any of them ran. Every result is an oracle upper bound under the
stated scope, never a served speedup. Results go to `evidence/frontier/`.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the public DFlash
drafter `z-lab/Qwen3.5-4B-DFlash` at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, greedy.
No GPU time is needed for any rule below.

## The requirement (`frame.py`, derived)

From `evidence/repair/stage_a_oracle_triton.json` (Triton GDN verify kernel, pending its
exactness classification): the DFlash block-16 baseline runs 7,301.8 us per cycle for
7.603 tokens with a 2.39% non-decode share, so 5x end to end needs 5.542x in decode, at
most 173.27 us per committed token (5.771 tokens per ms). Tokens a cycle must commit:

| B | cycle with DFlash drafting | free drafter (cycle minus draft) | verify + commit only |
|---|---|---|---|
| 16 | 42.1 (more than 16) | 28.6 | 26.9 |
| 64 | 61.1 (alpha 0.9985) | 46.4 (alpha 0.9892) | 44.4 |
| 256 | 132.2 (alpha 0.9941) | 114.1 (alpha 0.9925) | 111.6 |

(alpha: the constant per-position acceptance that commits that many tokens per cycle.)
B = 128 has no Triton measurement; `frame.json` gives an interpolated row (about 85
tokens, alpha 0.993) that no rule uses. The proposals' 111.6 at B = 256 is the last column,
which leaves out append and the rest of the cycle; the free-drafter column is 114.1.

A cheaper verify lowers every threshold: each millisecond removed from the cycle lowers
the tokens needed by 5.77. The JSON files report, beside each verdict, how much cycle time a
cheaper verify would have to save to reverse it. That is sensitivity only; the rules use the
measured cycles.

## Data

- Held-out continuations: the drafter workstream's block-16 DFlash trace on panel-v1
  (`~/vp-data/drafter/trace/b16`: 80 requests, greedy, thinking on, up to 2,048 new
  tokens; per-cycle drafts and target argmax rows). Prompts are rebuilt with the chat
  template as `experiments/drafter/support_screen.py` does and checked against every
  trace anchor (`interpreters.py sequences`). The panel contains benchmark test problems
  and is used for predictability only.
- The drafter workstream's per-cycle support table (`cycles.pt`, SHA-256 `a587d952...`,
  provenance in `evidence/drafter/README.md`): DFlash's top-16 candidates at every slot of
  the 21,067 kept cycles.
- n-gram training data: a frozen snapshot of the first 1,345 rows of the drafter
  workstream's training targets (`targets-v2.jsonl`), whose prompts are hash-checked
  disjoint from both panels (`evidence/drafter/data/prompts-v2.manifest.json`).

## Interpreters (declared before any measurement; `interpreters.py`)

- `copy`: the longest suffix (n <= 8) of the history that occurs earlier in it; predicts
  the token after its most recent earlier occurrence; no recurring suffix means no
  prediction.
- `mix`: `copy` when its match has n >= 3, otherwise a static 4-gram with backoff from
  the snapshot.

Both are table lookups. A GPU implementation still needs one dependent step per drafted
position; the rules charge c_G = 0 us per position, so they bound every implementation,
and `innovation_oracle.json` also reports c_G = 2 and 5 us.

## Rules

**A0 (P-A, descriptive).** Innovation density of each interpreter: the share of output
positions where it is wrong on the true prefix.

**A1 (P-A, the proposal's own rule).** Perfect-event oracle (`innovation_oracle.py`): a
controller supplies r perfect overrides per cycle for r in {0, 1, 2, 4, 8, 16, 32}, is
free, and the cycle costs the measured Triton cycle at width B minus its draft phase. A
whole-request dynamic program picks B in {16, 64, 256} per cycle with hindsight. Rule:
reject interpreter G at budget r if the upper end of the 95% request-bootstrap interval of
the end-to-end oracle speedup is below 5.0 at c_G = 0.

**A2 (P-A, controller class).** For controllers whose replacement tokens come from the
public DFlash drafter's top-16 candidates at that slot (or the interpreter's default), a
position is supported when the true token is in that set. At the stock block-16 anchors,
the mean conditional support over positions 5-15, alpha_bar, is compared with the constant
acceptance a free drafter needs. Rule: reject that controller class at width B if
alpha_bar < 0.9892 (B = 64) or < 0.9925 (B = 256).

**P-A decision.** Go to a learned-controller stage (which needs approval and a
written GPU budget, and the proposal's matched per-token causal controller as its
control) only if A1 leaves some (G, r <= 32) unrejected and A2 leaves the class
unrejected at B = 64 or at B = 256 for that G. Otherwise no-go; the escape is a controller that proposes the
true token outside DFlash's top-16 often enough, which is a better drafter.

Why A2 is needed: the canonical event records are a lossless re-encoding of the tokens
(`decode_encode` in the proposal's Lean file), so a record sequence is right exactly when
every token it covers is right. The proposal's example (8-token segments, 97% records)
is a per-token hazard of 1 - 0.97^(1/8) = 0.38%, i.e. 99.62% per-token acceptance, above
the 99.41% that B = 256 needs from any drafter. A1 gives the controller its events for
free and tests only the interpreter and the cycle cost; A2 tests where the events could
come from.

**B1 (P-B, first-gap bound; `anchor_bound.py`).** With isolation, an anchor influences
only its own and later positions, so with r uniform anchors spaced s = B // r apart and a
perfect controller, the first gap (positions 2..s) is filled from the prefix and the
first anchor only, and survival never increases afterwards:
E[G*] <= 2 + sum_{m<s} S'(m) + (B - 1 - s) S'(s - 1). Assumption: an adapted expander
drafts that gap no better than DFlash-16 does from a fresh cycle start or given its
correct first token, S'(m) = max(S_D(m), S_D(m+1) / S_D(1)) from the committed survival
`evidence/drafter/support/zlab_b16_panel_v1_survival.csv`, with no further decay past
slot 15 (most favourable). Rule: reject anchor count r at width B if the pooled bound is
below the verify + commit threshold (44.4 at B = 64, 111.6 at B = 256), the most lenient
column. P-B is a no-go at width B if r = 2, 4, 8 and 16 (the proposal's counts) are all
rejected; it goes to its Stage A (an adapted wide expander, GPU budget needed) only if one
of them survives. This bound uses inputs that were already committed when the rule was
written, and its rough size was anticipated from them; it is a derived bound, not a new
measurement. The escape is an expander that drafts inside a gap at the per-position
acceptance `gap_alpha_needed_for_threshold` in the JSON. (That field and
`independent_gaps_estimate`, the expected commit if every gap behaved like the first and gaps
failed independently, were added after the first run as reporting only; they replace a
`first_gap_survival_needed_lenient` field that was meaningless for gaps longer than the
threshold. The rule is unchanged.)

**B2 (P-B, optional smoke test, run only on request).** Plant true tokens at block
slots 4 and 8 of DFlash-4B's block-16 input at the panel-v1 anchors (the support-screen
pipeline, offline) and compare conditional acceptance after (and, because DFlash's last
layer is bidirectional, before) the planted slot with the unplanted draft. About 20
minutes under `scripts/gpu_lock.sh -s`. It cannot change B1's verdict, which depends only
on the first gap.

**C1 (P-C, one-sweep bound; `defect_oracle.py`).** At every post-rejection boundary of the
block-16 trace with old positions left after the correction, P-C's sweep from the audit's
target scores over the rejected draft, with a point-predictor interpreter, can only pick
G's new token, the target's old argmax, or the target's unrecorded second choice (counted
as right). The leading run of such positions bounds P-C's accepted drafts over the rest of
the old window for any boost. Control: the fresh DFlash draft the engine used next, capped
at the same horizon. Rule: reject P-C with these interpreters if the upper end of the 95%
request-bootstrap interval of mean(P-C bound - fresh DFlash) is below 0, i.e. the sweep
cannot match redrafting even within a 16-token block. Not covered: an interpreter with full
logits, or one that is itself a strong drafter.

**P-C decision.** No-go if C1 rejects both interpreters. Note on the proposal's budget:
in steady state the audit of x^k is also the anchor for x^{k+1}, so a cycle needs one
target pass, not two; P-C's cost requirement is then a drafter's (132 tokens at B = 256),
and C1 tests whether the sweep can deliver them.

## Commands

```sh
source scripts/sglang_env.sh
python experiments/frontier/interpreters.py sequences --trace ~/vp-data/drafter/trace/b16 \
    --panel experiments/drafter/panel-v1.jsonl --out ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl
python experiments/frontier/interpreters.py snapshot --targets ~/vp-data/drafter/data/targets-v2.jsonl \
    --rows 1345 --out ~/vp-data/frontier/data/ngram_train.jsonl
python experiments/frontier/frame.py --out evidence/frontier/frame.json
python experiments/frontier/anchor_bound.py --out evidence/frontier/anchor_bound.json
python experiments/frontier/innovation_oracle.py --sequences ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl \
    --ngram ~/vp-data/frontier/data/ngram_train.jsonl \
    --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt --out evidence/frontier/innovation_oracle.json
python experiments/frontier/defect_oracle.py --trace ~/vp-data/drafter/trace/b16 \
    --sequences ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl \
    --ngram ~/vp-data/frontier/data/ngram_train.jsonl --out evidence/frontier/defect_oracle.json
```

All of them are single-threaded CPU jobs of a few minutes; run them under
`scripts/gpu_lock.sh -s` when an exclusive hold is queued.
