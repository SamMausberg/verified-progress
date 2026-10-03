# Drafting-frontier oracles (P-A, P-B, P-C)

Three external proposals aim at a 5x end-to-end gain over optimized DFlash by verifying
blocks of 64 to 256 tokens without near-perfect per-position drafting accuracy:
innovation-clock drafting (P-A), prefix-isolated sparse planning (P-B) and causal defect
drafting (P-C). This directory holds the first decisive measurement of each, run against
rules committed in `experiments/frontier/README.md` (commit `adc4865`) before any of these
files existed. Every number here is an oracle upper bound under the stated scope (perfect
controllers, free drafting where stated, the measured Triton Stage A cycle), never a served
speedup. All runs are CPU only; no GPU time was used.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the public DFlash
drafter `z-lab/Qwen3.5-4B-DFlash` at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, greedy,
thinking on. Data: the drafter workstream's block-16 DFlash trace on panel-v1 (80 requests,
131,487 output positions), its per-cycle support table (21,067 cycles, top-16 candidates per
slot) and an n-gram snapshot of 1,345 training continuations whose prompts are disjoint from
the panel (provenance of all three in `evidence/drafter/README.md`). The Triton figures
carry the repair README's caveat that the Triton GDN verify kernel's exactness class was not measured.

## Verdicts

| Proposal | Rule | Result | Decision |
|---|---|---|---|
| P-A | A1: perfect-event oracle (the proposal's rule) | rejected for every budget r <= 16; not rejected at r = 32 (5.75x, 95% CI 5.60-5.90, with `copy`) | |
| P-A | A2: controllers drawing tokens from DFlash's top-16 or the interpreter | support 0.924 per position (needs 0.9892 at B = 64, 0.9925 at B = 256): rejected at both widths | **no-go** |
| P-B | B1: first-gap bound, anchor counts 2, 4, 8, 16 | E[G*] <= 14.4 to 37.0 at B = 64 (needs 44.4) and <= 43.4 at B = 256 (needs 111.6): all rejected | **no-go** |
| P-C | C1: one-sweep bound with point-predictor interpreters | at most 1.60 (`copy`) and 2.00 (`mix`) accepted drafts per boundary against 3.56 for fresh DFlash: rejected | **no-go** |

The common cause: each proposal still needs a drafter far more accurate per token than
DFlash-16, and none of the three mechanisms supplies that accuracy. P-A's controller must
supply the tokens its interpreter misses at nearly every second position, P-B's anchors
cannot help the positions before them, and P-C's sweep starts from target predictions that
were conditioned on a wrong token.

## What 5x requires (`frame.json`, derived)

From `evidence/repair/stage_a_oracle_triton.json`: the DFlash block-16 baseline on the
Triton verify kernel runs 7,301.8 us per cycle for 7.603 tokens, with a non-decode share of
2.39%, so 5x end to end needs 5.542x in decode, at most 173.27 us per committed token
(5.771 tokens per ms). A cycle of a perfect B-token block must commit:

| B | cycle with DFlash drafting | free drafter (cycle minus draft) | verify + commit only |
|---|---|---|---|
| 16 | 42.1 tokens (more than 16) | 28.6 (more than 16) | 26.9 |
| 64 | 61.1 (constant alpha 0.9985) | 46.4 (0.9892) | 44.4 (0.9877) |
| 128, interpolated | 84.7 (0.9930) | 69.0 (0.9891) | 66.8 (0.9884) |
| 256 | 132.2 (0.9941) | 114.1 (0.9925) | 111.6 (0.9923) |

Alpha is the constant per-position acceptance that commits that many tokens per cycle.
B = 128 has no Triton measurement: its verify time is interpolated between B = 64 and 256
and its other phases come from the FlashInfer B = 128 run; no rule uses it. The proposals'
111.6 at B = 256 is the verify-plus-commit column: it leaves out the append phase (0.14 ms)
and 0.31 ms of other cycle time, which a free drafter still pays (114.1). Adding 2 ms of
drafting to the measured cycle gives 143.7, as the proposals state. Each millisecond removed
from the cycle lowers the tokens needed by 5.77; the JSON files report, beside each verdict,
the saving that would reverse it.

## Checks of the proposals' own arithmetic and lemmas (derived)

- P-A's sufficiency example (31 eight-token records at 97% conditional accuracy) gives
  1 + 8 sum_{j=1}^{31} 0.97^j = 159.05 tokens, not the stated 161.7 (32 records give 162.07);
  either clears the 143.7 it is compared with. The event records are a lossless re-encoding
  of the tokens: a canonical program of KEEP and PUT commands reproduces the continuation
  with exactly as many overrides as the interpreter has errors on the true prefix, and no
  program uses fewer. A record sequence is therefore right exactly when every token it covers
  is right, and 97% per 8-token record is a per-token hazard of 1 - 0.97^(1/8) = 0.38%, i.e.
  99.62% per token, above the 99.41% any drafter needs at B = 256. The decomposition moves
  the accuracy requirement; it does not relax it.
- P-B's clipping law (with isolation and clamping, the learned plan's progress is the
  minimum of the true-anchor progress and the first wrong anchor) and its uniform-target
  obstruction (a codebook of 2^r blocks gives E[G] <= r + 2 - 2^-(n-r), because
  E[LCP] = sum_k |length-k prefixes of the codebook| / 2^k) are correct. Its sensitivity
  figures reproduce: with 8 perfect-expander anchors at 0.86 each, a 24 ms cycle commits
  138.5 tokens, exactly the 138.5 needed; an expander capped at 192 tokens with anchors at
  0.90 commits 135.9. Both assume an expander that is perfect between anchors (rule B1).
- P-C's budget reproduces exactly when each cycle pays two target passes and DFlash's
  draft: -7.08 ms at B = 64 and +2.27 ms (242.9 tokens) at B = 256. In steady state the
  audit of one proposal is also the anchor for the next sweep, so a cycle needs one target
  pass, and P-C's cost requirement is a drafter's (rule C1 tests whether the sweep meets it).
  Its convergence theorem bounds the sweeps by a weighted causal depth, but for a real target
  the defect F - G depends on every earlier position, so the bound is one sweep per position,
  the bound of plain Jacobi iteration. The operative claim is the margin lemma (an error
  whose oscillation stays below the target's margin keeps the winner); the repair
  workstream's P3 Stage B measured that certificate with G = 0 (anchor logits reused) and
  found it holds at 4.7% of changed positions (`evidence/repair/README.md`).

## `innovation_oracle.json` (P-A)

Interpreters, frozen before the run: `copy`, the token after the most recent earlier
occurrence of the longest recurring suffix (up to 8 tokens) of the history; `mix`, `copy`
when its match is at least 3 tokens long, otherwise a static 4-gram with backoff.

**A0.** Innovation density, the share of output positions where the interpreter is wrong on
the true prefix: `copy` 0.535 (chat 0.656, code 0.443, GSM8K 0.532, MATH-500 0.503); `mix`
0.471. Half the tokens are innovations; the median run between them is one token and the
90th percentile four.

**A1.** Perfect-event oracle: the controller supplies r perfect overrides per cycle at no
cost, the interpreter costs nothing per position, and each cycle costs the measured Triton
cycle at width B minus its draft phase; a whole-request dynamic program picks B in
{16, 64, 256} per cycle with hindsight. End-to-end speedup (95% request-bootstrap interval):

| r | `copy` | `mix` |
|---|---|---|
| 0 | 0.37 (0.35-0.39) | 0.42 (0.40-0.44) |
| 4 | 1.71 (1.64-1.79) | 1.91 (1.84-1.99) |
| 8 | 2.74 (2.65-2.84) | 2.96 (2.87-3.05) |
| 16 | 3.83 (3.71-3.97) | 4.09 (3.95-4.22) |
| 32 | 5.75 (5.60-5.90) | 6.15 (6.03-6.27) |

Rule A1 rejects every budget up to 16 for both interpreters and leaves r = 32. At a fixed
B = 256, r = 32 reaches only 2.8x (`copy`) and 3.1x (`mix`): 32 overrides cover about 64
tokens when half the tokens are innovations. Charging 2 or 5 us per drafted position for the
interpreter changes these figures by under 4%. Passing A1 at r = 32 means a controller that
predicts 32 of every ~64 tokens exactly (the positions where a copy rule fails), at no cost.

**A2.** Support at the stock DFlash-16 anchors: a position is supported when the true token
is among DFlash's top-K candidates at that slot or equals the interpreter's prediction on
the true prefix; alpha_bar is the mean conditional support over positions 5-15.

| Candidate set | mean supported run | alpha_bar | tokens per cycle if every position held alpha_bar (B = 64 / 256) |
|---|---|---|---|
| DFlash top-1 | 5.20 | 0.889 | 9.0 / 9.0 |
| DFlash top-16 | 9.14 | 0.914 | 11.6 / 11.7 |
| `copy` alone | 1.06 | 0.870 | 7.7 / 7.7 |
| `mix` alone | 1.30 | 0.858 | 7.1 / 7.1 |
| top-16 + `copy` | 9.53 | 0.923 | 12.9 / 13.0 |
| top-16 + `mix` | 9.59 | 0.924 | 13.1 / 13.2 |

The interpreter adds about 0.01 per position to the top-16 support (0.914 to 0.924). Rule A2
rejects the class at both widths: even a perfect choice among these candidates would commit
about 13 tokens per cycle against 46 (B = 64) and 114 (B = 256) needed; for that to be
enough, a cheaper verify would have to take 5.8 ms off the 8.0 ms cycle (B = 64) or 17.5 ms
off the 19.8 ms cycle (B = 256). By domain alpha_bar
for top-16 + `mix` is 0.845 (chat) to 0.945 (MATH-500). Scope: at the anchors the stock
trajectory visited, which follow rejections; a P-A program would visit other anchors.

**P-A decision: no-go.** A1 leaves r = 32, but A2 rejects controllers built from DFlash's
candidates. Going further needs a controller that proposes the true token outside DFlash's
top-16 at most about once per 130 positions, which is a much better drafter, not an event
encoding.

## `anchor_bound.json` (P-B)

With isolation an anchor influences only its own and later positions, so with r uniform
anchors spaced s = B / r apart and a perfect controller, the first gap (positions 2 to s) is
filled from the prefix and the first anchor alone, and survival never increases afterwards:
E[G*] <= 2 + sum_{m<s} S'(m) + (B - 1 - s) S'(s - 1). Assumption (rule B1): an adapted
expander drafts that first gap no better than DFlash-16 does from a fresh cycle start or
given its correct first token, S'(m) = max(S_D(m), S_D(m+1)/S_D(1)) from
`evidence/drafter/support/zlab_b16_panel_v1_survival.csv`, with no further decay after slot
15 (the most favourable extension).

| B | anchors r | spacing | E[G*] bound | best domain | each gap like the first (estimate) | gap acceptance needed | verdict |
|---|---|---|---|---|---|---|---|
| 64 | 2 | 32 | 14.4 | 21.9 (code) | 11.0 | 0.983 | rejected |
| 64 | 4 | 16 | 14.4 | 21.9 (code) | 8.4 | 0.972 | rejected |
| 64 | 8 | 8 | 23.3 | 29.8 (code) | 7.9 | 0.944 | rejected |
| 64 | 16 | 4 | 37.0 | 43.9 (MATH-500) | 8.0 | 0.879 | rejected |
| 64 | 32 | 2 | 55.3 | 58.1 (MATH-500) | 14.2 | 0.684 | not rejected |
| 256 | 2 to 16 | 128 to 16 | 43.4 | 75.3 (code) | 8.4 to 27.7 | 0.944 to 0.990 | rejected |
| 256 | 32 | 8 | 85.0 | 112.1 (code) | 7.9 | 0.885 | rejected |
| 256 | 64 | 4 | 144.1 | 172.6 (MATH-500) | 8.0 | 0.755 | not rejected |

(Thresholds 44.4 at B = 64 and 111.6 at B = 256, the most lenient accounting. At B = 256
with 32 anchors the pooled bound is rejected but the code domain's (112.1) just reaches the
threshold; 32 anchors is outside the proposal's counts. "Gap acceptance
needed" is the constant per-position acceptance inside the first gap at which the bound
would reach the threshold; DFlash-16 measures 0.80-0.91. The estimate column assumes every
gap behaves like the first and gaps fail independently; it is not a bound.)

**P-B decision: no-go.** Every anchor count the proposal considers (2, 4, 8, 16) is rejected
at both widths. The bound survives only when the controller supplies every second (B = 64)
or every fourth (B = 256) token, which is a per-token drafter again, and even then the
estimate with every gap like the first is 8 to 14 tokens per cycle. Going further would need
an expander that drafts each gap at 0.88-0.99 per position (0.94-0.99 at B = 256) with only
sparse anchors, more than DFlash-16 manages with the full prefix (0.80-0.91). The planted-anchor smoke test (B2) was not run:
it cannot change this verdict, which depends on the first gap only.

## `defect_oracle.json` (P-C)

At 18,022 post-rejection boundaries of the block-16 trace with old positions left after the
correction (mean horizon 10.4 positions), P-C's sweep from the audit's target scores over the
rejected draft, with a point-predictor interpreter, can only choose the interpreter's new
token, the target's old argmax or the target's unrecorded second choice (counted as right).
The leading run of such positions bounds the sweep's accepted drafts for any boost. Because a
deterministic causal proposal's free-running accepted prefix equals its teacher-forced one,
the interpreter is evaluated on the true prefix.

| Per boundary (95% request-bootstrap interval) | accepted drafts within the old window |
|---|---|
| target's old predictions alone (recycling) | 0.44 (0.41-0.47) |
| `copy` alone / `mix` alone | 0.59 / 0.83 |
| P-C bound, `copy` | 1.60 (1.48-1.74) |
| P-C bound, `mix` | 2.00 (1.86-2.18) |
| fresh DFlash, capped at the same horizon | 3.56 (3.33-3.82) |

**P-C decision: no-go.** The bound minus fresh DFlash is -1.95 (-2.13, -1.80) for `copy` and
-1.55 (-1.70, -1.41) for `mix`: the sweep cannot match redrafting within a 16-token block,
let alone carry the 114-132 tokens a B = 256 cycle needs. Not covered: an interpreter with
full logits, or one that is itself a strong drafter.

## Commands

The pre-registered generators ran at `adc4865` (`frame.json`, `innovation_oracle.json`,
`defect_oracle.json`). `anchor_bound.json` was regenerated twice after the first run, with
the rule unchanged: at `61bc9cd`, which adds two reporting fields, and at `f4a9665`, a
correction. The first run's extension past slot 15 multiplied the slot-1-conditioned survival
by DFlash's position-15 acceptance once more, an extra decay that the rule's no-further-decay
extension excludes; it understated the bound for gaps longer than 15 (B = 256: 39.7 instead
of 43.4; B = 64 with 2 or 4 anchors: 13.7 instead of 14.4). No verdict changed. Each JSON
records its repo commit and the SHA-256 of its generators and inputs.

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

Input hashes: sequences `19fc24e8...584f`, n-gram snapshot `b1174862...773d`, support table
`a587d952...c36b` (full values in the JSON files).
