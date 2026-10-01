# Long-window repair oracles (P2, P3)

Evidence for the repair workstream's kill tests of Sam's proposals P2 (long-window exact
repair of a DFlash window, 2026-09-30) and P3 (target-anchored residual decoding,
2026-09-30). Setting throughout: Qwen/Qwen3.5-4B
at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter z-lab/Qwen3.5-4B-DFlash at
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus the repair patches on
branch `engine/repair` (`engine/sglang/patches/repair/`): patch 0001 (engine commit
`101e52731b`) for the Stage A session and earlier runs, patches 0001 and 0002 (`5d8e00e3e1`)
for the verify decomposition; each run's `run.json` records its engine commit. One GH200,
greedy decoding, concurrency 1. Code is in
`experiments/repair/`; raw traces stay in `~/vp-data/repair/`.

## Results and verdicts

### Stage A (perfect continuations at width B)

Pre-registered rules (Sam, 2026-09-30). P3: "Stage A, perfect continuation blocks at
B = 32/64/128/256 with free repair: S_oracle(B) = B C_D / (A_D (C_anchor(B) + C_audit(B) +
C_state(B))); if it cannot reach the target end to end, stop the two-pass design." P2 Arm A:
perfect-continuation verification at widths 16-128 timed with the state commit; "if perfect
candidates cannot fit the budget, better drafting alone cannot give 5x with that verifier; if
state handling dominates, build a verifier/state kernel next". Target: 5x end to end over
optimized DFlash.

Baseline (measured, `fresh_b16` in `stage_a_timing.json`): stock DFlash at block 16 on the
same eight MATH-500 problems, C_D = 7.457 ms per cycle, A_D = 7.675 tokens per cycle, i.e.
0.972 ms per token (1,029 tokens/s pooled; the per-request median decode rate is 970
tokens/s); prefill and scheduling take f = 2.28% of a request (median). The target therefore
needs 5.51x in decode. DFlash at block 8 is slower (6.866 ms, 5.635 tokens per cycle).

Measured per width (one server per width, eight requests of 2,048 tokens: medians over 1,024
cycles at B = 16 down to 64 at B = 256; the p10 and p90 cycle periods are within 1.3% of the
median at every width):

| B | V(B) ms | commit ms | DFlash draft ms | cycle ms |
|---|---|---|---|---|
| 16 | 4.78 | 0.11 | 2.40 | 7.60 |
| 32 | 7.30 | 0.12 | 2.51 | 10.25 |
| 64 | 15.62 | 0.12 | 2.53 | 18.61 |
| 128 | 28.10 | 0.12 | 2.71 | 31.32 |
| 256 | 44.82 | 0.14 | 3.21 | 48.62 |

Derived (`stage_a_oracle.csv`; decode speedup over DFlash-16 / end to end):

| B | S_a ideal free drafter | perfect blocks with DFlash drafting | S_b estimate | S_b ceiling |
|---|---|---|---|---|
| 16 | 3.17 / 3.02 | 2.04 / 2.00 | 1.65 / 1.62 | 2.19 / 2.13 |
| 32 | 4.19 / 3.91 | 3.03 / 2.90 | 2.19 / 2.13 | 3.22 / 3.07 |
| 64 | 3.95 / 3.70 | 3.34 / 3.17 | 2.06 / 2.01 | 3.45 / 3.27 |
| 128 | 4.41 / 4.09 | 3.97 / 3.72 | 2.29 / 2.22 | 4.07 / 3.80 |
| 256 | 5.53 / 5.01 | 5.12 / 4.68 | 2.89 / 2.77 | 5.24 / 4.78 |

S_b estimate charges the anchor pass V(B) minus the per-position state writes (measured at
B = 16, 64 and 256 in the decomposition session below, bounded by their bytes at 3.0 TB/s at
B = 32 and 128) plus writing the anchor cache (every operator's input
and output at every block position, BF16, 3.95 MB per token). S_b ceiling charges the anchor
pass only one read of the 8.41 GB of weights at the measured 3.83 TB/s read peak
(`evidence/profiles/hbm_bandwidth.json`) plus the anchor cache at that rate, and the audit its
measured V(B) + commit; with no anchor pass at all the two-pass design is S_a.

- **P3 two-pass design: misses the pre-registered target at every width with SGLang's current
  verifier (refuted for this verifier).** The ceiling charges the anchor pass one weight read
  and the audit its measured V(B) + commit, and reaches 4.78x end to end at B = 256. That V(B)
  includes the per-position FP32 state writes: an audit 2.33 ms cheaper (5.2% of V(256),
  `audit_saving_needed_for_target_us`) would reach the target, and the writes' bytes bound them
  at 4.29 ms (9.6%); measured (FlashInfer verify kernel, cross-session), they are 4.10 ms
  (next section), so with a verifier that drops
  them and reconstructs only the accepted boundary state at no charge the ceiling could reach
  5.18x end to end. Such a verifier would
  also make the DFlash baseline cheaper (16 per-position writes per cycle), so a fair rerun gives
  both the same verifier. If both passes dropped the writes, the estimate (anchor pass = V(B)
  without them) would need them to be at least 50% of V(256). With SGLang's Triton verify kernel
  P3 passes Stage A at B = 256 (next section; pending exactness classification). Stage B below refutes P3 independently of the
  verifier.
- **P2 Arm A: 5x end to end only at B = 256 with zero drafting cost and every block accepted**
  (derived from measured V(B)): S_a = 5.01x end to end sits at the threshold; paying DFlash's
  own drafting cost at width B, perfect blocks reach 4.68x. Better drafting alone does not give
  5x with this verifier; with SGLang's Triton verify kernel it would, from B = 64 (next section;
  pending exactness classification).
- With this verifier the verify pass is the binding constraint. Past B = 16, V(B) grows by about
  0.167 ms per extra token, roughly five times what that token's GEMM work and 48 MiB
  per-position FP32 state write account for; the state commit is 0.11-0.14 ms at every width.
  The next section attributes most of it to the FlashInfer GDN verify kernel.

### What the wide-block verify pass spends its time on (measured, with derived oracles)

`stage_a_timing.json` (runs `force_nostate_b*`, `force_tritonverify_b*`, `fresh_tritonverify_b16`),
`stage_a_oracle.{json,csv}` (FlashInfer verify kernel, with measured state writes at
B = 16, 64, 256) and `stage_a_oracle_triton.{json,csv}` (Triton verify kernel). Session
`experiments/repair/runs/decomposition.sh` under the exclusive lock, 2026-10-01 04:49-05:12
UTC, same panel and flags as Stage A, engine build `5d8e00e3e1` (patches 0001 and 0002; the
Stage A session ran `101e52731b`, patch 0001 only, which leaves the verify path unchanged);
foreign CPU load below 0.5 cores; p10-p90 of every cycle period within 1.3% of its median.
The session's two Nsight Systems runs (`force_nsys_b64`, `force_nsys_b256`) failed at launch
and are excluded.

**Cross-session caveat.** The FlashInfer-with-states column comes from the Stage A session
(00:04-00:41 UTC), the other two from this session. The draft phase, which neither change
touches, is 1.8-2.6% shorter in this session (2.335 and 2.339 against 2.397 ms at B = 16, 3.148
and 3.131 against 3.207 ms at B = 256), so differences of a few percent between the sessions are
within drift; a
same-session control (all three variants at B = 16 and 256) is queued.

Which verify kernel the baseline used: at the pin SGLang's GDN verify kernel follows the decode
backend unless `--linear-attn-verify-backend` overrides it (FlashInfer if the decode backend is
FlashInfer, Triton otherwise). All Stage A runs, like the drafter workstream's shared trace,
use the DFlash model card's `--linear-attn-decode-backend flashinfer`, so they verify with
FlashInfer; with SGLang's default linear-attention backend (Triton, as in the bench arms) the
verifier is already Triton and `--linear-attn-verify-backend triton` changes nothing. The
comparison below is therefore FlashInfer verify against Triton verify, with everything else as
on the card.

With `--linear-attn-decode-backend flashinfer` SGLang verifies GDN layers with FlashInfer's
`gated_delta_rule_mtp` (code reading: at one request it runs its inline kernel with a value
tile of 8, so 512 CTAs per layer walk the T block positions one after another, and it compiles
once per T). Two runs separate its parts:

| B | V(B), FlashInfer kernel (Stage A session) | same, per-position states dropped | V(B), Triton kernel |
|---|---|---|---|
| 16 | 4.78 | 4.53 | 4.55 |
| 64 | 15.62 | 14.36 | 7.58 |
| 256 | 44.82 | 40.73 | 19.18 |

(ms, forced full acceptance; dropping the states uses engine patch 0002 and is timing only.)
With the FlashInfer kernel the per-position FP32 state writes cost 0.26, 1.26 and 4.10 ms
(5.3%, 8.1% and 9.1% of V(B); at B = 16 the difference is within the cross-session drift),
close to their bytes at 3 TB/s (0.27, 1.07, 4.29 ms). The larger cost is the FlashInfer verify
kernel itself: selecting SGLang's Triton GDN verify kernel instead
(`--linear-attn-verify-backend triton`, a stock option) makes the whole pass 2.06x faster at
B = 64 and 2.34x faster at B = 256. Whether the FlashInfer kernel's sequential walk is where
the time goes inside the pass (its kernel share) needs the queued Nsight Systems trace; the
ratios here are whole-pass times. **Exactness: pending.** The Triton verify kernel is a
different numerical path: stock DFlash at block 16 with it commits 7.603 tokens per cycle
against 7.675 with the FlashInfer kernel, so the greedy outputs differ, and every Triton-based
figure here (the speed ratios, the baseline below, the Triton Stage A table) is pending an
exactness classification of that kernel switch against plain decoding. That baseline runs
7.302 ms per cycle (1,041 tokens/s pooled, against 1,029 with FlashInfer in the Stage A
session): single runs from two sessions, a 2.1% shorter cycle and 0.9% fewer tokens per cycle,
within plausible run-to-run and cross-session variance, so the baseline gain is not
established; the wide blocks gain 2x.

Stage A with the Triton verify kernel and the DFlash baseline on the same kernel (derived from
the measured V(B) and baseline; `stage_a_oracle_triton.csv`; pending exactness classification;
f = 2.39%, so 5x end to end needs 5.54x in decode; decode / end to end):

| B | S_a ideal free drafter | perfect blocks with DFlash drafting | S_b estimate | S_b ceiling |
|---|---|---|---|---|
| 16 | 3.30 / 3.12 | 2.10 / 2.05 | 1.71 / 1.69 | 2.23 / 2.17 |
| 64 | 7.98 / 6.84 | 5.81 / 5.21 | 4.30 / 3.99 | 6.17 / 5.49 |
| 256 | 12.72 / 9.94 | 10.73 / 8.71 | 7.11 / 6.21 | 11.28 / 9.06 |

(S_b estimate bounds the Triton kernel's state writes by their bytes; no no-state Triton run.
`S_b_ceiling_stateless_audit_e2e` gives the ceiling with a state-free audit;
`S_b_ceiling_meets_target` marks the widths where the ceiling already reaches the target,
B = 64 and 256, where `audit_saving_needed_for_target_us` is 0.)

- **The Stage A verdicts depend on the verify kernel.** With the Triton kernel (pending
  exactness classification), perfect B-token blocks reach the 5x target at B = 64 even paying
  DFlash's drafting cost, and P3's two-pass design reaches it at B = 256 (its ceiling at
  B = 64). With the default FlashInfer kernel neither does (above). With the FlashInfer kernel
  and its measured state writes (4.10 ms, FlashInfer verify kernel, cross-session), P3's ceiling
  with a state-free audit and an uncharged boundary replay would also reach it at B = 256
  (5.18x end to end, `S_b_ceiling_stateless_audit_e2e`; 2.33 ms needed).
- With the Triton kernel (pending exactness classification), P2 therefore turns on drafting: a
  5x gain needs blocks of 64 or more tokens accepted almost entirely, and the repair mechanisms
  tested here do not produce them (one-step recycling below; exact Jacobi about one token per
  pass, Stage B). P3 stays refuted by Stage B.

### P9 support oracle: reuse a cached DFlash window once after a rejection (c = 1)

`p9_support_oracle.json` (derived from a measured per-cycle table and measured phases).
Sam's proposal P9 (2026-10-01) and its pre-registered rule: after a rejection, condition a program
compiled from the first cycle's cached candidate sets on the actual correction and propose the
rest of the old window, at most once; compare with fresh DFlash by the two-cycle aggregate
Delta = E[G2_R - G2_F] - r_F E[A + T2_R - T2_F], r_F = E[G1 + G2_F] / E[T1 + T2_F], and "reject
this finite-window reuse at the tested configuration if its upper confidence bound is <= 0".

Input: the drafter workstream's per-cycle support table for its shared block-16 DFlash trace
(`~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt`, SHA-256
`a587d952074d93f5d135c75760a32587f3e7a631360c4f20884be1c7bdddc36b`, raw data outside git; its
summary matches the committed `evidence/drafter/support/zlab_b16_panel_v1_summary.json`; 21,067
cycles of 80 requests; 207 cycles whose block runs past the output end are excluded by the
producer). It was written on 2026-10-01 at 08:27 UTC by `experiments/drafter/support_screen.py
--save-cycles` (on main since commit `1e66b99`; the run used `b631dc4`, an identical diff), run as
`scripts/gpu_lock.sh -s experiments/drafter/run_support_screen.sh` with the stock engine;
`evidence/drafter/README.md` records the full command and environment. Per
cycle it holds the engine's accepted length L, the realized greedy continuation (the committed
stream, which after a rejection follows the target given the corrected prefix), the engine's
drafted tokens and the leading supported length U_K of the continuation inside the drafter's top-K
candidate sets (K = 1, 2, 4, 8, 16). The candidate sets are an offline recomputation (Hugging Face
target features, SpecForge drafter module, BF16) whose top-1 token matches the engine's drafted
token at 97.4% of positions. Costs are the measured DFlash-16 phases at c = 1 in the drafter's
configuration (`stage_a_timing.json`, run `fresh_b16`): a reused cycle skips the 2.33 ms draft
phase and pays the rest of the 7.46 ms cycle, including the 4.73 ms verify of a full block of 16
positions (a padded verify; see the scope note below the table).

At the 18,274 post-rejection cycle boundaries (the correction at block position J = L + 1, with
m = 15 - J old positions left), P9's program conditions on the whole corrected prefix through the
frozen candidate sets of the old block, so reuse is possible when those top-K sets contain every
token of the corrected prefix (U_K >= J) and m >= 1, both observable at decision time. The greedy
oracle program then proposes the true token wherever the old candidate set contains it, so its
second cycle commits 1 + min(m, U_K - J) tokens; fresh DFlash commits 1 + L' (the next cycle's
accepted length). The oracle's extra first-cycle cost A and its conditioning cost are 0, so it
bounds from above P9's program over the same candidate sets when that program verifies a padded
block and reuses at every supported boundary. A program whose support starts at the correction,
conditioning on the suffix alone, is a different class and is not covered; with recomputed sets it
could also reuse at boundaries where the recomputation drops an already accepted token.

| K | corrected prefix supported | mean supported suffix | oracle G2 / fresh G2 (reused) | Delta (95% CI) | rule |
|---|---|---|---|---|---|
| 1 | 1.6% | 1.72 | 2.72 / 6.94 | -0.04 (-0.06, -0.03) | rejected |
| 2 | 48.1% | 2.10 | 3.10 / 6.07 | -0.66 (-0.81, -0.55) | rejected |
| 4 | 71.1% | 2.64 | 3.64 / 5.82 | -0.42 (-0.56, -0.30) | rejected |
| 8 | 82.4% | 3.34 | 4.34 / 5.66 | +0.22 (+0.12, +0.32) | not rejected |
| 16 | 88.8% | 4.08 | 5.08 / 5.55 | +0.99 (+0.89, +1.07) | not rejected |

(Delta in tokens per post-rejection boundary, r_F = 0.683 tokens per ms; 95% intervals from a
request-level bootstrap with 2,000 resamples, re-estimating r_F in every resample. Every verdict in
the table is for P9's program, which conditions on the whole corrected prefix, under an always-reuse
policy, one that reuses at every supported boundary, with a padded block-16 verify and the
offline-recomputed candidate sets.) Valuing the saved time at DFlash's overall rate (1.029 tokens
per ms) instead gives +0.89 (K = 8) and +1.70 (K = 16), with r_F then a constant. By domain at
K = 16, each with its own two-cycle rate r_F: chat +1.08, code +1.23, maths +1.02, MATH-500 +0.86.
The unchanged cached unary control (the old draft's tail after the correction, no fresh fill)
accepts 0.83 drafts and gives Delta = -1.93 (-2.22, -1.69): rejected.

Scope: padded verify. The table charges every reused cycle the full block-16 verify, as an
implementation that pads the remainder to the block width would pay. After a correction at J only
m = 15 - J old positions remain, so a program could verify m + 1 positions instead, and a narrower
verify that costs less lowers T2_R and raises Delta at every K. For a variable-width verify the
oracle is therefore not an upper bound, and the rejections of top-1, top-2 and top-4 do not follow.
With V_R the mean verify time of a reused cycle in place of 4.73 ms, and everything else unchanged,
Delta = Delta_padded + r_F p_K (4.73 ms - V_R), where p_K is the supported rate in the table. With
the variable-width verify cost at its lower bound of zero (V_R = 0, every other phase still charged
at its block-16 value: accept, commit and append 0.23 ms, and 0.16 ms of the cycle outside the timed
phases; `free_verify_always_reuse` in the JSON, the same request-level bootstrap and resamples as
the table), Delta would be +0.01 (+0.00, +0.02) at K = 1, +0.89 (+0.81, +0.98) at 2, +1.88
(+1.75, +2.05) at 4, +2.89 (+2.71, +3.10) at 8 and +3.86 (+3.64, +4.11) at 16. These bound P9's
always-reuse program at any verify width from above, provided its other phases cost no less than at
block 16, and none of the tested K would be rejected. On the point estimates (no intervals), top-1,
top-2 and top-4 stay negative only while V_R exceeds 0.89, 2.71 and 3.87 ms. The only measured
verify below width 16 at c = 1 is 4.16 ms at width 8 (`fresh_b8`, same session as `fresh_b16`), so
these thresholds are not settled by the data here; a sweep of the verify phase over widths 2 to 16
at c = 1 is queued. The top-8 and top-16 verdicts (not rejected) hold for either implementation.

Omniscient-gate oracle (`omniscient_gate_oracle` in the JSON; an oracle, since its gate knows fresh
DFlash's next accepted length). A gated program may choose fresh DFlash on supported boundaries
where reuse is unfavourable, which the always-reuse table does not allow. Taking, per supported
boundary, the better of oracle reuse and fresh drafting gives a Delta that is >= 0 and >= the
always-reuse Delta by construction, so it rejects nothing. It bounds from above P9's program with
any gate over the same candidate sets with a padded verify, and its excess over always-reuse is the
most a perfect gate could add (same resamples as the table, so the intervals are paired):

| K | gate reuses at | Delta (95% CI) | gain over always-reuse |
|---|---|---|---|
| 1 | 0.7% | +0.01 (+0.01, +0.01) | +0.05 |
| 2 | 24.9% | +0.45 (+0.42, +0.49) | +1.12 |
| 4 | 41.6% | +0.91 (+0.85, +0.98) | +1.33 |
| 8 | 53.9% | +1.44 (+1.35, +1.53) | +1.21 |
| 16 | 62.9% | +2.05 (+1.93, +2.18) | +1.06 |

The same gate applied to the free-verify scoring (`omniscient_gate_free_verify`, same resamples)
bounds gated programs at any verify width, with the other phases charged as above: +0.04
(+0.03, +0.05) at K = 1, +1.47 (+1.35, +1.61) at 2, +2.54 (+2.35, +2.76) at 4, +3.47 (+3.24, +3.73)
at 8 and +4.35 (+4.09, +4.64) at 16.

A perfect gate would add 1.06 to 1.33 tokens per post-rejection boundary at the tested K from 2 to
16 (2, 4, 8 and 16). A real gate decides before the second cycle and does not see fresh DFlash's
outcome, so a gate is worth building only if features known at decision time (the correction
position J, the remaining horizon m, the old sets' probabilities at the remaining positions) predict
which boundaries those are. The supported suffix length U_K - J is not one of them: it needs the
greedy tokens after the correction, so it is an oracle target that such features might predict.

Assumptions of this oracle, stated plainly: (1) the candidate sets are the offline recomputation,
whose top-1 token matches the engine's drafted token at 97.4% of positions, not the engine's own
sets; (2) the costs are the c = 1 phases of one DFlash-16 run (`fresh_b16`, the drafter workstream's
configuration), and a reused cycle is charged exactly a fresh cycle without its draft phase, so its
verify is padded to the full block; (3) the oracle's extra first-cycle cost A is 0: compiling the
program, retaining the candidate sets, conditioning on the correction and catching up the draft
cache are not priced; (4) the oracle knows the true token wherever it lies in the old candidate set,
so it bounds from above P9's program, which conditions on the whole corrected prefix, over the same
candidate sets with a padded verify that reuses at every supported boundary; a program whose support
starts at the correction is a different class and is not covered; (5) it reuses at every supported
boundary, so it does not bound a gated program, one that falls back to fresh DFlash on supported
boundaries it judges unfavourable, and its negative intervals cannot reject such a program (the
omniscient-gate oracle above bounds those).

- **Verdict at c = 1, for P9's program (conditioning on the whole corrected prefix) under
  always-reuse policies with a padded block-16 verify over the offline-recomputed sets: reuse is not
  rejected for top-8 and top-16 candidate sets; top-1, top-2 and top-4 are rejected for this class
  only.** A program that always found the true token inside the old top-16 sets would commit
  slightly fewer tokens than fresh DFlash (5.08 against 5.55) but skip the 2.33 ms draft, and come
  out ahead by about one token per post-rejection boundary. For always-reuse with a padded verify
  this is an upper bound: a real program must select the token (the unary control shows that the old
  top-1 choices fail), and its compile, conditioning and retention costs (A) are charged against the
  same margin. For gated programs with a padded verify the omniscient-gate oracle gives an upper
  bound of +2.05 (+1.93, +2.18) at top-16, and no tested K is rejected. A program whose support
  starts at the correction (suffix-only) is not covered. The c = 8 and 16 draft shares, which set
  T2_R - T2_F beyond c = 1, are queued.

```sh
python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
    --timing evidence/repair/stage_a_timing.json --bootstrap 2000 --out evidence/repair/p9_support_oracle.json
```

### P3 Stage B: anchored residual evaluation on real DFlash blocks (block 16)

`residual_eval_b16.json`, `residual_eval_b16.agree_by_distance.csv` (measured),
`p3_gate_b16.json` (derived). Sam's critique (2026-09-30) names three claims that must hold
separately: (1) fixed cheap operators stay accurate after candidate tokens change, (2) their
errors stay below the decision margins after attention, nonlinearities, recurrence and later
layers, (3) repair resolves enough positions at once to pay for the anchor, the sweeps and the
audit. **Claim (1) fails first, and (2) and (3) fail with it.**

Pre-registered criteria (Sam, 2026-09-30, P3 Stage B): "real DFlash proposals and real
corrections (isolated and cascading), dev/held-out split, fixed bases fitted on dev only, full
residual evaluator including full attention and GDN state; controls: full-target Jacobi from the
same initialization, a comparable standalone compact drafter, and the initializer without
repair. Key novelty ablation: does the anchored residual evaluator beat an ordinary compact
drafter with the same storage and compute?", with the critique's required measurements: two
cascading replays against the original anchor with R_i and actual disagreements, acceptance via
hazards and prefix advance per sweep, a few exact Jacobi sweeps, and the economic gate
mu_R / C_R > mu_0 / C_0 against tuned DFlash.

Setup: the Hugging Face Qwen3.5-4B target (BF16, pinned revision) run block by block from an
exact prefix cache; its argmax agrees with the engine's verify argmax on 99.1% of the
development blocks' positions. Every linear operator (GDN in_proj and out_proj, attention
q/k/v/o, MLP gate/up/down, tied head) is repaired as z-bar + (W U) U^T (x - x-bar) against the
anchor pass over the DFlash draft y0, with all nonlinearities, the convolution, the GDN
recurrence from the exact state and attention over the exact prefix run exactly. Bases are the
leading principal components of the exact input changes of each operator on 120 development
blocks (even-indexed requests); 80 held-out blocks come from odd-indexed requests. Blocks are
real DFlash drafts with a rejection from the drafter workstream's shared trace. Two cascading
replays against the original anchor: y1 = F(y0) and y2 = F(y1), both exact Jacobi iterates.

- (1) Operator accuracy: on held-out exact changes the median relative error
  ||W (I - U U^T) dx|| / ||W dx||, measured for the first projection that reads each operator
  input (GDN in_proj_qkv or attention q_proj, out_proj or o_proj, gate_proj, down_proj, and the
  head; the z, a, b, k, v and up projections share those inputs and bases but their own W is not
  scored), is 0.76-0.90 at rank 16, 0.64-0.81 at rank 128 and 0.51-0.73 at rank 512 (by operator
  class); on the development blocks themselves
  rank 128 captures a median 51-64% of the change energy. The changes caused by token
  corrections are not confined to a small fixed subspace.
- (2) Decisions: on replay 1 the repaired argmax equals the exact argmax at 35% of the changed
  positions with rank 0 (anchor outputs reused unchanged), 37% at rank 128 and 44% at rank 512;
  at the corrected position itself 11-14%; on replay 2, 24-35%. The certificate ratio
  R_i = 2 ||z~ - z||_inf / m_i is below 1 at 4.7% of the positions where it is defined (43 of
  921; the other 41 of the 962 changed positions are exact top-two ties of the exact logits,
  margin 0, where R_i is undefined), the same 43 positions at every rank including rank 0:
  positions whose exact top-two margin is large (median 12 logits, against 1.3 over all 962
  replayed positions), where even reusing the anchor logits stays within half the margin
  (`ratio_below_1_count`, `tied_positions_ratio_undefined`, `margin_median_*`,
  `ratio_below_1_same_positions_at_every_rank` in the JSON). These certificates are not earned
  by the repair; the median R_i is 13-16. Positions before the first change agree exactly in every block (harness check). The
  corrected position's decision is right in 9 of 80 blocks at ranks 0-256 and 11 at rank 512.
- (3) Progress: free-running repair from y1, audited exactly after every sweep, accepts 4.33
  drafts after 0 sweeps and 4.33-4.44 after 4 sweeps at every rank, while exact Jacobi from the
  same anchor accepts 2.98 (y0), 4.33, 5.35, 6.38, 7.34 and 8.30 after 0-5 exact passes: about one
  token per exact sweep, which also bounds any faithful cheap evaluator.
- Economic gate (Sam's mu_R / C_R > mu_0 / C_0 against DFlash-16; repair cost a bytes-only lower
  bound, audit = measured V(16) + commit): the anchored evaluator reaches at most 0.28 of
  DFlash's committed tokens per unit cost at any rank and sweep count. The faithful-evaluator
  reference (exact Jacobi, rows `exact_jacobi_faithful_reference`, sweeps charged only the bytes
  of reading the anchor cache and the GDN state) reaches 0.27, 0.47, 0.66, 0.85 and 1.03 after 0-4
  sweeps, so even a faithful cheap evaluator, one that reproduces the target's sweep exactly at
  that cost, would at best match DFlash after four sweeps. A non-faithful map could in principle
  jump ahead, as consistency-trained models do, but that would be better drafting rather than
  repair, and this evaluator does not. The rigorous bound (audit cost over the most tokens an attempt can add) does not reject
  at B = 16; the measured progress does.
- Controls: full-target Jacobi from the same initialization and the initializer without repair
  (rank 0, anchor outputs reused) were run; the standalone compact drafter control and the novelty
  ablation against it were not. They are unnecessary here: the evaluator's best case gains 0.11
  accepted drafts over no repair in four sweeps, so there is no gain for a compact drafter with
  the same storage and compute to explain.

```sh
scripts/gpu_lock.sh -s experiments/repair/runs/residual_b16.sh   # raw output in ~/vp-data/repair/residual/b16
python experiments/repair/summarize_residual.py ~/vp-data/repair/residual/b16 \
    --out evidence/repair/residual_eval_b16.json
python experiments/repair/p3_gate.py --stage-a evidence/repair/stage_a_oracle.json \
    --residual evidence/repair/residual_eval_b16.json --read-tbps 3.827 --out evidence/repair/p3_gate_b16.json
```

### P2 Arm B (one step): recycling the target's suffix predictions

Measured, offline, on the drafter workstream's shared trace (`one_step_recycling.json`,
`draft_source_accuracy.csv`; method below). At the 18,525 cycle boundaries after a rejection at
block 16, the next block drafted from the previous pass's target predictions accepts 0.52
drafts on average, against 4.44 for the fresh DFlash draft the engine used; keeping the
previous draft's tail gives 1.11, and choosing the best of the three with hindsight 4.55
(+2.5%). Block 8: 0.45 and 1.15 against 3.21 (best of three 3.30). The target's prediction
after a wrong draft token matches the continuation 14% of the time once that token is
corrected, against 84% for DFlash's first drafted token. Recycling the target's suffix does
not improve on fresh DFlash drafting.

## One-step recycling on the shared DFlash trace

`one_step_recycling.json`, `draft_source_accuracy.csv` (measured, offline).

Input: the drafter workstream's greedy DFlash trace on its shared 80-request panel
(32 MATH-500 problems and 16 chat, code and maths prompts each), which records the drafted
block and the target's argmax at every verify row of every cycle. The file records the
trace files' SHA-256, launch command and panel ids.

At every cycle boundary after a rejection, the next block is drafted three ways and
scored against the committed stream: the fresh DFlash draft the engine used, the previous
verify pass's target predictions after the first mismatch (a sliding Jacobi step,
"recycle"), and the previous draft's tail after the corrected token ("keep"). Where
recycle or keep has no token for a position the fresh draft is used. Accepted drafts are
the leading positions that match the committed stream, which is what greedy verification
accepts. `draft_source_accuracy.csv` gives each source's match rate by block index.

The block-16 trace holds 21,354 cycle records for 80 requests, 80 more than the 21,274 verify
cycles the server counted (`spec_verify_ct`): one record per request for the cycle the overlap
scheduler runs before it sees that the request has finished (the same extra cycle appears in our
timing logs). Its tokens lie past the output; it enters at most 80 of the 18,525 boundaries.

```sh
python experiments/repair/analyze_jacobi.py ~/vp-data/drafter/trace/b16 ~/vp-data/drafter/trace/b8 \
    --drop-last 0 --manifest ~/vp-data/drafter/trace/trace_manifest.json --out-dir evidence/repair
```

(`jacobi_summary.json`, which the same command writes, repeats the results without the
provenance and is not kept.)

## Stage A: perfect continuations at width B (P2 Arm A, P3 Stage A)

`stage_a_timing.json` (measured), `stage_a_oracle.json`, `stage_a_oracle.csv` (derived).

Session `experiments/repair/runs/stage_a.sh` (the same sequence was run as
`~/vp-data/repair/runs/timing1/run_timing.sh`, kept with the raw data) under the exclusive lock on
2026-10-01 00:04-00:41 UTC: SGLang with the DFlash drafter at block widths B = 16, 32, 64,
128 and 256, verification forced to accept the whole block (`SGLANG_SIMULATE_ACC_LEN=B`, so a
cycle commits exactly B tokens and costs what a perfect B-token candidate would cost), and the
real DFlash cycle at blocks 16 and 8. Eight MATH-500 problems (`panel.py math --n 8`), 2,048
tokens each (forced runs ignore EOS), concurrency 1, `--max-running-requests 1`, the
configuration of the drafter workstream's shared trace (flashinfer attention and GDN kernels,
overlapped plan stream, default prefill graphs, `--stream-interval 4`). Foreign CPU load
during every run stayed below 0.6 cores (`run.json` in each run directory). Per-cycle GPU
phase times come from CUDA events in the engine probe; the cycle period is measured on the
GPU timeline between consecutive cycle starts.

```sh
scripts/gpu_lock.sh -x experiments/repair/runs/stage_a.sh   # raw runs in ~/vp-data/repair/runs/timing1
scripts/gpu_lock.sh -x experiments/repair/runs/decomposition.sh   # raw runs in ~/vp-data/repair/runs/nsys1
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/timing1/force_b* \
    ~/vp-data/repair/runs/timing1/fresh_b* ~/vp-data/repair/runs/nsys1/force_nostate_b* \
    ~/vp-data/repair/runs/nsys1/force_tritonverify_b* ~/vp-data/repair/runs/nsys1/fresh_tritonverify_b16 \
    --out evidence/repair/stage_a_timing.json
python experiments/repair/stage_a.py --timing evidence/repair/stage_a_timing.json \
    --baseline fresh_b16 --verifier flashinfer --state-bytes-bound --out-dir evidence/repair
python experiments/repair/stage_a.py --timing evidence/repair/stage_a_timing.json \
    --baseline fresh_tritonverify_b16 --verifier triton --state-bytes-bound --suffix _triton \
    --out-dir evidence/repair
```

The decomposition session's two Nsight Systems runs failed at launch (an `nsys launch` option
that only `nsys start` accepts); the Triton-versus-FlashInfer runs answer the attribution
question causally.

The ReplaySSM spec protocol does not start with DFlash on this GDN model ("requires a KDA
model"), and the session's GDN kernel microbenchmark was stopped after 16 minutes of CPU-bound
kernel compilation without output. The per-position state writes inside V(B) are therefore
measured (forced acceptance without them, decomposition session) at B = 16, 64 and 256 and
bounded by their bytes at 3.0 TB/s at B = 32 and 128 in `stage_a_oracle.{json,csv}`, and bounded
by their bytes at every width in `stage_a_oracle_triton.{json,csv}`; `state_writes_source`
labels each row.
