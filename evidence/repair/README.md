# Repair oracles (P2, P3, P9, P12)

Evidence for the repair workstream's kill tests of Sam's proposals P2 (long-window exact
repair of a DFlash window, 2026-09-30), P3 (target-anchored residual decoding, 2026-09-30),
P9 (reusing a cached window once after a rejection, 2026-10-01) and P12 (compiling the final
classifier backward through the last FFN, 2026-10-01). Setting throughout: Qwen/Qwen3.5-4B
at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter z-lab/Qwen3.5-4B-DFlash at
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus the repair patches on
branch `engine/repair` (`engine/sglang/patches/repair/`): patch 0001 (engine commit
`101e52731b`) for the Stage A session and earlier runs, patches 0001 and 0002 (`5d8e00e3e1`)
for the verify decomposition; each run's `run.json` records its engine commit. One GH200,
greedy decoding, concurrency 1 except the P9 runs at 8 and 16. Code is in
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

**Cross-session caveat, now checked.** The FlashInfer-with-states column comes from the Stage A
session (00:04-00:41 UTC), the other two from this session. The draft phase, which neither change
touches, is 1.8-2.6% shorter in this session (2.335 and 2.339 against 2.397 ms at B = 16, 3.148
and 3.131 against 3.207 ms at B = 256), so differences of a few percent between the sessions are
within drift. The same-session control below confirms every value in the table to within 1.5%.

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
(5.3%, 8.1% and 9.1% of V(B); at B = 16 the difference is within the cross-session drift;
within one session, 0.18 and 3.98 ms at B = 16 and 256, below), close to their bytes at 3 TB/s
(0.27, 1.07, 4.29 ms). The larger cost is the FlashInfer verify
kernel itself: selecting SGLang's Triton GDN verify kernel instead
(`--linear-attn-verify-backend triton`, a stock option) makes the whole pass 2.06x faster at
B = 64 and 2.34x faster at B = 256; the kernel trace below attributes 72% of the FlashInfer
pass at B = 256 to the verify kernel itself. **Exactness: pending.** The Triton
verify kernel is a different numerical path: stock DFlash at block 16 with it commits 7.603 tokens per cycle
against 7.675 with the FlashInfer kernel, so the greedy outputs differ, and every Triton-based
figure here (the speed ratios, the baseline below, the Triton Stage A table) is pending an
exactness classification of that kernel switch against plain decoding. That baseline runs
7.302 ms per cycle (1,041 tokens/s pooled, against 1,029 with FlashInfer in the Stage A
session): single runs from two sessions, a 2.1% shorter cycle and 0.9% fewer tokens per cycle,
within plausible run-to-run and cross-session variance, so the baseline gain is not
established; the wide blocks gain 2x.

**Same-session control.** session_x2 reran all three variants at B = 16 and 256 one after
another in one exclusive hold (`verify_control.json`, `runs/verify_control.sh`, 2026-10-01
16:39-16:54 UTC; same panel, flags and engine build; foreign CPU load at most 0.64 cores):

| B | V(B), FlashInfer kernel | same, per-position states dropped | V(B), Triton kernel |
|---|---|---|---|
| 16 | 4.78 | 4.60 | 4.57 |
| 256 | 44.70 | 40.72 | 19.19 |

(ms, forced full acceptance, medians; p10-p90 within 1.1% of the median.) Every
cross-session value in the first table of this section is within 1.5% of its same-session
value (the largest gap is the B = 16 no-state run, 4.53 against 4.60 ms), so the conclusions
above stand. Within one session the per-position state writes cost 0.18 ms at B = 16 (3.8% of
V(16)) and 3.98 ms at B = 256 (8.9%), and the Triton kernel makes the pass 4.3% shorter at
B = 16 and 2.33x faster at B = 256.

**Kernel trace at B = 256.** `verify_kernels_b256.json` (`runs/verify_nsys.sh`, session_x2,
16:54-16:59 UTC): Nsight Systems over two forced-acceptance requests of 2,048 tokens with the
FlashInfer verify kernel. The profiler stretches the verify phase from 44.70 to 45.18 ms. The
window holds 19 complete verify cycles (456 launches of the GDN verify kernel over 24 GDN
layers, each cycle closed by its state commit) and a trailing cycle that `nsys stop` cut off
after 18 verify launches, whose 34.9 ms of kernel time is left out. All kernel time up to the
end of the last state commit, divided by the 19 cycles (the draft, prefill and warm-up kernels
in that span are spread over them), is 47.5 ms per cycle:

| category | ms per verify cycle | share |
|---|---|---|
| GDN verify kernel (`gdn_verify_kernel_mtp_inline`, 24 launches per cycle, 1.36 ms each) | 32.68 | 68.9% |
| GDN causal conv update (`_causal_conv1d_update_kernel`, 24 launches per cycle, 0.27 ms each) | 6.48 | 13.6% |
| GEMMs (target and drafter) | 5.87 | 12.4% |
| other GDN kernels, attention, normalization, head argmax, state commit and the rest | 2.43 | 5.1% |

The verify kernel takes 1.36 ms per layer for 256 positions, 5.3 us per position per layer, which
fits a walk over the positions one after another; it is 72% of the verify phase under the profiler
(32.68 of 45.18 ms) and is what the Triton kernel replaces. The conv update kernel, 0.27 ms per
layer, comes next. Derived, not measured: if the Triton path leaves the conv, GEMM and other
kernels unchanged, they take about 12.4 ms of the 44.70 ms pass without the profiler (45.18 -
32.68 ms, scaled by 44.70 / 45.18), so the Triton GDN kernel would take about 6.8 ms of its
19.19 ms pass and the conv update about a third of it. A trace of the Triton pass would settle
this.

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
  tested here do not produce them (one-step recycling and exact Jacobi sweeps in the engine
  below, about one token per exact pass; Stage B). P3 stays refuted by Stage B.

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
positions (a padded verify; see the scope note below the table). P9's protocol first picks the
fastest valid fresh DFlash among blocks 4, 8 and 16. At c = 1 that is block 16: in the bench's slot
T3 (serving workload, radix cache off, Triton attention) block 16 serves 851 tokens/s against 807
for block 8 and 498 for block 4 (`evidence/bench/README.md`). At c = 8, blocks 16 and 8 are within
1% (3,639 and 3,613), and from c = 32 block 8 or 4 is faster, so a P9 test at higher concurrency
may need the support remeasured at block 8.

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
(+1.75, +2.05) at 4, +2.89 (+2.71, +3.10) at 8 and +3.86 (+3.64, +4.11) at 16. These bound from
above P9's always-reuse program at any verify width whose non-verify phases cost at least their
block-16 values, and none of the tested K would be rejected. A narrower cycle can also cut those
phases (`fresh_b8` keeps 0.384 ms after draft and verify, `fresh_b16` 0.395 ms), so this is not a
bound for every implementation. On the point estimates (no intervals), top-1, top-2 and top-4 stay
negative only while V_R exceeds 0.89, 2.71 and 3.87 ms. The only measured verify below width 16 at
c = 1 was 4.16 ms at width 8 (`fresh_b8`, same session as `fresh_b16`). The sweep over widths 2
to 16 in the next section settles them: the narrowest verify costs 3.83 ms, and charged at their
measured widths the reused cycles keep top-1, top-2 and top-4 rejected. The top-8 and top-16
verdicts (not rejected) hold for either implementation.

Omniscient-gate oracle (`omniscient_gate_oracle` in the JSON; an oracle, since its gate knows fresh
DFlash's next accepted length). A gated program may choose fresh DFlash on supported boundaries
where reuse is unfavourable, which the always-reuse table does not allow. Taking, per supported
boundary, the better of oracle reuse and fresh drafting gives a Delta that is >= 0 and >= the
always-reuse Delta by construction, so it rejects nothing. It bounds from above P9's program with
any gate over the same candidate sets with a padded verify, and its Delta is the upper bound to
quote for them (same resamples as the table, so the intervals are paired). The last column, its
excess over the always-reuse oracle, is only the gain from gating the oracle's reuse arm: a real
program's reuse arm is no better than the oracle's, so gating it can gain more, and the column does
not bound that gain.

| K | gate reuses at | Delta (95% CI) | gain from gating the oracle reuse arm |
|---|---|---|---|
| 1 | 0.7% | +0.01 (+0.01, +0.01) | +0.05 |
| 2 | 24.9% | +0.45 (+0.42, +0.49) | +1.12 |
| 4 | 41.6% | +0.91 (+0.85, +0.98) | +1.33 |
| 8 | 53.9% | +1.44 (+1.35, +1.53) | +1.21 |
| 16 | 62.9% | +2.05 (+1.93, +2.18) | +1.06 |

The same gate applied to the free-verify scoring (`omniscient_gate_free_verify`, same resamples)
bounds P9's program with any gate, at any verify width whose non-verify phases cost at least their
block-16 values: +0.04 (+0.03, +0.05) at K = 1, +1.47 (+1.35, +1.61) at 2, +2.54 (+2.35, +2.76) at
4, +3.47 (+3.24, +3.73) at 8 and +4.35 (+4.09, +4.64) at 16.

Gating the oracle's reuse arm adds 1.06 to 1.33 tokens per post-rejection boundary at the tested K
from 2 to 16 (2, 4, 8 and 16). A real gate decides before the second cycle and does not see fresh
DFlash's outcome, so it pays off only if features known at decision time (the correction position J,
the remaining horizon m, the old sets' probabilities at the remaining positions) predict the
boundaries where its own reuse arm loses to fresh drafting. The supported suffix length U_K - J is
not one of them: it needs the greedy tokens after the correction, so it is an oracle target that
such features might predict.

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
  starts at the correction (suffix-only) is not covered. The next section prices the remainder
  verify, the reuse program and concurrency 8 and 16.

```sh
python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
    --timing evidence/repair/stage_a_timing.json --width-timing evidence/repair/p9_verify_widths.json \
    --bootstrap 2000 --out evidence/repair/p9_support_oracle.json
```

The `--width-timing` fields come from the next section's sweep; every other field is unchanged
from the run without it (the bootstrap draws the same resamples), apart from `kind` and the
added `baseline_run`, `draft_share_of_cycle` and phase fields.

### P9 costs: the remainder verify, the reuse program and concurrency 8 and 16

Session `session_x2` under the exclusive lock, 2026-10-01 16:13-16:59 UTC: `runs/p9_draft_share.sh`
and `runs/p9_verify_widths.sh` (`experiments/repair/`, with `p9_program_cost.py`) at repository
commit `d365673`, engine build `5d8e00e3e1`, the configuration of the Stage A runs. Every number in this section uses FlashInfer's GDN verify kernel, as the P9 baseline
`fresh_b16` does (`--linear-attn-decode-backend flashinfer`, the DFlash model card's setting);
SGLang's default on sm_90 is the Triton verify kernel, which the bench's tuned DFlash arms use.
Foreign CPU load averaged at most 0.8 cores in every run (`foreign_cpu_during` in each run's
`run.json`, kept with the raw data, and in the c > 1 rows of `p9_draft_share.json`).

**The remainder verify (c = 1).** After a correction at block position J, m = 15 - J old
positions remain, so a program could verify m + 1 positions instead of a padded block of 16.
Forced full acceptance at widths B = 2 to 16 (`p9_verify_widths.json`; four MATH-500 requests of
512 tokens per width, one server per width, c = 1) gives the verify phase (medians; p10-p90 within
1.6% of the median at every width):

| B | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V(B) ms | 3.83 | 3.86 | 3.95 | 3.98 | 4.08 | 4.08 | 4.20 | 4.38 | 4.42 | 4.45 | 4.52 | 4.57 | 4.65 | 4.76 | 4.78 |

Verifying two positions costs 80% of verifying sixteen; each further position adds about 68 us.
V(16) equals the Stage A session's 4.78 ms. Charging every reused cycle the verify of its own
width (`measured_width_verify` in `p9_support_oracle.json`: the reused cycle saves this session's
V(16) - V(m + 1), every other phase stays at its `fresh_b16` value; same bootstrap and resamples
as the first table) saves 0.30 ms of verify on the average reused cycle, because most corrections
come early and leave a wide remainder:

| K | padded verify (first table) | measured m + 1 verify | rule | measured, omniscient gate |
|---|---|---|---|---|
| 1 | -0.04 (-0.06, -0.03) | -0.04 (-0.05, -0.03) | rejected | +0.01 (+0.01, +0.01) |
| 2 | -0.66 (-0.81, -0.55) | -0.57 (-0.69, -0.46) | rejected | +0.49 (+0.45, +0.54) |
| 4 | -0.42 (-0.56, -0.30) | -0.27 (-0.39, -0.17) | rejected | +0.98 (+0.91, +1.06) |
| 8 | +0.22 (+0.12, +0.32) | +0.39 (+0.31, +0.47) | not rejected | +1.53 (+1.43, +1.64) |
| 16 | +0.99 (+0.89, +1.07) | +1.17 (+1.09, +1.26) | not rejected | +2.16 (+2.03, +2.30) |

(Delta in tokens per post-rejection boundary, 95% intervals.) For P9's always-reuse program
verifying m + 1 positions with SGLang's verify pass, the measured widths keep top-1, top-2 and
top-4 rejected; the narrower cycle's other phases would have to fall by about 0.35 ms more than
their block-16 values to lift top-4's upper bound to zero, against the 11 us that `fresh_b8`
saves over `fresh_b16`. The Triton verify kernel's widths were not swept.

**The reuse program.** `p9_program_cost.json` (`p9_program_cost.py`, run in the same hold after
the draft-share runs): the backward messages, the forward message over the corrected prefix and
the greedy walk of a fixed-shape program over top-16 sets (H = 15), with random values, at every
correction slot. Captured as one CUDA graph it takes 133-315 us per reuse (longest when the
correction comes first and the walk covers 14 slots), almost independent of the rank (2, 4, 8)
and of the batch (1, 8, 16 requests at once); run eagerly, 3.2-5.8 ms, longer than the draft it
would replace. It does not include building the matrices from the drafter's state. At c = 1 the
graph costs 6-14% of the 2.33 ms draft; charged on every reuse it lowers Delta by r_F p_K A,
about 0.2 tokens at K = 16 for A = 0.3 ms.

**Concurrency 8 and 16.** `p9_draft_share.json` (`runs/p9_draft_share.sh`): fresh DFlash at
block 16 kept at 8 and 16 requests in flight (96 and 192 decode checkpoints of the drafter's
shared panel, 512 new tokens each, natural stop), phases over the cycles that ran at the full
batch (463 and 252 cycles):

| c | draft ms | verify ms | commit ms | cycle ms | draft share of the cycle |
|---|---|---|---|---|---|
| 1 (`fresh_b16`, Stage A session) | 2.33 | 4.73 | 0.11 | 7.46 | 31% |
| 8 | 2.84 | 7.88 | 0.38 | 11.45 | 25% |
| 16 | 3.28 | 12.09 | 0.68 | 16.46 | 20% |

(Cycle p10-p90 within 1.5% of the median.) The batched draft grows by 73 us per added request from
1 to 8 and by 55 us from 8 to 16. At c > 1 one reusing request does not skip the batch's draft;
the batch drafts one row fewer, and the shorter batch period s serves every request, so a reuse
is worth c r_F s tokens (r_F per request). The oracle at c (`p9_support_oracle_c8.json`,
`p9_support_oracle_c16.json`) takes acceptance from the c = 1 support table, not remeasured at c,
and the phases from the c-run, and is scored two ways. With the request's even share of the
draft, s = draft(c) / c, it bounds P9 from above when the draft's cost is concave in the number
of rows, as the measured growth suggests (2.33 ms for the first request, then 73 and 55 us per
added request). With s set to that growth per added request (`_marginal` files: 73 us at c = 8,
55 us at c = 16), which bounds a single reusing request's s from above under the same condition,
it bounds what one request reusing on its own can gain:

| K | c = 8, even share | c = 8, per-request growth | c = 16, even share | c = 16, per-request growth |
|---|---|---|---|---|
| 1 | -0.05 (-0.06, -0.03) | -0.06 (-0.08, -0.05) | -0.05 (-0.07, -0.04) | -0.06 (-0.08, -0.05) |
| 2 | -0.82 (-0.98, -0.69) | -1.30 (-1.53, -1.12) | -0.94 (-1.12, -0.80) | -1.30 (-1.52, -1.12) |
| 4 | -0.65 (-0.81, -0.51) | -1.36 (-1.60, -1.16) | -0.83 (-1.01, -0.67) | -1.36 (-1.59, -1.15) |
| 8 | -0.05 (-0.17, +0.07) | -0.87 (-1.07, -0.70) | -0.25 (-0.39, -0.12) | -0.86 (-1.06, -0.69) |
| 16 | +0.70 (+0.59, +0.79) | -0.19 (-0.36, -0.05) | +0.48 (+0.36, +0.58) | -0.18 (-0.35, -0.04) |

(Always-reuse Delta, padded block-16 verify, 95% intervals. The omniscient-gate bounds with the
even share are +0.01, +0.37, +0.77, +1.26 and +1.84 at c = 8 and +0.01, +0.31, +0.67, +1.13 and
+1.69 at c = 16 for K = 1 to 16; with the per-request growth, +0.00, +0.17, +0.42, +0.78 and
+1.27 at c = 8 and the same to two decimals at c = 16, apart from +0.79 at K = 8. The JSON's
free-verify fields at c > 1 also credit the request's even share of the batched verify and are
not discussed here.) A real batch saves less still:
SGLang replays the draft from CUDA graphs captured at fixed batch sizes, so a batch that drafts
one row fewer may pay for the full graph. And one reuse step of the program (133-315 us) costs
more GPU time than the 55-73 us draft row it removes.

- **Verdict for P9 (oracle, upper bounds under the stated scopes).** At c = 1, with the measured
  remainder verify, reuse over top-16 and top-8 sets is not rejected (+1.17 and +0.39 tokens per
  post-rejection boundary) and top-1, top-2 and top-4 are rejected for always-reuse programs.
  Fresh DFlash commits about 10.2 tokens over the two cycles around a boundary, so the best
  always-reuse oracle gains about a tenth on them, and the omniscient gate about a fifth
  (+2.16), before the program, conditioning and retention costs and with the true token picked
  whenever the old set holds it. At c = 8 and 16 the even share keeps top-16 positive (+0.70
  and +0.48), and top-8 is not rejected at c = 8 only. Charging one reusing request the measured
  growth of the batched draft instead, always-reuse is rejected at every K, top-16 included
  (-0.19 and -0.18), and the reuse program alone costs more GPU time than the draft row it
  removes; the omniscient gate still bounds a gated program at +1.27 for top-16. Beyond c = 1,
  reuse would have to remove whole draft passes, that is, many requests of a batch reusing in the
  same cycle, which this oracle does not measure.

```sh
scripts/gpu_lock.sh -x bash -c 'for s in p9_draft_share p9_verify_widths verify_control verify_nsys; do
    experiments/repair/runs/$s.sh; done'   # raw runs in ~/vp-data/repair/runs/{p9share,p9width,control1,nsys2}
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/p9share/fresh_b16_c{8,16} \
    --out evidence/repair/p9_draft_share.json
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/p9width/force_b{2..16} \
    --out evidence/repair/p9_verify_widths.json
cp ~/vp-data/repair/runs/p9share/program_cost.json evidence/repair/p9_program_cost.json
for C in 8 16; do
  python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
      --timing evidence/repair/p9_draft_share.json --baseline-run fresh_b16_c$C \
      --bootstrap 2000 --out evidence/repair/p9_support_oracle_c$C.json
done
python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
    --timing evidence/repair/p9_draft_share.json --baseline-run fresh_b16_c8 --draft-saving-us 73.09 \
    --bootstrap 2000 --out evidence/repair/p9_support_oracle_c8_marginal.json
python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
    --timing evidence/repair/p9_draft_share.json --baseline-run fresh_b16_c16 --draft-saving-us 55.01 \
    --bootstrap 2000 --out evidence/repair/p9_support_oracle_c16_marginal.json
```

The hold ran the four scripts in that order (the verify control and kernel trace are in the
verify section above). The draft growth per added request is (2839.30 - 2327.65) / 7 = 73.09 us from `fresh_b16`
(`stage_a_timing.json`) to `fresh_b16_c8`, and (3279.38 - 2839.30) / 8 = 55.01 us from c = 8 to
c = 16 (draft medians).

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

### P2 Arm B in the engine: exact Jacobi sweeps from real DFlash windows (B = 16, 32)

`jacobi_probe_progress.csv`, `jacobi_probe_summary.json` (measured). Session `runs/jacobi_probes.sh`
under the exclusive lock on 2026-10-01, 21:53-22:05 UTC (exclusive for memory; nothing here is
timed), repository commit `17bbeac` (on main; each run's `run.json` records it), engine build
`5d8e00e3e1`. The probe follows the stock DFlash trajectory and, on
every block, runs four extra exact target passes from the same committed prefix of each kind: recycle
sweeps (the Jacobi map: the next candidate takes the target's predictions from the previous pass,
shifted by one) and keep sweeps (only the first mismatching draft token is replaced by the
target's token). The plain DFlash pass is then rerun and committed, so the trajectory stays stock
DFlash. Inputs: every second decode checkpoint of the drafter workstream's shared panel
(`checkpoints_probe_half.jsonl`, 67 requests, SHA-256
`5bb26409670cf3be159812c6003d3329781f04e5f7d575cfd50a51f4038b18b7`, raw data outside git), 384
new tokens each, greedy, one request at a time, FlashInfer GDN kernels. Each rerun from a prefix
restores the batch's conv and SSM states; rerunning the original draft reproduced the first
pass's argmax in all 4,085 (B = 16) and 3,968 (B = 32) cycles (`replay_mismatch_cycles` 0).

Accepted drafts after r exact sweeps, with committed tokens per exact target pass, (a_r + 1) / (r + 1):

| B | sweeps r | recycle: accepted | per pass | keep: accepted | per pass | blocks fully accepted (keep) |
|---|---|---|---|---|---|---|
| 16 | 0 (DFlash) | 5.49 | 6.49 | 5.49 | 6.49 | 14.7% |
| 16 | 1 | 6.72 | 3.86 | 7.08 | 4.04 | 18.6% |
| 16 | 2 | 7.66 | 2.89 | 8.16 | 3.05 | 21.6% |
| 16 | 3 | 8.53 | 2.38 | 9.08 | 2.52 | 24.7% |
| 16 | 4 | 9.37 | 2.07 | 9.93 | 2.19 | 27.9% |
| 32 | 0 (DFlash) | 5.69 | 6.69 | 5.69 | 6.69 | 0.05% |
| 32 | 1 | 7.18 | 4.09 | 7.59 | 4.30 | 0.2% |
| 32 | 2 | 8.33 | 3.11 | 8.94 | 3.31 | 0.4% |
| 32 | 3 | 9.41 | 2.60 | 10.12 | 2.78 | 0.7% |
| 32 | 4 | 10.48 | 2.30 | 11.25 | 2.45 | 1.1% |

Each exact sweep adds 0.84-1.58 accepted tokens at B = 16 and 1.07-1.90 at B = 32, less with
every sweep, which is what the Hugging Face replay of Stage B found (about one token per exact
pass). The mechanism shows in the target's own predictions: after one corrected token, the
prediction right after it changes in 87% of the cases (86% at B = 32), the next one in 49%, about
a quarter four positions on and about a seventh eight positions on (`one_correction` in the
JSON), so a correction invalidates many of the downstream guesses that a sweep would have to
confirm. Even P3's ceiling, which charges the sweeps nothing and pays
only an anchor and an audit pass, commits 5.18 (recycle) or 5.46 (keep) tokens per pass after
four sweeps at B = 16, below DFlash's 6.49, and 5.74 or 6.13 at B = 32 against 6.69
(`committed_per_pass_p3_ceiling` in the CSV). Derived, from Stage A's FlashInfer timings at
c = 1: a sweep is one verify pass (4.78 ms at B = 16), while fresh DFlash commits about 0.87 tokens
per ms (this panel's 6.49 tokens per Stage A's 7.46 ms cycle); the best sweep (the first keep sweep, +1.58) adds 0.33
tokens per ms of its own cost, so every sweep lowers throughput.

- **P2 verdict: long-window exact repair is rejected as a route to 5x.** Repair from a DFlash
  window converges at about one token per exact target pass and leaves 72% of B = 16 blocks and
  99% of B = 32 blocks short of full acceptance after four sweeps, while Stage A needs blocks of
  64 or more tokens accepted almost entirely (per-position acceptance about 0.9985 at B = 64 with
  the Triton verify kernel, `evidence/drafter/drafting_requirement.json`). What remains of P2 is
  a drafting problem, not a repair one.

```sh
scripts/gpu_lock.sh -x experiments/repair/runs/jacobi_probes.sh   # raw traces in ~/vp-data/repair/runs/probe1
python experiments/repair/analyze_jacobi.py ~/vp-data/repair/runs/probe1/probe_b16 \
    ~/vp-data/repair/runs/probe1/probe_b32 --out-dir ~/vp-data/repair/analysis/jacobi
cp ~/vp-data/repair/analysis/jacobi/jacobi_summary.json evidence/repair/jacobi_probe_summary.json
cp ~/vp-data/repair/analysis/jacobi/jacobi_progress.csv evidence/repair/jacobi_probe_progress.csv
```

### P12 static screen on the compiled last-FFN dictionary

`p12_static_screen.json` (measured; `p12_static_screen.py`). Sam's proposal P12 (2026-10-01)
compiles the tied head backward through the last decoder layer's FFN, so the greedy winner is an
argmax of t_v^T q over a dictionary t_v = [Gamma w_v; D^T Gamma w_v] with the query q = [x; a],
and asks whether a centre-plus-radius tile bound, mu_C^T q + R_C ||q||, can skip tiles of that
dictionary. The script measures the most favourable version of the screen: the winner's exact
score is taken as known, so its skip rate bounds from above what any screen with this bound can
skip.

Setup: the Hugging Face Qwen3.5-4B target at the pinned revision (BF16 weights, FP32 arithmetic for
the dictionary and the scores) on 1,000 queries, 25 positions from each of 40 outputs of the
drafter workstream's block-16 panel (`drafter_b16_outputs.jsonl`, SHA-256
`679240063371673782ca0fe6b7eeeb241c36bef2f183030bda3dcd9dceaffec4`, raw data outside git); 64-row
tiles in token-id order, and the same tiles after sorting the rows by a random projection (a cheap
clustering control, not k-means). Run under the shared lock on 2026-10-01, finishing at 18:51 UTC,
from a local worktree whose `p12_static_screen.py` and `runs/p12_screen.sh` are byte-identical to
the files on main.

| quantity | value |
|---|---|
| dictionary | 248,320 rows x 11,776 coefficients, 4.6 times the head's 2,560 |
| compiled argmax equals the model's argmax | 99.7% of queries |
| median margin between the top two scores, over \|\|q\|\| | 0.125 |
| rows in skippable tiles, token-id order | 0.077% (192 rows), the same for every query |
| rows in skippable tiles, random-projection order | 0.052% (128 rows), the same for every query |

- **Verdict: the tile screen does not prune the compiled dictionary.** Even knowing the winner's
  score, it skips the same two or three tiles for every query, 0.05-0.08% of the rows. That is
  the share the static l2 screen skips on the head alone with contiguous 64-row tiles (0.08%,
  which `evidence/head_geometry/README.md` traces to unused-token tiles; k-means tiles reach 0.64%
  there), while every remaining row costs 4.6 times a head row. The compiled form reproduces the
  model's argmax at 99.7% of the queries, so the obstruction is the bound: a tile is skipped only
  when its radius times ||q|| is below the winner's lead over the tile centre, and the winner's lead
  over the runner-up is a median 0.125 ||q||. Tighter tilings (k-means, smaller tiles) and per-row
  bounds were not tested on the dictionary.

```sh
scripts/gpu_lock.sh -s experiments/repair/runs/p12_screen.sh   # writes ~/vp-data/repair/p12/p12_static_screen.json
cp ~/vp-data/repair/p12/p12_static_screen.json evidence/repair/p12_static_screen.json
```

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
# same-session control and kernel trace (session_x2, raw runs in ~/vp-data/repair/runs/{control1,nsys2})
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/control1/force_{,nostate_,tritonverify_}b{16,256} \
    --out evidence/repair/verify_control.json
python experiments/repair/nsys_kernels.py ~/vp-data/repair/runs/nsys2/force_b256.nsys-rep \
    --out evidence/repair/verify_kernels_b256.json
```

The decomposition session's two Nsight Systems runs failed at launch (an `nsys launch` option
that only `nsys start` accepts); the Triton-versus-FlashInfer runs answer the attribution
question causally, and session_x2's B = 256 trace (`verify_kernels_b256.json`) gives the kernel
shares.

The ReplaySSM spec protocol does not start with DFlash on this GDN model ("requires a KDA
model"), and the session's GDN kernel microbenchmark was stopped after 16 minutes of CPU-bound
kernel compilation without output. The per-position state writes inside V(B) are therefore
measured (forced acceptance without them, decomposition session) at B = 16, 64 and 256 and
bounded by their bytes at 3.0 TB/s at B = 32 and 128 in `stage_a_oracle.{json,csv}`, and bounded
by their bytes at every width in `stage_a_oracle_triton.{json,csv}`; `state_writes_source`
labels each row.
