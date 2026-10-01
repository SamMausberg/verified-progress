# Long-window repair oracles (P2, P3)

Evidence for the repair workstream's kill tests of Sam's proposals P2 (long-window exact
repair of a DFlash window, 2026-09-30) and P3 (target-anchored residual decoding,
2026-09-30). Setting throughout: Qwen/Qwen3.5-4B
at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter z-lab/Qwen3.5-4B-DFlash at
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus
`engine/sglang/patches/repair/0001-*.patch` (engine commit `101e52731b` on branch
`engine/repair`), one GH200, greedy decoding, concurrency 1. Code is in
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
| 64 | 3.95 / 3.70 | 3.34 / 3.17 | 2.05 / 2.00 | 3.45 / 3.27 |
| 128 | 4.41 / 4.09 | 3.97 / 3.72 | 2.29 / 2.22 | 4.07 / 3.80 |
| 256 | 5.53 / 5.01 | 5.12 / 4.68 | 2.90 / 2.78 | 5.24 / 4.78 |

S_b estimate charges the anchor pass V(B) minus the per-position state writes bounded by their
bytes (12.9 GB at B = 256 at 3.0 TB/s) plus writing the anchor cache (every operator's input
and output at every block position, BF16, 3.95 MB per token). S_b ceiling charges the anchor
pass only one read of the 8.41 GB of weights at the measured 3.83 TB/s read peak
(`evidence/profiles/hbm_bandwidth.json`) plus the anchor cache at that rate, and the audit its
measured V(B) + commit; with no anchor pass at all the two-pass design is S_a.

- **P3 two-pass design: misses the pre-registered target at every width with SGLang's current
  verifier (refuted for this verifier).** The ceiling charges the anchor pass one weight read
  and the audit its measured V(B) + commit, and reaches 4.78x end to end at B = 256. That V(B)
  includes the per-position FP32 state writes: an audit 2.33 ms cheaper (5.2% of V(256),
  `audit_saving_needed_for_target_us`) would reach the target, and the writes' bytes bound them
  at 4.29 ms (9.6%), so with a verifier that drops them and reconstructs only the accepted
  boundary state at no charge the ceiling could reach 5.20x end to end. Such a verifier would
  also make the DFlash baseline cheaper (16 per-position writes per cycle), so a fair rerun gives
  both the same verifier. If both passes dropped the writes, the estimate (anchor pass = V(B)
  without them) would need them to be at least 50% of V(256). The measured no-state verify
  decides the ceiling case (queued). Stage B below refutes P3 independently of the verifier.
- **P2 Arm A: 5x end to end only at B = 256 with zero drafting cost and every block accepted**
  (derived from measured V(B)): S_a = 5.01x end to end sits at the threshold; paying DFlash's
  own drafting cost at width B, perfect blocks reach 4.68x. Better drafting alone does not give
  5x with this verifier.
- The verifier is the binding constraint. Past B = 16, V(B) grows by about 0.167 ms per extra
  token, roughly five times what that token's GEMM work and 48 MiB per-position FP32 state
  write account for; the state commit is 0.11-0.14 ms at every width. Its decomposition
  (no-state verify, Nsight Systems at B = 64 and 256) is queued.

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
  ||W (I - U U^T) dx|| / ||W dx|| over operator inputs is 0.76-0.90 at rank 16, 0.64-0.81 at
  rank 128 and 0.51-0.73 at rank 512 (by operator class); on the development blocks themselves
  rank 128 captures a median 51-64% of the change energy. The changes caused by token
  corrections are not confined to a small fixed subspace.
- (2) Decisions: on replay 1 the repaired argmax equals the exact argmax at 35% of the changed
  positions with rank 0 (anchor outputs reused unchanged), 37% at rank 128 and 44% at rank 512;
  at the corrected position itself 11-14%; on replay 2, 24-35%. The certificate ratio
  R_i = 2 ||z~ - z||_inf / m_i is below 1 at 4.7% of positions (43 of 962), the same 43 positions
  at every rank including rank 0: positions whose exact top-two margin is large (median 12
  logits, against 1.3 over all replayed positions), where even reusing the anchor logits stays
  within half the margin. These certificates are not earned by the repair; the median R_i is
  13-16. Positions before the first change agree exactly in every block (harness check). The
  corrected position's decision is right in 9 of 80 blocks at ranks 0-256 and 11 at rank 512.
- (3) Progress: free-running repair from y1, audited exactly after every sweep, accepts 4.33
  drafts after 0 sweeps and 4.33-4.44 after 4 sweeps at every rank, while exact Jacobi from the
  same anchor accepts 2.98 (y0), 4.33, 5.35, 6.38, 7.34 and 8.30 after 0-5 exact passes: about one
  token per exact sweep, which also bounds any faithful cheap evaluator.
- Economic gate (Sam's mu_R / C_R > mu_0 / C_0 against DFlash-16; repair cost a bytes-only lower
  bound, audit = measured V(16) + commit): the anchored evaluator reaches at most 0.28 of
  DFlash's committed tokens per unit cost at any rank and sweep count; exact Jacobi with sweeps
  charged only their anchor and state reads reaches 0.27, 0.47, 0.66, 0.85 and 1.03 after 0-4
  sweeps. That exact-Jacobi line charges each sweep only the bytes of reading the anchor cache
  and the GDN state, so even a perfect cheap evaluator would at best match DFlash after four
  sweeps. The rigorous bound (audit cost over the most tokens an attempt can add) does not reject
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
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/timing1/force_b* \
    ~/vp-data/repair/runs/timing1/fresh_b* --out evidence/repair/stage_a_timing.json
python experiments/repair/stage_a.py --timing evidence/repair/stage_a_timing.json \
    --baseline fresh_b16 --state-bytes-bound --out-dir evidence/repair
```

The ReplaySSM spec protocol does not start with DFlash on this GDN model ("requires a KDA
model"), and the session's GDN kernel microbenchmark was stopped after 16 minutes of CPU-bound
kernel compilation without output, so this table bounds the per-position state writes inside
V(B) by their bytes at 3.0 TB/s (labelled `state_writes_source`); measured no-state verify
times replace the bound when the decomposition session lands.
