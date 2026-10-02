# Lossy levers: evidence

Results of the study described and pre-registered in `experiments/lossy/README.md`: the INT4
quantization-aware-distilled target with its INT4 DFlash drafter, and the FP16 GDN state, against
bench's tuned exact arms. Both levers change the model's arithmetic. They are declared
approximations, not exact configurations, and each is charged against the quality budget declared
before any lossy measurement (`evidence/moonshot/README.md`, "Quality budget for the lossy stack").
Hardware, engine and client are as in `evidence/bench/README.md`; the engine is `57560de690`
(`bd66ce343e` + `engine/sglang/patches/lossy/0001`) for every arm, exact or lossy. Raw run
directories stay in `~/vp-data/lossy/` on the GH200 host.

Status: complete. Load test, checkpoint check, three timed sessions (`lossy-s1` to `-s3`), three
quality holds (`q1` to `q3`; the GSM8K run in `q1` failed, see below) and the exploratory GEMM
microbenchmark.

## Verdicts under the declared rules

| Lever | Speed (declared decision, three sessions) | Quality against the declared budget | Within budget |
|---|---|---|---|
| INT4 target + INT4 DFlash drafter | x at c = 1: 1.015 (1.014-1.016), no detectable change (y 1.046, faster). y slower at every c from 2 to 256: 0.72-0.87 against the best exact arm | Logit probe fails in both modes on both probed arms (top-1 agreement 0.969-0.975 against the 0.98 floor, KL 0.0120-0.0125 nats against the 0.01 ceiling). GSM8K: the declared run failed (time limit at 1,305 of 1,319 problems) | No (probe) |
| FP16 GDN state | y faster at c = 64, 128, 256: 1.163, 1.158, 1.171 against the best exact arm (`replayssm-cap256`), every session beyond +2% | Probe passes in both modes at about the reference's own noise. GSM8K misses the rule (at least -1.0 point against both references): -0.99 and -1.29 points (`plain-cap256-fp16`), -1.06 and -1.36 (`replayssm-cap256-fp16`); 95% intervals from -2.9 to -3.3 at their lower ends to +0.5 to +1.0 at their upper ends, McNemar p 0.18-0.36 | No (GSM8K) |

Both verdicts are measured outcomes of the declared rules. As the pre-registration anticipated, the
GSM8K part cannot separate a 1-point loss from noise: bench's exact arms span -1.36 to +1.06
points against the same two references, and `replayssm-cap256-fp16`'s -1.06 and -1.36 equal the
exact `mtp-stockverify`'s. So the FP16 state is outside the declared band, and its cost is the
trade stated below: about a point of GSM8K that this check cannot distinguish from run-to-run
variation, a small and measured change in the logits, for 16-17% more throughput at c >= 64. The
INT4 lever is outside the band on the probe alone, and on this GPU it is slower at every
concurrency above 1 and shows no detectable change in per-user speed at c = 1.

What is measured and what is derived. Measured: every ratio, accept length, probe statistic and
GSM8K accuracy below, each on the runs named. Derived: the full-split bounds of the failed INT4
GSM8K run (arithmetic on the finished problems), the per-pass quantities (y per accepted token;
the slow-launch check's ITL p50 times accept length), and the explanation of INT4's slowdown from the GEMM microbenchmark, which times the quantized
projections in isolation and is not an end-to-end result.

## Failures and deviations

- `q1`'s GSM8K run of `int4-dflash-b8` stopped at its 1,500 s limit (exit 124) with 1,305 of
  1,319 problems scored, so it wrote no `quality.json` and has no declared result. The limit was
  too short for this arm; the server did not fail: the INT4 drafted arm decodes at 0.75-0.77 of stock DFlash
  at c = 8-32 in the timed sessions, stock DFlash's GSM8K run took 933 s, and the INT4 run spent
  about 80 s launching and 1,419 s on 1,305 problems. Not rerun: the arm is already outside its
  band on the declared probe, so a GSM8K number cannot change its verdict.
- `quality/int4-dflash-b8-seed0-partial/` (not declared) reports what the finished problems show,
  and bounds that hold whatever the 14 unfinished problems would have scored.
- `quality_speed.png` was redrawn after the holds (commit `c25b2bc`): the earlier version shifted
  each reference's point by 0.12 points to separate them.
- No timed point was invalid, no launch failed a check, and the slow-launch check flagged nothing,
  so no decision has a second verdict.

## sessions/ (points.csv, launches.csv)

```sh
python -m bench.pareto ~/vp-data/lossy/sessions/*/2026* --out evidence/lossy/sessions --points-only
```

`hold.sh session lossy-s1` (05:08-05:46 UTC), `lossy-s2` (06:59-07:37, reversed order) and
`lossy-s3` (08:00-08:38), all on 2026-10-02 from repository `5f149d2`; the hold manifests are in
`holds/`. Validity (measured): 102 points, 34 per session, all valid under bench's rules (no failed
request, exact output lengths, aiperf exit 0, cache flushed, prompts as expected); mean foreign CPU
at most 0.58 cores at any point (maximum 1.54); all 30 launches passed every required launch check
(decode and prefill CUDA graphs covering the capacity, overlap scheduler, capacity, attention
backend, speculative settings), each on engine `57560de690` with a clean worktree.

Memory at the same `--mem-fraction-static` (from the timed launches' server logs, not timed; INT4 /
BF16 where they differ). The GDN state depends only on capacity and dtype; its sizes come from the
load test.

| Arms | Capacity | Weights | KV pool | KV tokens | GDN state |
|---|---|---|---|---|---|
| `int4-dflash-b16` / `dflash-tuned-b16` | 64 | 3.90 / 8.62 GB | 13.69 / 9.40 GB | 448,589 / 307,926 | 3.05 GB |
| `int4-dflash-b8` / `dflash-tuned` | 128 | 3.90 / 8.62 GB | 12.04 / 7.86 GB | 394,468 / 257,637 | 6.05 GB |
| `int4-plain-cap256` / `plain-cap256` | 256 | 3.90 / 8.62 GB | 30.52 GB | 1,000,000 (set) | 12.05 GB |
| `plain-cap256-fp16`, `replayssm-cap256-fp16` | 256 | 8.62 GB | 30.52 GB | 1,000,000 (set) | 6.02 GB (FP16) |

## decisions.json, ratios.csv

```sh
python -m experiments.lossy.analyze --points evidence/lossy/sessions/points.csv \
  --quality evidence/lossy/quality/plain-cap256-fp16-seed0/20261002-094130 \
            evidence/lossy/quality/replayssm-cap256-fp16-seed0/20261002-110013 \
  --references evidence/bench/quality/plain-tuned-a-seed0 evidence/bench/quality/plain-tuned-b-seed0 \
  --probes ~/vp-data/lossy/probes/int4-plain-cap256 ~/vp-data/lossy/probes/int4-dflash-b8 \
           ~/vp-data/lossy/probes/plain-cap256-fp16 ~/vp-data/lossy/probes/replayssm-cap256-fp16 \
  --probe-noise ~/vp-data/lossy/load_test/20261002T035539Z --out evidence/lossy
```

Run from commit `08c4357`; `analyze.py` and `plan.py` are unchanged since `ec91737`, the commit
the quality holds ran from. `decisions.json` holds the session means, matched pairs, envelope
ratios with runner-ups, the slow-launch check, GSM8K comparisons, the exact arms' GSM8K spread,
probe statistics and the probe's own noise; `ratios.csv` is the table of every ratio with its
decision. The probe inputs are raw files (6.6 MB each) kept outside git; `probes/` has their
summaries and SHA-256s.

Matched pairs (identical flags apart from the lever; mean of the three session ratios, range in
brackets; a decision needs all three beyond +-2%):

| Lossy arm / exact baseline | c | y | x |
|---|---|---|---|
| `int4-dflash-b16` / `dflash-tuned-b16` | 1 | 1.046 (1.045-1.046) faster | 1.015 (1.014-1.016) no detectable change |
| | 2 | 0.870 (0.870-0.871) slower | 0.838 slower |
| | 4 | 0.753 (0.751-0.754) slower | 0.734 slower |
| `int4-dflash-b8` / `dflash-tuned` | 8 | 0.757 (0.749-0.764) slower | 0.742 slower |
| | 16 | 0.774 (0.762-0.783) slower | 0.769 slower |
| | 32 | 0.754 (0.751-0.757) slower | 0.752 slower |
| `int4-plain-cap256` / `plain-cap256` | 64 | 0.782 (0.781-0.784) slower | 0.782 slower |
| | 128 | 0.840 (0.839-0.841) slower | 0.840 slower |
| | 256 | 0.812 (0.802-0.817) slower | 0.817 slower |
| `plain-cap256-fp16` / `plain-cap256` | 64 | 1.169 (1.168-1.171) faster | 1.169 faster |
| | 128 | 1.250 (1.248-1.252) faster | 1.250 faster |
| | 256 | 1.308 (1.306-1.309) faster | 1.308 faster |
| `replayssm-cap256-fp16` / `replayssm-cap256` | 64 | 1.098 (1.091-1.101) faster | 1.098 faster |
| | 128 | 1.132 (1.127-1.136) faster | 1.132 faster |
| | 256 | 1.171 (1.169-1.171) faster | 1.171 faster |

Envelope ratios (the lever's best arm over the best exact arm, each by mean y; headline points in
bold). INT4: **x 1.015 at c = 1** (no detectable change; y 1.046, faster), y 0.870 and 0.753 at
c = 2 and 4, **0.757 at c = 8**, 0.774 at 16, **0.754 at 32** (`int4-dflash-b8` over
`dflash-tuned`), 0.778, 0.778 and 0.715 at c = 64, 128, 256 (`int4-plain-cap256` over
`replayssm-cap256`), all slower. FP16 state: 1.163 at c = 64 and **1.158 at 128**
(`plain-cap256-fp16`), **1.171 at 256** (`replayssm-cap256-fp16`), all faster; against the
exact runner-up `plain-cap256` the same arms give 1.169, 1.250 and 1.330, and the lossy runner-ups
give 1.098, 1.132 and 1.151. Both FP16 arms have their own GSM8K run, so no FP16 row is
provisional. `plain-tuned` (capacity 128) ran at c = 64 and 128 in every session and ranks below
`replayssm-cap256` there, so the larger capacity does not inflate any ratio.
`replayssm-cap256` is bench's buffered plain decoding (`plain-tuned-replayssm`) at capacity 256;
`bench/arms.toml` classes it exact-up-to-rounding, from bench's equality report.

Accept length and time per verify pass (session means; the pass time is ITL p50 times accept
length, the slow-launch check's definition, so it is derived):

| Arms | c | Accept length, INT4 / BF16 | Time per pass, INT4 / BF16 |
|---|---|---|---|
| `int4-dflash-b16` / `dflash-tuned-b16` | 1 | 5.05 / 5.70 | 4.43 / 5.09 ms |
| | 2 | 5.03 / 5.69 | 5.97 / 5.56 ms |
| | 4 | 5.01 / 5.68 | 8.67 / 7.03 ms |
| `int4-dflash-b8` / `dflash-tuned` | 8 | 4.05 / 4.73 | 9.47 / 8.30 ms |
| | 16 | 3.95 / 4.67 | 12.88 / 11.72 ms |
| | 32 | 3.96 / 4.73 | 21.17 / 18.88 ms |

The INT4 drafter accepts 0.65-0.77 fewer tokens per verify pass at either block size (11-16%
fewer). At c = 1 the INT4 pass takes 0.87 of the BF16 time, which about cancels the lower
acceptance (x 1.015). From c = 2 the INT4 pass is itself slower (1.07 times at c = 2, 1.23 at
c = 4, 1.10-1.14 at c = 8-32), and the lower acceptance adds to that. y at c = 1 is 1.046: y
divides all output tokens by the session's span while x averages per-request rates, so the two
weight requests differently; x is the declared headline at c = 1.

Slow-launch check: no launch flagged. The largest TTFT p50 excess over an arm's median was 6.7 ms,
the largest per-pass excess 2.0%, never both beyond their thresholds at one point.

### Why INT4 is slower (derived, from the exploratory microbenchmark)

`gemm_w4a16.json` times the six quantized backbone projections of the target per forward pass,
Marlin W4A16 against BF16, under CUDA graphs with cold L2. W4A16 takes 0.69 of the BF16 time at
1-8 rows, 0.75 at 16, then 1.19 at 32, 1.76 at 64, 1.64 at 128 and 1.99 at 256 rows; the INT4
drafter crosses over at 32 rows too (1.01). A verify pass carries concurrency times the block size
in rows (16 for `-b16`, 8 for `-b8`): 16 rows at c = 1, 32 at c = 2, and 64 or more at every
other drafted point; plain decoding at c >= 64 has 64 or more rows. So every timed point except
c = 1 sits past the crossover, where the INT4 weights' smaller reads no longer pay for Marlin's
dequantization arithmetic. At the drafted points the microbenchmark predicts the per-pass change
well: the change in projection time at the pass's row count (target plus drafter) is -0.82,
+0.48 and +2.19 ms at c = 1, 2 and 4, against measured per-pass changes of -0.66, +0.41 and
+1.64 ms. The BF16 tied head (1.27 GB of the INT4 checkpoint's 3.29 GB read per
step) is not quantized, which limits the c = 1 gain as well. This accounts for the direction and
rough size of the served ratios; it does not decompose them, since attention, the GDN layers and
the head are not in the microbenchmark.

## quality/

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh quality q1   # then q2, q3; repository ec91737
python -m experiments.lossy.gsm8k_partial \
  ~/vp-data/lossy/quality/int4-dflash-b8-seed0/20261002-091151 \
  --out evidence/lossy/quality/int4-dflash-b8-seed0-partial
```

`plain-cap256-fp16-seed0/20261002-094130/` (`q2`, 09:41-09:59 UTC) and
`replayssm-cap256-fp16-seed0/20261002-110013/` (`q3`, 11:00-11:18) are `bench.quality`'s
`quality.json` and `problems.csv`, copied unchanged from `~/vp-data/lossy/quality/`. Settings as
declared: all 1,319 GSM8K test problems, thinking on, temperature 0.6, top-p 0.95, top-k 20,
seed 0, 16,384-token limit, 128 threads; every launch check passed, sgl-eval exited 0, and the
package versions equal the references' (`holds/q2.json`, `holds/q3.json`). Foreign CPU averaged
0.84 and 0.86 cores, which affects wall time, not accuracy.

| Run | Accuracy (Wilson 95%) | Truncated / no answer | vs `plain-tuned-a` (89.99%) | vs `plain-tuned-b` (90.30%) |
|---|---|---|---|---|
| `plain-cap256-fp16` | 89.01% (87.20-90.58) | 243 / 101 | -0.99 pt (-2.94, +0.97), p 0.36, 93 vs 80 | -1.29 pt (-3.17, +0.60), p 0.21, 89 vs 72 |
| `replayssm-cap256-fp16` | 88.93% (87.12-90.51) | 235 / 94 | -1.06 pt (-2.95, +0.83), p 0.31, 88 vs 74 | -1.36 pt (-3.27, +0.54), p 0.18, 91 vs 73 |

Each difference is paired by problem, with the 95% Wald interval of the paired difference, the
exact McNemar p-value and the discordant counts (solved only by the reference vs only by the FP16
run). The references truncate 219 and 235 problems and leave 85 and 84 without an answer.
bench's exact arms against the same references: `mtp-tuned` -0.45 and -0.76, `mtp-stockverify`
-1.06 and -1.36, `dflash-tuned` -0.30 and -0.61, `plain-tuned-replayssm` +1.06 and +0.76
(`decisions.json`, `gsm8k_exact_spread`).

Not declared, for context (`context/`): paired against `plain-tuned-replayssm`, the exact arm
with the highest accuracy (91.05%), both FP16 runs come out about 2.1 points lower (-2.05,
p 0.049; -2.12, p 0.032). The exact `mtp-stockverify` shows the same -2.12 points (p 0.031)
against it, so that gap reflects `plain-tuned-replayssm`'s high draw at least as much as anything
the FP16 state does. The two FP16 runs differ from each other by -0.08 points (p 1.0).

```sh
C=evidence/lossy/quality/context B=evidence/bench/quality L=evidence/lossy/quality
python -m bench.quality compare $B/plain-tuned-replayssm-seed0 \
  $L/plain-cap256-fp16-seed0/20261002-094130 --out $C/plain-cap256-fp16_vs_plain-tuned-replayssm.json
python -m bench.quality compare $B/plain-tuned-replayssm-seed0 \
  $L/replayssm-cap256-fp16-seed0/20261002-110013 \
  --out $C/replayssm-cap256-fp16_vs_plain-tuned-replayssm.json
python -m bench.quality compare $B/plain-tuned-replayssm-seed0 $B/mtp-stockverify-seed0 \
  --out $C/mtp-stockverify_vs_plain-tuned-replayssm.json
python -m bench.quality compare $L/plain-cap256-fp16-seed0/20261002-094130 \
  $L/replayssm-cap256-fp16-seed0/20261002-110013 \
  --out $C/replayssm-cap256-fp16_vs_plain-cap256-fp16.json
```

`int4-dflash-b8-seed0-partial/` (not declared): `problems.csv` holds the 1,305 finished problems
(bench.quality's columns) and `partial.json` the comparison. On the finished problems the INT4
run solves 89.43%: -0.69 points against `plain-tuned-a` (-2.61, +1.23; p 0.53), -0.92 against
`-b` (-2.83, +0.99; p 0.39) and -0.23 against stock `dflash-tuned` (p 0.88). The 14 unfinished
problems were the ones still generating at the limit, so they are not a random sample: the
references solve 11 and 12 of them and truncate 6 and 9. Whatever the INT4 run would have scored
on them, its full-split difference lies in [-1.52, -0.45] points against `plain-tuned-a` and
[-1.82, -0.76] against `-b`, so the declared GSM8K rule is undetermined for this arm. The source
predictions file's SHA-256 is recorded in `partial.json`.

## probes/

Written by the quality holds (`hold.sh quality q1`, `q2`, `q3`, which call `run_hold.py probe`
per arm; the commands are in `holds/`). `probe_summary.json` of each probed arm, copied unchanged
from `~/vp-data/lossy/probes/<arm>/`:
`experiments/moonshot/logit_probe.py`'s 48 prompts, 256 greedy tokens, top-20 logprobs, against the
reference `plain-ref-1` (stock `plain-cap256`, load test L0b, SHA-256 `5186472b...c968f7`, pinned
in `plan.REFERENCE_PROBE_SHA256` before `q1`). `decisions.json` (`probes`) has the declared
statistics computed from the raw files:

| Arm | Score mode: agreement, KL | Decode path: agreement, KL | Divergences per 1k shared tokens | Within budget |
|---|---|---|---|---|
| `int4-plain-cap256` | 0.9707, 0.0124 | 0.9737, 0.0122 | 27.1 | no |
| `int4-dflash-b8` | 0.9694, 0.0125 | 0.9750, 0.0120 | 25.6 | no |
| `plain-cap256-fp16` | 0.9938, 0.00027 | 0.9960, 0.00021 | 4.0 | yes |
| `replayssm-cap256-fp16` | 0.9937, 0.00027 | 0.9974, 0.00023 | 2.6 | yes |
| reference noise | 0.9940, 0.00025 | 0.9994, 0.00003 | 0.62 | |

Thresholds: agreement at least 0.98 and mean top-20 KL at most 0.01 nats, in both modes. The
noise row is `plain-ref-1` scored against itself (score mode, prefill against the decode path) and
a second launch's generate run against it (decode path). In score mode the FP16 arms disagree with
the reference at 76 and 77 of 12,240 positions against the reference's own 74. On the decode path
they diverge 4 to 6.5 times as often as a second launch of the reference does, still far inside the
budget. The two INT4 arms give nearly the same values, as expected: greedy speculative decoding
emits the target's tokens, so both probe the same INT4 target, which first diverges from the
reference at a median of 26-27 tokens per sequence.

Raw probe files (outside git), SHA-256:

| File | SHA-256 |
|---|---|
| `probes/int4-plain-cap256/probe_generate.json` | `acf5f4837861cd9cd4131c587e7c9cd7db9b7b01b708bac38dbddd09a9a56fd4` |
| `probes/int4-plain-cap256/probe_score.json` | `293819f0492ae49a7e45e8aa9bba03dbacf5563d34a6b6c193a0850b0634d8ea` |
| `probes/int4-dflash-b8/probe_generate.json` | `c94d3e1f8b3fbd78bc16791243cf58a3b08ad9efa18673cd47857980fc2b765b` |
| `probes/int4-dflash-b8/probe_score.json` | `92f23c493abde6123d7ca6e01e597f9f5370431d287e77b847413b82c19a95b8` |
| `probes/plain-cap256-fp16/probe_generate.json` | `a82ba8cd3ec24115cbc4d096dd858947d6f5aba4a6f87ea525d0aab4722f28e4` |
| `probes/plain-cap256-fp16/probe_score.json` | `baeaba18c3bfadb19a3ccbd76583d0412f24db796b1241d78d0e1a7b88713e66` |
| `probes/replayssm-cap256-fp16/probe_generate.json` | `936836f6c633e172948bfa5ec7332165951d31b819d0e508fbf58126b46f5edd` |
| `probes/replayssm-cap256-fp16/probe_score.json` | `7767f1b47d918ca9059630b6c17e53190b2c36f578ec32d8dd1f1439e2d57963` |
| `load_test/20261002T035539Z/plain-ref-1/probe_generate.json` | `5186472b7ae278bd4d7f9de1bd462e6126f03adb870cd0d98648cbaab0c968f7` |
| `load_test/20261002T035539Z/plain-ref-1/probe_score.json` | `84947211bcb4cb40853d652048b521427882b835558f17bb6908fb4afd3b5ace` |
| `load_test/20261002T035539Z/plain-ref-2/probe_generate.json` | `6aa9985c437ef39e78d8b182c557772dfde02cc3742217969a23d86edea20465` |
| `load_test/20261002T035539Z/plain-ref-2/probe_score.json` | `90022cad40ebf93b862c900433876d1abb306fc34ebe6b4da1b644b705a41a07` |

## holds/

`run_hold.py`'s manifests, copied unchanged from `~/vp-data/lossy/holds/`: for each hold, the
repository and engine commits, package versions, and every launch's command, timeout, times and
exit code. `q1.json` records the GSM8K launch's exit 124.

## Figures

```sh
~/sglang/.venv/bin/python -m experiments.lossy.figures evidence/lossy/decisions.json \
  --out evidence/lossy --gemm evidence/lossy/gemm_w4a16.json \
  --partial-gsm8k evidence/lossy/quality/int4-dflash-b8-seed0-partial/partial.json
```

Run from commit `c25b2bc`. `frontier.png`: session means of every arm, x (output tokens/s per
user, TTFT included) against y (output tokens/s per GPU), with the envelope of the exact arms
alone, and with each lever's arms added. The INT4 envelope leaves the exact one only at c = 1;
the FP16 envelope lies above it from c = 64. `quality_speed.png`: GSM8K difference against the
speed ratio at each lever's headline points, the -1.0-point budget line and the exact arms'
spread as a band; the FP16 arms with their 95% intervals per reference, the INT4 arm as its
full-split bounds (not declared; its x ratio at c = 1 is `int4-dflash-b16`'s, which shares the
weights). `gemm_w4a16.png`: the microbenchmark below.

## gemm_w4a16.json (exploratory, not declared)

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh gemm   # from a checkout at 90e45bf
```

Approved after `lossy-s1` as a mechanism check, run on 2026-10-02 at 08:40 UTC (after `lossy-s3`)
and copied unchanged from `~/vp-data/lossy/gemm/gemm_w4a16_20261002T084008Z.json`; it records its
repository commit, torch version, device and model snapshots. Per forward pass, BF16 against
W4A16, for the target's six quantized projection types and the drafters' layers, at 1-256 rows.

## checkpoint_check.json (CPU only)

```sh
SGLANG_WORKTREE=~/sglang-wt/lossy HF_HUB_OFFLINE=1 source scripts/sglang_env.sh
python -m experiments.lossy.checkpoint_check --out evidence/lossy/checkpoint_check.json
```

Run on 2026-10-02 in a CPU window (no GPU); the file records the repository commit it ran
from (`repo_commit`). From the safetensors
headers, a decode step of plain text decoding reads 8.41 GB of BF16 weights
(`Qwen/Qwen3.5-4B`, the tied 1.27 GB head included) and 3.29 GB of INT4 weights
(`nota-ai/Qwen3.5-4B-QAD-W4A16`: 1.78 GB packed, 0.22 GB group scales, 0.01 GB unquantized,
the same BF16 head), 2.56 times fewer. The vision tower and the MTP layer, not read here, are
excluded from both. The INT4 drafter is 0.28 GB against 1.27 GB for the BF16 DFlash drafter.

Chat template, vocabulary and merges are byte-identical; `tokenizer.json` and
`tokenizer_config.json` differ in serialization (the load test shows the server token ids are
identical). The end-of-sequence sets SGLang resolves differ: {248044, 248046} for BF16 and
{248046} for INT4 (its `config.json` sets a top-level `eos_token_id`). With the override the
INT4 arms pass (`json-model-override-args`, `bench/arms.toml`) the INT4 set is {248044, 248046}.

## load_test/ (untimed)

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh load                       # l0_*.json
scripts/gpu_lock.sh -x experiments/lossy/hold.sh load int4-dflash-b8 int4-plain-cap256 \
    plain-cap256-fp16 replayssm-cap256-fp16 replayssm-cap256 plain-ref-1 plain-ref-2  # l0b_*.json
```

`l0_20261002T033138Z.json` (repository `51808ce`) and `l0b_20261002T035539Z.json`
(`5f149d2`) are the load test's own records, copied unchanged. L0 ran its first two steps; the
other seven failed only because the server launched under Nsight Systems outlived its step and
held the port (a harness fault, fixed in `5f149d2`), and L0b ran them. Every arm then passed
every required launch check (decode and prefill CUDA graphs covering the capacity, overlap
scheduler on, capacity, attention backend, speculative settings), and on every arm the server's
token ids for 16 templated prompts equal those of the `Qwen/Qwen3.5-4B` tokenizer.

| Arm | Capacity | Accept length (8 greedy requests) | Weights in memory | GDN state |
|---|---|---|---|---|
| `int4-dflash-b16` | 64 | 4.48 | 3.90 GB target + 0.41 GB drafter | 3.05 GB |
| `int4-dflash-b8` | 128 | 3.81 | 3.90 + 0.41 GB | 6.05 GB |
| `int4-plain-cap256` | 256 | - | 3.90 GB | 12.05 GB |
| `plain-cap256-fp16` | 256 | - | 8.62 GB | 6.02 GB (FP16) |
| `replayssm-cap256-fp16` | 256 | - | 8.62 GB | 6.02 GB (FP16) |
| `replayssm-cap256` | 256 | - | 8.62 GB | 12.05 GB |
| `plain-cap256` (`plain-ref-1`, `-2`) | 256 | - | 8.62 GB | 12.05 GB |

Weights in memory include the vision tower (0.67 GB) and the MTP layer (0.24 GB), which text
decoding does not read. The accept lengths come from a smoke test of eight requests, not from
the timed runs.

`int4_dflash_b16_kernels.csv`: GPU kernels by total time over 20 scheduler steps of
`int4-dflash-b16` with four requests decoding (`nsys profile --cuda-graph-trace=node`, opened
by SGLang's `/start_profile` with the `CUDA_PROFILER` activity). The W4A16 Marlin kernels take
30.1% of the GPU time and dense BF16 GEMMs (`nvjet`, the BF16 head and the unquantized layers)
12.2%; Triton attention (`_fwd_kernel`) takes 30.8%. The quantized weights run on the low-bit
kernel, not as dequantized BF16 copies (which would also show in the weight memory).

Logit-probe reference (`plain-ref-1`, generate mode, SHA-256 `5186472b...c968f7`) and its own
noise, as `analyze.probe_noise` computes it: `plain-ref-1` scored against itself (prefill
against decode path) agrees on the top-1 token at 99.40% of positions with mean top-20 KL
0.00025 nats; a second launch (`plain-ref-2`) scored against it gives the same 99.40% and
0.00025; on the decode path the second launch agrees at 99.94% (7 of 48 sequences diverge,
0.62 per 1,000 shared tokens) with KL 0.00003.
