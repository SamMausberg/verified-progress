# speed-lowc confirmation: the exact levers that survived their probes, composed on the low-concurrency envelope

Declared on 2026-10-02 in the commit that adds this file, before any of its holds ran. Any change after
the equality hold starts is added below as a dated amendment with its reason.

**Question.** On the confirmed frontier's best exact arms at client concurrency 1-32
(`evidence/bench/README.md`), what do the three exact levers that passed their probes
(`evidence/speed_lowc/README.md`) give together, and what does each give alone, measured in the same
sessions with identical flags apart from the lever?

## Engine

`experiments/speed_lowc/build_engines.sh confirm`: the pin `bd66ce343e` plus `engine/sglang/patches/drafter/0001-0004`
and `engine/sglang/patches/speed-lowc/0001` and `0003`; tree `9a01a622f6e7f7f816ce6255ba5de56d52e09dbc`, every
switch off by default. Every hold checks the tree and refuses a worktree with local changes. Each server's
`launch.json` records the SGLang commit it imported.

## Arms (`experiments/speed_lowc/confirm_arms.sh`)

Two groups, each the envelope arm of its range (`bench/arms.toml`), with every bench flag unchanged:

- **L**: `dflash-tuned-b16` (block 16, Triton target and draft attention, capacity 64) at c = 1, 2, 4.
- **H**: `dflash-tuned` (block 8, FlashInfer target attention, FA4 draft attention, capacity 128) at c = 8, 16, 32.

Levers:

| Lever | Change | Groups | Probe evidence (one session each) |
|---|---|---|---|
| A | exact GDN fold: `--enable-linear-replayssm-spec`, `SGLANG_GDN_REPLAYSSM_FOLD=1` (drafter 0001-0004), with ring-verify tiles BV=4 up to 4 sequences (speed-lowc 0003) | L, H | fold with narrow tiles 1.020 / 1.012 / 1.018 at c = 1 / 2 / 4 on block 16; fold with wide tiles 1.061 / 1.058 at c = 8 on block 16 / 8 (`evidence/drafter/fold_narrow_tiles/timing/`, `evidence/drafter/fold_timing/`) |
| B | FA4 draft attention: `--speculative-draft-attention-backend fa4` | L (H already drafts with FA4) | x 1.029 / 1.027 / 1.039 / 1.043 at c = 1 / 2 / 4 / 8 (`evidence/speed_lowc/probe2/`) |
| C | FA4 target attention: `--attention-backend fa4` (needs speed-lowc 0001) | L, H | x 1.039 / 1.042 at c = 1 / 4 over B on block 16; 1.074 / 1.026 at c = 8 / 32 on block 8 (`evidence/speed_lowc/probe4/`) |

On L, an arm with C but not B keeps the drafter on Triton (`--speculative-draft-attention-backend triton`), so
each letter changes exactly one thing. FULL is ABC on L and AC on H. S0 is stock SGLang (`~/sglang` at the
pin); B0 is the confirm engine with every switch off.

Measured and not timed here: the FP8 draft head (`SGLANG_FP8_DRAFT_HEAD=1`, speed-bytes): x 1.012 at c = 1 and
1.032 at c = 4 on block 16, 0.995 at c = 8 on block 8 in one session, below the 2% rule at c = 1 and 8; split-KV
verify and a fused GDN chain (killed, `evidence/speed_lowc/README.md`).

## Step 1: equality (`experiments/speed_lowc/hold_confirm_equality.sh`, one exclusive hold)

State's runner (`experiments/state_safety/run_matrix.py`): the 320 state prompts, 256 greedy tokens, top-5
logprobs, c = 1, radix cache off, running limit 4, SGLang's own pools (one request at a time, so the pools
cannot change batch composition). Arms: L: S0, B0, A, B, C, ABC; H: S0, B0, A, C, AC. Each arm is compared
with S0 of its group by state's `compare.py`, and `experiments/speed_lowc/confirm_gate.py` decides:

- B0 must be bitwise equal to S0 (token ids and top-logprob arrays on all 320 prompts);
- every other arm must cover all 320 prompts with no length mismatch, and every first divergence must be
  classified tie, one_ulp or near (bench's rule as an allow-list; large, not_argmax and unknown fail);
- the sessions run only if B0 and every lever and FULL pass in both groups (`gate.json`, `ok: true`, levers
  `ABC`). Otherwise no session runs until a dated amendment decides.

Expected: A, B and C each change rounding (the fold's replay, FA4 against Triton or FlashInfer reductions), so
the expected class is exact up to rounding, with a few prompts diverging at ties; draft-side B can also move
verify-block boundaries. Reported per arm: identical prompts / 320 and the class counts.

## Step 2: three timed sessions (`experiments/speed_lowc/hold_confirm_session.sh <k>`, one exclusive hold each)

Per group, every arm launched once through `bench.sweep` (confirm split, 512 output tokens, bench's default
64 measured requests or 8 waves per point, quiet-host wait up to 300 s), in the order

    L: S0 ABC A B C ABC S0        H: S0 AC A C AC S0

with the single levers reversed in session 2, and group order L, H in sessions 1 and 3 and H, L in session 2.
Sessions refuse to start unless the equality gate passed for levers ABC.

## Analysis (`experiments/speed_lowc/confirm_analyze.py`, declared)

`python -m bench.pareto <the sessions' run dirs> --points-only` gives points.csv; then
`python experiments/speed_lowc/confirm_analyze.py --points points.csv --full L=ABC --full H=AC --out <dir>`.

- Session ratio of arm X at c: mean of X's launches over the mean of S0's two launches, for x_e2e and y.
- Across the three sessions: geometric mean and a 95% t interval on the logs (2 degrees of freedom).
  Speedup if the interval's lower end is above 1, slowdown if its upper end is below 1, otherwise no
  detectable change.
- A cell is void if any of its points is invalid by bench's rules (foreign CPU above 2 cores, failed
  requests, wrong lengths) or if the arm or S0 has a different number of launches than declared; fewer than
  three valid sessions leave the ratio undecided.
- Primary outcome: FULL against S0, x_e2e at c = 1, 2, 4 and y at c = 8, 16, 32. Single levers are
  secondary (attribution); they are not multiplied to predict FULL.
- Accepted tokens per cycle are reported per arm; a gain that comes from acceptance drift rather than a
  shorter cycle is named as such (x over accept length per arm).

## Commands

```sh
experiments/speed_lowc/build_engines.sh confirm
CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_equality.sh
for k in 1 2 3; do CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_session.sh $k; done
```

Outputs: `~/vp-data/speed-lowc/confirm/equality-<UTC>/` (with `current` pointing at the last passing one) and
`~/vp-data/speed-lowc/confirm/s<k>-<UTC>/`.

## Amendments

None yet.
