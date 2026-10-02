# BF16 against FP32 at two served positions: SGLang, transformers and the positions themselves

The served certified-head benchmark found two positions where stock SGLang's BF16 prefill and
decode paths disagree with each other, and one of them with FP32, by 9 to 15 nats
([`../certified_head/served/README.md`](../certified_head/served/README.md), "Reference paths
and FP32"): at `579ae7ce` position 439 batch-1 decoding puts token 1756 on top (-0.32) where FP32
puts it at -9.79, and at `a4db11ff` position 333 the prefills put FP32's top token 18299 at -13.4
to -15.3. Both positions lie in text the model wrote after its own end-of-text token, which the
benchmark's ignore_eos requests keep generating. This directory asks whether these errors are
specific to SGLang or whether any BF16 implementation of the model would make them, which decides
whether they belong in an SGLang issue. The code is in `experiments/bf16_paths/`.

Status: measured. Correctness only; nothing here is timed.

## Result

Neither error is shown to be specific to SGLang, so neither supports an SGLang issue. The
readings (each set before its run, in the scripts' docstrings) give three findings.

1. **`a4db11ff`/333 is ill-conditioned in BF16; any implementation can miss it.** transformers in
   BF16 misses FP32 there by as much as SGLang does: its torch GDN prefill of the whole output puts
   18299 at -17.16, and its decode with an FP32 cached state at -15.63, where FP32 has -0.45 and
   SGLang's prefills -13.4 to -15.3. FP32 itself, under relative noise within the BF16 rounding
   bound, flips its top-1 between 18299 and 5500 and puts 18299 as low as -11.65.
2. **At `579ae7ce`/439 only SGLang misses, and no single kernel carries the miss.** There FP32
   under the same noise keeps 1756 within 0.52 nats of -9.79, and transformers in BF16 (four
   configurations, four paths each) keeps FP32's top-1 68189 on top and 1756 between -10.11 and
   -7.60. SGLang's decode puts 1756 on top at -0.32, and swapping one kernel family at a time moves
   it between -8.9 and -0.32 on one path or another. One BF16 rounding of beta (the lines of the
   open upstream PRs #38977 and #40362) moves the decode to -4.97 and the prefills by up to 5.6
   nats without bringing any path to FP32, and at `a4db11ff`/333 it fixes the prefills (18299 at
   -0.29 to -0.33) while sending the decode from -0.28 to -7.26. SGLang's arithmetic at this
   position is therefore far more sensitive than FP32's or transformers', but that rests on one
   position chosen because SGLang erred there.
3. **On positions nobody selected, SGLang is as accurate as transformers.** Over 15,360 positions
   per path (12 workload prompts decoded for 512 tokens, then 12 prompts decoded for 768 tokens so
   that 5,426 positions lie after the end of text), SGLang's mean logprob error on FP32's top token
   (0.0039-0.0054 nats per path and sample) lies within transformers' range (0.0042-0.0057), and
   SGLang misses FP32's top by more than 0.5 nats at 3 positions against 2 and 1 for the two
   transformers configurations. The only miss above 2 nats, 3.2 nats at `6af1e245` position 247
   after the end of text, is shared by the prefill paths of all three implementations. The rule
   declared before the runs needed at least 5 positions where SGLang misses by more than 2 nats
   and 3 times transformers' count; both samples were inconclusive (0 against 0, then 1 against
   1). A position missed on both of a source's paths now counts once, a correction made after the
   runs (Codex on #222) that changes neither count.

So `579ae7ce`/439 is a real SGLang outlier at one position, after the model's end-of-text token,
but the samples cannot tell whether such outliers are more frequent in SGLang than in transformers:
with one miss above 2 nats per path in 15,360 positions, a several-fold difference in that rate
would go undetected. What they do show is that SGLang's typical BF16 accuracy on this model
matches transformers', and that large misses after the end of text occur in every BF16
implementation.

## The two positions in detail

Each path reads the recorded text: a prefill is one forward over the first n tokens (prompt plus
output), and the decode path prefills to 39 positions before the target and then feeds one
recorded token at a time. `positions.json` has every tracked token per path, the largest
absolute difference from FP32's one forward on them, and FP32's top logprob minus FP32's
logprob of the path's top-1.

At `579ae7ce` position 439, token 1756's logprob and, in parentheses, each path's top-1 (FP32
puts 68189 on top at -0.87):

| Source | Prefill to the target (514) | Prefill, 62 longer (576) | Whole output (587) | Decode from 400 |
|---|---|---|---|---|
| FP32, transformers on the CPU (one forward; recurrent from 400) | -9.79 (68189) | | | -9.79 (68189) |
| transformers BF16: torch GDN, BF16 state | -9.48 (68189) | -9.02 (68189) | -9.79 (68189) | -9.65 (68189) |
| transformers BF16: torch GDN, FP32 state | -9.48 (68189) | -9.02 (68189) | -9.79 (68189) | -9.82 (68189) |
| transformers BF16: fla GDN, BF16 state | -9.79 (68189) | -10.11 (68189) | -10.02 (68189) | -7.60 (68189) |
| transformers BF16: fla GDN, FP32 state | -9.79 (68189) | -10.11 (68189) | -10.02 (68189) | -9.60 (68189) |
| SGLang, benchmark engine, stock arm | -8.12 (68189) | -8.89 (68189) | -8.32 (8078) | -0.32 (1756) |
| SGLang pin: default | -8.12 (68189) | -8.89 (68189) | -8.32 (8078) | -0.32 (1756) |
| SGLang pin: no CUDA graphs | -8.12 (68189) | -8.89 (68189) | -8.32 (8078) | -0.32 (1756) |
| SGLang pin: Triton GDN prefill | -3.76 (5715) | -3.27 (5715) | -3.27 (5715) | -7.68 (8078) |
| SGLang pin: FlashInfer GDN decode | -8.12 (68189) | -8.89 (68189) | -8.32 (8078) | -0.94 (1756) |
| SGLang pin: Triton attention | -7.85 (68189) | -7.85 (68189) | -2.26 (5715) | -0.35 (1756) |
| SGLang pin: beta kept in FP32 | -4.41 (5715) | -7.55 (8078) | -2.76 (5715) | -4.97 (8078) |

At `a4db11ff` position 333, token 18299's logprob (FP32's top, at -0.45; 5500 is at -1.74):

| Source | Prefill to the target (393) | Prefill, 62 longer (455) | Whole output (572) | Decode from 294 |
|---|---|---|---|---|
| FP32, transformers on the CPU (one forward; recurrent from 294) | -0.45 (18299) | | | -0.45 (18299) |
| transformers BF16: torch GDN, BF16 state | -0.62 (18299) | -4.40 (5500) | -17.16 (5500) | -0.33 (18299) |
| transformers BF16: torch GDN, FP32 state | -0.62 (18299) | -4.40 (5500) | -17.16 (5500) | -15.63 (5500) |
| transformers BF16: fla GDN, BF16 state | -4.06 (5500) | -3.71 (5500) | -3.53 (5500) | -0.33 (18299) |
| transformers BF16: fla GDN, FP32 state | -4.06 (5500) | -3.71 (5500) | -3.53 (5500) | -0.25 (18299) |
| SGLang, benchmark engine, stock arm | -13.89 (5500) | -13.40 (5500) | -15.32 (5500) | -0.28 (18299) |
| SGLang pin: default | -13.89 (5500) | -13.40 (5500) | -15.32 (5500) | -0.28 (18299) |
| SGLang pin: no CUDA graphs | -13.89 (5500) | -13.40 (5500) | -15.32 (5500) | -0.28 (18299) |
| SGLang pin: Triton GDN prefill | -0.21 (18299) | -1.55 (5500) | -0.68 (18299) | -5.04 (5500) |
| SGLang pin: FlashInfer GDN decode | -13.89 (5500) | -13.40 (5500) | -15.32 (5500) | -0.29 (18299) |
| SGLang pin: Triton attention | -13.64 (5500) | -13.64 (5500) | -13.64 (5500) | -0.28 (18299) |
| SGLang pin: beta kept in FP32 | -0.33 (18299) | -0.29 (18299) | -0.33 (18299) | -7.26 (5500) |

The unpatched pin reproduces the benchmark engine's readings to every printed digit, with or
without CUDA graphs, so the observation stands on upstream code and the graphs play no part. The
SGLang variants each change one kernel family; the GDN dispatcher line of each server log
confirms the swap (`positions.json`, `meta`). The `beta_fp32` variant carries the one-line
changes of the open upstream PRs #38977 and #40362 (issue #38975: the packed GDN decode kernel
and the gating kernel round sigmoid(beta) through BF16).

In the terms of the pre-set readings (largest difference from FP32 on the tracked tokens,
`positions.json`): no variant brings `579ae7ce`'s decode within 1 nat of FP32 (the closest,
Triton GDN prefill, is 2.75 off). At `a4db11ff` the Triton GDN prefill brings one of the three
prefills within 1 nat (0.31; the others 1.18 and 1.34) and `beta_fp32` brings them to 1.06-1.72,
but both then break the decode path (4.60 and 6.81 off), so no kernel family carries either
error. For transformers, the reading called the error SGLang-specific only if every
transformers run stayed within 1 nat of FP32 at both positions; at `a4db11ff` they miss by up to
16.7 nats (not specific), and at `579ae7ce` two of sixteen readings exceed 1 nat (1.15 on 5715,
the torch 576-token prefill; 3.08 on 5715 and 2.19 on 1756, the fla decode with a BF16 state),
which the reading reports as a partial BF16 sensitivity rather than an SGLang fault.

On the 39 positions before each target every path gives the recorded token's logprob within 0.05
nats of FP32 (at most 0.047, SGLang's 576-token prefill at `579ae7ce` position 437). Those tokens
are near-certain (FP32 gives each a logprob of -0.103 or higher), so this agreement shows that the
paths read the same text, not that their distributions agree where the model is uncertain.

**How sensitive the positions are by themselves.** `perturb.py` repeats FP32's one forward with
BF16-sized relative noise, (1 + u) with u uniform in [-2^-8, 2^-8], 2^-8 being the bound on the
relative error of rounding to BF16: once on every decoder layer's output (the residual stream),
once on the inputs of every GDN layer's recurrence (query, key, value and beta, the quantities
both SGLang and transformers hold in BF16). Eight seeds each; ranges over the seeds at the target
(`positions.json`, `perturbation`). A first pair of runs drew u from [-2^-9, 2^-9], half the
bound, and is superseded; it gave the same picture with smaller ranges.

| FP32 forward | `579ae7ce`/439: 1756 | 68189 | top-1 | `a4db11ff`/333: 18299 | 5500 | top-1 |
|---|---|---|---|---|---|---|
| unperturbed | -9.79 | -0.87 | 68189 | -0.45 | -1.74 | 18299 |
| noise on every layer's output | -10.04 to -9.27 | -1.21 to -0.71 | 68189 in all 8 | -11.65 to -0.22 | -3.78 to -0.01 | 5500 in 4 of 8 |
| noise on the GDN inputs | -9.90 to -9.69 | -0.94 to -0.80 | 68189 in all 8 | -2.72 to -0.25 | -3.12 to -0.14 | 5500 in 4 of 8 |

At the positions before each target the same noise changes the recorded token's logprob by at
most 0.022 nats (the control). So `a4db11ff`/333 is ill-conditioned at BF16 rounding scale: noise
within one rounding flips FP32's own top-1 and can push 18299 down by 11 nats. `579ae7ce`/439 is
not: the same noise moves 1756 by at most 0.52 nats, while SGLang's paths put it anywhere from
-8.9 to -0.32.

**Selection-free error rates (`rates.json`, `rates_eot.json`).** The two positions were found
because SGLang erred there, so they cannot compare how often implementations err. `rates.py`
takes the first 12 prompts by hash of session 1's plain c = 128 point, decodes each greedily at
batch 1 on the unpatched pin with ignore_eos (the decode path), prefills the prompt and that
output in one request (the prefill path), reads the same text with transformers in BF16 (one
forward, and token by token through the cache) and scores every path's top-1 against an FP32
forward on the CPU. None of these 12 decodes reached an end-of-text token in 512 tokens, so
`rates_eot` repeats the comparison on the first 12 prompts by hash whose recorded output in that
point ends its text before position 400, decoding 768 tokens. Regret is FP32's top logprob minus
FP32's logprob of the path's top-1; the error is the absolute difference between the path's and
FP32's logprob of FP32's top-1 token.

| Path | Ordinary text (6,144 positions): regret > 0.5 | regret > 0.05 | top-1 differs | mean error | After the end of text (5,426 positions): regret > 2 | regret > 0.5 | top-1 differs | mean error, all 9,216 |
|---|---|---|---|---|---|---|---|---|
| SGLang, decode | 0 | 14 | 39 | 0.0054 | 0 | 1 | 14 | 0.0039 |
| SGLang, prefill | 0 | 13 | 36 | 0.0054 | 1 | 2 | 23 | 0.0045 |
| transformers, torch GDN, FP32 state, decode | 0 | 9 | 34 | 0.0057 | 0 | 1 | 14 | 0.0044 |
| transformers, torch GDN, FP32 state, prefill | 0 | 17 | 43 | 0.0056 | 1 | 1 | 16 | 0.0049 |
| transformers, fla GDN, FP32 state, decode | 0 | 7 | 31 | 0.0056 | 0 | 0 | 14 | 0.0042 |
| transformers, fla GDN, FP32 state, prefill | 0 | 15 | 41 | 0.0056 | 1 | 1 | 18 | 0.0047 |

Before the end of text in `rates_eot` (3,790 positions) no path misses by more than 0.2 nats.
transformers with its default BF16 cached state, run on the first sample only, gives 17 decode
positions above 0.05 nats and a mean error of 0.0063. The declared decision (`decide` in
`rates.py`, in both files under `decision`) counts misses above 2 nats over both paths and compares
SGLang with the worse of the two FP32-state transformers runs.

## What was run

All runs are correctness-only (shared GPU lane, nothing timed), on one GH200 (driver 570.195.03,
CUDA 13 compatibility libraries; `SETUP.md`), with `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in its checkpoint dtype, BF16.

- SGLang: the paper's pin `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` from `~/sglang` with no
  patches (version string 0.5.21.dev861), torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18,
  launched with the `plain-tuned` arm as `experiments/benchcert/rescore.py` runs it:
  `--attention-backend flashinfer --mm-attention-backend triton_attn --disable-radix-cache
  --max-running-requests 32 --max-mamba-cache-size 32 --mem-fraction-static 0.25
  --max-total-tokens 200000 --random-seed 0 --stream-interval 4 --enable-metrics`. On SM90 this
  resolves to FlashInfer's GDN kernel for prefill and SGLang's Triton GDN kernel (packed
  decode, fused projection unpack and conv) for decode, with the GDN state in FP32 (the
  checkpoint's `mamba_ssm_dtype`) and the conv state in BF16. Each request runs alone, the
  cache flushed before it. The `beta_fp32` variant runs the same flags on
  `engine/sglang/patches/upstream-bf16/0001` (engine head `4608661757`).
- transformers 5.12.1 (`AutoModelForCausalLM`, eager attention, TF32 off), BF16 on the GPU:
  its torch GDN implementation (FP32 inside the recurrence) and flash-linear-attention 0.5.2
  (`fla-core` and `flash-linear-attention`, installed with `uv pip install --no-deps --target`
  outside the venv and used only through `PYTHONPATH`; `causal-conv1d` absent, so the conv
  stays in torch). transformers stores the cached GDN state in the model dtype; the
  `fp32state` runs keep it in FP32 as SGLang does.
- FP32: the same transformers model in FP32 on the CPU (8 cores), as `paths.py fp32` ran it.

Commands, from the repository root in the SGLang environment (`source scripts/sglang_env.sh`):

```sh
# Hold 1 (repo d8522be): the SGLang variants (the transformers step of this hold failed on a
# trace-range bug, fixed in b6abd11, and is void).
scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh
# Hold 2 (repo 4923b0a): transformers BF16 at the two positions; FP32 with residual noise.
scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh hf perturb
# Hold 3 (repo aac2264): FP32 with noise on the GDN inputs; the selection-free rates.
scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh perturb_gdn rates
# Hold 4 (repo 43efc79): the same rates on text written after the end of text.
scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh rates_eot
# Hold 5 (repo 94460d6): both perturbations again at the full rounding bound, 2^-8. The runs of
# holds 2 and 3 used 2^-9 and are superseded; hold.sh refuses to overwrite an output, so they
# were moved aside first:
mkdir -p ~/vp-data/upstream/bf16/superseded_eps2m9
mv ~/vp-data/upstream/bf16/perturb.json ~/vp-data/upstream/bf16/perturb_gdn.json \
  ~/vp-data/upstream/bf16/superseded_eps2m9/
scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh perturb perturb_gdn
# (Into a fresh output directory at 94460d6 or later, holds 2 and 3 already perturb at 2^-8 and
# hold 5 is not needed.)
# Readouts (CPU), from the holds' outputs in ~/vp-data/upstream/bf16:
python -m experiments.bf16_paths.summary --paths ~/vp-data/exactness/paths \
  --out ~/vp-data/upstream/bf16 --json evidence/bf16_paths/positions.json
python -m experiments.bf16_paths.rates summary --out ~/vp-data/upstream/bf16/rates \
  --json evidence/bf16_paths/rates.json
python -m experiments.bf16_paths.rates summary --out ~/vp-data/upstream/bf16/rates_eot \
  --json evidence/bf16_paths/rates_eot.json
```

The targets file is `paths.py targets`' `targets.jsonl` from the serial-references hold
(sha256 `3a4351a4...`), copied into the output directory by the first hold and compared byte
for byte by the later ones. FP32 along both paths and the benchmark engine's stock reading
come from that hold (`../certified_head/served/reference_paths.json`).

## Files

| File | What it holds | Command |
|---|---|---|
| `positions.json` | Both positions along every path of every source: FP32 (one forward, recurrent), the benchmark engine's stock reading, SGLang's six variants, transformers BF16 in four configurations; per path the top-1, the tracked tokens' logprobs, the largest difference from FP32, FP32's regret of the path's top-1 and the largest difference on the recorded token before the target; and the two FP32 perturbation readouts | `python -m experiments.bf16_paths.summary --paths ~/vp-data/exactness/paths --out ~/vp-data/upstream/bf16 --json evidence/bf16_paths/positions.json` |
| `rates.json` | The selection-free comparison on 12 workload prompts (512 tokens): per path, positions by FP32 regret threshold before and after the first end of text, top-1 disagreements with FP32, the logprob error on FP32's top-1, the declared decision and the worst positions | `python -m experiments.bf16_paths.rates summary --out ~/vp-data/upstream/bf16/rates --json evidence/bf16_paths/rates.json` |
| `rates_eot.json` | The same on 12 prompts whose recorded output ends its text before position 400, decoded for 768 tokens so that the text after the end of text is sampled | `python -m experiments.bf16_paths.rates summary --out ~/vp-data/upstream/bf16/rates_eot --json evidence/bf16_paths/rates_eot.json` |

The raw outputs (per-path traces, top-20 logprobs per position, server logs and launch records)
stay in `~/vp-data/upstream/bf16/`.

## Limits

- Both samples come from one workload (session 1's plain c = 128 prompts), one model and one
  GPU. Misses above 2 nats are rare in every implementation (one per path in 15,360 positions),
  so the samples establish equal typical accuracy, not equal tails: a several-fold difference in
  the rate of large misses would not be detected.
- In the rate comparison every implementation reads the text of SGLang's own greedy decode; the
  transformers decode path is teacher-forced on it. The comparison is per position on a common
  text, not between free-running generations.
- transformers' torch GDN keeps the recurrence in FP32, a stronger core than a fused BF16 kernel;
  flash-linear-attention's kernels are therefore the second comparator. The conv stays in torch
  in both (no `causal-conv1d`).
- The `beta_fp32` patch is a diagnostic. It shows how sensitive these positions are to beta's
  rounding, not whether #38977 or #40362 improves accuracy in general.
- FP32 is transformers on the CPU; its two paths agree within 0.0024 nats at both targets
  (`../certified_head/served/reference_paths.json`). The perturbation runs are one forward each
  with 8 seeds of uniform noise within the rounding bound; they measure sensitivity to noise of
  rounding size, not to SGLang's particular roundings.
- The rate decision treats missed positions as independent; positions of one prompt are not, but
  with at most one missed position per source the question does not arise here.
- Hold 1's transformers step crashed on a trace-range bug (fixed in `b6abd11`) and is void; its
  SGLang variants are valid.
