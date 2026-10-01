# Real-head replay: transport versus self-evidence

This experiment decides H2 and H3 in `TASKS.md` on real hidden states. It captures the
exact inputs of the shared LM head from a running SGLang server, then evaluates, offline
and in FP64, two ways of certifying the head's decision without computing it densely:

- **Transport (H2, the paper's transport section, `paper/sections/transport.tex`):**
  draft-produced tile summaries `M_c^d`, `Z_c^d` moved to the target state with
  `|z_t - z_d - <mu_c, Delta>| <= eps_c`.
- **Self-evidence (H3):** a low-precision copy of the head with a rigorous per-row
  envelope `|z_i - z_hat_i| <= beta_i`, and exact rescoring of the rows whose interval
  can still win (greedy argmax, or Gumbel-max with the same noise field).

Results and their interpretation are in `evidence/head_geometry/README.md`.

## The transport-versus-int8 threshold

The comparison between the two mechanisms is stated per vocabulary row, in the l2
(Cauchy-Schwarz) family, as in `evidence/precision/head_constants.json` and
`experiments/precision_head_constants/head_constants.py`:

- `rho = ||h_t - h_d||_2 / ||h_t||_2`, where `h_d` and `h_t` are the exact LM-head inputs
  (after the final norm) behind the same verified draft token.
- Transport's error term for row `i` is `r_c ||Delta||_2`, with
  `r_c = max_{j in c} ||w_j - mu_c||_2` over the row's 64-row contiguous tile `c` and
  `mu_c` the tile mean.
- The int8 per-row self-evidence term is `||e_i||_2 ||h_t||_2`, `e_i = w_i - s_i q_i`.
- Transport's envelope is narrower for row `i` iff `rho < t_i := ||e_i||_2 / r_c(i)`.

The threshold `t_i` depends on the weights only; `rho` depends on the drafter.
`analyze_rho.py` measures both, the fraction of rows where transport is narrower for
every pair (also against int8 g128, int4 per-row and int4 g128), head-metric versions
of the drift that do not charge directions the head ignores, realized (not certified)
per-tile errors of both mechanisms, and the certification rates that follow. A ratio
below the threshold is necessary but not sufficient: tile certification also needs the
tile maxima to fall below a threshold score, which `analyze_transport.py` measures.

## Files

| File | Role |
|---|---|
| `build_prompts.py` | Seeded public prompt set (chat, code, maths, multilingual; 320 prompts) with an analysis/held-out split by prompt |
| `run_capture.sh` | Launches a capture-patched server for one arm, sends the prompts, stops the server |
| `capture_client.py` | Greedy `/generate` client; `rid = p<prompt_id>` joins records to prompts |
| `replay_data.py` | Loads the head at a pinned revision and the captured records; builds aligned pairs |
| `validate_alignment.py` | Recomputes draft and target argmaxes from the captured inputs |
| `bounds.py` | Tilings, transport and static bounds, quantizers, envelopes, candidate sets |
| `analyze_transport.py` | H2 metrics |
| `analyze_selfevidence.py` | H3 metrics |
| `analyze_rho.py` | The drift ratio rho, the per-row threshold and head-metric drift |
| `analyze_stats.py` | Norms, drift, margins, top-m mass, hidden-dimension outliers; envelope width against realized error |
| `analyze_rstock.py` | How often the stock head's decision (R-stock) needs the stock kernel: `stock_gap` rule and a bucket-exact rule |
| `export_candidate_ccdf.py` | Candidate-count CCDFs for plotting |
| `tail_killtest.py` | Proposal P1 kill test: INT8 surrogate of the final FFN versus certifying the head alone (HF forward on CPU) |
| `work_model.py` | Work and predicted time per mechanism from replay counts and measured primitive costs (model predictions) |

`tests/test_head_geometry_bounds.py` checks that every bound encloses exact values on
random BF16 heads (it skips without torch).

## Why an engine patch rather than `return_hidden_states`

SGLang's `--enable-return-hidden-states` returns the target's stored hidden states for
generated tokens. It does not return the MTP draft's head inputs, it returns the
auxiliary layer features rather than the head input for DFlash targets, and it carries
no draft tokens or accept lengths per verify step. The patch
(`engine/sglang/patches/geometry/0001-head-capture-replay-dumps.patch`) instead stashes the
tensor `LogitsProcessor` hands to the head and joins it, inside the speculative workers,
with the tokens and decisions of the same verify step. Alignment is then checked
end to end by recomputing argmaxes (below), not assumed.

## Alignment convention

A verify step for one request holds tokens `[x0, d1, ..., dS]` (`x0` is the last
committed token). Draft token `dk` was proposed from draft head input `h^d_k`; it is
accepted when the target argmax at verify position `k-1` equals it and all earlier drafts
were accepted. The pair `(h^d_k, h^t_{k-1})` is therefore one row of the transport
analysis, labelled accepted, rejected (reached, mismatch) or unreached. The target head
input at position `S` (the bonus position) has no draft anchor.

## Commands

Captures and every CPU-heavy analysis go through the GPU lock (`-s` for correctness work;
CPU-heavy jobs too, so they never overlap a timed exclusive run). Captures use
`SGLANG_WORKTREE=~/sglang-wt/geometry` with the patch applied; raw dumps go to
`~/vp-data/geometry/` (outside git). The exact command and code commit behind each
committed file are listed in `evidence/head_geometry/README.md`; the pattern is:

```sh
source scripts/sglang_env.sh
python experiments/head_geometry/build_prompts.py --out ~/vp-data/geometry \
    --manifest evidence/head_geometry/prompt_manifest.csv

# Captures (server start-up is serialized by scripts/gpu_startup_lock.sh inside the script)
scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh plain4b
scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh mtp4b
scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh dflash4b

cd experiments/head_geometry
EV=../../evidence/head_geometry
L=../../scripts/gpu_lock.sh
$L -s python validate_alignment.py --arm dflash4b --device cpu --out $EV/alignment_dflash4b.json
$L -s python analyze_rho.py --arm dflash4b --device cpu --max-rows 40000 --csv-pairs 15000 \
    --out $EV/rho_dflash4b.json
$L -s python analyze_transport.py --arm dflash4b --device cpu --max-rows 4000 --out $EV \
    --tag dflash4b
$L -s python analyze_selfevidence.py --device cpu --threads 40 --sets dflash_verify dflash_draft \
    --max-rows 4000 --heads int8_row int8_g128 int8_g32 fp8_row int4_g128 int4_g32 \
    --out $EV --tag dflash4b
$L -s python analyze_stats.py --arm dflash4b --device cpu --out $EV/stats_dflash4b.json
```

`--device cuda` runs the same analyses on the GPU (the transport analysis on 16,000 pairs
takes about three minutes there, against about six minutes for 4,000 pairs on 40 CPU
cores). The 27B DFlash2 arm (`dflash27b`) is wired into the capture script and the
analyses but was not run.

## Server configuration used for capture

Correctness capture, not a performance configuration: CUDA graphs and the overlap
scheduler are off because the hooks run in Python between forwards. Attention backend
`flashinfer` (no FA3 on aarch64). Greedy decoding, 384 new tokens, the model's chat
template with thinking enabled for one third of the prompts (assigned by seed).

| Arm | Model | Speculation | Concurrency |
|---|---|---|---|
| `plain4b` | Qwen3.5-4B @851bf6e8 | none | 16 |
| `mtp4b` | Qwen3.5-4B @851bf6e8 | NEXTN, 4 steps, topk 1, 5 draft tokens | 8 (24 GDN state slots) |
| `dflash4b` | Qwen3.5-4B @851bf6e8 | z-lab/Qwen3.5-4B-DFlash @9a1996cc, block 16 | 8 |
| `dflash27b` | Qwen3.8-27B @1d4bf0f2 | DFlash2 @015e7956, block 8 | 16 |

Four draft steps are one more than the bench and profile MTP arms (3 steps, 4 draft
tokens). The extra step does not change positions 1-3: the MTP draft chain for a step
depends only on earlier steps, and the target verify pass is causal, so its head inputs at
verify positions 0-2 are the same computation with one more token appended (up to
kernel-shape numerics). Position 4 adds a deeper, lower-acceptance slot; every result is
reported per position, so the 3-step configuration is the subset of positions 1-3.
