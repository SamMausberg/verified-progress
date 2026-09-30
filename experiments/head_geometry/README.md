# Real-head replay: transport versus self-evidence

This experiment decides H2 and H3 in `TASKS.md` on real hidden states. It captures the
exact inputs of the shared LM head from a running SGLang server, then evaluates, offline
and in FP64, two ways of certifying the head's decision without computing it densely:

- **Transport (H2, the manuscript's Section 4):** draft-produced tile summaries
  `M_c^d`, `Z_c^d` moved to the target state with `|z_t - z_d - <mu_c, Delta>| <= eps_c`.
- **Self-evidence (H3):** a low-precision copy of the head with a rigorous per-row
  envelope `|z_i - z_hat_i| <= beta_i`, and exact rescoring of the rows whose interval
  can still win (greedy argmax, or Gumbel-max with the same noise field).

Results and their interpretation are in `evidence/head_geometry/README.md`.

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
| `analyze_stats.py` | Norms, drift, margins, top-m mass, hidden-dimension outliers |

`tests/test_head_geometry_bounds.py` checks that every bound encloses exact values on
random BF16 heads (it skips without torch).

## Why an engine patch rather than `return_hidden_states`

SGLang's `--enable-return-hidden-states` returns the target's stored hidden states for
generated tokens. It does not return the MTP draft's head inputs, it returns the
auxiliary layer features rather than the head input for DFlash targets, and it carries
no draft tokens or accept lengths per verify step. The patch
(`engine/sglang/patches/0001-head-capture-replay-dumps.patch`) instead stashes the
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

All GPU commands go through the lock. Captures use `SGLANG_WORKTREE=~/sglang-wt/geometry`
with the patch applied; raw dumps go to `~/vp-data/geometry/` (outside git).

```sh
source scripts/sglang_env.sh
python experiments/head_geometry/build_prompts.py --out ~/vp-data/geometry \
    --manifest evidence/head_geometry/prompt_manifest.csv

scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh plain4b
scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh mtp4b
scripts/gpu_lock.sh -x experiments/head_geometry/run_capture.sh dflash27b

cd experiments/head_geometry
../../scripts/gpu_lock.sh -s python validate_alignment.py --arm plain4b \
    --out ../../evidence/head_geometry/alignment_plain4b.json
../../scripts/gpu_lock.sh -s python validate_alignment.py --arm mtp4b \
    --out ../../evidence/head_geometry/alignment_mtp4b.json
../../scripts/gpu_lock.sh -s python analyze_selfevidence.py --sets plain verify draft \
    --out ../../evidence/head_geometry
../../scripts/gpu_lock.sh -s python analyze_transport.py --arm mtp4b \
    --out ../../evidence/head_geometry
../../scripts/gpu_lock.sh -s python analyze_stats.py --arm mtp4b \
    --out ../../evidence/head_geometry/stats_4b.json
```

The 27B analyses use `--arm dflash27b` for `validate_alignment.py`,
`analyze_transport.py` and `analyze_stats.py`; the 27B head in FP64 is 10 GB, so they
run under `-x` when the shared budget (20 GB) is tight.

## Server configuration used for capture

Correctness capture, not a performance configuration: CUDA graphs and the overlap
scheduler are off because the hooks run in Python between forwards. Attention backend
`flashinfer` (no FA3 on aarch64). Greedy decoding, 384 new tokens, the model's chat
template with thinking enabled for one third of the prompts (assigned by seed).

| Arm | Model | Speculation | Concurrency |
|---|---|---|---|
| `plain4b` | Qwen3.5-4B @851bf6e8 | none | 16 |
| `mtp4b` | Qwen3.5-4B @851bf6e8 | NEXTN, 4 steps, topk 1, 5 draft tokens | 8 (24 GDN state slots) |
| `dflash27b` | Qwen3.8-27B @1d4bf0f2 | DFlash2 @015e7956, block 8 | 16 |

Four draft steps are one more than the untuned bench default (3); positions 1-3 are
computed identically, and position 4 adds a deeper, lower-acceptance slot.
