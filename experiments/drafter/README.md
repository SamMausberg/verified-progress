# Drafter experiments (moonshot M3)

Tools for serving, characterizing and fine-tuning a block drafter for
`Qwen/Qwen3.5-4B@851bf6e`. The starting point is the public DFlash drafter
`z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`
(Apache-2.0; `modal-labs/Qwen3.5-4B-DFlash@58aa4cb` has byte-identical weights,
sha256 `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42`).
Results and their exact commands are in `evidence/drafter/README.md`.

Run everything from the repository root in the SGLang venv
(`source scripts/sglang_env.sh`) and every GPU command under
`scripts/gpu_lock.sh` (`-s` for correctness work with `--mem 0.25`, `-x` for
anything timed).

## Files

Serving, probes and panels:

| File | What it does |
|---|---|
| `serve_run.py` | Starts one SGLang server (arms `plain`, `mtp`, `dflash`, matching the bench harness), runs client commands, stops it. Takes the start-up lock through `scripts/gpu_startup_lock.sh` (`--min-free-gb`), retries while shared jobs leave too little memory, and writes `launch.json` with the command, environment, engine revision and the pools the server resolved (KV tokens, mamba slots, running limit) |
| `accept_probe.py` | Greedy `/generate` probe: output ids, optional top-5 logprobs, per-request speculative counters; summary with acceptance by block position per domain. `--waves N` sends each wave of N requests as one batched request, so batch composition is the same in every run |
| `summarize_acceptance.py` | Probe summaries to `acceptance_by_position.csv` and `acceptance_summary.csv` |
| `compare_outputs.py` | First token divergence between two probe runs, classified with both runs' margins (#37's classes); counts sequences that are bitwise identical in tokens and top-5 logprobs |
| `build_panel.py`, `panel-v1.jsonl`, `panel-v2.jsonl` | The two panels: 32 MATH-500 problems (seeded sample) plus the first 16 chat, code and maths prompts of a bench confirm split (panel-v1: mixed-v1 at commit 8b4b7ab, shared with the repair workstream's trace; panel-v2: mixed-v2, the evaluation panel for trained drafters). Acceptance only, never quality |
| `mtbench-first-turn.jsonl` | The 80 MT-Bench first turns of the card-reproduction gate |
| `run_card_gate.sh` | The card gate: MT-Bench first turns with the model card's settings (block 16, up to 4,096 tokens, concurrency 1) |
| `run_trace.sh`, `trace_analysis.py` | Per-cycle trace (drafted tokens, target argmax at every block position, accepted length) on panel-v1 at blocks 16 and 8, needs engine patch 0001; tau by thinking/answer segment and output offset, and the first rejected draft's relation to the target's neighbouring tokens |
| `run_equality.sh` | Panel-v2 baselines with top-5 logprobs on every run: plain decoding, native MTP (3 steps), DFlash block 16, and the output comparison against plain |
| `run_eval_panel.sh` | Acceptance of one drafter checkpoint on panel-v2 and its output comparison with the plain reference |
| `run_timed_panel.sh` | Untraced DFlash runs on panel-v1 at concurrency 1 (blocks 16 and 8) for per-request cycle time and tokens per cycle; exclusive hold |

Buffered GDN verify (engine patches 0002 and 0003):

| File | What it does |
|---|---|
| `gdn_verify_parity.py`, `run_gdn_parity.sh` | Kernel-level parity at the Qwen3.5-4B GDN shape on random inputs: the fold verify and committed state, and the circular verify, against the stock recurrent verify, bitwise |
| `gdn_ring_tile_sweep.py`, `run_ring_tile_sweep.sh` | GPU time per layer of the fold's ring-writing verify at forced value tiles 4, 8, 16 and 32 for batches 1-64 and blocks 16 and 8 (24 layers per CUDA graph, repeated), with the stock per-position-state verify as reference and each tile's output checked bitwise against tile 32's; exclusive hold |
| `run_replay_check.sh`, `run_replay_check_mtp.sh` | Served exactness of the circular and fold arms against stock DFlash (block 16) and stock MTP s3 on panel-v2 at concurrency 1 and 8, with a stock-repeat control |
| `summarize_replay_check.py` | One table from those runs: bitwise-identical sequences, first-divergence classes, tokens per cycle, foreign CPU load and the resolved pools of each arm |
| `run_fold_localize.sh`, `fold_localize.py` | Fold against stock with identical pinned pools: traced runs at concurrency 1 located cycle by cycle, and deterministic batched waves (DFlash waves of 4, MTP waves of 8) with a stock repeat |
| `run_phase_timing.sh`, `phase_summary.py` | Per-phase GPU time of the DFlash cycle (draft, verify, commit, ...) for stock, circular and fold at concurrency 8 and 16 with identical pinned pools (the repair workstream's CUDA-event probe), and the GDN kernels each arm runs; exclusive hold |
| `run_fold_timing.sh`, `ab_timing_summary.py` | Serving A/B of the fold against stock verify on the bench's tuned DFlash arms (blocks 16 and 8), c = 1-32, order stock, fold, fold, stock; the summary gives per-run throughput, tokens per cycle, the ratio of the means and its range over run pairs, foreign CPU load and pools; exclusive hold |
| `run_fold_check.sh` | The fold's exactness after an engine change: the kernel parity check, then `run_fold_localize.sh`'s matched-pool served check; shared slot |
| `fold_timing_check.py` | Checks a `run_fold_timing.sh` output against its declared protocol (runs and their order, each server's resolved arguments, engine and repository commits, launch checks, pools, points, prompts, throughput recomputed from per-request records), then gives the fold/stock ratios with their ABBA pairs and, optionally, the comparison with an earlier session; writes the launch records; CPU |

Training data, training and the P6 screen:

| File | What it does |
|---|---|
| `build_train_prompts.py` | Training prompts (chat, code, maths) from permissive public sources; fails unless they are disjoint (text and id) from every bench split of mixed-v1 and mixed-v2, both panels, MT-Bench, GSM8K test and MATH-500 |
| `gen_targets.py`, `run_gen_segment.sh`, `run_gen_all.sh` | The target's own greedy thinking-mode responses to the training prompts, generated with SGLang in resumable segments, each under its own shared GPU-lock ticket |
| `train_dflash.py` | Fine-tunes a DFlash or DFlash 2 drafter against the frozen target; exports a checkpoint SGLang loads unchanged |
| `train_selector.py` | Trains a DFlash 2 candidate selector over the frozen public drafter's top-16 candidates (P6); several objectives side by side on the same data |
| `run_train_segments.sh`, `train_segment.sh` | Runs a resumable trainer in successive time-boxed shared GPU-lock segments |
| `run_train_smoke.sh` | Shared-slot smoke test of the training stack and the support screen |
| `support_screen.py`, `run_support_screen.sh` | Zero-training support screen (P6): U_K, the longest realized-greedy prefix inside the drafter's top-K candidates, per traced cycle; saves the per-cycle table `cycles.pt` for P9 |
| `run_selector_rewalk.sh` | A trained selector's greedy walk over the same frozen candidates (L_sel) and the corrected re-walk after early rejections for P9 (`selector_rewalk.pt`) |
| `rho_pairs.py` | Teacher-forced draft and target head inputs per block slot, and rho = relative head-input distance, with acceptance labels |
| `drafting_requirement.py` | Derived, no GPU: the tokens per cycle and the constant per-position acceptance a block drafter needs for a 5x end-to-end gain at each width, from the repair workstream's measured cycle periods, against the measured block-16 acceptance and its top-16 support bound |

## Engine patches

`engine/sglang/patches/drafter/` (apply commands and details in
`engine/sglang/README.md`):

- 0001 adds an opt-in trace (`SGLANG_DFLASH_TRACE_PATH`) to SGLang's DFlash worker.
  It changes nothing unless the variable is set; with it set, every verify cycle
  synchronizes the stream, so traced runs give no timings.
- 0002 lets `--enable-linear-replayssm-spec` run with DFLASH on GDN models (the
  circular ring commit; not bitwise).
- 0003 adds `SGLANG_GDN_REPLAYSSM_FOLD=1`: SGLang's fold-every-commit protocol for
  GDN pools, for DFLASH and EAGLE/MTP.
- 0004 takes the DFLASH ReplaySSM commit only under `--enable-linear-replayssm-spec`
  (plain `--enable-linear-replayssm` also allocates replay rings).
- 0005 gives the fold's ring-writing verify the stock verify's narrow value tiles on
  sm_90 at small batches.

## Training environment

`train_dflash.py` runs in the SGLang venv (torch 2.13.0+cu130, transformers
5.12.1) with two additions on `PYTHONPATH`, neither installed into the venv:

    pip install --no-deps --target ~/vp-data/drafter/pylib \
        fla-core==0.5.2 flash-linear-attention==0.5.2   # GDN kernels for the HF target
    git clone https://github.com/sgl-project/SpecForge ~/vp-data/drafter/src/SpecForge
    git -C ~/vp-data/drafter/src/SpecForge checkout 3cb0510
    export PYTHONPATH=~/vp-data/drafter/pylib:~/vp-data/drafter/src/SpecForge

The DFlash objective, anchor sampling, attention masks and per-position metrics
are SpecForge's `OnlineDFlashModel` (MIT); the script supplies the frozen
Hugging Face target, the data, the single-GPU optimizer loop, checkpointing and
the export.
