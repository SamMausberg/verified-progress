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

| File | What it does |
|---|---|
| `serve_run.py` | Starts one SGLang server (arms `plain`, `mtp`, `dflash`, matching the bench harness), runs client commands, stops it; writes `launch.json` with the command, environment and engine revision |
| `accept_probe.py` | Greedy `/generate` probe: output ids, optional top-2 logprobs, per-request speculative counters; summary with acceptance by block position per domain |
| `summarize_acceptance.py` | Probe summaries to `acceptance_by_position.csv` and `acceptance_summary.csv` |
| `compare_outputs.py` | First token divergence between two probe runs, classified by the reference run's top-2 logprob gap |
| `build_panel.py`, `panel-v1.jsonl` | Characterization panel shared with the repair workstream: 32 MATH-500 problems (seeded sample) and the first 16 chat, code and maths prompts of the bench confirm split |
| `run_trace.sh` | Per-cycle trace (drafted tokens, target argmax at every block position, accepted length) on the panel at blocks 16 and 8; needs the engine patch below |
| `build_train_prompts.py` | Training prompts (chat, code, maths) from permissive public sources, disjoint from every bench split and the panel |
| `gen_targets.py`, `run_gen_segment.sh` | The target's own greedy thinking-mode responses to the training prompts, generated with SGLang in resumable segments |
| `train_dflash.py` | Fine-tunes a DFlash or DFlash 2 drafter against the frozen target in resumable time-boxed segments; exports a checkpoint SGLang loads unchanged |

## Engine patch

`engine/sglang/patches/drafter-0001-dflash-cycle-trace.patch` adds an opt-in
trace (`SGLANG_DFLASH_TRACE_PATH`) to SGLang's DFlash worker. It changes nothing
unless the variable is set; with it set, every verify cycle synchronizes the
stream, so traced runs give no timings.

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
