# Lossy levers: quality for speed

Two levers that change the model's arithmetic, measured end to end against the best tuned
exact arms of `bench/` and charged against the quality budget declared in
`evidence/moonshot/README.md` ("Quality budget for the lossy stack"):

1. the INT4 quantization-aware-distilled target `nota-ai/Qwen3.5-4B-QAD-W4A16` with its INT4
   DFlash drafter `nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16` (moonshot portfolio row 6);
2. the GDN recurrent state stored in FP16 instead of the checkpoint's FP32, with capacity
   256 (moonshot portfolio row 3).

The arms are in `bench/arms.toml` (section "Lossy arms"). The INT4 DFlash arms need engine
patch `engine/sglang/patches/lossy/0001` (see `engine/sglang/README.md`); without it SGLang
loads the drafter's quantized context projection into nothing and serves an uninitialised
one. Every hold runs every arm, exact or lossy, on that one engine worktree.

| File | Role |
|---|---|
| `hold.sh` | Entry point of every GPU hold; checks that this checkout is clean and the engine worktree is at the declared commit |
| `load_test.py` | Load test L0: launch checks, tokenizer identity, greedy smoke requests with accept length, an Nsight Systems kernel table for the INT4 drafted arm, and the logit probe's reference runs |

## Load test L0 (untimed)

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh load
```

For each step in `load_test.STEPS` it launches the arm through `bench.server` (decode and
prefill CUDA graphs, overlap scheduler, capacity and backend checks, recorded but not
enforced, so a failure is a result), and then:

- reads the weight memory, quantization method and GDN state size from the server log;
- checks that the server's token ids for 16 templated tune-split prompts equal those of the
  `Qwen/Qwen3.5-4B` tokenizer (the INT4 checkpoint ships its own `tokenizer.json`; the chat
  template, vocabulary and merges files are byte-identical);
- sends 8 greedy 256-token requests (thinking on, 8 at once) and records output lengths,
  text excerpts and, under speculation, the accept length;
- for `int4-dflash-b16-nsys`, collects 20 scheduler steps under Nsight Systems with CUDA
  graph nodes traced and lists the GPU kernels by time, which shows whether the weight
  GEMMs run on the W4A16 Marlin kernel or on dense BF16 GEMMs;
- for `plain-ref-1` and `plain-ref-2` (two launches of the stock arm `plain-cap256`), runs
  `experiments/moonshot/logit_probe.py` in generate and score mode: the reference of the
  probe and its run-to-run noise. No lossy arm is probed in L0.

Nothing in L0 is timed. Its output is `~/vp-data/lossy/load_test/<UTC>/load_test.json`.
