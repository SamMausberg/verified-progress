# Lossy levers: evidence

Results of the study described and pre-registered in `experiments/lossy/README.md`: the INT4
quantization-aware-distilled target with its INT4 DFlash drafter, and the FP16 GDN state, against
bench's tuned exact arms. Hardware, engine and client are as in `evidence/bench/README.md`; the
engine is `57560de690` (`bd66ce343e` + `engine/sglang/patches/lossy/0001`) for every arm.
Raw run directories stay in `~/vp-data/lossy/` on the GH200 host.

Status: load test and checkpoint check done; the timed sessions and quality holds are pending.

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
