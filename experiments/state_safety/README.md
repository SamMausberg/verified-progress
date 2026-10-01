# State safety of speculative decoding (hypothesis H5)

These scripts test whether SGLang keeps the hybrid Gated DeltaNet (GDN) plus
attention state of Qwen3.5-4B correct under native MTP speculation, and measure
how often stock configurations that ought to give the same greedy output
disagree. Results and their interpretation are in
[`evidence/state_safety/README.md`](../../evidence/state_safety/README.md).

## What is compared

Every server gets the same 320 prompts as token IDs (built once by
`prompts.py` from pinned public datasets; manifest in
`evidence/state_safety/prompt_manifest.json`) and generates up to 256 tokens
greedily with top-5 target logprobs. Two runs of a prompt are compared
position by position. The first position where the tokens differ is a
divergence; up to that point both runs have identical prefixes, so each run's
own logprobs give its margin between the two competing tokens there.

Logits leave the LM head in BF16, so logprob gaps between tokens are integer
multiples of the BF16 spacing at the logit's magnitude (0.125 for logits in
[16, 32)). `compare.py` classifies each divergence:

| Class | Meaning |
|---|---|
| `tie` | one run had an exact BF16 tie between the two tokens; lowest-index tie-breaking picked its token |
| `one_ulp` | both runs preferred their token by at most one BF16 spacing |
| `near` | both margins at most 0.5 nats |
| `large` | anything else; not explained by rounding, investigated as a candidate state error |
| `not_argmax` | a run committed a token that is not its own top-1 (must never happen under greedy decoding) |

It also records the drift on the common prefix: the largest logprob
difference between the runs over competitive tokens (logprob above -4). A
state error perturbs the hidden state and shows up as drift even when no token
flips. `cycles.py` buckets positions of a speculative run by the previous
verify cycle's commit length (which draft position was rejected) and the
offset inside the current cycle, so a rollback error for one accept length
would stand out.

## Server configurations

`server.py` defines them. All share the model revision, FlashInfer attention,
the Triton GDN kernels, `--mem-fraction-static 0.25` (shared GPU lock),
`--max-running-requests 16` and `--mamba-full-memory-ratio 2` (so plain decode
and MTP have the same batch cap; the default cap differs), and
`--incremental-streaming-output`, which only changes how the HTTP stream
packages tokens. Each streamed chunk then carries one verify cycle, which is
how cycle boundaries are recovered; `compare.spec_cycles_consistent` checks
that the chunk count matches the server's `spec_verify_ct`.

| Name | Extra flags |
|---|---|
| `plain` | none (radix cache with the `extra_buffer` GDN strategy, overlap scheduler, CUDA graphs) |
| `plain_noradix` | `--disable-radix-cache` |
| `plain_nooverlap` | `--disable-overlap-schedule` |
| `plain_det` | `--enable-deterministic-inference` (with FlashInfer this also disables the radix cache) |
| `plain_fp32head` | `--enable-fp32-lm-head` |
| `mtp_s1`, `mtp_s3`, `mtp_s5` | `--speculative-algorithm EAGLE` (native MTP), steps 1/3/5, top-k 1, steps+1 draft tokens |
| `mtp_tree` | steps 3, top-k 2, 6 draft tokens |
| `mtp_s3_noradix`, `mtp_s3_nooverlap`, `mtp_s3_det`, `mtp_s3_fp32head` | `mtp_s3` plus the flag above |

A pass `cN` sends all prompts with N requests in flight after flushing the
radix cache; `cN_warm` repeats them without a flush, so every prompt prefix
the GDN checkpointing kept is served from the cache.

## Targeted tests

`targeted.py` (driven by `run_targeted.sh`):

- `truncation`: `max_new_tokens` chosen so the request ends after every
  possible number of tokens of its final verify cycle. The output must equal
  the prefix of an untruncated run on the same server, bitwise.
- `stops`: a stop token that is first generated at every possible index
  inside a verify cycle, so drafts after it were accepted and folded into the
  GDN state before the stop was detected. The output must equal the untruncated
  prefix; the conversation is then extended with a new user turn and served
  warm (radix cache) and cold (after a flush).
- `prefix`: long generations cross the GDN checkpoint interval (256 tokens)
  during speculative decode. Prefixes ending at and just past each checkpoint
  are served warm, restoring the checkpointed state, and cold; also for
  checkpoints taken in a cycle that finished the request.
- `abort`: requests are aborted mid-stream while the batch and the GDN pool
  (exactly four slots) are full, so the next request reuses the freed slot
  while the aborted request's last verify may still be in flight. Probe
  outputs are compared with the same probes served alone.
- `prefill`: prompt logprobs at every position of the 40 longest prompts,
  compared across `--chunked-prefill-size` settings by `compare_prefill.py`.

## Commands

Run from the repository root. GPU steps take the shared lock themselves.

```sh
source ~/verified-progress/scripts/sglang_env.sh
python experiments/state_safety/prompts.py \
    --out ~/vp-data/state/prompts/prompts.jsonl \
    --manifest evidence/state_safety/prompt_manifest.json
experiments/state_safety/run_all.sh          # differential matrix, about 2 h of GPU time
experiments/state_safety/run_targeted.sh     # targeted tests, about 1 h

cd experiments/state_safety
python compare.py --pairs pairs.json \
    --out-json ../../evidence/state_safety/noise_floor.json \
    --out-table ../../evidence/state_safety/noise_floor.csv \
    --out-csv ../../evidence/state_safety/divergences.csv
python cycles.py --ref plain/c1 --spec mtp_s3/c1 \
    --out ../../evidence/state_safety/cycles_mtp_s3.json
python summarize_targeted.py --out ../../evidence/state_safety/targeted.json
```

Raw outputs stay in `~/vp-data/state/` (`runs/`, `targeted/`); each run has a
`.meta.json` with the flags, resolved server settings, and the repository and
SGLang commits.
