# Moonshot: levers, ceilings and reformulations of the serving stack

Scripts for the moonshot portfolio: derived ceilings per lever stack, served measurements of
engine levers through the bench harness, their quality checks, and kernel-level tests of two
reformulations of the Gated DeltaNet (GDN) state, rounding-preserving replay (P4) and
block-parallel verification (P7). Results, the declared P4 test and the exact commands are in
[`evidence/moonshot/README.md`](../../evidence/moonshot/README.md); most scripts also give
their command in the docstring. The levers that need engine changes come from the patch series
`engine/sglang/patches/moonshot/`, applied in `~/sglang-wt/moonshot`
(`engine/sglang/README.md`); raw outputs stay in `~/vp-data/moonshot/`.

## Files

| File | Role | Evidence |
|---|---|---|
| `levers.py` | Each lever as server-flag and environment overrides on a bench arm, with what it changes numerically (`lossy`) | definitions, no output |
| `lever_sweep.py` | Runs `<arm>[+lever...]` configurations through `bench.sweep` in one exclusive hold | raw sweeps |
| `server_env.py` | bench's server plus a record of the server's numerics-relevant environment variables | `launch.json` of each run |
| `summarise.py` | Tables from raw results: `ceiling` (decode-step sweeps), `sweeps` (lever sweeps), `quality` (probe summaries) | `decode_ceiling_try1.csv`, `lever_sweeps_quick.csv`, `host_levers.csv` |
| `ceilings.py` | Derived step-time floors and throughput ceilings per lever stack under two execution models (a calculation, not a measurement) | `ceilings.json`, `ceilings.csv` |
| `decode_ceiling_sweep.py` | Engine-only decode-step latency against batch size per configuration (`sglang.benchmark.one_batch`) | raw, summarised into `decode_ceiling_try1.csv` |
| `gdn_exact_replay_check.py` | P4: bit-exactness (`check`) and kernel time (`bench`) of exact replay and ReplaySSM against SGLang's packed GDN decode | `gdn_exact_replay_check_*.json`, `gdn_exact_replay_bench.json` |
| `gdn_fast_verify_check.py` | P7: disagreement and time of SGLang's chunked GDN kernel used as a block-parallel verify path | `p7_fast_verify_check.json`, `p7_verify_width_bench.json` |
| `token_map_coverage.py` | Held-out coverage of bench's hot-vocabulary draft maps on plain outputs of the confirm split | `token_map_coverage.csv` |
| `build_token_map.py` | Builds hot-vocabulary maps for the MTP draft head from the model's own outputs on a calibration set | maps in `~/vp-data/moonshot/token_map/` |
| `logit_probe.py` | Quality proxy: first greedy divergence and top-k KL against a reference server, in decode and teacher-forced modes | pending |
| `quality_arms.py` | Runs the logit probe (and optionally the token-map calibration) for a list of lever stacks, one small server each | pending |
| `gsm8k_arms.py` | GSM8K accuracy per lever stack through `bench.quality`, for the declared quality budget | pending |
| `make_long_prompts.py` | The 2,048-token prompts of P4's served test, built from a bench split | input of the P4 test |
| `run_p4_admission.sh`, `check_admission.py` | P4's admission preflight: every arm's server must run 128 requests at once during the measured phase | pending |
| `run_p4b.sh` | P4's served test in one exclusive hold: kernel checks, admission, output probe, then the paired A/B | pending |
| `output_probe.py` | Outcome of P4's server output probe (refuted, undecided or no difference) | pending |
| `validate_p4_ab.py` | Validates one P4 A/B run and applies the declared decision rule | pending |

"Pending" means no committed evidence yet; `evidence/moonshot/README.md` gives each run's status.
Tests: `tests/test_moonshot_levers.py` (lever composition and the engine patches),
`tests/test_gdn_exact_replay.py` (the replay kernel) and `tests/test_moonshot_check_admission.py`,
`test_moonshot_output_probe.py` and `test_moonshot_validate_p4_ab.py` (the P4 checks); the engine
and kernel tests skip without SGLang or CUDA.
