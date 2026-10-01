"""Engine levers for the moonshot portfolio, as composable server-flag overrides.

Each lever is a set of `sglang.launch_server` flag overrides applied on top of one
of the bench arms (`plain`, `mtp` in bench/arms.toml). Levers compose by merging
their overrides, so every stacked configuration is a list of lever names and the
four-way pattern (baseline, A, B, A+B) is a list of four such lists.

`lossy` says what the lever changes numerically relative to the BF16 checkpoint
with an FP32 GDN state; an empty string means the lever keeps the target's
arithmetic up to floating-point reassociation (and a speculative lever keeps the
target's greedy decisions because the target verifies every token).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BENCH_TOKEN_MAPS = Path.home() / 'vp-data/bench/token_map'
DFLASH_DRAFT = 'z-lab/Qwen3.5-4B-DFlash'
DFLASH_DRAFT_REVISION = '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf'
NOTA_DRAFT = 'nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16'
NOTA_DRAFT_REVISION = 'c5fb290e47e30c81d06e48b0495ec06f2560dd4e'
QAD_TARGET = 'nota-ai/Qwen3.5-4B-QAD-W4A16'
QAD_TARGET_REVISION = 'a67b0fedb2b39cb057da6114e656b76b52d321b4'


@dataclass(frozen=True)
class Lever:
    flags: dict[str, str | int | float | bool]
    lossy: str = ''
    note: str = ''
    arm: str = ''  # restricts the lever to one base arm ('mtp' for speculative-only levers)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    model: tuple[str, str] | None = None  # replaces the target checkpoint (path, revision)
    env: dict[str, str] = field(default_factory=dict)  # server environment variables


LEVERS: dict[str, Lever] = {
    # --- GDN recurrent state ---
    'bf16_state': Lever(
        {'mamba-ssm-dtype': 'bfloat16'},
        lossy='GDN recurrent state rounded to BF16 after every update',
    ),
    'fp16_state': Lever(
        {'mamba-ssm-dtype': 'float16'},
        lossy='GDN recurrent state rounded to FP16 after every update (same bytes as BF16, '
        '3 more mantissa bits, narrower range)',
    ),
    'fp8_state': Lever(
        {},
        lossy='GDN recurrent state stored as unscaled FP8 E4M3 (experimental engine patch)',
        env={'SGLANG_MAMBA_SSM_DTYPE': 'float8_e4m3fn'},
    ),
    'replayssm': Lever(
        # ReplaySSM refuses the extra_buffer radix strategy the overlap scheduler
        # otherwise resolves to; no_buffer keeps the radix cache (3 slots/request).
        {'enable-linear-replayssm': True, 'mamba-radix-cache-strategy': 'no_buffer'},
        note='decode reads a checkpoint plus a 16-step ring and writes the state every 16 steps',
        conflicts=('replayssm_spec',),
    ),
    'exact_replay': Lever(
        {
            'enable-linear-replayssm': True,
            'linear-replayssm-cache-len': 4,
            'mamba-radix-cache-strategy': 'no_buffer',
        },
        note='rounding-preserving live replay (patch 0007): FP32 anchor written every 4 '
        'steps, ring of the packed decode operands; meant to be bit-identical',
        # The value tile is pinned here (not inherited from the caller's shell): 32 is the
        # packed decode's tile, the configuration the bit-exactness check covers.
        env={'SGLANG_GDN_EXACT_REPLAY': '1', 'SGLANG_GDN_EXACT_REPLAY_BV': '32'},
        conflicts=('replayssm', 'bf16_state', 'fp16_state', 'fp8_state'),
    ),
    # P4's served A/B pins the pools identically in both arms (BRIEF: equal running limit,
    # KV tokens and mamba slots): 128 requests x (2,048 prompt + 512 output) = 327,680 KV
    # tokens, plus headroom for chunked prefill.
    'p4_pools': Lever(
        {
            'max-running-requests': 128,
            # 360,448 admitted only 127 (run 20261001T082738Z, void): with ignore_eos the
            # scheduler reserves each request's full 512 output tokens and charges a
            # shared-mamba cost per request in token units.
            'max-total-tokens': 655360,
            # 128 slots admitted only 127 long prompts (runs 20261001T082738Z and
            # 20261001T104311Z): with chunked prefill the last admission saw no schedulable
            # mamba slot while one was free. 132 leaves headroom; the running limit stays 128.
            'max-mamba-cache-size': 132,
            'mamba-ssm-dtype': 'float32',
        },
        note='pinned pools for the P4 A/B: 128 running, 655,360 KV tokens, 132 mamba slots, '
        'FP32 state stated explicitly',
    ),
    'replayssm_spec': Lever(
        {'enable-linear-replayssm-spec': True},
        note='chain verify (MTP top-1, DFlash) stores per-draft inputs and folds the '
        'accepted prefix at commit instead of one state snapshot per draft token',
        conflicts=('replayssm', 'tree'),
    ),
    'no_radix': Lever(
        {'disable-radix-cache': True},
        note='one GDN state slot per request instead of five (no prefix reuse)',
    ),
    'tuned_noradix': Lever(
        {'disable-radix-cache': True, 'max-mamba-cache-size': 128},
        note="bench's tuned arms: radix off with one state slot per request (128 slots)",
        conflicts=('no_radix',),
    ),
    'gdn_triton': Lever(
        {'linear-attn-decode-backend': 'triton', 'linear-attn-verify-backend': 'triton'},
        note='Triton GDN decode and verify kernels, pinned so arms on different engines match',
        conflicts=('gdn_verify_triton', 'dflash'),
    ),
    # Exact buffered verify (GDN fold-every-commit): the verify runs the recurrent kernel and
    # writes the raw window to a ring; the commit replays the accepted prefix with a bitwise
    # clone of the recurrent update. It needs the drafter workstream's engine patch, which is
    # not on main yet. On an engine without it the flag alone runs stock ReplaySSM-spec, so
    # check the engine HEAD in each launch record. Exactness is the drafter's claim until its
    # evidence merges.
    'fold': Lever(
        {'enable-linear-replayssm-spec': True},
        note='GDN fold-every-commit buffered verify (exact by construction)',
        arm='mtp',
        conflicts=('replayssm', 'replayssm_spec', 'tree'),
        env={'SGLANG_GDN_REPLAYSSM_FOLD': '1'},
    ),
    # --- weights and KV ---
    'fp8_weights': Lever(
        {'quantization': 'fp8'},
        lossy='online FP8 E4M3 W8A8 linear layers (per-channel weights, dynamic activations)',
        # The aarch64 sgl-kernel CUTLASS FP8 GEMM aborts on this GH200 ("Arch
        # conditional MMA instruction used without targeting appropriate compute
        # capability"), so route the GEMMs to SGLang's Triton W8A8 kernel.
        env={'USE_TRITON_W8A8_FP8_KERNEL': '1'},
    ),
    'fp8_kv': Lever({'kv-cache-dtype': 'fp8_e4m3'}, lossy='FP8 E4M3 attention KV cache'),
    # --- speculation shape ---
    'mtp_s1': Lever(
        {'speculative-num-steps': 1, 'speculative-num-draft-tokens': 2},
        arm='mtp',
    ),
    'mtp_s2': Lever(
        {'speculative-num-steps': 2, 'speculative-num-draft-tokens': 3},
        arm='mtp',
    ),
    # Deeper chains and trees hold one FP32 intermediate GDN state per draft
    # token and request, so they carry a lower capacity.
    'mtp_s5': Lever(
        {
            'speculative-num-steps': 5,
            'speculative-num-draft-tokens': 6,
            'max-running-requests': 64,
            'max-mamba-cache-size': 320,
        },
        arm='mtp',
    ),
    'mtp_s7': Lever(
        {
            'speculative-num-steps': 7,
            'speculative-num-draft-tokens': 8,
            'max-running-requests': 64,
            'max-mamba-cache-size': 320,
        },
        arm='mtp',
    ),
    'tree': Lever(
        {
            'speculative-num-steps': 3,
            'speculative-eagle-topk': 4,
            'speculative-num-draft-tokens': 8,
            'max-running-requests': 64,
            'max-mamba-cache-size': 320,
        },
        arm='mtp',
        conflicts=('replayssm_spec',),
    ),
    'adaptive': Lever(
        {'speculative-adaptive': True},
        note='SGLang adaptive steps: candidate depths chosen by batch size and acceptance',
        arm='mtp',
    ),
    # Arrival batching (scheduling only). At c = 128 a held MTP batch cycles in ~21 ms but
    # the served cycle was estimated at ~35 ms; the hypothesis (untested) is that each
    # arrival's prefill pass interrupts decode. The delayer holds prefill until
    # min(running / 16, N) requests wait (5 s cap), so one pass admits several arrivals.
    # Costs TTFT; report it beside throughput.
    'prefill_delay4': Lever(
        {
            'enable-prefill-delayer': True,
            'prefill-delayer-queue-min-ratio': 0.0625,
            'prefill-max-requests': 4,
        },
        note='prefill delayer: batch arrivals, up to 4 per prefill pass',
    ),
    'prefill_delay8': Lever(
        {
            'enable-prefill-delayer': True,
            'prefill-delayer-queue-min-ratio': 0.0625,
            'prefill-max-requests': 8,
        },
        note='prefill delayer: batch arrivals, up to 8 per prefill pass',
        conflicts=('prefill_delay4',),
    ),
    # Host-overhead levers for the speculative cycle (profile: MTP at B=1 idles the
    # GPU ~25% of the cycle while the host prepares verify).
    'plan_stream': Lever(
        {},
        note='SGLANG_ENABLE_OVERLAP_PLAN_STREAM: verify metadata planning on its own stream',
        env={'SGLANG_ENABLE_OVERLAP_PLAN_STREAM': '1'},
    ),
    'glue_graph': Lever(
        {},
        note='SGLANG_ENABLE_METADATA_GLUE_GRAPH: attention-metadata prep captured in a graph',
        env={'SGLANG_ENABLE_METADATA_GLUE_GRAPH': '1'},
    ),
    'draft_attn_triton': Lever(
        {'speculative-draft-attention-backend': 'triton'},
        note='Triton attention for the MTP draft layer: no host-side FlashInfer plan '
        '(profile: kv_indptr .cpu() sync before every draft replay)',
    ),
    'attn_triton': Lever(
        {'attention-backend': 'triton'},
        note='Triton attention for target verify and draft: no FlashInfer plan syncs',
    ),
    'spec_attn_decode': Lever(
        {'speculative-attention-mode': 'decode'},
        note='verify and draft extend use the decode attention backend path',
    ),
    'ngram': Lever(
        {
            'speculative-algorithm': 'NGRAM',
            'speculative-num-draft-tokens': 12,
            # 12 FP32 intermediate GDN states per request bound capacity.
            'max-running-requests': 32,
            'max-mamba-cache-size': 160,
        },
        note='prompt/output n-gram drafts in a BFS tree (SGLang defaults)',
        arm='plain',
    ),
    'dflash': Lever(
        {
            'speculative-algorithm': 'DFLASH',
            'speculative-draft-model-path': DFLASH_DRAFT,
            'speculative-draft-model-revision': DFLASH_DRAFT_REVISION,
            'speculative-dflash-block-size': 16,
            # The drafter workstream's validated flags
            # (evidence/drafter/launch/zlab_b16_panel_trace.json).
            'linear-attn-prefill-backend': 'flashinfer',
            'linear-attn-decode-backend': 'flashinfer',
            # Block 16 at radix off caps capacity at 64 (bench's dflash arm).
            'disable-radix-cache': True,
            'max-running-requests': 64,
            'max-mamba-cache-size': 64,
        },
        note='public BF16 DFlash block drafter (z-lab); the target verifies, so exact',
        arm='plain',
        env={'SGLANG_ENABLE_OVERLAP_PLAN_STREAM': '1'},
    ),
    'gdn_verify_triton': Lever(
        {'linear-attn-verify-backend': 'triton'},
        note='GDN target verify on the Triton recurrent kernel instead of FlashInfer MTP '
        '(repair, evidence/repair/stage_a_timing.json: DFlash verify pass at one request '
        '4.78 -> 4.55 ms at B=16, 15.62 -> 7.58 at 64, 44.82 -> 19.18 at 256); class pending '
        "bench's equality classification (it changes the target's GDN verify kernel)",
    ),
    'dflash_nota': Lever(
        {
            'speculative-algorithm': 'DFLASH',
            'speculative-draft-model-path': NOTA_DRAFT,
            'speculative-draft-model-revision': NOTA_DRAFT_REVISION,
            'speculative-dflash-block-size': 16,
            'max-running-requests': 32,
            'max-mamba-cache-size': 160,
        },
        note='nota-ai INT4 DFlash drafter tuned for their QAD INT4 target; exact on any target',
        arm='plain',
        conflicts=('dflash',),
    ),
    'qad_target': Lever(
        {},
        lossy='target replaced by nota-ai QAD INT4 W4A16 (quantization-aware distilled)',
        model=(QAD_TARGET, QAD_TARGET_REVISION),
    ),
    'relax1': Lever(
        {},
        lossy='relaxed greedy verification: accept a draft within 1.0 logit of the argmax',
        note='chain drafts only (MTP top-1, DFlash); needs engine/moonshot patches 0002 and 0004',
        env={'SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP': '1.0'},
    ),
    'relax2': Lever(
        {},
        lossy='relaxed greedy verification: accept a draft within 2.0 logits of the argmax',
        note='chain drafts only (MTP top-1, DFlash); needs engine/moonshot patches 0002 and 0004',
        env={'SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP': '2.0'},
    ),
    # Hot-vocabulary draft heads. Maps from the bench workstream: token frequencies of
    # 3.1M natural-length MTP outputs on the mixed-v1 tune split
    # (~/vp-data/bench/token_map/hot32k_tune.json; 16,384 rows cover 99.8% of them).
    'hot8k': Lever(
        {'speculative-token-map': str(BENCH_TOKEN_MAPS / 'hot8192_tune.pt')},
        note='draft head restricted to 8,192 frequent rows (MTP: patch 0001; DFlash: 0005)',
        conflicts=('hot16k',),
    ),
    'hot23k': Lever(
        {'speculative-token-map': str(BENCH_TOKEN_MAPS / 'hot32k_tune.pt')},
        note='draft head restricted to the 22,936 tokens seen in the tune outputs '
        '(held-out coverage 97.5%, evidence/moonshot/token_map_coverage.csv)',
        conflicts=('hot8k', 'hot16k'),
    ),
    'hot16k': Lever(
        {'speculative-token-map': str(BENCH_TOKEN_MAPS / 'hot16384_tune.pt')},
        note='draft head restricted to 16,384 frequent rows (MTP: patch 0001; DFlash: 0005)',
        conflicts=('hot8k',),
    ),
}


def compose(names: list[str], base: str) -> dict[str, str | int | float | bool]:
    """Merged flag overrides for a lever list on a base arm, refusing conflicts."""
    flags: dict[str, str | int | float | bool] = {}
    for name in names:
        lever = LEVERS[name]
        if lever.arm and lever.arm != base:
            raise ValueError(f'lever {name} needs base arm {lever.arm}, got {base}')
        clash = set(lever.conflicts) & set(names)
        if clash:
            raise ValueError(f'lever {name} conflicts with {sorted(clash)}')
        flags.update(lever.flags)
    return flags


def target_model(names: list[str]) -> tuple[str, str] | None:
    models = {LEVERS[n].model for n in names if LEVERS[n].model}
    if len(models) > 1:
        raise ValueError(f'levers {names} replace the target with different checkpoints')
    return models.pop() if models else None


def environment(names: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in names:
        env.update(LEVERS[name].env)
    return env


def lossy_label(names: list[str]) -> str:
    return '; '.join(LEVERS[n].lossy for n in names if LEVERS[n].lossy)
