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

TOKEN_MAP_DIR = Path.home() / 'vp-data/moonshot/token_map'
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
        {'enable-linear-replayssm': True},
        note='decode reads a checkpoint plus a 16-step ring and writes the state every 16 steps',
        conflicts=('replayssm_spec',),
    ),
    'replayssm_spec': Lever(
        {'enable-linear-replayssm-spec': True},
        note='MTP verify stores raw per-draft inputs and folds the accepted prefix at commit',
        arm='mtp',
        conflicts=('replayssm', 'tree'),
    ),
    'no_radix': Lever(
        {'disable-radix-cache': True},
        note='one GDN state slot per request instead of five (no prefix reuse)',
    ),
    # --- weights and KV ---
    'fp8_weights': Lever(
        {'quantization': 'fp8'},
        lossy='online FP8 E4M3 W8A8 linear layers (per-channel weights, dynamic activations)',
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
            # 16 FP32 intermediate GDN states per request bound capacity.
            'max-running-requests': 32,
            'max-mamba-cache-size': 160,
        },
        note='public BF16 DFlash block drafter (z-lab); the target verifies, so exact',
        arm='plain',
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
    'hot32k': Lever(
        {'speculative-token-map': str(TOKEN_MAP_DIR / 'qwen3_5_4b_hot32768.pt')},
        note='draft head restricted to 32,768 frequent rows (needs the tied-head patch)',
        arm='mtp',
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
