"""Declared plan of the lossy-lever study: holds, arms, concurrencies, pins.

The pre-registration in experiments/lossy/README.md describes this plan in prose;
the hold runner (run_hold.py) and the analysis (analyze.py) read it from here,
so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE_WORKTREE = Path.home() / 'sglang-wt/lossy'
# bd66ce343e + engine/sglang/patches/lossy/0001; every arm, exact or lossy, runs on it.
ENGINE_COMMIT = '57560de6907090a34eb21774ba13dbe2a6b89b80'
ENGINE_BASE = 'bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824'
PORT = 30101
SESSIONS = ('lossy-s1', 'lossy-s2', 'lossy-s3')
OSL = 512


@dataclass(frozen=True)
class Launch:
    arm: str
    concurrency: tuple[int, ...]


# One timed session = one exclusive hold; lossy-s2 runs the list in reverse order.
# The two arms at the ends carry no headline point, so a hold that runs out of time
# loses a non-headline launch in either order (pre-run revision 2, 2026-10-02).
SESSION_LAUNCHES: tuple[Launch, ...] = (
    Launch('int4-plain-cap256', (64, 128, 256)),
    Launch('dflash-tuned-b16', (1, 2, 4)),
    Launch('int4-dflash-b16', (1, 2, 4, 8, 16, 32)),
    Launch('dflash-tuned', (8, 16, 32)),
    Launch('int4-dflash-b8', (8, 16, 32, 64, 128)),
    Launch('plain-tuned', (64, 128)),
    Launch('replayssm-cap256', (64, 128, 256)),
    Launch('plain-cap256-fp16', (64, 128, 256)),
    Launch('replayssm-cap256-fp16', (64, 128, 256)),
    Launch('plain-cap256', (64, 128, 256)),
)

# Arms that failed their load test (L0) launch checks; they are not timed.
DROPPED_ARMS: tuple[str, ...] = ()

EXACT_ARMS = (
    'dflash-tuned-b16',
    'dflash-tuned',
    'plain-tuned',
    'plain-cap256',
    'replayssm-cap256',
)

# Matched pairs: identical flags apart from the lever (model weights or state dtype).
PAIRS: tuple[tuple[str, str], ...] = (
    ('int4-dflash-b16', 'dflash-tuned-b16'),
    ('int4-dflash-b8', 'dflash-tuned'),
    ('int4-plain-cap256', 'plain-cap256'),
    ('plain-cap256-fp16', 'plain-cap256'),
    ('replayssm-cap256-fp16', 'replayssm-cap256'),
)

# The two levers and the arms that carry them (for the envelope comparison).
LEVERS: dict[str, tuple[str, ...]] = {
    'int4': ('int4-dflash-b16', 'int4-dflash-b8', 'int4-plain-cap256'),
    'fp16-state': ('plain-cap256-fp16', 'replayssm-cap256-fp16'),
}

# Decision band: a session-paired ratio counts as a change only beyond the largest
# session-to-session coefficient of variation of y in bench's confirmation (1.9%).
BAND = 0.02
# A point enters the envelope and the decisions only with this many valid sessions.
MIN_SESSIONS = 3

# Quality holds: GSM8K on one arm each, then the logit probe on the listed arms.
QUALITY_HOLDS: dict[str, dict[str, object]] = {
    'q1': {'gsm8k': 'int4-dflash-b8', 'probes': ('int4-plain-cap256', 'int4-dflash-b8')},
    'q2': {'gsm8k': 'plain-cap256-fp16', 'probes': ('plain-cap256-fp16',)},
    'q3': {'gsm8k': 'replayssm-cap256-fp16', 'probes': ('replayssm-cap256-fp16',)},
}
# Levers whose headline and envelope count only arms with their own GSM8K run.
GSM8K_REQUIRED_FOR_HEADLINE = ('fp16-state',)
GSM8K_SEED = 0
GSM8K_THREADS = 128
# The reference GSM8K runs (two launches of plain-tuned, same harness and settings),
# committed in evidence/bench/quality/.
GSM8K_REFERENCES = ('plain-tuned-a-seed0', 'plain-tuned-b-seed0')
# Reported beside them, not deciding: the INT4 drafted arm against stock DFlash with the
# same sampling and block size, and bench's exact arms against the references (the
# spread an exact configuration shows on this check).
GSM8K_INT4_DFLASH_REFERENCE = 'dflash-tuned-seed0'
GSM8K_EXACT_ARMS = (
    'mtp-tuned-seed0',
    'mtp-stockverify-seed0',
    'dflash-tuned-seed0',
    'plain-tuned-replayssm-seed0',
)
# Package versions the reference runs used (their dist-info files predate the runs).
REFERENCE_VERSIONS = {
    'torch': '2.13.0',
    'triton': '3.7.1',
    'flashinfer-python': '0.6.18',
    'transformers': '5.12.1',
    'sgl-eval': '0.1.2',
}

# Reference of the logit probe: plain-ref-1's generate run from the load test (stock
# plain-cap256, greedy, 48 prompts x 256 tokens, top-20), pinned by SHA-256 once L0 ran.
REFERENCE_PROBE: str | None = (
    '~/vp-data/lossy/load_test/20261002T035539Z/plain-ref-1/probe_generate.json'
)
REFERENCE_PROBE_SHA256: str | None = (
    '5186472b7ae278bd4d7f9de1bd462e6126f03adb870cd0d98648cbaab0c968f7'
)

# Seconds per launch (about twice the expected time) and per GSM8K run.
LAUNCH_TIMEOUT = 600
GSM8K_TIMEOUT = 1500
PROBE_TIMEOUT = 600
HOLD_BUDGET = 2640


def session_order(session: str) -> tuple[Launch, ...]:
    if session not in SESSIONS:
        raise ValueError(f'unknown session {session!r}; declared: {SESSIONS}')
    if session == 'lossy-s2':
        return tuple(reversed(SESSION_LAUNCHES))
    return SESSION_LAUNCHES
