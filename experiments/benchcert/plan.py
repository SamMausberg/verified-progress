"""The served certified-head benchmark as declared (experiments/benchcert/README.md).

One place holds the families, their tuned arms, the certified-head settings, the
concurrency levels, the pool pins, the launch order of every hold and the expected
engine. `run_session.py` runs a hold from it and `analyze.py` checks the runs
against it, so a missing or extra launch is an error, not a smaller sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE_WORKTREE = Path.home() / 'sglang-wt/benchcert'
# bd66ce343e + engine/sglang/patches/kernel/0001-0010 (git am), the series on main.
ENGINE_BASE = 'bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824'
ENGINE_TREE = '9cd14d901f532bb15ea1bc7d74155928969bc6c2'
PORT = 30081
# Holds run only at the commit recorded in the README on this branch ("Hold commit:"),
# so the code that ran is fixed before the first timed run.
PIN_REF = 'benchcert/served'
PIN_FILE = 'experiments/benchcert/README.md'
SESSION_PREFIX = 'bc-'

# Every certified launch: the per-position (column) fallback under the conservative
# stock error model, and the certified head only for batches of at most 64 rows,
# where the head microbenchmark has it ahead of the stock head
# (evidence/certified_head/head_path_table.md). Larger batches replay the same
# graph with the gate off, so the stock head runs in a conditional node.
CERT_ENV = {
    'SGLANG_CERTIFIED_HEAD_FALLBACK': 'columns',
    'SGLANG_CERTIFIED_HEAD_MODEL': 'conservative',
    'SGLANG_CERTIFIED_HEAD_MAX_ROWS': '64',
}
# Counters are read back (a device sync) only every STATS_EVERY glue calls: in timed
# launches once per 25-90 s of decoding (two to five calls per step or cycle), in
# check launches often enough that a point's snapshot lags it by at most 25 calls.
TIMED_STATS_EVERY = 20000
CHECK_STATS_EVERY = 25

# Sweep settings shared by every timed launch (bench/README.md, "Metrics"): the
# confirmation split, 512 greedy output tokens with ignore_eos, max(32, 8c) measured
# requests, prompt and output token ids returned for the token comparison.
TIMED_SWEEP = (
    '--osl',
    '512',
    '--min-requests',
    '32',
    '--waves',
    '8',
    '--quiet-cpu-wait',
    '120',
    '--return-token-ids',
)
# Check launches (untimed): fewer requests, every counter written per point.
CHECK_SWEEP = (
    '--osl',
    '512',
    '--min-requests',
    '16',
    '--waves',
    '2',
    '--quiet-cpu-wait',
    '120',
    '--return-token-ids',
)


@dataclass(frozen=True)
class Family:
    """One tuned arm and the certified paths it enables."""

    name: str
    arm: str
    flags: tuple[str, ...]  # SGLANG_CERTIFIED_HEAD_<flag>=1
    paths: tuple[str, ...]  # engine paths the server log must list
    concurrency: tuple[int, ...]
    check_concurrency: tuple[int, ...]
    # --set overrides applied to both arms of the pair (pool pins).
    sets: tuple[str, ...] = ()
    check_sets: tuple[str, ...] = ()
    # Head rows per request in each certified call: decode 1; MTP verify
    # steps + 1 and draft 1; DFlash verify block and draft projection block - 1.
    rows_per_request: dict[str, int] = field(default_factory=dict)
    max_concurrency: int = 128
    # The one concurrency whose paired ratio enters the family verdict and H4
    # (Holm-adjusted across the four families); every other point is descriptive.
    primary: int = 1

    def gated_off(self, c: int) -> bool:
        """Every certified path's batch exceeds MAX_ROWS at concurrency c."""
        limit = int(CERT_ENV['SGLANG_CERTIFIED_HEAD_MAX_ROWS'])
        return min(self.rows_per_request.values()) * c > limit

    @property
    def stock_label(self) -> str:
        return self.arm

    @property
    def cert_label(self) -> str:
        return f'{self.arm}+cert'

    @property
    def check_label(self) -> str:
        return f'{self.arm}+cert-check'


DRAFT_PATHS = ('draft', 'draft_extend', 'dflash_draft')
FAMILIES = {
    family.name: family
    for family in (
        Family(
            name='plain',
            arm='plain-tuned',
            flags=('DECODE',),
            paths=('decode',),
            concurrency=(1, 4, 8, 16, 32, 64, 128),
            check_concurrency=(1, 8, 64),
            rows_per_request={'decode': 1},
        ),
        Family(
            name='mtp',
            arm='mtp-tuned-triton',
            flags=('VERIFY', 'DRAFT'),
            paths=('verify', *DRAFT_PATHS),
            concurrency=(1, 2, 4, 8, 16, 32, 64),
            check_concurrency=(1, 4, 16, 64),
            rows_per_request={'verify': 4, 'draft': 1, 'draft_extend': 1},
        ),
        # DFlash pairs pin the KV pool on both arms. The certified graphs' work buffers
        # are allocated during graph capture, after SGLang has sized the pool from free
        # memory, and every captured graph of at most 256 rows gets them, whatever
        # MAX_ROWS is. With the stock GDN verify's per-draft-token state cache (about
        # 48 GB at these capacities), the stack workstream's certified block-16 server
        # ran out of memory at its sampling prewarm: 6.13 GB of verify graphs against
        # 2.91 GB stock, 2.29 MB per certified graph row. At that rate, verify plus
        # draft projection add about 6.2 GB (block 16: 1,408 + 1,320 rows) and 9.9 GB
        # (block 8: 2,304 + 2,016 rows). 60,000 tokens frees 13.2 GB and 10.6 GB
        # against the stock arms' own pools (308K and 258K tokens at 53.4 KB per token,
        # target and draft) and is 2.5 times what the largest point here holds (block 8
        # at c = 32: at most 23.6K tokens). Check launches add the stock head to
        # every certified graph, so they pin 40,000.
        Family(
            name='dflash16',
            arm='dflash-tuned-b16',
            flags=('VERIFY', 'DRAFT'),
            paths=('verify', *DRAFT_PATHS),
            concurrency=(1, 2, 4, 8),
            check_concurrency=(1, 2, 4),
            sets=('max-total-tokens=60000',),
            check_sets=('max-total-tokens=40000',),
            rows_per_request={'verify': 16, 'dflash_draft': 15},
            max_concurrency=64,
        ),
        Family(
            name='dflash8',
            arm='dflash-tuned',
            flags=('VERIFY', 'DRAFT'),
            paths=('verify', *DRAFT_PATHS),
            concurrency=(4, 8, 16, 32),
            check_concurrency=(1, 4, 8),
            primary=4,
            sets=('max-total-tokens=60000',),
            check_sets=('max-total-tokens=40000',),
            rows_per_request={'verify': 8, 'dflash_draft': 7},
        ),
    )
}

PART_A = ('plain', 'mtp')
PART_B = ('dflash16', 'dflash8')


def _part(families: tuple[str, str], reverse: bool) -> list[tuple[str, str]]:
    """Two pairs, the order alternating within the part (A B, B A)."""
    first, second = families
    order = [(first, 'stock'), (first, 'cert'), (second, 'cert'), (second, 'stock')]
    return order[::-1] if reverse else order


# Sessions: each pair (stock, certified) runs back to back; session 2 reverses the
# order of session 1, session 3 repeats it. A replacement session s4 runs only the
# families that s1-s3 left with fewer than three valid pairs (README, decision rule).
SESSIONS = {
    's1': {'A': _part(PART_A, False), 'B': _part(PART_B, False)},
    's2': {'A': _part(PART_A, True), 'B': _part(PART_B, True)},
    's3': {'A': _part(PART_A, False), 'B': _part(PART_B, False)},
}
DECISION_SESSIONS = ('s1', 's2', 's3')
REPLACEMENT_SESSION = 's4'

# The GPU holds, in submission order. A hold runs its steps in order; `check` is
# the untimed check-mode launch of every family.
HOLDS = {
    'h1': (('check', None), ('s1', 'B')),
    'h2': (('s1', 'A'),),
    'h3': (('s2', 'B'), ('s2', 'A')),
    'h4': (('s3', 'B'), ('s3', 'A')),
}


def launches(step: str, part: str | None) -> list[tuple[str, str]]:
    """(family, variant) launches of one hold step, in order."""
    if step == 'check':
        return [(name, 'check') for name in FAMILIES]
    if step == REPLACEMENT_SESSION:
        raise ValueError('s4 launches are chosen by analyze.py replacement-plan')
    if part is None:
        raise ValueError(f'session {step} needs a part (A or B)')
    return list(SESSIONS[step][part])


def label(family: Family, variant: str) -> str:
    return {
        'stock': family.stock_label,
        'cert': family.cert_label,
        'check': family.check_label,
    }[variant]


def certified_env(family: Family, variant: str, src: Path, stats: Path) -> dict[str, str]:
    """SGLANG_CERTIFIED_HEAD_* variables of a launch (none for the stock arm)."""
    if variant == 'stock':
        return {}
    env = {f'SGLANG_CERTIFIED_HEAD_{flag}': '1' for flag in family.flags}
    env.update(CERT_ENV)
    env['SGLANG_CERTIFIED_HEAD_SRC'] = str(src)
    env['SGLANG_CERTIFIED_HEAD_STATS'] = str(stats)
    every = CHECK_STATS_EVERY if variant == 'check' else TIMED_STATS_EVERY
    env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = str(every)
    if variant == 'check':
        env['SGLANG_CERTIFIED_HEAD_CHECK'] = '1'
    return env


def sweep_command(
    family: Family,
    variant: str,
    session: str,
    out: Path,
    python: str = 'python',
    src: Path | None = None,
) -> tuple[list[str], Path | None]:
    """The bench.sweep command of one launch and its stats file (None for stock)."""
    src = src or REPO / 'src'
    name = label(family, variant)
    stats = out / session / 'stats' / f'{name}.json' if variant != 'stock' else None
    command = [
        python,
        '-m',
        'bench.sweep',
        '--arm',
        family.arm,
        '--label',
        name,
        '--session',
        f'{SESSION_PREFIX}{session}',
        '--out',
        str(out / session),
        '--port',
        str(PORT),
        '--sglang-worktree',
        str(ENGINE_WORKTREE),
        *(CHECK_SWEEP if variant == 'check' else TIMED_SWEEP),
    ]
    for item in family.check_sets if variant == 'check' else family.sets:
        command += ['--set', item]
    if stats is not None:
        for key, value in certified_env(family, variant, src, stats).items():
            command += ['--env', f'{key}={value}']
        command += ['--snapshot-file', str(stats)]
    levels = family.check_concurrency if variant == 'check' else family.concurrency
    command += ['--concurrency', *(str(c) for c in levels)]
    return command, stats
