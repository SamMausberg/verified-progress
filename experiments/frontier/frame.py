"""The 5x requirement per verify width, re-derived from committed Stage A evidence (no GPU).

Inputs: `evidence/repair/stage_a_oracle_triton.json` (Triton GDN verify kernel, forced
full acceptance at B = 16, 64, 256, and the DFlash block-16 baseline on the same kernel)
and `evidence/repair/stage_a_oracle.json` (FlashInfer kernel; only its B = 128 run is
used, for the phases a Triton B = 128 cycle would share).

A 5x end-to-end gain needs decode speedup s = (1 - f) / (1/5 - f) over the baseline's
time per token, so a cycle of period C must commit C / (baseline us per token / s)
tokens. Three cycle accountings per width:

- `with_dflash_draft`: the measured cycle of a perfect block, including DFlash's own
  drafting at width B;
- `free_drafter`: the same cycle minus its draft phase (verify, commit, append and the
  rest of the cycle stay);
- `verify_commit_only`: verify plus commit, the accounting in the external proposals.

B = 128 has no Triton measurement: its verify time is interpolated linearly between the
Triton B = 64 and B = 256 runs and its other phases are taken from the FlashInfer B = 128
run. It is labelled and no rejection rule uses it.

    python experiments/frontier/frame.py --out evidence/frontier/frame.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TRITON = REPO / 'evidence/repair/stage_a_oracle_triton.json'
FLASHINFER = REPO / 'evidence/repair/stage_a_oracle.json'
TARGET_E2E = 5.0


def required_alpha(tokens: float, width: int) -> float | None:
    """Smallest constant per-position acceptance alpha with sum_{k<B} alpha^k >= tokens."""
    if tokens > width:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 - mid**width) / (1 - mid) >= tokens:
            hi = mid
        else:
            lo = mid
    return hi


def phases(row: dict[str, Any]) -> dict[str, float]:
    cycle = row['cycle_period_us']
    named = row['V_us'] + row['draft_us'] + row['commit_us'] + row['append_us']
    return {
        'cycle_us': cycle,
        'verify_us': row['V_us'],
        'draft_us': row['draft_us'],
        'commit_us': row['commit_us'],
        'append_us': row['append_us'],
        'other_us': cycle - named,
    }


def requirement(ph: dict[str, float], width: int, us_per_token_needed: float) -> dict[str, Any]:
    accountings = {
        'with_dflash_draft': ph['cycle_us'],
        'free_drafter': ph['cycle_us'] - ph['draft_us'],
        'verify_commit_only': ph['verify_us'] + ph['commit_us'],
    }
    out: dict[str, Any] = {}
    for name, period in accountings.items():
        tokens = period / us_per_token_needed
        alpha = required_alpha(tokens, width)
        out[name] = {
            'cycle_us': round(period, 1),
            'tokens_per_cycle_needed': round(tokens, 2),
            'constant_alpha_needed': None if alpha is None else round(alpha, 5),
        }
    return out


def frame() -> dict[str, Any]:
    tri = json.loads(TRITON.read_text())
    fi = json.loads(FLASHINFER.read_text())
    base = tri['baseline']
    f = base['f']
    decode_needed = (1 - f) / (1 / TARGET_E2E - f)
    us_per_token = base['C_D_us'] / base['A_D']
    us_needed = us_per_token / decode_needed
    widths: dict[str, Any] = {}
    rows = {r['B']: r for r in tri['rows']}
    for width, row in sorted(rows.items()):
        ph = phases(row)
        widths[str(width)] = {
            'source': f'triton run {row["run"]} (measured)',
            'phases_us': {k: round(v, 1) for k, v in ph.items()},
            **requirement(ph, width, us_needed),
        }
    fi128 = next(r for r in fi['rows'] if r['B'] == 128)
    v128 = rows[64]['V_us'] + (rows[256]['V_us'] - rows[64]['V_us']) * (128 - 64) / (256 - 64)
    ph128 = phases(fi128)
    ph128['cycle_us'] += v128 - ph128['verify_us']
    ph128['verify_us'] = v128
    widths['128'] = {
        'source': (
            'INTERPOLATED, not measured: Triton verify linear between B = 64 and 256; other '
            f'phases from FlashInfer run {fi128["run"]}'
        ),
        'phases_us': {k: round(v, 1) for k, v in ph128.items()},
        **requirement(ph128, 128, us_needed),
    }
    return {
        'kind': 'derived from committed Stage A measurements (no new runs)',
        'baseline': {
            'run': base['run'],
            'C_D_us': round(base['C_D_us'], 1),
            'A_D': round(base['A_D'], 4),
            'f': round(f, 5),
            'us_per_token': round(us_per_token, 2),
            'tokens_per_ms': round(1000 / us_per_token, 4),
        },
        'target_e2e': TARGET_E2E,
        'decode_speedup_needed': round(decode_needed, 4),
        'us_per_token_needed': round(us_needed, 2),
        'tokens_per_ms_needed': round(1000 / us_needed, 4),
        'tokens_needed_per_ms_of_cycle_saved': round(1000 / us_needed, 4),
        'widths': dict(sorted(widths.items(), key=lambda kv: int(kv[0]))),
        'note': (
            'Triton verify figures carry the repair README pending-exactness label. The '
            'external proposals quote 111.6 tokens at B = 256 for a free drafter; that is the '
            'verify_commit_only accounting, which leaves out append and the rest of the cycle '
            '(free_drafter: 114.1).'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = frame()
    out['sha256'] = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), TRITON, FLASHINFER)
    }
    out['repo_commit'] = subprocess.run(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['widths'], indent=1))


if __name__ == '__main__':
    main()
