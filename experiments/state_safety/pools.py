"""Record the pools both servers allocated for every comparison in the evidence.

Team rule: output comparisons pin the KV and GDN pools. The earlier runs did not,
so this records what each server had: for each noise-floor pair (matrix passes,
matched to the server that ran them), tapped mechanism run, tap check and tap
signature, the pools of both servers and whether they match.

Committed evidence names data directories by absolute path under the home
directory that produced it (e.g. /home/ubuntu/vp-data/state/tap/v3_plain_c1);
those are rebased onto the current data root, so the record regenerates under
any home directory.

    python experiments/state_safety/pools.py --root ~/vp-data/state \
        --evidence evidence/state_safety
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare import pass_server
from server import pools_known, resolved_pools

DATA_MARKER = ('vp-data', 'state')


def data_relative(path: str | Path, root: Path) -> Path:
    """A data path as a path under root.

    Paths already under root, or relative ones, are kept relative to root; an
    absolute path containing .../vp-data/state/ (from another home directory) is
    rebased at that point.
    """
    p = Path(path)
    if not p.is_absolute():
        return p
    try:
        return p.relative_to(root)
    except ValueError:
        parts = p.parts
        for i in range(len(parts) - 1):
            if parts[i : i + 2] == DATA_MARKER:
                return Path(*parts[i + 2 :])
    raise ValueError(f'{path} is not under {root} or any .../vp-data/state')


def build(root: Path, evidence: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def row(source, label, where_a, where_b, server_a, server_b, pa, pb):
        rows.append(
            {
                'evidence': source,
                'comparison': label,
                'server_a': where_a,
                'server_b': where_b,
                'same_server': server_a is not None and server_a == server_b,
                'pools_a': pa,
                'pools_b': pb,
                'pools_identical': pa == pb if pools_known(pa) and pools_known(pb) else None,
            }
        )

    def session(rel: Path) -> tuple[str, dict[str, int | None] | None]:
        # Tap sessions: one server per directory, started once by tap_runs.py.
        log = root / rel / 'server.log'
        return str(rel), resolved_pools(log) if log.exists() else None

    floors = (('noise_floor.json', 'runs'), ('noise_floor_pinned.json', 'runs_pinned'))
    for floor, sub in floors:
        if not (evidence / floor).exists():
            continue
        for label, s in json.loads((evidence / floor).read_text())['pairs'].items():
            # Matrix passes: the server that ran each pass, from the pass's own metadata.
            (ka, pa), (kb, pb) = (pass_server(root / sub, r) for r in (s['run_a'], s['run_b']))
            row(floor, label, f'{sub}/{s["run_a"]}', f'{sub}/{s["run_b"]}', ka, kb, pa, pb)
    for f in sorted(evidence.glob('mechanism_*.json')):
        s = json.loads(f.read_text())['summary']
        (a, pa), (b, pb) = (
            session(data_relative(s['a'], root)),
            session(data_relative(s['b'], root)),
        )
        row(f.name, f'{Path(a).name} vs {Path(b).name}', a, b, a, b, pa, pb)
    sig = evidence / 'tap_signature.json'
    if sig.exists():
        for e in json.loads(sig.read_text()):
            (a, pa), (b, pb) = session(Path('tap', e['a'])), session(Path('tap', e['b']))
            row(sig.name, f'{e["a"]} vs {e["b"]}', a, b, a, b, pa, pb)
    chk = evidence / 'tap_check.json'
    if chk.exists():
        for name, e in json.loads(chk.read_text()).items():
            a, pa = session(Path('tap', name))
            kb, pb = pass_server(root / 'runs', e['untapped_run'])
            label = f'{name} vs untapped {e["untapped_run"]}'
            row(chk.name, label, a, f'runs/{e["untapped_run"]}', a, kb, pa, pb)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=str(Path.home() / 'vp-data/state'))
    ap.add_argument('--evidence', required=True)
    args = ap.parse_args()
    evidence = Path(args.evidence)
    rows = build(Path(args.root), evidence)
    (evidence / 'pools.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
