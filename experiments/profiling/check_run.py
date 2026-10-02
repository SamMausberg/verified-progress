"""Check that a run directory holds every window record of the command it recorded.

    python experiments/profiling/check_run.py ~/vp-data/profile/dflash-tuned_none

``run_profiles.py`` writes its command line to ``run_meta.json`` when it starts; this
script parses that command and checks ``windows.jsonl`` (and, under nsys, the reports)
against it, as ``run_profiles.py --check-complete`` does, but for a directory
possibly moved since. Exit status: 0 complete, 10 incomplete or different, 11 no
records or no ``run_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_profiles import ABSENT, build_parser, print_status, run_status


def recorded_args(meta: dict) -> argparse.Namespace:
    """The arguments of the run_profiles.py command recorded in a run_meta.json."""
    return build_parser().parse_args(meta['argv'][1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_dir', type=Path)
    run_dir = parser.parse_args().run_dir.expanduser().resolve()
    meta = run_dir / 'run_meta.json'
    if not meta.exists():
        sys.exit(print_status(run_dir, ABSENT, ['no run_meta.json']))
    args = recorded_args(json.loads(meta.read_text()))
    args.out_dir = run_dir
    sys.exit(print_status(run_dir, *run_status(args)))


if __name__ == '__main__':
    main()
