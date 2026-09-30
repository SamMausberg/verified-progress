"""Copy a run directory's small records into evidence/profiles/windows/.

    python experiments/profiling/collect_run.py ~/vp-data/profile/plain_nsys --name plain_nsys

Writes ``<name>.jsonl`` (client window summaries), ``<name>_meta.json`` (the
exact server command, SGLang SHA and start time) and
``<name>_server_startup.log`` (the server log up to "fired up", without
progress bars: server arguments, backends, memory pools, CUDA graph batch
sizes). Raw traces stay in the run directory.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PROGRESS = re.compile(r'\d+%\|')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--evidence', type=Path, default=Path('evidence/profiles/windows'))
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.run_dir / 'windows.jsonl', args.evidence / f'{args.name}.jsonl')
    shutil.copyfile(args.run_dir / 'run_meta.json', args.evidence / f'{args.name}_meta.json')
    kept = []
    for line in (args.run_dir / 'server.log').read_text(errors='replace').splitlines():
        if PROGRESS.search(line) or 'Capturing' in line:
            continue
        kept.append(line[:20000])
        if 'fired up' in line:
            break
    (args.evidence / f'{args.name}_server_startup.log').write_text('\n'.join(kept) + '\n')


if __name__ == '__main__':
    main()
