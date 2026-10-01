"""Collect the buffered-verify exactness checks into one evidence table.

For each run directory written by run_replay_check.sh / run_replay_check_mtp.sh
(subdirectories c<conc>-<arm>, and c<conc>-<arm>-equality.json comparing an arm
with the stock arm), writes one row per (drafter, concurrency, arm): sequences,
sequences bitwise identical to stock (tokens and top-5 logprobs at every
position), sequences whose tokens differ, first-divergence classes (#37's
convention), the earliest position where logprobs first differ, tokens per
verify cycle, the server's foreign CPU load and the pools it resolved (KV
tokens, mamba slots, effective running limit: arms are comparable only if these
match). Also copies each run's launch record.

    python experiments/drafter/summarize_replay_check.py \
        --run dflash:~/vp-data/drafter/replay-check --run mtp:~/vp-data/drafter/replay-check-mtp \
        --out evidence/drafter/buffered_verify
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def pools(server_info: Path) -> dict[str, Any]:
    """KV tokens, mamba slots and the running limit the server resolved."""
    info = json.loads(server_info.read_text())
    internal = info.get('internal_states') or [{}]
    merged = {**info, **(internal[0] if isinstance(internal, list) else internal)}
    return {
        'kv_tokens': merged.get('max_total_num_tokens'),
        'mamba_slots': merged.get('max_mamba_cache_size'),
        'running_limit': merged.get('effective_max_running_requests_per_dp'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--run', action='append', required=True, help='DRAFTER:DIR')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'launch').mkdir(exist_ok=True)
    for spec in args.run:
        drafter, directory = spec.split(':', 1)
        root = Path(directory).expanduser()
        for run in sorted(p for p in root.iterdir() if p.is_dir() and re.match(r'c\d+-', p.name)):
            conc, arm = run.name[1:].split('-', 1)
            summary = json.loads((run / 'summary.json').read_text())['domains']['all']
            launch = json.loads((run / 'launch.json').read_text())
            cpu = (
                json.loads((run / 'cpu_load.json').read_text())
                if (run / 'cpu_load.json').exists()
                else {}
            )
            (args.out / 'launch' / f'{drafter}_{run.name}.json').write_text(
                json.dumps({'launch': launch, 'cpu_load': cpu}, indent=2) + '\n'
            )
            row: dict[str, Any] = {
                'drafter': drafter,
                'concurrency': int(conc),
                'arm': arm,
                'requests': summary['requests'],
                'completion_tokens': summary['completion_tokens'],
                'tau_pooled': summary['accept_length'],
                'tau_mean_per_request': summary.get('accept_length_mean_per_request'),
                'foreign_cpu_cores_mean': cpu.get('foreign_cores_mean'),
                **pools(run / 'server_info.json'),
            }
            equality = root / f'{run.name}-equality.json'
            if equality.exists():
                eq = json.loads(equality.read_text())
                firsts = list(eq.get('first_logprob_difference', {}).values())
                row.update(
                    bitwise_identical=eq.get('bitwise_identical_sequences'),
                    token_divergent=eq['sequences_diverged'],
                    divergences_per_1000=eq['divergences_per_1000_tokens'],
                    classes=json.dumps(eq['classes'], sort_keys=True),
                    logprob_differing=len(firsts),
                    earliest_logprob_difference=min(firsts) if firsts else None,
                )
            rows.append(row)
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with (args.out / 'exactness.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
