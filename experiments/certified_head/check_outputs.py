"""Check the evidence run's outputs, beyond the steps' exit codes.

Some steps isolate errors per measurement and still exit 0 (head_primitives,
micro_head arms), Nsight Compute exits 0 when the profiled program fails, and a
replay in which every batch size was refused by the self-test agrees with the
stock head trivially. This reads the outputs in OUT_DIR and exits 1, listing the
problems, if any of them is not what a valid run produces::

    python experiments/certified_head/check_outputs.py OUT_DIR
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def errors_in(obj: Any, path: str = '') -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'error':
                found.append(f'{path}: {str(v)[:120]}')
            else:
                found.extend(errors_in(v, f'{path}/{k}'))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(errors_in(v, f'{path}[{i}]'))
    return found


def main() -> None:
    out = Path(sys.argv[1])
    problems: list[str] = []

    def load(name: str) -> Any:
        p = out / name
        if not p.exists():
            problems.append(f'{name}: missing')
            return None
        return json.loads(p.read_text())

    replay = load('replay_decisions.json')
    if replay is not None:
        rows = replay['rows']
        counts = replay['agreement_counts']
        if rows < 60000:
            problems.append(f'replay: only {rows} rows')
        for key, n in counts.items():
            if key.endswith('_with_fallback_eq_reference') and n != rows:
                problems.append(f'replay: {key} = {n} of {rows}')
        decided = counts.get('bf16/conservative_decided', 0)
        if decided < 0.9 * rows:
            problems.append(
                f'replay: only {decided} of {rows} rows certified (self-test refusals?)'
            )
    for name in ('gemv_sweep_w8a16.json', 'gemv_sweep_w8a8.json'):
        sweep = load(name)
        if sweep is not None:
            for m, entry in sweep['batches'].items():
                if 'best' not in entry:
                    problems.append(f'{name}: no passing configuration at M={m}')
    micro = load('micro_head.json')
    if micro is not None:
        problems += [f'micro_head{e}' for e in errors_in(micro['batches'])]
        for head, report in micro.get('enclosure_self_test', {}).items():
            if not report.get('ok'):
                problems.append(
                    f'micro_head: self-test refused {head}: {report["refused_batch_sizes"]}'
                )
    prim = load('head_primitives.json')
    if prim is not None:
        problems += [f'head_primitives{e}' for e in errors_in(prim)]
    details = out / 'ncu_gemv_details.csv'
    if not details.exists():
        problems.append('ncu_gemv_details.csv: missing')
    else:
        with details.open(newline='') as f:
            ids = {r['ID'] for r in csv.DictReader(f)}
        if len(ids) != 8:
            problems.append(f'ncu: {len(ids)} profiled launches, expected 8')
    if problems:
        print('output check FAILED:')
        for p in problems:
            print(' ', p)
        sys.exit(1)
    print('output check passed')


if __name__ == '__main__':
    main()
