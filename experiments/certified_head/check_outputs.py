"""Check the evidence run's outputs, beyond the steps' exit codes.

Some steps isolate errors per measurement and still exit 0 (head_primitives,
micro_head arms), Nsight Compute exits 0 when the profiled program fails, and a
replay in which every batch size was refused by the self-test agrees with the
stock head trivially. This reads the outputs in OUT_DIR and exits 1, listing the
problems, if any of them is not what a valid run produces::

    python experiments/certified_head/check_outputs.py OUT_DIR
    python experiments/certified_head/check_outputs.py --engine ENGINE_ARMS_DIR
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


# Each micro_head arm timed on a batch its head decides, the arm that times the
# same head with a row forced to fall back, and whether that fallback runs the
# whole-batch stock head (column fallbacks run a small gathered GEMM instead).
DECIDED_ARMS = {
    'certified': ('certified_fallback', True),
    'certified_no_probe': ('certified_fallback', True),
    'certified_columns_mode': ('certified_columns_fallback', False),
    'certified_sample': ('certified_sample_fallback', True),
    'certified_sample_no_probe': ('certified_sample_fallback', True),
    'certified_w8a8': ('certified_w8a8_fallback', True),
    'certified_bf16': ('certified_bf16_fallback', True),
}


def micro_problems(micro: dict[str, Any]) -> list[str]:
    """A decided arm must time a batch its own head decides (statuses under
    ``batch_stats``), and its fallback arm must cost more: at least half the stock
    GEMM (``bf16_gemm``) more for a whole-batch fallback, which runs that GEMM on
    top of the decided path. x7's W8A8 arm at M = 32 timed the fallback on a batch
    chosen at other tiles: 679 us, against 228 us at M = 16."""
    found = []
    for m, entry in micro['batches'].items():
        stats = entry.get('batch_stats')
        if stats is None:
            found.append(f'micro_head/{m}: no batch statuses')
            continue
        gemm = entry.get('bf16_gemm', {}).get('warm', {}).get('median_us', 0.0)
        for arm, (fb_arm, dense) in DECIDED_ARMS.items():
            if arm not in entry:
                continue
            if arm not in stats:
                found.append(f'micro_head/{m}/{arm}: no batch status')
            elif stats[arm]['fallback_rows']:
                found.append(f'micro_head/{m}/{arm}: batch not decided: {stats[arm]}')
            t, t_fb = entry[arm].get('warm'), entry.get(fb_arm, {}).get('warm')
            if t is None or t_fb is None:
                continue
            margin = 0.5 * gemm if dense else 0.0
            if t_fb['median_us'] - t['median_us'] <= margin:
                found.append(
                    f'micro_head/{m}/{arm}: {t["median_us"]:.1f} us, within {margin:.1f} us '
                    f'of {fb_arm} ({t_fb["median_us"]:.1f} us): did it time its fallback?'
                )
    return found


MIN_CERTIFIED_SHARE = 0.5
MIN_CERTIFIED_ROWS = 1000


def engine_problems(root: Path) -> list[str]:
    """SGLang check arms (``engine_validate.sh`` output): every path must certify at
    least ``MIN_CERTIFIED_SHARE`` of its rows and ``MIN_CERTIFIED_ROWS`` rows, with
    no row differing from the stock head and none refused or probe-tripped, so
    "0 differing rows" cannot come from a path that certified almost nothing."""
    found = []
    arms = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.endswith('_check')]
    if not arms:
        found.append(f'{root}: no check arms')
    for d in arms:
        stats = d / 'certified_stats.json'
        if not stats.exists():
            found.append(f'{d.name}: no certified_stats.json')
            continue
        paths = json.loads(stats.read_text())['paths']
        active = {p: v for p, v in paths.items() if v.get('calls')}
        if not active:
            found.append(f'{d.name}: no certified calls')
        for p, v in active.items():
            certified = v['rows'] - v['fallback_rows']
            if v['mismatch_rows']:
                found.append(f'{d.name}/{p}: {v["mismatch_rows"]} rows differ from stock')
            if v.get('status_refused') or v.get('status_probe'):
                found.append(
                    f'{d.name}/{p}: refused {v.get("status_refused", 0)}, '
                    f'probe-tripped {v.get("status_probe", 0)} rows'
                )
            if certified < MIN_CERTIFIED_ROWS or certified < MIN_CERTIFIED_SHARE * v['rows']:
                found.append(f'{d.name}/{p}: only {certified} of {v["rows"]} rows certified')
    return found


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == '--engine':
        found = engine_problems(Path(sys.argv[2]))
        for p in found:
            print(' ', p)
        print('engine output check', 'FAILED' if found else 'passed')
        sys.exit(1 if found else 0)
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
    stress = load('stress_defaults.json')
    if stress is not None:
        for c in stress['configs']:
            name = f'stress_defaults: {c["arith"]} {c["config"]}'
            if not c['ok']:
                problems.append(f'{name}: missed {c["rows_missed"]}')
            if c['row_checks'] < stress['row_checks_target']:
                problems.append(f'{name}: only {c["row_checks"]} row-checks')
        if not stress.get('ok'):
            problems.append('stress_defaults: incomplete or failed')
    micro = load('micro_head.json')
    if micro is not None:
        problems += [f'micro_head{e}' for e in errors_in(micro['batches'])]
        problems += micro_problems(micro)
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
            grids: dict[int, str] = {}
            for r in csv.DictReader(f):
                grids[int(r['ID'])] = r['Grid Size']
        if len(grids) != 8:
            problems.append(f'ncu: {len(grids)} profiled launches, expected 8')
        expected_path = out / 'ncu_expected.json'
        if not expected_path.exists():
            problems.append('ncu_expected.json: missing (cannot tell which launches were profiled)')
        else:
            expected = [e['grid'] for e in json.loads(expected_path.read_text())]
            got = [int(grids[i].strip('()').split(',')[0]) for i in sorted(grids)]
            if got != expected:
                problems.append(f'ncu: launch grids {got}, expected {expected}')
    if problems:
        print('output check FAILED:')
        for p in problems:
            print(' ', p)
        sys.exit(1)
    print('output check passed')


if __name__ == '__main__':
    main()
