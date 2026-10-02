"""Readings of the h7 device ring: 579ae7ce's position-439 row and every certified row.

    python -m experiments.benchcert.ring_report --out DIR --json REPORT.json [--csv ROWS.csv]

The ring (replay_hook/benchcert_ring.py) holds, per target verify replay of a `certring`
launch, the gate the conditional node read, the device row count `valid`, the fallback
flags and, for the first 64 rows, the head's final id, status, candidate count, first 64
candidates and their refined bounds, with the verify's predicted ids and accept lengths;
a JSONL line per step holds the host's view (rows, batch size, intended gate, and per
request its prompt length, output length and last output ids).

Two readings, set before the run (README, "Ring-logged reruns"):

1. 579ae7ce's row for position 439. The request is the one whose prompt length and last
   output ids match the reference output (session 1's certified run; every run with the
   same prefix agrees through position 438) and whose verify covers position 439; the
   row is ``4 i + 439 - output_length``. The five readings: (1) device gate differs from
   the host's; (2) gate on, status 0 and an id outside the near tie (certificate fault);
   (3) gate on, a nonzero status (AMBIGUOUS: column fallback; other: dense merge) and an
   id outside the near tie (fallback fault), with whether 1756 was a candidate and
   whether the id had the largest bounds; (4) gate on and ``valid`` not covering the row;
   (5) gate off (the stock head inside the conditional node).
2. Every row of every certified step, two checks that need no stock logits:
   - consistency of the head: for a complete candidate list (status 0 or exactly
     AMBIGUOUS, count at most 64) the stock argmax ``t`` satisfies ``rhi[t] >= rlo[j]``
     for every candidate ``j``, so an id outside the list, or with ``rhi`` below the
     largest ``rlo``, is wrong whatever the stock rounding; with status 0 the id must
     have the largest ``rlo``;
   - emission: the verify's predicted ids for a request's accepted rows (accept length,
     bonus included) equal the head's ids on those rows.

Per launch it also lists the ring files, the deviations file, the graph check, and the
server time per pass of each c = 64 point beside h6a's certified launches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.benchcert import drain
from experiments.benchcert.analyze import server_ms_per_pass

POSITION = drain.TARGET[1]
NEAR_TIE = (8078, 5715, 68189)  # within 0.19 nats at batch 1 (teacher-forced top 3)
WRONG = drain.TARGET[2]
STATUS_AMBIGUOUS = 2
CANDS = 64
ROWS_PER_REQUEST = 4


def reference_output(runs: Path) -> list[int]:
    point = next(p for n, p in drain.point_dirs(runs / 'drain', runs) if n.startswith('s1/'))
    record = drain.target_record(point)
    assert record is not None and 'output' in record
    return list(record['output'])


def ring_files(ring: Path) -> list[tuple[Path, Path]]:
    return [(p, p.with_suffix('.jsonl')) for p in sorted(ring.glob('ring-*.npz'))]


def point_windows(launch_dir: Path) -> list[tuple[str, float, float, Path]]:
    """(rN/cNNN, first request start, last request end, point dir), wall clock seconds."""
    out = []
    for point in sorted(launch_dir.glob('*/2026*/r*/c*')):
        starts, ends = [], []
        with (point / 'requests.csv').open() as handle:
            for row in csv.DictReader(handle):
                if row.get('start_ns') and row.get('latency_ms'):
                    s = int(row['start_ns']) / 1e9
                    starts.append(s)
                    ends.append(s + float(row['latency_ms']) / 1e3)
        if starts:
            out.append((f'{point.parent.name}/{point.name}', min(starts), max(ends), point))
    return out


def which_point(t: float, windows: list[tuple[str, float, float, Path]]) -> str | None:
    for name, t0, t1, _ in windows:
        if t0 - 0.5 <= t <= t1 + 0.5:
            return name
    return None


def find_target(
    records: list[dict[str, Any]],
    predict: np.ndarray,
    accept: np.ndarray,
    ref: list[int],
    prompt_len: int,
) -> list[dict[str, Any]]:
    """Steps whose verify covers position 439 for the request that matches the reference.

    The host's output lengths lag the device under the overlap scheduler, so positions
    come from the device: each request with this prompt length is followed by its slot
    from its first verify, which covers positions 1-4 (the prefill produced position 0),
    and advances by its accept length (bonus included). A request is kept while every
    token it commits before position 439 equals the reference's. Offsets 0 and 2 are
    tracked as well, in case the first verify starts elsewhere.
    """
    offsets = (1, 0, 2)
    runs: dict[int, dict[str, Any]] = {}
    found = []
    for index, rec in enumerate(records):
        present = set()
        lists = zip(rec.get('slots') or [], rec.get('prompt_lens') or [], rec.get('output_lens') or [], strict=False)
        for i, (slot, pl, host_ol) in enumerate(lists):
            if pl != prompt_len or slot is None:
                continue
            present.add(slot)
            st = runs.get(slot)
            if st is None or host_ol < st['host_ol']:  # a new request in this slot
                st = runs[slot] = {'pos': dict.fromkeys(offsets, 0), 'ok': dict.fromkeys(offsets, True)}
                for o in offsets:
                    st['pos'][o] = o
            st['host_ol'] = host_ol
            acc = int(accept[index, i])
            pred = predict[index, ROWS_PER_REQUEST * i : ROWS_PER_REQUEST * (i + 1)]
            for o in offsets:
                pos = st['pos'][o]
                if st['ok'][o] and pos <= POSITION < pos + ROWS_PER_REQUEST:
                    found.append(
                        {
                            'index': index,
                            'request': i,
                            'slot': slot,
                            'offset': o,
                            'output_len': pos,
                            'host_output_len': host_ol,
                            'row': ROWS_PER_REQUEST * i + POSITION - pos,
                        }
                    )
                for j in range(min(acc, ROWS_PER_REQUEST)):
                    if pos + j < min(POSITION, len(ref)) and int(pred[j]) != ref[pos + j]:
                        st['ok'][o] = False
                st['pos'][o] = pos + acc
        for slot in [s for s in runs if s not in present]:
            del runs[slot]
    return found


def row_state(arrays: dict[str, np.ndarray], s: int, row: int) -> dict[str, Any]:
    n = int(arrays['count'][s, row])
    k = min(n, CANDS)
    cand = arrays['cand'][s, row, :k].tolist()
    rlo = arrays['rlo'][s, row, :k].tolist()
    rhi = arrays['rhi'][s, row, :k].tolist()
    head_id = int(arrays['ids'][s, row])
    state: dict[str, Any] = {
        'head_id': head_id,
        'status': int(arrays['status'][s, row]),
        'count': n,
        'cand': cand,
        'rlo': rlo,
        'rhi': rhi,
        'wrong_in_cand': WRONG in cand,
        'near_tie_in_cand': [t for t in NEAR_TIE if t in cand],
    }
    if cand:
        best_lo = max(rlo)
        state['max_rlo'] = best_lo
        state['id_in_cand'] = head_id in cand
        if head_id in cand:
            j = cand.index(head_id)
            state['id_bounds'] = [rlo[j], rhi[j]]
            state['id_has_max_rlo'] = rlo[j] == best_lo
            state['id_rhi_below_max_rlo'] = rhi[j] < best_lo
    return state


def readings(
    gate: bool, host_gate: bool, valid: int, row: int, state: dict[str, Any] | None
) -> list[int]:
    out = []
    if gate != host_gate:
        out.append(1)
    if not gate:
        out.append(5)
        return out
    if valid <= row:
        out.append(4)
    if state is not None and state['head_id'] not in NEAR_TIE:
        out.append(2 if state['status'] == 0 else 3)
    return out


def consistency(arrays: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Head consistency and emission checks over every certified row of one ring file."""
    gate = arrays['gate'].astype(bool)
    totals = {
        'certified_steps': 0,
        'certified_rows': 0,
        'complete_rows': 0,
        'column_rows': 0,
        'dense_rows': 0,
        'id_not_in_cand': 0,
        'id_rhi_below_max_rlo': 0,
        'status0_id_not_max_rlo': 0,
        'emission_rows': 0,
        'emission_mismatch': 0,
        'valid_differs_from_rows': 0,
        'gate_differs_from_host': 0,
    }
    examples: list[dict[str, Any]] = []
    for s, rec in enumerate(records):
        rows = int(rec.get('rows', 0))
        if bool(gate[s]) != bool(rec.get('host_gate')):
            totals['gate_differs_from_host'] += 1
            examples.append({'kind': 'gate', 'k': rec.get('k')})
        if not gate[s]:
            continue
        totals['certified_steps'] += 1
        if int(arrays['valid'][s]) != rows:
            totals['valid_differs_from_rows'] += 1
            examples.append(
                {'kind': 'valid', 'k': rec.get('k'), 'valid': int(arrays['valid'][s]), 'rows': rows}
            )
        r = min(rows, CANDS)
        totals['certified_rows'] += r
        st = arrays['status'][s, :r]
        cnt = arrays['count'][s, :r]
        ids = arrays['ids'][s, :r]
        complete = ((st == 0) | (st == STATUS_AMBIGUOUS)) & (cnt <= CANDS) & (cnt > 0)
        totals['complete_rows'] += int(complete.sum())
        totals['column_rows'] += int(((st == STATUS_AMBIGUOUS) & (cnt <= CANDS)).sum())
        totals['dense_rows'] += int(
            ((st != 0) & ~((st == STATUS_AMBIGUOUS) & (cnt <= CANDS))).sum()
        )
        for row in np.nonzero(complete)[0]:
            k = int(cnt[row])
            cand = arrays['cand'][s, row, :k]
            rlo = arrays['rlo'][s, row, :k]
            rhi = arrays['rhi'][s, row, :k]
            hit = np.nonzero(cand == ids[row])[0]
            best = rlo.max()
            problem = None
            if not len(hit):
                totals['id_not_in_cand'] += 1
                problem = 'id_not_in_cand'
            elif rhi[hit].max() < best:
                totals['id_rhi_below_max_rlo'] += 1
                problem = 'id_rhi_below_max_rlo'
            elif st[row] == 0 and rlo[hit].max() != best:
                totals['status0_id_not_max_rlo'] += 1
                problem = 'status0_id_not_max_rlo'
            if problem and len(examples) < 50:
                examples.append(
                    {
                        'kind': problem,
                        'k': rec.get('k'),
                        'row': int(row),
                        **row_state(arrays, s, int(row)),
                    }
                )
        # Emission: predicted ids of accepted rows equal the head's ids.
        bs = int(rec.get('batch_size', 0))
        for i in range(min(bs, CANDS // ROWS_PER_REQUEST)):
            acc = int(arrays['accept'][s, i])
            for j in range(min(acc, ROWS_PER_REQUEST)):
                row = ROWS_PER_REQUEST * i + j
                if row >= r:
                    break
                totals['emission_rows'] += 1
                if int(arrays['predict'][s, row]) != int(ids[row]):
                    totals['emission_mismatch'] += 1
                    if len(examples) < 50:
                        examples.append(
                            {
                                'kind': 'emission',
                                'k': rec.get('k'),
                                'row': row,
                                'predict': int(arrays['predict'][s, row]),
                                'head_id': int(ids[row]),
                            }
                        )
    return {'totals': totals, 'examples': examples}


def launch_report(launch: str, out: Path, ref: list[int], prompt_len: int) -> dict[str, Any]:
    launch_dir = out / launch
    ring = launch_dir / 'ring'
    windows = point_windows(launch_dir)
    report: dict[str, Any] = {'launch': launch, 'ring_files': [], 'target_steps': []}
    deviations = ring / 'deviations.jsonl'
    report['deviations'] = deviations.read_text().splitlines() if deviations.exists() else []
    checks = sorted(launch_dir.glob('*/2026*/graph_check.json'))
    report['graph_check'] = [json.loads(p.read_text()) for p in checks]
    totals: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for npz, jsonl in ring_files(ring):
        records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        steps = arrays['steps'].tolist()
        if [r.get('k') for r in records] != steps:
            report['ring_files'].append({'file': npz.name, 'error': 'jsonl and npz steps differ'})
            continue
        report['ring_files'].append({'file': npz.name, 'steps': len(steps)})
        result = consistency(arrays, records)
        for key, value in result['totals'].items():
            totals[key] = totals.get(key, 0) + value
        examples += result['examples'][: max(0, 50 - len(examples))]
        for hit in find_target(records, arrays['predict'], arrays['accept'], ref, prompt_len):
            s, rec = hit['index'], records[hit['index']]
            row, gate = hit['row'], bool(arrays['gate'][s])
            state = row_state(arrays, s, row) if gate and row < CANDS else None
            i = hit['request']
            entry = {
                'point': which_point(rec['t'], windows),
                'k': rec['k'],
                't': rec['t'],
                'batch_size': rec['batch_size'],
                'rows': rec['rows'],
                'request': i,
                'slot': hit['slot'],
                'offset': hit['offset'],
                'output_len': hit['output_len'],
                'output_len_host': hit['host_output_len'],
                'row': row,
                'gate': gate,
                'host_gate': bool(rec['host_gate']),
                'valid': int(arrays['valid'][s]),
                'any': bool(arrays['any'][s]),
                'any_cols': bool(arrays['any_cols'][s]),
                'any_dense': bool(arrays['any_dense'][s]),
                'predict': arrays['predict'][
                    s, ROWS_PER_REQUEST * i : ROWS_PER_REQUEST * (i + 1)
                ].tolist(),
                'accept': int(arrays['accept'][s, i]),
                'head_ids': arrays['ids'][
                    s, ROWS_PER_REQUEST * i : ROWS_PER_REQUEST * (i + 1)
                ].tolist()
                if gate and ROWS_PER_REQUEST * (i + 1) <= CANDS
                else None,
                'state': state,
            }
            entry['readings'] = readings(gate, entry['host_gate'], entry['valid'], row, state)
            report['target_steps'].append(entry)
    report['consistency'] = totals
    report['consistency_examples'] = examples
    tokens = {}
    for name, _, _, point in windows:
        record = drain.target_record(point)
        if record and 'output' in record and len(record['output']) > POSITION:
            tokens[name] = record['output'][POSITION]
    report['token_at_439_by_point'] = tokens
    report['events'] = sorted(name for name, tok in tokens.items() if tok == WRONG)
    report['pass_ms_c64'] = pass_times(launch_dir)
    return report


def pass_times(launch_dir: Path) -> list[dict[str, Any]]:
    out = []
    for sweep in sorted(launch_dir.glob('*/2026*/sweep.json')):
        for point in json.loads(sweep.read_text()).get('points', []):
            if int(point.get('concurrency', 0)) == drain.TOP:
                out.append(
                    {
                        'repeat': point.get('repeat'),
                        'server_ms_per_pass': server_ms_per_pass(point),
                        'foreign_cpu_mean': point.get('foreign_cpu_during_mean'),
                    }
                )
    return out


def report(out: Path, runs: Path) -> dict[str, Any]:
    ref = reference_output(runs)
    s1 = next(p for n, p in drain.point_dirs(out, runs) if n.startswith('s1/'))
    prompt_len = len(
        next(
            r
            for r in drain.requests(s1)
            if r['phase'] == 'profiling' and r['prompt'].startswith(drain.TARGET[0])
        )['input']
    )
    launches = {}
    for name, launch in drain.LAUNCHES.items():
        if launch.variant == 'certring' and (out / name / 'ring').exists():
            launches[name] = launch_report(name, out, ref, prompt_len)
    timing = {}
    for name, launch in drain.LAUNCHES.items():
        if launch.hold in ('h6a', 'h7a', 'h7b') and (out / name).exists():
            checks = sorted((out / name).glob('*/2026*/graph_check.json'))
            timing[name] = {
                'variant': launch.variant,
                'c64': pass_times(out / name),
                'graph_check': [json.loads(c.read_text())['problems'] for c in checks],
            }
    return {'prompt_len': prompt_len, 'launches': launches, 'pass_ms_c64': timing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True, help='the drain output directory')
    parser.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--csv', type=Path, help="579ae7ce's position-439 rows as CSV")
    args = parser.parse_args(argv)
    result = report(args.out, args.runs)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + '\n')
    rows = []
    for name, entry in result['launches'].items():
        print(
            f'{name}: events {entry["events"]}; consistency {entry["consistency"]}; deviations {len(entry["deviations"])}'
        )
        for t in entry['target_steps']:
            state = t['state'] or {}
            rows.append(
                {
                    'launch': name,
                    **{
                        k: t[k]
                        for k in (
                            'point',
                            'k',
                            'batch_size',
                            'rows',
                            'row',
                            'gate',
                            'host_gate',
                            'valid',
                            'any_cols',
                            'any_dense',
                            'accept',
                        )
                    },
                    'predict': ' '.join(map(str, t['predict'])),
                    'head_id': state.get('head_id'),
                    'status': state.get('status'),
                    'count': state.get('count'),
                    'wrong_in_cand': state.get('wrong_in_cand'),
                    'readings': ' '.join(map(str, t['readings'])),
                }
            )
            print(f'  {rows[-1]}')
    if args.csv is not None and rows:
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
