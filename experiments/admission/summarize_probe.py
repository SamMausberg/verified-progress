"""Summarize one admission probe or confirmation session directory into a CSV.

A directory holds one bench.sweep run per label (<dir>/<label>/<run>/r0/c<NNN>/point.json).
For every point the CSV row gives y, y_steady, x_e2e (includes TTFT), x_decode, TTFT
p50/p99, accept length, prefill batches, output tokens, span and foreign CPU load,
the ratios of y and x to the `--plain` label at the same concurrency, and, for each
`--pair TEST=BASE`, the ratios to BASE and the first-divergence count of greedy token
ids against BASE (needs --return-token-ids on both runs). The last three columns count
the point's prefill batches above, at and well below the 16-request prefill cap, from its
server's log. Every label and concurrency the hold script ran is declared with `--expect`:
the directory must hold exactly those points, each passing the validity checks (complete,
expected prompts, foreign CPU mean at most 2 cores), or the script stops before writing.

    python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/queue-delay \
        --expect plain-tuned=64,96 --expect plain-pd=64,96 --expect mtp-n0=64,96 \
        --expect mtp-pd=64,96 --expect replayssm=96 \
        --plain plain-tuned --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned \
        --out evidence/admission/probe2_prefill_delayer.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.results import iter_jsonl, prompt_hash

FIELDS = [
    'label', 'concurrency', 'session', 'y', 'y_steady', 'x_e2e', 'x_decode', 'ttft_p50_ms',
    'ttft_p99_ms', 'accept_length', 'prefill_batches', 'requests', 'output_tokens', 'span_s',
    'foreign_cpu_mean', 'foreign_cpu_max', 'y_vs_plain', 'x_vs_plain', 'base', 'y_vs_base',
    'x_vs_base', 'identical', 'diverged', 'length_mismatch', 'exposure_tokens',
    'divergences_per_1k', 'prefill_batches_over_16', 'prefill_batches_at_16',
    'prefill_batches_of_1_or_2',
]  # fmt: skip
# The delayed arms pass --prefill-max-requests 16 (run_queue_delay_probe.sh:20,
# run_natural_probe.sh:35, run_admission_confirm.sh:32), which caps every prefill batch at 16
# requests (PrefillAdder.add_one_req, python/sglang/srt/managers/schedule_policy.py:1350 at
# SGLang bd66ce34). Batches at 16 are the most the cap can have cut.
PREFILL_CAP = 16
_NEW_SEQ = re.compile(r'#new-seq: (\d+)')


def prefill_batch_sizes(run: Path, levels: list[int]) -> dict[int, list[int]]:
    """Requests per prefill batch at each concurrency of a run, from the server's log.

    SGLang logs one "Prefill batch" line per prefill forward with its request count. bench.sweep
    flushes the cache before every point, so the n-th flush starts the n-th concurrency of the
    run's sweep.json.
    """
    segments: list[list[int]] = []
    for line in (run / 'server/server.log').read_text(errors='replace').splitlines():
        if 'Cache flushed successfully' in line:
            segments.append([])
        elif segments and 'Prefill batch' in line:
            match = _NEW_SEQ.search(line)
            if match is None:
                raise SystemExit(f'{run}: prefill line without #new-seq: {line}')
            segments[-1].append(int(match.group(1)))
    if len(segments) != len(levels):
        raise SystemExit(f'{run}: {len(segments)} cache flushes for {len(levels)} concurrencies')
    return dict(zip(levels, segments, strict=True))


def output_ids(point_dir: Path) -> dict[str, list[int]]:
    """Prompt hash -> greedy output token ids of every measured request of a point."""
    out: dict[str, list[int]] = {}
    for record in iter_jsonl(point_dir / 'aiperf/profile_export_raw.jsonl.gz'):
        if record.get('metadata', {}).get('benchmark_phase') != 'profiling':
            continue
        messages = record.get('payload', {}).get('messages') or [{}]
        key = prompt_hash(messages[-1].get('content', ''))
        ids = None
        for response in record.get('responses', []):
            for packet in response.get('packets', []):
                value = packet.get('value')
                if isinstance(value, str) and value.startswith('{') and '_ids' in value:
                    ext = json.loads(value).get('sglext') or {}
                    if ext.get('output_ids'):
                        ids = list(ext['output_ids'][0])
        if ids is None:
            raise SystemExit(f'{point_dir}: a request has no output ids')
        if key in out:
            raise SystemExit(f'{point_dir}: prompt {key} measured twice')
        out[key] = ids
    return out


def compare(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, int]:
    """First-divergence counts of two runs over the same prompts."""
    if set(a) != set(b):
        raise SystemExit('the two runs measured different prompts')
    diverged = exposure = length_mismatch = 0
    for key in sorted(a):
        ta, tb = a[key], b[key]
        n = min(len(ta), len(tb))
        d = next((i for i in range(n) if ta[i] != tb[i]), None)
        if d is None:
            exposure += n
            length_mismatch += len(ta) != len(tb)
        else:
            diverged += 1
            exposure += d + 1
    return {
        'identical': len(a) - diverged - length_mismatch,
        'diverged': diverged,
        'length_mismatch': length_mismatch,
        'exposure_tokens': exposure,
    }


def load_points(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    points: dict[tuple[str, int], dict[str, Any]] = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # bench.sweep names each run YYYYMMDD-HHMMSS.
        runs = sorted(label_dir.glob('[0-9]*-[0-9]*/r0'))
        if not runs:
            continue
        if len(runs) != 1:
            raise SystemExit(f'{label_dir}: expected one run, found {len(runs)}')
        sweep_json = runs[0].parent / 'sweep.json'
        if not sweep_json.exists():
            raise SystemExit(f'{sweep_json} missing: the sweep did not finish')
        sweep = json.loads(sweep_json.read_text())
        # The session name pairs points across sessions (analyze_confirm.py): it must be set.
        if not sweep.get('session'):
            raise SystemExit(f'{sweep_json}: no session name')
        # Every concurrency the sweep was asked for must have a point.
        requested = {int(c) for c in sweep['concurrency']}
        observed = {
            int(json.loads(p.read_text())['concurrency']) for p in runs[0].glob('c*/point.json')
        }
        if observed != requested:
            raise SystemExit(
                f'{label_dir.name}: points at c = {sorted(observed)}, '
                f'sweep asked for c = {sorted(requested)}'
            )
        # bench.sweep runs one repeat's points in ascending concurrency, whatever the order on
        # its command line (Sweep.run, bench/sweep.py:429); a second repeat would reverse it.
        if sweep.get('repeats', 1) != 1:
            raise SystemExit(f'{sweep_json}: {sweep["repeats"]} repeats; one is supported')
        sizes = prefill_batch_sizes(runs[0].parent, sorted(requested))
        for point_json in sorted(runs[0].glob('c*/point.json')):
            p = json.loads(point_json.read_text())
            problems = []
            if p['failed'] or p['completed'] != p['requests'] or p['osl_mismatch']:
                problems.append('incomplete')
            if not p['prompts_as_expected']:
                problems.append('prompts')
            if p['foreign_cpu_during_mean'] > 2.0:
                problems.append('host_contention')
            if problems:
                raise SystemExit(f'{point_json}: invalid ({", ".join(problems)})')
            batch = sizes[int(p['concurrency'])]
            if len(batch) != p['server_log']['prefill_log_lines']:
                raise SystemExit(
                    f'{point_json}: {len(batch)} prefill lines after its cache flush, '
                    f'{p["server_log"]["prefill_log_lines"]} in point.json'
                )
            spec = p.get('spec') or {}
            points[(label_dir.name, p['concurrency'])] = {
                'label': label_dir.name,
                'concurrency': p['concurrency'],
                'session': sweep['session'],
                'y': p['y'],
                'y_steady': p['y_steady'],
                'x_e2e': p['x_e2e'],
                'x_decode': p['x_decode'],
                'ttft_p50_ms': p['ttft_ms']['p50'],
                'ttft_p99_ms': p['ttft_ms']['p99'],
                'accept_length': spec.get('accept_length'),
                'prefill_batches': p['server_log']['prefill_log_lines'],
                'requests': p['requests'],
                'output_tokens': p['output_tokens'],
                'span_s': p['span_s'],
                'foreign_cpu_mean': p['foreign_cpu_during_mean'],
                'foreign_cpu_max': p['foreign_cpu_during_max'],
                'prefill_batches_over_16': sum(n > PREFILL_CAP for n in batch),
                'prefill_batches_at_16': sum(n == PREFILL_CAP for n in batch),
                'prefill_batches_of_1_or_2': sum(n <= 2 for n in batch),
                '_dir': point_json.parent,
            }
    if not points:
        raise SystemExit(f'{root}: no points')
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root', type=Path)
    parser.add_argument(
        '--expect',
        action='append',
        required=True,
        metavar='LABEL=C1,C2,...',
        help='a label the hold script ran and its concurrencies; one per label, all required',
    )
    parser.add_argument('--plain', default='plain-tuned')
    parser.add_argument('--pair', action='append', default=[], metavar='TEST=BASE')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected: set[tuple[str, int]] = set()
    for item in args.expect:
        label, _, levels = item.partition('=')
        try:
            expected |= {(label, int(c)) for c in levels.split(',')}
        except ValueError:
            parser.error(f'--expect {item}: not LABEL=C1,C2,...')
        if not label:
            parser.error(f'--expect {item}: no label')
    pairs = dict(item.split('=', 1) for item in args.pair)
    declared = {label for label, _ in expected}
    undeclared = sorted({args.plain, *pairs, *pairs.values()} - declared)
    if undeclared:
        parser.error(f'not declared with --expect: {", ".join(undeclared)}')
    points = load_points(args.root.expanduser())
    # Every label and concurrency the hold script ran must be here and nothing else: a failed
    # sweep leaves no r0 directory, and its absence must stop the summary rather than drop an
    # arm or a comparison.
    found = set(points)
    if found != expected:
        raise SystemExit(
            f'{args.root}: missing {sorted(expected - found)}, '
            f'not declared {sorted(found - expected)}'
        )
    # Ratios and token comparisons pair arms of one hold: an arm rerun in another session
    # (a different server start, load and history) must not be mixed in.
    sessions = sorted({row['session'] for row in points.values()})
    if len(sessions) != 1:
        raise SystemExit(f'{args.root}: arms from several sessions {sessions}')
    # A paired test and its base must cover the same concurrencies: a sweep that aborted
    # part-way must not leave a comparison silently incomplete. Plain ratios are filled
    # wherever a plain point exists (arms may run at concurrencies plain does not).
    for test, base_label in pairs.items():
        levels = {c for label, c in points if label == test}
        base_levels = {c for label, c in points if label == base_label}
        if levels != base_levels:
            raise SystemExit(
                f'{test} ran at c = {sorted(levels)} but {base_label} at c = {sorted(base_levels)}'
            )
    for (label, c), row in points.items():
        plain = points.get((args.plain, c))
        if plain:
            row['y_vs_plain'] = row['y'] / plain['y']
            row['x_vs_plain'] = row['x_e2e'] / plain['x_e2e']
        base_label = pairs.get(label)
        base = points.get((base_label, c)) if base_label else None
        if base_label and base is None:
            raise SystemExit(f'{label} c={c}: no {base_label} point to pair with')
        if base:
            row['base'] = base_label
            row['y_vs_base'] = row['y'] / base['y']
            row['x_vs_base'] = row['x_e2e'] / base['x_e2e']
            cmp = compare(output_ids(base['_dir']), output_ids(row['_dir']))
            row.update(cmp)
            row['divergences_per_1k'] = 1000 * cmp['diverged'] / cmp['exposure_tokens']
    rows = sorted(points.values(), key=lambda r: (r['label'], r['concurrency']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()}
            )
    tmp.replace(args.out)
    print(f'{len(rows)} points -> {args.out}')


if __name__ == '__main__':
    main()
