"""Freeze per-prompt natural output lengths into a sensitivity workload.

Reads one natural-stopping sweep point that served every prompt of a workload
split exactly once (greedy, thinking on, `--no-ignore-eos`, a cap on output
tokens) and writes the split again with each record's `output_length` set to the
tokens that prompt generated, capped. bench.sweep sends those lengths per request
(aiperf `output_length`) with `ignore_eos`, so every arm generates the same
number of tokens for each prompt. A manifest beside the file records the source
run, the cap, the length distribution and the file hash.

    python -m bench.natural_workload ~/vp-data/bench/natural/natural-confirm-plain/<run> \\
        --workload bench/workloads/mixed-v2/confirm.jsonl --cap 2048 \\
        --out bench/workloads/mixed-v2-natural2048
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def natural_lengths(requests_csv: Path) -> dict[str, tuple[int, str]]:
    """prompt id -> (output tokens, finish reason) from a point's requests.csv."""
    lengths: dict[str, tuple[int, str]] = {}
    with requests_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row['ok'] != 'True':
                raise SystemExit(f'request {row["request_id"]} failed: {row["error"]}')
            if row['prompt_id'] in lengths:
                raise SystemExit(f'prompt {row["prompt_id"]} was served twice')
            lengths[row['prompt_id']] = (int(row['osl']), row['finish_reason'])
    return lengths


def build(
    workload: list[dict[str, Any]], lengths: dict[str, tuple[int, str]], cap: int
) -> list[dict[str, Any]]:
    missing = [item['id'] for item in workload if item['id'] not in lengths]
    if missing:
        raise SystemExit(f'{len(missing)} prompts have no natural length, e.g. {missing[:3]}')
    extra = set(lengths) - {item['id'] for item in workload}
    if extra:
        raise SystemExit(f'{len(extra)} served prompts are not in the workload')
    return [{**item, 'output_length': min(lengths[item['id']][0], cap)} for item in workload]


def distribution(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        'n': len(ordered),
        'mean': statistics.fmean(ordered),
        'p10': ordered[len(ordered) // 10],
        'p50': ordered[len(ordered) // 2],
        'p90': ordered[(9 * len(ordered)) // 10],
        'min': ordered[0],
        'max': ordered[-1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run', type=Path, help='natural-stopping sweep run directory')
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--cap', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True, help='output directory')
    args = parser.parse_args(argv)
    manifest = json.loads((args.run / 'sweep.json').read_text())
    if manifest.get('ignore_eos'):
        raise SystemExit('the source run used ignore_eos; natural lengths need natural stopping')
    if int(manifest['osl']) != args.cap:
        raise SystemExit(f'the source run capped outputs at {manifest["osl"]}, not {args.cap}')
    points = manifest.get('points') or []
    if len(points) != 1:
        raise SystemExit('expected one point in the source run')
    point_dir = args.run / f'r{points[0]["repeat"]}' / f'c{points[0]["concurrency"]:03d}'
    lengths = natural_lengths(point_dir / 'requests.csv')
    workload = [json.loads(line) for line in args.workload.read_text().splitlines() if line]
    records = build(workload, lengths, args.cap)
    args.out.mkdir(parents=True, exist_ok=True)
    data = ''.join(json.dumps(record) + '\n' for record in records)
    out_file = args.out / args.workload.name
    out_file.write_text(data)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_domain[record['domain']].append(record['output_length'])
    finish: dict[str, int] = defaultdict(int)
    for _, reason in lengths.values():
        finish[reason] += 1
    summary = {
        'name': args.out.name,
        'source_workload': str(args.workload),
        'source_workload_sha256': hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        'source_run': str(args.run),
        'source_arm': manifest['arm'],
        'request_body': manifest.get('request_body'),
        'cap': args.cap,
        'finish_reasons': dict(finish),
        'at_cap': sum(1 for r in records if r['output_length'] >= args.cap),
        'output_length': distribution([r['output_length'] for r in records]),
        'output_length_by_domain': {d: distribution(v) for d, v in sorted(by_domain.items())},
        'file': out_file.name,
        'sha256': hashlib.sha256(data.encode()).hexdigest(),
    }
    (args.out / 'manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('at_cap', 'output_length', 'sha256')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
