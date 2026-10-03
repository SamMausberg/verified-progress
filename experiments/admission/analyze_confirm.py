"""Declared analysis of the admission-batching confirmation (three sessions).

Reads one summarize_probe.py CSV per session and applies the rules declared before session 0
in experiments/admission/README.md ("Declared analysis of the confirmation"):

- primary, c = 48, 64, 96, 128: y of mtp-delay over the best non-speculative arm of the same
  session (the highest y of plain-tuned, plain-delay and replayssm); it leads if the ratio
  exceeds 1 in every session;
- c = 32 and 48: the same against the best of dflash and dflash-delay;
- c = 1 and 8: y and x of mtp-delay over mtp-n0; no harm if the mean ratio is at least 0.99
  and no session is below 0.98 (applied to y and to x separately).

Writes, only after every check passes:
- confirm_points.csv: every point of every session (the session CSVs, concatenated);
- confirm_arms.csv: per label and concurrency, the mean, min and max over sessions of y,
  x_e2e and TTFT p50/p99, mean prefill batches, mean divergence rate against the undelayed twin;
- confirm_analysis.csv: one row per declared comparison, concurrency and metric, with each
  session's ratio and base. For the y comparisons an x_e2e row follows against the same
  per-session base (reported, no rule).

    python experiments/admission/analyze_confirm.py s0.csv s1.csv s2.csv --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

# Rules copied from experiments/admission/README.md, "Declared analysis of the confirmation"
# (lines 53-60), declared in commit 7b35508 before session 0 and unchanged since.
PRIMARY_LEVELS = (48, 64, 96, 128)
NON_SPECULATIVE = ('plain-tuned', 'plain-delay', 'replayssm')
DFLASH_LEVELS = (32, 48)
DFLASH = ('dflash', 'dflash-delay')
LOW_LEVELS = (1, 8)
NO_HARM_MEAN = 0.99
NO_HARM_FLOOR = 0.98
TEST = 'mtp-delay'
LOW_BASE = 'mtp-n0'
# Every label and concurrency experiments/admission/run_admission_confirm.sh runs per session.
EXPECTED = {
    'plain-tuned': (32, 48, 64, 96, 128),
    'replayssm': (32, 48, 64, 96, 128),
    'plain-delay': (32, 48, 64, 96, 128),
    'mtp-n0': (1, 8, 32, 48, 64, 96, 128),
    'mtp-delay': (1, 8, 32, 48, 64, 96, 128),
    'dflash': (32, 48),
    'dflash-delay': (32, 48),
}
# Each delayed arm's undelayed twin, as the session CSVs must pair them (summarize_probe.py
# --pair), so every token-identity rate compares the same arms.
TWINS = {'mtp-delay': 'mtp-n0', 'plain-delay': 'plain-tuned', 'dflash-delay': 'dflash'}
METRICS = ('y', 'x_e2e', 'ttft_p50_ms', 'ttft_p99_ms')
Points = dict[tuple[str, str, int], dict[str, str]]


def load(paths: list[Path]) -> tuple[list[str], Points, list[str]]:
    sessions: list[str] = []
    points: Points = {}
    fields: list[str] = []
    expected = {(label, c) for label, levels in EXPECTED.items() for c in levels}
    for path in paths:
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if fields and reader.fieldnames != fields:
                raise SystemExit(f'{path}: columns differ from {paths[0]}')
            fields = list(reader.fieldnames or [])
        names = {row['session'] for row in rows}
        if len(names) != 1:
            raise SystemExit(f'{path}: expected one session, found {sorted(names)}')
        session = names.pop()
        if session in sessions:
            raise SystemExit(f'{path}: session {session} given twice')
        sessions.append(session)
        seen = set()
        for row in rows:
            key = (row['label'], int(row['concurrency']))
            if key in seen:
                raise SystemExit(f'{path}: {key} twice')
            seen.add(key)
            twin = TWINS.get(key[0], '')
            if row['base'] != twin or bool(twin) != bool(row['divergences_per_1k']):
                raise SystemExit(f'{path}: {key} paired with {row["base"]!r}, expected {twin!r}')
            points[(session, *key)] = row
        if seen != expected:
            raise SystemExit(
                f'{path}: missing {sorted(expected - seen)}, unexpected {sorted(seen - expected)}'
            )
    return sessions, points, fields


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ratio_row(
    comparison: str,
    c: int,
    metric: str,
    bases: tuple[str, ...],
    chosen: list[str],
    ratios: list[float],
    rule: str,
    verdict: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        'comparison': comparison,
        'concurrency': c,
        'metric': metric,
        'test': TEST,
        'bases': ' '.join(bases),
    }
    for i, (base, ratio) in enumerate(zip(chosen, ratios, strict=True)):
        row[f's{i}_base'] = base
        row[f's{i}_ratio'] = ratio
    row.update(
        mean=statistics.fmean(ratios), min=min(ratios), max=max(ratios), rule=rule, verdict=verdict
    )
    return row


def best_of(
    comparison: str, sessions: list[str], points: Points, c: int, bases: tuple[str, ...]
) -> list[dict[str, Any]]:
    """y of TEST over the highest-y arm of `bases` in each session, then x against that arm."""

    def best(session: str) -> str:
        return max(bases, key=lambda b: num(points[(session, b, c)], 'y'))

    chosen = [best(s) for s in sessions]
    rows = []
    for metric in ('y', 'x_e2e'):
        ratios = [
            num(points[(s, TEST, c)], metric) / num(points[(s, b, c)], metric)
            for s, b in zip(sessions, chosen, strict=True)
        ]
        if metric == 'y':
            rule = 'leads if every session > 1'
            verdict = 'leads' if min(ratios) > 1 else 'does not lead'
        else:
            rule, verdict = 'reported', ''
        rows.append(ratio_row(comparison, c, metric, bases, chosen, ratios, rule, verdict))
    return rows


def no_harm(sessions: list[str], points: Points, c: int) -> list[dict[str, Any]]:
    rows = []
    for metric in ('y', 'x_e2e'):
        ratios = [
            num(points[(s, TEST, c)], metric) / num(points[(s, LOW_BASE, c)], metric)
            for s in sessions
        ]
        ok = statistics.fmean(ratios) >= NO_HARM_MEAN and min(ratios) >= NO_HARM_FLOOR
        rows.append(
            ratio_row(
                'low concurrency',
                c,
                metric,
                (LOW_BASE,),
                [LOW_BASE] * len(sessions),
                ratios,
                f'no harm if mean >= {NO_HARM_MEAN} and every session >= {NO_HARM_FLOOR}',
                'no harm' if ok else 'harm',
            )
        )
    return rows


def arm_rows(sessions: list[str], points: Points) -> list[dict[str, Any]]:
    rows = []
    for label, levels in EXPECTED.items():
        for c in levels:
            group = [points[(s, label, c)] for s in sessions]
            row: dict[str, Any] = {'label': label, 'concurrency': c, 'n_sessions': len(group)}
            for metric in METRICS:
                values = [num(p, metric) for p in group]
                row[f'{metric}_mean'] = statistics.fmean(values)
                row[f'{metric}_min'] = min(values)
                row[f'{metric}_max'] = max(values)
            row['prefill_batches_mean'] = statistics.fmean(num(p, 'prefill_batches') for p in group)
            rates = [num(p, 'divergences_per_1k') for p in group if p['divergences_per_1k']]
            if rates and len(rates) != len(group):
                raise SystemExit(f'{label} c={c}: divergence rate missing in some sessions')
            row['base'] = group[0]['base']
            row['divergences_per_1k_mean'] = statistics.fmean(rates) if rates else ''
            row['foreign_cpu_mean_max'] = max(num(p, 'foreign_cpu_mean') for p in group)
            rows.append(row)
    return rows


def write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()}
            )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('sessions', type=Path, nargs='+', help='one summarize_probe.py CSV each')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--expect-sessions', type=int, default=3, help='3 as declared; fewer only for dry runs'
    )
    args = parser.parse_args()
    if len(args.sessions) != args.expect_sessions:
        raise SystemExit(f'{len(args.sessions)} session CSVs, expected {args.expect_sessions}')
    sessions, points, point_fields = load(args.sessions)
    analysis = []
    for c in PRIMARY_LEVELS:
        analysis += best_of('primary', sessions, points, c, NON_SPECULATIVE)
    for c in DFLASH_LEVELS:
        analysis += best_of('against dflash', sessions, points, c, DFLASH)
    for c in LOW_LEVELS:
        analysis += no_harm(sessions, points, c)
    per_session = [f's{i}_{k}' for i in range(len(sessions)) for k in ('base', 'ratio')]
    analysis_fields = [
        'comparison', 'concurrency', 'metric', 'test', 'bases', *per_session,
        'mean', 'min', 'max', 'rule', 'verdict',
    ]  # fmt: skip
    arms = arm_rows(sessions, points)
    all_points = [points[key] for key in sorted(points)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / 'confirm_points.csv', all_points, point_fields)
    write(args.out_dir / 'confirm_arms.csv', arms, list(arms[0]))
    write(args.out_dir / 'confirm_analysis.csv', analysis, analysis_fields)
    for row in analysis:
        ratios = ' '.join(f'{row[f"s{i}_ratio"]:.3f}' for i in range(len(sessions)))
        print(
            f'{row["comparison"]:<16} c={row["concurrency"]:>3} {row["metric"]:<5} '
            f'{ratios} mean {row["mean"]:.3f} {row["verdict"]}'
        )


if __name__ == '__main__':
    main()
