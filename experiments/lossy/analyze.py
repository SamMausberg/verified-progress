"""Declared analysis of the lossy-lever study (experiments/lossy/README.md).

    python -m experiments.lossy.analyze --points <dir>/points.csv \
        --quality <q1 GSM8K run dir> <q2 GSM8K run dir> \
        --probes <probe dir> [...] --probe-noise <load test dir> --out evidence/lossy

Speed: from bench.pareto's points.csv over the three session runs (only points it
marks valid), per arm and concurrency the session means of y and x; for each
matched pair (plan.PAIRS) and session the ratio of the lossy arm to its baseline;
and per lever the envelope ratio: the lever's best arm over the best exact arm,
each chosen by its mean y over the sessions, paired within each session. A
ratio is "faster" when every session's value exceeds 1 + plan.BAND, "slower"
when every one is below 1 - plan.BAND, otherwise "no detectable change". A point
needs plan.MIN_SESSIONS valid sessions to be ranked or decided.

Quality: each lever's GSM8K run against the two committed reference runs
(bench.quality.compare: accuracy difference, exact McNemar test, and a 95%
interval for the paired difference), and the logit probe's score-mode top-1
agreement and mean top-20 KL against the reference, beside the reference's own
run-to-run values. The budget is met when the GSM8K difference is at least
-1.0 point against both references, the agreement at least 0.98 and the KL at
most 0.01 nats.

Every output is written only after all inputs pass their checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.lossy import plan

GSM8K_BUDGET_PT = -1.0
# Slow-launch check (pre-run revision of 2026-10-02): see launch_flags().
SLOW_TTFT_MS = 3.0
SLOW_PASS_FRACTION = 0.02
# Sessions a ratio needs in the sensitivity analysis without flagged launches.
MIN_SESSIONS_WITHOUT_FLAGGED = 2
AGREEMENT_MIN = 0.98
KL_MAX = 0.01


def load_points(path: Path) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    """(label, c) -> session -> {'y', 'x'} for valid points of the declared sessions."""
    table: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row['session'] not in plan.SESSIONS or row['invalid_reason']:
                continue
            key = (row['label'], int(row['concurrency']))
            if row['session'] in table[key]:
                raise ValueError(f'two valid points for {key} in {row["session"]}')
            accept = float(row['accept_length']) if row.get('accept_length') else 1.0
            table[key][row['session']] = {
                'y': float(row['y']),
                'x': float(row['x_e2e']),
                'ttft_p50_ms': float(row['ttft_p50_ms']),
                # Time per forward pass: ITL is per output token, a verify pass emits
                # accept_length tokens on average (1 for plain decoding).
                'pass_ms': float(row['itl_p50_ms']) * accept,
            }
    return table


def check_plan(table: dict[tuple[str, int], dict[str, dict[str, float]]]) -> list[str]:
    """Declared points without the required number of valid sessions."""
    missing = []
    for launch in plan.SESSION_LAUNCHES:
        if launch.arm in plan.DROPPED_ARMS:
            continue
        for c in launch.concurrency:
            n = len(table.get((launch.arm, c), {}))
            if n < plan.MIN_SESSIONS:
                missing.append(f'{launch.arm} c={c}: {n} valid sessions')
    return missing


def classify(values: list[float], min_sessions: int = plan.MIN_SESSIONS) -> str:
    if len(values) < min_sessions:
        return 'not decided (fewer sessions)'
    if all(v > 1 + plan.BAND for v in values):
        return 'faster'
    if all(v < 1 - plan.BAND for v in values):
        return 'slower'
    return 'no detectable change'


def ratio_record(
    num: dict[str, dict[str, float]],
    den: dict[str, dict[str, float]],
    metric: str,
    min_sessions: int = plan.MIN_SESSIONS,
) -> dict[str, Any]:
    sessions = sorted(set(num) & set(den))
    values = [num[s][metric] / den[s][metric] for s in sessions]
    return {
        'sessions': sessions,
        'per_session': values,
        'mean': statistics.fmean(values) if values else math.nan,
        'min': min(values) if values else math.nan,
        'max': max(values) if values else math.nan,
        'decision': classify(values, min_sessions),
    }


def mean_y(entry: dict[str, dict[str, float]]) -> float:
    return statistics.fmean(v['y'] for v in entry.values())


def concurrencies() -> list[int]:
    return sorted({c for launch in plan.SESSION_LAUNCHES for c in launch.concurrency})


def launch_flags(
    table: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """The declared slow-launch check, per arm and session.

    At each of the launch's concurrencies, compare its TTFT p50 and time per pass
    with the median over the arm's sessions there. A launch is flagged when, at a
    majority of its concurrencies, TTFT p50 exceeds that median by more than
    SLOW_TTFT_MS and the time per pass exceeds it by more than SLOW_PASS_FRACTION.
    """
    per_launch: dict[str, dict[str, Any]] = {}
    for (arm, c), entry in sorted(table.items()):
        if len(entry) < 2:
            continue
        ttft_median = statistics.median(v['ttft_p50_ms'] for v in entry.values())
        pass_median = statistics.median(v['pass_ms'] for v in entry.values())
        for session, v in entry.items():
            record = per_launch.setdefault(
                f'{arm}@{session}', {'arm': arm, 'session': session, 'points': []}
            )
            d_ttft = v['ttft_p50_ms'] - ttft_median
            d_pass = v['pass_ms'] / pass_median - 1
            record['points'].append(
                {
                    'concurrency': c,
                    'ttft_excess_ms': d_ttft,
                    'pass_excess': d_pass,
                    'slow': d_ttft > SLOW_TTFT_MS and d_pass > SLOW_PASS_FRACTION,
                }
            )
    for record in per_launch.values():
        slow = sum(point['slow'] for point in record['points'])
        record['flagged'] = slow > len(record['points']) / 2
    return per_launch


def without(
    table: dict[tuple[str, int], dict[str, dict[str, float]]], launches: set[tuple[str, str]]
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    """The table without the given (arm, session) launches."""
    return {
        (arm, c): {s: v for s, v in entry.items() if (arm, s) not in launches}
        for (arm, c), entry in table.items()
    }


def speed(
    table: dict[tuple[str, int], dict[str, dict[str, float]]],
    min_sessions: int = plan.MIN_SESSIONS,
) -> dict[str, Any]:
    arms_at: dict[int, list[str]] = defaultdict(list)
    for launch in plan.SESSION_LAUNCHES:
        if launch.arm in plan.DROPPED_ARMS:
            continue
        for c in launch.concurrency:
            if len(table.get((launch.arm, c), {})) >= min_sessions:
                arms_at[c].append(launch.arm)
    means = [
        {
            'arm': arm,
            'concurrency': c,
            'sessions': len(entry),
            'y_mean': mean_y(entry),
            'y_sd': statistics.stdev([v['y'] for v in entry.values()]) if len(entry) > 1 else 0.0,
            'x_mean': statistics.fmean(v['x'] for v in entry.values()),
            'exact': arm in plan.EXACT_ARMS,
        }
        for (arm, c), entry in sorted(table.items(), key=lambda item: (item[0][0], item[0][1]))
        if entry
    ]
    pairs = []
    for test, base in plan.PAIRS:
        for c in concurrencies():
            if test in arms_at[c] and base in arms_at[c]:
                num, den = table[(test, c)], table[(base, c)]
                pairs.append(
                    {
                        'test': test,
                        'baseline': base,
                        'concurrency': c,
                        'y': ratio_record(num, den, 'y', min_sessions),
                        'x': ratio_record(num, den, 'x', min_sessions),
                    }
                )
    envelope = []
    for lever, lever_arms in plan.LEVERS.items():
        for c in concurrencies():
            exact = [a for a in arms_at[c] if a in plan.EXACT_ARMS]
            lossy = [a for a in arms_at[c] if a in lever_arms]
            if not exact or not lossy:
                continue
            best_exact = max(exact, key=lambda a: mean_y(table[(a, c)]))
            best_lossy = max(lossy, key=lambda a: mean_y(table[(a, c)]))
            num, den = table[(best_lossy, c)], table[(best_exact, c)]
            envelope.append(
                {
                    'lever': lever,
                    'concurrency': c,
                    'best_lossy': best_lossy,
                    'best_exact': best_exact,
                    'y': ratio_record(num, den, 'y', min_sessions),
                    'x': ratio_record(num, den, 'x', min_sessions),
                }
            )
    return {'means': means, 'pairs': pairs, 'envelope': envelope}


def paired_interval(only_ref: int, only_test: int, n: int) -> tuple[float, float]:
    """95% Wald interval of the paired accuracy difference (test - reference)."""
    if n <= 0:
        raise ValueError('no paired problems')
    d = (only_test - only_ref) / n
    variance_of_mean = ((only_test + only_ref) / n - d * d) / n
    half = 1.96 * math.sqrt(max(variance_of_mean, 0.0))
    return (d - half, d + half)


def gsm8k(run: Path, references: list[Path]) -> dict[str, Any]:
    from bench.quality import compare

    out = {'run': str(run), 'vs': []}
    for ref in references:
        result = compare(ref, run)
        low, high = paired_interval(
            result['correct_only_a'], result['correct_only_b'], result['problems']
        )
        result['delta_pt'] = 100 * result['accuracy_delta_b_minus_a']
        result['delta_ci95_pt'] = [100 * low, 100 * high]
        out['vs'].append(result)
    out['within_budget'] = all(r['delta_pt'] >= GSM8K_BUDGET_PT for r in out['vs'])
    return out


def probe(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    score = summary['score']
    generate = summary['generate']
    return {
        'arm': summary['arm'],
        'file': str(summary_path),
        'score_agreement': score['argmax_agreement'],
        'score_kl_mean': score['kl_mean'],
        'score_positions': score['positions'],
        'generate_first_divergence_median': generate.get('first_divergence_median'),
        'generate_divergences_per_1k': generate.get('divergences_per_1k_shared_tokens'),
        'generate_kl_mean_shared_prefix': generate.get('kl_mean'),
        'within_budget': score['argmax_agreement'] >= AGREEMENT_MIN
        and score['kl_mean'] <= KL_MAX,
    }


def probe_noise(load_test: Path) -> dict[str, Any]:
    """The reference's own values: ref-1 scored against itself, ref-2 against ref-1."""
    from experiments.moonshot.logit_probe import compare_runs

    ref1 = json.loads((load_test / 'plain-ref-1/probe_generate.json').read_text())
    out = {}
    for name, rel in (
        ('ref1_score_vs_ref1', 'plain-ref-1/probe_score.json'),
        ('ref2_score_vs_ref1', 'plain-ref-2/probe_score.json'),
        ('ref2_generate_vs_ref1', 'plain-ref-2/probe_generate.json'),
    ):
        result = compare_runs(ref1, json.loads((load_test / rel).read_text()))
        out[name] = {
            key: result.get(key)
            for key in (
                'argmax_agreement',
                'kl_mean',
                'positions',
                'first_divergence_median',
                'divergences_per_1k_shared_tokens',
            )
            if key in result
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--quality', type=Path, nargs='*', default=[])
    parser.add_argument('--references', type=Path, nargs='+', required=True)
    parser.add_argument('--probes', type=Path, nargs='*', default=[])
    parser.add_argument('--probe-noise', type=Path, default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help='write outputs although declared points lack valid sessions (they stay undecided)',
    )
    args = parser.parse_args(argv)
    table = load_points(args.points)
    missing = check_plan(table)
    if missing and not args.allow_missing:
        raise SystemExit('declared points without enough valid sessions:\n  ' + '\n  '.join(missing))
    flags = launch_flags(table)
    flagged = {(r['arm'], r['session']) for r in flags.values() if r['flagged']}
    result: dict[str, Any] = {'missing_points': missing, 'band': plan.BAND, **speed(table)}
    result['launch_flags'] = flags
    result['flagged_launches'] = sorted(f'{arm}@{session}' for arm, session in flagged)
    # Shown beside the primary verdict whenever a launch is flagged.
    result['without_flagged'] = (
        speed(without(table, flagged), MIN_SESSIONS_WITHOUT_FLAGGED) if flagged else None
    )
    result['gsm8k'] = [gsm8k(run, args.references) for run in args.quality]
    result['probes'] = [probe(path / 'probe_summary.json') for path in args.probes]
    if args.probe_noise is not None:
        result['probe_noise'] = probe_noise(args.probe_noise)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'decisions.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    rows = [
        ['kind', 'test', 'baseline', 'concurrency', 'metric', 'mean', 'min', 'max', 'n', 'decision']
    ]
    entries = []
    for prefix, part in (('', result), ('without-flagged-', result['without_flagged'])):
        if part is None:
            continue
        entries += [
            (f'{prefix}pair', p['test'], p['baseline'], p['concurrency'], p) for p in part['pairs']
        ]
        entries += [
            (f'{prefix}envelope-{e["lever"]}', e['best_lossy'], e['best_exact'], e['concurrency'], e)
            for e in part['envelope']
        ]
    for kind, test, base, c, record in entries:
        for metric in ('y', 'x'):
            r = record[metric]
            rows.append(
                [
                    kind,
                    test,
                    base,
                    c,
                    metric,
                    f'{r["mean"]:.4f}',
                    f'{r["min"]:.4f}',
                    f'{r["max"]:.4f}',
                    len(r['per_session']),
                    r['decision'],
                ]
            )
    with (args.out / 'ratios.csv').open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    print(json.dumps({'missing_points': missing}, indent=1))
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
