"""Figure data and figures for the stack campaign's results (evidence/stack/README.md).

Builds three tables from committed inputs only, then draws each figure from its table:

* frontier.csv (figure a): per arm and client concurrency, the mean over sessions of the
  per-user rate x_e2e and the throughput y (each session's value is the mean of the
  arm's launches at that concurrency), with their standard deviations over sessions and
  n. A session counts for an arm at a concurrency if analyze.py admitted it there and
  every launch of that arm is valid and present as often as the declared order runs it
  (S0's validity is not required here; it is for the ratios).
* ratios.csv (figure b): analyze.py's session-paired ratios against S0 (geometric mean,
  95% t interval, decision, session ratios) beside the declared expected range
  (expected.json) for F, G and the full stack. The declared full-stack range is FG's,
  because H was left out of the derivation; it is shown for both FG and FULL.
* gap_by_concurrency.csv and gap_by_block.csv (figure c): the measured full stack as a
  multiple of the tuned DFlash per-user rate next to the derived ceilings of ceiling.json,
  and the tokens a cycle must commit for 5x at each verify width from frame.json.
* last_lever.csv (no figure; not declared): the full stack against the stack without its
  last lever (FGH / FG when FULL = FGH) per session, from the same launches. The order runs
  the shorter stack once, in the middle, so this is a reading, not a test.

The tables need only the standard library; the figures need matplotlib (the SGLang
venv). Colours follow paper/figures/style.tex: the stock baseline in ink, arms with the
certified head in blue, the other arms in greys told apart by marker and dash pattern.

    python experiments/stack/figures.py --points evidence/stack/points.csv \
        --composition evidence/stack/composition.json --expected evidence/stack/expected.json \
        --ceiling evidence/stack/ceiling.json --frame evidence/frontier/frame.json \
        --out-dir evidence/stack
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

METRICS = ('x_e2e', 'y')
GOAL = 5.0  # the programme's goal: 5x the tuned DFlash per-user rate
INK, BLUE, MUTED = '#1B3A57', '#0072B2', '#6B7780'


def arm_order(full: str) -> list[str]:
    """S0, B0, each lever, the shorter cumulative stacks, then the full stack."""
    levers = list(full) if len(full) > 1 else []
    stacks = [full[:j] for j in range(2, len(full))]
    return ['S0', 'B0', *levers, *stacks, full]


def launches_expected(arm: str, full: str) -> int:
    return 2 if arm in ('S0', full) else 1


def _float(value: str) -> float:
    return float(value) if value not in ('', None) else math.nan


def session_means(
    points: list[dict[str, str]], comp: dict[str, Any]
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    """(arm, c) -> session -> metric -> mean over the arm's launches, for each session
    analyze.py admitted at c in which every launch of the arm is valid and present as often
    as the declared order runs it."""
    full = comp['full']
    valid: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    touched: set[tuple[str, int, str]] = set()
    for r in points:
        key = (r['session'], int(r['concurrency']), r['label'].removeprefix('stack-'))
        if r.get('invalid_reason'):
            touched.add(key)
        else:
            valid[key].append(r)
    out: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for arm in arm_order(full):
        for c_text, sessions in comp['sessions_admitted'].items():
            c = int(c_text)
            out[(arm, c)] = {}
            for s in sessions:
                launches = valid.get((s, c, arm), [])
                if (s, c, arm) in touched or len(launches) != launches_expected(arm, full):
                    continue
                out[(arm, c)][s] = {
                    m: statistics.fmean(_float(x[m]) for x in launches)
                    for m in (*METRICS, 'accept_length')
                }
    return out


def frontier_rows(points: list[dict[str, str]], comp: dict[str, Any]) -> list[dict[str, Any]]:
    means = session_means(points, comp)
    rows = []
    for arm in arm_order(comp['full']):
        for c in sorted(int(c) for c in comp['sessions_admitted']):
            per_session = list(means[(arm, c)].values())
            n = len(per_session)
            row: dict[str, Any] = {'arm': arm, 'c': c, 'n': n}
            for m in METRICS:
                vals = [v[m] for v in per_session]
                row[f'{m}_mean'] = round(statistics.fmean(vals), 2) if vals else ''
                row[f'{m}_std'] = round(statistics.stdev(vals), 2) if n > 1 else ''
            acc = [v['accept_length'] for v in per_session]
            row['accept_length_mean'] = round(statistics.fmean(acc), 3) if acc else ''
            rows.append(row)
    return rows


def last_lever_rows(points: list[dict[str, str]], comp: dict[str, Any]) -> list[dict[str, Any]]:
    """Not declared: the full stack against the stack without its last lever (FGH / FG),
    per session, from the same launches. The order is unbalanced (the shorter stack runs
    once, in the middle; the full stack second and second to last), so this is a reading
    of the last lever's effect, not a test: no interval and no decision."""
    full = comp['full']
    base = full[:-1]
    if len(full) < 2:
        return []
    means = session_means(points, comp)
    rows = []
    for c in sorted(int(c) for c in comp['sessions_admitted']):
        both = sorted(set(means[(full, c)]) & set(means[(base, c)]))
        for m in METRICS:
            ratios = [means[(full, c)][s][m] / means[(base, c)][s][m] for s in both]
            rows.append(
                {
                    'test': full,
                    'against': base,
                    'c': c,
                    'metric': m,
                    'n': len(ratios),
                    'geomean': round(math.exp(statistics.fmean(map(math.log, ratios))), 5)
                    if ratios
                    else '',
                    'min': round(min(ratios), 5) if ratios else '',
                    'max': round(max(ratios), 5) if ratios else '',
                    'sessions': ' '.join(f'{s}:{r:.5f}' for s, r in zip(both, ratios, strict=True)),
                    'status': 'not declared; unbalanced order; a reading, not a test',
                }
            )
    return rows


def ratio_rows(comp: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    full = comp['full']
    declared = {'F': 'F', 'G': 'G', 'FG': 'FG', full: 'FG'}
    rows = []
    for arm in arm_order(full)[1:]:
        for c_text, entry in sorted(comp['arms'][arm].items(), key=lambda kv: int(kv[0])):
            rng = expected['by_concurrency'][c_text]['expected_ratio'].get(declared.get(arm, ''))
            for m in METRICS:
                e = entry[m]
                rows.append(
                    {
                        'arm': arm,
                        'c': int(c_text),
                        'metric': m,
                        'n': e['n'],
                        'ratio': e.get('ratio', ''),
                        'lo': e.get('lo', ''),
                        'hi': e.get('hi', ''),
                        'decision': e.get('decision', ''),
                        'sessions': ' '.join(map(str, e['sessions'])),
                        'declared_lo': rng[0] if rng else '',
                        'declared_hi': rng[1] if rng else '',
                        'declared_for': ('FULL' if arm == full else arm) if rng else '',
                    }
                )
    return rows


def gap_rows(comp: dict[str, Any], ceiling: dict[str, Any]) -> list[dict[str, Any]]:
    full = comp['full']
    rows = []
    for c_text, ceil in sorted(ceiling['by_concurrency'].items(), key=lambda kv: int(kv[0])):
        e = comp['arms'][full][c_text]['x_e2e']
        ratio = e.get('ratio')
        rows.append(
            {
                'c': int(c_text),
                'baseline_x_e2e': ceil['baseline']['x_e2e'],
                'baseline_tau': ceil['baseline']['tau'],
                'full': full,
                'full_n': e['n'],
                'full_x_ratio': ratio if ratio is not None else '',
                'full_x_lo': e.get('lo', ''),
                'full_x_hi': e.get('hi', ''),
                'goal': GOAL,
                'full_short_of_goal': round(GOAL / ratio, 2) if ratio else '',
                'decode_ceiling_snapshot_free': ceil['decode_ceiling_x_snapshot_free'],
                'selector_bound_at_floor': ceil['selector_bound_at_floor_x'],
                'full_blocks_at_floor': ceil['full_blocks_at_floor_x'],
                'tau_for_5x_at_floor': ceil['tau_for_5x_at_floor_snapshot_free'],
            }
        )
    return rows


def block_rows(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for width, w in sorted(frame['widths'].items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                'block': int(width),
                'measured': 'INTERPOLATED' not in w['source'],
                'tokens_needed_with_draft': w['with_dflash_draft']['tokens_per_cycle_needed'],
                'tokens_needed_free_drafter': w['free_drafter']['tokens_per_cycle_needed'],
                'alpha_needed_with_draft': w['with_dflash_draft']['constant_alpha_needed'] or '',
                'cycle_us_with_draft': w['with_dflash_draft']['cycle_us'],
                'baseline_tokens_per_cycle': frame['baseline']['A_D'],
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


NAMES = {
    'S0': 'S0: tuned DFlash-16 (stock)',
    'B0': 'B0: composed, switches off',
    'F': 'F: snapshot-free verify',
    'G': 'G: GEMM routing table',
    'H': 'H: certified head',
}


def style(arm: str, full: str) -> dict[str, Any]:
    if arm == 'S0':
        return {'color': INK, 'marker': 'o', 'linestyle': '-', 'linewidth': 2.0}
    if arm == full:
        return {'color': BLUE, 'marker': 'D', 'linestyle': '-', 'linewidth': 2.0}
    if 'H' in arm:
        return {'color': BLUE, 'marker': 'v', 'linestyle': ':', 'linewidth': 1.3}
    others = {'B0': ('s', ':'), 'F': ('^', '--'), 'G': ('<', '-.'), 'FG': ('P', (0, (6, 2)))}
    marker, line = others.get(arm, ('x', '-'))
    return {'color': MUTED, 'marker': marker, 'linestyle': line, 'linewidth': 1.3}


def name(arm: str, full: str) -> str:
    if arm == full:
        return f'FULL = {full}'
    return NAMES.get(arm, arm)


def plot_frontier(rows: list[dict[str, str]], full: str, path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
    by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r['n'] not in ('', '0'):
            by_arm[r['arm']].append(r)
    for arm in arm_order(full):
        pts = sorted(by_arm.get(arm, []), key=lambda r: int(r['c']))
        if not pts:
            continue
        xs = [float(r['x_e2e_mean']) for r in pts]
        ys = [float(r['y_mean']) for r in pts]
        st = style(arm, full)
        ax.plot(xs, ys, markersize=6, label=name(arm, full), zorder=3 if arm in ('S0', full) else 2, **st)
    for r in by_arm.get('S0', []):
        ax.annotate(
            f'c = {r["c"]}', (float(r['x_e2e_mean']), float(r['y_mean'])), textcoords='offset points',
            xytext=(8, -12), fontsize=8, color=INK,
        )  # fmt: skip
    missing = [a for a in arm_order(full) if a not in by_arm]
    if missing:
        ax.text(
            0.02, 0.03, 'no valid session: ' + ', '.join(name(a, full) for a in missing),
            transform=ax.transAxes, fontsize=8, color=MUTED,
        )  # fmt: skip
    ax.set_xlabel('per-user rate x (output tokens/s per request, end to end)')
    ax.set_ylabel('throughput y (output tokens/s on the GPU)')
    ax.set_title('Stack arms on tuned DFlash-16, c = 1-8 (measured; mean over sessions)', fontsize=10)
    ax.grid(True, color='#d9dde1', linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ratios(rows: list[dict[str, str]], full: str, path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    arms = arm_order(full)[1:]
    cs = sorted({int(r['c']) for r in rows})
    titles = {'x_e2e': 'per-user rate x_e2e', 'y': 'throughput y'}
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True, layout='constrained')
    slot = 0.8 / len(arms)
    for ax, m in zip(axes, METRICS, strict=True):
        ax.axhline(1.0, color=INK, linewidth=0.8)
        for i, c in enumerate(cs):
            for j, arm in enumerate(arms):
                x = i + (j - (len(arms) - 1) / 2) * slot
                r = next(r for r in rows if r['arm'] == arm and int(r['c']) == c and r['metric'] == m)
                st = style(arm, full)
                if r['declared_lo']:
                    lo, hi = float(r['declared_lo']), float(r['declared_hi'])
                    ax.add_patch(
                        Rectangle(
                            (x - slot * 0.45, lo), slot * 0.9, max(hi - lo, 0.002),
                            color='#e3d3a8', alpha=0.9, linewidth=0, zorder=1,
                        )
                    )  # fmt: skip
                if r['sessions']:
                    sv = [float(v) for v in r['sessions'].split()]
                    ax.scatter([x] * len(sv), sv, s=9, color=st['color'], alpha=0.45, zorder=2)
                if r['ratio']:
                    ratio = float(r['ratio'])
                    if r['lo']:
                        ax.plot([x, x], [float(r['lo']), float(r['hi'])], color=st['color'], linewidth=1.4, zorder=3)
                    decided = r['decision'] in ('speedup', 'slowdown')
                    ax.plot(
                        x, ratio, marker=st['marker'], markersize=6, color=st['color'],
                        markerfacecolor=st['color'] if decided else 'white', zorder=4,
                    )  # fmt: skip
                else:
                    ax.text(x, 1.0, 'n=0', fontsize=6, color=MUTED, ha='center', va='bottom', rotation=90)
        ax.set_ylabel(f'{titles[m]} / S0')
        ax.grid(True, axis='y', color='#d9dde1', linewidth=0.6)
    axes[-1].set_xticks(range(len(cs)), [f'c = {c}' for c in cs])
    handles = [
        Line2D([], [], linestyle='none', marker=style(a, full)['marker'], color=style(a, full)['color'], label=name(a, full))
        for a in arms
    ]  # fmt: skip
    handles.append(Rectangle((0, 0), 1, 1, color='#e3d3a8', label='declared expected range'))
    axes[0].legend(handles=handles, fontsize=7, ncol=4, frameon=False, loc='upper left')
    axes[0].set_title(
        'Session-paired ratios against S0: geometric mean, 95% t interval, sessions (dots);\n'
        'filled marker = interval excludes 1 with at least three sessions (measured)',
        fontsize=9,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gap(gap: list[dict[str, str]], blocks: list[dict[str, str]], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.2), layout='constrained')
    cs = [int(r['c']) for r in gap]
    xs = range(len(cs))
    series = [
        ('decode_ceiling_snapshot_free', 'engine at the bandwidth floor, measured tau (derived)', 's', '--'),
        ('selector_bound_at_floor', '+ selector bound on tau (derived, upper bound)', '^', '-.'),
        ('full_blocks_at_floor', 'every 16-token block accepted, at the floor (derived)', 'v', ':'),
    ]
    for key, label, marker, line in series:
        ax.plot(xs, [float(r[key]) for r in gap], color=MUTED, marker=marker, linestyle=line, label=label)
    full = gap[0]['full']
    for i, r in enumerate(gap):
        if r['full_x_ratio']:
            if r['full_x_lo']:
                ax.plot([i, i], [float(r['full_x_lo']), float(r['full_x_hi'])], color=BLUE, linewidth=1.6)
            ax.plot(i, float(r['full_x_ratio']), marker='D', color=BLUE, markersize=6,
                    label=f'FULL = {full} (measured, 95% interval)' if i == 0 else None)  # fmt: skip
    ax.axhline(GOAL, color=INK, linestyle='--', linewidth=1.2)
    ax.text(len(cs) - 1, GOAL * 1.04, '5x goal', ha='right', fontsize=8, color=INK)
    ax.axhline(1.0, color=INK, linewidth=0.6)
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 3, 5, 8], ['1', '2', '3', '5', '8'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(list(xs), [f'c = {c}' for c in cs])
    ax.set_ylabel('multiple of tuned DFlash-16 per-user rate')
    ax.set_title('(a) Measured full stack and derived ceilings', fontsize=9)
    ax.grid(True, axis='y', color='#d9dde1', linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    widths = [int(r['block']) for r in blocks]
    bx.plot(widths, widths, color=INK, linewidth=1.0, label='block width (every token accepted)')
    for key, label, line in (
        ('tokens_needed_with_draft', 'needed for 5x, current drafter cost', '-'),
        ('tokens_needed_free_drafter', 'needed for 5x, free drafter', '--'),
    ):
        bx.plot(widths, [float(r[key]) for r in blocks], color=BLUE, linestyle=line, label=label)
        for r in blocks:
            bx.plot(int(r['block']), float(r[key]), marker='o', color=BLUE,
                    markerfacecolor=BLUE if r['measured'] == 'True' else 'white')  # fmt: skip
    tau = float(gap[0]['baseline_tau'])
    bx.plot(16, tau, marker='*', markersize=10, color=INK, linestyle='none', label=f'tuned DFlash-16 tau at c = 1 ({tau:.2f})')
    bx.set_xscale('log', base=2)
    bx.set_yscale('log', base=2)
    bx.set_xticks(widths, [str(w) for w in widths])
    bx.set_yticks([4, 8, 16, 32, 64, 128, 256], ['4', '8', '16', '32', '64', '128', '256'])
    bx.yaxis.set_minor_formatter(NullFormatter())
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.set_xlabel('verify width (tokens per block)')
    bx.set_ylabel('tokens committed per cycle')
    bx.set_title('(b) Tokens per cycle for 5x at c = 1 (derived; open: interpolated)', fontsize=9)
    bx.grid(True, color='#d9dde1', linewidth=0.6)
    bx.legend(fontsize=7, frameon=False, loc='upper left')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--points', type=Path, required=True, help="bench.pareto's points.csv")
    ap.add_argument('--composition', type=Path, required=True, help="analyze.py's JSON")
    ap.add_argument('--expected', type=Path, required=True)
    ap.add_argument('--ceiling', type=Path, required=True)
    ap.add_argument('--frame', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--no-plot', action='store_true', help='write the tables only')
    args = ap.parse_args()
    with args.points.open() as f:
        points = [r for r in csv.DictReader(f) if r['label'].startswith('stack-')]
    comp = json.loads(args.composition.read_text())
    tables = {
        'frontier.csv': frontier_rows(points, comp),
        'ratios.csv': ratio_rows(comp, json.loads(args.expected.read_text())),
        'gap_by_concurrency.csv': gap_rows(comp, json.loads(args.ceiling.read_text())),
        'gap_by_block.csv': block_rows(json.loads(args.frame.read_text())),
        'last_lever.csv': last_lever_rows(points, comp),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for file, rows in tables.items():
        if rows:
            write_csv(rows, args.out_dir / file)
    if args.no_plot:
        return 0
    # Each figure reads only its table as written, so the figure shows the committed data.
    full = comp['full']
    plot_frontier(read_csv(args.out_dir / 'frontier.csv'), full, args.out_dir / 'frontier.png')
    plot_ratios(read_csv(args.out_dir / 'ratios.csv'), full, args.out_dir / 'ratios.png')
    plot_gap(
        read_csv(args.out_dir / 'gap_by_concurrency.csv'),
        read_csv(args.out_dir / 'gap_by_block.csv'),
        args.out_dir / 'gap.png',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
