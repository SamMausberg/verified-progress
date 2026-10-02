"""Figures of the lossy-lever study, from analyze.py's decisions.json.

    python -m experiments.lossy.figures evidence/lossy/decisions.json --out evidence/lossy

frontier.png: output tokens/s per user (x) against tokens/s per GPU (y), session
means, for every arm, with three envelopes: exact arms only, exact arms plus the
INT4 arms, exact arms plus the FP16-state arms (the frontier without and with
each lever). quality_speed.png: each lever's GSM8K difference against the
reference (with its 95% interval, one point per reference) against the ratio of
its session-mean y (x at c = 1) to the best exact arm's at the declared headline
points, with the -1.0-point budget line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.lossy import plan

INK = '#2f2e2b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
# bench/pareto.py's categorical order: lever identity, not rank.
LEVER_COLOUR = {'exact': '#52514e', 'int4': '#2a78d6', 'fp16-state': '#eb6834'}
LEVER_NAME = {
    'exact': 'exact arms',
    'int4': 'INT4 target + INT4 DFlash',
    'fp16-state': 'FP16 GDN state',
}
HEADLINE = {'int4': ((1, 'x'), (8, 'y'), (32, 'y')), 'fp16-state': ((128, 'y'), (256, 'y'))}


def lever_of(arm: str) -> str:
    for lever, arms in plan.LEVERS.items():
        if arm in arms:
            return lever
    return 'exact'


def envelope(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pareto-optimal points (higher x and higher y are both better), by x."""
    front = [
        p
        for p in points
        if not any(
            q['x_mean'] >= p['x_mean'] and q['y_mean'] >= p['y_mean'] and q is not p
            and (q['x_mean'] > p['x_mean'] or q['y_mean'] > p['y_mean'])
            for q in points
        )
    ]  # fmt: skip
    return sorted(front, key=lambda p: p['x_mean'])


def style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def frontier(decisions: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    means = [m for m in decisions['means'] if m['sessions'] >= plan.MIN_SESSIONS]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    exact = [m for m in means if m['exact']]
    for lever, line in (('exact', ':'), ('int4', '-'), ('fp16-state', '--')):
        pool = exact + [m for m in means if lever_of(m['arm']) == lever and lever != 'exact']
        front = envelope(pool)
        label = (
            'envelope, exact arms' if lever == 'exact' else f'envelope, exact + {LEVER_NAME[lever]}'
        )
        ax.plot(
            [p['x_mean'] for p in front],
            [p['y_mean'] for p in front],
            color=LEVER_COLOUR[lever],
            linewidth=2,
            linestyle=line,
            label=label,
            zorder=2,
        )
    for m in means:
        lever = lever_of(m['arm'])
        ax.plot(
            m['x_mean'],
            m['y_mean'],
            marker='o' if lever == 'exact' else 's',
            markersize=5,
            color=LEVER_COLOUR[lever],
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            linestyle='none',
            zorder=3,
        )
    for p in envelope(means):
        ax.annotate(
            f'c={p["concurrency"]}',
            (p['x_mean'], p['y_mean']),
            textcoords='offset points',
            xytext=(5, 4),
            fontsize=7,
            color=MUTED,
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('output tokens/s per user (mean over requests, includes TTFT)')
    ax.set_ylabel('output tokens/s per GPU')
    ax.set_title(
        'Qwen3.5-4B on one GH200: frontier with and without each lossy lever',
        fontsize=10,
        color='#0b0b0b',
    )
    ax.plot([], [], marker='o', linestyle='none', color=MUTED, label='exact arm (session mean)')
    ax.plot([], [], marker='s', linestyle='none', color=MUTED, label='lossy arm (session mean)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def quality_speed(decisions: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    means = {(m['arm'], m['concurrency']): m for m in decisions['means']}
    best_exact = {e['concurrency']: e['best_exact'] for e in decisions['envelope']}
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axhline(-1.0, color=INK, linewidth=1, linestyle=':', zorder=1)
    ax.annotate('budget: -1.0 point', (1.0, -1.0), textcoords='offset points', xytext=(4, 4),
                fontsize=7, color=MUTED)  # fmt: skip
    ax.axhline(0.0, color=GRID, linewidth=1, zorder=1)
    seen: set[str] = set()
    for g in decisions['gsm8k']:
        lever = lever_of(g['arm'])
        for offset, ref in zip((-0.12, 0.12), g['vs'], strict=False):
            for c, metric in HEADLINE.get(lever, ()):
                arm, base = means.get((g['arm'], c)), best_exact.get(c)
                if arm is None or base is None:
                    continue
                speedup = arm[f'{metric}_mean'] / means[(base, c)][f'{metric}_mean']
                low, high = ref['delta_ci95_pt']
                ax.errorbar(
                    speedup,
                    ref['delta_pt'] + offset,
                    yerr=[[ref['delta_pt'] - low], [high - ref['delta_pt']]],
                    marker='s',
                    markersize=5,
                    color=LEVER_COLOUR[lever],
                    capsize=2,
                    linewidth=1.5,
                    zorder=3,
                )
                ax.annotate(
                    f'{g["arm"]} c={c} ({metric})',
                    (speedup, ref['delta_pt'] + offset),
                    textcoords='offset points',
                    xytext=(5, -10),
                    fontsize=6,
                    color=MUTED,
                )
        if lever not in seen:
            seen.add(lever)
            ax.plot([], [], marker='s', linestyle='none', color=LEVER_COLOUR[lever],
                    label=LEVER_NAME[lever])  # fmt: skip
    ax.set_xlabel('ratio of session means over the best exact arm (x at c=1, y elsewhere)')
    ax.set_ylabel('GSM8K accuracy difference (points, 95% interval)')
    ax.set_title('Quality for speed: each lever against the declared budget', fontsize=10,
                 color='#0b0b0b')  # fmt: skip
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('decisions', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    decisions = json.loads(args.decisions.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    frontier(decisions, args.out / 'frontier.png')
    if decisions.get('gsm8k'):
        quality_speed(decisions, args.out / 'quality_speed.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
