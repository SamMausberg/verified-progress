"""Figures of the lossy-lever study, from analyze.py's decisions.json.

    python -m experiments.lossy.figures evidence/lossy/decisions.json --out evidence/lossy

gemm_w4a16.png (with --gemm): the exploratory GEMM microbenchmark's time per forward
pass of the quantized projections against rows, BF16 and W4A16, target and drafter.
frontier.png: output tokens/s per user (x) against tokens/s per GPU (y), session
means, for every arm, with three envelopes: exact arms only, exact arms plus the
INT4 arms, exact arms plus the FP16-state arms (the frontier without and with
each lever). quality_speed.png: each GSM8K-measured arm's difference against the
two references (95% interval, one point per reference) against the ratio of its
session-mean y (x at c = 1) to the best exact arm's at its lever's headline
points, with the -1.0-point budget line and the exact arms' spread; with
--partial-gsm8k, also the full-split bounds of the INT4 run that stopped early
(not declared).
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
    # The exact envelope is drawn wide and light underneath, so a lever's envelope shows
    # where it leaves it and where it coincides.
    front = envelope(exact)
    ax.plot(
        [p['x_mean'] for p in front],
        [p['y_mean'] for p in front],
        color=INK,
        linewidth=6,
        alpha=0.22,
        solid_capstyle='round',
        label='envelope, exact arms',
        zorder=1,
    )
    for lever, line in (('int4', '-'), ('fp16-state', '--')):
        front = envelope(exact + [m for m in means if lever_of(m['arm']) == lever])
        ax.plot(
            [p['x_mean'] for p in front],
            [p['y_mean'] for p in front],
            color=LEVER_COLOUR[lever],
            linewidth=1.6,
            linestyle=line,
            label=f'envelope, exact + {LEVER_NAME[lever]}',
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


def quality_speed(
    decisions: dict[str, Any], path: Path, partial: dict[str, Any] | None = None
) -> None:
    """GSM8K difference against speed ratio at each lever's headline points.

    One point per reference run at its true value (filled: first reference, open:
    second), dodged sideways by a fixed amount so both stay visible; the band is the
    spread of bench's exact arms against the same references. `partial`
    (gsm8k_partial.py's output, not declared) adds the full-split bounds of an arm
    whose run stopped early, as bars without a point estimate.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    means = {(m['arm'], m['concurrency']): m for m in decisions['means']}
    best_exact = {e['concurrency']: e['best_exact'] for e in decisions['envelope']}

    def speedup(arm: str, c: int, metric: str) -> float | None:
        m, base = means.get((arm, c)), best_exact.get(c)
        if m is None or base is None:
            return None
        return float(m[f'{metric}_mean'] / means[(base, c)][f'{metric}_mean'])

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    spread = [r['delta_pt'] for r in decisions.get('gsm8k_exact_spread', [])]
    if spread:
        ax.axhspan(min(spread), max(spread), color=GRID, alpha=0.7, zorder=0,
                   label="exact arms' spread (bench, same references)")  # fmt: skip
    ax.axhline(-1.0, color=INK, linewidth=1, linestyle=':', zorder=1)
    ax.annotate('declared budget: -1.0 point', (0.2, -1.0), xycoords=('axes fraction', 'data'),
                textcoords='offset points', xytext=(0, 3), fontsize=7, color=MUTED)  # fmt: skip
    ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=1)
    dodge = 0.0015
    # Labels above the whiskers at some concurrencies and below at others, so that
    # neighbouring points (c = 8 and 32, c = 128 and 256) do not share a label slot.
    above = {1, 8, 128}

    def label_c(x: float, low: float, high: float, c: int) -> None:
        ax.annotate(f'c={c}', (x, high if c in above else low), textcoords='offset points',
                    xytext=(0, 3 if c in above else -9), ha='center', fontsize=6,
                    color=MUTED)  # fmt: skip

    for g in decisions['gsm8k']:
        lever = lever_of(g['arm'])
        marker = 'D' if 'replayssm' in g['arm'] else 's'
        for c, metric in HEADLINE.get(lever, ()):
            ratio = speedup(g['arm'], c, metric)
            if ratio is None:
                continue
            for k, ref in enumerate(g['vs']):
                low, high = ref['delta_ci95_pt']
                ax.errorbar(
                    ratio + (k - 0.5) * dodge,
                    ref['delta_pt'],
                    yerr=[[ref['delta_pt'] - low], [high - ref['delta_pt']]],
                    marker=marker,
                    markersize=5,
                    color=LEVER_COLOUR[lever],
                    markerfacecolor=LEVER_COLOUR[lever] if k == 0 else SURFACE,
                    capsize=2,
                    linewidth=1.2,
                    zorder=3,
                )
            lows, highs = zip(*(r['delta_ci95_pt'] for r in g['vs']), strict=True)
            label_c(ratio, min(lows), max(highs), c)
        ax.plot([], [], marker=marker, linestyle='none', color=LEVER_COLOUR[lever],
                label=f'{g["arm"]} ({LEVER_NAME[lever]}): 95% interval')  # fmt: skip
    if partial is not None:
        arm = 'int4-dflash-b8'
        lever = lever_of(arm)
        refs = [partial['vs'][name] for name in plan.GSM8K_REFERENCES]
        for c, metric in HEADLINE[lever]:
            # The x headline at c = 1 belongs to the b16 arm; GSM8K ran on b8, which
            # shares its target and drafter weights.
            ratio = speedup('int4-dflash-b16' if c == 1 else arm, c, metric)
            if ratio is None:
                continue
            for k, ref in enumerate(refs):
                low, high = ref['full_split_delta_bounds_pt']
                ax.plot([ratio + (k - 0.5) * 2 * dodge] * 2, [low, high], linewidth=3,
                        color=LEVER_COLOUR[lever], alpha=0.9 if k == 0 else 0.45,
                        solid_capstyle='butt', zorder=3)  # fmt: skip
            lows, highs = zip(*(r['full_split_delta_bounds_pt'] for r in refs), strict=True)
            label_c(ratio, min(lows), max(highs), c)
        ax.plot([], [], linewidth=3, color=LEVER_COLOUR[lever],
                label=f'{arm} ({LEVER_NAME[lever]}): bounds, run stopped at '
                f'{partial["finished"]}/{partial["task_problems"]} (not declared)')  # fmt: skip
    ax.margins(y=0.08)
    ax.set_xlabel('ratio of session means over the best exact arm (x at c=1, y elsewhere)')
    ax.set_ylabel('GSM8K accuracy difference (points)')
    ax.set_title('Quality for speed at the headline points (references: plain-tuned a, b)',
                 fontsize=10, color='#0b0b0b')  # fmt: skip
    ax.legend(frameon=False, fontsize=7, loc='lower center',
              title='dark or filled: reference a; light or open: reference b',
              title_fontsize=7)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gemm(result: dict[str, Any], path: Path) -> None:
    """Quantized-projection GEMM time per forward pass against rows, BF16 and W4A16."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, model, title in (
        (axes[0], 'target', 'target (Qwen3.5-4B backbone projections)'),
        (axes[1], 'drafter', 'DFlash drafter (BF16 z-lab vs INT4 nota)'),
    ):
        style(ax)
        rows = sorted(
            (t for t in result['per_forward_totals'] if t['model'] == model), key=lambda t: t['m']
        )
        ms = [t['m'] for t in rows]
        for side, colour, label in (
            ('bf16', MUTED, 'BF16 (cuBLAS)'),
            ('w4a16', LEVER_COLOUR['int4'], 'W4A16 (Marlin)'),
        ):
            ax.plot(
                ms,
                [t[f'{side}_us'] / 1000 for t in rows],
                marker='o',
                markersize=4,
                linewidth=2,
                color=colour,
                label=label,
            )
        ax.set_xscale('log', base=2)
        ax.set_xticks(ms, [str(m) for m in ms])
        ax.set_xlabel('rows per GEMM (M)')
        ax.set_ylabel('ms per forward pass')
        ax.set_title(title, fontsize=9, color='#0b0b0b')
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        'Quantized projections only, CUDA graphs, cold L2 (exploratory microbenchmark)',
        fontsize=10,
        color='#0b0b0b',
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('decisions', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gemm', type=Path, default=None, help='gemm_w4a16_bench output')
    parser.add_argument(
        '--partial-gsm8k', type=Path, default=None, help='gsm8k_partial output (not declared)'
    )
    args = parser.parse_args(argv)
    decisions = json.loads(args.decisions.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    frontier(decisions, args.out / 'frontier.png')
    if decisions.get('gsm8k'):
        partial = json.loads(args.partial_gsm8k.read_text()) if args.partial_gsm8k else None
        quality_speed(decisions, args.out / 'quality_speed.png', partial)
    if args.gemm is not None:
        gemm(json.loads(args.gemm.read_text()), args.out / 'gemm_w4a16.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
