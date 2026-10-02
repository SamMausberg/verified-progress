"""Figures of the served certified-head benchmark, drawn from the committed CSVs.

    python -m experiments.benchcert.figures evidence/certified_head/served

`frontier.png`: one panel per family, x = output tokens/s per user, y = output
tokens/s on the GPU, stock arm against the same arm with the certified head (mean
over valid sessions, bars one standard deviation), each point labelled with its
client concurrency. `ratios.png`: one panel per family, the paired y ratio
(certified / stock) by concurrency with its 95% interval, the derived prediction
as open markers, and the batch sizes above which every certified path is gated
off shaded.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

from experiments.benchcert import plan

STOCK = '#2a78d6'  # categorical slot 1 (validated pair, dataviz reference palette)
CERT = '#eb6834'  # categorical slot 2
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
TITLES = {
    'plain': 'Plain decoding (plain-tuned)',
    'mtp': 'MTP, 3-step chain (mtp-tuned-triton)',
    'dflash16': 'DFlash block 16 (dflash-tuned-b16)',
    'dflash8': 'DFlash block 8 (dflash-tuned)',
}


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _num(value: str) -> float:
    return float(value) if value not in ('', None) else math.nan


def _axes(plt: Any, n: int) -> tuple[Any, list[Any]]:
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows), facecolor=SURFACE)
    flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]
    for ax in flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def frontier(csv_path: Path, png: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read(csv_path)
    families = [f for f in plan.FAMILIES if any(r['family'] == f for r in rows)]
    fig, axes = _axes(plt, len(families))
    for ax, fam in zip(axes, families, strict=False):
        for variant, colour, name, marker in (
            ('stock', STOCK, 'stock head', 'o'),
            ('cert', CERT, 'certified head', 's'),
        ):
            pts = sorted(
                (r for r in rows if r['family'] == fam and r['variant'] == variant),
                key=lambda r: int(r['concurrency']),
            )
            if not pts:
                continue
            x = [_num(r['x_e2e_mean']) for r in pts]
            y = [_num(r['y_mean']) for r in pts]
            ax.errorbar(
                x,
                y,
                xerr=[_num(r['x_e2e_sd']) for r in pts],
                yerr=[_num(r['y_sd']) for r in pts],
                color=colour,
                linewidth=2,
                marker=marker,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
                capsize=0,
                label=name,
            )
            if variant == 'stock':
                for r, xi, yi in zip(pts, x, y, strict=True):
                    ax.annotate(
                        f'c={r["concurrency"]}',
                        (xi, yi),
                        textcoords='offset points',
                        xytext=(6, -10),
                        fontsize=8,
                        color=INK_2,
                    )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(TITLES.get(fam, fam), fontsize=10, color=INK, loc='left')
        ax.set_xlabel('output tokens/s per user', fontsize=9, color=INK_2)
        ax.set_ylabel('output tokens/s on the GPU', fontsize=9, color=INK_2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        'Qwen3.5-4B on one GH200: tuned arms with and without the certified head',
        fontsize=11,
        color=INK,
        x=0.01,
        ha='left',
    )
    fig.tight_layout()
    fig.savefig(png, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def gated_off_from(family: plan.Family) -> float:
    """Smallest concurrency at which every certified path's rows exceed MAX_ROWS."""
    limit = int(plan.CERT_ENV['SGLANG_CERTIFIED_HEAD_MAX_ROWS'])
    per_request = min(family.rows_per_request.values())
    return limit / per_request


def ratios(csv_path: Path, png: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read(csv_path)
    families = [f for f in plan.FAMILIES if any(r['family'] == f for r in rows)]
    fig, axes = _axes(plt, len(families))
    for ax, fam in zip(axes, families, strict=False):
        pts = sorted((r for r in rows if r['family'] == fam), key=lambda r: int(r['concurrency']))
        c = [int(r['concurrency']) for r in pts]
        mean = [_num(r['y_ratio']) for r in pts]
        low = [m - _num(r['y_low']) for r, m in zip(pts, mean, strict=True)]
        high = [_num(r['y_high']) - m for r, m in zip(pts, mean, strict=True)]
        ax.axhline(1.0, color=INK_2, linewidth=1)
        edge = gated_off_from(plan.FAMILIES[fam])
        top = max(c) * 1.5
        on, off = [v for v in c if v <= edge], [v for v in c if v > edge]
        if off:
            start = math.sqrt(max(on) * min(off)) if on else min(c) / 1.5
            ax.axvspan(
                start,
                top,
                color=GRID,
                alpha=0.6,
                linewidth=0,
                label='every certified path gated off',
            )
        ax.errorbar(
            c,
            mean,
            yerr=[low, high],
            color=CERT,
            linewidth=2,
            marker='s',
            markersize=6,
            markeredgecolor=SURFACE,
            capsize=3,
            label='measured y ratio, 95% interval',
        )
        for r, m in zip(pts, mean, strict=True):
            if r.get('role') == 'primary':
                ax.annotate(
                    f'primary: {r["decision"]}',
                    (int(r['concurrency']), m),
                    textcoords='offset points',
                    xytext=(8, 6),
                    fontsize=8,
                    color=INK,
                )
        predicted = [_num(r.get('predicted_y_ratio', '')) for r in pts]
        if any(math.isfinite(p) for p in predicted):
            ax.plot(
                c,
                predicted,
                linestyle='none',
                marker='o',
                markersize=7,
                markerfacecolor='none',
                markeredgecolor=INK_2,
                markeredgewidth=1.2,
                label='derived prediction',
            )
        ax.set_xscale('log', base=2)
        ax.set_xlim(min(c) / 1.5, top)
        ax.set_xticks(c)
        ax.set_xticklabels([str(v) for v in c])
        ax.set_title(TITLES.get(fam, fam), fontsize=10, color=INK, loc='left')
        ax.set_xlabel('client concurrency', fontsize=9, color=INK_2)
        ax.set_ylabel('y, certified / stock (same session)', fontsize=9, color=INK_2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        'Paired throughput ratio of the certified head', fontsize=11, color=INK, x=0.01, ha='left'
    )
    fig.tight_layout()
    fig.savefig(png, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    frontier(directory / 'frontier.csv', directory / 'frontier.png')
    ratios(directory / 'ratios.csv', directory / 'ratios.png')
