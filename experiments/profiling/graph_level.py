"""Step composition from a graph-level trace (``--cuda-graph-trace=graph``).

Calibration for the node-level attribution. With graph-level tracing, CUPTI
records each CUDA-graph replay as one activity
(``CUPTI_ACTIVITY_KIND_GRAPH_TRACE``) instead of one record per kernel node,
which costs less. Steps are cut at replays of the longest-running graph (the
target model; for speculation, cycles run verify to verify). Per step this
reports the step time, the time inside graph replays by graph, eager kernels
and copies, and GPU idle outside replays, for comparison with the node-level
numbers from ``attribute.py``.

    python experiments/profiling/graph_level.py \
        ~/vp-data/profile/mtp_nsys_graphtrace/mtp_bs1.nsys-rep --out evidence/profiles/graph_level_mtp_bs1.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from nsys_db import busy_union, export_sqlite, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for report in args.reports:
        trace = load(report)
        con = sqlite3.connect(export_sqlite(report))
        graphs = pd.read_sql_query(
            'select start, end, graphId as graph_id, correlationId as corr '
            'from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start',
            con,
        )
        con.close()
        dur = (graphs['end'] - graphs['start']).groupby(graphs['graph_id']).median()
        anchor_gid = int(dur.idxmax())
        anchors = graphs[graphs['graph_id'] == anchor_gid]['start'].to_list()
        steps = list(pairwise(anchors))
        eager = trace.kernels[trace.kernels['node_id'].isna()][['start', 'end']]
        extra = [df[['start', 'end']] for df in (trace.memcpy, trace.memset) if not df.empty]
        eager_all = pd.concat([eager, *extra])
        es, ee = eager_all['start'].to_numpy(), eager_all['end'].to_numpy()
        gs, ge = graphs['start'].to_numpy(), graphs['end'].to_numpy()
        gid = graphs['graph_id'].to_numpy()
        rows = []
        for s0, s1 in steps:
            gsel = (gs >= s0) & (gs < s1)
            esel = (es >= s0) & (es < s1)
            spans = list(zip(gs[gsel], np.minimum(ge[gsel], s1), strict=True))
            eag = list(zip(es[esel], np.minimum(ee[esel], s1), strict=True))
            busy = busy_union(spans + eag) / 1e3
            row = {
                'step_us': (s1 - s0) / 1e3,
                'graph_us': busy_union(spans) / 1e3,
                'busy_us': busy,
                'idle_outside_graphs_us': (s1 - s0) / 1e3 - busy,
            }
            for g in np.unique(gid[gsel]):
                sel = gsel & (gid == g)
                row[f'graph_{int(g)}_us'] = float(((np.minimum(ge[sel], s1) - gs[sel]) / 1e3).sum())
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0.0)
        mean = df.mean().to_dict()
        out[report.name] = {
            'steps': len(df),
            'anchor_graph': anchor_gid,
            'mean_us': mean,
            'idle_outside_graphs_pct': 100 * mean['idle_outside_graphs_us'] / mean['step_us'],
            'step_us_p10': float(df['step_us'].quantile(0.1)),
            'step_us_p90': float(df['step_us'].quantile(0.9)),
        }
        print(
            f'{report.name}: step {mean["step_us"]:.1f} us, idle outside graphs '
            f'{out[report.name]["idle_outside_graphs_pct"]:.1f}%'
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
