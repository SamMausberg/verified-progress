"""Load kernels, copies, runtime calls and NVTX ranges from an Nsight Systems report.

``load(report)`` exports ``<report>.sqlite`` with ``nsys export`` if needed and
returns pandas frames with times in nanoseconds on the report's clock.
"""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Trace:
    kernels: pd.DataFrame
    memcpy: pd.DataFrame
    memset: pd.DataFrame
    runtime: pd.DataFrame
    nvtx: pd.DataFrame
    sync: pd.DataFrame


def export_sqlite(report: Path) -> Path:
    db = report.with_suffix('.sqlite')
    if not db.exists() or db.stat().st_mtime < report.stat().st_mtime:
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
             "-o", str(db), str(report)],
            check=True, capture_output=True,
        )  # fmt: skip
    return db


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


def load(report: Path) -> Trace:
    db = export_sqlite(report) if report.suffix == '.nsys-rep' else report
    con = sqlite3.connect(db)
    tables = _tables(con)
    kcols = {r[1] for r in con.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    # Reports that start collecting after graph capture have no graphId column;
    # the upper 32 bits of graphNodeId identify the graph there.
    graph_col = 'k.graphId' if 'graphId' in kcols else '(k.graphNodeId >> 32)'
    kernels = pd.read_sql_query(
        f"""
        select k.start, k.end, k.streamId as stream, k.correlationId as corr,
               {graph_col} as graph_id, k.graphNodeId as node_id,
               s.value as name, d.value as demangled,
               k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
               k.registersPerThread as regs,
               k.staticSharedMemory + k.dynamicSharedMemory as smem
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id = k.shortName
        join StringIds d on d.id = k.demangledName
        order by k.start
        """,
        con,
    )

    def maybe(table: str, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, con) if table in tables else pd.DataFrame()

    memcpy = maybe(
        'CUPTI_ACTIVITY_KIND_MEMCPY',
        """select m.start, m.end, m.streamId as stream, m.correlationId as corr,
                  m.bytes, m.copyKind as kind, m.srcKind, m.dstKind,
                  m.graphNodeId as node_id
           from CUPTI_ACTIVITY_KIND_MEMCPY m order by m.start""",
    )
    memset = maybe(
        'CUPTI_ACTIVITY_KIND_MEMSET',
        """select m.start, m.end, m.streamId as stream, m.correlationId as corr,
                  m.bytes, m.graphNodeId as node_id
           from CUPTI_ACTIVITY_KIND_MEMSET m order by m.start""",
    )
    runtime = maybe(
        'CUPTI_ACTIVITY_KIND_RUNTIME',
        """select r.start, r.end, r.correlationId as corr, r.globalTid as tid,
                  s.value as name
           from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
           order by r.start""",
    )
    sync = maybe(
        'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION',
        """select y.start, y.end, y.streamId as stream, y.correlationId as corr,
                  y.syncType as sync_type
           from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION y order by y.start""",
    )
    nvtx = maybe(
        'NVTX_EVENTS',
        """select n.start, n.end, n.globalTid as tid,
                  coalesce(n.text, s.value) as text
           from NVTX_EVENTS n left join StringIds s on s.id = n.textId
           where n.end is not null order by n.start""",
    )
    con.close()
    return Trace(kernels, memcpy, memset, runtime, nvtx, sync)


def busy_union(intervals: list[tuple[int, int]]) -> int:
    """Total length of the union of half-open [start, end) intervals."""
    total = 0
    cur_s = cur_e = -1
    for s, e in sorted(intervals):
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    return total + cur_e - cur_s
