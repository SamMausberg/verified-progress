"""Achieved bandwidth of the head GEMM and the GDN kernels, by batch and by source.

    python experiments/profiling/kernel_bandwidth.py --evidence evidence/profiles \
        --out evidence/profiles/kernel_bandwidth.csv

One row per (kernel, batch, source), from four committed measurements:

* ``microbench``: the head GEMM replayed under CUDA graphs (``head_microbench.json``,
  warm L2), M = 1-256 rows;
* ``gdn_bench``: one GDN layer's decode, verify and verify-without-saves kernels
  under CUDA graphs with L2 evicted before each launch (``gdn_kernel_bench.json``);
  the eviction also writes back the dirty lines the previous launch left in L2, so
  these times include every byte the kernel writes;
* ``serving``: the kernel's mean duration per call in the nsys traces of the
  running server (``attribution/plain_bs*.json`` and ``mtp_bs*.json``); dirty
  lines still in L2 when a kernel ends are written back during later kernels,
  so a write-heavy kernel's duration excludes part of its write traffic;
* ``ncu``: one launch under Nsight Compute (``ncu_key_kernels.json``), with the
  DRAM bytes ncu counted during the launch instead of the modelled bytes.

``batch`` is the number of rows M for the head GEMM and the number of requests
for the GDN kernels (a verify request carries four positions). ``bytes`` is the
modelled traffic (head: weight, activations and BF16 logits; GDN: FP32 state
read and written, ``gdn_kernel_bench.py``) except in ``ncu`` rows, where it is
the measured DRAM read plus write. ``pct_of_read_peak`` divides by the read
bandwidth measured over the head's footprint (``hbm_bandwidth.json``,
``read_head_size``), the reference used throughout ``evidence/profiles``.

``regime`` is given for ncu rows only, by the rule Nsight Compute's
speed-of-light analysis uses: ``bandwidth`` when DRAM throughput is at least
80% of ncu's DRAM peak; ``latency`` when neither the memory nor the SM
throughput reaches 70% of its peak and the largest warp stall is
``long_scoreboard`` (waiting on memory); ``mixed`` otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HIDDEN, VOCAB = 2560, 248320
GDN_LAYERS = 24
KERNELS = {
    'gdn_decode': 'fused_recurrent_gated_delta_rule_packed_decode_kernel',
    'gdn_verify': 'fused_sigmoid_gating_delta_rule_update_kernel',
}
FIELDS = [
    'kernel', 'batch', 'source', 'us_per_call', 'bytes', 'bytes_counted', 'tb_per_s',
    'pct_of_read_peak', 'pct_of_ncu_dram_peak', 'tflop_per_s', 'sm_throughput_pct',
    'achieved_occupancy_pct', 'theoretical_occupancy_pct', 'top_stall', 'regime',
]  # fmt: skip


def head_bytes(m: int) -> float:
    return 2.0 * (VOCAB * HIDDEN + m * HIDDEN + m * VOCAB)


def row(kernel: str, batch: int, source: str, us: float, nbytes: float, peak: float) -> dict:
    if us <= 0 or nbytes <= 0:
        raise ValueError(f'{kernel} B={batch} {source}: non-positive time or bytes')
    tbs = nbytes / (us * 1e-6) / 1e12
    out = {
        'kernel': kernel,
        'batch': batch,
        'source': source,
        'us_per_call': round(us, 2),
        'bytes': int(nbytes),
        'bytes_counted': 'model',
        'tb_per_s': round(tbs, 3),
        'pct_of_read_peak': round(100 * tbs / peak, 1),
    }
    if kernel == 'head_gemm':
        out['tflop_per_s'] = round(2.0 * batch * HIDDEN * VOCAB / (us * 1e-6) / 1e12, 1)
    return out


def regime(n: dict) -> str:
    top = next(iter(n['top_stalls_cycles_per_issue']))
    if n['dram_pct_of_peak'] >= 80:
        return 'bandwidth'
    if max(n['memory_throughput_pct'], n['sm_throughput_pct']) < 70 and top == 'long_scoreboard':
        return 'latency'
    return 'mixed'


def load(path: Path) -> dict | list:
    if not path.exists():
        raise SystemExit(f'missing input {path}')
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ev = args.evidence
    hbm = load(ev / 'hbm_bandwidth.json')
    assert isinstance(hbm, dict)
    peak = hbm['kernels']['read_head_size']['tb_per_s_median']
    rows: list[dict] = []

    micro = load(ev / 'head_microbench.json')
    assert isinstance(micro, dict)
    gemm = [r for r in micro['rows'] if r['variant'] == 'gemm' and r['l2'] == 'warm']
    if not gemm:
        raise SystemExit('no warm-L2 GEMM rows in head_microbench.json')
    for r in sorted(gemm, key=lambda r: r['m']):
        rows.append(row('head_gemm', r['m'], 'microbench', r['median_us'], r['gemm_bytes'], peak))

    bench = load(ev / 'gdn_kernel_bench.json')
    assert isinstance(bench, dict)
    modes = {r['mode'] for r in bench['rows']}
    if modes != {'decode', 'verify', 'verify_nosave'}:
        raise SystemExit(f'gdn_kernel_bench.json has modes {sorted(modes)}')
    for r in bench['rows']:
        rows.append(
            row(f'gdn_{r["mode"]}', r['batch'], 'gdn_bench', r['median_us'], r['state_bytes'], peak)
        )
    state_bytes = {
        'gdn_decode': lambda b: 2 * b * 2**21,
        'gdn_verify': lambda b: 5 * b * 2**21,
    }

    for arm, kinds in (('plain', ('head_gemm', 'gdn_decode')), ('mtp', ('gdn_verify',))):
        reports = sorted(
            (ev / 'attribution').glob(f'{arm}_bs*.json'),
            key=lambda p: int(p.stem.split('_bs')[1]),
        )
        if not reports:
            raise SystemExit(f'no attribution/{arm}_bs*.json')
        for path in reports:
            batch = int(path.stem.split('_bs')[1])
            att = load(path)
            assert isinstance(att, dict)
            for kind in kinds:
                if kind == 'head_gemm':
                    us = att['summary']['lm_head_gemm']['median_us']
                    rows.append(row(kind, batch, 'serving', us, head_bytes(batch), peak))
                    continue
                k = [x for x in att['kernels'] if x['kernel'] == KERNELS[kind]]
                if len(k) != 1 or k[0]['calls_per_step'] != GDN_LAYERS:
                    raise SystemExit(f'{path.name}: expected one {KERNELS[kind]} x {GDN_LAYERS}')
                us = k[0]['raw_us_per_step'] / k[0]['calls_per_step']
                rows.append(row(kind, batch, 'serving', us, state_bytes[kind](batch), peak))

    ncu = load(ev / 'ncu_key_kernels.json')
    assert isinstance(ncu, list)
    names = {
        'head_gemm_m1': ('head_gemm', 1),
        'head_gemm_m32': ('head_gemm', 32),
        'gdn_decode_b32': ('gdn_decode', 32),
        'gdn_verify_b8': ('gdn_verify', 8),
    }
    seen = set()
    for n in ncu:
        stem = n['report'].removesuffix('.ncu-rep')
        if stem not in names:
            raise SystemExit(f'unexpected ncu report {n["report"]}')
        seen.add(stem)
        kernel, batch = names[stem]
        moved = n['dram_bytes_read'] + n['dram_bytes_write']
        r = row(kernel, batch, 'ncu', n['duration_us'], moved, peak)
        r.update(
            bytes_counted='dram_measured',
            pct_of_ncu_dram_peak=round(n['dram_pct_of_peak'], 1),
            sm_throughput_pct=round(n['sm_throughput_pct'], 1),
            achieved_occupancy_pct=round(n['achieved_occupancy_pct'], 1),
            theoretical_occupancy_pct=round(n['theoretical_occupancy_pct'], 1),
            top_stall=next(iter(n['top_stalls_cycles_per_issue'])),
            regime=regime(n),
        )
        rows.append(r)
    if seen != set(names):
        raise SystemExit(f'ncu reports missing: {sorted(set(names) - seen)}')

    order = {'head_gemm': 0, 'gdn_decode': 1, 'gdn_verify': 2, 'gdn_verify_nosave': 3}
    src = {'microbench': 0, 'gdn_bench': 0, 'serving': 1, 'ncu': 2}
    rows.sort(key=lambda r: (order[r['kernel']], src[r['source']], r['batch']))
    tmp = args.out.with_suffix('.tmp')
    with tmp.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.out)
    print(f'{len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
