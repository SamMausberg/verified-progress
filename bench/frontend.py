"""Find what limits streamed throughput at high concurrency: client or server.

Launches one server arm and runs the same load point with several aiperf client
settings against it. Each point records the client-observed throughput (y), the
server's own decode rate while the full batch runs (from its log), and the peak
CPU use of every server and client process, so a saturated single-threaded
process (tokenizer manager, detokenizer, an aiperf service) shows up directly.
Server-side settings (stream interval, incremental output, tokenizer workers, the
Rust server) are compared by running this once per server variant.

    scripts/gpu_lock.sh -x python -m bench.frontend --arm plain --label fe-plain \\
        --concurrency 256 --max-concurrency 256 --set max-running-requests=256 \\
        --set max-mamba-cache-size=1280 --variants default records-only workers-64
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from bench.server import Server
from bench.sweep import Sweep, build_parser, prepare

# Client settings per variant (attributes of the sweep's argument namespace).
VARIANTS: dict[str, dict[str, Any]] = {
    'default': {},
    'no-per-chunk-usage': {'per_chunk_usage': False},
    'records-only': {'per_chunk_usage': False, 'export_level': 'records'},
    'workers-64': {'aiperf_workers': 64},
    'non-streaming': {'streaming': False, 'per_chunk_usage': False, 'export_level': 'records'},
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    args = prepare(parser, argv)
    run_dir = args.out.expanduser() / args.label / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        args.arm_resolved,
        run_dir / 'server',
        args.port,
        host=args.host,
        sglang_worktree=args.sglang_worktree,
        strict=not args.no_strict,
    )
    baseline = {
        key: getattr(args, key)
        for key in ('per_chunk_usage', 'export_level', 'streaming', 'aiperf_workers')
    }
    results = []
    with server:
        sweep = Sweep(args, server, run_dir)
        sweep.server_warmup()
        for index, name in enumerate(args.variants):
            for key, value in {**baseline, **VARIANTS[name]}.items():
                setattr(args, key, value)
            for concurrency in args.concurrency:
                summary = sweep.point(index, concurrency)
                summary['client_variant'] = name
                sweep.points.append(summary)
                log = summary.get('server_log') or {}
                row = {
                    'variant': name,
                    'concurrency': concurrency,
                    'y_client': summary.get('y'),
                    'server_decode_tps_full_batch': log.get('logged_gen_tps_full_batch_p50'),
                    'ttft_p50_ms': (summary.get('ttft_ms') or {}).get('p50'),
                    'foreign_cpu_max': summary.get('foreign_cpu_during_max'),
                    'own_cpu_peak': summary.get('own_cpu_during_peak'),
                }
                results.append(row)
                print(json.dumps(row, default=str), flush=True)
                sweep.write_manifest({'client_variants': args.variants})
    (run_dir / 'frontend.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    print(f'done: {run_dir}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
