"""The gross positions on unpatched SGLang, under kernel-backend variants (shared lane).

    python -m experiments.bf16_paths.sglang_variants start --variant NAME --out DIR   # in gpu_startup_lock.sh
    python -m experiments.bf16_paths.sglang_variants read --variant NAME --out DIR --targets TARGETS
    python -m experiments.bf16_paths.sglang_variants stop --out DIR

`experiments/benchcert/paths.py stock` read 579ae7ce/439 and a4db11ff/333 on the
benchmark's engine (the SGLang pin plus the certified-head patches, which a stock arm
leaves off). This module reads them on the pin itself, the `~/sglang` checkout at
`PIN` with no patches, so that the observation stands on upstream code, and under
variants that each swap one kernel family the two paths use:

- `default`: the `plain-tuned` arm as `rescore.py` runs it (FlashInfer attention, radix
  cache off, small pools). On SM90 SGLang then picks FlashInfer's GDN kernel for prefill
  and its Triton GDN kernel for decode (the server log's "GDN kernel dispatcher" line).
- `prefill_triton`: `--linear-attn-prefill-backend triton`, the Triton chunked GDN kernel
  for prefill.
- `decode_flashinfer`: `--linear-attn-decode-backend flashinfer`, FlashInfer's GDN kernel
  for decode.
- `no_cuda_graph`: `--disable-cuda-graph`, eager decode and prefill.
- `attn_triton`: `--attention-backend triton`, Triton attention for the full-attention
  layers.
- `beta_fp32`: the default flags on the pin plus
  `engine/sglang/patches/upstream-bf16/0001-*.patch` (engine worktree
  `~/sglang-wt/upstream-bf16` at `BETA_FP32_HEAD`): the one-line changes of the open
  upstream PRs #38977 and #40362, which keep sigmoid(beta) in FP32 in the packed GDN decode
  kernel and in the gating kernel that feeds prefill, where the pin rounds it through BF16
  (upstream issue #38975).

`read` first checks that the server resolved the variant's kernels (`check_active`: the GDN
dispatcher line, the attention backend, CUDA graphs; added after the runs, which all
passed it), then runs `paths.stock_target` on every target (one request at a time, the
cache flushed before each: three prefills and one decode from TRACE positions before the
target) and writes `<variant>.json` with the server's resolved kernel choices; it refuses
an existing `<variant>.json`.

Readings (set before the run): `default` within 0.1 nats of the benchmark engine's stock
reading (`paths.py stock`) means the observation stands on the unpatched pin. A variant
that brings a wrong path within 1 nat of FP32 on the target's tracked tokens, where
`default` misses by several nats, names the kernel family that carries that error
(`prefill_triton`: FlashInfer's GDN prefill at a4db11ff/333; `decode_flashinfer`: the
Triton GDN decode at 579ae7ce/439; `no_cuda_graph`: graph capture or replay;
`attn_triton`: FlashInfer attention; `beta_fp32`: the BF16 rounding of beta). A variant
that leaves both errors in place clears its family at these positions.

Qualified after the run (review of #222): the swaps are not isolating interventions. Each
replacement is BF16 arithmetic too and several families may contribute, so a swap that
leaves an error in place does not clear its family, and one that removes an error on one path
while creating another does not name it. What the variants can show is whether any single
change restores every path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from bench.arms import ArgValue, resolve_arm
from bench.server import Server
from experiments.benchcert.paths import stock_target
from experiments.benchcert.rescore import OVERRIDES

PORT = 30240
PIN = 'bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824'
BETA_FP32_HEAD = '4608661757c5c3891f0a40d8980fd99fd0547abd'
VARIANTS: dict[str, dict[str, ArgValue]] = {
    'default': {},
    'prefill_triton': {'linear-attn-prefill-backend': 'triton'},
    'decode_flashinfer': {'linear-attn-decode-backend': 'flashinfer'},
    'no_cuda_graph': {'disable-cuda-graph': True},
    'attn_triton': {'attention-backend': 'triton'},
    'beta_fp32': {},
}
# Variants that run a patched engine worktree instead of ~/sglang: (worktree, its head).
ENGINES: dict[str, tuple[Path, str]] = {
    'beta_fp32': (Path.home() / 'sglang-wt' / 'upstream-bf16', BETA_FP32_HEAD),
}
# What each variant must resolve to: the GDN dispatcher's decode and extend kernels (server
# log), the attention backend and whether CUDA graphs are off (server_info.json).
GDN_TRITON_DECODE_FLASHINFER_PREFILL = 'decode=TritonGDNKernel, extend=FlashInferGDNKernel'
ACTIVE: dict[str, tuple[str, str, bool]] = {
    'default': (GDN_TRITON_DECODE_FLASHINFER_PREFILL, 'flashinfer', False),
    'prefill_triton': ('decode=TritonGDNKernel, extend=TritonGDNKernel', 'flashinfer', False),
    'decode_flashinfer': (
        'decode=FlashInferGDNKernel, extend=FlashInferGDNKernel',
        'flashinfer',
        False,
    ),
    'no_cuda_graph': (GDN_TRITON_DECODE_FLASHINFER_PREFILL, 'flashinfer', True),
    'attn_triton': (GDN_TRITON_DECODE_FLASHINFER_PREFILL, 'triton', False),
    'beta_fp32': (GDN_TRITON_DECODE_FLASHINFER_PREFILL, 'flashinfer', False),
}
_DISPATCHER = re.compile(r'GDN kernel dispatcher: (.*)$', re.MULTILINE)
_GDN_LINES = re.compile(r'^.*(?:GDN|Linear attention kernel backend).*$', re.MULTILINE)


def server_dir(out: Path, variant: str) -> Path:
    return out / 'servers' / variant


def start(out: Path, variant: str) -> int:
    """Start the variant, check its SGLang source, record its pid."""
    arm = resolve_arm('plain-tuned', {**OVERRIDES, **VARIANTS[variant]})
    arm = type(arm)(**{**arm.to_json(), 'max_concurrency': 32})
    directory = server_dir(out, variant)
    # Without a worktree the server imports the venv's editable install, ~/sglang at PIN.
    os.environ.pop('SGLANG_WORKTREE', None)
    worktree, head = ENGINES.get(variant, (None, PIN))
    server = Server(arm, directory, PORT, sglang_worktree=worktree, strict=False)
    server.start()
    try:
        source = server.launch_record['sglang_source']
        if source.get('head') != head or source.get('dirty_files'):
            raise SystemExit(f'SGLang source is not the clean {head}: {source}')
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (out / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def check_active(variant: str, directory: Path) -> list[str]:
    """Refuse a server whose resolved kernels are not the variant's (a flag the engine
    ignores or overrides would otherwise read as the swap); returns the dispatcher lines."""
    log = (directory / 'server.log').read_text(errors='replace')
    info = json.loads((directory / 'server_info.json').read_text())
    gdn, attention, graphs_off = ACTIVE[variant]
    lines = sorted(set(_DISPATCHER.findall(log)))
    found = (
        len(lines) == 1 and lines[0].startswith(gdn),
        info.get('attention_backend') == attention,
        bool(info.get('disable_cuda_graph')) == graphs_off,
    )
    if not all(found):
        raise SystemExit(
            f'{variant}: expected GDN {gdn!r}, attention {attention}, CUDA graphs off '
            f'{graphs_off}; the server resolved {lines}, {info.get("attention_backend")}, '
            f'{info.get("disable_cuda_graph")}'
        )
    return lines


def read(out: Path, variant: str, targets: Path) -> int:
    if (out / f'{variant}.json').exists():
        raise SystemExit(f'{out / f"{variant}.json"} exists')
    found = [json.loads(line) for line in targets.read_text().splitlines() if line]
    if not found:
        raise SystemExit(f'no targets in {targets}')
    directory = server_dir(out, variant)
    check_active(variant, directory)
    log = (directory / 'server.log').read_text(errors='replace')
    launch = json.loads((directory / 'launch.json').read_text())
    results = [stock_target(f'http://127.0.0.1:{PORT}', t) for t in found]
    record: dict[str, Any] = {
        'variant': variant,
        'flags': VARIANTS[variant],
        'command': launch['command'],
        'sglang_source': launch['sglang_source'],
        'server_version': launch.get('server_version'),
        'checks': launch.get('checks'),
        'gdn_dispatcher': _DISPATCHER.findall(log),
        'gdn_log_lines': sorted(set(line.split('] ', 1)[-1] for line in _GDN_LINES.findall(log))),
        'results': results,
    }
    tmp = out / f'{variant}.json.tmp'
    tmp.write_text(json.dumps(record, indent=1) + '\n')
    tmp.replace(out / f'{variant}.json')
    for r in results:
        pos = str(r['position'])
        summary = {
            name: {k: round(v, 3) for k, v in (trace.get(pos) or {}).get('tracked', {}).items()}
            for name, trace in r['paths'].items()
        }
        print(
            variant, r['id'], 'decode left text at', r['decode_left_text_at'], json.dumps(summary)
        )
    return 0


def stop(out: Path) -> int:
    pid_file = out / 'server.pid'
    if not pid_file.exists():
        return 0
    pid = int(pid_file.read_text())
    with contextlib.suppress(ProcessLookupError):
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM)
        for _ in range(30):
            time.sleep(1)
            os.killpg(group, 0)
        os.killpg(group, signal.SIGKILL)
    pid_file.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'read'):
        p = sub.add_parser(name)
        p.add_argument('--variant', choices=sorted(VARIANTS), required=True)
        p.add_argument('--out', type=Path, required=True)
        if name == 'read':
            p.add_argument('--targets', type=Path, required=True)
    s = sub.add_parser('stop')
    s.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'start':
        return start(args.out, args.variant)
    if args.command == 'read':
        return read(args.out, args.variant, args.targets)
    return stop(args.out)


if __name__ == '__main__':
    sys.exit(main())
