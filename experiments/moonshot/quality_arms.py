"""Run the logit probe (and optionally the token-map calibration) for a list of lever stacks.

One server per configuration, launched through the bench harness (`bench.server`)
at a small memory footprint so the job can share the GPU. No timing is taken or
reported here; this is the quality side of every lever. The first configuration
should be the reference (`plain` with no levers); every later one is compared
against it in both probe modes.

    scripts/gpu_lock.sh -s python experiments/moonshot/quality_arms.py \
        --out ~/vp-data/moonshot/quality --configs plain plain+bf16_state plain+fp8_weights

A configuration is `<base arm>[+lever...]`, with levers from `levers.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from levers import compose, environment, lossy_label, target_model

# Small, shareable footprint (5 GDN state slots per running request, plus one
# intermediate state per draft token under speculation).
SHARED_FLAGS: dict[str, dict[str, object]] = {
    'plain': {
        'mem-fraction-static': 0.25,
        'max-running-requests': 32,
        'max-mamba-cache-size': 170,
        'cuda-graph-max-bs-decode': 32,
    },
    'speculative': {
        'mem-fraction-static': 0.25,
        'max-running-requests': 8,
        'max-mamba-cache-size': 45,
        'cuda-graph-max-bs-decode': 8,
    },
}


@contextlib.contextmanager
def startup_lock() -> Iterator[None]:
    """Hold ~/.gpu.lock.startup while a shared-mode server starts (scripts/gpu_startup_lock.sh).

    SGLang sizes its pools from the free memory it sees while loading, so concurrent
    start-ups of shared jobs race; the lock serialises launch-and-wait-healthy only.
    """
    path = Path(os.environ.get('GPU_LOCK_FILE', Path.home() / '.gpu.lock')).with_suffix(
        '.lock.startup'
    )
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_config(text: str) -> tuple[str, list[str]]:
    """`base[+lever...][#tag]`; the tag only distinguishes repeated runs."""
    base, *levers = text.split('#')[0].split('+')
    return base, levers


def launch(config: str, out_dir: Path, port: int, extra: dict[str, object]):
    from bench.arms import Arm, resolve_arm
    from bench.server import Server

    base, levers = parse_config(config)
    lever_flags = compose(levers, base)
    speculative = base == 'mtp' or 'speculative-algorithm' in lever_flags
    shared = SHARED_FLAGS['speculative' if speculative else 'plain']
    flags = {**lever_flags, **shared, **extra}
    if flags.get('disable-radix-cache'):
        # One slot per request: drop the arm default's slot count (640 FP32 slots
        # would not fit a shared-mode budget), and size the pool by max-running.
        flags['max-mamba-cache-size'] = False
    arm = resolve_arm(base, flags)
    model = target_model(levers)
    arm = Arm(
        **{
            **arm.to_json(),
            **({'model': model[0], 'revision': model[1]} if model else {}),
            'env': {**arm.env, **environment(levers)},
            'name': config,
            'max_concurrency': int(shared['max-running-requests']),  # type: ignore[call-overload]
            'lossy': lossy_label(levers),
            'require_full_graph_coverage': False,
        }
    )
    return Server(arm, out_dir / 'server', port, strict=False)


def probe(
    url: str, out: Path, label: str, mode: str, reference: Path | None, concurrency: int
) -> None:
    command = [
        sys.executable,
        str(HERE / 'logit_probe.py'),
        'run',
        '--url',
        url,
        '--mode',
        mode,
        '--out',
        str(out),
        '--label',
        label,
        '--concurrency',
        str(concurrency),
    ]
    if reference is not None:
        command += ['--reference', str(reference)]
    subprocess.run(command, check=True)


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(HERE / 'logit_probe.py'), 'compare', str(reference), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, object] = json.loads(result.stdout)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--port', type=int, default=30071)
    parser.add_argument('--reference', default='plain', help='label of the reference config')
    parser.add_argument('--probe-concurrency', type=int, default=16)
    parser.add_argument(
        '--calibrate-token-map',
        action='store_true',
        help='also collect token-map calibration outputs on the reference server',
    )
    args = parser.parse_args()
    out_root = args.out.expanduser()
    ref_dir = out_root / args.reference
    summary_path = out_root / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for config in args.configs:
        run_dir = out_root / config
        run_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, object] = {}
        server = None
        try:
            # Shared-lock neighbours can hold memory at our startup; retry a few times.
            for attempt in range(3):
                server = launch(config, run_dir, args.port, {})
                try:
                    with startup_lock():
                        server.start()
                        server.wait_ready()
                    server.record_and_verify()
                    break
                except Exception as exc:
                    server.stop()
                    entry[f'launch_attempt_{attempt}'] = repr(exc)[-300:]
                    time.sleep(60)
            else:
                raise RuntimeError('server did not start in three attempts')
            try:
                url = server.base_url
                is_reference = config == args.reference
                if is_reference and args.calibrate_token_map:
                    subprocess.run(
                        [
                            sys.executable,
                            str(HERE / 'build_token_map.py'),
                            'generate',
                            '--url',
                            url,
                            '--out',
                            str(out_root.parent / 'token_map/outputs.jsonl'),
                        ],
                        check=True,
                    )
                conc = args.probe_concurrency
                probe(url, run_dir / 'generate.json', config, 'generate', None, conc)
                gen_ref = (ref_dir if not is_reference else run_dir) / 'generate.json'
                probe(url, run_dir / 'score.json', config, 'score', gen_ref, conc)
                entry['server_info'] = server.server_info().get('internal_states')
            finally:
                server.stop()
            if config != args.reference:
                for mode in ('generate', 'score'):
                    entry[mode] = compare(ref_dir / f'{mode}.json', run_dir / f'{mode}.json')
        except Exception as exc:  # record and continue with the next configuration
            entry['error'] = repr(exc)[-2000:]
        entry['launch'] = server.launch_record.get('command') if server else None
        summary[config] = entry
        summary_path.write_text(json.dumps(summary, indent=1))
        print(json.dumps({config: entry}, indent=1), flush=True)
    # Exit non-zero when any configuration of this invocation failed or lacks its comparison.
    failed = [
        config
        for config in args.configs
        if 'error' in summary[config]
        or (
            config != args.reference
            and not all(m in summary[config] for m in ('generate', 'score'))
        )
    ]
    if failed:
        sys.exit(f'quality_arms: failed or incomplete configurations: {failed}')


if __name__ == '__main__':
    main()
