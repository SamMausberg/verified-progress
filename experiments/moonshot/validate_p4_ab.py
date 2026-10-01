"""Validate P4's served A/B run and apply the declared decision rule (README 2c).

Reads one fresh lever_sweep output directory and checks, for this run only:
  - lever_sweep_log.jsonl lists exactly the expected configurations in the declared
    A B B A A B B A order, each with status 'exit 0';
  - every configuration has exactly one sweep.json with one concurrency-128 point that ran
    the configured number of requests (256) to completion: requests == completed == 256,
    failed 0, AIPerf exit code 0, the prompts sent as expected and no output-length
    mismatch; and that carries the token-weighted server full-batch decode rate measured
    with at least 128 requests running (server_log.max_running_logged >= 128);
  - every arm ran the declared workload (long2048.jsonl and its warm-up pool by SHA-256,
    512 prompts, mean input length 2,040-2,049 tokens, OSL 512 with ignore_eos, greedy
    request body) and resolved the pinned pools from its server log: max_running_requests
    128, max_mamba_cache_size 128 and a KV pool (max_total_num_tokens; 360,448 requested)
    of at least 327,680 tokens (128 x 2,560), identical in all eight arms;
  - every exact-replay server log shows the exact-replay kernel dispatch line, and no
    dense server log does; every exact-replay server was launched with
    SGLANG_GDN_EXACT_REPLAY_BV=32 (bench's launch.json records the arm's environment,
    which overrides anything inherited);
  - the run forms exactly four complete dense/exact pairs (labels r1-r4);
  - environment: each server's SGLANG_/FLASHINFER_/TRITON_/TORCH_/PYTORCH_/NCCL_ variables
    (read from the process by RecordingServer, launch.json `env_prefixed`) equal the declared
    set: SGLANG_WORKTREE only for dense arms, plus SGLANG_GDN_EXACT_REPLAY=1 and
    SGLANG_GDN_EXACT_REPLAY_BV=32 for exact arms;
  - state dtype: every server ran with mamba_ssm_dtype float32 (server_info.json) and its
    log reports an SSM pool of FP32 size for its slot count;
  - provenance: the run's provenance file (written by run_p4b.sh after its clean-tree
    preflight) names both HEADs, and every arm's launch record (bench's launch.json) shows
    the same repository HEAD and the same engine HEAD (the imported sglang package lies in
    that engine tree), with no modified tracked files in either.
Any failed check prints FAILED and exits 1. Otherwise it prints, per pair, the exact / dense
ratio of the primary metric (`server_log.logged_gen_tps_full_batch`) and of client y, the
mean log ratio with a t(3) 95% interval, and the decision at 1.10x (rejected if the upper
end < 1.10, supported if the lower end >= 1.10, otherwise inconclusive).

    python experiments/moonshot/validate_p4_ab.py <out dir> --json <verdict.json>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

DENSE = 'plain+no_radix+p4_pools'
EXACT = 'plain+no_radix+p4_pools+exact_replay'
PAIRS = ['r1', 'r2', 'r3', 'r4']
# Execution order declared in README 2c: A B B A A B B A.
ORDER = [
    f'{DENSE}#r1', f'{EXACT}#r1', f'{EXACT}#r2', f'{DENSE}#r2',
    f'{DENSE}#r3', f'{EXACT}#r3', f'{EXACT}#r4', f'{DENSE}#r4',
]  # fmt: skip
REQUESTS = 256  # --min-requests 256 at concurrency 128
WORKLOAD_SHA256 = 'db376fa3aadf75a30933a649b5ded1dfcafac8289b8e2aed1dde7201afd2659c'
WARMUP_SHA256 = 'b4b5b4e43b53f3c64083263113904868cccf23767aa0b3c5f1b45740c13130a6'
POOLS = {'max_running_requests': 128, 'max_mamba_cache_size': 128}
MIN_KV_TOKENS = 128 * (2048 + 512)
POOL_PATTERNS = {
    'max_running_requests': re.compile(r'max_running_requests=(\d+)'),
    'max_total_num_tokens': re.compile(r'max_total_num_tokens=(\d+)'),
    'max_mamba_cache_size': re.compile(r'max_mamba_cache_size: (\d+)'),
}
DISPATCH = 'GDN decode: exact replay kernel'
T3_975 = 3.182446305284263  # Student t, 3 degrees of freedom, two-sided 95%
THRESHOLD = 1.10


def fail(message: str) -> None:
    print(f'FAILED: {message}', flush=True)
    sys.exit(1)


def check_manifest(label: str, data: dict[str, Any]) -> None:
    workload = data.get('workload') or {}
    if workload.get('sha256') != WORKLOAD_SHA256 or workload.get('prompts') != 512:
        fail(f'{label}: workload {workload.get("file")} is not the declared long2048.jsonl')
    if workload.get('warmup_pool_sha256') != WARMUP_SHA256:
        fail(f'{label}: warm-up pool is not the declared long2048_warmup.jsonl')
    body = data.get('request_body') or {}
    if data.get('osl') != 512 or data.get('ignore_eos') is not True:
        fail(f'{label}: osl {data.get("osl")}, ignore_eos {data.get("ignore_eos")}')
    if body.get('temperature') != 0.0 or body.get('ignore_eos') is not True:
        fail(f'{label}: request body is not greedy with ignore_eos: {body}')


def resolved_pools(label: str, server_log: Path) -> dict[str, int]:
    text = server_log.read_text(errors='replace') if server_log.exists() else ''
    pools = {}
    for key, pattern in POOL_PATTERNS.items():
        values = {int(v) for v in pattern.findall(text)}
        if len(values) != 1:
            fail(f'{label}: {key} in the server log: {sorted(values)}')
        pools[key] = values.pop()
    if any(pools[key] != value for key, value in POOLS.items()):
        fail(f'{label}: resolved pools {pools}, pinned {POOLS}')
    if pools['max_total_num_tokens'] < MIN_KV_TOKENS:
        fail(f'{label}: KV pool of {pools["max_total_num_tokens"]} tokens < {MIN_KV_TOKENS}')
    return pools


EXACT_ENV = {'SGLANG_GDN_EXACT_REPLAY': '1', 'SGLANG_GDN_EXACT_REPLAY_BV': '32'}
FP32_STATE_GIB_PER_SLOT = 24 * 32 * 128 * 128 * 4 / 2**30  # 24 GDN layers, 32 heads, 128 x 128
SSM_SIZE = re.compile(r'max_mamba_cache_size: (\d+), .*?ssm_state size: ([\d.]+)GB')


def check_environment(label: str, exact: bool, launch: dict[str, Any], engine: str) -> None:
    declared = {'SGLANG_WORKTREE': engine, **(EXACT_ENV if exact else {})}
    recorded = launch.get('env_prefixed')
    if recorded != declared:
        fail(f'{label}: server environment {recorded} != declared {declared}')


def check_state_dtype(label: str, server_dir: Path, server_log_text: str) -> None:
    info_json = server_dir / 'server_info.json'
    info = json.loads(info_json.read_text()) if info_json.exists() else {}
    if info.get('mamba_ssm_dtype') != 'float32':
        fail(f'{label}: mamba_ssm_dtype {info.get("mamba_ssm_dtype")!r}, not float32')
    sizes = {(int(n), float(g)) for n, g in SSM_SIZE.findall(server_log_text)}
    if len(sizes) != 1:
        fail(f'{label}: SSM pool lines in the server log: {sorted(sizes)}')
    slots, gib = sizes.pop()
    # The pool holds the slots plus one reserved slot; FP16 or FP8 would be half or less.
    if not any(abs(gib - FP32_STATE_GIB_PER_SLOT * n) <= 0.01 for n in (slots, slots + 1)):
        fail(f'{label}: SSM pool {gib} GiB for {slots} slots is not FP32')


def check_provenance(label: str, launch: dict[str, Any], provenance: dict[str, str]) -> None:
    repo = launch.get('repo') or {}
    engine = launch.get('sglang_source') or {}
    if repo.get('head') != provenance['repo_head'] or repo.get('dirty_files'):
        fail(f'{label}: repository {repo.get("head")} dirty={repo.get("dirty_files")}')
    if engine.get('head') != provenance['engine_head'] or engine.get('dirty_files'):
        fail(f'{label}: engine {engine.get("head")} dirty={engine.get("dirty_files")}')
    module = str(engine.get('module_file') or '')
    if not module.startswith(provenance['engine'].rstrip('/') + '/'):
        fail(f'{label}: sglang imported from {module!r}, not {provenance["engine"]}')


def point_of(out: Path, label: str) -> tuple[dict[str, Any], Path]:
    runs = sorted((out / label).glob('*/sweep.json'))
    if len(runs) != 1:
        fail(f'{label}: expected one sweep.json, found {len(runs)}')
    data = json.loads(runs[0].read_text())
    check_manifest(label, data)
    points = [p for p in data.get('points', []) if p.get('concurrency') == 128]
    if len(points) != 1:
        fail(f'{label}: expected one concurrency-128 point, found {len(points)}')
    point = points[0]
    # `requests` counts exported rows, so an AIPerf run that stops early can show
    # completed == requests below the configured count; check the count itself.
    requests, completed = int(point.get('requests') or 0), int(point.get('completed') or 0)
    if requests != REQUESTS or completed != REQUESTS or int(point.get('failed') or 0) != 0:
        fail(
            f'{label}: {completed}/{requests} completed of {REQUESTS}, {point.get("failed")} failed'
        )
    if point.get('aiperf_exit_code') != 0:
        fail(f'{label}: AIPerf exit code {point.get("aiperf_exit_code")}')
    if point.get('prompts_as_expected') is not True:
        fail(f'{label}: prompts not as expected')
    if int(point.get('osl_mismatch') or 0) != 0:
        fail(f'{label}: {point.get("osl_mismatch")} outputs with the wrong length')
    if not 2040 <= float(point.get('isl_mean') or 0) <= 2049:
        fail(f'{label}: mean input length {point.get("isl_mean")}, expected about 2,048')
    if int((point.get('server_log') or {}).get('max_running_logged') or 0) < 128:
        fail(f'{label}: at most {point["server_log"].get("max_running_logged")} requests running')
    if not (point.get('server_log') or {}).get('logged_gen_tps_full_batch'):
        fail(f'{label}: no token-weighted server full-batch decode rate')
    return point, runs[0].parent / 'server/server.log'


def interval(ratios: list[float]) -> tuple[float, float, float]:
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    half = T3_975 * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    out = args.out.expanduser()
    if not args.provenance.exists():
        fail(f'provenance file {args.provenance} missing')
    provenance = json.loads(args.provenance.read_text())
    for key in ('repo', 'repo_head', 'engine', 'engine_head'):
        if not provenance.get(key):
            fail(f'provenance lacks {key}')
    expected = ORDER

    log = out / 'lever_sweep_log.jsonl'
    if not log.exists():
        fail(f'{log} missing')
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if [r['config'] for r in records] != expected:
        fail(f'configurations in the log {[r["config"] for r in records]} != {expected} (order)')
    bad = [r['config'] for r in records if r['status'] != 'exit 0']
    if bad:
        fail(f'arms did not exit 0: {bad}')

    rows = []
    for pair in PAIRS:
        row: dict[str, Any] = {'pair': pair}
        for arm, key in ((DENSE, 'dense'), (EXACT, 'exact')):
            label = f'{arm}#{pair}'.replace('#', '_')
            point, server_log = point_of(out, label)
            text = server_log.read_text(errors='replace') if server_log.exists() else ''
            dispatched = DISPATCH in text
            if dispatched != (arm == EXACT):
                fail(f'{label}: exact-replay dispatch line present={dispatched}')
            launch_json = server_log.parent / 'launch.json'
            if not launch_json.exists():
                fail(f'{label}: launch.json missing')
            launch = json.loads(launch_json.read_text())
            check_provenance(label, launch, provenance)
            check_environment(label, arm == EXACT, launch, provenance['engine'])
            check_state_dtype(label, server_log.parent, text)
            if arm == EXACT:
                tile = (launch.get('env_overrides') or {}).get('SGLANG_GDN_EXACT_REPLAY_BV')
                if tile != '32':
                    fail(f'{label}: launched with SGLANG_GDN_EXACT_REPLAY_BV={tile!r}, not 32')
            row[f'{key}_pools'] = resolved_pools(label, server_log)
            row[f'{key}_server_tps'] = float(point['server_log']['logged_gen_tps_full_batch'])
            row[f'{key}_client_y'] = float(point['y'])
        row['server_ratio'] = row['exact_server_tps'] / row['dense_server_tps']
        row['client_ratio'] = row['exact_client_y'] / row['dense_client_y']
        rows.append(row)

    kv_pools = {
        r[f'{key}_pools']['max_total_num_tokens'] for r in rows for key in ('dense', 'exact')
    }
    if len(kv_pools) != 1:
        fail(f'KV pools differ between arms: {sorted(kv_pools)}')
    server = interval([r['server_ratio'] for r in rows])
    client = interval([r['client_ratio'] for r in rows])
    if server[2] < THRESHOLD:
        decision = 'rejected'
    elif server[1] >= THRESHOLD:
        decision = 'supported'
    else:
        decision = 'inconclusive'
    verdict = {
        'provenance': provenance,
        'pairs': rows,
        'primary_metric': 'server_log.logged_gen_tps_full_batch',
        'server_ratio_mean_ci95': server,
        'client_y_ratio_mean_ci95': client,
        'threshold': THRESHOLD,
        'decision': decision,
    }
    print(json.dumps(verdict, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(verdict, indent=1) + '\n')


if __name__ == '__main__':
    main()
