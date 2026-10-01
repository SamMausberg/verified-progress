"""validate_p4_ab.py must fail closed: partial points, failed arms and a wrong order are FAILED."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/moonshot/validate_p4_ab.py'
DENSE = 'plain+no_radix+p4_pools'
EXACT = 'plain+no_radix+p4_pools+exact_replay'
ORDER = [
    (DENSE, 'r1'),
    (EXACT, 'r1'),
    (EXACT, 'r2'),
    (DENSE, 'r2'),
    (DENSE, 'r3'),
    (EXACT, 'r3'),
    (EXACT, 'r4'),
    (DENSE, 'r4'),
]
MANIFEST: dict[str, Any] = {
    'workload': {
        'file': 'long2048.jsonl',
        'sha256': 'db376fa3aadf75a30933a649b5ded1dfcafac8289b8e2aed1dde7201afd2659c',
        'prompts': 512,
        'warmup_pool_sha256': 'b4b5b4e43b53f3c64083263113904868cccf23767aa0b3c5f1b45740c13130a6',
    },
    'osl': 512,
    'ignore_eos': True,
    'request_body': {'temperature': 0.0, 'ignore_eos': True},
}
REPO_HEAD = 'a' * 40
ENGINE_HEAD = 'b' * 40
ENGINE = '/engine'
POOL_LOG = (
    'max_total_num_tokens=360448, max_running_requests=128\n'
    'Mamba Cache is allocated. max_mamba_cache_size: 128, conv_state size: 0.10GB, '
    'ssm_state size: 6.05GB intermediate_ssm_state_cache size: 0.00GB\n'
)


def point(rate: float, **overrides: Any) -> dict[str, Any]:
    base = {
        'concurrency': 128,
        'requests': 256,
        'completed': 256,
        'failed': 0,
        'osl_mismatch': 0,
        'aiperf_exit_code': 0,
        'prompts_as_expected': True,
        'isl_mean': 2047.9,
        'y': rate * 0.6,
        'server_log': {'logged_gen_tps_full_batch': rate, 'max_running_logged': 128},
    }
    base.update(overrides)
    return base


def make_run(
    root: Path,
    order: list[tuple[str, str]] = ORDER,
    manifest: dict[str, Any] = MANIFEST,
    pool_log: str = POOL_LOG,
    tile: str = '32',
    engine_head: str = ENGINE_HEAD,
    dirty: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    state_dtype: str = 'float32',
    **overrides: Any,
) -> Path:
    """A synthetic A/B directory; `overrides` apply to the first exact-replay arm's point,
    `manifest` and `pool_log` to every arm."""
    records = []
    first_exact = True
    for arm, pair in order:
        config = f'{arm}#{pair}'
        records.append({'config': config, 'status': 'exit 0', 'seconds': 1})
        run = root / config.replace('#', '_') / '20261001-000000'
        (run / 'server').mkdir(parents=True)
        exact = arm.endswith('exact_replay')
        extra = overrides if exact and first_exact else {}
        first_exact = first_exact and not exact
        rate = 10_000.0 * (1.2 if exact else 1.0) * (1.0 + 0.001 * int(pair[1]))
        sweep = {**manifest, 'points': [point(rate, **extra)]}
        (run / 'sweep.json').write_text(json.dumps(sweep))
        log = 'GDN decode: exact replay kernel, ring length 4\n' if exact else 'decode\n'
        (run / 'server/server.log').write_text(pool_log + log)
        env = {'SGLANG_GDN_EXACT_REPLAY': '1', 'SGLANG_GDN_EXACT_REPLAY_BV': tile} if exact else {}
        launch = {
            'env_overrides': env,
            'env_prefixed': {'SGLANG_WORKTREE': ENGINE, **env, **(extra_env or {})},
            'repo': {'head': REPO_HEAD, 'dirty_files': dirty or []},
            'sglang_source': {
                'module_file': f'{ENGINE}/python/sglang/__init__.py',
                'head': engine_head,
                'dirty_files': [],
            },
        }
        (run / 'server/launch.json').write_text(json.dumps(launch))
        (run / 'server/server_info.json').write_text(json.dumps({'mamba_ssm_dtype': state_dtype}))
    (root / 'lever_sweep_log.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    provenance = {
        'repo': '/repo',
        'repo_head': REPO_HEAD,
        'engine': ENGINE,
        'engine_head': ENGINE_HEAD,
    }
    (root / 'provenance.json').write_text(json.dumps(provenance))
    return root


def run(root: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        str(root),
        '--provenance',
        str(root / 'provenance.json'),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_complete_run_is_decided(tmp_path: Path) -> None:
    result = run(make_run(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['decision'] == 'supported'


def test_partial_point_fails(tmp_path: Path) -> None:
    # AIPerf stopped early: 128 exported rows, all successful.
    result = run(make_run(tmp_path, requests=128, completed=128))
    assert result.returncode == 1
    assert 'FAILED' in result.stdout


def test_aiperf_error_fails(tmp_path: Path) -> None:
    result = run(make_run(tmp_path, aiperf_exit_code=1))
    assert result.returncode == 1


def test_unexpected_prompts_fail(tmp_path: Path) -> None:
    result = run(make_run(tmp_path, prompts_as_expected=False))
    assert result.returncode == 1


def test_wrong_order_fails(tmp_path: Path) -> None:
    swapped = [ORDER[1], ORDER[0], *ORDER[2:]]
    result = run(make_run(tmp_path, order=swapped))
    assert result.returncode == 1
    assert 'order' in result.stdout


def test_failed_arm_fails(tmp_path: Path) -> None:
    root = make_run(tmp_path)
    log = root / 'lever_sweep_log.jsonl'
    log.write_text(log.read_text().replace('"exit 0"', '"exit 1"', 1))
    assert run(root).returncode == 1


def test_other_workload_fails(tmp_path: Path) -> None:
    manifest = {**MANIFEST, 'workload': {**MANIFEST['workload'], 'sha256': '0' * 64}}
    assert run(make_run(tmp_path, manifest=manifest)).returncode == 1


def test_sampled_request_body_fails(tmp_path: Path) -> None:
    manifest = {**MANIFEST, 'request_body': {'temperature': 0.6, 'ignore_eos': True}}
    assert run(make_run(tmp_path, manifest=manifest)).returncode == 1


def test_batch_below_128_fails(tmp_path: Path) -> None:
    server_log = {'logged_gen_tps_full_batch': 12_000.0, 'max_running_logged': 120}
    assert run(make_run(tmp_path, server_log=server_log)).returncode == 1


def test_unpinned_pools_fail(tmp_path: Path) -> None:
    pool_log = POOL_LOG.replace('max_running_requests=128', 'max_running_requests=133')
    assert run(make_run(tmp_path, pool_log=pool_log)).returncode == 1


def test_small_kv_pool_fails(tmp_path: Path) -> None:
    pool_log = POOL_LOG.replace('max_total_num_tokens=360448', 'max_total_num_tokens=300000')
    assert run(make_run(tmp_path, pool_log=pool_log)).returncode == 1


def test_other_value_tile_fails(tmp_path: Path) -> None:
    assert run(make_run(tmp_path, tile='16')).returncode == 1


def test_other_engine_commit_fails(tmp_path: Path) -> None:
    assert run(make_run(tmp_path, engine_head='c' * 40)).returncode == 1


def test_dirty_repository_fails(tmp_path: Path) -> None:
    assert run(make_run(tmp_path, dirty=[' M experiments/moonshot/levers.py'])).returncode == 1


def test_missing_provenance_fails(tmp_path: Path) -> None:
    root = make_run(tmp_path)
    (root / 'provenance.json').unlink()
    assert run(root).returncode == 1


def test_inherited_variable_fails(tmp_path: Path) -> None:
    extra = {'SGLANG_MAMBA_SSM_DTYPE': 'float8_e4m3fn'}
    assert run(make_run(tmp_path, extra_env=extra)).returncode == 1


def test_other_state_dtype_fails(tmp_path: Path) -> None:
    assert run(make_run(tmp_path, state_dtype='float16')).returncode == 1


def test_half_size_state_pool_fails(tmp_path: Path) -> None:
    pool_log = POOL_LOG.replace('ssm_state size: 6.05GB', 'ssm_state size: 3.02GB')
    assert run(make_run(tmp_path, pool_log=pool_log)).returncode == 1
