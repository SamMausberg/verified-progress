"""Unit tests for the state-safety server helpers: pinned pools (CPU only)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

import server
from server import POOL_PIN, expected_pools, pool_flags, resolved_pools


def test_pool_flags_and_overrides():
    flags = pool_flags()
    assert expected_pools(flags) == POOL_PIN
    # A later flag (the retraction test's smaller KV pool) overrides the pin.
    assert expected_pools([*flags, '--max-total-tokens', '6000']) == {
        **POOL_PIN,
        'max_total_tokens': 6000,
    }


def test_resolved_pools_reads_the_last_allocation(tmp_path):
    log = tmp_path / 'server.log'
    log.write_text(
        'Mamba Cache is allocated. max_mamba_cache_size: 99, conv_state size: 0.11GB\n'
        'max_total_num_tokens=1000, chunked_prefill_size=8192, max_running_requests=16, x=1\n'
        'Mamba Cache is allocated. max_mamba_cache_size: 40, conv_state size: 0.05GB\n'
        'max_total_num_tokens=49152, chunked_prefill_size=8192, max_prefill_tokens=16384, '
        'max_running_requests=8, context_len=262144\n'
    )
    assert resolved_pools(log) == {
        'max_total_tokens': 49152,
        'max_running_requests': 8,
        'max_mamba_cache_size': 40,
    }
    log.write_text('no allocation yet\n')
    assert set(resolved_pools(log).values()) == {None}


def test_startup_lock_waits_for_free_memory(tmp_path, monkeypatch):
    monkeypatch.setenv('GPU_LOCK_FILE', str(tmp_path / 'lock'))
    monkeypatch.setenv('GPU_STARTUP_RETRY_WAIT', '0')
    monkeypatch.setenv('GPU_STARTUP_TRIES', '3')
    readings = iter([10.0, 70.0])
    monkeypatch.setattr(server, 'gpu_free_gb', lambda: next(readings))
    with server.startup_lock(min_free=65.0):
        pass
    monkeypatch.setattr(server, 'gpu_free_gb', lambda: 10.0)
    with pytest.raises(TimeoutError, match='memory failure'), server.startup_lock(min_free=65.0):
        pass


def test_run_matrix_refuses_a_root_of_the_other_pool_regime(tmp_path):
    import json

    from server import mixed_pin_runs

    legacy = tmp_path / 'legacy' / 'plain'
    legacy.mkdir(parents=True)
    (legacy / 'c1.meta.json').write_text(json.dumps({'config': 'plain'}))  # no pool_pin key
    root = tmp_path / 'campaign'
    root.mkdir()
    (root / 'plain').symlink_to(legacy)  # a linked reference counts too
    assert mixed_pin_runs(root, pin=True) == ['plain']
    assert mixed_pin_runs(root, pin=False) == []
    arm = root / 'mtp_s3__x'
    arm.mkdir()
    (arm / 'c1.meta.json').write_text(json.dumps({'pool_pin': POOL_PIN}))
    assert mixed_pin_runs(root, pin=False) == ['mtp_s3__x']


def test_compare_flags_pinned_state(tmp_path):
    import json

    from compare import pinned

    (tmp_path / 'a.meta.json').write_text(json.dumps({'pool_pin': None}))
    (tmp_path / 'b.meta.json').write_text(json.dumps({'pool_pin': POOL_PIN}))
    assert pinned(tmp_path / 'a.meta.json') is False
    assert pinned(tmp_path / 'b.meta.json') is True
    assert pinned(tmp_path / 'missing.meta.json') is None


def test_pools_match_uses_each_pass_server_and_treats_missing_fields_as_unknown(tmp_path):
    import json

    from compare import pools_match

    run = tmp_path / 'plain'
    run.mkdir()
    pools = {'max_total_tokens': 100, 'max_running_requests': 8, 'max_mamba_cache_size': 40}

    def meta(name, **info):
        (run / f'{name}.meta.json').write_text(
            json.dumps({'started_at': '2026-10-01T10:05:00', 'server_info': info})
        )

    # New runs: one server id per process; passes of two servers are told apart.
    meta('c1', server_id='s1', resolved_pools=pools)
    meta('c32', server_id='s2', resolved_pools={**pools, 'max_total_tokens': 200})
    assert pools_match(tmp_path, 'plain/c1', 'plain/c32') == (False, False)
    meta('c32', server_id='s1', resolved_pools=pools)
    assert pools_match(tmp_path, 'plain/c1', 'plain/c32') == (True, True)
    # A pool field that was not found is unknown, not equal.
    missing = {**pools, 'max_mamba_cache_size': None}
    meta('c1', server_id='s1', resolved_pools=missing)
    meta('c32', server_id='s2', resolved_pools=missing)
    assert pools_match(tmp_path, 'plain/c1', 'plain/c32') == (False, None)
    # Older runs: the session log only counts for passes that started inside its span.
    (run / 'server.log').write_text(
        '[2026-10-01 10:00:00] Mamba Cache is allocated. max_mamba_cache_size: 40, x\n'
        '[2026-10-01 10:00:01] max_total_num_tokens=100, a=1, max_running_requests=8, b=2\n'
        '[2026-10-01 10:30:00] Decode batch\n'
    )
    meta('c1')
    meta('c32')
    assert pools_match(tmp_path, 'plain/c1', 'plain/c32') == (True, True)
    (run / 'c32.meta.json').write_text(
        json.dumps({'started_at': '2026-10-01T09:00:00', 'server_info': {}})
    )
    assert pools_match(tmp_path, 'plain/c1', 'plain/c32') == (False, None)
