"""Tests for the repair workstream's Stage A oracle helpers (CPU only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'repair'


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(*flags: str) -> dict[str, list[str]]:
    return {'command': ['python', '-m', 'sglang.launch_server', *flags]}


def test_verifier_follows_sglang_resolution() -> None:
    stage_a = load('stage_a')
    verifier_of = stage_a.verifier_of
    # The repair runs: FlashInfer decode backend, verify follows it unless overridden.
    assert verifier_of(run('--linear-attn-decode-backend', 'flashinfer')) == 'flashinfer'
    assert (
        verifier_of(
            run(
                '--linear-attn-decode-backend',
                'flashinfer',
                '--linear-attn-verify-backend',
                'triton',
            )
        )
        == 'triton'
    )
    # A Triton decode backend (explicit or through --linear-attn-backend) gives a Triton verifier.
    assert verifier_of(run('--linear-attn-decode-backend', 'triton')) == 'triton'
    assert verifier_of(run('--linear-attn-backend', 'flashinfer')) == 'flashinfer'
    assert (
        verifier_of(
            run('--linear-attn-backend', 'flashinfer', '--linear-attn-decode-backend', 'triton')
        )
        == 'triton'
    )
    # No flags: SGLang's default linear-attention backend is Triton.
    assert verifier_of(run()) == 'triton'
    assert verifier_of({}) == 'triton'


def test_flag_value_accepts_equals_form() -> None:
    stage_a = load('stage_a')
    assert stage_a.verifier_of(run('--linear-attn-verify-backend=triton')) == 'triton'
    assert stage_a.verifier_of(run('--linear-attn-decode-backend=flashinfer')) == 'flashinfer'


def timing_row(name: str, mode: str, block: int, *flags: str) -> dict[str, object]:
    def q(value: float) -> dict[str, float]:
        return {'median': value, 'mean': value}

    return {
        'run': f'/runs/{name}',
        'mode': mode,
        'block': block,
        'command': ['python', '-m', 'sglang.launch_server', *flags],
        'probe_env': {},
        'phase_us': {
            'verify': q(4000.0),
            'commit': q(100.0),
            'draft': q(2000.0),
            'append': q(100.0),
        },
        'cycle_period_us': q(6500.0),
        'commit_per_cycle': q(7.0 if mode == 'fresh' else float(block)),
        'unaffected_fraction': q(0.02),
    }


def test_baseline_verifier_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage_a = load('stage_a')
    timing = tmp_path / 'timing.json'
    timing.write_text(
        json.dumps(
            [
                timing_row('fresh_b16', 'fresh', 16, '--linear-attn-decode-backend', 'flashinfer'),
                timing_row(
                    'force_tritonverify_b64', 'force', 64, '--linear-attn-verify-backend', 'triton'
                ),
            ]
        )
    )
    base_args = [
        'stage_a.py',
        '--timing',
        str(timing),
        '--verifier',
        'triton',
        '--out-dir',
        str(tmp_path),
    ]
    # A FlashInfer baseline for a Triton calculation is rejected ...
    monkeypatch.setattr(sys, 'argv', base_args)
    with pytest.raises(SystemExit):
        stage_a.main()
    # ... unless C_D, A_D and f are all given, so the named baseline contributes nothing.
    monkeypatch.setattr(sys, 'argv', [*base_args, '--cd-us', '6500', '--ad', '7', '--f', '0.02'])
    stage_a.main()
    out = json.loads((tmp_path / 'stage_a_oracle.json').read_text())
    assert [row['B'] for row in out['rows']] == [64]


def test_replayssm_spec_runs_get_their_own_label() -> None:
    stage_a = load('stage_a')
    assert (
        stage_a.verifier_of(
            run('--linear-attn-decode-backend', 'flashinfer', '--enable-linear-replayssm-spec')
        )
        == 'replayssm_spec'
    )


def test_no_state_comparator_must_be_flashinfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_a = load('stage_a')
    bad = timing_row(
        'force_nostate_tritonverify_b64', 'force', 64, '--linear-attn-verify-backend', 'triton'
    )
    bad['probe_env'] = {'SGLANG_REPAIR_DROP_VERIFY_STATES': '1'}
    timing = tmp_path / 'timing.json'
    timing.write_text(
        json.dumps(
            [
                timing_row(
                    'fresh_tritonverify_b16', 'fresh', 16, '--linear-attn-verify-backend', 'triton'
                ),
                timing_row(
                    'force_tritonverify_b64', 'force', 64, '--linear-attn-verify-backend', 'triton'
                ),
                bad,
            ]
        )
    )
    args = [
        'stage_a.py',
        '--timing',
        str(timing),
        '--verifier',
        'triton',
        '--baseline',
        'fresh_tritonverify_b16',
        '--out-dir',
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit, match='not a no-state run'):
        stage_a.main()


def test_gdn_microbenchmark_only_describes_flashinfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_a = load('stage_a')
    timing = tmp_path / 'timing.json'
    timing.write_text(
        json.dumps(
            [
                timing_row(
                    'fresh_tritonverify_b16', 'fresh', 16, '--linear-attn-verify-backend', 'triton'
                )
            ]
        )
    )
    gdn = tmp_path / 'gdn.json'
    gdn.write_text(json.dumps({'blocks': {}}))
    args = [
        'stage_a.py',
        '--timing',
        str(timing),
        '--verifier',
        'triton',
        '--baseline',
        'fresh_tritonverify_b16',
        '--gdn',
        str(gdn),
        '--out-dir',
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit, match='gated_delta_rule_mtp'):
        stage_a.main()
