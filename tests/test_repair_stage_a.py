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


def test_replayssm_spec_runs_get_their_own_label() -> None:
    stage_a = load('stage_a')
    assert (
        stage_a.verifier_of(
            run('--linear-attn-decode-backend', 'flashinfer', '--enable-linear-replayssm-spec')
        )
        == 'replayssm_spec'
    )


FI = ('--linear-attn-decode-backend', 'flashinfer')
TR = ('--linear-attn-verify-backend', 'triton')


def no_state(row: dict[str, object]) -> dict[str, object]:
    row['probe_env'] = {'SGLANG_REPAIR_DROP_VERIFY_STATES': '1'}
    return row


def run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]], *extra: str
) -> Path:
    stage_a = load('stage_a')
    timing = tmp_path / 'timing.json'
    timing.write_text(json.dumps(rows))
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(
        sys, 'argv', ['stage_a.py', '--timing', str(timing), '--out-dir', str(out), *extra]
    )
    stage_a.main()
    return out


FAILURES = {
    'no rows for the verifier': (
        [timing_row('fresh_b16', 'fresh', 16, *FI)],
        ['--verifier', 'flashinfer'],
    ),
    'baseline on another verifier': (
        [timing_row('fresh_b16', 'fresh', 16, *FI), timing_row('force_tv_b64', 'force', 64, *TR)],
        ['--verifier', 'triton'],
    ),
    'no-state row on the triton kernel': (
        [
            timing_row('fresh_tv_b16', 'fresh', 16, *TR),
            timing_row('force_tv_b64', 'force', 64, *TR),
            no_state(timing_row('force_nostate_tv_b64', 'force', 64, *TR)),
        ],
        ['--verifier', 'triton', '--baseline', 'fresh_tv_b16'],
    ),
    'microbenchmark with the triton kernel': (
        [
            timing_row('fresh_tv_b16', 'fresh', 16, *TR),
            timing_row('force_tv_b64', 'force', 64, *TR),
        ],
        ['--verifier', 'triton', '--baseline', 'fresh_tv_b16', '--gdn', 'GDN'],
    ),
    'baseline is not a fresh run': (
        [timing_row('force_b64', 'force', 64, *FI)],
        ['--verifier', 'flashinfer', '--baseline', 'force_b64'],
    ),
    'missing baseline': ([timing_row('force_b64', 'force', 64, *FI)], ['--verifier', 'flashinfer']),
    'duplicate run names': (
        [timing_row('fresh_b16', 'fresh', 16, *FI), timing_row('fresh_b16', 'fresh', 16, *FI)],
        ['--verifier', 'flashinfer'],
    ),
}


@pytest.mark.parametrize('case', sorted(FAILURES))
def test_bad_inputs_fail_without_writing(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, extra = FAILURES[case]
    gdn = tmp_path / 'gdn.json'
    gdn.write_text(json.dumps({'blocks': {}}))
    extra = [str(gdn) if x == 'GDN' else x for x in extra]
    out = tmp_path / 'out'
    out.mkdir()
    stale = out / 'stage_a_oracle.csv'
    stale.write_text('stale\n')
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, rows, *extra)
    assert stale.read_text() == 'stale\n'
    assert not (out / 'stage_a_oracle.json').exists()


@pytest.mark.parametrize('overridden', [False, True])
def test_outputs_are_written_together(
    overridden: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        timing_row('fresh_b16', 'fresh', 16, *FI),
        timing_row('force_b64', 'force', 64, *FI),
        no_state(timing_row('force_nostate_b64', 'force', 64, *FI)),
        timing_row('force_tv_b64', 'force', 64, *TR),
    ]
    extra = ['--verifier', 'flashinfer']
    if overridden:
        # A fully overridden baseline contributes nothing, so its verify kernel is not checked.
        extra = ['--verifier', 'triton', '--cd-us', '6500', '--ad', '7', '--f', '0.02']
    out = run_main(tmp_path, monkeypatch, rows, *extra)
    data = json.loads((out / 'stage_a_oracle.json').read_text())
    csv_rows = (out / 'stage_a_oracle.csv').read_text().strip().splitlines()
    assert len(csv_rows) == len(data['rows']) + 1 == 2
    assert data['verifier'] == ('triton' if overridden else 'flashinfer')
    if not overridden:
        assert data['rows'][0]['state_writes_source'].startswith('measured')
