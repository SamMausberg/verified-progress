"""Tests for the repair workstream's Stage A oracle helpers (CPU only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
