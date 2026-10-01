"""Tests for the per-cycle kernel attribution of experiments/repair/nsys_kernels.py (CPU only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'repair'
VERIFY = 'kernel_cutlass_gdn_verify_kernel_mtp_inline_tensorptrf32'
COMMIT = 'fused_mamba_state_scatter_with_mask_kernel'


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('nsys_kernels', ROOT / 'nsys_kernels.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['nsys_kernels'] = module
    spec.loader.exec_module(module)
    return module


def cycle(
    t: int, verify_layers: int = 24, commits: int = 2
) -> tuple[list[tuple[int, int, str]], int]:
    """One draft GEMM, verify_layers verify launches of 100 ns, then the state commit."""
    out = [(t, t + 50, 'nvjet_sm90_tst_draft')]
    t += 100
    for _ in range(verify_layers):
        out.append((t, t + 100, VERIFY))
        t += 200
    for _ in range(commits):
        out.append((t, t + 10, COMMIT))
        t += 20
    return out, t + 1000


def test_categories_keep_verify_and_norm_out_of_gemm() -> None:
    nk = load()
    assert nk.categorize(VERIFY) == 'gdn_verify'
    assert nk.categorize('fused_sigmoid_gating_delta_rule_update_kernel') == 'gdn_verify'
    assert nk.categorize('chunk_gated_delta_rule_fwd_kernel_h') == 'gdn_other'
    assert nk.categorize('_causal_conv1d_update_kernel') == 'gdn_conv'
    assert nk.categorize('kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm') == 'norm'
    assert nk.categorize('nvjet_sm90_tst_320x128_64x3_4x2_h_bz_coopB_TNT') == 'gemm'
    assert nk.categorize(COMMIT) == 'state_commit'


def test_truncated_trailing_cycle_is_not_charged() -> None:
    nk = load()
    kernels: list[tuple[int, int, str]] = []
    t = 0
    for _ in range(2):
        ks, t = cycle(t)
        kernels += ks
    tail, _ = cycle(t, verify_layers=18, commits=0)  # cut off by nsys stop
    kernels += tail
    out = nk.summarize(Path('r.nsys-rep'), kernels)
    assert out['cycles'] == 2
    assert out['per_cycle_us']['gdn_verify'] == pytest.approx(24 * 100 / 1e3)
    assert out['per_cycle_us']['gemm'] == pytest.approx(50 / 1e3)
    assert out['per_cycle_us']['state_commit'] == pytest.approx(2 * 10 / 1e3)
    assert out['kernels_after_last_commit'] == 19
    assert out['kernel_ms_after_last_commit_not_charged'] == pytest.approx((50 + 18 * 100) / 1e6)


def test_partial_windows_fail_loudly() -> None:
    nk = load()
    ks, t = cycle(0, verify_layers=12)  # the window opened inside a cycle
    more, _ = cycle(t)
    with pytest.raises(SystemExit, match='not a whole number'):
        nk.summarize(Path('r.nsys-rep'), ks + more)
    no_commit, _ = cycle(0, commits=0)
    with pytest.raises(SystemExit, match='no state commit'):
        nk.summarize(Path('r.nsys-rep'), no_commit)
