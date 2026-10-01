"""CPU test of the SGLang glue's fixed-noise eligibility (patch 0009). It runs in a
subprocess against the engine worktree with the kernel series applied
(``SGLANG_WORKTREE``, default ``~/sglang-wt/kernel``) and is skipped without it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ENGINE = Path(os.environ.get('SGLANG_WORKTREE', Path.home() / 'sglang-wt/kernel')) / 'python'
GLUE = ENGINE / 'sglang/srt/layers/certified_head.py'

CHECK = r"""
import types

import torch

import sglang.srt.layers.certified_head as glue
import sglang.srt.runtime_context as runtime

runtime.get_spec = lambda: types.SimpleNamespace(speculative_use_rejection_sampling=False)
glue._adjusts_logits = lambda batch, observer: False
runner = types.SimpleNamespace(
    spec_algorithm=types.SimpleNamespace(is_dflash_family=lambda: False),
    sampling_observer=None,
)


def batch(**flags):
    info = dict(
        is_all_greedy=False,
        is_any_greedy=False,
        sampling_seed=torch.zeros(2, dtype=torch.int64),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    info.update(flags)
    return types.SimpleNamespace(sampling_info=types.SimpleNamespace(**info))


print(
    glue._fixed_noise_eligible(runner, batch()),
    glue._fixed_noise_eligible(runner, batch(is_any_greedy=True)),
)
"""


def test_a_batch_with_a_greedy_row_is_not_fixed_noise_eligible() -> None:
    """A batch mixing greedy and sampled rows must never take fixed-noise sampled
    verify, whatever its other flags say (here need_top_k_sampling is unset)."""
    pytest.importorskip('torch')
    if not GLUE.exists():
        pytest.skip(f'no engine worktree with the kernel series at {ENGINE}')
    env = dict(os.environ, PYTHONPATH=str(ENGINE), CUDA_VISIBLE_DEVICES='')
    run = subprocess.run(
        [sys.executable, '-c', CHECK], env=env, capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.stdout.split()[-2:] == ['True', 'False']
