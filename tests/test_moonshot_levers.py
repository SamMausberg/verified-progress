"""Tests for the moonshot lever composition and the engine/moonshot patches.

The lever tests are pure Python. The engine tests import SGLang from the
engine/moonshot worktree (SGLANG_WORKTREE) and skip cleanly when SGLang or torch
is not importable (the repository's CPU venv has neither).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/moonshot'))

from levers import LEVERS, compose, environment, lossy_label, target_model


def test_compose_merges_and_refuses_conflicts() -> None:
    flags = compose(['fp16_state', 'replayssm', 'fp8_kv'], 'plain')
    assert flags['mamba-ssm-dtype'] == 'float16'
    assert flags['enable-linear-replayssm'] is True
    assert flags['kv-cache-dtype'] == 'fp8_e4m3'
    with pytest.raises(ValueError, match='conflicts'):
        compose(['mtp_s5', 'replayssm', 'replayssm_spec'], 'mtp')
    with pytest.raises(ValueError, match='needs base arm'):
        compose(['mtp_s5'], 'plain')


def test_lossy_labels_and_model_swaps() -> None:
    assert lossy_label(['replayssm_spec', 'hot16k']) == ''
    assert 'FP8' in lossy_label(['fp8_weights'])
    assert target_model(['fp8_kv']) is None
    assert target_model(['qad_target']) == (
        'nota-ai/Qwen3.5-4B-QAD-W4A16',
        'a67b0fedb2b39cb057da6114e656b76b52d321b4',
    )
    assert environment(['relax1'])['SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP'] == '1.0'
    # Every lever that changes numerics says so; exact levers say nothing.
    for name in ('bf16_state', 'fp16_state', 'fp8_state', 'fp8_weights', 'fp8_kv'):
        assert LEVERS[name].lossy


torch_spec = importlib.util.find_spec('torch')
sglang_spec = importlib.util.find_spec('sglang')
needs_sglang = pytest.mark.skipif(
    torch_spec is None or sglang_spec is None, reason='needs torch and SGLang (sglang venv)'
)


@needs_sglang
def test_relaxed_greedy_chain_accepts_near_argmax_drafts() -> None:
    import torch
    from sglang.srt.speculative.eagle_utils import relax_greedy_chain_predict

    bs, width, vocab = 2, 4, 32
    logits = torch.full((bs * width, vocab), -10.0)
    candidates = torch.tensor([[1, 5, 6, 7], [2, 8, 9, 10]])
    view = logits.view(bs, width, vocab)
    view[:, :, 0] = 5.0  # argmax everywhere is token 0
    view[0, 0, 5] = 4.5  # draft 5 is 0.5 below the max: accepted at gap 1
    view[0, 1, 6] = 3.0  # draft 6 is 2.0 below: rejected at gap 1
    view[1, 0, 8] = 5.0  # a tie with the argmax: accepted
    argmax = logits.argmax(-1).view(bs, width)
    relaxed = relax_greedy_chain_predict(logits, candidates, argmax, 1.0)
    assert relaxed[0].tolist() == [5, 0, 0, 0]
    assert relaxed[1].tolist() == [8, 0, 0, 0]
    # The last row is the bonus token and always keeps the argmax.
    assert relaxed[:, -1].tolist() == argmax[:, -1].tolist()
    assert relax_greedy_chain_predict(logits, candidates, argmax, 1e-9)[0, 0] == 0


@needs_sglang
def test_reduced_draft_head_unties_only_for_a_reduced_vocabulary() -> None:
    import torch
    from sglang.srt.models import qwen3_5_mtp

    head = qwen3_5_mtp._ReducedDraftHead(torch.randn(8, 4))
    assert head.weight.shape == (8, 4)
    assert not head.weight.requires_grad
    assert getattr(head, 'quant_method', None) is None


@needs_sglang
def test_mamba_state_carries_the_exact_replay_beta_ring() -> None:
    """Patch 0007's beta ring reaches the per-layer cache view, and is None when off.

    A review flagged `replayssm_beta` as a local that never reaches the layer cache; the
    pool passes it to `MambaPool.State`, whose field defaults to None, and
    `at_layer_idx` slices every field, which this test pins down.
    """
    import torch
    from sglang.srt.mem_cache.memory_pool import MambaPool

    layers, slots, heads, ring = 3, 5, 2, 4
    state = MambaPool.State(
        conv=[torch.zeros(layers, slots, 2, 3)],
        temporal=torch.zeros(layers, slots, heads, 2, 2),
        replayssm_beta=torch.arange(layers * slots * heads * ring, dtype=torch.float32).view(
            layers, slots, heads, ring
        ),
    )
    view = state.at_layer_idx(1)
    assert view.replayssm_beta is not None
    assert view.replayssm_beta.shape == (slots, heads, ring)
    assert torch.equal(view.replayssm_beta, state.replayssm_beta[1])
    plain = MambaPool.State(conv=[torch.zeros(layers, slots, 2, 3)], temporal=torch.zeros(1))
    assert plain.at_layer_idx(0).replayssm_beta is None
