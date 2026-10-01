"""CPU tests for the P13 study's projected GDN rule on a tiny random Qwen3.5 model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers.models.qwen3_5')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'moonshot'))

import gdn_state_rank_study as study


def tiny_model() -> Any:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_key_head_dim=study.K, linear_value_head_dim=16, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=['linear_attention', 'full_attention'],
    )  # fmt: skip
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


def patched(model: Any) -> list[dict[str, Any]]:
    layers = study.gdn_modules(model)
    states: list[dict[str, Any]] = [{} for _ in layers]
    for module, state in zip(layers, states, strict=True):
        module.chunk_gated_delta_rule = study.make_rule(state)
    return states


def test_full_rank_rotation_reproduces_the_unprojected_rule() -> None:
    model = tiny_model()
    states = patched(model)
    ids = torch.randint(0, 64, (1, 24))
    heads = study.gdn_modules(model)[0].num_v_heads
    rotation = torch.linalg.qr(torch.randn(heads, study.K, study.K))[0]
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits
        for state in states:
            state['basis'] = rotation.contiguous()
        rotated = model(ids, use_cache=False).logits
    assert torch.allclose(rotated, reference, atol=1e-4)


def test_reduced_rank_forward_runs_without_the_cache() -> None:
    # The study calls the model with use_cache=False: it never reuses a cache, and with the
    # cache on HF would also have the rule return and store each layer's r-wide final state.
    model = tiny_model()
    states = patched(model)
    ids = torch.randint(0, 64, (1, 24))
    heads = study.gdn_modules(model)[0].num_v_heads
    basis = torch.linalg.qr(torch.randn(heads, study.K, study.K))[0][..., :32].contiguous()
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits
        for state in states:
            state['basis'] = basis
        reduced = model(ids, use_cache=False).logits
    assert reduced.shape == reference.shape
    assert torch.isfinite(reduced).all()
    assert not torch.allclose(reduced, reference)
