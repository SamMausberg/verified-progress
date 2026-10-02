"""The SM90 regression test offered in the sgl-project/sglang#35757 comment, runnable on any tree.

The test function below is the comment's code block; only this repository's formatter changed it
(quotes and line breaks; the syntax tree is identical). It needs the helpers of SGLang's FA4 test
file, which it imports from the tree under test, so the same file runs against an unpatched tree
(fail before) and a patched one (pass after):

    PYTHONPATH=<tree>/python SGLANG_TREE=<tree> \
        python -m pytest -p no:cacheprovider experiments/upstream_fa4/regression_test_sm90.py
"""

import importlib.util
import os

import pytest
import torch
from einops import rearrange
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.utils import is_sm90_supported

_spec = importlib.util.spec_from_file_location(
    'sglang_fa4_tests',
    os.path.join(
        os.environ['SGLANG_TREE'], 'test/registered/kernels/ops/attention/test_flash_attention_4.py'
    ),
)
assert _spec is not None and _spec.loader is not None
_fa4_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa4_tests)
attention_ref = _fa4_tests.attention_ref
_generate_block_kvcache = _fa4_tests._generate_block_kvcache


@pytest.mark.skipif(
    not is_sm90_supported(),
    reason='Covers the SM90 head_dim 256 tile (128 x 80).',
)
@pytest.mark.parametrize('page_size', [1, 16])
def test_flash_attn_paged_non_tma_partial_loader_tile_sm90(page_size):
    """The SM90 head_dim 256 tile has 80 KV rows, fewer than the 128 threads of
    the cp.async paged-KV loader; a page size other than 80 must still work."""
    device = 'cuda'
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 3, 16, 600
    nheads, nheads_k, d = 16, 4, 256
    torch.random.manual_seed(0)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, _ = _generate_block_kvcache(
        seqlen_k, page_size, batch_size, nheads_k, d, d, device, dtype, dtype
    )
    # Lengths that end inside a KV tile, so the last tile is partially loaded.
    cache_seqlens = torch.tensor([seqlen_k, 97, 333], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seqlen_q

    out = flash_attn_with_kvcache(
        q=rearrange(q, 'b s h d -> (b s) h d'),
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seqlen_q,
        causal=True,
        ver=4,
    )
    out = rearrange(out, '(b s) h d -> b s h d', b=batch_size)

    key_padding_mask = rearrange(torch.arange(seqlen_k, device=device), 's -> 1 s') < rearrange(
        cache_seqlens, 'b -> b 1'
    )
    out_ref, _ = attention_ref(q, k_cache, v_cache, None, key_padding_mask, causal=True)
    out_pt, _ = attention_ref(
        q,
        k_cache,
        v_cache,
        None,
        key_padding_mask,
        causal=True,
        upcast=False,
        reorder_ops=True,
    )
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (out - out_ref).abs().mean().item() <= 1.5 * (out_pt - out_ref).abs().mean().item()
