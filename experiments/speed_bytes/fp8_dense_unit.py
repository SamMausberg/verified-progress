"""GPU unit check of fp8_dense_online: conversion, both activation modes, row independence, graph capture."""

import os

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization import fp8_dense_online as m
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

torch.manual_seed(0)
failures = []


class Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.02
        )
        self.quant_method = UnquantizedLinearMethod()


for act in ('token', 'tensor'):
    os.environ['SGLANG_FP8_DENSE'] = 'target'
    os.environ['SGLANG_FP8_DENSE_ACT'] = act
    root = torch.nn.Module()
    root.a = Lin(12288, 2560)
    root.in_proj_ba = Lin(64, 2560)
    wref = root.a.weight.detach().clone()
    m.maybe_convert_dense_to_fp8(root, is_draft_worker=False)
    assert type(root.in_proj_ba.quant_method) is UnquantizedLinearMethod
    assert root.a.weight.numel() == 0
    for M in (1, 16, 64):
        x = torch.randn(M, 2560, device='cuda', dtype=torch.bfloat16)
        x[0] *= 50  # one outlier row
        y = root.a.quant_method.apply(root.a, x)
        ref = F.linear(x.float(), wref.float())
        rel = (y.float() - ref).norm(dim=1) / ref.norm(dim=1)
        # row independence: row 1 alone vs inside the batch
        if M > 1:
            y1 = root.a.quant_method.apply(root.a, x[1:2].contiguous())
            same = torch.equal(y1[0], y[1])
            d = (y1[0].float() - y[1].float()).abs().max().item()
        else:
            same, d = None, 0.0
        others = rel[1:].max().item() if M > 1 else float('nan')
        print(
            f'act={act} M={M} dtype={y.dtype} rel_err row0 {rel[0]:.4f} others {others:.4f} '
            f'row1 alone==in-batch {same} maxdiff {d:.3g}'
        )
        if rel.max().item() > 0.06:  # e4m3 rounding gives about 0.04 on these inputs
            failures.append(f'act={act} M={M}: relative error {rel.max().item():.4f}')
        if act == 'token' and M > 1 and not same:
            failures.append(f'act=token M={M}: row 1 depends on the batch')
    # graph capture
    x = torch.randn(16, 2560, device='cuda', dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        root.a.quant_method.apply(root.a, x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = root.a.quant_method.apply(root.a, x)
    g.replay()
    torch.cuda.synchronize()
    graph_equal = torch.equal(yg, root.a.quant_method.apply(root.a, x))
    print(f'act={act} graph replay equal eager: {graph_equal}')
    if not graph_equal:
        failures.append(f'act={act}: CUDA-graph replay differs from eager')
if failures:
    raise SystemExit('FAILED: ' + '; '.join(failures))
print('OK')
