"""GPU check of SGLANG_FP8_DENSE_ACT=static: error, a row alone equals the row in a batch, saturation, graph replay."""

import json
import os
import tempfile

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

torch.manual_seed(0)


class Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.02
        )
        self.quant_method = UnquantizedLinearMethod()


calib = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)  # noqa: SIM115
json.dump({'amax': {'a': 4.0}}, calib)
calib.close()
os.environ.update(
    SGLANG_FP8_DENSE='target', SGLANG_FP8_DENSE_ACT='static', SGLANG_FP8_DENSE_CALIB=calib.name
)
from sglang.srt.layers.quantization import fp8_dense_online as m

root = torch.nn.Module()
root.a = Lin(12288, 2560)
wref = root.a.weight.detach().clone()
m.maybe_convert_dense_to_fp8(root, is_draft_worker=False)
ok = True
for M in (1, 16, 64):
    x = torch.randn(
        M, 2560, device='cuda', dtype=torch.bfloat16
    )  # |x| mostly below the calibrated 4.0
    y = root.a.quant_method.apply(root.a, x)
    ref = F.linear(x.float(), wref.float())
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    same = (
        True
        if M == 1
        else torch.equal(root.a.quant_method.apply(root.a, x[1:2].contiguous())[0], y[1])
    )
    print(f'M={M}: rel err {rel:.4f}, row 1 alone == in batch: {same}')
    ok &= rel < 0.06 and same
x = torch.randn(4, 2560, device='cuda', dtype=torch.bfloat16)
x[0, 0] = 1000.0  # beyond the calibrated maximum: must saturate, not overflow
y = root.a.quant_method.apply(root.a, x)
print('saturation finite:', bool(torch.isfinite(y).all()))
ok &= bool(torch.isfinite(y).all())
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
same = torch.equal(yg, root.a.quant_method.apply(root.a, x))
print('graph replay equal eager:', same)
assert ok and same, 'static FP8 check failed'
print('OK')
