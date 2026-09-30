"""Soundness tests for the replay bound library (experiments/head_geometry/bounds.py).

Each test draws a small random BF16 head and hidden states and checks that every
transport, static and self-evidence bound encloses the exact FP64 values. Skips when
torch is unavailable (the repository's CPU venv has only numpy).
"""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'head_geometry'))

import bounds as B

V, D = 1536, 256


@pytest.fixture(scope='module')
def head():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(V, D, generator=g) * 0.05).bfloat16()
    # A few heavy columns, as in real heads.
    w[:, :4] = (w[:, :4].float() * 8).bfloat16()
    return w


def _hidden(n, seed, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, D, generator=g) * scale
    h[:, 7] *= 20  # an outlier dimension
    return h.bfloat16()


def _tilings(w):
    return [
        B.contiguous_tiling(V, 64, 'cpu'),
        B.permuted_tiling(V, 128, 1, 'cpu'),
        B.kmeans_tiling(w, 96, iters=5, chunk=512),
    ]


def test_transport_and_static_bounds_enclose_exact_scores(head):
    w64 = head.double()
    hd = _hidden(8, 1).double()
    ht = (hd + _hidden(8, 2, scale=0.5).double()).bfloat16().double()
    delta = ht - hd
    zd, zt = hd @ w64.T, ht @ w64.T
    for tiling in _tilings(head):
        basis = B.top_basis(w64.T @ w64, 16)
        geo = B.tile_geometry(head, tiling, groups=(32, 128), bases={'w16': basis}, chunk=500)
        a = delta @ geo.mu.T
        err = (zt - zd - a[:, tiling.tile_of_row]).abs()
        worst = B.seg_max(err, tiling)
        for name, eps in B.tile_epsilons(geo, delta).items():
            assert bool((worst <= eps * (1 + 1e-12) + 1e-12).all()), (tiling.name, name)
        static_err = (zt - (ht @ geo.mu.T)[:, tiling.tile_of_row]).abs()
        static_worst = B.seg_max(static_err, tiling)
        for name, eps in B.tile_epsilons(geo, ht).items():
            assert bool((static_worst <= eps * (1 + 1e-12) + 1e-12).all()), (tiling.name, name)


def test_segment_reductions_match_brute_force(head):
    z = _hidden(3, 5).double() @ head.double().T
    tiling = B.kmeans_tiling(head, 128, iters=3, chunk=512)
    m = B.seg_max(z, tiling)
    lse = B.seg_logsumexp(z, tiling)
    for c in range(tiling.num_tiles):
        rows = tiling.tile_of_row == c
        assert torch.allclose(m[:, c], z[:, rows].max(1).values)
        assert torch.allclose(lse[:, c], torch.logsumexp(z[:, rows], 1))


def _heads(w):
    w64 = w.double()
    centre = w64.mean(0)
    basis = B.top_basis(_hidden(64, 9).double().T @ _hidden(64, 9).double(), 16)
    int8 = B.quantize(w, 'int8', None, 'int8_row')
    return [
        int8,
        B.quantize(w, 'int8', 128, 'int8_g128'),
        B.quantize(w, 'int8', 32, 'int8_g32'),
        B.quantize(w, 'int4', 128, 'int4_g128'),
        B.quantize(w, 'fp8', None, 'fp8_row'),
        B.quantize(w, 'int8', None, 'int8_row_centred', centre=centre),
        B.outlier_head(w, torch.tensor([7, 0, 1]), 'int8', None, 'int8_out3'),
        B.rotated_head(w, basis, int8, 'int8_rot16'),
    ]


def _envelope(head, h):
    return head.envelope(h) if hasattr(head, 'envelope') else B.quant_envelope(head, h)


def test_quant_envelopes_enclose_exact_logits(head):
    h = _hidden(16, 3).double()
    z = h @ head.double().T
    for qh in _heads(head):
        env = _envelope(qh, h)
        err = (z - env.z_hat).abs()
        for name, half in env.quant.items():
            assert bool((err <= half * (1 + 1e-12) + 1e-12).all()), (qh.name, name)


def test_quant_metadata(head):
    h = _hidden(16, 3).double()
    for qh in _heads(head)[:6]:
        if qh.free_einf_block is not None:
            assert bool((qh.einf_block <= qh.free_einf_block * (1 + 1e-12)).all()), qh.name
        # The accumulation scale bounds sum_j |w_hat_ij h_j|.
        w_hat = torch.cat([qh.dequant(0, V)])
        abs_sum = h.abs() @ w_hat.abs().T
        env = B.quant_envelope(qh, h)
        assert bool((abs_sum <= env.acc_scale * (1 + 1e-12)).all()), qh.name
        # Dequantized values are exact in FP32 (so a kernel sees the same weights).
        assert torch.equal(w_hat.float().double(), w_hat), qh.name


def test_candidate_sets_contain_the_exact_winner(head):
    qh = B.quantize(head, 'int4', 128, 'int4_g128')
    h = _hidden(16, 4).double()
    z = h @ head.double().T
    env = B.quant_envelope(qh, h)
    half = env.quant['block_l2']
    cand = B.candidate_mask(env.z_hat, half)
    assert bool(cand.gather(1, z.argmax(1, keepdim=True)).all())
    noise = B.gumbel_noise(tuple(z.shape), 7, 'cpu')
    for temp in (1.0, 0.7):
        cand_g = B.candidate_mask(env.z_hat / temp + noise, half / temp)
        winner = (z / temp + noise).argmax(1, keepdim=True)
        assert bool(cand_g.gather(1, winner).all())


def test_unresolved_probability_matches_equation(head):
    # Eq. (unknown): P = min(1, w/(q Z-)) - min(1, w/(q Z+)).
    w_x, q_x, z_true, z_lo, z_hi = 2.0, 0.3, 10.0, 8.0, 13.0
    direct = min(1, w_x / (q_x * z_lo)) - min(1, w_x / (q_x * z_hi))
    got = B.unresolved_probability(
        torch.tensor(math.log(w_x / z_true), dtype=torch.float64),
        torch.tensor(math.log(q_x), dtype=torch.float64),
        torch.tensor(math.log(z_true / z_lo), dtype=torch.float64),
        torch.tensor(math.log(z_true / z_hi), dtype=torch.float64),
    )
    assert abs(float(got) - direct) < 1e-12
