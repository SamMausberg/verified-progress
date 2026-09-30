"""Transport and self-evidence bounds for a real LM head, in FP64 torch.

Everything here is a real-arithmetic bound evaluated in FP64 on BF16 inputs. BF16
products are exact in FP64 and a 5,120-term FP64 sum has relative error below 1e-12,
which is far below every bound width measured, so FP64 values stand in for the exact
real logits. The floating-point cost of computing a bound *online* (FP32 or tensor-core
accumulation) is modelled explicitly by :func:`accumulation_gamma` and the ``A`` term of
the self-evidence envelope; it is not hidden in the FP64 evaluation.

Notation follows ``paper/paper.tex`` Section 4 and ``~/vp-coord/notes/theory.md``:
``W`` is the [V, D] head, ``h`` a head input, ``Delta = h_target - h_draft``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

F64 = torch.float64

# ---------------------------------------------------------------------------
# Tilings of the vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Tiling:
    """A partition of the V rows into C tiles; ``tile_of_row[i]`` is row i's tile."""

    name: str
    tile_of_row: torch.Tensor  # [V] int64
    num_tiles: int
    sizes: torch.Tensor  # [C] int64

    @property
    def mean_size(self) -> float:
        return float(self.sizes.double().mean())


def _make_tiling(name: str, tile_of_row: torch.Tensor) -> Tiling:
    num_tiles = int(tile_of_row.max()) + 1
    sizes = torch.bincount(tile_of_row, minlength=num_tiles)
    if bool((sizes == 0).any()):
        # Renumber to drop empty tiles (k-means can empty a cluster).
        keep = sizes > 0
        remap = torch.cumsum(keep.long(), 0) - 1
        tile_of_row = remap[tile_of_row]
        num_tiles = int(keep.sum())
        sizes = sizes[keep]
    return Tiling(name, tile_of_row, num_tiles, sizes)


def contiguous_tiling(vocab: int, size: int, device: torch.device | str) -> Tiling:
    return _make_tiling(
        f'contig{size}', torch.arange(vocab, device=device, dtype=torch.long) // size
    )


def permuted_tiling(vocab: int, size: int, seed: int, device: torch.device | str) -> Tiling:
    """Control: tiles of `size` rows chosen uniformly at random."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(vocab, generator=g)
    tile = torch.empty(vocab, dtype=torch.long)
    tile[perm] = torch.arange(vocab) // size
    return _make_tiling(f'random{size}', tile.to(device))


def kmeans_tiling(
    w: torch.Tensor, size: int, iters: int = 25, seed: int = 0, chunk: int = 16384
) -> Tiling:
    """Lloyd k-means on the rows of W with k = V/size clusters (FP32, GPU).

    The paper's favourable case: rows grouped by geometry rather than by token id.
    Cluster sizes are unequal; ``Tiling.sizes`` records them.
    """
    vocab = w.shape[0]
    k = max(1, vocab // size)
    x = w.float()
    g = torch.Generator(device='cpu').manual_seed(seed)
    centres = x[torch.randperm(vocab, generator=g)[:k].to(w.device)].clone()
    x_sq = (x * x).sum(1)
    assign = torch.empty(vocab, dtype=torch.long, device=w.device)
    for _ in range(iters):
        c_sq = (centres * centres).sum(1)
        for s in range(0, vocab, chunk):
            d = x_sq[s : s + chunk, None] - 2 * x[s : s + chunk] @ centres.T + c_sq[None]
            assign[s : s + chunk] = d.argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()
        nonempty = counts > 0
        centres[nonempty] = sums[nonempty] / counts[nonempty, None]
    return _make_tiling(f'kmeans{size}', assign)


# ---------------------------------------------------------------------------
# Segment reductions over tiles
# ---------------------------------------------------------------------------


def seg_max(z: torch.Tensor, tiling: Tiling) -> torch.Tensor:
    """[n, V] -> [n, C] per-tile maxima."""
    out = torch.full((z.shape[0], tiling.num_tiles), -math.inf, dtype=z.dtype, device=z.device)
    idx = tiling.tile_of_row.unsqueeze(0).expand_as(z)
    return out.scatter_reduce_(1, idx, z, reduce='amax', include_self=True)


def seg_logsumexp(z: torch.Tensor, tiling: Tiling) -> torch.Tensor:
    """[n, V] -> [n, C] per-tile log-sum-exp; a tile of all -inf rows gives -inf."""
    m = seg_max(z, tiling)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = torch.exp(z - m_safe[:, tiling.tile_of_row])
    s = torch.zeros_like(m).scatter_add_(1, tiling.tile_of_row.unsqueeze(0).expand_as(z), shifted)
    return torch.where(s > 0, m_safe + torch.log(s), torch.full_like(m, -math.inf))


# ---------------------------------------------------------------------------
# Tile geometry (paper Eq. geometry and groupbound) and transport bounds
# ---------------------------------------------------------------------------


@dataclass
class LowRankTile:
    """w_i = mu_c + Q u_i + e_i with Q orthonormal [D, r], e_i orthogonal to span(Q)."""

    basis: torch.Tensor  # [D, r]
    rho: torch.Tensor  # [C, r] max_i |u_ik| over the tile
    eta: torch.Tensor  # [C] max_i ||e_i||_2 over the tile


@dataclass
class TileGeometry:
    tiling: Tiling
    mu: torch.Tensor  # [C, D] centre (coordinate midrange: minimizes the l_inf radius)
    r_coord: torch.Tensor  # [C, D] max_i |w_ij - mu_cj|
    r_inf: torch.Tensor  # [C] max_j r_coord
    r_group: dict[int, torch.Tensor] = field(default_factory=dict)  # G -> [C, D/G]
    lowrank: dict[str, LowRankTile] = field(default_factory=dict)


def _scatter_rows(values: torch.Tensor, tiling: Tiling, reduce: str) -> torch.Tensor:
    init = -math.inf if reduce == 'amax' else math.inf
    out = torch.full(
        (tiling.num_tiles, values.shape[1]), init, dtype=values.dtype, device=values.device
    )
    idx = tiling.tile_of_row.unsqueeze(1).expand_as(values)
    return out.scatter_reduce_(0, idx, values, reduce=reduce, include_self=True)


def tile_geometry(
    w: torch.Tensor,
    tiling: Tiling,
    groups: tuple[int, ...] = (32, 128),
    bases: dict[str, torch.Tensor] | None = None,
    chunk: int = 32768,
) -> TileGeometry:
    """Centres, scalar/coordinate/grouped radii and low-rank tile radii for W (FP64)."""
    vocab, dim = w.shape
    mx = torch.full((tiling.num_tiles, dim), -math.inf, dtype=F64, device=w.device)
    mn = torch.full((tiling.num_tiles, dim), math.inf, dtype=F64, device=w.device)
    for s in range(0, vocab, chunk):
        rows = w[s : s + chunk].to(F64)
        sub = Tiling(tiling.name, tiling.tile_of_row[s : s + chunk], tiling.num_tiles, tiling.sizes)
        mx = torch.maximum(mx, _scatter_rows(rows, sub, 'amax'))
        mn = torch.minimum(mn, _scatter_rows(rows, sub, 'amin'))
    mu = (mx + mn) / 2
    r_coord = torch.maximum(mx - mu, mu - mn)
    geo = TileGeometry(tiling, mu, r_coord, r_coord.max(1).values)
    for g in groups:
        acc = torch.full((tiling.num_tiles, dim // g), -math.inf, dtype=F64, device=w.device)
        for s in range(0, vocab, chunk):
            t = tiling.tile_of_row[s : s + chunk]
            dev = w[s : s + chunk].to(F64) - mu[t]
            norms = dev.view(dev.shape[0], dim // g, g).norm(dim=2)
            sub = Tiling(tiling.name, t, tiling.num_tiles, tiling.sizes)
            acc = torch.maximum(acc, _scatter_rows(norms, sub, 'amax'))
        geo.r_group[g] = acc
    for name, q in (bases or {}).items():
        r = q.shape[1]
        rho = torch.full((tiling.num_tiles, r), -math.inf, dtype=F64, device=w.device)
        eta = torch.full((tiling.num_tiles, 1), -math.inf, dtype=F64, device=w.device)
        for s in range(0, vocab, chunk):
            t = tiling.tile_of_row[s : s + chunk]
            dev = w[s : s + chunk].to(F64) - mu[t]
            u = dev @ q
            e = (dev - u @ q.T).norm(dim=1, keepdim=True)
            sub = Tiling(tiling.name, t, tiling.num_tiles, tiling.sizes)
            rho = torch.maximum(rho, _scatter_rows(u.abs(), sub, 'amax'))
            eta = torch.maximum(eta, _scatter_rows(e, sub, 'amax'))
        geo.lowrank[name] = LowRankTile(q, rho, eta[:, 0])
    return geo


def tile_epsilons(geo: TileGeometry, delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Half-widths eps_c >= max_{i in c} |<w_i - mu_c, delta>| for every bound family.

    ``delta`` is [n, D] FP64: Delta for transport, or h itself for the static screen.
    Returns name -> [n, C].
    """
    n, dim = delta.shape
    out: dict[str, torch.Tensor] = {
        'scalar': geo.r_inf[None] * delta.abs().sum(1, keepdim=True),
        'coord': delta.abs() @ geo.r_coord.T,
    }
    for g, radii in geo.r_group.items():
        out[f'group{g}'] = delta.view(n, dim // g, g).norm(dim=2) @ radii.T
    for name, lr in geo.lowrank.items():
        proj = delta @ lr.basis
        resid = (delta - proj @ lr.basis.T).norm(dim=1, keepdim=True)
        out[f'lowrank_{name}'] = proj.abs() @ lr.rho.T + resid * lr.eta[None]
    return out


def top_basis(second_moment: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-`rank` eigenvectors [D, rank] of a symmetric PSD matrix (FP64)."""
    evals, evecs = torch.linalg.eigh(second_moment.to(F64))
    order = torch.argsort(evals, descending=True)[:rank]
    return evecs[:, order].contiguous()


def centred_second_moment(w: torch.Tensor, geo: TileGeometry, chunk: int = 32768) -> torch.Tensor:
    """sum_i (w_i - mu_c(i))(w_i - mu_c(i))^T in FP64."""
    dim = w.shape[1]
    acc = torch.zeros((dim, dim), dtype=F64, device=w.device)
    for s in range(0, w.shape[0], chunk):
        dev = w[s : s + chunk].to(F64) - geo.mu[geo.tiling.tile_of_row[s : s + chunk]]
        acc += dev.T @ dev
    return acc


# ---------------------------------------------------------------------------
# Partition and acceptance statistics (paper Theorem guard, Eq. unknown)
# ---------------------------------------------------------------------------


def unresolved_probability(
    log_p_x: torch.Tensor,
    log_q_x: torch.Tensor,
    log_r_plus: torch.Tensor,
    log_r_minus: torch.Tensor,
) -> torch.Tensor:
    """P_? = min(1, p_x r+/q_x) - min(1, p_x r-/q_x), with r+ = Z/Z^- and r- = Z/Z^+.

    Equivalent to Eq. (unknown) since w_x / (q_x Z^-) = (p_x / q_x)(Z / Z^-).
    """
    ratio = log_p_x - log_q_x
    hi = torch.clamp(ratio + log_r_plus, max=0).exp()
    lo = torch.clamp(ratio + log_r_minus, max=0).exp()
    return hi - lo


def expected_unresolved(
    log_p: torch.Tensor, log_q: torch.Tensor, log_r_plus: torch.Tensor, log_r_minus: torch.Tensor
) -> torch.Tensor:
    """E_{x ~ q}[P_?(x)] = sum_x min(q_x, p_x r+) - min(q_x, p_x r-) for [n, V] log-probs."""
    q = log_q.exp()
    hi = torch.minimum(q, (log_p + log_r_plus[:, None]).exp())
    lo = torch.minimum(q, (log_p + log_r_minus[:, None]).exp())
    return (hi - lo).sum(1)


# ---------------------------------------------------------------------------
# Quantized heads and self-evidence envelopes
# ---------------------------------------------------------------------------

BLOCK = 128  # block size for the blockwise and Hoelder envelopes
ROW_CHUNK = 16384


def fp16_up(x: torch.Tensor) -> torch.Tensor:
    """Smallest FP16 value >= x for positive FP64 x (a scale that never clips)."""
    s = x.clamp_min(2.0**-24).half()
    bumped = (s.view(torch.int16) + 1).view(torch.float16)
    return torch.where(s.double() < x, bumped, s)


def _blocks(x: torch.Tensor) -> torch.Tensor:
    """[n, d] -> [n, ceil(d/BLOCK), BLOCK], zero padded."""
    d = x.shape[1]
    pad = (-d) % BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.view(x.shape[0], -1, BLOCK)


@dataclass
class QuantHead:
    """A low-precision copy of W (or of W minus a fixed centre, or of a column subset).

    Codes are int8 (int8 and int4 values) or float8_e4m3fn with FP16 scales per row or
    per group; every dequantized value ``code * scale`` is exact in FP32 and FP64. The
    error metadata is exact FP64; a deployment stores it rounded up.
    """

    name: str
    codes: torch.Tensor  # [V, d]
    scale: torch.Tensor  # [V, d / group] FP16
    group: int
    bits: int
    centre: torch.Tensor | None  # [d] FP64; the head approximates w_i - centre
    e2: torch.Tensor  # [V] ||e_i||_2
    e2_block: torch.Tensor  # [V, nb] ||e_ig||_2 over BLOCK-wide column blocks
    einf_block: torch.Tensor  # [V, nb] max_j |e_ij| per block
    free_einf_block: torch.Tensor | None  # [V, nb] scale / 2 (integer RTN, no clipping)
    sq2: torch.Tensor  # [V] ||w_hat_i||_2, bounds sum_j |w_hat_ij h_j| / ||h||_2
    cache: torch.Tensor | None = None  # optional dense FP64 w_hat (CPU runs)

    def materialize(self) -> None:
        """Keep the dequantized FP64 head in memory (fast on CPU, 8 bytes/weight)."""
        self.cache = self.dequant(0, self.codes.shape[0])

    @property
    def weight_bytes_per_row(self) -> float:
        d = self.codes.shape[1]
        return d * self.bits / 8 + self.scale.shape[1] * 2

    def dequant(self, start: int, stop: int) -> torch.Tensor:
        q = self.codes[start:stop].to(F64)
        sc = self.scale[start:stop].to(F64)
        rows = stop - start
        return (q.view(rows, -1, self.group) * sc[..., None]).view(rows, -1)

    def z_hat(self, h: torch.Tensor) -> torch.Tensor:
        """Exact <w_hat_i, h> for all rows (FP64), plus <centre, h> if centred."""
        v = self.codes.shape[0]
        if self.cache is not None:
            out = h @ self.cache.T
            if self.centre is not None:
                out += (h @ self.centre)[:, None]
            return out
        out = torch.empty((h.shape[0], v), dtype=F64, device=h.device)
        for s in range(0, v, ROW_CHUNK):
            e = min(v, s + ROW_CHUNK)
            out[:, s:e] = h @ self.dequant(s, e).T
        if self.centre is not None:
            out += (h @ self.centre)[:, None]
        return out


def quantize(
    w: torch.Tensor,
    kind: str,
    group: int | None,
    name: str,
    centre: torch.Tensor | None = None,
) -> QuantHead:
    """Symmetric RTN int8/int4 or FP8 e4m3 quantization of W [V, d] (BF16).

    ``group=None`` means one scale per row. Scales are FP16 rounded up so that no code
    clips; for the integer kinds this gives |e_ij| <= scale / 2 exactly.
    """
    v, d = w.shape
    group = group or d
    if d % group:
        raise ValueError('group must divide the row length')
    qmax = {'int8': 127.0, 'int4': 7.0, 'fp8': 448.0}[kind]
    code_dtype = torch.float8_e4m3fn if kind == 'fp8' else torch.int8
    codes = torch.empty((v, d), dtype=code_dtype, device=w.device)
    scale = torch.empty((v, d // group), dtype=torch.float16, device=w.device)
    nb = (d + BLOCK - 1) // BLOCK
    e2 = torch.empty(v, dtype=F64, device=w.device)
    e2_block = torch.empty((v, nb), dtype=F64, device=w.device)
    einf_block = torch.empty((v, nb), dtype=F64, device=w.device)
    sq2 = torch.empty(v, dtype=F64, device=w.device)
    for s in range(0, v, ROW_CHUNK):
        e = min(v, s + ROW_CHUNK)
        x = w[s:e].to(F64)
        if centre is not None:
            x = x - centre
        g = x.view(e - s, d // group, group)
        sc = fp16_up(g.abs().amax(2) / qmax)
        scale[s:e] = sc
        ratio = g / sc.double()[..., None]
        if kind == 'fp8':
            q = ratio.float().to(torch.float8_e4m3fn)
            q64 = q.to(F64)
        else:
            q64 = torch.clamp(torch.round(ratio), -qmax, qmax)
            q = q64.to(torch.int8)
        codes[s:e] = q.view(e - s, d)
        w_hat = (q64 * sc.double()[..., None]).view(e - s, d)
        err = x - w_hat
        e2[s:e] = err.norm(dim=1)
        eb = _blocks(err)
        e2_block[s:e] = eb.norm(dim=2)
        einf_block[s:e] = eb.abs().amax(dim=2)
        sq2[s:e] = w_hat.norm(dim=1)
    free = None
    if kind != 'fp8' and group in (d, BLOCK):
        half = scale.double() / 2
        free = half.expand(v, nb).contiguous() if group == d else half
    bits = 4 if kind == 'int4' else 8
    return QuantHead(name, codes, scale, group, bits, centre, e2, e2_block, einf_block, free, sq2)


@dataclass
class ResidualHead:
    """Two planes: ``first`` approximates w, ``second`` quantizes w - first.

    Progressive precision as bit planes: a cascade reads ``first`` for every row and
    ``second`` only for rows the first plane could not exclude. The combined error
    metadata is ``second``'s (its input was the first plane's exact error).
    """

    name: str
    first: QuantHead
    second: QuantHead

    def envelope(self, h: torch.Tensor) -> Envelope:
        z_hat = self.first.z_hat(h) + self.second.z_hat(h)
        acc = h.norm(dim=1, keepdim=True) * (self.first.sq2 + self.second.sq2)[None]
        return Envelope(z_hat, _quant_terms(self.second, h), acc)


def residual_head(w: torch.Tensor, first: QuantHead, kind: str, group: int, name: str):
    """Quantize the first plane's error (FP64, exact) as a second plane."""
    v = w.shape[0]
    resid = torch.empty(w.shape, dtype=F64, device=w.device)
    for s in range(0, v, ROW_CHUNK):
        e = min(v, s + ROW_CHUNK)
        resid[s:e] = w[s:e].to(F64) - first.dequant(s, e)
    second = quantize(resid, kind, group, name + '_plane2')
    del resid
    return ResidualHead(name, first, second)


@dataclass
class Envelope:
    """Self-evidence for inputs h: ``quant`` bounds |z - z_hat| (real arithmetic);
    ``acc_scale`` times an accumulation gamma bounds the online floating-point error of
    computing z_hat."""

    z_hat: torch.Tensor  # [n, V]
    quant: dict[str, torch.Tensor]  # variant -> [n, V]
    acc_scale: torch.Tensor  # [n, V]


def block_norms(h: torch.Tensor, p: int) -> torch.Tensor:
    b = _blocks(h)
    return b.norm(dim=2) if p == 2 else b.abs().sum(2)


def _quant_terms(qh: QuantHead, h: torch.Tensor) -> dict[str, torch.Tensor]:
    hn2 = h.norm(dim=1, keepdim=True)
    terms = {
        'row_cs': hn2 * qh.e2[None],
        'block_l2': block_norms(h, 2) @ qh.e2_block.T,
        'hoelder_block': block_norms(h, 1) @ qh.einf_block.T,
    }
    if qh.free_einf_block is not None:
        terms['hoelder_scale'] = block_norms(h, 1) @ qh.free_einf_block.T
    return terms


def quant_envelope(qh: QuantHead, h: torch.Tensor) -> Envelope:
    """Envelope of a whole-head quantization for h [n, D] (FP64)."""
    return Envelope(qh.z_hat(h), _quant_terms(qh, h), h.norm(dim=1, keepdim=True) * qh.sq2[None])


@dataclass
class OutlierHead:
    """Columns ``exact_cols`` evaluated exactly from BF16 W; the rest quantized."""

    name: str
    exact_cols: torch.Tensor  # [k]
    rest_cols: torch.Tensor  # [D - k]
    w_exact: torch.Tensor  # [V, k] BF16
    rest: QuantHead

    def envelope(self, h: torch.Tensor) -> Envelope:
        h_o, h_r = h[:, self.exact_cols], h[:, self.rest_cols]
        z_hat = h_o @ self.w_exact.to(F64).T + self.rest.z_hat(h_r)
        wn = self.w_exact.to(F64).norm(dim=1)
        acc = (
            h_o.norm(dim=1, keepdim=True) * wn[None]
            + h_r.norm(dim=1, keepdim=True) * self.rest.sq2[None]
        )
        return Envelope(z_hat, _quant_terms(self.rest, h_r), acc)


def outlier_head(w: torch.Tensor, cols: torch.Tensor, kind: str, group: int | None, name: str):
    mask = torch.ones(w.shape[1], dtype=torch.bool, device=w.device)
    mask[cols] = False
    rest_cols = torch.nonzero(mask)[:, 0]
    rest = quantize(w[:, rest_cols].contiguous(), kind, group, name + '_rest')
    return OutlierHead(name, cols, rest_cols, w[:, cols].contiguous(), rest)


@dataclass
class RotatedHead:
    """z_i = <a_i, y> + <w_i, h_perp> with y = P^T h (FP32), h_perp = h - P y exactly.

    ``a = W P`` is stored in BF16 (rounding error ``da2``); the residual h_perp goes
    through the quantized full head as two BF16 columns (hi + lo), whose representation
    error ``delta`` is charged with ``wn2 = ||w_i||_2``.
    """

    name: str
    basis32: torch.Tensor  # [D, r] FP32 values, used as exact reals
    a_bf16: torch.Tensor  # [V, r]
    da2: torch.Tensor  # [V]
    wn2: torch.Tensor  # [V]
    rest: QuantHead

    def envelope(self, h: torch.Tensor) -> Envelope:
        p32 = self.basis32
        y32 = h.float() @ p32
        hp32 = h.float() - y32 @ p32.T
        hi = hp32.bfloat16()
        lo = (hp32 - hi.float()).bfloat16()
        h_perp = h - y32.double() @ p32.double().T  # exact residual for this y
        rep = hi.double() + lo.double()
        delta = (h_perp - rep).norm(dim=1, keepdim=True)
        y = y32.double()
        yn = y.norm(dim=1, keepdim=True)
        z_hat = y @ self.a_bf16.to(F64).T + self.rest.z_hat(rep)
        extra = yn * self.da2[None] + delta * self.wn2[None]
        quant = {k: v + extra for k, v in _quant_terms(self.rest, rep).items()}
        acc = (
            yn * self.a_bf16.to(F64).norm(dim=1)[None]
            + (hi.double().norm(dim=1, keepdim=True) + lo.double().norm(dim=1, keepdim=True))
            * self.rest.sq2[None]
        )
        return Envelope(z_hat, quant, acc)


def rotated_head(w: torch.Tensor, basis: torch.Tensor, rest: QuantHead, name: str):
    p32 = basis.float()
    a = w.to(F64) @ p32.double()
    a_bf16 = a.bfloat16()
    return RotatedHead(
        name,
        p32,
        a_bf16,
        (a - a_bf16.double()).norm(dim=1),
        w.to(F64).norm(dim=1),
        rest,
    )


def accumulation_gamma(dim: int, model: str) -> float:
    """gamma for |fl(sum) - sum| <= gamma * sum |terms| (theory notes, 2026-09-30).

    ``tensor_core``: conservative alignment-and-truncation model, D * 2^-22 * 1.001.
    ``fp32_tree``: IEEE FP32 on CUDA cores, tree depth D/128 + log2(128) (+1 for the
    scale multiply), gamma = (d+1)u / (1 - (d+1)u) with u = 2^-24.
    """
    if model == 'tensor_core':
        return dim * 2.0**-22 * 1.001
    if model == 'fp32_tree':
        depth = math.ceil(dim / 128) + 7 + 1
        u = 2.0**-24
        return (depth + 1) * u / (1 - (depth + 1) * u)
    raise ValueError(model)


# ---------------------------------------------------------------------------
# Certified argmax and Gumbel-max candidate sets
# ---------------------------------------------------------------------------


def candidate_mask(centre: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Rows whose upper bound reaches the best lower bound (non-strict keeps ties)."""
    best_lower = (centre - half).max(dim=1, keepdim=True).values
    return centre + half >= best_lower


def gumbel_noise(shape: tuple[int, int], seed: int, device: torch.device | str) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(shape, generator=g, dtype=F64, device=device)
    u = u.clamp(min=torch.finfo(F64).tiny)
    return -torch.log(-torch.log(u))
