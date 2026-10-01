"""Build the low-precision copy of the LM head and its exact error metadata.

The head of Qwen3.5-4B is tied to the input embedding
(``model.language_model.embed_tokens.weight``, 248,320 x 2,560 BF16). This
module reads it from the pinned checkpoint, quantizes each row to symmetric
int8 (``w_i ~ s_i * q_i``), and computes, in FP64 from the values actually
stored, squared norms of the error ``e_i = w_i - s_i q_i``, of ``q_i`` and of
``w_i`` on groups of :data:`BASE_GROUP` coordinates. Every FP64 value is an
upper bound of the exact quantity (see :mod:`certified_head.bounds`).

The error itself is computed exactly: ``s_i`` has 24 significant bits and
``q_ij`` at most 7, so ``s_i q_ij`` is exact in FP64, and the subtraction from
the BF16 weight is verified to be exact with an error-free TwoSum check.

Results are cached under ``~/vp-data/kernel/`` (override with
``VP_KERNEL_CACHE``), keyed by the SHA-256 of the BF16 head bytes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from certified_head.bounds import sumsq_upper

MODEL_ID = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
HEAD_TENSOR = 'model.language_model.embed_tokens.weight'
BASE_GROUP = 128
"""Coordinates per stored metadata group; coarser groups are derived from it."""

SCHEME = 'int8-sym-row-v2'


def cache_dir() -> Path:
    return Path(os.environ.get('VP_KERNEL_CACHE', '~/vp-data/kernel')).expanduser()


def load_head_weight(model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> torch.Tensor:
    """Read the tied LM head (BF16, CPU) from the pinned local checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(
        model_id, 'model.safetensors.index.json', revision=revision, local_files_only=True
    )
    import json

    with open(index_path) as f:
        shard = json.load(f)['weight_map'][HEAD_TENSOR]
    path = hf_hub_download(model_id, shard, revision=revision, local_files_only=True)
    with safe_open(path, 'pt') as st:
        w = st.get_tensor(HEAD_TENSOR)
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError(f'unexpected head tensor {w.dtype} {tuple(w.shape)}')
    return w.contiguous()


def head_sha256(w: torch.Tensor, chunk_rows: int = 16384) -> str:
    """SHA-256 of a BF16 head's bytes in row-major order. The rows are read in
    chunks, so a device tensor reaches the host one chunk at a time."""
    digest = hashlib.sha256()
    bits = w.contiguous().view(torch.int16)
    for r0 in range(0, bits.shape[0], chunk_rows):
        digest.update(bits[r0 : r0 + chunk_rows].cpu().numpy().tobytes())
    return digest.hexdigest()


def quantize_int8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: ``s_i = RN_fp32(max_j |w_ij| / 127)``, ``q = round(w / s)``.

    Any positive FP32 scale is admissible for correctness because the error is
    measured afterwards from the stored values; this choice only affects how
    tight the envelope is. All-zero rows get ``s_i = 1`` and ``q_i = 0``.
    """
    w64 = w.to(torch.float64)
    amax = w64.abs().amax(dim=1)
    scale = (amax / 127.0).to(torch.float32)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(w64 / scale.to(torch.float64)[:, None]).clamp_(-127, 127)
    return q.to(torch.int8), scale


def duplicate_representatives(w: torch.Tensor) -> np.ndarray:
    """For each row, the smallest index of a row with identical bits (int32)."""
    a = w.contiguous().view(torch.int16).numpy()
    rows = np.ascontiguousarray(a).view(np.dtype((np.void, a.shape[1] * a.dtype.itemsize))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    return first[inverse].astype(np.int32)


def _two_sum_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rounding error of ``fl(a + b)`` in FP64 (Knuth's TwoSum; exact)."""
    s = a + b
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def error_metadata(
    w: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, chunk_rows: int = 16384
) -> dict[str, np.ndarray]:
    """FP64 upper bounds of squared group norms of ``e``, ``q`` and ``w``.

    Returns arrays of shape ``[V, K / BASE_GROUP]`` plus exact per-row maxima
    ``err_linf`` (FP64) for diagnostics.
    """
    v, k = w.shape
    if k % BASE_GROUP:
        raise ValueError(f'hidden size {k} not divisible by {BASE_GROUP}')
    g = k // BASE_GROUP
    out = {
        'err_sumsq': np.empty((v, g), np.float64),
        'q_sumsq': np.empty((v, g), np.float64),
        'w_sumsq': np.empty((v, g), np.float64),
        'err_linf': np.empty((v,), np.float64),
    }
    for r0 in range(0, v, chunk_rows):
        r1 = min(v, r0 + chunk_rows)
        w64 = w[r0:r1].to(torch.float64)
        sq = scale[r0:r1].to(torch.float64)[:, None] * q[r0:r1].to(torch.float64)
        e = w64 - sq
        if torch.count_nonzero(_two_sum_error(w64, -sq)).item() != 0:
            raise ArithmeticError(f'inexact FP64 quantization error in rows {r0}:{r1}')
        n = r1 - r0
        out['err_sumsq'][r0:r1] = e.square().view(n, g, BASE_GROUP).sum(-1).numpy()
        out['q_sumsq'][r0:r1] = (
            q[r0:r1].to(torch.float64).square().view(n, g, BASE_GROUP).sum(-1).numpy()
        )
        out['w_sumsq'][r0:r1] = w64.square().view(n, g, BASE_GROUP).sum(-1).numpy()
        out['err_linf'][r0:r1] = e.abs().amax(dim=1).numpy()
    # q squares are small integers summed exactly; the others get the FP64 bound.
    out['err_sumsq'] = sumsq_upper(out['err_sumsq'], BASE_GROUP)
    out['w_sumsq'] = sumsq_upper(out['w_sumsq'], BASE_GROUP)
    return out


@dataclass(frozen=True)
class QuantizedHead:
    """Int8 head plus the metadata needed to build envelopes (all on CPU)."""

    q: torch.Tensor
    """int8 ``[V, K]``."""
    scale: torch.Tensor
    """FP32 ``[V]``."""
    err_sumsq: np.ndarray
    q_sumsq: np.ndarray
    w_sumsq: np.ndarray
    err_linf: np.ndarray
    dup_rep: np.ndarray
    """int32 ``[V]``: smallest index of a bitwise-identical row."""
    info: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        v, k = self.q.shape
        return int(v), int(k)


def build_quantized_head(w: torch.Tensor, info: dict[str, Any] | None = None) -> QuantizedHead:
    """Quantize ``w`` and compute its envelope metadata. ``info`` records the
    SHA-256 of ``w``, which ``CertifiedHead.from_quantized`` checks against the
    weight it is handed. A ``head_sha256`` supplied in ``info`` must equal it:
    metadata copied from another head would otherwise label these codes and
    bounds with that head's digest and pass the check for the wrong weight."""
    info = dict(info or {})
    digest = head_sha256(w)
    supplied = info.get('head_sha256')
    if supplied is not None and supplied != digest:
        raise ValueError('the supplied head_sha256 is not the digest of the weight being quantized')
    info['head_sha256'] = digest
    q, scale = quantize_int8_rows(w)
    meta = error_metadata(w, q, scale)
    return QuantizedHead(
        q=q,
        scale=scale,
        err_sumsq=meta['err_sumsq'],
        q_sumsq=meta['q_sumsq'],
        w_sumsq=meta['w_sumsq'],
        err_linf=meta['err_linf'],
        dup_rep=duplicate_representatives(w),
        info=dict(info, scheme=SCHEME, base_group=BASE_GROUP),
    )


def load_or_build(
    model_id: str = MODEL_ID, revision: str = MODEL_REVISION, rebuild: bool = False
) -> tuple[torch.Tensor, QuantizedHead]:
    """Return ``(w_bf16, quantized)`` for the pinned checkpoint, using the cache."""
    w = load_head_weight(model_id, revision)
    digest = head_sha256(w)
    path = cache_dir() / f'{model_id.replace("/", "--")}-{revision[:12]}-{SCHEME}.pt'
    if path.exists() and not rebuild:
        blob = torch.load(path, weights_only=False)
        if blob['info'].get('head_sha256') == digest:
            return w, QuantizedHead(**blob)
    info = {'model_id': model_id, 'revision': revision, 'head_sha256': digest}
    qh = build_quantized_head(w, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(qh.__dict__, tmp)
    tmp.replace(path)
    return w, qh
