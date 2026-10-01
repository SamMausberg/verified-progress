"""Shared loader for real head inputs captured by the geometry workstream.

The capture patch writes a stream of pickled records per server process; each
plain-decode record holds one decode step: the BF16 head inputs of the whole
batch (``uint16`` bit patterns), the engine's argmax and top-2 logits.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_PLAIN = '~/vp-data/geometry/plain4b/heads'
DEFAULT_MTP = '~/vp-data/geometry/mtp4b/heads'


def iter_records(heads_dir: str | Path, kind: str) -> Iterator[dict[str, Any]]:
    for path in sorted(Path(heads_dir).expanduser().glob(f'{kind}_*.pkl')):
        with path.open('rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def as_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def plain_decode_steps(
    heads_dir: str | Path | None = None, limit_rows: int | None = None
) -> Iterator[tuple[torch.Tensor, np.ndarray]]:
    """Yield ``(hidden [bs, D] bf16, engine_argmax [bs])`` per decode step."""
    d = heads_dir or os.environ.get('VP_PLAIN_DECODE_DIR', DEFAULT_PLAIN)
    rows = 0
    for rec in iter_records(d, 'plain_decode'):
        if rec['forward_mode'] != 'DECODE':
            continue
        yield as_bf16(rec['hidden']), np.asarray(rec['engine_argmax'])
        rows += len(rec['rids'])
        if limit_rows is not None and rows >= limit_rows:
            return
