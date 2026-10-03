"""The runtime a hold or the GEMM probe runs on, for summarize.py to check.

The interpreter, torch and its CUDA, the version of every serving package SETUP.md lists, and
the sha256 of sgl-kernel's compiled libraries (the FP8 routes and the missing sm_90a code are
properties of those binaries, which a rebuild can change without changing the version).

    python experiments/speed_bytes/runtime_record.py    # prints "runtime <JSON>"
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# SETUP.md, "Key versions" (the serving packages; torch is recorded as torch.__version__).
PACKAGES = (
    'triton',
    'flashinfer-python',
    'sglang-kernel',
    'flash-attn-4',
    'transformers',
    'xgrammar',
)


def sgl_kernel_libraries() -> str:
    """One sha256 over every compiled library of the installed sgl_kernel package, by path."""
    spec = importlib.util.find_spec('sgl_kernel')
    if spec is None or spec.origin is None:
        raise SystemExit('sgl_kernel is not installed')
    root = Path(spec.origin).parent
    h = hashlib.sha256()
    for lib in sorted(root.rglob('*.so')):
        h.update(
            f'{lib.relative_to(root)} {hashlib.sha256(lib.read_bytes()).hexdigest()}\n'.encode()
        )
    return h.hexdigest()


def record() -> dict[str, Any]:
    import torch

    return {
        'python': sys.executable,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'packages': {name: importlib.metadata.version(name) for name in PACKAGES},
        'sgl_kernel_libraries': sgl_kernel_libraries(),
    }


if __name__ == '__main__':
    print('runtime', json.dumps(record(), sort_keys=True))
