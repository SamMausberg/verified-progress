"""Certified low-precision LM head: exact greedy decisions from an int8 head.

See :mod:`certified_head.head` for the API and :mod:`certified_head.bounds`
for the error model and the exactness contract. The GPU classes are imported
lazily so that :mod:`certified_head.bounds` works without torch.
"""

from typing import Any

__all__ = ['CertifiedHead', 'GemvConfig', 'HeadStats']


def __getattr__(name: str) -> Any:
    if name in __all__:
        from certified_head import head

        return getattr(head, name)
    raise AttributeError(name)
