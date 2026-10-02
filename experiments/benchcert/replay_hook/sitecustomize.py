"""Per-replay log of the certified head in an SGLang server (debug, untimed).

Imported at interpreter start-up when this directory is on PYTHONPATH (control_waves.py's
`certlog` variant). With BENCHCERT_REPLAY_LOG set, it wraps the certified-head glue's
`after_replay` once `sglang.srt.layers.certified_head` is imported, and appends one JSON
line per target verify replay: the gated path, rows, request slots, sequence lengths,
the verify input ids (last committed token and drafts), the certified ids, the stock
logits' top 5 per row (valid in check mode, which computes the stock head too) and every
path's gate. Nothing in the engine changes; without the variable this does nothing.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sys
import threading
from typing import Any

TARGET = 'sglang.srt.layers.certified_head'


def _patch(glue: Any) -> None:
    import torch

    out = os.environ['BENCHCERT_REPLAY_LOG']
    lock, counter = threading.Lock(), [0]
    original = glue.after_replay

    def after_replay(model_runner: Any, forward_batch: Any, logits_output: Any) -> None:
        original(model_runner, forward_batch, logits_output)
        if model_runner.is_draft_worker or not forward_batch.forward_mode.is_target_verify():
            return
        record: dict[str, Any] = {'n': counter[0]}
        counter[0] += 1
        try:
            rows = int(forward_batch.input_ids.shape[0])
            ids = getattr(logits_output, 'certified_ids', None)
            top = torch.topk(logits_output.next_token_logits[:rows].float(), 5, dim=-1)
            record.update(
                rows=rows,
                batch_size=int(forward_batch.batch_size),
                path=getattr(forward_batch, 'certified_path', None),
                gates={p: bool(st.gate.item()) for p, st in glue._state.items()},
                req_pool_indices=forward_batch.req_pool_indices.tolist(),
                seq_lens=forward_batch.seq_lens.tolist(),
                input_ids=forward_batch.input_ids.tolist(),
                certified_ids=ids.tolist() if ids is not None else None,
                top5_ids=top.indices.tolist(),
                top5_logits=[[round(v, 4) for v in row] for row in top.values.tolist()],
            )
        except Exception as exc:  # a debug log must not stop the server
            record['error'] = repr(exc)
        with lock, open(out, 'a') as handle:
            handle.write(json.dumps(record) + '\n')

    glue.after_replay = after_replay


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                exec_module = spec.loader.exec_module

                def run(module: Any, _exec: Any = exec_module) -> None:
                    _exec(module)
                    _patch(module)

                spec.loader.exec_module = run  # type: ignore[method-assign]
                return spec
        return None


if os.environ.get('BENCHCERT_REPLAY_LOG'):
    sys.meta_path.insert(0, _Finder())
