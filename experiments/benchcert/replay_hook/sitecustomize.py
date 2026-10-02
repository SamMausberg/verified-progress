"""Per-replay log of the certified head in an SGLang server (debug, untimed).

Imported at interpreter start-up when this directory is on PYTHONPATH (control_waves.py's
`certlog` variant). With BENCHCERT_REPLAY_LOG set, it wraps the certified-head glue's
`after_replay` once `sglang.srt.layers.certified_head` is imported, and appends one JSON
line per target verify replay: the time, the gated path, rows, request slots, sequence
lengths, each row's position, the verify input ids (last committed token and drafts),
the certified ids, the stock logits' top 5 per row (valid in check mode, which computes
the stock head too) and every path's gate. Each row's position and input id say which
token a request's verify read at that position. It also wraps the verify's `eagle_sample`
(as imported by `sglang.srt.speculative.eagle_worker_common`) and logs what each verify
committed: the predicted ids, accept lengths (bonus included) and accept index, with the
host's view of each request (slot, prompt length, output length, last output ids).
Nothing in the engine changes; without the variable this does nothing.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sys
import threading
import time
from typing import Any

TARGET = 'sglang.srt.layers.certified_head'
SAMPLE_TARGET = 'sglang.srt.speculative.eagle_worker_common'
_LOCK = threading.Lock()
_COUNTER = [0]


def _write(record: dict[str, Any]) -> None:
    with _LOCK, open(os.environ['BENCHCERT_REPLAY_LOG'], 'a') as handle:
        handle.write(json.dumps(record) + '\n')


def _patch(glue: Any) -> None:
    import torch

    original = glue.after_replay

    def after_replay(model_runner: Any, forward_batch: Any, logits_output: Any) -> None:
        original(model_runner, forward_batch, logits_output)
        if model_runner.is_draft_worker or not forward_batch.forward_mode.is_target_verify():
            return
        record: dict[str, Any] = {'kind': 'replay', 'n': _COUNTER[0], 't': time.time()}
        _COUNTER[0] += 1
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
        try:
            record['positions'] = forward_batch.positions[
                : int(forward_batch.input_ids.shape[0])
            ].tolist()
        except Exception as exc:
            record['positions_error'] = repr(exc)
        _write(record)

    glue.after_replay = after_replay


def _patch_sample(module: Any) -> None:
    original = module.eagle_sample

    def eagle_sample(verify_input: Any, batch: Any, logits_output: Any, *args: Any, **kwargs: Any):
        result = original(verify_input, batch, logits_output, *args, **kwargs)
        record: dict[str, Any] = {'kind': 'sample', 'n': _COUNTER[0], 't': time.time()}
        reqs = list(getattr(batch, 'reqs', None) or [])
        fields = {
            'req_pool_indices': lambda: [getattr(r.kv, 'req_pool_idx', None) for r in reqs],
            'prompt_lens': lambda: [len(r.origin_input_ids) for r in reqs],
            'output_lens': lambda: [len(r.output_ids) for r in reqs],
            'last_output_ids': lambda: [list(r.output_ids[-4:]) for r in reqs],
            'predict': lambda: result[0].tolist(),
            'accept_lens': lambda: result[1].tolist(),
            'accept_index': lambda: result[2].tolist(),
        }
        # One guard per field: a field this engine lacks must not drop the others.
        for key, get in fields.items():
            try:
                record[key] = get()
            except Exception as exc:  # a debug log must not stop the server
                record[f'{key}_error'] = repr(exc)
        _write(record)
        return result

    module.eagle_sample = eagle_sample


_PATCHES = {TARGET: _patch, SAMPLE_TARGET: _patch_sample}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name not in _PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                exec_module = spec.loader.exec_module

                def run(module: Any, _exec: Any = exec_module, _name: str = name) -> None:
                    _exec(module)
                    try:
                        _PATCHES[_name](module)
                    except Exception as exc:  # never break the engine's import
                        _write({'kind': 'patch_error', 'module': _name, 'error': repr(exc)})

                spec.loader.exec_module = run  # type: ignore[method-assign]
                return spec
        return None


if os.environ.get('BENCHCERT_REPLAY_LOG'):
    sys.meta_path.insert(0, _Finder())
