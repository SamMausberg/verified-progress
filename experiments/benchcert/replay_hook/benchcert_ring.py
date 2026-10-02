"""Device ring log of the certified MTP verify (hold h7, ring mode): no per-step readback.

With BENCHCERT_RING=<dir>, sitecustomize.py installs these wrappers in the SGLang server.
After every target verify replay (outside the graph, once it has been launched), the
wrapper of the glue's `after_replay` issues asynchronous device copies, on the stream the
replay ran on, into ring tensors allocated once on the device:

- the gate the conditional node read (`gate`) and the device row count (`valid`);
- which fallback ran: `_any`, `_any_cols` (column fallback), `_any_dense` (dense merge);
- for the first ROWS rows, the head's final id, status bits and candidate count, and the
  first CANDS candidates with their refined bounds (`rlo`, `rhi`). CANDS is the column
  fallback's cap, so a row the column fallback serves is logged in full.

The `eagle_sample` wrapper copies the verify's predicted ids and accept lengths (bonus
included) into the same slot. The host keeps, per step, plain Python values only: step,
time, rows, batch size, its intended gate (`forward_batch.certified_path`) and each
request's slot, prompt length, output length and last four output ids.

A daemon thread writes the steps not yet written once no verify replay has run for IDLE_S
seconds, which happens only between points: it waits on an event recorded after the last
step's copies, copies the slices to the host on its own stream, and writes
<dir>/ring-<first>-<last>.npz with <dir>/ring-<first>-<last>.jsonl (host records). No
device value is read on the serving thread. A point with more than STEPS verify replays
would overwrite its own start; the thread records that in <dir>/deviations.jsonl, as it
does any error.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

STEPS = 8192  # verify replays held between writes (a c = 64 point has about 1,800)
ROWS = 64  # the certified verify's largest batch (MAX_ROWS)
CANDS = 64  # certified_head.head.COLS_CAP
PRED = 256  # predicted ids per verify (rows)
REQS = 128  # accept lengths per verify (requests)
IDLE_S = 1.0


class Ring:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.lock = threading.Lock()
        self.k = 0  # verify replays seen
        self.written = 0  # steps written to disk
        self.host: dict[int, dict[str, Any]] = {}
        self.t_last = 0.0
        self.last_event: Any = None
        self.dev: dict[str, Any] | None = None
        self.device: Any = None
        self.thread: threading.Thread | None = None

    def deviation(self, record: dict[str, Any]) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / 'deviations.jsonl').open('a') as handle:
            handle.write(json.dumps({'t': time.time(), **record}) + '\n')

    def alloc(self, torch: Any, device: Any) -> None:
        def z(*shape: int, dtype: Any) -> Any:
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.device = device
        self.dev = {
            'ids': z(STEPS, ROWS, dtype=torch.int64),
            'status': z(STEPS, ROWS, dtype=torch.int32),
            'count': z(STEPS, ROWS, dtype=torch.int32),
            'cand': z(STEPS, ROWS, CANDS, dtype=torch.int32),
            'rlo': z(STEPS, ROWS, CANDS, dtype=torch.float32),
            'rhi': z(STEPS, ROWS, CANDS, dtype=torch.float32),
            'gate': z(STEPS, dtype=torch.bool),
            'valid': z(STEPS, dtype=torch.int32),
            'any': z(STEPS, dtype=torch.bool),
            'any_cols': z(STEPS, dtype=torch.bool),
            'any_dense': z(STEPS, dtype=torch.bool),
            'predict': z(STEPS, PRED, dtype=torch.int32),
            'accept': z(STEPS, REQS, dtype=torch.int32),
        }

    def record_verify(self, torch: Any, glue: Any, forward_batch: Any) -> None:
        """Device copies of one verify replay's head state (no readback)."""
        st = glue._state.get('verify')
        heads = glue._heads
        if st is None or heads is None:
            return
        head = heads.get('verify').head
        if self.dev is None:
            self.alloc(torch, st.gate.device)
        assert self.dev is not None
        with self.lock:
            k = self.k
            self.k += 1
        s = k % STEPS
        rows = int(forward_batch.input_ids.shape[0])  # a host shape, not a device read
        r = min(rows, ROWS)
        d = self.dev
        d['gate'][s].copy_(st.gate)
        d['valid'][s].copy_(st.valid)
        d['any'][s].copy_(head._any)
        d['any_cols'][s].copy_(head._any_cols)
        d['any_dense'][s].copy_(head._any_dense)
        d['ids'][s, :r].copy_(head._ids[:r])
        d['status'][s, :r].copy_(head._status[:r])
        d['count'][s, :r].copy_(head._count[:r])
        d['cand'][s, :r].copy_(head._cand[:r, :CANDS])
        d['rlo'][s, :r].copy_(head._rlo[:r, :CANDS])
        d['rhi'][s, :r].copy_(head._rhi[:r, :CANDS])
        event = torch.cuda.Event()
        event.record()
        record = {
            'k': k,
            't': time.time(),
            'rows': rows,
            'batch_size': int(forward_batch.batch_size),
            'host_gate': getattr(forward_batch, 'certified_path', None) == 'verify',
        }
        with self.lock:
            self.host[k] = record
            self.last_event = event
            self.t_last = time.monotonic()

    def record_sample(self, torch: Any, batch: Any, result: Any) -> None:
        """Device copies of what the verify committed; the host's view of each request."""
        if self.dev is None:
            return
        predict, accept_lens = result[0], result[1]
        with self.lock:
            k = self.k - 1
        s = k % STEPS
        n = min(int(predict.shape[0]), PRED)
        m = min(int(accept_lens.shape[0]), REQS)
        if n:
            self.dev['predict'][s, :n].copy_(predict[:n].to(torch.int32))
        if m:
            self.dev['accept'][s, :m].copy_(accept_lens[:m].to(torch.int32))
        reqs = list(getattr(batch, 'reqs', None) or [])
        extra: dict[str, Any] = {}
        fields = {
            'slots': lambda: [getattr(r.kv, 'req_pool_idx', None) for r in reqs],
            'prompt_lens': lambda: [len(r.origin_input_ids) for r in reqs],
            'output_lens': lambda: [len(r.output_ids) for r in reqs],
            'last_output_ids': lambda: [list(r.output_ids[-4:]) for r in reqs],
        }
        for key, get in fields.items():
            try:
                extra[key] = get()
            except Exception as exc:  # a debug log must not stop the server
                extra[f'{key}_error'] = repr(exc)
        event = torch.cuda.Event()
        event.record()
        with self.lock:
            self.host.setdefault(k, {'k': k}).update(extra, predict_n=n, accept_n=m)
            self.last_event = event
            self.t_last = time.monotonic()

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name='benchcert-ring', daemon=True)
            self.thread.start()

    def _run(self) -> None:
        import numpy as np
        import torch

        stream = None
        while True:
            time.sleep(0.2)
            with self.lock:
                end, start, event = self.k, self.written, self.last_event
                idle = time.monotonic() - self.t_last
            if end == start or event is None or idle < IDLE_S or self.dev is None:
                continue
            try:
                event.synchronize()
                if end - start > STEPS:
                    self.deviation({'overflow': [start, end]})
                    start = end - STEPS
                if stream is None:
                    stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    index = torch.tensor([i % STEPS for i in range(start, end)], device=self.device)
                    arrays = {
                        name: t.index_select(0, index).cpu().numpy() for name, t in self.dev.items()
                    }
                stem = self.out / f'ring-{start:07d}-{end - 1:07d}'
                np.savez(stem.with_suffix('.npz'), steps=np.arange(start, end), **arrays)
                with self.lock:
                    records = [self.host.pop(i, {'k': i}) for i in range(start, end)]
                with stem.with_suffix('.jsonl').open('w') as handle:
                    for record in records:
                        handle.write(json.dumps(record) + '\n')
                with self.lock:
                    self.written = end
            except Exception as exc:
                self.deviation({'write_error': repr(exc), 'steps': [start, end]})
                with self.lock:
                    self.written = end


RING: Ring | None = None


def ring(out: str) -> Ring:
    global RING
    if RING is None:
        RING = Ring(Path(out))
        RING.out.mkdir(parents=True, exist_ok=True)
    return RING


def patch_glue(glue: Any, out: str) -> None:
    import torch

    log = ring(out)
    original = glue.after_replay

    def after_replay(model_runner: Any, forward_batch: Any, logits_output: Any) -> None:
        original(model_runner, forward_batch, logits_output)
        if model_runner.is_draft_worker or not forward_batch.forward_mode.is_target_verify():
            return
        try:
            log.record_verify(torch, glue, forward_batch)
        except Exception as exc:  # a debug log must not stop the server
            log.deviation({'verify_error': repr(exc)})

    glue.after_replay = after_replay
    log.start()


def patch_sample(module: Any, out: str) -> None:
    import torch

    log = ring(out)
    original = module.eagle_sample

    def eagle_sample(verify_input: Any, batch: Any, logits_output: Any, *args: Any, **kwargs: Any):
        result = original(verify_input, batch, logits_output, *args, **kwargs)
        try:
            log.record_sample(torch, batch, result)
        except Exception as exc:  # a debug log must not stop the server
            log.deviation({'sample_error': repr(exc)})
        return result

    module.eagle_sample = eagle_sample
