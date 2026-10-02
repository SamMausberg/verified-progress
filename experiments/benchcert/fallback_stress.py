"""Stress test of the certified MTP verify head and its fallbacks, without the scheduler.

    python -m experiments.benchcert.fallback_stress context --point DIR --out OUT
    python -m experiments.benchcert.fallback_stress start --out OUT   # inside gpu_startup_lock.sh
    python -m experiments.benchcert.fallback_stress fetch --out OUT [--contexts C.jsonl]
    python -m experiments.benchcert.fallback_stress stop --out OUT
    python -m experiments.benchcert.fallback_stress stress --out OUT --replay REPLAY --seconds S

The question: can the certified verify head, as the engine captures and replays it,
return a token other than the stock head's for one row of a batch? The drain reruns
committed a wrong token (1756, 3.8 nats below the top) at 579ae7ce's position 439, a
three-way near tie, while the certified verify ran (at most 64 rows). This test builds
the head as the engine does (``certified_head.engine.EngineHeads`` from the BF16 head:
verify plus the MTP draft and draft-extend siblings, column fallback, conservative
model) and captures one CUDA graph per verify size in one shared memory pool, the way
``sglang.srt.layers.certified_head.gated_logits`` records it: the certified head under
the path's device gate and the stock head (``torch.matmul`` into the float logits
buffer) under its negation, with the column fallback and dense merge as conditional
nodes. Then it replays, back to back on one stream with no host read between replays:

- the drain's real sequence of verify batch sizes and gates (the h6b ``certlog``
  replay log of the c = 64 point: bs 60..18 with the gate off, then 16..1 with it on),
  padded to the captured graph sizes as the graph runner pads them;
- the drain-like sizes 64, 48, 32, 16, 8, 64 rows, each with the gate on and then off.

Each replay sets the gate and the device row count with ``fill_`` right before it, as
``set_gate`` does, between MTP draft (two certified steps) and draft-extend replays on
the same stream and pool. Each verify batch mixes real verify hidden states
(``experiments/head_geometry`` captures), near ties synthesized from them (two to four
tokens nudged to within a BF16 ulp of each other along ``w_a - w_b``), rows the dense
merge serves, and the 579ae7ce context's own rows (prompt plus output, from a plain
server's ``return_hidden_states``; the position-439 row sits where its request's row
for 439 would be).

After every replay, device copies (no readback) record the gate and ``valid`` the graph
read, the fallback flags, and per row the committed id (``_ids[:rows]``, as
``attach_ids`` clones it), status, candidate count, candidates and refined bounds, plus
the stock argmax of the same batch at the same shape computed eagerly
(``torch.matmul(h, W.T)``, the engine's ``_compute_lm_head``). Every ``CHUNK`` steps
the host compares them bitwise and logs each mismatch with the row's full state and an
eager re-run of the same batch. Certified replays also audit the envelope against those
stock logits (every candidate inside its refined interval, the stock argmax a candidate,
no excluded token reaching the winner's lower bound). Gate-off replays compare the
graph's stock argmax with the eager one; the draft paths are compared the same way.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.benchcert import plan

PORT = 30091
ROWS_PER_REQUEST = 4  # MTP: speculative_num_draft_tokens
DRAFT_STEPS = 2  # certified head calls in one MTP draft graph (num_steps - 1)
# The verify graph sizes of the drain reruns (server log: "Capture target verify CUDA graph").
GRAPH_BS = (*range(1, 9), *range(10, 33, 2), *range(40, 65, 4), *range(72, 129, 8))
MAX_ROWS = 64  # SGLANG_CERTIFIED_HEAD_MAX_ROWS of the timed certified runs
HEAD_ROWS = 256  # certified_head.engine.MAX_HEAD_BATCH: graphs up to this carry the head
CANDS = 64  # certified_head.head.COLS_CAP
CHUNK = 256  # replays between host comparisons
DETAILED = (
    200  # mismatch records with the batch's slots and an eager re-run; later ones are compact
)
SYNTHETIC_ROWS = (64, 48, 32, 16, 8, 64)
STATUS_AMBIGUOUS = 2
GEOMETRY = Path.home() / 'vp-data/geometry/mtp4b/heads'
TARGET = ('579ae7ce', 439)


# -- sizes and gates (CPU) --------------------------------------------------------------


def padded_bs(bs: int) -> int:
    """The captured graph a batch of ``bs`` requests replays (smallest size >= bs)."""
    for size in GRAPH_BS:
        if size >= bs:
            return size
    raise ValueError(f'batch size {bs} above the largest graph {GRAPH_BS[-1]}')


@dataclass(frozen=True)
class Step:
    bs: int  # requests
    gate: bool  # host's intended gate
    mode: str  # drain or synthetic

    @property
    def rows(self) -> int:
        return ROWS_PER_REQUEST * self.bs

    @property
    def m(self) -> int:
        """Rows of the padded graph, the head's batch."""
        return ROWS_PER_REQUEST * padded_bs(self.bs)


def engine_gate(bs: int, eligible: bool = True, max_rows: int = MAX_ROWS) -> bool:
    """``set_gate``: certify iff eligible and the real rows fit the row limit and a graph
    that carries the head (``_PathState.max_rows``)."""
    rows = ROWS_PER_REQUEST * bs
    covered = ROWS_PER_REQUEST * padded_bs(bs) <= HEAD_ROWS
    return bool(eligible) and covered and rows <= max_rows


def drain_steps(
    records: Iterable[dict[str, Any]], tail_from: int = 60, lead: int = 64
) -> list[Step]:
    """The c = 64 point's verify sequence from a ``certlog`` replay log (its longest point,
    not the warmup burst that also reaches the size): ``lead``
    steady steps before the drain, then the drain from the last step with at least
    ``tail_from`` requests to the end of the point. The gate is the one the log recorded,
    checked against ``engine_gate``."""
    # Every replay record is a target verify; 'path' names the certified path and is
    # None when the gate was off, so it does not select verify replays.
    replays = [r for r in records if r.get('kind') == 'replay']
    points: list[list[dict[str, Any]]] = []
    for r in replays:
        if points and points[-1] and r['t'] - points[-1][-1]['t'] > 0.5:
            points.append([])
        if not points:
            points.append([])
        points[-1].append(r)
    full = [p for p in points if max(r['batch_size'] for r in p) >= tail_from]
    if not full:
        raise ValueError('no point reaches the drain size')
    point = max(full, key=len)  # the measured point, not the short warmup burst
    start = max(i for i, r in enumerate(point) if r['batch_size'] >= tail_from)
    steps = []
    for r in point[max(0, start - lead) :]:
        gate = bool(r['gates']['verify'])
        if gate != engine_gate(r['batch_size']):
            raise ValueError(f'step {r["n"]}: logged gate {gate} differs from the engine rule')
        steps.append(Step(int(r['batch_size']), gate, 'drain'))
    return steps


def synthetic_steps() -> list[Step]:
    """64, 48, 32, 16, 8 and 64 rows, each with the gate on and then off."""
    steps = []
    for rows in SYNTHETIC_ROWS:
        bs = rows // ROWS_PER_REQUEST
        steps += [Step(bs, True, 'synthetic'), Step(bs, False, 'synthetic')]
    return steps


def gate_toggles(steps: Sequence[Step]) -> int:
    return sum(1 for a, b in itertools.pairwise(steps) if a.gate != b.gate)


# -- batch composition (CPU) ------------------------------------------------------------


@dataclass
class Pools:
    """Index ranges of each row class in the concatenated device pool."""

    clear: np.ndarray
    column: np.ndarray
    dense: np.ndarray
    context: np.ndarray  # 579ae7ce rows by output position (index = position), may be empty
    draft: np.ndarray

    def target_rows(self) -> np.ndarray | None:
        """The context rows of the request's verify at position 439: hidden states that
        predict output positions 438-441 (the request's rows in that verify)."""
        p = TARGET[1]
        if len(self.context) <= p + 2:
            return None
        return self.context[p - 1 : p + 3]


CLASS_NAMES = ('clear', 'column', 'dense', 'context')


def compose(rng: np.random.Generator, rows: int, pools: Pools) -> tuple[np.ndarray, np.ndarray]:
    """Pool indices and class codes (CLASS_NAMES) for one verify batch of ``rows`` rows:
    mostly real rows, about 8% near ties for the column fallback, dense-merge rows in
    about a third of batches, and in half the batches 579ae7ce's four rows at a random
    request slot (its position-439 row at offset 1)."""
    idx = rng.choice(pools.clear, rows)
    cls = np.zeros(rows, np.int8)
    if len(pools.column):
        n = rng.binomial(rows, 0.08)
        at = rng.choice(rows, n, replace=False)
        idx[at] = rng.choice(pools.column, n)
        cls[at] = 1
    if len(pools.dense) and rng.random() < 1 / 3:
        n = min(rows, int(rng.integers(1, 3)))
        at = rng.choice(rows, n, replace=False)
        idx[at] = rng.choice(pools.dense, n)
        cls[at] = 2
    target = pools.target_rows()
    if target is not None and rows >= ROWS_PER_REQUEST and rng.random() < 0.5:
        slot = int(rng.integers(rows // ROWS_PER_REQUEST)) * ROWS_PER_REQUEST
        idx[slot : slot + ROWS_PER_REQUEST] = target
        cls[slot : slot + ROWS_PER_REQUEST] = 3
    return idx, cls


# -- near ties (CPU) --------------------------------------------------------------------


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 bit patterns, round to nearest even (finite inputs)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return (b >> 16).astype(np.uint16)


def bf16_value(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_ulp(x: float) -> float:
    """Spacing of BF16 values at |x| (8 significant bits)."""
    if x == 0:
        return 2.0**-133
    return float(2.0 ** (np.floor(np.log2(abs(x))) - 7))


def nudge(h: np.ndarray, w: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Minimal change of ``h`` (FP64) that sets ``w[0] . h - w[j] . h = gaps[j-1]`` for
    j >= 1 (least squares in the span of ``w[0] - w[j]``), returned as BF16 bits."""
    d = w[0][None, :] - w[1:]
    target = np.asarray(gaps, np.float64) - d @ h
    step = d.T @ np.linalg.solve(d @ d.T, target)
    return bf16_bits((h + step).astype(np.float32))


# -- server: the 579ae7ce context's hidden states ----------------------------------------


def context(point: Path, out: Path) -> dict[str, Any]:
    """579ae7ce's prompt and output token ids from a point's client export."""
    from experiments.benchcert.drain import requests

    for item in requests(point):
        if item['phase'] == 'profiling' and item['prompt'].startswith(TARGET[0]):
            record = {'point': str(point), 'input_ids': item['input'], 'output_ids': item['output']}
            out.mkdir(parents=True, exist_ok=True)
            (out / 'context.json').write_text(json.dumps(record) + '\n')
            return record
    raise SystemExit(f'{TARGET[0]} not in {point}')


def start(out: Path) -> int:
    """rescore.py's stock plain-tuned server, returning hidden states, on PORT."""
    from bench.arms import resolve_arm
    from bench.server import Server
    from experiments.benchcert.rescore import OVERRIDES

    arm = resolve_arm('plain-tuned', {**OVERRIDES, 'enable-return-hidden-states': True})
    arm = type(arm)(**{**arm.to_json(), 'max_concurrency': 32})
    server = Server(arm, out / 'server', PORT, sglang_worktree=plan.ENGINE_WORKTREE)
    server.start()
    try:
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (out / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def post(url: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.loads(response.read())
        return result


def context_rows(hidden_states: list[Any], prompt_len: int, output_len: int) -> np.ndarray:
    """The rows that predict output positions 0..output_len-1 from a ``return_hidden_states``
    response to prompt + output[:-1]: one list per prefill chunk (one row per input
    position), then one row per generated token. Input position prompt_len - 1 + j holds
    the hidden state that predicts output j."""
    flat: list[Any] = []
    for item in hidden_states:
        if item and isinstance(item[0], list):
            flat.extend(item)
        else:
            flat.append(item)
    if len(flat) < prompt_len - 1 + output_len:
        raise ValueError(f'{len(flat)} hidden rows for {prompt_len} + {output_len} positions')
    return np.asarray(flat[prompt_len - 1 : prompt_len - 1 + output_len], np.float32)


REF_TOKENS = (1756, 8078, 5715, 68189)  # the wrong token and the three near-tied ones


def choose_planted(
    paths: dict[str, dict[int, float]], candidates: dict[str, int], wrong: int = 1756
) -> tuple[str, dict[str, float]]:
    """The candidate suffix whose logprob is closest to the wrong token's under every
    reference path: the smallest worst-case distance (ties to the smaller token id). Paths
    without both logprobs are skipped."""
    scores: dict[str, float] = {}
    for suffix, token in candidates.items():
        gaps = [abs(lp[token] - lp[wrong]) for lp in paths.values() if token in lp and wrong in lp]
        if gaps:
            scores[suffix] = max(gaps)
    if not scores:
        raise ValueError('no reference path scored the candidates')
    best = min(scores, key=lambda k: (scores[k], candidates[k]))
    return best, scores


def _logprobs(entries: Any) -> dict[int, float]:
    """{token: logprob} from one position of SGLang's *_token_ids_logprobs."""
    return {int(e[1]): float(e[0]) for e in entries or [] if e and e[0] is not None}


def refs(out: Path, runs: Path, url: str, candidates: dict[str, int], plant: Path) -> int:
    """579ae7ce's distribution at position 439 under four batch-1 stock paths on one
    server: prefills ending at absolute position 514 (the context only), 576 and 587 (the
    scorer's shape), and decoding from position 400; then the planted suffix
    (choose_planted) written to ``plant``."""
    from experiments.benchcert.drain import point_dirs, requests

    point = next(p for n, p in point_dirs(runs / 'drain', runs) if n.startswith('s1/'))
    item = next(
        r for r in requests(point) if r['phase'] == 'profiling' and r['prompt'].startswith(TARGET[0])
    )
    prompt, output = item['input'], item['output']
    pos, n = TARGET[1], len(prompt)
    track = sorted({*REF_TOKENS, *candidates.values()})
    common = {'return_logprob': True, 'top_logprobs_num': 5, 'token_ids_logprob': track}
    result: dict[str, Any] = {'prompt_len': n, 'position': pos, 'paths': {}}
    paths: dict[str, dict[int, float]] = {}
    body: dict[str, Any] = {
        'input_ids': prompt + output[:pos],
        'sampling_params': {'max_new_tokens': 1, 'temperature': 0.0},
        **common,
    }
    meta = post(f'{url}/generate', body)['meta_info']
    paths['prefill_514'] = _logprobs((meta.get('output_token_ids_logprobs') or [None])[0])
    result['paths']['prefill_514'] = {'top5': (meta.get('output_top_logprobs') or [None])[0]}
    for end in (n + 501, n + len(output)):
        body = {
            'input_ids': (prompt + output)[:end],
            'sampling_params': {'max_new_tokens': 0, 'temperature': 0.0},
            'logprob_start_len': n + pos - 1,
            **common,
        }
        meta = post(f'{url}/generate', body)['meta_info']
        key = f'prefill_{end}'
        # Entry 0 is position n + pos - 1 (SGLang leaves it empty); entry 1 is n + pos.
        ids = meta.get('input_token_logprobs') or []
        checked = len(ids) > 1 and ids[1] and int(ids[1][1]) == output[pos]
        paths[key] = _logprobs((meta.get('input_token_ids_logprobs') or [None, None])[1])
        result['paths'][key] = {
            'top5': (meta.get('input_top_logprobs') or [None, None])[1],
            'aligned': bool(checked),
        }
    start = 400
    body = {
        'input_ids': prompt + output[:start],
        'sampling_params': {'max_new_tokens': pos - start + 1, 'temperature': 0.0},
        **common,
    }
    response = post(f'{url}/generate', body)
    meta = response['meta_info']
    generated = list(response.get('output_ids') or [])
    same = generated[: pos - start] == output[start:pos]
    entries = meta.get('output_token_ids_logprobs') or []
    if same and len(entries) > pos - start:
        paths['decode_from_400'] = _logprobs(entries[pos - start])
    result['paths']['decode_from_400'] = {
        'same_prefix': same,
        'top5': (meta.get('output_top_logprobs') or [None] * (pos - start + 1))[pos - start]
        if len(meta.get('output_top_logprobs') or []) > pos - start
        else None,
    }
    for key, lp in paths.items():
        result['paths'][key]['ref_tokens'] = {str(t): lp.get(t) for t in REF_TOKENS}
    suffix, scores = choose_planted(paths, candidates)
    result['chosen'] = {
        'suffix': suffix,
        'token': candidates[suffix],
        'worst_gap_to_1756': scores[suffix],
        'per_path': {k: lp.get(candidates[suffix]) for k, lp in paths.items()},
    }
    result['scores'] = scores
    out.mkdir(parents=True, exist_ok=True)
    (out / 'refs.json').write_text(json.dumps(result, indent=1) + '\n')
    plant.parent.mkdir(parents=True, exist_ok=True)
    plant.write_text(json.dumps(result['chosen']) + '\n')
    print(json.dumps({k: v.get('ref_tokens') for k, v in result['paths'].items()}))
    print(json.dumps(result['chosen']))
    return 0


def fetch(out: Path, contexts: Path | None) -> int:
    """Hidden states of every output position of the context (one prefill), and the
    batch-1 top-5 re-score of the gross contexts (rescore.score_one) if given."""
    url = f'http://127.0.0.1:{PORT}'
    record = json.loads((out / 'context.json').read_text())
    prompt, output = record['input_ids'], record['output_ids']
    body = {
        'input_ids': prompt + output[:-1],
        'sampling_params': {'max_new_tokens': 1, 'temperature': 0.0},
        'return_hidden_states': True,
        'return_logprob': True,
        'logprob_start_len': len(prompt) - 1,
        'top_logprobs_num': 5,
    }
    meta = post(f'{url}/generate', body)['meta_info']
    rows = context_rows(meta['hidden_states'], len(prompt), len(output))
    if rows.shape != (len(output), 2560):
        raise SystemExit(f'unexpected hidden-state shape {rows.shape}')
    bits = bf16_bits(rows)
    # The head reads BF16; rows that are not BF16 values are not the head's input, which
    # the stress run's context check (argmax against the server's top-1) would show.
    exact = bool(np.array_equal(bf16_value(bits), rows))
    np.save(out / 'context_hidden_bits.npy', bits)
    (out / 'context_meta.json').write_text(
        json.dumps({'rows': len(rows), 'bf16_values': exact, 'prompt_len': len(prompt)}) + '\n'
    )
    tops = meta.get('input_top_logprobs') or []
    (out / 'context_top5.json').write_text(
        json.dumps([[[float(e[0]), int(e[1])] for e in (t or [])] for t in tops]) + '\n'
    )
    print(f'context: {len(output)} hidden rows (BF16 values: {exact}), {len(tops)} top-5 lists')
    if contexts is not None and contexts.exists():
        from experiments.benchcert.rescore import score_one

        lines = [json.loads(line) for line in contexts.read_text().splitlines() if line.strip()]
        with (out / 'gross_rescored.jsonl').open('w') as handle:
            for c in lines:
                try:
                    handle.write(json.dumps(score_one(url, c)) + '\n')
                except Exception as exc:  # recorded per context
                    handle.write(json.dumps({'id': c['id'], 'error': repr(exc)}) + '\n')
        print(f'rescored {len(lines)} gross contexts')
    return 0


# -- GPU ---------------------------------------------------------------------------------


def segment_pool(snapshot: dict[str, Any], ptr: int) -> list[int] | None:
    """The memory pool (segment_pool_id) owning device address ``ptr``; (0, 0) is the
    general caching allocator, a graph's private pool has another id."""
    for seg in snapshot['segments']:
        if seg['address'] <= ptr < seg['address'] + seg['total_size']:
            return list(seg['segment_pool_id'])
    return None


def probe_minimal() -> dict[str, Any]:
    """Where an allocation on the capture stream after a conditional node lands, and
    whether an eager tensor allocated after capture can share its memory.

    A graph is captured into a private pool: a temporary allocated before an if-node, a
    small allocation inside the node, then a temporary of size S allocated after the node
    and freed before the capture ends. Before the capture a block of size S is freed into
    the general allocator, so a post-node allocation that the general allocator serves
    takes it. After the capture an eager tensor of size S is allocated and zeroed, the
    graph is replayed (it writes 3s into its post-node temporary), and the eager tensor is
    checked."""
    import torch
    from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

    dev = torch.device('cuda')
    size = 37 * 1024 * 1024 + 4096
    pred = torch.ones((), dtype=torch.bool, device=dev)
    primer = torch.empty(size, dtype=torch.uint8, device=dev)
    primer_ptr = primer.data_ptr()
    del primer
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    ptrs: dict[str, int] = {}
    with torch.cuda.graph(g, pool=pool, stream=stream):
        before = torch.empty(size, dtype=torch.uint8, device=dev)
        before.fill_(1)
        ptrs['before_node'] = before.data_ptr()
        with _if_body(pred):
            inside = torch.empty(4096, dtype=torch.uint8, device=dev)
            inside.fill_(2)
            ptrs['inside_node'] = inside.data_ptr()
        after = torch.empty(size, dtype=torch.uint8, device=dev)
        after.fill_(3)
        ptrs['after_node'] = after.data_ptr()
        del before, inside, after
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    result: dict[str, Any] = {
        'graph_pool': list(pool),
        'pools': {k: segment_pool(snap, v) for k, v in ptrs.items()},
        'after_node_took_freed_general_block': ptrs['after_node'] == primer_ptr,
    }
    eager = torch.zeros(size, dtype=torch.uint8, device=dev)
    result['eager_shares_after_node_memory'] = eager.data_ptr() == ptrs['after_node']
    g.replay()
    torch.cuda.synchronize()
    result['eager_overwritten_by_replay'] = bool((eager != 0).any())
    del eager, g
    torch.cuda.synchronize()
    return result


class Rig:
    """The heads, graphs, pools and device logs of one stress run (CUDA)."""

    def __init__(self, out: Path, seed: int) -> None:
        import torch

        sys.path.insert(0, str(plan.REPO / 'src'))
        from certified_head import engine
        from certified_head.quantize import load_head_weight

        self.torch = torch
        self.out = out
        self.rng = np.random.default_rng(seed)
        self.dev = torch.device('cuda')
        t0 = time.perf_counter()
        self.weight = load_head_weight().to(self.dev).contiguous()
        self.vocab, self.k = self.weight.shape
        flags = engine.Flags(verify=True, draft=True, fallback='columns', model='conservative')
        self.heads = engine.EngineHeads(
            self.weight, flags, {p: engine.MAX_HEAD_BATCH for p in engine.PATHS}
        )
        # PathHead instances (Any: certified_head is imported at run time, not by mypy).
        self.verify: Any = self.heads.get('verify')
        self.draft: Any = self.heads.get('draft')
        self.extend: Any = self.heads.get('draft_extend')
        if self.verify is None or self.draft is None or self.extend is None:
            raise RuntimeError('the engine heads lack the verify or draft paths')
        self.setup: dict[str, Any] = {'load_s': round(time.perf_counter() - t0, 1)}

        def state() -> tuple[Any, Any]:
            return (
                torch.zeros((), dtype=torch.bool, device=self.dev),
                torch.zeros((), dtype=torch.int32, device=self.dev),
            )

        self.vgate, self.vvalid = state()
        self.dgate, self.dvalid = state()
        self.egate, self.evalid = state()
        k = self.k
        self.v_in = torch.zeros(HEAD_ROWS, k, dtype=torch.bfloat16, device=self.dev)
        self.d_in = torch.zeros(DRAFT_STEPS, 64, k, dtype=torch.bfloat16, device=self.dev)
        self.e_in = torch.zeros(64, k, dtype=torch.bfloat16, device=self.dev)
        self.logits_buf = torch.zeros(HEAD_ROWS, self.vocab, dtype=torch.float32, device=self.dev)
        self.d_out = torch.zeros(DRAFT_STEPS, 64, dtype=torch.int64, device=self.dev)
        self.slot_src = np.zeros(HEAD_ROWS, np.int64)  # pool row held by each verify input slot
        self.graphs: dict[tuple[str, int], Any] = {}
        self.detailed = 0

    # pools ------------------------------------------------------------------------

    def build_pools(self, context_bits: Path | None, n_real: int, n_ties: int) -> Pools:
        torch = self.torch
        from experiments.head_geometry.replay_data import iter_records

        target, draft = [], []
        for rec in iter_records(GEOMETRY, 'mtp_verify'):
            th, dh = rec['target_hidden'], rec['draft_hidden']
            target.append(th.reshape(-1, th.shape[-1]))
            draft.append(dh.reshape(-1, dh.shape[-1]))
            if sum(len(t) for t in target) >= n_real:
                break
        real = np.concatenate(target)[:n_real]
        drafts = np.concatenate(draft)[:4096]
        ctx = np.load(context_bits) if context_bits is not None and context_bits.exists() else None
        parts = [real]
        if ctx is not None:
            parts.append(ctx)
        ties, tie_meta = self.synthesize_ties(real, n_ties)
        if ctx is not None and len(ctx) > TARGET[1]:
            # Near-tie variants of 579ae7ce's own position-439 row (its top tokens are the
            # three-way tie), moved within an ulp of each other.
            own, _ = self.synthesize_ties(np.repeat(ctx[TARGET[1] : TARGET[1] + 1], 128, 0), 128)
            ties = np.concatenate([ties, own])
            tie_meta['context_439_variants'] = len(own)
        dense = self.synthesize_dense(real)
        parts += [ties, dense, drafts]
        bits = np.concatenate(parts)
        self.pool = (
            torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(self.dev).contiguous()
        )
        status, count = self.classify(self.pool)
        n_ctx = 0 if ctx is None else len(ctx)
        base = np.arange(len(bits))
        real_idx = base[: len(real)]
        ctx_idx = base[len(real) : len(real) + n_ctx]
        tie_idx = base[len(real) + n_ctx : len(real) + n_ctx + len(ties)]
        dense_idx = base[len(real) + n_ctx + len(ties) : len(real) + n_ctx + len(ties) + len(dense)]
        draft_idx = base[len(bits) - len(drafts) :]
        column = (status == STATUS_AMBIGUOUS) & (count <= CANDS)
        dense_mask = (status != 0) & ~column
        pools = Pools(
            clear=real_idx[status[real_idx] == 0],
            column=np.concatenate([tie_idx[column[tie_idx]], real_idx[column[real_idx]]]),
            dense=np.concatenate(
                [dense_idx[dense_mask[dense_idx]], real_idx[dense_mask[real_idx]]]
            ),
            context=ctx_idx,
            draft=draft_idx,
        )
        self.pool_status, self.pool_count = status, count
        self.setup['pools'] = {
            'real_rows': len(real),
            'real_clear': int((status[real_idx] == 0).sum()),
            'real_column': int(column[real_idx].sum()),
            'real_dense': int(dense_mask[real_idx].sum()),
            'tie_candidates': len(ties),
            'tie_column': int(column[tie_idx].sum()),
            'tie_meta': tie_meta,
            'dense_candidates': len(dense),
            'dense_rows': int(dense_mask[dense_idx].sum()),
            'context_rows': n_ctx,
            'context_439_status': None if ctx is None else int(status[ctx_idx[TARGET[1]]]),
            'context_439_count': None if ctx is None else int(count[ctx_idx[TARGET[1]]]),
            'draft_rows': len(drafts),
        }
        if ctx is not None:
            try:
                self.setup['context_check'] = self.check_context(ctx_idx)
            except Exception as exc:  # a diagnostic; the run goes on without it
                self.setup['context_check'] = {'error': repr(exc)}
        return pools

    def exact_top(self, h: Any, k: int) -> tuple[Any, Any]:
        """Top-k tokens and FP64 logits of BF16 rows (chunks of 64 rows)."""
        torch = self.torch
        w64 = None
        vals, idx = [], []
        for r0 in range(0, h.shape[0], 64):
            hh = h[r0 : r0 + 64].to(torch.float64)
            best_v = torch.full((hh.shape[0], k), -torch.inf, dtype=torch.float64, device=self.dev)
            best_i = torch.zeros((hh.shape[0], k), dtype=torch.int64, device=self.dev)
            for c0 in range(0, self.vocab, 32768):
                w64 = self.weight[c0 : c0 + 32768].to(torch.float64)
                z = hh @ w64.T
                v, i = torch.topk(torch.cat([best_v, z], 1), k, dim=1)
                cand = torch.cat(
                    [best_i, torch.arange(c0, c0 + z.shape[1], device=self.dev).expand_as(z)], 1
                )
                best_v, best_i = v, torch.gather(cand, 1, i)
            vals.append(best_v)
            idx.append(best_i)
        return torch.cat(vals), torch.cat(idx)

    def synthesize_ties(self, real: np.ndarray, n: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Near ties from real rows: the top 2, 3 or 4 tokens nudged to within about one
        BF16 ulp of the top logit (gaps uniform in +-0.75 ulp)."""
        torch = self.torch
        pick = self.rng.choice(len(real), n, replace=False)
        h = torch.from_numpy(real[pick].view(np.int16)).view(torch.bfloat16).to(self.dev)
        vals, idx = self.exact_top(h, 4)
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        out = np.empty((n, self.k), np.uint16)
        ways = self.rng.choice([2, 3, 4], n, p=[0.25, 0.5, 0.25])
        for r in range(n):
            tokens = idx[r, : ways[r]]
            w = self.weight[torch.from_numpy(tokens).to(self.dev)].to(torch.float64).cpu().numpy()
            ulp = bf16_ulp(float(vals[r, 0]))
            gaps = self.rng.uniform(-0.75, 0.75, ways[r] - 1) * ulp
            h64 = bf16_value(real[pick[r]]).astype(np.float64)
            out[r] = nudge(h64, w, gaps)
        return out, {f'{w}_way': int((ways == w).sum()) for w in (2, 3, 4)}

    def synthesize_dense(self, real: np.ndarray) -> np.ndarray:
        """Rows the dense merge serves: scaled-down real rows (wide candidate lists) and
        Gaussian rows; classify() keeps those whose status routes them to the dense merge."""
        pick = self.rng.choice(len(real), 256, replace=False)
        rows = bf16_value(real[pick])
        scaled = [rows[i] * s for i, s in enumerate(self.rng.choice([0.02, 0.05, 0.1, 0.2], 256))]
        gauss = self.rng.normal(0, 2, (64, self.k)).astype(np.float32)
        return bf16_bits(np.concatenate([np.stack(scaled), gauss]))

    def classify(self, pool: Any) -> tuple[np.ndarray, np.ndarray]:
        """Status and candidate count of every pool row, eagerly at 64 rows, no fallback."""
        torch = self.torch
        head = self.verify.head
        status, count = [], []
        for r0 in range(0, pool.shape[0], 64):
            h = pool[r0 : r0 + 64]
            pad = 64 - h.shape[0]
            if pad:
                h = torch.cat([h, pool[:pad]])
            _, stats = head.argmax(h.contiguous(), fallback=False)
            status.append(stats.status[: 64 - pad].cpu().numpy().copy())
            count.append(stats.candidates[: 64 - pad].cpu().numpy().copy())
        return np.concatenate(status), np.concatenate(count)

    def check_context(self, ctx_idx: np.ndarray) -> dict[str, Any]:
        """The context rows are head inputs: their stock argmax should match the plain
        server's teacher-forced top-1 (context_top5.json) at every output position."""
        torch = self.torch
        top_path = self.out / 'context_top5.json'
        h = self.pool[torch.from_numpy(ctx_idx).to(self.dev)]
        ids = torch.matmul(h, self.weight.T).argmax(-1).cpu().numpy()
        result: dict[str, Any] = {'rows': len(ids), 'argmax_439': int(ids[TARGET[1]])}
        if top_path.exists():
            tops = json.loads(top_path.read_text())
            # Entry 0 is the prompt's last position (no logprob); output j follows at j + 1.
            server = [t[0][1] if t else None for t in tops[1 : 1 + len(ids)]]
            result['server_top1_match'] = int(
                sum(1 for a, b in zip(ids, server, strict=False) if a == b)
            )
            result['server_top5_439'] = tops[1 + TARGET[1]] if len(tops) > 1 + TARGET[1] else None
        return result

    # graphs ------------------------------------------------------------------------

    def capture(self) -> None:
        """Warm and capture every graph with the head, largest first, in one pool:
        target verify, then MTP draft, then draft extend (the server's order)."""
        torch = self.torch
        from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

        t0 = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        warm_rows = self.pool[:HEAD_ROWS].contiguous()
        sizes = [bs for bs in GRAPH_BS if ROWS_PER_REQUEST * bs <= HEAD_ROWS]
        w = self.weight
        for bs in reversed(sizes):
            m = ROWS_PER_REQUEST * bs
            self.verify.warm(warm_rows[:m].clone())
            self.v_in[:m].copy_(warm_rows[:m])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                h = self.v_in[:m].clone()
                self.verify.argmax(h, gate=self.vgate, valid=self.vvalid)
                with _if_body(torch.logical_not(self.vgate)):
                    self.logits_buf[:m].copy_(torch.matmul(h, w.T))
            self.graphs[('verify', bs)] = g
        draft_sizes = [bs for bs in GRAPH_BS if bs <= 64]
        for bs in reversed(draft_sizes):
            self.draft.warm(warm_rows[:bs].clone())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                for step in range(DRAFT_STEPS):
                    h = self.d_in[step, :bs].clone()
                    ids = self.draft.argmax(h, gate=self.dgate, valid=self.dvalid)
                    with _if_body(torch.logical_not(self.dgate)):
                        self.d_out[step, :bs].copy_(torch.matmul(h, w.T).argmax(-1))
                    with _if_body(self.dgate):
                        self.d_out[step, :bs].copy_(ids)
            self.graphs[('draft', bs)] = g
        for bs in reversed(draft_sizes):
            self.extend.warm(warm_rows[:bs].clone())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                h = self.e_in[:bs].clone()
                self.extend.argmax(h, gate=self.egate, valid=self.evalid)
            self.graphs[('extend', bs)] = g
        torch.cuda.synchronize()
        self.setup['capture_s'] = round(time.perf_counter() - t0, 1)
        self.setup['graphs'] = len(self.graphs)
        self.setup['column_report'] = self.verify.column_report
        self.setup['self_test'] = self.verify.head.self_test_summary()

    # replay ------------------------------------------------------------------------

    def alloc_log(self) -> None:
        torch = self.torch

        def z(*shape: int, dtype: Any) -> Any:
            return torch.zeros(*shape, dtype=dtype, device=self.dev)

        n, r = CHUNK, HEAD_ROWS
        self.log = {
            'gate': z(n, dtype=torch.bool),
            'valid': z(n, dtype=torch.int32),
            'any': z(n, dtype=torch.bool),
            'any_cols': z(n, dtype=torch.bool),
            'any_dense': z(n, dtype=torch.bool),
            'ids': z(n, r, dtype=torch.int64),
            'committed': z(n, r, dtype=torch.int64),
            'status': z(n, r, dtype=torch.int32),
            'count': z(n, r, dtype=torch.int32),
            'cand': z(n, MAX_ROWS, CANDS, dtype=torch.int32),
            'rlo': z(n, MAX_ROWS, CANDS, dtype=torch.float32),
            'rhi': z(n, MAX_ROWS, CANDS, dtype=torch.float32),
            'ref': z(n, r, dtype=torch.int64),
            'graph_stock': z(n, r, dtype=torch.int64),
            'draft_ids': z(n, DRAFT_STEPS, 64, dtype=torch.int64),
            'draft_ref': z(n, DRAFT_STEPS, 64, dtype=torch.int64),
            'extend_ids': z(n, 64, dtype=torch.int64),
            'extend_ref': z(n, 64, dtype=torch.int64),
            'dgate': z(n, dtype=torch.bool),
            # Envelope audit (certified replays, first 64 rows): candidates whose stock
            # logit lies outside [rlo, rhi], whether the stock argmax is a candidate, the
            # largest stock logit among excluded tokens and the winner's lower bound.
            'env_outside': z(n, MAX_ROWS, dtype=torch.int32),
            'env_top_in': z(n, MAX_ROWS, dtype=torch.bool),
            'env_excl_max': z(n, MAX_ROWS, dtype=torch.float32),
            'env_best_rlo': z(n, MAX_ROWS, dtype=torch.float32),
            # Stock logits of the stock argmax and of the head's id (certified replays),
            # to class a mismatch by size: exact tie, one BF16 ulp, or more.
            'z_ref': z(n, r, dtype=torch.float32),
            'z_head': z(n, r, dtype=torch.float32),
        }

    def argmax_stock(self, h: Any) -> Any:
        return self.torch.matmul(h, self.weight.T).argmax(-1)

    def alloc_sentinels(self, which: range | None = None) -> None:
        """Long-lived eager tensors allocated after capture (the engine allocates per-batch
        and per-request state after its graphs are captured), filled with a pattern whose
        sum is kept: a graph replay that writes a temporary into their memory changes it."""
        torch = self.torch
        sizes = [4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]
        if which is None:
            self.sentinels: list[tuple[Any, Any]] = [None] * (6 * len(sizes))  # type: ignore[list-item]
            which = range(len(self.sentinels))
        for i in which:
            n = sizes[i % len(sizes)] // 4
            t = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device=self.dev)
            self.sentinels[i] = (t, t.to(torch.int64).sum())

    def check_sentinels(self) -> list[dict[str, Any]]:
        """Sentinels whose sum changed (one host read of all sums)."""
        torch = self.torch
        now = torch.stack([t.to(torch.int64).sum() for t, _ in self.sentinels])
        ref = torch.stack([s for _, s in self.sentinels])
        bad = (now != ref).nonzero().flatten().tolist()
        return [
            {'kind': 'sentinel', 'index': i, 'bytes': self.sentinels[i][0].numel() * 4} for i in bad
        ]

    def audit(self, s: int, full: Any, r: int) -> None:
        """Envelope containment against the stock logits of the same batch (device ops).

        The refined bounds enclose the reference's BF16 logit, which is what the stock
        GEMM at this shape returns, so every candidate's stock logit must lie in
        [rlo, rhi], and every excluded token's stock logit must be below the winner's
        lower bound (a row with status 0 or exactly AMBIGUOUS passed the threshold test,
        which implies that)."""
        torch = self.torch
        head = self.verify.head
        log = self.log
        cand = head._cand[:r, :CANDS].long()
        cnt = head._count[:r].clamp(max=CANDS)
        valid = torch.arange(CANDS, device=self.dev)[None, :] < cnt[:, None]
        z = full[:r]
        zc = torch.gather(z, 1, cand).float()
        rlo, rhi = head._rlo[:r, :CANDS], head._rhi[:r, :CANDS]
        outside = valid & ((zc < rlo) | (zc > rhi))
        log['env_outside'][s, :r].copy_(outside.sum(1).to(torch.int32))
        top = z.argmax(-1)
        log['env_top_in'][s, :r].copy_((valid & (cand == top[:, None])).any(1))
        log['env_best_rlo'][s, :r].copy_(torch.where(valid, rlo, float('-inf')).amax(1))
        excluded = z.clone()
        excluded.scatter_(1, torch.where(valid, cand, cand[:, :1]), float('-inf'))
        log['env_excl_max'][s, :r].copy_(excluded.amax(1).float())

    def run_step(self, s: int, step: Step, pools: Pools) -> dict[str, Any]:
        """One MTP cycle: draft graph, verify graph, draft-extend graph; device logs only."""
        torch = self.torch
        log = self.log
        bs, rows, m = step.bs, step.rows, step.m
        pb = padded_bs(bs)
        # Draft: certified (every drain batch fits 64 rows).
        for k in range(DRAFT_STEPS):
            d_idx = torch.from_numpy(self.rng.choice(pools.draft, pb)).to(
                self.dev, non_blocking=True
            )
            torch.index_select(self.pool, 0, d_idx, out=self.d_in[k, :pb])
        self.dgate.fill_(True)
        self.dvalid.fill_(bs)
        self.graphs[('draft', pb)].replay()
        log['dgate'][s].copy_(self.dgate)
        for k in range(DRAFT_STEPS):
            log['draft_ids'][s, k, :pb].copy_(self.d_out[k, :pb])
            log['draft_ref'][s, k, :pb].copy_(self.argmax_stock(self.d_in[k, :pb]))
        # Verify.
        idx, cls = compose(self.rng, rows, pools)
        self.slot_src[:rows] = idx
        torch.index_select(
            self.pool,
            0,
            torch.from_numpy(idx).to(self.dev, non_blocking=True),
            out=self.v_in[:rows],
        )
        self.vgate.fill_(step.gate)
        self.vvalid.fill_(rows)
        self.graphs[('verify', pb)].replay()
        head = self.verify.head
        log['gate'][s].copy_(self.vgate)
        log['valid'][s].copy_(self.vvalid)
        log['any'][s].copy_(head._any)
        log['any_cols'][s].copy_(head._any_cols)
        log['any_dense'][s].copy_(head._any_dense)
        log['ids'][s, :m].copy_(head._ids[:m])
        log['committed'][s, :rows].copy_(head._ids[:rows].clone())  # attach_ids
        log['status'][s, :m].copy_(head._status[:m])
        log['count'][s, :m].copy_(head._count[:m])
        r = min(m, MAX_ROWS)
        log['cand'][s, :r].copy_(head._cand[:r, :CANDS])
        log['rlo'][s, :r].copy_(head._rlo[:r, :CANDS])
        log['rhi'][s, :r].copy_(head._rhi[:r, :CANDS])
        if step.gate:
            full = self.torch.matmul(self.v_in[:m], self.weight.T)  # the stock logits
            ref = full.argmax(-1)
            log['ref'][s, :m].copy_(ref)
            log['z_ref'][s, :m].copy_(full.gather(1, ref[:, None])[:, 0].float())
            log['z_head'][s, :m].copy_(full.gather(1, head._ids[:m, None])[:, 0].float())
            self.audit(s, full, r)
        else:
            log['ref'][s, :m].copy_(self.argmax_stock(self.v_in[:m]))
        if not step.gate:
            log['graph_stock'][s, :m].copy_(self.logits_buf[:m].argmax(-1))
        # Draft extend.
        e_idx = torch.from_numpy(self.rng.choice(pools.draft, pb)).to(self.dev, non_blocking=True)
        torch.index_select(self.pool, 0, e_idx, out=self.e_in[:pb])
        self.egate.fill_(True)
        self.evalid.fill_(bs)
        self.graphs[('extend', pb)].replay()
        log['extend_ids'][s, :bs].copy_(self.extend.head._ids[:bs].clone())
        log['extend_ref'][s, :pb].copy_(self.argmax_stock(self.e_in[:pb]))
        return {'step': step, 'classes': cls, 'slots': self.slot_src[:m].copy()}

    def check_chunk(
        self, steps: list[dict[str, Any]], totals: dict[str, Any], start: int
    ) -> list[dict[str, Any]]:
        """Compare one chunk on the host; return mismatch records with row state."""
        torch = self.torch
        torch.cuda.synchronize()
        host = {k: v[: len(steps)].cpu().numpy() for k, v in self.log.items()}
        problems = []
        for s, entry in enumerate(steps):
            step: Step = entry['step']
            rows, m, bs = step.rows, step.m, step.bs
            totals['steps'] += 1
            totals['by_mode'][step.mode] = totals['by_mode'].get(step.mode, 0) + 1
            dev_gate, valid = bool(host['gate'][s]), int(host['valid'][s])
            if dev_gate != step.gate:
                problems.append(self.record('gate', start + s, entry, host, s, None))
            if valid != rows:
                problems.append(self.record('valid', start + s, entry, host, s, None))
            if step.gate:
                totals['verify_certified_steps'] += 1
                totals['verify_certified_rows'] += rows
                totals['verify_padding_rows'] += m - rows
                totals['rows_by_size'][m] = totals['rows_by_size'].get(m, 0) + rows
                st, cnt = host['status'][s, :rows], host['count'][s, :rows]
                col = (st == STATUS_AMBIGUOUS) & (cnt <= CANDS)
                totals['column_rows'] += int(col.sum())
                totals['dense_rows'] += int(((st != 0) & ~col).sum())
                totals['steps_column_ran'] += int(host['any_cols'][s])
                totals['steps_dense_ran'] += int(host['any_dense'][s])
                for c, name in enumerate(CLASS_NAMES):
                    totals['rows_by_class'][name] += int((entry['classes'] == c).sum())
                bad = np.nonzero(host['committed'][s, :rows] != host['ref'][s, :rows])[0]
                for row in bad:
                    problems.append(
                        self.record('verify_committed', start + s, entry, host, s, int(row))
                    )
                    zr, zh = float(host['z_ref'][s, row]), float(host['z_head'][s, row])
                    size = 'tie' if zr == zh else 'one_ulp' if zr - zh <= bf16_ulp(zr) else 'larger'
                    totals['mismatch_by_size'][size] += 1
                self.check_envelope(start + s, entry, host, s, totals, problems)
                pad_bad = np.nonzero(host['ids'][s, rows:m] != host['ref'][s, rows:m])[0]
                for row in pad_bad:
                    problems.append(
                        self.record('verify_padding', start + s, entry, host, s, rows + int(row))
                    )
            else:
                totals['verify_stock_steps'] += 1
                totals['verify_stock_rows'] += rows
                bad = np.nonzero(host['graph_stock'][s, :m] != host['ref'][s, :m])[0]
                for row in bad:
                    problems.append(self.record('stock_graph', start + s, entry, host, s, int(row)))
            pb = padded_bs(bs)
            for k in range(DRAFT_STEPS):
                bad = np.nonzero(host['draft_ids'][s, k, :pb] != host['draft_ref'][s, k, :pb])[0]
                totals['draft_rows'] += pb
                for row in bad:
                    problems.append(
                        {'kind': 'draft', 'step': start + s, 'draft_step': k, 'row': int(row)}
                    )
            bad = np.nonzero(host['extend_ids'][s, :bs] != host['extend_ref'][s, :bs])[0]
            totals['extend_rows'] += bs
            for row in bad:
                problems.append({'kind': 'draft_extend', 'step': start + s, 'row': int(row)})
        for name in self.log:
            self.log[name].zero_()
        return problems

    def check_envelope(
        self,
        step_no: int,
        entry: dict[str, Any],
        host: dict[str, Any],
        s: int,
        totals: dict[str, Any],
        problems: list[dict[str, Any]],
    ) -> None:
        """Host side of the envelope audit, for rows with a complete candidate list."""
        step: Step = entry['step']
        r = min(step.m, MAX_ROWS)
        st, cnt = host['status'][s, :r], host['count'][s, :r]
        complete = ((st == 0) | (st == STATUS_AMBIGUOUS)) & (cnt > 0) & (cnt <= CANDS)
        totals['audited_rows'] += int(complete.sum())
        rows_c = np.nonzero(complete)[0]
        excl, best = host['env_excl_max'][s, rows_c], host['env_best_rlo'][s, rows_c]
        for name, bad in (
            ('envelope_candidate', host['env_outside'][s, rows_c] > 0),
            ('envelope_top_not_candidate', ~host['env_top_in'][s, rows_c]),
            ('envelope_excluded', excl > best),
            ('envelope_excluded_tie', excl == best),
        ):
            totals[name] += int(bad.sum())
            for row in rows_c[bad]:
                rec = self.record(name, step_no, entry, host, s, int(row))
                rec['env'] = {
                    'outside': int(host['env_outside'][s, row]),
                    'top_in': bool(host['env_top_in'][s, row]),
                    'excluded_max': float(host['env_excl_max'][s, row]),
                    'best_rlo': float(host['env_best_rlo'][s, row]),
                }
                problems.append(rec)

    def record(
        self,
        kind: str,
        step_no: int,
        entry: dict[str, Any],
        host: dict[str, Any],
        s: int,
        row: int | None,
    ) -> dict[str, Any]:
        step: Step = entry['step']
        rec: dict[str, Any] = {
            'kind': kind,
            'step': step_no,
            'mode': step.mode,
            'bs': step.bs,
            'rows': step.rows,
            'm': step.m,
            'host_gate': step.gate,
            'device_gate': bool(host['gate'][s]),
            'valid': int(host['valid'][s]),
            'any': bool(host['any'][s]),
            'any_cols': bool(host['any_cols'][s]),
            'any_dense': bool(host['any_dense'][s]),
        }
        detailed = self.detailed < DETAILED
        self.detailed += 1
        if detailed:
            rec['slots'] = entry['slots'].tolist()
        if row is None:
            return rec
        n = int(host['count'][s, row])
        rec.update(
            row=row,
            committed=int(host['committed'][s, row]) if row < step.rows else None,
            head_id=int(host['ids'][s, row]),
            stock_id=int(host['ref'][s, row]),
            graph_stock_id=int(host['graph_stock'][s, row]),
            status=int(host['status'][s, row]),
            count=n,
            pool_row=int(entry['slots'][row]),
            z_ref=float(host['z_ref'][s, row]),
            z_head=float(host['z_head'][s, row]),
            row_class=CLASS_NAMES[int(entry['classes'][row])] if row < step.rows else 'padding',
        )
        if row < MAX_ROWS:
            k = min(n, CANDS)
            rec.update(
                cand=host['cand'][s, row, :k].tolist(),
                rlo=host['rlo'][s, row, :k].tolist(),
                rhi=host['rhi'][s, row, :k].tolist(),
            )
        if not detailed:
            return rec
        try:
            rec['eager'] = self.eager_rerun(entry['slots'], row, rec.get('cand', []))
        except Exception as exc:  # the mismatch is logged either way
            rec['eager'] = {'error': repr(exc)}
        return rec

    def eager_rerun(self, slots: np.ndarray, row: int, cand: list[int]) -> dict[str, Any]:
        """The same batch through the head outside any graph, and the stock logits of the
        row's candidates (full GEMM and the column fallback's gathered GEMM)."""
        torch = self.torch
        h = self.pool[torch.from_numpy(np.asarray(slots)).to(self.dev)].contiguous()
        ids, stats = self.verify.head.argmax(h)
        full = torch.matmul(h, self.weight.T)
        out: dict[str, Any] = {
            'id': int(ids[row]),
            'status': int(stats.status[row]),
            'stock_id': int(full[row].argmax()),
        }
        if cand:
            c = torch.tensor(cand, device=self.dev)
            out['stock_logits_of_cands'] = full[row, c].float().cpu().tolist()
            gathered = torch.matmul(h, self.weight.index_select(0, c).T)
            out['gathered_logits_of_cands'] = gathered[row].float().cpu().tolist()
        return out


def sentinels(rig: Rig, totals: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the long-lived sentinels after a chunk, then reallocate half of them (eager
    state is allocated and freed between batches in the engine too)."""
    found = rig.check_sentinels()
    totals['sentinel_checks'] += len(rig.sentinels)
    totals['sentinel_corrupt'] += len(found)
    half = totals['sentinel_checks'] // len(rig.sentinels) % 2
    rig.alloc_sentinels(range(half, len(rig.sentinels), 2))
    return found


def stress(out: Path, replay: Path, seconds: float, seed: int, n_real: int, n_ties: int) -> int:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in replay.read_text().splitlines() if line.strip()]
    drain = drain_steps(records)
    synthetic = synthetic_steps()
    try:
        minimal = probe_minimal()
    except Exception as exc:  # recorded; the stress run goes on
        minimal = {'error': repr(exc)}
    (out / 'probe_minimal.json').write_text(json.dumps(minimal, indent=1) + '\n')
    print(f'probe_minimal: {json.dumps(minimal)}', flush=True)
    rig = Rig(out, seed)
    context_bits = out / 'context_hidden_bits.npy'
    pools = rig.build_pools(context_bits if context_bits.exists() else None, n_real, n_ties)
    if not len(pools.column):
        raise SystemExit('no column-fallback rows in the pool')
    rig.capture()
    rig.alloc_log()
    rig.alloc_sentinels()
    totals: dict[str, Any] = {
        'steps': 0,
        'by_mode': {},
        'verify_certified_steps': 0,
        'verify_certified_rows': 0,
        'verify_padding_rows': 0,
        'verify_stock_steps': 0,
        'verify_stock_rows': 0,
        'rows_by_size': {},
        'rows_by_class': dict.fromkeys(CLASS_NAMES, 0),
        'column_rows': 0,
        'dense_rows': 0,
        'steps_column_ran': 0,
        'steps_dense_ran': 0,
        'audited_rows': 0,
        'envelope_candidate': 0,
        'envelope_top_not_candidate': 0,
        'envelope_excluded': 0,
        'envelope_excluded_tie': 0,
        'mismatch_by_size': {'tie': 0, 'one_ulp': 0, 'larger': 0},
        'draft_rows': 0,
        'extend_rows': 0,
        'gate_toggles': 0,
        'iterations': 0,
        'sentinel_checks': 0,
        'sentinel_corrupt': 0,
    }
    mismatches = out / 'mismatches.jsonl'
    mismatches.write_text('')
    problems_total = 0
    stream = torch.cuda.Stream()
    t_end = time.monotonic() + seconds
    done = 0
    previous: Step | None = None
    with torch.cuda.stream(stream):
        while time.monotonic() < t_end:
            plan_steps = drain + synthetic
            totals['iterations'] += 1
            chunk: list[dict[str, Any]] = []
            for step in plan_steps:
                if previous is not None and previous.gate != step.gate:
                    totals['gate_toggles'] += 1
                previous = step
                chunk.append(rig.run_step(len(chunk), step, pools))
                if len(chunk) == CHUNK:
                    found = rig.check_chunk(chunk, totals, done) + sentinels(rig, totals)
                    done += len(chunk)
                    chunk = []
                    problems_total += len(found)
                    with mismatches.open('a') as handle:
                        for p in found:
                            handle.write(json.dumps(p) + '\n')
            if chunk:
                found = rig.check_chunk(chunk, totals, done) + sentinels(rig, totals)
                done += len(chunk)
                problems_total += len(found)
                with mismatches.open('a') as handle:
                    for p in found:
                        handle.write(json.dumps(p) + '\n')
    summary = {
        'setup': rig.setup,
        'drain_steps_per_iteration': len(drain),
        'drain_gate_toggles_per_iteration': gate_toggles(drain),
        'synthetic_steps_per_iteration': len(synthetic),
        'totals': {
            **totals,
            'rows_by_size': {str(k): v for k, v in sorted(totals['rows_by_size'].items())},
        },
        'mismatch_records': problems_total,
        'probe_stats': rig.verify.head.probe_stats(),
        'seed': seed,
        'seconds': seconds,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(
        json.dumps({k: summary[k] for k in ('mismatch_records',)} | {'totals': summary['totals']})
    )
    return 0 if problems_total == 0 else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('context', help="579ae7ce's token ids from a point directory (CPU)")
    c.add_argument('--point', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    for name in ('start', 'stop'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    r = sub.add_parser('refs', help="579ae7ce's distribution at 439 under stock reference paths")
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    r.add_argument('--url', required=True)
    r.add_argument('--plant', type=Path, required=True, help='where to write the chosen suffix')
    f = sub.add_parser('fetch')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--contexts', type=Path, help='gross contexts (score_report --contexts)')
    s = sub.add_parser('stress')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--replay', type=Path, required=True, help="h6b certlog's replay.jsonl")
    s.add_argument('--seconds', type=float, default=480)
    s.add_argument('--seed', type=int, default=20261002)
    s.add_argument('--real-rows', type=int, default=16384)
    s.add_argument('--ties', type=int, default=2048)
    args = parser.parse_args(argv)
    if args.command == 'context':
        context(args.point, args.out)
        return 0
    if args.command == 'start':
        args.out.mkdir(parents=True, exist_ok=True)
        return start(args.out)
    if args.command == 'stop':
        from experiments.benchcert.rescore import stop

        return stop(args.out)
    if args.command == 'fetch':
        return fetch(args.out, args.contexts)
    if args.command == 'refs':
        from experiments.benchcert.drain import PLANT_CANDIDATES

        return refs(args.out, args.runs, args.url, PLANT_CANDIDATES, args.plant)
    return stress(args.out, args.replay, args.seconds, args.seed, args.real_rows, args.ties)


if __name__ == '__main__':
    sys.exit(main())
