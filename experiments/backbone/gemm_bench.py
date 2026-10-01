"""Time Qwen3.5-4B's backbone weight GEMMs at decode batch sizes, and alternatives.

SGLang computes every backbone projection with ``F.linear(x, W)`` (BF16 in and
out, FP32 accumulation), which PyTorch sends to cuBLAS (``cublasGemmEx``). This
script replays that call with the real checkpoint weights under CUDA graphs for
each projection and row count M, and compares it with kernels that read the same
BF16 weights:

* ``cublas``: the stock call (PyTorch's default BLAS backend is cuBLAS).
* ``cublas_nored``: the same with
  ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False``
  (reports whether the stock split-K kernels reduce in reduced precision).
* ``cublaslt``: ``preferred_blas_library('cublaslt')``, i.e. cublasLtMatmul with
  the first heuristic algorithm (what ``TORCH_BLAS_PREFER_CUBLASLT=1`` selects).
* ``lt``: every algorithm cuBLASLt's heuristic returns (up to ``--lt-count``),
  through ``lt_algos.cpp``; the best one and the best without split-K are kept.
* ``sgl_gemv``: SGLang's own Hopper GEMV (``hopper_bf16_gemv``, M = 1 only).
* ``triton``: the skinny GEMM of the engine patch
  (``sglang/srt/layers/backbone_gemm.py`` in ``~/sglang-wt/backbone``) over a
  configuration sweep, screened with few replays and the best few re-timed.

Timing: one CUDA graph calls the arm once per layer, each layer with its own
weight and input (``L`` calls; the weights of all layers together exceed the
60 MB L2, as in the model), and is replayed ``--inner`` times between two CUDA
events; the per-call time is the elapsed time over ``inner * L``. It therefore
includes the gap between consecutive kernels inside a graph and any split-K
reduction kernel. ``--repeats`` such measurements give the median and spread.
Efficiency is weight bytes over per-call time, against the measured HBM read
peak (``--hbm-json``).

The ``norm`` and ``chain`` subcommands time the RMSNorm kernels and short layer
skeletons (norm, projections, activation) in which launches, PDL and the fused
prologues interact.

Run under the exclusive GPU lock, with the engine worktree on the path:

    SGLANG_WORKTREE=~/sglang-wt/backbone source scripts/sglang_env.sh
    scripts/gpu_lock.sh -x python experiments/backbone/gemm_bench.py gemm \\
        --out evidence/backbone/gemm_microbench.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = Path.home() / (
    '.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/'
    '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
LAYER_PREFIX = 'model.language_model.layers.'
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
SCRATCH_FLOATS = 8 << 20  # backbone_gemm's default split-K workspace
REPO = Path(__file__).resolve().parents[2]

# name: (layers it occurs in, checkpoint tensors concatenated along N). The
# concatenation order follows SGLang's packed parameters; it does not affect time.
PROJECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    'gdn_in_proj_qkvz': (
        'linear_attention',
        ('linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight'),
    ),
    'gdn_in_proj_ba': (
        'linear_attention',
        ('linear_attn.in_proj_b.weight', 'linear_attn.in_proj_a.weight'),
    ),
    'gdn_in_proj_merged': (
        'linear_attention',
        (
            'linear_attn.in_proj_qkv.weight',
            'linear_attn.in_proj_z.weight',
            'linear_attn.in_proj_b.weight',
            'linear_attn.in_proj_a.weight',
        ),
    ),
    'gdn_out_proj': ('linear_attention', ('linear_attn.out_proj.weight',)),
    'attn_qkv_proj': (
        'full_attention',
        ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'),
    ),
    'attn_o_proj': ('full_attention', ('self_attn.o_proj.weight',)),
    'mlp_gate_up': ('all', ('mlp.gate_proj.weight', 'mlp.up_proj.weight')),
    'mlp_down': ('all', ('mlp.down_proj.weight',)),
    'mtp_fc': ('mtp', ('mtp.fc.weight',)),
}


# ----------------------------------------------------------------------------
# Weights and environment


class Checkpoint:
    def __init__(self, device: str) -> None:
        self.device = device
        index = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text())
        self.weight_map: dict[str, str] = index['weight_map']
        config = json.loads((MODEL_DIR / 'config.json').read_text())['text_config']
        self.config = config
        self.layer_types: list[str] = config['layer_types']

    def tensor(self, key: str) -> torch.Tensor:
        with safe_open(
            str(MODEL_DIR / self.weight_map[key]), framework='pt', device=self.device
        ) as f:
            return f.get_tensor(key)

    def layers(self, kind: str) -> list[int]:
        if kind == 'all':
            return list(range(len(self.layer_types)))
        if kind == 'mtp':
            return [0]
        return [i for i, t in enumerate(self.layer_types) if t == kind]

    def projection(self, name: str) -> list[torch.Tensor]:
        kind, parts = PROJECTIONS[name]
        out = []
        for layer in self.layers(kind):
            prefix = '' if kind == 'mtp' else f'{LAYER_PREFIX}{layer}.'
            ws = [self.tensor(prefix + p) for p in parts]
            w = torch.cat(ws, dim=0).contiguous() if len(ws) > 1 else ws[0].contiguous()
            assert w.dtype == torch.bfloat16
            out.append(w)
        return out

    def norm_weight(self, layer: int, which: str) -> torch.Tensor:
        return self.tensor(f'{LAYER_PREFIX}{layer}.{which}.weight').contiguous()


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def git_dirty(path: Path) -> bool:
    try:
        out = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


def environment(args: argparse.Namespace) -> dict[str, Any]:
    import sglang
    import triton

    sglang_dir = Path(sglang.__file__).resolve().parents[2]
    clocks = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw',
            '--format=csv,noheader',
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        'command': ' '.join([sys.executable, *sys.argv]),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'gpu': torch.cuda.get_device_name(),
        'sm_count': torch.cuda.get_device_properties(0).multi_processor_count,
        'clocks_at_start': clocks,
        'torch': torch.__version__,
        'triton': triton.__version__,
        'cuda': torch.version.cuda,
        'cublaslt_version': cublaslt_version(),
        'repo_commit': git_sha(REPO),
        'repo_dirty': git_dirty(REPO),
        'sglang_dir': str(sglang_dir),
        'sglang_commit': git_sha(sglang_dir),
        'sglang_dirty': git_dirty(sglang_dir),
        'model': f'Qwen/Qwen3.5-4B@{MODEL_REVISION}',
        'inner': args.inner,
        'repeats': args.repeats,
        'peak_tb_per_s': args.peak,
    }


def cublaslt_version() -> int:
    import ctypes

    import nvidia

    lib = ctypes.CDLL(str(Path(nvidia.__path__[0]) / 'cu13/lib/libcublasLt.so.13'))
    lib.cublasLtGetVersion.restype = ctypes.c_size_t
    return int(lib.cublasLtGetVersion())


def read_peak(path: str) -> float:
    data = json.loads(Path(path).read_text())
    return float(data['kernels']['read_head_size']['tb_per_s_median'])


# ----------------------------------------------------------------------------
# Timing


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, inner: int, repeats: int) -> list[float]:
    """Microseconds per replay, one value per repeat."""
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1e3 / inner)
    return out


def stats(values: Sequence[float], scale: float = 1.0) -> dict[str, float]:
    ordered = sorted(v * scale for v in values)
    n = len(ordered)
    return {
        'median_us': statistics.median(ordered),
        'p10_us': ordered[n // 10],
        'p90_us': ordered[(9 * n) // 10],
        'min_us': ordered[0],
        'n': n,
    }


def kernels_of(fn: Callable[[], object]) -> list[dict[str, Any]]:
    """CUDA kernels launched by one eager call: name, grid, block, duration."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        prof.export_chrome_trace(tmp.name)
        trace = json.loads(Path(tmp.name).read_text())
    out = []
    for ev in trace.get('traceEvents', []):
        if ev.get('cat') != 'kernel':
            continue
        a = ev.get('args', {})
        out.append(
            {
                'name': ev['name'],
                'grid': a.get('grid'),
                'block': a.get('block'),
                'eager_us': ev.get('dur'),
            }
        )
    return out


# ----------------------------------------------------------------------------
# Arms


def load_lt(build_dir: Path) -> Any:
    import nvidia
    import torch.utils.cpp_extension as ce

    cu = Path(nvidia.__path__[0]) / 'cu13'
    build_dir.mkdir(parents=True, exist_ok=True)
    return ce.load(
        name='lt_algos',
        sources=[str(Path(__file__).with_name('lt_algos.cpp'))],
        extra_include_paths=[str(cu / 'include')],
        extra_ldflags=[f'-L{cu / "lib"}', '-l:libcublasLt.so.13', f'-Wl,-rpath,{cu / "lib"}'],
        build_directory=str(build_dir),
        with_cuda=True,
    )


def triton_candidates(m: int, n: int, k: int) -> list[Any]:
    """Configurations worth screening for one problem (decode-sized M).

    One tile height per M range (the dot path pads M to 16), plus an FMA path
    without tensor-core padding for M <= 2; tile width, depth, split-K and
    pipeline depth are swept, keeping grids of roughly one to six waves.
    """
    from sglang.srt.layers.backbone_gemm import GemmConfig

    cands = []
    if m <= 2:
        for bn, bk, sk in itertools.product((8, 16, 32), (256, 512), (1, 2, 4)):
            if m * bn * bk // 128 <= 64:  # FP32 partials per thread at 4 warps
                cands.append(GemmConfig(m, bn, bk, sk, False, 4, 3))
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    for bn, bk, sk, nw, ns in itertools.product(
        (16, 32, 64, 128), (64, 128, 256), (1, 2, 4, 8), (4, 8), (3, 5)
    ):
        if (bk == 64 and bm < 64) or (nw == 8 and bm < 64):
            continue
        if (bm + bn) * bk * 2 * ns > 200 * 1024:
            continue
        cands.append(GemmConfig(bm, bn, bk, sk, True, nw, ns))
    out = []
    for c in cands:
        if not c.valid_for(m, n, k):
            continue
        gx, gy, gz = c.grid(m, n)
        if c.split_k > 1 and n * gz * c.block_m * c.split_k > SCRATCH_FLOATS:
            continue  # would not fit the kernel's split-K workspace
        if 100 <= gx * gy * gz <= 800:
            out.append(c)
    return out


def precompile(configs: list[Any], workers: int, verified_path: Path) -> float:
    """Compile and check every configuration, with and without PDL, in worker
    processes; the configurations that pass are appended to ``verified_path``.

    Each worker runs its configurations on a small problem against a float64
    product (split-K also for run-to-run bitwise equality) and, for PDL, inside a
    captured graph. A configuration that faults kills only its worker, and the
    ones after it in that worker's share stay unverified, so the main process
    never launches an unchecked kernel.
    """
    t0 = time.time()
    uniq = sorted({json.dumps(asdict(c), sort_keys=True) for c in configs})
    verified_path.write_text('')
    procs = []
    for i in range(workers):
        chunk = uniq[i::workers]
        if chunk:
            procs.append(
                subprocess.Popen(
                    [sys.executable, __file__, '_compile', json.dumps(chunk), str(verified_path)],
                    stdout=subprocess.DEVNULL,
                )
            )
    for proc in procs:
        proc.wait()
    return time.time() - t0


def repeat_gemm(x: torch.Tensor, w: torch.Tensor, cfg: Any, out: torch.Tensor, n: int) -> None:
    from sglang.srt.layers import backbone_gemm as bg

    for _ in range(n):
        bg.skinny_gemm(x, w, cfg, out=out)


def read_verified(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def config_key(cfg: Any) -> str:
    return json.dumps(asdict(cfg), sort_keys=True)


def compile_worker(chunk_json: str, verified_path: str) -> None:
    from sglang.srt.layers import backbone_gemm as bg

    torch.manual_seed(0)
    for spec in json.loads(chunk_json):
        base = json.loads(spec)
        for pdl in (False, True):
            cfg = bg.GemmConfig(**{**base, 'pdl': pdl})
            n, k = cfg.block_n * 4, cfg.block_k * cfg.split_k * 4
            ok = True
            for m in sorted({1, cfg.block_m, cfg.block_m + 1}):
                if not cfg.valid_for(m, n, k):
                    continue
                x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
                w = (torch.randn(n, k, device='cuda') * 0.05).to(torch.bfloat16)
                ref = x.double() @ w.double().T
                try:
                    y1 = bg.skinny_gemm(x, w, cfg)
                    y2 = bg.skinny_gemm(x, w, cfg)
                    if pdl:
                        out = torch.empty_like(y1)
                        graph = capture(functools.partial(repeat_gemm, x, w, cfg, out, 3))
                        graph.replay()
                        torch.cuda.synchronize()
                        y2 = out
                except Exception as e:
                    print(f'failed {cfg}: {e!r}', file=sys.stderr)
                    ok = False
                    break
                err = float((y1.double() - ref).abs().max() / ref.abs().max())
                if err > 1e-2 or not torch.equal(y1, y2):
                    print(f'wrong {cfg}: rel err {err:.3g}', file=sys.stderr)
                    ok = False
                    break
            if ok:
                with open(verified_path, 'a') as f:
                    f.write(json.dumps(asdict(cfg), sort_keys=True) + '\n')


def error_stats(y: torch.Tensor, ref64: torch.Tensor, stock: torch.Tensor) -> dict[str, float]:
    err = (y.double() - ref64).abs()
    return {
        'max_abs_err': float(err.max()),
        'max_rel_err': float(err.max() / ref64.abs().max().clamp_min(1e-30)),
        'bitwise_equal_to_cublas': float(
            (y.view(torch.int16) == stock.view(torch.int16)).float().mean()
        ),
    }


class Problem:
    """One projection at one M: its layers' weights, inputs and the stock output."""

    def __init__(self, args: argparse.Namespace, weights: list[torch.Tensor], m: int) -> None:
        self.args = args
        self.weights = weights
        self.m = m
        self.n, self.k = weights[0].shape
        self.layers = len(weights)
        self.wbytes = self.n * self.k * 2
        self.xs = [
            torch.randn(m, self.k, device='cuda', dtype=torch.bfloat16) for _ in range(self.layers)
        ]
        self.ys = [
            torch.empty(m, self.n, device='cuda', dtype=torch.bfloat16) for _ in range(self.layers)
        ]
        self.ref64 = self.xs[0].double() @ weights[0].double().T
        self.stock = F.linear(self.xs[0], weights[0])
        self.floor_us = self.wbytes / (args.peak * 1e12) * 1e6

    def linear(self, i: int) -> torch.Tensor:
        return F.linear(self.xs[i], self.weights[i])

    def record(
        self,
        arm: str,
        call: Callable[[int], torch.Tensor],
        extra: dict[str, Any] | None = None,
        *,
        repeats: int,
        kernels: bool = True,
        setup: Callable[[], object] | None = None,
        teardown: Callable[[], object] | None = None,
    ) -> dict[str, Any]:
        """Check one arm against the float64 product, then time it in a graph."""
        layers = range(self.layers)
        if setup:
            setup()
        try:
            errs = error_stats(call(0), self.ref64, self.stock)
            graph = capture(lambda: [call(i) for i in layers])
            per = time_graph(graph, self.args.inner, repeats)
            del graph
            kern = kernels_of(lambda: call(0)) if kernels else None
        except Exception as e:  # an algorithm that cannot run is reported, not fatal
            return {'m': self.m, 'arm': arm, 'error': repr(e)[:400], **(extra or {})}
        finally:
            if teardown:
                teardown()
        st = stats(per, 1.0 / self.layers)
        return {
            'm': self.m,
            'arm': arm,
            **(extra or {}),
            'us_per_call': st,
            'tb_per_s': self.wbytes / (st['median_us'] * 1e-6) / 1e12,
            'frac_of_peak': self.floor_us / st['median_us'],
            **errs,
            'kernels': kern,
        }

    def stock_arms(self) -> list[dict[str, Any]]:
        reps = self.args.repeats
        rows = [self.record('cublas', self.linear, repeats=reps)]

        def no_reduced() -> None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        def reduced() -> None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

        rows.append(
            self.record(
                'cublas_nored', self.linear, repeats=reps, setup=no_reduced, teardown=reduced
            )
        )

        def lt_on() -> None:
            torch.backends.cuda.preferred_blas_library('cublaslt')

        def lt_off() -> None:
            torch.backends.cuda.preferred_blas_library('cublas')

        rows.append(
            self.record('cublaslt', self.linear, repeats=reps, setup=lt_on, teardown=lt_off)
        )
        if self.m == 1 and not self.args.skip_sgl_gemv:
            from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv

            def gemv(i: int) -> torch.Tensor:
                return hopper_bf16_gemv(self.xs[i], self.weights[i])

            rows.append(self.record('sgl_gemv', gemv, repeats=reps))
        return rows

    def lt_call(
        self, lt: Any, blob: torch.Tensor, ws: torch.Tensor
    ) -> Callable[[int], torch.Tensor]:
        def call(i: int) -> torch.Tensor:
            lt.run(blob, self.xs[i], self.weights[i], self.ys[i], ws)
            return self.ys[i]

        return call

    def lt_arms(self, lt: Any, ws: torch.Tensor) -> list[dict[str, Any]]:
        """Screen every heuristic algorithm, then re-time the best two and the best
        one without split-K."""
        algos, attrs, waves, attr_names = lt.heuristics(
            self.m, self.n, self.k, ws.numel(), self.args.lt_count
        )
        screened = []
        for a in range(algos.shape[0]):
            info = dict(zip(attr_names, attrs[a].tolist(), strict=True))
            info['waves'] = float(waves[a])
            info['heuristic_rank'] = a
            row = self.record(
                f'lt{a}',
                self.lt_call(lt, algos[a].clone(), ws),
                {'lt': info, 'screen': True},
                repeats=self.args.screen_repeats,
                kernels=False,
            )
            screened.append(row)
        ok = sorted(
            (r for r in screened if 'us_per_call' in r), key=lambda r: r['us_per_call']['median_us']
        )
        no_split = [r for r in ok if r['lt']['splitk_num'] in (0, 1)]
        chosen = ok[:2] + [r for r in no_split[:1] if r not in ok[:2]]
        rows = list(screened)
        for r in chosen:
            info = {k: v for k, v in r['lt'].items()}
            row = self.record(
                'lt_best' if r is ok[0] else 'lt_retime',
                self.lt_call(lt, algos[info['heuristic_rank']].clone(), ws),
                {'lt': info, 'lt_no_split_best': bool(no_split) and r is no_split[0]},
                repeats=self.args.repeats,
                kernels=True,
            )
            rows.append(row)
        return rows

    def triton_call(self, cfg: Any) -> Callable[[int], torch.Tensor]:
        from sglang.srt.layers import backbone_gemm as bg

        def call(i: int) -> torch.Tensor:
            return bg.skinny_gemm(self.xs[i], self.weights[i], cfg, out=self.ys[i])

        return call

    def triton_arms(self) -> list[dict[str, Any]]:
        """Screen the sweep, then re-time the best few with and without PDL."""
        from sglang.srt.layers import backbone_gemm as bg

        screened = [
            self.record(
                'triton_screen',
                self.triton_call(cfg),
                {'config': asdict(cfg)},
                repeats=self.args.screen_repeats,
                kernels=False,
            )
            for cfg in triton_candidates(self.m, self.n, self.k)
            if config_key(cfg) in self.args.verified
        ]
        ok = sorted(
            (r for r in screened if 'us_per_call' in r and r['max_rel_err'] < 1e-2),
            key=lambda r: r['us_per_call']['median_us'],
        )
        rows = list(screened)
        for rank, r in enumerate(ok[: self.args.retime]):
            for pdl in (False, True):
                cfg = bg.GemmConfig(**{**r['config'], 'pdl': pdl})
                if config_key(cfg) not in self.args.verified:
                    continue
                rows.append(
                    self.record(
                        'triton_pdl' if pdl else 'triton',
                        self.triton_call(cfg),
                        {
                            'config': asdict(cfg),
                            'screen_rank': rank,
                            'grid': list(cfg.grid(self.m, self.n)),
                        },
                        repeats=self.args.repeats,
                        kernels=False,
                    )
                )
        return rows


def projection_shape(ckpt: Checkpoint, name: str) -> tuple[int, int]:
    kind, parts = PROJECTIONS[name]
    prefix = '' if kind == 'mtp' else f'{LAYER_PREFIX}{ckpt.layers(kind)[0]}.'
    n = sum(ckpt.tensor(prefix + p).shape[0] for p in parts)
    return n, ckpt.tensor(prefix + parts[0]).shape[1]


def bench_gemm(args: argparse.Namespace) -> dict[str, Any]:
    ckpt = Checkpoint('cuda')
    lt = None if args.skip_lt else load_lt(Path(args.build_dir))
    lt_ws = torch.empty(32 << 20, dtype=torch.uint8, device='cuda')
    names = args.projections or list(PROJECTIONS)
    m_values = args.m or list(M_VALUES)
    torch.manual_seed(0)

    # Compile the whole Triton sweep first, in parallel processes.
    shapes = {name: projection_shape(ckpt, name) for name in names}
    compile_s = 0.0
    verified_path = Path(args.out).with_suffix('.verified_configs')
    verified_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_triton:
        sweep = [c for nm in names for m in m_values for c in triton_candidates(m, *shapes[nm])]
        compile_s = precompile(sweep, args.compile_workers, verified_path)
        args.verified = read_verified(verified_path)
        print(
            f'triton: {len(args.verified)} of {2 * len({config_key(c) for c in sweep})} '
            f'configurations verified in {compile_s:.0f} s',
            flush=True,
        )

    result: dict[str, Any] = {
        'meta': {**environment(args), 'triton_compile_s': compile_s},
        'projections': {},
    }
    for name in names:
        weights = ckpt.projection(name)
        n, k = weights[0].shape
        entry: dict[str, Any] = {
            'n': n,
            'k': k,
            'layers': len(weights),
            'weight_bytes_per_layer': n * k * 2,
            'rows': [],
        }
        result['projections'][name] = entry
        for m in m_values:
            prob = Problem(args, weights, m)
            entry['rows'].extend(prob.stock_arms())
            if lt is not None:
                entry['rows'].extend(prob.lt_arms(lt, lt_ws))
            if not args.skip_triton:
                entry['rows'].extend(prob.triton_arms())
            summary = summarize_m(entry['rows'], m)
            print(
                f'{name:20s} m={m:4d} '
                + ' '.join(f'{a}={v["us"]:.2f}' for a, v in summary.items()),
                flush=True,
            )
            del prob
        del weights
        torch.cuda.empty_cache()
    result['summary'] = {
        name: {str(m): summarize_m(e['rows'], m) for m in m_values}
        for name, e in result['projections'].items()
    }
    return result


def summarize_m(rows: list[dict[str, Any]], m: int) -> dict[str, Any]:
    """Best timed row per arm family at one M (screen rows excluded)."""
    out: dict[str, Any] = {}
    fam = {
        'cublas': 'cublas',
        'cublas_nored': 'cublas_nored',
        'cublaslt': 'cublaslt',
        'sgl_gemv': 'sgl_gemv',
        'lt_best': 'lt',
        'lt_retime': 'lt',
        'triton': 'triton',
        'triton_pdl': 'triton_pdl',
    }
    for r in rows:
        if r.get('m') != m or r.get('arm') not in fam or 'us_per_call' not in r:
            continue
        f = fam[r['arm']]
        us = r['us_per_call']['median_us']
        if f not in out or us < out[f]['us']:
            out[f] = {
                'us': us,
                'tb_per_s': r['tb_per_s'],
                'frac_of_peak': r['frac_of_peak'],
                'max_rel_err': r['max_rel_err'],
                'bitwise_equal_to_cublas': r['bitwise_equal_to_cublas'],
                **({'config': r['config']} if 'config' in r else {}),
                **({'lt': r['lt']} if 'lt' in r else {}),
            }
    return out


# ----------------------------------------------------------------------------
# RMSNorm kernels


NormFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]


def run_norms(
    fn: NormFn, xs: list[torch.Tensor], rs: list[torch.Tensor], gws: list[torch.Tensor]
) -> None:
    for x, r, g in zip(xs, rs, gws, strict=True):
        fn(x, r, g)


def bench_norm(args: argparse.Namespace) -> dict[str, Any]:
    import flashinfer.norm as fnorm
    from sgl_kernel import gemma_fused_add_rmsnorm

    ckpt = Checkpoint('cuda')
    hidden = ckpt.config['hidden_size']
    eps = ckpt.config['rms_norm_eps']
    n_layers = 2 * len(ckpt.layer_types)
    gws = [
        ckpt.norm_weight(i // 2, 'input_layernorm' if i % 2 == 0 else 'post_attention_layernorm')
        for i in range(n_layers)
    ]
    cuda_mod = fnorm.get_norm_module()
    arms: dict[str, NormFn] = {
        'flashinfer_cute_pdl': lambda x, r, w: gemma_fused_add_rmsnorm(x, r, w, eps),
        'flashinfer_cute_nopdl': lambda x, r, w: fnorm.gemma_fused_add_rmsnorm(
            x, r, w, eps, enable_pdl=False
        ),
        'flashinfer_cuda_pdl': lambda x, r, w: cuda_mod.gemma_fused_add_rmsnorm(x, r, w, eps, True),
        'flashinfer_cuda_nopdl': lambda x, r, w: cuda_mod.gemma_fused_add_rmsnorm(
            x, r, w, eps, False
        ),
    }
    result: dict[str, Any] = {
        'meta': environment(args),
        'norm_calls_per_graph': n_layers,
        'rows': [],
    }
    for m in args.m or list(M_VALUES):
        torch.manual_seed(m)
        x0 = [torch.randn(m, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(n_layers)]
        r0 = [
            (torch.randn(m, hidden, device='cuda') * 4).to(torch.bfloat16) for _ in range(n_layers)
        ]
        ref_x = ref_r = None
        for arm, fn in arms.items():
            xs = [t.clone() for t in x0]
            rs = [t.clone() for t in r0]
            try:
                fn(xs[0], rs[0], gws[0])
                if ref_x is None:
                    ref_x, ref_r = xs[0].clone(), rs[0].clone()
                assert ref_r is not None
                eq_x = float((xs[0].view(torch.int16) == ref_x.view(torch.int16)).float().mean())
                eq_r = float((rs[0].view(torch.int16) == ref_r.view(torch.int16)).float().mean())
                # In place: every replay normalizes the previous replay's output,
                # which does not change the work done.
                graph = capture(functools.partial(run_norms, fn, xs, rs, gws))
                per = time_graph(graph, args.inner, args.repeats)
                kern = kernels_of(functools.partial(fn, xs[0], rs[0], gws[0]))
            except Exception as e:
                result['rows'].append({'m': m, 'arm': arm, 'error': repr(e)[:400]})
                continue
            st = stats(per, 1.0 / n_layers)
            result['rows'].append(
                {
                    'm': m,
                    'arm': arm,
                    'us_per_call': st,
                    'bitwise_equal_out_to_stock': eq_x,
                    'bitwise_equal_residual_to_stock': eq_r,
                    'kernels': kern,
                }
            )
            print(f'norm m={m:4d} {arm:24s} {st["median_us"]:.2f} us  eq={eq_x:.4f}', flush=True)
    return result


# ----------------------------------------------------------------------------
# Layer skeletons


def best_configs(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """(projection, m) -> best Triton configuration from a gemm result file."""
    data = json.loads(path.read_text())
    out = {}
    for name, per_m in data['summary'].items():
        for m, arms in per_m.items():
            if 'triton' in arms:
                out[(name, int(m))] = arms['triton']['config']
    return out


class Skeletons:
    """Two layer skeletons, chained over the model's layers, at one M.

    mlp: post-attention norm (fused residual add) -> gate_up -> SiLU-mul -> down
         over the 32 layers (each down output feeds the next norm).
    gdn: input norm -> in_proj (qkvz and, on a side stream, ba; or the merged
         weight) -> out_proj on the first 4096 columns of qkvz, over the 24 GDN
         layers (convolution, recurrence and gated norm omitted).
    Variants replace the GEMMs with the Triton kernel (best isolated config),
    add PDL, and fold the norm and the SiLU-mul into the GEMM prologues.
    """

    def __init__(self, ckpt: Checkpoint, cfgs: dict[tuple[str, int], dict[str, Any]]):
        self.cfgs = cfgs
        self.m = 0  # set before each use
        self.eps = ckpt.config['rms_norm_eps']
        self.all_layers = list(range(len(ckpt.layer_types)))
        self.gdn_layers = ckpt.layers('linear_attention')
        self.w_gu = ckpt.projection('mlp_gate_up')
        self.w_dn = ckpt.projection('mlp_down')
        self.w_qkvz = ckpt.projection('gdn_in_proj_qkvz')
        self.w_ba = ckpt.projection('gdn_in_proj_ba')
        self.w_inm = [
            torch.cat([a, b]).contiguous() for a, b in zip(self.w_qkvz, self.w_ba, strict=True)
        ]
        self.w_out = ckpt.projection('gdn_out_proj')
        self.g_post = [ckpt.norm_weight(i, 'post_attention_layernorm') for i in self.all_layers]
        self.g_in = [ckpt.norm_weight(i, 'input_layernorm') for i in self.gdn_layers]
        self.alt = torch.cuda.Stream()

    def cfg(self, name: str, pdl: bool) -> Any:
        from sglang.srt.layers.backbone_gemm import GemmConfig

        c = self.cfgs.get((name, self.m))
        if c is None:
            raise KeyError(f'no Triton config for {name} m={self.m}')
        return GemmConfig(**{**c, 'pdl': pdl})

    def mlp_stock(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        from sgl_kernel import gemma_fused_add_rmsnorm
        from sglang.kernels.ops.activation.activation import silu_and_mul

        x, r = state['x'], state['r']
        for i in self.all_layers:
            gemma_fused_add_rmsnorm(x, r, self.g_post[i], self.eps)
            x = F.linear(silu_and_mul(F.linear(x, self.w_gu[i])), self.w_dn[i])
        return x

    def mlp_triton(self, state: dict[str, torch.Tensor], pdl: bool) -> torch.Tensor:
        from sgl_kernel import gemma_fused_add_rmsnorm
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.srt.layers import backbone_gemm as bg

        x, r = state['x'], state['r']
        c_gu, c_dn = self.cfg('mlp_gate_up', pdl), self.cfg('mlp_down', pdl)
        for i in self.all_layers:
            gemma_fused_add_rmsnorm(x, r, self.g_post[i], self.eps)
            gu = bg.skinny_gemm(x, self.w_gu[i], c_gu)
            x = bg.skinny_gemm(silu_and_mul(gu), self.w_dn[i], c_dn)
        return x

    def mlp_fused(self, state: dict[str, torch.Tensor], pdl: bool) -> torch.Tensor:
        from sglang.srt.layers import backbone_gemm as bg

        x, r, r2 = state['x'], state['r'], state['r2']
        c_gu, c_dn = self.cfg('mlp_gate_up', pdl), self.cfg('mlp_down', pdl)
        for i in self.all_layers:
            gu = bg.skinny_gemm(
                x,
                self.w_gu[i],
                c_gu,
                prologue=bg.PROLOGUE_ADD_RMSNORM,
                residual=r,
                norm_weight=self.g_post[i],
                eps=self.eps,
                residual_out=r2,
            )
            r, r2 = r2, r
            x = bg.skinny_gemm(gu, self.w_dn[i], c_dn, prologue=bg.PROLOGUE_SILU_MUL)
        return x

    def gdn_stock(self, state: dict[str, torch.Tensor], merged: bool) -> torch.Tensor:
        from sgl_kernel import gemma_fused_add_rmsnorm

        x, r = state['x'], state['r']
        for j, _ in enumerate(self.gdn_layers):
            gemma_fused_add_rmsnorm(x, r, self.g_in[j], self.eps)
            if merged:
                qkvz = F.linear(x, self.w_inm[j])
            else:
                cur = torch.cuda.current_stream()
                self.alt.wait_stream(cur)
                qkvz = F.linear(x, self.w_qkvz[j])
                with torch.cuda.stream(self.alt):
                    F.linear(x, self.w_ba[j])
                cur.wait_stream(self.alt)
            x = F.linear(qkvz[:, :4096], self.w_out[j])
        return x

    def gdn_triton(self, state: dict[str, torch.Tensor], pdl: bool, fused: bool) -> torch.Tensor:
        from sgl_kernel import gemma_fused_add_rmsnorm
        from sglang.srt.layers import backbone_gemm as bg

        x, r, r2 = state['x'], state['r'], state['r2']
        c_in, c_out = self.cfg('gdn_in_proj_merged', pdl), self.cfg('gdn_out_proj', pdl)
        for j, _ in enumerate(self.gdn_layers):
            if fused:
                qkvz = bg.skinny_gemm(
                    x,
                    self.w_inm[j],
                    c_in,
                    prologue=bg.PROLOGUE_ADD_RMSNORM,
                    residual=r,
                    norm_weight=self.g_in[j],
                    eps=self.eps,
                    residual_out=r2,
                )
                r, r2 = r2, r
            else:
                gemma_fused_add_rmsnorm(x, r, self.g_in[j], self.eps)
                qkvz = bg.skinny_gemm(x, self.w_inm[j], c_in)
            x = bg.skinny_gemm(qkvz[:, :4096], self.w_out[j], c_out)
        return x

    def variants(self) -> dict[str, tuple[int, Callable[[dict[str, torch.Tensor]], torch.Tensor]]]:
        nm, ng = len(self.all_layers), len(self.gdn_layers)
        return {
            'mlp_stock': (nm, self.mlp_stock),
            'mlp_triton': (nm, functools.partial(self.mlp_triton, pdl=False)),
            'mlp_triton_pdl': (nm, functools.partial(self.mlp_triton, pdl=True)),
            'mlp_fused': (nm, functools.partial(self.mlp_fused, pdl=False)),
            'mlp_fused_pdl': (nm, functools.partial(self.mlp_fused, pdl=True)),
            'gdn_stock': (ng, functools.partial(self.gdn_stock, merged=False)),
            'gdn_stock_merged': (ng, functools.partial(self.gdn_stock, merged=True)),
            'gdn_triton': (ng, functools.partial(self.gdn_triton, pdl=False, fused=False)),
            'gdn_triton_pdl': (ng, functools.partial(self.gdn_triton, pdl=True, fused=False)),
            'gdn_fused': (ng, functools.partial(self.gdn_triton, pdl=False, fused=True)),
            'gdn_fused_pdl': (ng, functools.partial(self.gdn_triton, pdl=True, fused=True)),
        }


def bench_chain(args: argparse.Namespace) -> dict[str, Any]:
    ckpt = Checkpoint('cuda')
    cfgs = best_configs(Path(args.gemm_json))
    hidden = ckpt.config['hidden_size']
    result: dict[str, Any] = {
        'meta': environment(args),
        'gemm_configs_from': args.gemm_json,
        'rows': [],
    }
    sk = Skeletons(ckpt, cfgs)
    for m in args.m or list(M_VALUES):
        sk.m = m
        torch.manual_seed(100 + m)
        x_init = torch.randn(m, hidden, device='cuda', dtype=torch.bfloat16)
        r_init = (torch.randn(m, hidden, device='cuda') * 4).to(torch.bfloat16)
        ref_out: dict[str, torch.Tensor] = {}
        for name, (n_layers, fn) in sk.variants().items():
            state = {'x': x_init.clone(), 'r': r_init.clone(), 'r2': torch.empty_like(r_init)}
            try:
                # One eager pass from the initial state, for the numerical comparison.
                out = fn({key: v.clone() for key, v in state.items()}).clone()
                ref = ref_out.setdefault(name.split('_')[0], out)
                rel = float((out.float() - ref.float()).abs().max() / ref.float().abs().max())
                graph = capture(functools.partial(fn, state))
                per = time_graph(graph, args.inner, args.repeats)
                del graph
            except Exception as e:
                result['rows'].append({'m': m, 'variant': name, 'error': repr(e)[:400]})
                print(f'chain m={m} {name} failed: {e!r}', flush=True)
                continue
            st = stats(per, 1.0 / n_layers)
            result['rows'].append(
                {
                    'm': m,
                    'variant': name,
                    'layers': n_layers,
                    'us_per_layer': st,
                    'max_rel_diff_to_stock_after_chain': rel,
                }
            )
            print(f'chain m={m:4d} {name:18s} {st["median_us"]:.2f} us/layer', flush=True)
    return result


KEPT_ARMS = (
    'cublas',
    'cublas_nored',
    'cublaslt',
    'sgl_gemv',
    'lt_best',
    'lt_retime',
    'triton',
    'triton_pdl',
)


def compact_gemm(result: dict[str, Any]) -> dict[str, Any]:
    """The committed form: timed rows only; screens reduced to their timings."""
    out: dict[str, Any] = {'meta': result['meta'], 'projections': {}, 'summary': result['summary']}
    for name, e in result['projections'].items():
        rows = [r for r in e['rows'] if r.get('arm') in KEPT_ARMS]
        lt_screen = [
            {
                'm': r['m'],
                'rank': r['lt']['heuristic_rank'],
                'splitk_num': r['lt']['splitk_num'],
                'tile_id': r['lt']['tile_id'],
                'stages_id': r['lt']['stages_id'],
                'reduction_scheme': r['lt']['reduction_scheme'],
                'us': round(r['us_per_call']['median_us'], 3) if 'us_per_call' in r else None,
            }
            for r in e['rows']
            if r.get('screen') and str(r.get('arm', '')).startswith('lt')
        ]
        tri_screen: dict[str, Any] = {}
        for r in e['rows']:
            if r.get('arm') == 'triton_screen':
                d = tri_screen.setdefault(str(r['m']), {'screened': 0, 'failed': 0})
                d['screened'] += 1
                d['failed'] += 'us_per_call' not in r
        out['projections'][name] = {
            **{k: v for k, v in e.items() if k != 'rows'},
            'rows': rows,
            'lt_screen': lt_screen,
            'triton_screen': tri_screen,
        }
    return out


def main() -> None:
    if len(sys.argv) > 3 and sys.argv[1] == '_compile':
        compile_worker(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('what', choices=('gemm', 'norm', 'chain'))
    ap.add_argument('--out', required=True, help='full result (raw, may exceed 1 MB)')
    ap.add_argument('--evidence', help='compact result for evidence/ (gemm)')
    ap.add_argument('--m', type=int, nargs='*')
    ap.add_argument('--projections', nargs='*', choices=list(PROJECTIONS))
    ap.add_argument('--inner', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--screen-repeats', type=int, default=5)
    ap.add_argument('--retime', type=int, default=3, help='Triton configs re-timed per problem')
    ap.add_argument('--lt-count', type=int, default=32)
    ap.add_argument('--compile-workers', type=int, default=24)
    ap.add_argument('--build-dir', default=str(Path.home() / 'vp-data/backbone/build'))
    ap.add_argument('--hbm-json', default=str(REPO / 'evidence/profiles/hbm_bandwidth.json'))
    ap.add_argument('--gemm-json', help='gemm result with the Triton configs (chain)')
    ap.add_argument('--skip-lt', action='store_true')
    ap.add_argument('--skip-triton', action='store_true')
    ap.add_argument('--skip-sgl-gemv', action='store_true')
    args = ap.parse_args()
    args.peak = read_peak(args.hbm_json)
    if args.what == 'gemm':
        result = bench_gemm(args)
    elif args.what == 'norm':
        result = bench_norm(args)
    else:
        if not args.gemm_json:
            ap.error('chain needs --gemm-json')
        result = bench_chain(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
    print('wrote', args.out)
    if args.evidence:
        compact = compact_gemm(result) if args.what == 'gemm' else result
        Path(args.evidence).parent.mkdir(parents=True, exist_ok=True)
        Path(args.evidence).write_text(json.dumps(compact, indent=1) + '\n')
        print('wrote', args.evidence)


if __name__ == '__main__':
    main()
