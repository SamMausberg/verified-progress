"""Load test of the lossy arms: does each one serve, and on which kernels?

Untimed. For every step it launches one arm through the bench harness (the same
launch checks as a sweep: decode and prefill CUDA graphs, overlap scheduler,
capacity, backends), then:

* reads from the server log the weight memory of the target (and drafter), the
  quantization method and the GDN state dtype;
* checks that the server tokenizes like the reference checkpoint: the token ids
  the server reports for 16 templated prompts must equal those of the
  Qwen/Qwen3.5-4B tokenizer (the INT4 checkpoint ships its own tokenizer.json);
* sends 8 greedy requests of 256 tokens (thinking on, concurrency 8) and records
  the output lengths, a text excerpt and, under speculation, the accept length
  (output tokens per verify pass) from the server's meta information;
* optionally (``nsys``) collects 20 scheduler steps with Nsight Systems
  (CUDA graph nodes traced) and lists the GPU kernels by time, to show which
  GEMM kernels the quantized weights actually run on;
* optionally (``probe``) runs the logit probe of experiments/moonshot/logit_probe.py
  in generate mode (and score mode against a named reference), which is how the
  reference's own run-to-run noise is measured.

    python -m experiments.lossy.load_test --out ~/vp-data/lossy/load_test

Steps are fixed in ``STEPS``. A step that fails is recorded and the next one runs;
the exit status is 1 if any step failed. Output: ``<out>/<UTC>/load_test.json``
plus one directory per step (server log, launch record, smoke outputs, nsys
report and kernel table, probe files).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.arms import Arm, resolve_arm, server_command
from bench.server import Server, git_state, gpu_snapshot, host_id, http_post, sglang_source
from experiments.moonshot.logit_probe import chat_ids, select_prompts

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / 'bench/workloads/mixed-v2/tune.jsonl'
REFERENCE_TOKENIZER = ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
PORT = 30101
SMOKE_PROMPTS = 8
SMOKE_TOKENS = 256
TOKENIZER_PROMPTS = 16
NSYS_STEPS = 20
PROBE = REPO / 'experiments/moonshot/logit_probe.py'

_WEIGHT_LINE = re.compile(
    r'Load weight end\. elapsed=[0-9.]+ s, type=(\w+), quant=([\w-]+|None), '
    r'avail mem=([0-9.]+) GB, mem usage=([0-9.]+) GB'
)
_STATE_LINE = re.compile(r'ssm_state size: ([0-9.]+)GB')
_INTERESTING = re.compile(
    r'(Compressed Tensors|compressed-tensors|marlin|Marlin|DFLASH|GDN kernel dispatcher|'
    r'Linear attention kernel backend|mamba_ssm_dtype|ssm_state size|KV Cache is allocated|'
    r'max_total_num_tokens=)'
)


@dataclass(frozen=True)
class Step:
    name: str
    arm: str
    sets: tuple[str, ...] = ()
    nsys: bool = False
    # 'generate' writes <step>/probe_generate.json; a score run needs a reference step.
    probe_generate: bool = False
    probe_score_against: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Load test L0 (experiments/lossy/README.md). The INT4 steps answer "does it
# serve, with graphs and overlap, on low-bit kernels"; the FP16-state steps
# check that the capacity-256 arms launch; plain-ref-1/2 measure the logit
# probe's reference and its run-to-run noise (no lossy arm is probed here).
STEPS: tuple[Step, ...] = (
    Step('int4-dflash-b16', 'int4-dflash-b16'),
    Step('int4-dflash-b16-nsys', 'int4-dflash-b16', nsys=True),
    Step('int4-dflash-b8', 'int4-dflash-b8'),
    Step('int4-plain-cap256', 'int4-plain-cap256'),
    Step('plain-cap256-fp16', 'plain-cap256-fp16'),
    Step('replayssm-cap256-fp16', 'replayssm-cap256-fp16'),
    Step('replayssm-cap256', 'replayssm-cap256'),
    Step('plain-ref-1', 'plain-cap256', probe_generate=True, probe_score_against='plain-ref-1'),
    Step('plain-ref-2', 'plain-cap256', probe_generate=True, probe_score_against='plain-ref-1'),
)


def reference_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER[0], revision=REFERENCE_TOKENIZER[1])


def log_facts(text: str) -> dict[str, Any]:
    weights = [
        {'type': m[0], 'quant': m[1], 'avail_gb': float(m[2]), 'usage_gb': float(m[3])}
        for m in _WEIGHT_LINE.findall(text)
    ]
    states = [float(x) for x in _STATE_LINE.findall(text)]
    lines = [line.strip()[:300] for line in text.splitlines() if _INTERESTING.search(line)]
    return {'weights': weights, 'ssm_state_gb': states, 'log_lines': lines[:40]}


def check_tokenizer(base_url: str, tok: Any, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """Token ids the server derives from the templated text vs the reference tokenizer."""
    mismatches = []
    for row in prompts:
        templated = tok.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
        )
        expected = chat_ids(tok, row['text'], True)
        body = {
            'text': templated,
            'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1},
            'return_logprob': True,
            'logprob_start_len': 0,
        }
        meta = json.loads(http_post(f'{base_url}/generate', body, timeout=120))['meta_info']
        got = [int(entry[1]) for entry in meta['input_token_logprobs']]
        if got != expected:
            mismatches.append({'id': row['id'], 'expected_len': len(expected), 'got_len': len(got)})
    return {'prompts': len(prompts), 'mismatches': mismatches, 'ok': not mismatches}


def smoke(base_url: str, tok: Any, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy fixed-length requests; output lengths, accept length, text excerpts."""

    def one(row: dict[str, Any]) -> dict[str, Any]:
        body = {
            'input_ids': chat_ids(tok, row['text'], True),
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': SMOKE_TOKENS,
                'ignore_eos': True,
            },
        }
        started = time.monotonic()
        result = json.loads(http_post(f'{base_url}/generate', body, timeout=600))
        meta = result['meta_info']
        return {
            'id': row['id'],
            'completion_tokens': meta.get('completion_tokens'),
            'spec_verify_ct': meta.get('spec_verify_ct'),
            'seconds': round(time.monotonic() - started, 2),
            'text_head': result.get('text', '')[:240],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        rows = list(pool.map(one, prompts))
    tokens = sum(int(r['completion_tokens'] or 0) for r in rows)
    verifies = sum(int(r['spec_verify_ct'] or 0) for r in rows)
    return {
        'requests': rows,
        'all_full_length': all(r['completion_tokens'] == SMOKE_TOKENS for r in rows),
        'accept_length': tokens / verifies if verifies else None,
    }


class NsysServer(Server):
    """A bench server launched under `nsys profile`, collecting only when the
    scheduler calls cudaProfilerStart (SGLang's /start_profile CUDA_PROFILER).
    Collection ends after NSYS_STEPS steps and nsys then shuts the server down."""

    report: Path

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        env = self.environment()
        self.report = self.out_dir / 'profile'
        nsys = [
            'nsys',
            'profile',
            '--trace=cuda',
            '--cuda-graph-trace=node',
            '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop-shutdown',
            '--force-overwrite=true',
            '-o',
            str(self.report),
        ]
        command = nsys + server_command(self.arm, self.python, self.host, self.port)
        self.launch_record = {
            'arm': self.arm.to_json(),
            'command': command,
            'env_overrides': self.arm.env,
            'pythonpath': env.get('PYTHONPATH', ''),
            'sglang_worktree': env.get('SGLANG_WORKTREE'),
            'sglang_source': sglang_source(self.python, env),
            'repo': git_state(REPO),
            'host_id': host_id(),
            'gpu_before_start': gpu_snapshot(),
            'start_time_unix': time.time(),
        }
        self._write_json('launch.json', self.launch_record)
        self._log_handle = self.log_path.open('wb')
        self.proc = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def kernel_table(report: Path, out_dir: Path) -> dict[str, Any]:
    """`nsys stats` GPU kernel summary; the top kernels and the GEMM families."""
    rep = report.with_suffix('.nsys-rep')
    if not rep.exists():
        return {'error': f'{rep} not written'}
    prefix = out_dir / 'kernels'
    subprocess.run(
        [
            'nsys',
            'stats',
            '--report',
            'cuda_gpu_kern_sum',
            '--format',
            'csv',
            '--force-export=true',
            '--output',
            str(prefix),
            str(rep),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(out_dir.glob('kernels*cuda_gpu_kern_sum*.csv'))
    if not files:
        return {'error': 'nsys stats wrote no kernel table'}
    with files[0].open() as handle:
        rows = list(csv.DictReader(handle))
    total = sum(float(r['Total Time (ns)']) for r in rows) or 1.0

    def share(pattern: str) -> float:
        rx = re.compile(pattern, re.IGNORECASE)
        return sum(float(r['Total Time (ns)']) for r in rows if rx.search(r['Name'])) / total

    return {
        'csv': str(files[0]),
        'kernels': len(rows),
        'top': [
            {'name': r['Name'][:160], 'share': float(r['Total Time (ns)']) / total}
            for r in rows[:25]
        ],
        'share_marlin': share(r'marlin'),
        # dense BF16 GEMM families SGLang/cuBLAS use on sm_90 (the LM head stays BF16)
        'share_dense_gemm': share(r'nvjet|sm90_xmma|cutlass.*gemm|gemm.*bf16|ampere_bf16'),
    }


def run_probe(base_url: str, out: Path, mode: str, label: str, reference: Path | None) -> None:
    command = [
        sys.executable,
        str(PROBE),
        'run',
        '--url',
        base_url,
        '--mode',
        mode,
        '--out',
        str(out),
        '--label',
        label,
        '--concurrency',
        '16',
    ]
    if reference is not None:
        command += ['--reference', str(reference)]
    subprocess.run(command, check=True, cwd=REPO)


def run_step(step: Step, root: Path, worktree: Path, tok: Any) -> dict[str, Any]:
    out = root / step.name
    arm: Arm = resolve_arm(step.arm)
    record: dict[str, Any] = {'step': step.name, 'arm': step.arm, 'start_unix': time.time()}
    cls = NsysServer if step.nsys else Server
    server = cls(arm, out / 'server', PORT, sglang_worktree=worktree, strict=False)
    try:
        with server:
            record['checks'] = [
                {'name': c.name, 'ok': c.ok, 'required': c.required, 'detail': c.detail}
                for c in server.checks
            ]
            record['launch_ok'] = all(c.ok for c in server.checks if c.required)
            record['final_limits'] = server.launch_record.get('final_limits')
            record['sglang_source'] = server.launch_record.get('sglang_source')
            prompts = select_prompts(WORKLOAD, 6)
            record['tokenizer'] = check_tokenizer(
                server.base_url, tok, prompts[:TOKENIZER_PROMPTS]
            )
            record['smoke'] = smoke(server.base_url, tok, prompts[:SMOKE_PROMPTS])
            if step.probe_generate:
                gen = out / 'probe_generate.json'
                run_probe(server.base_url, gen, 'generate', step.name, None)
                record['probe_generate'] = str(gen)
            if step.probe_score_against:
                ref = root / step.probe_score_against / 'probe_generate.json'
                score = out / 'probe_score.json'
                run_probe(server.base_url, score, 'score', step.name, ref)
                record['probe_score'] = str(score)
            if step.nsys:
                record['nsys'] = profile_kernels(server, out, tok, prompts)
            record['log_facts'] = log_facts(server.log_text())
    except Exception as exc:  # noqa: BLE001 - a failed step is a result
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['traceback'] = traceback.format_exc()[-3000:]
        if server.log_path.exists():
            record['log_facts'] = log_facts(server.log_text())
            record['log_tail'] = server.log_tail(30)
    record['end_unix'] = time.time()
    record['ok'] = bool(record.get('launch_ok')) and 'error' not in record
    return record


def profile_kernels(
    server: Server, out: Path, tok: Any, prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep 4 requests decoding, collect NSYS_STEPS scheduler steps, list kernels."""
    assert isinstance(server, NsysServer) and server.proc is not None

    def long_request(row: dict[str, Any]) -> None:
        body = {
            'input_ids': chat_ids(tok, row['text'], True),
            'sampling_params': {'temperature': 0.0, 'max_new_tokens': 2048, 'ignore_eos': True},
        }
        try:
            http_post(f'{server.base_url}/generate', body, timeout=600)
        except Exception:  # noqa: BLE001 - nsys shuts the server down mid-request
            pass

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    for row in prompts[:4]:
        pool.submit(long_request, row)
    time.sleep(5.0)
    body = {'activities': ['CUDA_PROFILER'], 'num_steps': NSYS_STEPS}
    started = http_post(f'{server.base_url}/start_profile', body, timeout=60)
    try:
        server.proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(server.proc.pid), signal.SIGINT)
        server.proc.wait(timeout=120)
    pool.shutdown(wait=False, cancel_futures=True)
    return {'start_profile': started.strip(), 'steps': NSYS_STEPS, **kernel_table(server.report, out)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=Path.home() / 'vp-data/lossy/load_test')
    parser.add_argument('--steps', nargs='+', default=None, help='subset of STEPS, in order')
    args = parser.parse_args(argv)
    worktree_env = os.environ.get('SGLANG_WORKTREE')
    if not worktree_env:
        raise SystemExit('set SGLANG_WORKTREE to the engine worktree with the lossy patch')
    worktree = Path(worktree_env)
    known = {step.name: step for step in STEPS}
    names = args.steps or [step.name for step in STEPS]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f'unknown steps {unknown}; known: {sorted(known)}')
    root = args.out.expanduser() / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    root.mkdir(parents=True, exist_ok=False)
    tok = reference_tokenizer()
    summary: dict[str, Any] = {
        'repo': git_state(REPO),
        'engine': git_state(worktree),
        'engine_worktree': str(worktree),
        'steps': [],
    }
    path = root / 'load_test.json'
    for name in names:
        print(f'=== {name}', flush=True)
        record = run_step(known[name], root, worktree, tok)
        summary['steps'].append(record)
        path.write_text(json.dumps(summary, indent=1, default=str) + '\n')
        status = 'ok' if record['ok'] else f'FAILED {record.get("error", "launch checks")}'
        smoke_info = record.get('smoke') or {}
        print(
            f'    {status}; accept {smoke_info.get("accept_length")}; '
            f'tokenizer {(record.get("tokenizer") or {}).get("ok")}',
            flush=True,
        )
    failed = [s['step'] for s in summary['steps'] if not s['ok']]
    summary['failed'] = failed
    path.write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print(f'wrote {path}; failed: {failed}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
