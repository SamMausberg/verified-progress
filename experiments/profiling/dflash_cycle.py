"""Split the DFlash speculative cycle into phases, traced and untraced.

    python experiments/profiling/dflash_cycle.py --evidence evidence/profiles \
        --out evidence/profiles/dflash_cycle.json --csv evidence/profiles/dflash_cycle.csv

Inputs, for the bench's two tuned DFlash arms (``run_all.sh dflash``):

* ``attribution/<arm>_bs<C>.json`` (``attribute.py --kind dflash``): one traced
  window per concurrency C, a cycle running from one target-verify replay to the next;
* ``windows/<arm>_nsys.jsonl`` and ``windows/<arm>_none.jsonl`` with their
  ``_meta.json`` (``collect_run.py``): the client windows of the traced server
  (uncollected and collected) and of the untraced server (repeated windows).

Nothing is written unless every window the two runs' recorded commands planned is
present (``run_profiles.window_problems``), every window held its batch at C with no
request finishing inside it, and every planned concurrency has an attribution of its
own report with complete eager kernel records. A partial windows file therefore
stops the analysis instead of yielding a partial table. The traced and untraced runs
of an arm must also share the repository and SGLang commits, the GPU, the server
command and its environment (``pair_problems``), since the derived terms combine them.

Phases (``PHASES``) group the attribution's categories, which for DFlash are tied
to the graph or eager section a kernel runs in, so the phases and the two idle
terms sum to the traced cycle:

* ``draft_forward``: the block drafter's layers inside the draft graph;
* ``draft_context_kv``: the eager projection of the verified target features into
  the drafter's KV cache after each verify (GEMMs, norms, RoPE, KV writes);
* ``draft_head``: the drafter's projection through the target head and its argmax
  (folded into the draft graph);
* ``verify_forward``: the target-verify graph except its head (weight GEMMs, the GDN
  kernels writing one state per block position, attention, norms);
* ``verify_head``: the target head GEMM over the block, the FP32 logits copy and the
  eager argmax over the verify logits;
* ``gdn_commit``: the scatter of each request's accepted state into its slot;
* ``sampling_bookkeeping``: acceptance, other small eager kernels, copies, memsets;
* ``idle_in_graph``: GPU gaps between the nodes of graph replays;
* ``host_gap``: GPU idle outside graph replays, while the GPU waits for the host.

Untraced quantities come from the windows (measured): the cycle estimate
``ms_per_cycle_est`` = C x accept length / output tokens per second, with the accept
length from the server's log lines inside the window, and the mean completion length
at mid-window (``mid_window_tokens``). Derived quantities combine the two servers:

* ``untraced_minus_traced_work_ms``: the untraced cycle minus the traced cycle's time
  inside graph replays and eager kernels. It equals the untraced host gap only if both
  windows do the same GPU work; the untraced windows run at longer contexts, which adds
  attention work, so it overestimates that gap, and it carries the cycle estimator's
  error (``traced_cycle_over_collected_window_estimate`` compares the estimator with the
  trace on the collected window);
* ``head_share_untraced``: the head chains' traced time over the untraced cycle, with
  its Amdahl ceiling 1 / (1 - f).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_profiles import BENCH_ARMS, build_parser, read_windows, window_problems

PHASES: list[tuple[str, list[str]]] = [
    ('draft_forward', ['draft_model']),
    ('draft_context_kv', ['draft_context_kv']),
    ('draft_head', ['draft_lm_head_gemm', 'draft_logits_cast', 'spec_draft_topk',
                    'draft_argmax_eager']),
    ('verify_forward', ['gdn_in_proj_gemm', 'gdn_out_proj_gemm', 'gdn_conv', 'gdn_recurrent',
                        'gdn_gated_norm', 'gdn_state_track', 'attn_qkv_gemm',
                        'attn_o_proj_gemm', 'full_attention', 'attn_qk_norm_rope_gate',
                        'attn_output_gate', 'kv_cache_store', 'mlp_gate_up_gemm',
                        'mlp_down_gemm', 'mlp_activation', 'norm', 'embedding', 'other_gemm',
                        'other_in_graph']),
    ('verify_head', ['lm_head_gemm', 'logits_cast', 'sampling_argmax']),
    ('gdn_commit', ['spec_gdn_state']),
    ('sampling_bookkeeping', ['spec_verify', 'runtime_eager_small', 'memcpy_h2d', 'memcpy_d2h',
                              'memcpy_d2d', 'memset']),
    ('idle_in_graph', ['idle_in_graph']),
    ('host_gap', ['idle_outside_graph']),
]  # fmt: skip
HEAD_PHASES = ('verify_head', 'draft_head')
MIN_EAGER_RECORDS = 0.98  # attribute.py warns below this: CUPTI dropped records
FOREIGN_CPU_LIMIT = 2.0  # cores; the team's limit for host-sensitive timings


def load_run(evidence: Path, name: str, mode: str, problems: list[str]) -> tuple[dict, list]:
    """A run's recorded arguments and its window records, checked against its command."""
    wpath = evidence / 'windows' / f'{name}.jsonl'
    mpath = evidence / 'windows' / f'{name}_meta.json'
    if not wpath.exists() or not mpath.exists():
        problems.append(f'{name}: missing {wpath.name} or {mpath.name}')
        return {}, []
    meta = json.loads(mpath.read_text())
    args = build_parser().parse_args(meta['argv'][1:])
    rows, bad = read_windows(wpath)
    if args.mode != mode:
        problems.append(f'{name}: recorded mode {args.mode}, expected {mode}')
    problems += [
        f'{name}: {p}'
        for p in bad + window_problems(rows, args.arm, args.mode, args.concurrency, args.repeats)
    ]
    for r in rows:
        c = r['concurrency']
        if r.get('requests_finished_before_window_end'):
            problems.append(f'{name} c={c}: a request finished inside the window')
        if r.get('log_running_reqs') != [c]:
            problems.append(f'{name} c={c}: running batch {r.get("log_running_reqs")}, not {c}')
        if 'log_accept_len_mean' not in r:
            problems.append(f'{name} c={c}: no accept length in the server log')
    return {'args': args, 'meta': meta}, rows


def pair_problems(traced: dict, untraced: dict) -> list[str]:
    """Why a traced and an untraced run cannot be combined: the derived terms subtract one
    from the other, so both must come from the same code, engine, GPU and server command."""
    problems = []
    for key in ('repo_sha', 'sglang_sha'):
        if not traced.get(key) or traced.get(key) != untraced.get(key):
            problems.append(
                f'{key} differs ({traced.get(key)} traced, {untraced.get(key)} untraced)'
            )
    if traced.get('gpu') != untraced.get('gpu'):
        problems.append('the runs were made on different GPUs or clocks')
    server = shlex.split(untraced.get('server_command', ''))
    if not server or shlex.split(traced.get('server_command', ''))[-len(server) :] != server:
        problems.append('the traced server command is not the untraced one under nsys')
    if traced.get('env', {}) != untraced.get('env', {}):
        problems.append('the server environments differ')
    return problems


def window_stats(rows: list[dict]) -> dict:
    ms = [r['ms_per_cycle_est'] for r in rows]
    tps = [r['output_tokens_per_s'] for r in rows]
    c = rows[0]['concurrency']
    return {
        'windows': len(rows),
        'ms_per_cycle_mean': statistics.mean(ms),
        'ms_per_cycle_sd': statistics.stdev(ms) if len(ms) > 1 else 0.0,
        'output_tok_per_s_mean': statistics.mean(tps),
        'output_tok_per_s_sd': statistics.stdev(tps) if len(tps) > 1 else 0.0,
        'output_tok_per_s_per_user': statistics.mean(tps) / c,
        'accept_len_mean': statistics.mean(r['log_accept_len_mean'] for r in rows),
        'mid_window_tokens': statistics.mean(
            r['mean_completion_tokens_at_window_start'] + r['window_output_tokens'] / (2 * c)
            for r in rows
        ),
        'foreign_cpu_cores_max': max(r['cpu_cores_busy_foreign'] for r in rows),
    }


def phase_split(attr: dict, name: str, problems: list[str]) -> dict[str, float]:
    """Microseconds per traced cycle by phase; every category must belong to one phase."""
    phase_of = {cat: phase for phase, cats in PHASES for cat in cats}
    out = dict.fromkeys((p for p, _ in PHASES), 0.0)
    for row in attr['categories']:
        phase = phase_of.get(row['category'])
        if phase is None:
            problems.append(f'{name}: category {row["category"]} belongs to no phase')
            continue
        out[phase] += row['us_per_step']
    return out


def load_inputs(evidence: Path) -> dict[str, dict]:
    """Every input of every arm, after all checks; raises SystemExit listing each problem."""
    problems: list[str] = []
    inputs: dict[str, dict] = {}
    for arm in BENCH_ARMS:
        traced_run, traced = load_run(evidence, f'{arm}_nsys', 'nsys', problems)
        untraced_run, untraced = load_run(evidence, f'{arm}_none', 'none', problems)
        if not traced_run or not untraced_run:
            continue
        cs = traced_run['args'].concurrency
        if sorted(cs) != sorted(untraced_run['args'].concurrency):
            problems.append(f'{arm}: traced and untraced runs planned different concurrencies')
        problems += [f'{arm}: {p}' for p in pair_problems(traced_run['meta'], untraced_run['meta'])]
        attrs = {}
        for c in cs:
            path = evidence / 'attribution' / f'{arm}_bs{c}.json'
            if not path.exists():
                problems.append(f'{arm} c={c}: no {path.name}')
                continue
            attr = json.loads(path.read_text())
            s = attr['summary']
            if s.get('kind') != 'dflash' or s.get('report') != f'{arm}_bs{c}.nsys-rep':
                problems.append(
                    f'{arm} c={c}: {path.name} is not the DFlash attribution of {arm}_bs{c}'
                )
            if s['eager_kernel_records_per_launch_call'] < MIN_EAGER_RECORDS:
                problems.append(f'{arm} c={c}: CUPTI dropped eager kernel records')
            attr['phases_us'] = phase_split(attr, f'{arm}_bs{c}', problems)
            attrs[c] = attr
        for r in traced + untraced:
            if r['cpu_cores_busy_foreign'] > FOREIGN_CPU_LIMIT:
                problems.append(
                    f'{arm} c={r["concurrency"]} {r["mode"]}/{r["window_kind"]}: '
                    f'{r["cpu_cores_busy_foreign"]:.2f} foreign cores'
                )
        inputs[arm] = {
            'traced_run': traced_run,
            'traced': traced,
            'untraced_run': untraced_run,
            'untraced': untraced,
            'attrs': attrs,
        }
    if problems:
        raise SystemExit('refusing to write:\n  ' + '\n  '.join(problems))
    return inputs


def cycle_row(arm: str, c: int, attr: dict, traced: list[dict], untraced: list[dict]) -> dict:
    s = attr['summary']
    phases = attr['phases_us']
    cycle = s['step_us_mean']
    in_graphs_and_kernels = cycle - phases['host_gap']
    head = sum(phases[p] for p in HEAD_PHASES)
    un = window_stats([r for r in untraced if r['concurrency'] == c])
    unc = window_stats([r for r in traced if r['concurrency'] == c and r['window_kind'] == 'none'])
    col = window_stats([r for r in traced if r['concurrency'] == c and r['window_kind'] == 'nsys'])
    un_us = 1e3 * un['ms_per_cycle_mean']
    f_un = head / un_us
    return {
        'concurrency': c,
        'traced': {
            'cycles': s['steps'],
            'cycle_us': cycle,
            'cycle_us_p10': s['step_us_p10'],
            'cycle_us_p90': s['step_us_p90'],
            'gpu_busy_pct': s['gpu_busy_pct'],
            'phases_us': phases,
            'phases_pct': {p: 100 * v / cycle for p, v in phases.items()},
            'head_share': head / cycle,
            'head_share_of_gpu_busy': head / (cycle * s['gpu_busy_pct'] / 100),
            'host_lead_us_median_by_graph': s['host_lead_us_median_by_graph'],
            'verify_head_gemm': s['lm_head_gemm'],
            'draft_head_gemm': s['draft_lm_head_gemm'],
        },
        'collected_window': col,
        'uncollected_window': unc,
        'untraced': un,
        'derived': {
            'untraced_minus_traced_work_ms': (un_us - in_graphs_and_kernels) / 1e3,
            'untraced_minus_traced_work_share': (un_us - in_graphs_and_kernels) / un_us,
            'head_share_untraced': f_un,
            'head_ceiling_untraced': 1 / (1 - f_un),
            'verify_head_ceiling_untraced': 1 / (1 - phases['verify_head'] / un_us),
            'draft_head_ceiling_untraced': 1 / (1 - phases['draft_head'] / un_us),
            'traced_cycle_over_collected_window_estimate': cycle / (1e3 * col['ms_per_cycle_mean']),
        },
    }


def csv_row(arm: str, r: dict) -> dict:
    t, un, d = r['traced'], r['untraced'], r['derived']
    return {
        'arm': arm,
        'batch': r['concurrency'],
        'accept_len_untraced': round(un['accept_len_mean'], 3),
        'cycle_ms_untraced': round(un['ms_per_cycle_mean'], 3),
        'cycle_ms_untraced_sd': round(un['ms_per_cycle_sd'], 3),
        'cycle_ms_traced': round(t['cycle_us'] / 1e3, 3),
        'mid_window_tokens_untraced': round(un['mid_window_tokens']),
        'mid_window_tokens_traced': round(r['collected_window']['mid_window_tokens']),
        **{f'{p}_us': round(v, 1) for p, v in t['phases_us'].items()},
        'untraced_minus_traced_work_us': round(1e3 * d['untraced_minus_traced_work_ms'], 1),
        'head_pct_traced': round(100 * t['head_share'], 2),
        'head_pct_untraced': round(100 * d['head_share_untraced'], 2),
        'head_ceiling_untraced': round(d['head_ceiling_untraced'], 4),
    }


def analyze(evidence: Path) -> tuple[dict, list[dict]]:
    arms = {}
    rows = []
    for arm, x in load_inputs(evidence).items():
        cycles = [
            cycle_row(arm, c, x['attrs'][c], x['traced'], x['untraced'])
            for c in x['traced_run']['args'].concurrency
        ]
        rows += [csv_row(arm, r) for r in cycles]
        meta = x['untraced_run']['meta']
        arms[arm] = {
            'traced_server': x['traced_run']['meta']['server_command'],
            'untraced_server': meta['server_command'],
            'repo_sha': meta.get('repo_sha'),
            'sglang_sha': meta.get('sglang_sha'),
            'rows': cycles,
        }
    if not rows:
        raise SystemExit('no DFlash runs in the evidence')
    return {'phases': dict(PHASES), 'arms': arms}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--evidence', type=Path, default=Path('evidence/profiles'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    result, rows = analyze(args.evidence)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    with args.csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f'{r["arm"]:17s} B={r["batch"]:3d} untraced {r["cycle_ms_untraced"]:7.3f} ms '
            f'traced {r["cycle_ms_traced"]:7.3f} ms head {r["head_pct_untraced"]:5.1f}% '
            f'untraced - traced work {r["untraced_minus_traced_work_us"]:7.1f} us'
        )


if __name__ == '__main__':
    main()
