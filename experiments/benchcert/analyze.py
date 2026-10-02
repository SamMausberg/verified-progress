"""Analysis of the served certified-head benchmark (experiments/benchcert/README.md).

    python -m experiments.benchcert.analyze predict --out evidence/certified_head/served/predictions.json
    python -m experiments.benchcert.analyze report --runs ~/vp-data/benchcert \\
        --out evidence/certified_head/served
    python -m experiments.benchcert.analyze replacement --runs ~/vp-data/benchcert

`predict` turns the head microbenchmark and the confirmation frontier (both
committed) into the expected decode-rate ratio of each certified arm; it ran
before any timed launch. `report` reads the holds' manifests, checks every launch
and point against the plan (plan.py) and writes the evidence: paired ratios with
95% intervals and the pre-registered decisions, token comparisons against the
stock arm and the stock noise floor, the check-mode counters, the certified
head's counters in the timed launches, the graph-capture memory of every launch,
the figure data and the figures. `replacement` prints the families whose
sessions s1-s3 left fewer than three valid pairs at some concurrency (the only
case in which session s4 runs).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.divergence import rate_interval, ratio_interval
from bench.pareto import invalid_reason
from bench.results import iter_jsonl, prompt_hash
from experiments.benchcert import plan

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
DECISION_N = 3
# t(1 - 0.05 / 8, 2): two-sided 95% across the four primary points (displayed intervals).
T_BONFERRONI4 = 8.860
NEAR_NATS = 0.5  # experiments/state_safety/compare.py
CLASSES = ('tie', 'one_ulp', 'near', 'large')

_CERT_LINE = re.compile(
    r'Certified LM head on (?P<paths>[\w, ]+) \(fallback (?P<fallback>\w+), '
    r'stock error model (?P<model>[\w-]+), check (?P<check>\w+)\)'
)
_CAPTURE_END = re.compile(
    r'Capture (?P<graph>(?:target|draft) \w+) CUDA graph end\. elapsed=(?P<elapsed>[\d.]+) s, '
    r'mem usage=(?P<mem>[\d.]+) GB, avail mem=(?P<avail>[\d.]+) GB'
)
_FINAL = re.compile(r'max_total_num_tokens=(\d+),.*?available_gpu_mem=([\d.]+) GB')
_MAMBA = re.compile(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)')
_OOM = ('OutOfMemoryError', 'CUDA out of memory')


# ---------------------------------------------------------------------------
# Predictions (derived, written before the timed runs)
# ---------------------------------------------------------------------------


def _interp(table: dict[int, float], m: float) -> float:
    """Linear interpolation of a per-M table (M within its range)."""
    keys = sorted(table)
    if m <= keys[0]:
        return table[keys[0]]
    for low, high in itertools.pairwise(keys):
        if low <= m <= high:
            return table[low] + (table[high] - table[low]) * (m - low) / (high - low)
    raise ValueError(f'M = {m} outside the head table ({keys[0]}-{keys[-1]})')


def head_calls(family: plan.Family, c: int) -> list[tuple[str, int]]:
    """(path, rows) of every head call in one decode step or speculative cycle."""
    if family.name == 'plain':
        return [('decode', c)]
    if family.name == 'mtp':
        # Three-step chain: two draft steps inside the draft graph, the draft-extend
        # token, and one verify over steps + 1 rows per request.
        return [('draft', c), ('draft', c), ('draft_extend', c), ('verify', 4 * c)]
    rows = family.rows_per_request
    return [('dflash_draft', rows['dflash_draft'] * c), ('verify', rows['verify'] * c)]


def predict(head_csv: Path, frontier_csv: Path) -> dict[str, Any]:
    """Expected decode-rate and y ratios of each certified arm (see the README)."""
    with head_csv.open() as handle:
        head = list(csv.DictReader(handle))
    stock = {int(r['M']): float(r['stock_us']) for r in head}
    cert = {int(r['M']): float(r['columns_expected_us']) for r in head}
    with frontier_csv.open() as handle:
        frontier = {(r['label'], int(r['concurrency'])): r for r in csv.DictReader(handle)}
    limit = int(plan.CERT_ENV['SGLANG_CERTIFIED_HEAD_MAX_ROWS'])
    out: dict[str, Any] = {
        'inputs': {'head': str(head_csv), 'frontier': str(frontier_csv)},
        'model': (
            'per call: saving = stock_us(M) - columns_expected_us(M) for M <= MAX_ROWS, 0 above;'
            ' cycle time T = accept_length / x_decode (1 / x_decode for plain);'
            ' decode ratio = T / (T - sum of savings); y ratio = 1 / (f + (1 - f) / decode ratio)'
            ' with f = 1 - x_e2e / x_decode, the time share before the first token'
        ),
        'points': [],
    }
    for family in plan.FAMILIES.values():
        for c in family.concurrency:
            row = frontier.get((family.arm, c))
            if row is None:
                continue
            x_decode, x_e2e = float(row['x_decode_mean']), float(row['x_e2e_mean'])
            accept = float(row['accept_length_mean'] or 1.0) if family.name != 'plain' else 1.0
            cycle_us = 1e6 * accept / x_decode
            calls = head_calls(family, c)
            saving = sum(_interp(stock, m) - _interp(cert, m) for _, m in calls if m <= limit)
            decode_ratio = cycle_us / (cycle_us - saving)
            f = 1.0 - x_e2e / x_decode
            out['points'].append(
                {
                    'family': family.name,
                    'arm': family.arm,
                    'concurrency': c,
                    'head_calls': [
                        {'path': p, 'rows': m, 'certified': m <= limit} for p, m in calls
                    ],
                    'cycle_us': cycle_us,
                    'saving_us': saving,
                    'decode_ratio': decode_ratio,
                    'prefill_share': f,
                    'y_ratio': 1.0 / (f + (1.0 - f) / decode_ratio),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Loading and checking launches
# ---------------------------------------------------------------------------


def load_holds(runs: Path) -> list[dict[str, Any]]:
    """Every launch recorded by the holds' manifests, tagged with its hold."""
    entries = []
    for path in sorted((runs / 'holds').glob('*.json')):
        record = json.loads(path.read_text())
        for entry in record.get('launches', []):
            entries.append(
                {**entry, 'hold': record['hold'], 'provenance': record.get('provenance', {})}
            )
    return entries


def index_launches(entries: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict]:
    """(step, family, variant) -> its successful launch; two successes is an error."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (entry['step'], entry['family'], entry['variant'])
        if entry.get('exit_code') != 0:
            index.setdefault(key, entry)
            continue
        if key in index and index[key].get('exit_code') == 0:
            raise SystemExit(f'two successful launches of {key}')
        index[key] = entry
    return index


def parse_server_log(text: str) -> dict[str, Any]:
    cert = _CERT_LINE.search(text)
    final = _FINAL.findall(text)
    mamba = _MAMBA.findall(text)
    captures: dict[str, dict[str, float]] = {}
    for match in _CAPTURE_END.finditer(text):
        captures[match['graph']] = {
            'elapsed_s': float(match['elapsed']),
            'mem_gb': float(match['mem']),
            'avail_after_gb': float(match['avail']),
        }
    return {
        'cert_paths': tuple(p.strip() for p in cert['paths'].split(',')) if cert else None,
        'cert_fallback': cert['fallback'] if cert else None,
        'cert_model': cert['model'] if cert else None,
        'cert_check': cert['check'] if cert else None,
        'cert_disabled': 'Certified LM head disabled' in text,
        'max_total_num_tokens': int(final[-1][0]) if final else None,
        'available_gpu_mem_gb': float(final[-1][1]) if final else None,
        'mamba_slots': int(mamba[-1]) if mamba else None,
        'captures': captures,
        'oom': any(marker in text for marker in _OOM),
    }


def load_launch(entry: dict[str, Any]) -> dict[str, Any]:
    """The launch's records (sweep manifest, launch record, server log)."""
    info: dict[str, Any] = {**entry, 'points': [], 'problems': []}
    run_dir = Path(entry['run_dir']) if entry.get('run_dir') else None
    info['run'] = run_dir.name if run_dir else None
    log_text = ''
    if run_dir and (run_dir / 'server/server.log').exists():
        log_text = (run_dir / 'server/server.log').read_text(errors='replace')
    info['server'] = parse_server_log(log_text)
    if run_dir is None or not (run_dir / 'sweep.json').exists():
        info['problems'].append('no sweep manifest')
        return info
    manifest = json.loads((run_dir / 'sweep.json').read_text())
    info['manifest'] = manifest
    info['points'] = manifest.get('points', [])
    launch = manifest.get('launch') or {}
    info['final_limits'] = launch.get('final_limits') or {}
    info['memory_usage'] = launch.get('memory_usage') or {}
    info['ready_after_s'] = launch.get('ready_after_s')
    info['sglang_source'] = launch.get('sglang_source') or {}
    info['graph_sizes'] = {
        key: value.get('sizes') for key, value in (launch.get('graph_captures') or {}).items()
    }
    info['repo_state'] = launch.get('repo') or {}
    info['checks_failed'] = [
        c['name'] for c in manifest.get('checks', []) if c.get('required') and not c.get('ok')
    ]
    gpu = (launch.get('gpu_after_ready') or {}).get('values') or []
    info['gpu_used_after_ready_mib'] = _mib(gpu[3]) if len(gpu) > 3 else None
    return info


def _mib(text: str) -> float | None:
    match = re.match(r'([\d.]+)\s*MiB', str(text))
    return float(match.group(1)) if match else None


def config_problems(
    info: dict[str, Any], family: plan.Family, variant: str, all_levels: bool = True
) -> list[str]:
    """Ways a launch did not run its declared configuration (head, engine, levels).

    A launch with any of these cannot stand for its arm in a token comparison either:
    a certified launch without the head would trivially match the stock arm. With
    `all_levels=False` (token comparisons) a launch cut short after some levels
    still counts for the levels it completed.
    """
    problems = []
    server = info['server']
    if variant == 'stock' and server['cert_paths'] is not None:
        problems.append('stock launch ran the certified head')
    if variant in ('cert', 'check'):
        if server['cert_disabled'] or server['cert_paths'] is None:
            problems.append('certified head not installed')
        else:
            if tuple(sorted(server['cert_paths'])) != tuple(sorted(family.paths)):
                problems.append(f'certified paths {server["cert_paths"]}, declared {family.paths}')
            if server['cert_fallback'] != plan.CERT_ENV['SGLANG_CERTIFIED_HEAD_FALLBACK']:
                problems.append(f'fallback {server["cert_fallback"]}')
            if server['cert_model'] != plan.CERT_ENV['SGLANG_CERTIFIED_HEAD_MODEL']:
                problems.append(f'stock error model {server["cert_model"]}')
            if (server['cert_check'] == 'True') != (variant == 'check'):
                problems.append(f'check mode {server["cert_check"]}')
    source = info.get('sglang_source') or {}
    prov = info.get('provenance') or {}
    if source:
        if source.get('head') != prov.get('engine_commit') or source.get('dirty_files'):
            problems.append(
                f'engine {source.get("head")} (hold declared {prov.get("engine_commit")})'
            )
    elif info.get('manifest'):
        problems.append('no engine record')
    if info.get('points'):
        levels = sorted(int(p['concurrency']) for p in info['points'])
        wanted = sorted(family.check_concurrency if variant == 'check' else family.concurrency)
        if levels != wanted if all_levels else not set(levels) <= set(wanted):
            problems.append(f'concurrency levels {levels}, declared {wanted}')
    return problems


def launch_problems(info: dict[str, Any], family: plan.Family, variant: str) -> list[str]:
    """Why a launch cannot stand for its arm in a timed comparison (empty when it can)."""
    problems = list(info.get('problems', []))
    if info.get('exit_code') != 0:
        problems.append(f'exit {info.get("exit_code")}')
    if info['server']['oom']:
        problems.append('out of memory')
    if info.get('checks_failed'):
        problems.append(f'launch checks failed: {", ".join(info["checks_failed"])}')
    return problems + config_problems(info, family, variant)


def outputs_complete(point: dict[str, Any]) -> bool:
    """Every measured request finished with the requested length (timing aside)."""
    return (
        point.get('failed') == 0
        and not point.get('osl_mismatch')
        and point.get('aiperf_exit_code') == 0
        and int(point.get('completed') or 0) > 0
    )


def point_problems(point: dict[str, Any], capacity: int) -> str:
    """bench.pareto's validity rule plus the pool and record checks of this campaign."""
    reasons = [invalid_reason(point)] if invalid_reason(point) else []
    if point.get('foreign_cpu_during_mean') is None:
        reasons.append('no foreign-load record')
    log = point.get('server_log') or {}
    if log.get('kv_retractions'):
        reasons.append(f'{log["kv_retractions"]} KV retractions')
    c = int(point['concurrency'])
    top = log.get('max_running_logged')
    if top is None:
        reasons.append('no running-batch record')
    elif top < min(c, capacity):
        reasons.append(f'running requests peaked at {top} (c = {c})')
    return '; '.join(reasons)


def pools(info: dict[str, Any]) -> tuple[Any, ...]:
    server = info['server']
    limits = info.get('final_limits') or {}
    return (
        limits.get('max_total_num_tokens', server['max_total_num_tokens']),
        limits.get('max_running_requests'),
        server['mamba_slots'],
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def ratio_summary(ratios: list[float]) -> dict[str, Any]:
    """Geometric mean of paired ratios with a 95% t interval on the log scale."""
    n = len(ratios)
    if n == 0:
        return {'n': 0, 'mean': math.nan, 'low': math.nan, 'high': math.nan, 'sd_log': math.nan}
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    if n == 1:
        return {
            'n': 1,
            'mean': math.exp(mean),
            'low': math.nan,
            'high': math.nan,
            'sd_log': math.nan,
        }
    sd = statistics.stdev(logs)
    half = T95[n - 1] * sd / math.sqrt(n)
    return {
        'n': n,
        'mean': math.exp(mean),
        'low': math.exp(mean - half),
        'high': math.exp(mean + half),
        'sd_log': sd,
    }


def _adjusted(summary: dict[str, Any], t: float, side: int) -> float:
    """One end of a t interval with quantile `t` (NaN below three pairs)."""
    if summary['n'] != DECISION_N:
        return math.nan
    half = t * summary['sd_log'] / math.sqrt(DECISION_N)
    return math.exp(math.log(summary['mean']) + side * half)


def t_test_p(summary: dict[str, Any]) -> float:
    """Two-sided p of mean log ratio = 0 for three pairs (Student t, 2 degrees of freedom).

    With 2 degrees of freedom the t distribution has the closed form
    P(|T| > t) = 1 - t / sqrt(t^2 + 2).
    """
    if summary['n'] != DECISION_N:
        return 1.0
    mean = math.log(summary['mean'])
    if summary['sd_log'] == 0:
        return 0.0 if mean != 0 else 1.0
    t = abs(mean) / (summary['sd_log'] / math.sqrt(DECISION_N))
    return 1.0 - t / math.sqrt(t * t + 2.0)


def holm(primaries: dict[str, dict[str, Any]], alpha: float = 0.05) -> dict[str, str]:
    """gain / loss / null per family from Holm's procedure over the declared primaries.

    The family size is the number of declared primaries whether or not each is
    complete; an incomplete primary counts as not rejected and reads `incomplete`.
    """
    m = len(primaries)
    order = sorted(primaries, key=lambda fam: t_test_p(primaries[fam]))
    decisions = {
        fam: 'incomplete' if primaries[fam]['n'] < DECISION_N else 'null' for fam in primaries
    }
    for rank, fam in enumerate(order):
        if primaries[fam]['n'] < DECISION_N or t_test_p(primaries[fam]) > alpha / (m - rank):
            break
        decisions[fam] = 'gain' if primaries[fam]['mean'] > 1.0 else 'loss'
    return decisions


def interval_reading(summary: dict[str, Any]) -> str:
    """Where a descriptive point's 95% interval lies (not a decision)."""
    if summary['n'] < DECISION_N:
        return 'incomplete'
    if summary['low'] > 1.0:
        return 'above 1'
    if summary['high'] < 1.0:
        return 'below 1'
    return 'includes 1'


def exactness_status(c1: bool | None, c1_required: bool, check: bool | None) -> str:
    """`fails` on positive evidence of a difference, `established` on positive
    evidence of equality (check mode, and concurrency-1 identity where the family has
    that point), `incomplete` otherwise."""
    if c1 is False or check is False:
        return 'fails'
    if check is True and (c1 is True or not c1_required):
        return 'established'
    return 'incomplete'


def family_verdict(primary: str, exact: str) -> str:
    if exact == 'fails':
        return 'fails exactness'
    if primary == 'incomplete':
        return 'incomplete'
    if primary == 'gain':
        return 'improves' if exact == 'established' else 'gain, exactness incomplete'
    if primary == 'loss':
        return 'loses'
    return 'no detectable change'


def h4_verdict(verdicts: dict[str, str]) -> str:
    """Supported if a family improves (gain at its primary point, exactness
    established); incomplete while a family's verdict is pending; refuted otherwise."""
    if any(v == 'improves' for v in verdicts.values()):
        return 'supported'
    if any(v in ('incomplete', 'gain, exactness incomplete') for v in verdicts.values()):
        return 'incomplete'
    return 'refuted'


# ---------------------------------------------------------------------------
# Token comparisons
# ---------------------------------------------------------------------------


def request_ids(point_dir: Path) -> dict[str, dict[str, list[int]]]:
    """Prompt hash -> prompt and output token ids of every measured request of a point."""
    raw = point_dir / 'aiperf/profile_export_raw.jsonl.gz'
    if not raw.exists():
        raw = point_dir / 'aiperf/profile_export_raw.jsonl'
    out: dict[str, dict[str, list[int]]] = {}
    for record in iter_jsonl(raw):
        if record.get('metadata', {}).get('benchmark_phase') != 'profiling':
            continue
        messages = record.get('payload', {}).get('messages') or [{}]
        key = prompt_hash(messages[-1].get('content', ''))
        found: dict[str, list[int]] = {}
        for response in record.get('responses', []):
            for packet in response.get('packets', []):
                value = packet.get('value')
                if isinstance(value, str) and value.startswith('{') and '_ids' in value:
                    ext = json.loads(value).get('sglext') or {}
                    if ext.get('output_ids'):
                        found['output'] = list(ext['output_ids'][0])
                    if ext.get('input_ids'):
                        found['input'] = list(ext['input_ids'])
        if 'output' not in found:
            raise ValueError(f'{point_dir}: a request has no output ids (--return-token-ids?)')
        if key in out:
            raise ValueError(f'{point_dir}: prompt {key} measured twice')
        out[key] = found
    return out


def output_ids(point_dir: Path) -> dict[str, list[int]]:
    """Prompt hash -> output token ids of every measured request of a point."""
    return {key: ids['output'] for key, ids in request_ids(point_dir).items()}


def compare(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, Any]:
    """First-divergence counts of two runs over the same prompts."""
    if set(a) != set(b):
        raise ValueError('the two runs measured different prompts')
    diverged, exposure, length_mismatch = 0, 0, 0
    positions: list[int] = []
    events: list[tuple[str, int, int, int]] = []
    for key in sorted(a):
        ta, tb = a[key], b[key]
        n = min(len(ta), len(tb))
        d = next((i for i in range(n) if ta[i] != tb[i]), None)
        if d is None:
            exposure += n
            length_mismatch += len(ta) != len(tb)
        else:
            diverged += 1
            exposure += d + 1
            positions.append(d)
            events.append((key, d, ta[d], tb[d]))
    return {
        'prompts': len(a),
        'identical': len(a) - diverged - length_mismatch,
        'diverged': diverged,
        'length_mismatch': length_mismatch,
        'exposure_tokens': exposure,
        'first_divergence_median': statistics.median(positions) if positions else None,
        'positions': positions,
        'events': events,
    }


def context_id(input_ids: list[int], prefix: list[int], tokens: tuple[int, int]) -> str:
    """Identity of a divergence context: the prompt, the common prefix, the two tokens."""
    first, second = sorted(tokens)
    text = ','.join(map(str, input_ids)) + '|' + ','.join(map(str, prefix))
    return prompt_hash(f'{text}|{first},{second}')


def classify_margin(margin: float, ulp: float | None) -> str:
    """compare.py's classes applied to the stock margin between the two tokens."""
    if not math.isfinite(margin):
        return 'large'
    if margin == 0:
        return 'tie'
    if ulp is not None and margin <= ulp * 1.0001:
        return 'one_ulp'
    if margin <= NEAR_NATS:
        return 'near'
    return 'large'


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def stats_delta(after: dict[str, Any], before: dict[str, Any] | None) -> dict[str, dict]:
    """Per-path counter differences between two certified-head stats snapshots."""
    out = {}
    for path, counters in (after.get('paths') or {}).items():
        prior = ((before or {}).get('paths') or {}).get(path, {})
        out[path] = {
            key: value - prior.get(key, 0)
            for key, value in counters.items()
            if isinstance(value, int)
        }
    return out


def _snapshot(point_dir: Path, when: str, name: str) -> dict[str, Any] | None:
    path = point_dir / f'snapshot_{when}' / name
    return json.loads(path.read_text()) if path.exists() else None


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def hold_records(runs: Path) -> list[dict[str, Any]]:
    """Each hold's provenance and time span (UTC), from its manifest."""
    out = []
    for path in sorted((runs / 'holds').glob('*.json')):
        record = json.loads(path.read_text())
        prov = record.get('provenance') or {}
        out.append(
            {
                'hold': record.get('hold'),
                'start_utc': _utc(record.get('start_unix')),
                'end_utc': _utc(record.get('end_unix')),
                'failed_launches': record.get('failed_launches'),
                **{
                    key: prov.get(key)
                    for key in (
                        'repo_commit',
                        'engine_commit',
                        'engine_tree',
                        'certified_head_digest',
                    )
                },
            }
        )
    return out


def _utc(stamp: float | None) -> str | None:
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, tz=UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def load_classes(path: Path) -> dict[str, str]:
    """Context id -> class from rescore.py's output (empty before the re-score)."""
    if not path.exists():
        return {}
    return {row['id']: row['class'] for row in iter_jsonl(path)}


def report(
    runs: Path,
    out: Path,
    predictions: Path | None,
    plot: bool,
    export_contexts: Path | None = None,
) -> dict[str, Any]:
    entries = load_holds(runs)
    index = index_launches(entries)
    launches: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, entry in index.items():
        info = load_launch(entry)
        info['problems'] = launch_problems(info, plan.FAMILIES[key[1]], key[2])
        launches[key] = info
    sessions = [
        s
        for s in (*plan.DECISION_SESSIONS, plan.REPLACEMENT_SESSION)
        if any(k[0] == s for k in launches)
    ]

    # Points of every timed launch.
    point_rows, by_key = [], {}
    for (step, fam, variant), info in sorted(launches.items()):
        if variant == 'check':
            continue
        family = plan.FAMILIES[fam]
        for point in info['points']:
            reason = point_problems(point, family.max_concurrency)
            row = {
                'session': step,
                'family': fam,
                'arm': family.arm,
                'variant': variant,
                'label': info['label'],
                'run': info['run'],
                'concurrency': int(point['concurrency']),
                'invalid_reason': '; '.join(filter(None, [reason, *info['problems']])),
                # Token comparisons need complete outputs from a correctly configured
                # launch, not a valid timing.
                'outputs_complete': outputs_complete(point)
                and not config_problems(info, family, variant, all_levels=False)
                and bool(info.get('run_dir')),
                'x_e2e': point.get('x_e2e'),
                'x_decode': point.get('x_decode'),
                'y': point.get('y'),
                'y_steady': point.get('y_steady'),
                'ttft_p50_ms': (point.get('ttft_ms') or {}).get('p50'),
                'itl_p50_ms': (point.get('itl_ms') or {}).get('p50'),
                'server_ms_per_pass': server_ms_per_pass(point),
                'accept_length': (point.get('spec') or {}).get('accept_length'),
                'max_running_logged': (point.get('server_log') or {}).get('max_running_logged'),
                'kv_retractions': (point.get('server_log') or {}).get('kv_retractions'),
                'foreign_cpu_mean': point.get('foreign_cpu_during_mean'),
                'foreign_cpu_max': point.get('foreign_cpu_during_max'),
                'requests': point.get('requests'),
            }
            point_rows.append(row)
            by_key[(step, fam, variant, row['concurrency'])] = row

    # Pairs and decisions.
    pair_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for fam, family in plan.FAMILIES.items():
        for c in family.concurrency:
            ys: list[float] = []
            xs: list[float] = []
            used: list[str] = []
            for step in sessions:
                s_row, c_row = (
                    by_key.get((step, fam, 'stock', c)),
                    by_key.get((step, fam, 'cert', c)),
                )
                if s_row is None and c_row is None:
                    continue
                problems = []
                if s_row is None or c_row is None:
                    problems.append('unpaired')
                else:
                    for side in (s_row, c_row):
                        if side['invalid_reason']:
                            problems.append(f'{side["variant"]}: {side["invalid_reason"]}')
                    stock_info = launches[(step, fam, 'stock')]
                    cert_info = launches[(step, fam, 'cert')]
                    if pools(stock_info) != pools(cert_info):
                        problems.append(f'pools differ {pools(stock_info)} {pools(cert_info)}')
                    if stock_info.get('graph_sizes') != cert_info.get('graph_sizes'):
                        problems.append('captured graph sizes differ')
                # Sessions are in order s1, s2, s3, s4: the first three valid pairs count,
                # so s4 counts only where s1-s3 left fewer than three.
                counted = not problems and len(ys) < DECISION_N
                pair: dict[str, Any] = {
                    'family': fam,
                    'concurrency': c,
                    'session': step,
                    'y_stock': s_row['y'] if s_row else None,
                    'y_cert': c_row['y'] if c_row else None,
                    'x_stock': s_row['x_e2e'] if s_row else None,
                    'x_cert': c_row['x_e2e'] if c_row else None,
                    'y_ratio': None,
                    'x_ratio': None,
                    'accept_stock': s_row['accept_length'] if s_row else None,
                    'accept_cert': c_row['accept_length'] if c_row else None,
                    'accept_ratio': None,
                    'cycle_rate_ratio': None,
                    'foreign_cpu_stock': s_row['foreign_cpu_mean'] if s_row else None,
                    'foreign_cpu_cert': c_row['foreign_cpu_mean'] if c_row else None,
                    'counted': counted,
                    'invalid_reason': '; '.join(problems),
                }
                if s_row and c_row and not problems:
                    pair['y_ratio'] = float(c_row['y']) / float(s_row['y'])
                    pair['x_ratio'] = float(c_row['x_e2e']) / float(s_row['x_e2e'])
                    if s_row['accept_length'] and c_row['accept_length']:
                        # y = cycles/s x tokens per cycle: split a head effect (cycle
                        # rate) from acceptance drift.
                        pair['accept_ratio'] = float(c_row['accept_length']) / float(
                            s_row['accept_length']
                        )
                        pair['cycle_rate_ratio'] = pair['y_ratio'] / pair['accept_ratio']
                pair_rows.append(pair)
                if counted:
                    ys.append(pair['y_ratio'])
                    xs.append(pair['x_ratio'])
                    used.append(step)
            y_sum, x_sum = ratio_summary(ys), ratio_summary(xs)
            by_session = {
                pr['session']: pr['y_ratio']
                for pr in pair_rows
                if pr['family'] == fam and pr['concurrency'] == c and pr['counted']
            }
            order = None
            if all(k in by_session for k in plan.DECISION_SESSIONS):
                # Session 2 runs each pair in the other order: its log ratio against the
                # mean of sessions 1 and 3 shows a position effect.
                order = math.log(by_session['s2']) - 0.5 * (
                    math.log(by_session['s1']) + math.log(by_session['s3'])
                )
            if c == family.primary:
                role = 'primary'
            elif family.gated_off(c):
                role = 'gate overhead'
            else:
                role = 'descriptive'
            ratio_rows.append(
                {
                    'family': fam,
                    'arm': family.arm,
                    'concurrency': c,
                    'role': role,
                    'n': y_sum['n'],
                    'sessions': ' '.join(used),
                    'y_ratio': y_sum['mean'],
                    'y_low': y_sum['low'],
                    'y_high': y_sum['high'],
                    'x_ratio': x_sum['mean'],
                    'x_low': x_sum['low'],
                    'x_high': x_sum['high'],
                    'p_value': t_test_p(y_sum),
                    'y_low_bonferroni4': _adjusted(y_sum, T_BONFERRONI4, -1),
                    'y_high_bonferroni4': _adjusted(y_sum, T_BONFERRONI4, 1),
                    'interval_reading': interval_reading(y_sum),
                    'decision': '',
                    'order_effect_log': order,
                    '_summary': y_sum,
                }
            )
    primaries = {r['family']: r['_summary'] for r in ratio_rows if r['role'] == 'primary'}
    holm_decisions = holm(primaries)
    for row in ratio_rows:
        if row['role'] == 'primary':
            row['decision'] = holm_decisions[row['family']]
        del row['_summary']

    pred = {}
    if predictions and predictions.exists():
        for p in json.loads(predictions.read_text())['points']:
            pred[(p['family'], p['concurrency'])] = p
    for row in ratio_rows:
        p = pred.get((row['family'], row['concurrency']))
        row['predicted_y_ratio'] = p['y_ratio'] if p else None
        row['predicted_decode_ratio'] = p['decode_ratio'] if p else None

    # Token comparisons, with the re-scored classes when they exist.
    classes = load_classes(runs / 'rescore' / 'classes.jsonl')
    equality_rows, exactness, contexts = token_comparisons(launches, by_key, sessions, classes)
    if export_contexts is not None:
        export_contexts.parent.mkdir(parents=True, exist_ok=True)
        with export_contexts.open('w') as handle:
            for record in contexts.values():
                handle.write(json.dumps(record) + '\n')

    # Check-mode launches.
    check_rows, check_verdict = check_counters(launches)

    # Certified head counters in the timed launches; launch and capture records.
    stats_rows, launch_rows, capture_rows = launch_records(launches)

    verdicts: dict[str, str] = {}
    exact_status: dict[str, str] = {}
    for fam, family in plan.FAMILIES.items():
        exact_status[fam] = exactness_status(
            exactness['families'].get(fam, {}).get('c1_identical'),
            1 in family.concurrency,
            check_verdict.get(fam),
        )
        verdicts[fam] = family_verdict(holm_decisions[fam], exact_status[fam])
    summary = {
        'sessions': sessions,
        'h4': h4_verdict(verdicts),
        'verdicts': verdicts,
        'exactness_status': exact_status,
        'primary': {fam: f.primary for fam, f in plan.FAMILIES.items()},
        'decisions': ratio_rows,
        'exactness': exactness,
        'check': check_verdict,
        'plan': {
            name: {'arm': f.arm, 'flags': f.flags, 'concurrency': f.concurrency, 'sets': f.sets}
            for name, f in plan.FAMILIES.items()
        },
        'certified_env': plan.CERT_ENV,
        'analysis_commit': _git_head(plan.REPO),
        'holds': hold_records(runs),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(point_rows, out / 'points.csv')
    write_csv(pair_rows, out / 'pairs.csv')
    write_csv(ratio_rows, out / 'ratios.csv')
    write_csv(equality_rows, out / 'equality.csv')
    write_csv(check_rows, out / 'check.csv')
    write_csv(stats_rows, out / 'certified_stats.csv')
    write_csv(launch_rows, out / 'launches.csv')
    write_csv(capture_rows, out / 'capture_memory.csv')
    write_csv(frontier_rows(point_rows), out / 'frontier.csv')
    slow_rows, slow_launches = slow_launch_diagnostic(point_rows)
    write_csv(slow_rows, out / 'launch_outliers.csv')
    summary['post_hoc_launch_outliers'] = outlier_effects(slow_launches, pair_rows, ratio_rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    if plot:
        from experiments.benchcert import figures

        figures.frontier(out / 'frontier.csv', out / 'frontier.png')
        figures.ratios(out / 'ratios.csv', out / 'ratios.png')
    return summary


def token_comparisons(
    launches: dict[tuple[str, str, str], dict[str, Any]],
    by_key: dict[tuple[str, str, str, int], dict[str, Any]],
    sessions: list[str],
    classes: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Certified against stock in each session, and stock against stock across them.

    Returns the comparison rows, the per-family exactness summary and the contexts
    of every first divergence of the two compared kinds (for the re-score), keyed
    by `context_id`. `classes` maps context ids to their re-scored class.
    """
    rows: list[dict[str, Any]] = []
    families: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    cache: dict[tuple[str, str, str, int], dict[str, dict[str, list[int]]]] = {}

    def ids(step: str, fam: str, variant: str, c: int) -> dict[str, dict[str, list[int]]] | None:
        key = (step, fam, variant, c)
        row = by_key.get(key)
        if row is None or not row['outputs_complete']:
            return None
        if key not in cache:
            run_dir = Path(launches[(step, fam, variant)]['run_dir'])
            cache[key] = request_ids(run_dir / 'r0' / f'c{c:03d}')
        return cache[key]

    for fam, family in plan.FAMILIES.items():
        totals = {'cert_vs_stock': [0, 0], 'stock_vs_stock': [0, 0]}
        counts = {kind: dict.fromkeys((*CLASSES, 'unscored'), 0) for kind in totals}
        c1 = []
        for c in family.concurrency:
            kinds = [('cert_vs_stock', (s, 'cert'), (s, 'stock')) for s in sessions]
            kinds += [
                ('stock_vs_stock', (a, 'stock'), (b, 'stock'))
                for i, a in enumerate(sessions)
                for b in sessions[i + 1 :]
            ]
            kinds += [
                ('cert_vs_cert', (a, 'cert'), (b, 'cert'))
                for i, a in enumerate(sessions)
                for b in sessions[i + 1 :]
            ]
            for kind, (sa, va), (sb, vb) in kinds:
                left, right = ids(sa, fam, va, c), ids(sb, fam, vb, c)
                if left is None or right is None:
                    continue
                result = compare(
                    {k: v['output'] for k, v in left.items()},
                    {k: v['output'] for k, v in right.items()},
                )
                positions, events = result.pop('positions'), result.pop('events')
                rate, low, high = rate_interval(result['diverged'], result['exposure_tokens'])
                row = {
                    'family': fam,
                    'concurrency': c,
                    'kind': kind,
                    'a': f'{sa}/{va}',
                    'b': f'{sb}/{vb}',
                    **result,
                    'rate_per_1k': rate,
                    'rate_low': low,
                    'rate_high': high,
                }
                if kind in totals:
                    found = dict.fromkeys((*CLASSES, 'unscored'), 0)
                    for prompt, d, tok_a, tok_b in events:
                        prompt_ids = left[prompt].get('input')
                        if prompt_ids is None:
                            found['unscored'] += 1
                            continue
                        prefix = left[prompt]['output'][:d]
                        cid = context_id(prompt_ids, prefix, (tok_a, tok_b))
                        contexts.setdefault(
                            cid,
                            {
                                'id': cid,
                                'input_ids': prompt_ids + prefix,
                                'tokens': sorted((tok_a, tok_b)),
                            },
                        )
                        found[(classes or {}).get(cid, 'unscored')] += 1
                    row.update({f'class_{name}': n for name, n in found.items()})
                    if c > 1:
                        for name, n in found.items():
                            counts[kind][name] += n
                row['first_positions'] = ' '.join(map(str, positions[:50]))
                rows.append(row)
                if c == 1 and kind == 'cert_vs_stock':
                    c1.append(result['identical'] == result['prompts'])
                if c > 1 and kind in totals:
                    totals[kind][0] += result['diverged']
                    totals[kind][1] += result['exposure_tokens']
        (k1, e1), (k2, e2) = totals['cert_vs_stock'], totals['stock_vs_stock']
        families[fam] = {
            'c1_pairs': len(c1),
            'c1_identical': (all(c1) if c1 else None),
            'cert_vs_stock_c_gt_1': {
                'diverged': k1,
                'exposure': e1,
                'rate': rate_interval(k1, e1),
                'classes': counts['cert_vs_stock'],
            },
            'stock_vs_stock_c_gt_1': {
                'diverged': k2,
                'exposure': e2,
                'rate': rate_interval(k2, e2),
                'classes': counts['stock_vs_stock'],
            },
            'ratio_to_floor': ratio_interval(k1, e1, k2, e2),
        }
    return rows, {'families': families}, contexts


def check_counters(
    launches: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool | None]]:
    """Per point and path: certified calls, rows, fallbacks and differing rows.

    The verdict per family is False on positive evidence (a differing row, or a
    declared path never certified by a launch that completed), True when a complete
    launch certified every declared path with no differing row, and None (exactness
    incomplete) when the launch is missing, failed or left no counters.
    """
    rows: list[dict[str, Any]] = []
    verdict: dict[str, bool | None] = {}
    for fam, family in plan.FAMILIES.items():
        info = launches.get(('check', fam, 'check'))
        if info is None:
            verdict[fam] = None
            continue
        complete = not info['problems']
        mismatch = 0
        name = Path(info['stats_file']).name if info.get('stats_file') else ''
        seen_paths: set[str] = set()
        for point in info['points']:
            c = int(point['concurrency'])
            point_dir = Path(info['run_dir']) / 'r0' / f'c{c:03d}'
            after = _snapshot(point_dir, 'after', name)
            if after is None:
                complete = False
                continue
            delta = stats_delta(after, _snapshot(point_dir, 'before', name))
            for path, counters in delta.items():
                if counters.get('calls', 0) > 0:
                    seen_paths.add(path)
                mismatch += counters.get('mismatch_rows', 0)
                rows.append({'family': fam, 'concurrency': c, 'path': path, **counters})
        if mismatch > 0:
            verdict[fam] = False
        elif not complete or not info['points']:
            verdict[fam] = None
        else:
            verdict[fam] = set(family.rows_per_request) <= seen_paths
    return rows, verdict


def launch_records(
    launches: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stats_rows, launch_rows, capture_rows = [], [], []
    for (step, fam, variant), info in sorted(launches.items()):
        server = info['server']
        tokens, running, mamba = pools(info)
        launch_rows.append(
            {
                'session': step,
                'family': fam,
                'variant': variant,
                'label': info['label'],
                'run': info['run'],
                'hold': info['hold'],
                'exit_code': info.get('exit_code'),
                'problems': '; '.join(info['problems']),
                'max_total_num_tokens': tokens,
                'max_running_requests': running,
                'mamba_slots': mamba,
                'available_gpu_mem_gb': server['available_gpu_mem_gb'],
                'gpu_used_after_ready_mib': info.get('gpu_used_after_ready_mib'),
                'ready_after_s': info.get('ready_after_s'),
                'cert_paths': ' '.join(server['cert_paths'] or ()),
                'engine_commit': (info.get('sglang_source') or {}).get('head'),
                'repo_commit': (info.get('provenance') or {}).get('repo_commit'),
            }
        )
        for graph, record in sorted(server['captures'].items()):
            capture_rows.append(
                {'session': step, 'family': fam, 'variant': variant, 'graph': graph, **record}
            )
        if variant == 'cert' and info.get('stats_file') and Path(info['stats_file']).exists():
            stats = json.loads(Path(info['stats_file']).read_text())
            for path, counters in (stats.get('paths') or {}).items():
                calls, rows_ = counters.get('calls', 0), counters.get('rows', 0)
                host = counters.get('host_steps') or {}
                stats_rows.append(
                    {
                        'session': step,
                        'family': fam,
                        'path': path,
                        'calls': calls,
                        'rows': rows_,
                        'fallback_rows': counters.get('fallback_rows', 0),
                        'fallback_calls': counters.get('fallback_calls', 0),
                        'fallback_row_rate': counters.get('fallback_rows', 0) / rows_
                        if rows_
                        else None,
                        'fallback_call_rate': counters.get('fallback_calls', 0) / calls
                        if calls
                        else None,
                        'certified_steps': host.get('certified'),
                        'stock_graph_steps': host.get('stock_graph'),
                        'eager_steps': host.get('eager'),
                        'max_certified_rows': counters.get('max_certified_rows'),
                        'refused_rows': counters.get('status_refused', 0),
                    }
                )
    return stats_rows, launch_rows, capture_rows


def server_ms_per_pass(point: dict[str, Any]) -> float | None:
    """Scheduler time per decode or verify pass at (nearly) full batch, from the log:
    running requests x tokens per pass / logged generation rate."""
    log = point.get('server_log') or {}
    tps, running = log.get('logged_gen_tps_full_batch'), log.get('max_running_logged')
    if not tps or not running:
        return None
    return 1000.0 * running * (log.get('logged_accept_len_mean') or 1.0) / tps


# Post hoc, not declared (2026-10-02, after h1): the machine has shown launch-level slow
# states (TTFT p50 about +4 ms at every concurrency, decode 2-6% slower, foreign load
# and clocks normal). A launch is listed when, at three quarters or more of its levels,
# its TTFT p50 is at least TTFT_MS above the median of the same arm's other sessions or
# its time per pass is at least PASS_FRACTION above theirs. Reported, never excluded.
OUTLIER_TTFT_MS = 3.0
OUTLIER_PASS_FRACTION = 0.02
OUTLIER_SHARE = 0.75


def slow_launch_diagnostic(
    point_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Per point: TTFT p50 and time per pass against the same arm's other sessions."""
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in point_rows:
        groups[(row['family'], row['variant'], row['concurrency'])][row['session']] = row
    rows: list[dict[str, Any]] = []
    flags: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for (fam, variant, c), by_session in sorted(groups.items()):
        for session, row in by_session.items():
            others = [r for s, r in by_session.items() if s != session]
            entry: dict[str, Any] = {
                'family': fam,
                'variant': variant,
                'concurrency': c,
                'session': session,
                'others': len(others),
            }
            slow = False
            for key, label in (('ttft_p50_ms', 'ttft'), ('server_ms_per_pass', 'pass')):
                mine = row.get(key)
                theirs = [float(r[key]) for r in others if r.get(key) is not None]
                if mine is None or not theirs:
                    entry[f'{label}_excess'] = None
                    continue
                ref = statistics.median(theirs)
                excess = float(mine) - ref if label == 'ttft' else float(mine) / ref - 1.0
                entry[f'{label}_excess'] = excess
                limit = OUTLIER_TTFT_MS if label == 'ttft' else OUTLIER_PASS_FRACTION
                slow = slow or excess >= limit
            rows.append(entry)
            if others:
                flags[(session, fam, variant)].append(slow)
    flagged = [
        key for key, marks in flags.items() if marks and sum(marks) >= OUTLIER_SHARE * len(marks)
    ]
    for row in rows:
        row['launch_flagged'] = (row['session'], row['family'], row['variant']) in flagged
    return rows, sorted(flagged)


def outlier_effects(
    flagged: list[tuple[str, str, str]],
    pair_rows: list[dict[str, Any]],
    ratio_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each flagged launch, its family's primary ratio with and without that
    session's pair (a sensitivity check; the declared decision is unchanged)."""
    out = []
    for session, fam, variant in flagged:
        primary = plan.FAMILIES[fam].primary
        kept = [
            pr['y_ratio']
            for pr in pair_rows
            if pr['family'] == fam
            and pr['concurrency'] == primary
            and pr['counted']
            and pr['session'] != session
        ]
        declared = next(r for r in ratio_rows if r['family'] == fam and r['concurrency'] == primary)
        without = ratio_summary(kept)
        out.append(
            {
                'launch': f'{session}/{fam}/{variant}',
                'primary_concurrency': primary,
                'declared_ratio': declared['y_ratio'],
                'declared_decision': declared['decision'],
                'without_ratio': without['mean'],
                'without_interval': [without['low'], without['high']],
                'without_n': without['n'],
            }
        )
    return out


def frontier_rows(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and sd over valid sessions of x and y per family, variant and c."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        if not row['invalid_reason']:
            groups[(row['family'], row['variant'], row['concurrency'])].append(row)
    out = []
    for (fam, variant, c), rows in sorted(groups.items()):
        entry: dict[str, Any] = {
            'family': fam,
            'variant': variant,
            'concurrency': c,
            'n': len(rows),
        }
        for key in ('x_e2e', 'y'):
            values = [float(r[key]) for r in rows]
            entry[f'{key}_mean'] = statistics.fmean(values)
            entry[f'{key}_sd'] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(entry)
    return out


def replacement_families(runs: Path) -> list[str]:
    """Families whose sessions s1-s3 left fewer than three valid pairs at some c."""
    with_runs = [
        s
        for s in plan.DECISION_SESSIONS
        if (runs / 'holds').exists() and any(e['step'] == s for e in load_holds(runs))
    ]
    if len(with_runs) < len(plan.DECISION_SESSIONS):
        raise SystemExit(f'sessions run so far: {with_runs}; s4 is decided after s1-s3')
    summary = report(runs, runs / 'analysis-replacement', None, plot=False)
    return sorted({r['family'] for r in summary['decisions'] if r['n'] < DECISION_N})


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return '' if math.isnan(value) else f'{value:.6g}'
    if isinstance(value, tuple | list):
        return ' '.join(map(str, value))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict')
    p.add_argument(
        '--head', type=Path, default=plan.REPO / 'evidence/certified_head/head_path_time.csv'
    )
    p.add_argument(
        '--frontier', type=Path, default=plan.REPO / 'evidence/bench/confirm/frontier.csv'
    )
    p.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('report')
    r.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument(
        '--predictions',
        type=Path,
        default=plan.REPO / 'evidence/certified_head/served/predictions.json',
    )
    r.add_argument('--no-plot', action='store_true')
    r.add_argument(
        '--export-contexts',
        type=Path,
        default=None,
        help='write every first-divergence context (JSONL) for rescore.py',
    )
    s = sub.add_parser('replacement')
    s.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    args = parser.parse_args(argv)
    if args.command == 'predict':
        result = predict(args.head, args.frontier)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')
        for point in result['points']:
            print(
                f'{point["family"]:9} c={point["concurrency"]:>3} decode x{point["decode_ratio"]:.3f}'
                f' y x{point["y_ratio"]:.3f} (saves {point["saving_us"]:.0f} of'
                f' {point["cycle_us"]:.0f} us)'
            )
        return 0
    if args.command == 'report':
        summary = report(
            args.runs.expanduser(),
            args.out,
            args.predictions,
            not args.no_plot,
            args.export_contexts,
        )
        print(json.dumps(summary['verdicts'], indent=1))
        return 0
    print(' '.join(replacement_families(args.runs.expanduser())) or 'none')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
