"""Turn one aiperf run (one sweep point) into per-request rows and a point summary.

Inputs are aiperf 0.13 artifacts: `profile_export.jsonl` (per-request metrics),
`profile_export_raw.jsonl` (the SSE stream of every request, needed for the
server's per-request speculative statistics and the per-chunk token counts) and
the profiling-phase summary. Only profiling-phase requests are summarised.

Frontier metrics follow the paper (Section "Frontier metrics and useful
throughput"). For request i with n_i server-reported output tokens and e_i the
time from request send to its last non-empty streamed output:

- `x_e2e` = mean_i n_i / e_i: per-user output rate including TTFT (paper x);
- `y` = sum_i n_i / (last completion - first send): output tokens/s on the one
  GPU (paper y);
- `x_decode` = mean_i 1 / ITL_i with ITL_i = (e_i - TTFT_i) / (n_i - first-chunk
  tokens): aiperf's per-user decode rate, excluding TTFT;
- `y_steady` = output tokens streamed between the first and the last request
  send, divided by that interval: throughput while the client holds its full
  concurrency, excluding the drain after the last send.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PERCENTILES = (0.5, 0.9, 0.99)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _open(path: Path) -> Any:
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default), NaN for no data."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    summary = {
        'mean': statistics.fmean(finite) if finite else math.nan,
        'std': statistics.pstdev(finite) if len(finite) > 1 else 0.0 if finite else math.nan,
    }
    for q in PERCENTILES:
        summary[f'p{round(q * 100)}'] = quantile(finite, q)
    return summary


def _metric(record: dict[str, Any], name: str) -> Any:
    entry = record.get('metrics', {}).get(name)
    return entry.get('value') if isinstance(entry, dict) else None


def parse_stream(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract usage timeline, finish reason and spec stats from one raw record."""
    timeline: list[tuple[int, int]] = []
    finish_reason = None
    spec: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for response in raw.get('responses', []):
        perf_ns = int(response['perf_ns'])
        for packet in response.get('packets', []):
            value = packet.get('value')
            if not isinstance(value, str) or not value.startswith('{'):
                continue
            chunk = json.loads(value)
            if chunk.get('usage'):
                usage = chunk['usage']
                if chunk.get('choices'):
                    timeline.append((perf_ns, int(usage['completion_tokens'])))
            for choice in chunk.get('choices') or []:
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']
            extension = chunk.get('sglext') or {}
            if extension.get('spec_tokens_details'):
                spec = extension['spec_tokens_details']
    messages = raw.get('payload', {}).get('messages') or [{}]
    text = messages[-1].get('content', '')
    return {
        'x_request_id': raw.get('metadata', {}).get('x_request_id'),
        'start_perf_ns': int(raw.get('start_perf_ns') or 0),
        'prompt_sha': prompt_hash(text if isinstance(text, str) else json.dumps(text)),
        'timeline': timeline,
        'finish_reason': finish_reason,
        'usage': usage,
        'spec': spec,
        'status': raw.get('status'),
    }


def load_requests(
    artifact_dir: Path, prompt_index: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Merge metrics records with raw streams for the profiling phase."""
    raw_path = artifact_dir / 'profile_export_raw.jsonl'
    if not raw_path.exists():
        raw_path = artifact_dir / 'profile_export_raw.jsonl.gz'
    streams: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for raw in iter_jsonl(raw_path):
            if raw.get('metadata', {}).get('benchmark_phase') != 'profiling':
                continue
            parsed = parse_stream(raw)
            streams[parsed['x_request_id']] = parsed
    rows = []
    for record in iter_jsonl(artifact_dir / 'profile_export.jsonl'):
        meta = record.get('metadata', {})
        if meta.get('benchmark_phase') != 'profiling':
            continue
        stream = streams.get(meta.get('x_request_id'), {})
        latency_ms = _metric(record, 'request_latency')
        osl = _metric(record, 'output_sequence_length')
        itl_ms = _metric(record, 'inter_token_latency')
        spec = stream.get('spec') or {}
        prompt = (prompt_index or {}).get(stream.get('prompt_sha', ''), {})
        error = record.get('error')
        rows.append(
            {
                'request_id': meta.get('x_request_id'),
                'prompt_id': prompt.get('id'),
                'domain': prompt.get('domain'),
                'prompt_sha': stream.get('prompt_sha'),
                'start_ns': meta.get('request_start_ns'),
                'start_perf_ns': stream.get('start_perf_ns'),
                'ok': error is None and latency_ms is not None and osl is not None,
                'error': json.dumps(error)[:300] if error else None,
                'isl': _metric(record, 'input_sequence_length'),
                'osl': osl,
                'ttft_ms': _metric(record, 'time_to_first_token'),
                'latency_ms': latency_ms,
                'itl_ms': itl_ms,
                'e2e_tps': osl / (latency_ms / 1e3) if osl and latency_ms else None,
                'decode_tps': 1e3 / itl_ms if itl_ms else None,
                'chunks': len(stream.get('timeline') or []),
                'finish_reason': stream.get('finish_reason'),
                'spec_verify_ct': spec.get('spec_verify_ct'),
                'spec_accept_length': spec.get('spec_accept_length'),
                'spec_correct_drafts': spec.get('spec_num_correct_drafts'),
                'spec_proposed_drafts': spec.get('spec_num_proposed_drafts'),
                'spec_histogram': spec.get('spec_correct_drafts_histogram'),
                '_timeline': stream.get('timeline') or [],
            }
        )
    return rows


def steady_throughput(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Output tokens streamed between the first and last send, per second."""
    sends = [row['start_perf_ns'] for row in rows if row.get('start_perf_ns')]
    if len(sends) < 2 or not any(row['_timeline'] for row in rows):
        return {'y_steady': math.nan, 'steady_window_s': math.nan}
    first, last = min(sends), max(sends)
    if last <= first:
        return {'y_steady': math.nan, 'steady_window_s': 0.0}
    tokens = 0
    for row in rows:
        streamed = [count for when, count in row['_timeline'] if when <= last]
        tokens += max(streamed) if streamed else 0
    window = (last - first) / 1e9
    return {'y_steady': tokens / window, 'steady_window_s': window}


def summarise_point(
    rows: list[dict[str, Any]],
    target_osl: int | dict[str, int] | None,
    concurrency: int,
    aiperf_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Point-level metrics for one concurrency; see the module docstring."""
    ok = [row for row in rows if row['ok']]
    summary: dict[str, Any] = {
        'concurrency': concurrency,
        'requests': len(rows),
        'completed': len(ok),
        'failed': len(rows) - len(ok),
        'finish_reasons': dict(Counter(str(row['finish_reason']) for row in rows)),
    }
    if not ok:
        return summary
    osl = [row['osl'] for row in ok]
    summary['osl_min'] = min(osl)
    summary['osl_max'] = max(osl)
    if isinstance(target_osl, dict):
        # Per-prompt lengths (natural-length workload), matched by prompt id.
        summary['osl_mismatch'] = sum(
            1 for row in ok if row['osl'] != target_osl.get(str(row.get('prompt_id')), -1)
        )
    else:
        summary['osl_mismatch'] = (
            sum(1 for value in osl if value != target_osl) if target_osl is not None else None
        )
    starts = [row['start_ns'] for row in ok]
    ends = [row['start_ns'] + row['latency_ms'] * 1e6 for row in ok]
    span_s = (max(ends) - min(starts)) / 1e9
    total = sum(osl)
    summary.update(
        {
            'output_tokens': total,
            'span_s': span_s,
            'x_e2e': statistics.fmean(row['e2e_tps'] for row in ok),
            'x_e2e_p50': quantile([row['e2e_tps'] for row in ok], 0.5),
            'x_decode': (
                statistics.fmean(rates)
                if (rates := [row['decode_tps'] for row in ok if row['decode_tps']])
                else math.nan
            ),
            'y': total / span_s if span_s > 0 else math.nan,
            **steady_throughput(ok),
            'isl_mean': statistics.fmean(row['isl'] for row in ok if row['isl'] is not None),
            'ttft_ms': describe([row['ttft_ms'] for row in ok]),
            'latency_ms': describe([row['latency_ms'] for row in ok]),
            'itl_ms': describe([row['itl_ms'] for row in ok if row['itl_ms'] is not None]),
        }
    )
    verify = [row for row in ok if row['spec_verify_ct']]
    if verify:
        steps = sum(row['spec_verify_ct'] for row in verify)
        proposed = sum(row['spec_proposed_drafts'] or 0 for row in verify)
        histogram: list[int] = []
        for row in verify:
            for index, count in enumerate(row['spec_histogram'] or []):
                if index >= len(histogram):
                    histogram.extend([0] * (index + 1 - len(histogram)))
                histogram[index] += count
        summary['spec'] = {
            'requests_with_stats': len(verify),
            'verify_steps': steps,
            # Tokens per verify step including the bonus token, token-weighted.
            'accept_length': sum(row['osl'] for row in verify) / steps,
            'accept_length_request_mean': statistics.fmean(
                row['spec_accept_length'] for row in verify
            ),
            'accept_rate': (
                sum(row['spec_correct_drafts'] or 0 for row in verify) / proposed
                if proposed
                else math.nan
            ),
            'correct_drafts_histogram': histogram,
            'accept_length_by_domain': _by_domain(verify),
        }
    if aiperf_summary:
        summary['aiperf'] = {
            key: aiperf_summary.get(key, {}).get('avg')
            for key in (
                'output_token_throughput',
                'request_throughput',
                'benchmark_duration',
                'e2e_output_token_throughput',
                'output_token_throughput_per_user',
            )
            if isinstance(aiperf_summary.get(key), dict)
        }
    return summary


def _by_domain(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    tokens: Counter[str] = Counter()
    steps: Counter[str] = Counter()
    for row in rows:
        domain = row.get('domain') or 'unknown'
        tokens[domain] += row['osl']
        steps[domain] += row['spec_verify_ct']
    return {domain: tokens[domain] / steps[domain] for domain in sorted(steps)}


REQUEST_COLUMNS = (
    'request_id',
    'prompt_id',
    'domain',
    'ok',
    'isl',
    'osl',
    'ttft_ms',
    'latency_ms',
    'itl_ms',
    'e2e_tps',
    'decode_tps',
    'chunks',
    'finish_reason',
    'spec_verify_ct',
    'spec_accept_length',
    'start_ns',
    'error',
)


def write_requests_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f'{value:.4f}' if isinstance(value, float) else value)
                    for key, value in row.items()
                    if key in REQUEST_COLUMNS
                }
            )


# ---------------------------------------------------------------------------
# Server-side counters from /metrics (Prometheus text format)
# ---------------------------------------------------------------------------


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name plus label set; skip comments and histograms."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name_labels, _, value = line.rpartition(' ')
        try:
            values[name_labels] = values.get(name_labels, 0.0) + float(value)
        except ValueError:
            continue
    return values


def counter_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Differences of the sglang counters we report, keyed by a short name."""

    def total(values: dict[str, float], name: str, contains: str = '') -> float:
        return sum(
            value for key, value in values.items() if key.split('{')[0] == name and contains in key
        )

    wanted = {
        'generation_tokens': ('sglang:generation_tokens_total', ''),
        'prompt_tokens': ('sglang:prompt_tokens_total', ''),
        'spec_verify_calls': ('sglang:spec_verify_calls_total', ''),
        'decode_graph_passes': ('sglang:cuda_graph_passes_total', 'mode="decode_cuda_graph"'),
        'decode_eager_passes': ('sglang:cuda_graph_passes_total', 'mode="decode_none"'),
        'prefill_graph_passes': ('sglang:cuda_graph_passes_total', 'mode="prefill_cuda_graph"'),
        'prefill_eager_passes': ('sglang:cuda_graph_passes_total', 'mode="prefill_none"'),
        'num_requests': ('sglang:num_requests_total', ''),
        'retracted_requests': ('sglang:num_retracted_requests_total', ''),
    }
    deltas = {
        short: total(after, name, contains) - total(before, name, contains)
        for short, (name, contains) in wanted.items()
    }
    if deltas['spec_verify_calls'] > 0:
        deltas['server_accept_length'] = deltas['generation_tokens'] / deltas['spec_verify_calls']
    decode_passes = deltas['decode_graph_passes'] + deltas['decode_eager_passes']
    if decode_passes > 0:
        deltas['decode_graph_fraction'] = deltas['decode_graph_passes'] / decode_passes
    return deltas
