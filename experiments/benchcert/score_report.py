"""Per-point summary of the teacher-forced scores and the gross-event report (after h6s).

    python -m experiments.benchcert.score_report summarize --out DIR --csv POINTS.csv
    python -m experiments.benchcert.score_report report --out DIR --json REPORT.json \
        [--events EVENTS.csv] [--contexts CONTEXTS.jsonl] [--rescored CLASSES.jsonl]

`drain.py score` (hold h6s) writes one JSONL file per point under DIR/score: per request
its phase, prompt hash and every output position where the committed token is not the
teacher-forced top-1, with both logprobs. The gap is the top-1 logprob minus the
committed token's. Under the README's reading rule ("Drain reruns") a gap of at least
`drain.GROSS_NATS` (2) is a gross wrong-token event and a gap above `NEAR_NATS` (0.5) and
below 2 is near.

`summarize` writes one row per point: requests, scored positions, near and gross counts
over all requests and over the measured ones, and the gap at 579ae7ce's position 439.

`report` gives, as JSON:
- the rate comparisons, near and gross separately, as an exact conditional (binomial)
  95% interval on the Poisson rate ratio: h6a's certified c = 64 points against its stock
  ones (the declared comparison), every MTP c = 64 point of sessions 1-3 and h6a, and,
  per family, every timed point of sessions 1-3;
- the positive control (session 1's certified 1756 at 579ae7ce/439) and every point that
  committed 1756 there, with that request's other disagreements at positions 440-511:
  the outputs continue identically after 1756 or 68189, so a gap there would mean the
  model's state held another token than the one emitted;
- every gross event, with its time from the client's streamed chunks (the first chunk
  that carries the position), the requests in flight then (the drain's running batch
  as the client saw it), the time to the point's last response, and the co-batched
  requests' disagreements whose tokens streamed within `WINDOW_MS` of it, by class
  (a row-shift fault would put wrong tokens in other requests of the same verify).

`--contexts` writes each gross event's context in rescore.py's format (prompt, output
up to the event, the committed and top-1 tokens), for a batch-1 top-5 re-score that
says whether the position was a near tie; `--rescored` merges that re-score's classes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from bench.results import prompt_hash
from experiments.benchcert import plan
from experiments.benchcert.analyze import NEAR_NATS
from experiments.benchcert.drain import FAMILY, GROSS_NATS, LAUNCHES, TARGET, TOP, point_dirs

WINDOW_MS = 300.0
UNFINISHED_SLACK_MS = (0, 20, 50, 100)
ROUNDING = 1e-6  # a gap below this is an exact tie at the BF16 logit
CONFIDENCE = 0.95


# -- point names ------------------------------------------------------------------


def describe(name: str) -> dict[str, Any]:
    """Group, family, variant, launch, repeat and concurrency from a point name.

    Names (drain.point_dirs): ``h6/<launch>/<rN>/cNNN`` (drain reruns, MTP),
    ``s1/<arm label>/cNNN`` (sessions 1-3), ``check/<check label>/cNNN``.
    """
    parts = name.split('/')
    group, level = parts[0], parts[-1]
    out: dict[str, Any] = {
        'group': group,
        'family': None,
        'variant': None,
        'launch': None,
        'repeat': None,
        'concurrency': int(level[1:]) if level.startswith('c') and level[1:].isdigit() else None,
    }
    if group in ('h6', 'h7') and len(parts) == 4:
        launch = LAUNCHES.get(parts[1])
        out.update(
            family='mtp',
            launch=parts[1],
            repeat=parts[2],
            variant=launch.variant if launch is not None else None,
        )
        return out
    label = parts[1] if len(parts) == 3 else None
    for key, family in plan.FAMILIES.items():
        if label == family.stock_label:
            out.update(family=key, variant='stock')
        elif label == family.cert_label:
            out.update(family=key, variant='cert')
        elif label == family.check_label:
            out.update(family=key, variant='check')
    return out


def score_file(out: Path, name: str) -> Path:
    return out / 'score' / (name.replace('/', '__') + '.jsonl')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gap(entry: dict[str, Any]) -> float:
    return float(entry['top1_logprob']) - float(entry['logprob'])


def gap_class(value: float) -> str:
    if value < ROUNDING:
        return 'tie'
    if value <= NEAR_NATS:
        return 'rounding'
    if value < GROSS_NATS:
        return 'near'
    return 'gross'


# -- per point ----------------------------------------------------------------------


def target_gap(records: Iterable[dict[str, Any]]) -> tuple[int | None, float | None]:
    """(token, gap) at 579ae7ce's position 439 in the measured phase, if it disagreed
    with the teacher-forced top-1; (None, 0.0) if it agreed; (None, None) if absent."""
    prompt, position, _ = TARGET
    for record in records:
        if record.get('phase') != 'profiling' or not record.get('prompt', '').startswith(prompt):
            continue
        if 'error' in record:
            return None, None
        for entry in record.get('disagree', []):
            if entry['position'] == position:
                return int(entry['token']), gap(entry)
        return None, 0.0
    return None, None


def summarize_point(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {'point': name, **describe(name)}
    for scope, keep in (
        ('all', lambda r: True),
        ('measured', lambda r: r.get('phase') == 'profiling'),
    ):
        chosen = [r for r in records if keep(r)]
        ok = [r for r in chosen if 'error' not in r]
        suffix = '' if scope == 'all' else '_measured'
        row[f'requests{suffix}'] = len(chosen)
        row[f'errors{suffix}'] = len(chosen) - len(ok)
        row[f'positions{suffix}'] = sum(int(r['positions']) for r in ok)
        row[f'near{suffix}'] = sum(int(r['near']) for r in ok)
        row[f'gross{suffix}'] = sum(int(r['gross']) for r in ok)
    ok = [r for r in records if 'error' not in r]
    row['max_gap'] = round(max((float(r['max_gap']) for r in ok), default=0.0), 4)
    token, value = target_gap(records)
    row['token_439_if_not_top1'] = token
    row['gap_439'] = None if value is None else round(value, 4)
    return row


def summarize(out: Path, runs: Path) -> list[dict[str, Any]]:
    rows = []
    for name, _ in point_dirs(out, runs):
        path = score_file(out, name)
        if path.exists():
            rows.append(summarize_point(name, read_jsonl(path)))
    return rows


# -- rates --------------------------------------------------------------------------


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    terms = [_log_binom_pmf(i, n, p) for i in range(k + 1)]
    top = max(terms)
    if top == -math.inf:
        return 0.0
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def _solve(f: Callable[[float], float], target: float, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    alpha = 1 - confidence
    if n == 0:
        return 0.0, 1.0
    lower = 0.0 if k == 0 else _solve(lambda p: 1 - binom_cdf(k - 1, n, p), alpha / 2, True)
    upper = 1.0 if k == n else _solve(lambda p: binom_cdf(k, n, p), alpha / 2, False)
    return lower, upper


def rate_ratio(k1: int, e1: int, k2: int, e2: int) -> dict[str, Any]:
    """Poisson rate ratio (1 over 2) with the exact conditional 95% interval: given
    k1 + k2 events, k1 is binomial with p = r e1 / (r e1 + e2)."""
    out: dict[str, Any] = {'events': [k1, k2], 'exposure': [e1, e2]}
    if min(e1, e2) <= 0:
        return {**out, 'ratio': None, 'ci95': [None, None]}
    n = k1 + k2
    lo_p, hi_p = clopper_pearson(k1, n)

    def to_ratio(p: float) -> float:
        return math.inf if p >= 1 else (p / (1 - p)) * (e2 / e1)

    ratio = None if k2 == 0 else (k1 / e1) / (k2 / e2)
    if n == 0:
        return {**out, 'ratio': None, 'ci95': [0.0, None]}
    upper = to_ratio(hi_p)
    return {
        **out,
        'ratio': None if ratio is None else round(ratio, 4),
        'ci95': [round(to_ratio(lo_p), 4), None if math.isinf(upper) else round(upper, 4)],
    }


def compare(
    rows: list[dict[str, Any]],
    first: Callable[[dict[str, Any]], bool],
    second: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suffix in ('', '_measured'):
        a = [r for r in rows if first(r)]
        b = [r for r in rows if second(r)]
        scope = 'all' if not suffix else 'measured'
        out[scope] = {'points': [len(a), len(b)]}
        for kind in ('near', 'gross'):
            out[scope][kind] = rate_ratio(
                sum(r[f'{kind}{suffix}'] for r in a),
                sum(r[f'positions{suffix}'] for r in a),
                sum(r[f'{kind}{suffix}'] for r in b),
                sum(r[f'positions{suffix}'] for r in b),
            )
    return out


def comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def is_(group: str | tuple[str, ...], variant: str, top: bool = True, family: str = 'mtp'):
        groups = (group,) if isinstance(group, str) else group

        def keep(row: dict[str, Any]) -> bool:
            return (
                row['group'] in groups
                and row['variant'] == variant
                and row['family'] == family
                and (not top or row['concurrency'] == TOP)
            )

        return keep

    sessions = plan.DECISION_SESSIONS
    out = {
        'h6a_c64_cert_vs_stock': compare(rows, is_('h6', 'cert'), is_('h6', 'stock')),
        'sessions_and_h6a_mtp_c64_cert_vs_stock': compare(
            rows, is_(('h6', *sessions), 'cert'), is_(('h6', *sessions), 'stock')
        ),
    }
    if any(r['group'] == 'h7' for r in rows):
        for variant in ('certring', 'cert0'):
            out[f'h7_c64_{variant}_vs_stock'] = compare(
                rows, is_('h7', variant), is_('h7', 'stock')
            )
    for key in plan.FAMILIES:
        out[f'sessions_{key}_all_points_cert_vs_stock'] = compare(
            rows, is_(sessions, 'cert', False, key), is_(sessions, 'stock', False, key)
        )
    return out


# -- client timelines ---------------------------------------------------------------


def timeline(point_dir: Path) -> list[dict[str, Any]]:
    """Every request of a point in export order: phase, prompt hash, conversation, token
    ids, start and end (client perf counter, ns) and streamed chunks as (time, tokens so
    far)."""
    raw = point_dir / 'aiperf/profile_export_raw.jsonl.gz'
    out = []
    with gzip.open(raw, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            messages = record.get('payload', {}).get('messages') or [{}]
            item: dict[str, Any] = {
                'phase': record.get('metadata', {}).get('benchmark_phase'),
                'prompt': prompt_hash(messages[-1].get('content', '')),
                'cid': record.get('metadata', {}).get('conversation_id'),
                'start': record.get('start_perf_ns'),
                'chunks': [],
            }
            responses = record.get('responses', [])
            if responses:
                item['end'] = responses[-1].get('perf_ns')
            for response in responses:
                tokens = None
                for packet in response.get('packets', []):
                    value = packet.get('value')
                    if not (isinstance(value, str) and value.startswith('{')):
                        continue
                    chunk = json.loads(value)
                    usage = chunk.get('usage') or {}
                    if usage.get('completion_tokens') is not None:
                        tokens = int(usage['completion_tokens'])
                    ext = chunk.get('sglext') or {}
                    if ext.get('output_ids'):
                        item['output'] = list(ext['output_ids'][0])
                    if ext.get('input_ids'):
                        item['input'] = list(ext['input_ids'])
                if tokens is not None:
                    item['chunks'].append((int(response['perf_ns']), tokens))
            out.append(item)
    return out


def arrival(item: dict[str, Any], position: int) -> int | None:
    """Client time (ns) of the first chunk that carries output position ``position``."""
    for t, tokens in item['chunks']:
        if tokens > position:
            return t
    return None


def positions_between(item: dict[str, Any], t0: int, t1: int) -> list[int]:
    """Output positions whose chunk arrived in [t0, t1]."""
    found: list[int] = []
    before = 0
    for t, tokens in item['chunks']:
        if t0 <= t <= t1:
            found.extend(range(before, tokens))
        before = max(before, tokens)
    return found


def event_context(
    records: list[dict[str, Any]], items: list[dict[str, Any]], index: int, position: int
) -> dict[str, Any]:
    """Timing and co-batched disagreements around one gross event (request ``index``)."""
    item = items[index]
    t = arrival(item, position)
    if t is None:
        return {'time_found': False}
    ends = [i['end'] for i in items if i.get('end') is not None]
    in_flight = sum(
        1
        for i in items
        if i.get('start') is not None and i.get('end') is not None and i['start'] <= t <= i['end']
    )
    # Requests still unfinished at the server when it produced the event's token: their
    # final chunk reaches the client after the event's chunk. Chunks of one server step
    # can arrive out of order, so requests whose final chunk came up to `slack` before
    # the event's are counted too: an upper bound on the server's running batch.
    unfinished = {
        f'{slack}ms': sum(
            1
            for i in items
            if i.get('start') is not None
            and i.get('end') is not None
            and i['start'] <= t
            and i['end'] >= t - slack * 1_000_000
        )
        for slack in UNFINISHED_SLACK_MS
    }
    half = int(WINDOW_MS * 1e6)
    co: dict[str, int] = {'tie': 0, 'rounding': 0, 'near': 0, 'gross': 0}
    co_events = []
    co_requests = 0
    for j, (other, rec) in enumerate(zip(items, records, strict=True)):
        if j == index:
            continue
        window = set(positions_between(other, t - half, t + half))
        if not window:
            continue
        co_requests += 1
        for entry in rec.get('disagree', []):
            if entry['position'] in window:
                value = gap(entry)
                co[gap_class(value)] += 1
                if value > NEAR_NATS:
                    co_events.append(
                        {
                            'prompt': rec.get('prompt'),
                            'phase': rec.get('phase'),
                            'position': entry['position'],
                            'token': entry['token'],
                            'top1': entry['top1'],
                            'gap': round(value, 4),
                            'dt_ms': round(((arrival(other, entry['position']) or t) - t) / 1e6, 1),
                        }
                    )
    return {
        'time_found': True,
        'ms_before_point_end': round((max(ends) - t) / 1e6, 1) if ends else None,
        'in_flight': in_flight,
        'running_upper_bound': unfinished,
        'co_batched_requests_in_window': co_requests,
        'co_batched_disagreements_by_class': co,
        'co_batched_events_above_near': co_events,
    }


# -- report -------------------------------------------------------------------------


def gross_events(name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for index, record in enumerate(records):
        for entry in record.get('disagree', []):
            value = gap(entry)
            if value >= GROSS_NATS:
                events.append(
                    {
                        'point': name,
                        **describe(name),
                        'request_index': index,
                        'phase': record.get('phase'),
                        'prompt': record.get('prompt'),
                        'position': int(entry['position']),
                        'token': int(entry['token']),
                        'top1': int(entry['top1']),
                        'gap': round(value, 4),
                        'control': (
                            record.get('prompt', '').startswith(TARGET[0])
                            and int(entry['position']) == TARGET[1]
                            and int(entry['token']) == TARGET[2]
                        ),
                    }
                )
    return events


def after_1756(name: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """579ae7ce's disagreements after position 439, in a point that committed 1756 there."""
    prompt, position, token = TARGET
    for record in records:
        if record.get('phase') != 'profiling' or not record.get('prompt', '').startswith(prompt):
            continue
        entries = record.get('disagree', [])
        if not any(e['position'] == position and e['token'] == token for e in entries):
            return None
        later = [e for e in entries if e['position'] > position]
        return {
            'point': name,
            'disagreements_440_511': [
                {
                    'position': e['position'],
                    'token': e['token'],
                    'top1': e['top1'],
                    'gap': round(gap(e), 4),
                }
                for e in later
            ],
            'max_gap_440_511': round(max((gap(e) for e in later), default=0.0), 4),
        }
    return None


def report(
    out: Path,
    runs: Path,
    contexts: Path | None = None,
    rescored: Path | None = None,
) -> dict[str, Any]:
    points = [(n, p) for n, p in point_dirs(out, runs) if score_file(out, n).exists()]
    rows, events, prefix = [], [], []
    control_name = f's1/{FAMILY.cert_label}/c{TOP:03d}'
    context_lines = []
    classes = {}
    if rescored is not None and rescored.exists():
        classes = {r['id']: r for r in read_jsonl(rescored)}
    for name, point in points:
        records = read_jsonl(score_file(out, name))
        rows.append(summarize_point(name, records))
        found = after_1756(name, records)
        if found is not None:
            prefix.append(found)
        point_events = gross_events(name, records)
        if not point_events:
            continue
        items = timeline(point)
        aligned = len(items) == len(records) and all(
            i['prompt'] == r.get('prompt') and i['phase'] == r.get('phase')
            for i, r in zip(items, records, strict=False)
        )
        for event in point_events:
            if aligned:
                event.update(
                    event_context(records, items, event['request_index'], event['position'])
                )
                item = items[event['request_index']]
                if 'input' in item and 'output' in item:
                    cid = f'{name}|{event["phase"]}|{event["prompt"]}|{event["position"]}'
                    event['context_id'] = cid
                    context_lines.append(
                        {
                            'id': cid,
                            'input_ids': item['input'] + item['output'][: event['position']],
                            'tokens': [event['token'], event['top1']],
                        }
                    )
                    if cid in classes:
                        event['rescore'] = {
                            k: classes[cid].get(k)
                            for k in ('margin', 'ulp', 'class', 'stock_top1', 'top5')
                        }
            else:
                event['timeline_error'] = 'export and score records do not align'
            events.append(event)
    if contexts is not None:
        contexts.parent.mkdir(parents=True, exist_ok=True)
        contexts.write_text(''.join(json.dumps(c) + '\n' for c in context_lines))
    control = [e for e in events if e['point'] == control_name and e['control']]
    return {
        'points_scored': len(rows),
        'errors': sum(r['errors'] for r in rows),
        'positions': sum(r['positions'] for r in rows),
        'positive_control': control[0] if control else None,
        'points_with_1756_at_439': prefix,
        'gross_events': events,
        'gross_by_variant': _count_by(events, ('group', 'family', 'variant')),
        'comparisons': comparisons(rows),
    }


def _count_by(events: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], int] = {}
    for event in events:
        key = tuple(event.get(k) for k in keys)
        counts[key] = counts.get(key, 0) + 1
    return [
        {**dict(zip(keys, key, strict=True)), 'gross_events': n}
        for key, n in sorted(counts.items(), key=str)
    ]


EVENT_FIELDS = (
    'point',
    'group',
    'family',
    'variant',
    'concurrency',
    'phase',
    'prompt',
    'position',
    'token',
    'top1',
    'gap',
    'control',
    'ms_before_point_end',
    'in_flight',
    'co_batched_requests_in_window',
)


def write_csv(rows: list[dict[str, Any]], path: Path, fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields) if fields is not None else list(rows[0]) if rows else []
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


EOT_ID = 248044  # <|endoftext|> (config.json eos_token_id; checked against 579ae7ce's output)


def eot_split(out: Path, runs: Path, eot: int = EOT_ID) -> list[dict[str, Any]]:
    """Near and gross events before and after each request's first <|endoftext|>, per
    group, family and variant. Both gross contexts follow an end of turn that the
    ignore_eos runs decode past (the model then regenerates a prompt)."""
    acc: dict[tuple[Any, ...], dict[str, int]] = {}
    for name, point in point_dirs(out, runs):
        path = score_file(out, name)
        if not path.exists():
            continue
        records = read_jsonl(path)
        items = timeline(point)
        if len(items) != len(records):
            continue
        d = describe(name)
        key = (d['group'], d['family'], d['variant'])
        bucket = acc.setdefault(key, {})
        for item, rec in zip(items, records, strict=True):
            if 'error' in rec:
                continue
            output = item.get('output') or []
            first = output.index(eot) if eot in output else len(output)
            bucket['positions_before'] = bucket.get('positions_before', 0) + min(first, 512)
            bucket['positions_after'] = bucket.get('positions_after', 0) + max(0, 512 - first)
            for entry in rec.get('disagree', []):
                value = gap(entry)
                kind = gap_class(value)
                if kind in ('near', 'gross'):
                    side = 'before' if entry['position'] < first else 'after'
                    bucket[f'{kind}_{side}'] = bucket.get(f'{kind}_{side}', 0) + 1
    return [
        {'group': k[0], 'family': k[1], 'variant': k[2], **v} for k, v in sorted(acc.items(), key=str)
    ]


def companions(out: Path, runs: Path) -> list[dict[str, Any]]:
    """The requests in flight (client view) when 579ae7ce's position-439 token arrived, in
    every MTP c = 64 point whose output reached that position with session 1's prefix."""
    from experiments.benchcert.drain import target_record

    names = [(n, p) for n, p in point_dirs(out, runs) if n.endswith(f'c{TOP:03d}')]
    names = [(n, p) for n, p in names if describe(n)['family'] == 'mtp']
    control = next(p for n, p in names if n.startswith('s1/') and describe(n)['variant'] == 'cert')
    ref = (target_record(control) or {}).get('output') or []
    rows = []
    for name, point in names:
        items = timeline(point)
        target = next(
            (i for i in items if i['phase'] == 'profiling' and i['prompt'].startswith(TARGET[0])), None
        )
        if target is None or 'output' not in target or target['output'][:TARGET[1]] != ref[:TARGET[1]]:
            continue
        t = arrival(target, TARGET[1])
        if t is None:
            continue
        flying = sorted(
            f'{i["phase"]}:{i["prompt"]}'
            for i in items
            if i is not target and i.get('start') is not None and i.get('end') is not None
            and i['start'] <= t <= i['end']
        )
        rows.append(
            {
                'point': name,
                **describe(name),
                'token_at_439': target['output'][TARGET[1]],
                'in_flight_others': flying,
            }
        )
    events = [set(r['in_flight_others']) for r in rows if r['token_at_439'] == TARGET[2]]
    for r in rows:
        mine = set(r['in_flight_others'])
        r['overlap_with_events'] = [len(mine & e) for e in events]
        r['same_set_as_an_event'] = any(mine == e for e in events)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('summarize', 'report', 'context'):
        p = sub.add_parser(command)
        p.add_argument('--out', type=Path, required=True, help='the drain output directory')
        p.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    sub.choices['summarize'].add_argument('--csv', type=Path, required=True)
    sub.choices['context'].add_argument('--json', type=Path, required=True)
    sub.choices['context'].add_argument('--eot', type=int, default=EOT_ID)
    r = sub.choices['report']
    r.add_argument('--json', type=Path, required=True)
    r.add_argument('--events', type=Path, help='also write the gross events as CSV')
    r.add_argument('--contexts', type=Path, help='write the gross contexts for rescore.py')
    r.add_argument('--rescored', type=Path, help="rescore.py's classes for those contexts")
    args = parser.parse_args(argv)
    if args.command == 'summarize':
        write_csv(summarize(args.out, args.runs), args.csv)
        return 0
    if args.command == 'context':
        result = {
            'eot_id': args.eot,
            'eot_split': eot_split(args.out, args.runs, args.eot),
            'companions_at_439': companions(args.out, args.runs),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + '\n')
        for row in result['companions_at_439']:
            print(row['point'], row['token_at_439'], len(row['in_flight_others']), row['overlap_with_events'])
        return 0
    result = report(args.out, args.runs, args.contexts, args.rescored)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + '\n')
    if args.events is not None:
        write_csv(result['gross_events'], args.events, EVENT_FIELDS)
    control = result['positive_control']
    print(
        f'{result["points_scored"]} points, {result["positions"]} positions, '
        f'{result["errors"]} errors; control '
        f'{"found, gap " + str(control["gap"]) if control else "NOT FOUND"}; '
        f'{len(result["gross_events"])} gross events'
    )
    for entry in result['gross_by_variant']:
        print(f'  {entry}')
    if result['errors']:
        # Counts from points with unscored requests understate the events: rescore them
        # (drain.py score retries any point file holding an error record).
        print(f'ERROR: {result["errors"]} requests were not scored')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
