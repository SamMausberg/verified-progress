"""Compare the streamed greedy text of two bench sweeps, prompt by prompt.

bench.sweep keeps aiperf's raw export, the SSE stream of every request. With greedy
decoding and ignore_eos, two servers that commit the same tokens stream the same text.
This rebuilds each profiling-phase request's text from its chunks (the `reasoning_content`
and `content` deltas in one buffer, in arrival order, noting each character's channel),
matches requests across two runs by prompt id (the point's requests.csv), and reports for
each sweep point how many prompts streamed identical text and, for the others, where the
first difference falls: its character offset and the output tokens streamed before the
chunk that holds it (the smaller count of the two runs, from the chunks' cumulative usage).

A prompt is identical when both runs streamed the same characters in the same order, each
on the same channel. This is streamed-text identity, not token-id or logprob identity: two
token sequences that detokenize to the same text count as identical. Completion token
counts do not enter it; prompts whose counts differ are reported separately.

A run is LABEL=<sweep run dir>; a pair is TEST:BASELINE. A missing raw export, a request
that failed or streamed no usage, a prompt repeated within a point, or two runs whose
points or prompt sets differ is an error, and nothing is written.

    T=~/vp-data/backbone/e2e/mtp-triton-v1
    python experiments/backbone/stream_text_identity.py \\
        --run B1=$T/backbone-mtp-triton-v1-B/20261002-123648 \\
        --run A1=$T/backbone-mtp-triton-v1-A/20261002-124002 \\
        --pair B1:A1 --out <out.json>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHANNELS = {'reasoning_content': 'r', 'content': 'c'}


@dataclass
class Stream:
    text: str = ''  # every delta of both channels, in arrival order
    channels: str = ''  # one letter per character of text: 'r' reasoning, 'c' content
    # (characters streamed so far, cumulative completion tokens)
    marks: list[tuple[int, int]] = field(default_factory=list)
    completion_tokens: int | None = None


def read_stream(raw: dict[str, Any]) -> Stream:
    """Rebuild one request's streamed text and token marks from its raw aiperf record."""
    s = Stream()
    for response in raw.get('responses', []):
        for packet in response.get('packets', []):
            value = packet.get('value')
            if not isinstance(value, str) or not value.startswith('{'):
                continue
            chunk = json.loads(value)
            for choice in chunk.get('choices') or []:
                # Within one delta, keep the order in which its fields were sent.
                for key, piece in (choice.get('delta') or {}).items():
                    if key in CHANNELS and piece:
                        s.text += piece
                        s.channels += CHANNELS[key] * len(piece)
            usage = chunk.get('usage')
            if usage:
                s.completion_tokens = int(usage['completion_tokens'])
                s.marks.append((len(s.text), s.completion_tokens))
    return s


def _raw_records(point: Path) -> list[dict[str, Any]]:
    for name in ('profile_export_raw.jsonl', 'profile_export_raw.jsonl.gz'):
        path = point / 'aiperf' / name
        if path.exists():
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rt') as f:
                return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f'{point}: no raw aiperf export')


def load_point(point: Path) -> dict[str, Stream]:
    """Streams of one sweep point's profiling requests, keyed by prompt id."""
    with (point / 'requests.csv').open() as f:
        rows = {r['request_id']: r for r in csv.DictReader(f)}
    streams: dict[str, Stream] = {}
    seen: set[str] = set()
    for raw in _raw_records(point):
        meta = raw.get('metadata', {})
        if meta.get('benchmark_phase') != 'profiling':
            continue
        rid = meta.get('x_request_id')
        row = rows.get(rid or '')
        if row is None:
            raise ValueError(f'{point}: request {rid} is not in requests.csv')
        if row['ok'] != 'True' or raw.get('status') != 200:
            raise ValueError(f'{point}: request {rid} failed')
        pid = row['prompt_id']
        if not pid or pid in streams:
            raise ValueError(f'{point}: prompt id {pid!r} missing or repeated')
        stream = read_stream(raw)
        if stream.completion_tokens is None:
            raise ValueError(f'{point}: request {rid} streamed no usage')
        streams[pid] = stream
        seen.add(rid or '')
    if seen != set(rows):
        raise ValueError(f'{point}: {len(set(rows) - seen)} requests.csv rows have no raw record')
    return streams


def load_run(run: Path) -> dict[str, dict[str, Stream]]:
    """Every point of a sweep run directory, keyed by '<repeat>/<concurrency dir>'."""
    points = sorted(p for p in run.glob('r*/c*') if p.is_dir())
    if not points:
        raise ValueError(f'{run}: no sweep points')
    return {f'{p.parent.name}/{p.name}': load_point(p) for p in points}


def _first_mismatch(x: str, y: str) -> int | None:
    """First index where x and y differ (the shorter length for a strict prefix), or None."""
    offset = next((i for i, (u, v) in enumerate(zip(x, y, strict=False)) if u != v), None)
    if offset is None and len(x) != len(y):
        offset = min(len(x), len(y))
    return offset


def first_divergence(a: Stream, b: Stream) -> tuple[int, int]:
    """Character offset of the first difference, and tokens streamed before its chunk."""
    offset = _first_mismatch(a.text, b.text)
    if offset is None:  # same text, split differently between the channels
        offset = _first_mismatch(a.channels, b.channels)
    if offset is None:
        raise ValueError('the two streams are identical')

    def before(s: Stream) -> int:
        return max((tokens for chars, tokens in s.marks if chars <= offset), default=0)

    return offset, min(before(a), before(b))


def compare_point(a: dict[str, Stream], b: dict[str, Stream]) -> dict[str, Any]:
    if set(a) != set(b):
        raise ValueError(f'prompt sets differ: {len(set(a) ^ set(b))} ids in only one run')
    divergences: list[dict[str, Any]] = []
    token_counts_differ: list[str] = []
    for pid in sorted(a):
        sa, sb = a[pid], b[pid]
        if sa.completion_tokens != sb.completion_tokens:
            token_counts_differ.append(pid)
        if sa.text != sb.text or sa.channels != sb.channels:
            offset, tokens = first_divergence(sa, sb)
            divergences.append({'prompt_id': pid, 'char_offset': offset, 'tokens_before': tokens})
    divergences.sort(key=lambda d: (d['tokens_before'], d['prompt_id']))
    tokens_before: list[int] = [d['tokens_before'] for d in divergences]
    return {
        'prompts': len(a),
        'identical': len(a) - len(divergences),
        'differing': len(divergences),
        'completion_tokens': sorted({s.completion_tokens for s in (*a.values(), *b.values())}),
        'completion_tokens_differ': token_counts_differ,
        'tokens_before_min': min(tokens_before) if tokens_before else None,
        'tokens_before_median': statistics.median(tokens_before) if tokens_before else None,
        'first_divergences': divergences,
    }


def _repo_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True
    )
    return head.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    ap.add_argument('--pair', action='append', required=True, metavar='TEST:BASELINE')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run_dirs: dict[str, Path] = {}
    for item in args.run:
        label, sep, path = item.partition('=')
        if not sep or not label or label in run_dirs:
            raise SystemExit(f'bad or repeated --run {item!r}')
        run_dirs[label] = Path(path).expanduser()
    pairs = []
    for item in args.pair:
        test, sep, base = item.partition(':')
        if not sep or test not in run_dirs or base not in run_dirs:
            raise SystemExit(f'--pair {item!r} names a run not given with --run')
        pairs.append((test, base))
    try:
        runs = {label: load_run(path) for label, path in run_dirs.items()}
        results = []
        for test, base in pairs:
            if set(runs[test]) != set(runs[base]):
                raise ValueError(f'{test} and {base} cover different sweep points')
            for point in sorted(runs[test]):
                s = compare_point(runs[test][point], runs[base][point])
                results.append({'test': test, 'baseline': base, 'point': point, **s})
                print(
                    f'{test}:{base} {point} identical {s["identical"]}/{s["prompts"]}', flush=True
                )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    out = {
        'command': ' '.join(sys.argv),
        'repo_commit': _repo_commit(),
        'comparison': 'streamed text per prompt (reasoning_content and content deltas in one '
        'buffer, in arrival order, each character on the same channel); not token ids or '
        'logprobs; completion_tokens_differ lists prompts whose token counts differ, which '
        'does not enter the identity',
        'tokens_before': 'output tokens streamed before the chunk that holds the first '
        'difference, the smaller count of the two runs; the first differing token is at '
        'or after this index',
        'runs': {label: path.name for label, path in run_dirs.items()},
        'pairs': results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
