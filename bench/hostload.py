"""Host CPU load from other processes during a timed run.

The GPU lock serialises GPU use, not the CPU. Serving timings depend on
single-threaded CPU loops (SGLang's scheduler, tokenizer manager and detokenizer,
and the load client), so a timed run is only valid on a quiet host. This module
samples, once per second, the CPU cores used by every process outside a given
process tree (the run's own processes), and reports the mean, the maximum and the
busiest foreign processes. A run whose mean foreign load exceeds
`CONTENTION_CORES` is treated as contended.

Use it around any command (the command's whole process tree counts as its own):

    python -m bench.hostload record --out load.json -- <command ...>
    python -m bench.hostload wait --max-cores 2 --timeout 600   # block until quiet

`record` exits with the command's exit status and writes the load summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Idle-host background (IDE server, agent CLIs, kernel threads) measures 0.3-0.6
# cores here. A run needs its four single-threaded loops (scheduler, tokenizer
# manager, detokenizer, client timing manager) to keep a core each and the memory
# system to itself; 2 cores admits the background and rejects any real job. It is
# a heuristic threshold, not a derived one.
CONTENTION_CORES = 2.0
_TICK = os.sysconf('SC_CLK_TCK')


def cpu_by_pid() -> dict[int, tuple[int, float, float]]:
    """pid -> (parent pid, own CPU seconds, CPU seconds of its reaped children).

    When a process exits and its parent reaps it, the kernel adds its CPU time to
    the parent's cutime/cstime, so (own + reaped children) summed over a process
    tree is continuous across child exits.
    """
    table: dict[int, tuple[int, float, float]] = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
        except OSError:
            continue
        # After the command name: state ppid ... utime(11) stime(12) cutime(13) cstime(14).
        fields = stat.rsplit(')', 1)[-1].split()
        own = (int(fields[11]) + int(fields[12])) / _TICK
        children = (int(fields[13]) + int(fields[14])) / _TICK
        table[int(entry.name)] = (int(fields[1]), own, children)
    return table


def busy_cpu_seconds() -> float:
    """CPU time spent busy on all cores since boot (everything but idle and iowait)."""
    fields = [int(value) for value in Path('/proc/stat').read_text().split('\n', 1)[0].split()[1:9]]
    # user nice system idle iowait irq softirq steal
    return (sum(fields) - fields[3] - fields[4]) / _TICK


def process_tree(root: int, table: dict[int, tuple[int, float, float]]) -> set[int]:
    """`root` and all of its descendants in `table`."""
    children: dict[int, list[int]] = {}
    for pid, (ppid, *_) in table.items():
        children.setdefault(ppid, []).append(pid)
    tree, frontier = {root}, [root]
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    return tree


def title(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    except OSError:
        return ''


def sample(root: int, interval: float = 1.0) -> dict[str, Any]:
    """CPU cores used over `interval` s outside and inside the process tree of `root`.

    Foreign load is the host's total busy CPU time (/proc/stat) minus the CPU time
    of the run's own process tree, including its children reaped during the
    interval, so short-lived foreign processes (builds, hooks, compiles) count as
    foreign and short-lived processes of the run itself do not. The per-process
    lists only cover processes alive at both ends and are diagnostic.
    """
    busy_before = busy_cpu_seconds()
    before = cpu_by_pid()
    time.sleep(interval)
    busy_after = busy_cpu_seconds()
    after = cpu_by_pid()
    tree_before, tree_after = process_tree(root, before), process_tree(root, after)
    own = tree_before | tree_after

    def tree_cpu(table: dict[int, tuple[int, float, float]], tree: set[int]) -> float:
        return sum(table[pid][1] + table[pid][2] for pid in tree if pid in table)

    # Own CPU includes processes of the run that started or exited within the
    # interval: their time is in a living ancestor's reaped-children counters.
    own_cpu = max(0.0, tree_cpu(after, tree_after) - tree_cpu(before, tree_before))
    usage = {
        pid: max(0.0, (seconds - before[pid][1]) / interval)
        for pid, (_, seconds, _) in after.items()
        if pid in before
    }
    foreign = {pid: cores for pid, cores in usage.items() if pid not in own}
    mine = {pid: cores for pid, cores in usage.items() if pid in own}
    total = (busy_after - busy_before) / interval
    foreign_total = max(0.0, total - own_cpu / interval)

    def top(items: dict[int, float], limit: int) -> list[dict[str, Any]]:
        ranked = sorted(items.items(), key=lambda item: -item[1])[:limit]
        return [
            {'pid': pid, 'cores': round(cores, 2), 'cmd': title(pid)[:120]}
            for pid, cores in ranked
            if cores >= 0.05
        ]

    return {
        'cores': round(foreign_total, 2),
        'host_busy_cores': round(total, 2),
        'own_cores': round(own_cpu / interval, 2),
        'top': top(foreign, 5),
        'own': top(mine, 8),
    }


class HostLoadSampler:
    """Samples foreign CPU load in a background thread until stopped."""

    def __init__(self, root: int | None = None, interval: float = 1.0) -> None:
        self.root = root if root is not None else os.getpid()
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(sample(self.root, self.interval))

    def __enter__(self) -> HostLoadSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, Any]:
        return summarise(self.samples, self.interval)


def summarise(samples: list[dict[str, Any]], interval: float = 1.0) -> dict[str, Any]:
    """Mean and maximum foreign cores, the busiest foreign and own processes."""
    cores = [entry['cores'] for entry in samples]
    foreign: dict[str, float] = {}
    own_peak: dict[str, float] = {}
    for entry in samples:
        for proc in entry.get('top', []):
            foreign[proc['cmd']] = foreign.get(proc['cmd'], 0.0) + proc['cores'] / len(samples)
        for proc in entry.get('own', []):
            key = proc['cmd'][:60]
            own_peak[key] = max(own_peak.get(key, 0.0), proc['cores'])
    mean = sum(cores) / len(cores) if cores else None
    return {
        'interval_s': interval,
        'samples': len(cores),
        'foreign_cores_mean': round(mean, 3) if mean is not None else None,
        'foreign_cores_max': max(cores) if cores else None,
        'contended': mean is not None and mean > CONTENTION_CORES,
        'contention_threshold_cores': CONTENTION_CORES,
        'top_foreign_mean_cores': dict(
            sorted(((k, round(v, 2)) for k, v in foreign.items()), key=lambda kv: -kv[1])[:5]
        ),
        'own_peak_cores': dict(sorted(own_peak.items(), key=lambda kv: -kv[1])[:8]),
    }


def wait_for_quiet(
    root: int | None = None, max_cores: float = CONTENTION_CORES, max_wait_s: float = 600.0
) -> dict[str, Any]:
    """Block until foreign load (2 s average) is at most `max_cores`, or time out."""
    root = root if root is not None else os.getpid()
    started = time.monotonic()
    while True:
        entry = sample(root, 2.0)
        waited = time.monotonic() - started
        if entry['cores'] <= max_cores or waited >= max_wait_s:
            return {**entry, 'waited_s': round(waited, 1), 'quiet': entry['cores'] <= max_cores}
        time.sleep(10.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record', help='run a command and record foreign CPU load')
    record.add_argument('--out', type=Path, required=True)
    record.add_argument('--interval', type=float, default=1.0)
    record.add_argument('argv', nargs=argparse.REMAINDER, help='-- command ...')
    wait = commands.add_parser('wait', help='block until the host is quiet')
    wait.add_argument('--max-cores', type=float, default=CONTENTION_CORES)
    wait.add_argument('--timeout', type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.command == 'wait':
        result = wait_for_quiet(os.getpid(), args.max_cores, args.timeout)
        print(json.dumps(result))
        return 0 if result['quiet'] else 1
    command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not command:
        parser.error('record needs a command after --')
    with HostLoadSampler(os.getpid(), args.interval) as sampler:
        code = subprocess.run(command, check=False).returncode
    report = {'command': command, 'exit_code': code, **sampler.summary()}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps({k: report[k] for k in ('foreign_cores_mean', 'foreign_cores_max', 'contended')})
    )
    return code


if __name__ == '__main__':
    sys.exit(main())
