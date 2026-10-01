"""CPU test of the SGLang check arms' verdict (experiments/certified_head/check_arm_stats.py):
every expected path must make certified calls, and no row on any path may differ, be
refused or trip a probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/certified_head/check_arm_stats.py'
spec = importlib.util.spec_from_file_location('check_arm_stats', SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def stats(**paths: dict[str, int]) -> dict[str, Any]:
    base = {'calls': 0, 'mismatch_rows': 0, 'status_refused': 0, 'status_probe': 0}
    return {'paths': {p: {**base, **v} for p, v in paths.items()}}


def test_an_idle_expected_path_fails() -> None:
    s = stats(draft={'calls': 10}, draft_extend={'calls': 0})
    assert check.problems(s, ['draft', 'draft_extend'])[1]


def test_an_enabled_path_the_arm_does_not_run_is_reported_not_failed() -> None:
    s = stats(draft={'calls': 10}, draft_extend={'calls': 5}, dflash_draft={'calls': 0})
    line, failed = check.problems(s, ['draft', 'draft_extend'])
    assert not failed and "['dflash_draft']" in line


def test_differing_refused_or_probed_rows_fail_on_any_path() -> None:
    for field in ('mismatch_rows', 'status_refused', 'status_probe'):
        s = stats(verify={'calls': 10}, dflash_draft={'calls': 0, field: 1})
        assert check.problems(s, ['verify'])[1], field


def test_an_arm_without_expected_paths_fails() -> None:
    assert check.problems(stats(verify={'calls': 10}), [])[1]
