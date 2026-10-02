"""Tests of the backbone workstream's served-evidence tools (CPU only)."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'backbone'))

from bitwise_runs import compare
from insitu_gemm import complete_replays
from stream_text_identity import compare_point, load_point


def test_complete_replays_drops_cut_and_boundary_replays() -> None:
    # Replay 1 was cut by the window's start, replay 6 by its end.
    order = [1, 2, 3, 4, 5, 6]
    sizes = {1: 3, 2: 10, 3: 10, 4: 10, 5: 10, 6: 7}
    kept, full = complete_replays(order, sizes)
    assert full == 10
    # Full-count replays are 2-5; the first and last of them are dropped too.
    assert kept == [3, 4]


def test_complete_replays_counts_a_missing_replay_as_empty() -> None:
    kept, full = complete_replays([1, 2, 3, 4], {2: 5, 3: 5, 4: 5})
    assert full == 5
    assert kept == [3]


def rec(ids, tops):
    return {'output_ids': ids, 'top_logprobs': tops}


def test_bitwise_counts_tokens_and_full_logprob_arrays() -> None:
    a = {
        'p0': rec([1, 2], [[[-0.1, 1], [-9.0, 7]], [[-0.2, 2], [-8.0, 5]]]),
        'p1': rec([3], [[[-0.3, 3], [-9.5, 4]]]),
        'p2': rec([4], [[[-0.4, 4]]]),
    }
    b = {
        'p0': rec([1, 2], [[[-0.1, 1], [-9.0, 7]], [[-0.2, 2], [-8.0, 5]]]),
        # Same tokens; a tail entry below -4 nats differs (drift ignores it).
        'p1': rec([3], [[[-0.3, 3], [-9.25, 4]]]),
        'p2': rec([5], [[[-0.4, 5]]]),
    }
    s = compare(a, b)
    assert s['prompts'] == 3
    assert s['tokens_equal'] == 2
    assert s['top_logprobs_equal'] == 1
    assert s['bitwise_equal'] == 1
    assert s['differing_ids_first_10'] == ['p1', 'p2']


def test_bitwise_refuses_different_prompt_sets() -> None:
    with pytest.raises(ValueError, match='prompt sets differ'):
        compare({'p0': rec([1], [[]])}, {'p1': rec([1], [[]])})


def test_bitwise_refuses_runs_without_logprobs() -> None:
    with pytest.raises(ValueError, match='no top_logprobs'):
        compare({'p0': {'output_ids': [1]}}, {'p0': rec([1], [[]])})


def test_bitwise_cli_fails_on_a_missing_run(tmp_path: Path) -> None:
    import subprocess

    pairs = tmp_path / 'pairs.json'
    pairs.write_text(json.dumps([['x', 'a/c1', 'b/c1']]))
    script = Path(__file__).resolve().parents[1] / 'experiments' / 'backbone' / 'bitwise_runs.py'
    run = subprocess.run(
        [sys.executable, str(script), '--runs', str(tmp_path), '--pairs', str(pairs),
         '--out', str(tmp_path / 'out.json')],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert run.returncode != 0
    assert 'missing run' in run.stderr
    assert not (tmp_path / 'out.json').exists()


Chunk = str | tuple[str, str]


def write_point(
    point: Path, texts: dict[str, list[Chunk]], *, ok: bool = True, tokens_per_chunk: int = 1
) -> None:
    """A sweep point with one profiling request per prompt.

    A chunk is a reasoning delta or a (delta field, text) pair; each streams tokens_per_chunk.
    """
    (point / 'aiperf').mkdir(parents=True)
    rows: list[dict[str, str]] = []
    raws: list[dict[str, Any]] = []
    for i, (pid, chunks) in enumerate(texts.items()):
        rid = f'r{i}'
        rows.append({'request_id': rid, 'prompt_id': pid, 'ok': str(ok)})
        deltas = [dict([c]) if isinstance(c, tuple) else {'reasoning_content': c} for c in chunks]
        packets = [
            {'value': json.dumps({'choices': [{'delta': d}],
                                  'usage': {'completion_tokens': (n + 1) * tokens_per_chunk}})}
            for n, d in enumerate(deltas)
        ] + [{'value': '[DONE]'}]  # fmt: skip
        raws.append({'metadata': {'benchmark_phase': 'warmup', 'x_request_id': 'w'}})
        raws.append({'metadata': {'benchmark_phase': 'profiling', 'x_request_id': rid},
                     'status': 200, 'responses': [{'packets': packets}]})  # fmt: skip
    with (point / 'requests.csv').open('w') as f:
        f.write('request_id,prompt_id,ok\n')
        f.writelines(f'{r["request_id"]},{r["prompt_id"]},{r["ok"]}\n' for r in rows)
    with (point / 'aiperf' / 'profile_export_raw.jsonl').open('w') as f:
        f.writelines(json.dumps(r) + '\n' for r in raws)


def test_stream_text_identity_finds_the_first_divergent_chunk(tmp_path: Path) -> None:
    write_point(tmp_path / 'a', {'p0': ['ab', 'cd', 'ef'], 'p1': ['xy', 'z']})
    write_point(tmp_path / 'b', {'p0': ['ab', 'cX', 'ef'], 'p1': ['xy', 'z']})
    s = compare_point(load_point(tmp_path / 'a'), load_point(tmp_path / 'b'))
    assert (s['prompts'], s['identical'], s['differing']) == (2, 1, 1)
    # 'abcd' and 'abcX' first differ at character 3, inside the second chunk: one token before it.
    assert s['first_divergences'] == [{'prompt_id': 'p0', 'char_offset': 3, 'tokens_before': 1}]


def test_stream_text_identity_refuses_different_prompts_and_failed_requests(tmp_path: Path) -> None:
    write_point(tmp_path / 'a', {'p0': ['ab']})
    write_point(tmp_path / 'b', {'p1': ['ab']})
    with pytest.raises(ValueError, match='prompt sets differ'):
        compare_point(load_point(tmp_path / 'a'), load_point(tmp_path / 'b'))
    write_point(tmp_path / 'c', {'p0': ['ab']}, ok=False)
    with pytest.raises(ValueError, match='failed'):
        load_point(tmp_path / 'c')


def test_stream_text_identity_keeps_the_arrival_order_of_both_channels(tmp_path: Path) -> None:
    # Reasoning 'a', content 'c', reasoning 'b' streams 'acb'; buffering the channels
    # separately would rebuild both requests as 'abc' and call them identical.
    r, c = 'reasoning_content', 'content'
    write_point(tmp_path / 'a', {'p0': [(r, 'a'), (c, 'c'), (r, 'b')]})
    write_point(tmp_path / 'b', {'p0': [(r, 'a'), (r, 'b'), (c, 'c')]})
    s = compare_point(load_point(tmp_path / 'a'), load_point(tmp_path / 'b'))
    assert s['first_divergences'] == [{'prompt_id': 'p0', 'char_offset': 1, 'tokens_before': 1}]
    # Same characters in the same order, but the second one on another channel.
    write_point(tmp_path / 'c', {'p0': [(r, 'a'), (c, 'b'), (c, 'c')]})
    write_point(tmp_path / 'd', {'p0': [(r, 'a'), (r, 'b'), (c, 'c')]})
    s = compare_point(load_point(tmp_path / 'c'), load_point(tmp_path / 'd'))
    assert s['first_divergences'] == [{'prompt_id': 'p0', 'char_offset': 1, 'tokens_before': 1}]


def test_stream_text_identity_reports_token_counts_outside_the_identity(tmp_path: Path) -> None:
    write_point(tmp_path / 'a', {'p0': ['ab', 'cd']}, tokens_per_chunk=1)
    write_point(tmp_path / 'b', {'p0': ['ab', 'cd']}, tokens_per_chunk=2)
    s = compare_point(load_point(tmp_path / 'a'), load_point(tmp_path / 'b'))
    assert (s['identical'], s['differing']) == (1, 0)
    assert s['completion_tokens_differ'] == ['p0']
