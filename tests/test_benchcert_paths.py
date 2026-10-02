"""CPU tests of the serial references and the seeded MTP control (benchcert, after h8)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchcert import control_waves, paths, score_report


def test_prefill_ends_match_the_reference_step() -> None:
    # 579ae7ce: 512 output tokens, target 439 -> absolute 514, 576 and 587 with 75 prompt ids.
    assert [75 + e for e in paths.prefill_ends(439, 512)] == [514, 576, 587]
    assert paths.prefill_ends(500, 512) == [500, 512]


def test_position_entry_reads_top5_and_tracked_tokens() -> None:
    top = [[-0.3, 1756, None], [-2.7, 5715, None]]
    tracked = [[-0.3, 1756, None], [-5.3, 68189, None], [None, 9471, None]]
    entry = paths.position_entry(top, tracked, 68189)
    assert entry['top1'] == 1756 and entry['top1_logprob'] == -0.3
    assert entry['tracked'] == {'1756': -0.3, '68189': -5.3}
    assert entry['token_logprob'] == -5.3


def test_serial_readout_reclasses_each_event_by_the_serial_gap(tmp_path: Path) -> None:
    contexts = tmp_path / 'contexts.jsonl'
    rescored = tmp_path / 'rescored.jsonl'
    lines = [
        {
            'id': 'a',
            'input_ids': [1, 2],
            'tokens': [7, 9],
            'events': [
                {'point': 'p1', 'kind': 'gross', 'token': 7, 'h6s_gap': 3.8, 'position': 5},
                {'point': 'p2', 'kind': 'gross', 'token': 7, 'h6s_gap': 3.8, 'position': 5},
            ],
        },
        {
            'id': 'b',
            'input_ids': [3],
            'tokens': [4, 5],
            'events': [{'point': 'p1', 'kind': 'near', 'token': 4, 'h6s_gap': 0.9, 'position': 2}],
        },
        {
            'id': 't',
            'input_ids': [6],
            'tokens': [68189, 1756],
            'events': [{'point': 'p1', 'kind': 'target', 'token': 68189, 'h6s_gap': 0.0, 'position': 439}],
        },
    ]
    contexts.write_text(''.join(json.dumps(x) + '\n' for x in lines))
    classes = [
        {'id': 'a', 'tokens': [7, 9], 'logprobs': [-8.1, -1.3], 'top5': [[-1.3, 9]], 'stock_top1': 9},
        {'id': 'b', 'tokens': [4, 5], 'logprobs': [-1.0, -1.2], 'top5': [[-1.0, 4]], 'stock_top1': 4},
        {'id': 't', 'tokens': [68189, 1756], 'logprobs': [-1.3, -8.1], 'top5': [[-1.3, 68189]], 'stock_top1': 68189},
    ]
    rescored.write_text(''.join(json.dumps(x) + '\n' for x in classes))
    result = score_report.serial_readout(contexts, rescored)
    assert result['gross_h6s'] == 2 and result['gross_serial'] == 2
    assert result['near_h6s'] == 1 and result['near_serial'] == 0
    assert result['class_moves'] == {'gross->gross': 2, 'near->tie': 1}
    assert result['target_439'][0]['serial_gap'] == 0.0
    assert result['missing'] == 0


def test_seeded_plan_is_fixed_and_never_holds_the_partner() -> None:
    keys = [f'{i:08x}' for i in range(40)]
    keys[30] = '579ae7ce' + keys[30][8:]
    plan_a = control_waves.small_plan(keys, partner=3)
    plan_b = control_waves.small_plan(keys, partner=3)
    assert plan_a == plan_b
    assert [w['size'] for w in plan_a] == [1, 8, 8, 8, 12, 12, 12, 16, 16, 16]
    for wave in plan_a:
        assert wave['members'][0] == 30 and 3 not in wave['members']
        assert len(set(wave['members'])) == wave['size']


def test_seeded_logprobs_keep_the_span_by_output_position() -> None:
    tops = [[[-0.1 * i, 100 + i, None]] for i in range(60)]
    ids = [[[-1.0, 1756, None], [-2.0, 9471, None]] for _ in range(60)]
    kept = control_waves.seeded_logprobs(
        {'output_top_logprobs': tops, 'output_token_ids_logprobs': ids}, first=400
    )
    assert sorted(map(int, kept)) == list(range(400, 451))
    assert kept['439']['top5'] == [[-0.1 * 39, 139]]
    assert kept['439']['tracked'] == {'1756': -1.0, '9471': -2.0}
