"""Prompt-set manifests: the committed files regenerate byte for byte (CPU only)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments' / 'state_safety'))

from prompts import MAIN_SELECTION_TEXT, build_manifest

DATA = Path.home() / 'vp-data' / 'state' / 'prompts'
SETS = {
    'main': ('prompts.jsonl', 'prompt_manifest.json'),
    'fresh': ('prompts_fresh.jsonl', 'prompt_manifest_fresh.json'),
}


def _items():
    return [
        {'id': 'gsm8k-0000', 'source': 'gsm8k', 'thinking': False, 'input_ids': [1, 2, 3]},
        {'id': 'gsm8k-0001', 'source': 'gsm8k', 'thinking': True, 'input_ids': [4, 5]},
    ]


def test_main_manifest_keeps_the_original_layout():
    m = build_manifest(_items(), 'main')
    assert 'prompt_set' not in m
    assert m['selection'] == {
        **MAIN_SELECTION_TEXT,
        'thinking': 'every third prompt of each source (index % 3 == 2)',
    }
    assert list(m)[:4] == ['model', 'model_revision', 'sources', 'selection']


def test_other_sets_name_themselves_before_the_selection():
    m = build_manifest(_items(), 'fresh')
    assert list(m)[:5] == ['model', 'model_revision', 'sources', 'prompt_set', 'selection']
    assert m['selection']['mt_bench'] == 'none'


@pytest.mark.parametrize('prompt_set', sorted(SETS))
def test_committed_manifest_regenerates_byte_for_byte(prompt_set):
    prompts_file, manifest_file = SETS[prompt_set]
    path = DATA / prompts_file
    if not path.exists():
        pytest.skip(f'{path} is not available (built by prompts.py on the GPU host)')
    items = [json.loads(line) for line in path.read_text().splitlines()]
    text = json.dumps(build_manifest(items, prompt_set), indent=2) + '\n'
    assert text == (REPO / 'evidence' / 'state_safety' / manifest_file).read_text()
