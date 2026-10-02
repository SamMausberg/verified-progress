"""CPU-only comparison of the INT4 checkpoints with the BF16 reference checkpoint.

    python -m experiments.lossy.checkpoint_check --out evidence/lossy/checkpoint_check.json

Reads only local files from the Hugging Face cache (no GPU, no download):

* weight bytes per tensor group from each safetensors header, and the bytes a
  decode step reads (language-model weights and the tied head; the vision tower
  and the MTP layer are not read when serving text without speculation);
* SHA-256 of the tokenizer and chat-template files;
* the end-of-sequence ids SGLang derives from each checkpoint
  (ModelConfig.hf_eos_token_id) and each tokenizer's eos/pad ids, because the
  GSM8K runs stop at EOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

CACHE = Path.home() / '.cache/huggingface/hub'
CHECKPOINTS = {
    'bf16_target': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'int4_target': ('nota-ai/Qwen3.5-4B-QAD-W4A16', 'a67b0fedb2b39cb057da6114e656b76b52d321b4'),
    'bf16_drafter': ('z-lab/Qwen3.5-4B-DFlash', '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf'),
    'int4_drafter': (
        'nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16',
        'c5fb290e47e30c81d06e48b0495ec06f2560dd4e',
    ),
}
INT4_EOS_OVERRIDE = '{"eos_token_id": 248044}'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                   'chat_template.jinja')  # fmt: skip


def snapshot(repo: str, revision: str) -> Path:
    path = CACHE / f'models--{repo.replace("/", "--")}' / 'snapshots' / revision
    if not path.is_dir():
        raise SystemExit(f'{repo}@{revision} is not in the local cache ({path})')
    return path


def header(path: Path) -> dict[str, Any]:
    with path.open('rb') as handle:
        (size,) = struct.unpack('<Q', handle.read(8))
        meta: dict[str, Any] = json.loads(handle.read(size))
    meta.pop('__metadata__', None)
    return meta


def group(name: str) -> str:
    if 'visual' in name:
        return 'vision'
    if name.startswith('mtp.'):
        return 'mtp'
    if 'embed_tokens' in name:
        return 'embedding (tied head)'
    if name.endswith('weight_packed'):
        return 'packed int4 weights'
    if name.endswith('weight_scale'):
        return 'group scales'
    if name.endswith('weight_shape'):
        return 'shape records'
    return 'other weights'


def weight_bytes(root: Path) -> dict[str, Any]:
    files = sorted(root.glob('*.safetensors'))
    if not files:
        raise SystemExit(f'no safetensors files in {root}')
    groups: Counter[str] = Counter()
    tensors = 0
    for file in files:
        for name, info in header(file).items():
            start, end = info['data_offsets']
            groups[group(name)] += end - start
            tensors += 1
    decode = sum(b for g, b in groups.items() if g not in ('vision', 'mtp'))
    return {
        'tensors': tensors,
        'bytes_by_group': dict(sorted(groups.items())),
        'total_bytes': sum(groups.values()),
        'decode_step_bytes': decode,
    }


def eos_ids(repo: str, revision: str, override: str = '{}') -> dict[str, Any]:
    """The token ids that end a request in SGLang (Req._check_token_based_finish):
    ModelConfig.hf_eos_token_id, the tokenizer's eos id and its additional stop ids."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.utils.hf_transformers import attach_additional_stop_token_ids
    from transformers import AutoTokenizer

    config = ModelConfig(repo, revision=revision, model_override_args=override)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    attach_additional_stop_token_ids(tok)
    hf_eos = set(config.hf_eos_token_id or [])
    stop = hf_eos | {tok.eos_token_id} | set(tok.additional_stop_token_ids or [])
    return {
        'model_override_args': override,
        'sglang_hf_eos_token_id': sorted(hf_eos),
        'tokenizer_eos': [tok.eos_token, tok.eos_token_id],
        'tokenizer_pad': [tok.pad_token, tok.pad_token_id],
        'additional_stop_token_ids': sorted(tok.additional_stop_token_ids or []),
        'stop_set': sorted(stop),
        'generation_config_present': (snapshot(repo, revision) / 'generation_config.json').exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    repo_dir = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        'repo_commit': subprocess.run(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'repo_dirty': bool(
            subprocess.run(
                ['git', '-C', str(repo_dir), 'status', '--porcelain', '--untracked-files=no'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        ),
    }
    for key, (repo, revision) in CHECKPOINTS.items():
        root = snapshot(repo, revision)
        entry: dict[str, Any] = {'repo': repo, 'revision': revision, **weight_bytes(root)}
        if key.endswith('target'):
            entry['tokenizer_sha256'] = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in TOKENIZER_FILES
                if (root / name).exists()
            }
            entry['eos'] = eos_ids(repo, revision)
        result[key] = entry
    # The override the INT4 arms in bench/arms.toml pass (json-model-override-args).
    int4_repo, int4_revision = CHECKPOINTS['int4_target']
    result['int4_target']['eos_with_arm_override'] = eos_ids(
        int4_repo, int4_revision, INT4_EOS_OVERRIDE
    )
    result['stop_sets_equal_without_override'] = (
        result['bf16_target']['eos']['stop_set'] == result['int4_target']['eos']['stop_set']
    )
    result['stop_sets_equal_with_override'] = (
        result['bf16_target']['eos']['stop_set']
        == result['int4_target']['eos_with_arm_override']['stop_set']
    )
    bf16 = result['bf16_target']['decode_step_bytes']
    int4 = result['int4_target']['decode_step_bytes']
    result['decode_step_ratio_bf16_over_int4'] = bf16 / int4
    result['tokenizer_files_identical'] = {
        name: result['bf16_target']['tokenizer_sha256'].get(name)
        == result['int4_target']['tokenizer_sha256'].get(name)
        for name in TOKENIZER_FILES
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    keys = (
        'decode_step_ratio_bf16_over_int4',
        'tokenizer_files_identical',
        'stop_sets_equal_without_override',
        'stop_sets_equal_with_override',
    )
    print(json.dumps({k: result[k] for k in keys}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
