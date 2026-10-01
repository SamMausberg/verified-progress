"""The stack's equality gate: one decision, written once and checked before every timed run.

`build` (exit status 1 when the gate is rejected, i.e. ok is false) reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py (plus the certified head's check-mode statistics) and writes
gate.json. `check` is the single precondition every timed hold calls: it recomputes the
decision from the same files, compares it with gate.json, verifies the routing table and
the certified-head package on disk, and prints the plan (`full=<arm>`, `table=<path>`);
it exits non-zero, printing why, unless every precondition holds.

The decision is all or nothing:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree), and F, G
  and FG must each be exact against both B0 and stock DFlash block 16. A comparison is
  usable only if it covers all 320 prompts with no output-length mismatch; it is exact
  only if every first divergence is classified `tie`, `one_ulp` or `near` (anything else,
  including `unknown` when a run lacks the logprobs, fails); bitwise only if the two
  runs' raw outputs are identical, token ids and complete top-logprob arrays, prompt by
  prompt, and both runs carry all TOP_K entries at every output position (missing or
  truncated logprobs fail; the comparator's drift statistic ignores low-probability
  entries).
* The routing table G ran with is recorded by its SHA-256; a session must use that file.
* The certified head (H) joins FG only if the tokens-only B0 run reproduces the logprob
  B0 run, H and FGH reproduce B0 and FG (tokens and lengths, all 320 prompts), both
  check-mode statistics show certified verify rows with mismatch_rows exactly 0, and the
  package's SHA-256 (over its files) is recorded. A session whose gate includes H must
  name that exact package.

Engine provenance is part of the gate. Every hold first runs `preflight`: the repository
running it, the stock SGLang checkout S0 runs (~/sglang, which must be at the pin
bd66ce343e) and the composed worktree (whose tree must be STACK_TREE) must all have no
uncommitted changes to tracked files and no untracked files under python/. `build`
records that identity (commits, trees, the SGLang venv's key package versions) in
gate.json, requires it unchanged since the hold's preflight, and requires every equality
run's own record to name those commits with a clean tree. `check` refuses a session
whose current identity differs from the gate's in any field.

    python experiments/stack/equality_gate.py preflight --repo . --s0 ~/sglang \
        --stack-engine ~/sglang-wt/stack --stack-tree <tree> --out <run>/identity.json
    python experiments/stack/equality_gate.py build ~/vp-data/stack/equality/<run> \
        --table <run>/backbone_table_v1.json [--cert-src ~/vp-wt/stack-cert/src] \
        --repo . --s0 ~/sglang --stack-engine ~/sglang-wt/stack --stack-tree <tree>
    python experiments/stack/equality_gate.py check --gate ~/vp-data/stack/equality/current/gate.json \
        [--cert-src ~/vp-wt/stack-cert/src] [--pin ~/vp-data/stack/campaign_gate.json] \
        --repo . --s0 ~/sglang --stack-engine ~/sglang-wt/stack --stack-tree <tree>

`--pin` binds a campaign of sessions to one gate: the first check that passes every
precondition writes the gate's and its comparison summary's SHA-256 and the run
directory there, and every later check refuses a different gate. A failed check never
writes a pin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXACT_CLASSES = ('tie', 'one_ulp', 'near')
PROMPTS = 320
TOP_K = 5  # top logprobs requested at every position by the equality runs
TABLE = 'backbone_table_v1.json'
S0_COMMIT = 'bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824'  # the paper's SGLang pin
PACKAGES = ('torch', 'triton', 'flashinfer-python', 'sgl-kernel', 'transformers')
# Environment variables that can change an engine's arithmetic or kernels; holds clear them
# (arms.sh), so only ENV_ALLOWED may remain in the identity.
ENV_PREFIXES = ('SGLANG_', 'TORCH_', 'PYTORCH_', 'TRITON_', 'FLASHINFER_', 'NCCL_', 'CUDA_')
ENV_ALLOWED = ('CUDA_HOME',)
# The equality runs the plan declares (hold_equality.sh), and the environment each lever adds.
CORE_RUNS = ('S0', 'B0', 'F', 'G', 'FG')
H_RUNS = ('B0_tokens', 'H_tokens', 'FGH_tokens')
LEVER_ENV = {
    'F': ('SGLANG_GDN_REPLAYSSM_FOLD',),
    'G': (
        'SGLANG_BACKBONE_GEMM',
        'SGLANG_BACKBONE_PDL',
        'SGLANG_BACKBONE_MERGE_IN_PROJ',
        'SGLANG_BACKBONE_GEMM_TABLE',
    ),
    'H': (
        'SGLANG_CERTIFIED_HEAD_VERIFY',
        'SGLANG_CERTIFIED_HEAD_SRC',
        'SGLANG_CERTIFIED_HEAD_FALLBACK',
        'SGLANG_CERTIFIED_HEAD_MODEL',
        'SGLANG_CERTIFIED_HEAD_MAX_ROWS',
        'SGLANG_CERTIFIED_HEAD_CHECK',
        'SGLANG_CERTIFIED_HEAD_STATS',
    ),
}
FOLD_FLAG = '--enable-linear-replayssm-spec'
# Bench's reference configuration for DFlash block 16 (bench/campaigns/equality_tuned.sh):
# the two reused stock runs and every stack equality run (plus its lever flags) use it.
DFLASH_B16_FLAGS = [
    '--speculative-algorithm', 'DFLASH',
    '--speculative-draft-model-path', 'z-lab/Qwen3.5-4B-DFlash',
    '--speculative-draft-model-revision', '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf',
    '--speculative-dflash-block-size', '16',
    '--max-running-requests', '4',
    '--disable-radix-cache',
]  # fmt: skip
TRITON_FLAGS = ['--attention-backend', 'triton']
REF_FLAGS = {
    'ref_dflash_b16': DFLASH_B16_FLAGS,
    'ref_dflash_b16_triton': DFLASH_B16_FLAGS + TRITON_FLAGS,
}
# The pass every equality run makes (state's runner): one request at a time, fresh cache.
RUN_SETTINGS = {'pass': 'c1', 'concurrency': 1, 'max_new_tokens': 256, 'warm': False}


class GateError(Exception):
    """A precondition of a timed run does not hold."""


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as f:
        return {r['id']: r for r in (json.loads(line) for line in f if line.strip())}


def runs_identical(run: Path, label: str) -> bool:
    """Both runs of pair `label` have identical token ids and top-logprob arrays."""
    pairs = {p[0]: p[1:] for p in json.loads((run / 'pairs.json').read_text())}
    if label not in pairs:
        return False
    a, b = (load_run(run / 'runs' / f'{r}.jsonl') for r in pairs[label])
    if len(a) != PROMPTS or a.keys() != b.keys():
        return False
    return all(
        full_coverage(a[k])
        and full_coverage(b[k])
        and a[k]['output_ids'] == b[k]['output_ids']
        and a[k]['top_logprobs'] == b[k]['top_logprobs']
        for k in a
    )


def full_coverage(rec: dict[str, Any]) -> bool:
    """The record carries TOP_K logprob entries at every output position."""
    tops = rec.get('top_logprobs')
    ids = rec.get('output_ids')
    return (
        isinstance(tops, list)
        and isinstance(ids, list)
        and len(ids) > 0
        and len(tops) == len(ids)
        and all(isinstance(t, list) and len(t) == TOP_K for t in tops)
    )


def usable(pair: dict[str, Any] | None) -> bool:
    return pair is not None and pair['prompts'] == PROMPTS and pair['length_mismatch'] == 0


def identical_tokens(pair: dict[str, Any] | None) -> bool:
    return pair is not None and usable(pair) and pair['diverged'] == 0


def bitwise(pair: dict[str, Any] | None) -> bool:
    return pair is not None and identical_tokens(pair) and pair['drift_max'] == 0


def exact(pair: dict[str, Any] | None) -> bool:
    if pair is None or not usable(pair):
        return False
    classes = pair['classes']
    return (
        all(k in EXACT_CLASSES for k, n in classes.items() if n)
        and sum(classes.values()) == pair['diverged']
    )


def lever_class(run: Path, pairs: dict[str, Any], lever: str) -> str:
    stock = pairs.get(f'{lever} vs bench stock b16')
    own = pairs.get(f'{lever} vs B0')
    if not (exact(stock) and exact(own)):
        return 'not exact'
    if bitwise(own) and runs_identical(run, f'{lever} vs B0'):
        return 'bitwise'
    return 'exact-up-to-rounding'


def certified_check(path: Path) -> dict[str, Any]:
    """The verify path's check-mode counters: rows certified and rows differing."""
    out: dict[str, Any] = {'file': path.name, 'ok': False}
    if not path.is_file():
        out['reason'] = 'missing'
        return out
    verify = json.loads(path.read_text()).get('paths', {}).get('verify')
    if not isinstance(verify, dict) or 'mismatch_rows' not in verify:
        out['reason'] = 'no verify counters'
        return out
    out.update(rows=verify.get('rows', 0), mismatch_rows=verify['mismatch_rows'])
    out['fallback_rows'] = verify.get('fallback_rows')
    out['ok'] = out['rows'] > 0 and out['mismatch_rows'] == 0
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(src: Path) -> str:
    """SHA-256 over the certified_head package's files (paths and bytes, sorted)."""
    h = hashlib.sha256()
    pkg = src / 'certified_head'
    if not pkg.is_dir():
        raise GateError(f'no certified_head package under {src}')
    for f in sorted(p for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(str(f.relative_to(pkg)).encode() + b'\0' + f.read_bytes() + b'\0')
    return h.hexdigest()


def _git(path: Path, *args: str) -> str:
    out = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise GateError(f'git {" ".join(args)} failed in {path}: {out.stderr.strip()}')
    return out.stdout.rstrip('\n')  # keep porcelain's leading status columns


def tree_state(path: Path, untracked_under: str | None) -> dict[str, Any]:
    """HEAD, tree and every uncommitted change of a checkout: tracked files, plus untracked
    (not gitignored) files under `untracked_under` ('.' for the whole checkout)."""
    dirty = _git(path, 'status', '--porcelain', '--untracked-files=no').splitlines()
    if untracked_under:
        listing = _git(
            path, 'status', '--porcelain', '--untracked-files=all', '--', untracked_under
        )
        dirty += [line for line in listing.splitlines() if line.startswith('??')]
    return {
        'path': str(path),
        'head': _git(path, 'rev-parse', 'HEAD'),
        'tree': _git(path, 'rev-parse', 'HEAD^{tree}'),
        'dirty': sorted(set(dirty)),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def engine_env() -> dict[str, str]:
    import os

    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


def identity(
    repo: Path, s0: Path, stack_engine: Path, cert_src: Path | None = None
) -> dict[str, Any]:
    """What a timed run's numbers depend on outside the gate's own files."""
    return {
        'cert_package': fingerprint(cert_src) if cert_src else None,
        # Any untracked file in the repository (a stray sitecustomize.py, a local module)
        # can be imported by every hold process.
        'repo': tree_state(repo, '.'),
        's0': tree_state(s0, 'python'),
        'stack_engine': tree_state(stack_engine, 'python'),
        'packages': package_versions(),
        'env': engine_env(),
    }


def require_clean(ident: dict[str, Any], stack_tree: str) -> None:
    """The absolute conditions: no checkout dirty, S0 at the pin, the composed tree declared."""
    problems = [
        f'{k} has uncommitted changes {ident[k]["dirty"][:5]}'
        for k in ('repo', 's0', 'stack_engine')
        if ident[k]['dirty']
    ]
    if ident['s0']['head'] != S0_COMMIT:
        problems.append(f's0 is at {ident["s0"]["head"]}, not the pin {S0_COMMIT}')
    if ident['stack_engine']['tree'] != stack_tree:
        problems.append(f'composed tree {ident["stack_engine"]["tree"]} is not {stack_tree}')
    stray = sorted(set(ident['env']) - set(ENV_ALLOWED))
    if stray:
        problems.append(f'engine environment variables set: {stray}')
    if problems:
        raise GateError('; '.join(problems))


def identity_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Fields (other than checkout paths) in which two identities differ."""
    diffs = [
        f'{k}.{f}'
        for k in ('repo', 's0', 'stack_engine')
        for f in ('head', 'tree', 'dirty')
        if old[k][f] != new[k][f]
    ]
    for key in ('packages', 'env', 'cert_package'):
        if old[key] != new[key]:
            diffs.append(key)
    return diffs


REFERENCES = ('ref_dflash_b16', 'ref_dflash_b16_triton')  # bench's stock runs, reused


def _prompt_tokens(path: Path) -> dict[str, Any]:
    """id -> prompt token count of a run's records."""
    return {r['id']: r.get('prompt_tokens') for r in load_run(path).values()}


def runs_provenance(
    run: Path, ident: dict[str, Any], prompt_ids: set[str] | None = None
) -> list[str]:
    """Each declared equality run and each reused reference: its record matches the
    declared configuration (engine commit, clean tree, flags, pass, logprobs, prompt set
    and model), and every run answered exactly the declared prompts with the same prompt
    token counts as S0's run."""
    problems = []
    s0_meta = run / 'runs' / 'plain__stack_S0' / 'c1.meta.json'
    model = json.loads(s0_meta.read_text()).get('model_revision') if s0_meta.is_file() else None
    s0_out = run / 'runs' / 'plain__stack_S0' / 'c1.jsonl'
    s0_tokens = _prompt_tokens(s0_out) if s0_out.is_file() else {}
    if prompt_ids is None or set(s0_tokens) != prompt_ids:
        problems.append('S0 did not answer exactly the declared prompts')

    def same_prompts(name: str, out_path: Path) -> None:
        if not out_path.is_file() or _prompt_tokens(out_path) != s0_tokens:
            problems.append(f'{name}: prompts or prompt token counts differ from S0')

    for ref in REFERENCES:
        meta_path = run / 'runs' / ref / 'c1.meta.json'
        if not meta_path.is_file():
            problems.append(f'{ref}: no run record')
            continue
        meta = json.loads(meta_path.read_text())
        expect = {
            **RUN_SETTINGS,
            'flags': REF_FLAGS[ref],
            'top_logprobs_num': TOP_K,
            'num_prompts': PROMPTS,
        }
        for key, value in expect.items():
            if meta.get(key) != value:
                problems.append(f'{ref}: {key} {meta.get(key)!r}, declared {value!r}')
        same_prompts(ref, run / 'runs' / ref / 'c1.jsonl')
        if meta.get('sglang_sha') != ident['s0']['head'] or meta.get('sglang_dirty') is not False:
            problems.append(
                f'{ref}: engine {meta.get("sglang_sha")} dirty={meta.get("sglang_dirty")}'
            )
        if model is None or meta.get('model_revision') != model:
            problems.append(f'{ref}: model {meta.get("model_revision")}, S0 ran {model}')
    plan_path = run / 'plan.jsonl'
    if not plan_path.is_file():
        return [*problems, 'no plan.jsonl (the declared equality runs)']
    plan = {}
    for line in plan_path.read_text().splitlines():
        row = json.loads(line)
        if row['run'] in plan:
            problems.append(f'{row["run"]}: declared twice')
        plan[row['run']] = row
    declared = {f'plain__stack_{t}' for t in CORE_RUNS}
    h_runs = {f'plain__stack_{t}' for t in H_RUNS}
    missing = sorted(declared - set(plan))
    if missing:
        problems.append(f'plan lacks {missing}')
    if set(plan) & h_runs and not h_runs <= set(plan):
        problems.append(f'plan has only part of the certified-head runs {sorted(h_runs)}')
    undeclared = sorted(set(plan) - declared - h_runs)
    if undeclared:
        problems.append(f'plan has runs the hold does not declare: {undeclared}')
    present = {p.name for p in (run / 'runs').glob('plain__stack_*') if p.is_dir()}
    if present - set(plan):
        problems.append(f'runs not in the plan: {sorted(present - set(plan))}')
    for name, row in sorted(plan.items()):
        tag = name.removeprefix('plain__stack_')
        levers = tag.removesuffix('_tokens') if tag not in ('S0', 'B0', 'B0_tokens') else ''
        want_env = sorted(e for lever in levers for e in LEVER_ENV[lever])
        if row['env'] != want_env:
            problems.append(f'{name}: declared environment {row["env"]}, lever wants {want_env}')
        if (FOLD_FLAG in row['flags']) != ('F' in levers):
            problems.append(f'{name}: fold flag does not match the lever')
        want_engine = 's0' if tag == 'S0' else 'stack_engine'
        if row['engine'] != want_engine:
            problems.append(f'{name}: declared on {row["engine"]}, not {want_engine}')
        meta_path = run / 'runs' / name / 'c1.meta.json'
        if not meta_path.is_file():
            problems.append(f'{name}: no run record')
            continue
        meta = json.loads(meta_path.read_text())
        want = ident[want_engine]['head']
        if meta.get('sglang_sha') != want or meta.get('sglang_dirty') is not False:
            problems.append(
                f'{name}: engine {meta.get("sglang_sha")} dirty={meta.get("sglang_dirty")}'
            )
        if meta.get('repo_sha') != ident['repo']['head']:
            problems.append(f'{name}: repository {meta.get("repo_sha")}')
        expect = {
            **RUN_SETTINGS,
            'flags': row['flags'],
            'top_logprobs_num': row['top_logprobs'],
            'num_prompts': PROMPTS,
            'model_revision': model,
        }
        for key, value in expect.items():
            if meta.get(key) != value:
                problems.append(f'{name}: {key} {meta.get(key)!r}, declared {value!r}')
        same_prompts(name, run / 'runs' / name / 'c1.jsonl')
        base = DFLASH_B16_FLAGS + TRITON_FLAGS
        if row['flags'][: len(base)] != base or set(row['flags'][len(base) :]) - {FOLD_FLAG}:
            problems.append(f'{name}: declared flags are not the reference configuration')
    return problems


def evaluate(
    run: Path,
    table_sha: str | None,
    package_sha: str | None,
    ident: dict[str, Any],
    prompts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`prompts`: {'sha256': ..., 'ids': [...]} of the declared prompt file (recorded in
    gate.json; check() passes the recorded value back)."""
    """The gate decision for an equality run directory (pure: reads only that directory)."""
    pairs = json.loads((run / 'summary.json').read_text())['pairs']
    gate: dict[str, Any] = {
        'b0_bitwise_to_s0': bitwise(pairs.get('B0 vs S0')) and runs_identical(run, 'B0 vs S0'),
        'classes': {x: lever_class(run, pairs, x) for x in ('F', 'G', 'FG')},
        'table_sha256': table_sha,
        'identity': ident,
        'prompts': prompts,
        'provenance_problems': runs_provenance(
            run, ident, set(prompts['ids']) if prompts else None
        ),
        'references_sha256': {
            ref: sha256_file(run / 'runs' / ref / 'c1.jsonl')
            for ref in REFERENCES
            if (run / 'runs' / ref / 'c1.jsonl').is_file()
        },
    }
    gate['ok'] = bool(
        gate['b0_bitwise_to_s0']
        and all(c != 'not exact' for c in gate['classes'].values())
        and table_sha
        and not gate['provenance_problems']
    )
    timed = ['F', 'G'] if gate['ok'] else []
    checks = [certified_check(run / f'certified_stats_{n}.json') for n in ('H', 'FGH')]
    gate['certified'] = {
        'tokens_identical': all(
            identical_tokens(pairs.get(k))
            for k in ('B0 tokens vs B0', 'H tokens vs B0 tokens', 'FGH tokens vs FG')
        ),
        'check_mode': checks,
        'package_sha256': package_sha,
    }
    if (
        timed
        and package_sha
        and gate['certified']['tokens_identical']
        and all(c['ok'] for c in checks)
    ):
        timed.append('H')
    gate['timed_levers'] = timed
    return gate


def check(
    gate_path: Path,
    cert_src: Path | None,
    pin: Path | None,
    ident: dict[str, Any],
    stack_tree: str,
) -> tuple[str, Path]:
    """Every precondition of a timed run; returns (full arm, routing table) or raises.
    `ident` is the identity of the checkouts this run would use, measured now."""
    require_clean(ident, stack_tree)
    if not gate_path.is_file():
        raise GateError(f'no gate at {gate_path}')
    stored = json.loads(gate_path.read_text())
    run = gate_path.resolve().parent
    if 'identity' not in stored:
        raise GateError('the gate records no engine identity')
    changed = identity_changes(stored['identity'], ident)
    if changed:
        raise GateError(f'engine identity differs from the equality run in {changed}')
    recomputed = evaluate(
        run,
        stored.get('table_sha256'),
        stored.get('certified', {}).get('package_sha256'),
        stored['identity'],
        stored.get('prompts'),
    )
    if recomputed != stored:
        raise GateError('gate.json does not match the decision recomputed from its run')
    if not stored['ok']:
        raise GateError(f'gate not passed: {json.dumps(stored)}')
    levers = stored['timed_levers']
    if levers[:2] != ['F', 'G'] or not set(levers) <= {'F', 'G', 'H'}:
        raise GateError(f'unexpected timed levers {levers}')
    table = run / TABLE
    if not table.is_file() or sha256_file(table) != stored['table_sha256']:
        raise GateError(f'routing table {table} is missing or not the one that passed')
    if 'H' in levers:
        if cert_src is None:
            raise GateError('the gate includes H but no certified_head package was named')
        if fingerprint(cert_src) != stored['certified']['package_sha256']:
            raise GateError(f'certified_head package under {cert_src} is not the one that passed')
    # Only a gate that passed every check above may pin or continue a campaign.
    if pin is not None:
        digest = campaign_digest(gate_path)
        if pin.is_file():
            if json.loads(pin.read_text()) != digest:
                raise GateError(f'this campaign is pinned to another gate ({pin})')
        else:
            pin.write_text(json.dumps(digest, indent=1) + '\n')
    return ''.join(levers), table


def campaign_digest(gate_path: Path) -> dict[str, str]:
    run = gate_path.resolve().parent
    return {
        'gate_sha256': sha256_file(gate_path),
        'summary_sha256': sha256_file(run / 'summary.json'),
        'run': str(run),
    }


def pinned_gate(pin: Path) -> dict[str, Any]:
    """The gate a campaign is pinned to, after checking the pinned files are unchanged."""
    digest = json.loads(pin.read_text())
    gate_path = Path(digest['run']) / 'gate.json'
    if campaign_digest(gate_path) != digest:
        raise GateError(f'the gate pinned in {pin} has changed on disk')
    gate: dict[str, Any] = json.loads(gate_path.read_text())
    return gate


def prompt_record(path: Path) -> dict[str, Any]:
    ids = [json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != PROMPTS or len(set(ids)) != PROMPTS:
        raise GateError(f'{path} does not hold {PROMPTS} distinct prompts')
    return {'path': str(path), 'sha256': sha256_file(path), 'ids': ids}


def build(
    run: Path,
    table: Path,
    cert_src: Path | None,
    ident: dict[str, Any],
    stack_tree: str,
    prompts: Path,
) -> int:
    """Write gate.json for an equality run; 1 if the gate is rejected."""
    require_clean(ident, stack_tree)
    if table.resolve() != (run / TABLE).resolve():
        raise GateError(f'the table must be {run / TABLE}')
    preflight = run / 'identity.json'
    if not preflight.is_file():
        raise GateError(f'no preflight identity at {preflight}')
    before = json.loads(preflight.read_text())
    changed = identity_changes(before, ident)  # includes the certified-head package
    if changed:
        raise GateError(f'engine identity changed during the hold in {changed}')
    record = prompt_record(prompts)
    if before.get('prompts_sha256') != record['sha256']:
        raise GateError('the equality prompts changed during the hold')
    gate = evaluate(run, sha256_file(table), ident['cert_package'], ident, record)
    # Written either way so a rejected gate can be inspected; check() refuses it.
    (run / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print(json.dumps(gate))
    if not gate['ok']:
        print('gate: rejected (ok is false)', file=sys.stderr)
        return 1
    return 0


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--repo', type=Path, required=True, help='the repository running the hold')
    p.add_argument('--s0', type=Path, required=True, help='the stock SGLang checkout (~/sglang)')
    p.add_argument('--stack-engine', type=Path, required=True, help='the composed worktree')
    p.add_argument('--stack-tree', required=True, help='the declared composed tree (STACK_TREE)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    pf = sub.add_parser('preflight', help='check the checkouts and record their identity')
    _engine_args(pf)
    pf.add_argument('--out', type=Path, required=True)
    pf.add_argument('--cert-src', type=Path, help='certified_head source dir, if H runs')
    pf.add_argument('--prompts', type=Path, help='the equality prompt file (equality hold only)')
    b = sub.add_parser('build', help='write gate.json for an equality run directory')
    b.add_argument('run', type=Path)
    b.add_argument('--table', type=Path, required=True, help='the routing table G ran with')
    b.add_argument('--cert-src', type=Path, help='certified_head source dir the H runs used')
    b.add_argument('--prompts', type=Path, required=True, help='the equality prompt file')
    _engine_args(b)
    c = sub.add_parser('check', help='verify every precondition of a timed run')
    c.add_argument('--gate', type=Path, required=True)
    c.add_argument('--cert-src', type=Path)
    c.add_argument('--pin', type=Path, help='campaign file binding sessions to one gate')
    _engine_args(c)
    f = sub.add_parser('fingerprint', help='print the certified_head package fingerprint')
    f.add_argument('src', type=Path)
    e = sub.add_parser('env', help='record the ambient engine environment (ENV_PREFIXES)')
    e.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    try:
        if args.cmd == 'fingerprint':
            print(fingerprint(args.src))
            return 0
        if args.cmd == 'env':
            args.out.write_text(json.dumps(engine_env(), indent=1) + '\n')
            return 0
        ident = identity(args.repo, args.s0, args.stack_engine, args.cert_src)
        if args.cmd == 'preflight':
            require_clean(ident, args.stack_tree)
            record = dict(ident)
            if args.prompts:
                record['prompts_sha256'] = prompt_record(args.prompts)['sha256']
            args.out.write_text(json.dumps(record, indent=1) + '\n')
            print(json.dumps(record))
        elif args.cmd == 'build':
            return build(args.run, args.table, args.cert_src, ident, args.stack_tree, args.prompts)
        else:
            full, table = check(args.gate, args.cert_src, args.pin, ident, args.stack_tree)
            print(f'full={full}')
            print(f'table={table}')
    except (GateError, OSError, KeyError, ValueError) as err:
        print(f'gate: {err}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
