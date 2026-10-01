"""Declarative server arms (bench/arms.toml) and the SGLang command they produce.

An arm is a model revision plus a set of `sglang.launch_server` flags. Every arm
inherits `[defaults]`; its own `args` override the default flags, and command-line
overrides (`--set flag=value`, `--unset flag`) override both. Flags are written
without the leading dashes. Values map to the command line as follows:

- `true` adds a bare flag, `false` (or `--unset`) removes it;
- a list adds one flag followed by several values;
- anything else adds the flag followed by its string value.

Every arm declares an exactness class (`EXACTNESS_CLASSES`): how its greedy outputs
relate to plain decoding with the default flags. `stock` keeps the reference
arithmetic (FlashInfer target attention, stock GDN state handling); speculative
arms with stock verify are `stock`, and draft-only kernels do not change the
class. Stock arms may set only flags in `NEUTRAL_FLAGS` (an allowlist) and
FlashInfer target attention. `pending` arms change something else
(`numerics_changes`) and wait for classification of their greedy outputs against
their matched stock reference (bench/README.md, "Exactness classes"):
`exact-up-to-rounding` arms diverge from it only at rounding-level positions;
`lossy` arms do not, or change the model's arithmetic by design, and need a
quality measurement. An arm without a class, or
a `stock` arm in arms.toml that sets a numerics-changing flag, is an error. A
`--set` override that adds such a flag to a `stock` arm makes it `pending`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS_FILE = Path(__file__).with_name('arms.toml')

EXACTNESS_CLASSES = ('stock', 'exact-up-to-rounding', 'pending', 'lossy')
# Classes whose outputs match plain decoding as closely as batch shape allows.
EXACT_CLASSES = frozenset({'stock', 'exact-up-to-rounding'})
# Flags known to leave the target's arithmetic as in the reference configuration
# (plain decoding, FlashInfer target attention, stock GDN state handling). Any other
# flag, any environment variable outside NEUTRAL_ENV, a target attention backend
# other than FlashInfer, or another model or revision counts as a numerics change.
NEUTRAL_FLAGS = frozenset(
    {
        # capacity and memory
        'max-running-requests',
        'max-mamba-cache-size',
        'mem-fraction-static',
        'max-total-tokens',
        # prefix cache (the workload flushes it before every point)
        'disable-radix-cache',
        # CUDA graphs replay the same kernels; padding a batch to a captured size
        # changes batch shape only, which the batch-shape floor covers
        'cuda-graph-max-bs',
        'cuda-graph-bs',
        # observability, seeding (greedy decoding draws no random numbers), frontend
        'enable-metrics',
        'random-seed',
        'stream-interval',
        'incremental-streaming-output',
        'tokenizer-worker-num',
        'detokenizer-worker-num',
        # the vision encoder, which text requests never reach
        'mm-attention-backend',
        # speculation with SGLang's stock verify; draft-side kernels never change
        # which tokens the target accepts under greedy verification
        'speculative-algorithm',
        'speculative-num-steps',
        'speculative-eagle-topk',
        'speculative-num-draft-tokens',
        'speculative-draft-model-path',
        'speculative-draft-model-revision',
        'speculative-dflash-block-size',
        'speculative-draft-attention-backend',
        'speculative-adaptive',
    }
)
NEUTRAL_ENV = frozenset({'SGLANG_FLASHINFER_WORKSPACE_SIZE'})
REFERENCE_ATTENTION_BACKEND = 'flashinfer'

ArgValue = str | int | float | bool | list[str | int | float]

# Flags the launcher sets itself; an arm may not override them.
RESERVED_FLAGS = frozenset({'model-path', 'revision', 'host', 'port'})


@dataclass(frozen=True)
class Arm:
    """A fully resolved server configuration."""

    name: str
    description: str
    model: str
    revision: str
    args: dict[str, ArgValue]
    env: dict[str, str] = field(default_factory=dict)
    max_concurrency: int = 128
    # What the arm changes numerically relative to the BF16 checkpoint (FP8
    # weights, FP8 KV, BF16 GDN state, ...); empty for arms that keep the model's
    # arithmetic. Reports must pair a lossy arm with a quality measurement.
    lossy: str = ''
    # One of EXACTNESS_CLASSES (see the module docstring), and for an
    # exact-up-to-rounding arm the evidence for that class. `lossy` stays empty on
    # stock and exact arms, so a lossy note added in code always changes the class.
    exactness: str = ''
    exactness_note: str = ''
    # Decode/verify CUDA graphs must cover every batch size up to capacity. An arm
    # whose capacity exceeds the graph range it can afford sets this to false and
    # then runs its largest batches eagerly (reported, not hidden).
    require_full_graph_coverage: bool = True

    def __post_init__(self) -> None:
        # An arm built in code with a lossy note but no class, or with a stock,
        # exact or pending class inherited from its base arm, takes its class from
        # the note.
        if self.lossy and self.exactness in ('', 'stock', 'exact-up-to-rounding', 'pending'):
            note = self.lossy.lower()
            object.__setattr__(
                self, 'exactness', 'pending' if note.startswith('pending') else 'lossy'
            )

    @property
    def speculative(self) -> bool:
        return bool(self.args.get('speculative-algorithm'))

    def to_json(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'model': self.model,
            'revision': self.revision,
            'args': dict(self.args),
            'env': dict(self.env),
            'max_concurrency': self.max_concurrency,
            'lossy': self.lossy,
            'exactness': self.exactness,
            'exactness_note': self.exactness_note,
            'require_full_graph_coverage': self.require_full_graph_coverage,
        }


def parse_value(text: str) -> ArgValue:
    """Parse a `--set` value: true/false, int, float, comma list, else string."""
    lowered = text.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if ',' in text:
        return [_scalar(part) for part in text.split(',') if part]
    return _scalar(text)


def _scalar(text: str) -> str | int | float:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_overrides(sets: list[str], unsets: list[str]) -> dict[str, ArgValue]:
    """Turn `flag=value` strings and bare flag names into an override mapping."""
    overrides: dict[str, ArgValue] = {}
    for item in sets:
        flag, sep, value = item.partition('=')
        if not sep or not flag:
            raise ValueError(f'override {item!r} is not of the form flag=value')
        overrides[flag.lstrip('-')] = parse_value(value)
    for flag in unsets:
        overrides[flag.lstrip('-')] = False
    return overrides


def numerics_changes(
    args: dict[str, Any],
    env: dict[str, str] | None = None,
    model: tuple[str, str] | None = None,
    path: Path = ARMS_FILE,
) -> list[str]:
    """What in a configuration may change the target's arithmetic (see NEUTRAL_FLAGS).

    `model` is (model, revision); it counts when it differs from [defaults].
    """
    changes = []
    for flag, value in args.items():
        if value is None or value is False:
            continue
        if flag == 'attention-backend':
            if value != REFERENCE_ATTENTION_BACKEND:
                changes.append(f'attention-backend={value}')
        elif flag not in NEUTRAL_FLAGS:
            changes.append(flag if value is True else f'{flag}={value}')
    changes += [f'env {name}' for name in (env or {}) if name not in NEUTRAL_ENV]
    if model is not None:
        defaults = load_arms(path).get('defaults', {})
        if tuple(model) != (defaults.get('model'), defaults.get('revision')):
            changes.append(f'model {model[0]}@{model[1]}')
    return changes


def load_arms(path: Path = ARMS_FILE) -> dict[str, Any]:
    with path.open('rb') as handle:
        return tomllib.load(handle)


def arm_names(path: Path = ARMS_FILE) -> list[str]:
    return list(load_arms(path).get('arms', {}))


def resolve_arm(
    name: str,
    overrides: dict[str, ArgValue] | None = None,
    env_overrides: dict[str, str] | None = None,
    path: Path = ARMS_FILE,
) -> Arm:
    """Merge defaults, the named arm and overrides into one `Arm`."""
    spec = load_arms(path)
    arms = spec.get('arms', {})
    if name not in arms:
        known = ', '.join(sorted(arms))
        raise KeyError(f'unknown arm {name!r}; known arms: {known}')
    defaults = spec.get('defaults', {})
    entry = arms[name]
    args: dict[str, ArgValue] = {}
    for source in (defaults.get('args', {}), entry.get('args', {}), overrides or {}):
        for flag, value in source.items():
            if flag in RESERVED_FLAGS:
                raise ValueError(f'flag --{flag} is set by the launcher, not by an arm')
            args[flag] = value
    args = {flag: value for flag, value in args.items() if value is not False}
    env = {**defaults.get('env', {}), **entry.get('env', {}), **(env_overrides or {})}
    exactness = str(entry.get('exactness', ''))
    lossy = str(entry.get('lossy', '')).strip()
    if exactness not in EXACTNESS_CLASSES:
        raise ValueError(f'arm {name!r}: exactness must be one of {EXACTNESS_CLASSES}')
    exactness_note = str(entry.get('exactness_note', '')).strip()
    if exactness in ('pending', 'lossy') and not lossy:
        raise ValueError(f'arm {name!r}: a {exactness} arm needs a lossy note saying why')
    if exactness in EXACT_CLASSES and lossy:
        raise ValueError(f'arm {name!r}: a {exactness} arm takes exactness_note, not lossy')
    if exactness == 'exact-up-to-rounding' and not exactness_note:
        raise ValueError(f'arm {name!r}: exact-up-to-rounding needs an exactness_note')
    model = (
        str(entry.get('model', defaults['model'])),
        str(entry.get('revision', defaults['revision'])),
    )
    own_env = {**defaults.get('env', {}), **entry.get('env', {})}
    own = numerics_changes(
        {**defaults.get('args', {}), **entry.get('args', {})}, own_env, model, path
    )
    if exactness == 'stock' and own:
        raise ValueError(f'arm {name!r} is stock but sets {", ".join(own)}')
    changes = numerics_changes(args, env, model, path)
    added = [change for change in changes if change not in own]
    if added and exactness in EXACT_CLASSES:
        # A classification holds for the arm's own flags only.
        exactness = 'pending'
        lossy = f'pending: override sets {", ".join(added)}'
    return Arm(
        name=name,
        description=str(entry.get('description', '')).strip(),
        model=str(entry.get('model', defaults['model'])),
        revision=str(entry.get('revision', defaults['revision'])),
        args=args,
        env={key: str(value) for key, value in env.items()},
        max_concurrency=int(entry.get('max_concurrency', defaults.get('max_concurrency', 128))),
        lossy=lossy,
        exactness=exactness,
        exactness_note=exactness_note if exactness == 'exact-up-to-rounding' else '',
        require_full_graph_coverage=bool(entry.get('require_full_graph_coverage', True)),
    )


def flag_tokens(args: dict[str, ArgValue]) -> list[str]:
    """Render resolved args as command-line tokens, in insertion order."""
    tokens: list[str] = []
    for flag, value in args.items():
        if value is False:
            continue
        tokens.append(f'--{flag}')
        if value is True:
            continue
        if isinstance(value, list):
            tokens.extend(str(item) for item in value)
        else:
            tokens.append(str(value))
    return tokens


def server_command(arm: Arm, python: str, host: str, port: int) -> list[str]:
    """The full `sglang.launch_server` command for an arm."""
    return [
        python,
        '-m',
        'sglang.launch_server',
        '--model-path',
        arm.model,
        '--revision',
        arm.revision,
        '--host',
        host,
        '--port',
        str(port),
        *flag_tokens(arm.args),
    ]
