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
class. `pending` arms set a numerics-changing flag (`numerics_changes`) and wait
for classification of their outputs against the batch-shape floor;
`exact-up-to-floor` arms passed it; `lossy` arms did not, or change the model's
arithmetic by design, and need a quality measurement. An arm without a class, or
a `stock` arm in arms.toml that sets a numerics-changing flag, is an error. A
`--set` override that adds such a flag to a `stock` arm makes it `pending`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS_FILE = Path(__file__).with_name('arms.toml')

EXACTNESS_CLASSES = ('stock', 'exact-up-to-floor', 'pending', 'lossy')
# Classes whose outputs match plain decoding as closely as batch shape allows.
EXACT_CLASSES = frozenset({'stock', 'exact-up-to-floor'})
# Flags that change the target model's arithmetic relative to the reference
# configuration (any value other than unset or false counts).
NUMERICS_FLAGS = (
    'enable-linear-replayssm',
    'enable-linear-replayssm-spec',
    'linear-attn-decode-backend',
    'kv-cache-dtype',
    'quantization',
    'enable-fp32-lm-head',
    'enable-deterministic-inference',
)
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
    # One of EXACTNESS_CLASSES (see the module docstring).
    exactness: str = ''
    # Decode/verify CUDA graphs must cover every batch size up to capacity. An arm
    # whose capacity exceeds the graph range it can afford sets this to false and
    # then runs its largest batches eagerly (reported, not hidden).
    require_full_graph_coverage: bool = True

    def __post_init__(self) -> None:
        # An arm built in code with a lossy note but no class (or the stock class
        # inherited from its base arm) takes its class from the note.
        if self.lossy and self.exactness in ('', 'stock'):
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


def numerics_changes(args: dict[str, Any]) -> list[str]:
    """The flags in `args` that change the target's arithmetic."""
    changes = [flag for flag in NUMERICS_FLAGS if args.get(flag) not in (None, False)]
    backend = args.get('attention-backend', REFERENCE_ATTENTION_BACKEND)
    if backend != REFERENCE_ATTENTION_BACKEND:
        changes.append(f'attention-backend={backend}')
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
    if exactness != 'stock' and not lossy:
        raise ValueError(f'arm {name!r}: a {exactness} arm needs a lossy note saying why')
    changes = numerics_changes(args)
    if exactness == 'stock' and changes:
        own = {**defaults.get('args', {}), **entry.get('args', {})}
        if numerics_changes(own):
            raise ValueError(f'arm {name!r} is stock but sets {", ".join(changes)}')
        exactness = 'pending'
        lossy = f'pending: override sets {", ".join(changes)}'
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
