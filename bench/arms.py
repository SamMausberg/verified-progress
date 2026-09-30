"""Declarative server arms (bench/arms.toml) and the SGLang command they produce.

An arm is a model revision plus a set of `sglang.launch_server` flags. Every arm
inherits `[defaults]`; its own `args` override the default flags, and command-line
overrides (`--set flag=value`, `--unset flag`) override both. Flags are written
without the leading dashes. Values map to the command line as follows:

- `true` adds a bare flag, `false` (or `--unset`) removes it;
- a list adds one flag followed by several values;
- anything else adds the flag followed by its string value.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS_FILE = Path(__file__).with_name('arms.toml')

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
    # Decode/verify CUDA graphs must cover every batch size up to capacity. An arm
    # whose capacity exceeds the graph range it can afford sets this to false and
    # then runs its largest batches eagerly (reported, not hidden).
    require_full_graph_coverage: bool = True

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
    return Arm(
        name=name,
        description=str(entry.get('description', '')).strip(),
        model=str(entry.get('model', defaults['model'])),
        revision=str(entry.get('revision', defaults['revision'])),
        args=args,
        env={key: str(value) for key, value in env.items()},
        max_concurrency=int(entry.get('max_concurrency', defaults.get('max_concurrency', 128))),
        lossy=str(entry.get('lossy', '')).strip(),
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
