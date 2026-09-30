#!/usr/bin/env python3
"""Exact-arithmetic reference for the research proposal, not a serving kernel.

Python >= 3.10; standard library only. Run:
    python prefix_value_reference.py --output validation_results.json

The recurrence optimizes a supplied substochastic Markov lattice. It does not
establish that pretrained selector scores are calibrated acceptance probabilities,
that a GPU implementation is correct, or that this policy improves a real model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Callable, Sequence

Matrix = tuple[tuple[F, ...], ...]
Lattice = tuple[Matrix, ...]


@dataclass(frozen=True)
class Certificate:
    """Bellman values and a maximizing action at every state and position."""
    values: tuple[tuple[F, ...], ...]
    choices: tuple[tuple[int, ...], ...]


def validate_lattice(r: Lattice, k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")
    for matrix in r:
        if len(matrix) != k or any(len(row) != k for row in matrix):
            raise ValueError("every transition matrix must have shape (k, k)")
        for row in matrix:
            if any(not isinstance(p, F) or p < 0 or p > 1 for p in row):
                raise ValueError("transitions must be Fractions in [0, 1]")
            if sum(row, F(0)) > 1:
                raise ValueError("a row's mass must be at most one")


def prefix_value(r: Lattice, path: Sequence[int], anchor: int = 0) -> F:
    if len(path) != len(r):
        raise ValueError("path length must equal the number of draft slots")
    survival, value, previous = F(1), F(0), anchor
    for matrix, current in zip(r, path):
        survival *= matrix[previous][current]
        value += survival
        previous = current
    return value


def whole_path_survival(r: Lattice, path: Sequence[int], anchor: int = 0) -> F:
    survival, previous = F(1), anchor
    for matrix, current in zip(r, path):
        survival *= matrix[previous][current]
        previous = current
    return survival


def optimize(r: Lattice, k: int, anchor: int = 0) -> tuple[tuple[int, ...], Certificate]:
    validate_lattice(r, k)
    if not 0 <= anchor < k:
        raise ValueError("anchor index out of range")
    values = [tuple(F(0) for _ in range(k)) for _ in range(len(r) + 1)]
    choices = [tuple(0 for _ in range(k)) for _ in r]
    for t in reversed(range(len(r))):
        rows, acts = [], []
        for a in range(k):
            rewards = tuple(r[t][a][b] * (1 + values[t + 1][b]) for b in range(k))
            b = max(range(k), key=lambda j: rewards[j])  # smallest index on ties
            rows.append(rewards[b])
            acts.append(b)
        values[t], choices[t] = tuple(rows), tuple(acts)
    path, a = [], anchor
    for actions in choices:
        a = actions[a]
        path.append(a)
    return tuple(path), Certificate(tuple(values), tuple(choices))


def check_certificate(r: Lattice, k: int, c: Certificate) -> bool:
    """Independent finite checker: upper bound for all actions, equality for one.

    Exact Fractions make these literal inequalities, not floating-point tests.
    This checks one supplied lattice. It is not a Lean theorem or a GPU proof.
    """
    try:
        validate_lattice(r, k)
        if len(c.values) != len(r) + 1 or len(c.choices) != len(r):
            return False
        if any(len(v) != k for v in c.values) or any(len(v) != k for v in c.choices):
            return False
        if any(v != 0 for v in c.values[-1]):
            return False
        for t, matrix in enumerate(r):
            for a, row in enumerate(matrix):
                b_star = c.choices[t][a]
                if not 0 <= b_star < k:
                    return False
                for b, p in enumerate(row):
                    q = p * (1 + c.values[t + 1][b])
                    if q > c.values[t][a]:
                        return False
                    if b == b_star and q != c.values[t][a]:
                        return False
        return True
    except (TypeError, ValueError, IndexError):
        return False


def greedy_path(r: Lattice, anchor: int = 0) -> tuple[int, ...]:
    path, a = [], anchor
    for matrix in r:
        a = max(range(len(matrix[a])), key=lambda b: matrix[a][b])
        path.append(a)
    return tuple(path)


def random_lattice(rng: random.Random, length: int, k: int) -> Lattice:
    layers = []
    for _ in range(length):
        rows = []
        for _ in range(k):
            counts = [rng.randrange(10) for _ in range(k)]
            outside = rng.randrange(1, 10)
            denominator = sum(counts) + outside
            rows.append(tuple(F(n, denominator) for n in counts))
        layers.append(tuple(rows))
    return tuple(layers)


def counterexample() -> dict[str, object]:
    first = (F(45, 100), F(35, 100), F(20, 100))
    second, third = (F(1, 10), F(7, 10), F(99, 100)), (F(1, 10), F(1, 2), F(99, 100))
    diag = lambda weights: tuple(tuple(weights[a] if a == b else F(0) for b in range(3)) for a in range(3))
    r: Lattice = (tuple(first for _ in range(3)), diag(second), diag(third))
    dp, c = optimize(r, 3)
    greedy = greedy_path(r)
    whole = max(product(range(3), repeat=3), key=lambda p: whole_path_survival(r, p))
    assert greedy == (0, 0, 0) and dp == (1, 1, 1) and whole == (2, 2, 2)
    assert check_certificate(r, 3, c)
    expected = [F(4995, 10000), F(7175, 10000), F(59402, 100000)]
    assert [prefix_value(r, (b, b, b)) for b in range(3)] == expected
    return {"label": "constructed mathematical example; not model measurements", "greedy_path": list(greedy), "prefix_optimal_path": list(dp), "whole_survival_optimal_path": list(whole), "prefix_values": {"A": str(expected[0]), "B": str(expected[1]), "C": str(expected[2])}}


def commit_greedy(prefix: tuple[int, ...], draft: Sequence[int], target: Callable[[tuple[int, ...]], int]) -> tuple[int, ...]:
    """Abstract greedy verifier, with a bonus token and no EOS/length truncation."""
    committed: list[int] = []
    for proposed in draft:
        gold = target(prefix + tuple(committed))
        if proposed != gold:
            return tuple(committed + [gold])
        committed.append(proposed)
    committed.append(target(prefix + tuple(committed)))
    return tuple(committed)


def run_tests() -> dict[str, object]:
    rng, started = random.Random(20260930), perf_counter()
    optimality_cases, enumerated_paths, certificate_rejections = 300, 0, 0
    for _ in range(optimality_cases):
        length, k = rng.randrange(0, 6), rng.randrange(1, 5)
        r = random_lattice(rng, length, k)
        path, cert = optimize(r, k)
        candidates = list(product(range(k), repeat=length))
        enumerated_paths += len(candidates)
        brute = max(prefix_value(r, p) for p in candidates)
        assert prefix_value(r, path) == brute == cert.values[0][0]
        assert check_certificate(r, k, cert)
        bad_values = list(cert.values)
        bad_values[0] = tuple(v + 1 for v in bad_values[0])
        assert not check_certificate(r, k, Certificate(tuple(bad_values), cert.choices))
        certificate_rejections += 1

    bound_cases, bounded_paths = 100, 0
    for _ in range(bound_cases):
        length, k = rng.randrange(1, 6), rng.randrange(1, 4)
        r, s = random_lattice(rng, length, k), random_lattice(rng, length, k)
        mixing = F(1, 20)
        rh: Lattice = tuple(tuple(tuple((1-mixing)*r[t][a][b] + mixing*s[t][a][b] for b in range(k)) for a in range(k)) for t in range(length))
        eps = [max(abs(r[t][a][b] - rh[t][a][b]) for a in range(k) for b in range(k)) for t in range(length)]
        delta = sum((length-t)*eps[t] for t in range(length))
        for p in product(range(k), repeat=length):
            assert abs(prefix_value(r, p) - prefix_value(rh, p)) <= delta
            bounded_paths += 1
        star, _ = optimize(r, k)
        chosen, _ = optimize(rh, k)
        assert prefix_value(r, star) - prefix_value(r, chosen) <= 2*delta

    verifier_cases = 2000
    forced_accept_lengths: set[int] = set()
    for i in range(verifier_cases):
        prefix = tuple(rng.randrange(11) for _ in range(rng.randrange(1, 12)))
        # Arbitrary deterministic causal toy target, not a language model.
        def target(history: tuple[int, ...]) -> int:
            return (3 + len(history) + sum((j+1)*v for j, v in enumerate(history))) % 11
        gold: list[int] = []
        for _ in range(8):
            gold.append(target(prefix + tuple(gold)))
        accepted = i % 8
        draft = gold[:7]
        if accepted < 7:
            draft[accepted] = (draft[accepted] + 1) % 11
        out = commit_greedy(prefix, draft, target)
        assert out == tuple(gold[:accepted+1])
        forced_accept_lengths.add(accepted)
    assert forced_accept_lengths == set(range(8))

    return {
        "status": "PASS",
        "scope": "Synthetic exact-rational CPU checks only; no GPU/model/Lean verification",
        "seed": 20260930,
        "python": sys.version.split()[0],
        "optimality_lattices": optimality_cases,
        "exhaustively_enumerated_paths": enumerated_paths,
        "valid_certificates_checked": optimality_cases,
        "corrupted_certificates_rejected": certificate_rejections,
        "perturbation_bound_lattices": bound_cases,
        "perturbation_bound_paths": bounded_paths,
        "abstract_greedy_verifier_cases": verifier_cases,
        "forced_accepted_draft_lengths": sorted(forced_accept_lengths),
        "counterexample": counterexample(),
        "seconds": round(perf_counter()-started, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    results = run_tests()
    serialized = json.dumps(results, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
