"""Exact witnesses about recurrent-state structure and cross-request sharing (P4, P5).

P4 (minimal recurrent state) and P5 (sharing across requests) are the proposals listed in
TASKS.md.

Each test checks one small counterexample from the P4 and P5 analyses in exact rational
arithmetic (Fractions), or with numpy float32 where the point is floating-point rounding.
They are witnesses to specific claims, not measurements of the model: a witness shows that a
shortcut is not valid in general, not how often it fails on real states.

Run as a script to write evidence/state_structure/state_structure_tests.json.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import unittest
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WITNESSES: dict[str, object] = {}

Matrix = list[list[Q]]


def eye(d: int) -> Matrix:
    return [[Q(int(i == j)) for j in range(d)] for i in range(d)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [
        [sum((a[i][t] * b[t][j] for t in range(len(b))), Q(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def scale(c: Q, a: Matrix) -> Matrix:
    return [[c * x for x in row] for row in a]


def outer(u: list[Q], v: list[Q]) -> Matrix:
    return [[x * y for y in v] for x in u]


def transition(alpha: Q, beta: Q, k: list[Q]) -> Matrix:
    """A = alpha (I - beta k k^T), the key-space factor of one gated-delta step."""
    d = len(k)
    return scale(alpha, add(eye(d), scale(-beta, outer(k, k))))


def gdn_step(s: Matrix, alpha: Q, beta: Q, k: list[Q], v: list[Q]) -> Matrix:
    """S' = S A + beta v k^T with S of shape (d_v, d_k), the Gated DeltaNet update."""
    return add(matmul(s, transition(alpha, beta, k)), scale(beta, outer(v, k)))


def rank(rows: Matrix) -> int:
    """Exact rank by Gaussian elimination over the rationals."""
    a = [list(r) for r in rows]
    r = 0
    for c in range(len(a[0]) if a else 0):
        pivot = next((i for i in range(r, len(a)) if a[i][c] != 0), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        for i in range(len(a)):
            if i != r and a[i][c] != 0:
                f = a[i][c] / a[r][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[r], strict=True)]
        r += 1
    return r


def det(a: Matrix) -> Q:
    a = [list(r) for r in a]
    n, sign, prod = len(a), 1, Q(1)
    for c in range(n):
        pivot = next((i for i in range(c, n) if a[i][c] != 0), None)
        if pivot is None:
            return Q(0)
        if pivot != c:
            a[c], a[pivot] = a[pivot], a[c]
            sign = -sign
        prod *= a[c][c]
        for i in range(c + 1, n):
            f = a[i][c] / a[c][c]
            a[i] = [x - f * y for x, y in zip(a[i], a[c], strict=True)]
    return sign * prod


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest BF16 value (ties to even), returned as float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


class StateStructure(unittest.TestCase):
    def test_query_invisible_is_not_unobservable(self) -> None:
        # A state difference orthogonal to today's query becomes visible after one step.
        alpha, beta, k = Q(1, 2), Q(1, 2), [Q(3, 5), Q(4, 5)]
        q, ds = [Q(1), Q(0)], [Q(0), Q(1)]
        a = transition(alpha, beta, k)
        now = sum((x * y for x, y in zip(q, ds, strict=True)), Q(0))
        later = sum((q[i] * a[i][j] * ds[j] for i in range(2) for j in range(2)), Q(0))
        self.assertEqual(now, 0)
        self.assertEqual(later, Q(-3, 25))
        WITNESSES['query_invisible'] = {'q_dot_ds': str(now), 'q_A_ds': str(later)}

    def test_coordinate_key_writes_reach_any_matrix(self) -> None:
        # With unit coordinate keys and gates inside (0, 1), d writes set any target matrix.
        alpha, beta = Q(1, 2), Q(1, 2)
        target = [[Q(3), Q(-1), Q(2, 7)], [Q(0), Q(5, 3), Q(-4)], [Q(1, 9), Q(2), Q(6)]]
        s = [[Q(1), Q(2), Q(3)], [Q(4), Q(5), Q(6)], [Q(7), Q(8), Q(10)]]
        d = 3
        for j in range(d):
            key = [Q(int(t == j)) for t in range(d)]
            # Column j is scaled by alpha at each of the d - 1 - j later steps.
            want = [target[i][j] / alpha ** (d - 1 - j) for i in range(d)]
            value = [(want[i] - alpha * (1 - beta) * s[i][j]) / beta for i in range(d)]
            s = gdn_step(s, alpha, beta, key, value)
        self.assertEqual(s, target)
        WITNESSES['coordinate_key_reachability'] = {'dimension': d, 'reached_target': True}

    def test_transition_is_contraction_not_erasure(self) -> None:
        # det(alpha (I - beta k k^T)) = alpha^d (1 - beta ||k||^2) > 0 when beta ||k||^2 < 1.
        alpha, beta, k = Q(9, 10), Q(2, 3), [Q(2, 7), Q(3, 7), Q(6, 7)]
        norm2 = sum((x * x for x in k), Q(0))
        value = det(transition(alpha, beta, k))
        self.assertEqual(norm2, 1)
        self.assertEqual(value, alpha**3 * (1 - beta * norm2))
        self.assertGreater(value, 0)
        WITNESSES['transition_determinant'] = {'det': str(value)}

    def test_nonorthogonal_keys_do_not_commute(self) -> None:
        a1 = transition(Q(1, 2), Q(1, 2), [Q(1), Q(0)])
        a2 = transition(Q(1, 2), Q(1, 2), [Q(3, 5), Q(4, 5)])
        self.assertNotEqual(matmul(a1, a2), matmul(a2, a1))
        WITNESSES['noncommuting_transitions'] = {'commute': False}

    def test_silu_raises_rank(self) -> None:
        # G = log 2 [[1, 2], [2, 4]] has rank one; SiLU(c log 2) = c log 2 / (1 + 2^-c).
        g = [[Q(1), Q(2)], [Q(2), Q(4)]]
        silu_over_log2 = [[c / (1 + Q(1, 2) ** int(c)) for c in row] for row in g]
        self.assertEqual(rank(g), 1)
        self.assertEqual(silu_over_log2, [[Q(2, 3), Q(8, 5)], [Q(8, 5), Q(64, 17)]])
        self.assertEqual(det(silu_over_log2), Q(-64, 1275))
        self.assertEqual(rank(silu_over_log2), 2)
        # The float64 SiLU agrees with the exact form to rounding.
        x = math.log(2) * np.array([[1.0, 2.0], [2.0, 4.0]])
        silu = x / (1.0 + np.exp(-x))
        exact = math.log(2) * np.array([[float(c) for c in row] for row in silu_over_log2])
        self.assertTrue(np.allclose(silu, exact, rtol=1e-15, atol=0.0))
        WITNESSES['silu_rank'] = {
            'rank_before': 1,
            'rank_after': 2,
            'det_over_log2_squared': str(det(silu_over_log2)),
        }

    def test_one_gdn_step_separates_identical_states(self) -> None:
        # Four requests share one state; after one step their states span all 2x2 matrices.
        s = [[Q(1), Q(2)], [Q(0), Q(1)]]
        requests = [
            (Q(1, 2), Q(1, 2), [Q(1), Q(0)], [Q(0), Q(1)]),
            (Q(3, 4), Q(1, 3), [Q(0), Q(1)], [Q(1), Q(0)]),
            (Q(1, 2), Q(2, 3), [Q(3, 5), Q(4, 5)], [Q(1), Q(1)]),
            (Q(9, 10), Q(1, 2), [Q(4, 5), Q(-3, 5)], [Q(1), Q(-1)]),
        ]
        stacked = [[x for row in gdn_step(s, a, b, k, v) for x in row] for a, b, k, v in requests]
        self.assertEqual(rank([[x for row in s for x in row]] * 4), 1)
        self.assertEqual(rank(stacked), 4)
        WITNESSES['batch_rank_after_one_step'] = {'before': 1, 'after': 4, 'requests': 4}

    def test_rmsnorm_hides_no_null_space(self) -> None:
        # o1 and o2 differ only in the null space of W_o = [1, 0], but RMSNorm precedes W_o.
        # RMSNorm here is x / sqrt(mean(x^2)) with no epsilon and no weight.
        def rmsnorm(o: list[Q]) -> list[Q]:
            ms = sum((x * x for x in o), Q(0)) / len(o)
            root = Q(math.isqrt(ms.numerator), math.isqrt(ms.denominator))
            self.assertEqual(root * root, ms)  # rational root for these inputs
            return [x / root for x in o]

        w_o = [Q(1), Q(0)]
        o1, o2 = [Q(1), Q(1)], [Q(1), Q(7)]
        raw = [sum((w * x for w, x in zip(w_o, o, strict=True)), Q(0)) for o in (o1, o2)]
        normed = [
            sum((w * x for w, x in zip(w_o, rmsnorm(o), strict=True)), Q(0)) for o in (o1, o2)
        ]
        self.assertEqual(raw[0], raw[1])
        self.assertNotEqual(normed[0], normed[1])
        WITNESSES['rmsnorm_before_projection'] = {
            'projection_without_norm': [str(x) for x in raw],
            'projection_after_norm': [str(x) for x in normed],
        }

    def test_fp32_lazy_decay_is_one_ulp_off(self) -> None:
        # Deferring a decay reassociates two FP32 roundings: fl(fl(a1 x) a2) != fl(fl(a1 a2) x).
        # Arithmetic is IEEE binary32 with round to nearest, ties to even (NumPy scalar
        # multiplies, no FMA contraction); a kernel that contracts or reorders may differ.
        a1, a2, x = np.float32(0.9), np.float32(0.9), np.float32(1.3)
        eager = (a1 * x) * a2
        lazy = (a1 * a2) * x
        ulps = abs(float(eager) - float(lazy)) / float(np.spacing(max(eager, lazy)))
        self.assertNotEqual(eager, lazy)
        self.assertEqual(ulps, 1.0)
        real = Q(float(a1)) * Q(float(a2)) * Q(float(x))
        WITNESSES['fp32_lazy_decay'] = {
            'a1': float(a1),
            'a2': float(a2),
            'x': float(x),
            'eager': float(eager),
            'lazy': float(lazy),
            'ulps_apart': ulps,
            'eager_error': float(abs(Q(float(eager)) - real)),
            'lazy_error': float(abs(Q(float(lazy)) - real)),
        }

    def test_bf16_rounding_breaks_rank_one(self) -> None:
        # BF16 operands give an exact rank-one FP32 outer product; rounding it to BF16 does not.
        rng = np.random.default_rng(0)
        u = bf16(rng.uniform(-1, 1, 32).astype(np.float32))
        v = bf16(rng.uniform(-1, 1, 32).astype(np.float32))
        product = np.outer(u, v).astype(np.float32)
        exact = [[Q(float(a)) * Q(float(b)) for b in v] for a in u]
        self.assertEqual([[Q(float(x)) for x in row] for row in product], exact)
        rounded = bf16(product)
        r_rounded = rank([[Q(float(x)) for x in row] for row in rounded])
        self.assertEqual(rank(exact), 1)
        self.assertEqual(r_rounded, 28)
        WITNESSES['bf16_rounded_outer_product'] = {'size': 32, 'seed': 0, 'rank': r_rounded}


if __name__ == '__main__':
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StateStructure)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'status': 'pass' if result.wasSuccessful() else 'fail',
        'methods': result.testsRun,
        'witnesses': WITNESSES,
        'seconds': round(time.perf_counter() - start, 3),
        'repo_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        'numpy': np.__version__,
        'python': sys.version.split()[0],
        'scope': 'exact rational witnesses and float32 rounding cases; not model measurements',
    }
    out = ROOT / 'evidence' / 'state_structure' / 'state_structure_tests.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(not result.wasSuccessful())
