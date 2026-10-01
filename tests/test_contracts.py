"""Exact witnesses that separate the exactness contracts (P7 in TASKS.md).

Each test checks one small counterexample showing that two of the paper's contracts differ:
identical internal state, identical greedy outputs, identical seeded samples, the exact
sampling distribution, and empirical quality equivalence. Exact cases use Fractions; floating-point cases use NumPy scalar and
elementwise float32 arithmetic (IEEE binary32, round to nearest, ties to even, no FMA
contraction), with every reduction written as an explicit sequential loop so that the
evaluation order is the one stated. The query-invisible state witness (equal readout now,
different later) is in tests/test_state_structure.py.

Run as a script to write evidence/contracts/contract_witnesses.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from collections.abc import Sequence
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WITNESSES: dict[str, object] = {}
F32 = np.float32


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest BF16 value (ties to even), returned as float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def first_argmax(values: Sequence[Any] | np.ndarray) -> int:
    """torch.argmax's rule: the first index among equal maxima."""
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best


def matvec_rows(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """m @ x in float32, accumulating over columns sequentially (fixed order)."""
    acc = np.zeros(m.shape[0], dtype=np.float32)
    for j in range(m.shape[1]):
        acc = (acc + m[:, j] * x[j]).astype(np.float32)
    return acc


def sequential_norm(x: np.ndarray) -> np.float32:
    """Euclidean norm with the squares summed left to right in float64, rounded to float32."""
    total = 0.0
    for value in x.astype(np.float64):
        total += float(value) * float(value)
    return F32(np.sqrt(total))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b in float32 with a sequential reduction over the inner dimension."""
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for t in range(a.shape[1]):
        acc = (acc + np.outer(a[:, t], b[t, :]).astype(np.float32)).astype(np.float32)
    return acc


class Contracts(unittest.TestCase):
    def test_reassociation_flips_a_greedy_winner(self) -> None:
        # Row 0 is one term; row 1 sums three terms. Sequential FP32 accumulation of row 1
        # loses both small terms and ties row 0, so the first-index rule picks row 0; the
        # regrouped sum keeps them and row 1 wins, as it does in real arithmetic.
        tiny = F32(2.0**-24)
        row0 = F32(1.0)
        seq = (F32(1.0) + tiny) + tiny
        regrouped = F32(1.0) + (tiny + tiny)
        real = [Q(1), Q(1) + 2 * Q(2) ** -24]
        self.assertEqual(first_argmax([row0, seq]), 0)
        self.assertEqual(first_argmax([row0, regrouped]), 1)
        self.assertEqual(first_argmax(real), 1)
        WITNESSES['reassociation_flips_greedy'] = {
            'sequential_logits': [float(row0), float(seq)],
            'regrouped_logits': [float(row0), float(regrouped)],
            'sequential_token': 0,
            'regrouped_token': 1,
        }

    def test_distinct_fp32_values_share_a_bf16_cell(self) -> None:
        # 1 and 1 + 2^-10 differ in FP32 but both round to 1 in BF16 (spacing 2^-7 there),
        # so a BF16-output head ties them and the index rule decides.
        a, b = F32(1.0), F32(1.0 + 2.0**-10)
        self.assertNotEqual(a, b)
        self.assertEqual(bf16(np.array([a]))[0], bf16(np.array([b]))[0])
        self.assertEqual(first_argmax([a, b]), 1)
        self.assertEqual(first_argmax(bf16(np.array([a, b]))), 0)
        WITNESSES['bf16_cell'] = {'fp32': [float(a), float(b)], 'bf16': 1.0}

    def test_same_argmax_different_law(self) -> None:
        # Logits (1, 0) and (2, 0) in the base-2 mass model: same greedy token, different law.
        law1 = [Q(2, 3), Q(1, 3)]
        law2 = [Q(4, 5), Q(1, 5)]
        self.assertEqual(first_argmax(law1), first_argmax(law2))
        self.assertNotEqual(law1, law2)
        WITNESSES['same_argmax_different_law'] = {
            'law_1_0': [str(x) for x in law1],
            'law_2_0': [str(x) for x in law2],
        }

    def test_same_law_different_seeded_tokens(self) -> None:
        # Two exact inverse-CDF samplers for p = (1/2, 1/2) that list the tokens in different
        # orders have the same law, but the same uniform draw gives different tokens.
        p = [Q(1, 2), Q(1, 2)]
        u = Q(1, 4)

        def inverse_cdf(order: list[int]) -> int:
            cumulative = Q(0)
            for token in order:
                cumulative += p[token]
                if u < cumulative:
                    return token
            return order[-1]

        # Law check: the token is order[0] exactly when u < p[order[0]] = 1/2.
        self.assertEqual(inverse_cdf([0, 1]), 0)
        self.assertEqual(inverse_cdf([1, 0]), 1)
        WITNESSES['same_law_different_seeded_tokens'] = {
            'uniform': str(u),
            'token_order_01': 0,
            'token_order_10': 1,
        }

    def test_regrouped_gdn_recurrence_changes_outputs(self) -> None:
        # The gated-delta recurrence S_t = S_{t-1} A_t + B_t, with A_t = alpha_t (I - beta_t k k^T),
        # evaluated sequentially and in the split form S_t = S_0 P_t + W_t (P_t the product of
        # the transitions, W_t the recurrence started from zero). The two are equal in exact
        # arithmetic; in FP32 many outputs o_t = S_t q_t differ, far fewer after BF16 rounding.
        rng = np.random.default_rng(7)
        steps, d = 50, 32
        s0 = rng.standard_normal((d, d)).astype(F32)
        seq_state = s0.copy()
        prod = np.eye(d, dtype=F32)
        zero_start = np.zeros((d, d), dtype=F32)
        seq_out, split_out = [], []
        for _ in range(steps):
            k = rng.standard_normal(d).astype(F32)
            k = (k / sequential_norm(k)).astype(F32)
            v = rng.standard_normal(d).astype(F32)
            q = rng.standard_normal(d).astype(F32)
            alpha = F32(rng.uniform(0.9, 1.0))
            beta = F32(rng.uniform(0.1, 0.9))
            a = (alpha * (np.eye(d, dtype=F32) - beta * np.outer(k, k).astype(F32))).astype(F32)
            b = (beta * np.outer(v, k)).astype(F32)
            seq_state = (matmul(seq_state, a) + b).astype(F32)
            prod = matmul(prod, a)
            zero_start = (matmul(zero_start, a) + b).astype(F32)
            split_state = (matmul(s0, prod) + zero_start).astype(F32)
            seq_out.append(matvec_rows(seq_state, q))
            split_out.append(matvec_rows(split_state, q))
        seq_all = np.concatenate(seq_out)
        split_all = np.concatenate(split_out)
        differ_fp32 = int(np.sum(seq_all != split_all))
        differ_bf16 = int(np.sum(bf16(seq_all) != bf16(split_all)))
        self.assertGreater(differ_fp32, 0)
        self.assertLess(differ_bf16, differ_fp32)
        WITNESSES['regrouped_gdn_outputs'] = {
            'outputs': int(seq_all.size),
            'differ_fp32': differ_fp32,
            'differ_after_bf16': differ_bf16,
            'steps': steps,
            'dim': d,
            'seed': 7,
        }

    def test_equal_quality_different_outputs(self) -> None:
        # Empirical quality equivalence compares a score on a fixed set, not the outputs.
        # Two greedy decoders answer four problems; each is right on two, so their accuracies
        # are equal (difference 0, within any margin), yet they give different answers on
        # every problem. Equal task scores therefore imply none of the stronger contracts.
        reference = ['a', 'b', 'c', 'd']
        decoder_x = ['a', 'b', 'x', 'y']
        decoder_y = ['z', 'w', 'c', 'd']

        def accuracy(answers: list[str]) -> Q:
            correct = sum(1 for got, want in zip(answers, reference, strict=True) if got == want)
            return Q(correct, len(reference))

        only_x = sum(
            1 for x, y, r in zip(decoder_x, decoder_y, reference, strict=True) if x == r != y
        )
        only_y = sum(
            1 for x, y, r in zip(decoder_x, decoder_y, reference, strict=True) if y == r != x
        )
        differing = sum(1 for x, y in zip(decoder_x, decoder_y, strict=True) if x != y)
        self.assertEqual(accuracy(decoder_x), accuracy(decoder_y))
        self.assertEqual(differing, len(reference))
        self.assertEqual((only_x, only_y), (2, 2))
        WITNESSES['equal_quality_different_outputs'] = {
            'accuracy_x': str(accuracy(decoder_x)),
            'accuracy_y': str(accuracy(decoder_y)),
            'problems_with_different_answers': differing,
            'discordant_pairs_x_only_y_only': [only_x, only_y],
        }

    def test_transition_norm_and_error_growth(self) -> None:
        # (I - beta k k^T) k = (1 - beta ||k||^2) k and vectors orthogonal to k are fixed, so
        # the transition's spectral norm is max(1, |1 - beta ||k||^2|): at most 1 exactly when
        # 0 <= beta ||k||^2 <= 2, which gives ||E_t|| <= |alpha_t| ||E_{t-1}|| + ||eta_t||.
        k = [Q(3, 5), Q(4, 5)]
        orth = [Q(-4, 5), Q(3, 5)]
        results = {}
        for beta in (Q(1, 2), Q(2), Q(5, 2)):

            def apply(x: list[Q], beta: Q = beta) -> list[Q]:
                dot = sum((a * b for a, b in zip(k, x, strict=True)), Q(0))
                return [xi - beta * ki * dot for xi, ki in zip(x, k, strict=True)]

            self.assertEqual(apply(k), [(1 - beta) * ki for ki in k])
            self.assertEqual(apply(orth), orth)
            results[str(beta)] = str(max(Q(1), abs(1 - beta)))
        self.assertEqual(results['1/2'], '1')
        self.assertEqual(results['2'], '1')
        self.assertEqual(results['5/2'], '3/2')
        WITNESSES['transition_norm_by_beta'] = results


if __name__ == '__main__':
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
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
        'scope': 'exact and float32 witnesses separating the contracts; not model measurements',
    }
    out = ROOT / 'evidence' / 'contracts' / 'contract_witnesses.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(not result.wasSuccessful())
