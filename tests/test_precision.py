"""Exact tests for certified low-precision head decisions; CPU only.

Run from the repository root: `python tests/test_precision.py` writes
evidence/precision/precision_tests.json; pytest also collects this file.
Every floating-point rounding is emulated exactly on Fractions, so each check
compares a bound or a decision with the real-arithmetic value.
"""

from __future__ import annotations

import itertools
import json
import platform
import random
import struct
import subprocess
import sys
import time
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

from precision_reference import *

SEED = 20260930
R = random.Random(SEED)
COUNTS: dict[str, int] = {}
STATS: dict[str, object] = {}
WITNESSES: dict[str, object] = {}


def count(key: str, n: int = 1) -> None:
    COUNTS[key] = COUNTS.get(key, 0) + n


def bf16_value(emin: int, emax: int) -> Q:
    return Q(R.choice((-1, 1)) * (128 + R.randrange(128)), 128) * pow2(R.randint(emin, emax))


def bf16_vector(n: int, emin: int, emax: int) -> list[Q]:
    return [bf16_value(emin, emax) for _ in range(n)]


def random_head(V: int, D: int) -> list[list[Q]]:
    return [bf16_vector(D, -7, -3) for _ in range(V)]


def near_tie_head(V: int, D: int) -> list[list[Q]]:
    """Rows one BF16 ulp apart in one coordinate, plus exact duplicates."""
    base = bf16_vector(D, -7, -3)
    rows = []
    for _ in range(V):
        row = list(base)
        j = R.randrange(D)
        row[j] = round_to(row[j] + R.choice((-1, 0, 1)) * BF16.quantum(row[j]), BF16)
        rows.append(row)
    R.shuffle(rows)
    return rows


def with_outliers(h: list[Q], k: int) -> tuple[list[Q], list[int]]:
    """Massive activations in k columns, as in real residual streams."""
    cols = R.sample(range(len(h)), k)
    for j in cols:
        h[j] = bf16_value(5, 7)
    return h, cols


ACCUMULATORS = (
    Accumulator('sequential', FP32),
    Accumulator('pairwise', FP32),
    Accumulator('blocked', FP32, block=4),
    Accumulator('blocked', FP32, block=4, splits=3),
    Accumulator('sequential', FP32, rounding='toward_zero'),
    Accumulator('fused', FP32, rounding='toward_zero', block=4),
    Accumulator('fused', FP32, rounding='toward_zero', block=8, ftz=True),
    Accumulator('sequential', FP32, ftz=True),
)


def int_head(V: int, D: int, span: int = 2) -> tuple[list[list[Q]], list[Q]]:
    """Small integer logits keep base-2 softmax laws far from one-hot."""
    W = [[Q(R.randint(-span, span)) for _ in range(D)] for _ in range(V)]
    h = [Q(R.randint(-1, 1)) for _ in range(D)]
    if not any(h):
        h[0] = Q(1)
    return W, h


def mass_evaluator(W: list[list[Q]], h: list[Q], bits: int) -> MassEvaluator:
    rule = EnvelopeRule(accumulator=R.choice((None, FP32_SEQUENTIAL)))
    return MassEvaluator(Evaluator(quantize_head(W, bits=bits), tuple(h), rule))


def law(n: int, support: list[int]) -> list[Q]:
    a = [R.randint(1, 9) for _ in support]
    q = [Q(0)] * n
    for i, v in zip(support, a, strict=True):
        q[i] = Q(v, sum(a))
    return q


class Precision(unittest.TestCase):
    def test_rounding_emulation(self) -> None:
        for _ in range(4000):
            x = R.uniform(-1e3, 1e3) * 2.0 ** R.randint(-150, 100)
            self.assertEqual(round_to(Q(x), FP32), Q(float(np.float32(x))))
            bits = struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]
            rne = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
            bf = struct.unpack('<f', struct.pack('<I', rne & 0xFFFFFFFF))[0]
            self.assertEqual(round_to(Q(float(np.float32(x))), BF16), Q(bf))
            down, up = round_to(Q(x), FP32, 'down'), round_to(Q(x), FP32, 'up')
            self.assertTrue(down <= Q(x) <= up)
            self.assertTrue(up == down or next_up(down, FP32) == up)
            count('rounding_cases')
        for _ in range(1500):
            x = R.uniform(0, 1e3) * 2.0 ** R.randint(-300, 300)
            self.assertEqual(sqrt_round(Q(x), FP64), Q(float(np.sqrt(x))))
            x32 = np.float32(R.uniform(0, 1e3) * 2.0 ** R.randint(-140, 110))
            self.assertEqual(sqrt_round(Q(float(x32)), FP32), Q(float(np.sqrt(x32))))
            count('sqrt_cases')

    def test_accumulation_bound(self) -> None:
        worst: dict[str, float] = {}
        for case in range(700):
            n = R.randint(1, 40)
            if case % 3 == 0:  # cancellation around a large term
                big = bf16_value(8, 12)
                terms = [big, *(bf16_value(-12, 2) for _ in range(n)), -big]
            else:
                terms = [bf16_value(-8, 0) * bf16_value(-20, 6) for _ in range(n)]
            exact = sum(terms, Q())
            abs_sum = sum((abs(t) for t in terms), Q())
            for acc in ACCUMULATORS:
                err = abs(acc.sum(terms) - exact)
                bound = acc.error_bound(abs_sum, len(terms))
                self.assertLessEqual(err, bound, (acc, terms))
                if bound:
                    key = f'{acc.order}-{acc.rounding}-b{acc.block}-s{acc.splits}-ftz{acc.ftz}'
                    worst[key] = max(worst.get(key, 0.0), float(err / bound))
                count('accumulation_bound_checks')
        STATS['accumulation_max_error_over_bound'] = worst

    def test_envelope_encloses(self) -> None:
        for case in range(260):
            V, D = R.randint(2, 12), R.randint(2, 24)
            W = random_head(V, D)
            h = bf16_vector(D, -3, 3)
            exact_cols: list[int] = []
            if case % 3 == 0:
                h, exact_cols = with_outliers(h, R.randint(1, max(1, D // 4)))
            head = quantize_head(
                W,
                bits=R.choice((3, 4, 8)),
                group_size=R.choice((None, 4, 8)),
                exact_columns=exact_cols,
            )
            acc = R.choice((None, *ACCUMULATORS))
            rule = EnvelopeRule(accumulator=acc)
            ev = Evaluator(head, tuple(h), rule, ladder=(None,))
            for i in range(V):
                z = dot(W[i], h)
                lo, hi = ev.interval(0, i)
                self.assertTrue(lo <= z <= hi, (case, i))
                inner = abs(dot(head.error(i), h))
                self.assertLessEqual(inner, head.error_l2[i] * ev.approx_norms.l2)
                self.assertLessEqual(inner, head.error_linf[i] * ev.approx_norms.l1)
                block = sum(
                    (
                        a * b
                        for a, b in zip(
                            head.error_block_l2[i], ev.approx_norms.block_l2, strict=True
                        )
                    ),
                    Q(),
                )
                self.assertLessEqual(inner, block)
                count('envelope_rows')
            count('envelope_heads')

    def test_envelope_terms_are_needed(self) -> None:
        # (a) Zero quantization error, yet FP32 accumulation is inexact: the
        # accumulation term is what keeps z inside the interval.
        found = 0
        for _ in range(200):
            D = R.randint(3, 10)
            W = [[Q(R.randint(-127, 127), 128) for _ in range(D)] for _ in range(3)]
            for w in W:
                w[0] = Q(127, 128)
            h = [bf16_value(-30, 8) for _ in range(D)]
            head = quantize_head(W, bits=8)
            if any(head.error(i) != tuple(Q(0) for _ in range(D)) for i in range(3)):
                continue
            ev = Evaluator(
                head, tuple(h), EnvelopeRule(accumulator=FP32_SEQUENTIAL), ladder=(None,)
            )
            for i in range(3):
                zt = approx_logit(head, i, h, FP32_SEQUENTIAL)
                z = dot(W[i], h)
                lo, hi = ev.interval(0, i)
                self.assertTrue(lo <= z <= hi)
                if zt != z:
                    found += 1
                    WITNESSES.setdefault('accumulation_term_needed', {'z': str(z), 'zt': str(zt)})
        self.assertGreater(found, 0)
        count('accumulation_term_witnesses', found)
        # (b) Without outward rounding, fl(zt - beta) can exceed the true lower end.
        zt, beta = Q(1), pow2(-26)
        naive = round_to(zt - beta, FP32)
        lo, _ = outward_interval(zt, beta)
        self.assertGreater(naive, zt - beta)
        self.assertLessEqual(lo, zt - beta)
        WITNESSES['outward_rounding_needed'] = {'zt': '1', 'beta': '2^-26', 'naive_lo': str(naive)}
        # (c) An uninflated FP32 norm can undercut ||h||, and an error vector
        # parallel to h then escapes an envelope built from it.
        escapes = 0
        for _ in range(300):
            h = bf16_vector(R.randint(2, 30), -3, 3)
            naive_norm = sqrt_round(FP32_SEQUENTIAL.sum([h_j * h_j for h_j in h]), FP32)
            e = [x * Q(1, 1024) for x in h]
            inner = dot(e, h)
            e2 = round_to(sqrt_up(sum((x * x for x in e), Q())), FP32, 'up')
            if e2 * naive_norm < inner:
                escapes += 1
            self.assertGreaterEqual(e2 * token_norms(h, (), FP32_SEQUENTIAL).l2, inner)
            count('norm_inflation_checks')
        self.assertGreater(escapes, 0)
        count('norm_inflation_witnesses', escapes)

    def test_bounds_are_attained(self) -> None:
        """Blockwise Cauchy-Schwarz and the transport Hoelder bound are the
        suprema of their information classes (so tighter needs more data)."""
        for _ in range(300):
            D = R.randint(2, 16)
            h = [Q(R.randint(-9, 9), R.randint(1, 4)) for _ in range(D)]
            cut = sorted(R.sample(range(1, D), R.randint(0, D - 1))) if D > 1 else []
            blocks = [list(range(a, b)) for a, b in zip([0, *cut], [*cut, D], strict=True)]
            coef = [Q(R.randint(1, 9), R.randint(1, 9)) for _ in blocks]
            e = [Q(0)] * D
            for c, b in zip(coef, blocks, strict=True):
                for j in b:
                    e[j] = c * h[j]
            attained = sum(
                (c * sum((h[j] ** 2 for j in b), Q()) for c, b in zip(coef, blocks, strict=True)),
                Q(),
            )
            self.assertEqual(dot(e, h), attained)  # = sum_g ||e_g||_2 ||h_g||_2
            r = Q(R.randint(1, 9), 8)
            mu = [Q(R.randint(-9, 9), 4) for _ in range(D)]
            w = [m + r * (1 if x > 0 else -1 if x < 0 else 0) for m, x in zip(mu, h, strict=True)]
            self.assertEqual(
                dot([a - b for a, b in zip(w, mu, strict=True)], h), transport_halfwidth(r, h)
            )
            count('supremum_attainments')

    def test_certified_argmax(self) -> None:
        levels: dict[int, int] = {}
        ties = 0
        for case in range(420):
            V, D = R.randint(2, 40), R.randint(2, 16)
            W = near_tie_head(V, D) if case % 2 else random_head(V, D)
            h = bf16_vector(D, -3, 3)
            exact_cols: list[int] = []
            if case % 5 == 0:
                h, exact_cols = with_outliers(h, 1)
            head = quantize_head(
                W, bits=R.choice((4, 8)), group_size=R.choice((None, 4)), exact_columns=exact_cols
            )
            rule = EnvelopeRule(
                accumulator=R.choice((None, FP32_SEQUENTIAL, FP32_BLOCKED, ACCUMULATORS[5]))
            )
            ladder = R.choice(
                ((FP32_SEQUENTIAL, FP64_SEQUENTIAL, None), (FP32_BLOCKED, None), (None,))
            )
            ev = Evaluator(head, tuple(h), rule, ladder)
            z = [dot(w, h) for w in W]
            bias = None
            masked: set[int] = set()
            if case % 7 == 0:
                bias = [Q(R.randint(-2, 2), 64) for _ in range(V)]
                masked = set(R.sample(range(V), R.randint(0, V - 1)))
            cert = certify_argmax(ev, bias, masked)
            assert cert is not None
            scores = [z[i] + (bias[i] if bias else 0) for i in range(V)]
            rows = [i for i in range(V) if i not in masked]
            self.assertEqual(cert.token, exact_argmax(scores, rows))
            self.assertEqual(cert.rows[0], len(rows))
            self.assertLessEqual(cert.candidates, len(rows))
            levels[cert.level] = levels.get(cert.level, 0) + 1
            ties += cert.exact_tie
            count('certified_argmax_cases')
        STATS['argmax_certified_level_histogram'] = {str(k): v for k, v in sorted(levels.items())}
        count('certified_argmax_exact_ties', ties)
        self.assertGreater(ties, 0)

    def test_subnormal_products(self) -> None:
        """Products of subnormal BF16 operands can underflow FP32.

        The ladder must round them as the modelled accumulator does (gradual
        underflow or flush to zero), keep every interval an enclosure of the
        exact logit, and still resolve an exact tie between identical rows to
        the smallest index (review comment on PR #7).
        """
        t = pow2(-133)  # the smallest positive BF16 subnormal
        ties = 0
        for ftz in (False, True):
            ladder = (Accumulator('sequential', FP32, ftz=ftz), FP64_SEQUENTIAL, None)
            for _ in range(80):
                D = R.randint(2, 8)

                def sub_row(d: int) -> list[Q]:
                    return [Q(R.randint(-127, 127)) * t for _ in range(d)]

                row = sub_row(D)
                W = [list(row), list(row), sub_row(D)]
                if R.random() < 0.5:  # a normal-range weight next to subnormal ones
                    W[2][0] = bf16_value(-7, -3)
                h = sub_row(D)
                ev = Evaluator(quantize_head(W), tuple(h), ladder=ladder)
                for i in range(len(W)):
                    lo, hi = ev.interval(1, i)
                    self.assertTrue(lo <= dot(W[i], h) <= hi)
                cert = certify_argmax(ev)
                self.assertIsNotNone(cert)
                assert cert is not None
                self.assertEqual(cert.token, exact_argmax([dot(w, h) for w in W]))
                ties += int(cert.exact_tie)
                count('subnormal_product_cases')
        self.assertGreater(ties, 0)
        count('subnormal_exact_ties', ties)
        t2 = Q(1, 2**133)
        with self.assertRaises(ValueError):  # an inexact normal-range leaf is still an input error
            FP32_SEQUENTIAL.sum([Q(1) + t2])

    def test_certificate_refuses(self) -> None:
        refused = 0
        for _ in range(300):
            V, D = R.randint(2, 24), R.randint(2, 12)
            W = near_tie_head(V, D) if R.random() < 0.5 else random_head(V, D)
            h = bf16_vector(D, -3, 3)
            ev = Evaluator(quantize_head(W, bits=R.choice((4, 8))), tuple(h))
            cert = certify_argmax(ev, max_level=0)
            if cert is None:
                refused += 1
            else:
                self.assertEqual(cert.token, exact_argmax([dot(w, h) for w in W]))
            count('approx_only_attempts')
        self.assertGreater(refused, 0)
        self.assertLess(refused, 300)
        count('approx_only_refusals', refused)
        W = random_head(4, 4)
        h = bf16_vector(4, -3, 3)
        head = quantize_head(W)
        with self.assertRaises(ValueError):
            Evaluator(head, tuple(h), namespace='other-adapter')
        with self.assertRaises(ValueError):
            Evaluator(head, (*h[:3], h[3] + pow2(-40)))  # not the BF16 vector the head consumes
        with self.assertRaises(ValueError):
            Evaluator(head, tuple(h[:3]))
        with self.assertRaises(ValueError):
            Evaluator(head, tuple(h), ladder=(FP32_SEQUENTIAL,))  # no exact completion
        with self.assertRaises(ValueError):
            quantize_head([[Q(1, 3)]])  # not a BF16 weight
        count('invalid_input_rejections', 5)

    def test_candidate_set_bounds(self) -> None:
        for _ in range(250):
            V, D = R.randint(2, 30), R.randint(2, 6)
            W, h = int_head(V, D)
            ev = Evaluator(quantize_head(W, bits=R.choice((3, 4))), tuple(h))
            iv = [ev.interval(0, i) for i in range(V)]
            z = [int(dot(w, h)) for w in W]
            a = exact_argmax([Q(v) for v in z])
            tau = max(lo for lo, _ in iv)
            C = [i for i in range(V) if iv[i][1] >= tau]
            width = [hi - lo for lo, hi in iv]
            self.assertIn(a, C)
            for i in C:
                self.assertGreaterEqual(z[i] + width[i], z[a] - width[a])
            t = max(width[i] for i in C) + width[a]
            Z = sum((pow2(v) for v in z), Q())
            p_max = pow2(z[a]) / Z
            self.assertLessEqual(len(C) * p_max, exp2_bounds(t)[1])
            count('candidate_set_bounds')

    def test_stock_contract(self) -> None:
        engine = (
            Accumulator('blocked', FP32, block=4, splits=2),
            Accumulator('sequential', FP32),
            Accumulator('fused', FP32, rounding='toward_zero', block=4),
        )
        certified, differs = 0, 0
        for _ in range(400):
            V, D = R.randint(2, 16), R.randint(2, 8)
            W = near_tie_head(V, D)
            h = bf16_vector(D, 0, 4)
            z = [dot(w, h) for w in W]
            a = exact_argmax(z)
            for acc in engine:
                G = [
                    acc.error_bound(sum((abs(x * y) for x, y in zip(w, h, strict=True)), Q()), D)
                    for w in W
                ]
                stock = stock_argmax(stock_logits(W, h, acc))
                if stock_agrees(z, a, G):
                    self.assertEqual(stock, a)
                    certified += 1
                elif stock != a:
                    differs += 1
                count('stock_contract_checks')
        count('stock_agreement_certified', certified)
        count('stock_differs_from_exact_argmax', differs)
        self.assertGreater(differs, 0)
        # BF16 output rounding: a later, strictly larger logit ties with an
        # earlier one and torch.argmax returns the earlier token.
        W = [[Q(16), Q(0)], [Q(16), Q(1, 32)]]
        h = [Q(1), Q(1)]
        z = [dot(w, h) for w in W]
        self.assertEqual(exact_argmax(z), 1)
        self.assertEqual(stock_argmax(stock_logits(W, h, FP32_SEQUENTIAL)), 0)
        WITNESSES['bf16_output_tie'] = {'z': list(map(str, z)), 'stock': 0, 'exact': 1}
        # Reduced-precision split-K: the gap exceeds delta under the FP32 model,
        # yet BF16 partials flip the token.
        W = [[Q(256), Q(5, 2), Q(-256), Q(0)], [Q(0), Q(9, 4), Q(0), Q(0)]]
        h = [Q(1)] * 4
        z = [dot(w, h) for w in W]
        split = Accumulator('sequential', FP32, splits=2)
        G = [split.error_bound(sum((abs(x) for x in w), Q()), 4) for w in W]
        self.assertTrue(stock_agrees(z, 0, G))
        self.assertEqual(stock_argmax(stock_logits(W, h, split)), 0)
        self.assertEqual(stock_argmax(stock_logits(W, h, split, bf16_partials=True)), 1)
        WITNESSES['bf16_partials_flip'] = {
            'z': list(map(str, z)),
            'stock_fp32': 0,
            'stock_bf16_partials': 1,
        }

    def test_integer_pass(self) -> None:
        """W8A8-style pass: exact integer accumulation, activation error term."""
        for case in range(200):
            V, D = R.randint(2, 24), R.randint(2, 24)
            W = near_tie_head(V, D) if case % 2 else random_head(V, D)
            h = bf16_vector(D, -3, 3)
            if case % 3 == 0:
                h, _ = with_outliers(h, 1)
            head = quantize_head(W, bits=R.choice((4, 8)))
            rule = EnvelopeRule(accumulator=FP32_SEQUENTIAL, activation_bits=R.choice((6, 8)))
            ev = Evaluator(head, tuple(h), rule)
            z = [dot(w, h) for w in W]
            for i in range(V):
                lo, hi = ev.interval(0, i)
                self.assertTrue(lo <= z[i] <= hi, (case, i))
                count('integer_pass_rows')
            cert = certify_argmax(ev)
            assert cert is not None
            self.assertEqual(cert.token, exact_argmax(z))
            count('integer_pass_argmax')

    def test_bf16_contract(self) -> None:
        """R-bf16 (argmax of BF16-rounded exact logits, lowest index on BF16
        ties) against the emulated stock head (FP32 accumulation, BF16 output).
        At these small widths the emulated stock rounding almost always equals
        exact rounding, so agreement here exercises the tie rule only; it does
        not show that R-bf16 is stock equality (see the next test)."""
        engine = (
            Accumulator('blocked', FP32, block=4, splits=2),
            Accumulator('sequential', FP32),
            Accumulator('fused', FP32, rounding='toward_zero', block=4),
        )
        agree_bf16, agree_real, rounding_matches = 0, 0, 0
        levels: dict[int, int] = {}
        for _ in range(300):
            V, D = R.randint(2, 24), R.randint(2, 8)
            W = near_tie_head(V, D)
            h = bf16_vector(D, 4, 6)  # logits of order 10-30, where BF16 spacing is 1/16-1/8
            ev = Evaluator(quantize_head(W, bits=8), tuple(h))
            z = [dot(w, h) for w in W]
            L = [round_to(v, BF16) for v in z]
            cert = certify_argmax(ev, contract='bf16')
            assert cert is not None
            self.assertEqual(cert.token, stock_argmax(L))
            levels[cert.level] = levels.get(cert.level, 0) + 1
            for acc in engine:
                stock_L = stock_logits(W, h, acc)
                stock = stock_argmax(stock_L)
                if stock_L == L:
                    self.assertEqual(stock, cert.token)
                    rounding_matches += 1
                agree_bf16 += stock == cert.token
                agree_real += stock == exact_argmax(z)
                count('bf16_contract_stock_comparisons')
            count('bf16_contract_cases')
        count('bf16_contract_agrees_with_stock', agree_bf16)
        count('real_contract_agrees_with_stock', agree_real)
        count('stock_rounding_identical_to_exact_rounding', rounding_matches)
        STATS['bf16_contract_level_histogram'] = {str(k): v for k, v in sorted(levels.items())}
        self.assertGreater(agree_bf16, agree_real)

    def test_bf16_contract_is_not_stock(self) -> None:
        """Witness that R-bf16 is not stock equality, and that the R-stock gap
        condition refuses to certify it (so the stock head would run)."""
        W = [[Q(16), Q(0), Q(0)], [Q(16), Q(1, 16), pow2(-30)]]
        h = [Q(1)] * 3
        z = [dot(w, h) for w in W]
        self.assertEqual(z, [Q(16), Q(16) + Q(1, 16) + pow2(-30)])
        self.assertEqual([round_to(v, BF16) for v in z], [Q(16), Q(16) + Q(1, 8)])
        cert = certify_argmax(Evaluator(quantize_head(W), tuple(h)), contract='bf16')
        assert cert is not None
        self.assertEqual(cert.token, 1)
        stock_L = stock_logits(W, h, FP32_SEQUENTIAL)
        self.assertEqual(stock_L, [Q(16), Q(16)])  # 16.0625 rounds to even
        self.assertEqual(stock_argmax(stock_L), 0)
        for g in (Q(0), pow2(-20), pow2(-10)):
            self.assertFalse(stock_agrees(z, 1, [g, g]))
        count('bf16_not_stock_witnesses')

    def test_race(self) -> None:
        levels: dict[int, int] = {}
        for case in range(360):
            V, D = R.randint(2, 32), R.randint(2, 12)
            W = near_tie_head(V, D) if case % 2 else random_head(V, D)
            h = bf16_vector(D, -3, 3)
            head = quantize_head(W, bits=R.choice((4, 8)))
            rule = EnvelopeRule(accumulator=R.choice((None, FP32_SEQUENTIAL)))
            ev = Evaluator(head, tuple(h), rule)
            z = [dot(w, h) for w in W]
            T = R.choice((Q(1), Q(1, 2), Q(3, 2), Q(7, 10)))
            spread = R.choice((Q(16), Q(1), Q(1, 64)))
            noise = [(counter_noise(case, 7, 'bonus', i) - Q(1, 2)) * spread for i in range(V)]
            excluded = set(R.sample(range(V), R.randint(0, V - 1)))
            top_k = R.choice((None, 1, max(1, (V - len(excluded)) // 2)))
            admissible = sorted(
                (i for i in range(V) if i not in excluded), key=lambda i: (-z[i], i)
            )
            admissible = admissible[:top_k] if top_k else admissible
            if case % 3 == 0 and len(admissible) > 1:  # exact tie at the top of the race
                w = dense_race(z, noise, T, excluded, top_k)
                j = R.choice([i for i in admissible if i != w])
                noise[j] = z[w] / T + noise[w] - z[j] / T
                count('race_forced_top_ties')
            cert = certify_race(ev, noise, T, excluded, top_k)
            self.assertEqual(cert.token, dense_race(z, noise, T, excluded, top_k))
            levels[cert.level] = levels.get(cert.level, 0) + 1
            count('race_cases')
        STATS['race_certified_level_histogram'] = {str(k): v for k, v in sorted(levels.items())}

    def test_topk(self) -> None:
        for _ in range(200):
            V, D = R.randint(1, 30), R.randint(2, 8)
            W = near_tie_head(V, D)
            h = bf16_vector(D, -3, 3)
            ev = Evaluator(quantize_head(W, bits=4), tuple(h))
            z = [dot(w, h) for w in W]
            k = R.randint(1, V)
            self.assertEqual(
                sorted(certify_topk(ev, k)), sorted(sorted(range(V), key=lambda i: (-z[i], i))[:k])
            )
            count('topk_cases')

    def test_rng_contract(self) -> None:
        """A counter-based field gives the same race winner in any evaluation
        order; a stateful stream consumed in evaluation order does not."""
        differ = 0
        for case in range(300):
            V = R.randint(3, 16)
            z = [Q(R.randint(-8, 8), 4) for _ in range(V)]
            field = [(counter_noise(case, 0, 'race', i) - Q(1, 2)) * 16 for i in range(V)]
            order = sorted(range(V), key=lambda i: -z[i])  # e.g. compacted candidates first
            by_order = [Q(0)] * V
            stream = random.Random(case)
            dense_stream = [Q(stream.randrange(2**16), 2**12) for _ in range(V)]
            for rank, i in enumerate(order):
                by_order[i] = dense_stream[rank]
            self.assertEqual(dense_race(z, field, Q(1)), dense_race(z, list(field), Q(1)))
            if dense_race(z, dense_stream, Q(1)) != dense_race(z, by_order, Q(1)):
                differ += 1
            count('rng_contract_cases')
        self.assertGreater(differ, 0)
        count('stateful_stream_divergences', differ)

    def test_exp2_bounds(self) -> None:
        for _ in range(400):
            n, d = R.randint(-40, 40), R.randint(1, 12)
            lo, hi = exp2_bounds(Q(n, d))
            # lo <= 2^(n/d) <= hi  <=>  lo^d <= 2^n <= hi^d (for positive values)
            self.assertLessEqual(lo**d, pow2(n))
            self.assertLessEqual(pow2(n), hi**d)
            self.assertLessEqual(hi / lo, Q(1025, 1000) if d > 1 else Q(1))
            count('exp2_enclosures')

    def test_acceptance(self) -> None:
        steps_total, rows_total = 0, 0
        for _ in range(300):
            V, D = R.randint(2, 20), R.randint(2, 6)
            W, h = int_head(V, D)
            me = mass_evaluator(W, h, R.choice((3, 4)))
            z = [int(dot(w, h)) for w in W]
            x = R.randrange(V)
            qx = Q(R.randint(1, 16), 16)
            u = Q(R.randrange(256), 256)
            threshold = pow2(z[x]) / (qx * sum((pow2(v) for v in z), Q()))
            if R.random() < 0.5 and threshold < 1:  # adversarial: u next to the threshold
                u = min(max(Q(0), threshold + Q(R.randint(-3, 3), 4096)), Q(4095, 4096))
                count('acceptance_adversarial_uniforms')
            d, rows, steps = certify_acceptance(me, x, qx, u)
            self.assertEqual(d, dense_acceptance(z, x, qx, u))
            self.assertLessEqual(rows, V)
            steps_total += steps
            rows_total += rows
            count('acceptance_decisions')
            count('acceptance_rows_total', V)
        STATS['acceptance_rows_rescored_fraction'] = rows_total / COUNTS['acceptance_rows_total']
        STATS['acceptance_mean_refinement_steps'] = steps_total / COUNTS['acceptance_decisions']
        # Unresolved measure with an uncertain numerator, and its tail bound.
        for _ in range(200):
            n_lo = Q(R.randint(1, 20))
            n_hi = n_lo + R.randint(0, 6)
            z_lo = Q(R.randint(1, 40))
            z_hi = z_lo + R.randint(0, 20)
            qx = Q(R.randint(1, 10), 10)
            cuts = sorted(
                {Q(0), Q(1), min(Q(1), n_lo / (qx * z_hi)), min(Q(1), n_hi / (qx * z_lo))}
            )
            measure = sum(
                (
                    b - a
                    for a, b in itertools.pairwise(cuts)
                    if a < b and guard(n_lo, n_hi, z_lo, z_hi, qx, (a + b) / 2) == 'unknown'
                ),
                Q(),
            )
            self.assertEqual(measure, unresolved_probability(n_lo, n_hi, z_lo, z_hi, qx))
            count('unresolved_measures')
        for _ in range(200):
            V, D = R.randint(2, 16), R.randint(2, 5)
            W, h = int_head(V, D)
            me = mass_evaluator(W, h, 4)
            z = [int(dot(w, h)) for w in W]
            x = R.randrange(V)
            for i in [x, *R.sample(range(V), R.randint(0, V - 1))]:
                me.logit(i)
            qx = Q(R.randint(1, 8), 8)
            z_lo, z_hi = me.partition()
            wx = pow2(z[x])
            Z = sum((pow2(v) for v in z), Q())
            tail = [i for i in range(V) if i not in me.rescored]
            pi_tail = sum((pow2(z[i]) for i in tail), Q()) / Z
            rho = max((me.mass(i)[1] / me.mass(i)[0] for i in tail), default=Q(1))
            bound = (wx / Z) / qx * (rho - 1) * pi_tail / (1 - pi_tail * (1 - 1 / rho))
            self.assertLessEqual(unresolved_probability(wx, wx, z_lo, z_hi, qx), bound)
            count('unresolved_tail_bounds')

    def test_residual_race(self) -> None:
        for _ in range(300):
            V, D = R.randint(2, 16), R.randint(2, 6)
            W, h = int_head(V, D)
            me = mass_evaluator(W, h, R.choice((3, 4)))
            z = [int(dot(w, h)) for w in W]
            support = R.sample(range(V), R.randint(1, min(3, V)))
            q = law(V, support)
            prio = [Q(R.randint(1, 50), R.randint(1, 50)) for _ in range(V)]
            try:
                expected = dense_residual_race(z, q, prio)
            except ValueError:
                with self.assertRaises(ValueError):
                    certify_residual_race(me, q, prio)
                count('zero_residual_refusals')
                continue
            token, rows, _ = certify_residual_race(me, q, prio)
            self.assertEqual(token, expected)
            self.assertLessEqual(rows, V)
            count('residual_race_cases')
            count('residual_race_rows_rescored', rows)
            count('residual_race_rows_total', V)
        # p == q makes the residual zero; both paths must refuse.
        W, h = int_head(4, 3)
        z = [int(dot(w, h)) for w in W]
        Z = sum((pow2(v) for v in z), Q())
        p = [pow2(v) / Z for v in z]
        prio = [Q(1)] * 4
        with self.assertRaises(ValueError):
            dense_residual_race(z, p, prio)
        with self.assertRaises(ValueError):
            certify_residual_race(mass_evaluator(W, h, 4), p, prio)
        count('zero_residual_refusals')

    def test_sibling_verify(self) -> None:
        for _ in range(300):
            V, D = R.randint(3, 16), R.randint(2, 6)
            W, h = int_head(V, D)
            me = mass_evaluator(W, h, R.choice((3, 4)))
            z = [int(dot(w, h)) for w in W]
            siblings = R.sample(range(V), R.randint(1, min(3, V - 1)))
            u = Q(R.randrange(256), 256)
            if R.random() < 0.5:  # adversarial: u next to a cumulative threshold
                Z = sum((pow2(v) for v in z), Q())
                cut = sum((pow2(z[x]) for x in siblings[: R.randint(1, len(siblings))]), Q()) / Z
                u = min(max(Q(0), cut + Q(R.randint(-3, 3), 4096)), Q(4095, 4096))
            prio = [Q(R.randint(1, 50), R.randint(1, 50)) for _ in range(V)]
            j, token, rows = certify_sibling_verify(me, siblings, u, prio)
            self.assertEqual((j, token), dense_sibling_verify(z, siblings, u, prio))
            self.assertLessEqual(rows, V)
            count('sibling_verify_cases')
            count('sibling_verify_rows_rescored', rows)
            count('sibling_verify_rows_total', V)

    def test_transport_versus_self_evidence(self) -> None:
        for _ in range(300):
            D, n = R.randint(2, 12), R.randint(2, 8)
            tile = [[Q(R.randint(-64, 64), 256) for _ in range(D)] for _ in range(n)]
            mean = [sum((w[j] for w in tile), Q()) / n for j in range(D)]
            centre = R.choice((mean, [Q(R.randint(-64, 64), 256) for _ in range(D)]))
            radius = max(abs(w[j] - centre[j]) for w in tile for j in range(D))
            diam = linf_diameter(tile)
            self.assertGreaterEqual(2 * radius, diam)
            hd = [Q(R.randint(-40, 40), 8) for _ in range(D)]
            h = [Q(R.randint(-40, 40), 8) for _ in range(D)]
            delta = [a - b for a, b in zip(h, hd, strict=True)]
            bits = R.choice((4, 8))
            l1 = sum((abs(x) for x in h), Q())
            if l1 == 0:
                continue
            for w in tile:
                if max(abs(x) for x in w) == 0:
                    continue
                s = max(abs(x) for x in w) / (2 ** (bits - 1) - 1)
                e = [x - round(x / s) * s for x in w]
                self.assertLessEqual(abs(dot(e, h)), rtn_holder_halfwidth(w, bits, h))
                if diam and sum((abs(x) for x in delta), Q()) / l1 >= s / diam:
                    self.assertLessEqual(
                        rtn_holder_halfwidth(w, bits, h), transport_halfwidth(radius, delta)
                    )
                    count('crossover_implications')
            # Proposition: with a centre in the tile's convex hull, transport's
            # tile bound exceeds the static Hoelder screen by r(|D|_1 - |h|_1) or more.
            radius_m = max(abs(w[j] - mean[j]) for w in tile for j in range(D))
            M_d = max(dot(w, hd) for w in tile)
            U_t = M_d + dot(mean, delta) + transport_halfwidth(radius_m, delta)
            U_s = dot(mean, h) + radius_m * l1
            self.assertGreaterEqual(U_t - U_s, radius_m * (sum((abs(x) for x in delta), Q()) - l1))
            count('transport_comparisons')


if __name__ == '__main__':
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Precision)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'status': 'pass' if result.wasSuccessful() else 'fail',
        'test_methods': result.testsRun,
        'seed': SEED,
        'repo_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'counts': COUNTS,
        'stats': STATS,
        'witnesses': WITNESSES,
        'seconds': round(time.perf_counter() - start, 1),
        'scope': 'exact rational emulation of BF16/FP32/FP64 rounding, synthetic heads; '
        'no GPU, no model weights; the fused tensor-core adder is a model, not hardware',
    }
    out = ROOT / 'evidence' / 'precision' / 'precision_tests.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(not result.wasSuccessful())
