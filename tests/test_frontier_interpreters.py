"""The frontier oracles' interpreters against naive references (experiments/frontier)."""

from __future__ import annotations

import random
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/frontier'))

from interpreters import MAX_COPY, NO_PREDICTION, NGram, copy_walk, mix_token


def naive_copy(hist: list[int]) -> tuple[int, int]:
    for n in range(min(MAX_COPY, len(hist)), 0, -1):
        suffix = hist[len(hist) - n :]
        for e in range(len(hist) - 2, n - 2, -1):  # occurrence ending at e, e + 1 in history
            if hist[e - n + 1 : e + 1] == suffix:
                return hist[e + 1], n
    return NO_PREDICTION, 0


def test_copy_walk_matches_naive_including_altered_histories() -> None:
    rng = random.Random(0)
    for trial in range(40):
        vocab = rng.choice([2, 3, 5, 12])
        seq = [rng.randrange(vocab) for _ in range(rng.randrange(2, 60))]
        tails = {}
        for _ in range(3):
            t = rng.randrange(1, len(seq) + 1)
            tails[t] = [rng.randrange(vocab) for _ in range(rng.randrange(1, 8))]
        preds, lens, altered = copy_walk(seq, tails)
        for p in range(1, len(seq)):
            assert (preds[p], lens[p]) == naive_copy(seq[:p]), (trial, p)
        for t, tail in tails.items():
            for u in range(1, len(tail) + 1):
                hist = seq[:t] + tail[:u]
                assert altered[t][u - 1] == naive_copy(hist), (trial, t, u)


def test_ngram_backoff_and_mix() -> None:
    ngram = NGram([[1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 5]])
    assert ngram.predict([1, 2, 3]) == 5  # 5 follows (1, 2, 3) twice, 4 once
    assert ngram.predict([9, 2, 3]) == 5  # backoff to (2, 3)
    assert ngram.predict([9, 9, 4]) == 1  # backoff to (4,)
    assert ngram.predict([9, 9, 9]) == 1  # unigram mode (1, 2, 3 tie at 3: smallest id)
    assert mix_token(7, 3, ngram, [1, 2, 3]) == 7
    assert mix_token(7, 2, ngram, [1, 2, 3]) == 5


def test_anchor_proxy_extension_past_slot_15() -> None:
    from anchor_bound import proxy

    s_d = [0.9 * 0.8**k for k in range(15)]  # S_D(1..15); every late acceptance is 0.8
    flat = proxy(s_d, 20, decayed=False)
    assert flat[13] == max(s_d[13], s_d[14] / s_d[0])  # m = 14: the measured forms
    assert flat[14] == max(s_d[14], s_d[14] / s_d[0])  # m = 15: S_D(16) = S_D(15)
    assert flat[15:] == [flat[14]] * 5  # no decay past slot 15
    assert all(a >= b for a, b in pairwise(flat))
    decayed = proxy(s_d, 17, decayed=True)
    assert abs(decayed[14] - max(s_d[14], s_d[14] * 0.8 / s_d[0])) < 1e-12
    assert abs(decayed[16] - decayed[14] * 0.8**2) < 1e-12
