"""Vectorised Omaha evaluator: a (2-card hole pair) x (3-card board triple) score table.

TABLE[pair_id, triple_id] is the official 5-card score of those 5 cards (0 when they overlap),
so the best Omaha score of a hand is a max over 10 x 10 table cells.
The table is loaded from data/omaha_table.npy (read only). If that file is missing or fails
the spot check against the official evaluator, it is rebuilt in memory from evaluator.get_table().
"""
import os
import random
from itertools import combinations

import numpy as np

import evaluator as _official

_DIR = os.path.dirname(os.path.abspath(__file__))
_TABLE_PATH = os.path.join(_DIR, "data", "omaha_table.npy")

PAIRS = list(combinations(range(36), 2))
TRIPLES = list(combinations(range(36), 3))
NP = len(PAIRS)
NT = len(TRIPLES)

PAIR_IDX = np.full((36, 36), -1, dtype=np.int64)
for _k, (_a, _b) in enumerate(PAIRS):
    PAIR_IDX[_a, _b] = _k
    PAIR_IDX[_b, _a] = _k

TRIP_IDX = np.full((36, 36, 36), -1, dtype=np.int64)
for _k, _t in enumerate(TRIPLES):
    for _p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        TRIP_IDX[_t[_p[0]], _t[_p[1]], _t[_p[2]]] = _k

P2 = np.array(list(combinations(range(5), 2)))
P0, P1 = P2[:, 0], P2[:, 1]
T3 = np.array(list(combinations(range(5), 3)))
T0, T1, T2 = T3[:, 0], T3[:, 1], T3[:, 2]


def _build_from_official():
    table = _official.get_table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    masks = (np.int64(1) << combos).sum(axis=1)
    scores = np.fromiter((table[int(m)] for m in masks), dtype=np.uint32, count=len(masks))
    out = np.zeros((NP, NT), dtype=np.uint32)
    for i, j in combinations(range(5), 2):
        rest = [k for k in range(5) if k not in (i, j)]
        pid = PAIR_IDX[combos[:, i], combos[:, j]]
        tid = TRIP_IDX[combos[:, rest[0]], combos[:, rest[1]], combos[:, rest[2]]]
        out[pid, tid] = scores
    return out


def _spot_check(t, n=300):
    rng = random.Random(1)
    for _ in range(n):
        c = rng.sample(range(36), 5)
        a, b = sorted(c[:2])
        x, y, z = sorted(c[2:])
        if int(t[PAIR_IDX[a, b], TRIP_IDX[x, y, z]]) != _official.score_five(c):
            return False
    return True


def _load():
    try:
        t = np.load(_TABLE_PATH)
        if t.shape == (NP, NT) and t.dtype == np.uint32 and _spot_check(t):
            return t, "file"
    except Exception:
        pass
    return _build_from_official(), "rebuilt"


TABLE, TABLE_SOURCE = _load()
FLAT = np.ascontiguousarray(TABLE).ravel()


def omaha_score(hole, board):
    h = list(hole)
    b = list(board)
    pids = np.array([PAIR_IDX[h[i], h[j]] for i, j in combinations(range(len(h)), 2)])
    tids = np.array([TRIP_IDX[b[i], b[j], b[k]] for i, j, k in combinations(range(len(b)), 3)])
    return int(FLAT[(pids[:, None] * NT + tids[None, :])].max())
