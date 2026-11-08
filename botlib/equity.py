# build: dh7q_equityedge_k41z
"""Monte Carlo equity against a uniformly random opponent hand."""
import numpy as np

from .fast_eval import FLAT, NT, PAIR_IDX, TRIP_IDX, P0, P1, T0, T1, T2

_ALL = np.arange(36)


def mc_equity(hole, board, n, rng):
    nb = len(board)
    need = 5 - nb
    mask = np.ones(36, dtype=bool)
    mask[list(hole)] = False
    if nb:
        mask[list(board)] = False
    deck = _ALL[mask]
    k = 5 + need
    r = rng.random((n, deck.shape[0]))
    idx = np.argpartition(r, k - 1, axis=1)[:, :k]
    opp = deck[idx[:, :5]]
    boards = np.empty((n, 5), dtype=np.int64)
    if nb:
        boards[:, :nb] = board
    if need:
        boards[:, nb:] = deck[idx[:, 5:]]
    return _equity_from(np.asarray(hole, dtype=np.int64), opp, boards, n)


def _equity_from(h, opp, boards, n):
    bt = TRIP_IDX[boards[:, T0], boards[:, T1], boards[:, T2]]
    op = PAIR_IDX[opp[:, P0], opp[:, P1]]
    mp = PAIR_IDX[h[P0], h[P1]]
    bt3 = bt[:, None, :]
    mine = FLAT[mp[None, :, None] * NT + bt3].max(axis=(1, 2))
    theirs = FLAT[op[:, :, None] * NT + bt3].max(axis=(1, 2))
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / n)
