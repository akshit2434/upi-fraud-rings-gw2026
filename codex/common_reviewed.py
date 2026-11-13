import numpy as np
from sklearn.metrics import f1_score

from features import LABELS


def macro_f1(y, pred):
    return f1_score(y, pred, average='macro', labels=list(range(len(LABELS))))


def tune_weights(P, y, iters=3, grid=None):
    """Coordinate ascent on per-class multipliers applied to probabilities (argmax decision)."""
    k = P.shape[1]
    w = np.ones(k)
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.05), np.log(20), 41))
    best = macro_f1(y, (P * w).argmax(1))
    for _ in range(iters):
        for c in range(1, k):
            bw = w[c]
            for g in grid:
                w[c] = g
                f = macro_f1(y, (P * w).argmax(1))
                if f > best:
                    best, bw = f, g
            w[c] = bw
    return w, best
