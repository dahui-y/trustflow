"""Small dependency-free statistics for the analysis (rank correlation, AUROC, binning)."""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties handled, like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (Mann-Whitney)."""
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index 0..n_bins-1 by quantiles of x (equal-count bins)."""
    x = np.asarray(x, dtype=np.float64)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")
