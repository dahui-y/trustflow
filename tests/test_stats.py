import numpy as np
import pytest

from trustflow.stats import auroc, quantile_bins, spearman


def test_spearman_matches_scipy_when_available():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300); y = x + rng.normal(size=300); y[::7] = y[0]  # ties
    r = spearman(x, y)
    scipy_stats = pytest.importorskip("scipy.stats")
    assert r == pytest.approx(scipy_stats.spearmanr(x, y).correlation, abs=1e-12)


def test_spearman_perfect_and_nan():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 2], [1, 2]))


def test_auroc():
    score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    pos = np.array([False, False, True, True, True])
    assert auroc(score, pos) == pytest.approx(5 / 6)  # one inversion (0.35 < 0.4)
    assert auroc(score, np.ones(5, bool)) != auroc(score, np.ones(5, bool))  # nan


def test_quantile_bins_equal_counts():
    x = np.random.default_rng(1).normal(size=1000)
    b = quantile_bins(x, 5)
    counts = np.bincount(b, minlength=5)
    assert b.min() == 0 and b.max() == 4 and counts.min() >= 195
