import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from trustflow.uq import QuantileGBM, QuantileGBMConfig, interval_coverage


def _heteroscedastic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 4)).astype(np.float32)
    noise_scale = 0.05 + 0.5 * (X[:, 0] > 0)  # noisier when x0 > 0
    y = X[:, 1] + rng.normal(0, noise_scale)
    return X, y, noise_scale


def test_fit_predict_coverage_and_heteroscedasticity(tmp_path):
    X, y, scale = _heteroscedastic()
    m = QuantileGBM().fit(X[:3000], y[:3000])  # default 500 rounds (UQ-MLIP setting)
    out = m.predict_uncertainty(X[3000:])
    cov = interval_coverage(y[3000:], out["lower"], out["upper"])
    assert 0.7 <= cov <= 0.97  # nominal 0.90; loose because it varies across xgboost versions
    hi = out["uncertainty"][scale[3000:] > 0.3].mean()
    lo = out["uncertainty"][scale[3000:] < 0.3].mean()
    assert hi > 2.5 * lo  # the model must rank the noisy region as more uncertain

    m.save(tmp_path)
    m2 = QuantileGBM.load(tmp_path)
    out2 = m2.predict_uncertainty(X[3000:3010])
    np.testing.assert_allclose(out2["uncertainty"], out["uncertainty"][:10], rtol=1e-5)


def test_feature_count_mismatch_raises():
    X, y, _ = _heteroscedastic(n=200)
    m = QuantileGBM(QuantileGBMConfig(n_estimators=5)).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_uncertainty(X[:, :3])
