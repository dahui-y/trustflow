"""Per-atom uncertainty via quantile gradient boosting on MLIP atom features.

Re-implements the UQ-MLIP recipe (Bilbrey et al.) in ~60 lines so the MLIP
environment needs only numpy + xgboost:

    features X : (N_atoms, F)   per-atom embedding (CHGNet ``atom_fea``, F=64)
    target   y : (N_atoms,)     per-atom quantity (default: CHGNet ``site_energies``)
    model      : XGBoost ``reg:quantileerror`` at alphas (lower, upper)
    uncertainty = |q_upper - q_lower| / 2

The target can be any per-atom array, so the same class also serves a
force-disagreement target later (Experiment B) without code changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class QuantileGBMConfig:
    lower_alpha: float = 0.05
    upper_alpha: float = 0.95
    n_estimators: int = 500
    learning_rate: float = 0.04
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    device: str = "cpu"
    seed: int = 0


class QuantileGBM:
    """Two-quantile XGBoost regressor with save/load."""

    def __init__(self, config: QuantileGBMConfig | None = None):
        self.config = config or QuantileGBMConfig()
        self.booster = None
        self.n_features: int | None = None

    def _params(self) -> dict:
        c = self.config
        return {
            "objective": "reg:quantileerror",
            "quantile_alpha": [c.lower_alpha, c.upper_alpha],
            "tree_method": "hist",
            "device": c.device,
            "learning_rate": c.learning_rate,
            "max_depth": c.max_depth,
            "subsample": c.subsample,
            "colsample_bytree": c.colsample_bytree,
            "seed": c.seed,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileGBM":
        import xgboost as xgb

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        self.n_features = X.shape[1]
        dtrain = xgb.QuantileDMatrix(X, y)
        self.booster = xgb.train(self._params(), dtrain, num_boost_round=self.config.n_estimators)
        return self

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import xgboost as xgb

        if self.booster is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        out = self.booster.inplace_predict(X)  # (N, 2)
        return out[:, 0], out[:, 1]

    def predict_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        lo, hi = self.predict_quantiles(X)
        return {"lower": lo, "upper": hi, "uncertainty": np.abs(hi - lo) / 2.0}

    # ------------------------------------------------------------ io
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(d / "quantile_gbm.json")
        (d / "config.json").write_text(json.dumps({**asdict(self.config), "n_features": self.n_features}, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "QuantileGBM":
        import xgboost as xgb

        d = Path(directory)
        meta = json.loads((d / "config.json").read_text())
        n_features = meta.pop("n_features")
        obj = cls(QuantileGBMConfig(**meta))
        obj.booster = xgb.Booster()
        obj.booster.load_model(d / "quantile_gbm.json")
        obj.n_features = n_features
        return obj


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of targets inside [lower, upper]; ~0.90 expected for alphas 0.05/0.95."""
    y = np.asarray(y).reshape(-1)
    return float(np.mean((y >= lower) & (y <= upper)))
