import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
spec = importlib.util.spec_from_file_location("cross_mlip_analysis", Path(__file__).resolve().parents[1] / "scripts_analysis" / "cross_mlip_analysis.py")
mod = importlib.util.module_from_spec(spec); sys.modules["cross_mlip_analysis"] = mod; spec.loader.exec_module(mod)


def test_merge_and_stats():
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"sample": np.repeat(np.arange(40), 10), "step": 500, "t": 0.5, "atom": np.tile(np.arange(10), 40), "Z": 6})
    F = rng.normal(size=(n, 3))
    chg = base.assign(fx=F[:, 0], fy=F[:, 1], fz=F[:, 2], force_norm=np.linalg.norm(F, axis=1), uq=rng.uniform(size=n), min_distance=1.0)
    noise = rng.normal(size=(n, 3)) * (0.1 + chg.uq.to_numpy()[:, None])  # disagreement grows with uq
    Fd = F + noise
    dpa = base.assign(fx_dpa=Fd[:, 0], fy_dpa=Fd[:, 1], fz_dpa=Fd[:, 2], force_norm_dpa=np.linalg.norm(Fd, axis=1), e_atom_dpa=-1.0)
    m = mod.merge_tables(chg, dpa)
    assert len(m) == n
    np.testing.assert_allclose(m.D_F, np.linalg.norm(noise, axis=1))
    s = mod.stats_block(m)
    assert s["spearman_uq_DF"] > 0.4 and s["auroc_uq_topdecile_DF"] > 0.7


def test_merge_detects_misalignment():
    base = pd.DataFrame({"sample": [0, 0], "step": [1, 1], "atom": [0, 1], "Z": [6, 8]})
    chg = base.assign(fx=0.0, fy=0.0, fz=0.0, force_norm=0.0, uq=0.0)
    dpa = base.assign(Z=[6, 7], fx_dpa=0.0, fy_dpa=0.0, fz_dpa=0.0, force_norm_dpa=0.0)
    with pytest.raises(ValueError):
        mod.merge_tables(chg, dpa)
