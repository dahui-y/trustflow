"""Attach per-atom uncertainty to a trajectory reliability analysis table.

Reads <eval_dir>/atoms.parquet and <eval_dir>/atom_fea.npz (rows aligned), applies
the quantile GBM from <uq_dir>, and writes <eval_dir>/atoms_uq.parquet with
columns uq_lower, uq_upper, uq, plus is_high_uq (uq above the in-distribution
reference percentile, default p95).

    python scripts_analysis/uq_apply.py outputs/m0/chgnet_ep1300 outputs/m0/uq_chgnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustflow.uq import QuantileGBM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("uq_dir", type=Path)
    ap.add_argument("--high_uq_percentile", default="p95", help="key in uq meta reference_uq_percentiles")
    args = ap.parse_args()

    atoms = pd.read_parquet(args.eval_dir / "atoms.parquet")
    fea = np.load(args.eval_dir / "atom_fea.npz")["atom_fea"]
    if len(atoms) != fea.shape[0]:
        raise SystemExit(f"row mismatch: atoms.parquet has {len(atoms)} rows, atom_fea has {fea.shape[0]}")

    gbm = QuantileGBM.load(args.uq_dir)
    out = gbm.predict_uncertainty(fea)
    atoms["uq_lower"], atoms["uq_upper"], atoms["uq"] = out["lower"], out["upper"], out["uncertainty"]
    meta = json.loads((args.uq_dir / "meta.json").read_text())
    thr = meta["reference_uq_percentiles"][args.high_uq_percentile]
    atoms["is_high_uq"] = atoms["uq"] > thr
    atoms.to_parquet(args.eval_dir / "atoms_uq.parquet", index=False)

    summary = atoms.groupby("t").agg(n=("uq", "size"), uq_median=("uq", "median"), uq_p90=("uq", lambda s: s.quantile(0.9)), high_uq_frac=("is_high_uq", "mean"), force_median=("force_norm", "median"))
    print(f"high-UQ threshold ({args.high_uq_percentile} of in-distribution MP-20 val): {thr:.4f}")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    (args.eval_dir / "uq_summary_by_t.csv").write_text(summary.to_csv())


if __name__ == "__main__":
    main()
