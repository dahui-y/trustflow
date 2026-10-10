"""Train the per-atom quantile GBM (UQ-MLIP recipe) on CHGNet features of MP-20 crystals.

MP-20 structures are inside CHGNet's training distribution (MPtrj), so this is the
"in-distribution" reference the trajectory audit is compared against. Runs in the
flowmm env (chgnet 0.3.1 present).

    python scripts_analysis/uq_train_mp20.py data/mp_20 outputs/m0/uq_chgnet \
        --n_train 3000 --n_val 500 --seed 0

Outputs in <out_dir>: quantile_gbm.json, config.json, reference_uq.npz (per-atom
uncertainty on held-out MP-20 val structures), meta.json (coverage, thresholds).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from trustflow.chgnet_eval import evaluate_structures, load_chgnet
from trustflow.uq import QuantileGBM, QuantileGBMConfig, interval_coverage


def _structures_from_csv(path: Path, n: int, seed: int) -> list[Structure]:
    df = pd.read_csv(path, usecols=["material_id", "cif"])
    df = df.sample(n=min(n, len(df)), random_state=seed)
    out = []
    for cif in df["cif"]:
        try:
            out.append(Structure.from_str(cif, fmt="cif"))
        except Exception:  # noqa: BLE001
            continue
    return out


def _extract(structures, model, batch_size):
    res = evaluate_structures(structures, model=model, batch_size=batch_size)
    X = np.concatenate([r.atom_fea for r in res if r.ok], axis=0)
    y = np.concatenate([r.site_energies for r in res if r.ok], axis=0)
    n_fail = sum(not r.ok for r in res)
    return X, y, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory with train.csv / val.csv")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()

    model = load_chgnet(device=args.device)
    train_s = _structures_from_csv(args.data_dir / "train.csv", args.n_train, args.seed)
    val_s = _structures_from_csv(args.data_dir / "val.csv", args.n_val, args.seed + 1)
    print(f"parsed {len(train_s)} train / {len(val_s)} val structures [{time.time() - t0:.0f}s]")

    Xtr, ytr, ftr = _extract(train_s, model, args.batch_size)
    Xva, yva, fva = _extract(val_s, model, args.batch_size)
    print(f"features: train {Xtr.shape}, val {Xva.shape}; CHGNet failures {ftr}/{fva} [{time.time() - t0:.0f}s]")

    gbm = QuantileGBM(QuantileGBMConfig(n_estimators=args.n_estimators, seed=args.seed)).fit(Xtr, ytr)
    out = gbm.predict_uncertainty(Xva)
    cov = interval_coverage(yva, out["lower"], out["upper"])
    u = out["uncertainty"]
    thresholds = {f"p{p}": float(np.percentile(u, p)) for p in (50, 75, 90, 95, 99)}
    print(f"val coverage of 5-95 interval: {cov:.3f}; reference uq percentiles: {thresholds}")

    gbm.save(args.out_dir)
    np.savez_compressed(args.out_dir / "reference_uq.npz", uncertainty=u, lower=out["lower"], upper=out["upper"], y=yva)
    meta = {
        "n_train_structures": len(train_s), "n_val_structures": len(val_s),
        "n_train_atoms": int(Xtr.shape[0]), "n_val_atoms": int(Xva.shape[0]),
        "target": "chgnet_site_energy", "features": "chgnet_atom_fea_64",
        "val_interval_coverage": cov, "reference_uq_percentiles": thresholds,
        "seed": args.seed, "n_estimators": args.n_estimators, "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
