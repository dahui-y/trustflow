"""Experiment B, step 2: does CHGNet per-atom UQ predict CHGNet-vs-DPA force disagreement?

Inputs: <chgnet_dir>/atoms_uq.parquet (uq_apply.py) and <dpa_dir>/dpa_atoms.parquet.
Joins on (sample, step, atom), computes

    D_F     = |F_CHGNet - F_DPA|                      (eV/A)
    D_F_rel = D_F / (|F_CHGNet| + |F_DPA| + 1e-6)     in [0, 1]

and reports Spearman(uq, D_F), AUROC of uq for flagging the top-decile D_F atoms,
per t and pooled, plus uq-decile-binned D_F. Writes cross_mlip_summary.json and
fig5 / fig6. Needs numpy, pandas, matplotlib only (either environment).

    python scripts_analysis/cross_mlip_analysis.py outputs/m0/chgnet_ep1300 outputs/m0/dpa_ep1300 --out figures/m0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trustflow.stats import auroc, quantile_bins, spearman  # noqa: E402

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
})


def merge_tables(chg: pd.DataFrame, dpa: pd.DataFrame) -> pd.DataFrame:
    m = chg.merge(dpa, on=["sample", "step", "atom"], suffixes=("", "_d"), how="inner")
    if "Z_d" in m and not (m["Z"] == m["Z_d"]).all():
        raise ValueError("atomic numbers disagree between CHGNet and DPA tables: row alignment broken")
    df = np.stack([m.fx - m.fx_dpa, m.fy - m.fy_dpa, m.fz - m.fz_dpa], axis=1)
    m["D_F"] = np.linalg.norm(df, axis=1)
    m["D_F_rel"] = m["D_F"] / (m["force_norm"] + m["force_norm_dpa"] + 1e-6)
    return m


def stats_block(g: pd.DataFrame, top_frac: float = 0.1) -> dict:
    thr = g["D_F"].quantile(1 - top_frac)
    return {
        "n": int(len(g)),
        "D_F_median": float(g["D_F"].median()),
        "D_F_rel_median": float(g["D_F_rel"].median()),
        "spearman_uq_DF": spearman(g["uq"], g["D_F"]),
        "spearman_uq_DFrel": spearman(g["uq"], g["D_F_rel"]),
        "spearman_Fnorm_DF": spearman(g["force_norm"], g["D_F"]),
        "auroc_uq_topdecile_DF": auroc(g["uq"], g["D_F"] >= thr),
        "auroc_Fnorm_topdecile_DF": auroc(g["force_norm"], g["D_F"] >= thr),
    }


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chgnet_dir", type=Path)
    ap.add_argument("dpa_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figures/m0"))
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    chg = pd.read_parquet(args.chgnet_dir / "atoms_uq.parquet")
    dpa = pd.read_parquet(args.dpa_dir / "dpa_atoms.parquet")
    m = merge_tables(chg, dpa)
    print(f"joined {len(m)} atoms (chgnet {len(chg)}, dpa {len(dpa)})")

    summary = {"pooled": stats_block(m), "by_t": {}}
    for t, g in m.groupby("t"):
        summary["by_t"][f"{t:.2f}"] = stats_block(g)
    tbl = pd.DataFrame(summary["by_t"]).T
    print(tbl[["n", "D_F_median", "spearman_uq_DF", "spearman_Fnorm_DF", "auroc_uq_topdecile_DF", "auroc_Fnorm_topdecile_DF"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("pooled:", json.dumps(summary["pooled"], indent=2))

    # fig5: disagreement along the flow
    t = np.array(sorted(m["t"].unique()))
    g = m.groupby("t")["D_F"]
    med, q25, q75 = (g.quantile(q).reindex(t).to_numpy() for q in (0.5, 0.25, 0.75))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.fill_between(t, q25, q75, color=BLUE, alpha=0.18, linewidth=0, label="IQR")
    ax.plot(t, med, color=BLUE, linewidth=2, marker="o", markersize=5, label="median")
    ax.set_yscale("log"); ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("flow time t"); ax.set_ylabel("per-atom |F_CHGNet − F_DPA|  (eV/Å)")
    ax.set_title("Cross-model force disagreement along the flow", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, args.out, "fig5_disagreement_vs_t")

    # fig6: D_F binned by uq decile, for early / late halves
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for label, color, sel in (("early  t ≤ 0.5", ORANGE, m["t"] <= 0.5), ("late  t > 0.5", BLUE, m["t"] > 0.5)):
        g = m[sel]
        if len(g) < 10 * args.n_bins:
            continue
        b = quantile_bins(g["uq"].to_numpy(), args.n_bins)
        med_b = np.array([np.median(g["D_F"].to_numpy()[b == k]) for k in range(args.n_bins)])
        ax.plot(np.arange(1, args.n_bins + 1), med_b, color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xlabel(f"CHGNet per-atom UQ decile (1 = lowest … {args.n_bins} = highest)")
    ax.set_ylabel("median |F_CHGNet − F_DPA|  (eV/Å)")
    ax.set_yscale("log")
    ax.set_title("Does CHGNet uncertainty rank cross-model disagreement?", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, args.out, "fig6_disagreement_by_uq_decile")

    (args.out / "cross_mlip_summary.json").write_text(json.dumps(summary, indent=2))
    m[["sample", "step", "t", "atom", "Z", "uq", "force_norm", "force_norm_dpa", "D_F", "D_F_rel", "min_distance"]].to_parquet(
        args.chgnet_dir / "atoms_cross_mlip.parquet", index=False)


if __name__ == "__main__":
    main()
