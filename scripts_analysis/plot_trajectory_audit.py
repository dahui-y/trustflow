"""Milestone 0 figures from a trajectory audit directory.

Input: <eval_dir>/structures.parquet, <eval_dir>/atoms_uq.parquet (from uq_apply.py;
falls back to atoms.parquet without UQ panels), optional <uq_dir>/reference_uq.npz.

    python scripts_analysis/plot_trajectory_audit.py outputs/m0/chgnet_ep1300 \
        --uq_dir outputs/m0/uq_chgnet --out figures/m0

Figures (PNG + SVG; remember `git add -f` because *.png is gitignored):
  fig1_uq_vs_t          per-atom CHGNet uncertainty along the flow (median, IQR, p90)
  fig2_high_uq_frac     fraction of atoms above the in-distribution p95 threshold vs t
  fig3_force_vs_t       per-atom |F| along the flow
  fig4_prefilter_vs_t   fraction of intermediate structures that are physically evaluable
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

# reference palette (dataviz skill): series-1 blue, series-2 orange; ink tokens
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
})


def _band(ax, df: pd.DataFrame, col: str, label: str):
    g = df.groupby("t")[col]
    t = np.array(sorted(df["t"].unique()))
    med, q25, q75, p90 = (g.quantile(q).reindex(t).to_numpy() for q in (0.5, 0.25, 0.75, 0.9))
    ax.fill_between(t, q25, q75, color=BLUE, alpha=0.18, linewidth=0, label="IQR")
    ax.plot(t, med, color=BLUE, linewidth=2, marker="o", markersize=5, label="median")
    ax.plot(t, p90, color=BLUE, linewidth=1.2, linestyle="--", label="p90")
    ax.set_xlabel("flow time t  (0 = noise, 1 = generated crystal)")
    ax.set_ylabel(label)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return t, med


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--uq_dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures/m0"))
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    structs = pd.read_parquet(args.eval_dir / "structures.parquet")
    atoms_path = args.eval_dir / "atoms_uq.parquet"
    has_uq = atoms_path.exists()
    atoms = pd.read_parquet(atoms_path if has_uq else args.eval_dir / "atoms.parquet")
    n_traj = structs["sample"].nunique()
    summary = {}

    if has_uq:
        ref_line = None
        if args.uq_dir and (args.uq_dir / "reference_uq.npz").exists():
            ref = np.load(args.uq_dir / "reference_uq.npz")["uncertainty"]
            ref_line = (float(np.median(ref)), float(np.percentile(ref, 95)))
        fig, ax = plt.subplots(figsize=(6.4, 4))
        t, med = _band(ax, atoms, "uq", "per-atom CHGNet uncertainty  (eV, quantile half-width)")
        if ref_line:
            ax.axhline(ref_line[0], color=ORANGE, linewidth=1.5, label="MP-20 val median (in-distribution)")
            ax.axhline(ref_line[1], color=ORANGE, linewidth=1.2, linestyle=":", label="MP-20 val p95")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax.set_title(f"MLIP uncertainty along FlowMM trajectories (n={n_traj}){args.title_suffix}", color=INK, loc="left")
        _save(fig, args.out, "fig1_uq_vs_t")
        summary["uq_median_by_t"] = dict(zip(map(float, t), map(float, med)))

        frac = atoms.groupby("t")["is_high_uq"].mean()
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(frac.index, frac.values, color=BLUE, linewidth=2, marker="o", markersize=5)
        ax.axhline(0.05, color=ORANGE, linewidth=1.2, linestyle=":", label="5% expected in-distribution")
        ax.set_xlabel("flow time t"); ax.set_ylabel("fraction of atoms above in-distribution p95 UQ")
        ax.set_ylim(0, max(1.0, frac.max() * 1.05)); ax.set_xlim(-0.02, 1.02); ax.legend(frameon=False)
        ax.set_title(f"High-uncertainty atom fraction along the flow{args.title_suffix}", color=INK, loc="left")
        _save(fig, args.out, "fig2_high_uq_frac")
        summary["high_uq_frac_by_t"] = {float(k): float(v) for k, v in frac.items()}

    fig, ax = plt.subplots(figsize=(6.4, 4))
    t, med = _band(ax, atoms, "force_norm", "per-atom |F_CHGNet|  (eV/Å)")
    ax.set_yscale("log")
    ax.set_title(f"CHGNet force magnitude along the flow{args.title_suffix}", color=INK, loc="left")
    _save(fig, args.out, "fig3_force_vs_t")
    summary["force_median_by_t"] = dict(zip(map(float, t), map(float, med)))

    rate = structs.groupby("t").agg(prefilter=("passes_prefilter", "mean"), evaluated=("evaluated", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(rate.index, rate["prefilter"], color=BLUE, linewidth=2, marker="o", markersize=5, label="passes prefilter (d_min ≥ 0.5 Å, V ≥ 0.1 Å³)")
    ax.plot(rate.index, rate["evaluated"], color=ORANGE, linewidth=1.5, linestyle="--", marker="s", markersize=4, label="CHGNet evaluated OK")
    ax.set_xlabel("flow time t"); ax.set_ylabel("fraction of structures"); ax.set_ylim(0, 1.02); ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Physically evaluable intermediate structures{args.title_suffix}", color=INK, loc="left")
    _save(fig, args.out, "fig4_prefilter_vs_t")
    summary["prefilter_rate_by_t"] = {float(k): float(v) for k, v in rate["prefilter"].items()}

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
