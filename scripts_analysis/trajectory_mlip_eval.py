"""Milestone 0, Experiment A: evaluate CHGNet along FlowMM generation trajectories.

Run inside the MLIP environment (needs numpy, pandas, pymatgen, chgnet; pyarrow
for parquet). Produces one row per (sample, step, atom) plus a per-structure
table, and a JSON sidecar with the exact settings.

    python scripts_analysis/trajectory_mlip_eval.py outputs/m0/traj.npz outputs/m0/chgnet \
        --n_steps 11 --atom_types_from final

No guidance is computed here. STOP after the audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trustflow.chgnet_eval import evaluate_structures, load_chgnet
from trustflow.trajectory import TrajectoryBundle, per_atom_min_distance, structure_diagnostics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj_npz", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n_steps", type=int, default=11, help="evenly spaced saved states incl. both ends")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="explicit step indices (overrides --n_steps)")
    ap.add_argument("--atom_types_from", choices=["final", "step"], default="final")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--chgnet_model", default="0.3.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_prefilter", action="store_true", help="send even unphysical structures to CHGNet")
    ap.add_argument("--save_atom_fea", action="store_true", help="also dump (n_rows, 64) atom features to npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    traj = TrajectoryBundle.load_npz(args.traj_npz)
    steps = np.asarray(args.steps, dtype=np.int64) if args.steps else traj.select_steps(args.n_steps)
    n_samples = traj.num_samples if args.max_samples is None else min(args.max_samples, traj.num_samples)
    print(f"trajectory: {traj.num_samples} samples, T={traj.num_steps}; evaluating {n_samples} samples at steps {steps.tolist()}")

    model = load_chgnet(args.chgnet_model, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    struct_rows, atom_rows, fea_chunks = [], [], []
    for step in steps:
        t = traj.time_of_step(int(step))
        structures, keys = [], []
        for i in range(n_samples):
            s = traj.structure_at(i, int(step), args.atom_types_from)
            base = {"sample": i, "step": int(step), "t": t, "num_atoms": int(traj.num_atoms[i])}
            if s is None:
                struct_rows.append({**base, "invalid_z": True, "passes_prefilter": False, "evaluated": False, "error": "invalid atomic number"})
                continue
            diag = structure_diagnostics(s)
            row = {**base, "invalid_z": False, **diag, "evaluated": False, "error": None}
            if diag["passes_prefilter"] or args.no_prefilter:
                structures.append(s)
                keys.append(len(struct_rows))
            struct_rows.append(row)

        results = evaluate_structures(structures, model=model, batch_size=args.batch_size) if structures else []
        n_ok = 0
        for s, k, r in zip(structures, keys, results):
            struct_rows[k]["evaluated"] = r.ok
            struct_rows[k]["error"] = r.error
            if not r.ok:
                continue
            n_ok += 1
            struct_rows[k]["energy_per_atom"] = r.energy_per_atom
            fnorm = np.linalg.norm(r.forces, axis=1)
            struct_rows[k]["mean_force_norm"] = float(fnorm.mean())
            struct_rows[k]["max_force_norm"] = float(fnorm.max())
            dmin = per_atom_min_distance(s)
            z = np.array([sp.Z for sp in s.species])
            for a in range(len(s)):
                atom_rows.append({
                    "sample": struct_rows[k]["sample"], "step": int(step), "t": t, "atom": a, "Z": int(z[a]),
                    "force_norm": float(fnorm[a]), "fx": r.forces[a, 0], "fy": r.forces[a, 1], "fz": r.forces[a, 2],
                    "site_energy": float(r.site_energies[a]), "min_distance": float(dmin[a]),
                })
            if args.save_atom_fea:
                fea_chunks.append(r.atom_fea)
        print(f"step {int(step):5d} (t={t:.2f}): {len(structures)}/{n_samples} passed prefilter, {n_ok} evaluated  [{time.time() - t0:.0f}s]")

    structs = pd.DataFrame(struct_rows)
    atoms = pd.DataFrame(atom_rows)
    structs.to_parquet(args.out_dir / "structures.parquet", index=False)
    atoms.to_parquet(args.out_dir / "atoms.parquet", index=False)
    if args.save_atom_fea and fea_chunks:
        np.savez_compressed(args.out_dir / "atom_fea.npz", atom_fea=np.concatenate(fea_chunks, axis=0))
    meta = {
        "argv": sys.argv, "traj_npz": str(args.traj_npz.resolve()), "steps": steps.tolist(),
        "atom_types_from": args.atom_types_from, "n_samples": n_samples, "chgnet_model": args.chgnet_model,
        "prefilter": not args.no_prefilter, "n_struct_rows": len(structs), "n_atom_rows": len(atoms),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
