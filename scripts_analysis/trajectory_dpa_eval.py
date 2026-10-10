"""Experiment B, step 1: DPA forces on the same intermediate structures as the CHGNet audit.

Runs in the *mlip* environment (deepmd-kit[torch], pymatgen, pandas, pyarrow). Reads
the same traj.npz, rebuilds structures with the same atom-type policy and the same
prefilter, so rows align 1:1 with the CHGNet atom table on (sample, step, atom).

    python scripts_analysis/trajectory_dpa_eval.py outputs/m0/traj.npz outputs/m0/dpa_ep1300 \
        --n_steps 11 --model DPA-3.1-3M --head MP_traj_v024_alldata_mixu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trustflow.dpa_eval import DEFAULT_HEAD, DEFAULT_MODEL, evaluate_structures_dpa, load_dpa
from trustflow.trajectory import TrajectoryBundle, structure_diagnostics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj_npz", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n_steps", type=int, default=11)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--atom_types_from", choices=["final", "step"], default="final")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--head", default=DEFAULT_HEAD)
    ap.add_argument("--no_prefilter", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    traj = TrajectoryBundle.load_npz(args.traj_npz)
    steps = np.asarray(args.steps, dtype=np.int64) if args.steps else traj.select_steps(args.n_steps)
    n_samples = traj.num_samples if args.max_samples is None else min(args.max_samples, traj.num_samples)
    dp = load_dpa(args.model, args.head)
    print(f"DPA {args.model} head={args.head}: {len(dp.get_type_map())} element types, rcut={dp.get_rcut()}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atom_rows, struct_rows = [], []
    for step in steps:
        t = traj.time_of_step(int(step))
        structures, idx = [], []
        for i in range(n_samples):
            s = traj.structure_at(i, int(step), args.atom_types_from)
            if s is None:
                continue
            if not (structure_diagnostics(s)["passes_prefilter"] or args.no_prefilter):
                continue
            structures.append(s); idx.append(i)
        results = evaluate_structures_dpa(structures, dp)
        n_ok = 0
        for s, i, r in zip(structures, idx, results):
            struct_rows.append({"sample": i, "step": int(step), "t": t, "evaluated": r.ok, "error": r.error,
                                "energy_per_atom_dpa": (r.energy / len(s)) if r.ok else None})
            if not r.ok:
                continue
            n_ok += 1
            for a in range(len(s)):
                atom_rows.append({"sample": i, "step": int(step), "t": t, "atom": a, "Z": s.species[a].Z,
                                  "fx_dpa": r.forces[a, 0], "fy_dpa": r.forces[a, 1], "fz_dpa": r.forces[a, 2],
                                  "force_norm_dpa": float(np.linalg.norm(r.forces[a])), "e_atom_dpa": float(r.atomic_energies[a])})
        print(f"step {int(step):5d} (t={t:.2f}): {len(structures)} structures, {n_ok} evaluated  [{time.time() - t0:.0f}s]")

    pd.DataFrame(atom_rows).to_parquet(args.out_dir / "dpa_atoms.parquet", index=False)
    pd.DataFrame(struct_rows).to_parquet(args.out_dir / "dpa_structures.parquet", index=False)
    meta = {"argv": sys.argv, "traj_npz": str(args.traj_npz.resolve()), "steps": steps.tolist(), "model": args.model,
            "head": args.head, "type_map_size": len(dp.get_type_map()), "atom_types_from": args.atom_types_from,
            "n_samples": n_samples, "n_atom_rows": len(atom_rows), "elapsed_s": round(time.time() - t0, 1)}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
