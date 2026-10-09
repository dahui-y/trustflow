"""Convert a FlowMM ``consolidated_gen_trajectory.pt`` into a plain-numpy npz.

Run inside the ``flowmm`` environment (the .pt pickles torch_geometric Batch
objects). The npz is the hand-off format to the MLIP environment; its layout is
documented in ``src/trustflow/trajectory.py``.

    python scripts_analysis/export_trajectory_npz.py \
        <ckpt_dir>/m0_audit/consolidated_gen_trajectory.pt \
        outputs/m0/traj.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("consolidated_pt", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--eval_index", type=int, default=0, help="which gen_trajectory_?? run to export")
    args = ap.parse_args()

    d = torch.load(args.consolidated_pt, map_location="cpu")
    k = args.eval_index
    frac = d["frac_coords"][k].numpy()  # (T+1, N, 3)
    latt = d["lattices"][k].numpy()  # (T+1, B, 3, 3)
    atom_types = d["atom_types"][k].numpy()  # (T+1, N, 7) analog bits or (T+1, N)
    num_atoms = d["num_atoms"][k].numpy().reshape(-1)
    num_steps_txt = args.consolidated_pt.parent / "num_steps.txt"
    num_steps = int(num_steps_txt.read_text()) if num_steps_txt.exists() else frac.shape[0] - 1
    assert frac.shape[0] == num_steps + 1, (frac.shape, num_steps)
    assert num_atoms.sum() == frac.shape[1], (num_atoms.sum(), frac.shape)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        frac_coords=frac.astype(np.float32),
        lattices=latt.astype(np.float32),
        atom_types=atom_types.astype(np.float32) if atom_types.ndim == 3 else atom_types.astype(np.int64),
        num_atoms=num_atoms.astype(np.int64),
        num_steps=np.int64(num_steps),
    )
    meta = {
        "source": str(args.consolidated_pt.resolve()),
        "eval_index": k,
        "num_samples": int(len(num_atoms)),
        "num_atoms_total": int(num_atoms.sum()),
        "num_steps": num_steps,
        "atom_types_encoding": "analog_bits" if atom_types.ndim == 3 else "int",
    }
    args.out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
