"""Load exported FlowMM generation trajectories and build intermediate crystals.

Data contract (npz written by ``scripts_analysis/export_trajectory_npz.py``):

    frac_coords   float32 (T+1, N_total, 3)   fractional, on the torus [0, 1)
    lattices      float32 (T+1, B, 3, 3)      rows are lattice vectors; cart = frac @ L
    atom_types    float32 (T+1, N_total, 7)   analog bits   -- OR --
                  int64   (T+1, N_total)      already-decoded atomic numbers
    num_atoms     int64   (B,)
    num_steps     int64   scalar              T (so T+1 saved states, t = step / T)

Crystal ``i`` owns atoms ``offsets[i]:offsets[i+1]`` with
``offsets = concatenate([[0], cumsum(num_atoms)])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pymatgen.core import Lattice, Structure

from trustflow.analog_bits import NUM_ATOMIC_BITS, analog_bits_to_int, is_valid_z

AtomTypesFrom = Literal["final", "step"]


@dataclass
class TrajectoryBundle:
    frac_coords: np.ndarray  # (T+1, N_total, 3)
    lattices: np.ndarray  # (T+1, B, 3, 3)
    atom_types_int: np.ndarray  # (T+1, N_total) decoded atomic numbers
    num_atoms: np.ndarray  # (B,)
    num_steps: int

    def __post_init__(self) -> None:
        T1, N, three = self.frac_coords.shape
        if three != 3:
            raise ValueError(f"frac_coords last dim must be 3, got {self.frac_coords.shape}")
        if self.lattices.shape[0] != T1 or self.lattices.shape[2:] != (3, 3):
            raise ValueError(f"lattices shape {self.lattices.shape} inconsistent with frac_coords {self.frac_coords.shape}")
        if self.atom_types_int.shape != (T1, N):
            raise ValueError(f"atom_types_int shape {self.atom_types_int.shape} != {(T1, N)}")
        if int(self.num_atoms.sum()) != N:
            raise ValueError(f"num_atoms sums to {int(self.num_atoms.sum())} but N_total={N}")
        if self.lattices.shape[1] != len(self.num_atoms):
            raise ValueError("lattices batch dim must equal len(num_atoms)")
        if T1 != self.num_steps + 1:
            raise ValueError(f"expected num_steps+1={self.num_steps + 1} saved states, got {T1}")
        self.offsets = np.concatenate([[0], np.cumsum(self.num_atoms)]).astype(np.int64)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_arrays(
        cls,
        frac_coords: np.ndarray,
        lattices: np.ndarray,
        atom_types: np.ndarray,
        num_atoms: np.ndarray,
        num_steps: int,
    ) -> "TrajectoryBundle":
        atom_types = np.asarray(atom_types)
        if atom_types.ndim == 3 and atom_types.shape[-1] == NUM_ATOMIC_BITS:
            atom_types_int = analog_bits_to_int(atom_types)
        elif atom_types.ndim == 2:
            atom_types_int = atom_types.astype(np.int64)
        else:
            raise ValueError(f"unrecognized atom_types shape {atom_types.shape}")
        return cls(
            frac_coords=np.asarray(frac_coords, dtype=np.float64),
            lattices=np.asarray(lattices, dtype=np.float64),
            atom_types_int=atom_types_int,
            num_atoms=np.asarray(num_atoms, dtype=np.int64).reshape(-1),
            num_steps=int(num_steps),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "TrajectoryBundle":
        with np.load(path) as d:
            return cls.from_arrays(
                d["frac_coords"], d["lattices"], d["atom_types"], d["num_atoms"], int(d["num_steps"])
            )

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            frac_coords=self.frac_coords.astype(np.float32),
            lattices=self.lattices.astype(np.float32),
            atom_types=self.atom_types_int,
            num_atoms=self.num_atoms,
            num_steps=np.int64(self.num_steps),
        )

    # ------------------------------------------------------------ queries
    @property
    def num_samples(self) -> int:
        return len(self.num_atoms)

    def time_of_step(self, step: int) -> float:
        """Flow time t in [0, 1] of saved state ``step`` (0 = noise, T = final)."""
        return step / self.num_steps

    def select_steps(self, n_select: int) -> np.ndarray:
        """``n_select`` step indices evenly spaced in [0, T], always including both ends."""
        if n_select < 2:
            raise ValueError("n_select must be >= 2 to include both endpoints")
        return np.unique(np.round(np.linspace(0, self.num_steps, n_select)).astype(np.int64))

    def atom_slice(self, sample: int) -> slice:
        return slice(int(self.offsets[sample]), int(self.offsets[sample + 1]))

    def atom_types_for(self, sample: int, step: int, atom_types_from: AtomTypesFrom = "final") -> np.ndarray:
        """Atomic numbers of crystal ``sample`` at ``step``.

        ``"final"``: types decoded from the last saved state (default; isolates
        geometric OOD-ness). ``"step"``: types decoded from the state at ``step``.
        """
        src_step = self.num_steps if atom_types_from == "final" else step
        return self.atom_types_int[src_step, self.atom_slice(sample)]

    def structure_at(
        self, sample: int, step: int, atom_types_from: AtomTypesFrom = "final"
    ) -> Structure | None:
        """pymatgen Structure for crystal ``sample`` at ``step``; None if any Z invalid."""
        z = self.atom_types_for(sample, step, atom_types_from)
        if not is_valid_z(z).all():
            return None
        sl = self.atom_slice(sample)
        lattice = Lattice(self.lattices[step, sample])
        return Structure(lattice, species=z.tolist(), coords=self.frac_coords[step, sl], coords_are_cartesian=False)


# ------------------------------------------------------------ diagnostics
def structure_diagnostics(structure: Structure) -> dict[str, float | bool]:
    """Cheap geometric sanity numbers used to pre-filter MLIP input.

    Thresholds follow DiffCSP ``structure_validity``: min pair distance >= 0.5 A and
    volume >= 0.1 A^3.
    """
    n = len(structure)
    dm = structure.distance_matrix
    if n > 1:
        off = dm + np.eye(n) * 1e6
        min_dist = float(off.min())
    else:
        min_dist = float("inf")
    vol = float(structure.volume)
    return {
        "min_pair_distance": min_dist,
        "volume_per_atom": vol / n,
        "passes_prefilter": bool(min_dist >= 0.5 and vol >= 0.1),
    }


def per_atom_min_distance(structure: Structure) -> np.ndarray:
    """(N,) nearest-neighbour distance per atom under periodic boundary conditions."""
    n = len(structure)
    if n == 1:
        return np.array([float("inf")])
    dm = structure.distance_matrix + np.eye(n) * 1e6
    return dm.min(axis=1)
