"""Per-atom DPA (DeePMD-kit) evaluation of pymatgen Structures.

Used as the *independent reference* in Experiment B: ``D_F = |F_CHGNet - F_DPA|``.
deepmd is imported lazily; this module needs only numpy + pymatgen otherwise.

Pretrained multi-task DPA models require a ``head``; the MPtrj head is the fair
comparison to CHGNet 0.3.0 (same training distribution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Structure

DEFAULT_MODEL = "DPA-3.1-3M"
DEFAULT_HEAD = "MP_traj_v024_alldata_mixu"


@dataclass
class DPAResult:
    energy: float | None  # eV, total
    forces: np.ndarray | None  # (N, 3) eV/A
    atomic_energies: np.ndarray | None  # (N,) eV
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def load_dpa(model: str = DEFAULT_MODEL, head: str | None = DEFAULT_HEAD) -> Any:
    """``DeepPot`` for a registry name (auto-download) or a local .pt/.pth path."""
    from deepmd.infer import DeepPot

    try:
        return DeepPot(model, head=head)
    except AssertionError as exc:  # head missing / misspelled: surface the available heads
        raise SystemExit(f"DPA head selection failed for head={head!r}: {exc}") from exc


def type_index_map(dp: Any) -> dict[str, int]:
    return {sym: i for i, sym in enumerate(dp.get_type_map())}


def evaluate_structures_dpa(structures: Sequence[Structure], dp: Any) -> list[DPAResult]:
    """One frame per call (natoms varies across crystals). Elements outside the
    model's type_map give an error entry instead of raising."""
    tmap = type_index_map(dp)
    out: list[DPAResult] = []
    for s in structures:
        syms = [sp.symbol for sp in s.species]
        missing = sorted({x for x in syms if x not in tmap})
        if missing:
            out.append(DPAResult(None, None, None, error=f"elements not in type_map: {missing}"))
            continue
        coords = np.asarray(s.cart_coords, dtype=np.float64).reshape(1, -1, 3)
        cell = np.asarray(s.lattice.matrix, dtype=np.float64).reshape(1, 9)
        atype = np.array([tmap[x] for x in syms], dtype=np.int64)
        try:
            e, f, _v, ae, _av = dp.eval(coords, cell, atype, atomic=True)
        except Exception as exc:  # noqa: BLE001
            out.append(DPAResult(None, None, None, error=f"eval: {type(exc).__name__}: {exc}"))
            continue
        f = np.asarray(f, dtype=np.float64).reshape(-1, 3)
        out.append(DPAResult(
            energy=float(np.asarray(e).reshape(-1)[0]),
            forces=f,
            atomic_energies=np.asarray(ae, dtype=np.float64).reshape(-1),
            error=None if np.isfinite(f).all() else "non-finite forces",
        ))
    return out
