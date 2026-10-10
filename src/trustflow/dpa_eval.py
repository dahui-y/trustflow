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


def load_dpa(
    model: str = DEFAULT_MODEL,
    head: str | None = DEFAULT_HEAD,
    device: str | None = None,
    nlist_backend: str = "native",
) -> Any:
    """``DeepPot`` for a registry name (auto-download) or a local .pt/.pth path.

    - ``device="cpu"`` must be decided before deepmd is imported (it reads ``DEVICE``).
    - TorchScript GPU operator fusion is disabled: it JIT-compiles kernels through
      nvrtc, which is absent in pip-installed CUDA-13 torch builds.
    - ``nlist_backend="native"`` avoids the fixed-capacity vesin CUDA neighbor list,
      which overflows on dense early-timestep structures.
    """
    import os

    if device == "cpu":
        os.environ["DEVICE"] = "cpu"
    import torch

    for fn in ("_jit_override_can_fuse_on_gpu", "_jit_set_texpr_fuser_enabled", "_jit_set_nvfuser_enabled"):
        try:
            getattr(torch._C, fn)(False)
        except Exception:  # noqa: BLE001 - not every torch build has every switch
            pass
    from deepmd.infer import DeepPot

    try:
        return DeepPot(model, head=head, nlist_backend=nlist_backend)
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
            res = dp.eval(coords, cell, atype, atomic=True)
            e, f = res[0], res[1]
            ae = res[3] if len(res) > 3 else np.full(len(s), np.nan)
            f = np.asarray(f, dtype=np.float64).reshape(-1, 3)
            ae = np.asarray(ae, dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            out.append(DPAResult(None, None, None, error=f"eval: {type(exc).__name__}: {str(exc)[:300]}"))
            continue
        out.append(DPAResult(
            energy=float(np.asarray(e).reshape(-1)[0]),
            forces=f,
            atomic_energies=ae,
            error=None if np.isfinite(f).all() else "non-finite forces",
        ))
    return out
