"""Per-atom CHGNet evaluation of a list of pymatgen Structures.

Wraps ``CHGNet.predict_graph`` with the flags needed for Milestone 0:
energy, forces, per-site energies and the 64-d atom features that UQ-MLIP's
quantile-GBM consumes. chgnet is imported lazily so that the rest of
``trustflow`` stays importable in environments without it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Structure


@dataclass
class AtomwiseResult:
    """Per-structure CHGNet outputs. ``None`` fields mean the structure failed."""

    energy_per_atom: float | None  # eV/atom
    forces: np.ndarray | None  # (N, 3) eV/A
    site_energies: np.ndarray | None  # (N,) eV
    atom_fea: np.ndarray | None  # (N, 64)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def load_chgnet(model_name: str = "0.3.0", device: str | None = None, on_isolated_atoms: str = "warn") -> Any:
    """Load CHGNet and relax the isolated-atom policy (default 'error' would raise on
    early-timestep structures with large cells)."""
    from chgnet.model.model import CHGNet

    model = CHGNet.load(model_name=model_name, use_device=device, verbose=False)
    model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
    return model


def evaluate_structures(
    structures: Sequence[Structure],
    model: Any = None,
    batch_size: int = 32,
) -> list[AtomwiseResult]:
    """Evaluate structures one graph at a time for conversion, batched for the forward pass.

    Structures whose graph conversion fails (e.g. bond-graph RuntimeError) get an
    ``AtomwiseResult`` with ``error`` set instead of aborting the batch.
    """
    if model is None:
        model = load_chgnet()

    graphs, idx_ok, results = [], [], [None] * len(structures)
    for i, s in enumerate(structures):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graphs.append(model.graph_converter(s))
            idx_ok.append(i)
        except Exception as exc:  # noqa: BLE001 - we want to record any conversion failure
            results[i] = AtomwiseResult(None, None, None, None, error=f"graph: {type(exc).__name__}: {exc}")

    if graphs:
        preds = model.predict_graph(
            graphs,
            task="ef",
            return_site_energies=True,
            return_atom_feas=True,
            batch_size=batch_size,
        )
        if isinstance(preds, dict):  # single graph returns a bare dict
            preds = [preds]
        for i, p in zip(idx_ok, preds):
            f = np.asarray(p["f"], dtype=np.float64)
            bad = not np.isfinite(f).all()
            results[i] = AtomwiseResult(
                energy_per_atom=float(p["e"]),
                forces=f,
                site_energies=np.asarray(p["site_energies"], dtype=np.float64).reshape(-1),
                atom_fea=np.asarray(p["atom_fea"], dtype=np.float32),
                error="non-finite forces" if bad else None,
            )
    return results
