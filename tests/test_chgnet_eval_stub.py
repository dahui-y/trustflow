"""Exercise trustflow.chgnet_eval with a stub model that mimics CHGNet's interface."""

import numpy as np
from pymatgen.core import Lattice, Structure

from trustflow.chgnet_eval import evaluate_structures


class _Converter:
    def __call__(self, structure):
        if len(structure) == 1:
            raise RuntimeError("bond graph failed")
        return structure  # pass the structure through as the "graph"


class _StubModel:
    graph_converter = _Converter()

    def predict_graph(self, graphs, task, return_site_energies, return_atom_feas, batch_size):
        assert task == "ef" and return_site_energies and return_atom_feas
        out = []
        for g in graphs:
            n = len(g)
            out.append({
                "e": -1.0 * n,
                "f": np.ones((n, 3)) * 0.1,
                "site_energies": np.full(n, -1.0),
                "atom_fea": np.zeros((n, 64), dtype=np.float32),
            })
        return out[0] if len(out) == 1 else out


def _s(n):
    return Structure(Lattice.cubic(5.0), ["C"] * n, np.random.default_rng(n).uniform(size=(n, 3)))


def test_evaluate_structures_handles_single_and_failed():
    res = evaluate_structures([_s(1), _s(3)], model=_StubModel())
    assert not res[0].ok and res[0].error.startswith("graph:")
    assert res[1].ok and res[1].forces.shape == (3, 3) and res[1].atom_fea.shape == (3, 64)
    assert res[1].site_energies.shape == (3,)


def test_evaluate_structures_single_graph_bare_dict():
    res = evaluate_structures([_s(2)], model=_StubModel())
    assert len(res) == 1 and res[0].ok and res[0].energy_per_atom == -2.0


def test_evaluate_structures_empty():
    assert evaluate_structures([], model=_StubModel()) == []
