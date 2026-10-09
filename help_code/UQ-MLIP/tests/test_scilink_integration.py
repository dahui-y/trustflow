"""Tests for the SciLink / orchestrator tool plug-in.

These exercise the parts of the plug-in that do not require a heavy MLIP
backend: spec/schema shape, spec-to-callable pairing (the contract SciLink's
registry relies on), and the end-to-end extract → train → evaluate flow with a
stub extractor so no MACE/UMA/CHGNet install is needed.
"""

import numpy as np
import pytest

from uq_mlip.data import EmbeddingData
from uq_mlip.integrations import scilink


def test_specs_and_callables_pair_by_name():
    # Registry contract: every spec.name resolves to a module-level callable.
    names = {spec.name for spec in scilink.TOOL_SPECS}
    assert names == {
        "uq_extract_embeddings",
        "uq_train_model",
        "uq_evaluate_uncertainty",
    }
    for spec in scilink.TOOL_SPECS:
        assert callable(getattr(scilink, spec.name))
        assert callable(scilink.get_tool_function(spec.name))


def test_openai_schemas_have_required_fields():
    schemas = scilink.openai_tool_schemas()
    assert len(schemas) == len(scilink.TOOL_SPECS)
    by_name = {s["function"]["name"]: s for s in schemas}
    req = by_name["uq_evaluate_uncertainty"]["function"]["parameters"]["required"]
    assert req == ["backend", "uq_model", "structures"]


def test_unknown_tool_raises():
    with pytest.raises(LookupError):
        scilink.get_tool_function("does_not_exist")


class _StubExtractor:
    """Deterministic embeddings so train/evaluate run without an MLIP."""

    def extract(self, atoms_list):
        feats, energies, node_type, num_atoms = [], [], [], []
        for a in atoms_list:
            n = len(a)
            feats.append(np.random.RandomState(n).rand(n, 8))
            energies.append(np.random.RandomState(n + 1).rand(n))
            node_type.append(a.get_atomic_numbers())
            num_atoms.append(n)
        return EmbeddingData(
            node_feats=np.vstack(feats),
            node_type=np.concatenate(node_type),
            num_atoms=np.asarray(num_atoms),
            node_energies=np.concatenate(energies),
        )


def test_extract_train_evaluate_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    from ase import Atoms
    from ase.io import write

    monkeypatch.setattr(scilink, "get_extractor", lambda backend, **k: _StubExtractor())

    frames = [
        Atoms("H2O", positions=np.random.rand(3, 3)),
        Atoms("H2", positions=np.random.rand(2, 3)),
        Atoms("CO2", positions=np.random.rand(3, 3)),
    ]
    xyz = tmp_path / "val.xyz"
    write(str(xyz), frames)

    # Train directly from a stub-extracted bundle (extract_file needs the
    # real extractor's file plumbing; the tool path is covered by evaluate).
    bundle = _StubExtractor().extract(frames)
    from uq_mlip.data import save_embeddings

    npz = save_embeddings(bundle, tmp_path / "embedding_info_val.npz")
    model_path = scilink.uq_train_model(str(npz), str(tmp_path / "uqm"))
    assert model_path.endswith(".pkl")

    report = scilink.uq_evaluate_uncertainty(
        backend="mace",
        uq_model=str(tmp_path / "uqm"),
        structures=str(xyz),
        model_size="medium-0b",
    )
    assert report["method"] == "uq-mlip-quantile-gbm"
    assert len(report["per_structure"]) == 3
    for ps, expected_n in zip(report["per_structure"], [3, 2, 3]):
        assert ps["n_atoms"] == expected_n
        assert len(ps["per_atom_uncertainty"]) == expected_n
        assert len(ps["elements"]) == expected_n
    assert set(report).issuperset(
        {"n_extrapolating", "extrapolation_indices", "mean_energy_uncertainty"}
    )
