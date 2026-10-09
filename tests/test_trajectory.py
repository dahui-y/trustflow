"""Unit tests for trustflow.trajectory and trustflow.analog_bits (no torch / chgnet)."""

import numpy as np
import pytest

from trustflow.analog_bits import analog_bits_to_int, int_to_analog_bits, is_valid_z
from trustflow.trajectory import TrajectoryBundle, per_atom_min_distance, structure_diagnostics


def test_analog_bits_roundtrip():
    z = np.arange(0, 128)
    assert np.array_equal(analog_bits_to_int(int_to_analog_bits(z)), z)
    # continuous bits decode by sign, independent of magnitude
    noisy = int_to_analog_bits(np.array([26, 8])) * np.array([0.1, 3.0])[:, None]
    assert np.array_equal(analog_bits_to_int(noisy), [26, 8])


def test_is_valid_z():
    assert np.array_equal(is_valid_z(np.array([0, 1, 94, 95, 127])), [False, True, True, False, False])


def _toy_bundle(T: int = 10, encode_bits: bool = True) -> TrajectoryBundle:
    """Two crystals (2 and 3 atoms). Atoms start spread out and converge to a NaCl-like cell."""
    rng = np.random.default_rng(0)
    num_atoms = np.array([2, 3])
    N = num_atoms.sum()
    final_z = np.array([11, 17, 6, 6, 6])
    frac = np.zeros((T + 1, N, 3))
    frac[-1, :2] = [[0, 0, 0], [0.5, 0.5, 0.5]]
    frac[-1, 2:] = [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25]]
    frac[0] = rng.uniform(size=(N, 3))
    for s in range(1, T):
        frac[s] = (frac[0] + (frac[-1] - frac[0]) * s / T) % 1.0
    latt = np.zeros((T + 1, 2, 3, 3))
    for s in range(T + 1):
        a = 1.0 + 4.6 * s / T  # tiny cell at t=0 -> atoms overlap
        latt[s] = np.eye(3) * a
    if encode_bits:
        z_traj = np.tile(final_z, (T + 1, 1))
        z_traj[0] = [100, 17, 6, 6, 6]  # garbage type at t=0
        atom_types = int_to_analog_bits(z_traj)
    else:
        atom_types = np.tile(final_z, (T + 1, 1))
    return TrajectoryBundle.from_arrays(frac, latt, atom_types, num_atoms, T)


@pytest.mark.parametrize("encode_bits", [True, False])
def test_bundle_shapes_and_offsets(encode_bits):
    b = _toy_bundle(encode_bits=encode_bits)
    assert b.num_samples == 2
    assert b.offsets.tolist() == [0, 2, 5]
    assert b.atom_types_int.shape == (11, 5)
    assert b.time_of_step(0) == 0.0 and b.time_of_step(10) == 1.0


def test_select_steps_includes_endpoints():
    b = _toy_bundle(T=1000)
    steps = b.select_steps(11)
    assert steps[0] == 0 and steps[-1] == 1000 and len(steps) == 11
    assert np.all(np.diff(steps) > 0)


def test_atom_types_final_vs_step():
    b = _toy_bundle()
    assert b.atom_types_for(0, 0, "final").tolist() == [11, 17]
    assert b.atom_types_for(0, 0, "step").tolist() == [100, 17]
    assert b.structure_at(0, 0, "step") is None  # Z=100 invalid for CHGNet
    s = b.structure_at(0, 0, "final")
    assert s is not None and len(s) == 2


def test_structure_geometry_matches_arrays():
    b = _toy_bundle()
    s = b.structure_at(1, 10)
    assert s is not None
    np.testing.assert_allclose(s.lattice.matrix, np.eye(3) * 5.6)
    np.testing.assert_allclose(s.frac_coords, b.frac_coords[10, 2:5])
    assert [sp.Z for sp in s.species] == [6, 6, 6]


def test_diagnostics_prefilter_rejects_overlap_and_accepts_final():
    b = _toy_bundle()
    early = structure_diagnostics(b.structure_at(0, 0))
    late = structure_diagnostics(b.structure_at(0, 10))
    assert late["passes_prefilter"]
    assert late["min_pair_distance"] == pytest.approx(5.6 * np.sqrt(3) / 2)
    assert late["volume_per_atom"] == pytest.approx(5.6**3 / 2)
    assert early["volume_per_atom"] == pytest.approx(0.5)
    assert early["min_pair_distance"] < 1.0 or not early["passes_prefilter"]


def test_per_atom_min_distance_pbc():
    b = _toy_bundle()
    s = b.structure_at(1, 10)
    d = per_atom_min_distance(s)
    assert d.shape == (3,)
    # atom at (0.25,0.25,0.25) is equidistant from (0,0,0) and (0.5,0.5,0)
    assert d[2] == pytest.approx(min(s.get_distance(2, 0), s.get_distance(2, 1)))


def test_npz_roundtrip(tmp_path):
    b = _toy_bundle()
    p = tmp_path / "traj.npz"
    b.save_npz(p)
    b2 = TrajectoryBundle.load_npz(p)
    np.testing.assert_allclose(b2.frac_coords, b.frac_coords, atol=1e-6)
    assert np.array_equal(b2.atom_types_int, b.atom_types_int)
    assert b2.num_steps == b.num_steps


def test_inconsistent_num_atoms_raises():
    b = _toy_bundle()
    with pytest.raises(ValueError):
        TrajectoryBundle.from_arrays(b.frac_coords, b.lattices, b.atom_types_int, np.array([2, 2]), b.num_steps)
