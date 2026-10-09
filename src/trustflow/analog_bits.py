"""Decode FlowMM analog-bit atom types without importing flowmm.

FlowMM encodes atomic number z (0..127) as 7 analog bits in {-b, +b}
(``src/flowmm/rfm/manifolds/analog_bits.py``). During sampling the bits are
continuous; decoding takes the sign of each bit, bit i carrying weight 2**i.
"""

from __future__ import annotations

import numpy as np

NUM_ATOMIC_BITS = 7
MAX_CHGNET_Z = 94  # CHGNet AtomEmbedding(max_num_elements=94)


def analog_bits_to_int(bits: np.ndarray) -> np.ndarray:
    """(..., 7) analog bits -> (...) integer atomic numbers (sign rule)."""
    bits = np.asarray(bits)
    if bits.shape[-1] != NUM_ATOMIC_BITS:
        raise ValueError(f"last dim must be {NUM_ATOMIC_BITS}, got {bits.shape}")
    b = (np.sign(bits) + 1) / 2  # {0, 1}; sign(0) -> 0.5 -> treated as 0 below
    b = (b > 0.5).astype(np.int64)
    weights = 2 ** np.arange(NUM_ATOMIC_BITS, dtype=np.int64)
    return (b * weights).sum(axis=-1)


def int_to_analog_bits(z: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Inverse of :func:`analog_bits_to_int` for tests. (...) -> (..., 7)."""
    z = np.asarray(z, dtype=np.int64)
    shifted = z[..., None] >> np.arange(NUM_ATOMIC_BITS, dtype=np.int64)
    bits = shifted % 2
    return (bits * 2 - 1).astype(np.float64) * scale


def is_valid_z(z: np.ndarray, z_max: int = MAX_CHGNET_Z) -> np.ndarray:
    """Element-wise mask: 1 <= z <= z_max."""
    z = np.asarray(z)
    return (z >= 1) & (z <= z_max)
