"""Multiword Fock lift: exact agreement with the int64 path, and beyond-62-mode physics."""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.fock.multiword import MWSignedBlockCycle, pack_bits, propagator_mw
from pca3d.fock.signed import SignedBlockCycle
from pca3d.models import conditional as C


def swap_perm():
    p = np.empty(16, dtype=np.int64)
    for c in range(16):
        sw = lambda v: ((v & 1) << 1) | ((v >> 1) & 1)
        p[c] = C.compose(sw(C.system_state(c)), sw(C.env_state(c)))
    return p


@pytest.mark.parametrize("rule_idx", [891, 261])
def test_multiword_matches_int64_bitwise(rule_idx):
    """Same configs, same substeps: configs AND signs must agree exactly."""
    perm = C.enumerate_conditional_rules()[rule_idx]
    m64 = SignedBlockCycle(n_sites=30, block_perm=perm, boundary="open")
    mmw = MWSignedBlockCycle(n_sites=30, block_perm=perm)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(300, 60)).astype(bool)
    c64 = np.zeros(300, dtype=np.int64)
    for m in range(60):
        c64 |= bits[:, m].astype(np.int64) << m
    cmw = pack_bits(bits)
    s64 = np.ones(300, dtype=np.int64)
    smw = s64.copy()
    for t in range(8):
        c64, s64 = m64.substep(c64, s64, t % 2)
        cmw, smw = mmw.substep(cmw, smw, t % 2)
    assert np.array_equal(c64, cmw[:, 0].astype(np.int64))
    assert np.array_equal(s64, smw)


def test_free_streaming_coherent_beyond_62_modes():
    """L=100 -> 200 modes: the exact massless propagator, impossible on the old path."""
    m = MWSignedBlockCycle(n_sites=100, block_perm=swap_perm())
    g, surv = propagator_mw(m, n_substeps=10, y_site=30, ensemble=800, seed=1)
    for t in range(11):
        row = g[t, :, 0]
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        assert list(nz) == [30 + t]
        assert abs(row[30 + t]) == pytest.approx(surv[t], abs=1e-12)
        assert surv[t] == pytest.approx(surv[0], abs=1e-12)
