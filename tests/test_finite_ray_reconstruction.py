"""Finite-ray theorem, closed against the instrument (review F17).

Three checks the original suite lacked:
  1. The Bloch operator of a deterministic legal rule is RECONSTRUCTED from
     real-space propagation (not constructed to be monomial) and is monomial
     with exactly linear eigenphases.
  2. The coincube itself, at FIXED media, is a signed permutation of the
     (site, channel) basis per cycle -- the theorem's unique-jump hypothesis
     holds per realization (sign structure included).
  3. The media-ensemble average (the dressed operator actually measured) is
     NOT monomial: the documented evasion of the finite-ray theorem is the
     ensemble average over a non-empty Bernoulli vacuum, not a loophole in
     the theorem.
"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pca3d.models.coincube import (COIN_D, PERMS, SIGNS, annealed_u,
                                   layer_env)

L = 4


def ballistic_propagate(g):
    """One cycle of the coinless (ballistic) walk: channel c translates by
    COIN_D[a][c] along each axis a, twice per axis. g: (4, L, L, L)."""
    out = g.copy()
    for a in range(3):
        for _ in range(2):
            for c in range(4):
                out[c] = np.roll(out[c], int(COIN_D[a][c]), axis=a)
    return out


def test_bloch_reconstructed_from_propagation_is_monomial_linear():
    # reconstruct U(k) by Fourier-contracting the propagated basis fields
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = rng.uniform(-np.pi, np.pi, 3)
        # lattice-commensurate momenta so the contraction is exact
        k = 2 * np.pi * np.round(k * L / (2 * np.pi)) / L
        U = np.zeros((4, 4), dtype=complex)
        x = np.arange(L)
        px, py, pz = (np.exp(-1j * k[a] * x) for a in range(3))
        for c0 in range(4):
            g = np.zeros((4, L, L, L))
            g[c0, 0, 0, 0] = 1.0
            g1 = ballistic_propagate(g)
            U[:, c0] = np.einsum("cxyz,x,y,z->c", g1, px, py, pz)
        # monomial: exactly one unit-modulus entry per row and column
        mags = np.abs(U)
        assert np.allclose(np.sort(mags, axis=0)[-1], 1, atol=1e-12)
        assert np.allclose(np.sort(mags, axis=0)[:-1], 0, atol=1e-12)
        assert np.allclose(np.sort(mags, axis=1)[:, :-1], 0, atol=1e-12)
        # exact linearity: eigenphases double when k doubles (mod 2pi)
        U2 = np.zeros((4, 4), dtype=complex)
        px2, py2, pz2 = (np.exp(-2j * k[a] * x) for a in range(3))
        for c0 in range(4):
            g = np.zeros((4, L, L, L))
            g[c0, 0, 0, 0] = 1.0
            g1 = ballistic_propagate(g)
            U2[:, c0] = np.einsum("cxyz,x,y,z->c", g1, px2, py2, pz2)
        ph1 = np.sort(np.angle(np.linalg.eigvals(U)) % (2 * np.pi))
        ph2 = np.angle(np.linalg.eigvals(U2)) % (2 * np.pi)
        for p in ph1:
            assert np.min(np.abs(np.exp(2j * p) - np.exp(1j * ph2))) < 1e-9


def coincube_fixed_media_cycle(env):
    """Full one-cycle single-particle matrix on the (channel, site) basis
    for a FIXED environment realization; env is consumed (streamed)."""
    dim = 4 * L ** 3
    M = np.zeros((dim, dim))

    def idx(c, x, y, z):
        return ((c * L + x) * L + y) * L + z

    # basis-vector propagation through the exact signed layers
    for c0 in range(4):
        for x0 in range(L):
            for y0 in range(L):
                for z0 in range(L):
                    e = [env[a].copy() for a in range(3)]
                    c, pos, sg = c0, [x0, y0, z0], 1.0
                    for a in range(3):
                        for o in (0, 1):
                            if e[a][tuple(pos)]:
                                sg *= SIGNS[a][c]
                                c = int(PERMS[a][c])
                            pos[a] = (pos[a] + int(COIN_D[a][c])) % L
                            e[a] = layer_env(e[a], a, o)
                    M[idx(c, *pos), idx(c0, x0, y0, z0)] = sg
    return M


def test_fixed_media_cycle_is_signed_permutation():
    rng = np.random.default_rng(1)
    env = [(rng.random((L, L, L)) < 0.3) for _ in range(3)]
    M = coincube_fixed_media_cycle(env)
    a = np.abs(M)
    assert np.allclose(a.sum(axis=0), 1) and np.allclose(a.sum(axis=1), 1)
    assert np.allclose(a[a > 0], 1)          # entries are +-1: signed perm


def test_ensemble_average_is_not_monomial():
    # the measured (dressed) operator: ensemble average -> NOT monomial,
    # which is exactly why the finite-ray theorem does not constrain it
    U = annealed_u(np.array([0.3, 0.7, 0.1]), 0.08)
    nonzero_per_row = (np.abs(U) > 1e-9).sum(axis=1)
    assert nonzero_per_row.max() >= 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
