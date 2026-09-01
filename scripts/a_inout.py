#!/usr/bin/env python
"""Route A: the in-out (two-boundary) propagator is relatively unitary.

The damped propagator is the IN-IN object (bra = evolved vacuum): per
conversion event the 1p transfer is the convex mixture (1-q) + qC (phase ~ q,
decoherence ~ 2q(1-q), Q pinned O(1)). Choosing a UNIFORM final env boundary
term (Wetterich's transfer-matrix formalism is a two-boundary problem — the
final boundary factor is ours to choose) replaces probabilities by
amplitudes: per fresh read the transfer is proportional to

    O_a(k) = E_a(k) (c + s C_a),   c = sqrt(1-q)/D, s = sqrt(q)/D,
                                   D = sqrt(1-q) + sqrt(q)

and (sqrt(1-q) + sqrt(q) C) is EXACTLY orthogonal (C^2 = -1): all relative
damping vanishes — the quasiparticle evolution is unitary up to one global
scalar per time step (wave-function renormalisation), with omega0 ~ sqrt(q).

Checks:
 1. 3D annealed: orthogonality exact; the cone at the corners: degeneracy,
    slope isotropy (100/110/111 + random), omega0(q), all moduli EQUAL.
 2. Substrate (15-mode 1D coincube, exact Fock): the measured in-out
    correlator G_io(t)/Z(t) vs the fresh-read transfer prediction; t = 1
    comparison and the re-read (1D companion) deviations at t >= 2,
    with the in-in moduli spread as the comparison baseline.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from w3c_composite_check import (DIM, LX, NCAR, NM, full_cycle,  # noqa: E402
                                 single_particle_reference)

from pca3d.models.coincube import COIN_C, COIN_D, annealed_u  # noqa: E402

Q = 0.08


# -- 1: 3D annealed orthogonal cone -------------------------------------------

def o_cycle(kvec, q):
    c, s = np.sqrt(1 - q), np.sqrt(q)
    u = np.eye(4, dtype=complex)
    for a in range(3):
        t = np.diag(np.exp(1j * kvec[a] * COIN_D[a])) @ (
            c * np.eye(4) + s * COIN_C[a])
        u = t @ t @ u
    return u


def check_3d():
    c, s = np.sqrt(1 - Q), np.sqrt(Q)
    for a in range(3):
        m = c * np.eye(4) + s * COIN_C[a]
        assert np.abs(m @ m.T - np.eye(4)).max() < 1e-14
    print("1. (sqrt(1-q) + sqrt(q) C) exactly orthogonal  [OK]")
    U0 = o_cycle(np.zeros(3), Q)
    w = np.linalg.eigvals(U0)
    assert np.abs(np.abs(w) - 1).max() < 1e-12
    up = sorted([l for l in w if l.imag > 0], key=lambda l: -np.angle(l))
    lam0 = np.mean(up)
    gap = max(abs(l - lam0) for l in up)
    print(f"   node: |lam| = 1 (max dev {np.abs(np.abs(w) - 1).max():.1e}), "
          f"omega0 = {np.angle(lam0):.5f} "
          f"(sqrt-q scale: arctan sqrt(q/(1-q)) = "
          f"{np.arctan(np.sqrt(Q / (1 - Q))):.5f}), gap = {gap:.2e}")
    dirs = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1),
            "r1": (0.276, 0.850, 0.448), "r2": (0.732, 0.214, 0.647)}
    rep = {}
    for name, dv in dirs.items():
        u = np.array(dv, float)
        u /= np.linalg.norm(u)
        vs = []
        for h in (0.01, 0.005):
            lams = np.linalg.eigvals(o_cycle(h * u, Q))
            pair = sorted(lams, key=lambda l: abs(l - lam0))[:2]
            vs.append(abs(np.angle(pair[0] / pair[1])) / (2 * h))
        rep[name] = 2 * vs[1] - vs[0]
    v = np.array(list(rep.values()))
    print("   slopes: " + "  ".join(f"{n}:{x:.4f}" for n, x in rep.items()))
    print(f"   isotropy max/min = {v.max() / v.min():.5f}   ALL |lam| = 1: "
          f"UNDAMPED CONE")
    assert v.max() / v.min() < 1.001


# -- 2: substrate in-out correlator -------------------------------------------

STATES = np.arange(DIM)


def env_product_state(amp0, amp1):
    v = np.zeros(DIM)
    for e in range(1 << LX):
        a = 1.0
        for x in range(LX):
            a *= amp1 if (e >> x) & 1 else amp0
        v[e << NCAR] = a
    return v


def create(j, v):
    out = np.zeros_like(v)
    below = np.zeros(DIM, dtype=np.int64)
    for m in range(j):
        below ^= (STATES >> m) & 1
    sgn = np.where(below == 1, -1.0, 1.0)
    src = ((STATES >> j) & 1) == 0
    out[STATES[src] | (1 << j)] = sgn[src] * v[src]
    return out


def fresh_read_transfer(q):
    """Predicted 1p in-out transfer per cycle (fresh reads), 12x12 real."""
    D = np.sqrt(1 - q) + np.sqrt(q)
    c, s = np.sqrt(1 - q) / D, np.sqrt(q) / D
    permC = np.argmax(np.abs(COIN_C[0]), axis=0)
    M = np.eye(4 * LX)
    for o in (0, 1):
        conv = np.zeros((4 * LX, 4 * LX))
        for x in range(LX):
            for ch in range(4):
                j = 4 * x + ch
                conv[j, j] += c
                i_c = int(permC[ch])
                conv[4 * x + i_c, j] += s * COIN_C[0][i_c, ch]
        shift = np.zeros((4 * LX, 4 * LX))
        for x in range(LX):
            for ch in range(4):
                xt = (x + int(COIN_D[0][ch])) % LX
                shift[4 * xt + ch, 4 * x + ch] = 1.0
        M = shift @ conv @ M
    return M


def main():
    check_3d()

    perm, sign = full_cycle()
    ketv = env_product_state(np.sqrt(1 - Q), np.sqrt(Q))     # sqrt-P vacuum
    brav = env_product_state(1 / np.sqrt(2), 1 / np.sqrt(2))  # uniform bra
    Mpred = fresh_read_transfer(Q)

    def ev(v, t):
        for _ in range(t):
            out = np.zeros_like(v)
            out[perm] = sign * v
            v = out
        return v

    print("\n2. substrate in-out correlator (1D, re-reads present):")
    n1 = 4 * LX
    for t in (1, 2, 3):
        Z = float(brav @ ev(ketv.copy(), t))
        G = np.zeros((n1, n1))
        for j in range(n1):
            kv = ev(create(j, ketv), t)
            for i in range(n1):
                G[i, j] = float(create(i, brav) @ kv)
        Gn = G / Z
        pred = np.linalg.matrix_power(Mpred, t)
        dev = np.abs(Gn - pred).max()
        ev_mod = np.abs(np.linalg.eigvals(Gn))
        ev_mod = ev_mod[ev_mod > 1e-8]
        spread = ev_mod.max() / ev_mod.min() if len(ev_mod) else np.nan
        print(f"   t={t}: max|G/Z - pred| = {dev:.2e}   "
              f"eigen-moduli spread = {spread:.4f}")
    # in-in comparison baseline: moduli spread of the damped propagator
    from w3c_composite_check import sector_matrix
    env0 = [0, 0, 0]
    # (use the mixture in-in from J38 land: coherent in-in via evolved bra)
    vac_t = ev(ketv.copy(), 2)
    Gin = np.zeros((n1, n1))
    for j in range(n1):
        kv = ev(create(j, ketv), 2)
        for i in range(n1):
            Gin[i, j] = float(create(i, vac_t) @ kv)
    m_in = np.abs(np.linalg.eigvals(Gin))
    m_in = m_in[m_in > 1e-8]
    print(f"   baseline (in-in, t=2): eigen-moduli spread = "
          f"{m_in.max() / m_in.min():.4f}, max modulus = {m_in.max():.4f} "
          f"(vs in-out max {np.abs(np.linalg.eigvals(Gn)).max():.4f})")
    print("\nRoute A: the two-boundary propagator is unitary up to a global "
          "scalar at the fresh-read level; 1D deviations quantified above "
          "are the known re-read (companion) artifact.")


if __name__ == "__main__":
    main()
