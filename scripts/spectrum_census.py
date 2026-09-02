#!/usr/bin/env python
"""Full gapless census of the coincube Bloch walks + exact structural facts.

Self-contained machine checks (assertions run BEFORE the results file is
written; results -> results/spectrum_census.json):

  1. FACTORIZATION PROPOSITION: for the per-axis factor
     T_a(k) = E_a(k) (alpha I + beta C_a) with C_a real antisymmetric
     orthogonal (the quaternionic coin), T_a(k)^dag T_a(k) =
     (alpha^2 + beta^2) I identically in k. Hence the six-factor cycle obeys

         U(k) = [alpha^2 + beta^2]^3 x (exactly unitary matrix),  for all k.

     For the in-state (chord) walk (alpha, beta) = (1-q, q) the scalar is
     rho^6 = [(1-q)^2 + q^2]^3: the annealed "damping" is a mode-blind,
     k-independent amplitude factor (an interference-visibility decay), not
     a broadening of any mode relative to another. For the two-boundary
     (arc) walk (alpha, beta) = (sqrt(1-q), sqrt(q)) the scalar is 1.
  2. One-line corollaries, machine-checked over random momenta:
     pi-periodicity  U(k + pi e_a) = U(k)   (E_a(pi) = -I since d = +-1, and
     the sign cancels in T_a^2), and  |lam|^2 = [(1-q)^2+q^2]^6 for EVERY
     eigenvalue at EVERY k (immediate from the factorization).
  3. GAPLESS CENSUS over the reduced zone [0, pi)^3, both walks, q in
     {0.08, 0.15}:
       - corner node k = 0: two-fold, isotropic (50 directions), chirality
         chi = -1;
       - three HALF-POINT Weyl nodes at (pi/2,0,0) & permutations: two-fold,
         chirality chi = +1, anisotropic (axis slopes and full-direction
         spread reported), at a different quasienergy (arc walk: ~3pi/4);
       - three exact NODAL LINES along (t, pi/2, pi/2) & permutations
         (degenerate for all t);
       - the point (pi/2,pi/2,pi/2): U = -rho^6 I exactly (arc: U = -I).
     Point-charge sum on the +omega branch: -1 + 3(+1) = +2, compensated on
     the nodal-line network; the +-omega conjugate branches double the
     census with opposite chiralities (particle/antiparticle pairing).
  4. MASSIVE MODEL (8-channel inversion-doubled walk, mass layer
     M = (1-q_m) + q_m (1 (x) XZ)): the corner AND all three half-point
     nodes gap by exactly 2 arctan[q_m/(1-q_m)]; the nodal lines split only
     weakly (2m/9.8 at (0.7, pi/2, pi/2), q = 0.08) -> sub-gap remnant
     states inside the Dirac gap.
  5. Mass no-go anticommutants in M4(R), by SVD of the linear constraint:
     the anticommutant of the quaternionic triple {C_a} is ZERO-dimensional
     (no real mass matrix of any square sign at n = 4); for a Cl(3,0)
     triple with squares +1 the anticommutant is two-dimensional with all
     squares negative (wrong-sign only). Either way: no real Dirac mass at
     n = 4; the minimal real Clifford mass needs n = 8.

Run:  PYTHONPATH=src .venv/bin/python scripts/spectrum_census.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_C, COIN_D  # noqa: E402

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "spectrum_census.json"
C4 = [np.array(c) for c in COIN_C]
DD = [np.array(d) for d in COIN_D]
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]


def walk(kv, al, be):
    u = np.eye(4, dtype=complex)
    for a in range(3):
        T = np.diag(np.exp(1j * np.asarray(kv, float)[a] * DD[a])) @ (
            al * np.eye(4) + be * C4[a])
        u = T @ T @ u
    return u


def weights(kind, q):
    return (1 - q, q) if kind == "chord" else (np.sqrt(1 - q), np.sqrt(q))


# ----------------------------------------------------------------------------

def part_factorization(rng):
    res = {}
    dev_fac = dev_mod = 0.0
    for q in (0.05, 0.08, 0.15, 0.30):
        for kind in ("chord", "arc"):
            al, be = weights(kind, q)
            rho2 = al**2 + be**2
            for _ in range(50):
                kv = rng.uniform(-np.pi, np.pi, 3)
                for a in range(3):
                    T = np.diag(np.exp(1j * kv[a] * DD[a])) @ (
                        al * np.eye(4) + be * C4[a])
                    dev_fac = max(dev_fac, np.abs(
                        T.conj().T @ T - rho2 * np.eye(4)).max())
                U = walk(kv, al, be)
                dev_mod = max(dev_mod, np.abs(
                    np.abs(np.linalg.eigvals(U)) - rho2**3).max())
    res["factor_unitarity_dev"] = float(dev_fac)
    res["all_eigenvalue_moduli_dev"] = float(dev_mod)
    # pi-periodicity as exact operator identity
    dev_pi = 0.0
    q = 0.08
    al, be = weights("chord", q)
    for _ in range(20):
        kv = rng.uniform(-np.pi, np.pi, 3)
        for a in range(3):
            kp = kv.copy()
            kp[a] += np.pi
            dev_pi = max(dev_pi, np.abs(walk(kp, al, be)
                                        - walk(kv, al, be)).max())
    res["pi_periodicity_dev"] = float(dev_pi)
    return res


# ----------------------------------------------------------------------------

def node_data(al, be, k0, dirs):
    """(lam, mult, slopes over dirs, chirality) of the +Im pair at k0"""
    U0 = walk(k0, al, be)
    w, V = np.linalg.eig(U0)
    Vi = np.linalg.inv(V)
    lam = w[np.argmax(w.imag)]
    up = [i for i in range(4) if abs(w[i] - lam) < 1e-9]
    eps = 1e-6
    Ms = []
    B = np.zeros((3, 3), complex)
    for a in range(3):
        kp = np.array(k0, float); kp[a] += eps
        km = np.array(k0, float); km[a] -= eps
        M = Vi[up, :] @ ((walk(kp, al, be) - walk(km, al, be))
                         / (2 * eps)) @ V[:, up]
        Ms.append(M)
        h = (-1j) * M / lam
        h = h - np.trace(h) / 2 * np.eye(2)
        B[a, :] = [np.trace(h @ s) / 2 for s in PAULI]
    sl = np.array([abs(np.diff(np.linalg.eigvals(
        n[0] * Ms[0] + n[1] * Ms[1] + n[2] * Ms[2]))[0]) for n in dirs])
    chi = int(np.sign(np.linalg.det(B).real))
    return lam, len(up), sl, chi


def part_census(rng):
    dirs = [np.array(v, float) / np.linalg.norm(v) for v in
            ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
             (0, 1, 1), (1, 1, 1))]
    for _ in range(43):
        v = rng.normal(size=3)
        dirs.append(v / np.linalg.norm(v))
    res = {}
    for kind in ("chord", "arc"):
        for q in (0.08, 0.15):
            al, be = weights(kind, q)
            entry = {}
            # corner
            lam, mult, sl, chi = node_data(al, be, (0, 0, 0), dirs)
            entry["corner"] = dict(
                mult=mult, chi=chi, quasienergy_over_pi=float(np.angle(lam) / np.pi),
                mod=float(abs(lam)), v=float(sl.mean() / 2 / abs(lam)),
                spread=float(sl.max() / sl.min()))
            # half-points (three, symmetry-related; verify all three)
            hp = []
            for k0 in ((np.pi / 2, 0, 0), (0, np.pi / 2, 0), (0, 0, np.pi / 2)):
                lam, mult, sl, chi = node_data(al, be, k0, dirs)
                hp.append(dict(
                    k0=[float(x) for x in k0], mult=mult, chi=chi,
                    quasienergy_over_pi=float(np.angle(lam) / np.pi),
                    axis_slopes=[float(x / 2 / abs(lam)) for x in sl[:3]],
                    spread=float(sl.max() / sl.min())))
            entry["half_points"] = hp
            # nodal lines
            worst = 0.0
            for t in (0.3, 0.7, 1.1, 2.0):
                for k0 in ((t, np.pi / 2, np.pi / 2),
                           (np.pi / 2, t, np.pi / 2),
                           (np.pi / 2, np.pi / 2, t)):
                    w = np.linalg.eigvals(walk(k0, al, be))
                    mind = min(abs(a - b) for a, b in
                               itertools.combinations(w, 2))
                    worst = max(worst, mind)
            entry["nodal_line_max_gap"] = float(worst)
            # R point: U = -rho^6 I exactly
            rho6 = (al**2 + be**2)**3
            UR = walk((np.pi / 2, np.pi / 2, np.pi / 2), al, be)
            entry["R_minus_rho6_I_dev"] = float(np.abs(UR + rho6 * np.eye(4)).max())
            entry["point_charge_sum_plus_omega"] = int(
                entry["corner"]["chi"] + sum(h["chi"] for h in hp))
            res[f"{kind}_q{q}"] = entry
    return res


# ----------------------------------------------------------------------------

def part_massive():
    """8-channel inversion-doubled massive walk: gaps and line splitting"""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0.]])
    Z = np.diag([1., -1])
    XZ = X @ Z
    C8 = [np.kron(c, I2) for c in C4]
    D8 = [np.kron(d, [1., -1]) for d in DD]
    CM = np.kron(np.eye(4), XZ)

    def U8(kv, q, qm):
        u = np.eye(8, dtype=complex)
        for a in range(3):
            T = np.diag(np.exp(1j * np.asarray(kv, float)[a] * D8[a])) @ (
                (1 - q) * np.eye(8) + q * C8[a])
            u = T @ T @ u
        return ((1 - qm) * np.eye(8) + qm * CM) @ u

    res = {}
    for q, qm in ((0.08, 0.05), (0.15, 0.05)):
        m = np.arctan(qm / (1 - qm))
        entry = {"m": float(m)}
        # gap at corner and at all three half-points: exactly 2m
        worst = 0.0
        for k0 in ((0, 0, 0), (np.pi / 2, 0, 0), (0, np.pi / 2, 0),
                   (0, 0, np.pi / 2)):
            w = np.linalg.eigvals(U8(k0, q, qm))
            # the massless node quartet splits into two 2-fold groups sharing
            # the maximal modulus; their frequency gap is 2m
            groups = []
            for z in w:
                for g in groups:
                    if abs(z - g[0]) < 1e-9:
                        g[1] += 1
                        break
                else:
                    groups.append([z, 1])
            twos = [g for g in groups if g[1] == 2]
            mods = [abs(g[0]) for g in twos]
            top = [g for g in twos if abs(abs(g[0]) - max(mods)) < 1e-9]
            # pick the +Im-side pair of top-modulus groups
            args = sorted(np.angle(g[0]) for g in top if g[0].imag > 0)
            if len(args) < 2:
                args = sorted(np.angle(g[0]) for g in top)[:2]
            gap = abs(args[1] - args[0])
            worst = max(worst, abs(gap - 2 * m))
        entry["node_gap_minus_2m_max"] = float(worst)
        # nodal-line splitting at (0.7, pi/2, pi/2)
        kL = (0.7, np.pi / 2, np.pi / 2)
        w0 = np.linalg.eigvals(U8(kL, q, 0.0))
        wm = np.linalg.eigvals(U8(kL, q, qm))
        g0 = []
        for z in w0:
            for g in g0:
                if abs(z - g[0]) < 1e-9:
                    g[1] += 1
                    break
            else:
                g0.append([z, 1])
        splits = []
        for gz, gm_ in g0:
            if gm_ < 4:
                continue
            idx = np.argsort(np.abs(wm - gz))[:4]
            args = np.angle(wm[idx])
            dst = []
            for a_ in args:
                if not any(abs(a_ - b_) < 1e-9 for b_ in dst):
                    dst.append(a_)
            if len(dst) == 2:
                splits.append(abs(dst[1] - dst[0]))
        entry["line_split"] = float(max(splits))
        entry["line_split_over_2m"] = float(max(splits) / (2 * m))
        entry["line_split_subgap"] = bool(max(splits) < 2 * m)
        res[f"q{q}_qm{qm}"] = entry
    return res


# ----------------------------------------------------------------------------

def part_anticommutant():
    """SVD of {X : X A + A X = 0 for all A in triple} in M4(R)"""
    def anticomm_dim_and_squares(triple):
        rows = [np.kron(np.eye(4), A) + np.kron(A.T, np.eye(4))
                for A in triple]
        _, sv, vt = np.linalg.svd(np.vstack(rows))
        dim = int(np.sum(sv < 1e-10))
        sq = []
        for i in range(dim):
            M = np.real(vt[-i - 1].reshape(4, 4).T)
            S = M @ M
            assert np.abs(S - S[0, 0] * np.eye(4)).max() < 1e-8
            sq.append(float(S[0, 0]))
        return dim, sq

    res = {}
    dimQ, _ = anticomm_dim_and_squares(C4)               # quaternionic triple
    res["quaternionic_triple_anticommutant_dim"] = dimQ  # = 0
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0.]])
    Z = np.diag([1., -1])
    A3 = [np.kron(Z, I2), np.kron(X, Z), np.kron(X, X)]  # Cl(3,0), squares +1
    dimA, sqA = anticomm_dim_and_squares(A3)
    res["cl30_triple_anticommutant_dim"] = dimA          # = 2
    res["cl30_anticommutant_squares"] = sqA              # all negative
    return res


# ----------------------------------------------------------------------------

def main():
    out = {"seed": SEED}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    for name, fn in [("factorization", lambda: part_factorization(rng)),
                     ("census", lambda: part_census(rng)),
                     ("massive", part_massive),
                     ("anticommutant", part_anticommutant)]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.1f}s] {name} done")

    # ---- assertions (before the results file is written) -------------------
    f = out["factorization"]
    assert f["factor_unitarity_dev"] < 1e-13
    assert f["all_eigenvalue_moduli_dev"] < 1e-12
    assert f["pi_periodicity_dev"] < 1e-12
    for key, e in out["census"].items():
        assert e["corner"]["mult"] == 2 and e["corner"]["chi"] == -1
        assert e["corner"]["spread"] < 1 + 1e-6          # isotropic
        for h in e["half_points"]:
            assert h["mult"] == 2 and h["chi"] == +1
            assert h["spread"] > 1.5                     # anisotropic
        assert e["nodal_line_max_gap"] < 1e-12
        assert e["R_minus_rho6_I_dev"] < 1e-12
        assert e["point_charge_sum_plus_omega"] == 2
    # arc half-points sit at ~3pi/4 at the working point q = 0.08
    # (quasienergies are q-dependent: 0.749 pi at q = 0.08, 0.663 pi at 0.15)
    for h in out["census"]["arc_q0.08"]["half_points"]:
        assert abs(abs(h["quasienergy_over_pi"]) - 0.75) < 0.01
    for key, e in out["massive"].items():
        assert e["node_gap_minus_2m_max"] < 1e-12        # all four nodes: 2m
        assert e["line_split_subgap"]
    e = out["massive"]["q0.08_qm0.05"]
    assert abs(e["line_split_over_2m"] - 1 / 9.78) < 0.01
    a = out["anticommutant"]
    assert a["quaternionic_triple_anticommutant_dim"] == 0
    assert a["cl30_triple_anticommutant_dim"] == 2
    assert all(s < 0 for s in a["cl30_anticommutant_squares"])
    print("all headline assertions passed")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
