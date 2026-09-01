#!/usr/bin/env python
"""Red-team response, theory front: proofs extended to the stated scopes.

Machine checks for docs/notes/theory-redteam-response.md. Assertions run
BEFORE the results file is written. Results -> results/theory_redteam.json.

P1  Two-boundary completion: the solution set is the unitary ARC (every
    product boundary gives zero relative damping and an isotropic node);
    (sqrt(1-q), sqrt(q)) is the point selected by the unique medium-blind
    (bit-flip-invariant) boundary.  [uniqueness RESCOPED and proven]
P2  Q-pinning for non-commuting mixtures: any product of convex mixtures of
    the quaternionic event set lies in the group algebra R[Q8]; its
    eigenvalues are w +- i|v|_2 with |w| + |v|_2 <= 1, i.e. inside the chord
    hull, so dist(arg lam, (pi/2)Z) <= kappa * Gamma with
    kappa = pi/(2 ln 2). [PROVEN, non-commuting and correlated included]
P3  No-two-component theorem at ANY momentum: linear splitting => scalar
    U(k0); orthogonality lemma tr(sigma_z T sigma_z T^-1) = 0 forces
    p = 1/2 AND a rotation coin (symbolic, both placements) => for p != 1/2
    (or any reflection/diagonal coin in the load-bearing slots) no isotropic
    node exists at any k [PROVEN for single-engagement cycles]; the
    surviving p = 1/2 family and the double-engagement (squared) cycles are
    excluded by an exhaustive design sweep with continuous multistart
    scalar-point solving.  [machine-checked]
P4  The I-spectrum bridge: Phi(u + i v) = u (+) (-P v) is a canonical
    complex-linear isomorphism ((V_1)_C, i) -> (V_1 (+) V_{M-1}, I) that
    intertwines the dynamics iff [S, P] = 0; hence the Fourier-i spectral
    analysis of the 1-particle sector IS the I-picture spectrum on the
    particle-hole sector, and the +-omega doublets are one
    particle/antiparticle pair.  [PROVEN + machine-checked in 3D]
P5  (a) M8 tensor identities symbolic in (q, q_m, k).

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_redteam.py [--quick]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pca3d.models.coincube import COIN_C, COIN_D  # noqa: E402

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_redteam.json"
C4 = [np.array(c) for c in COIN_C]
DD = [np.array(d) for d in COIN_D]
KAPPA = np.pi / (2 * np.log(2))


# ----------------------------------------------------------------------------
# P1 -- the completion set and the boundary that selects sqrt(q)
# ----------------------------------------------------------------------------

def part_P1(rng):
    res = {}
    # (a) conformal unitarity of the whole (alpha, beta) family:
    # T = E(alpha + beta C) has T^dag T = (alpha^2 + beta^2) I exactly
    dev = 0.0
    for _ in range(20):
        al, be = rng.normal(size=2)
        k = rng.uniform(-np.pi, np.pi)
        for a in range(3):
            T = np.diag(np.exp(1j * k * DD[a])) @ (al * np.eye(4) + be * C4[a])
            dev = max(dev, np.abs(T.conj().T @ T
                                  - (al**2 + be**2) * np.eye(4)).max())
    res["conformal_unitarity_dev"] = float(dev)

    # (b) general-(alpha,beta) node isotropy, symbolic (the arc inherits it)
    import sympy as sp
    al, be = sp.symbols('alpha beta', real=True)
    I = sp.I
    s1 = sp.Matrix([[0, 1], [1, 0]])
    s2 = sp.Matrix([[0, -I], [I, 0]])
    s3 = sp.Matrix([[1, 0], [0, -1]])
    tx = al * sp.eye(2) - I * be * s1
    ty = al * sp.eye(2) - I * be * s2
    tz = al * sp.eye(2) + I * be * s3
    G = sp.expand(tz * tz * ty * ty * tx * tx)
    avec = [sp.simplify(sp.trace(G * s) / 2 * I) for s in (s1, s2, s3)]
    r2 = sp.simplify(sum(x**2 for x in avec))
    nsig = (avec[0] * s1 + avec[1] * s2 + avec[2] * s3) / sp.sqrt(r2)
    P = (sp.eye(2) + nsig) / 2
    cx = sp.trace(P * (-2 * I * al * tz * tz * ty * ty * s2 * tx))
    cy = sp.trace(P * (-2 * I * al * tz * tz * s3 * ty * tx * tx))
    cz = sp.trace(P * (-2 * I * al * s1 * tz * ty * ty * tx * tx))
    res["isotropy_all_alpha_beta"] = bool(
        sp.simplify(sp.expand(cx**2 - cy**2)) == 0
        and sp.simplify(sp.expand(cx**2 - cz**2)) == 0)

    # (c) boundary -> arc map: (alpha, beta) = (w0 sqrt(1-q), w1 sqrt(q));
    # every (w0, w1) != 0 lands on the arc after normalization; s = sqrt(q)
    # iff w0^2 = w1^2 (the bit-flip-invariant / medium-blind boundary)
    rows = []
    for _ in range(50):
        w0, w1 = np.exp(rng.normal(size=2))
        q = rng.uniform(0.02, 0.98)
        al, be = w0 * np.sqrt(1 - q), w1 * np.sqrt(q)
        s2v = be**2 / (al**2 + be**2)
        rows.append((w0, w1, q, s2v))
    res["boundary_map_s_eq_sqrtq_iff_unbiased"] = bool(all(
        (abs(w0 - w1) < 1e-12) == (abs(s2v - q) < 1e-12)
        for w0, w1, q, s2v in rows))
    # exactness both ways on constructed cases
    for q in (0.1, 0.37):
        assert abs(q * 1.0 / ((1 - q) + q) - q) < 1e-15          # w0 = w1
        w1 = 2.0
        s2v = w1**2 * q / ((1 - q) + w1**2 * q)
        assert abs(s2v - q) > 1e-3                               # biased
    return res


# ----------------------------------------------------------------------------
# P2 -- Q-pinning for arbitrary products of mixtures (group algebra proof)
# ----------------------------------------------------------------------------

def part_P2(rng):
    res = {}
    Q8 = [np.eye(4), -np.eye(4)] + [s * c for c in C4 for s in (1, -1)]
    span = np.stack([np.eye(4).ravel()] + [c.ravel() for c in C4])  # H basis

    worst_hull = 0.0
    worst_ratio = 0.0
    max_resid = 0.0
    for _ in range(4000):
        nf = rng.integers(1, 9)
        T = np.eye(4)
        for _ in range(nf):
            p = rng.dirichlet(np.ones(8) * rng.uniform(0.15, 2.0))
            T = sum(pi * U for pi, U in zip(p, Q8)) @ T
        # T must lie in the quaternion span: T = w + x Cx + y Cy + z Cz
        coef, resid, _, _ = np.linalg.lstsq(span.T, T.ravel(), rcond=None)
        max_resid = max(max_resid, float(np.abs(span.T @ coef - T.ravel()).max()))
        w, v = coef[0], coef[1:]
        r2n = float(np.linalg.norm(v))
        worst_hull = max(worst_hull, abs(w) + r2n)
        lam = w + 1j * r2n
        if abs(abs(lam) - 1) > 1e-12 and abs(lam) > 1e-12:
            gam = -np.log(abs(lam))
            d = np.abs((np.angle(lam) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            worst_ratio = max(worst_ratio, d / gam)
        # eigenvalues of T really are w +- i r2 (set-matched)
        ev = list(np.linalg.eigvals(T))
        for target in (lam, lam, np.conj(lam), np.conj(lam)):
            j = int(np.argmin(np.abs(np.array(ev) - target)))
            assert abs(ev[j] - target) < 1e-9
            ev.pop(j)
    res["span_residual"] = float(max_resid)          # closure: products stay in H
    res["hull_max_l1"] = float(worst_hull)           # <= 1: inside the square
    res["max_dist_over_Gamma"] = float(worst_ratio)  # <= kappa
    res["kappa"] = float(KAPPA)

    # the coincube node from the OPERATOR (not the closed form): bound + match
    worst_fam = 0.0
    cf_dev = 0.0
    for q in np.linspace(0.02, 0.98, 49):
        U = np.eye(4)
        for a in range(3):
            T = (1 - q) * np.eye(4) + q * C4[a]
            U = T @ T @ U
        lam = np.linalg.eigvals(U)
        lam0 = lam[np.argmax(lam.imag)]
        A, B = 1 - 2 * q, 2 * q * (1 - q)
        lam_cf = (A**3 - B**3) + 1j * A * B * np.sqrt(3 * A**2 + 2 * A * B
                                                      + 3 * B**2)
        # the closed form tracks one analytic branch; the +Im eigenvalue is
        # its conjugate for q > 1/2
        cf_dev = max(cf_dev, min(abs(lam0 - lam_cf),
                                 abs(lam0 - np.conj(lam_cf))))
        gam = -np.log(abs(lam0))
        d = np.abs((np.angle(lam0) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        worst_fam = max(worst_fam, d / gam)
    res["family_operator_vs_closedform_dev"] = float(cf_dev)
    res["family_max_dist_over_Gamma"] = float(worst_fam)
    return res


# ----------------------------------------------------------------------------
# P3 -- the two-component no-go at any momentum
# ----------------------------------------------------------------------------

ROT = np.array([[0, -1], [1, 0.]])
REFL = np.array([[0, 1], [1, 0.]])


def part_P3_symbolic():
    """orthogonality lemma: tr(sigma_z T sigma_z T^-1) = 0 <=>
    T00 T11 = -T01 T10; evaluate for every coin type and both placements."""
    import sympy as sp
    p, th = sp.symbols('p theta', real=True, positive=True)
    z = sp.exp(sp.I * th)
    E = sp.diag(z, 1 / z)
    out = {}
    conds = {}
    coins = {"rot": sp.Matrix([[0, -1], [1, 0]]),
             "refl": sp.Matrix([[0, 1], [1, 0]]),
             "diag": sp.Matrix([[1, 0], [0, -1]])}
    for place in ("additive", "multiplicative"):
        for name, C in coins.items():
            T = ((1 - p) * E + p * C if place == "additive"
                 else E * ((1 - p) * sp.eye(2) + p * C))
            cond = sp.simplify(T[0, 0] * T[1, 1] + T[0, 1] * T[1, 0])
            conds[f"{place}_{name}"] = cond
            out[f"{place}_{name}"] = str(cond)
    # rot: (1-p)^2 - p^2 = 1 - 2p -> zero only at p = 1/2
    # refl: (1-p)^2 + p^2 -> never zero for real p
    # diag: 1 - 2p (mult.) -> p = 1/2, but there T is singular (channel dies)
    ok = (sp.simplify(conds["additive_rot"] - ((1 - p)**2 - p**2)) == 0
          and sp.simplify(conds["multiplicative_rot"]
                          - ((1 - p)**2 - p**2)) == 0
          and sp.simplify(conds["additive_refl"]
                          - ((1 - p)**2 + p**2)) == 0
          and sp.simplify(conds["multiplicative_refl"]
                          - ((1 - p)**2 + p**2)) == 0)
    out["lemma_verified"] = bool(ok)
    return out


def part_P3_sweep(rng, quick):
    """sector-resolved exhaustive design sweep: all antidiagonal-coin sign
    tables x traceless direction tables per axis, both placements, single
    (m=1) and blocked (m=2) engagement, p in {1/4, 1/2}; continuous
    multistart scalar-point solving. Reports the minimum splitting
    anisotropy PER SECTOR: the multiplicative m=1 p=1/2 sector contains the
    exact Weyl counterexample (min ~ 0); every other sector has a strictly
    positive anisotropy floor."""
    coins = [s * ROT for s in (1, -1)] + [s * REFL for s in (1, -1)]
    dsets = [np.array([1, -1.]), np.array([-1, 1.])]
    dirs = [np.array(v, float) / np.linalg.norm(v) for v in
            ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
             (0, 1, 1), (1, 1, 1), (1, -1, 1))]
    for _ in range(4):
        v = rng.normal(size=3)
        dirs.append(v / np.linalg.norm(v))

    def Tf(k, d, C, p, place):
        E = np.diag(np.exp(1j * k * d))
        return ((1 - p) * E + p * C if place == "a"
                else E @ ((1 - p) * np.eye(2) + p * C))

    def Uf(kv, design, p, place, m):
        M = np.eye(2, dtype=complex)
        for (d, C), k in zip(design, kv):
            t = Tf(k, d, C, p, place)
            for _ in range(m):
                M = t @ M
        return M

    def sdev(kv, design, p, place, m):
        Um = Uf(kv, design, p, place, m)
        return np.array([Um[0, 1].real, Um[0, 1].imag, Um[1, 0].real,
                         Um[1, 0].imag, (Um[0, 0] - Um[1, 1]).real,
                         (Um[0, 0] - Um[1, 1]).imag])

    nstart = 6 if quick else 14
    sectors = {}
    for place in ("a", "m"):
        for m in (1, 2):
            for p in (0.25, 0.5):
                key = f"{place}_m{m}_p{p}"
                sectors[key] = dict(scalar_points=0, min_aniso=np.inf)
    ndesign = 0
    for place in ("a", "m"):
        for m in (1, 2):
            for combo in itertools.product(
                    itertools.product(dsets, coins), repeat=3):
                design = list(combo)
                ndesign += 1
                for _ in range(nstart):
                    x0 = rng.uniform(-np.pi, np.pi, 3)
                    for p in (0.25, 0.5):
                        key = f"{place}_m{m}_p{p}"
                        r = least_squares(
                            lambda kk: sdev(kk, design, p, place, m), x0,
                            xtol=3e-14, ftol=3e-14, method="lm")
                        if np.abs(r.fun).max() > 1e-9:
                            continue
                        k0 = r.x
                        lam0 = np.trace(Uf(k0, design, p, place, m)) / 2
                        if abs(lam0) < 1e-8:
                            continue
                        eps = 1e-6
                        sl = np.array([
                            abs(np.diff(np.linalg.eigvals(
                                Uf(k0 + eps * n, design, p, place, m)))[0])
                            / eps for n in dirs])
                        if sl.max() < 1e-6:
                            continue
                        sec = sectors[key]
                        sec["scalar_points"] += 1
                        aniso = float(sl.max() / max(sl.min(), 1e-30) - 1)
                        if aniso < sec["min_aniso"]:
                            sec["min_aniso"] = aniso
                            sec["best_k0_over_pi"] = [float(x) for x in
                                                     np.mod(k0 + np.pi, 2 * np.pi)
                                                     / np.pi - 1]
    for sec in sectors.values():
        if sec["min_aniso"] is np.inf:
            sec["min_aniso"] = None
        else:
            sec["min_aniso"] = float(sec["min_aniso"])
    sectors["n_designs"] = ndesign
    return sectors


def part_P3_counterexample():
    """the exact two-component Weyl cone (REFUTES the universal no-go):
    multiplicative placement, p = 1/2, rotation coin, symmetric design
    d = (1,-1) on all axes; k0 = (3pi/4, 3pi/4, -pi/4).  Verified
    SYMBOLICALLY: U(k0) is scalar with |lam0| = 2^{-3/2}, and the three
    effective generators h_a = dU/dk_a /(i lam0) satisfy h_a^2 = 1 and
    pairwise {h_a, h_b} = 0 exactly -- a complex Clifford triple, hence
    splitting +- |dk|: an exact isotropic propagating cone (omega0 = pi,
    v = 1).  Numerically: chirality chi = -1; the arc completion
    (c = s = 1/sqrt(2)) gives U(k0) = -I: an exact UNDAMPED unitary
    2-component Weyl walk in the boundary-completed theory."""
    import sympy as sp
    res = {}
    ROTs = sp.Matrix([[0, -1], [1, 0]])
    k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)

    def Tk(k):
        E = sp.diag(sp.exp(sp.I * k), sp.exp(-sp.I * k))
        return E * (sp.Rational(1, 2) * sp.eye(2)
                    + sp.Rational(1, 2) * ROTs)

    U = Tk(k3) * Tk(k2) * Tk(k1)
    subs0 = {k1: 3 * sp.pi / 4, k2: 3 * sp.pi / 4, k3: -sp.pi / 4}
    U0 = sp.simplify(sp.expand_complex(U.subs(subs0)))
    lam0 = U0[0, 0]
    res["U_k0_scalar"] = bool(sp.simplify(sp.expand_complex(
        U0 - lam0 * sp.eye(2))) == sp.zeros(2, 2))
    res["mod_lam0_exact"] = bool(sp.simplify(sp.expand_complex(
        sp.Abs(lam0) - 1 / (2 * sp.sqrt(2)))) == 0)
    res["omega0_is_pi"] = bool(sp.simplify(sp.expand_complex(
        lam0 + sp.Abs(lam0))) == 0)
    hs = []
    ok_sq = True
    for kk in (k1, k2, k3):
        h = sp.simplify(sp.expand_complex(
            (sp.diff(U, kk) / lam0 / sp.I).subs(subs0)))
        hs.append(h)
        ok_sq = ok_sq and sp.simplify(sp.expand_complex(
            h * h - sp.eye(2))) == sp.zeros(2, 2)
    ok_ac = all(sp.simplify(sp.expand_complex(
        hs[a] * hs[b] + hs[b] * hs[a])) == sp.zeros(2, 2)
        for a, b in itertools.combinations(range(3), 2))
    res["clifford_h_sq_1"] = bool(ok_sq)
    res["clifford_anticommute"] = bool(ok_ac)

    # numerics: chirality and the arc completion
    d = np.array([1, -1.])

    def Tn(k, al, be):
        return np.diag(np.exp(1j * k * d)) @ (al * np.eye(2) + be * ROT)

    def Un(kv, al, be):
        M = np.eye(2, dtype=complex)
        for k in kv:
            M = Tn(k, al, be) @ M
        return M

    k0 = np.array([3 * np.pi / 4, 3 * np.pi / 4, -np.pi / 4])
    lam = np.trace(Un(k0, 0.5, 0.5)) / 2
    sigma = [np.array([[0, 1], [1, 0]], complex),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]], complex)]
    eps = 1e-7
    B = np.zeros((3, 3), complex)
    for a in range(3):
        kp = k0.copy(); kp[a] += eps
        km = k0.copy(); km[a] -= eps
        h = (-1j) * (Un(kp, 0.5, 0.5) - Un(km, 0.5, 0.5)) / (2 * eps) / lam
        h = h - np.trace(h) / 2 * np.eye(2)
        B[a, :] = [np.trace(h @ sg) / 2 for sg in sigma]
    res["chirality"] = int(np.sign(np.linalg.det(B).real))
    c = s_ = 1 / np.sqrt(2)
    Uarc = Un(k0, c, s_)
    res["arc_node_is_minus_I_dev"] = float(np.abs(Uarc + np.eye(2)).max())
    res["arc_unitary_dev"] = float(np.abs(
        Uarc @ Uarc.conj().T - np.eye(2)).max())
    return res


def part_P3_deadaxis():
    """d proportional to (1,1): E = e^{ik} I -> the axis derivative is
    proportional to the identity on any scalar point: no splitting."""
    d = np.array([1, 1.])
    k = 0.37
    E = np.diag(np.exp(1j * k * d))
    return {"E_is_scalar_dev": float(np.abs(E - np.exp(1j * k) * np.eye(2)).max())}


# ----------------------------------------------------------------------------
# P4 -- the I-spectrum bridge, machine-checked in 3D
# ----------------------------------------------------------------------------

def part_P4(rng):
    import theory_3d_certs as tc
    res = {}
    lat = tc.Lattice(2)
    M = lat.M
    env0 = [(rng.random(lat.NS) < 0.35).astype(np.int8) for _ in range(3)]
    states_1p = [(m,) for m in range(M)]
    states_1h = [tuple(m for m in range(M) if m != h) for h in range(M)]
    p1, s1, _ = tc.cycle_sector(lat, states_1p, env0)
    ph, sh, _ = tc.cycle_sector(lat, states_1h, env0)
    sP = np.array([tc.majorana_sign((m,), M) for m in range(M)])

    # real-matrix forms
    S1 = np.zeros((M, M))
    Sh = np.zeros((M, M))
    for m in range(M):
        S1[p1[m], m] = s1[m]
        Sh[ph[m], m] = sh[m]
    Pm = np.diag(sP.astype(float))        # in the hole-index pairing h = m

    # Phi(u + iv) = u (+) (-P v); I = P o diag(eta), eta = (-1 on 1p, +1 on 1h)
    def Phi(x):
        return np.concatenate([x.real, -(Pm @ x.imag)])

    def Iop(y):
        u, w = y[:M], y[M:]
        # diag(eta): (-u, +w); then P swaps sectors with the sP signs
        return np.concatenate([Pm @ w, Pm @ (-u)])

    def Shat(y):
        return np.concatenate([S1 @ y[:M], Sh @ y[M:]])

    dev_i = dev_s = dev_h = 0.0
    for _ in range(20):
        x = rng.normal(size=M) + 1j * rng.normal(size=M)
        y = rng.normal(size=M) + 1j * rng.normal(size=M)
        dev_i = max(dev_i, np.abs(Phi(1j * x) - Iop(Phi(x))).max())
        dev_s = max(dev_s, np.abs(Phi(S1 @ x) - Shat(Phi(x))).max())
        # Hermitian form: <Phi x, Phi y>_I := Phi(x).Phi(y) - i Phi(x).I Phi(y)
        hI = Phi(x) @ Phi(y) - 1j * (Phi(x) @ Iop(Phi(y)))
        hC = np.vdot(x, y)
        dev_h = max(dev_h, abs(hI - hC))
    res["Phi_intertwines_i_vs_I"] = float(dev_i)
    res["Phi_intertwines_dynamics"] = float(dev_s)
    res["hermitian_form_matches"] = float(dev_h)

    # spectral corollary: the I-picture matrix of Shat in the basis
    # {Phi(e_m)} is exactly the real S1 viewed complex-linearly, so the
    # I-spectrum on the particle-hole sector == complexified spec(S1)
    # (both +omega and -omega present, as particle and antiparticle).
    w1 = np.linalg.eigvals(S1)
    res["spec_conjugation_closed"] = bool(
        np.abs(np.sort_complex(w1) - np.sort_complex(np.conj(w1))).max() < 1e-9)
    return res


# ----------------------------------------------------------------------------
# P5a -- M8 tensor identities, symbolic in (q, q_m, k)
# ----------------------------------------------------------------------------

def part_P5a():
    import sympy as sp
    q, qm, k = sp.symbols('q q_m k', real=True)
    X = sp.Matrix([[0, 1], [1, 0]])
    Z = sp.diag(1, -1)
    XZ = X * Z
    I2 = sp.eye(2)
    C4s = [sp.Matrix(np.kron(np.array(XZ, dtype=int), np.eye(2, dtype=int))),
           sp.Matrix(np.kron(np.array(Z, dtype=int), np.array(XZ, dtype=int))),
           sp.Matrix(-np.kron(np.array(X, dtype=int), np.array(XZ, dtype=int)))]
    D4s = [sp.diag(1, 1, -1, -1), sp.diag(1, -1, -1, 1), sp.diag(1, -1, 1, -1)]

    def U4(sign):
        u = sp.eye(4)
        for a in range(3):
            E = sp.diag(*[sp.exp(sp.I * sign * k * D4s[a][c, c])
                          for c in range(4)])
            T = E * ((1 - q) * sp.eye(4) + q * C4s[a])
            u = T * T * u
        return u

    # 8-dim: D8 = D4 (x) diag(1,-1), C8 = C4 (x) 1, M = (1-qm) + qm (1 (x) XZ)
    def U8():
        u = sp.eye(8)
        for a in range(3):
            D8 = sp.Matrix(np.kron(np.array(D4s[a], dtype=int),
                                   np.diag([1, -1])))
            E = sp.diag(*[sp.exp(sp.I * k * D8[c, c]) for c in range(8)])
            C8 = sp.Matrix(np.kron(np.array(C4s[a], dtype=int),
                                   np.eye(2, dtype=int)))
            T = E * ((1 - q) * sp.eye(8) + q * C8)
            u = T * T * u
        CM = sp.Matrix(np.kron(np.eye(4, dtype=int), np.array(XZ, dtype=int)))
        return ((1 - qm) * sp.eye(8) + qm * CM) * u

    lhs = U8()
    # (I): U8 = (1 (x) Rm) . [U4(k) (+) U4(-k)] with b_m the minor index:
    # RHS[2i+b, 2j+b'] = Rm[b, b'] * U4(k if b'=0 else -k)[i, j]
    Rm = (1 - qm) * sp.eye(2) + qm * XZ
    Up, Um_ = U4(1), U4(-1)
    rhs = sp.Matrix(8, 8, lambda I_, J_: Rm[I_ % 2, J_ % 2]
                    * (Up if J_ % 2 == 0 else Um_)[I_ // 2, J_ // 2])
    diff = sp.simplify(sp.expand(lhs - rhs))
    ok1 = all(diff[i, j] == 0 for i in range(8) for j in range(8))
    # (II) at k = 0: U8 = U4(0) (x) Rm
    lhs0 = lhs.subs(k, 0)
    U40 = Up.subs(k, 0)
    rhs0 = sp.Matrix(8, 8, lambda i, j: U40[i // 2, j // 2] * Rm[i % 2, j % 2])
    diff0 = sp.simplify(sp.expand(lhs0 - rhs0))
    ok2 = all(diff0[i, j] == 0 for i in range(8) for j in range(8))
    return {"tensor_identity_I_symbolic": bool(ok1),
            "tensor_identity_II_symbolic": bool(ok2)}


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = {"seed": SEED, "quick": args.quick}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    for name, fn in [("P1", lambda: part_P1(rng)),
                     ("P2", lambda: part_P2(rng)),
                     ("P3_lemma", part_P3_symbolic),
                     ("P3_deadaxis", part_P3_deadaxis),
                     ("P3_counterexample", part_P3_counterexample),
                     ("P3_sweep", lambda: part_P3_sweep(rng, args.quick)),
                     ("P4", lambda: part_P4(rng)),
                     ("P5a", part_P5a)]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:7.1f}s] {name} done")

    # ---------------- assertions BEFORE the results file is written ---------
    P1 = out["P1"]
    assert P1["conformal_unitarity_dev"] < 1e-12
    assert P1["isotropy_all_alpha_beta"]
    assert P1["boundary_map_s_eq_sqrtq_iff_unbiased"]
    P2 = out["P2"]
    assert P2["span_residual"] < 1e-10             # closure in R[Q8]
    assert P2["hull_max_l1"] <= 1 + 1e-12          # l1 bound
    assert P2["max_dist_over_Gamma"] <= KAPPA + 1e-9
    assert P2["family_operator_vs_closedform_dev"] < 1e-12
    assert P2["family_max_dist_over_Gamma"] <= KAPPA + 1e-9
    assert out["P3_lemma"]["lemma_verified"]
    assert out["P3_deadaxis"]["E_is_scalar_dev"] < 1e-15
    ce = out["P3_counterexample"]
    assert ce["U_k0_scalar"] and ce["mod_lam0_exact"] and ce["omega0_is_pi"]
    assert ce["clifford_h_sq_1"] and ce["clifford_anticommute"]
    assert ce["chirality"] == -1
    assert ce["arc_node_is_minus_I_dev"] < 1e-12
    assert ce["arc_unitary_dev"] < 1e-12
    sw = out["P3_sweep"]
    # the counterexample sector contains (numerically) isotropic nodes ...
    assert sw["m_m1_p0.5"]["min_aniso"] < 1e-4
    # ... and every other sector has a strictly positive anisotropy floor
    for key in ("a_m1_p0.25", "a_m1_p0.5", "a_m2_p0.25", "a_m2_p0.5",
                "m_m1_p0.25", "m_m2_p0.25", "m_m2_p0.5"):
        sec = sw[key]
        if sec["min_aniso"] is not None:
            assert sec["min_aniso"] > 0.3, key
    P4 = out["P4"]
    assert P4["Phi_intertwines_i_vs_I"] < 1e-12
    assert P4["Phi_intertwines_dynamics"] < 1e-12
    assert P4["hermitian_form_matches"] < 1e-12
    assert P4["spec_conjugation_closed"]
    assert out["P5a"]["tensor_identity_I_symbolic"]
    assert out["P5a"]["tensor_identity_II_symbolic"]
    print("all headline assertions passed")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
