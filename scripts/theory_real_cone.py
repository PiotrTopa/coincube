#!/usr/bin/env python
"""W3 theory strike: real Clifford structure of the dynamical-coin cone.

Machine checks for the real-coin cone theory (free-spectrum section of the
paper). Results are written to
results/theory_real_cone.json.

Model (annealed dynamical coin, one full cycle = two blocking origins per axis):

    T_a(k)  = (1-p) E_a(k_a) + p C_a,   E_a = diag(exp(i k_a d_a)),  d_a in {+-1}^n
    U(k)    = T_z(k_z)^2 T_y(k_y)^2 T_x(k_x)^2

with C_a a REAL signed permutation (fermionic-lift sign gauge).

Parts:
  n2_algebra    -- signature (2,1) of the quadratic form behind M2(R): no real
                   2-dim Cl(3,0), machine-checked.
  n2_scan       -- exhaustive scan of all 32768 n=2 designs: no isotropic cone.
  n4_construction - the quaternion-coin design: exact isotropic Weyl cones,
                   signed permutations only; node census; controls.
  n4_symbolic   -- sympy proof c_x^2 = c_y^2 = c_z^2 (exact isotropy at all p).
  mass          -- anticommutant of the Cl(3,0) triple (no real mass at n=4),
                   Cl(4,0) quadruple at n=8, Weyl-node protection, p=1/2 merged
                   Dirac point and its (anisotropic) gap openers.
  quenched      -- lane-media class-locked 32-dim transfer operator: MC
                   validation, pole census, exact diamond at eta=1, exact cone
                   for all eta<1 (companion-memory interpolation).

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_real_cone.py
      (add --quick to skip the two slow parts: n2_scan refinement, MC)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_real_cone.json"

# ----------------------------------------------------------------------------
# shared algebra
# ----------------------------------------------------------------------------

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0.]])
SZ = np.array([[1, 0], [0, -1.]])
SYR = np.array([[0, 1], [-1, 0.]])          # i*sigma_y, real
SY = np.array([[0, -1j], [1j, 0]])

# quaternion left/right multiplications in basis (1, i, j, k)
LI = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], float)
LJ = np.array([[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]], float)
LK = np.array([[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], float)
RI = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], float)
RJ = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], float)
RK = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], float)
D1 = np.array([1, 1, -1, -1.])   # diag = sigma_z (x) 1
D2 = np.array([1, -1, 1, -1.])   # diag = 1 (x) sigma_z
D3 = np.array([1, -1, -1, 1.])   # diag = sigma_z (x) sigma_z

# THE winning design: C = (L_i, L_j, L_k), d = (delta_j, delta_k, delta_i)
DESIGN = [(D2, LI), (D3, LJ), (D1, LK)]


def is_signed_perm(M):
    A = np.abs(M)
    return (np.allclose(A.sum(0), 1) and np.allclose(A.sum(1), 1)
            and np.allclose(A * (A != 0), np.abs(M)) and np.allclose(np.abs(M[M != 0]), 1))


def signed_perms(n):
    out = []
    for perm in itertools.permutations(range(n)):
        for signs in itertools.product((1, -1), repeat=n):
            M = np.zeros((n, n))
            for c, (r, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            out.append(M)
    return out


def U_of_k(kv, design, p, extra=None):
    """one cycle U(k) = Tz^2 Ty^2 Tx^2 (optionally followed by an extra coin block)"""
    n = len(design[0][0])
    M = np.eye(n, dtype=complex)
    for (d, C), k in zip(design, kv):
        t = (1 - p) * np.diag(np.exp(1j * k * d)) + p * C
        M = t @ t @ M
    if extra is not None:
        M = extra @ M
    return M


def direction_set(rng, nrand=6):
    ds = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
          (1, -1, 0), (1, 1, 1), (1, -1, 1), (1, 1, -1)]
    ds = [np.array(d, float) / np.linalg.norm(d) for d in ds]
    for _ in range(nrand):
        v = rng.normal(size=3)
        ds.append(v / np.linalg.norm(v))
    return ds


def degenerate_pairs(w, tol=1e-9):
    """group exactly repeated eigenvalues; return list (value, index-list)"""
    groups = []
    for i, z in enumerate(w):
        for g in groups:
            if abs(z - g[0]) < tol:
                g[1].append(i)
                break
        else:
            groups.append([z, [i]])
    return groups


def multiplet_slopes(Ufun, k0, dirs, eps=1e-6, tol=1e-9):
    """exact-degenerate perturbation theory: for each degenerate eigengroup of
    U(k0), project dU onto it and return per-direction eigenvalue diameters."""
    U0 = Ufun(k0)
    w0, V = np.linalg.eig(U0)
    Vinv = np.linalg.inv(V)
    dUs = []
    for a in range(3):
        kp = np.array(k0, float); kp[a] += eps
        km = np.array(k0, float); km[a] -= eps
        dUs.append((Ufun(kp) - Ufun(km)) / (2 * eps))
    out = []
    for lam, idx in degenerate_pairs(w0, tol):
        if len(idx) < 2:
            continue
        R = V[:, idx]
        L = Vinv[idx, :]
        Ms = [L @ dU @ R for dU in dUs]
        diams = []
        for n in dirs:
            mu = np.linalg.eigvals(n[0] * Ms[0] + n[1] * Ms[1] + n[2] * Ms[2])
            diams.append(max(abs(a - b) for a, b in itertools.combinations(mu, 2)))
        out.append((lam, len(idx), np.array(diams), Ms))
    return out


# ----------------------------------------------------------------------------
# Part 1a: the n=2 obstruction, algebraically
# ----------------------------------------------------------------------------

def part_n2_algebra(rng):
    res = {}
    # Quadratic form N(A) = -det A on sl2(R), basis (h, e, f) with
    # A = a*h + b*e + c*f = [[a, b], [c, -a]]:  N = a^2 + b*c.
    # Gram matrix (2x polarization) in integer arithmetic:
    G = np.array([[2, 0, 0], [0, 0, 1], [0, 1, 0]], float) / 2.0
    ev = np.linalg.eigvalsh(2 * G)   # integer matrix [[2,0,0],[0,0,1],[0,1,0]]
    signature = (int(np.sum(ev > 0)), int(np.sum(ev < 0)))
    res["signature_of_N_on_sl2R"] = signature       # expect (2, 1)
    # Cayley-Hamilton check A^2 = N(A) I for random traceless A (exact identity):
    dev = 0.0
    for _ in range(200):
        a, b, c = rng.integers(-9, 10, 3)
        A = np.array([[a, b], [c, -a]], float)
        dev = max(dev, np.abs(A @ A - (a * a + b * c) * np.eye(2)).max())
    res["cayley_hamilton_maxdev"] = float(dev)
    # random pairwise-anticommuting invertible real triples: sign pattern of
    # squares is always (+,+,-) up to order (never (+,+,+) / (-,-,-)).
    patterns = {}
    tries = 0
    while sum(patterns.values()) < 500 and tries < 100000:
        tries += 1
        A = rng.normal(size=(2, 2)); A -= np.trace(A) / 2 * np.eye(2)
        B = rng.normal(size=(2, 2)); B -= np.trace(B) / 2 * np.eye(2)
        # project B orthogonal to A wrt polarization: B -= A * <A,B>/<A,A>
        pol = lambda X, Y: np.trace(X @ Y + Y @ X).real / 2 / 2 * 2  # tr(XY) sym
        nA = -np.linalg.det(A)
        if abs(nA) < 1e-6:
            continue
        B = B - A * (np.trace(A @ B) / np.trace(A @ A))
        nB = -np.linalg.det(B)
        if abs(nB) < 1e-6:
            continue
        Cm = A @ B  # anticommutes with both (2x2 special fact: AB = -BA up to..)
        # verify pairwise anticommutation numerically
        if max(np.abs(A @ B + B @ A).max(), np.abs(A @ Cm + Cm @ A).max(),
               np.abs(B @ Cm + Cm @ B).max()) > 1e-8:
            continue
        signs = tuple(int(s) for s in sorted(
            np.sign([-np.linalg.det(A), -np.linalg.det(B),
                     -np.linalg.det(Cm)]).astype(int), reverse=True))
        patterns[signs] = patterns.get(signs, 0) + 1
    res["square_sign_patterns"] = {str(k): v for k, v in patterns.items()}
    res["ppp_triple_found"] = any(k == (1, 1, 1) for k in patterns)
    return res


# ----------------------------------------------------------------------------
# Part 1b: exhaustive n=2 transfer-model scan
# ----------------------------------------------------------------------------

def part_n2_scan(rng, quick=False):
    res = {}
    SP2 = signed_perms(2)
    DS2 = [np.array(d, float) for d in itertools.product((1, -1), repeat=2)]
    dirs = direction_set(rng)
    vals = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    p = 0.25

    # precompute per-axis squared factors on the k-grid
    combos = list(itertools.product(range(4), range(8)))     # (d index, C index)
    Tsq = np.zeros((len(combos), len(vals), 2, 2), complex)
    for ci, (di, si) in enumerate(combos):
        for ki, kval in enumerate(vals):
            t = (1 - p) * np.diag(np.exp(1j * kval * DS2[di])) + p * SP2[si]
            Tsq[ci, ki] = t @ t

    def U2(design_ci, kidx):
        return (Tsq[design_ci[2], kidx[2]] @ Tsq[design_ci[1], kidx[1]]
                @ Tsq[design_ci[0], kidx[0]])

    kidx_all = list(itertools.product(range(4), repeat=3))
    n_diabolic = 0
    min_aniso_mod = np.inf
    min_aniso_om = np.inf
    best = None
    for dci in itertools.product(range(len(combos)), repeat=3):
        # find on-grid scalar (diabolic) points: at n=2 a two-fold degeneracy
        # of a 2x2 with linear splitting REQUIRES U(k0) = lam*I (else it is an
        # exceptional point with sqrt splitting, not a cone).
        found_k = None
        for kidx in kidx_all:
            Um = U2(dci, kidx)
            lam = (Um[0, 0] + Um[1, 1]) / 2
            if (abs(Um[0, 1]) < 1e-9 and abs(Um[1, 0]) < 1e-9
                    and abs(Um[0, 0] - Um[1, 1]) < 1e-9 and abs(lam) > 1e-12):
                found_k = np.array([vals[i] for i in kidx])
                break
        if found_k is None:
            continue
        design = [(DS2[combos[c][0]], SP2[combos[c][1]]) for c in dci]
        lam0 = (U_of_k(found_k, design, p)[0, 0] + U_of_k(found_k, design, p)[1, 1]) / 2
        eps = 1e-5
        sm, so = [], []
        for n in dirs:
            w = np.linalg.eigvals(U_of_k(found_k + eps * n, design, p))
            z = (w[0] - w[1]) / eps
            sm.append(abs(z))
            so.append(abs((z / lam0).imag) / 2)
        sm, so = np.array(sm), np.array(so)
        if sm.max() < 1e-6:
            continue                      # trivial (scalar model): v = 0 everywhere
        n_diabolic += 1
        am = sm.max() / max(sm.min(), 1e-30) - 1
        ao = so.max() / max(so.min(), 1e-30) - 1
        if am < min_aniso_mod:
            min_aniso_mod = am
        if ao < min_aniso_om:
            min_aniso_om = ao
            best = dict(k0=[float(x) for x in found_k],
                        vo_min=float(so.min()), vo_max=float(so.max()))
    res["n_designs"] = 32768
    res["n_nontrivial_diabolic"] = n_diabolic
    res["min_aniso_modulus"] = float(min_aniso_mod)
    res["min_aniso_omega"] = float(min_aniso_om)
    res["best_design"] = best

    # off-grid check: continuous k0 refinement on a random design subsample
    if not quick:
        from scipy.optimize import minimize
        n_ref, best_off = 0, np.inf
        for _ in range(300):
            design = [(DS2[rng.integers(4)], SP2[rng.integers(8)]) for _ in range(3)]

            def dev(kk):
                Um = U_of_k(kk, design, p)
                lam = (Um[0, 0] + Um[1, 1]) / 2
                return np.abs(Um - lam * np.eye(2)).max()

            bd, bk = np.inf, None
            for x0 in [rng.uniform(-np.pi, np.pi, 3) for _ in range(4)]:
                r = minimize(dev, x0, method="Nelder-Mead",
                             options=dict(xatol=1e-10, fatol=1e-12, maxiter=300))
                if r.fun < bd:
                    bd, bk = r.fun, r.x
            if bd < 1e-8:
                eps = 1e-5
                sl = []
                lam0 = np.trace(U_of_k(bk, design, p)) / 2
                dirs_l = direction_set(rng, 3)
                for n in dirs_l:
                    w = np.linalg.eigvals(U_of_k(bk + eps * n, design, p))
                    sl.append(abs((w[0] - w[1]) / eps))
                sl = np.array([abs(s) for s in sl])
                if sl.max() > 1e-6:
                    n_ref += 1
                    best_off = min(best_off, sl.max() / max(sl.min(), 1e-30) - 1)
        res["refined_subsample"] = dict(n_designs=300, n_nontrivial=n_ref,
                                        min_aniso=float(best_off))
    return res


# ----------------------------------------------------------------------------
# Part 2: the n=4 quaternion-coin construction
# ----------------------------------------------------------------------------

def part_n4_construction(rng):
    res = {}
    # given Cl(3,0) triple from the task
    A1 = np.kron(SZ, I2); A2 = np.kron(SX, SZ); A3 = np.kron(SX, SX)
    ok = all(np.abs(a @ b + b @ a).max() < 1e-14
             for a, b in itertools.combinations([A1, A2, A3], 2))
    ok &= all(np.allclose(a @ a, np.eye(4)) for a in (A1, A2, A3))
    res["A_triple_cl30"] = dict(anticommute_and_square_plus1=bool(ok),
                                all_signed_permutations=bool(all(
                                    is_signed_perm(a) for a in (A1, A2, A3))))
    # quaternion tables
    assert np.allclose(LI @ LJ, LK) and np.allclose(LJ @ LK, LI) and np.allclose(LK @ LI, LJ)
    for L in (LI, LJ, LK):
        assert np.allclose(L @ L, -np.eye(4)) and is_signed_perm(L)
        for R in (RI, RJ, RK):
            assert np.allclose(L @ R, R @ L)
    assert np.allclose(np.diag(D1), -LI @ RI)
    assert np.allclose(np.diag(D2), -LJ @ RJ)
    assert np.allclose(np.diag(D3), -LK @ RK)
    res["quaternion_tables_verified"] = True
    res["design"] = dict(
        d=dict(x=[int(v) for v in D2], y=[int(v) for v in D3], z=[int(v) for v in D1]),
        C=dict(x="L_i", y="L_j", z="L_k"),
        signed_permutation_only=True,
        note="C_a = quaternion units (real signed permutations, no fixed point, "
             "C^2=-1); d_a = the delta_v diagonal NOT paired with C_a's unit "
             "(v_a != u_a), cyclic assignment")

    dirs = direction_set(rng)
    # closed form lam0(p) and per-p cone data
    table = []
    for p in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        A = 1 - 2 * p; B = 2 * p * (1 - p)
        lam0_cf = (A**3 + B**3) + 1j * A * B * np.sqrt(3 * A**2 - 2 * A * B + 3 * B**2)
        groups = multiplet_slopes(lambda kv: U_of_k(kv, DESIGN, p), np.zeros(3), dirs)
        # the +Im Weyl pair
        g = [gg for gg in groups if gg[0].imag > 0][0]
        lam0, mult, diams, Ms = g
        # frequency/damping decomposition (direction x)
        mu = np.linalg.eigvals(Ms[0])
        z = (mu[0] - mu[1]) / 2 / lam0
        vo, vg = abs(z.imag), abs(z.real)
        aniso = float(diams.max() / diams.min() - 1)
        assert vo <= 2.0 + 1e-9, "cone bound violated"       # <= 2 sites/cycle
        table.append(dict(p=p, lam0=[lam0.real, lam0.imag],
                          lam0_closed_form_dev=float(abs(lam0 - lam0_cf)),
                          mod_lam0=float(abs(lam0)),
                          rho6=float(((1 - p)**2 + p**2)**3),
                          mult=int(mult), split_slope=float(diams.mean()),
                          aniso=aniso, v_omega=float(vo), v_gamma=float(vg)))
    res["cone_table"] = table
    res["max_aniso_over_p"] = float(max(t["aniso"] for t in table))
    res["max_v_gamma_over_p"] = float(max(t["v_gamma"] for t in table))

    # direct eigenvalue line fits (batched): linearity + isotropy at p=0.25
    p = 0.25
    U0 = U_of_k(np.zeros(3), DESIGN, p)
    w0 = np.linalg.eigvals(U0)
    lamref = w0[np.argmax(w0.imag)]
    lines = {}
    for lbl, n in [("100", (1, 0, 0)), ("110", (1, 1, 0)), ("110b", (1, -1, 0)),
                   ("111", (1, 1, 1)), ("r1", None), ("r2", None), ("r3", None)]:
        if n is None:
            v = rng.normal(size=3); n = v / np.linalg.norm(v)
        else:
            n = np.array(n, float) / np.linalg.norm(n)
        slopes = []
        for eps in (1e-2, 1e-3, 1e-4):
            w = np.linalg.eigvals(U_of_k(eps * n, DESIGN, p))
            idx = np.argsort(np.abs(w - lamref))[:2]
            slopes.append(abs(w[idx[0]] - w[idx[1]]) / eps)
        lines[lbl] = dict(slope=float(slopes[-1]),
                          linearity_ratio=float(slopes[-1] / slopes[-2]))
    res["line_fits_p025"] = lines
    sl = [v["slope"] for v in lines.values()]
    res["line_fit_aniso"] = float(max(sl) / min(sl) - 1)

    # exact degeneracy + right-module protection
    res["exact_pair_degeneracy"] = float(max(
        abs(a - b) for g in degenerate_pairs(w0) if len(g[1]) == 2
        for a, b in [(w0[g[1][0]], w0[g[1][1]])]))

    # corner node census with helicity
    sigma = [SX.astype(complex), SY, SZ.astype(complex)]
    census = []
    for kc in itertools.product((0.0, np.pi), repeat=3):
        k0 = np.array(kc)
        U0 = U_of_k(k0, DESIGN, p)
        w0, V = np.linalg.eig(U0); Vinv = np.linalg.inv(V)
        idx = [i for i in range(4) if w0[i].imag > 0]
        i, j = idx
        lam0 = w0[i]
        Bm = np.zeros((3, 3), complex)
        eps = 1e-6
        slp = []
        for a in range(3):
            kp = k0.copy(); kp[a] += eps; km = k0.copy(); km[a] -= eps
            dU = (U_of_k(kp, DESIGN, p) - U_of_k(km, DESIGN, p)) / (2 * eps)
            M = Vinv[[i, j], :] @ dU @ V[:, [i, j]]
            h = (-1j) * M / lam0
            h = h - np.trace(h) / 2 * np.eye(2)
            Bm[a, :] = [np.trace(h @ s) / 2 for s in sigma]
        det = np.linalg.det(Bm)
        diams = multiplet_slopes(lambda kv: U_of_k(kv, DESIGN, p), k0, dirs)
        g = [gg for gg in diams if gg[0].imag > 0][0]
        census.append(dict(k=[float(x) for x in kc], lam0=[lam0.real, lam0.imag],
                           chi=int(np.sign(det.real)),
                           v_omega=float(abs(det) ** (1 / 3)),
                           aniso=float(g[2].max() / g[2].min() - 1)))
    res["corner_node_census"] = census
    res["chirality_sum"] = int(sum(c["chi"] for c in census))
    res["max_corner_aniso"] = float(max(c["aniso"] for c in census))

    # controls: breaking the cyclic (u, v) pairing destroys isotropy
    controls = {}
    for name, dd in (("v_z_equals_u_z", [(D2, LI), (D1, LJ), (D3, LK)]),
                     ("v_repeated", [(D3, LI), (D3, LJ), (D1, LK)])):
        g = [gg for gg in multiplet_slopes(lambda kv: U_of_k(kv, dd, p),
                                           np.zeros(3), dirs) if gg[0].imag > 0][0]
        controls[name] = dict(aniso=float(g[2].max() / max(g[2].min(), 1e-30) - 1))
    res["controls_broken_designs"] = controls
    return res


def part_n4_symbolic():
    """sympy: prove c_x^2 = c_y^2 = c_z^2 exactly (isotropy at every p).

    Left-factor (H = C^2 x C^2 splitting) computation: U(0)_left =
    t3^2 t2^2 t1^2 with t_u = (1-p) - i p sigma_u; the k-derivative of T_a^2
    collapses via {sigma_v, t_u} = 2(1-p) sigma_v (u != v) to
    2i(1-p)^2 sigma_{v_a} inserted at the axis slot. c_a = projection onto the
    lam0 eigenvector. Isotropy <=> c_x^2 = c_y^2 = c_z^2 as functions of p.
    """
    import sympy as sp
    p = sp.symbols('p', positive=True)
    I = sp.I
    s1 = sp.Matrix([[0, 1], [1, 0]]); s2 = sp.Matrix([[0, -I], [I, 0]])
    s3 = sp.Matrix([[1, 0], [0, -1]])
    t = {u: (1 - p) * sp.eye(2) - I * p * s for u, s in ((1, s1), (2, s2), (3, s3))}
    G = sp.expand(t[3] * t[3] * t[2] * t[2] * t[1] * t[1])
    a0 = sp.simplify(sp.trace(G) / 2)
    avec = [sp.simplify(sp.trace(G * s) / 2 * I) for s in (s1, s2, s3)]
    r2 = sp.simplify(sum(x**2 for x in avec))
    nsig = (avec[0] * s1 + avec[1] * s2 + avec[2] * s3) / sp.sqrt(r2)
    P = (sp.eye(2) + nsig) / 2                     # lam0 = a0 - i sqrt(r2) projector
    Mx = 2 * I * (1 - p)**2 * t[3] * t[3] * t[2] * t[2] * s2
    My = 2 * I * (1 - p)**2 * t[3] * t[3] * s3 * t[1] * t[1]
    Mz = 2 * I * (1 - p)**2 * s1 * t[2] * t[2] * t[1] * t[1]
    cx = sp.trace(P * Mx); cy = sp.trace(P * My); cz = sp.trace(P * Mz)
    d1 = sp.simplify(sp.expand(cx**2 - cy**2))
    d2 = sp.simplify(sp.expand(cx**2 - cz**2))
    # numeric agreement of the derived slope 2|cx| with the measured splitting
    num = {}
    for pv in (sp.Rational(1, 20), sp.Rational(1, 10), sp.Rational(1, 4)):
        num[str(pv)] = float(2 * abs(complex(cx.subs(p, pv))))
    return dict(cx2_minus_cy2=str(d1), cx2_minus_cz2=str(d2),
                isotropy_proven=bool(d1 == 0 and d2 == 0),
                slope_2cx=num,
                lam0_closed_form="lam0 = A^3+B^3 + i A B sqrt(3A^2-2AB+3B^2), "
                                 "A = 1-2p, B = 2p(1-p); |lam0|^2 = (A^2+B^2)^3 "
                                 "= ((1-p)^2+p^2)^6 -- uniform damping")


# ----------------------------------------------------------------------------
# Part 3: the mass question
# ----------------------------------------------------------------------------

def part_mass(rng):
    res = {}
    A1 = np.kron(SZ, I2); A2 = np.kron(SX, SZ); A3 = np.kron(SX, SX)

    # anticommutant of the triple: {X : XA_a = -A_a X} via SVD nullspace
    rows = []
    for A in (A1, A2, A3):
        rows.append(np.kron(np.eye(4), A) + np.kron(A.T, np.eye(4)))
    S = np.vstack(rows)
    _, sv, vt = np.linalg.svd(S)
    null_dim = int(np.sum(sv < 1e-10))
    basis = [vt[-i - 1].reshape(4, 4).T for i in range(null_dim)]
    sq_coeffs = []
    for al, be in [(1, 0), (0, 1), (1, 1), (1, -1), (0.3, 0.7), (2, -5)]:
        if null_dim == 2:
            M = al * np.real(basis[0]) + be * np.real(basis[1])
            sq = M @ M
            assert np.abs(sq - sq[0, 0] * np.eye(4)).max() < 1e-8
            sq_coeffs.append(float(sq[0, 0].real))
    res["anticommutant"] = dict(dim=null_dim, all_squares_negative=bool(
        all(c < 0 for c in sq_coeffs)), square_coeff_samples=sq_coeffs)

    # real tensor words: no Cl(4,0) quadruple at 4x4, one exists at 8x8
    words1 = [I2, SX, SZ, SYR]; names1 = ["1", "X", "Z", "Y"]

    def words(nfac):
        out = []
        for combo in itertools.product(range(4), repeat=nfac):
            M = words1[combo[0]]
            for c in combo[1:]:
                M = np.kron(M, words1[c])
            out.append(("".join(names1[c] for c in combo), M))
        return out

    def quadruples(ws):
        sq = [(nm, M) for nm, M in ws if np.allclose(M @ M, np.eye(len(M)))]
        found = []
        for combo in itertools.combinations(range(len(sq)), 4):
            if all(np.abs(sq[i][1] @ sq[j][1] + sq[j][1] @ sq[i][1]).max() < 1e-12
                   for i, j in itertools.combinations(combo, 2)):
                found.append([sq[i][0] for i in combo])
                if len(found) > 3:
                    break
        return found

    res["cl40_word_quadruples_4x4"] = quadruples(words(2))
    q8 = quadruples(words(3))
    res["cl40_word_quadruple_8x8"] = q8[0] if q8 else None
    if q8:
        # verify signed-permutation property of the n=8 quadruple
        wmap = dict(words(3))
        res["cl40_8x8_all_signed_perms"] = bool(all(is_signed_perm(wmap[nm])
                                                    for nm in q8[0]))

    # Weyl-node protection at p<1/2: coin perturbations move the node, never gap
    from scipy.optimize import minimize
    p, pm = 0.25, 0.1
    prot = {}
    for name, Sm in (("R_i", RI), ("L_k", LK), ("delta_1", np.diag(D1))):
        extra = (1 - pm) * np.eye(4) + pm * Sm

        def gapf(kv):
            w = np.linalg.eigvals(U_of_k(kv, DESIGN, p, extra))
            return min(abs(a - b) for a, b in itertools.combinations(w, 2))

        g0 = gapf(np.zeros(3))
        entry = dict(gap_at_k0=float(g0))
        if g0 > 1e-10:
            best = min((minimize(gapf, x0, method="Nelder-Mead",
                                 options=dict(xatol=1e-12, fatol=1e-14, maxiter=2000))
                        for x0 in (np.array([0.01, 0.01, -0.09]),
                                   np.array([0.05, -0.05, 0.05]), np.zeros(3) + 0.02)),
                       key=lambda r: r.fun)
            entry["relocated_node"] = dict(k=[float(x) for x in best.x],
                                           residual_gap=float(best.fun))
        prot[name] = entry
    res["node_protection"] = prot

    # p = 1/2: exact 4-fold scalar Dirac point; mass openers exist but are
    # anisotropic (the merged point's kinetic term is already anisotropic)
    p = 0.5
    U0 = U_of_k(np.zeros(3), DESIGN, p)
    res["p_half_scalar_point_dev"] = float(np.abs(U0 - 0.125 * np.eye(4)).max())
    dirs = direction_set(rng)
    sl = []
    for n in dirs:
        w = np.linalg.eigvals(U_of_k(1e-4 * n, DESIGN, p))
        om = np.sort(np.angle(w / 0.125))
        sl.append((om.max() - om.min()) / 2e-4)
    sl = np.array(sl)
    res["p_half_kinetic_slopes"] = dict(min=float(sl.min()), max=float(sl.max()))
    # 384-scan for gap openers
    pm = 0.1
    openers = []
    for idx, Sm in enumerate(signed_perms(4)):
        extra = (1 - pm) * np.eye(4) + pm * Sm
        w0 = np.sort_complex(np.linalg.eigvals(U_of_k(np.zeros(3), DESIGN, p, extra)))
        if (abs(w0[0] - w0[1]) < 1e-9 and abs(w0[2] - w0[3]) < 1e-9
                and abs(np.angle(w0[0] / w0[2])) > 1e-6):
            m = abs(np.angle(w0[0]) - np.angle(w0[2])) / 2

            def gap_at(kv):
                om = np.sort(np.angle(np.linalg.eigvals(U_of_k(kv, DESIGN, p, extra))))
                return (om[2] + om[3]) / 2 - (om[0] + om[1]) / 2

            g0 = gap_at(np.zeros(3))
            v2 = {}
            for lbl, n in (("100", (1, 0, 0)), ("111", (1, 1, 1))):
                n = np.array(n, float) / np.linalg.norm(n)
                kk = 0.1
                v2[lbl] = float((gap_at(kk * n)**2 - g0**2) / kk**2)
            openers.append(dict(index=idx, m=float(m), v2=v2))
    res["p_half_gap_openers"] = dict(
        count=len(openers),
        example=openers[0] if openers else None,
        min_v2_aniso=float(min(abs(o["v2"]["100"] / o["v2"]["111"] - 1)
                               for o in openers)) if openers else None)
    return res


# ----------------------------------------------------------------------------
# Part 4: quenched (lane-media, class-locked) vs annealed
# ----------------------------------------------------------------------------

P0 = np.array([[1, 0], [0, 0.]]); P1 = np.array([[0, 0], [0, 1.]])
SM = np.array([[0, 1], [0, 0.]]); SP = np.array([[0, 0], [1, 0.]])


def Tq_axis(k, d, C, q, axis, eta=1.0):
    """per-engagement transfer on coin (x) c_x (x) c_y (x) c_z (32-dim).

    Companion (class) bit c_a per axis; fresh medium bit b ~ Bernoulli(q):
      b == c_a: coin-conditioned move by d_a[coin]
      b != c_a: conversion (coin <- C_a coin), c_a <- b
    eta < 1: companion resampled from stationarity with prob (1 - eta) before
    the engagement (eta = 1 strict class-locking, eta = 0 memoryless =
    annealed with p = 2q(1-q))."""
    E = np.diag(np.exp(1j * k * d))
    W = eta * np.eye(2) + (1 - eta) * np.array([[1 - q, 1 - q], [q, q]])
    T8 = (np.kron(E, (1 - q) * P0 + q * P1)
          + np.kron(C, (1 - q) * SM + q * SP)) @ np.kron(np.eye(4), W)
    full = np.zeros((32, 32), complex)
    T8r = T8.reshape(4, 2, 4, 2)
    for spr in range(4):
        for s in range(4):
            blk = T8r[spr, :, s, :]
            mats = [I2, I2, I2]
            mats[axis] = blk
            full[spr * 8:(spr + 1) * 8, s * 8:(s + 1) * 8] = np.kron(
                np.kron(mats[0], mats[1]), mats[2])
    return full


def Uq(kv, q, eta=1.0):
    M = np.eye(32, dtype=complex)
    for axis, ((d, C), k) in enumerate(zip(DESIGN, kv)):
        t = Tq_axis(k, d, C, q, axis, eta)
        M = t @ t @ M
    return M


def mc_validate(q, T, E, seed=7):
    """direct quenched Monte Carlo of the signed coin walk vs Uq^T contraction"""
    r = np.random.default_rng(seed)
    Cs = [LI, LJ, LK]; ds = [D2, D3, D1]
    perm = [np.argmax(np.abs(C), axis=0) for C in Cs]
    colsgn = [C[np.argmax(np.abs(C), axis=0), np.arange(4)] for C in Cs]
    kv = np.array([0.3, -0.2, 0.5])
    G = np.zeros((4, 4), complex)
    for s0 in range(4):
        s = np.full(E, s0); sign = np.ones(E); X = np.zeros((E, 3))
        c = (r.random((E, 3)) < q).astype(int)
        for _ in range(T):
            for axis in (0, 1, 2):
                for _sub in range(2):
                    b = (r.random(E) < q).astype(int)
                    mv = b == c[:, axis]
                    X[mv, axis] += ds[axis][s[mv]]
                    cv = ~mv
                    sign[cv] *= colsgn[axis][s[cv]]
                    s[cv] = perm[axis][s[cv]]
                    c[cv, axis] = b[cv]
        ph = sign * np.exp(1j * (X @ kv))
        for sf in range(4):
            G[sf, s0] = ph[s == sf].sum() / E
    Um = np.linalg.matrix_power(Uq(kv, q), T)
    pi = np.array([1 - q, q])
    Cin = np.kron(np.eye(4), np.kron(np.kron(pi, pi), pi).reshape(8, 1))
    Cout = np.kron(np.eye(4), np.ones((1, 8)))
    G_ex = Cout @ Um @ Cin
    return float(np.abs(G - G_ex).max()), float(np.abs(G_ex).max())


def part_quenched(rng, quick=False):
    res = {}
    dirs3 = [np.array(d, float) / np.linalg.norm(d)
             for d in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]]

    if not quick:
        err, scale = mc_validate(0.2, T=2, E=2_000_000)
        res["mc_validation"] = dict(q=0.2, cycles=2, ensemble=2_000_000,
                                    max_err=err, scale=scale)

    # pole census at eta=1: per-sub-step modulus law and diamond multiplets
    census = {}
    for q in (0.10, 0.20, 0.30):
        w0 = np.linalg.eigvals(Uq(np.zeros(3), q))
        mods = np.sort(np.unique(np.round(np.abs(w0), 9)))[::-1]
        groups = multiplet_slopes(lambda kv: Uq(kv, q), np.zeros(3), dirs3)
        fams = []
        for lam, m, diams, _ in groups:
            if lam.imag <= 1e-9 or diams[0] < 1e-9:
                continue
            fams.append(dict(lam=[lam.real, lam.imag], mult=int(m),
                             r110=float(diams[1] / diams[0]),
                             r111=float(diams[2] / diams[0])))
        census[str(q)] = dict(
            top_moduli=[float(x) for x in mods[:4]],
            law_2q1mq_cubed=float((2 * q * (1 - q))**3),
            annealed_rho6=float(((1 - q)**2 + q**2)**3),
            families=fams)
    res["eta1_census"] = census
    # the diamond signature: sqrt(2) and 2/sqrt(3)
    res["diamond_reference"] = dict(r110=float(np.sqrt(2)), r111=float(2 / np.sqrt(3)))

    # off-zero BZ scan at eta=1: degeneracies form extended manifolds
    q = 0.2
    grid = np.linspace(-np.pi, np.pi, 7, endpoint=False)
    face_pts = 0; diag_pts = 0; other_pts = 0; tot_deg = 0
    for kx in grid:
        for ky in grid:
            for kz in grid:
                kv = np.array([kx, ky, kz])
                if np.abs(kv).max() < 1e-12:
                    continue
                w = np.linalg.eigvals(Uq(kv, q))
                w = w[np.abs(w) > 0.3 * np.abs(w).max()]
                mind = min(abs(a - b) for a, b in itertools.combinations(w, 2))
                if mind < 1e-12:
                    tot_deg += 1
                    if np.abs(np.abs(kv) - np.pi).min() < 1e-9:
                        face_pts += 1
                    elif (abs(abs(kv[0]) - abs(kv[1])) < 1e-9
                          and abs(abs(kv[0]) - abs(kv[2])) < 1e-9):
                        diag_pts += 1
                    else:
                        other_pts += 1
    res["eta1_offzero_degeneracies"] = dict(
        grid="7^3", count=tot_deg, on_pi_faces=face_pts,
        on_body_diagonals=diag_pts, elsewhere=other_pts,
        note="all found degeneracies lie on extended manifolds (pi-faces and "
             "BZ body diagonals = the diamond's chiral-plane crossing loci), "
             "no isolated point nodes")

    # companion-memory interpolation: exact cone for all eta < 1
    interp = {}
    for q in (0.10, 0.20):
        rows = []
        for eta in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
            groups = multiplet_slopes(lambda kv: Uq(kv, q, eta), np.zeros(3), dirs3)
            doms = [g for g in groups if g[0].imag > 1e-9 and g[2][0] > 1e-9]
            doms.sort(key=lambda g: -abs(g[0]))
            lam, m, diams, _ = doms[0]
            rows.append(dict(eta=eta, lam=[lam.real, lam.imag], mult=int(m),
                             diam100=float(diams[0]),
                             r110=float(diams[1] / diams[0]),
                             r111=float(diams[2] / diams[0])))
        interp[str(q)] = rows
    res["memory_interpolation"] = interp

    # eta=0 equals the annealed model at p = 2q(1-q) (naive parity rate), exactly
    q = 0.2
    p = 2 * q * (1 - q)
    A = 1 - 2 * p; B = 2 * p * (1 - p)
    lam_cf = (A**3 + B**3) + 1j * A * B * np.sqrt(3 * A**2 - 2 * A * B + 3 * B**2)
    groups = multiplet_slopes(lambda kv: Uq(kv, q, 0.0), np.zeros(3), dirs3)
    lam_q0 = max((g[0] for g in groups if g[0].imag > 0), key=abs)
    res["eta0_equals_annealed_parity_rate"] = dict(
        q=q, p=p, lam_eta0=[lam_q0.real, lam_q0.imag],
        lam_annealed_closed_form=[lam_cf.real, lam_cf.imag],
        dev=float(abs(lam_q0 - lam_cf)))
    return res


# ----------------------------------------------------------------------------
# Part 5 (addendum): the layered static-scatterer model  T_a = E_a . [(1-q)+qC_a]
# ----------------------------------------------------------------------------

# main-thread tables (ADR 0011 / w3c): tensor basis
XZ_ = SX @ SZ
CS_T = [np.kron(XZ_, I2), np.kron(SZ, XZ_), -np.kron(SX, XZ_)]
DS_T = [np.diag(np.kron(SZ, I2)), np.diag(np.kron(SZ, SZ)), np.diag(np.kron(I2, SZ))]
# equivalent quaternion-basis design: the note's winner with C_z -> -L_k
DESIGN_MULT = [(D2, LI), (D3, LJ), (D1, -LK)]


def U_mult(kv, q, CS=None, DS=None):
    """layered cycle: per sub-step convert-then-move, U = Tz^2 Ty^2 Tx^2"""
    if CS is None:
        CS = [c for _, c in DESIGN_MULT]
        DS = [d for d, _ in DESIGN_MULT]
    M = np.eye(4, dtype=complex)
    for a in range(3):
        t = np.diag(np.exp(1j * kv[a] * np.array(DS[a]))) @ (
            (1 - q) * np.eye(4) + q * CS[a])
        M = t @ t @ M
    return M


def lam0_mult(q):
    """closed form at the corner node (anti-cyclic triple: A^3 - B^3 family)"""
    A = 1 - 2 * q
    B = 2 * q * (1 - q)
    return (A**3 - B**3) + 1j * A * B * np.sqrt(3 * A**2 + 2 * A * B + 3 * B**2)


def part_layered(rng):
    res = {}
    q = 0.15

    # (0) model identities
    assert np.allclose(CS_T[0] @ CS_T[1], -CS_T[2])          # anti-cyclic triple
    ks = [rng.uniform(-1, 1, 3) for _ in range(4)]
    match = all(np.allclose(
        np.sort_complex(np.linalg.eigvals(U_mult(k, q))),
        np.sort_complex(np.linalg.eigvals(U_mult(k, q, CS_T, DS_T)))) for k in ks)
    res["quaternion_basis_equivalent"] = bool(match)
    kv = np.array([0.3, -0.7, 1.1])
    res["pi_periodicity_dev"] = float(max(
        np.abs(U_mult(kv + np.pi * np.eye(3)[a], q) - U_mult(kv, q)).max()
        for a in range(3)))
    w0 = np.linalg.eigvals(U_mult(np.zeros(3), q))
    lam0 = w0[np.argmax(w0.imag)]
    res["lam0_closed_form_dev"] = float(abs(lam0 - lam0_mult(q)))
    res["lam0_q015"] = [float(lam0.real), float(lam0.imag)]
    res["mod_lam0_equals_rho6_dev"] = float(abs(abs(lam0) - ((1 - q)**2 + q**2)**3))

    # (1) cone table: isotropy, v_omega(q), damping, cone bound
    dirs = direction_set(rng)
    table = []
    for qq in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        groups = multiplet_slopes(lambda kv: U_mult(kv, qq), np.zeros(3), dirs)
        g = [gg for gg in groups if gg[0].imag > 0][0]
        lam, mult, diams, Ms = g
        mu = np.linalg.eigvals(Ms[0])
        z = (mu[0] - mu[1]) / 2 / lam
        vo, vg = abs(z.imag), abs(z.real)
        assert vo <= 2.0 + 1e-9
        table.append(dict(q=qq, lam0=[lam.real, lam.imag],
                          lam0_closed_form_dev=float(abs(lam - lam0_mult(qq))),
                          mult=int(mult), aniso=float(diams.max() / diams.min() - 1),
                          v_omega=float(vo), v_gamma=float(vg)))
    res["cone_table"] = table
    res["max_aniso_over_q"] = float(max(t["aniso"] for t in table))
    res["max_v_gamma_over_q"] = float(max(t["v_gamma"] for t in table))

    # (2) symbolic isotropy proof (left-factor 2x2 with insertion collapse)
    import sympy as sp
    p = sp.symbols('q', positive=True)
    I = sp.I
    s1 = sp.Matrix([[0, 1], [1, 0]]); s2 = sp.Matrix([[0, -I], [I, 0]])
    s3 = sp.Matrix([[1, 0], [0, -1]])
    # t_x = (1-q) - i q s1, t_y = (1-q) - i q s2, t_z = (1-q) + i q s3 (-L_k)
    tx = (1 - p) * sp.eye(2) - I * p * s1
    ty = (1 - p) * sp.eye(2) - I * p * s2
    tz = (1 - p) * sp.eye(2) + I * p * s3
    G = sp.expand(tz * tz * ty * ty * tx * tx)
    avec = [sp.simplify(sp.trace(G * s) / 2 * I) for s in (s1, s2, s3)]
    r2 = sp.simplify(sum(x**2 for x in avec))
    nsig = (avec[0] * s1 + avec[1] * s2 + avec[2] * s3) / sp.sqrt(r2)
    P = (sp.eye(2) + nsig) / 2
    # dT_a^2|0 (left) = -2i(1-q) sigma_{v_a} t_a inserted at the axis slot,
    # v = (j, k, i) -> (s2, s3, s1)
    cx = sp.trace(P * (-2 * I * (1 - p) * tz * tz * ty * ty * s2 * tx))
    cy = sp.trace(P * (-2 * I * (1 - p) * tz * tz * s3 * ty * tx * tx))
    cz = sp.trace(P * (-2 * I * (1 - p) * s1 * tz * ty * ty * tx * tx))
    dxy = sp.simplify(sp.expand(cx**2 - cy**2))
    dxz = sp.simplify(sp.expand(cx**2 - cz**2))
    res["symbolic_isotropy"] = dict(
        cx2_minus_cy2=str(dxy), cx2_minus_cz2=str(dxz),
        isotropy_proven=bool(dxy == 0 and dxz == 0))
    # derived slope vs measured
    cx_num = complex(cx.subs(p, sp.Rational(3, 20)))
    g15 = [t for t in table if abs(t["q"] - 0.15) < 1e-12][0]
    slope_meas = 2 * g15["v_omega"] * abs(lam0_mult(0.15))
    res["symbolic_isotropy"]["slope_2cx_q015"] = float(2 * abs(cx_num))
    res["symbolic_isotropy"]["slope_measured_q015"] = float(slope_meas)

    # small-q limit of v_omega
    vo_small = []
    for qq in (0.01, 0.005, 0.002):
        g = [gg for gg in multiplet_slopes(lambda kv: U_mult(kv, qq),
                                           np.zeros(3), [np.array([1., 0, 0])])
             if gg[0].imag > 0][0]
        mu = np.linalg.eigvals(g[3][0])
        vo_small.append(float(abs(((mu[0] - mu[1]) / 2 / g[0]).imag)))
    res["v_omega_small_q"] = vo_small                     # -> 2/sqrt(3)

    # (3) reduced-zone node census with helicities (+Im branch), q = 0.15
    sigma = [SX.astype(complex), SY, SZ.astype(complex)]
    census = []
    for kc in itertools.product((0.0, np.pi / 2), repeat=3):
        k0 = np.array(kc)
        U0 = U_mult(k0, q)
        w0, V = np.linalg.eig(U0)
        Vinv = np.linalg.inv(V)
        eps = 1e-6
        dUs = []
        for a in range(3):
            kp = k0.copy(); kp[a] += eps
            km = k0.copy(); km[a] -= eps
            dUs.append((U_mult(kp, q) - U_mult(km, q)) / (2 * eps))
        for lam, idx in degenerate_pairs(w0):
            if lam.imag < -1e-9 or len(idx) < 2:
                continue
            entry = dict(k=[float(x) for x in kc], lam=[lam.real, lam.imag],
                         mult=len(idx))
            if len(idx) == 2:
                R = V[:, idx]; L = Vinv[idx, :]
                Ms = [L @ dU @ R for dU in dUs]
                diams = []
                for n in direction_set(rng, 2):
                    mu = np.linalg.eigvals(n[0] * Ms[0] + n[1] * Ms[1] + n[2] * Ms[2])
                    diams.append(abs(mu[0] - mu[1]))
                diams = np.array(diams)
                Bm = np.zeros((3, 3), complex)
                for a in range(3):
                    h = (-1j) * Ms[a] / lam
                    h = h - np.trace(h) / 2 * np.eye(2)
                    Bm[a, :] = [np.trace(h @ s) / 2 for s in sigma]
                det = np.linalg.det(Bm)
                entry["aniso"] = float(diams.max() / max(diams.min(), 1e-30) - 1)
                entry["chi"] = int(np.sign(det.real)) if abs(det) > 1e-9 else 0
                entry["detB"] = [float(det.real), float(det.imag)]
            census.append(entry)
    res["reduced_zone_census_q015"] = census
    point_chis = [c["chi"] for c in census if c.get("chi") is not None and c["chi"] != 0]
    res["point_node_chi_sum_gamma_and_X"] = int(sum(point_chis))

    # (4) Fock-lift legality of the layers
    res["fock_lift"] = fock_lift_checks(q)
    return res


def fock_lift_checks(q):
    """L2 = env-controlled Givens exp((pi/2)(a_c+ a_c' - a_c'+ a_c)); verify the
    Fock-level statements and that the composite circuit's single-particle
    sector equals T_a(k) after env-averaging."""
    from scipy.linalg import expm
    out = {}

    def fermion_ops(nmodes):
        sz = 2**nmodes
        ops = []
        for m in range(nmodes):
            mats = []
            for j in range(nmodes):
                if j < m:
                    mats.append(SZ)          # JW string
                elif j == m:
                    mats.append(np.array([[0, 1], [0, 0.]]))   # annihilator
                else:
                    mats.append(I2)
            M = mats[0]
            for x in mats[1:]:
                M = np.kron(M, x)
            ops.append(M)
        return ops

    # 2-mode Givens
    a, b = fermion_ops(2)
    G = expm((np.pi / 2) * (a.T.conj() @ b - b.T.conj() @ a))
    v10 = a.T.conj() @ np.eye(4)[:, 0]      # |10>
    v01 = b.T.conj() @ np.eye(4)[:, 0]      # |01>
    v11 = a.T.conj() @ b.T.conj() @ np.eye(4)[:, 0]
    sp_block = np.array([[v10 @ G @ v10, v10 @ G @ v01],
                         [v01 @ G @ v10, v01 @ G @ v01]]).T
    Npar = a.T.conj() @ a + b.T.conj() @ b
    parity = expm(1j * np.pi * Npar).real
    out["givens_single_particle_block"] = np.round(sp_block.real, 12).tolist()
    out["givens_11_eigenvalue"] = float((v11 @ G @ v11).real)
    out["givens_vacuum_fixed"] = float(G[0, 0].real)
    out["givens_parity_even"] = float(np.abs(G @ parity - parity @ G).max())

    # controlled Givens on (a, b, env): acts iff n_env = 1; parity even; local
    a3, b3, e3 = fermion_ops(3)
    ne = e3.T.conj() @ e3
    Gc = expm((np.pi / 2) * ne @ (a3.T.conj() @ b3 - b3.T.conj() @ a3))
    # sector checks: with env empty -> identity on carriers
    P0e = np.eye(8) - ne
    dev_id = np.abs(P0e @ (Gc - np.eye(8)) @ P0e).max()
    N3 = a3.T.conj() @ a3 + b3.T.conj() @ b3 + ne
    par3 = expm(1j * np.pi * N3).real
    out["controlled_identity_when_env_empty"] = float(dev_id)
    out["controlled_parity_even"] = float(np.abs(Gc @ par3 - par3 @ Gc).max())
    out["controlled_is_signed_permutation"] = bool(
        np.allclose(np.abs(np.abs(Gc.real) - np.round(np.abs(Gc.real))), 0,
                    atol=1e-12) and np.abs(Gc.imag).max() < 1e-12)

    # C_a decomposition into two disjoint +-90-degree Givens blocks
    dec = {}
    for name, C in zip("xyz", CS_T):
        P = np.abs(C)
        pairs = []
        seen = set()
        for c in range(4):
            c2 = int(np.argmax(P[:, c]))
            if c not in seen:
                pairs.append((c, c2))
                seen |= {c, c2}
        ok = all(int(np.argmax(P[:, c2])) == c for c, c2 in pairs)  # involution
        blocks_ok = True
        for c, c2 in pairs:
            blk = C[np.ix_([c, c2], [c, c2])]
            rot = np.array([[0, -1], [1, 0.]])
            blocks_ok &= (np.allclose(blk, rot) or np.allclose(blk, -rot))
        dec[name] = dict(pairs=pairs, involution=bool(ok),
                         plus_minus_90_givens=bool(blocks_ok),
                         det_blocks_plus1=bool(all(
                             abs(np.linalg.det(C[np.ix_([c, c2], [c, c2])]) - 1)
                             < 1e-12 for c, c2 in pairs)))
    out["conversion_blocks"] = dec

    # composite circuit single-particle sector on a ring: env-average of
    # (shift after conversion) has Bloch symbol E_a(k) [(1-q) + q C_a]
    L = 6
    devs = []
    for axis in range(3):
        C = CS_T[axis]; d = DS_T[axis]
        # single-particle space: site (x) channel, ordering |x, c>
        conv_mean = np.kron(np.eye(L), (1 - q) * np.eye(4) + q * C)
        S = np.zeros((4 * L, 4 * L))
        for x in range(L):
            for c in range(4):
                S[4 * ((x + int(d[c])) % L) + c, 4 * x + c] = 1.0
        M = S @ conv_mean
        for m in range(L):
            k = 2 * np.pi * m / L
            psi = np.zeros((L, 4), complex)
            # plane waves per channel; symbol in channel basis
            sym = np.zeros((4, 4), complex)
            # Fourier convention psi_k(x) = e^{-ikx}: shift x -> x + d then has
            # symbol e^{+ik d}, matching E_a(k) = diag(e^{i k d_a})
            for c in range(4):
                v = np.zeros(4 * L, complex)
                for x in range(L):
                    v[4 * x + c] = np.exp(-1j * k * x)
                w = M @ v
                for c2 in range(4):
                    sym[c2, c] = np.array(
                        [w[4 * x + c2] * np.exp(1j * k * x) for x in range(L)]
                    ).sum() / L
            T_expected = np.diag(np.exp(1j * k * np.array(d))) @ (
                (1 - q) * np.eye(4) + q * C)
            devs.append(np.abs(sym - T_expected).max())
    out["ring_symbol_equals_Ta_maxdev"] = float(max(devs))
    return out


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the n2 off-grid refinement and the MC validation")
    args = ap.parse_args()

    out = {"seed": SEED, "quick": args.quick}
    t0 = time.time()
    for name, fn in [
        ("n2_algebra", lambda: part_n2_algebra(np.random.default_rng(SEED))),
        ("n2_scan", lambda: part_n2_scan(np.random.default_rng(SEED), args.quick)),
        ("n4_construction", lambda: part_n4_construction(np.random.default_rng(SEED))),
        ("n4_symbolic", part_n4_symbolic),
        ("mass", lambda: part_mass(np.random.default_rng(SEED))),
        ("quenched", lambda: part_quenched(np.random.default_rng(SEED), args.quick)),
        ("layered", lambda: part_layered(np.random.default_rng(SEED))),
    ]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:7.1f}s] {name} done ({out[name + '_seconds']}s)")


    # headline assertions (fail loudly if the theory note's claims break)
    assert out["n2_algebra"]["signature_of_N_on_sl2R"] == (2, 1)
    assert not out["n2_algebra"]["ppp_triple_found"]
    assert out["n2_scan"]["min_aniso_omega"] > 1.0
    assert out["n4_construction"]["max_aniso_over_p"] < 1e-9
    assert out["n4_construction"]["max_v_gamma_over_p"] < 1e-9
    assert out["n4_construction"]["chirality_sum"] == 0
    assert out["n4_symbolic"]["isotropy_proven"]
    assert out["mass"]["anticommutant"]["dim"] == 2
    assert out["mass"]["anticommutant"]["all_squares_negative"]
    assert out["mass"]["cl40_word_quadruples_4x4"] == []
    assert out["mass"]["cl40_word_quadruple_8x8"] is not None
    assert out["quenched"]["eta1_offzero_degeneracies"]["elsewhere"] == 0
    lay = out["layered"]
    assert lay["quaternion_basis_equivalent"]
    assert lay["pi_periodicity_dev"] < 1e-12
    assert lay["max_aniso_over_q"] < 1e-9 and lay["max_v_gamma_over_q"] < 1e-9
    assert lay["symbolic_isotropy"]["isotropy_proven"]
    assert lay["fock_lift"]["givens_11_eigenvalue"] == 1.0
    assert lay["fock_lift"]["givens_parity_even"] < 1e-12
    assert lay["fock_lift"]["ring_symbol_equals_Ta_maxdev"] < 1e-12
    for rows in out["quenched"]["memory_interpolation"].values():
        for r in rows:
            if r["eta"] < 1.0:
                assert abs(r["r110"] - 1) < 1e-6 and abs(r["r111"] - 1) < 1e-6
            else:
                assert abs(r["r110"] - np.sqrt(2)) < 1e-3
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
