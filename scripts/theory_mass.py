#!/usr/bin/env python
"""M8 theory: exact structure of the inversion-doubled massive coincube.

Machine checks for the mass-sector theory (all headline claims re-asserted
at the end of every run). Results -> results/theory_mass.json.

Model (main thread, scripts/m8_mass_exact.py): 8 channels = coin (x) mass bit;
  C8_a = C_a (x) 1  (conversions blind to b_m),
  D8_a = d_a (x) diag(1,-1)  (b_m = 1 = spatial-inversion sector),
  M = (1-q_m) + q_m C_m,  C_m = 1 (x) XZ,
  U(k) = M . Prod_a [ E_a(k) ((1-q) + q C8_a) ]^2.

Central exact identities (all machine-checked to ~1e-16):
  (I)  U(k) = (1 (x) R_m) . [ U4(k) (+) U4(-k) ]      (b_m block form)
  (II) U(0) = U4(0) (x) M2,  M2 = [[1-q_m, -q_m], [q_m, 1-q_m]] = r R(theta),
       r = sqrt((1-q_m)^2 + q_m^2),  theta = arctan(q_m / (1-q_m))
  => spectrum at k=0 is {lam4} x {r e^{+-i theta}}: gap EXACTLY
     2 arctan(q_m/(1-q_m)), centre omega_0(q) unshifted, damping x r.
  The same factorization holds at every k with all components in {0, pi/2}
  (mod pi), because there -k == k (mod pi) and U is pi-periodic.

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_mass.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_mass.json"

I2 = np.eye(2)
X = np.array([[0.0, 1], [1, 0]])
Z = np.diag([1.0, -1])
XZ = X @ Z
C4 = [np.kron(XZ, I2), np.kron(Z, XZ), -np.kron(X, XZ)]
D4 = [np.diag(np.kron(Z, I2)), np.diag(np.kron(Z, Z)), np.diag(np.kron(I2, Z))]
C8 = [np.kron(c, I2) for c in C4]
D8 = [np.kron(d, [1.0, -1]) for d in D4]
CM = np.kron(np.eye(4), XZ)
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]


def U4(kv, q, flip=False):
    u = np.eye(4, dtype=complex)
    for a in range(3):
        C = -C4[a] if flip else C4[a]
        t = np.diag(np.exp(1j * kv[a] * D4[a])) @ ((1 - q) * np.eye(4) + q * C)
        u = t @ t @ u
    return u


def mass_layer(theta, scale=1.0):
    """scale * (cos theta + sin theta C_m) -- a legal annealed layer for any
    theta in (0, pi/2): equals gamma [(1-q')I + q'C_m] with tan theta =
    q'/(1-q')."""
    return scale * (np.cos(theta) * np.eye(8) + np.sin(theta) * CM)


def U8(kv, q, qm, placement="end"):
    th = np.arctan(qm / (1 - qm))
    rm = np.hypot(1 - qm, qm)
    axes = []
    for a in range(3):
        t = np.diag(np.exp(1j * kv[a] * D8[a])) @ ((1 - q) * np.eye(8) + q * C8[a])
        axes.append(t)
    u = np.eye(8, dtype=complex)
    if placement == "end":
        for a in range(3):
            u = axes[a] @ axes[a] @ u
        u = rm * mass_layer(th) @ u
    elif placement == "sym":
        u = mass_layer(th / 2) @ u
        for a in range(3):
            u = axes[a] @ axes[a] @ u
        u = rm * mass_layer(th / 2) @ u
    elif placement == "interleave6":
        for a in range(3):
            u = mass_layer(th / 6) @ axes[a] @ u
            u = mass_layer(th / 6) @ axes[a] @ u
        u = rm * u
    return u


def groups_of(w, tol=1e-9):
    gs = []
    for i, z in enumerate(w):
        for g in gs:
            if abs(z - g[0]) < tol:
                g[1].append(i)
                break
        else:
            gs.append([z, [i]])
    return gs


def eig_set_dev(A, B):
    wa = list(np.linalg.eigvals(A))
    d = 0.0
    for x in np.linalg.eigvals(B):
        i = int(np.argmin(np.abs(np.array(wa) - x)))
        d = max(d, abs(wa[i] - x))
        wa.pop(i)
    return d


def massless_v(q):
    eps = 1e-6
    U0 = U4(np.zeros(3), q)
    w, V = np.linalg.eig(U0)
    Vi = np.linalg.inv(V)
    up = [i for i in range(4) if w[i].imag > 0]
    dU = (U4(np.array([eps, 0, 0]), q) - U4(np.array([-eps, 0, 0]), q)) / (2 * eps)
    mu = np.linalg.eigvals(Vi[up, :] @ dU @ V[:, up])
    return abs(((mu[0] - mu[1]) / 2 / w[up[0]]).imag)


# ----------------------------------------------------------------------------

def part_identities(rng):
    """(I), (II) and the exact mass law as operator identities"""
    res = {}
    q, qm = 0.08, 0.05
    Rm = (1 - qm) * I2 + qm * XZ
    P0, P1 = np.diag([1.0, 0]), np.diag([0.0, 1])
    dev = 0.0
    for _ in range(6):
        k = rng.uniform(-2, 2, 3)
        B = np.kron(U4(k, q), P0) + np.kron(U4(-k, q), P1)
        dev = max(dev, np.abs(U8(k, q, qm) - np.kron(np.eye(4), Rm) @ B).max())
    res["block_identity_dev"] = float(dev)
    res["k0_tensor_dev"] = float(np.abs(
        U8(np.zeros(3), q, qm) - np.kron(U4(np.zeros(3), q), Rm)).max())
    # sympy: eigenvalues of M2 are (1-qm) +- i qm = r e^{+-i theta}
    import sympy as sp
    t = sp.symbols('q_m', positive=True)
    M2 = sp.Matrix([[1 - t, -t], [t, 1 - t]])
    ev = M2.eigenvals()
    res["mass_block_eigs"] = sorted(str(e) for e in ev)
    ok = set(sp.simplify(e - ((1 - t) + s * sp.I * t)) == 0
             for e, s in zip(ev, (1, -1))) | set(
        sp.simplify(e - ((1 - t) - s * sp.I * t)) == 0
        for e, s in zip(ev, (1, -1)))
    res["mass_block_eigs_verified"] = bool(True in ok)
    # exact mass law across (q, qm): the operator identity U8(0) = U4(0) (x) M2
    # implies spectrum {lam4} x {r e^{+-i theta}} with no further input
    dev_max = 0.0
    for qq in (0.05, 0.08, 0.15, 0.25, 0.35):
        for qmm in (0.01, 0.05, 0.1, 0.2):
            Rm2 = (1 - qmm) * I2 + qmm * XZ
            dev_max = max(dev_max, np.abs(
                U8(np.zeros(3), qq, qmm)
                - np.kron(U4(np.zeros(3), qq), Rm2)).max())
    res["mass_law_spectrum_dev_max"] = float(dev_max)
    res["mass_law_note"] = ("spectrum at k=0 is exactly {lam4} x {r e^+-itheta}:"
                            " gap 2 arctan(qm/(1-qm)), centre omega0(q)"
                            " unshifted, damping x r -- independent of q; when"
                            " theta > omega0 the naive Im>0 frequency filter"
                            " wraps (bookkeeping, not physics)")
    # C (x) Z doubling fails: -C flips the triple orientation (cyclic <->
    # anti-cyclic), i.e. the OTHER frequency family -- different omega0, so
    # the two sectors would not share a node
    qf = 0.2
    w = np.linalg.eigvals(U4(np.zeros(3), qf, flip=True))
    lamf = w[np.argmax(w.imag)]
    w0 = np.linalg.eigvals(U4(np.zeros(3), qf))
    lam0 = w0[np.argmax(w0.imag)]
    res["minusC_is_other_family"] = dict(
        arg_C=float(np.angle(lam0)), arg_minusC=float(np.angle(lamf)),
        distinct=bool(abs(np.angle(lam0) - np.angle(lamf)) > 0.05))
    return res


def part_clifford(rng):
    """Dirac-Clifford on the 4-dim node space, tensor-structurally, per q"""
    res = {"per_q": []}
    for q in (0.05, 0.15, 0.3):
        U0 = U8(np.zeros(3), q, 0.0)
        w, V = np.linalg.eig(U0)
        Vi = np.linalg.inv(V)
        up = [i for i in range(8) if w[i].imag > 0]
        lam0 = np.mean(w[up])
        assert max(abs(w[i] - lam0) for i in up) < 1e-9 and len(up) == 4
        R = V[:, up]
        L = Vi[up, :]
        h = 1e-5
        hs = []
        for a in range(3):
            e = np.zeros(3); e[a] = h
            dU = (U8(e, q, 0.0) - U8(-e, q, 0.0)) / (2 * h)
            hs.append((-1j / lam0) * (L @ dU @ R))
        dUm = (U8(np.zeros(3), q, h) - U8(np.zeros(3), q, -h)) / (2 * h)
        hm = (-1j / lam0) * (L @ dUm @ R)
        hm = hm - np.trace(hm) / 4 * np.eye(4)
        devs = []
        v2 = None
        for a in range(3):
            for b in range(a, 3):
                ac = hs[a] @ hs[b] + hs[b] @ hs[a]
                if a == b:
                    sq = ac / 2
                    devs.append(np.abs(sq - sq[0, 0] * np.eye(4)).max())
                    v2 = sq[0, 0].real if v2 is None else v2
                    devs.append(abs(sq[0, 0] - v2))
                else:
                    devs.append(np.abs(ac).max())
            devs.append(np.abs(hs[a] @ hm + hm @ hs[a]).max())
        hm2 = hm @ hm
        devs.append(np.abs(hm2 - hm2[0, 0] * np.eye(4)).max())
        mu = np.sqrt(hm2[0, 0].real)
        res["per_q"].append(dict(q=q, v=float(np.sqrt(v2)), mu=float(mu),
                                 mu_minus_1=float(abs(mu - 1)),
                                 max_algebra_dev=float(max(devs))))
    res["structure"] = ("h_a = h_a^(4) (x) sigma_z^(m) (opposite sector"
                        " velocities), h_m = 1 (x) sigma_y^(m)-type from"
                        " C_m projection; {h_a,h_b} = 2v^2 delta_ab from the"
                        " massless c_a^2-equality (sympy-proven in"
                        " theory_real_cone), {h_a,h_m} = 0 and h_m^2 = 1 from"
                        " the Pauli algebra of the b_m factor -- identically"
                        " in q")
    res["mu_exactly_1"] = bool(all(r["mu_minus_1"] < 1e-6 for r in res["per_q"]))
    return res


def part_dispersion(rng):
    """omega(k) = omega_c +- sqrt(m^2 + v^2 k^2) at leading order; the exact
    composition law cos(omega - omega_c) = cos(m) cos(phi(k)); corrections."""
    res = {}
    q, qm = 0.08, 0.05
    m = np.arctan(qm / (1 - qm))
    v = massless_v(q)
    w4 = np.linalg.eigvals(U4(np.zeros(3), q))
    om0 = np.angle(w4[np.argmax(w4.imag)])
    u_dir = np.array([0.267, 0.535, 0.802])
    u_dir /= np.linalg.norm(u_dir)

    def branch_dw(kv, qmv):
        lams = np.linalg.eigvals(U8(kv, q, qmv))
        return sorted(np.angle(l) for l in lams if l.imag > 0.02)[-1] - om0

    def phi_exact(kv):
        lams4 = np.linalg.eigvals(U4(kv, q))
        return sorted(np.angle(l) for l in lams4 if l.imag > 0.02)[-1] - om0

    rows = []
    for s in (0.02, 0.05, 0.1, 0.15):
        dw = branch_dw(s * u_dir, qm)
        rows.append(dict(k=s, dw=float(dw),
                         dev_sqrt=float(dw - np.sqrt(m**2 + (v * s)**2)),
                         dev_cos_vk=float(dw - np.arccos(np.cos(m) * np.cos(v * s))),
                         dev_cos_phi=float(dw - np.arccos(np.cos(m)
                                                          * np.cos(phi_exact(s * u_dir))))))
    res["dispersion_rows"] = rows
    res["cos_phi_absorbs"] = bool(all(abs(r["dev_cos_phi"]) < 0.05 * abs(r["dev_sqrt"])
                                      for r in rows if r["k"] >= 0.1))

    # scaling at fixed vk/m: absolute sqrt-deviation is O(qm^2) (relative O(qm))
    scal = []
    for qm_i in (0.1, 0.05, 0.025, 0.0125):
        m_i = np.arctan(qm_i / (1 - qm_i))
        s = 1.1 * m_i / v
        dw = branch_dw(s * u_dir, qm_i)
        scal.append(dict(qm=qm_i,
                         rel_dev_sqrt=float(abs(dw - np.sqrt(m_i**2 + (v * s)**2)) / m_i),
                         abs_dev_sqrt=float(abs(dw - np.sqrt(m_i**2 + (v * s)**2)))))
    res["fixed_ratio_scaling"] = scal
    r_last = scal[-1]["rel_dev_sqrt"] / scal[-2]["rel_dev_sqrt"]
    res["rel_dev_halving_ratio"] = float(r_last)      # ~0.5: relative linear

    # direction spread at fixed |k| is O(k^2) (curvature, not anisotropy)
    spreads = {}
    for s in (0.005, 0.01, 0.02):
        oms = []
        rng_d = np.random.default_rng(17)
        for _ in range(8):
            uu = rng_d.normal(size=3); uu /= np.linalg.norm(uu)
            oms.append(branch_dw(s * uu, qm))
        spreads[s] = (max(oms) - min(oms)) / np.mean(oms)
    res["direction_spread"] = {str(k): float(vv) for k, vv in spreads.items()}
    res["spread_k2_ratio"] = float(spreads[0.01] / spreads[0.005])

    # Trotter question: symmetrized placement is EXACTLY spectrum-neutral
    devs_sym, devs_int = [], []
    for _ in range(4):
        k = rng.uniform(-1.5, 1.5, 3)
        A = U8(k, q, qm, "end")
        devs_sym.append(eig_set_dev(A, U8(k, q, qm, "sym")))
        devs_int.append(eig_set_dev(A, U8(k, q, qm, "interleave6")))
    res["symmetrized_spectrum_dev"] = float(max(devs_sym))     # ~1e-15
    res["interleaved_spectrum_dev"] = float(max(devs_int))     # genuinely != 0
    # interleaved: same exact mass law at k=0 (rotations compose), but no
    # improvement of the finite-k deviation
    comp = []
    for qm_i in (0.1, 0.05):
        m_i = np.arctan(qm_i / (1 - qm_i))
        s = 1.1 * m_i / v
        de = abs(sorted(np.angle(l) for l in
                        np.linalg.eigvals(U8(s * u_dir, q, qm_i, "end"))
                        if l.imag > 0.02)[-1] - om0 - np.sqrt(m_i**2 + (v * s)**2))
        di = abs(sorted(np.angle(l) for l in
                        np.linalg.eigvals(U8(s * u_dir, q, qm_i, "interleave6"))
                        if l.imag > 0.02)[-1] - om0 - np.sqrt(m_i**2 + (v * s)**2))
        comp.append(dict(qm=qm_i, dev_end=float(de), dev_interleaved=float(di)))
    res["interleave_comparison"] = comp
    return res


def part_census(rng):
    """massive node census, spectators, doubling degeneracy locus"""
    res = {}
    q, qm = 0.15, 0.05
    m = np.arctan(qm / (1 - qm))
    rho6 = ((1 - q)**2 + q**2)**3
    rm = np.hypot(1 - qm, qm)

    # factorization locus {0, pi/2}^3: every point U4(k) (x) M2 exactly
    dev = 0.0
    Rm2 = (1 - qm) * I2 + qm * XZ
    for kc in itertools.product((0.0, np.pi / 2), repeat=3):
        k0 = np.array(kc)
        dev = max(dev, np.abs(U8(k0, q, qm)
                              - np.kron(U4(k0, q), Rm2)).max())
    res["census_set_factorization_dev"] = float(dev)

    # gaps at Gamma and X exactly 2m
    gaps = {}
    for name, kc in (("Gamma", (0, 0, 0)), ("X", (np.pi / 2, 0, 0))):
        w = np.linalg.eigvals(U8(np.array(kc, float), q, qm))
        gs = [g for g in groups_of(w) if len(g[1]) == 2 and g[0].imag > 0]
        gs.sort(key=lambda g: -abs(g[0]))
        # the split quartet: the two 2-fold groups sharing max modulus
        mods = [abs(g[0]) for g in gs]
        pair = [g for g in gs if abs(abs(g[0]) - max(mods)) < 1e-9]
        gap = abs(np.angle(pair[0][0]) - np.angle(pair[1][0]))
        gaps[name] = float(abs(gap - 2 * m))
    res["gap_minus_2m"] = gaps

    # R point: 4-fold spectator at -rho6 rm e^{+-i theta}
    wR = np.linalg.eigvals(U8(np.array([np.pi / 2] * 3), q, qm))
    gsR = sorted(groups_of(wR), key=lambda g: -g[0].imag)
    predR = -rho6 * rm * np.exp(-1j * m)
    res["R_point"] = dict(mults=[len(g[1]) for g in gsR],
                          dev=float(min(abs(g[0] - predR) for g in gsR)))

    # nodal lines survive as 2-fold spectators
    kline = np.array([0.7, np.pi / 2, np.pi / 2])
    mults = sorted(len(g[1]) for g in groups_of(np.linalg.eigvals(U8(kline, q, qm))))
    res["line_mult_pattern"] = mults

    # doubling degeneracy: exact on zone edges (>= 2 components in {0, pi/2}),
    # O(k^2)-split at generic k
    def min_gap_p(kv):
        w = np.linalg.eigvals(U8(np.array(kv), q, qm))
        wp = w[w.imag > 0.02]
        return min(abs(a - b) for a, b in itertools.combinations(wp, 2))

    res["doubling_exact_on_edges"] = float(max(
        min_gap_p((0, 0, 0.9)), min_gap_p((0, 0.7, np.pi / 2)),
        min_gap_p((np.pi / 2, 0.3, np.pi / 2))))
    res["doubling_split_generic"] = dict(
        k_02=float(min_gap_p(0.2 * np.array([0.37, 0.65, 0.93])
                             / np.linalg.norm([0.37, 0.65, 0.93]))),
        k_04=float(min_gap_p(0.4 * np.array([0.37, 0.65, 0.93])
                             / np.linalg.norm([0.37, 0.65, 0.93]))))

    # no new gapless points, two-part statement:
    # (a) no exact degeneracies off the known loci: the minimum pairwise +Im
    #     gap over the grid (excluding points with >= 2 components in
    #     {0, pi/2}) is nonzero, and its argmin sits adjacent to a known
    #     exact-degeneracy edge (the doubling splitting -> 0 continuously
    #     approaching the edges -- within-band, not a mass-gap closing);
    grid = np.linspace(0, np.pi, 10, endpoint=False)
    floor_intra, argmin = 1e9, None
    for kx in grid:
        for ky in grid:
            for kz in grid:
                kv = np.array([kx, ky, kz])
                nspecial = np.sum((np.abs(kv - np.pi / 2) < 1e-9)
                                  | (np.abs(kv) < 1e-9))
                if nspecial >= 2:
                    continue
                g = min_gap_p(kv)
                if g < floor_intra:
                    floor_intra, argmin = g, kv
    dist_edge = sorted(min(abs(x), abs(x - np.pi / 2), abs(x - np.pi))
                       for x in argmin)[1]      # 2nd-smallest comp. distance
    res["min_pairwise_gap_off_spectator_loci"] = float(floor_intra)
    res["argmin_second_distance_to_special"] = float(dist_edge)
    # (b) the mass gap does not close near the (gapped) nodes: min frequency
    #     gap between the two split bands over |k| <= 0.3 around Gamma is 2m
    #     (attained at k = 0, sqrt law)
    wmin = 1e9
    for _ in range(40):
        uu = rng.normal(size=3)
        uu *= rng.uniform(0, 0.3) / np.linalg.norm(uu)
        w = np.linalg.eigvals(U8(uu, q, qm))
        phs = sorted(np.angle(l) for l in w if l.imag > 0.02)
        wmin = min(wmin, phs[-1] - phs[0])
    res["min_freq_gap_near_Gamma"] = float(wmin)
    res["scale_2m"] = float(2 * m)
    res["scale_2m_rho6"] = float(2 * m * rho6)

    # sector chiralities at Gamma and X: structural doubling chi -> -chi
    def sector_chi(q, k0, sector):
        ds = [d if sector == 0 else -d for d in D4]

        def u4s(kv):
            u = np.eye(4, dtype=complex)
            for a in range(3):
                t = np.diag(np.exp(1j * kv[a] * ds[a])) @ (
                    (1 - q) * np.eye(4) + q * C4[a])
                u = t @ t @ u
            return u

        U0 = u4s(k0)
        w, V = np.linalg.eig(U0)
        Vi = np.linalg.inv(V)
        up = [i for i in range(4) if w[i].imag > 0]
        lam0 = w[up[0]]
        R = V[:, up]
        L = Vi[up, :]
        h = 1e-5
        B = np.zeros((3, 3), complex)
        for a in range(3):
            e = np.array(k0, float); e2 = np.array(k0, float)
            e[a] += h; e2[a] -= h
            dU = (u4s(e) - u4s(e2)) / (2 * h)
            hp = (-1j / lam0) * (L @ dU @ R)
            hp = hp - np.trace(hp) / 2 * np.eye(2)
            B[a, :] = [np.trace(hp @ s) / 2 for s in PAULI]
        return int(np.sign(np.linalg.det(B).real))

    chis = {}
    for name, kc in (("Gamma", np.zeros(3)), ("X", np.array([np.pi / 2, 0, 0]))):
        chis[name] = [sector_chi(q, kc, 0), sector_chi(q, kc, 1)]
    res["sector_chiralities"] = chis
    res["pointwise_chi_net_zero"] = bool(all(c[0] == -c[1] for c in chis.values()))
    return res


# ----------------------------------------------------------------------------

def main():
    out = {"seed": SEED}
    t0 = time.time()
    for name, fn in [
        ("identities", lambda: part_identities(np.random.default_rng(SEED))),
        ("clifford", lambda: part_clifford(np.random.default_rng(SEED))),
        ("dispersion", lambda: part_dispersion(np.random.default_rng(SEED))),
        ("census", lambda: part_census(np.random.default_rng(SEED))),
    ]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.1f}s] {name} done")


    # headline assertions
    ident = out["identities"]
    assert ident["block_identity_dev"] < 1e-14
    assert ident["k0_tensor_dev"] < 1e-14
    assert ident["mass_law_spectrum_dev_max"] < 1e-12
    assert ident["minusC_is_other_family"]["distinct"]
    cl = out["clifford"]
    assert cl["mu_exactly_1"]
    assert all(r["max_algebra_dev"] < 1e-5 for r in cl["per_q"])
    disp = out["dispersion"]
    assert disp["symmetrized_spectrum_dev"] < 1e-12       # placement-neutral
    assert disp["interleaved_spectrum_dev"] > 1e-6        # genuinely different
    assert disp["cos_phi_absorbs"]
    assert 0.35 < disp["rel_dev_halving_ratio"] < 0.65    # relative-linear
    assert disp["spread_k2_ratio"] > 2.0
    cen = out["census"]
    assert cen["census_set_factorization_dev"] < 1e-14
    assert max(cen["gap_minus_2m"].values()) < 1e-12
    assert cen["R_point"]["dev"] < 1e-12
    assert cen["line_mult_pattern"] == [2, 2, 2, 2]
    assert cen["doubling_exact_on_edges"] < 1e-12
    assert cen["min_pairwise_gap_off_spectator_loci"] > 1e-8   # no new exact degeneracies
    assert cen["argmin_second_distance_to_special"] < 0.35     # argmin hugs a known edge
    assert cen["min_freq_gap_near_Gamma"] >= cen["scale_2m"] - 1e-9
    assert cen["pointwise_chi_net_zero"]
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
