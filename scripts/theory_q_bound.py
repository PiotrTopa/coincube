#!/usr/bin/env python
"""Q-bound theorem for quaternionic-event probabilistic CAs -- with its
exact scope boundary exhibited.

Self-contained machine checks; assertions run before the results file is
written; results -> results/theory_q_bound.json.

THEOREM (Q-pinning, rescoped). Let every conversion event lie in the
quaternionic event set R.Q8 = {+-1, +-C_x, +-C_y, +-C_z} (the coincube's
certified set; eigenvalue angles in (pi/2)Z). Then for ANY product of ANY
convex mixtures of events -- non-commuting factors and correlated weights
included -- every eigenvalue lambda of the composite obeys

    dist(arg lambda, (pi/2)Z) <= kappa * (-ln|lambda|),
    kappa = pi/(2 ln 2) = 2.26618...

Proof: products of mixtures over a group are mixtures over the group
(convolution), so the composite lies in the group algebra R[Q8]; in the
quaternionic representation T = w + x C_x + y C_y + z C_z with
|w| + l1-norm(v) <= 1; the eigenvalues are w +- i l2-norm(v), and l2 <= l1
places them inside the convex hull of {+-1, +-i}; the chord maximization
of the CORRECT quantity dist(arg lambda, (pi/2)Z)/(-ln|lambda|) -- using
the square-hull reduction and the pi/2 symmetry -- gives kappa exactly, at
the chord midpoint.  (The naive quantity arg lambda/(-ln|lambda|) is NOT
maximized at the midpoint: it diverges at the chord endpoints; part A
exhibits the divergence explicitly.)

SCOPE (transport-phase boundary, checked in part scope_k): the bound governs
momenta where the transport phases vanish -- the node.  At k != 0 the
factors E_a(k_a) T_a are not mixtures over R.Q8 and the bound FAILS for
the coincube operator itself (q = 0.02, k = (0.4, 0, 0): min-mode
dist/Gamma = 6.06 > kappa; part scope_k).  The phase-advance consequence
is therefore scoped to the node.

SCOPE BOUNDARY (checked here as a REPORTED counterexample): the theorem
fails for event sets not closed into R.Q8. Transpositions S12, S23 on >= 3
modes are signed permutations with angles {0, pi}, but their mixtures'
products develop 3-cycle eigenvalues near e^{2 pi i/3}: the ratio
dist/Gamma is ~4.9 at eps = 0.05, ~26 at eps = 0.01, unbounded as eps -> 0.
The bound is therefore a statement about the quaternionic class, not about
all signed-permutation event sets; for event sets with finest angle
2 pi/ell the chord constant grows as kappa(ell) ~ 2 ell/pi (tabulated).

Parts (all asserted):
  A. chord bound: kappa exact at the chord midpoint; 20000-mixture hull scan.
  B. products: 4000 random NON-COMMUTING products of mixtures over R.Q8:
     group-algebra closure, l1 bound, eigenvalue form, dist <= kappa*Gamma.
  B2. the S3/transposition counterexample (the scope boundary, reported).
  C. the coincube family at every q, from the OPERATOR eigenvalues (the
     closed form is re-derived as a spectral branch, not assumed).
  D. node-unitarity characterization: the annealed node map E[C^N] is
     unitary iff the window count N is a.s. constant mod 4 (the sharp law,
     with teeth: |E[i^N]| < 1 strictly for every non-degenerate count
     distribution); for adjacent-pair read windows the deterministic-count
     translation-invariant binary media are, ON RINGS AND ON MEDIA THAT
     FACTORIZE OVER AXIS LINES, exactly the per-line crystals (enumerated
     per line).  The classification is 1D/line-factorized: general 3D
     media are NOT classified (see part media3d below); the theorem's statement is
     rescoped accordingly.
  E-G. deterministic media kill the cone (crystal -> diamond fan; versor
     words -> dead axes; the order-random dimer map, which would need a
     stochastic env rule, still gives the diamond); the in-out arc
     T = E(cI + sC), c^2+s^2 = 1, is unitary with the isotropic cone for
     all weights (symbolic).
  H (part media3d). the reviewer 3D counterexample family -- staggered along the read
     axis with an independent random phase per transverse line: (i) its
     stated properties CONFIRMED (translation-invariant in measure,
     deterministic per-line window counts, not one of the four global
     crystals; closed under the model's env streaming); (ii) the theorem's
     CONCLUSION nevertheless survives on it: the ensemble node map is
     damped EXACTLY by e^3 per cycle, e = P(adjacent line phases equal)
     = r^2 + (1-r)^2 < 1 (cross-streaming couples adjacent lines, so
     per-line determinism does not give ensemble determinism), and an
     exhaustive quenched scan of ALL 4096 members on the 2^3 supercell
     (folded corner momentum = Gamma + all 8 physical BZ corners) finds
     every degenerate band group fails the isotropic-pair witness.
     Nodes at interior supercell momenta are not exhaustively excluded
     (open, stated).

Hypotheses ledger: the bound uses (i) convex probability weights and
(ii) the R.Q8 event set; media correlations are free; it governs the NODE
(transport phases zero), not general k (part scope_k). The cone => Gamma > 0
statement additionally uses the coincube read geometry (adjacent-pair
windows, both directions coin-reachable) and a deterministic env CA with
randomness only in the initial measure; its media classification input is
proven for rings and line-factorized media only.

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_q_bound.py
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
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_q_bound.json"
C4 = [np.array(c) for c in COIN_C]
DD = [np.array(d) for d in COIN_D]
KAPPA = np.pi / (2 * np.log(2))


def dist_vertex(theta, ell=4):
    """distance of an angle to the available event-angle lattice (pi/ell')Z
    with ell' = ell/2 spacing pi/(ell/2)... spacing = pi/(ell/2): for the
    quaternionic set ell=4 the lattice is (pi/2)Z."""
    spacing = 2 * np.pi / ell
    return np.abs((theta + spacing / 2) % spacing - spacing / 2)


# ----------------------------------------------------------------------------

def part_chord(rng):
    res = {}
    # kappa on the 1 -> i chord: midpoint exact value pi/(2 ln 2)
    ps = np.linspace(1e-6, 1 - 1e-6, 20001)
    lam = (1 - ps) + 1j * ps
    q = dist_vertex(np.angle(lam)) / (-np.log(np.abs(lam)))
    res["chord_max"] = float(q.max())
    res["chord_argmax_p"] = float(ps[int(np.argmax(q))])
    res["kappa_exact"] = float(KAPPA)
    # the NAIVE quantity arg lambda/(-ln|lambda|) (the earlier, wrong
    # appendix phrasing) is NOT maximized at the midpoint: it diverges at
    # the chord endpoint t -> 1 (arg -> pi/2, Gamma -> 0). Exhibited:
    naive = np.angle(lam) / (-np.log(np.abs(lam)))
    res["naive_ratio_at_midpoint"] = float(naive[len(ps) // 2])
    res["naive_ratio_near_endpoint"] = float(naive[-1])
    res["naive_diverges"] = bool(naive[-1] > 1e3 * naive[len(ps) // 2])
    # hull scan: random mixtures over {1, i, -1, -i}
    worst = 0.0
    for _ in range(20000):
        w = rng.dirichlet(np.ones(4) * rng.uniform(0.2, 2))
        z = w[0] + 1j * w[1] - w[2] - 1j * w[3]
        if abs(z) < 1e-12 or abs(abs(z) - 1) < 1e-12:
            continue
        worst = max(worst, dist_vertex(np.angle(z)) / (-np.log(abs(z))))
    res["hull_scan_max"] = float(worst)
    # kappa(ell): chord between adjacent ell-th roots (event angle 2pi/ell)
    tab = {}
    for ell in (4, 8, 16, 32):
        z0, z1 = 1.0, np.exp(2j * np.pi / ell)
        lam = (1 - ps) * z0 + ps * z1
        qv = dist_vertex(np.angle(lam), ell) / (-np.log(np.abs(lam)))
        tab[ell] = float(qv.max())
    res["kappa_of_ell"] = {str(k): v for k, v in tab.items()}
    res["kappa_scaling_2ell_over_pi"] = {str(ell): float(tab[ell] / (4 * ell / (2 * np.pi)))
                                         for ell in tab}
    return res


def part_products(rng):
    """random non-commuting products of mixtures over R.Q8: the theorem"""
    Q8 = [np.eye(4), -np.eye(4)] + [sg * c for c in C4 for sg in (1, -1)]
    span = np.stack([np.eye(4).ravel()] + [c.ravel() for c in C4])
    worst_hull = worst_ratio = max_resid = 0.0
    for _ in range(4000):
        nf = rng.integers(1, 9)
        T = np.eye(4)
        for _ in range(nf):
            p = rng.dirichlet(np.ones(8) * rng.uniform(0.15, 2.0))
            T = sum(pi * U for pi, U in zip(p, Q8)) @ T
        coef, _, _, _ = np.linalg.lstsq(span.T, T.ravel(), rcond=None)
        max_resid = max(max_resid, float(np.abs(span.T @ coef
                                                - T.ravel()).max()))
        w, v = coef[0], coef[1:]
        r2n = float(np.linalg.norm(v))
        worst_hull = max(worst_hull, abs(w) + r2n)
        lam = w + 1j * r2n
        if 1e-12 < abs(lam) and abs(abs(lam) - 1) > 1e-12:
            gam = -np.log(abs(lam))
            d = np.abs((np.angle(lam) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            worst_ratio = max(worst_ratio, d / gam)
    return dict(span_residual=float(max_resid),
                hull_max_l1=float(worst_hull),
                max_dist_over_Gamma=float(worst_ratio), kappa=float(KAPPA))


def part_boundary_S3():
    """the reviewer-style counterexample OUTSIDE R.Q8: transposition
    mixtures develop 3-cycle eigenvalues; the bound fails and is unbounded
    as eps -> 0. Reported as the theorem's exact scope boundary."""
    S12 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.]])
    S23 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.]])
    rows = []
    for eps in (0.05, 0.01, 0.002):
        M1 = (1 - eps) * S12 + eps * np.eye(3)
        M2 = (1 - eps) * S23 + eps * np.eye(3)
        w = np.linalg.eigvals(M2 @ M1)
        # the complex pair approaching e^{2 pi i/3}
        lam = w[np.argmax(np.abs(np.angle(w)) * (np.abs(w.imag) > 1e-12))]
        gam = -np.log(abs(lam))
        d = np.abs((np.angle(lam) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        rows.append(dict(eps=float(eps), arg_over_pi=float(np.angle(lam) / np.pi),
                         Gamma=float(gam), ratio=float(d / gam)))
    return dict(rows=rows, kappa=float(KAPPA))


def part_scope_k():
    """the transport-phase boundary of the theorem, exhibited: at k != 0
    the factors E_a(k_a) T_a leave the R.Q8 mixture class and the bound
    FAILS for the coincube operator itself.  The bound therefore governs
    the node (where transport phases vanish), not general momenta."""
    from pca3d.models.coincube import annealed_u
    q, kv = 0.02, np.array([0.4, 0.0, 0.0])
    lam = np.linalg.eigvals(annealed_u(kv, q))
    ratios = []
    for lm in lam:
        gam = -np.log(abs(lm))
        d = np.abs((np.angle(lm) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        ratios.append(float(d / gam))
    return dict(q=q, k=[float(x) for x in kv],
                min_mode_ratio=float(min(ratios)),
                all_ratios=sorted(ratios), kappa=float(KAPPA))


def part_family():
    """the coincube node at every q, from the OPERATOR (closed form
    re-derived as a spectral branch)"""
    res = {"rows": []}
    worst = cf_dev = 0.0
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
        cf_dev = max(cf_dev, min(abs(lam0 - lam_cf),
                                 abs(lam0 - np.conj(lam_cf))))
        gam = -np.log(abs(lam0))
        d = np.abs((np.angle(lam0) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        Q = d / gam
        worst = max(worst, Q)
        if abs(q - 0.1) < 1e-9 or abs(q - 0.2) < 1e-9:
            res["rows"].append(dict(q=float(q), omega0=float(np.angle(lam0)),
                                    Gamma=float(gam), Q_vertex=float(Q)))
    res["family_max_Q_vertex"] = float(worst)
    res["operator_vs_closedform_dev"] = float(cf_dev)
    res["kappa"] = float(KAPPA)
    return res


def part_count(rng):
    res = {}
    # node-window law WITH TEETH (the earlier check asserted the arithmetic
    # identity |E[i^N]|^2 = (p0-p2)^2 + (p1-p3)^2, true for any reals --
    # vacuous; replaced): the law that carries weight is
    #   |E[i^N]| = 1  <=>  the count distribution is a point mass mod 4,
    # i.e. the node map is unitary iff N is a.s. constant mod 4.
    strict_ok = True
    for _ in range(400):
        p = rng.dirichlet(np.ones(4) * rng.uniform(0.05, 3.0))
        z = abs(np.sum(p * np.array([1, 1j, -1, -1j])))
        if (z < 1 - 1e-12) != (p.max() < 1 - 1e-12):
            strict_ok = False
    for hot in range(4):                       # the point masses: |z| = 1
        p = np.zeros(4)
        p[hot] = 1
        if abs(abs(np.sum(p * np.array([1, 1j, -1, -1j]))) - 1) > 1e-14:
            strict_ok = False
    res["unitary_iff_deterministic_count"] = bool(strict_ok)
    # Bernoulli(q) media: window N = b_i + b_{i+1} is NOT deterministic;
    # the per-window node factor is 1 - 2q + 2iq(1-q), strictly inside the
    # disc for all q in (0, 1): the law quantifies the damping directly.
    dev = 0.0
    min_gap = np.inf
    for q in np.linspace(0.05, 0.95, 19):
        pN = np.array([(1 - q)**2, 2 * q * (1 - q), q * q, 0.0])
        z = np.sum(pN * np.array([1, 1j, -1, -1j]))
        dev = max(dev, abs(z - ((1 - 2 * q) + 2j * q * (1 - q))))
        min_gap = min(min_gap, 1 - abs(z))
    res["bernoulli_window_factor_dev"] = float(dev)
    res["bernoulli_min_unitarity_gap"] = float(min_gap)
    # ring enumeration: TI binary configs with constant adjacent-pair count
    counts = {}
    for L in (6, 8, 10, 7):
        good = []
        for cfg in range(1 << L):
            b = [(cfg >> i) & 1 for i in range(L)]
            sums = {b[i] + b[(i + 1) % L] for i in range(L)}
            if len(sums) == 1:
                good.append(cfg)
        counts[L] = len(good)
    res["deterministic_count_configs"] = {str(k): v for k, v in counts.items()}
    # even rings: exactly 4 (all-0, all-1, two staggered); odd rings: 2
    return res


# --- crystal / versor / idealized-map models --------------------------------

def crystal_ops():
    def idx(r, c):
        return 4 * (r[0] + 2 * (r[1] + 2 * r[2])) + c

    def conv_op(a, phi):
        M = np.eye(32, dtype=complex)
        for r in itertools.product((0, 1), repeat=3):
            if (r[a] + phi) % 2 == 1:
                for c in range(4):
                    for c2 in range(4):
                        M[idx(r, c2), idx(r, c)] = C4[a][c2, c]
        return M

    def shift_op(a, k):
        M = np.zeros((32, 32), dtype=complex)
        for r in itertools.product((0, 1), repeat=3):
            for c in range(4):
                d = int(DD[a][c])
                r2 = list(r)
                r2[a] += d
                ph = 1.0
                if r2[a] == 2:
                    r2[a] = 0
                    ph = np.exp(2j * k[a])
                elif r2[a] == -1:
                    r2[a] = 1
                    ph = np.exp(-2j * k[a])
                M[idx(tuple(r2), c), idx(r, c)] = ph
        return M

    def U(kv, phis=(0, 0, 0)):
        M = np.eye(32, dtype=complex)
        for a in range(3):
            for _ in range(2):
                M = shift_op(a, kv) @ conv_op(a, phis[a]) @ M
        return M

    return U


def fan_of(Ufun, dim, n, eps=1e-6):
    U0 = Ufun(np.zeros(3))
    Ms = []
    for a in range(3):
        kp = np.zeros(3); kp[a] = eps
        km = np.zeros(3); km[a] = -eps
        dU = (Ufun(kp) - Ufun(km)) / (2 * eps)
        H = 1j * np.linalg.inv(U0) @ dU
        Ms.append((H + H.conj().T) / 2)
    H = sum(n[a] * Ms[a] for a in range(3))
    return np.sort(np.linalg.eigvalsh(H))


def part_crystal(rng):
    res = {}
    U = crystal_ops()
    k = rng.uniform(-1, 1, 3)
    Um = U(k)
    res["unitary_dev"] = float(np.abs(Um @ Um.conj().T - np.eye(32)).max())
    res["U0_is_minus_I_dev"] = float(np.abs(U(np.zeros(3)) + np.eye(32)).max())
    # the first-order fan is strongly anisotropic (ballistic diamond
    # signature, no isotropic cone pair): along the axes {+-2 x8, 0 x16},
    # along 111 the extreme slope is 2 sqrt(3) (the corner-mover cone bound);
    # the max-slope ratio 111/100 is sqrt(3) (a cone would give 1).
    ev100 = fan_of(U, 32, np.array([1.0, 0, 0]))
    ev111 = fan_of(U, 32, np.array([1.0, 1, 1]) / np.sqrt(3))
    res["fan_100_pattern_dev"] = float(np.abs(
        np.sort(ev100) - np.array([-2.0] * 8 + [0.0] * 16 + [2.0] * 8)).max())
    res["fan_111_max"] = float(np.abs(ev111).max())
    res["max_slope_ratio_111_over_100"] = float(
        np.abs(ev111).max() / np.abs(ev100).max())
    return res


def part_versor():
    """two-stripe media (staggered x and y, env_z = 0): deterministic word
    C_yC_x = +C_z -> U(0) = +-i pairs, but dead/chiral axes, no cone."""
    def idx(rx, ry, c):
        return 4 * (rx + 2 * ry) + c

    def U(kv):
        M = np.eye(16, dtype=complex)
        for a in range(3):
            for _ in range(2):
                S = np.zeros((16, 16), dtype=complex)
                Cm = np.eye(16, dtype=complex)
                for rx in (0, 1):
                    for ry in (0, 1):
                        r = (rx, ry)
                        if a < 2 and r[a] % 2 == 1:
                            for c in range(4):
                                for c2 in range(4):
                                    Cm[idx(rx, ry, c2), idx(rx, ry, c)] = C4[a][c2, c]
                for rx in (0, 1):
                    for ry in (0, 1):
                        for c in range(4):
                            d = int(DD[a][c])
                            r2 = [rx, ry]
                            ph = 1.0
                            if a < 2:
                                r2[a] += d
                                if r2[a] == 2:
                                    r2[a] = 0
                                    ph = np.exp(2j * kv[a])
                                elif r2[a] == -1:
                                    r2[a] = 1
                                    ph = np.exp(-2j * kv[a])
                            else:
                                ph = np.exp(1j * kv[2] * d)
                            S[idx(r2[0], r2[1], c), idx(rx, ry, c)] = ph
                M = S @ Cm @ M
        return M

    res = {}
    U0 = U(np.zeros(3))
    res["unitary_dev"] = float(np.abs(U0 @ U0.conj().T - np.eye(16)).max())
    w = np.linalg.eigvals(U0)
    res["spectrum_pm_i"] = bool(np.allclose(np.sort(np.abs(np.angle(w))),
                                            np.pi / 2, atol=1e-9))
    # per-axis projected slope ranges on the +i eigengroup
    wv, V = np.linalg.eig(U0)
    sel = np.abs(wv - 1j) < 1e-9
    Qm, _ = np.linalg.qr(V[:, sel])
    eps = 1e-6
    slopes = []
    for a in range(3):
        kp = np.zeros(3); kp[a] = eps
        km = np.zeros(3); km[a] = -eps
        dU = (U(kp) - U(km)) / (2 * eps)
        H = 1j * np.linalg.inv(U0) @ dU
        h = Qm.conj().T @ ((H + H.conj().T) / 2) @ Qm
        ev = np.linalg.eigvalsh(h)
        slopes.append([float(ev.min()), float(ev.max())])
    res["axis_slope_ranges"] = slopes    # x and z dead, y chiral +-2
    return res


def part_idealized(rng):
    """order-random dimer map T2 = (E^2 C + E C E)/2 per axis (would need a
    stochastic env CA): node-unitary, Gamma = k^2/2, but diamond fan."""
    def U(kv):
        M = np.eye(4, dtype=complex)
        for a in range(3):
            E = np.diag(np.exp(1j * kv[a] * DD[a]))
            M = 0.5 * (E @ E @ C4[a] + E @ C4[a] @ E) @ M
        return M

    res = {}
    res["U0_minus_I_dev"] = float(np.abs(U(np.zeros(3)) + np.eye(4)).max())
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    gs = []
    for s in (0.05, 0.1, 0.2):
        w = np.linalg.eigvals(U(s * n))
        gs.append(float(-np.log(np.abs(w)).max()))
    res["Gamma_over_khalf2"] = [g / (s**2 / 2) for g, s in
                                zip(gs, (0.05, 0.1, 0.2))]
    ev = fan_of(U, 4, np.array([1, 1, 1]) / np.sqrt(3))
    res["fan_111"] = [float(x) for x in ev]     # corner movers: -1/sqrt3 x3, sqrt3
    return res


# --- H: the reviewer 3D counterexample family (per-line phases) --------------

def _media3d_ops():
    """exact one-cycle supercell machinery on the 2^3 torus: 32-dim
    (site, channel) space, real conversion layers controlled by the actual
    streamed env fields, Bloch wrap phases exp(+-2ik_a) (supercell period
    2, so the folded k = 0 point carries Gamma AND all 8 physical BZ
    corners)."""
    sites = list(itertools.product((0, 1), repeat=3))

    def idx(r, c):
        return 4 * (r[0] + 2 * (r[1] + 2 * r[2])) + c

    PERM = [np.argmax(np.abs(c), axis=0) for c in C4]
    SGNv = [c[np.argmax(np.abs(c), axis=0), np.arange(4)] for c in C4]

    def conv_mat(enva, a):
        M = np.zeros((32, 32))
        for r in sites:
            for c in range(4):
                if enva[r]:
                    M[idx(r, int(PERM[a][c])), idx(r, c)] = SGNv[a][c]
                else:
                    M[idx(r, c), idx(r, c)] = 1
        return M

    def shift_mat(a, k):
        M = np.zeros((32, 32), complex)
        for r in sites:
            for c in range(4):
                d = int(DD[a][c])
                r2 = list(r)
                r2[a] += d
                ph = 1.0
                if r2[a] == 2:
                    r2[a] = 0
                    ph = np.exp(2j * k[a])
                elif r2[a] == -1:
                    r2[a] = 1
                    ph = np.exp(-2j * k[a])
                M[idx(tuple(r2), c), idx(r, c)] = ph
        return M

    def stream_env(enva, a):
        sa = (a + 1) % 3
        out = {}
        for r in sites:
            r2 = list(r)
            r2[sa] = 1 - r2[sa]
            out[tuple(r2)] = enva[r]
        return out

    def U_of(k, envs0):
        envs = [dict(e) for e in envs0]
        M = np.eye(32, dtype=complex)
        for a in range(3):
            S = shift_mat(a, k)
            for o in (0, 1):
                M = S @ conv_mat(envs[a], a) @ M
                envs[a] = stream_env(envs[a], a)
        return M

    def staggered_medium(phis):
        """env_a(r) = (r_a + phi_a(r_perp)) mod 2: the reviewer family"""
        envs = []
        for a, phi in enumerate(phis):
            e = {}
            for r in sites:
                perp = tuple(r[b] for b in range(3) if b != a)
                e[r] = (r[a] + phi[perp]) % 2
            envs.append(e)
        return envs

    def line_medium(types):
        """general line-factorized media: per (axis, transverse line) type
        in {0: all-0, 1: all-1, 2: staggered phi=0, 3: staggered phi=1}"""
        envs = []
        for a, tp in enumerate(types):
            e = {}
            for r in sites:
                perp = tuple(r[b] for b in range(3) if b != a)
                t = tp[perp]
                e[r] = (0 if t == 0 else 1 if t == 1
                        else (r[a] + (t - 2)) % 2)
            envs.append(e)
        return envs

    return sites, U_of, staggered_medium, line_medium, stream_env


def _iso_witness(U0, Hs, dirs):
    """for every degenerate eigenvalue group of U0: does the projected
    first-order fan contain a direction-independent +-v pair?  Returns
    (n_groups, n_excluded, n_candidate)."""
    w, V = np.linalg.eig(U0)
    used = np.zeros(len(w), bool)
    n_grp = n_exc = n_cand = 0
    for i in range(len(w)):
        if used[i]:
            continue
        grp = np.where(np.abs(w - w[i]) < 1e-9)[0]
        used[grp] = True
        if len(grp) < 2:
            continue
        n_grp += 1
        Q, _ = np.linalg.qr(V[:, grp])
        specs = []
        for n in dirs:
            H = sum(n[a] * Hs[a] for a in range(3))
            specs.append(np.linalg.eigvalsh(Q.conj().T @ H @ Q))
        cand = [v for v in np.abs(specs[0]) if v > 1e-4]
        hit = any(all(any(abs(e - v) < 1e-5 for e in s)
                      and any(abs(e + v) < 1e-5 for e in s) for s in specs)
                  for v in cand)
        if hit:
            n_cand += 1
        else:
            n_exc += 1
    return n_grp, n_exc, n_cand


def part_media3d(rng):
    """(i) confirm the reviewer counterexample's properties; (ii) show the
    theorem's CONCLUSION survives on the family: exact ensemble damping
    e^3 per cycle, and an exhaustive quenched no-isotropic-node scan at
    the folded corner momentum."""
    res = {}
    sites, U_of, staggered_medium, line_medium, stream_env = _media3d_ops()
    perp2 = list(itertools.product((0, 1), repeat=2))
    allphi = [dict(zip(perp2, bits))
              for bits in itertools.product((0, 1), repeat=4)]

    # (i.a) deterministic PER-LINE window counts: every staggered line has
    # adjacent-pair count exactly 1 in every window, any phase, any length
    ok_lines = True
    for L in (4, 6, 8):
        for phi in (0, 1):
            line = [(x + phi) % 2 for x in range(L)]
            counts = {line[i] + line[(i + 1) % L] for i in range(L)}
            ok_lines &= counts == {1}
    res["per_line_counts_deterministic"] = bool(ok_lines)

    # (i.b) translation invariance in measure: the uniform measure on the
    # per-axis phase family is permuted bijectively by every lattice
    # translation (shift along the read axis flips all phases; transverse
    # shifts permute the lines), and by the model's env streaming
    members = []
    for phx in allphi:
        members.append(staggered_medium((phx, allphi[0], allphi[0]))[0])
    fam_keys = {tuple(sorted(m.items())) for m in members}
    closed_shift = closed_stream = True
    for m in members:
        for ax in range(3):                 # translations by 1 along ax
            sh = {}
            for r in sites:
                r2 = list(r)
                r2[ax] = (r2[ax] + 1) % 2
                sh[tuple(r2)] = m[r]
            if tuple(sorted(sh.items())) not in fam_keys:
                closed_shift = False
        st = stream_env(m, 0)               # env_x streams along y
        if tuple(sorted(st.items())) not in fam_keys:
            closed_stream = False
    res["family_closed_under_translations"] = bool(closed_shift)
    res["family_closed_under_streaming"] = bool(closed_stream)

    # (i.c) not a crystal: only 2 of the 16 phase choices per axis give a
    # globally staggered (crystal) field
    n_crystal = sum(1 for phx in allphi
                    if len(set(phx.values())) == 1)
    res["crystal_members_per_axis"] = int(n_crystal)
    res["noncrystal_members_per_axis"] = int(len(allphi) - n_crystal)

    # (ii.a) ENSEMBLE: the averaged one-cycle operator at the folded
    # corner momentum has top |eigenvalue| EXACTLY e^3 with
    # e = r^2 + (1-r)^2 (cross-streaming reads adjacent-line phase
    # differences; equal-phase windows give one conversion, unequal-phase
    # windows give 0 or 2 with C^2 = -1 interference)
    k0 = np.zeros(3)
    rows = []
    for r_ in (0.5, 0.3):
        acc = np.zeros((32, 32), complex)
        for bx in itertools.product((0, 1), repeat=4):
            wx = np.prod([r_ if b else 1 - r_ for b in bx])
            for by in itertools.product((0, 1), repeat=4):
                wy = np.prod([r_ if b else 1 - r_ for b in by])
                for bz in itertools.product((0, 1), repeat=4):
                    wz = np.prod([r_ if b else 1 - r_ for b in bz])
                    phis = (dict(zip(perp2, bx)), dict(zip(perp2, by)),
                            dict(zip(perp2, bz)))
                    acc += wx * wy * wz * U_of(k0, staggered_medium(phis))
        top = float(np.abs(np.linalg.eigvals(acc)).max())
        e = r_ * r_ + (1 - r_)**2
        rows.append(dict(r=float(r_), top_mod_eig=top, e_cubed=float(e**3),
                         Gamma_per_cycle=float(-3 * np.log(e))))
    res["ensemble_rows"] = rows

    # (ii.b) QUENCHED, exhaustive: all 4096 members; unitary; every
    # degenerate band group at the folded corner momentum fails the
    # isotropic-pair witness
    eps = 1e-6
    dirs = [np.array(v, float) / np.linalg.norm(v) for v in
            ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
             (0, 1, 1), (1, 1, 1), (1, -1, 1))]
    for _ in range(3):
        v = rng.normal(size=3)
        dirs.append(v / np.linalg.norm(v))
    n_media = n_scalar = n_grp = n_exc = n_cand = 0
    udev = 0.0
    for bx in itertools.product((0, 1), repeat=4):
        for by in itertools.product((0, 1), repeat=4):
            for bz in itertools.product((0, 1), repeat=4):
                phis = (dict(zip(perp2, bx)), dict(zip(perp2, by)),
                        dict(zip(perp2, bz)))
                envs = staggered_medium(phis)
                n_media += 1
                U0 = U_of(k0, envs)
                udev = max(udev, float(np.abs(
                    U0 @ U0.conj().T - np.eye(32)).max()))
                if np.abs(U0 - U0[0, 0] * np.eye(32)).max() < 1e-12:
                    n_scalar += 1
                Hs = []
                for a in range(3):
                    kp = np.zeros(3); kp[a] = eps
                    km = np.zeros(3); km[a] = -eps
                    dU = (U_of(kp, envs) - U_of(km, envs)) / (2 * eps)
                    Ha = 1j * np.linalg.inv(U0) @ dU
                    Hs.append((Ha + Ha.conj().T) / 2)
                g, x, cnd = _iso_witness(U0, Hs, dirs)
                n_grp += g
                n_exc += x
                n_cand += cnd
    res["quenched_exhaustive"] = dict(
        n_media=n_media, unitary_dev=udev, n_scalar_at_corner=n_scalar,
        n_band_groups=n_grp, n_groups_excluded=n_exc,
        n_groups_isotropy_candidate=n_cand,
        scope_note=("folded corner momentum only (= Gamma + all 8 physical"
                    " BZ corners); interior supercell momenta not"
                    " exhaustively excluded"))

    # (ii.c) general line-factorized media (mixed all-0/all-1/staggered
    # per line): random sample with the same witness
    n_media = n_grp = n_exc = n_cand = 0
    udev = 0.0
    for _ in range(400):
        types = tuple({p: int(rng.integers(0, 4)) for p in perp2}
                      for _ in range(3))
        envs = line_medium(types)
        n_media += 1
        U0 = U_of(k0, envs)
        udev = max(udev, float(np.abs(
            U0 @ U0.conj().T - np.eye(32)).max()))
        Hs = []
        for a in range(3):
            kp = np.zeros(3); kp[a] = eps
            km = np.zeros(3); km[a] = -eps
            dU = (U_of(kp, envs) - U_of(km, envs)) / (2 * eps)
            Ha = 1j * np.linalg.inv(U0) @ dU
            Hs.append((Ha + Ha.conj().T) / 2)
        g, x, cnd = _iso_witness(U0, Hs, dirs)
        n_grp += g
        n_exc += x
        n_cand += cnd
    res["line_factorized_sample"] = dict(
        n_media=n_media, unitary_dev=udev, n_band_groups=n_grp,
        n_groups_excluded=n_exc, n_groups_isotropy_candidate=n_cand)

    # TEETH: the witness must DETECT a genuine isotropic node -- the
    # unitary in-out arc (part_arc) at s = 1/2 on the same supercell:
    # constant staggered-free medium is not applicable, so run the witness
    # on the 4-dim annealed arc operator directly
    s = 0.5
    c = np.sqrt(1 - s * s)

    def Uarc(kv):
        M = np.eye(4, dtype=complex)
        for a in range(3):
            T = np.diag(np.exp(1j * kv[a] * DD[a])) @ (
                c * np.eye(4) + s * C4[a])
            M = T @ T @ M
        return M

    U0 = Uarc(np.zeros(3))
    Hs = []
    for a in range(3):
        kp = np.zeros(3); kp[a] = eps
        km = np.zeros(3); km[a] = -eps
        dU = (Uarc(kp) - Uarc(km)) / (2 * eps)
        Ha = 1j * np.linalg.inv(U0) @ dU
        Hs.append((Ha + Ha.conj().T) / 2)
    g, x, cnd = _iso_witness(U0, Hs, dirs)
    res["witness_teeth_arc"] = dict(n_groups=g, excluded=x, candidates=cnd)
    return res


def part_arc(rng):
    """the in-out arc T = E (c + s C): unitary, isotropic cone, closed forms;
    symbolic isotropy for GENERAL (alpha, beta)."""
    res = {}
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
    res["symbolic_isotropy_general_alpha_beta"] = bool(
        sp.simplify(sp.expand(cx**2 - cy**2)) == 0
        and sp.simplify(sp.expand(cx**2 - cz**2)) == 0)

    def U(kv, s):
        c = np.sqrt(1 - s * s)
        M = np.eye(4, dtype=complex)
        for a in range(3):
            T = np.diag(np.exp(1j * kv[a] * DD[a])) @ (c * np.eye(4) + s * C4[a])
            M = T @ T @ M
        return M

    dirs = [np.array(v, float) / np.linalg.norm(v)
            for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1))]
    for _ in range(5):
        v = rng.normal(size=3)
        dirs.append(v / np.linalg.norm(v))
    rows = []
    for s in (0.1, 0.2, np.sqrt(0.15), 0.5, 0.6):
        U0 = U(np.zeros(3), s)
        udev = np.abs(U0 @ U0.conj().T - np.eye(4)).max()
        w, V = np.linalg.eig(U0)
        Vi = np.linalg.inv(V)
        lam = w[np.argmax(w.imag)]
        up = [i for i in range(4) if abs(w[i] - lam) < 1e-9]
        eps = 1e-6
        Ms = []
        for a in range(3):
            kp = np.zeros(3); kp[a] = eps
            km = np.zeros(3); km[a] = -eps
            Ms.append(Vi[up, :] @ ((U(kp, s) - U(km, s)) / (2 * eps)) @ V[:, up])
        sl = []
        for n in dirs:
            mu = np.linalg.eigvals(n[0] * Ms[0] + n[1] * Ms[1] + n[2] * Ms[2])
            sl.append(abs(mu[0] - mu[1]))
        sl = np.array(sl)
        c = np.sqrt(1 - s * s)
        A, B = c * c - s * s, 2 * c * s
        lam_cf = (A**3 - B**3) + 1j * A * B * np.sqrt(3 * A**2 + 2 * A * B + 3 * B**2)
        v_omega = float(sl.mean() / 2 / abs(lam))
        assert v_omega <= 2.0 + 1e-9
        rows.append(dict(s=float(s), unitary_dev=float(udev),
                         mult=len(up), omega0=float(np.angle(lam)),
                         cf_dev=float(abs(lam - lam_cf)),
                         v=v_omega, aniso=float(sl.max() / sl.min() - 1)))
    res["arc_rows"] = rows
    res["max_aniso"] = float(max(r["aniso"] for r in rows))
    res["max_unitary_dev"] = float(max(r["unitary_dev"] for r in rows))
    return res


# ----------------------------------------------------------------------------

def main():
    out = {"seed": SEED}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    for name, fn in [("chord", lambda: part_chord(rng)),
                     ("products", lambda: part_products(rng)),
                     ("boundary_S3", part_boundary_S3),
                     ("scope_k", part_scope_k),
                     ("family", part_family),
                     ("count", lambda: part_count(rng)),
                     ("crystal", lambda: part_crystal(rng)),
                     ("versor", part_versor),
                     ("idealized", lambda: part_idealized(rng)),
                     ("media3d", lambda: part_media3d(rng)),
                     ("arc", lambda: part_arc(rng))]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.1f}s] {name} done")

    ch = out["chord"]
    # the CORRECT maximized quantity dist(arg, (pi/2)Z)/Gamma: kappa exact,
    # at the chord midpoint; the naive arg/Gamma diverges at the endpoint
    assert abs(ch["chord_max"] - KAPPA) < 1e-6
    assert abs(ch["chord_argmax_p"] - 0.5) < 1e-3
    assert ch["hull_scan_max"] <= KAPPA + 1e-9
    assert ch["naive_diverges"]
    assert ch["naive_ratio_near_endpoint"] > 1e3
    pr = out["products"]
    assert pr["span_residual"] < 1e-10
    assert pr["hull_max_l1"] <= 1 + 1e-12
    assert pr["max_dist_over_Gamma"] <= KAPPA + 1e-9
    bs = out["boundary_S3"]["rows"]
    assert bs[0]["ratio"] > KAPPA          # the boundary is real ...
    assert bs[1]["ratio"] > 2 * bs[0]["ratio"]   # ... and unbounded
    # the transport-phase boundary: the bound FAILS at k != 0 (documented
    # scope: the theorem governs the node)
    sk = out["scope_k"]
    assert sk["min_mode_ratio"] > 2 * KAPPA
    assert abs(sk["min_mode_ratio"] - 6.06) < 0.05
    fam = out["family"]
    assert fam["family_max_Q_vertex"] <= KAPPA + 1e-9
    assert fam["operator_vs_closedform_dev"] < 1e-12
    cnt = out["count"]
    assert cnt["unitary_iff_deterministic_count"]
    assert cnt["bernoulli_window_factor_dev"] < 1e-14
    assert cnt["bernoulli_min_unitarity_gap"] > 0.05   # q in [0.05, 0.95]
    assert cnt["deterministic_count_configs"] == {"6": 4, "8": 4,
                                                 "10": 4, "7": 2}
    cr = out["crystal"]
    assert cr["unitary_dev"] < 1e-12 and cr["U0_is_minus_I_dev"] < 1e-12
    assert cr["fan_100_pattern_dev"] < 1e-6
    assert abs(cr["fan_111_max"] - 2 * np.sqrt(3)) < 1e-6
    assert cr["max_slope_ratio_111_over_100"] > 1.7    # cone would give 1
    vs = out["versor"]
    assert vs["spectrum_pm_i"]
    assert max(abs(x) for x in vs["axis_slope_ranges"][0]) < 1e-6   # x dead
    assert max(abs(x) for x in vs["axis_slope_ranges"][2]) < 1e-6   # z dead
    assert abs(vs["axis_slope_ranges"][1][1] - 2) < 1e-6            # y chiral
    idl = out["idealized"]
    assert idl["U0_minus_I_dev"] < 1e-12
    assert all(abs(r - 1) < 0.05 for r in idl["Gamma_over_khalf2"])
    assert abs(idl["fan_111"][-1] - np.sqrt(3)) < 1e-5              # diamond
    md = out["media3d"]
    assert md["per_line_counts_deterministic"]
    assert md["family_closed_under_translations"]
    assert md["family_closed_under_streaming"]
    assert md["crystal_members_per_axis"] == 2                # counterexample
    assert md["noncrystal_members_per_axis"] == 14            # ... confirmed
    for row in md["ensemble_rows"]:
        assert abs(row["top_mod_eig"] - row["e_cubed"]) < 1e-9   # EXACT e^3
        assert row["Gamma_per_cycle"] > 0.5
    qe = md["quenched_exhaustive"]
    assert qe["n_media"] == 4096 and qe["unitary_dev"] < 1e-12
    assert qe["n_groups_isotropy_candidate"] == 0             # no iso node
    assert qe["n_groups_excluded"] == qe["n_band_groups"] > 4000
    lf = md["line_factorized_sample"]
    assert lf["unitary_dev"] < 1e-12
    assert lf["n_groups_isotropy_candidate"] == 0
    wt = md["witness_teeth_arc"]
    assert wt["candidates"] > 0                               # teeth
    arc = out["arc"]
    assert arc["symbolic_isotropy_general_alpha_beta"]
    assert arc["max_aniso"] < 1e-9 and arc["max_unitary_dev"] < 1e-12
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
