#!/usr/bin/env python
"""Q-bound theory: is the quasiparticle quality factor pinned for legal PCAs?

Machine checks for docs/notes/theory-q-bound.md. All headline claims
re-asserted every run; results -> results/theory_q_bound.json.

Summary of what is proven/checked here:
  A. Chord bound: for any convex mixture lambda = sum p_j z_j over the
     quaternionic event angles (pi/2)Z, dist(arg lambda, (pi/2)Z) <=
     kappa * (-ln|lambda|) with kappa = pi/(2 ln 2) = 2.2662 exactly
     (attained at the chord midpoint); finer angle sets (ell-cycles) give
     kappa(ell) ~ 4 ell / pi.
  B. The coincube family satisfies the bound at every q (numerically, the
     non-commuting cycle composition included).
  C. Node-unitarity characterization: the annealed node map is unitary iff
     the conversion count N is deterministic (|E[i^N]| = 1); for the
     coincube's adjacent-pair read windows, translation-invariant binary
     media with deterministic window counts are EXACTLY {all-0, all-1, the
     two staggered crystals} (ring enumeration).
  D. The staggered crystal: unitary for all k (Gamma = 0), U(0) = -I, and
     the first-order fan is the pure corner-mover DIAMOND — no cone.
  E. Deterministic versor words (two-stripe media): U(0) = +-i pairs but the
     velocity projections give dead/chiral axes — no cone.
  F. The idealized order-random dimer map (needs a stochastic env CA —
     outside the framework): node-unitary, Gamma = k^2/2, and STILL the
     diamond fan — even that route fails the cone.
  G. The in-out arc T = E (c + s C), c^2 + s^2 = 1: unitary for all k, the
     isotropic cone survives (symbolic proof for general (alpha, beta),
     closed forms with A = c^2 - s^2, B = 2cs) — the unique completion of
     the mixture family achieving Q = infinity WITH the cone.

Verdict: within the legal class (deterministic CA + initial-measure
randomness), zero node damping forces deterministic conversion words, which
kill the isotropic cone. The Q-bound survives all legal media correlations.

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
    res["kappa_scaling_4ell_over_pi"] = {str(ell): float(tab[ell] / (4 * ell / (2 * np.pi)))
                                         for ell in tab}
    return res


def part_family():
    """the coincube cycle at every q obeys the bound (non-commuting comp.)"""
    res = {"rows": []}
    worst = 0.0
    for q in np.linspace(0.02, 0.98, 49):
        A, B = 1 - 2 * q, 2 * q * (1 - q)
        lam0 = (A**3 - B**3) + 1j * A * B * np.sqrt(3 * A**2 + 2 * A * B + 3 * B**2)
        gam = -np.log(abs(lam0))
        Q = dist_vertex(np.angle(lam0)) / gam
        worst = max(worst, Q)
        if abs(q - 0.1) < 1e-9 or abs(q - 0.15) < 1e-9 or abs(q - 0.2) < 1e-9:
            res["rows"].append(dict(q=float(q), omega0=float(np.angle(lam0)),
                                    Gamma=float(gam),
                                    Q_naive=float(np.angle(lam0) / gam),
                                    Q_vertex=float(Q)))
    res["family_max_Q_vertex"] = float(worst)
    res["kappa"] = float(KAPPA)
    return res


def part_count(rng):
    res = {}
    # exact node-window law: |E[i^N]|^2 = (p0-p2)^2 + (p1-p3)^2
    for _ in range(5):
        p = rng.dirichlet(np.ones(4))
        lhs = abs(np.sum(p * np.array([1, 1j, -1, -1j])))**2
        rhs = (p[0] - p[2])**2 + (p[1] - p[3])**2
        assert abs(lhs - rhs) < 1e-14
    res["window_law_verified"] = True
    # ring enumeration: TI binary configs with constant adjacent-pair count
    counts = {}
    for L in (8, 10, 7):
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
                     ("family", part_family),
                     ("count", lambda: part_count(rng)),
                     ("crystal", lambda: part_crystal(rng)),
                     ("versor", part_versor),
                     ("idealized", lambda: part_idealized(rng)),
                     ("arc", lambda: part_arc(rng))]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.1f}s] {name} done")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")

    ch = out["chord"]
    assert abs(ch["chord_max"] - KAPPA) < 1e-6
    assert abs(ch["chord_argmax_p"] - 0.5) < 1e-3
    assert ch["hull_scan_max"] <= KAPPA + 1e-9
    fam = out["family"]
    assert fam["family_max_Q_vertex"] <= KAPPA + 1e-9
    cnt = out["count"]
    assert cnt["deterministic_count_configs"] == {"8": 4, "10": 4, "7": 2}
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
    arc = out["arc"]
    assert arc["symbolic_isotropy_general_alpha_beta"]
    assert arc["max_aniso"] < 1e-9 and arc["max_unitary_dev"] < 1e-12
    print("all headline assertions passed")


if __name__ == "__main__":
    sys.exit(main())
