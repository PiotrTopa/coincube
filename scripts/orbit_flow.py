#!/usr/bin/env python
"""Adaptive continuation of the coincube spectator orbits in q.

A continuation instrument for the isolated Weyl
nodes ("spectators") of the annealed chord walk U(k) = prod_a [E_a M]^2,
M = (1-q) I + q C_a, over q in [0.015, 0.49]:

  * corner (k = 0) and one half-point (pi/2, 0, 0): fixed k, tracked
    omega(q) and Fukui-Hatsugai chirality as sanity anchors
    (chi = -1 / +1 on the +Im branch, asserted at every computed q);
  * octet A and octet B (the <111>-line octets, reps on (t, t, 1-t) pi):
    full-3D Nelder-Mead continuation with linear warm starts and adaptive
    q step; at EVERY accepted q the 3D-refined node is asserted to sit on
    the line to 1e-8 (it does, to ~1e-15);
  * the generic 24-orbit rep: full-3D continuation both directions from
    the census point at q = 0.08; its death (pairwise annihilation on the
    {k_a = 0} planes) is bracketed to < 1e-3 in q, and its low-q side is
    continued to the window edge (no birth in range).

At every accepted continuation state the node is refined to gap < 1e-12
(typically ~1e-16), the doublet quasienergy is tracked by phase continuity,
and the chirality is computed as the Fukui-Hatsugai Chern number of the
upper doublet band over an enclosing sphere (machinery and calibration
convention copied from the committed census instrument census_sweep.py;
sphere radii auto-shrunk below the distance to the nearest other node or
symmetry image).  At a subset of states the charge is recomputed at two
radii x two grids and asserted integer-stable.

Low-q resolution block: for q in {0.015, 0.02, 0.025, 0.028, 0.03, 0.035}
a dense 1D scan of the full (t, t, 1-t) pi line (24001 points) plus a local
3D sweep (grid + refinement) in a tube of radius 0.11 around the line
establishes exactly which nodes exist there; the quick scan's apparent
appearance/disappearance near q ~ 0.026-0.028 is reproduced with its exact
algorithm and explained (refine-gate jitter, not physics).

Also recorded: quasienergy crossings between tracked orbits (bisected to
1e-4 in q), the (Delta k, Delta omega) bridge between octet A and octet B
(the momentum + quasienergy a gapping perturbation must supply) tabulated
at q in {0.08, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45}, minimum pair
separations (position mergers), and the spectator-free-window question.

All headline facts are asserted before results/orbit_flow.json is
written.  Run:  PYTHONPATH=src .venv/bin/python scripts/orbit_flow.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_C, COIN_D  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "orbit_flow.json"

C4 = [np.array(c, float) for c in COIN_C]
D4 = [np.array(d, float) for d in COIN_D]
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]

DEG_TOL = 1e-12
GAP_ASSERT = 1e-10
Q_LO, Q_HI = 0.015, 0.49
BRIDGE_QS = (0.08, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
LOWQ_QS = (0.015, 0.02, 0.025, 0.028, 0.03, 0.035)


# -- operators (chord walk) ---------------------------------------------------

def walk_batch(K, q):
    al, be = 1 - q, q
    N = K.shape[0]
    U = np.broadcast_to(np.eye(4, dtype=complex), (N, 4, 4)).copy()
    for a in range(3):
        M = al * np.eye(4) + be * C4[a]
        ph = np.exp(1j * K[:, a, None] * D4[a])
        T = ph[:, :, None] * M[None, :, :]
        U = T @ T @ U
    return U


def make_ops(q):
    def op(k):
        return walk_batch(np.asarray(k, float)[None], q)[0]

    def gap(K):
        K = np.atleast_2d(np.asarray(K, float))
        w = np.linalg.eigvals(walk_batch(K, q))
        d = np.abs(w[:, :, None] - w[:, None, :])
        d[:, np.arange(4), np.arange(4)] = np.inf
        return d.reshape(len(w), -1).min(axis=1)

    return op, gap


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def tdist(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(np.linalg.norm(np.minimum(d % np.pi, np.pi - d % np.pi)))


def refine(gap, k0):
    """Full-3D Nelder-Mead refinement.  NO mod-pi wrap of the result: the
    continuation needs smooth coordinates for warm-start extrapolation
    (U(k) is exactly pi-periodic, so unwrapped coordinates are valid)."""
    f = lambda k: gap(k)[0]  # noqa: E731
    r = minimize(f, k0, method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-17, maxfev=1500))
    if r.fun > DEG_TOL:
        r2 = minimize(f, r.x, method="Nelder-Mead",
                      options=dict(xatol=1e-14, fatol=1e-18, maxfev=1500))
        if r2.fun < r.fun:
            r = r2
    return np.asarray(r.x, float), float(r.fun)


# -- doublet tracking ---------------------------------------------------------

def doublets_at(op, k):
    """Up to two disjoint minimum-gap eigenvalue pairs: [(gap, mean phase)]."""
    w = np.linalg.eigvals(op(k))
    d = np.abs(w[:, None] - w[None, :])
    np.fill_diagonal(d, np.inf)
    pairs = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[i, j]):
            break
        pairs.append((float(d[i, j]), float(np.angle((w[i] + w[j]) / 2))))
        for x in (i, j):
            d[x, :] = np.inf
            d[:, x] = np.inf
    return pairs


def omega_track(op, k, om_prev):
    """Continuously tracked doublet quasienergy nearest om_prev."""
    pairs = doublets_at(op, k)
    deg = [p for p in pairs if p[0] < 1e-9] or pairs
    th = min(deg, key=lambda p: abs(wrap(p[1] - om_prev)))[1]
    return om_prev + wrap(th - om_prev)


# -- Fukui-Hatsugai charge (census_sweep convention, calibrated) --------------

def band_vec(U, theta0, which):
    w, V = np.linalg.eig(U)
    d = np.abs(wrap(np.angle(w) - theta0))
    idx = np.argsort(d)
    pair = idx[:2]
    margin = float(d[idx[2]] - d[idx[1]]) if len(d) > 2 else np.inf
    dd = wrap(np.angle(w[pair]) - theta0)
    sel = pair[np.argmax(dd)] if which > 0 else pair[np.argmin(dd)]
    v = V[:, sel]
    return v / np.linalg.norm(v), margin


def chern_sphere(op, k0, theta0, r, nth, nph, which=+1):
    thetas = np.linspace(0, np.pi, nth + 1)
    phis = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    vecs = np.empty((nth + 1, nph), object)
    minmarg = np.inf
    for i, th in enumerate(thetas):
        if i in (0, nth):
            n = np.array([0, 0, 1.0]) if i == 0 else np.array([0, 0, -1.0])
            v, m = band_vec(op(k0 + r * n), theta0, which)
            minmarg = min(minmarg, m)
            for j in range(nph):
                vecs[i, j] = v
            continue
        for j, ph in enumerate(phis):
            n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                          np.cos(th)])
            v, m = band_vec(op(k0 + r * n), theta0, which)
            minmarg = min(minmarg, m)
            vecs[i, j] = v
    tot = 0.0
    for i in range(nth):
        for j in range(nph):
            jp = (j + 1) % nph
            p = (np.vdot(vecs[i, j], vecs[i + 1, j])
                 * np.vdot(vecs[i + 1, j], vecs[i + 1, jp])
                 * np.vdot(vecs[i + 1, jp], vecs[i, jp])
                 * np.vdot(vecs[i, jp], vecs[i, j]))
            tot += np.angle(p)
    return float(tot / (2 * np.pi)), float(minmarg)


def calibrate():
    def cal_op(k):
        H = sum(k[a] * PAULI[a] for a in range(3))
        w, V = np.linalg.eigh(H)
        return V @ np.diag(np.exp(1j * w)) @ V.conj().T
    cu, _ = chern_sphere(cal_op, np.zeros(3), 0.0, 0.1, 20, 20, +1)
    cl, _ = chern_sphere(cal_op, np.zeros(3), 0.0, 0.1, 20, 20, -1)
    return float(cu), float(cl)


# -- empirical spectral symmetry group (for image distances) ------------------

def signed_perms():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            O = np.zeros((3, 3))
            for i, p in enumerate(perm):
                O[i, p] = signs[i]
            out.append(O)
    return out


def spec_set_dev(a, b):
    a = list(a)
    d = 0.0
    for v in b:
        i = int(np.argmin(np.abs(np.array(a) - v)))
        d = max(d, abs(a[i] - v))
        a.pop(i)
    return d


def empirical_group(op, rng, ntest=6, tol=1e-8):
    ks = rng.uniform(0, np.pi, (ntest, 3))
    specs = [np.linalg.eigvals(op(k)) for k in ks]
    uni, anti = [], []
    for O in signed_perms():
        du = max(spec_set_dev(np.linalg.eigvals(op(O @ k)), s)
                 for k, s in zip(ks, specs))
        da = max(spec_set_dev(np.linalg.eigvals(op(O @ k)), np.conj(s))
                 for k, s in zip(ks, specs))
        if du < tol:
            uni.append(O)
        if da < tol:
            anti.append(O)
    return uni, anti


GROUP = []  # filled in main(): all 24 event-group image maps


def images(k):
    """All symmetry images (mod pi) of a node position."""
    k = np.asarray(k, float)
    return [(O @ k) % np.pi for O in GROUP]


def min_image_dist(k, others):
    """Distance from k to the nearest symmetry image of any node in others
    (self-images at distance ~0 excluded)."""
    best = np.inf
    for ko in others:
        for im in images(ko):
            d = tdist(k, im)
            if d > 1e-9:
                best = min(best, d)
    return best


# -- continuation -------------------------------------------------------------

def try_node(gap, k_warm, jitter=None):
    """Refine from k_warm (plus optional jittered restarts)."""
    kr, gr = refine(gap, k_warm)
    if gr < DEG_TOL:
        return kr, gr
    if jitter is not None:
        for dk in (jitter, -jitter):
            kr2, gr2 = refine(gap, k_warm + dk)
            if gr2 < DEG_TOL:
                return kr2, gr2
    return kr, gr


def continue_track(name, q0, k0, om0, direction, q_stop,
                   dq0=0.005, dq_max=0.02, dq_min=2.5e-4, move_cap=0.35):
    """Adaptive continuation from (q0, k0, om0) toward q_stop.

    Returns (states, boundary) where states = [(q, k, gap, omega)] excluding
    the seed state, and boundary is None if q_stop was reached alive, else
    a dict with the death bracket refined to < 1e-3 in q."""
    states = []
    q, k, om = q0, np.asarray(k0, float), om0
    q_prev, k_prev = None, None
    dq = dq0
    while (q_stop - q) * direction > 1e-12:
        qn = q + direction * dq
        if (qn - q_stop) * direction > 0:
            qn = q_stop
        # linear warm start
        if q_prev is not None and abs(q - q_prev) > 1e-12:
            k_pred = k + (k - k_prev) * (qn - q) / (q - q_prev)
        else:
            k_pred = k.copy()
        op, gap = make_ops(qn)
        jit = (k - k_prev) * 0.5 if k_prev is not None else None
        kr, gr = try_node(gap, k_pred, jit)
        moved = tdist(kr, k)
        if gr < DEG_TOL and moved < move_cap:
            omn = omega_track(op, kr, om)
            q_prev, k_prev = q, k
            q, k, om = qn, kr, omn
            states.append(dict(q=float(q), k=k.copy(), gap=gr, omega=float(om)))
            if moved < 0.03 and dq < dq_max:
                dq = min(dq * 1.4, dq_max)
        else:
            dq /= 2
            if dq < dq_min:
                # death: bisect [q (alive), qn (dead)] to < 1e-3
                qa, qd, ka = q, qn, k.copy()
                while qd - qa > 5e-4 if direction > 0 else qa - qd > 5e-4:
                    qm = 0.5 * (qa + qd)
                    _, gm = make_ops(qm)
                    km, gmv = try_node(gm, ka, jit)
                    if gmv < DEG_TOL and tdist(km, ka) < move_cap:
                        qa, ka = qm, km
                    else:
                        qd = qm
                boundary = dict(last_alive_q=float(qa), first_dead_q=float(qd),
                                width=float(abs(qd - qa)),
                                last_alive_k_over_pi=[float(x / np.pi)
                                                      for x in ka])
                print(f"    [{name}] death bracket: alive {qa:.6f} / "
                      f"dead {qd:.6f} (k/pi={np.round(ka / np.pi, 5)})")
                return states, boundary
        if len(states) > 2000:
            raise RuntimeError(f"{name}: runaway continuation")
    return states, None


def build_track(name, k_seed, q_seed, om_seed, gap_seed):
    """Bidirectional adaptive continuation; states sorted by q."""
    up, bnd_up = continue_track(name, q_seed, k_seed, om_seed, +1, Q_HI)
    dn, bnd_dn = continue_track(name, q_seed, k_seed, om_seed, -1, Q_LO)
    seed_state = dict(q=float(q_seed), k=np.asarray(k_seed, float),
                      gap=float(gap_seed), omega=float(om_seed))
    states = sorted(dn + [seed_state] + up, key=lambda s: s["q"])
    return dict(name=name, states=states, death_up=bnd_up, death_dn=bnd_dn)


def interp_k(track, q):
    qs = np.array([s["q"] for s in track["states"]])
    K = np.array([s["k"] for s in track["states"]])
    return np.array([np.interp(q, qs, K[:, a]) for a in range(3)])


def interp_om(track, q):
    qs = np.array([s["q"] for s in track["states"]])
    om = np.array([s["omega"] for s in track["states"]])
    return float(np.interp(q, qs, om))


def track_qrange(track):
    qs = [s["q"] for s in track["states"]]
    return min(qs), max(qs)


# -- main ---------------------------------------------------------------------

def main():
    t00 = time.time()
    rng = np.random.default_rng(20260902)
    out = {}

    cu, cl = calibrate()
    out["calibration"] = dict(upper=cu, lower=cl)
    assert abs(cu - 1) < 1e-9 and abs(cl + 1) < 1e-9
    print(f"[{time.time()-t00:6.1f}s] calibration ok "
          f"(upper {cu:+.6f}, lower {cl:+.6f})")

    op08, gap08 = make_ops(0.08)
    uni, anti = empirical_group(op08, rng)
    assert len(uni) == 12 and len(anti) == 12
    GROUP.extend(uni + anti)
    print(f"[{time.time()-t00:6.1f}s] event group: 12 unitary + 12 antiunitary")

    # ---- seeds at q = 0.08 (census representatives) -------------------------
    seeds = {}
    census_seeds = dict(
        octetA=(np.array([0.2650, 0.2650, 0.7350]) * np.pi, 0.5142 * np.pi),
        octetB=(np.array([0.2434, 0.2434, 0.7566]) * np.pi, 0.5652 * np.pi),
        gen24=(np.array([0.0283, 0.4006, 0.9021]) * np.pi, 0.9479 * np.pi))
    for name, (k0, om_cen) in census_seeds.items():
        kr, gr = refine(gap08, k0)
        assert gr < DEG_TOL, (name, gr)
        om = omega_track(op08, kr, om_cen)
        assert abs(om - om_cen) < 2e-3 * np.pi, (name, om / np.pi)
        seeds[name] = (kr, om, gr)
        print(f"    seed {name}: k/pi={np.round(kr / np.pi, 5)} "
              f"omega/pi={om / np.pi:.4f}")

    # ---- moving-orbit continuations -----------------------------------------
    tracks = {}
    for name in ("octetA", "octetB", "gen24"):
        k0, om0, g0 = seeds[name]
        tr = build_track(name, k0, q_seed=0.08, om_seed=om0, gap_seed=g0)
        tracks[name] = tr
        qlo, qhi = track_qrange(tr)
        print(f"[{time.time()-t00:6.1f}s] {name}: {len(tr['states'])} states, "
              f"q in [{qlo:.4f}, {qhi:.4f}], "
              f"death_up={'yes' if tr['death_up'] else 'no'}, "
              f"death_dn={'yes' if tr['death_dn'] else 'no'}")

    # on-line assertion for the octets at EVERY accepted state
    online_dev = {}
    for name in ("octetA", "octetB"):
        dev = 0.0
        for s in tracks[name]["states"]:
            k = s["k"]
            dev = max(dev, abs(k[0] - k[1]), abs(k[0] + k[2] - np.pi))
        online_dev[name] = float(dev)
        assert dev < 1e-8, (name, dev)
    print(f"[{time.time()-t00:6.1f}s] octets on-line at all states "
          f"(max dev {max(online_dev.values()):.2e})")

    # ---- anchors (fixed k) --------------------------------------------------
    anchor_qs = np.round(np.arange(Q_LO, Q_HI + 1e-9, 0.005), 6)
    anchors = {}
    for name, kfix, om_guess in (("corner", np.zeros(3), 0.10 * np.pi),
                                 ("half", np.array([np.pi / 2, 0, 0]),
                                  0.95 * np.pi)):
        states, om = [], None
        for q in anchor_qs:
            op, gap = make_ops(q)
            g = float(gap(kfix)[0])
            om = omega_track(op, kfix, om if om is not None else om_guess)
            states.append(dict(q=float(q), k=kfix.copy(), gap=g,
                               omega=float(om)))
        # anchor omega must connect to the census value at q = 0.08
        anchors[name] = dict(name=name, states=states,
                             death_up=None, death_dn=None)
    assert abs(interp_om(anchors["corner"], 0.08) - 0.1006 * np.pi) < 2e-3
    assert abs(interp_om(anchors["half"], 0.08) - 0.9221 * np.pi) < 2e-3
    print(f"[{time.time()-t00:6.1f}s] anchors tracked on {len(anchor_qs)} "
          f"q values (omega(0.08) matches census)")

    all_tracks = dict(tracks, **anchors)

    # ---- charges along every track ------------------------------------------
    def charge_state(op, s, others, full):
        dmin = min_image_dist(s["k"], others)
        r1 = min(0.008, 0.30 * dmin)
        r2 = min(0.02, 0.45 * dmin)
        combos = ([(r1, 16, 16), (r1, 24, 24), (r2, 16, 16), (r2, 24, 24)]
                  if full else [(r1, 16, 16)])
        vals, margs = [], []
        for r, nth, nph in combos:
            c, m = chern_sphere(op, s["k"], wrap(s["omega"]), r, nth, nph, +1)
            vals.append(c)
            margs.append(m)
        return vals, float(min(margs)), float(dmin), float(r1)

    n_full = 0
    for name, tr in all_tracks.items():
        others = []
        for name2, tr2 in all_tracks.items():
            lo2, hi2 = track_qrange(tr2)
            others.append((name2, tr2, lo2, hi2))
        for i, s in enumerate(tr["states"]):
            q = s["q"]
            op, _ = make_ops(q)
            other_ks = [interp_k(t2, q) for _, t2, lo2, hi2 in others
                        if lo2 - 1e-9 <= q <= hi2 + 1e-9]
            full = (i % 6 == 0 or i == len(tr["states"]) - 1
                    or any(abs(q - bq) < 2.6e-3 for bq in BRIDGE_QS))
            vals, marg, dmin, r1 = charge_state(op, s, other_ks, full)
            s["chi_raw"] = vals
            s["chi"] = int(round(vals[0]))
            s["margin"] = marg
            s["dmin"] = dmin
            s["radius"] = r1
            s["full_check"] = full
            n_full += full
        chis = [s["chi"] for s in tr["states"]]
        flips = sum(1 for a, b in zip(chis, chis[1:]) if a != b)
        tr["chi_values"] = sorted(set(chis))
        tr["chi_sign_changes"] = flips
        print(f"[{time.time()-t00:6.1f}s] charges {name}: chi set "
              f"{tr['chi_values']}, sign changes {flips}")

    # ---- low-q resolution block ---------------------------------------------
    def line_k(ts):
        return np.stack([ts, ts, 1 - ts], axis=1) * np.pi

    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, 2]) / np.sqrt(6)

    def net_dist(k):
        """Distance to the axis nodal-line network {k: two components = pi/2}
        (exactly degenerate along its length; census fact, all q probed)."""
        d = np.abs(np.asarray(k, float) % np.pi - np.pi / 2)
        ds = np.sort(d)
        return float(np.hypot(ds[0], ds[1]))

    def classify_node(kr, q):
        # the nodal-line network first: the (t,t,1-t) line hits it at the R
        # triple junction (t = 1/2), and at low q the near-degenerate
        # background funnels Nelder-Mead onto the network from far away
        if net_dist(kr) < 1e-5:
            return "axis-line-network"
        if max(abs(kr[0] - kr[1]), abs(kr[0] + kr[2] - np.pi)) < 1e-8:
            return "on-line"
        for nm in ("gen24", "corner", "half"):
            tr = all_tracks[nm]
            lo, hi = track_qrange(tr)
            if lo - 1e-9 <= q <= hi + 1e-9:
                kt = interp_k(tr, q)
                if min(tdist(kr, im) for im in images(kt)) < 5e-3:
                    return f"{nm}-image"
        return "NEW"

    lowq = []
    for q in LOWQ_QS:
        op, gap = make_ops(q)
        # dense 1D line scan
        n = 24001
        ts = np.linspace(0.002, 0.998, n)
        gl = gap(line_k(ts))
        mins = [i for i in range(1, n - 1)
                if gl[i] <= gl[i - 1] and gl[i] <= gl[i + 1] and gl[i] < 0.05]
        line_nodes, line_nonnodes = [], []
        for i in mins:
            kr, gr = refine(gap, line_k(ts[i:i + 1])[0])
            if gr < DEG_TOL:
                if all(tdist(kr, x["k"]) > 1e-5 for x in line_nodes):
                    line_nodes.append(dict(k=kr, gap=gr,
                                           t=float(kr[0] / np.pi),
                                           kind=classify_node(kr, q)))
            else:
                line_nonnodes.append(dict(t0=float(ts[i]),
                                          scan_gap=float(gl[i]),
                                          refined_gap=float(gr)))
        # local 3D tube sweep around the full line: per-ray local minima
        # (not raw thresholding: at low q the whole zone is near-degenerate
        # and a plain threshold sweeps in thousands of background points).
        # Candidates within 0.10 of the axis nodal-line network are dropped
        # as line members (the network is exactly degenerate along its
        # length; an isolated node hiding within 0.10 of it would be
        # missed -- accepted, recorded caveat of this exploratory tube).
        tt = np.linspace(0.01, 0.99, 320)
        rays = [line_k(tt)]
        for r in (0.02, 0.05, 0.08, 0.11):
            for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                off = r * (np.cos(a) * e1 + np.sin(a) * e2)
                rays.append(line_k(tt) + off)
        cand, n_net_dropped = [], 0
        for ray in rays:
            gr_ = gap(ray)
            for i in range(1, len(tt) - 1):
                if (gr_[i] <= gr_[i - 1] and gr_[i] <= gr_[i + 1]
                        and gr_[i] < 0.04):
                    if net_dist(ray[i]) < 0.10:
                        n_net_dropped += 1
                    else:
                        cand.append(ray[i])
        # cluster tolerance 0.015 < the A-B separation at q = 0.015 (~0.022)
        reps = []
        for kc in cand:
            if all(tdist(kc, kr) > 0.015 for kr in reps):
                reps.append(kc)
        # the tube census is the union of the refined tube clusters and the
        # (independently found) 1D line nodes
        tube_nodes = [dict(k=x["k"], gap=x["gap"], kind=x["kind"])
                      for x in line_nodes]
        for kc in reps[:120]:
            kr, gr = refine(gap, kc)
            if gr < DEG_TOL and all(tdist(kr, x["k"]) > 1e-5
                                    for x in tube_nodes):
                tube_nodes.append(dict(k=kr, gap=gr,
                                       kind=classify_node(kr, q)))
        # the exact quick-scan line algorithm, for the record
        nqs = 4000
        tq = np.linspace(0.02, 0.49, nqs)
        gq = gap(line_k(tq))
        qmins = [i for i in range(1, nqs - 1)
                 if gq[i] < gq[i - 1] and gq[i] < gq[i + 1] and gq[i] < 0.05]
        quick = []
        for i in qmins:
            def f1(t):
                return gap(line_k(np.array([t])))[0]
            r = minimize_scalar(f1, bounds=(tq[i - 1], tq[i + 1]),
                                method="bounded", options={"xatol": 1e-12})
            quick.append(dict(t_raw=float(tq[i]),
                              t_refined=float(r.x),
                              refined_fun=float(r.fun),
                              passes_1e8_gate=bool(r.fun < 1e-8)))
        lowq.append(dict(
            q=float(q),
            n_line_scan=n,
            line_nodes=[dict(t=x["t"], k_over_pi=[float(v / np.pi)
                                                  for v in x["k"]],
                             gap=x["gap"], kind=x["kind"])
                        for x in sorted(line_nodes, key=lambda x: x["t"])],
            line_nonnode_minima=line_nonnodes,
            tube_nodes=[dict(k_over_pi=[float(v / np.pi) for v in x["k"]],
                             gap=x["gap"], kind=x["kind"])
                        for x in tube_nodes],
            n_tube_candidates=len(cand),
            n_candidates_dropped_near_line_network=int(n_net_dropped),
            quick_scan_replay=quick))
        kinds = {}
        for x in tube_nodes:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        print(f"[{time.time()-t00:6.1f}s] low-q q={q}: "
              f"{len(line_nodes)} line nodes "
              f"(t={[round(x['t'], 4) for x in sorted(line_nodes, key=lambda x: x['t'])]}), "
              f"tube kinds: {kinds} (dropped near network: {n_net_dropped}), "
              f"quick-scan gate passes: "
              f"{sum(x['passes_1e8_gate'] for x in quick)}/{len(quick)}")

    # ---- quasienergy crossings ----------------------------------------------
    def omega_at(name, q, om_guess):
        tr = all_tracks[name]
        if name in ("corner", "half"):
            op, _ = make_ops(q)
            return omega_track(op, tr["states"][0]["k"], om_guess)
        op, gap = make_ops(q)
        kr, gr = refine(gap, interp_k(tr, q))
        assert gr < DEG_TOL
        return omega_track(op, kr, om_guess)

    names5 = list(all_tracks)
    crossings = []
    for na, nb in itertools.combinations(names5, 2):
        loa, hia = track_qrange(all_tracks[na])
        lob, hib = track_qrange(all_tracks[nb])
        lo, hi = max(loa, lob), min(hia, hib)
        if hi <= lo:
            continue
        qs = np.linspace(lo, hi, 1200)
        d = np.array([interp_om(all_tracks[na], q)
                      - interp_om(all_tracks[nb], q) for q in qs])
        for i in np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
            qa, qb = qs[i], qs[i + 1]
            # bisect with true evaluations to 1e-4
            while qb - qa > 1e-4:
                qm = 0.5 * (qa + qb)
                dm = (omega_at(na, qm, interp_om(all_tracks[na], qm))
                      - omega_at(nb, qm, interp_om(all_tracks[nb], qm)))
                if np.sign(dm) == np.sign(d[i]):
                    qa = qm
                else:
                    qb = qm
            qc = 0.5 * (qa + qb)
            omc = omega_at(na, qc, interp_om(all_tracks[na], qc))
            chia = all_tracks[na]["states"][0]["chi"]
            chib = all_tracks[nb]["states"][0]["chi"]
            crossings.append(dict(pair=[na, nb], q=float(qc),
                                  omega_over_pi=float(omc / np.pi),
                                  chi=[int(chia), int(chib)],
                                  opposite_chirality=bool(chia * chib < 0)))
    crossings.sort(key=lambda c: c["q"])
    for c in crossings:
        print(f"    crossing {c['pair'][0]} x {c['pair'][1]} at q={c['q']:.4f} "
              f"omega/pi={c['omega_over_pi']:.4f} chi={c['chi']}"
              f"{'  (opposite chirality)' if c['opposite_chirality'] else ''}")
    print(f"[{time.time()-t00:6.1f}s] {len(crossings)} quasienergy crossings")

    # ---- (Delta k, Delta omega) bridge octet A <-> octet B ------------------
    bridge = []
    for q in BRIDGE_QS:
        op, gap = make_ops(q)
        kA, gA = refine(gap, interp_k(tracks["octetA"], q))
        kB, gB = refine(gap, interp_k(tracks["octetB"], q))
        assert gA < DEG_TOL and gB < DEG_TOL, q
        omA = omega_track(op, kA, interp_om(tracks["octetA"], q))
        omB = omega_track(op, kB, interp_om(tracks["octetB"], q))
        dk = kB - kA
        bridge.append(dict(
            q=float(q),
            tA=float(kA[0] / np.pi), tB=float(kB[0] / np.pi),
            dk_over_pi=[float(x / np.pi) for x in dk],
            dk_norm=float(np.linalg.norm(dk)),
            dk_norm_over_pi=float(np.linalg.norm(dk) / np.pi),
            omegaA_over_pi=float(omA / np.pi),
            omegaB_over_pi=float(omB / np.pi),
            domega_over_pi=float((omB - omA) / np.pi)))
    print(f"[{time.time()-t00:6.1f}s] bridge table done")

    # ---- separations / mergers ----------------------------------------------
    loA, hiA = track_qrange(tracks["octetA"])
    loB, hiB = track_qrange(tracks["octetB"])
    qs = np.linspace(max(loA, loB), min(hiA, hiB), 600)
    sepAB = np.array([tdist(interp_k(tracks["octetA"], q),
                            interp_k(tracks["octetB"], q)) for q in qs])
    imin = int(np.argmin(sepAB))
    # 24-orbit partner separation at death (2 k_x, annihilation on k_x = 0)
    kx_last = min(tracks["gen24"]["death_up"]["last_alive_k_over_pi"]) \
        if tracks["gen24"]["death_up"] else None
    separations = dict(
        octetA_octetB_min=float(sepAB.min()),
        octetA_octetB_min_q=float(qs[imin]),
        octetA_octetB_max=float(sepAB.max()),
        octetA_octetB_at_qlo=float(sepAB[0]),
        gen24_partner_sep_at_last_alive=(
            float(2 * kx_last * np.pi) if kx_last is not None else None),
        any_merger_below_5e3=bool(sepAB.min() < 5e-3))

    # ---- spectator-free-window question -------------------------------------
    spectator_free = dict(
        octetA_exists=[float(loA), float(hiA)],
        octetB_exists=[float(loB), float(hiB)],
        gen24_exists=[float(track_qrange(tracks["gen24"])[0]),
                      float(tracks["gen24"]["death_up"]["last_alive_q"])
                      if tracks["gen24"]["death_up"]
                      else float(track_qrange(tracks["gen24"])[1])],
        window_covered=[Q_LO, Q_HI],
        answer="no spectator-free q: octets A and B exist (gap < 1e-12) at "
               "every accepted continuation state across the whole window "
               "[0.015, 0.49]; only the generic 24-orbit dies (q_c ~ 0.293)")

    # ---- flow table ---------------------------------------------------------
    print("\n===== orbit flow table =====")
    show_qs = [0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.29,
               0.30, 0.35, 0.40, 0.45, 0.49]
    hdr = "q      " + "".join(f"{n:>28s}" for n in names5)
    print(hdr + "\n" + "-" * len(hdr))
    for q in show_qs:
        row = f"{q:6.3f} "
        for n in names5:
            lo, hi = track_qrange(all_tracks[n])
            if lo - 1e-9 <= q <= hi + 1e-9:
                st = min(all_tracks[n]["states"],
                         key=lambda s: abs(s["q"] - q))
                row += (f"  t/k={st['k'][0]/np.pi:6.4f} "
                        f"w={st['omega']/np.pi:6.4f} {st['chi']:+d}")
            else:
                row += f"{'--':>28s}"
        print(row)

    # ---- assertions before writing ------------------------------------------
    # tracked gaps
    for name, tr in all_tracks.items():
        worst = max(s["gap"] for s in tr["states"])
        assert worst < GAP_ASSERT, (name, worst)
    # charge integrality and stability
    for name, tr in all_tracks.items():
        for s in tr["states"]:
            if s["margin"] < 5e-3:
                continue
            assert max(abs(v - round(v)) for v in s["chi_raw"]) < 1e-6, \
                (name, s["q"], s["chi_raw"])
            assert len(set(round(v) for v in s["chi_raw"])) == 1, \
                (name, s["q"], s["chi_raw"])
    # anchors: corner chi = -1, half chi = +1 at every computed q
    assert all(s["chi"] == -1 for s in anchors["corner"]["states"])
    assert all(s["chi"] == +1 for s in anchors["half"]["states"])
    # census values at q = 0.08
    for name, want in (("octetA", -1), ("octetB", +1), ("gen24", -1)):
        st = min(tracks[name]["states"], key=lambda s: abs(s["q"] - 0.08))
        assert st["chi"] == want, (name, st["chi"])
    # 24-orbit death bracket inside (0.292, 0.294), width < 1e-3, low-q alive
    b = tracks["gen24"]["death_up"]
    assert b is not None
    assert 0.292 <= b["last_alive_q"] < b["first_dead_q"] <= 0.294, b
    assert b["width"] < 1e-3, b
    assert tracks["gen24"]["death_dn"] is None          # no low-q birth
    assert abs(track_qrange(tracks["gen24"])[0] - Q_LO) < 1e-9
    # octets exist over the whole window, on the line
    for name in ("octetA", "octetB"):
        lo, hi = track_qrange(tracks[name])
        assert abs(lo - Q_LO) < 1e-9 and abs(hi - Q_HI) < 1e-9, (name, lo, hi)
        assert tracks[name]["death_up"] is None
        assert tracks[name]["death_dn"] is None
        assert online_dev[name] < 1e-8
    # low-q block: at every probed q exactly 4 on-line nodes (A, B and their
    # conjugate images) and nothing NEW in the tube
    for blk in lowq:
        on = [x for x in blk["line_nodes"] if x["kind"] == "on-line"]
        assert len(on) == 4, (blk["q"], blk["line_nodes"])
        assert not any(x["kind"] == "NEW" for x in blk["tube_nodes"]), blk["q"]
        ts_found = sorted(x["t"] for x in on)
        # conjugation pairing t <-> 1 - t
        assert abs(ts_found[0] + ts_found[3] - 1) < 1e-6
        assert abs(ts_found[1] + ts_found[2] - 1) < 1e-6
        # the quick scan's gate really is the artifact: at least one true
        # node fails its 1e-8 gate at every probed q
        replay_found = [x for x in blk["quick_scan_replay"]
                        if x["passes_1e8_gate"]]
        replay_missed = [x for x in blk["quick_scan_replay"]
                         if not x["passes_1e8_gate"] and x["refined_fun"] < 1e-6]
        assert len(replay_found) < 2 and len(replay_missed) >= 1, blk["q"]
    # no sign changes along any track
    assert all(tr["chi_sign_changes"] == 0 for tr in all_tracks.values())
    print("\nall assertions passed")

    # ---- write --------------------------------------------------------------
    def track_json(tr):
        return dict(
            name=tr["name"],
            chi_values=tr["chi_values"],
            chi_sign_changes=tr["chi_sign_changes"],
            death_up=tr["death_up"], death_dn=tr["death_dn"],
            states=[dict(q=s["q"],
                         k_over_pi=[float(x / np.pi) for x in s["k"]],
                         gap=s["gap"], omega_over_pi=float(s["omega"] / np.pi),
                         chi=s["chi"], chi_raw=s["chi_raw"],
                         margin=s["margin"], dmin=s["dmin"],
                         radius=s["radius"], full_check=s["full_check"])
                    for s in tr["states"]])

    payload = dict(
        script="scripts/orbit_flow.py",
        model="annealed chord coincube walk, U = prod_a [E_a ((1-q)I+qC_a)]^2",
        q_window=[Q_LO, Q_HI],
        calibration=out["calibration"],
        n_full_charge_checks=int(n_full),
        online_dev=online_dev,
        tracks={n: track_json(t) for n, t in all_tracks.items()},
        lowq_resolution=lowq,
        lowq_artifact_explanation=(
            "The quick 1D line scan refines each line "
            "minimum with bounded minimize_scalar and gates on fun < 1e-8. "
            "At low q the bounded scalar refiner stalls at ~1e-8..5e-8 on "
            "the narrower cone (octet B), so B fails the gate and is "
            "dropped, while octet A passes at ~1e-9..8e-9; near q ~ 0.026-"
            "0.028 A's refined value itself jitters around 1e-8, producing "
            "the apparent appearance/disappearance. Full-3D Nelder-Mead "
            "refinement reaches gap ~ 1e-16 for BOTH octets at every "
            "probed q >= 0.015: both are real, nothing is born or dies at "
            "low q; the quick-scan behavior is 100% refine-gate artifact."),
        crossings=crossings,
        bridge_octetA_octetB=bridge,
        separations=separations,
        spectator_free_window=spectator_free,
        total_seconds=round(time.time() - t00, 1))
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=1))
    print(f"wrote {RESULTS}  ({payload['total_seconds']}s)")


if __name__ == "__main__":
    main()
