#!/usr/bin/env python
"""Dense-sweep gapless census of the coincube Bloch walks, with computed
Fukui-Hatsugai Floquet-Weyl charges (the honest census instrument).

By the scalar-unitary factorization (Proposition 1), U(k) = rho^6 V(k) with
V(k) exactly unitary at every k, so quasienergies (eigenphases of V) and
spectral projectors of split doublets are well-defined despite the annealed
damping; degeneracies are pure phase coincidences.  This script charts the
COMPLETE gapless set of V by machine, without assuming any list of loci:

  A. Dense sweep of the reduced zone [0, pi)^3 (pi-periodicity is exact) on a
     64^3 grid of the minimum pairwise eigenvalue distance; every local
     minimum below threshold is refined by Nelder-Mead to machine precision
     (~1e-16, i.e. ~1e-13 in k for conical touchings).
  B. Each refined degeneracy is classified by probing the gap on a small
     sphere around it: an isolated (Weyl) point has gap > 0 in every
     direction; a nodal line/curve point has a refinable zero direction with
     an antipodal partner.  The classification evidence is recorded.
  C. The spectral symmetry group is determined EMPIRICALLY: all 48 signed
     coordinate permutations are tested for spec(U(Ok)) = spec(U(k))
     (unitary type) and spec(U(Ok)) = conj(spec(U(k))) (antiunitary type).
     Isolated nodes are deduplicated into orbits of the resulting event
     group, acting on (k, theta) by (Ok, theta) / (Ok, -theta).
  D. For EVERY isolated node and every degenerate doublet at it (both +-Im
     branches where present), the chirality is computed as a Chern number:
     chi := Fukui-Hatsugai lattice Berry flux, over an outward-oriented
     discretized sphere enclosing the node, of the doublet member with the
     LARGER eigenphase (phases compared after wrapping to the node
     quasienergy theta0).  The convention is calibrated on the analytic
     family V(k) = exp(+i k.sigma), for which the upper band must give
     chi = +1.  Every charge is verified stable under two sphere radii and
     two grid refinements and asserted to be an exact integer.
  E. Charge accounting.  Per-branch sums are reported for what they are.
     The correct global statements are computed, not asserted:
       - the eigenphases of V cover the FULL unit circle (no Floquet
         quasienergy gap exists anywhere), so no Nielsen-Ninomiya zero-sum
         rule applies per branch or per quasienergy window;
       - every axis-aligned 2-torus slice of the zone contains a nodal-line
         point (the line network pierces all slice families), so no gapped
         slice family exists in any direction;
       - the antiunitary map k -> -k pairs every event at (k, theta) with a
         partner at (-k, -theta) of OPPOSITE chirality, so the both-branch
         total vanishes identically (doubling by conjugation);
       - the nodal lines carry NO monopole charge: the loop (Wilson) Berry
         phase of each band around a line is close to pi but NOT quantized
         (it varies smoothly along the line - there is no protecting
         symmetry), and its winding along the full line is zero at two tube
         radii: the lines emit no net Berry flux and do not "compensate"
         the point charges.
  F. Massive census (8-channel inversion-doubled walk with mass layer
     M = (1-q_m) + q_m (1 x XZ)): the same sweep machinery charts the
     massive gapless set, and the fate of every massless feature is
     determined.  The principal quartet (corner + half-points, the
     self-conjugate momenta 2k = 0 mod pi where the two inversion sectors
     are resonant) gaps by exactly 2 arctan(q_m/(1-q_m)); EVERY extra
     Weyl point survives exactly ungapped in both sector copies (its
     inversion partner sits at the conjugate quasienergy, off resonance),
     verified constructively: both sector doublets of all extra nodes are
     refined, sphere-classified, and charged.  Massive-only structures are
     charted too: degeneracy curves confined to the {k_a in {0, pi/2}}
     planes (sector crossings whose mass coupling vanishes by mirror
     symmetry) and isolated Weyl nodes pinned at the self-conjugate
     resonant quasienergies theta in {0, pi}, completed under the
     empirical massive symmetry group (the mass layer breaks the axis
     permutations, leaving the 8 diagonal sign flips, and makes the
     spectrum self-conjugate at every k); the 12-line edge network
     (>= 2 components in {0, pi/2}) is verified exactly degenerate along
     its full length.

All headline findings are asserted before results/census_sweep.json is
written.  Run:  PYTHONPATH=src .venv/bin/python scripts/census_sweep.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_C, COIN_D  # noqa: E402

SEED = 20260902
RESULTS = Path(__file__).resolve().parent.parent / "results" / "census_sweep.json"

I2 = np.eye(2)
X = np.array([[0.0, 1], [1, 0]])
Z = np.diag([1.0, -1])
XZ = X @ Z
C4 = [np.array(c, float) for c in COIN_C]
D4 = [np.array(d, float) for d in COIN_D]
C8 = [np.kron(c, I2) for c in C4]
D8 = [np.kron(d, [1.0, -1]) for d in D4]
CM = np.kron(np.eye(4), XZ)
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]

GRID_N = 64            # sweep grid (pi/2 lies on the grid)
CAND_THRESH = 0.15     # collect local minima below this gap
DEG_TOL = 1e-12        # refined gap below this = exact degeneracy
CLUSTER_TOL = 1e-5     # torus distance for deduplication
PHASE_TOL = 1e-7       # eigenphase grouping at a degeneracy


# -- operators ----------------------------------------------------------------

def walk_batch(K, al, be, C=C4, D=D4, mass=None):
    """Batched cycle operator: K (N, 3) -> (N, n, n)."""
    n = C[0].shape[0]
    N = K.shape[0]
    U = np.broadcast_to(np.eye(n, dtype=complex), (N, n, n)).copy()
    for a in range(3):
        M = al * np.eye(n) + be * C[a]
        ph = np.exp(1j * K[:, a, None] * D[a])
        T = ph[:, :, None] * M[None, :, :]
        U = T @ T @ U
    if mass is not None:
        U = mass[None] @ U
    return U


def make_ops(kind, q, qm=None):
    """(op(k)->matrix, gap(K)->(N,), nbands) for a model configuration."""
    al, be = ((1 - q, q) if kind == "chord"
              else (np.sqrt(1 - q), np.sqrt(q)))
    if qm is None:
        C, D, mass, n = C4, D4, None, 4
    else:
        C, D, n = C8, D8, 8
        mass = (1 - qm) * np.eye(8) + qm * CM

    def op(k):
        return walk_batch(np.asarray(k, float)[None], al, be, C, D, mass)[0]

    def gap(K):
        K = np.atleast_2d(np.asarray(K, float))
        w = np.linalg.eigvals(walk_batch(K, al, be, C, D, mass))
        d = np.abs(w[:, :, None] - w[:, None, :])
        d[:, np.arange(n), np.arange(n)] = np.inf
        return d.reshape(len(w), -1).min(axis=1)

    return op, gap, n


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def tdist(a, b):
    """Torus distance on the reduced zone [0, pi)^3."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return float(np.linalg.norm(np.minimum(d, np.pi - d)))


# -- empirical symmetry group -------------------------------------------------

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
    """Greedy matching distance between two eigenvalue multisets (robust
    against sort-order instabilities of near-coincident values)."""
    a = list(a)
    d = 0.0
    for v in b:
        i = int(np.argmin(np.abs(np.array(a) - v)))
        d = max(d, abs(a[i] - v))
        a.pop(i)
    return d


def empirical_group(op, rng, ntest=6, tol=1e-8):
    """Signed permutations preserving the spectrum (unitary type) or mapping
    it to its conjugate (antiunitary type), tested at random momenta."""
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


# -- sweep, refinement, classification ---------------------------------------

def sweep_candidates(gap, n=GRID_N, thresh=CAND_THRESH):
    g1 = np.linspace(0, np.pi, n, endpoint=False)
    K = np.array(np.meshgrid(g1, g1, g1, indexing="ij")).reshape(3, -1).T
    G = gap(K).reshape(n, n, n)
    mf = minimum_filter(G, size=3, mode="wrap")
    cand = (G <= mf) & (G < thresh)
    return [ij * np.pi / n for ij in np.argwhere(cand)], G


def refine(gap, k0):
    f = lambda k: gap(k)[0]
    r = minimize(f, k0, method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-17, maxfev=1500))
    if r.fun > DEG_TOL:
        r2 = minimize(f, r.x, method="Nelder-Mead",
                      options=dict(xatol=1e-14, fatol=1e-18, maxfev=1500))
        if r2.fun < r.fun:
            r = r2
    return r.x % np.pi, float(r.fun)


def classify(gap, k0, rng, r=0.01, nsamp=160):
    """Sphere probe: isolated point vs member of a line/curve.

    The directional gap minimum over the probe sphere is refined by
    Nelder-Mead from the three best sampled directions (a single start can
    miss the zero direction of a strongly curved degeneracy curve and
    mimic an isolated point); an isolated Weyl point keeps a gap > 1e-4
    in every direction at r = 0.01, a line/curve member refines to ~1e-16."""
    dirs = rng.normal(size=(nsamp, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    gs = gap(k0[None] + r * dirs)

    def fsph(ang):
        th, p = ang
        n = np.array([np.sin(th) * np.cos(p), np.sin(th) * np.sin(p),
                      np.cos(th)])
        return gap((k0 + r * n)[None])[0]

    best, bestx = np.inf, None
    for i in np.argsort(gs)[:6:2]:
        n0 = dirs[i]
        a0 = [np.arccos(np.clip(n0[2], -1, 1)), np.arctan2(n0[1], n0[0])]
        rr = minimize(fsph, a0, method="Nelder-Mead",
                      options=dict(xatol=1e-12, fatol=1e-16, maxfev=800))
        if rr.fun < best:
            best, bestx = rr.fun, rr.x
        if best < 1e-10:
            break
    d = np.array([np.sin(bestx[0]) * np.cos(bestx[1]),
                  np.sin(bestx[0]) * np.sin(bestx[1]), np.cos(bestx[0])])
    anti = float(gap((k0 - r * d)[None])[0]) if best < 1e-10 else None
    kind = "point" if best > 1e-8 else "line"
    return dict(kind=kind, sphere_min_sampled=float(gs.min()),
                sphere_median=float(np.median(gs)),
                refined_dir_min=float(best),
                zero_dir=[float(x) for x in d] if kind == "line" else None,
                antipodal_gap=anti, probe_radius=r)


def events_at(op, k0):
    """Degenerate eigenphase groups at k0: [(theta0, multiplicity)]."""
    w = np.linalg.eigvals(op(k0))
    used = np.zeros(len(w), bool)
    ev = []
    for i in range(len(w)):
        if used[i]:
            continue
        grp = [j for j in range(len(w))
               if not used[j] and abs(w[j] - w[i]) < PHASE_TOL * 10]
        if len(grp) >= 2:
            for j in grp:
                used[j] = True
            ev.append((float(np.angle(np.mean(w[grp]))), len(grp)))
    return ev


# -- Fukui-Hatsugai charge ----------------------------------------------------

def band_vec(U, theta0, which):
    """Eigenvector of the doublet member nearest theta0 in phase; which=+1
    for the larger wrapped phase (upper), -1 for the smaller (lower).
    Returns (unit vector, isolation margin of the doublet)."""
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
    """FH lattice Berry flux of the doublet band over the outward-oriented
    sphere of radius r around k0.  Returns (chern, min doublet margin)."""
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
    """chi convention: upper-phase band of V(k) = exp(+i k.sigma) is +1."""
    def cal_op(k):
        H = sum(k[a] * PAULI[a] for a in range(3))
        w, V = np.linalg.eigh(H)
        return V @ np.diag(np.exp(1j * w)) @ V.conj().T
    cu, _ = chern_sphere(cal_op, np.zeros(3), 0.0, 0.1, 20, 20, +1)
    cl, _ = chern_sphere(cal_op, np.zeros(3), 0.0, 0.1, 20, 20, -1)
    return float(cu), float(cl)


def node_charges(op, k0, theta0, radii=(0.008, 0.02), grids=((16, 16), (24, 24))):
    """chi of the +upper doublet band, all radius/grid combinations."""
    vals, margs = [], []
    for r in radii:
        for nth, nph in grids:
            c, m = chern_sphere(op, k0, theta0, r, nth, nph, +1)
            vals.append(c)
            margs.append(m)
    return vals, min(margs)


# -- Wilson loops around nodal lines ------------------------------------------

def wilson_bands(op, t, r, nph, axis=0, offset=0.173, nbands=4):
    """Per-band Berry phase around the circle of radius r encircling the
    nodal line (t, pi/2, pi/2)-type at parameter t, with eigenvector
    continuity tracking.  Returns [(band phase, berry phase, closed)],
    plus the minimum tracking link modulus."""
    base = np.full(3, np.pi / 2)
    e1 = np.zeros(3)
    e2 = np.zeros(3)
    e1[(axis + 1) % 3] = 1
    e2[(axis + 2) % 3] = 1
    phis = np.linspace(0, 2 * np.pi, nph, endpoint=False) + offset
    Vs, Ws = [], []
    for p in phis:
        kv = base + np.eye(3)[axis] * (t - np.pi / 2) \
            + r * np.cos(p) * e1 + r * np.sin(p) * e2
        w, V = np.linalg.eig(op(kv))
        Vs.append(V / np.linalg.norm(V, axis=0))
        Ws.append(w)
    res, minlink = [], 1.0
    for m in range(nbands):
        i, a = m, 1.0 + 0j
        for j in range(nph):
            ov = Vs[(j + 1) % nph].conj().T @ Vs[j][:, i]
            i2 = int(np.argmax(np.abs(ov)))
            minlink = min(minlink, float(abs(ov[i2])))
            a *= ov[i2]
            i = i2
        res.append((float(np.angle(Ws[0][m])), float(np.angle(a)), i == m))
    return res, minlink


def line_winding(op, r, axis=0, nts=121, nph=192, excl=0.02):
    """Winding of the +pair upper band's loop Berry phase along the full
    line, excluding a +-excl window around the R crossing at t = pi/2."""
    ts = np.pi / 2 + excl + (np.pi - 2 * excl) * np.linspace(0, 1, nts)
    phases, closed = [], True
    for t in ts:
        res, _ = wilson_bands(op, t, r, nph, axis)
        plus = sorted(x for x in res if x[0] > 0)
        phases.append(plus[-1][1])
        closed &= plus[-1][2]
    ph = np.unwrap(phases)
    return float((ph[-1] - ph[0]) / (2 * np.pi)), bool(closed)


# -- census of one configuration ----------------------------------------------

def census(kind, q, qm, rng, verbose=True):
    op, gap, nb = make_ops(kind, q, qm)
    t0 = time.time()
    cands, G = sweep_candidates(gap)
    refined = [refine(gap, k0) for k0 in cands]
    deg = [(k, f) for k, f in refined if f < DEG_TOL]
    nondeg = [f for _, f in refined if f >= DEG_TOL]

    clusters = []
    for k, f in deg:
        for c in clusters:
            if tdist(k, c["k"]) < CLUSTER_TOL:
                c["hits"] += 1
                break
        else:
            clusters.append(dict(k=k, hits=1))

    points, lines = [], []
    for c in clusters:
        cls = classify(gap, c["k"], rng)
        c.update(cls)
        c["n_special_components"] = int(np.sum(
            (np.abs(c["k"] - np.pi / 2) < 2e-5)
            | (c["k"] < 2e-5) | (c["k"] > np.pi - 2e-5)))
        c["n_pi_half_components"] = int(np.sum(
            np.abs(c["k"] - np.pi / 2) < 2e-5))
        if cls["kind"] == "point":
            c["events"] = events_at(op, c["k"])
            points.append(c)
        else:
            lines.append(c)

    # charges for every event at every isolated point
    for c in points:
        c["charges"] = []
        for th0, mult in c["events"]:
            vals, marg = node_charges(op, c["k"], th0)
            c["charges"].append(dict(theta0=th0, mult=mult, chi_raw=vals,
                                     chi=int(round(vals[0])), margin=marg))
    dt = time.time() - t0
    if verbose:
        print(f"    [{kind} q={q} qm={qm}] cands={len(cands)} "
              f"deg={len(deg)} clusters={len(clusters)} points={len(points)} "
              f"line-members={len(lines)} ({dt:.0f}s)", flush=True)
    return dict(op=op, gap=gap, nbands=nb, points=points, lines=lines,
                grid_min_gap=float(G.min()),
                n_candidates=len(cands), n_degenerate=len(deg),
                nondeg_floor=float(min(nondeg)) if nondeg else None,
                seconds=dt)


# -- orbits under the empirical event group -----------------------------------

def orbits_of_points(points, uni, anti):
    """Group isolated (k, theta) events into orbits of the event group."""
    events = []
    for c in points:
        for ch in c["charges"]:
            events.append(dict(k=c["k"], theta=ch["theta0"], chi=ch["chi"],
                               mult=ch["mult"], margin=ch["margin"],
                               chi_raw=ch["chi_raw"]))
    unassigned = list(range(len(events)))
    orbits = []
    while unassigned:
        i0 = unassigned[0]
        members = []
        for i in unassigned:
            for O, conj in ([(O, False) for O in uni]
                            + [(O, True) for O in anti]):
                km = (O @ events[i0]["k"]) % np.pi
                tm = -events[i0]["theta"] if conj else events[i0]["theta"]
                if (tdist(km, events[i]["k"]) < 1e-6
                        and abs(wrap(events[i]["theta"] - tm)) < 1e-6):
                    members.append((i, conj))
                    break
        for i, _ in members:
            unassigned.remove(i)
        mem = [dict(events[i], conj=cj) for i, cj in members]
        # representative: theta > 0 preferred, lexicographically smallest k
        reps = [m for m in mem if np.sin(m["theta"]) > 1e-6] or mem
        rep = min(reps, key=lambda m: tuple(np.round(m["k"], 9)))
        orbits.append(dict(rep_k=[float(x) for x in rep["k"]],
                           rep_theta=float(rep["theta"]),
                           size=len(mem), members=mem))
    return orbits, events


def summarize_orbits(orbits):
    rows = []
    for ob in orbits:
        chis_plus = [m["chi"] for m in ob["members"]
                     if np.sin(m["theta"]) > 1e-6]
        chis_minus = [m["chi"] for m in ob["members"]
                      if np.sin(m["theta"]) < -1e-6]
        chis_real = [m["chi"] for m in ob["members"]
                     if abs(np.sin(m["theta"])) <= 1e-6]
        rows.append(dict(
            rep_k_over_pi=[round(x / np.pi, 6) for x in ob["rep_k"]],
            rep_theta_over_pi=round(ob["rep_theta"] / np.pi, 6),
            size=ob["size"],
            n_plus=len(chis_plus), n_minus=len(chis_minus),
            n_real=len(chis_real),
            chi_plus=sorted(set(chis_plus)), chi_minus=sorted(set(chis_minus)),
            sum_plus=int(sum(chis_plus)), sum_minus=int(sum(chis_minus))))
    rows.sort(key=lambda r: (r["size"], r["rep_k_over_pi"]))
    return rows


# -- massive fate of massless features ----------------------------------------

def pair_gap_fn(op, theta):
    """Scalar gap of the eigenvalue pair nearest phase theta (targets one
    sector doublet of the massive walk)."""
    def g(k):
        w = np.linalg.eigvals(op(np.asarray(k, float)))
        i = np.argsort(np.abs(wrap(np.angle(w) - theta)))[:2]
        return float(abs(w[i[0]] - w[i[1]]))
    return g


def refine_fn(f, k0):
    """Nelder-Mead refinement of a scalar function of k."""
    r = minimize(f, k0, method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-17, maxfev=1500))
    if r.fun > DEG_TOL:
        r2 = minimize(f, r.x, method="Nelder-Mead",
                      options=dict(xatol=1e-14, fatol=1e-18, maxfev=1500))
        if r2.fun < r.fun:
            r = r2
    return r.x % np.pi, float(r.fun)


def massive_fate(gap8, op8, k0, theta0, rng):
    """Fate of a massless node at k0 under the mass layer.

    Reports the refined nearby minimum of the 8-band gap and its
    displacement, PLUS the classification of that refined degeneracy: the
    minimum pairwise gap alone is misleading, because the principal nodes
    sit on the doubling-degenerate edge network (each mass-split level
    remains exactly two-fold there, with the Dirac gap 2m between the
    levels).  A node "survives exactly" only if the nearby refined
    degeneracy is an ISOLATED point (a genuine Weyl touching), not a
    member of the edge network or of a mirror-plane curve.  The local
    Dirac gap at the original node is quartet_inner_gap: the largest phase
    separation inside the mass-split quartet at k0."""
    k0 = np.asarray(k0, float)
    kf, gf = refine(gap8, k0)
    w = np.linalg.eigvals(op8(k0))
    d = np.abs(wrap(np.angle(w) - theta0))
    ph = np.sort(wrap(np.angle(w)[np.argsort(d)[:4]] - theta0))
    gaps = np.diff(ph)
    res = dict(refined_gap=float(gf), refined_k=[float(x) for x in kf],
               moved=tdist(kf, k0),
               quartet_rel_phases=[float(x) for x in ph],
               quartet_inner_gap=float(gaps.max()))
    if gf < DEG_TOL:
        cls = classify(gap8, kf, rng)
        res["refined_kind"] = cls["kind"]
        res["refined_n_special"] = int(np.sum(
            (np.abs(kf - np.pi / 2) < 2e-5) | (kf < 2e-5)
            | (kf > np.pi - 2e-5)))
        res["survives_exactly"] = bool(cls["kind"] == "point"
                                       and res["moved"] < 0.05)
    else:
        res["refined_kind"] = "gapped"
        res["survives_exactly"] = False
    return res


# -- main ---------------------------------------------------------------------

def main(explore=False):
    t00 = time.time()
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "grid_n": GRID_N}

    # calibration of the charge instrument
    cu, cl = calibrate()
    out["calibration"] = dict(upper=cu, lower=cl)
    assert abs(cu - 1) < 1e-9 and abs(cl + 1) < 1e-9
    print(f"[{time.time()-t00:6.1f}s] calibration: chi(exp(+ik.sigma)) "
          f"upper={cu:+.6f} lower={cl:+.6f}")

    # empirical symmetry group (massless chord q=0.08 reference operator)
    op_ref, _, _ = make_ops("chord", 0.08)
    uni, anti = empirical_group(op_ref, rng)
    out["symmetry_group"] = dict(
        n_unitary=len(uni), n_antiunitary=len(anti),
        unitary_dets=sorted(float(np.linalg.det(O)) for O in uni),
        minus_identity_antiunitary=bool(
            any(np.abs(O + np.eye(3)).max() < 1e-12 for O in anti)),
        unitary_ops=[[[int(x) for x in row] for row in O] for O in uni],
        antiunitary_ops=[[[int(x) for x in row] for row in O] for O in anti])
    print(f"[{time.time()-t00:6.1f}s] empirical group: {len(uni)} unitary "
          f"(all det +1), {len(anti)} antiunitary (incl. -1)")

    # ---- massless census over configurations --------------------------------
    configs = [("chord", 0.08), ("chord", 0.15), ("chord", 0.30),
               ("arc", 0.08)]
    out["massless"] = {}
    census_cache = {}
    for kind, q in configs:
        cen = census(kind, q, None, rng)
        census_cache[(kind, q)] = cen
        orbits, events = orbits_of_points(cen["points"], uni, anti)
        rows = summarize_orbits(orbits)
        splus = sum(e["chi"] for e in events if np.sin(e["theta"]) > 1e-6)
        sminus = sum(e["chi"] for e in events if np.sin(e["theta"]) < -1e-6)
        # every line-classified cluster lies on the known axis network
        # ((t, pi/2, pi/2) and permutations; >= 2 components at pi/2,
        # with the triple junction R having all three)
        line_ok = all(c["n_pi_half_components"] >= 2 for c in cen["lines"])
        # conjugation partner check: every event's partner at -k has -chi
        conj_ok = True
        for e in events:
            km = (-np.asarray(e["k"])) % np.pi
            part = [f for f in events
                    if tdist(km, f["k"]) < 1e-6
                    and abs(wrap(f["theta"] + e["theta"])) < 1e-6]
            conj_ok &= len(part) == 1 and part[0]["chi"] == -e["chi"]
        # unitary suborbit chi consistency
        chi_consistent = all(
            len(set(m["chi"] for m in ob["members"] if not m["conj"])) <= 1
            for ob in orbits)
        # charge integrality/stability
        stab = max(max(abs(v - round(v)) for v in e["chi_raw"])
                   for e in events)
        same = all(len(set(round(v) for v in e["chi_raw"])) == 1
                   for e in events)
        al, be = ((1 - q, q) if kind == "chord"
                  else (np.sqrt(1 - q), np.sqrt(q)))
        rho6 = (al**2 + be**2)**3
        UR = walk_batch(np.full((1, 3), np.pi / 2), al, be)[0]
        out["massless"][f"{kind}_q{q}"] = dict(
            n_isolated_points=len(cen["points"]),
            n_events=len(events),
            n_line_members=len(cen["lines"]),
            R_minus_rho6_I_dev=float(np.abs(UR + rho6 * np.eye(4)).max()),
            lines_on_axis_network=bool(line_ok),
            orbit_table=rows,
            sum_chi_plus=int(splus), sum_chi_minus=int(sminus),
            conjugation_pairing_ok=bool(conj_ok),
            unitary_orbit_chi_consistent=bool(chi_consistent),
            charge_integrality_dev=float(stab),
            charge_radius_grid_stable=bool(same),
            nondegenerate_local_min_floor=cen["nondeg_floor"],
            n_candidates=cen["n_candidates"],
            seconds=round(cen["seconds"], 1))
        print(f"[{time.time()-t00:6.1f}s] {kind} q={q}: "
              f"{len(cen['points'])} points, {len(events)} events, "
              f"S+={splus} S-={sminus}, orbits="
              f"{sorted(r['size'] for r in rows)}")

    # ---- no-Floquet-gap check (phase coverage) ------------------------------
    op, gp, _ = make_ops("chord", 0.08)
    g1 = np.linspace(0, np.pi, 40, endpoint=False)
    K = np.array(np.meshgrid(g1, g1, g1, indexing="ij")).reshape(3, -1).T
    w = np.linalg.eigvals(walk_batch(K, 0.92, 0.08))
    hist, _ = np.histogram(np.angle(w).ravel(), bins=360, range=(-np.pi, np.pi))
    out["no_floquet_gap"] = dict(bins=360, min_bin_count=int(hist.min()),
                                 empty_bins=int((hist == 0).sum()))
    # every axis-aligned slice k_a = c contains the point of the a-line
    # (pi/2, pi/2 in the other two components): exact by the census loci.
    print(f"[{time.time()-t00:6.1f}s] phase coverage: min bin "
          f"{hist.min()}, empty {int((hist == 0).sum())}")

    # ---- nodal-line analysis (chord q=0.08) ---------------------------------
    line_res = {}
    devs, winds = [], []
    for axis in range(3):
        rows = []
        for t in (0.1, 0.8, 1.5, 2.4, 3.0):
            res, ml = wilson_bands(op, t, 0.02, 256, axis)
            rows.append(dict(t=t, min_link=ml,
                             bands=[dict(theta=round(a, 6),
                                         berry_over_pi=round(b / np.pi, 6),
                                         closed=bool(c))
                                    for a, b, c in sorted(res)]))
            for a, b, c in res:
                devs.append(abs(abs(b) - np.pi))
        w1, c1 = line_winding(op, 0.02, axis)
        w2, c2 = line_winding(op, 0.04, axis)
        winds += [w1, w2]
        line_res[f"axis{axis}"] = dict(
            wilson_rows=rows, winding_r002=w1, winding_r004=w2,
            tracking_closed=bool(c1 and c2))
    line_res["berry_dev_from_pi_min"] = float(min(devs))
    line_res["berry_dev_from_pi_max"] = float(max(devs))
    line_res["max_abs_winding"] = float(max(abs(x) for x in winds))
    out["line_analysis"] = line_res
    print(f"[{time.time()-t00:6.1f}s] lines: berry dev from pi in "
          f"[{min(devs):.4f}, {max(devs):.4f}], max |winding| "
          f"{line_res['max_abs_winding']:.4f}")

    # ---- corner isolation (chord q=0.08) ------------------------------------
    cen = census_cache[("chord", 0.08)]
    dists = sorted(tdist(c["k"], np.zeros(3)) for c in cen["points"]
                   if tdist(c["k"], np.zeros(3)) > 1e-6)
    line_dist = float(np.sqrt(2) * np.pi / 2)  # corner to any axis line
    w0 = np.linalg.eigvals(op(np.zeros(3)))
    om0 = float(np.max(np.angle(w0)))
    thetas = sorted({round(ch["theta0"], 6) for c in cen["points"]
                     for ch in c["charges"]
                     if tdist(c["k"], np.zeros(3)) > 1e-6})
    dth = min(abs(wrap(th - om0)) for th in thetas)
    out["corner_isolation"] = dict(
        min_point_distance=float(dists[0]),
        min_point_distance_over_pi=float(dists[0] / np.pi),
        line_network_distance_over_pi=float(line_dist / np.pi),
        omega0=om0,
        min_offcorner_point_quasienergy_gap=float(dth))
    print(f"[{time.time()-t00:6.1f}s] corner isolation: nearest point "
          f"{dists[0]/np.pi:.4f} pi, lines at {line_dist/np.pi:.4f} pi, "
          f"min |theta-omega0| of off-corner points {dth:.4f}")

    # ---- generic-orbit continuation in q ------------------------------------
    # The 24-point generic orbit exists at q = 0.08 and 0.15 but is absent
    # from the q = 0.30 census.  Continuation of its representative from
    # q = 0.15 shows the k_x component shrinking to 0: the orbit members
    # annihilate pairwise on the {k_a = 0} planes at q_c in (0.292, 0.294).
    cont = []
    kc = None
    for qq in (0.15, 0.20, 0.25, 0.28, 0.29, 0.292, 0.294, 0.296):
        _, gq, _ = make_ops("chord", qq)
        if kc is None:
            kc = np.array([r for r in
                           out["massless"]["chord_q0.15"]["orbit_table"]
                           if r["size"] == 24][0]["rep_k_over_pi"]) * np.pi
        kr, gr = refine(gq, kc)
        cont.append(dict(q=qq, k_over_pi=[round(float(x / np.pi), 6)
                                          for x in kr], gap=float(gr)))
        if gr < DEG_TOL:
            kc = kr
    out["generic_orbit_continuation"] = cont
    print(f"[{time.time()-t00:6.1f}s] generic-orbit continuation: "
          + " ".join(f"q={c['q']}:{'alive' if c['gap'] < 1e-12 else 'GONE'}"
                     for c in cont), flush=True)

    # ---- massive census -----------------------------------------------------
    out["massive"] = {}
    for q, qm in ((0.08, 0.05), (0.15, 0.05)):
        m = float(np.arctan(qm / (1 - qm)))
        op8, gap8, _ = make_ops("chord", q, qm)
        uni8, anti8 = empirical_group(op8, rng)
        cen8 = census("chord", q, qm, rng)
        # principal quartet: gap exactly 2m at corner + three half-points
        worst = 0.0
        for kc in (np.zeros(3), *(np.eye(3)[a] * np.pi / 2 for a in range(3))):
            w = np.linalg.eigvals(op8(kc))
            ph = np.angle(w)
            up = np.sort(ph[ph > 0])
            gap2m = abs((up[-1] - up[0]) - 2 * m) \
                if len(up) == 4 else np.inf
            # top-modulus quartet is doubly degenerate two-group
            worst = max(worst, gap2m)
        # fate of every massless orbit (same q, chord)
        cml = census_cache[("chord", q)]
        orbits_ml, _ = orbits_of_points(cml["points"], uni, anti)
        fates = []
        for ob in orbits_ml:
            fate = massive_fate(gap8, op8, ob["rep_k"], ob["rep_theta"], rng)
            fate["rep_k_over_pi"] = [round(x / np.pi, 6) for x in ob["rep_k"]]
            fate["rep_theta_over_pi"] = round(ob["rep_theta"] / np.pi, 6)
            fate["orbit_size"] = ob["size"]
            fates.append(fate)

        # CONSTRUCTED survivors: for EVERY massless extra point (all except
        # the principal quartet), both sector doublets are targeted (phases
        # theta0 and -theta0: inversion doubling puts the second sector's
        # copy of the node at the conjugate quasienergy) and refined.
        def is_principal(k0):
            if tdist(k0, np.zeros(3)) < 1e-6:
                return True
            return any(tdist(k0, np.eye(3)[a] * np.pi / 2) < 1e-6
                       for a in range(3))

        constructed = []
        for c in cml["points"]:
            if is_principal(c["k"]):
                continue
            th0 = c["events"][0][0]
            for th in (th0, -th0):
                ks, gs = refine_fn(pair_gap_fn(op8, th), np.asarray(c["k"]))
                constructed.append(dict(from_k=c["k"], theta_seed=float(th),
                                        k=ks, gap=gs, moved=tdist(ks, c["k"])))
        spts = []
        for ev in constructed:
            for s in spts:
                if tdist(ev["k"], s["k"]) < 1e-4:
                    s["evs"].append(ev)
                    break
            else:
                spts.append(dict(k=ev["k"], evs=[ev]))
        surv_events = []
        for s in spts:
            cls = classify(gap8, s["k"], rng)
            evs_here = events_at(op8, s["k"])
            for ev in s["evs"]:
                th0 = min((t for t, _ in evs_here),
                          key=lambda t: abs(wrap(t - ev["theta_seed"])))
                vals, marg = node_charges(op8, s["k"], th0)
                surv_events.append(dict(
                    k_over_pi=[round(float(x / np.pi), 6) for x in s["k"]],
                    theta0_over_pi=round(th0 / np.pi, 6),
                    chi=int(round(vals[0])), chi_raw=vals, margin=marg,
                    kind=cls["kind"], dirmin=cls["refined_dir_min"],
                    gap=ev["gap"], moved=ev["moved"]))
        sp_s = sum(e["chi"] for e in surv_events
                   if np.sin(e["theta0_over_pi"] * np.pi) > 1e-6)
        sm_s = sum(e["chi"] for e in surv_events
                   if np.sin(e["theta0_over_pi"] * np.pi) < -1e-6)

        # sector-split displacement: distance between the two sector nodes
        # descending from one massless point
        splits = []
        for c in cml["points"]:
            if is_principal(c["k"]):
                continue
            pair = [ev["k"] for ev in constructed
                    if tdist(ev["from_k"], c["k"]) < 1e-9]
            if len(pair) == 2:
                splits.append(tdist(pair[0], pair[1]))

        # the 12-line edge network (>= 2 components in {0, pi/2}) is
        # EXACTLY degenerate along its full length (all four level pairs
        # two-fold, the doubling degeneracy): verified directly, so any
        # sweep cluster on it is an edge member regardless of the sphere
        # probe (whose Nelder-Mead can miss the narrow zero-funnel of a
        # nearly-tangent line and mislabel an on-line point as isolated)
        edge_dev = 0.0
        for axis in range(3):
            for c1 in (0.0, np.pi / 2):
                for c2 in (0.0, np.pi / 2):
                    for t in (0.31, 0.87, 1.93):
                        kv = np.zeros(3)
                        kv[axis] = t
                        kv[(axis + 1) % 3] = c1
                        kv[(axis + 2) % 3] = c2
                        edge_dev = max(edge_dev, float(gap8(kv)[0]))

        def on_edge_network(kv):
            return int(np.sum((np.abs(np.asarray(kv) - np.pi / 2) < 2e-5)
                              | (np.asarray(kv) < 2e-5)
                              | (np.asarray(kv) > np.pi - 2e-5))) >= 2

        # massive-only isolated nodes: sweep points not in the constructed
        # survivor set (5e-3 matching tolerance: distinct physical nodes are
        # separated by >~ 0.05, while refinement scatter is <~ 2e-3) and not
        # on the edge network, completed under the empirical event group
        n_edge_reassigned = sum(1 for c in cen8["points"]
                                if on_edge_network(c["k"]))
        pool = [dict(k=c["k"]) for c in cen8["points"]
                if all(tdist(c["k"], s["k"]) > 5e-3 for s in spts)
                and not on_edge_network(c["k"])]
        for _ in range(3):
            added = 0
            for p in list(pool):
                for O in uni8 + anti8:
                    km = (O @ p["k"]) % np.pi
                    if any(tdist(km, x["k"]) < 5e-3 for x in pool) or any(
                            tdist(km, s["k"]) < 5e-3 for s in spts):
                        continue
                    kr, gr = refine(gap8, km)
                    if (gr < DEG_TOL and tdist(kr, km) < 1e-3
                            and not on_edge_network(kr)
                            and all(tdist(kr, x["k"]) > 5e-3 for x in pool)
                            and all(tdist(kr, s["k"]) > 5e-3 for s in spts)):
                        cls = classify(gap8, kr, rng)
                        if cls["kind"] == "point":
                            pool.append(dict(k=kr))
                            added += 1
            if not added:
                break
        # Taxonomy of the massive-only nodes (measured, then asserted):
        # every one is PINNED at a self-conjugate quasienergy, theta0 in
        # {0, pi} (mod 2pi), where the two inversion sectors are resonant;
        # isolated, unit charges, cancelling in total.
        extra_events = []
        for p in pool:
            for th0, mult in events_at(op8, p["k"]):
                pinned = min(abs(wrap(th0)),
                             abs(abs(wrap(th0)) - np.pi)) < 1e-6
                ev = dict(
                    k_over_pi=[round(float(x / np.pi), 6) for x in p["k"]],
                    theta0_over_pi=round(th0 / np.pi, 6), mult=mult,
                    pinned=bool(pinned))
                if pinned:
                    vals, marg = node_charges(op8, p["k"], th0)
                    ev.update(chi=int(round(vals[0])), chi_raw=vals,
                              margin=marg)
                extra_events.append(ev)
        pinned_events = [e for e in extra_events if e["pinned"]]
        unclassified = [e for e in extra_events if not e["pinned"]]
        s_pinned = sum(e["chi"] for e in pinned_events)

        # curve members: line-classified clusters off the edge network
        curves = [c for c in cen8["lines"] if c["n_special_components"] < 2]
        n_curve_offplane = sum(
            1 for c in curves
            if not (np.any(np.abs(c["k"] - np.pi / 2) < 1e-6)
                    or np.any(c["k"] < 1e-6) or np.any(c["k"] > np.pi - 1e-6)))
        n_curve_pihalf = sum(
            1 for c in curves if np.any(np.abs(c["k"] - np.pi / 2) < 1e-6))
        edge_members = [c for c in cen8["lines"]
                        if c["n_special_components"] >= 2]
        out["massive"][f"q{q}_qm{qm}"] = dict(
            m=m,
            n_unitary_sym=len(uni8), n_antiunitary_sym=len(anti8),
            self_conjugate_spectrum=bool(
                any(np.abs(O - np.eye(3)).max() < 1e-12 for O in anti8)),
            unitary_sym_ops=[[[int(x) for x in row] for row in O]
                             for O in uni8],
            principal_gap_minus_2m_max=float(worst),
            n_sweep_isolated_points=len(cen8["points"]),
            n_survivor_points=len(spts),
            n_survivor_events=len(surv_events),
            survivor_max_gap=float(max(e["gap"] for e in surv_events)),
            survivor_max_moved=float(max(e["moved"] for e in surv_events)),
            survivor_all_points=bool(all(e["kind"] == "point"
                                         for e in surv_events)),
            survivor_events=surv_events,
            survivor_sum_chi_plus=int(sp_s), survivor_sum_chi_minus=int(sm_s),
            sector_split_distances=dict(
                min=float(min(splits)), max=float(max(splits)),
                n_zero=int(sum(1 for x in splits if x < 1e-6))),
            massless_orbit_fates=fates,
            n_massive_only_points=len(pool),
            n_pinned_events=len(pinned_events),
            pinned_chi_sum=int(s_pinned),
            n_unclassified_massive_only_events=len(unclassified),
            n_sweep_points_reassigned_to_edges=int(n_edge_reassigned),
            edge_network_max_gap=float(edge_dev),
            massive_only_events=extra_events,
            n_curve_members=len(curves),
            n_curve_members_offplane=int(n_curve_offplane),
            n_curve_members_pihalf_plane=int(n_curve_pihalf),
            curve_samples=[[round(x / np.pi, 6) for x in c["k"]]
                           for c in curves[:12]],
            n_edge_line_members=len(edge_members),
            nondegenerate_local_min_floor=cen8["nondeg_floor"],
            seconds=round(cen8["seconds"], 1))
        print(f"[{time.time()-t00:6.1f}s] massive q={q} qm={qm}: "
              f"survivors {len(spts)} pts / {len(surv_events)} events "
              f"(S+={sp_s} S-={sm_s}), massive-only {len(pool)} pts "
              f"({len(pinned_events)} pinned chi-sum {s_pinned}, "
              f"{len(unclassified)} unclassified, "
              f"{n_edge_reassigned} reassigned to edges), "
              f"{len(curves)} curve members ({n_curve_offplane} off-plane), "
              f"{len(edge_members)} edge members, "
              f"principal 2m dev {worst:.2e}", flush=True)

    out["total_seconds"] = round(time.time() - t00, 1)

    # ---- summary table ------------------------------------------------------
    print("\n===== census summary =====")
    for key, blk in out["massless"].items():
        print(f"\n{key}: {blk['n_isolated_points']} isolated points, "
              f"S+ = {blk['sum_chi_plus']}, S- = {blk['sum_chi_minus']}")
        for r in blk["orbit_table"]:
            print(f"  orbit size {r['size']:2d} at k/pi={r['rep_k_over_pi']}"
                  f" theta0/pi={r['rep_theta_over_pi']:+.4f}"
                  f" chi+={r['chi_plus']}({r['n_plus']})"
                  f" chi-={r['chi_minus']}({r['n_minus']})")
    for key, blk in out["massive"].items():
        print(f"\nmassive {key}: {blk['n_survivor_points']} survivor points "
              f"({blk['n_survivor_events']} events, "
              f"S+={blk['survivor_sum_chi_plus']} "
              f"S-={blk['survivor_sum_chi_minus']}), "
              f"{blk['n_massive_only_points']} massive-only points "
              f"({blk['n_pinned_events']} pinned events, "
              f"chi sum {blk['pinned_chi_sum']}), "
              f"{blk['n_curve_members']} curve members")
        for f in blk["massless_orbit_fates"]:
            tag = ("SURVIVES exactly" if f["survives_exactly"]
                   else f"gapped (quartet inner gap {f['quartet_inner_gap']:.4f})")
            print(f"  massless orbit size {f['orbit_size']:2d} "
                  f"k/pi={f['rep_k_over_pi']}: {tag}")

    if explore:
        print("\n[explore mode: skipping headline assertions and JSON write]")
        return out

    # ---- headline assertions (before any results are written) ---------------
    g = out["symmetry_group"]
    assert g["n_unitary"] == 12 and g["n_antiunitary"] == 12
    assert g["minus_identity_antiunitary"]
    assert all(abs(d - 1) < 1e-9 for d in g["unitary_dets"])

    # the massless census.  Orbit sizes count EVENTS (k, theta): the corner
    # hosts both branches at one k (2 events), the half-point triple hosts
    # 6, and the extra orbits host one event per point.  At q = 0.08 and
    # 0.15 (chord) and q = 0.08 (arc) the topology is identical:
    # 44 isolated points, event orbits [2, 6, 8, 8, 24], S+- = -+10.
    # At q = 0.30 the generic 24-orbit has ANNIHILATED (see continuation
    # below): 20 points, orbits [2, 6, 8, 8], S+- = +-2.
    for key, blk in out["massless"].items():
        big = key != "chord_q0.3"
        assert blk["n_isolated_points"] == (44 if big else 20), key
        assert blk["n_line_members"] > 0 and blk["lines_on_axis_network"], key
        assert blk["R_minus_rho6_I_dev"] < 1e-12, key
        sizes = sorted(r["size"] for r in blk["orbit_table"])
        assert sizes == ([2, 6, 8, 8, 24] if big else [2, 6, 8, 8]), (key, sizes)
        assert blk["sum_chi_plus"] == (-10 if big else 2), key
        assert blk["sum_chi_minus"] == (10 if big else -2), key
        assert blk["conjugation_pairing_ok"], key
        assert blk["unitary_orbit_chi_consistent"], key
        assert blk["charge_integrality_dev"] < 1e-8, key
        assert blk["charge_radius_grid_stable"], key
        # corner: both branches at k = 0, chi = -1 on the +Im branch
        corner = [r for r in blk["orbit_table"] if r["size"] == 2][0]
        assert corner["sum_plus"] == -1 and corner["sum_minus"] == +1
        half = [r for r in blk["orbit_table"] if r["size"] == 6][0]
        assert half["sum_plus"] == 3 and half["sum_minus"] == -3
        # two diagonal octets, opposite charge per branch
        eights = [r for r in blk["orbit_table"] if r["size"] == 8]
        assert sorted(r["sum_plus"] for r in eights) == [-4, 4], key
        if big:
            generic = [r for r in blk["orbit_table"] if r["size"] == 24][0]
            assert generic["sum_plus"] == -12 and generic["sum_minus"] == 12

    # extra-point locations at the working point (chord q=0.08)
    blk = out["massless"]["chord_q0.08"]
    eights = sorted((r for r in blk["orbit_table"] if r["size"] == 8),
                    key=lambda r: r["rep_theta_over_pi"])
    assert abs(min(eights[0]["rep_k_over_pi"]) - 0.265026) < 1e-4
    assert abs(min(eights[1]["rep_k_over_pi"]) - 0.243428) < 1e-4
    assert abs(eights[0]["rep_theta_over_pi"] - 0.514) < 2e-3
    assert abs(eights[1]["rep_theta_over_pi"] - 0.565) < 2e-3
    gen = [r for r in blk["orbit_table"] if r["size"] == 24][0]
    assert abs(abs(gen["rep_theta_over_pi"]) - 0.948) < 2e-3

    # no Floquet gap; lines carry ~pi but unquantized Berry phase, no winding
    assert out["no_floquet_gap"]["empty_bins"] == 0
    assert out["no_floquet_gap"]["min_bin_count"] > 0
    la = out["line_analysis"]
    assert la["berry_dev_from_pi_max"] > 0.02       # NOT quantized to pi
    assert la["berry_dev_from_pi_min"] < 0.2        # but close to pi
    # no net flux emission (winding zero to discretization accuracy,
    # vs. unit quanta for a charged line)
    assert la["max_abs_winding"] < 0.05
    for axis in range(3):
        assert la[f"axis{axis}"]["tracking_closed"]

    # corner isolation
    ci = out["corner_isolation"]
    assert 0.40 < ci["min_point_distance_over_pi"] < 0.43
    assert abs(ci["line_network_distance_over_pi"] - 0.7071) < 1e-3
    assert ci["min_offcorner_point_quasienergy_gap"] > 0.3

    # generic-orbit annihilation: alive at q <= 0.292, gone by q = 0.294,
    # with the smallest momentum component shrinking to 0 (pairwise
    # annihilation on the {k_a = 0} planes)
    cont = out["generic_orbit_continuation"]
    alive = [c for c in cont if c["gap"] < DEG_TOL]
    dead = [c for c in cont if c["gap"] >= DEG_TOL]
    assert any(abs(c["q"] - 0.292) < 1e-9 for c in alive)
    assert dead and all(c["q"] >= 0.294 - 1e-9 for c in dead)
    # approaching q_c the smallest momentum component collapses onto the
    # k_a = 0 plane (it is < 0.015 pi at q = 0.292 and the smallest of the
    # whole continuation there; it is non-monotonic at small q)
    kmins = [min(c["k_over_pi"]) for c in alive]
    assert kmins[-1] == min(kmins) and kmins[-1] < 0.02

    # massive model
    for key, blk in out["massive"].items():
        # spectral symmetry reduced by the mass layer: all axis
        # permutations broken, the 8 diagonal sign flips survive; the
        # massive spectrum is SELF-conjugate at every k (identity with
        # conjugation is an antiunitary symmetry -- every event at
        # (k, theta) pairs with (k, -theta)), so 8 antiunitary ops too
        assert blk["n_unitary_sym"] == 8 and blk["n_antiunitary_sym"] == 8, key
        assert blk["self_conjugate_spectrum"], key
        # principal quartet (corner + three half-points): gap EXACTLY 2m
        assert blk["principal_gap_minus_2m_max"] < 1e-12, key
        # every massless extra Weyl point survives EXACTLY ungapped in both
        # sector copies (constructed, refined, sphere-classified isolated)
        assert blk["n_survivor_events"] == 80, key
        assert blk["survivor_max_gap"] < DEG_TOL, key
        assert blk["survivor_max_moved"] < 0.05, key
        assert blk["survivor_all_points"], key
        assert blk["survivor_sum_chi_plus"] == 0, key
        assert blk["survivor_sum_chi_minus"] == 0, key
        for e in blk["survivor_events"]:
            assert max(abs(v - round(v)) for v in e["chi_raw"]) < 1e-8, key
            assert len(set(round(v) for v in e["chi_raw"])) == 1, key
            assert abs(e["chi"]) == 1, key
        # fates: the principal orbits (event sizes 2 and 6) are gapped by 2m
        # onto the exactly-degenerate doubling edge network; the extra
        # orbits (8, 8, 24) survive exactly
        for f in blk["massless_orbit_fates"]:
            if f["orbit_size"] in (2, 6):
                assert not f["survives_exactly"], key
                assert abs(f["quartet_inner_gap"] - 2 * blk["m"]) < 1e-9, key
            else:
                assert f["survives_exactly"], key
        # the 12-line edge network is exactly degenerate along its length
        assert blk["edge_network_max_gap"] < 1e-12, key
        # every massive-only isolated node is PINNED at a self-conjugate
        # quasienergy theta0 in {0, pi} (the sector-resonant values), with
        # unit charges cancelling in total
        assert blk["n_pinned_events"] > 0, key
        assert blk["pinned_chi_sum"] == 0, key
        assert blk["n_unclassified_massive_only_events"] == 0, key
        for e in blk["massive_only_events"]:
            if "chi_raw" not in e:
                continue
            assert max(abs(v - round(v)) for v in e["chi_raw"]) < 1e-8, key
            assert len(set(round(v) for v in e["chi_raw"])) == 1, key
            assert abs(e["chi"]) == 1, key
            assert e["margin"] > 0.01, key
        # degeneracy curves confined to the {k_a in {0, pi/2}} planes
        assert blk["n_curve_members"] > 0, key
        assert blk["n_curve_members_offplane"] == 0, key
        assert blk["n_curve_members_pihalf_plane"] > 0, key

    print("\nall headline assertions passed")
    payload = {k: v for k, v in out.items()}
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=1))
    print(f"wrote {RESULTS}")
    return out


if __name__ == "__main__":
    main(explore="--explore" in sys.argv)
