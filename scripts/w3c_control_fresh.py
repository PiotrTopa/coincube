#!/usr/bin/env python
"""Broken-triple positive control under the FRESH-TAPE (F1) schedule.

Adaptation of control B of scripts/w3c_positive_control.py to the fresh-tape
rebuild: the legal broken-triple quenched automaton (conversions (C_x, C_y,
identity); strongly anisotropic exact X-point splitting) is evolved with the
F1 streaming schedule -- THREE phase-continuing pair swaps per axis substep
(per-field phase counters; the n-th swap ever applied to a field has origin
n mod 2) -- and the full media measurement must match the exact-operator
prediction computed through the SAME pooled pipeline.

Under F1 the fresh-tape theorem applies to this model as well (same layer
types, same slot counting; the z field is never physically read since the z
conversion is the identity), so for TCYC <= ceil(L/8) the quenched ensemble
propagator IS the annealed operator exactly -- the control checks that the
estimator recovers the anisotropic truth from fresh-schedule media, i.e.
that ratios are not pinned to 1 by an instrument artifact under the new
schedule. TCYC = 6 = ceil(48/8): the sharp wrap-free torus horizon.

Estimator functions (zk, u_fit, split_of) are asserted source-identical to
scripts/w3c_corner.py, exactly as in the original control. The synthetic
control A of the original file is schedule-independent (no automaton is
evolved) and is not repeated here.

Gates (hard assertions, as in the original): measured ratios within
max(0.03, 3 sigma_jk) of the exact pooled-pipeline prediction; every
strongly anisotropic channel resolved away from 1 by more than 3 sigma.

Statistics note: the fresh-tape horizon forces TCYC = 6 (the original
control used TCYC = 8), which shortens the Bloch fit window and roughly
doubles the jackknife noise of the two-direction off-symmetry channels
(a first run at the original R = 1200 measured r1 = 0.727 +- 0.140:
matched the exact prediction at 0.5 sigma but resolved the anisotropy at
only 2.0 sigma -- gate 2 failed on statistics, not physics). R is raised
to 3600 to restore > 3 sigma resolution in the weakest channel.

Run:  PYTHONPATH=src .venv/bin/python scripts/w3c_control_fresh.py
Output: results/w3c_control_fresh.json (+ .log copied in by the caller).
"""
import ast
import json
import pathlib
import time

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

from pca3d.models.coincube import COIN_C, COIN_D, perm_sign

L, R, TCYC, NB = 48, 3600, 6, 10
Q = 0.08
DELTAS = (0.03, 0.05, 0.08)
T0 = 1
assert TCYC <= -(-L // 8), "fresh-tape torus horizon violated (T <= ceil(L/8))"

# broken-triple control coins: quaternionic pair + identity on z
COINS_CTL = [COIN_C[0], COIN_C[1], np.eye(4)]
PERMS_CTL, SIGNS_CTL = zip(*(perm_sign(c) for c in COINS_CTL))

X_POINTS = [np.array(v, float) * np.pi for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]


def _norm(v):
    u = np.array(v, float)
    return u / np.linalg.norm(u)


R1 = (0.276, 0.850, 0.448)
R2 = (0.732, 0.214, 0.647)
FAMILIES = {
    "100": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "110": [(a, b, 0) for a in (1, -1) for b in (1, -1)] +
           [(a, 0, b) for a in (1, -1) for b in (1, -1)] +
           [(0, a, b) for a in (1, -1) for b in (1, -1)],
    "111": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
    "r1": [R1, tuple(-x for x in R1)],
    "r2": [R2, tuple(-x for x in R2)],
}


# --- estimator, source-identical to w3c_corner.py (asserted below) -----------

def zk(G, kvec):
    x = np.arange(G.shape[-1])
    px, py, pz = (np.exp(-1j * kvec[a] * x) for a in range(3))
    return np.einsum("btcxyz,x,y,z->btc", G, px, py, pz, optimize=True)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(T0, Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(T0 + 1, Gmat.shape[0])], axis=1)
    return B @ np.linalg.pinv(A, rcond=1e-8)


def split_of(poles, lam_ref):
    sel = [l for l in poles
           if 0.5 * abs(lam_ref) < abs(l) < 1.5 * abs(lam_ref)
           and np.angle(l) > 0.0]
    if len(sel) < 2:
        return np.nan
    sel = sorted(sel, key=lambda l: abs(abs(l) - abs(lam_ref)))[:2]
    return abs(np.angle(sel[0]) - np.angle(sel[1]))


def _assert_estimator_identity():
    """The three estimator functions above must be source-identical to the
    production instrument in scripts/w3c_corner.py."""
    here = pathlib.Path(__file__)
    prod = here.parent / "w3c_corner.py"

    def defs(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n) for n in tree.body
                if isinstance(n, ast.FunctionDef)}
    dh, dp = defs(here), defs(prod)
    for f in ("zk", "u_fit", "split_of"):
        assert dh[f] == dp[f], f"estimator drift vs w3c_corner: {f}"


# --- exact annealed operator of the control model ----------------------------

def u_ctl(kvec, q):
    u = np.eye(4, dtype=complex)
    for a in range(3):
        t = np.diag(np.exp(1j * kvec[a] * COIN_D[a])) @ (
            (1 - q) * np.eye(4) + q * COINS_CTL[a])
        u = t @ t @ u
    return u


# --- quenched evolve under F1 (local copy; fresh three-swap batches) ---------

def _swap(field, sa, o):
    """One pair-swap streaming application along axis sa, origin o."""
    m = xp.moveaxis(field, sa, -1)
    if o:
        m = xp.roll(m, -1, axis=-1)
    m0 = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = m0
    if o:
        m = xp.roll(m, 1, axis=-1)
    return xp.moveaxis(m, -1, sa)


def evolve_ctl_fresh(L, R, TCYC, q, seed, launch, n_blocks):
    """Table-driven broken-triple evolve, F1 streaming: three PHASE-
    CONTINUING pair swaps per axis substep, per-field phase counters."""
    perms = [xp.asarray(np.asarray(p)) for p in PERMS_CTL]
    signs = [xp.asarray(np.asarray(s, dtype=float)) for s in SIGNS_CTL]
    r = np.random.default_rng(seed)
    acc = np.zeros((n_blocks, TCYC + 1, 4, L, L, L))
    per_block = max(1, R // n_blocks)
    for rep in range(R):
        env = xp.asarray(r.random((3, L, L, L)) < q)
        phase = [0, 0, 0]
        g = xp.zeros((4, L, L, L))
        g[launch, L // 2, L // 2, L // 2] = 1.0
        out = [g.copy()]
        for _ in range(TCYC):
            for a in (0, 1, 2):
                for o in (0, 1):
                    mask = env[a]
                    new = xp.empty_like(g)
                    for c in range(4):
                        cp = int(perms[a][c])
                        new[cp] = xp.where(mask, float(signs[a][c]) * g[c],
                                           g[cp])
                    g = new
                    for c in range(4):
                        g[c] = xp.roll(g[c], int(COIN_D[a][c]), axis=a)
                    sa = (a + 1) % 3
                    e = env[a]
                    for _s in range(3):
                        e = _swap(e, sa, phase[a] % 2)
                        phase[a] += 1
                    env[a] = e
            out.append(g.copy())
        stack = xp.stack(out)
        b = min(rep // per_block, n_blocks - 1)
        acc[b] += stack.get() if GPU else stack
    acc /= (R / n_blocks)
    return acc


# --- the pooled pipeline (identical staging to w3c_corner.analyse) -----------

def pipeline_from_series(series_of):
    """series_of(key) -> (T+1, 4, 4) mean series for that momentum key."""
    cps = [np.poly(u_fit(series_of(("X", tuple(kp))))) for kp in X_POINTS]
    poles0 = np.roots(np.mean(cps, axis=0))
    prop = [l for l in poles0 if abs(np.angle(l)) > 0.005 and abs(l) > 0.1]
    lam_ref = max(prop, key=abs) if prop else max(poles0, key=abs)
    lam_ref = abs(lam_ref) * np.exp(1j * abs(np.angle(lam_ref)))
    out = {"lam_ref": lam_ref}
    for fname, dirs in FAMILIES.items():
        vs = {}
        for dlt in DELTAS:
            cps_d = []
            for d in dirs:
                for kp in X_POINTS:
                    key = (fname, dlt, tuple(_norm(d)), tuple(kp))
                    cps_d.append(np.poly(u_fit(series_of(key))))
            vs[dlt] = split_of(np.roots(np.mean(cps_d, axis=0)),
                               lam_ref) / (2 * dlt)
        good = [d for d in DELTAS if np.isfinite(vs[d])]
        if len(good) > 1:
            d1, d2 = good[0], good[1]
            v0 = (vs[d1] * d2 ** 2 - vs[d2] * d1 ** 2) / (d2 ** 2 - d1 ** 2)
        elif good:
            v0 = vs[good[0]]
        else:
            v0 = np.nan
        out[fname] = v0
    return out


def all_keys():
    keys = [("X", tuple(kp)) for kp in X_POINTS]
    for fname, dirs in FAMILIES.items():
        for dlt in DELTAS:
            for d in dirs:
                for kp in X_POINTS:
                    keys.append((fname, dlt, tuple(_norm(d)), tuple(kp)))
    return keys


def key_momentum(key):
    if key[0] == "X":
        return np.array(key[1])
    _, dlt, u, kp = key
    return np.array(kp) + dlt * np.array(u)


def main():
    _assert_estimator_identity()
    t0 = time.time()

    # exact prediction: noiseless exact-operator series through the pipeline
    def exact_series(key):
        k = key_momentum(key)
        U = u_ctl(k, Q)
        G = np.empty((TCYC + 1, 4, 4), dtype=complex)
        G[0] = np.eye(4)
        for t in range(1, TCYC + 1):
            G[t] = U @ G[t - 1]
        return G
    ex = pipeline_from_series(exact_series)
    ex_ratio = {f: ex[f] / ex["100"] for f in FAMILIES}
    print("exact pooled-pipeline prediction (broken-triple control, "
          f"q={Q}, TCYC={TCYC}):")
    print("  " + "  ".join(f"{f}:{ex_ratio[f]:.4f}" for f in FAMILIES))

    # quenched F1 measurement, 4 launches, NB blocks
    Gs = [evolve_ctl_fresh(L, R, TCYC, Q, 11 + 100 * c, c, NB)
          for c in range(4)]
    print(f"quenched F1 evolve done ({time.time() - t0:.0f}s, R={R}, L={L}, "
          f"NB={NB}, TCYC={TCYC})", flush=True)
    kser = {key: np.stack([zk(G, key_momentum(key)) for G in Gs], axis=-1)
            for key in all_keys()}

    def measured(mask):
        return pipeline_from_series(lambda key: kser[key][mask].mean(axis=0))

    full = measured(np.ones(NB, dtype=bool))
    jk = []
    for i in range(NB):
        m = np.ones(NB, dtype=bool)
        m[i] = False
        jk.append(measured(m))

    rows = {}
    print(f"  {'fam':>4} {'exact_r':>8} {'meas_r':>8} {'sig_jk':>7} "
          f"{'pull':>6}")
    for f in FAMILIES:
        mr = full[f] / full["100"]
        rjk = np.array([s[f] / s["100"] for s in jk])
        ok = np.isfinite(rjk)
        nn = ok.sum()
        sr = np.sqrt((nn - 1) / nn * ((rjk[ok] - rjk[ok].mean()) ** 2).sum())
        pull = (mr - ex_ratio[f]) / sr if sr > 0 else np.nan
        rows[f] = {"exact_ratio": float(ex_ratio[f]), "ratio": float(mr),
                   "sig_ratio": float(sr), "pull": float(pull),
                   "n_jk": int(nn)}
        print(f"  {f:>4} {ex_ratio[f]:>8.4f} {mr:>8.4f} {sr:>7.4f} "
              f"{pull:>6.2f}")

    # GATES (as in the original control B)
    for f in FAMILIES:
        r_ = rows[f]
        tol = max(0.03, 3 * r_["sig_ratio"])
        assert abs(r_["ratio"] - r_["exact_ratio"]) < tol, \
            f"CONTROL FAILED: {f} measured {r_['ratio']:.4f} vs exact " \
            f"{r_['exact_ratio']:.4f} (tol {tol:.3f})"
        if abs(r_["exact_ratio"] - 1.0) > 0.05:
            assert abs(r_["ratio"] - 1.0) > 3 * r_["sig_ratio"], \
                f"CONTROL FAILED: {f} anisotropy not resolved"
    print("[control PASSED] estimator recovers anisotropic truth from "
          "fresh-schedule (F1) media; ratios are not pinned to 1")

    out = {"model": "coins (C_x, C_y, identity), q=0.08, chord weights",
           "streaming": "fresh (F1: 3 phase-continuing swaps per substep)",
           "L": L, "R": R, "TCYC": TCYC, "NB": NB, "T0": T0,
           "wall_s": float(time.time() - t0),
           "rows": rows,
           "note": "TCYC = 6 = ceil(L/8): sharp wrap-free fresh-tape torus "
                   "horizon; exact prediction = annealed operator through "
                   "the identical pooled pipeline (fresh-tape theorem makes "
                   "quenched == annealed exact for this model)"}
    pathlib.Path("results/w3c_control_fresh.json").write_text(
        json.dumps(out, indent=1))
    print("written: results/w3c_control_fresh.json")


if __name__ == "__main__":
    main()
