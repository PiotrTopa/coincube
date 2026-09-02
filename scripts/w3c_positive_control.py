#!/usr/bin/env python
"""Positive controls for the w3c cone estimator: known-ANISOTROPIC truths.

The w3c gate validates the instrument only at ratio = 1 (the annealed
coincube), which is also the measured quenched value; an artifact pinning
ratios to 1 would pass every gate. Two controls close that hole, both run
through the IDENTICAL estimator (functions verified below to be
source-identical to scripts/w3c_corner.py):

A. Synthetic diamond truth. An operator family whose doublet splitting is
   exactly the l1 (diamond) law, s(k) = v * sum_a |sin k_a|, at every X
   representative. The pipeline must return the diamond ratios
   (1.414/1.732/1.574/1.593) -- noiseless AND with block noise through the
   full jackknife. This is an estimator phantom (not an automaton): its
   only role is to prove the estimator reports the diamond when the truth
   IS the diamond.

B. Legal broken-triple walk, end to end. Conversions use (C_x, C_y,
   identity) -- the z axis carries no conversion -- a legal quenched
   automaton whose X-point doublet has strongly anisotropic exact
   splitting. The full media measurement must match the exact-operator
   prediction computed through the SAME pooled pipeline.

Two documented facts frame the controls:
  (i) orbit pooling symmetrizes the cubic stars: for a truth that BREAKS
      cubic symmetry the pooled 110/111 ratios average to ~1 by
      construction, so the burden of symmetry-free anisotropy detection is
      carried by the off-symmetry channels r1/r2 (exact ratios 0.651 and
      1.196 for control B); control A carries the cubic-star burden with a
      cubic-symmetric anisotropic truth;
 (ii) a FULLY factorized (coinless, diagonal-transport) walk has no usable
      doublet at all -- its X-point eigenvalues all sit at angle 0 mod pi
      (this is the two-component obstruction at work), which is why the
      diamond row of the production instrument is a transport-level
      prediction and why control A is synthetic.

Gates (hard assertions): control A ratios within max(0.02, 3 sigma) of the
diamond values; control B measured ratios within max(0.03, 3 sigma_jk) of
the exact-pipeline prediction, with every strongly anisotropic channel
resolved away from 1 by more than 3 sigma.
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

L, R, TCYC, NB = 48, 1200, 8, 10
Q = 0.08
DELTAS = (0.03, 0.05, 0.08)
T0 = 1

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


# --- quenched evolve, table-driven (mirrors evolve_field_cc) -----------------

def evolve_ctl(L, R, TCYC, q, seed, launch, n_blocks):
    perms = [xp.asarray(np.asarray(p)) for p in PERMS_CTL]
    signs = [xp.asarray(np.asarray(s, dtype=float)) for s in SIGNS_CTL]
    r = np.random.default_rng(seed)
    acc = np.zeros((n_blocks, TCYC + 1, 4, L, L, L))
    per_block = max(1, R // n_blocks)
    for rep in range(R):
        env = xp.asarray(r.random((3, L, L, L)) < q)
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
                    m = xp.moveaxis(env[a], sa, -1)
                    if o:
                        m = xp.roll(m, -1, axis=-1)
                    m0 = m[..., 0::2].copy()
                    m[..., 0::2] = m[..., 1::2]
                    m[..., 1::2] = m0
                    if o:
                        m = xp.roll(m, 1, axis=-1)
                    env[a] = xp.moveaxis(m, -1, sa)
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


DIAMOND = {f: float(np.abs(_norm(FAMILIES[f][0])).sum()) for f in FAMILIES}


def u_syn(kvec, v=1.18, om0=0.316, rho=0.62):
    """Synthetic diamond-splitting truth: doublet at rho*e^{i(om0 +- s)},
    s(k) = v * sum_a |sin k_a| (exact l1 law at every X representative),
    plus two damped spectator poles."""
    s = v * np.abs(np.sin(np.asarray(kvec))).sum()
    return np.diag([rho * np.exp(1j * (om0 + s)),
                    rho * np.exp(1j * (om0 - s)), 0.3, 0.2]).astype(complex)


def control_A():
    rng = np.random.default_rng(7)

    def series_noisy(noise):
        def s(key):
            k = key_momentum(key)
            U = u_syn(k)
            G = np.empty((NB, TCYC + 1, 4, 4), dtype=complex)
            for b in range(NB):
                g = np.eye(4, dtype=complex)
                G[b, 0] = g + noise * (rng.standard_normal((4, 4)) +
                                       1j * rng.standard_normal((4, 4)))
                for t in range(1, TCYC + 1):
                    g = U @ g
                    G[b, t] = g + noise * (rng.standard_normal((4, 4)) +
                                           1j * rng.standard_normal((4, 4)))
            return G
        return s

    for noise in (0.0, 3e-3):
        ser = series_noisy(noise)
        cache = {key: ser(key) for key in all_keys()}

        def measured(mask):
            return pipeline_from_series(
                lambda key: cache[key][mask].mean(axis=0))
        full = measured(np.ones(NB, dtype=bool))
        jk = [measured(np.arange(NB) != i) for i in range(NB)]
        print(f"  control A (noise {noise}):")
        for f in FAMILIES:
            r_ = full[f] / full["100"]
            rjk = np.array([s_[f] / s_["100"] for s_ in jk])
            sr = np.sqrt((NB - 1) / NB * ((rjk - rjk.mean()) ** 2).sum())
            tol = max(0.02, 3 * sr)
            print(f"    {f:>4} ratio {r_:.4f}  diamond {DIAMOND[f]:.4f}  "
                  f"sig {sr:.4f}")
            assert abs(r_ - DIAMOND[f]) < tol, \
                f"CONTROL A FAILED: {f} {r_:.4f} vs {DIAMOND[f]:.4f}"
            if DIAMOND[f] > 1.05:
                assert abs(r_ - 1.0) > max(0.1, 3 * sr), \
                    f"CONTROL A FAILED: {f} pinned near 1"
    print("  [control A PASSED] estimator returns the diamond on a "
          "diamond truth")


def main():
    _assert_estimator_identity()
    t0 = time.time()
    control_A()

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
          f"q={Q}):")
    print("  " + "  ".join(f"{f}:{ex_ratio[f]:.4f}" for f in FAMILIES))

    # documented fact (ii): the coinless factorized walk has no usable doublet
    def u_diag(kvec):
        u = np.eye(4, dtype=complex)
        for a in range(3):
            t = np.diag(np.exp(1j * kvec[a] * COIN_D[a]))
            u = t @ t @ u
        return u
    lam = np.linalg.eigvals(u_diag(X_POINTS[0]))
    assert np.allclose(np.abs(lam), 1) and \
        np.allclose(np.sin(np.angle(lam)), 0), "diag walk structure changed"

    # quenched measurement, 4 launches, NB blocks
    Gs = [evolve_ctl(L, R, TCYC, Q, 11 + 100 * c, c, NB) for c in range(4)]
    print(f"quenched evolve done ({time.time() - t0:.0f}s, R={R}, L={L}, "
          f"NB={NB})", flush=True)
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

    # GATES
    for f in FAMILIES:
        r_ = rows[f]
        tol = max(0.03, 3 * r_["sig_ratio"])
        assert abs(r_["ratio"] - r_["exact_ratio"]) < tol, \
            f"CONTROL FAILED: {f} measured {r_['ratio']:.4f} vs exact " \
            f"{r_['exact_ratio']:.4f} (tol {tol:.3f})"
        if abs(r_["exact_ratio"] - 1.0) > 0.05:
            assert abs(r_["ratio"] - 1.0) > 3 * r_["sig_ratio"], \
                f"CONTROL FAILED: {f} anisotropy not resolved"
    print("[control PASSED] estimator recovers anisotropic truth; "
          "ratios are not pinned to 1")

    out = {"model": "coins (C_x, C_y, identity), q=0.08, chord weights",
           "L": L, "R": R, "TCYC": TCYC, "NB": NB, "T0": T0,
           "rows": rows,
           "note_factorized_walk": "coinless diagonal-transport walk has no "
           "usable doublet (all X-point eigenvalues at angle 0 mod pi); the "
           "diamond row of w3c_corner is a transport-level prediction"}
    pathlib.Path("results/w3c_positive_control.json").write_text(
        json.dumps(out, indent=1))
    print("written: results/w3c_positive_control.json")


if __name__ == "__main__":
    main()
