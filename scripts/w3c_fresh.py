#!/usr/bin/env python
"""W3C-FRESH (campaign instrument): cone spectroscopy under fresh-tape (F1) streaming.

Claim under test (J48): with F1 streaming (one extra phase-continuing swap
pair per axis substep; bits move +-6 sites/cycle), no path re-reads a bit,
so the quenched ensemble propagator equals the annealed operator EXACTLY.
Consequence at scale: the quenched node parameters must equal the annealed
closed forms within statistics -- the production-schedule shifts
(|lambda_0| +2.8%, omega_0 +8.6% at q=0.08) must vanish within statistics.

Instrument: the w3c v6 pipeline (estimator functions source-identical to
scripts/w3c_corner.py, asserted), NB-block jackknife, annealed known-answer
gate row. Wrap constraint: bits travel 6*TCYC sites; requires L > 6*TCYC,
hence TCYC = ceil(L/8) = 6 at L=48 (sharp horizon, freshtape_proof).

Gates (hard): annealed row reproduces the exact operator (as in w3c);
quenched-F1 node modulus and frequency match the ANNEALED closed forms
within max(0.5%, 3 sigma_jk) -- an order of magnitude inside the
production-schedule shifts -- and all slope ratios equal 1 within
max(2%, 3 sigma).
"""
import ast
import json
import os
import pathlib
import time

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

from pca3d.models.coincube import annealed_u, evolve_field_cc

L, R, TCYC, NB = 48, 3000, 6, 10
# replication overrides (independent seed stream, separate output file):
R = int(os.environ.get("W3CF_R", R))
SEED0 = int(os.environ.get("W3CF_SEED", 11))
OUT = os.environ.get("W3CF_OUT", "results/w3c_fresh.json")
Q = 0.08
DELTAS = (0.03, 0.05, 0.08)
T0 = 1
X_POINTS = [np.array(v, float) * np.pi for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]


def _norm(v):
    u = np.array(v, float)
    return u / np.linalg.norm(u)


R1 = (0.276, 0.850, 0.448)
R2 = (0.732, 0.214, 0.647)

#: factorized-transport (diamond) slope-ratio prediction per unit direction,
#: source-identical to w3c_corner.py
def _diamond(v):
    return float(np.abs(_norm(v)).sum())
FAMILIES = {
    "100": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "110": [(a, b, 0) for a in (1, -1) for b in (1, -1)] +
           [(a, 0, b) for a in (1, -1) for b in (1, -1)] +
           [(0, a, b) for a in (1, -1) for b in (1, -1)],
    "111": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
    "r1": [R1, tuple(-x for x in R1)],
    "r2": [R2, tuple(-x for x in R2)],
}


# --- estimator, source-identical to w3c_corner.py (asserted) -----------------

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
    here = pathlib.Path(__file__)
    prod = here.parent / "w3c_corner.py"

    def defs(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n) for n in tree.body
                if isinstance(n, ast.FunctionDef)}
    dh, dp = defs(here), defs(prod)
    for f in ("zk", "u_fit", "split_of"):
        assert dh[f] == dp[f], f"estimator drift vs w3c_corner: {f}"


# --- pipeline (w3c staging) --------------------------------------------------

def analyse(mode, seed=None):
    if seed is None:
        seed = SEED0
    t0 = time.time()
    Gs = [evolve_field_cc(L, R, TCYC, Q, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c,
                          n_blocks=NB, streaming="fresh") for c in range(4)]
    print(f"\n--- {mode} q={Q}  (R={R} x 4 launches, {NB} blocks, TCYC={TCYC}, "
          f"{time.time() - t0:.0f}s evolve) ---", flush=True)

    kseries = {}
    for kp in X_POINTS:
        kseries[("X", tuple(kp))] = np.stack(
            [zk(G, kp) for G in Gs], axis=-1)
    for fname, dirs in FAMILIES.items():
        for dlt in DELTAS:
            for d in dirs:
                for kp in X_POINTS:
                    kv = kp + dlt * _norm(d)
                    kseries[(fname, dlt, tuple(_norm(d)), tuple(kp))] = \
                        np.stack([zk(G, kv) for G in Gs], axis=-1)

    def pipeline(mask):
        def ms(key):
            return kseries[key][mask].mean(axis=0)
        cps = [np.poly(u_fit(ms(("X", tuple(kp))))) for kp in X_POINTS]
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
                        cps_d.append(np.poly(u_fit(ms(key))))
                vs[dlt] = split_of(np.roots(np.mean(cps_d, axis=0)),
                                   lam_ref) / (2 * dlt)
            good = [d for d in DELTAS if np.isfinite(vs[d])]
            if len(good) > 1:
                d1, d2 = good[0], good[1]
                out[fname] = (vs[d1] * d2 ** 2 - vs[d2] * d1 ** 2) / \
                    (d2 ** 2 - d1 ** 2)
            else:
                out[fname] = vs[good[0]] if good else np.nan
            out[fname + "_per_delta"] = {str(d): float(vs[d]) for d in DELTAS}
        return out

    full = pipeline(np.ones(NB, dtype=bool))
    jk = [pipeline(np.arange(NB) != i) for i in range(NB)]
    return full, jk


def jackknife(vals):
    v = np.array(vals)
    n = len(v)
    return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))


def main():
    _assert_estimator_identity()
    th = np.linalg.eigvals(annealed_u(-X_POINTS[0], Q))
    thp = max((l for l in th if l.imag > 0), key=abs)
    mod_th, om_th = abs(thp), abs(np.angle(thp))
    print(f"annealed closed forms: |lam0|={mod_th:.4f}  om0={om_th:.4f}  "
          f"v={1.1806}")

    results = {}
    for mode in ("annealed", "quenchedF1"):
        full, jk = analyse(mode)
        lam = full["lam_ref"]
        smod = jackknife([abs(s["lam_ref"]) for s in jk])
        som = jackknife([abs(np.angle(s["lam_ref"])) for s in jk])
        rows = {}
        print(f"  node: |lam0|={abs(lam):.4f}+-{smod:.4f} "
              f"om0={np.angle(lam):.4f}+-{som:.4f}  "
              f"(shift vs annealed: {100*(abs(lam)/mod_th-1):+.2f}% / "
              f"{100*(np.angle(lam)/om_th-1):+.2f}%)")
        for f in FAMILIES:
            ratio = full[f] / full["100"]
            sr = jackknife([s[f] / s["100"] for s in jk])
            dia = _diamond(FAMILIES[f][0])
            zd = (dia - ratio) / sr if sr > 0 else float("nan")
            rows[f] = {"v": float(full[f]), "ratio": float(ratio),
                       "sig_ratio": sr, "diamond": dia,
                       "z_diamond": float(zd),
                       "per_delta": full[f + "_per_delta"]}
            print(f"    {f:>4}: v={full[f]:.4f}  ratio={ratio:.4f}+-{sr:.4f}"
                  f"  z_diamond={zd:.1f}")
        results[mode] = {
            "lam_mod": float(abs(lam)), "sig_mod": smod,
            "om0": float(abs(np.angle(lam))), "sig_om": som, "rows": rows}
        # GATES
        for f in ("110", "111", "r1", "r2"):
            rr = rows[f]
            tol = max(0.02, 3 * rr["sig_ratio"])
            assert abs(rr["ratio"] - 1.0) < tol, \
                f"GATE FAILED [{mode}] ratio {f}: {rr['ratio']:.4f}"
        mtol = max(0.005 * mod_th, 3 * smod)
        otol = max(0.005 * om_th, 3 * som)
        assert abs(abs(lam) - mod_th) < mtol, \
            f"GATE FAILED [{mode}] |lam0| {abs(lam):.4f} vs {mod_th:.4f}"
        assert abs(abs(np.angle(lam)) - om_th) < otol, \
            f"GATE FAILED [{mode}] om0 {abs(np.angle(lam)):.4f} vs {om_th:.4f}"
        print(f"  [gate PASSED: {mode} node at the annealed closed forms]")

    print("\n[T2 GATE PASSED] fresh-tape quenched node parameters equal the "
          "annealed closed forms within statistics -- the production-schedule "
          "shifts are eliminated.")
    pathlib.Path(OUT).write_text(json.dumps(results, indent=1))
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
