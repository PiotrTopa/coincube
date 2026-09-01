#!/usr/bin/env python
"""W3c: quenched cone spectroscopy of the LEGAL layered coincube CA.

Same estimator as w3_corner v5 (4 basis launches -> full measured propagator;
char-poly pooling over the symmetry orbit; off-grid k probes; delta -> 0
Richardson; jackknife; annealed known-answer gate). Model: coincube — the
manifestly legal layered architecture of ADR 0011's follow-up (single-site
env-controlled Givens conversions + coin-steered shifts + env streaming).

Annealed reference: annealed_u(k, q) carries the exact isotropic cone at all
8 corners (scratch check: iso 1.0002 at finite h). Quenched question: does it
survive the streaming env with single-bit conversion control?
"""
import json, pathlib, time
import numpy as np
from pca3d.models.coincube import annealed_u, evolve_field_cc

L, R, TCYC = 48, 3000, 8
RUNS = [("annealed", 0.08, (0.03, 0.05, 0.08)),
        ("quenched", 0.08, (0.03, 0.05, 0.08)),
        ("quenched", 0.15, (0.03, 0.05, 0.08))]
X_POINTS = [np.array(v, float) * np.pi for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
FAMILIES = {
    "100": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "110": [(a, b, 0) for a in (1, -1) for b in (1, -1)] +
           [(a, 0, b) for a in (1, -1) for b in (1, -1)] +
           [(0, a, b) for a in (1, -1) for b in (1, -1)],
    "111": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
}
BOUND = 2 * np.sqrt(3) + 0.3


def zk(G, kvec):
    x = np.arange(G.shape[-1])
    px, py, pz = (np.exp(-1j * kvec[a] * x) for a in range(3))
    return np.einsum("tcxyz,x,y,z->tc", G, px, py, pz, optimize=True)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(1, Gmat.shape[0])], axis=1)
    return B @ np.linalg.pinv(A, rcond=1e-8)


def analyse(mode, q, deltas, seed=11):
    t0 = time.time()
    Gs = [evolve_field_cc(L, R, TCYC, q, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c)
          for c in range(4)]
    print(f"\n--- {mode} q={q}  (R={R} x 4 launches, "
          f"{time.time() - t0:.0f}s evolve) ---", flush=True)

    def gmat(kvec):
        return np.stack([np.stack([zk(G, kvec)[t] for G in Gs], axis=1)
                         for t in range(TCYC + 1)])

    cps = [np.poly(u_fit(gmat(kp))) for kp in X_POINTS]
    poles0 = np.roots(np.mean(cps, axis=0))
    prop = [l for l in poles0 if abs(np.angle(l)) > 0.005 and abs(l) > 0.1]
    lam_ref = max(prop, key=abs) if prop else max(poles0, key=abs)
    lam_ref = abs(lam_ref) * np.exp(1j * abs(np.angle(lam_ref)))
    th = np.linalg.eigvals(annealed_u(-X_POINTS[0], q))
    thp = max([l for l in th if l.imag > 0], key=abs)
    print(f"  X poles: {np.array2string(np.sort_complex(poles0), precision=3)}")
    print(f"  reference: |lam0|={abs(lam_ref):.3f}  om0={np.angle(lam_ref):.4f}  "
          f"(annealed theory {abs(thp):.3f}/{abs(np.angle(thp)):.4f})  "
          f"cone range ~{np.angle(lam_ref) / 1.2:.3f}")

    def split_of(poles):
        sel = [l for l in poles
               if 0.5 * abs(lam_ref) < abs(l) < 1.5 * abs(lam_ref)
               and np.angle(l) > 0.0]
        if len(sel) < 2:
            return np.nan
        sel = sorted(sel, key=lambda l: abs(abs(l) - abs(lam_ref)))[:2]
        return abs(np.angle(sel[0]) - np.angle(sel[1]))

    rows = {}
    for fname, dirs in FAMILIES.items():
        vs = {}
        for dlt in deltas:
            cps_all, cps_a, cps_b = [], [], []
            for i, (kp, d) in enumerate([(kp, d) for kp in X_POINTS for d in dirs]):
                u = np.array(d, float)
                u /= np.linalg.norm(u)
                cp = np.poly(u_fit(gmat(kp + dlt * u)))
                cps_all.append(cp)
                (cps_a if i % 2 == 0 else cps_b).append(cp)
            v = split_of(np.roots(np.mean(cps_all, axis=0))) / (2 * dlt)
            va = split_of(np.roots(np.mean(cps_a, axis=0))) / (2 * dlt)
            vb = split_of(np.roots(np.mean(cps_b, axis=0))) / (2 * dlt)
            err = abs(va - vb) / 2 if np.isfinite(va + vb) else np.nan
            vs[dlt] = (v, err)
        good = [d for d in deltas if np.isfinite(vs[d][0])]
        per_delta = {str(d): [float(x) if np.isfinite(x) else None for x in vs[d]]
                     for d in deltas}
        if not good:
            rows[fname] = {"v": None, "sem": None, "lin": None,
                           "delta_used": None, "per_delta": per_delta}
            continue
        if len(good) > 1:
            d1, d2 = good[0], good[1]
            v0 = (vs[d1][0] * d2 ** 2 - vs[d2][0] * d1 ** 2) / (d2 ** 2 - d1 ** 2)
            lin = abs(vs[d2][0] - vs[d1][0]) / vs[d1][0]
        else:
            v0, lin = vs[good[0]][0], np.nan
        rows[fname] = {"v": float(v0), "sem": float(vs[good[0]][1])
                       if np.isfinite(vs[good[0]][1]) else None,
                       "lin": float(lin) if np.isfinite(lin) else None,
                       "delta_used": good[0], "per_delta": per_delta}
        assert not (v0 > BOUND), f"cone bound violated: {fname} v={v0}"
    v100 = rows["100"]["v"] if rows["100"]["v"] else np.nan
    print(f"  {'fam':>4} {'v(d->0)':>8} {'sem':>7} {'ratio':>7} {'lin':>6}"
          f"   per-delta v")
    for fname, r in rows.items():
        pd = "  ".join(
            f"{d}:{r['per_delta'][str(d)][0]:.3f}"
            if r['per_delta'][str(d)][0] is not None else f"{d}:unres"
            for d in deltas)
        vv = r["v"] if r["v"] is not None else np.nan
        sem = r["sem"] if r["sem"] is not None else np.nan
        lin = r["lin"] if r["lin"] is not None else np.nan
        print(f"  {fname:>4} {vv:>8.4f} {sem:>7.4f} {vv / v100:>7.3f} "
              f"{lin:>6.3f}   {pd}")
    return {"mode": mode, "q": q, "lam_ref": [float(lam_ref.real),
            float(lam_ref.imag)], "rows": rows}


results = [analyse(mode, q, deltas) for mode, q, deltas in RUNS]
pathlib.Path("results/w3c_corner.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/w3c_corner.json  (cone: ratios ~1; diamond: 1.41/1.73)")
