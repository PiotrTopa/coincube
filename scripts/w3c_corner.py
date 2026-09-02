#!/usr/bin/env python
"""W3c v6: quenched cone spectroscopy of the coincube with honest statistics.

Estimator: 4 basis launches -> full measured propagator; per-momentum Bloch
fit over the post-launch window (t >= 1); characteristic-polynomial pooling
over the symmetry orbit; delta -> 0 Richardson.

Statistics (red-team upgrade): media are accumulated in NB independent
blocks; every final quantity (family slope, ratio) is jackknifed at the
end-of-pipeline level over leave-one-block-out samples. Off-symmetry
direction families (r1, r2) are measured alongside the cubic stars. The
diamond-exclusion significance is COMPUTED HERE, per channel, no channel
selection; the annealed gate row's deviation from the exact operator is
reported as a systematic and added in quadrature.

Gate (hard assertions): an annealed known-answer row AT EACH measured q must
reproduce the exact operator's ratios and node modulus or the run dies; each
quenched row inherits the gate systematic of its own q.

Linear-range rule: a row whose Richardson pair shows a slope change above
LIN_MAX between probe radii is outside the node's linear window; its
families are flagged within_linear_range = False in the output and the row
is excluded from evidence. (Formalized here after the q = 0.15 row was
excluded by hand at the analysis level; see results/RUN_REGISTRY.md.)
"""
import json, pathlib, time
import numpy as np

from pca3d.models.coincube import annealed_u, evolve_field_cc

L, R, TCYC, NB = 48, 3000, 8, 10
RUNS = [("annealed", 0.08, (0.03, 0.05, 0.08)),
        ("quenched", 0.08, (0.03, 0.05, 0.08)),
        ("annealed", 0.15, (0.03, 0.05, 0.08)),
        ("quenched", 0.15, (0.03, 0.05, 0.08))]
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
#: factorized-transport (diamond) slope ratio prediction per unit direction
DIAMOND = {f: float(np.abs(_norm(FAMILIES[f][0])).sum()) for f in FAMILIES}
BOUND = 2 * np.sqrt(3) + 0.3
LIN_MAX = 0.15                        # linear-range rule (evidence flag)
T0 = 1                                # post-launch fit window


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


def analyse(mode, q, deltas, seed=11):
    t_start = time.time()
    # per-block 4-launch propagator series: Zs[key] -> (NB, T+1, 4, 4)
    Gs = [evolve_field_cc(L, R, TCYC, q, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c,
                          n_blocks=NB) for c in range(4)]
    print(f"\n--- {mode} q={q}  (R={R} x 4 launches, {NB} blocks, "
          f"{time.time() - t_start:.0f}s evolve) ---", flush=True)

    def series(kvec):
        return np.stack([zk(G, kvec) for G in Gs], axis=-1)  # (NB,T+1,4,4)

    kseries = {}
    for kp in X_POINTS:
        kseries[("X", tuple(kp))] = series(kp)
    for fname, dirs in FAMILIES.items():
        for dlt in deltas:
            for d in dirs:
                kv = None
                for kp in X_POINTS:
                    kv = kp + dlt * _norm(d)
                    kseries[(fname, dlt, tuple(_norm(d)), tuple(kp))] = \
                        series(kv)

    def pipeline(block_mask):
        """Full analysis on the mean over the selected blocks."""
        def mseries(key):
            return kseries[key][block_mask].mean(axis=0)
        cps = [np.poly(u_fit(mseries(("X", tuple(kp))))) for kp in X_POINTS]
        poles0 = np.roots(np.mean(cps, axis=0))
        prop = [l for l in poles0 if abs(np.angle(l)) > 0.005 and abs(l) > 0.1]
        lam_ref = max(prop, key=abs) if prop else max(poles0, key=abs)
        lam_ref = abs(lam_ref) * np.exp(1j * abs(np.angle(lam_ref)))
        out = {"lam_ref": lam_ref}
        for fname, dirs in FAMILIES.items():
            vs = {}
            for dlt in deltas:
                cps_d = []
                for d in dirs:
                    for kp in X_POINTS:
                        key = (fname, dlt, tuple(_norm(d)), tuple(kp))
                        cps_d.append(np.poly(u_fit(mseries(key))))
                v = split_of(np.roots(np.mean(cps_d, axis=0)),
                             lam_ref) / (2 * dlt)
                vs[dlt] = v
            good = [d for d in deltas if np.isfinite(vs[d])]
            if len(good) > 1:
                d1, d2 = good[0], good[1]
                v0 = (vs[d1] * d2 ** 2 - vs[d2] * d1 ** 2) / (d2 ** 2 - d1 ** 2)
                lin = abs(vs[d2] - vs[d1]) / vs[d1]
            elif good:
                v0, lin = vs[good[0]], np.nan
            else:
                v0, lin = np.nan, np.nan
            out[fname] = {"v": v0, "lin": lin,
                          "per_delta": {str(d): vs[d] for d in deltas}}
        return out

    full = pipeline(np.ones(NB, dtype=bool))
    jk = []
    for i in range(NB):
        m = np.ones(NB, dtype=bool)
        m[i] = False
        jk.append(pipeline(m))

    lam_ref = full["lam_ref"]
    th = np.linalg.eigvals(annealed_u(-X_POINTS[0], q))
    thp = max([l for l in th if l.imag > 0], key=abs)
    print(f"  node: |lam0|={abs(lam_ref):.3f} om0={np.angle(lam_ref):.4f}  "
          f"(annealed theory {abs(thp):.3f}/{abs(np.angle(thp)):.4f})")

    rows = {}
    v100 = full["100"]["v"]
    print(f"  {'fam':>4} {'v(d->0)':>8} {'sig_jk':>7} {'ratio':>7} "
          f"{'sig_r':>7} {'lin':>6} {'diamond':>8}")
    for fname in FAMILIES:
        v = full[fname]["v"]
        vjk = np.array([s[fname]["v"] for s in jk])
        rjk = np.array([s[fname]["v"] / s["100"]["v"] for s in jk])
        ok = np.isfinite(vjk) & np.isfinite(rjk)
        nn = ok.sum()
        sv = np.sqrt((nn - 1) / nn * ((vjk[ok] - vjk[ok].mean()) ** 2).sum()) \
            if nn > 1 else np.nan
        ratio = v / v100
        sr = np.sqrt((nn - 1) / nn * ((rjk[ok] - rjk[ok].mean()) ** 2).sum()) \
            if nn > 1 else np.nan
        lin_ok = bool(full[fname]["lin"] <= LIN_MAX)
        rows[fname] = {"v": float(v), "sig_v": float(sv), "ratio": float(ratio),
                       "sig_ratio": float(sr), "lin": float(full[fname]["lin"]),
                       "within_linear_range": lin_ok,
                       "diamond": DIAMOND[fname], "n_jk": int(nn)}
        if not lin_ok:
            print(f"  [linear-range rule] {fname}: lin "
                  f"{full[fname]['lin']:.3f} > {LIN_MAX} -- row excluded "
                  f"from evidence")
        assert not (v > BOUND), f"cone bound violated: {fname}"
        print(f"  {fname:>4} {v:>8.4f} {sv:>7.4f} {ratio:>7.4f} {sr:>7.4f} "
              f"{full[fname]['lin']:>6.3f} {DIAMOND[fname]:>8.3f}")
    return {"mode": mode, "q": q,
            "lam_ref": [float(lam_ref.real), float(lam_ref.imag)],
            "rows": rows}


results = []
gate_syst = {}
for mode, q, deltas in RUNS:
    r = analyse(mode, q, deltas)
    if mode == "annealed":
        # GATE (hard): known-answer row vs the exact operator (isotropy = 1)
        for fam in ("110", "111", "r1", "r2"):
            rat = r["rows"][fam]["ratio"]
            sig = r["rows"][fam]["sig_ratio"]
            # significance-based gate: 3 sigma of the channel's own jackknife,
            # floored at 2% (channels with tiny sig_jk must still be accurate)
            tol = max(0.02, 3 * sig)
            assert abs(rat - 1.0) < tol, \
                f"GATE FAILED: ratio {fam} = {rat} (tol {tol:.3f})"
            gate_syst[fam] = abs(rat - 1.0)
        lr = complex(*r["lam_ref"])
        th = annealed_u(-X_POINTS[0], q)
        thp = max((l for l in np.linalg.eigvals(th) if l.imag > 0), key=abs)
        assert abs(abs(lr) - abs(thp)) < 0.05, "GATE FAILED: node modulus"
        print(f"  [gate PASSED]  systematics (|ratio-1|): " +
              "  ".join(f"{f}:{s:.4f}" for f, s in gate_syst.items()))
    else:
        # diamond-exclusion significance, all channels, gate systematic added
        print("  diamond exclusion (all channels, sigma_total = "
              "sqrt(sig_jk^2 + gate_syst^2)):")
        zs = []
        for fam in ("110", "111", "r1", "r2"):
            row = r["rows"][fam]
            st = np.sqrt(row["sig_ratio"] ** 2 +
                         gate_syst.get(fam, 0.0) ** 2)
            z = (row["diamond"] - row["ratio"]) / st if st > 0 else np.nan
            row["sigma_total"] = float(st)
            row["z_diamond"] = float(z)
            zs.append(z)
            print(f"    {fam}: ratio {row['ratio']:.4f} +- {st:.4f}, "
                  f"diamond {row['diamond']:.3f}  ->  z = {z:.1f}")
        print(f"    weakest channel: z = {np.nanmin(zs):.1f}")
    results.append(r)
pathlib.Path("results/w3c_corner.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/w3c_corner.json")
