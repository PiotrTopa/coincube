#!/usr/bin/env python
"""M8: quenched spectroscopy of the massive coincube.

Instrument: 8 basis launches -> full measured 8x8 propagator per k; char-poly
pooling over the 3 X-point orbit and direction families; annealed gate row.

Verdict variables:
  - node gap: the two +Im pole-pair phases at the X points -> m_meas and
    omega_c; compare m_meas to arctan(q_m_eff/(1-q_m_eff)) (quenched
    renormalisation of q_m reported, not assumed away);
  - massive dispersion: upper-branch omega_+(delta) - omega_c per direction
    family vs sqrt(m^2 + v^2 delta^2); family agreement = isotropy.
"""
import json, pathlib, time
import numpy as np

from pca3d.models.coincube import annealed_u8, evolve_field_m8

L, R, TCYC, NB = 48, 3000, 8, 8
Q, QM = 0.08, 0.05
RUNS = [("annealed",), ("quenched",)]
DELTAS = (0.04, 0.07, 0.10)
X_POINTS = [np.array(v, float) * np.pi for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
FAMILIES = {
    "100": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "110": [(a, b, 0) for a in (1, -1) for b in (1, -1)] +
           [(a, 0, b) for a in (1, -1) for b in (1, -1)] +
           [(0, a, b) for a in (1, -1) for b in (1, -1)],
    "111": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
}


def zk(G, kvec):
    x = np.arange(G.shape[-1])
    px, py, pz = (np.exp(-1j * kvec[a] * x) for a in range(3))
    return np.einsum("btcxyz,x,y,z->btc", G, px, py, pz, optimize=True)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(1, Gmat.shape[0])], axis=1)
    return B @ np.linalg.pinv(A, rcond=1e-8)


def analyse(mode, seed=11):
    t0 = time.time()
    Gs = [evolve_field_m8(L, R, TCYC, Q, QM, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c,
                          n_blocks=NB) for c in range(8)]
    print(f"\n--- {mode} q={Q} qm={QM}  (R={R} x 8 launches, "
          f"{time.time() - t0:.0f}s evolve) ---", flush=True)

    kcache = {}

    def series(kvec):
        key = tuple(np.round(kvec, 9))
        if key not in kcache:
            zs = [zk(G, kvec) for G in Gs]          # each (NB, T+1, 8)
            kcache[key] = np.stack(zs, axis=-1)      # (NB, T+1, 8, 8)
        return kcache[key]

    def gmat(kvec, mask):
        return series(kvec)[mask].mean(axis=0)

    def pooled_poles(kvecs, mask=None):
        if mask is None:
            mask = np.ones(NB, dtype=bool)
        cps = [np.poly(u_fit(gmat(kv, mask))) for kv in kvecs]
        return np.roots(np.mean(cps, axis=0))

    def node_stats(mask=None):
        p0 = pooled_poles(X_POINTS, mask)
        upl = sorted([l for l in p0 if l.imag > 0.02 and abs(l) > 0.1],
                     key=lambda l: np.angle(l))
        ph = [np.angle(l) for l in upl]
        return 0.5 * (ph[0] + ph[-1]), 0.5 * (ph[-1] - ph[0]), \
            [abs(l) for l in upl]

    # node: two +Im pole pairs -> omega_c, m (with block jackknife)
    om_c, m_meas, mods = node_stats()
    jkn = []
    for i in range(NB):
        msk = np.ones(NB, dtype=bool)
        msk[i] = False
        jkn.append(node_stats(msk)[:2])
    omjk = np.array([x[0] for x in jkn])
    mjk = np.array([x[1] for x in jkn])
    sig_om = np.sqrt((NB - 1) / NB * ((omjk - omjk.mean()) ** 2).sum())
    sig_m = np.sqrt((NB - 1) / NB * ((mjk - mjk.mean()) ** 2).sum())
    poles0 = pooled_poles(X_POINTS)
    up = sorted([l for l in poles0 if l.imag > 0.02 and abs(l) > 0.1],
                key=lambda l: np.angle(l))
    phs = [np.angle(l) for l in up]
    th = np.linalg.eigvals(annealed_u8(X_POINTS[0], Q, QM))
    thp = sorted([np.angle(l) for l in th if l.imag > 0.02])
    m_th = 0.5 * (thp[-1] - thp[0])
    print(f"  node poles (+Im): phases {[f'{p:.4f}' for p in phs]} "
          f"moduli {[f'{x:.3f}' for x in mods]}")
    print(f"  omega_c = {om_c:.4f} +- {sig_om:.4f}, "
          f"m_meas = {m_meas:.5f} +- {sig_m:.5f}  "
          f"(annealed theory m = {m_th:.5f}; arctan law "
          f"{np.arctan(QM / (1 - QM)):.5f})")

    # massive dispersion per family
    mod_ref = np.mean(mods)
    rows = {}
    for fname, dirs in FAMILIES.items():
        oms = {}
        for dlt in DELTAS:
            kvecs = [kp + dlt * (np.array(d, float) / np.linalg.norm(d))
                     for kp in X_POINTS for d in dirs]

            def om_of(mask=None):
                poles = pooled_poles(kvecs, mask)
                sel = [l for l in poles if l.imag > 0.02
                       and 0.45 * mod_ref < abs(l) < 1.6 * mod_ref]
                return (max(np.angle(l) for l in sel) - om_c) if sel \
                    else np.nan
            v = om_of()
            if not np.isfinite(v):
                oms[dlt] = None
                continue
            ojk = np.array([om_of(np.arange(NB) != i) for i in range(NB)])
            ok = np.isfinite(ojk)
            so = np.sqrt((ok.sum() - 1) / ok.sum() *
                         ((ojk[ok] - ojk[ok].mean()) ** 2).sum())
            oms[dlt] = (float(v), float(so))
        rows[fname] = oms
    print(f"  upper branch omega_+ - omega_c  (pred = sqrt(m^2 + v^2 d^2), "
          f"v = 1.1806):")
    hdr = "   fam  " + "  ".join(f"d={d}" for d in DELTAS)
    print(hdr)
    for fname, oms in rows.items():
        cells = []
        for d in DELTAS:
            v = oms[d]
            cells.append(f"{v[0]:.4f}({v[1]:.4f})" if v is not None
                         else "unres")
        print(f"   {fname:>4}  " + "  ".join(cells))
    print("   pred  " + "  ".join(
        f"{np.sqrt(m_meas ** 2 + (1.1806 * d) ** 2):.4f}" for d in DELTAS))
    return {"mode": mode, "om_c": om_c, "m_meas": m_meas, "m_th": float(m_th),
            "rows": rows, "sig_m": float(sig_m),
            "sig_om": float(sig_om)}


results = []
for (mode,) in RUNS:
    r = analyse(mode)
    if mode == "annealed":
        # GATE (hard): gap and centre must match the exact operator
        assert abs(r["m_meas"] - r["m_th"]) < 0.30 * r["m_th"], \
            "GATE FAILED: mass gap"
        th = np.linalg.eigvals(annealed_u8(X_POINTS[0], Q, QM))
        phs = sorted(np.angle(l) for l in th if l.imag > 0.02)
        assert abs(r["om_c"] - 0.5 * (phs[0] + phs[-1])) < 0.02, \
            "GATE FAILED: multiplet centre"
        print("  [gate PASSED]")
    results.append(r)
pathlib.Path("results/m8_corner.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/m8_corner.json")
