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

L, R, TCYC = 48, 3000, 8
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
    return np.einsum("tcxyz,x,y,z->tc", G, px, py, pz, optimize=True)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(1, Gmat.shape[0])], axis=1)
    return B @ np.linalg.pinv(A, rcond=1e-8)


def analyse(mode, seed=11):
    t0 = time.time()
    Gs = [evolve_field_m8(L, R, TCYC, Q, QM, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c)
          for c in range(8)]
    print(f"\n--- {mode} q={Q} qm={QM}  (R={R} x 8 launches, "
          f"{time.time() - t0:.0f}s evolve) ---", flush=True)

    def gmat(kvec):
        return np.stack([np.stack([zk(G, kvec)[t] for G in Gs], axis=1)
                         for t in range(TCYC + 1)])

    def pooled_poles(kvecs):
        cps = [np.poly(u_fit(gmat(kv))) for kv in kvecs]
        return np.roots(np.mean(cps, axis=0))

    # node: two +Im pole pairs -> omega_c, m
    poles0 = pooled_poles(X_POINTS)
    up = sorted([l for l in poles0 if l.imag > 0.02 and abs(l) > 0.1],
                key=lambda l: np.angle(l))
    phs = [np.angle(l) for l in up]
    mods = [abs(l) for l in up]
    om_lo, om_hi = phs[0], phs[-1]
    om_c = 0.5 * (om_lo + om_hi)
    m_meas = 0.5 * (om_hi - om_lo)
    th = np.linalg.eigvals(annealed_u8(X_POINTS[0], Q, QM))
    thp = sorted([np.angle(l) for l in th if l.imag > 0.02])
    m_th = 0.5 * (thp[-1] - thp[0])
    print(f"  node poles (+Im): phases {[f'{p:.4f}' for p in phs]} "
          f"moduli {[f'{x:.3f}' for x in mods]}")
    print(f"  omega_c = {om_c:.4f}, m_meas = {m_meas:.5f}  "
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
            poles = pooled_poles(kvecs)
            sel = [l for l in poles if l.imag > 0.02
                   and 0.45 * mod_ref < abs(l) < 1.6 * mod_ref]
            if not sel:
                oms[dlt] = None
                continue
            oms[dlt] = float(max(np.angle(l) for l in sel) - om_c)
        rows[fname] = oms
    print(f"  upper branch omega_+ - omega_c  (pred = sqrt(m^2 + v^2 d^2), "
          f"v = 1.1806):")
    hdr = "   fam  " + "  ".join(f"d={d}" for d in DELTAS)
    print(hdr)
    for fname, oms in rows.items():
        cells = []
        for d in DELTAS:
            v = oms[d]
            cells.append(f"{v:.4f}" if v is not None else "unres")
        print(f"   {fname:>4}  " + "  ".join(cells))
    print("   pred  " + "  ".join(
        f"{np.sqrt(m_meas ** 2 + (1.1806 * d) ** 2):.4f}" for d in DELTAS))
    return {"mode": mode, "om_c": om_c, "m_meas": m_meas, "m_th": float(m_th),
            "rows": rows}


results = [analyse(mode) for (mode,) in RUNS]
pathlib.Path("results/m8_corner.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/m8_corner.json")
