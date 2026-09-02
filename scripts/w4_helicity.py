#!/usr/bin/env python
"""W4b: measured helicity residues of the coincube Weyl node.

Per (X-point, direction u, delta): fit the 4x4 one-cycle U from the 4-launch
propagator (v5 instrument), take the + branch spectral projector |r><l|/<l|r>,
express it in the EXACT node frame (results/w4_frame.npz, from
w4_helicity_exact.py), and read off the helicity vector n_meas (complex Pauli
coefficients, bilinear normalisation). Across 3 X-points x 26 directions this
gives the measured map n(u); fit n = M u and check:

  - frame: M^T M proportional to I (bilinear), chirality = sign(det M);
  - pointwise angular agreement with the exact n(u) = R u.

Gate: the annealed row must reproduce the exact frame (chi = -1, small
residuals) before the quenched row is believed.
"""
import json, pathlib, time
import numpy as np

from pca3d.models.coincube import evolve_field_cc

L, R, TCYC, NB = 48, 3000, 8, 6
DELTA = 0.08
RUNS = [("annealed", 0.08), ("quenched", 0.08)]
X_POINTS = [np.array(v, float) * np.pi for v in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
DIRS = [np.array(d, float) for d in
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        + [(a, b, 0) for a in (1, -1) for b in (1, -1)]
        + [(a, 0, b) for a in (1, -1) for b in (1, -1)]
        + [(0, a, b) for a in (1, -1) for b in (1, -1)]
        + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]

FR = np.load("results/w4_frame.npz")
VR, VL, S, Sinv = FR["VR"], FR["VL"], FR["S"], FR["Sinv"]
R_EX, V_EX, CHI_EX, LAM0 = FR["R"], float(FR["v"]), int(FR["chi"]), FR["lam0"]


def zk(G, kvec):
    # e^{+ikx} so the fitted matrix estimates U(k) itself: eigenVECTOR work
    # needs the true node, not its conjugate. (The e^{-ikx} convention probes
    # conj U = the opposite-helicity doublet — caught by the annealed gate as
    # a global n -> -n flip with det M sign reversed.)
    x = np.arange(G.shape[-1])
    px, py, pz = (np.exp(1j * kvec[a] * x) for a in range(3))
    return np.einsum("btcxyz,x,y,z->btc", G, px, py, pz, optimize=True)


T0 = 1                    # post-launch fit window (launch transient excluded)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(T0, Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(T0 + 1, Gmat.shape[0])], axis=1)
    return B @ np.linalg.pinv(A, rcond=1e-8)


def plus_projector(U, lam_ref):
    """Spectral projector of the +branch (larger phase of the cone pair)."""
    w, Vr = np.linalg.eig(U)
    wl, Vlc = np.linalg.eig(U.conj().T)
    sel = [i for i in range(4)
           if 0.5 * abs(lam_ref) < abs(w[i]) < 1.5 * abs(lam_ref)
           and np.angle(w[i]) > 0]
    if len(sel) < 2:
        return None
    sel = sorted(sel, key=lambda i: abs(abs(w[i]) - abs(lam_ref)))[:2]
    i_p = max(sel, key=lambda i: np.angle(w[i]))
    rv = Vr[:, i_p]
    lv = Vlc[:, np.argmin(np.abs(wl.conj() - w[i_p]))]
    return np.outer(rv, lv.conj()) / (lv.conj() @ rv)


def analyse(mode, q, seed=11):
    t0 = time.time()
    Gs = [evolve_field_cc(L, R, TCYC, q, seed + 100 * c,
                          annealed=(mode == "annealed"), launch=c,
                          n_blocks=NB) for c in range(4)]
    print(f"\n--- {mode} q={q}  ({time.time() - t0:.0f}s evolve) ---",
          flush=True)

    kcache = {}

    def series(kvec):
        key = tuple(np.round(kvec, 9))
        if key not in kcache:
            kcache[key] = np.stack([zk(G, kvec) for G in Gs], axis=-1)
        return kcache[key]

    def gmat(kvec, mask=None):
        if mask is None:
            mask = np.ones(NB, dtype=bool)
        return series(kvec)[mask].mean(axis=0)

    def measure(mask=None):
        lam_ms = []
        for kp in X_POINTS:
            w = np.linalg.eigvals(u_fit(gmat(kp, mask)))
            up = [l for l in w if l.imag > 0.02]
            if up:
                lam_ms.append(max(up, key=abs))
        lam_ref = np.mean(lam_ms)
        us, ns, ang_errs, unres = [], [], [], 0
        for kp in X_POINTS:
            for d in DIRS:
                u = d / np.linalg.norm(d)
                P4 = plus_projector(u_fit(gmat(kp + DELTA * u, mask)), lam_ref)
                if P4 is None:
                    unres += 1
                    continue
                P2 = Sinv @ (VL.conj().T @ P4 @ VR) @ S
                n = np.array([0.5 * np.trace(PAULI[j] @ (2 * P2 - np.eye(2)))
                              for j in range(3)])
                nn = np.sqrt(np.sum(n * n) + 0j)
                if abs(nn) < 1e-6:
                    unres += 1
                    continue
                n = n / nn
                n_ex = R_EX @ u
                n_ex = n_ex / np.sqrt(np.sum(n_ex * n_ex) + 0j)
                ov = np.sum(n * n_ex)
                ang_errs.append(abs(1 - ov.real))
                us.append(u)
                ns.append(n)
        usa, nsa = np.array(us), np.array(ns)
        M = nsa.T @ usa @ np.linalg.inv(usa.T @ usa)
        MtM = M.T @ M
        scale = np.trace(MtM).real / 3
        iso = np.abs(MtM / scale - np.eye(3)).max()
        detM = np.linalg.det(M)
        return {"lam_ref": lam_ref, "chi": int(np.sign(detM.real)),
                "iso": float(iso), "detM": detM,
                "ang": float(np.mean(ang_errs)),
                "ang_max": float(np.max(ang_errs)),
                "nres": len(us), "nunres": unres}

    full = measure()
    jks = [measure(np.arange(NB) != i) for i in range(NB)]
    sig = {}
    for key in ("iso", "ang"):
        a = np.array([j[key] for j in jks])
        sig[key] = float(np.sqrt((NB - 1) / NB * ((a - a.mean()) ** 2).sum()))
    chi_stable = all(j["chi"] == full["chi"] for j in jks)
    print(f"  node pole: {full['lam_ref']:.4f}  (exact annealed {LAM0:.4f})")
    print(f"  samples: {full['nres']} resolved, {full['nunres']} unresolved "
          f"(26 cubic-star directions x 3 pi-copies)")
    print(f"  helicity map: ||M^T M / s - I|| = {full['iso']:.3f} "
          f"+- {sig['iso']:.3f},  det M = {full['detM'].real:+.4f}"
          f"{full['detM'].imag:+.4f}i  ->  chi = {full['chi']} "
          f"(exact {CHI_EX}; stable over all jackknife samples: {chi_stable})")
    print(f"  pointwise |1 - n.n_exact|: mean {full['ang']:.3f} "
          f"+- {sig['ang']:.3f}  max {full['ang_max']:.3f}")
    return {"mode": mode, "q": q, "chi": full["chi"],
            "chi_jk_stable": bool(chi_stable),
            "iso": full["iso"], "sig_iso": sig["iso"],
            "ang_err_mean": full["ang"], "sig_ang": sig["ang"],
            "n_resolved": full["nres"], "n_unresolved": full["nunres"],
            "lam_ref": [full["lam_ref"].real, full["lam_ref"].imag]}


results = []
for mode, q in RUNS:
    r = analyse(mode, q)
    if mode == "annealed":
        # GATE (hard): the known-answer row must reproduce the exact frame
        assert r["chi"] == CHI_EX, "GATE FAILED: chirality"
        assert r["iso"] < 0.10, "GATE FAILED: helicity-map isotropy"
        assert r["ang_err_mean"] < 0.05, "GATE FAILED: pointwise agreement"
        print("  [gate PASSED]")
    results.append(r)
pathlib.Path("results/w4_helicity.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/w4_helicity.json")
