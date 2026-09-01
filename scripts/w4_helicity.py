#!/usr/bin/env python
"""W4b: measured helicity residues of the coincube Weyl node.

Per (X-point, direction u, delta): fit the 4x4 one-cycle U from the 4-launch
propagator (v5 instrument), take the + branch spectral projector |r><l|/<l|r>,
express it in the EXACT node frame (results/w4_frame.npz, from
w4_helicity_exact.py), and read off the helicity vector n_meas (complex Pauli
coefficients, bilinear normalisation). Across 3 X-points x 26 directions this
gives the measured map n(u); fit n = M u and check:

  - purity: Tr P ~ 1, Tr P^2 ~ 1 (rank-1 residue);
  - frame: M^T M proportional to I (bilinear), chirality = sign(det M);
  - pointwise angular agreement with the exact n(u) = R u.

Gate: the annealed row must reproduce the exact frame (chi = -1, small
residuals) before the quenched row is believed.
"""
import json, pathlib, time
import numpy as np

from pca3d.models.coincube import evolve_field_cc

L, R, TCYC = 48, 3000, 8
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
    return np.einsum("tcxyz,x,y,z->tc", G, px, py, pz, optimize=True)


def u_fit(Gmat):
    A = np.concatenate([Gmat[t] for t in range(Gmat.shape[0] - 1)], axis=1)
    B = np.concatenate([Gmat[t] for t in range(1, Gmat.shape[0])], axis=1)
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
                          annealed=(mode == "annealed"), launch=c)
          for c in range(4)]
    print(f"\n--- {mode} q={q}  ({time.time() - t0:.0f}s evolve) ---",
          flush=True)

    def gmat(kvec):
        return np.stack([np.stack([zk(G, kvec)[t] for G in Gs], axis=1)
                         for t in range(TCYC + 1)])

    # measured lam0 from the X points (for branch selection)
    lam_ms = []
    for kp in X_POINTS:
        w = np.linalg.eigvals(u_fit(gmat(kp)))
        up = [l for l in w if l.imag > 0.02]
        if up:
            lam_ms.append(max(up, key=abs))
    lam_ref = np.mean(lam_ms)
    print(f"  node pole: {lam_ref:.4f}  (exact annealed {LAM0:.4f})")

    us, ns, purities, ang_errs, unres = [], [], [], [], 0
    for kp in X_POINTS:
        for d in DIRS:
            u = d / np.linalg.norm(d)
            P4 = plus_projector(u_fit(gmat(kp + DELTA * u)), lam_ref)
            if P4 is None:
                unres += 1
                continue
            P2 = Sinv @ (VL.conj().T @ P4 @ VR) @ S
            tr, tr2 = np.trace(P2), np.trace(P2 @ P2)
            n = np.array([0.5 * np.trace(PAULI[j] @ (2 * P2 - np.eye(2)))
                          for j in range(3)])
            nn = np.sqrt(np.sum(n * n) + 0j)
            if abs(nn) < 1e-6:
                unres += 1
                continue
            n = n / nn
            n_ex = R_EX @ u
            n_ex = n_ex / np.sqrt(np.sum(n_ex * n_ex) + 0j)
            # bilinear overlap: +-1 for aligned/anti-aligned helicity
            ov = np.sum(n * n_ex)
            ang_errs.append(abs(1 - ov.real))
            purities.append(abs(tr2 / max(abs(tr), 1e-9) ** 2))
            us.append(u)
            ns.append(n)
    us, ns = np.array(us), np.array(ns)
    # least-squares complex M: n = M u
    M = ns.T @ us @ np.linalg.inv(us.T @ us)
    MtM = M.T @ M
    scale = np.trace(MtM).real / 3
    iso = np.abs(MtM / scale - np.eye(3)).max()
    detM = np.linalg.det(M)
    chi = int(np.sign(detM.real))
    print(f"  samples: {len(us)} resolved, {unres} unresolved")
    print(f"  purity Tr P^2 / (Tr P)^2: mean {np.mean(purities):.3f} "
          f"(rank-1 -> 1.000)")
    print(f"  helicity map: ||M^T M / s - I|| = {iso:.3f},  "
          f"det M = {detM.real:+.4f}{detM.imag:+.4f}i  ->  chi = {chi} "
          f"(exact {CHI_EX})")
    print(f"  pointwise |1 - n.n_exact|: mean {np.mean(ang_errs):.3f}  "
          f"max {np.max(ang_errs):.3f}")
    return {"mode": mode, "q": q, "chi": chi, "iso": float(iso),
            "purity": float(np.mean(purities)),
            "ang_err_mean": float(np.mean(ang_errs)),
            "n_resolved": len(us), "n_unresolved": unres,
            "lam_ref": [lam_ref.real, lam_ref.imag]}


results = [analyse(mode, q) for mode, q in RUNS]
pathlib.Path("results/w4_helicity.json").write_text(json.dumps(results, indent=1))
print("\nwritten: results/w4_helicity.json")
