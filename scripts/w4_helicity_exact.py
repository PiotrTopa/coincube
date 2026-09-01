#!/usr/bin/env python
"""W4a: exact helicity/residue structure of the coincube Weyl node (annealed).

At the node (Gamma; all corners are exact copies) the +Im doublet of U(0)
spans a 2-dim subspace V. First-order perturbation gives the effective
generator h(u) = sum_a u_a h_a on V (h_a from left/right eigvec sandwiches of
dU/dk_a). Machine checks:

  1. splitting is purely in phase: eig(h(u)) real (+-v(u));
  2. Clifford structure: {h_a, h_b} = 2 v^2 delta_ab (isotropy at the
     generator level, the residue-side statement of the cone);
  3. the Pauli-frame map n(u): h(u) = |Ru| * (n.sigma), n = Ru/|Ru|:
     R^T R = v^2 I and chirality chi = sign(det R);
  4. residues: the split branches' spectral projectors are rank-1 and equal
     (1 +- n.sigma)/2 in the V-frame.

Writes the frame (V_R, V_L, sigma basis, R, v, chi) to results/w4_frame.npz
for the measured analysis to reuse.
"""
import numpy as np

from pca3d.models.coincube import annealed_u

Q = 0.08
K0 = np.zeros(3)
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]


def doublet_frame(U0):
    w, VR = np.linalg.eig(U0)
    wl, VLc = np.linalg.eig(U0.conj().T)
    up = [i for i in range(4) if w[i].imag > 0]
    lam0 = np.mean(w[up])
    assert max(abs(w[i] - lam0) for i in up) < 1e-9, "node not degenerate"
    R = VR[:, up]                                     # 4x2 right
    upl = [i for i in range(4) if wl[i].imag < 0]     # left eigvals conj
    L = VLc[:, upl]                                   # 4x2 left
    # biorthonormalize: L^dag R = I
    M = L.conj().T @ R
    L = L @ np.linalg.inv(M).conj().T
    assert np.allclose(L.conj().T @ R, np.eye(2), atol=1e-9)
    return lam0, R, L


def du(a, q, h=1e-5):
    e = np.zeros(3)
    e[a] = h
    return (annealed_u(K0 + e, q) - annealed_u(K0 - e, q)) / (2 * h)


def main():
    U0 = annealed_u(K0, Q)
    lam0, VR, VL = doublet_frame(U0)
    print(f"node: lam0 = {lam0:.6f}  |lam0| = {abs(lam0):.6f}  "
          f"om0 = {np.angle(lam0):.6f}")

    # effective generators h_a = -i * VL^dag (dU/dk_a) VR / lam0
    hs = [(-1j / lam0) * (VL.conj().T @ du(a, Q) @ VR) for a in range(3)]

    # 1: reality of the splitting (phase-type) for random directions
    rng = np.random.default_rng(3)
    for _ in range(5):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        ev = np.linalg.eigvals(sum(u[a] * hs[a] for a in range(3)))
        assert np.abs(ev.imag).max() < 1e-7, ev
    print("1. splitting purely in phase: eig(h(u)) real  [OK]")

    # 2: Clifford algebra
    v2 = None
    for a in range(3):
        for b in range(3):
            ac = hs[a] @ hs[b] + hs[b] @ hs[a]
            if a == b:
                sq = ac / 2
                assert np.allclose(sq, sq[0, 0] * np.eye(2), atol=1e-7)
                if v2 is None:
                    v2 = sq[0, 0].real
                assert abs(sq[0, 0] - v2) < 1e-7
            else:
                assert np.abs(ac).max() < 1e-7
    v = np.sqrt(v2)
    print(f"2. Clifford: {{h_a, h_b}} = 2 v^2 delta_ab, v = {v:.6f}  [OK]")

    # 3: Pauli frame and chirality. Choose the 2-dim basis so h_z is diagonal.
    # the V-frame is not unitary (non-normal U), so the Pauli coefficients
    # are COMPLEX with the bilinear orthogonality M^T M = v^2 I (complex
    # orthogonal); det M = +-v^3 is real and its sign is the chirality.
    wz, S = np.linalg.eig(hs[2])
    order = np.argsort(-wz.real)
    S = S[:, order]
    S /= np.sqrt(np.linalg.det(S) + 0j)
    Sinv = np.linalg.inv(S)
    hp = [Sinv @ h @ S for h in hs]
    Rmat = np.zeros((3, 3), complex)
    for a in range(3):
        for j in range(3):
            Rmat[j, a] = 0.5 * np.trace(PAULI[j] @ hp[a])
        assert abs(0.5 * np.trace(hp[a])) < 1e-7
        rebuilt = sum(Rmat[j, a] * PAULI[j] for j in range(3))
        assert np.abs(rebuilt - hp[a]).max() < 1e-6, a
    detR = np.linalg.det(Rmat)
    assert abs(detR.imag) < 1e-7 * abs(detR)
    chi = np.sign(detR.real)
    ortho = np.abs(Rmat.T @ Rmat - v2 * np.eye(3)).max()
    print(f"3. n(u) = R u frame (complex-orthogonal): "
          f"||R^T R - v^2 I|| = {ortho:.2e}, det R = {detR.real:.6f} "
          f"(= chi v^3, v^3 = {v ** 3:.6f}), chi = {int(chi)}  [OK]")
    assert ortho < 1e-6

    # 4: residues are the helicity projectors
    for _ in range(4):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        eps = 1e-3
        Ue = annealed_u(K0 + eps * u, Q)
        w, Vr = np.linalg.eig(Ue)
        wl, Vlc = np.linalg.eig(Ue.conj().T)
        n = Rmat @ u
        n = n / np.sqrt(np.sum(n * n) + 0j)        # bilinear normalisation
        Pth = 0.5 * (np.eye(2) + sum(n[j] * PAULI[j] for j in range(3)))
        # + branch: eigenvalue with larger phase among the split pair
        up = sorted([i for i in range(4) if w[i].imag > 0],
                    key=lambda i: -np.angle(w[i]))
        i_p = up[0]
        rv = Vr[:, i_p]
        lv = Vlc[:, np.argmin(np.abs(wl.conj() - w[i_p]))]
        P4 = np.outer(rv, lv.conj()) / (lv.conj() @ rv)
        P2 = Sinv @ (VL.conj().T @ P4 @ VR) @ S
        assert np.abs(P2 - Pth).max() < 5e-3, np.abs(P2 - Pth).max()
    print("4. residue projectors = (1 + n.sigma)/2, rank-1  [OK]")

    np.savez("results/w4_frame.npz", VR=VR, VL=VL, S=S, Sinv=Sinv, R=Rmat,
             v=v, chi=chi, lam0=lam0, q=Q)
    print(f"\nframe written: results/w4_frame.npz  "
          f"(v = {v:.4f}, chi = {int(chi)})")


if __name__ == "__main__":
    main()
