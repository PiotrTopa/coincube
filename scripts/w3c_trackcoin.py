#!/usr/bin/env python
"""W3c: annealed spectrum of the LEGAL track-coin block CA.

Block rule per axis-a sub-step (manifestly legal partitioned permutation):
  env parity even -> swap all species between the block's two sites (motion,
  coin-blind, track-encoded direction exactly as in rule 891);
  env parity odd  -> apply the signed channel permutation C_a on site (stall +
  conversion); env bits swap always.
The coin is a passively carried quaternion: it never steers the bits, only
weights paths (non-Abelian carried gauge -- not excluded by ADR 0010, which
covers Abelian position-periodic gauges).

Annealed state: coin (4) x track (t in {+-1}^3, 8) = 32 dims.
  T_a(k) = (1-p) e^{i k_a t_a} (motion; diag in track, identity on coin)
         + p (flip t_a) (x) C_a (stall).
Question: does the cycle U = Tz^2 Ty^2 Tx^2 still carry the isotropic cone?
Firewall: propagating-degeneracy criterion, linearity, modulus-split,
cone-bound assert (threat classes #6/#7 protocol).
"""
import itertools
import numpy as np

I2 = np.eye(2)
X = np.array([[0.0, 1], [1, 0]])
Z = np.diag([1.0, -1])
XZ = X @ Z

CS = [np.kron(XZ, I2), np.kron(Z, XZ), -np.kron(X, XZ)]   # quaternion triple

# track space: 8 states, t_a = +-1 read from bit a (state index bit a)
TRK = 8


def t_val(state, a):
    return 1 - 2 * ((state >> a) & 1)


def flip_mat(a):
    m = np.zeros((TRK, TRK))
    for s in range(TRK):
        m[s ^ (1 << a), s] = 1.0
    return m


FLIPS = [flip_mat(a) for a in range(3)]
TDIAG = [np.diag([t_val(s, a) for s in range(TRK)]).astype(float)
         for a in range(3)]


def t_a_mat(kva, a, p, C):
    d = np.diag(np.exp(1j * kva * np.diag(TDIAG[a])))
    move = np.kron(np.eye(4), d) * (1 - p)
    stall = p * np.kron(C, FLIPS[a])
    return move + stall


def cycle_u(kvec, p):
    u = np.eye(32, dtype=complex)
    for a in range(3):
        t = t_a_mat(kvec[a], a, p, CS[a])
        u = t @ t @ u
    return u


def propagating_pairs(lams, tol=1e-9):
    """Degenerate complex pairs (upper half plane), grouped."""
    up = sorted([l for l in lams if l.imag > 0.02 * max(abs(l), 1e-12)],
                key=lambda l: -abs(l))
    groups = []
    used = [False] * len(up)
    for i in range(len(up)):
        if used[i]:
            continue
        grp = [up[i]]
        used[i] = True
        for j in range(i + 1, len(up)):
            if not used[j] and abs(up[i] - up[j]) < tol:
                grp.append(up[j])
                used[j] = True
        if len(grp) >= 2:
            groups.append((abs(grp[0]), np.mean(grp), len(grp)))
    return sorted(groups, key=lambda g: -g[0])


def split_slope(k0, u, p, lam0, h):
    lams = np.linalg.eigvals(cycle_u(k0 + h * u, p))
    pair = sorted(lams, key=lambda l: abs(l - lam0))[:2]
    ph = abs(np.angle(pair[0] / pair[1]))
    md = abs(abs(pair[0]) - abs(pair[1])) / max(abs(lam0), 1e-12)
    return ph / (2 * h), md


def main():
    p = 0.15
    corners = [np.array(c, float) * np.pi
               for c in itertools.product((0, 1), repeat=3)]
    dirs = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1),
            "r1": (0.276, 0.850, 0.448), "r2": (0.732, 0.214, 0.647)}

    print(f"track-coin legal CA, annealed spectrum at p={p}")
    for k0 in corners:
        lams = np.linalg.eigvals(cycle_u(k0, p))
        groups = propagating_pairs(lams)
        if not groups:
            print(f"k0/pi={[int(x / np.pi) for x in k0]}: no propagating "
                  f"degeneracy (top |lam| {max(abs(lams)):.3f})")
            continue
        mod, lam0, mult = groups[0]
        print(f"k0/pi={[int(x / np.pi) for x in k0]}: degeneracy x{mult} at "
              f"|lam0|={abs(lam0):.4f} om0={np.angle(lam0):.4f}")
        rep = {}
        ok = True
        for name, dv in dirs.items():
            u = np.array(dv, float)
            u /= np.linalg.norm(u)
            v1, md1 = split_slope(k0, u, p, lam0, 0.01)
            v2, md2 = split_slope(k0, u, p, lam0, 0.005)
            v = 2 * v2 - v1
            rep[name] = v
            if v > 2 * np.sqrt(3) + 0.1 or max(md1, md2) > 0.02:
                ok = False
        vs = np.array(list(rep.values()))
        print(f"   slopes: " + "  ".join(f"{n}:{v:.4f}" for n, v in rep.items()))
        print(f"   iso max/min = {vs.max() / max(vs.min(), 1e-12):.4f}  "
              f"firewall={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
