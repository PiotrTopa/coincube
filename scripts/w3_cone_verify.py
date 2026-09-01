#!/usr/bin/env python
"""W3a fine verification of the quaternionic cone candidate.

Winner of w3_transfer_scan v2:
  C_x = (X(x)1)(Z(x)1) = XZ(x)1,  C_y = (1(x)X)(Z(x)Z) = Z(x)XZ,
  C_z = -(X(x)X)(1(x)Z) = -X(x)XZ
  d_x = diag(Z(x)1), d_y = diag(Z(x)Z), d_z = diag(1(x)Z),  k0 = (pi, 0, 0)
The C's are a real quaternion triple (pairwise anticommuting, squaring to -I):
the real irrep of Cl(0,3), as the n=2 obstruction argument demands.

Checks: (1) h -> 0 extrapolation of directional slopes (is the residual
anisotropy curvature contamination or genuine?); (2) 50-direction isotropy
map; (3) p-sweep of gap, slopes, damping; (4) band structure along lines
(linearity range). Firewall: cone bound asserted on every slope.
"""
import json, pathlib
import numpy as np

I2 = np.eye(2)
X = np.array([[0.0, 1], [1, 0]])
Z = np.diag([1.0, -1])
XZ = X @ Z

CS = [np.kron(XZ, I2), np.kron(Z, XZ), -np.kron(X, XZ)]
DS = [np.diag(np.kron(Z, I2)), np.diag(np.kron(Z, Z)), np.diag(np.kron(I2, Z))]
K0 = np.array([np.pi, 0.0, 0.0])


def cycle_u(kvec, p):
    u = np.eye(4, dtype=complex)
    for a in range(3):
        t = (1 - p) * np.diag(np.exp(1j * kvec[a] * DS[a])) + p * CS[a]
        u = t @ t @ u
    return u


def lam0_at(p):
    lams = np.linalg.eigvals(cycle_u(K0, p))
    up = sorted([l for l in lams if l.imag > 0.02], key=lambda l: -abs(l))
    gap = abs(up[0] - up[1]) if len(up) >= 2 else np.inf
    return 0.5 * (up[0] + up[1]) if len(up) >= 2 else None, gap


def split(kvec, p, lam0):
    lams = np.linalg.eigvals(cycle_u(kvec, p))
    pair = sorted(lams, key=lambda l: abs(l - lam0))[:2]
    ph = abs(np.angle(pair[0] / pair[1]))
    md = abs(abs(pair[0]) - abs(pair[1])) / abs(lam0)
    return ph, md


def v_dir(u, p, lam0, h):
    ph, md = split(K0 + h * u, p, lam0)
    return ph / (2 * h), md


def main():
    quat = all(abs(CS[i] @ CS[j] + CS[j] @ CS[i]).max() < 1e-14
               for i in range(3) for j in range(i + 1, 3)) and \
           all(abs(CS[i] @ CS[i] + np.eye(4)).max() < 1e-14 for i in range(3))
    print(f"quaternion triple check (anticommuting, C^2 = -I): {quat}")

    p = 0.15
    lam0, gap = lam0_at(p)
    print(f"p={p}: k0 gap = {gap:.2e}, |lam0| = {abs(lam0):.4f}, "
          f"omega0 = {np.angle(lam0):.4f}, Gamma = {-np.log(abs(lam0)):.4f}/cycle")

    # 1 -- h -> 0 extrapolation along the symmetry directions
    dirs = {"100": (1, 0, 0), "010": (0, 1, 0), "001": (0, 0, 1),
            "110": (1, 1, 0), "101": (1, 0, 1), "011": (0, 1, 1),
            "111": (1, 1, 1)}
    print("\nh-extrapolation of slopes (Richardson from h, h/2):")
    print(f"{'dir':>5} {'v(h=.04)':>9} {'v(.02)':>8} {'v(.01)':>8} {'v(.005)':>8} "
          f"{'v(h->0)':>8} {'mod':>6}")
    v0 = {}
    for name, dv in dirs.items():
        u = np.array(dv, float); u /= np.linalg.norm(u)
        vs = []
        for h in (0.04, 0.02, 0.01, 0.005):
            v, md = v_dir(u, p, lam0, h)
            vs.append(v)
        rich = 2 * vs[3] - vs[2]        # h^2 Richardson from the two finest
        v0[name] = rich
        assert rich <= 2 * np.sqrt(3) + 0.1, f"cone bound violated on {name}"
        print(f"{name:>5} {vs[0]:>9.4f} {vs[1]:>8.4f} {vs[2]:>8.4f} {vs[3]:>8.4f} "
              f"{rich:>8.4f} {md:>6.3f}")
    r = {k: v0[k] / v0['100'] for k in v0}
    print("extrapolated ratios: " +
          "  ".join(f"{k}:{r[k]:.4f}" for k in ('010', '001', '110', '111')))

    # 2 -- 50 random directions at h -> 0
    rng = np.random.default_rng(7)
    vs = []
    for _ in range(50):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v1, _ = v_dir(u, p, lam0, 0.01)
        v2, _ = v_dir(u, p, lam0, 0.005)
        vs.append(2 * v2 - v1)
    vs = np.array(vs)
    print(f"\n50 random directions (h->0): mean v = {vs.mean():.4f}, "
          f"min = {vs.min():.4f}, max = {vs.max():.4f}, "
          f"max/min = {vs.max() / vs.min():.4f}")

    # 3 -- p-sweep
    print("\np-sweep (gap, damping, extrapolated v100/v111 ratio):")
    for pp in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        l0, g = lam0_at(pp)
        if l0 is None or g > 1e-6:
            print(f"  p={pp:.2f}  no propagating degeneracy (gap={g:.2e})")
            continue
        vv = {}
        for name in ("100", "111"):
            u = np.array(dirs[name], float); u /= np.linalg.norm(u)
            v1, _ = v_dir(u, pp, l0, 0.01)
            v2, _ = v_dir(u, pp, l0, 0.005)
            vv[name] = 2 * v2 - v1
        print(f"  p={pp:.2f}  gap={g:.1e}  |lam0|={abs(l0):.3f}  "
              f"v100={vv['100']:.4f}  r111={vv['111'] / vv['100']:.4f}")

    # 4 -- band structure along 100 and 111 through k0 (linearity range)
    print("\nband omega(k) - omega0 along 100 and 111 (propagating branches):")
    for name in ("100", "111"):
        u = np.array(dirs[name], float); u /= np.linalg.norm(u)
        row = []
        for s in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
            ph, md = split(K0 + s * u, p, lam0)
            row.append(f"k={s:.2f}:{ph / 2:.4f}{'*' if md > 0.02 else ''}")
        print(f"  {name}: " + "  ".join(row) + "   (* = modulus-split > 2%)")

    json.dump({"ratios_h0": r, "v_random_maxmin": float(vs.max() / vs.min()),
               "gamma": float(-np.log(abs(lam0)))},
              open("results/w3_cone_verify.json", "w"), indent=1)
    print("\nwritten: results/w3_cone_verify.json")


if __name__ == "__main__":
    main()
