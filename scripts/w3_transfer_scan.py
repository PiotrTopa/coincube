#!/usr/bin/env python
"""W3a v2: annealed transfer-matrix scan for a real-signed split-step Dirac cone.

Model: n-component coin field. Per axis-a sub-step the annealed (i.i.d. parity)
signed evolution has Bloch matrix

    T_a(k) = (1-p) * diag(exp(i k_a d_a)) + p * C_a

with d_a in {+-1}^n the coin-conditioned move direction and C_a a real signed
permutation (conversion block: carrier stalls, coin channel changes). One cycle:
U(k) = T_z^2 T_y^2 T_x^2 (two blocking origins per axis, matching the CA).

v2 criterion (v1 accepted exceptional points -- real double eigenvalues of the
damped operator, sqrt splitting, cone-bound violations up to v=16):
a PROPAGATING Dirac point at a corner k0 (U(k0) real there) is a degenerate
COMPLEX pair {lam0, lam0} (plus its conjugate pair), i.e. a phase crossing.
Splitting must be linear (h vs 2h within 5%), phase-type (modulus split < 2%),
respect the cone bound (v100 <= 2.05, others <= 2*sqrt(3)), and be isotropic
including two off-symmetry directions.

n=2 corollary checked: a real 2x2 U has only conjugate eigenvalue pairs, which
can only meet on the real axis -- no propagating corner Dirac point exists.
The n=2 scan doubles as a verification of that obstruction.
"""
import itertools, json, pathlib
import numpy as np

I2 = np.eye(2)
X = np.array([[0.0, 1], [1, 0]])
Z = np.diag([1.0, -1])

P_CYCLE = 0.15
GAP_TOL = 1e-8
IM_MIN = 0.05           # |Im lam0|/|lam0|: propagating (phase) degeneracy only
H = 0.02
DIRS = {
    "100": (1.0, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1),
    "r1": (0.276, 0.850, 0.448), "r2": (0.732, 0.214, 0.647),
}
BOUND = {"100": 2.05, "110": 2 * np.sqrt(3), "111": 2 * np.sqrt(3),
         "r1": 2 * np.sqrt(3), "r2": 2 * np.sqrt(3)}


def signed_perms(n):
    out = []
    if n == 2:
        for pi, P in (("1", I2), ("X", X)):
            for si, S in (("1", I2), ("Z", Z)):
                for s in (1, -1):
                    out.append((f"{'+' if s > 0 else '-'}{pi}{si}", s * P @ S))
    else:
        for p1, P1 in (("1", I2), ("X", X)):
            for p2, P2 in (("1", I2), ("X", X)):
                for s1, S1 in (("1", I2), ("Z", Z)):
                    for s2, S2 in (("1", I2), ("Z", Z)):
                        for s in (1, -1):
                            lbl = f"{'+' if s > 0 else '-'}{p1}{p2}.{s1}{s2}"
                            out.append((lbl, s * np.kron(P1, P2) @ np.kron(S1, S2)))
    return out


def d_choices(n):
    if n == 2:
        return [("Z", np.array([1.0, -1]))]
    return [("Z1", np.diag(np.kron(Z, I2))), ("1Z", np.diag(np.kron(I2, Z))),
            ("ZZ", np.diag(np.kron(Z, Z)))]


def cycle_u(kvec, ds, cs, p=P_CYCLE):
    n = cs[0].shape[0]
    u = np.eye(n, dtype=complex)
    for a in range(3):
        t = (1 - p) * np.diag(np.exp(1j * kvec[a] * ds[a])) + p * cs[a]
        u = t @ t @ u
    return u


def propagating_degeneracy(lams):
    """Best degenerate pair among complex (upper-half-plane) eigenvalues."""
    up = [l for l in lams if l.imag > IM_MIN * max(abs(l), 1e-12)]
    best = (np.inf, None)
    for i in range(len(up)):
        for j in range(i + 1, len(up)):
            g = abs(up[i] - up[j])
            if g < best[0]:
                best = (g, 0.5 * (up[i] + up[j]))
    return best


def pair_split(kvec, ds, cs, lam0, p=P_CYCLE):
    """(phase split rate proxy, modulus split fraction) of the pair nearest lam0."""
    lams = np.linalg.eigvals(cycle_u(kvec, ds, cs, p))
    pair = sorted(lams, key=lambda l: abs(l - lam0))[:2]
    ph = abs(np.angle(pair[0] / pair[1]))
    md = abs(abs(pair[0]) - abs(pair[1])) / max(abs(lam0), 1e-12)
    return ph, md


def slope_report(k0, ds, cs, lam0, p=P_CYCLE):
    """Slopes per direction with linearity, modulus-split and bound checks."""
    rep = {}
    ok = True
    for name, dv in DIRS.items():
        u = np.array(dv, float)
        u /= np.linalg.norm(u)
        ph1, md1 = pair_split(k0 + H * u, ds, cs, lam0, p)
        ph2, md2 = pair_split(k0 + 2 * H * u, ds, cs, lam0, p)
        v = ph1 / (2 * H)
        lin = abs(ph2 / (4 * H) - v) / max(v, 1e-12)
        rep[name] = v
        if v > BOUND[name] or lin > 0.05 or max(md1, md2) > 0.02:
            ok = False
        rep[name + "_lin"], rep[name + "_mod"] = lin, max(md1, md2)
    return rep, ok


def main():
    corners = [np.array(c, dtype=float) * np.pi
               for c in itertools.product((0, 1), repeat=3)]

    for n in (2, 4):
        cset = signed_perms(n)
        dset = d_choices(n)
        print(f"\n=== n = {n}: {len(cset)}^3 C-triples ===", flush=True)

        # -- stage 1 (corners: E_a = +-I there, so degeneracy depends only on C's)
        survivors = []
        for (lx, cx), (ly, cy), (lz, cz) in itertools.product(cset, repeat=3):
            for k0 in corners:
                u = np.eye(n)
                for a, c in enumerate((cx, cy, cz)):
                    t = (1 - P_CYCLE) * np.cos(k0[a]) * np.eye(n) + P_CYCLE * c
                    u = t @ t @ u
                lams = np.linalg.eigvals(u.astype(complex))
                gap, lam0 = propagating_degeneracy(lams)
                if gap < GAP_TOL:
                    survivors.append(((lx, ly, lz), (cx, cy, cz), k0, lam0))
        print(f"stage 1: {len(survivors)} (C-triple, corner) with a PROPAGATING "
              f"degeneracy (complex pair, gap < {GAP_TOL})", flush=True)

        # -- stage 2: firewalled slope isotropy
        rows, rejected = [], 0
        for (labels, cs, k0, lam0) in survivors:
            dcombos = ([[dset[0]] * 3] if n == 2 else
                       [[dset[0], a, b] for a in dset for b in dset])
            for combo in dcombos:
                dlab = [c[0] for c in combo]
                ds = [c[1] for c in combo]
                rep, ok = slope_report(k0, ds, cs, lam0)
                if not ok or rep["100"] < 1e-3:
                    rejected += 1
                    continue
                vs = [rep[m] for m in DIRS]
                iso = (max(vs) - min(vs)) / np.mean(vs)
                rows.append({"n": n, "C": labels, "d": tuple(dlab),
                             "k0": (k0 / np.pi).astype(int).tolist(),
                             "v100": rep["100"], "r110": rep["110"] / rep["100"],
                             "r111": rep["111"] / rep["100"],
                             "rr1": rep["r1"] / rep["100"],
                             "rr2": rep["r2"] / rep["100"],
                             "iso": iso, "mod": abs(lam0),
                             "om0": float(np.angle(lam0))})
        rows.sort(key=lambda r: r["iso"])
        print(f"stage 2: {len(rows)} passed the firewall (linear, phase-type, "
              f"cone-bound); {rejected} rejected. Top 12 by isotropy:")
        print(f"{'C(x,y,z)':>28} {'d(x,y,z)':>12} {'k0/pi':>10} {'v100':>7} "
              f"{'r110':>6} {'r111':>6} {'rr1':>6} {'rr2':>6} {'iso':>7} {'|lam|':>6}")
        for r in rows[:12]:
            print(f"{','.join(r['C']):>28} {','.join(r['d']):>12} "
                  f"{str(r['k0']):>10} {r['v100']:>7.4f} {r['r110']:>6.3f} "
                  f"{r['r111']:>6.3f} {r['rr1']:>6.3f} {r['rr2']:>6.3f} "
                  f"{r['iso']:>7.4f} {r['mod']:>6.3f}")
        pathlib.Path(f"results/w3_transfer_scan_n{n}.json").write_text(
            json.dumps(rows[:200], indent=1))

        # -- p-stability of the best candidate
        if rows and rows[0]["iso"] < 0.10:
            best = rows[0]
            cmap, dmap = dict(cset), dict(dset)
            cs = [cmap[l] for l in best["C"]]
            ds = [dmap[l] for l in best["d"]]
            k0 = np.array(best["k0"], float) * np.pi
            print("p-sweep of the winner (cone ratios must be p-stable, bound obeyed):")
            for p in (0.05, 0.10, 0.15, 0.25):
                u = cycle_u(k0, ds, cs, p)
                gap, lam0 = propagating_degeneracy(np.linalg.eigvals(u))
                if lam0 is None or gap > 1e-6:
                    print(f"  p={p:.2f}  degeneracy lost (gap={gap:.2e})")
                    continue
                rep, ok = slope_report(k0, ds, cs, lam0, p)
                print(f"  p={p:.2f}  v100={rep['100']:.4f}  "
                      f"r110={rep['110'] / rep['100']:.3f}  "
                      f"r111={rep['111'] / rep['100']:.3f}  "
                      f"rr1={rep['r1'] / rep['100']:.3f}  "
                      f"firewall={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
