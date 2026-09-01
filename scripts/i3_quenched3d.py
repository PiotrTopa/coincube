#!/usr/bin/env python
"""I3 hardening: quenched-3D verification of the interaction damping law.

Paired common-noise walker estimator on the massless coincube (q = 0.08) with
the ADR 0014 imprint (C3 rotating pair, permutation lift). Each walker carries
its own media + iota field and is evolved TWICE with identical randomness:
branch A ignores imprints (g = 0), branch B applies them. The flat-contraction
identity (J38) makes E[sign . e^{-ikx}] the canonical coherent-vacuum
propagator, so the theory prediction is

    Z_B(t, k) = (1 - 2 g q^2)^t  Z_A(t, k)         [permutation lift]

for EVERY k: the paired ratio rho(t, k) = Z_B/Z_A must be k-independent
(scalar law = cone/residue survival in one shot), real-positive (no frequency
drift), and match the closed form. Paired noise cancels the caustic variance.

k set: the three X points + X + delta*(100/110/111) at delta = 0.05 —
k-independence across these IS the isotropy certificate at g > 0.
"""
import sys, time

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

from pca3d.models.coincube import COIN_C, COIN_D, PAIRS, perm_sign

L, E, TCYC, Q = 12, 120000, 6, 0.08
GS = (0.3, 0.6)
LAUNCHES = (2, 3)
PERMS, SIGNS = zip(*(perm_sign(c) for c in COIN_C))

KLIST = []
for kp in ([np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]):
    KLIST.append(np.array(kp, float))
for d in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
    u = np.array(d, float)
    u /= np.linalg.norm(u)
    KLIST.append(np.array([np.pi, 0, 0]) + 0.05 * u)


def stream(field, axis, o):
    m = xp.moveaxis(field, axis + 1, -1)          # field: (E, L, L, L)
    if o:
        m = xp.roll(m, -1, axis=-1)
    m0 = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = m0
    if o:
        m = xp.roll(m, 1, axis=-1)
    return xp.moveaxis(m, -1, axis + 1)


def run(g, launch, seed):
    r = np.random.default_rng(seed)
    envA = xp.asarray(r.random((3, E, L, L, L)) < Q)   # pristine (branch A)
    envB = envA.copy()                                  # imprinted (branch B)
    u = r.random((E, L, L, L))
    iota = xp.asarray(np.minimum((u < g).astype(np.int64) *
                                 (1 + (u * 3 / g).astype(np.int64) % 3), 3))
    pos = {b: xp.zeros((E, 3), dtype=xp.int64) for b in "AB"}
    ch = {b: xp.full(E, launch, dtype=xp.int64) for b in "AB"}
    sg = {b: xp.ones(E, dtype=xp.float64) for b in "AB"}
    idx = xp.arange(E)
    perms = [xp.asarray(p) for p in PERMS]
    signs = [xp.asarray(s) for s in SIGNS]
    envs = {"A": envA, "B": envB}
    zA = np.zeros((TCYC, len(KLIST)), dtype=complex)
    zB = np.zeros((TCYC, len(KLIST)), dtype=complex)

    def substep(a, o):
        for b in "AB":
            p = pos[b]
            bit = envs[b][a][idx, p[:, 0] % L, p[:, 1] % L, p[:, 2] % L]
            conv = bit
            sg[b] *= xp.where(conv, signs[a][ch[b]].astype(xp.float64), 1.0)
            ch[b] = xp.where(conv, perms[a][ch[b]], ch[b])
            pos[b] = p.copy()
            pos[b][:, a] += xp.asarray(COIN_D[a])[ch[b]]
        envs["A"][a] = stream(envs["A"][a], (a + 1) % 3, o)
        envs["B"][a] = stream(envs["B"][a], (a + 1) % 3, o)

    for t in range(TCYC):
        for a in (0, 1, 2):
            for o in (0, 1):
                substep(a, o)
        # imprint (branch B only): rotating pair, permutation lift
        p = pos["B"]
        site = (p[:, 0] % L, p[:, 1] % L, p[:, 2] % L)
        pval = iota[idx, site[0], site[1], site[2]]
        for pp, (ea, eb) in enumerate(PAIRS, start=1):
            fire = pval == pp
            ba = envs["B"][ea][idx, site[0], site[1], site[2]]
            bb = envs["B"][eb][idx, site[0], site[1], site[2]]
            sg["B"] *= xp.where(fire & ba & bb, -1.0, 1.0)
            na = xp.where(fire, bb, ba)
            nb = xp.where(fire, ba, bb)
            envs["B"][ea][idx, site[0], site[1], site[2]] = na
            envs["B"][eb][idx, site[0], site[1], site[2]] = nb
        for b, acc in (("A", zA), ("B", zB)):
            x = pos[b].get() if GPU else np.asarray(pos[b])
            s = sg[b].get() if GPU else np.asarray(sg[b])
            for ki, kv in enumerate(KLIST):
                acc[t, ki] += (s * np.exp(-1j * (x @ kv))).sum()
    return zA / E, zB / E


def main():
    print(f"paired quenched-3D damping check: L={L}, E={E} x "
          f"{len(LAUNCHES)} launches, q={Q}, T={TCYC}  (GPU={GPU})")
    for g in GS:
        t0 = time.time()
        zA = np.zeros((TCYC, len(KLIST)), dtype=complex)
        zB = np.zeros_like(zA)
        for li, launch in enumerate(LAUNCHES):
            a, b = run(g, launch, 100 + li)
            zA += a
            zB += b
        law = 1 - 2 * g * Q * Q
        print(f"\n g = {g}  (evolve {time.time() - t0:.0f}s)   "
              f"law per cycle = {law:.5f}")
        noise = 0.5 / np.sqrt(E * len(LAUNCHES))
        print(f" {'t':>3} {'|rho| wmean':>12} {'law^t':>8} {'k-spread':>9} "
              f"{'arg(rho)':>9} {'n_k':>4}")
        for t in range(TCYC):
            gate = np.abs(zA[t]) > 0.05               # ratio-conditioning gate
            # (8x shot noise is not enough: near free-beat amplitude
            # nodes the RATIO needs |Z_A| bounded well away from zero)
            if gate.sum() == 0:
                print(f" {t + 1:>3} {'below gate':>12}")
                continue
            rho = zB[t][gate] / zA[t][gate]
            w = np.abs(zA[t][gate]) ** 2
            wmean = float((np.abs(rho) * w).sum() / w.sum())
            wspread = float(np.sqrt(((np.abs(rho) - wmean) ** 2 * w).sum()
                                    / w.sum()))
            warg = float((np.abs(np.angle(rho)) * w).sum() / w.sum())
            print(f" {t + 1:>3} {wmean:>12.5f} {law ** (t + 1):>8.5f} "
                  f"{wspread:>9.5f} {warg:>9.5f} {int(gate.sum()):>4}")



if __name__ == "__main__":
    main()
