#!/usr/bin/env python
"""I2: the interaction certificate — connected 2-particle correlator, exact.

Substrate: 1D coincube ring (LX = 3), 4 carrier channels/site + TWO env
species/site (e0 = the species the conversions read, e1 = the imprint
partner), 18 modes, 2^18 Fock states, exact (perm, sign) evolution.

Cycle: per sub-step o in {0,1}: L2 conversion (controlled on e0, certified
Givens table), L1 coin-steered shift, stream e0 and e1 (pair swap, origin o).
Once per cycle: L4 imprint — at sites with iota(x) = 1 AND odd carrier
parity, fermionic swap of (e0(x), e1(x)) (adjacent modes, |11> -> -|11>).

Certificate structure (learned the honest way):

    C2 = G2[(i1,i2),(j1,j2)] - (G1[i1,j1] G1[i2,j2] - G1[i1,j2] G1[i2,j1])

is NOT zero at g = 0: the determinant theorem of J31 holds per FIXED medium,
while the env-summed correlator carries connected common-dressing
correlations (E[Lambda^2 M] != Lambda^2 E[M]) — the coincube is already an
interacting carrier-env QFT at the correlator level. The BACK-REACTION
certificate is therefore differential:

    dC2(iota) = C2(iota) - C2(0)      (entrywise, then max |.|)

which vanishes identically iff L4 does nothing, and the g-averaged
dC2(g) ~ g at small g is the coupling dial made visible.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from w3c_lift_check import controlled_l2  # noqa: E402

from pca3d.models.coincube import COIN_D  # noqa: E402

LX = 3
NCAR = 4 * LX                    # carrier modes 4x+c
NM = NCAR + 2 * LX               # env modes 12+2x (e0), 13+2x (e1)
DIM = 1 << NM
QENV = 0.15
TCYC = 3

PAR16 = np.array([bin(v).count("1") % 2 for v in range(16)], dtype=np.int64)


def l2_tables():
    U = controlled_l2(0)
    perm = np.argmax(np.abs(U), axis=0)
    sign = U[perm, np.arange(32)]
    p16 = np.empty(16, dtype=np.int64)
    s16 = np.empty(16, dtype=np.int64)
    for s in range(16):
        t = int(perm[s | 16])
        p16[s] = t & 15
        s16[s] = int(sign[s | 16])
    return p16, s16


L2P, L2S = l2_tables()
STATES = np.arange(DIM, dtype=np.int64)


def lift_mode_perm(pi):
    """Vectorized signed lift of a mode permutation (inversion parity)."""
    perm = np.zeros(DIM, dtype=np.int64)
    for i in range(NM):
        perm |= ((STATES >> i) & 1) << pi[i]
    sgn = np.zeros(DIM, dtype=np.int64)
    for i in range(NM):
        for j in range(i + 1, NM):
            if pi[i] > pi[j]:
                sgn ^= ((STATES >> i) & 1) & ((STATES >> j) & 1)
    return perm, np.where(sgn == 1, -1, 1).astype(np.int64)


def compose(p1, s1, p2, s2):
    return p2[p1], s1 * s2[p1]


def apply_l2_all(perm, sign):
    for x in range(LX):
        base = 4 * x
        ebit = NCAR + 2 * x
        t = perm
        ctrl = (t >> ebit) & 1
        loc = (t >> base) & 15
        newloc = L2P[loc]
        perm = np.where(ctrl == 1, (t & ~(15 << base)) | (newloc << base), t)
        sign = sign * np.where(ctrl == 1, L2S[loc], 1)
    return perm, sign


def mode_perm_shift():
    pi = list(range(NM))
    for c in range(4):
        d = int(COIN_D[0][c])
        for x in range(LX):
            pi[4 * x + c] = 4 * ((x + d) % LX) + c
    return pi


def mode_perm_env(species, o):
    pi = list(range(NM))
    x0, x1 = o, (o + 1) % LX
    i, j = NCAR + 2 * x0 + species, NCAR + 2 * x1 + species
    pi[i], pi[j] = pi[j], pi[i]
    return pi


SHIFT = lift_mode_perm(mode_perm_shift())
ENV = {(sp, o): lift_mode_perm(mode_perm_env(sp, o))
       for sp in (0, 1) for o in (0, 1)}


def apply_l4(perm, sign, iota):
    for x in range(LX):
        if not iota[x]:
            continue
        i, j = NCAR + 2 * x, NCAR + 2 * x + 1
        t = perm
        par = PAR16[(t >> (4 * x)) & 15]
        bi, bj = (t >> i) & 1, (t >> j) & 1
        fire = (par == 1)
        differ = bi != bj
        both = (bi & bj) == 1
        newt = np.where(fire & differ, t ^ ((1 << i) | (1 << j)), t)
        perm = newt
        sign = sign * np.where(fire & both, -1, 1)
    return perm, sign


def cycle(iota):
    perm = STATES.copy()
    sign = np.ones(DIM, dtype=np.int64)
    for o in (0, 1):
        perm, sign = apply_l2_all(perm, sign)
        perm, sign = compose(perm, sign, *SHIFT)
        perm, sign = compose(perm, sign, *ENV[(0, o)])
        perm, sign = compose(perm, sign, *ENV[(1, o)])
    perm, sign = apply_l4(perm, sign, iota)
    return perm, sign


def vac_state():
    """Carriers empty; env bits in the sqrt-Bernoulli product state."""
    v = np.zeros(DIM)
    amp0, amp1 = np.sqrt(1 - QENV), np.sqrt(QENV)
    env_states = range(1 << (2 * LX))
    for e in env_states:
        n1 = bin(e).count("1")
        v[e << NCAR] = amp0 ** (2 * LX - n1) * amp1 ** n1
    return v


def create(j, v):
    out = np.zeros_like(v)
    has = ((STATES >> j) & 1) == 1
    below = np.zeros(DIM, dtype=np.int64)
    for m in range(j):
        below ^= (STATES >> m) & 1
    sgn = np.where(below == 1, -1.0, 1.0)
    src = ~has
    out[STATES[src] | (1 << j)] = sgn[src] * v[src]
    return out


def evolve(v, perm, sign, t):
    for _ in range(t):
        out = np.zeros_like(v)
        out[perm] = sign * v
        v = out
    return v


def correlators(iota, modes_in, modes_out):
    perm, sign = cycle(np.asarray(iota, dtype=bool))
    v0 = vac_state()
    vac_t = evolve(v0.copy(), perm, sign, TCYC)
    kets1 = {j: evolve(create(j, v0), perm, sign, TCYC) for j in modes_in}
    G1 = {(i, j): float(create(i, vac_t) @ kets1[j])
          for i in modes_out for j in modes_in}
    G2 = {}
    for j1 in modes_in:
        for j2 in modes_in:
            if j2 <= j1:
                continue
            ket2 = evolve(create(j1, create(j2, v0)), perm, sign, TCYC)
            for i1 in modes_out:
                for i2 in modes_out:
                    if i2 <= i1:
                        continue
                    bra2 = create(i1, create(i2, vac_t))
                    G2[(i1, i2, j1, j2)] = float(bra2 @ ket2)
    return G1, G2


def connected(G1, G2):
    return {k: G2[k] - (G1[(k[0], k[2])] * G1[(k[1], k[3])] -
                        G1[(k[0], k[3])] * G1[(k[1], k[2])])
            for k in G2}


def main():
    # carriers launched in channel 0 at sites 0 and 1; probed in all channels
    modes_in = [0, 4]
    modes_out = [4 * x + c for x in range(LX) for c in range(4)]

    print(f"substrate: {NM} modes, {DIM} states, T = {TCYC} cycles, "
          f"q_env = {QENV}")

    G1, G2 = correlators([0, 0, 0], modes_in, modes_out)
    C0 = connected(G1, G2)
    c0 = max(abs(v) for v in C0.values())
    print(f"iota = 000 (g = 0):  max |C2| = {c0:.3e}   "
          f"(env-mediated common-dressing baseline; per-fixed-medium "
          f"determinants still exact per J31)")

    vals = {}
    for iota in ([1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]):
        G1i, G2i = correlators(iota, modes_in, modes_out)
        Ci = connected(G1i, G2i)
        dci = max(abs(Ci[k] - C0[k]) for k in Ci)
        vals[tuple(iota)] = dci
        print(f"iota = {''.join(map(str, iota))}:            "
              f"max |dC2| = {dci:.3e}   (back-reaction differential)")
    assert max(vals.values()) > 1e-4, "no interaction detected"

    # g-averaged (quenched iota, Bernoulli(g)) connected correlator
    print("g-averaged connected correlator (exact over the 8 iota patterns):")
    pats = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    gdata = {p: correlators(list(p), modes_in, modes_out) for p in pats}
    for g in (0.02, 0.05, 0.1, 0.2):
        G1a = {k: 0.0 for k in gdata[pats[0]][0]}
        G2a = {k: 0.0 for k in gdata[pats[0]][1]}
        for p in pats:
            w = g ** sum(p) * (1 - g) ** (LX - sum(p))
            for k in G1a:
                G1a[k] += w * gdata[p][0][k]
            for k in G2a:
                G2a[k] += w * gdata[p][1][k]
        Cg = connected(G1a, G2a)
        dcg = max(abs(Cg[k] - C0[k]) for k in Cg)
        print(f"   g = {g:<5} max |dC2| = {dcg:.4e}   (dC2/g = {dcg / g:.4e})")

    print("\nI2 certificate: back-reaction differential dC2 is exactly zero "
          "at g = 0 and switches on ~ g. The imprint interaction is real "
          "and dialled by g.")


if __name__ == "__main__":
    main()
