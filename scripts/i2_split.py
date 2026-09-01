#!/usr/bin/env python
"""I2 hardening: separate the parity-blocking contact artifact from the
mediated interaction by a CONTROL-VARIANT experiment.

The production L4 control fires on ODD site carrier parity (PH-even, E1-safe)
and therefore never fires on coincident carrier pairs — a kinematic blocking
that contributes a contact artifact to dC2 (theory-interaction.md item 4).
Control experiment: an occupancy-controlled variant (fires when any carrier
is present; breaks PH, used ONLY as an instrument control) has no blocking.
The difference

    dC2_parity - dC2_occupancy   (entrywise)

isolates the blocking artifact; dC2_occupancy is the mediated part (plus
double-fire effects, O(g) identical). Exact on the 18-mode substrate.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
import i2_connected as I  # noqa: E402

OCC16 = np.array([1 if v else 0 for v in range(16)], dtype=np.int64)
OCC16[0] = 0
OCC16[1:] = 1


def apply_l4_occ(perm, sign, iota):
    for x in range(I.LX):
        if not iota[x]:
            continue
        i, j = I.NCAR + 2 * x, I.NCAR + 2 * x + 1
        t = perm
        occ = OCC16[(t >> (4 * x)) & 15]
        bi, bj = (t >> i) & 1, (t >> j) & 1
        fire = occ == 1
        differ = bi != bj
        both = (bi & bj) == 1
        perm = np.where(fire & differ, t ^ ((1 << i) | (1 << j)), t)
        sign = sign * np.where(fire & both, -1, 1)
    return perm, sign


def cycle_occ(iota):
    perm = I.STATES.copy()
    sign = np.ones(I.DIM, dtype=np.int64)
    for o in (0, 1):
        perm, sign = I.apply_l2_all(perm, sign)
        perm, sign = I.compose(perm, sign, *I.SHIFT)
        perm, sign = I.compose(perm, sign, *I.ENV[(0, o)])
        perm, sign = I.compose(perm, sign, *I.ENV[(1, o)])
    return apply_l4_occ(perm, sign, np.asarray(iota, dtype=bool))


def correlators_with(cycle_fn, iota, modes_in, modes_out):
    perm, sign = cycle_fn(iota)
    v0 = I.vac_state()
    vac_t = I.evolve(v0.copy(), perm, sign, I.TCYC)
    kets1 = {j: I.evolve(I.create(j, v0), perm, sign, I.TCYC)
             for j in modes_in}
    G1 = {(i, j): float(I.create(i, vac_t) @ kets1[j])
          for i in modes_out for j in modes_in}
    G2 = {}
    for j1 in modes_in:
        for j2 in modes_in:
            if j2 <= j1:
                continue
            ket2 = I.evolve(I.create(j1, I.create(j2, v0)), perm, sign, I.TCYC)
            for i1 in modes_out:
                for i2 in modes_out:
                    if i2 <= i1:
                        continue
                    bra2 = I.create(i1, I.create(i2, vac_t))
                    G2[(i1, i2, j1, j2)] = float(bra2 @ ket2)
    return G1, G2


def main():
    modes_in = [0, 4]
    modes_out = [4 * x + c for x in range(I.LX) for c in range(4)]
    base_p = correlators_with(lambda i: I.cycle(np.asarray(i, bool)),
                              [0, 0, 0], modes_in, modes_out)
    C0 = I.connected(*base_p)
    print("control-variant split of dC2 (substrate, single-site iota):")
    print(f"{'iota':>6} {'|dC2| parity':>13} {'|dC2| occup.':>13} "
          f"{'|blocking|':>11} {'block/total':>12}")
    for iota in ([1, 0, 0], [0, 1, 0], [1, 1, 1]):
        Gp = correlators_with(lambda i: I.cycle(np.asarray(i, bool)),
                              iota, modes_in, modes_out)
        Go = correlators_with(cycle_occ, iota, modes_in, modes_out)
        Cp = I.connected(*Gp)
        Co = I.connected(*Go)
        dp = {k: Cp[k] - C0[k] for k in Cp}
        do = {k: Co[k] - C0[k] for k in Co}
        blk = {k: dp[k] - do[k] for k in dp}
        mp = max(abs(v) for v in dp.values())
        mo = max(abs(v) for v in do.values())
        mb = max(abs(v) for v in blk.values())
        print(f"{''.join(map(str, iota)):>6} {mp:>13.4e} {mo:>13.4e} "
              f"{mb:>11.4e} {mb / mp:>12.3f}")
    print("\n(the occupancy variant is an instrument CONTROL only — it breaks "
          "PH and is never part of the model; the blocking artifact is the "
          "parity-vs-occupancy difference, separable as predicted.)")


if __name__ == "__main__":
    main()
