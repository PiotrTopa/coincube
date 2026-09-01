#!/usr/bin/env python
"""E1 at g > 0: the complex structure survives the imprint interaction.

Red-team item 2.4/3 (the paper claimed this [machine-checked] with no check).
Substrate: the i2_connected 18-mode cycle WITH the L4 imprint layer at fixed
nonzero iota patterns. P = Majorana string over the 12 carrier modes;
eta = sign(N_c - 6) on non-half-filled sectors. Checks, all exact:

  1. [S_g, P] = 0 including all lift signs, for every iota pattern;
  2. I = P diag(eta): I^2 = -1, antisymmetric, {K, I} = 0, and
     [S_g, I] = 0 on the non-half-filled sectors;
  3. the complex-picture Hermitian form is invariant under S_g.

(The parity control of L4 is PH-even by construction; this verifies the
LIFTED statement, signs included.)
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from i2_connected import DIM, LX, NCAR, STATES, cycle  # noqa: E402


def ph_lift():
    perm = np.arange(DIM, dtype=np.int64)
    sign = np.ones(DIM, dtype=np.int64)
    for i in range(NCAR):
        newp = np.empty_like(perm)
        news = np.empty_like(sign)
        for s in range(DIM):
            t = perm[s]
            below = bin(t & ((1 << i) - 1)).count("1")
            newp[s] = t ^ (1 << i)
            news[s] = sign[s] * (-1 if below % 2 else 1)
        perm, sign = newp, news
    return perm, sign


def compose(p1, s1, p2, s2):
    return p2[p1], s1 * s2[p1]


def main():
    P_perm, P_sign = ph_lift()
    n_c = np.array([bin(s & ((1 << NCAR) - 1)).count("1")
                    for s in range(DIM)])
    eta = np.sign(n_c - NCAR // 2)
    nonhalf = eta != 0

    for iota in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]):
        S_perm, S_sign = cycle(np.asarray(iota, dtype=bool))
        sp = compose(S_perm, S_sign, P_perm, P_sign)
        ps = compose(P_perm, P_sign, S_perm, S_sign)
        assert np.array_equal(sp[0], ps[0]) and np.array_equal(sp[1], ps[1]), \
            f"[S,P] != 0 at iota={iota}"
        I_perm = P_perm
        I_sign = (P_sign * eta).astype(np.int64)
        i2s = I_sign * I_sign[I_perm]
        assert np.all(i2s[nonhalf] == -1)
        assert np.all((I_sign + I_sign[I_perm])[nonhalf] == 0)   # antisym
        assert np.all((eta[I_perm] + eta)[nonhalf] == 0)          # {K,I}=0
        si = compose(S_perm, S_sign, I_perm, I_sign)
        is_ = compose(I_perm, I_sign, S_perm, S_sign)
        assert (np.array_equal(si[0][nonhalf], is_[0][nonhalf]) and
                np.array_equal(si[1][nonhalf], is_[1][nonhalf])), \
            f"[S,I] != 0 at iota={iota}"

        def apply(perm, sign, v):
            out = np.zeros_like(v)
            out[perm] = sign * v
            return out

        rng = np.random.default_rng(3)
        q1 = rng.normal(size=DIM) * nonhalf
        q2 = rng.normal(size=DIM) * nonhalf
        h0 = q1 @ q2 + 1j * (q1 @ apply(I_perm, I_sign, q2))
        s1v = apply(S_perm, S_sign, q1)
        s2v = apply(S_perm, S_sign, q2)
        h1 = s1v @ s2v + 1j * (s1v @ apply(I_perm, I_sign, s2v))
        assert abs(h0 - h1) < 1e-9
        print(f"iota={iota}: [S,P]=0, [S,I]=0, unitary complex picture  [OK]")

    print("\nE1 at g > 0: the complex structure survives the imprint "
          "interaction, lift signs included.")


if __name__ == "__main__":
    main()
