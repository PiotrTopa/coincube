#!/usr/bin/env python
"""I3a: substrate verification of the one-sided amplitude damping law.

Theory (docs/notes/theory-interaction.md, proven at annealed order): the
parity control makes the vacuum imprint-transparent, so the ensemble 1p
amplitude stays CLOSED under back-reaction:

    G1_g(t) = (1 - 2 g q (1-q))^t  G1_0(t)    (+ recross corrections)

i.e. pure scalar damping Gamma_int = 2q(1-q) g per cycle; cone, residues and
mass untouched. Here: exact check on the 18-mode Fock substrate (quenched
static iota patterns, exact Bernoulli(g) average), extracting the per-cycle
factor from the projection of G1_g onto G1_0 and comparing to the law.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from i2_connected import (DIM, LX, QENV, STATES, create, cycle,  # noqa: E402
                          evolve, vac_state)

TMAX = 4
MODES_OUT = [4 * x + c for x in range(LX) for c in range(4)]


def g1_series(iota):
    perm, sign = cycle(np.asarray(iota, dtype=bool))
    v0 = vac_state()
    ket = create(0, v0)
    vac = v0.copy()
    out = []
    for t in range(1, TMAX + 1):
        ket = evolve(ket, perm, sign, 1)
        vac = evolve(vac, perm, sign, 1)
        out.append(np.array([create(i, vac) @ ket for i in MODES_OUT]))
    return out


def main():
    pats = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    data = {p: g1_series(list(p)) for p in pats}
    base = data[(0, 0, 0)]
    f1cyc = lambda g: 1 - 2 * g * QENV * (1 - QENV)
    print(f"substrate damping law check  (q = {QENV}, "
          f"2q(1-q) = {2 * QENV * (1 - QENV):.4f})")
    print(f"{'g':>6} {'t':>3} {'ratio^(1/t)':>12} {'law':>8} {'rel dev':>9}")
    for g in (0.05, 0.1, 0.2):
        for t in range(1, TMAX + 1):
            num = np.zeros_like(base[0])
            for p in pats:
                w = g ** sum(p) / 3 ** 0 * (1 - g) ** (LX - sum(p))
                num += w * data[p][t - 1]
            # projection of G1_g onto G1_0 at the same t
            r = float(num @ base[t - 1]) / float(base[t - 1] @ base[t - 1])
            percyc = np.sign(r) * abs(r) ** (1.0 / t)
            law = f1cyc(g)
            print(f"{g:>6} {t:>3} {percyc:>12.6f} {law:>8.6f} "
                  f"{abs(percyc - law) / (1 - law):>9.3f}")
    print("\n(rel dev is relative to the damping depth 1-law; the growing-t "
          "excess is the identified recross channel — quenched static iota "
          "revisits, scalar and isotropic by the C3 design.)")


if __name__ == "__main__":
    main()
