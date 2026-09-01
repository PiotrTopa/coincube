#!/usr/bin/env python
"""Paper item: the damping story as exact prediction, not error.

Every width in the model is a closed form, and every speed is computable
exactly from the annealed operator:
  free quasiparticle:  |lam0(q)|^2 = ((1-q)^2+q^2)^6
                       -> Gamma(q) = -3 ln((1-q)^2+q^2) per cycle
  mass layer:          modulus factor sqrt((1-qm)^2+qm^2) per cycle
  interaction:         Gamma_int = 2 q^2 g per cycle (permutation lift)
Unitary limit q, qm, g -> 0: Gamma -> 6q -> 0 while v(q) -> 2/sqrt(3) stays
finite and the cone/chirality/mass-law structure is untouched. The
quasiparticle width is an exact prediction of the vacuum engineering, with
quality factor Q = omega0/Gamma rising from 0.60 to 0.90 across the working
range (propagation and damping share their origin in the conversion events,
but omega0 grows faster).
"""
import numpy as np

from pca3d.models.coincube import annealed_u


def node_v(q, h=1e-4):
    """Exact cone speed at Gamma from the splitting slope."""
    lams0 = np.linalg.eigvals(annealed_u(np.zeros(3), q))
    lam0 = max((l for l in lams0 if l.imag > 0), key=abs)
    u = np.array([1.0, 0, 0])
    lams = np.linalg.eigvals(annealed_u(h * u, q))
    pair = sorted(lams, key=lambda l: abs(l - lam0))[:2]
    return abs(np.angle(pair[0] / pair[1])) / (2 * h), lam0


def main():
    print(f"{'q':>6} {'Gamma/cyc':>10} {'omega0':>8} {'Q':>6} {'v(q)':>7} "
          f"{'Gam_int(g=.1)':>14} {'int/free':>9}")
    for q in (0.02, 0.05, 0.08, 0.15, 0.25):
        gam_cf = -3 * np.log((1 - q) ** 2 + q ** 2)
        v, lam0 = node_v(q)
        gam_op = -np.log(abs(lam0))
        assert abs(gam_cf - gam_op) < 1e-9          # closed form == operator
        om0 = abs(np.angle(lam0))
        gi = 2 * q ** 2 * 0.1
        print(f"{q:>6} {gam_cf:>10.5f} {om0:>8.5f} {om0 / gam_cf:>6.3f} "
              f"{v:>7.4f} {gi:>14.6f} {gi / gam_cf:>9.4f}")
    print("\nunitary limit: Gamma ~ 6q -> 0, v -> 2/sqrt(3) = 1.1547; "
          "interaction cost is O(q^2 g) — vanishing faster than the free "
          "width itself.")


if __name__ == "__main__":
    main()
