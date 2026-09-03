#!/usr/bin/env python
"""Global overdamping of the coincube node: Q <= pi/(2 ln 2) proven,
sup Q = pi/(3 ln 2) machine-checked.

The chain, with every finite step asserted here:

1. LEMMA (quaternion triangle inequality) [proven]. At k = 0 each sub-step
   factor is rho * u_a with u_a a UNIT quaternion of angle
   theta(q) = arctan[q/(1-q)] (the factor (1-q) + q C_a has eigenvalues
   rho e^{+-i theta}), and the cycle at k = 0 is rho^6 times LEFT
   MULTIPLICATION by the product unit quaternion u = u_z^2 u_y^2 u_x^2.
   The bi-invariant metric on unit quaternions (S^3 ~ SU(2)) gives the
   triangle inequality for rotation angles: angle(u v) <= angle(u) +
   angle(v). Hence omega_0 = angle(u) <= 6 theta(q).
   Machine check: omega_0(q) <= 6 theta(q) on a dense density grid.

2. COROLLARY. Q(q) = omega_0 / Gamma <= 6 theta / (-3 ln rho^2)
   = 2 theta(q) / (-ln rho^2(q)) =: g(q). The function g is increasing on
   (0, 1/2] with g(1/2) = (pi/2)/ln 2 = pi/(2 ln 2) = kappa -- the same
   constant as the Q-pinning theorem. Machine check: g monotone on a dense
   grid; g(1/2) == kappa to machine precision. Hence Q <= kappa for every
   density.

3. SHARP VALUE [machine-checked]. The model's actual supremum is smaller:
   Q(q) is monotone increasing with sup_q Q = pi/(3 ln 2) = 1.51099
   approached as q -> 1/2 (where omega_0 -> pi and
   Gamma -> -3 ln(1/2) = 3 ln 2). Checked on a dense grid up to
   q = 0.4999; Q(0.08) = 0.6614 at the working density.

4. PHYSICAL FORM. Per full oscillation period the in-state excitation
   retains at most exp(-2 pi / Q) <= exp(-2 pi * 3 ln 2 / pi) = 2^{-6}
   = 1/64 of its visibility, at EVERY density: the quasiparticle never
   survives one period. This is the global statement behind
   "an isotropic cone forces overdamping".

Output: results/theory_q_global.json
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from pca3d.models.coincube import annealed_u

T0 = time.time()
KAPPA = np.pi / (2 * np.log(2))
Q_SUP = np.pi / (3 * np.log(2))


def node_q(q):
    lam = np.linalg.eigvals(annealed_u(np.zeros(3), q))
    ph = np.angle(lam)
    om0 = np.sort(ph[ph > 1e-12])[0]
    gam = -3 * np.log((1 - q) ** 2 + q ** 2)
    return om0, gam


qs = np.linspace(0.002, 0.4999, 800)
Qv, viol_tri, viol_g = [], 0, 0
gprev = 0.0
for q in qs:
    om0, gam = node_q(q)
    th = np.arctan(q / (1 - q))
    if om0 > 6 * th + 1e-12:
        viol_tri += 1
    g = 2 * th / (-np.log((1 - q) ** 2 + q ** 2))
    if g < gprev - 1e-12:
        viol_g += 1
    gprev = g
    Qv.append(om0 / gam)
Qv = np.array(Qv)

assert viol_tri == 0, viol_tri
print(f"1. omega_0 <= 6 theta(q): holds at {len(qs)}/{len(qs)} densities")
assert viol_g == 0, viol_g
g_half = 2 * np.arctan(1.0) / np.log(2)
assert abs(g_half - KAPPA) < 1e-14
print(f"2. g(q) monotone; g(1/2) = {g_half:.10f} = kappa "
      f"({KAPPA:.10f}) -> Q <= kappa at every density")
mono = np.all(np.diff(Qv) > -1e-12)
assert mono
assert Qv.max() < Q_SUP
assert Q_SUP - Qv.max() < 2e-3
print(f"3. Q(q) monotone increasing; max on grid {Qv.max():.6f} < "
      f"sup = pi/(3 ln 2) = {Q_SUP:.6f} (gap {Q_SUP-Qv.max():.1e} at "
      f"q = {qs[-1]})")
om008, gam008 = node_q(0.08)
q008 = om008 / gam008
print(f"   Q(0.08) = {q008:.4f}")
per_period = np.exp(-2 * np.pi / Q_SUP)
assert abs(per_period - 2.0 ** -6) < 1e-12
print(f"4. per-period visibility <= exp(-2 pi / sup Q) = 2^-6 = 1/64 "
      f"({per_period:.6f}) at every density")

out = {"kappa": KAPPA, "Q_sup": Q_SUP, "Q_grid_max": float(Qv.max()),
       "Q_q008": float(q008), "per_period_bound": float(per_period),
       "grid_points": len(qs), "elapsed_s": time.time() - T0}
json.dump(out, open("results/theory_q_global.json", "w"), indent=1)
print(f"\n[ALL ASSERTIONS PASSED]  ({out['elapsed_s']:.0f}s)  "
      "-> results/theory_q_global.json")
