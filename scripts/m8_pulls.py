#!/usr/bin/env python
"""M8 pull check: the measured massive dispersion against the EXACT operator
branches (not the leading-order sqrt(m^2 + v^2 k^2) form), from the committed
instrument output results/m8_corner.json.

The exact upper branch is computed from annealed_u8 at the same momenta; the
center convention matches the figure (exact-operator multiplet center at the
corner). BOTH instrument rows are reported: the annealed (gate) row's pulls
expose the estimator's own bias with its smaller error bars; the quenched
row is the physics claim. PULL_MAX is a catastrophe gate in the sense of the
paper's gate policy (it does not certify accuracy), and the nine points per
row share media blocks and the center estimate, so they are correlated --
the worst pull is reported, not a chi^2.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import annealed_u8  # noqa: E402

Q, QM = 0.08, 0.05
K0 = np.array([np.pi, 0.0, 0.0])
PULL_MAX = 2.5


def upper_and_center(kvec):
    lams = np.linalg.eigvals(annealed_u8(kvec, Q, QM))
    phs = sorted(np.angle(l) for l in lams if l.imag > 0.02)
    return phs[-1], 0.5 * (phs[0] + phs[-1])


data = json.load(open("results/m8_corner.json"))
om_c = upper_and_center(K0)[1]
worsts = {}
for mode in ("annealed", "quenched"):
    row = [r for r in data if r["mode"] == mode][0]
    worst = 0.0
    print(f"-- {mode} row --")
    print(f"{'fam':>4} {'delta':>6} {'meas':>8} {'exact':>8} {'sig':>7} "
          f"{'pull':>6}")
    for name, dv in (("100", (1, 0, 0)), ("110", (1, 1, 0)),
                     ("111", (1, 1, 1))):
        u = np.array(dv, float)
        u /= np.linalg.norm(u)
        for dstr, (val, sig) in sorted(row["rows"][name].items(),
                                       key=lambda kv: float(kv[0])):
            d = float(dstr)
            ex = upper_and_center(K0 + d * u)[0] - om_c
            pull = (val - ex) / sig
            worst = max(worst, abs(pull))
            print(f"{name:>4} {d:>6.3f} {val:>8.4f} {ex:>8.4f} {sig:>7.4f} "
                  f"{pull:>6.2f}")
    worsts[mode] = worst
    print(f"  worst |pull| ({mode}) = {worst:.2f}")
assert worsts["quenched"] < PULL_MAX, \
    f"pull check FAILED: quenched worst |pull| = {worsts['quenched']:.2f}"
assert worsts["annealed"] < PULL_MAX, \
    f"pull check FAILED: gate-row worst |pull| = {worsts['annealed']:.2f}"
print(f"[gate PASSED] both rows within {PULL_MAX} jackknife errors of the "
      "exact branches (correlated points; worst-pull statistic)")
