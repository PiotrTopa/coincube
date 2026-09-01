#!/usr/bin/env python
"""P0/A3: the EXACT correlator of a conditional-propagation rule on small rings.

No Monte Carlo anywhere. All 2**(2L) configurations of an L-site ring are evolved
simultaneously (the update is a permutation of packed integers), and the density
correlator

    C(t, x) = (1/N) sum_c  sigma_x(P^t c) sigma_0(c)

is accumulated as an exact INTEGER count: sigma sigma' = +-1, so

    N * C(t, x) = N - 2 * #{configs where the two bits disagree}

The packet centroid is then an exact rational number. If centroid(t)/t equals a fixed
small rational (e.g. 3/2) in exact arithmetic, at every computable t, on every ring
size, that is the rational-speed conjecture confirmed at machine-proof grade for this
rule -- no estimator, no noise, no fit.

Limits: light cone is 2 sites/cycle, so a ring of L sites gives ~L/8 clean cycles
before wrap-around contaminates the packet. Small-L exactness complements (does not
replace) the large-L Monte Carlo.

    .venv/bin/python scripts/exact_ring.py [L ...]
"""

from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np

from pca3d.models import conditional as C

# bit layout matches Lattice.bit_index with n_species=2:
#   site s: system bit = 2s, environment bit = 2s+1
# block b (origin 0) covers sites (2b, 2b+1): nibble bits 4b..4b+3 in exactly the
# (n_psi, n_phi, n_psi', n_phi') order of conditional.encode -- no translation needed.


def step_all(states: np.ndarray, table: np.ndarray, L: int, origin: int) -> np.ndarray:
    """Apply one block sub-step to every packed configuration at once."""
    nbits = 2 * L
    mask = (1 << nbits) - 1
    if origin:  # shift so that origin-1 blocks become origin-0 blocks
        states = ((states >> 2) | (states << (nbits - 2))) & mask
    out = np.zeros_like(states)
    for b in range(L // 2):
        nib = (states >> (4 * b)) & 15
        out |= table[nib] << (4 * b)
    if origin:  # undo the shift
        out = ((out << 2) | (out >> (nbits - 2))) & mask
    return out


def exact_correlator(perm_table: np.ndarray, L: int, n_substeps: int) -> np.ndarray:
    """Integer counts ``N*C(t, x)`` for the system channel, shape (n_substeps+1, L)."""
    nbits = 2 * L
    n = 1 << nbits
    dtype = np.uint32 if nbits <= 32 else np.uint64
    states = np.arange(n, dtype=dtype)
    table = perm_table.astype(dtype)

    b0 = ((states >> np.uint32(0)) & 1).astype(bool)  # system bit of site 0 at t=0

    counts = np.empty((n_substeps + 1, L), dtype=np.int64)
    cur = states
    for t in range(n_substeps + 1):
        if t:
            cur = step_all(cur, table, L, origin=(t - 1) % 2)
        for x in range(L):
            bx = ((cur >> np.uint32(2 * x)) & 1).astype(bool)
            disagree = int(np.count_nonzero(bx ^ b0))
            counts[t, x] = n - 2 * disagree
    return counts


def centroid_fraction(row: np.ndarray, L: int) -> Fraction:
    """Exact |x|-centroid of one time slice, over entries >= 10% of the peak."""
    x = [min(i, L - i) for i in range(L)]  # minimal-image |displacement|
    mags = [abs(int(v)) for v in row]
    peak = max(mags)
    if peak == 0:
        return Fraction(0)
    num = Fraction(0)
    den = Fraction(0)
    for xi, m in zip(x, mags):
        if 10 * m >= peak:  # same 10%-of-peak rule as the MC estimator
            num += Fraction(m) * xi
            den += Fraction(m)
    return num / den


def main() -> None:
    args = sys.argv[1:]
    rule_idx = None
    if args and args[0].startswith("rule="):
        rule_idx = int(args[0].split("=")[1]); args = args[1:]
    sizes = [int(a) for a in args] or [8, 10, 12]
    if rule_idx is None:
        perm = C.wetterich_cpa_perm(); name = "Wetterich table CPA"
    else:
        perm = C.enumerate_conditional_rules()[rule_idx]; name = f"enumerated rule {rule_idx}"

    # sanity: the sub-step really is a permutation of packed states (R2, again)
    probe = exact_correlator  # noqa: F841  (import-time guard below is the real check)
    st = np.arange(1 << 16, dtype=np.uint32)
    img = step_all(st, perm.astype(np.uint32), 8, origin=0)
    assert len(np.unique(img)) == len(st), "block sub-step is not a bijection on the ring"
    img = step_all(st, perm.astype(np.uint32), 8, origin=1)
    assert len(np.unique(img)) == len(st), "shifted sub-step is not a bijection on the ring"

    print(f"{name}, exact all-configuration correlator (integer arithmetic)")
    print("velocity per CYCLE = 2 sub-steps\n")

    for L in sizes:
        n_sub = max(2, (L // 2) - 2)  # stay well inside the wrap
        counts = exact_correlator(perm, L, n_sub)
        print(f"ring L={L}  ({1 << (2*L):,} configurations, {n_sub} sub-steps)")
        for t in range(1, n_sub + 1):
            c = centroid_fraction(counts[t], L)
            per_cycle = c / t * 2  # sub-step index -> cycles
            approx = float(per_cycle)
            tag = "  == 3/2" if per_cycle == Fraction(3, 2) else ""
            print(f"  t={t} sub-steps: centroid = {c}  -> v = {per_cycle} = {approx:.6f}{tag}")
        print()


if __name__ == "__main__":
    main()
