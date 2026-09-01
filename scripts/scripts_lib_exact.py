"""Exact q-weighted correlator machinery shared by exact-ring scripts."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from pca3d.models import conditional as C


def step_all(states, table, L, origin):
    nbits = 2 * L
    mask = (1 << nbits) - 1
    if origin:
        states = ((states >> 2) | (states << (nbits - 2))) & mask
    out = np.zeros_like(states)
    for b in range(L // 2):
        nib = (states >> (4 * b)) & 15
        out |= table[nib] << (4 * b)
    if origin:
        out = ((out << 2) | (out >> (nbits - 2))) & mask
    return out


def exact_env_weighted_centroids(perm, L, n_substeps, q: Fraction):
    """Exact |x|-centroids of the system-channel correlator at env density q.

    Weight per config: q^Ne (1-q)^(L-Ne); the system channel's 1/2 factors are common
    to all configs and cancel. All arithmetic in Fraction: results are exact.
    """
    nbits = 2 * L
    n = 1 << nbits
    dtype = np.uint32 if nbits <= 32 else np.uint64
    states = np.arange(n, dtype=dtype)
    table = perm.astype(dtype)

    env_mask = 0
    for site in range(L):
        env_mask |= 1 << (2 * site + 1)
    ne = np.zeros(n, dtype=np.int64)
    v = (states & np.int64(env_mask)).astype(np.uint64)
    while np.any(v):
        ne += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)

    # spin at site x = 2*bit - 1; correlate with site 0 at t=0
    s0 = (2 * ((states >> np.uint64(0)) & 1).astype(np.int64) - 1)

    # accumulate weighted sums per (t, x, Ne) as INTEGER counts, then attach q powers
    cur = states
    out = []
    for t in range(n_substeps + 1):
        if t:
            cur = step_all(cur, table, L, origin=(t - 1) % 2)
        counts = np.zeros((L, L + 1), dtype=np.int64)  # [x, Ne] -> sum of s0*sx
        for x in range(L):
            sx = (2 * ((cur >> np.uint64(2 * x)) & 1).astype(np.int64) - 1)
            prod = s0 * sx
            for e in range(L + 1):
                m = ne == e
                counts[x, e] = int(prod[m].sum())
        out.append(counts)

    def centroid(counts) -> Fraction:
        # C(x) = sum_e counts[x,e] q^e (1-q)^(L-e), exact
        vals = []
        for x in range(L):
            tot = Fraction(0)
            for e in range(L + 1):
                if counts[x, e]:
                    tot += counts[x, e] * (q ** e) * ((1 - q) ** (L - e))
            vals.append(tot)
        mags = [abs(vv) for vv in vals]
        peak = max(mags)
        if peak == 0:
            return Fraction(0)
        num = den = Fraction(0)
        for x, m in enumerate(mags):
            if 10 * m >= peak:
                num += m * min(x, L - x)
                den += m
        return num / den

    return [centroid(cs) for cs in out]
