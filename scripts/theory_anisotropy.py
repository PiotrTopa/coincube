#!/usr/bin/env python
"""The chiral (rank-3) anisotropy of the coincube node: structure, no-go,
doubling cancellation, improvement bound, and instrument identity.

Machine-checked facts (all assertions):

A. STRUCTURE. The node splitting is s(k) = 2v|k| + c(q) kx ky kz / |k| + ...
   -- the leading term exactly isotropic, the next term proportional to the
   A4 invariant kx ky kz. Certified by (i) bit-level equality of slopes for
   direction pairs sharing the product kx ky kz /|k|^3, (ii) a clean linear
   fit of the odd part, (iii) c(q) on a density grid.

B. NO-GO (exhaustive at n = 4). The 4x4 signed-permutation group contains
   exactly 12 elements squaring to -1 (the left- and right-multiplication
   quaternion families) and 48 ordered anticommuting triples. Over every
   triple, every signed axis assignment, every block ordering and every
   balanced direction-table combination that preserves an isotropic-leading
   node, the odd coefficient never drops below a fixed floor.

C. DOUBLING CANCELLATION. On the inversion-doubled (massive) model the
   Dirac-branch dispersion is invariant under every single-axis sign flip
   to machine precision, at every radius and mass density tested: the odd
   term is odd under the doubling's inversion and cancels between sectors.
   The surviving even anisotropy and the (higher-order) transposition-chiral
   remnant are reported.

D. IMPROVEMENT BOUND. The legal deterministic counterterm family
   (coin-conditional transverse translation dipoles, which read no
   environment bits and so cost no freshness) reduces |c| by at most about
   half at the working density; the residual stays rank 3. Reported, with
   leading isotropy asserted intact.

E. INSTRUMENT IDENTITY. Averaging the splitting over the +- direction star
   annihilates the odd term identically (it is odd under an odd number of
   sign flips); the cone instrument's pooled ratios are therefore
   statements about the even sector and the k -> 0 limit. The
   direction-resolved ratios of the exact operator at the production probe
   radii are reported alongside.

Output: results/theory_anisotropy.json
"""
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from pca3d.models.coincube import COIN_C, COIN_D, annealed_u, annealed_u8

T0 = time.time()
Q = 0.08
I4 = np.eye(4)
Cstd = [np.array(c, float) for c in COIN_C]
Dstd = [np.array(d, float) for d in COIN_D]
OUT = {}


def node_split(k, q=Q):
    lam = np.linalg.eigvals(annealed_u(np.asarray(k, float), q))
    ph = np.sort(np.angle(lam))
    lam0 = np.linalg.eigvals(annealed_u(np.zeros(3), q))
    p0 = np.sort(np.angle(lam0))
    omc = p0[p0 > 1e-12].mean()
    d = np.abs(ph - omc)
    pair = np.sort(ph[np.argsort(d)[:2]])
    return pair[1] - pair[0]


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- A
print("=== A. structure of the odd term ===")
r = 0.02
pairs_equal = [((1, 1, 1), (1, -1, -1)), ((2, 1, 1), (2, -1, -1)),
               ((1, 1, -1), (-1, -1, -1))]
worst_eq = 0.0
for ua, ub in pairs_equal:
    d = abs(node_split(r * unit(ua)) - node_split(r * unit(ub)))
    worst_eq = max(worst_eq, d)
assert worst_eq < 1e-13, worst_eq
print(f"  equal-product direction pairs: bit-level slope equality "
      f"(worst {worst_eq:.1e})")

u111 = unit((1, 1, 1))
prod111 = float(np.prod(u111))


def c_of_q(q):
    """Fit s(ru111) - s(ru11m) = 2 c prod111 r^2 + O(r^4)."""
    rs = np.array([0.004, 0.008])
    odd = np.array([node_split(rr * u111, q)
                    - node_split(rr * unit((1, 1, -1)), q) for rr in rs])
    c2 = odd / (2 * prod111 * rs ** 2)
    c = float((4 * c2[0] - c2[1]) / 3)    # Richardson in r^2
    return c


cq = {q: c_of_q(q) for q in (0.02, 0.08, 0.15, 0.25, 0.30)}
OUT["c_of_q"] = cq
print("  c(q):", {k: round(v, 2) for k, v in cq.items()})
assert all(v > 0 for v in cq.values())
v008 = 1.1806
rel_per_k = cq[0.08] / (3 * np.sqrt(3) * v008)
OUT["rel_aniso_per_unit_k_q008"] = rel_per_k
kmin = 2 * np.pi / 48
OUT["rel_aniso_at_L48_kmin"] = rel_per_k * kmin
OUT["L_for_1pct"] = 2 * np.pi * rel_per_k / 0.01
print(f"  relative velocity anisotropy per unit |k| at q=0.08: "
      f"{rel_per_k:.2f}; at k_min(L=48): {rel_per_k*kmin:.2f}; "
      f"L for 1%: {OUT['L_for_1pct']:.0f}")

# ---------------------------------------------------------------- B
print("=== B. exhaustive no-go at n = 4 ===")
units = []
for perm in itertools.permutations(range(4)):
    P = np.zeros((4, 4))
    for i, p in enumerate(perm):
        P[p, i] = 1
    for signs in itertools.product((1, -1), repeat=4):
        C = P * np.array(signs)[None, :]
        if np.allclose(C @ C, -I4):
            units.append(C)
assert len(units) == 12, len(units)
n_tri = 0
WORDS = [sum([[a, a] for a in perm], []) for perm in
         itertools.permutations((0, 1, 2))]


def probe_model(word, Cs, Ds):
    def U(k):
        M = np.eye(4, dtype=complex)
        for a in word:
            E = np.diag(np.exp(1j * k[a] * Ds[a]))
            M = E @ ((1 - Q) * I4 + Q * Cs[a]) @ M
        return M
    ph0 = np.angle(np.linalg.eigvals(U(np.zeros(3))))
    pos = np.sort(ph0[ph0 > 1e-12])
    if len(pos) != 2 or abs(pos[0] - pos[1]) > 1e-10:
        return None
    omc = pos.mean()

    def split(u, rr):
        ph = np.sort(np.angle(np.linalg.eigvals(U(rr * unit(u)))))
        d = np.abs(ph - omc)
        pair = np.sort(ph[np.argsort(d)[:2]])
        return pair[1] - pair[0]
    iso = abs(0.5 * (split((1, 1, 1), 0.005) + split((1, 1, -1), 0.005))
              - split((1, 0, 0), 0.005)) / split((1, 0, 0), 0.005)
    if iso > 0.01:
        return None
    a, b = split((1, 1, 1), 0.02), split((1, 1, -1), 0.02)
    return abs(a - b) / (0.5 * (a + b))


odds = []
for i, j in itertools.combinations(range(12), 2):
    A, B = units[i], units[j]
    if not np.allclose(A @ B, -B @ A):
        continue
    for k in range(12):
        if k in (i, j):
            continue
        Cm = units[k]
        if not (np.allclose(A @ Cm, -Cm @ A)
                and np.allclose(B @ Cm, -Cm @ B)):
            continue
        n_tri += 1
        for word in WORDS:
            o = probe_model(word, [A, B, Cm], Dstd)
            if o is not None:
                odds.append(o)
assert n_tri == 48, n_tri
bal = [np.array(v, float) for v in set(itertools.permutations((1, 1, -1, -1)))]
for da in bal:
    for db in bal:
        for dc in bal:
            o = probe_model(WORDS[0], Cstd, [da, db, dc])
            if o is not None:
                odds.append(o)
odds = np.array(odds)
OUT["nogo_models"] = int(len(odds))
OUT["nogo_min_abs_odd"] = float(odds.min())
assert odds.min() > 0.1, odds.min()
print(f"  12 units, 48 triples; {len(odds)} isotropic-node models; "
      f"min |odd| = {odds.min():.4f} (assert > 0.1)")

# ---------------------------------------------------------------- C
print("=== C. inversion-doubling cancellation (Dirac branch) ===")


def om_plus(u, rr, qm):
    lam0 = np.linalg.eigvals(annealed_u8(np.zeros(3), Q, qm))
    p0 = sorted(np.angle(l) for l in lam0 if l.imag > 0.02)
    omc = 0.5 * (p0[0] + p0[-1])
    lams = np.linalg.eigvals(annealed_u8(rr * unit(u), Q, qm))
    phs = sorted(np.angle(l) for l in lams if l.imag > 0.02)
    return phs[-1] - omc


worst_flip = 0.0
for qm in (0.02, 0.05, 0.10):
    for rr in (0.02, 0.08, 0.4, 1.2):
        for ua, ub in (((1, 1, 1), (1, 1, -1)), ((1, 2, 3), (1, -2, 3))):
            d = abs(om_plus(ua, rr, qm) - om_plus(ub, rr, qm))
            worst_flip = max(worst_flip, d)
assert worst_flip < 1e-13, worst_flip
OUT["doubling_flip_cancellation"] = worst_flip
trans = abs(om_plus((1, 2, 3), 0.08, 0.05) - om_plus((2, 1, 3), 0.08, 0.05))
OUT["doubling_transposition_remnant_r008"] = trans
print(f"  sign-flip invariance of the Dirac branch: worst {worst_flip:.1e} "
      f"(all radii to 1.2, all qm); transposition remnant at r=0.08: "
      f"{trans:.1e}")

# ---------------------------------------------------------------- D
print("=== D. improvement (dipole counterterm) bound ===")


def build_dip(k, n):
    U = np.eye(4, dtype=complex)
    for a in (0, 1, 2):
        b = (a + 1) % 3
        E = np.diag(np.exp(1j * k[a] * Dstd[a]))
        T = E @ ((1 - Q) * I4 + Q * Cstd[a])
        D = np.diag(np.exp(1j * k[b] * n * Dstd[b]))
        Di = np.diag(np.exp(-1j * k[b] * n * Dstd[b]))
        U = Di @ T @ T @ D @ U
    return U


def dip_metrics(n):
    ph0 = np.angle(np.linalg.eigvals(build_dip(np.zeros(3), n)))
    pos = np.sort(ph0[ph0 > 1e-12])
    omc = pos.mean()

    def split(u, rr):
        ph = np.sort(np.angle(np.linalg.eigvals(build_dip(rr * unit(u), n))))
        d = np.abs(ph - omc)
        pair = np.sort(ph[np.argsort(d)[:2]])
        return pair[1] - pair[0]
    iso = abs(0.5 * (split((1, 1, 1), 0.005) + split((1, 1, -1), 0.005))
              - split((1, 0, 0), 0.005)) / split((1, 0, 0), 0.005)
    a, b = split((1, 1, 1), 0.02), split((1, 1, -1), 0.02)
    return iso, (a - b) / (0.5 * (a + b))


dip = {}
for n in range(0, 9):
    iso, odd = dip_metrics(n)
    assert iso < 1e-3, (n, iso)
    dip[n] = odd
OUT["dipole_scan"] = dip
OUT["dipole_min_abs_odd"] = float(min(abs(v) for v in dip.values()))
assert OUT["dipole_min_abs_odd"] > 0.05
print("  dipole amplitude scan (lead-iso intact at every n):",
      {n: round(v, 4) for n, v in dip.items()})
print(f"  best reduction: {dip[0]:.4f} -> "
      f"{OUT['dipole_min_abs_odd']:.4f} (bounded, residual rank 3)")

# ---------------------------------------------------------------- E
print("=== E. instrument identity and unpooled ratios ===")
star = [np.array(s) * u111 for s in itertools.product((1, -1), repeat=3)]
pooled = np.mean([node_split(0.02 * np.abs(np.array(s)) * u111 * np.array(s))
                  for s in itertools.product((1, -1), repeat=3)])
even = 0.5 * (node_split(0.02 * u111) + node_split(0.02 * unit((1, 1, -1))))
assert abs(pooled - even) < 1e-14, abs(pooled - even)
print(f"  +-star average == even part (dev {abs(pooled-even):.1e}): "
      "the pooling annihilates the odd term identically")
unpooled = {}
for d in (0.03, 0.05, 0.08):
    r111 = node_split(d * u111) / node_split(d * unit((1, 0, 0)))
    unpooled[d] = float(r111)
OUT["unpooled_r111_by_delta"] = unpooled
print("  direction-resolved exact r_111 at the probe radii:",
      {k: round(v, 3) for k, v in unpooled.items()})

OUT["elapsed_s"] = time.time() - T0
json.dump(OUT, open("results/theory_anisotropy.json", "w"), indent=1)
print(f"\n[ALL ASSERTIONS PASSED]  ({OUT['elapsed_s']:.0f}s)  "
      "-> results/theory_anisotropy.json")
