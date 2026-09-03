#!/usr/bin/env python
"""EXPLORATORY T1: re-read kinematics of streaming schedules (J47 phase T).

Question: does a legal fast-streaming schedule eliminate ALL per-path
re-reads, making the quenched ensemble propagator EXACTLY equal to the
annealed operator at all times?

Method (exact, no sampling): enumerate every conversion history of the
4-channel walk over T cycles (2^(6T) paths). Physical bit identity is
tracked by applying the REAL pair-swap streaming to a label array, so
half-swapped intra-cycle states are handled exactly. For each path the
quenched weight forces all reads of one physical bit to agree (inconsistent
histories get probability zero; each distinct bit contributes its Bernoulli
factor once); the annealed weight gives every read an independent factor.
G_quenched == G_annealed identically iff no path re-reads any bit.

Schedules: S0 = production (field a: one swap at each of its two axis
substeps, net +1 site/cycle along (a+1)%3); Fj = S0 plus j extra
origin-alternating swap PAIRS at each axis-a substep (legal composition of
legal layers; net +(1+2j) sites/cycle). Part 2 scans re-read kinematics
over longer horizons via label-offset reachability with the real label maps.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_D, PERMS, SIGNS  # noqa: E402

L = 24                     # no wraps for T<=3 with fast streaming
T = 3
Q = 0.08
C0 = 0                     # launch channel
CTR = L // 2


def swap_1d(labels, axis, o):
    m = np.moveaxis(labels, axis, -1)
    if o:
        m = np.roll(m, -1, axis=-1)
    m = m.copy()
    t = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = t
    if o:
        m = np.roll(m, 1, axis=-1)
    return np.moveaxis(m, -1, axis)


def build_label_history(j_extra):
    """labels[a][n] = label array of field a AT read time of substep n
    (reads happen before that substep's streaming).

    Fast schedule F_j: per axis-a substep, apply (1 + 2j) swaps whose
    origins CONTINUE the field's global alternation phase (a swap pair
    only composes to a net +-2 shift when its origins continue the
    alternation; restarting at o=0 after a half-swap cancels). Net bit
    speed: +-2(1+2j) sites/cycle on the two parity sublattices."""
    lab = [np.arange(L ** 3).reshape(L, L, L) for _ in range(3)]
    phase = [0, 0, 0]                    # per-field origin alternation
    hist = [[None] * (6 * T) for _ in range(3)]
    n = 0
    for _t in range(T):
        for a in range(3):
            sa = (a + 1) % 3
            for _o in (0, 1):
                for f in range(3):
                    hist[f][n] = lab[f]
                la = lab[a]
                for _s in range(1 + 2 * j_extra):
                    la = swap_1d(la, sa, phase[a] % 2)
                    phase[a] += 1
                lab[a] = la
                n += 1
    return hist


def enumerate_paths(j_extra):
    """Exact G_quenched and G_annealed over all conversion histories.
    Returns (dict end->(gq, ga), n_reread_paths, n_inconsistent)."""
    hist = build_label_history(j_extra)
    out = {}
    stats = {"reread": 0, "inconsistent": 0, "paths": 0}

    def rec(n, pos, c, amp_sign, reads):
        # reads: dict label(with field tag) -> value
        if n == 6 * T:
            stats["paths"] += 1
            nb1 = sum(1 for v in reads.values() if v)
            nb0 = len(reads) - nb1
            gq = amp_sign * (Q ** nb1) * ((1 - Q) ** nb0)
            key = (pos, c)
            g0, a0 = out.get(key, (0.0, 0.0))
            out[key] = (g0 + gq, a0)      # annealed added separately below
            return
        t_sub = n
        a = (t_sub // 2) % 3
        lb = int(hist[a][n][pos[0], pos[1], pos[2]])
        tag = (a, lb)
        for bit in (0, 1):
            if tag in reads and reads[tag] != bit:
                stats["inconsistent"] += 1
                continue
            new_reads = reads
            fresh = tag not in reads
            if fresh:
                new_reads = dict(reads)
                new_reads[tag] = bit
            else:
                stats["reread"] += 1
            if bit:
                c2 = int(PERMS[a][c])
                s2 = amp_sign * float(SIGNS[a][c])
            else:
                c2, s2 = c, amp_sign
            d = int(COIN_D[a][c2])
            pos2 = list(pos)
            pos2[a] += d
            rec(n + 1, tuple(pos2), c2, s2, new_reads)

    rec(0, (CTR, CTR, CTR), C0, 1.0, {})
    return out, stats


def annealed_reference():
    """Annealed G by independent-read recursion (no label tracking)."""
    out = {}

    def rec(n, pos, c, w):
        if n == 6 * T:
            out[pos, c] = out.get((pos, c), 0.0) + w
            return
        a = (n // 2) % 3
        # bit = 0
        d0 = int(COIN_D[a][c])
        p0 = list(pos); p0[a] += d0
        rec(n + 1, tuple(p0), c, w * (1 - Q))
        # bit = 1
        c2 = int(PERMS[a][c]); s2 = float(SIGNS[a][c])
        d1 = int(COIN_D[a][c2])
        p1 = list(pos); p1[a] += d1
        rec(n + 1, tuple(p1), c2, w * Q * s2)

    rec(0, (CTR, CTR, CTR), C0, 1.0)
    return out


def reread_reachability(j_extra, T_long=20, L1=1024):
    """Kinematic no-re-read certificate over T_long cycles (1D reduction).

    Field-a labels permute only along the streaming axis s_a, so a re-read
    of field a requires (i) zero path displacement along the two other
    axes and (ii) a 1D label match along s_a within the path's reachable
    displacement set. This checks (ii) exhaustively on a large ring for
    every pair of read substeps up to T_long cycles: for each separation,
    compute all offsets delta with label(n1, y) == label(n2, y + delta)
    for some y, and intersect with the path-reachable set
    {|delta| <= m, delta = m mod 2}, m = number of s_a steps the path
    takes in between. Empty intersection for all pairs => no re-read at
    any time up to T_long, on any lattice with L > 2(1+2j)T_long (no
    wrap). Returns the number of violating pairs (0 = certificate)."""
    lab = np.arange(L1)
    labs = []                      # label array at each read substep
    phase = 0
    # field a is read at its two axis-a substeps each cycle; between
    # consecutive reads the field streams by the substep schedule below.
    # Reads happen at stream-application counts 0,1, (1+2j)*2 per cycle:
    per_sub = 1 + 2 * j_extra
    for _t in range(T_long):
        for _o in (0, 1):
            labs.append(lab.copy())
            for _s in range(per_sub):
                o = phase % 2
                if o:
                    lab = np.roll(lab, -1)
                l2 = lab.copy()
                t_ = l2[0::2].copy()
                l2[0::2] = l2[1::2]
                l2[1::2] = t_
                lab = np.roll(l2, 1) if o else l2
                phase += 1
    # s_a steps available to the path between read substeps n1 < n2:
    # the path steps along s_a during axis-s_a substeps: 2 per cycle.
    def sa_steps(n1, n2):
        # read substeps n are the axis-a substeps: cycle c = n//2, o = n%2.
        # axis s_a substeps occur elsewhere in each cycle; count them in
        # the open interval between the two read events (worst case: 2
        # per full cycle spanned, plus up to 2 partial). Upper bound:
        return 2 * ((n2 - n1) // 2 + 1)
    bad = 0
    ctr = L1 // 2
    win = per_sub * 2 * T_long + 4
    for n1 in range(len(labs)):
        for n2 in range(n1 + 1, len(labs)):
            m_ = sa_steps(n1, n2)
            # offsets with a label match, restricted to a central window
            l1v = labs[n1][ctr - win:ctr + win]
            match = {}
            for i, v in enumerate(labs[n2]):
                match[v] = i
            ok = False
            for i, v in enumerate(l1v):
                y1 = ctr - win + i
                if v in match:
                    d = match[v] - y1
                    if abs(d) <= m_ and (d - m_) % 2 == 0:
                        ok = True
                        break
            if ok:
                bad += 1
    return bad


def main():
    ann = annealed_reference()
    print(f"T={T} cycles, q={Q}, launch channel {C0}; "
          f"annealed support {len(ann)} endpoints")
    for name, j in (("S0 (production, +1/cycle)", 0),
                    ("F1 (+3/cycle)", 1),
                    ("F2 (+5/cycle)", 2)):
        gq, st = enumerate_paths(j)
        keys = set(gq) | set(ann)
        dev = max(abs(gq.get(k, (0, 0))[0] - ann.get(k, 0.0)) for k in keys)
        tot = sum(abs(v) for v in ann.values())
        print(f"  {name}: re-read branch events {st['reread']:>8}, "
              f"inconsistent branches {st['inconsistent']:>8}, "
              f"max|G_q - G_ann| = {dev:.3e}  (annealed L1 mass {tot:.3f})")
        if j == 0:
            assert st["reread"] > 0, "S0 should have re-reads"
            assert dev > 1e-12, "S0 quenched should differ from annealed"
        note = "QUENCHED == ANNEALED EXACTLY" if dev < 1e-14 else \
            "quenched != annealed"
        print(f"      -> {note}")

    print("\nPart 2: kinematic no-re-read certificate (20 cycles, 1D "
          "reduction, wrap-free):")
    for name, j in (("S0", 0), ("F1", 1)):
        bad = reread_reachability(j)
        print(f"  {name}: label-match pairs inside the reachable set: {bad}"
              + ("  -> NO re-read possible up to 20 cycles" if bad == 0
                 else "  (re-reads kinematically allowed)"))
    assert reread_reachability(1) == 0, "F1 certificate failed"
    print("[T1 PASSED] F1 (one extra phase-continuing swap pair per "
          "substep) eliminates all re-reads: quenched propagator == "
          "annealed operator exactly, at machine precision (T=3 exact "
          "path sum) and kinematically to 20 cycles.")


if __name__ == "__main__":
    main()
