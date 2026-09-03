#!/usr/bin/env python
"""T2/TASK 1+3: the fresh-tape theorem for F1 streaming (all-T proof) + rebuild
certificates.  Companion of scripts/x_reread_kinematics.py (T1 instrument).

THEOREM (fresh tape, g = 0).  Coincube, single-particle sector, product
Bernoulli(q) vacuum, F1 streaming (per axis-a substep the axis-a env field
receives 3 pair-swap layers whose origins CONTINUE the field's global
alternation phase).  Then on the infinite lattice NO conversion history reads
the same physical environment bit twice, at ANY time; on the L^3 torus
(L even) the same holds for every evolution of T <= ceil(L/8) cycles, and
this horizon is SHARP (the first wrap re-read joins two reads separated by
Dn* = 2*ceil(L/8) field batches, i.e. ceil(L/8) cycles).

Proof structure, every step machine-checked below:

  (a) 1D reduction.  A pair-swap layer of the axis-a field permutes bits
      only along its streaming axis s_a = (a+1) mod 3 (checked: the layer
      acts within 1D fibers).  A bit's identity is (transverse coords;
      1D label along s_a).  The path reads at its own site, so a re-read of
      field a at substeps tau1 < tau2 requires (i) zero net path displacement
      along BOTH transverse axes over [tau1, tau2) and (ii) the path's s_a
      displacement to equal the bit's s_a displacement.

  (b) Monotone transport (LEMMA 1 + LEMMA 2).
      L1: a single pair-swap layer with origin o moves the bit at site y to
          y+1 if y = o (mod 2), else to y-1  [4 finite cases].
      L2: under any ORIGIN-ALTERNATING swap sequence, eps = (-1)^(y + phase)
          is conserved, so every bit moves eps * 1 per swap FOREVER: after
          k swaps the displacement is EXACTLY eps*k, direction fixed at t=0.
          Phase continuity across substeps is exactly what keeps the global
          sequence alternating (F1 batches have odd length 3, so a restarted
          origin would repeat and cancel: the J48 negative control below
          shows the broken schedule has |speed| < 3/batch).
      Under F1 the axis-a field receives 3 swaps per axis-a substep, all
      phase-continuing: bit displacement is EXACTLY 3*eps per batch
      (+-6 sites/cycle).  The carrier moves +-1 per substep of the active
      axis only: at most 1 per s_a-substep.

  (c) Slot counting (LEMMA 3).  In the cycle the slots seen by field a are
      the periodic word  A A S S X X  (A = axis-a substep: read, then batch;
      S = axis-s_a substep: carrier +-1 along s_a; X = third axis).  For any
      two A-slots with Dn = #A in [tau1, tau2) (= batches applied between
      the reads, since reads precede their own slot's batch), the number of
      S-slots in between is m <= Dn + 1; the number of A-slots in between is
      Dn (each forcing a +-1 carrier move along axis a, so transverse return
      along a requires Dn even, in which case m = Dn and the X-count is even).

  (d) The theorem.  A re-read needs |path s_a displacement| = |bit
      displacement| = 3*Dn with the path bounded by m <= Dn + 1:
      3*Dn > Dn + 1 for every Dn >= 1.  No re-read, at any separation,
      on the infinite lattice.  (No residual window: the exhaustive T=3
      path enumeration below re-verifies the FULL dynamics -- labels through
      the real permutations, signs, weights -- for every separation
      Dn <= 5, i.e. all separations up to 2 cycles and more, catching any
      modeling error in the reduction.)

  (e) Torus horizon (sharp).  On the ring of size L the match condition
      becomes eps*3*Dn = Dy (mod L) with |Dy| <= m, Dy = m (mod 2), and
      Dn even (transverse-a return; transverse wraps need Dn >= L, beyond
      every horizon considered).  With m = Dn the first solution is
      Dn* = min{even n >= L/4} = 2*ceil(L/8)  (Dy = 3Dn - L, parity
      automatic for even L).  Checked by ring certificate at
      L = 12, 16, 24, 32 and EXHIBITED by full path enumeration at L = 16:
      T = 2 clean, T = 3 = ceil(16/8) + 1 has its first re-read at exactly
      Dn = 4 batches.

T1 hardening: the exact path-sum equality G_quenched == G_annealed is
asserted for launch channels 0..3 at T = 3 (recursion) and for channel 0 at
T = 4 (vectorized over all 2^24 conversion histories, wrap-free).

TASK 3 certificates (rebuild):
  - F1's extra swaps are the SAME certified legal layer (layer_env with a
    chosen origin): asserted bit-for-bit.  layer_env is env-supported and
    parity-even, so every F1 layer commutes with the P string by the
    string-factorization lemma, case (E), of scripts/theory_sp_proof.py --
    composition of certified layers, nothing new to certify.
  - even-L constraint unchanged (pair-swap streaming): assert raise.
  - the annealed operator is schedule-independent: annealed_u never sees the
    env schedule; the annealed path sum here is checked against U(k)^T.

Assertions run BEFORE the results file is written.
Results -> results/freshtape_proof.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/freshtape_proof.py
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import (  # noqa: E402
    COIN_D, PERMS, SIGNS, annealed_u, evolve_field_cc, layer_env)

Q = 0.08
RESULTS = {}


# ---------------------------------------------------------------- primitives

def swap_1d(labels, axis, o):
    """One pair-swap layer along `axis` with origin o (label transport)."""
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


def swap1(lab, o):
    """1D version of swap_1d."""
    return swap_1d(lab, 0, o)


def build_tapes(L, T, j):
    """3D label histories under F_j: hist[a][n] = labels at read time of
    substep n (reads precede the slot's batch).  Phase-continuing origins."""
    lab = [np.arange(L ** 3).reshape(L, L, L) for _ in range(3)]
    phase = [0, 0, 0]
    hist = [[None] * (6 * T) for _ in range(3)]
    n = 0
    for _t in range(T):
        for a in range(3):
            sa = (a + 1) % 3
            for _o in (0, 1):
                for f in range(3):
                    hist[f][n] = lab[f]
                la = lab[a]
                for _s in range(1 + 2 * j):
                    la = swap_1d(la, sa, phase[a] % 2)
                    phase[a] += 1
                lab[a] = la
                n += 1
    return hist


# ------------------------------------------- exact path enumeration (small T)

def enumerate_exact(L, T, j, c0, q):
    """Exact quenched path sum with physical-bit tracking (backtracking).

    Returns (out, stats): out[(dx,dy,dz,c)] = G_quenched; stats counts
    re-read branch events, inconsistent branches, and the minimal re-read
    separation in field-batch units (None if no re-read)."""
    hist = build_tapes(L, T, j)
    ctr = L // 2
    out = {}
    st = {"reread": 0, "inconsistent": 0, "minsep": None, "paths": 0}
    reads = {}          # (field, label) -> (bit, field_slot_index)
    qp = [q ** i for i in range(6 * T + 1)]
    q1 = [(1 - q) ** i for i in range(6 * T + 1)]
    perms = [list(map(int, p)) for p in PERMS]
    signs = [list(map(int, s)) for s in SIGNS]
    dvec = [list(map(int, d)) for d in COIN_D]

    def rec(n, x, y, z, c, s, nb1, nb0):
        if n == 6 * T:
            st["paths"] += 1
            k = (x - ctr, y - ctr, z - ctr, c)
            out[k] = out.get(k, 0.0) + s * qp[nb1] * q1[nb0]
            return
        a = (n // 2) % 3
        ridx = 2 * (n // 6) + (n % 2)
        tag = (a, int(hist[a][n][x % L, y % L, z % L]))
        prev = reads.get(tag)
        for bit in (0, 1):
            if prev is not None:
                if prev[0] != bit:
                    st["inconsistent"] += 1
                    continue
                st["reread"] += 1
                sep = ridx - prev[1]
                if st["minsep"] is None or sep < st["minsep"]:
                    st["minsep"] = sep
                fresh = False
            else:
                fresh = True
                reads[tag] = (bit, ridx)
            if bit:
                c2 = perms[a][c]
                s2 = s * signs[a][c]
                b1, b0 = nb1 + fresh, nb0
            else:
                c2, s2 = c, s
                b1, b0 = nb1, nb0 + fresh
            d = dvec[a][c2]
            if a == 0:
                rec(n + 1, x + d, y, z, c2, s2, b1, b0)
            elif a == 1:
                rec(n + 1, x, y + d, z, c2, s2, b1, b0)
            else:
                rec(n + 1, x, y, z + d, c2, s2, b1, b0)
            if fresh:
                del reads[tag]

    rec(0, ctr, ctr, ctr, c0, 1.0, 0, 0)
    return out, st


def annealed_paths(T, q, c0):
    """Annealed path sum (independent reads), relative coordinates."""
    out = {}
    perms = [list(map(int, p)) for p in PERMS]
    signs = [list(map(int, s)) for s in SIGNS]
    dvec = [list(map(int, d)) for d in COIN_D]

    def rec(n, x, y, z, c, w):
        if n == 6 * T:
            k = (x, y, z, c)
            out[k] = out.get(k, 0.0) + w
            return
        a = (n // 2) % 3
        for bit in (0, 1):
            if bit:
                c2 = perms[a][c]
                w2 = w * q * signs[a][c]
            else:
                c2, w2 = c, w * (1 - q)
            d = dvec[a][c2]
            if a == 0:
                rec(n + 1, x + d, y, z, c2, w2)
            elif a == 1:
                rec(n + 1, x, y + d, z, c2, w2)
            else:
                rec(n + 1, x, y, z + d, c2, w2)

    rec(0, 0, 0, 0, c0, 1.0)
    return out


def dict_dev(g1, g2):
    keys = set(g1) | set(g2)
    return max(abs(g1.get(k, 0.0) - g2.get(k, 0.0)) for k in keys)


# ----------------------------------------------------- vectorized T=4 (2^24)

def vec_quenched(T, c0, q):
    """All 2^(6T) conversion histories under F1, vectorized; wrap-free
    (ring 512 for tapes, raw integer positions).  Returns (grid, S, off,
    n_reread_path_pairs)."""
    nsub = 6 * T
    N = 1 << nsub
    R = 512
    # 1D tapes per field at own-slot read times (phase-continuing F1)
    taps = []
    for _a in range(3):
        lab = np.arange(R, dtype=np.int16)
        snaps = []
        phase = 0
        for _t in range(T):
            for _o in (0, 1):
                snaps.append(lab.copy())
                for _s in range(3):
                    lab = swap1(lab, phase % 2)
                    phase += 1
        taps.append(snaps)
    idx = np.arange(N, dtype=np.int64)
    c = np.full(N, c0, np.int8)
    sign = np.ones(N, np.int8)
    pos = np.zeros((3, N), np.int16)
    cnt = np.zeros(N, np.uint8)
    keys = [[] for _ in range(3)]
    reread = np.zeros(N, bool)
    perms = [np.asarray(p, np.int8) for p in PERMS]
    signs = [np.asarray(s, np.int8) for s in SIGNS]
    dvec = [np.asarray(d, np.int16) for d in COIN_D]
    for n in range(nsub):
        a = (n // 2) % 3
        sa, ua = (a + 1) % 3, (a + 2) % 3
        ridx = 2 * (n // 6) + (n % 2)
        lab = taps[a][ridx][(pos[sa] + R // 2)]
        key = (lab.astype(np.int32) << 12) \
            | ((pos[a].astype(np.int32) + 32) << 6) \
            | (pos[ua].astype(np.int32) + 32)
        for k0 in keys[a]:
            reread |= key == k0
        keys[a].append(key)
        b = ((idx >> n) & 1).astype(bool)
        sign = np.where(b, sign * signs[a][c], sign)
        c = np.where(b, perms[a][c], c)
        pos[a] += dvec[a][c]
        cnt += b
    n_re = int(reread.sum())
    qp = q ** np.arange(nsub + 1)
    q1 = (1 - q) ** np.arange(nsub + 1)
    amp = sign.astype(np.float64) * qp[cnt] * q1[nsub - cnt]
    S = 4 * T + 5
    off = 2 * T + 2
    ek = ((c.astype(np.int64) * S + pos[0] + off) * S
          + pos[1] + off) * S + pos[2] + off
    grid = np.bincount(ek, weights=amp,
                       minlength=4 * S ** 3).reshape(4, S, S, S)
    return grid, S, off, n_re


def annealed_dp(T, c0, q, S, off):
    """Annealed evolution as an exact lattice DP (deterministic operator)."""
    g = np.zeros((4, S, S, S))
    g[c0, off, off, off] = 1.0
    for _t in range(T):
        for a in range(3):
            for _o in (0, 1):
                conv = np.empty_like(g)
                for cc in range(4):
                    conv[int(PERMS[a][cc])] = float(SIGNS[a][cc]) * g[cc]
                g = (1 - q) * g + q * conv
                for cc in range(4):
                    g[cc] = np.roll(g[cc], int(COIN_D[a][cc]), axis=a)
    return g


# ----------------------------------------------------------- torus tightness

def ring_first_wrap(L, Tmax=10):
    """First kinematically valid wrap re-read on the ring of size L (F1).

    Scans all read-slot pairs up to Tmax cycles; a pair is a valid re-read
    iff Dn is even (transverse-a return), and some bit displacement
    (verified from the REAL tape, both eps classes) equals a path-reachable
    s_a displacement mod L.  Returns (min Dn, witness pair) or (None, None).
    """
    lab = np.arange(L)
    snaps = []
    phase = 0
    for _t in range(Tmax):
        for _o in (0, 1):
            snaps.append(lab.copy())
            for _s in range(3):
                lab = swap1(lab, phase % 2)
                phase += 1
    # slot type word per cycle for one field: A A S S X X
    word = ["A", "A", "S", "S", "X", "X"] * Tmax
    apos = [i for i, w in enumerate(word) if w == "A"]
    best, wit = None, None
    for i1 in range(len(apos)):
        for i2 in range(i1 + 1, len(apos)):
            dn = i2 - i1
            t1, t2 = apos[i1], apos[i2]
            m = sum(1 for k in range(t1, t2) if word[k] == "S")
            n_a = sum(1 for k in range(t1, t2) if word[k] == "A")
            n_x = sum(1 for k in range(t1, t2) if word[k] == "X")
            assert n_a == dn
            if dn % 2 == 1:
                continue           # transverse-a return impossible (no wrap)
            assert m == dn and n_x % 2 == 0     # Lemma 3, even case
            # real bit displacements between the two snapshots (mod L)
            p1 = np.argsort(snaps[i1])          # label -> position
            p2 = np.argsort(snaps[i2])
            disp = set((p2 - p1) % L)
            assert disp <= {(3 * dn) % L, (-3 * dn) % L}, disp
            hit = any((dy - d) % L == 0
                      for d in disp for dy in range(-m, m + 1, 2))
            if hit and (best is None or dn < best):
                best, wit = dn, (i1, i2)
    return best, wit


# ------------------------------------------------------------------- driver

def main():
    t0 = time.time()
    print("== T2/TASK 1: fresh-tape theorem, machine-checked lemmas ==\n")

    # ---- LEMMA 1: single-swap displacement table (4 finite cases)
    R = 64
    lab = np.arange(R)
    for o in (0, 1):
        after = swap1(lab, o)
        posn = np.argsort(after)               # label -> new position
        for y in range(2, R - 2):
            expect = y + 1 if y % 2 == o else y - 1
            assert posn[y] == expect, (o, y)
    print("L1  single swap: y -> y+1 iff y = o (mod 2), else y-1   [OK]")

    # ---- LEMMA 2: fixed direction under phase-continuing alternation,
    #      including across odd-length (3-swap) batch boundaries
    R = 4096
    K = 300
    for phase0 in (0, 1):
        lab = np.arange(R)
        for k in range(1, K + 1):
            lab = swap1(lab, (phase0 + k - 1) % 2)
            posn = np.argsort(lab)
            ys = np.arange(K + 2, R - K - 2)
            eps = np.where(ys % 2 == phase0, 1, -1)
            assert np.array_equal(posn[ys], ys + eps * k), (phase0, k)
    print("L2  alternating origins: displacement = eps*k exactly, eps fixed"
          " by (y0 + phase0) mod 2   [OK]")

    # negative control (the J48 failure): restarting each 3-swap batch at
    # o = 0 repeats an origin at the boundary and cancels the extra pair
    lab = np.arange(R)
    for _b in range(8):                        # 8 batches, restart each
        for s in range(3):
            lab = swap1(lab, s % 2)
    posn = np.argsort(lab)
    ys = np.arange(40, R - 40)
    dis = np.abs(posn[ys] - ys)
    broken_speed = float(dis.max()) / 8.0
    assert broken_speed == 0.0, broken_speed
    print(f"L2' negative control (phase restart): |speed| = {broken_speed}"
          " per 3-swap batch == 0 exactly (extra pair cancels)   [OK]")

    # ---- LEMMA 3: slot counting on the real substep schedule
    Tm = 12
    word = ["A", "A", "S", "S", "X", "X"] * Tm
    apos = [i for i, w in enumerate(word) if w == "A"]
    for i1 in range(len(apos)):
        for i2 in range(i1 + 1, len(apos)):
            dn = i2 - i1
            m = sum(1 for k in range(apos[i1], apos[i2]) if word[k] == "S")
            n_x = sum(1 for k in range(apos[i1], apos[i2]) if word[k] == "X")
            assert m <= dn + 1
            assert 3 * dn > m                  # the theorem inequality
            if dn % 2 == 0:
                assert m == dn and n_x % 2 == 0
    print("L3  slot count: m <= Dn+1 (m = Dn, X even when Dn even); "
          "3*Dn > m for all Dn >= 1  ->  no re-read at ANY separation  [OK]")

    # the same word describes all three fields (cyclic shifts of AASSXX):
    # axis order per cycle is 0,0,1,1,2,2 and s_a = a+1, so field a sees
    # A at its own slots, S two slots later (cyclically) -- identical word.

    # ---- T1 hardening: exact equality, all launch channels, T = 3
    print("\n== T1 hardening: exact path-sum equality ==")
    L, T = 24, 3
    devs = {}
    for c0 in range(4):
        gq, st = enumerate_exact(L, T, 1, c0, Q)
        ga = annealed_paths(T, Q, c0)
        dev = dict_dev(gq, ga)
        devs[c0] = dev
        assert st["reread"] == 0 and st["inconsistent"] == 0, st
        assert dev < 1e-14, (c0, dev)
        print(f"  F1 T=3 channel {c0}: re-reads 0, "
              f"max|G_q - G_ann| = {dev:.3e}   [EXACT]")
    # S0 control (channel 0): re-reads present, quenched != annealed
    gq0, st0 = enumerate_exact(L, T, 0, 0, Q)
    dev0 = dict_dev(gq0, annealed_paths(T, Q, 0))
    assert st0["reread"] > 0 and dev0 > 1e-12
    print(f"  S0 control: {st0['reread']} re-read branches, "
          f"dev = {dev0:.3e}, min separation {st0['minsep']} batches")

    # ---- T = 4, channel 0, vectorized over all 2^24 histories, wrap-free
    tv = time.time()
    g3v, S3, o3, re3 = vec_quenched(3, 0, Q)
    gq3, _ = enumerate_exact(24, 3, 1, 0, Q)
    grid3 = np.zeros_like(g3v)
    for (dx, dy, dz, c), v in gq3.items():
        grid3[c, dx + o3, dy + o3, dz + o3] += v
    xdev = float(np.abs(g3v - grid3).max())
    assert re3 == 0 and xdev < 1e-14, (re3, xdev)
    print(f"  vectorized engine cross-check at T=3: dev vs recursion = "
          f"{xdev:.1e}   [OK]")
    g4, S4, o4, re4 = vec_quenched(4, 0, Q)
    a4 = annealed_dp(4, 0, Q, S4, o4)
    dev4 = float(np.abs(g4 - a4).max())
    l1_4 = float(np.abs(a4).sum())
    assert re4 == 0, re4
    assert dev4 < 1e-13, dev4
    print(f"  F1 T=4 channel 0 (2^24 histories, {time.time()-tv:.0f}s): "
          f"re-read paths 0, max|G_q - G_ann| = {dev4:.3e} "
          f"(annealed L1 mass {l1_4:.3f})   [EXACT]")

    # ---- torus horizon: sharpness
    print("\n== torus horizon: Dn* = 2*ceil(L/8), T_safe = ceil(L/8) ==")
    torus = {}
    for Lr in (12, 16, 24, 32):
        pred = 2 * math.ceil(Lr / 8)
        dn, wit = ring_first_wrap(Lr)
        torus[Lr] = dn
        assert dn == pred, (Lr, dn, pred)
        print(f"  L={Lr}: first valid wrap re-read at Dn = {dn} batches "
              f"(= {dn//2} cycles; predicted {pred})   [TIGHT]")
    # full-dynamics exhibit at L = 16: T = 2 clean, T = 3 wraps at Dn = 4
    gq, st = enumerate_exact(16, 2, 1, 0, Q)
    assert st["reread"] == 0, st
    gq, st = enumerate_exact(16, 3, 1, 0, Q)
    devw = dict_dev(gq, annealed_paths(3, Q, 0))
    assert st["reread"] > 0 and st["minsep"] == 4, st
    print(f"  L=16 full enumeration: T=2 re-reads 0; T=3 re-read branches "
          f"{st['reread']}, first at Dn = {st['minsep']} = 2*ceil(16/8), "
          f"quenched-annealed dev {devw:.2e}   [SHARP]")

    # ---- TASK 3 certificates
    print("\n== TASK 3: certificates for the rebuild ==")
    # (1) F1 layers are the certified legal layer type: every extra swap is
    #     bit-for-bit a layer_env application (env-supported, parity-even ->
    #     [S,P] = 0 by theory_sp_proof string-factorization lemma, case E).
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 2, size=(8, 8, 8)).astype(bool)
    for a in range(3):
        sa = (a + 1) % 3
        for o in (0, 1):
            le = layer_env(arr, a, o)
            sw = swap_1d(arr.astype(np.int64), sa, o).astype(bool)
            assert np.array_equal(le, sw), (a, o)
    print("  F1 swaps == layer_env (certified legal layer; theory_sp_proof "
          "case E covers any composition)   [OK]")
    # (2) even-L constraint unchanged
    try:
        evolve_field_cc(5, 1, 1, Q)
        raise AssertionError("odd L accepted")
    except ValueError:
        pass
    print("  even-L constraint unchanged (odd L raises)   [OK]")
    # (3) annealed operator schedule-independent: the annealed path sum
    #     (which never sees a schedule) matches U(k)^T at random k
    ga = annealed_paths(3, Q, 0)
    kdev = 0.0
    for kv in (np.array([np.pi, 0, 0]), rng.uniform(-np.pi, np.pi, 3),
               rng.uniform(-np.pi, np.pi, 3)):
        u3 = np.linalg.matrix_power(annealed_u(kv, Q), 3)
        ghat = np.zeros(4, complex)
        for (dx, dy, dz, c), v in ga.items():
            ghat[c] += v * np.exp(1j * (kv[0] * dx + kv[1] * dy
                                        + kv[2] * dz))
        kdev = max(kdev, float(np.abs(ghat - u3[:, 0]).max()))
    assert kdev < 1e-12, kdev
    print(f"  annealed path sum == U(k)^3 (schedule-independent), "
          f"dev {kdev:.1e}   [OK]")

    RESULTS.update({
        "q": Q,
        "lemma_broken_speed_per_batch": broken_speed,
        "t3_dev_by_channel": {str(k): v for k, v in devs.items()},
        "s0_control": {"reread": st0["reread"], "dev": dev0,
                       "minsep_batches": st0["minsep"]},
        "t4_channel0": {"reread_paths": re4, "dev": dev4,
                        "annealed_l1": l1_4, "engine_crosscheck_dev": xdev},
        "torus_first_wrap_dn": torus,
        "torus_exhibit_L16": {"T2_rereads": 0, "T3_rereads": st["reread"],
                              "T3_minsep_batches": st["minsep"],
                              "T3_dev": devw},
        "annealed_uk_dev": kdev,
        "elapsed_s": time.time() - t0,
    })
    out = Path(__file__).resolve().parent.parent / "results" / \
        "freshtape_proof.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n[T2 TASK 1+3 PASSED] fresh-tape theorem machine-checked at "
          f"every step; T=3 all channels + T=4 exact; torus horizon "
          f"T_safe = ceil(L/8) sharp.  ({time.time()-t0:.0f}s)  -> {out}")


if __name__ == "__main__":
    main()
