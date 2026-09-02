#!/usr/bin/env python
"""Paper hardening: the J31/J33 certificates in genuine 3D geometry.

Full Fock space at 3D is unreachable (2^(4L^3+...)); but for FIXED classical
env every sector evolution of the lifted coincube cycle is a SIGNED
PERMUTATION, so the 0/1/2-carrier sectors and their particle-hole duals are
exactly computable at L = 2 and L = 3 per axis.

Checks (all exact, integer signs):
  1. composite coherence in 3D (J31 lift):
     0p: vacuum -> vacuum with sign +1;
     1p: the lifted cycle == the instrument rule (signed C_a at env_a sites,
         then coin-steered shift, per sub-step) — convention alignment;
     2p: the lifted cycle == Lambda^2 of the 1p signed permutation — the
         genuine certificate: conversion JW strings (spectator between the
         Givens pair) + translation crossing/wrap parities compose coherently
         across all three axes, both origins, and periodic wraps.
  2. E1 in 3D (J33 lift) via particle-hole duality:
     P = Majorana string over the 4L^3 carrier modes (segregated gauge);
     P^2 = +1 (M = 32, 108: M(M-1)/2 even);
     [S, P] = 0 verified on the sector pairs the physical construction uses:
       0p <-> Mp   (sigma_full = +1 = sigma_vac; also at L = 4),
       1p <-> (M-1)p = 1h   (per LAYER and for the full cycle; L = 2, 3, 4),
       2p <-> (M-2)p = 2h   (full cycle; L = 2 and 3).
     The eta = sign(N_c - M/2) grading argument is size- and dimension-
     independent (every layer conserves carrier number) — restated in the
     note; these checks make (K, I) explicit in 3D on the checked sectors.
  3. ALL-sector, ALL-size coverage is not inferred from these samples: it is
     PROVEN per layer, block-locally, in scripts/theory_sp_proof.py (string-
     factorization lemma + finite block checks + the translation sign
     formula).  The sector runs here are the independent belt-and-braces:
     0/1/2/3-carrier sectors and PH duals at L = 2, 3 and 1p/2p/1h at L = 4,
     with >= 4 env draws at each of q = 0.15 and q = 0.3 for the extended
     checks (bitmask engine, cross-validated against the tuple engine and
     the instrument before use).

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_3d_certs.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_C, COIN_D, PERMS, SIGNS  # noqa: E402

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_3d_certs.json"

PERM = [np.array(p) for p in PERMS]
SGN = [np.array(s) for s in SIGNS]
DD = [np.array(d) for d in COIN_D]

# involution channel pairs of each C_a, with the signed-swap entries
PAIRS = []
for a in range(3):
    seen = set()
    pr = []
    for c in range(4):
        c2 = int(PERM[a][c])
        if c not in seen:
            pr.append((min(c, c2), max(c, c2)))
            seen |= {c, c2}
    PAIRS.append(pr)


class Lattice:
    def __init__(self, L):
        self.L = L
        self.NS = L**3
        self.M = 4 * self.NS

    def site(self, x, y, z):
        L = self.L
        return (x % L) + L * ((y % L) + L * (z % L))

    def coords(self, s):
        L = self.L
        return s % L, (s // L) % L, s // (L * L)

    def mode(self, s, c):
        return 4 * s + c

    def shift_mode(self, m, a):
        """image of carrier mode m under the axis-a coin-steered shift"""
        s, c = m // 4, m % 4
        x, y, z = self.coords(s)
        d = int(DD[a][c])
        if a == 0:
            x += d
        elif a == 1:
            y += d
        else:
            z += d
        return self.mode(self.site(x, y, z), c)


def stream_env(lat, field, ax, origin):
    """axis-a env streams along `ax`: even L = full pair swap; odd L =
    single plane transposition (o, o+1) (the i2 substrate convention)."""
    L = lat.L
    f = field.reshape(L, L, L)
    if L % 2 == 0:
        idxA = (np.arange(0, L, 2) + origin) % L
        idxB = (idxA + 1) % L
    else:
        idxA = np.array([origin % L])
        idxB = np.array([(origin + 1) % L])
    sl = [slice(None)] * 3
    slA = list(sl)
    slB = list(sl)
    slA[ax] = idxA
    slB[ax] = idxB
    tmp = f[tuple(slA)].copy()
    f[tuple(slA)] = f[tuple(slB)]
    f[tuple(slB)] = tmp
    return f.reshape(-1)


# ----------------------------------------------------------------------------
# sector layer rules (signed permutations on occupied-mode tuples)
# ----------------------------------------------------------------------------

def conv_state(lat, occ, sign, a, env_a):
    """L2 layer on a sorted occupied tuple: env-controlled signed pair swaps
    with the spectator JW string (-1)^{# occupied strictly between}."""
    occ = list(occ)
    for s in range(lat.NS):
        if not env_a[s]:
            continue
        for (c1, c2) in PAIRS[a]:
            m1, m2 = lat.mode(s, c1), lat.mode(s, c2)
            o1, o2 = m1 in occ, m2 in occ
            if o1 == o2:
                continue                       # both (Pauli, +1) or none
            src, dst = (m1, m2) if o1 else (m2, m1)
            csrc = c1 if o1 else c2
            btw = sum(1 for m in occ if m1 < m < m2 and m != src)
            sign *= int(SGN[a][csrc]) * (-1 if btw % 2 else 1)
            occ[occ.index(src)] = dst
            occ.sort()
    return tuple(occ), sign


def shift_state(lat, occ, sign, a):
    """L1 layer: species-conditioned translation; sign = inversion parity of
    the image list (the stored CA crossing/wrap rule)."""
    img = [lat.shift_mode(m, a) for m in occ]
    inv = 0
    for i in range(len(img)):
        for j in range(i + 1, len(img)):
            if img[i] > img[j]:
                inv += 1
    return tuple(sorted(img)), sign * (-1 if inv % 2 else 1)


def cycle_sector(lat, states, env0):
    """evolve every basis state of a sector through one full cycle
    (both origins per axis); env fields evolve deterministically alongside.
    Returns (perm indices, signs, env_out)."""
    index = {st: i for i, st in enumerate(states)}
    env = [f.copy() for f in env0]
    out_p = np.empty(len(states), dtype=np.int64)
    out_s = np.empty(len(states), dtype=np.int64)
    for i, st in enumerate(states):
        occ, sg = st, 1
        env_run = [f.copy() for f in env0]
        for a in range(3):
            for sub in range(2):
                occ, sg = conv_state(lat, occ, sg, a, env_run[a])
                occ, sg = shift_state(lat, occ, sg, a)
                env_run[a] = stream_env(lat, env_run[a], (a + 1) % 3,
                                        (2 * a + sub) % 2)
        out_p[i] = index[occ]
        out_s[i] = sg
        if i == 0:
            env = env_run
    return out_p, out_s, env


def layer_sequence(lat, env0):
    """the cycle as an explicit list of (kind, a, env-snapshot) layers"""
    seq = []
    env_run = [f.copy() for f in env0]
    for a in range(3):
        for sub in range(2):
            seq.append(("conv", a, env_run[a].copy()))
            seq.append(("shift", a, None))
            env_run[a] = stream_env(lat, env_run[a], (a + 1) % 3,
                                    (2 * a + sub) % 2)
    return seq


def apply_layer(lat, occ, sign, layer):
    kind, a, env_a = layer
    if kind == "conv":
        return conv_state(lat, occ, sign, a, env_a)
    return shift_state(lat, occ, sign, a)


# ----------------------------------------------------------------------------
# instrument reference (1p) and the Majorana-string PH lift on small sectors
# ----------------------------------------------------------------------------

def instrument_1p(lat, env0):
    """signed permutation of the instrument rule: per sub-step, signed C_a at
    env_a sites, then the coin-steered shift; env streams alongside."""
    M = lat.M
    perm = np.arange(M)
    sign = np.ones(M, dtype=np.int64)
    env_run = [f.copy() for f in env0]
    for a in range(3):
        for sub in range(2):
            newp = perm.copy()
            news = sign.copy()
            for j in range(M):
                m = perm[j]
                s, c = m // 4, m % 4
                if env_run[a][s]:
                    news[j] = sign[j] * int(SGN[a][c])
                    m = lat.mode(s, int(PERM[a][c]))
                newp[j] = lat.shift_mode(m, a)
            perm, sign = newp, news
            env_run[a] = stream_env(lat, env_run[a], (a + 1) % 3,
                                    (2 * a + sub) % 2)
    return perm, sign


def majorana_sign(occ, M):
    """P = prod_{i=0}^{M-1} (a_i + a_i^dag) applied in increasing i (the E1
    convention): returns the sign; the image is the complement of occ."""
    state = set(occ)
    sg = 1
    for i in range(M):
        below = sum(1 for m in state if m < i)
        if below % 2:
            sg = -sg
        if i in state:
            state.remove(i)
        else:
            state.add(i)
    return sg


# ----------------------------------------------------------------------------

def check_lattice(lat, rng, ntrial, do_2h):
    res = {"L": lat.L, "M": lat.M}
    M = lat.M
    q = 0.3

    states_1p = [(m,) for m in range(M)]
    states_2p = list(itertools.combinations(range(M), 2))
    states_1h = [tuple(m for m in range(M) if m != h) for h in range(M)]
    full = tuple(range(M))

    # PH matrices between dual sectors
    sP1 = np.array([majorana_sign((m,), M) for m in range(M)])
    p2idx = {st: i for i, st in enumerate(states_2p)}

    dev_1p = dev_2p = 0
    dual1_ok = dual2_ok = layer_ok = True
    sig_full_all = []
    for trial in range(ntrial):
        env0 = [(rng.random(lat.NS) < q).astype(np.int8) for _ in range(3)]

        # --- 1: composite coherence ---
        # 0p
        occ, sg = (), 1
        for lay in layer_sequence(lat, env0):
            occ, sg = apply_layer(lat, occ, sg, lay)
        assert occ == () and sg == 1
        # 1p vs instrument
        p1, s1, _ = cycle_sector(lat, states_1p, env0)
        ip, isg = instrument_1p(lat, env0)
        dev_1p = max(dev_1p, int(np.abs(p1 - ip).max()),
                     int(np.abs(s1 - isg).max()))
        # 2p vs Lambda^2(1p)
        p2, s2, _ = cycle_sector(lat, states_2p, env0)
        for idx, (i, j) in enumerate(states_2p):
            a1, a2 = int(p1[i]), int(p1[j])
            flip = -1 if a1 > a2 else 1
            pred = p2idx[(min(a1, a2), max(a1, a2))]
            preds = int(s1[i]) * int(s1[j]) * flip
            if p2[idx] != pred or s2[idx] != preds:
                dev_2p += 1

        # --- 2: PH duality ---
        # 0p <-> Mp: sigma_full must equal sigma_vac = +1
        occ, sg = full, 1
        for lay in layer_sequence(lat, env0):
            occ, sg = apply_layer(lat, occ, sg, lay)
        assert occ == full
        sig_full_all.append(int(sg))

        # 1p <-> 1h, per layer and full cycle
        for lay in layer_sequence(lat, env0):
            for m in range(M):
                o1, g1 = apply_layer(lat, (m,), 1, lay)         # 1p side
                oh, gh = apply_layer(lat, states_1h[m], 1, lay)  # 1h side
                mp = o1[0]
                hp = [h for h in range(M) if h not in oh][0]
                # duality: hole image must be mp with the P-sign relation
                if hp != mp or gh * sP1[m] != sP1[mp] * g1:
                    layer_ok = False
        ph, sh_, _ = cycle_sector(lat, states_1h, env0)
        # states_1h[m] indexes the hole at m; its image is the state with
        # hole at index found via the sector permutation
        hole_of = {st: h for h, st in enumerate(states_1h)}
        for m in range(M):
            himg = int(ph[m])            # index into states_1h list order
            if himg != int(p1[m]) or int(sh_[m]) * sP1[m] != sP1[int(p1[m])] * int(s1[m]):
                dual1_ok = False

        # 2p <-> 2h (optional, heavier)
        if do_2h:
            sP2 = np.array([majorana_sign(st, M) for st in states_2p])
            states_2h = [tuple(m for m in range(M) if m not in st)
                         for st in states_2p]
            p2h, s2h, _ = cycle_sector(lat, states_2h, env0)
            for idx in range(len(states_2p)):
                if (int(p2h[idx]) != int(p2[idx])
                        or int(s2h[idx]) * int(sP2[idx])
                        != int(sP2[int(p2[idx])]) * int(s2[idx])):
                    dual2_ok = False

    # P^2 = +1 on the 1p sector (via the M-1 sector)
    p2sq = []
    for m in range(M):
        s1_ = majorana_sign((m,), M)
        s2_ = majorana_sign(states_1h[m], M)
        p2sq.append(s1_ * s2_)
    res["P_squared_1p"] = sorted(set(p2sq))

    res["trials"] = ntrial
    res["dev_1p_vs_instrument"] = int(dev_1p)
    res["n_2p_mismatch_vs_wedge"] = int(dev_2p)
    res["n_2p_states"] = len(states_2p)
    res["sigma_full"] = sig_full_all
    res["duality_1h_per_layer"] = bool(layer_ok)
    res["duality_1h_cycle"] = bool(dual1_ok)
    if do_2h:
        res["duality_2h_cycle"] = bool(dual2_ok)
    return res


# ----------------------------------------------------------------------------
# fast bitmask sector engine (extended coverage; cross-validated against the
# tuple engine and the instrument below before use)
# ----------------------------------------------------------------------------

def conv_ops_of(lat, env_a, a):
    """precompute the env=1 signed-pair-swap ops of one conversion layer:
    (m1, m2, sgn_if_src_m1, sgn_if_src_m2, between-bits mask)"""
    ops = []
    for s in range(lat.NS):
        if not env_a[s]:
            continue
        for (c1, c2) in PAIRS[a]:
            m1, m2 = lat.mode(s, c1), lat.mode(s, c2)
            btwm = ((1 << m2) - 1) & ~((1 << (m1 + 1)) - 1)
            ops.append((m1, m2, int(SGN[a][c1]), int(SGN[a][c2]), btwm))
    return ops


def conv_mask(occ, sign, ops):
    """L2 on an occupancy bitmask: identical rule to conv_state."""
    for (m1, m2, s1_, s2_, btwm) in ops:
        o1 = (occ >> m1) & 1
        if o1 == ((occ >> m2) & 1):
            continue
        b = bin(occ & btwm).count("1")
        sign *= (s1_ if o1 else s2_) * (-1 if b & 1 else 1)
        occ ^= (1 << m1) | (1 << m2)
    return occ, sign


def shift_mask(smap, occ, sign, M):
    """L1 on an occupancy bitmask: identical rule to shift_state."""
    img = [int(smap[m]) for m in range(M) if (occ >> m) & 1]
    n = len(img)
    if n > 24:               # hole sectors: O(n log n) parity via argsort
        arr = np.array(img)
        order = np.argsort(arr, kind="stable")
        par = 0
        seen = np.zeros(n, dtype=bool)
        for i in range(n):
            if seen[i]:
                continue
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = int(order[j])
                ln += 1
            par ^= (ln - 1) & 1
        inv = par
    else:
        inv = 0
        for i in range(n):
            vi = img[i]
            for j in range(i + 1, n):
                if vi > img[j]:
                    inv += 1
        inv &= 1
    occ2 = 0
    for m in img:
        occ2 |= 1 << m
    return occ2, sign * (-1 if inv else 1)


def mask_layer_seq(lat, env0, smaps):
    seq = []
    env_run = [f.copy() for f in env0]
    for a in range(3):
        for sub in range(2):
            seq.append(("conv", conv_ops_of(lat, env_run[a], a)))
            seq.append(("shift", smaps[a]))
            env_run[a] = stream_env(lat, env_run[a], (a + 1) % 3,
                                    (2 * a + sub) % 2)
    return seq


def mask_cycle_one(lat, seq, occ0):
    occ, sg = occ0, 1
    M = lat.M
    for kind, arg in seq:
        if kind == "conv":
            occ, sg = conv_mask(occ, sg, arg)
        else:
            occ, sg = shift_mask(arg, occ, sg, M)
    return occ, sg


def majorana_sign_mask(occ, M):
    """majorana_sign for a bitmask state (same convention)."""
    sg = 1
    state = occ
    for i in range(M):
        if bin(state & ((1 << i) - 1)).count("1") & 1:
            sg = -sg
        state ^= (1 << i)
    return sg


def smaps_of(lat):
    return [np.array([lat.shift_mode(m, a) for m in range(lat.M)])
            for a in range(3)]


def wedge_predict(modes, p1, s1):
    """Lambda^n prediction: image modes, product sign x inversion parity."""
    img = [int(p1[m]) for m in modes]
    sg = 1
    for m in modes:
        sg *= int(s1[m])
    inv = sum(1 for i in range(len(img)) for j in range(i + 1, len(img))
              if img[i] > img[j])
    return tuple(sorted(img)), sg * (-1 if inv % 2 else 1)


def check_extended(rng, ndraw=4, qs=(0.15, 0.3)):
    """Extended coverage (belt and braces on top of the per-layer proof of
    scripts/theory_sp_proof.py): 3-carrier sectors at L = 2 and 3; the
    M-2 sector at L = 3 via PH duality to 2 holes; 1p/2p (+ 1h duality and
    sigma_full) at L = 4; ndraw env draws at each q in qs."""
    res = {}

    # engine cross-validation at L = 2, one draw: bitmask engine == tuple
    # engine on 1p and 2p (perm and sign, state by state)
    lat = Lattice(2)
    env0 = [(rng.random(lat.NS) < 0.3).astype(np.int8) for _ in range(3)]
    smaps = smaps_of(lat)
    seq = mask_layer_seq(lat, env0, smaps)
    states_1p = [(m,) for m in range(lat.M)]
    states_2p = list(itertools.combinations(range(lat.M), 2))
    p1, s1, _ = cycle_sector(lat, states_1p, env0)
    p2, s2, _ = cycle_sector(lat, states_2p, env0)
    dev = 0
    for m in range(lat.M):
        occ, sg = mask_cycle_one(lat, seq, 1 << m)
        if occ != (1 << int(p1[m])) or sg != int(s1[m]):
            dev += 1
    for idx, (i, j) in enumerate(states_2p):
        occ, sg = mask_cycle_one(lat, seq, (1 << i) | (1 << j))
        st = states_2p[int(p2[idx])]
        if occ != ((1 << st[0]) | (1 << st[1])) or sg != int(s2[idx]):
            dev += 1
    # majorana_sign_mask == majorana_sign
    for st in states_2p[:50]:
        if majorana_sign_mask((1 << st[0]) | (1 << st[1]), lat.M) != \
                majorana_sign(st, lat.M):
            dev += 1
    res["engine_crossval_mismatches"] = int(dev)

    for L in (2, 3, 4):
        lat = Lattice(L)
        M = lat.M
        smaps = smaps_of(lat)
        entry = {"M": M, "draws_per_q": ndraw, "qs": list(qs)}
        n3_mis = n2_mis = n1_mis = dual1_mis = dual2_mis = 0
        sig_full = []
        n3_states = n2_states = 0
        for q in qs:
            for _ in range(ndraw):
                env0 = [(rng.random(lat.NS) < q).astype(np.int8)
                        for _ in range(3)]
                seq = mask_layer_seq(lat, env0, smaps)
                # 1p: bitmask engine vs the instrument (exact reference)
                ip, isg = instrument_1p(lat, env0)
                p1v = np.empty(M, dtype=np.int64)
                s1v = np.empty(M, dtype=np.int64)
                for m in range(M):
                    occ, sg = mask_cycle_one(lat, seq, 1 << m)
                    p1v[m] = occ.bit_length() - 1
                    s1v[m] = sg
                n1_mis += int((p1v != ip).sum() + (s1v != isg).sum())

                if L <= 3:
                    # 3p wedge certificate: full sector
                    for st in itertools.combinations(range(M), 3):
                        occ0 = (1 << st[0]) | (1 << st[1]) | (1 << st[2])
                        occ, sg = mask_cycle_one(lat, seq, occ0)
                        pmodes, psg = wedge_predict(st, p1v, s1v)
                        pocc = 0
                        for m in pmodes:
                            pocc |= 1 << m
                        if occ != pocc or sg != psg:
                            n3_mis += 1
                        n3_states += 1
                if L >= 3:
                    # 2p wedge certificate (needed for the 2h duality too)
                    st2 = list(itertools.combinations(range(M), 2))
                    p2v = {}
                    s2v = {}
                    for st in st2:
                        occ0 = (1 << st[0]) | (1 << st[1])
                        occ, sg = mask_cycle_one(lat, seq, occ0)
                        pmodes, psg = wedge_predict(st, p1v, s1v)
                        pocc = (1 << pmodes[0]) | (1 << pmodes[1])
                        if occ != pocc or sg != psg:
                            n2_mis += 1
                        n2_states += 1
                        p2v[st] = pmodes
                        s2v[st] = sg
                    if L == 3:
                        # M-2 sector via PH duality to 2 holes:
                        # S(complement of st) must be complement of S(st)
                        # with the P-sign relation
                        full = (1 << M) - 1
                        for st in st2:
                            occ0 = full ^ ((1 << st[0]) | (1 << st[1]))
                            occ, sg = mask_cycle_one(lat, seq, occ0)
                            img = p2v[st]
                            pocc = full ^ ((1 << img[0]) | (1 << img[1]))
                            sP_in = majorana_sign_mask(
                                (1 << st[0]) | (1 << st[1]), M)
                            sP_out = majorana_sign_mask(
                                (1 << img[0]) | (1 << img[1]), M)
                            if occ != pocc or sg * sP_in != sP_out * s2v[st]:
                                dual2_mis += 1
                if L == 4:
                    # 1h duality + sigma_full
                    full = (1 << M) - 1
                    for m in range(M):
                        occ0 = full ^ (1 << m)
                        occ, sg = mask_cycle_one(lat, seq, occ0)
                        mp = int(p1v[m])
                        pocc = full ^ (1 << mp)
                        sP_in = majorana_sign_mask(1 << m, M)
                        sP_out = majorana_sign_mask(1 << mp, M)
                        if occ != pocc or sg * sP_in != sP_out * int(s1v[m]):
                            dual1_mis += 1
                    occ, sg = mask_cycle_one(lat, seq, full)
                    assert occ == full
                    sig_full.append(int(sg))
        entry["n_1p_mismatch_vs_instrument"] = int(n1_mis)
        if L <= 3:
            entry["n_3p_mismatch_vs_wedge"] = int(n3_mis)
            entry["n_3p_states_checked"] = int(n3_states)
        if L >= 3:
            entry["n_2p_mismatch_vs_wedge"] = int(n2_mis)
            entry["n_2p_states_checked"] = int(n2_states)
        if L == 3:
            entry["n_Mminus2_duality_mismatch"] = int(dual2_mis)
        if L == 4:
            entry["n_1h_duality_mismatch"] = int(dual1_mis)
            entry["sigma_full"] = sig_full
        res[f"L{L}"] = entry
    return res


def mutation_controls(rng):
    """the checks must FAIL under wrong sign conventions (teeth check):
    (a) dropping the conversion spectator string (L = 2);
    (b) dropping the translation inversion parity (L = 3 — at L = 2 the two
        sub-step shifts per axis make net shift crossings even, so this
        mutation is invisible there; L = 3's 3-cycles expose it)."""
    def conv_nostring(lat, occ, sign, a, env_a):
        occ = list(occ)
        for s in range(lat.NS):
            if not env_a[s]:
                continue
            for (c1, c2) in PAIRS[a]:
                m1, m2 = lat.mode(s, c1), lat.mode(s, c2)
                o1, o2 = m1 in occ, m2 in occ
                if o1 == o2:
                    continue
                src, dst = (m1, m2) if o1 else (m2, m1)
                sign *= int(SGN[a][c1 if o1 else c2])
                occ[occ.index(src)] = dst
                occ.sort()
        return tuple(occ), sign

    def shift_nopar(lat, occ, sign, a):
        return tuple(sorted(lat.shift_mode(m, a) for m in occ)), sign

    out = {}
    for name, L, cf, sf in (("no_string", 2, conv_nostring, shift_state),
                            ("no_shift_parity", 3, conv_state, shift_nopar)):
        lat = Lattice(L)
        M = lat.M
        env0 = [(rng.random(lat.NS) < 0.4).astype(np.int8) for _ in range(3)]
        states_2p = list(itertools.combinations(range(M), 2))
        p1, s1, _ = cycle_sector(lat, [(m,) for m in range(M)], env0)
        mis = 0
        for (i, j) in states_2p:
            occ, sg = (i, j), 1
            env_run = [f.copy() for f in env0]
            for a in range(3):
                for sub in range(2):
                    occ, sg = cf(lat, occ, sg, a, env_run[a])
                    occ, sg = sf(lat, occ, sg, a)
                    env_run[a] = stream_env(lat, env_run[a], (a + 1) % 3,
                                            (2 * a + sub) % 2)
            a1, a2 = int(p1[i]), int(p1[j])
            flip = -1 if a1 > a2 else 1
            if (occ != (min(a1, a2), max(a1, a2))
                    or sg != int(s1[i]) * int(s1[j]) * flip):
                mis += 1
        out[name] = dict(L=L, mismatches=int(mis), n=len(states_2p))
    return out


def main():
    out = {"seed": SEED}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out["L2"] = check_lattice(Lattice(2), rng, ntrial=4, do_2h=True)
    print(f"[{time.time() - t0:6.1f}s] L = 2 done")
    out["L3"] = check_lattice(Lattice(3), rng, ntrial=2, do_2h=False)
    print(f"[{time.time() - t0:6.1f}s] L = 3 done")
    out["mutation_controls"] = mutation_controls(rng)
    print(f"[{time.time() - t0:6.1f}s] mutation controls done")
    out["extended"] = check_extended(rng, ndraw=4, qs=(0.15, 0.3))
    print(f"[{time.time() - t0:6.1f}s] extended coverage done "
          "(3p at L=2,3; M-2 at L=3; 1p/2p/1h at L=4)")


    for key in ("L2", "L3"):
        r = out[key]
        assert r["dev_1p_vs_instrument"] == 0
        assert r["n_2p_mismatch_vs_wedge"] == 0
        assert all(s == 1 for s in r["sigma_full"])
        assert r["duality_1h_per_layer"] and r["duality_1h_cycle"]
        assert r["P_squared_1p"] == [1]
    assert out["L2"]["duality_2h_cycle"]
    # the checks must have teeth: wrong conventions fail
    assert out["mutation_controls"]["no_string"]["mismatches"] > 0
    assert out["mutation_controls"]["no_shift_parity"]["mismatches"] > 0
    # extended coverage (4 draws at each of q = 0.15, 0.3 per check)
    ext = out["extended"]
    assert ext["engine_crossval_mismatches"] == 0
    for L in (2, 3, 4):
        assert ext[f"L{L}"]["n_1p_mismatch_vs_instrument"] == 0
    assert ext["L2"]["n_3p_mismatch_vs_wedge"] == 0
    assert ext["L3"]["n_3p_mismatch_vs_wedge"] == 0
    assert ext["L3"]["n_2p_mismatch_vs_wedge"] == 0
    assert ext["L3"]["n_Mminus2_duality_mismatch"] == 0
    assert ext["L4"]["n_2p_mismatch_vs_wedge"] == 0
    assert ext["L4"]["n_1h_duality_mismatch"] == 0
    assert all(s == 1 for s in ext["L4"]["sigma_full"])
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
