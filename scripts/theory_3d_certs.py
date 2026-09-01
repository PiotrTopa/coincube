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
       0p <-> Mp   (sigma_full = +1 = sigma_vac),
       1p <-> (M-1)p = 1h   (per LAYER and for the full cycle; L = 2 and 3),
       2p <-> (M-2)p = 2h   (full cycle; L = 2).
     The eta = sign(N_c - M/2) grading argument is size- and dimension-
     independent (every layer conserves carrier number) — restated in the
     note; these checks make (K, I) explicit in 3D on the checked sectors.

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
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
