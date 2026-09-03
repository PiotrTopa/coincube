#!/usr/bin/env python
"""T2/TASK 2: g > 0 write events (imprint layer L4) under fresh-tape
streaming.  Companion of scripts/freshtape_proof.py.

The imprint layer WRITES: once per cycle, at the carrier's site (parity
control is trivial in the 1p sector), it swaps the two env-species bits of
the pair selected by the 4-valued iota field, with the production
permutation-lift sign (-1 iff both swapped bits are 1: the lift sign READS
the two bits' values into the amplitude).  Writes relocate bits across
species and their values enter the weight, so freshness at g > 0 means:
no conversion read, no lift sign-read, and no iota read ever touches a bit
whose value already entered the weight.

KINEMATICS under F1 (machine-checked in scripts/freshtape_proof.py
lemmas; the counting below is asserted here by exact enumeration):

  (i)  read -> write rendezvous is IMPOSSIBLE: between a conversion read of
       field b (slot 2b+o of cycle t) and any later write (cycle end t'),
       the bit moves exactly 3*(2-o+2(t'-t)) sites along s_b while the
       carrier has at most 2+2(t'-t) s_b-substeps: 3*Dn > Dn+o' for all
       windows.  So writes only ever touch never-read bits, and the lift
       sign-reads are always fresh.
  (ii) write -> read: NOT excluded by F1 alone.  The written cells sit at
       the carrier's site and the next conversion read of each field
       happens with ZERO intervening batches of that field (reads precede
       their own slot's batch): for the field read at the next cycle's
       matching slot the carrier need not even move.  Whenever iota fires a
       pair containing field 0, the very next substep re-reads a bit whose
       value entered the lift sign => structural correlated read.  Plain F1
       therefore does NOT give the exact law beyond cycle 1 (refuted below,
       with the exact deviation).
  (iii) THE FIX (part of the model, "F1i"): one post-imprint FLUSH batch
       (3 phase-continuing pair swaps, the same certified layer_env type)
       per env field per cycle.  Every written/sign-read bit then moves
       >= 3 sites along its species' streaming axis before any field is
       read again, while the carrier window stays <= 2: all reads fresh at
       ALL times (same monotone-transport lemma).  Then per cycle the
       imprint contributes the exact scalar (1-g) + g(1-2q^2) = 1 - 2gq^2:

           U_g(k) = (1 - 2 g q^2) U(k)   EXACTLY at every cycle.

  (iv) iota and mass fields: read once per cycle at the carrier's site, so
       their bit speed must exceed the carrier's 2 sites/cycle along the
       streaming axis.  Certificates below: production mass streaming
       (1 swap/cycle, axis 0) admits re-reads at Dt = 2 cycles; production
       iota streaming (1 swap/cycle, rotating axis) at Dt = 6; the F1
       treatment = 3 phase-continuing swaps/cycle along a FIXED axis is
       fresh at all times (a rotating-axis schedule needs >= 7 swaps per
       activation; 3 and 5 fail at Dt = 6, 6 fails at Dt = 3).

EXACT ENUMERATION (T = 3 cycles): every conversion history x every iota
outcome (2^18 x 4^3 = 16.7M leaves), physical bits tracked through the real
streaming permutations AND through the write swaps (cell-content overrides);
lift sign-reads handled exactly by deferred integration (a pending pair's
unread members integrate to their Bernoulli expectation at accumulation
time; members later read bind to the read value).  Quenched G at t = 1,2,3
is compared endpoint-by-endpoint against (1-2gq^2)^t * G_annealed(t).

Assertions run BEFORE the results file is written.
Results -> results/freshtape_interaction.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/freshtape_interaction.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_D, PAIRS, PERMS, SIGNS  # noqa: E402

L, T, Q, G, C0 = 48, 3, 0.08, 0.3, 0
CTR = L // 2
L3 = L ** 3
RESULTS = {}


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


def swap1(lab, o):
    return swap_1d(lab, 0, o)


def build_env_tapes(flush):
    """envread[a][2t+o] = labels at the axis-a read of cycle t, substep o;
    envwr[a][t] = labels at write time (post substep 5, pre flush).
    F1 phase-continuing origins; optional post-imprint flush batch."""
    lab = [np.arange(L3).reshape(L, L, L) for _ in range(3)]
    phase = [0, 0, 0]
    envread = [[None] * (2 * T) for _ in range(3)]
    envwr = [[None] * T for _ in range(3)]
    for t in range(T):
        for a in range(3):
            sa = (a + 1) % 3
            for o in (0, 1):
                envread[a][2 * t + o] = lab[a].ravel()
                la = lab[a]
                for _s in range(3):
                    la = swap_1d(la, sa, phase[a] % 2)
                    phase[a] += 1
                lab[a] = la
        for f in range(3):
            envwr[f][t] = lab[f].ravel()
        if flush:
            for f in range(3):
                sf = (f + 1) % 3
                lf = lab[f]
                for _s in range(3):
                    lf = swap_1d(lf, sf, phase[f] % 2)
                    phase[f] += 1
                lab[f] = lf
    return envread, envwr


def build_iota_tape():
    """F1-iota: 1D labels along axis 0; read at cycle end, then 3
    phase-continuing swaps per cycle."""
    lab = np.arange(L)
    ph = 0
    snaps = []
    for _t in range(T):
        snaps.append(lab.copy())
        for _s in range(3):
            lab = swap1(lab, ph % 2)
            ph += 1
    return snaps


def enumerate_g(flush, g):
    """Exact quenched path sum of the interacting model.

    Returns (out, cnt): out[t] for t = 1..T maps (dx,dy,dz,c) -> G; cnt
    diagnoses every correlation channel."""
    envread, envwr = build_env_tapes(flush)
    iot = build_iota_tape()
    perms = [list(map(int, p)) for p in PERMS]
    signs = [list(map(int, s)) for s in SIGNS]
    dvec = [list(map(int, d)) for d in COIN_D]
    out = [dict() for _ in range(T + 1)]
    cnt = {"env_reread": 0, "pending_read": 0, "relocated_read": 0,
           "write_on_read": 0, "iota_reread": 0, "writes": 0}
    reads = {}       # physical env bit id -> value
    ioreads = {}     # iota bit id -> value
    content = {}     # cell id -> physical bit id (override only)
    pending = []     # (bitA, bitB) lift sign pairs
    pmem = set()     # members of pending pairs
    e2q = 1 - 2 * Q * Q
    e1q = 1 - 2 * Q

    def pending_factor():
        f = 1.0
        for ba, bb in pending:
            va = reads.get(ba)
            vb = reads.get(bb)
            if va is None and vb is None:
                f *= e2q
            elif va is None:
                if vb:
                    f *= e1q
            elif vb is None:
                if va:
                    f *= e1q
            elif va and vb:
                f = -f
        return f

    def cycle_end(n, x, y, z, c, amp):
        t = n // 6 - 1
        xm, ym, zm = x % L, y % L, z % L
        flat = (xm * L + ym) * L + zm
        ikey = (int(iot[t][xm]) * L + ym) * L + zm
        forced = ioreads.get(ikey)
        if forced is not None:
            cnt["iota_reread"] += 1
            branches = ((forced, 1.0),)
            fresh_io = False
        else:
            branches = ((0, 1 - g), (1, g / 3), (2, g / 3), (3, g / 3)) \
                if g > 0 else ((0, 1.0),)
            fresh_io = True
        okey = (x - CTR, y - CTR, z - CTR, c)
        for p, w in branches:
            if fresh_io:
                ioreads[ikey] = p
            a2 = amp * w
            if p == 0:
                o = out[t + 1]
                o[okey] = o.get(okey, 0.0) + a2 * pending_factor()
                if n < 6 * T:
                    body(n, x, y, z, c, a2)
            else:
                cnt["writes"] += 1
                ea, eb = PAIRS[p - 1]
                ca = ea * L3 + int(envwr[ea][t][flat])
                cb = eb * L3 + int(envwr[eb][t][flat])
                ba = content.get(ca, ca)
                bb = content.get(cb, cb)
                if ba in reads or bb in reads:
                    cnt["write_on_read"] += 1
                if ba in pmem or bb in pmem:
                    raise AssertionError(
                        "pending collision: shared unknown sign bit")
                pending.append((ba, bb))
                pmem.add(ba)
                pmem.add(bb)
                olda = content.get(ca)
                oldb = content.get(cb)
                content[ca] = bb
                content[cb] = ba
                o = out[t + 1]
                o[okey] = o.get(okey, 0.0) + a2 * pending_factor()
                if n < 6 * T:
                    body(n, x, y, z, c, a2)
                if olda is None:
                    del content[ca]
                else:
                    content[ca] = olda
                if oldb is None:
                    del content[cb]
                else:
                    content[cb] = oldb
                pending.pop()
                pmem.discard(ba)
                pmem.discard(bb)
            if fresh_io:
                del ioreads[ikey]

    def body(n, x, y, z, c, amp):
        a = (n // 2) % 3
        ridx = 2 * (n // 6) + (n % 2)
        flat = ((x % L) * L + (y % L)) * L + (z % L)
        cell = a * L3 + int(envread[a][ridx][flat])
        bit = content.get(cell, cell)
        if bit != cell:
            cnt["relocated_read"] += 1
        prev = reads.get(bit)
        if prev is None:
            vals = (0, 1)
            fresh = True
            if bit in pmem:
                cnt["pending_read"] += 1
        else:
            vals = (prev,)
            fresh = False
            cnt["env_reread"] += 1
        n2 = n + 1
        atend = n2 % 6 == 0
        for v in vals:
            if fresh:
                reads[bit] = v
                a2 = amp * (Q if v else 1 - Q)
            else:
                a2 = amp
            if v:
                c2 = perms[a][c]
                a2 = a2 * signs[a][c]
            else:
                c2 = c
            d = dvec[a][c2]
            x2, y2, z2 = x, y, z
            if a == 0:
                x2 += d
            elif a == 1:
                y2 += d
            else:
                z2 += d
            if atend:
                cycle_end(n2, x2, y2, z2, c2, a2)
            else:
                body(n2, x2, y2, z2, c2, a2)
            if fresh:
                del reads[bit]

    body(0, CTR, CTR, CTR, C0, 1.0)
    return out, cnt


def annealed_percycle():
    """g = 0 annealed path sum, accumulated at every cycle end."""
    out = [dict() for _ in range(T + 1)]
    perms = [list(map(int, p)) for p in PERMS]
    signs = [list(map(int, s)) for s in SIGNS]
    dvec = [list(map(int, d)) for d in COIN_D]

    def rec(n, x, y, z, c, w):
        if n % 6 == 0 and n > 0:
            o = out[n // 6]
            k = (x, y, z, c)
            o[k] = o.get(k, 0.0) + w
            if n == 6 * T:
                return
        a = (n // 2) % 3
        for v in (0, 1):
            if v:
                c2 = perms[a][c]
                w2 = w * Q * signs[a][c]
            else:
                c2, w2 = c, w * (1 - Q)
            d = dvec[a][c2]
            if a == 0:
                rec(n + 1, x + d, y, z, c2, w2)
            elif a == 1:
                rec(n + 1, x, y + d, z, c2, w2)
            else:
                rec(n + 1, x, y, z + d, c2, w2)

    rec(0, 0, 0, 0, C0, 1.0)
    return out


def devs_vs_law(out, ann, law):
    """max |G_quenched(t) - law^t G_ann(t)| per cycle, plus scale."""
    res = []
    for t in range(1, T + 1):
        keys = set(out[t]) | set(ann[t])
        d = max(abs(out[t].get(k, 0.0) - law ** t * ann[t].get(k, 0.0))
                for k in keys)
        scale = max(abs(v) for v in ann[t].values())
        res.append((d, scale))
    return res


# --------------------- once-per-cycle-read fields (mass, iota): speed certs

def cyclefield_first_reread(k, rotating, Tmax=60):
    """First Dt admitting a kinematic re-read for a field read once per
    cycle at the carrier's site, streamed with k phase-continuing pair
    swaps per activation (fixed axis 0, or rotating axis t mod 3).

    Bit displacements are taken from REAL tape simulations (both parity
    classes move symmetrically, asserted); the carrier reach along each
    axis over Dt cycles is {|d| <= 2 Dt, d even}.  Returns Dt or None."""
    R = 2048
    tapes = [np.arange(R) for _ in range(3)]
    phases = [0, 0, 0]
    # per-axis displacement of the two parity classes after each cycle end
    disp = [[0], [0], [0]]      # class eps=+1 label displacement, cumulative
    probe = [R // 2, R // 2 + 1]
    pos = [[list(probe)], [list(probe)], [list(probe)]]
    cur = [list(probe) for _ in range(3)]
    for t in range(Tmax):
        axes = (t % 3,) if rotating else (0,)
        for j in range(3):
            if j in axes:
                la = tapes[j]
                for _s in range(k):
                    la = swap1(la, phases[j] % 2)
                    phases[j] += 1
                tapes[j] = la
                posn = np.argsort(la)
                cur[j] = [int(posn[probe[0]]), int(posn[probe[1]])]
            pos[j].append(list(cur[j]))
    for j in range(3):
        for s in range(Tmax + 1):
            d0 = pos[j][s][0] - probe[0]
            d1 = pos[j][s][1] - probe[1]
            assert d0 == -d1, (j, s)           # symmetric parity classes
            disp[j].append(abs(d0))
    for dt in range(1, Tmax):
        for s1 in range(Tmax - dt):
            ok = True
            for j in range(3):
                dj = abs((pos[j][s1 + dt][0] - probe[0])
                         - (pos[j][s1][0] - probe[0]))
                if dj > 2 * dt or dj % 2:
                    ok = False
                    break
            if ok:
                return dt
    return None


def main():
    t0 = time.time()
    law = 1 - 2 * G * Q * Q
    print(f"== T2/TASK 2: imprint writes under fresh-tape streaming ==")
    print(f"   L={L}, T={T}, q={Q}, g={G}, launch {C0}; "
          f"law/cycle = 1-2gq^2 = {law:.6f}\n")
    ann = annealed_percycle()

    # gate: g = 0 must reduce to the fresh-tape identity, with and without
    # the flush batch (the flush is itself a legal fresh schedule)
    for flush in (False, True):
        outg, cg = enumerate_g(flush, 0.0)
        dv = devs_vs_law(outg, ann, 1.0)
        for t, (d, _s) in enumerate(dv, 1):
            assert d < 1e-14, (flush, t, d)
        assert all(v == 0 for v in cg.values()), cg
        print(f" [gate PASSED] g=0, flush={flush}: quenched == annealed "
              f"exactly (max dev {max(d for d, _ in dv):.1e})")

    # config A: F1i = F1 + post-imprint flush batch (the model)
    ta = time.time()
    outA, cA = enumerate_g(True, G)
    dvA = devs_vs_law(outA, ann, law)
    print(f"\n config A (F1i: flush batch, {time.time()-ta:.0f}s): "
          f"counters {cA}")
    for t, (d, s) in enumerate(dvA, 1):
        print(f"   t={t}: max|G_q - law^t G_ann| = {d:.3e}  "
              f"(scale {s:.3e})")
    assert cA["writes"] > 0
    assert cA["env_reread"] == 0 and cA["pending_read"] == 0, cA
    assert cA["relocated_read"] == 0 and cA["write_on_read"] == 0, cA
    assert cA["iota_reread"] == 0, cA
    for t, (d, _s) in enumerate(dvA, 1):
        assert d < 1e-12, (t, d)
    print("   -> ALL reads fresh; U_g = (1-2gq^2) U EXACT at every "
          "computed cycle   [EXACT LAW]")

    # config B: plain F1, production layer order (no flush) -- REFUTATION
    tb = time.time()
    outB, cB = enumerate_g(False, G)
    dvB = devs_vs_law(outB, ann, law)
    print(f"\n config B (plain F1, no flush, {time.time()-tb:.0f}s): "
          f"counters {cB}")
    for t, (d, s) in enumerate(dvB, 1):
        print(f"   t={t}: max|G_q - law^t G_ann| = {d:.3e}  "
              f"(scale {s:.3e})")
    assert cB["env_reread"] == 0 and cB["write_on_read"] == 0, cB
    assert cB["pending_read"] > 0 and cB["relocated_read"] > 0, cB
    assert dvB[0][0] < 1e-12, dvB          # cycle 1 exact even without flush
    assert dvB[1][0] > 1e-9 and dvB[2][0] > 1e-9, dvB
    print("   -> lift sign-read bits ARE re-read (zero-batch window): the "
          "exact law FAILS from cycle 2 under plain F1   [REFUTED]")

    # once-per-cycle fields: required speeds
    print("\n== once-per-cycle-read fields (mass, iota): speed "
          "certificates ==")
    certs = {}
    for name, k, rot, expect in (
            ("mass production (1/cycle, fixed axis)", 1, False, 2),
            ("mass F1 (3/cycle, fixed axis)", 3, False, None),
            ("iota production (1/cycle, rotating axis)", 1, True, 6),
            ("iota rotating 3/cycle", 3, True, 6),
            ("iota rotating 5/cycle", 5, True, 6),
            ("iota rotating 6/cycle", 6, True, 3),
            ("iota rotating 7/cycle", 7, True, None),
            ("iota F1 (3/cycle, fixed axis)", 3, False, None)):
        first = cyclefield_first_reread(k, rot)
        certs[name] = first
        assert first == expect, (name, first, expect)
        verdict = "fresh to 60 cycles" if first is None \
            else f"re-read window at Dt = {first} cycles"
        print(f"  {name}: {verdict}")
    print("  -> F1 treatment: 3 phase-continuing swaps/cycle on a FIXED "
        "axis (mass and iota); a rotating-axis schedule needs 7/activation")

    RESULTS.update({
        "L": L, "T": T, "q": Q, "g": G, "launch": C0,
        "law_per_cycle": law,
        "configA_flush": {"counters": cA,
                          "dev_per_cycle": [d for d, _ in dvA],
                          "scale_per_cycle": [s for _, s in dvA]},
        "configB_noflush": {"counters": cB,
                            "dev_per_cycle": [d for d, _ in dvB],
                            "scale_per_cycle": [s for _, s in dvB]},
        "cyclefield_first_reread": certs,
        "elapsed_s": time.time() - t0,
    })
    out = Path(__file__).resolve().parent.parent / "results" / \
        "freshtape_interaction.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n[T2 TASK 2 PASSED] interacting sector EXACT under F1i "
          f"(flush batch + F1-iota); plain-F1 residual channel identified, "
          f"quantified, and refuted as exact.  ({time.time()-t0:.0f}s)  "
          f"-> {out}")


if __name__ == "__main__":
    main()
