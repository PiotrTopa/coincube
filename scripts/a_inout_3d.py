#!/usr/bin/env python
"""Route A, decisive check: EXACT quenched-3D in-out propagator by coherent
path enumeration.

The in-out 1p amplitude with the uniform final env boundary is a coherent sum
over carrier branch histories (convert / not-convert per sub-step), where a
branch's amplitude is the product over DISTINCT env bits it read of
sqrt(q) (read as 1) or sqrt(1-q) (read as 0), divided by D = sqrt(1-q)+sqrt(q)
per read (the Z-normalisation), times the lift sign. Consistency: every read
of the same physical bit (tracked through the cross-streaming by an id field)
must see the same value; inconsistent branch sets contribute nothing.

Fresh-read law prediction: G_io/Z = [M_io]^t with per-sub-step transfer
E(k)(c + sC), c = sqrt(1-q)/D, s = sqrt(q)/D. Deviations = genuine 3D
recross corrections — expected small (cross-streaming), unlike the 1D
substrate where re-reads dominate.

Exact, no sampling: branch tree with consistency pruning, t <= 3 cycles.
"""
import numpy as np

from pca3d.models.coincube import COIN_C, COIN_D, annealed_u, perm_sign

Q = 0.08
L = 16
TCYC = 3
PERMS, SIGNS = zip(*(perm_sign(c) for c in COIN_C))
SQ0, SQ1 = np.sqrt(1 - Q), np.sqrt(Q)
DNORM = SQ0 + SQ1


def stream_ids(ids, axis, o):
    m = np.moveaxis(ids, axis, -1)
    if o:
        m = np.roll(m, -1, axis=-1)
    m = m.copy()
    m0 = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = m0
    if o:
        m = np.roll(m, 1, axis=-1)
    return np.moveaxis(m, -1, axis)


def enumerate_paths():
    """Coherent branch sum. Returns dict (dx, dy, dz, ch) -> amplitude,
    per cycle count 1..TCYC, for launch channel 0 at the origin."""
    ids = [np.arange(L ** 3).reshape(L, L, L) + a * L ** 3 for a in range(3)]
    # branches: (pos(3), ch, sign, frozenset of (bit_id, val))
    branches = {(0, 0, 0, 0): [(1.0, {})]}   # key: pos+ch; val: [(sign, reads)]
    out = []
    for t in range(TCYC):
        for a in (0, 1, 2):
            for o in (0, 1):
                new = {}
                for (x, y, z, ch), lst in branches.items():
                    site_id = int(ids[a][x % L, y % L, z % L])
                    for (sg, reads) in lst:
                        known = reads.get(site_id)
                        for val in ((0, 1) if known is None else (known,)):
                            r2 = reads if known is not None else {**reads,
                                                                  site_id: val}
                            if val == 1:
                                ch2 = int(PERMS[a][ch])
                                sg2 = sg * float(SIGNS[a][ch])
                            else:
                                ch2, sg2 = ch, sg
                            p2 = [x, y, z]
                            p2[a] += int(COIN_D[a][ch2])
                            key = (p2[0], p2[1], p2[2], ch2)
                            new.setdefault(key, []).append((sg2, r2))
                    # (no per-branch amplitude yet: applied at readout)
                branches = new
                ids[a] = stream_ids(ids[a], (a + 1) % 3, o)
        amp = {}
        for key, lst in branches.items():
            tot = 0.0
            for (sg, reads) in lst:
                n1 = sum(reads.values())
                n = len(reads)
                tot += sg * (SQ1 ** n1) * (SQ0 ** (n - n1)) / (DNORM ** n)
            amp[key] = tot
        out.append(amp)
    return out


def fresh_read_prediction(t):
    """[M_io]^t at k as a real-space kernel via the annealed-form operator:
    evaluate on a k-grid and inverse transform is overkill — instead compare
    IN K-SPACE at a few k points."""
    return None


def main():
    print(f"exact coherent 3D path sum: q = {Q}, up to T = {TCYC} cycles")
    amps = enumerate_paths()
    c, s = SQ0 / DNORM, SQ1 / DNORM

    def m_io(kvec):
        u = np.eye(4, dtype=complex)
        for a in range(3):
            tmat = np.diag(np.exp(1j * kvec[a] * COIN_D[a])) @ (
                c * np.eye(4) + s * COIN_C[a])
            u = tmat @ tmat @ u
        return u

    ks = [np.array(v, float) for v in
          ([np.pi, 0, 0], [0.4, 0.1, 0.7], [1.1, 2.0, 0.3])]
    for t in range(1, TCYC + 1):
        amp = amps[t - 1]
        for kv in ks:
            zt = np.zeros(4, dtype=complex)
            for (dx, dy, dz, ch), a in amp.items():
                zt[ch] += a * np.exp(-1j * (kv[0] * dx + kv[1] * dy +
                                            kv[2] * dz))
            pred = np.linalg.matrix_power(m_io(-kv), t)[:, 0]
            dev = np.abs(zt - pred).max()
            mod_meas = np.linalg.norm(zt)
            mod_pred = np.linalg.norm(pred)
            print(f"  t={t} k={np.array2string(kv, precision=2)}: "
                  f"rel dev = {dev / mod_pred:.4f}   "
                  f"|Z| meas/pred = {mod_meas / mod_pred:.4f}")
    print(f"\n(endpoint classes at t={TCYC}: {len(amps[-1])})")


if __name__ == "__main__":
    main()
