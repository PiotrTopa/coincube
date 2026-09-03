#!/usr/bin/env python
"""Two-boundary (in--out) amplitudes under the fresh-tape schedule: exact check.

Claim (paper, Sec. Damping): under fresh-tape streaming the postselected
two-boundary amplitude inherits the no-re-read property of the fresh-tape
theorem -- the quenched two-boundary path sum is the SAME path sum as the
propagator's, with boundary weights (sqrt(1-q), sqrt(q)) in place of
(1-q, q) -- so the in--out family equals powers of the unitary
arc operator exactly at every cycle.

Method: exact quenched enumeration. Every conversion history over TCYC
cycles is expanded as a branch tree with per-bit read records; bits are
transported by the F1 streaming (three phase-continuing pair-swap batches
per axis substep, cross-axis (a+1)%3), so a branch's read set is the true
tape read set. Each read is then integrated against the in--out boundary
weights sqrt(1-q), sqrt(q) (normalized by D = sqrt(1-q)+sqrt(q)), and the
resulting momentum-space amplitude is compared to the t-th power of the
arc operator M(k) = prod_a [E_a (c + s C_a)]^2, c = sqrt(1-q)/D,
s = sqrt(q)/D.

Gate: worst relative deviation over cycles t=1..TCYC and probe momenta
must be < 1e-12 (machine precision expected). Torus scope: free-model
horizon T <= ceil(L/8); L=24, TCYC=3 sits exactly at the horizon.

Output: results/inout_fresh.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pca3d.models.coincube import COIN_C, COIN_D, perm_sign

Q, L, TCYC = 0.08, 24, 3
PERMS, SIGNS = zip(*(perm_sign(c) for c in COIN_C))
SQ0, SQ1 = np.sqrt(1 - Q), np.sqrt(Q)
DNORM = SQ0 + SQ1


def stream_ids(ids, axis, o):
    """One certified pair-swap batch on the bit-identity lattice."""
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
    ids = [np.arange(L ** 3).reshape(L, L, L) + a * L ** 3 for a in range(3)]
    phase = [0, 0, 0]
    branches = {(0, 0, 0, 0): [(1.0, {})]}
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
                            r2 = (reads if known is not None
                                  else {**reads, site_id: val})
                            if val == 1:
                                ch2 = int(PERMS[a][ch])
                                sg2 = sg * float(SIGNS[a][ch])
                            else:
                                ch2, sg2 = ch, sg
                            p2 = [x, y, z]
                            p2[a] += int(COIN_D[a][ch2])
                            new.setdefault((p2[0], p2[1], p2[2], ch2),
                                           []).append((sg2, r2))
                branches = new
                for _s in range(3):  # F1: three phase-continuing batches
                    ids[a] = stream_ids(ids[a], (a + 1) % 3, phase[a] % 2)
                    phase[a] += 1
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


def m_io(kvec):
    c, s = SQ0 / DNORM, SQ1 / DNORM
    u = np.eye(4, dtype=complex)
    for a in range(3):
        tmat = (np.diag(np.exp(1j * kvec[a] * COIN_D[a]))
                @ (c * np.eye(4) + s * COIN_C[a]))
        u = tmat @ tmat @ u
    return u


def main():
    amps = enumerate_paths()
    ks = [np.array(v, float) for v in
          ([np.pi, 0, 0], [0.4, 0.1, 0.7], [1.1, 2.0, 0.3])]
    rows = []
    worst = 0.0
    for t in range(1, TCYC + 1):
        amp = amps[t - 1]
        for kv in ks:
            zt = np.zeros(4, dtype=complex)
            for (dx, dy, dz, ch), a in amp.items():
                zt[ch] += a * np.exp(-1j * (kv[0] * dx + kv[1] * dy
                                            + kv[2] * dz))
            pred = np.linalg.matrix_power(m_io(-kv), t)[:, 0]
            dev = float(np.abs(zt - pred).max() / np.linalg.norm(pred))
            ratio = float(np.linalg.norm(zt) / np.linalg.norm(pred))
            worst = max(worst, dev)
            rows.append({"t": t, "k": list(kv), "rel_dev": dev,
                         "norm_ratio": ratio})
            print(f"t={t} k={np.round(kv, 2)}: rel dev = {dev:.3e}  "
                  f"|Z| ratio = {ratio:.12f}")
    gate = worst < 1e-12
    print(f"WORST rel dev: {worst:.3e}  "
          f"[{'GATE PASSED' if gate else 'GATE FAILED'}]")
    out = {"q": Q, "L": L, "TCYC": TCYC, "rows": rows,
           "worst_rel_dev": worst, "gate_passed": gate}
    pathlib.Path("results/inout_fresh.json").write_text(
        json.dumps(out, indent=1))
    print("written: results/inout_fresh.json")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
