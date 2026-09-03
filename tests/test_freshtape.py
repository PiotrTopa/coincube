"""Fresh-tape (F1) schedule: model-level tests.

(1) Parameter validation; (2) deterministic checksum regressions for BOTH
schedules; (3) the decisive identity, exactly: a self-contained T=2 path
sum (all 4096 conversion histories, physical bit labels tracked through
the schedule's streaming permutations) shows the quenched ensemble
propagator EQUALS the annealed one under 'fresh' and DIFFERS under
'production'.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pca3d.models.coincube import (COIN_D, PERMS, SIGNS,  # noqa: E402
                                   evolve_field_cc)

L, T, Q = 16, 2, 0.08


def test_streaming_param_validation():
    with pytest.raises(ValueError):
        evolve_field_cc(4, 1, 1, 0.1, streaming="bogus")


def test_schedule_checksums_stable():
    a = evolve_field_cc(8, 4, 2, 0.08, seed=5, launch=2, n_blocks=1,
                        streaming="production")
    b = evolve_field_cc(8, 4, 2, 0.08, seed=5, launch=2, n_blocks=1,
                        streaming="fresh")
    w = np.sin(np.arange(a[2].size, dtype=float)).reshape(a[2].shape)
    sa, sb = float((a[2] * w).sum()), float((b[2] * w).sum())
    assert abs(sa - sb) > 1e-6            # schedules genuinely differ
    # regression anchors (update only with a deliberate semantic change)
    assert np.isclose(sa, 0.16443548990164017, atol=1e-12)
    assert np.isclose(sb, 0.09497605765470823, atol=1e-12)


def _swap1d(lab, o):
    m = np.roll(lab, -1) if o else lab.copy()
    t = m[0::2].copy()
    m[0::2] = m[1::2]
    m[1::2] = t
    return np.roll(m, 1) if o else m


def _label_history(swaps_per_substep):
    lab = [np.arange(L ** 3).reshape(L, L, L) for _ in range(3)]
    phase = [0, 0, 0]
    hist = [[None] * (6 * T) for _ in range(3)]
    n = 0
    for _t in range(T):
        for a in range(3):
            sa = (a + 1) % 3
            for o in (0, 1):
                for f in range(3):
                    hist[f][n] = lab[f]
                la = np.moveaxis(lab[a], sa, -1)
                shp = la.shape
                la = la.reshape(-1, L)
                for _s in range(swaps_per_substep):
                    oo = (phase[a] if swaps_per_substep > 1 else o) % 2
                    la = np.apply_along_axis(_swap1d, 1, la, oo)
                    phase[a] += 1
                lab[a] = np.moveaxis(la.reshape(shp), -1, sa)
                n += 1
    return hist


def _path_sums(swaps_per_substep):
    hist = _label_history(swaps_per_substep)
    ctr = L // 2
    gq, ga = {}, {}

    def rec(n, pos, c, s, reads):
        if n == 6 * T:
            key = (pos, c)
            nb1 = sum(reads.values())
            nb0 = len(reads) - nb1
            gq[key] = gq.get(key, 0.0) + s * Q ** nb1 * (1 - Q) ** nb0
            return
        a = (n // 2) % 3
        lb = (a, int(hist[a][n][pos]))
        for bit in (0, 1):
            if lb in reads and reads[lb] != bit:
                continue
            rd = reads if lb in reads else {**reads, lb: bit}
            c2 = int(PERMS[a][c]) if bit else c
            s2 = s * (float(SIGNS[a][c]) if bit else 1.0)
            d = int(COIN_D[a][c2])
            p2 = list(pos)
            p2[a] += d
            rec(n + 1, tuple(p2), c2, s2, rd)

    def rec_ann(n, pos, c, w):
        if n == 6 * T:
            ga[pos, c] = ga.get((pos, c), 0.0) + w
            return
        a = (n // 2) % 3
        for bit in (0, 1):
            c2 = int(PERMS[a][c]) if bit else c
            w2 = w * (Q * float(SIGNS[a][c]) if bit else (1 - Q))
            d = int(COIN_D[a][c2])
            p2 = list(pos)
            p2[a] += d
            rec_ann(n + 1, tuple(p2), c2, w2)

    rec(0, (ctr, ctr, ctr), 0, 1.0, {})
    rec_ann(0, (ctr, ctr, ctr), 0, 1.0)
    keys = set(gq) | set(ga)
    return max(abs(gq.get(k, 0.0) - ga.get(k, 0.0)) for k in keys)


def test_fresh_exact_production_not():
    dev_fresh = _path_sums(3)
    dev_prod = _path_sums(1)
    assert dev_fresh < 1e-14, f"fresh schedule not exact: {dev_fresh}"
    assert dev_prod > 1e-5, f"production unexpectedly exact: {dev_prod}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
