#!/usr/bin/env python
"""W2 (restored evidence code): scalar-sign-gauge spectroscopy of the
dressed split-step walk — the measurement behind the Abelian-gauge no-go
(paper Sec. III: base/stall identical diamond spectra, KS gauges likewise
diamond). Restored from tag v1.1-hardened after the red-team review noted
the pruned tree carried the claim without its implementing code.

Full-BZ omega(k) and persistence maps via 3D FFT per cycle + Prony-1 lambda fit
(omega = -arg lambda, robust to gauge sign structure). The Dirac point k0 per gauge is
found EMPIRICALLY: the k maximizing spectral persistence weight among low-|omega|
candidates. Slopes are then measured from k0 along (100), (110), (111) grid steps.
Cone: ratios ~ 1,1. Diamond: ~ 1.41, 1.73. Cone-bound assertion on every slope.
"""
import json, pathlib
import numpy as np
from pca3d.models.walk3d import make_evolve, GPU

L, E, TCYC = 48, 5000, 11
# lane media + interleaved streaming: the class-locking architecture (see J27) --
# the only schedule under which the 1D rule-891 ballistic law transfers per axis.
evolve = make_evolve(L, E, TCYC, continuous=False, lane_media=True)

def gfield(traj, signs):
    out = np.zeros((len(traj), L, L, L))
    for t, (p, s) in enumerate(zip(traj, signs)):
        p_np = p.get() if GPU else p
        s_np = s.get() if GPU else s
        np.add.at(out[t], (p_np[:, 0], p_np[:, 1], p_np[:, 2]), s_np)
    return out / p_np.shape[0]

def analyse(q, gauge, seed=3):
    traj, signs = evolve(q, gauge, seed)
    G = gfield(traj, signs)
    T1 = G.shape[0]
    Z = np.stack([np.fft.fftn(G[t]) for t in range(T1)])          # (T, L, L, L)
    # Prony-1 per k, EXCLUDING the launch transient (t < 3): the t=0 delta has flat
    # phase and biased the slope 20% low against the exact 1D anchor v = 2(1-2q)
    Z = Z[3:]
    num = (np.conj(Z[:-1]) * Z[1:]).sum(axis=0)
    den = (np.abs(Z[:-1]) ** 2).sum(axis=0) + 1e-30
    lam = num / den
    om = -np.angle(lam)                                            # per cycle
    weight = (np.abs(Z) ** 2).sum(axis=0)
    persist = np.abs(lam)                                          # ~1 = long-lived
    # Dirac point: among top-weight ks, the one with min |omega| and high persistence
    flat_w = weight.reshape(-1)
    cand = np.argsort(-flat_w)[:64]
    score = [abs(om.reshape(-1)[c]) - 0.5 * persist.reshape(-1)[c] for c in cand]
    k0_flat = cand[int(np.argmin(score))]
    k0 = np.array(np.unravel_index(k0_flat, (L, L, L)))
    dirs = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1)}
    slopes = {}
    for name, d in dirs.items():
        d = np.array(d)
        oms = []
        for n in (1, 2):
            kp = tuple((k0 + n * d) % L)
            km = tuple((k0 - n * d) % L)
            do = (om[kp] - om[km]) / 2.0                            # symmetric difference
            dk = n * np.linalg.norm(d) * 2 * np.pi / L
            oms.append(abs(do) / dk)
        slopes[name] = float(np.mean(oms))
    vmax_cycle = 2 * np.sqrt(3)  # per cycle, any direction
    ok = all(v <= vmax_cycle + 0.3 for v in slopes.values())
    return k0, slopes, ok, float(persist[tuple(k0)]), float(om[tuple(k0)])

print(f"{'q':>5} {'gauge':>9} {'k0':>12} {'v100':>7} {'v110':>7} {'v111':>7} "
      f"{'r110':>6} {'r111':>6} {'persist':>8} {'om(k0)':>8} {'bound':>6}")
out = {}
print("CALIBRATION ANCHOR: v100 must equal 2(1-2q) = 1.600 at q=0.10, 1.800 at q=0.05")
for q, gauge in [(0.10, "base"), (0.05, "base"), (0.10, "ks"), (0.10, "stall"),
                 (0.10, "ks+stall"), (0.05, "ks+stall")]:
    k0, s, ok, per, om0 = analyse(q, gauge)
    r110 = s["110"] / max(s["100"], 1e-9)
    r111 = s["111"] / max(s["100"], 1e-9)
    out[f"{q}_{gauge}"] = {"k0": k0.tolist(), "slopes": s, "r110": r110, "r111": r111,
                           "persist": per, "om0": om0, "bound_ok": ok}
    print(f"{q:>5.2f} {gauge:>9} {str(k0.tolist()):>12} {s['100']:>7.3f} {s['110']:>7.3f} "
          f"{s['111']:>7.3f} {r110:>6.2f} {r111:>6.2f} {per:>8.3f} {om0:>8.3f} "
          f"{'OK' if ok else 'VIOL':>6}", flush=True)
pathlib.Path("results/w2_corner.json").write_text(json.dumps(out, indent=2))
print("written: results/w2_corner.json   (cone: r~1,1; diamond: r~1.41,1.73)")
