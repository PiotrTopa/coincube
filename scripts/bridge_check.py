#!/usr/bin/env python
"""EXPLORATORY: the I-spectrum bridge machine-checked at L = 3 (F11 closure).

The intertwiner Phi(u + iv) = u (+) (-P v) connects the Fourier-i spectral
analysis to the automaton's (K, I) quantum mechanics. Review round 4 noted
the direct check ran only at L = 2 -- the size that is provably blind to
translation-sign errors (mutation controls: 0/496 mismatches at L = 2 vs
2772/5778 at L = 3). The bridge's content follows from [S,P] = 0 (proven at
all sizes), but this closes the empirical gap: the identical check at
L = 3, a sign-sensitive size, over multiple environment draws and densities.

The check's content is streaming-schedule agnostic (streaming layers are
environment-supported; equivariance proven for any composition of
layer_env-type swaps -- theory_sp_proof), so it covers both the production
and fresh-tape (F1) cycles.
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(Path := __import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.parent / "src"))
import theory_3d_certs as tc  # noqa: E402


def bridge_check(L, q_env, seed):
    rng = np.random.default_rng(seed)
    lat = tc.Lattice(L)
    M = lat.M
    env0 = [(rng.random(lat.NS) < q_env).astype(np.int8) for _ in range(3)]
    states_1p = [(m,) for m in range(M)]
    states_1h = [tuple(m for m in range(M) if m != h) for h in range(M)]
    p1, s1, _ = tc.cycle_sector(lat, states_1p, env0)
    ph, sh, _ = tc.cycle_sector(lat, states_1h, env0)
    sP = np.array([tc.majorana_sign((m,), M) for m in range(M)])

    S1 = np.zeros((M, M))
    Sh = np.zeros((M, M))
    for m in range(M):
        S1[p1[m], m] = s1[m]
        Sh[ph[m], m] = sh[m]
    Pm = np.diag(sP.astype(float))

    def Phi(x):
        return np.concatenate([x.real, -(Pm @ x.imag)])

    def Iop(y):
        u, w = y[:M], y[M:]
        return np.concatenate([Pm @ w, Pm @ (-u)])

    def Shat(y):
        return np.concatenate([S1 @ y[:M], Sh @ y[M:]])

    dev_i = dev_s = dev_h = 0.0
    for _ in range(20):
        x = rng.normal(size=M) + 1j * rng.normal(size=M)
        y = rng.normal(size=M) + 1j * rng.normal(size=M)
        dev_i = max(dev_i, np.abs(Phi(1j * x) - Iop(Phi(x))).max())
        dev_s = max(dev_s, np.abs(Phi(S1 @ x) - Shat(Phi(x))).max())
        hI = Phi(x) @ Phi(y) - 1j * (Phi(x) @ Iop(Phi(y)))
        dev_h = max(dev_h, abs(hI - np.vdot(x, y)))
    return {"L": L, "q_env": q_env, "seed": seed, "M": M,
            "Phi_intertwines_i_vs_I": float(dev_i),
            "Phi_intertwines_dynamics": float(dev_s),
            "hermitian_form_matches": float(dev_h)}


def main():
    out = []
    t0 = time.time()
    for L, q_env, seed in ((3, 0.08, 1), (3, 0.08, 2), (3, 0.35, 3),
                           (2, 0.08, 4)):
        r = bridge_check(L, q_env, seed)
        out.append(r)
        print(f"L={L} q={q_env} seed={seed} (M={r['M']}): "
              f"i-vs-I {r['Phi_intertwines_i_vs_I']:.2e}  "
              f"dynamics {r['Phi_intertwines_dynamics']:.2e}  "
              f"form {r['hermitian_form_matches']:.2e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        assert r["Phi_intertwines_i_vs_I"] < 1e-12
        assert r["Phi_intertwines_dynamics"] < 1e-12
        assert r["hermitian_form_matches"] < 1e-12
    print("[PASSED] bridge exact at L=3 (sign-sensitive size), "
          "three draws incl. working density q=0.08")
    pathlib.Path("results/bridge_check.json").write_text(
        json.dumps(out, indent=1))
    print("written: results/bridge_check.json")


if __name__ == "__main__":
    main()
