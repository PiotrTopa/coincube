#!/usr/bin/env python
"""Mass identity of the massive (inversion-doubled) coincube, proved
structurally and machine-checked symbolically with a full VECTOR momentum.

IDENTITY (I).  With b_m the minor (mass-bit) index,

    U8(k) = (1 (x) R_m) [ U4(k) (+) U4(-k) ],
    R_m   = (1 - q_m) 1_2 + q_m XZ,

where the direct sum is in the b_m basis (interleaved: RHS[2i+b, 2j+b'] =
R_m[b, b'] U4(b'-signed k)[i, j]).

STRUCTURAL PROOF (each ingredient checked below):
  (1) d^(8) = d (x) diag(1, -1): the mass bit reverses ALL direction
      assignments, so the transport phase factorizes,
      E8_a(k_a) = E4_a(k_a) (+) E4_a(-k_a) in the b_m basis;
  (2) C8_a = C4_a (x) 1_2: conversions are blind to b_m -- every
      conversion layer is block-DIAGONAL in b_m and acts identically on
      both blocks;
  (3) hence the whole pre-mass cycle is U4(k) (+) U4(-k);
  (4) the mass layer is 1_4 (x) R_m (a b_m rotation blind to the coin),
      applied once per cycle.
The similarity R_m of the paper's statement is exactly this mass factor:
after moving to the b_m basis the identity reads
U8 = (1 (x) R_m) [U4(k) (+) U4(-k)].

IDENTITY (II), corollary at k = 0: U8(0) = U4(0) (x) R_m.

Both identities are verified SYMBOLICALLY in (k_x, k_y, k_z, q, q_m) --
independent momentum components, not a scalar k.  Assertions run BEFORE
the results file is written.  Results -> results/theory_mass_identity.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_mass_identity.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import (COIN_C, COIN_C8, COIN_D, COIN_D8,  # noqa: E402
                                   MASS_C, annealed_u8)

SEED = 20260902
RESULTS = (Path(__file__).resolve().parent.parent / "results"
           / "theory_mass_identity.json")


def main():
    import sympy as sp
    t0 = time.time()
    out = {}

    # ---- structural ingredients, from the model source itself ----------
    C4 = [np.array(c, dtype=int) for c in COIN_C]
    D4 = [np.array(d, dtype=int) for d in COIN_D]
    C8 = [np.array(c, dtype=int) for c in COIN_C8]
    D8 = [np.array(d, dtype=int) for d in COIN_D8]
    out["d8_is_d4_tensor_diag1m1"] = bool(all(
        np.array_equal(D8[a], np.kron(D4[a], np.array([1, -1])))
        for a in range(3)))
    out["c8_is_c4_tensor_id"] = bool(all(
        np.array_equal(C8[a], np.kron(C4[a], np.eye(2, dtype=int)))
        for a in range(3)))
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])
    out["mass_is_id_tensor_XZ"] = bool(
        np.array_equal(np.array(MASS_C, dtype=int), np.kron(np.eye(4, dtype=int),
                                                            X @ Z)))
    # blindness: conversions commute with EVERY operator on the b_m factor
    dev = 0.0
    for a in range(3):
        for M2 in (X, Z, X @ Z):
            B = np.kron(np.eye(4), M2)
            dev = max(dev, np.abs(C8[a] @ B - B @ C8[a]).max())
    out["conversion_bm_blindness_dev"] = float(dev)

    # ---- symbolic identities with vector momentum ----------------------
    q, qm = sp.symbols('q q_m', real=True)
    ks = sp.symbols('k_x k_y k_z', real=True)
    XZs = sp.Matrix([[0, -1], [1, 0]])
    C4s = [sp.Matrix(c) for c in C4]
    D4s = [sp.diag(*[int(x) for x in d]) for d in D4]

    def U4(sign):
        u = sp.eye(4)
        for a in range(3):
            E = sp.diag(*[sp.exp(sp.I * sign * ks[a] * D4s[a][c, c])
                          for c in range(4)])
            T = E * ((1 - q) * sp.eye(4) + q * C4s[a])
            u = T * T * u
        return u

    def U8():
        u = sp.eye(8)
        for a in range(3):
            D8v = np.kron(D4[a], np.array([1, -1]))
            E = sp.diag(*[sp.exp(sp.I * ks[a] * int(D8v[c]))
                          for c in range(8)])
            C8a = sp.Matrix(np.kron(C4[a], np.eye(2, dtype=int)))
            T = E * ((1 - q) * sp.eye(8) + q * C8a)
            u = T * T * u
        CM = sp.Matrix(np.kron(np.eye(4, dtype=int),
                               np.array([[0, -1], [1, 0]])))
        return ((1 - qm) * sp.eye(8) + qm * CM) * u

    lhs = U8()
    Rm = (1 - qm) * sp.eye(2) + qm * XZs
    Up, Um = U4(1), U4(-1)
    rhs = sp.Matrix(8, 8, lambda I_, J_: Rm[I_ % 2, J_ % 2]
                    * (Up if J_ % 2 == 0 else Um)[I_ // 2, J_ // 2])
    diff = sp.expand(lhs - rhs)
    out["identity_I_vector_k_symbolic"] = bool(all(
        sp.simplify(diff[i, j]) == 0 for i in range(8) for j in range(8)))
    print(f"[{time.time() - t0:5.1f}s] identity I checked (vector k)")

    lhs0 = lhs.subs({ks[0]: 0, ks[1]: 0, ks[2]: 0})
    U40 = Up.subs({ks[0]: 0, ks[1]: 0, ks[2]: 0})
    rhs0 = sp.Matrix(8, 8, lambda i, j: U40[i // 2, j // 2] * Rm[i % 2, j % 2])
    diff0 = sp.expand(lhs0 - rhs0)
    out["identity_II_k0_symbolic"] = bool(all(
        sp.simplify(diff0[i, j]) == 0 for i in range(8) for j in range(8)))
    print(f"[{time.time() - t0:5.1f}s] identity II checked (k = 0)")

    # ---- numeric spot check against the production operator ------------
    rng = np.random.default_rng(SEED)
    dev = 0.0
    for _ in range(6):
        kv = rng.uniform(-np.pi, np.pi, 3)
        qv, qmv = rng.uniform(0.02, 0.6), rng.uniform(0.02, 0.6)
        lhs_n = annealed_u8(kv, qv, qmv)
        subs = {ks[0]: kv[0], ks[1]: kv[1], ks[2]: kv[2], q: qv, qm: qmv}
        rhs_n = np.array(sp.N(rhs.subs(subs)), dtype=complex)
        dev = max(dev, float(np.abs(lhs_n - rhs_n).max()))
    out["production_operator_dev"] = dev

    # -------- assertions BEFORE the results file is written -------------
    assert out["d8_is_d4_tensor_diag1m1"]
    assert out["c8_is_c4_tensor_id"]
    assert out["mass_is_id_tensor_XZ"]
    assert out["conversion_bm_blindness_dev"] == 0.0
    assert out["identity_I_vector_k_symbolic"]
    assert out["identity_II_k0_symbolic"]
    assert out["production_operator_dev"] < 1e-12
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
