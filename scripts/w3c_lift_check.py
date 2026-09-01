#!/usr/bin/env python
"""W3c lift certificate: the conversion layer is an exact local fermionic
operation whose single-particle sector is C_a.

Setup: one site, 5 modes (4 species channels + 1 env), dense Fock space
(32-dim). For each axis a, L2 is the product of two env-controlled Givens
rotations G = exp((pi/2)(a_c^dag a_cp - a_cp^dag a_c)) over the two channel
pairs of C_a's involution, oriented so the single-particle signs match C_a.

Machine checks:
 1. U is a SIGNED PERMUTATION of the Fock basis (classical CA + sign gauge);
 2. U is unitary and commutes with total fermion parity (R4-1/2);
 3. env empty  -> U acts as identity;
    env filled -> <1_i(env)| U a_j^dag |1_env> single-particle matrix == C_a;
 4. U^2 restricted to the single-particle sector == C_a^2 = -I (the
    quaternion structure is the double-Givens sign, not an insertion);
 5. doubly-occupied pair: G|1_c 1_cp> = +|1_c 1_cp> (SO(2) determinant).
"""
import numpy as np
from scipy.linalg import expm

from pca3d.models.coincube import COIN_C

NM = 5          # modes: channels 0..3, env = 4
DIM = 1 << NM


def a_op(i):
    """Annihilation operator, Jordan-Wigner, mode order 0..4 (same site)."""
    m = np.zeros((DIM, DIM))
    for s in range(DIM):
        if (s >> i) & 1:
            sgn = (-1) ** bin(s & ((1 << i) - 1)).count("1")
            m[s ^ (1 << i), s] = sgn
    return m


A = [a_op(i) for i in range(NM)]
N_ENV = A[4].T @ A[4]
PARITY = np.diag([(-1) ** bin(s).count("1") for s in range(DIM)]).astype(float)


def controlled_l2(a):
    C = COIN_C[a]
    U = np.eye(DIM)
    done = set()
    for j in range(4):
        i = int(np.argmax(np.abs(C[:, j])))
        if (i, j) in done or (j, i) in done:
            continue
        done.add((j, i))
        s = C[i, j]                      # a_j^dag -> s * a_i^dag
        gen = s * (A[i].T @ A[j]) - s * (A[j].T @ A[i])
        G = expm((np.pi / 2) * gen)
        U = U @ ((np.eye(DIM) - N_ENV) + N_ENV @ G)
    return U


def single_particle(U):
    """<vac,env=1| a_i U a_j^dag |vac,env=1>."""
    vac_env = np.zeros(DIM)
    vac_env[1 << 4] = 1.0
    out = np.zeros((4, 4))
    for j in range(4):
        v = U @ A[j].T @ vac_env
        for i in range(4):
            out[i, j] = vac_env @ (A[i] @ v)
    return out


def is_signed_permutation(U, tol=1e-10):
    absU = np.abs(U)
    return (np.all(np.isclose(absU.sum(axis=0), 1, atol=tol)) and
            np.all(np.isclose(absU.max(axis=0), 1, atol=tol)))


def main():
    ok_all = True
    for a in range(3):
        U = controlled_l2(a)
        c1 = is_signed_permutation(U)
        c2 = np.allclose(U @ U.T, np.eye(DIM)) and np.allclose(
            U @ PARITY, PARITY @ U)
        # env empty -> identity on the no-env sector
        no_env = [s for s in range(DIM) if not (s >> 4) & 1]
        c3a = np.allclose(U[np.ix_(no_env, no_env)], np.eye(len(no_env)))
        sp = single_particle(U)
        c3b = np.allclose(sp, COIN_C[a], atol=1e-12)
        sp2 = single_particle(U @ U)
        c4 = np.allclose(sp2, -np.eye(4), atol=1e-12)
        # doubly occupied pair -> +1 (check every pair state with env)
        c5 = True
        C = COIN_C[a]
        for j in range(4):
            i = int(np.argmax(np.abs(C[:, j])))
            st = np.zeros(DIM)
            st[(1 << i) | (1 << j) | (1 << 4)] = 1.0
            c5 &= abs(st @ (U @ st) - 1.0) < 1e-12
        ok = all([c1, c2, c3a, c3b, c4, c5])
        ok_all &= ok
        print(f"axis {a}: signed-perm={c1} unitary+parity={c2} "
              f"env0-identity={c3a} single-particle==C={c3b} "
              f"U^2==-I(1p)={c4} pair|11>->+|11>={c5}   "
              f"=> {'CERTIFIED' if ok else 'FAIL'}")
    print("\nL2 lift certificate:", "ALL PASS" if ok_all else "FAILURES")
    assert ok_all


if __name__ == "__main__":
    main()
