#!/usr/bin/env python
"""W3c composite-cycle sign coherence (the GF(2) obligation, operator form).

Build the FULL lifted cycle of the 1D coincube (L=3 ring, 4 channels + env =
15 modes, 2^15 Fock states) as a signed permutation of the basis, layer by
layer, using exactly the sign rules a CA implementation would store:
  - L2: per-site env-conditioned 16-entry carrier table from the dense
    certified controlled-Givens (env control is a stringless number operator);
  - L1/L3: standard signed-permutation lift (sign = parity of the induced
    reordering of occupied modes).
JW mode ordering is the segregated gauge (all carriers first, env last); the
interleaved site-major ordering decorates the instrument signs with pure-gauge
carrier-env crossing signs (verified, then retired).
Then check against the fermionic reference:
  (1) the composite is a signed permutation (bijective, signs +-1);
  (2) for random classical env configs, the 1-particle sector equals the
      instrument's signed evolution (conversion C_a then shift, per sub-step);
  (3) the 2-particle sector equals the antisymmetrised square Lambda^2 M of
      the 1-particle matrix — the free-fermion determinant structure, i.e.
      the crossing signs are exactly coherent.
"""
import itertools
import numpy as np

from pca3d.models.coincube import COIN_C, COIN_D
from w3c_lift_check import controlled_l2

LX = 3
NM = 5 * LX                      # carriers m = 4x+c (0..11), env m = 12+x
NCAR = 4 * LX
DIM = 1 << NM

# JW MODE ORDERING (gauge choice, and it matters): all carrier modes first,
# all env modes last. Then carrier translations cross no env modes, the L2
# control is a stringless number operator, and env swaps never cross carriers.
# (The site-major interleaved ordering was tried first: it decorates the
# instrument signs with carrier-env crossing signs -- a pure JW artifact.)


def local_tables():
    """Env-conditioned 16-dim carrier tables of the certified Givens (axis 0).

    From the dense 32-dim controlled operator with local order (c0..c3, env):
    env=0 sector must be identity; env=1 sector is the double Givens on the
    4 carrier modes (its JW strings involve carrier modes only).
    """
    U = controlled_l2(0)
    perm = np.argmax(np.abs(U), axis=0)
    sign = U[perm, np.arange(32)]
    p16 = np.empty(16, dtype=np.int64)
    s16 = np.empty(16, dtype=np.int64)
    for s in range(16):
        assert perm[s] == s and sign[s] == 1          # env=0: identity
        t = int(perm[s | 16])                          # env=1 sector
        assert t & 16
        p16[s] = t & 15
        s16[s] = int(sign[s | 16])
    return p16, s16


L2_PERM16, L2_SIGN16 = local_tables()


def apply_l2(perm, sign):
    for x in range(LX):
        base = 4 * x
        ebit = 1 << (NCAR + x)
        for s in range(DIM):
            t = perm[s]
            if not (t & ebit):
                continue
            loc = (t >> base) & 15
            perm[s] = (t & ~(15 << base)) | (int(L2_PERM16[loc]) << base)
            sign[s] *= int(L2_SIGN16[loc])
    return perm, sign


def lift_mode_perm(pi):
    """Standard signed lift of a mode permutation: sign = reordering parity."""
    perm = np.empty(DIM, dtype=np.int64)
    sign = np.empty(DIM, dtype=np.int64)
    for s in range(DIM):
        occ = [m for m in range(NM) if (s >> m) & 1]
        img = [pi[m] for m in occ]
        t = 0
        for i in range(len(img)):
            for j in range(i + 1, len(img)):
                if img[i] > img[j]:
                    t += 1
        perm[s] = sum(1 << m for m in img)
        sign[s] = -1 if t % 2 else 1
    return perm, sign


def compose(p1, s1, p2, s2):
    """(p2, s2) AFTER (p1, s1)."""
    return p2[p1], s1 * s2[p1]


def mode_perm_shift():
    pi = list(range(NM))
    for c in range(4):
        d = int(COIN_D[0][c])
        for x in range(LX):
            pi[4 * x + c] = 4 * ((x + d) % LX) + c
    return pi


def mode_perm_env(o):
    pi = list(range(NM))
    # pair swap along x with origin o (LX=3 odd ring: swap the full pairing
    # that fits; the leftover site maps to itself)
    x0 = o
    x1 = (o + 1) % LX
    pi[NCAR + x0], pi[NCAR + x1] = pi[NCAR + x1], pi[NCAR + x0]
    return pi


SH_PERM, SH_SIGN = lift_mode_perm(mode_perm_shift())
ENV_PERM = {o: lift_mode_perm(mode_perm_env(o)) for o in (0, 1)}


def full_cycle():
    perm = np.arange(DIM, dtype=np.int64)
    sign = np.ones(DIM, dtype=np.int64)
    for o in (0, 1):
        perm, sign = apply_l2(perm, sign)
        perm, sign = compose(perm, sign, SH_PERM, SH_SIGN)
        ep, es = ENV_PERM[o]
        perm, sign = compose(perm, sign, ep, es)
    return perm, sign


def carrier_mode(x, c):
    return 4 * x + c


def single_particle_reference(env_bits):
    """Instrument rule: per sub-step, signed C at env sites then shift."""
    n1 = 4 * LX                  # 1p states indexed (x, c) -> 4x + c
    M = np.eye(n1)
    env = list(env_bits)
    permC = np.argmax(np.abs(COIN_C[0]), axis=0)
    for o in (0, 1):
        step = np.zeros((n1, n1))
        for x in range(LX):
            for c in range(4):
                j = 4 * x + c
                if env[x]:
                    i_c = int(permC[c])
                    amp = COIN_C[0][i_c, c]
                    xt = (x + int(COIN_D[0][i_c])) % LX
                    step[4 * xt + i_c, j] = amp
                else:
                    xt = (x + int(COIN_D[0][c])) % LX
                    step[4 * xt + c, j] = 1.0
        M = step @ M
        x0, x1 = o, (o + 1) % LX
        env[x0], env[x1] = env[x1], env[x0]
    return M, env


def sector_matrix(perm, sign, env_in, env_out, k):
    """k-carrier sector block for env_in -> env_out."""
    env_mask_in = sum(1 << (NCAR + x) for x in range(LX) if env_in[x])
    states = []
    for occ in itertools.combinations(range(NCAR), k):
        states.append((occ, sum(1 << m for m in occ) | env_mask_in))
    idx = {st[0]: i for i, st in enumerate(states)}
    env_mask_out = sum(1 << (NCAR + x) for x in range(LX) if env_out[x])
    M = np.zeros((len(states), len(states)))
    for occ, s in states:
        t = int(perm[s])
        assert (t & (((1 << LX) - 1) << NCAR)) == env_mask_out
        carr = [m for m in range(NCAR) if (t >> m) & 1]
        M[idx[tuple(sorted(carr))], idx[occ]] = sign[s]
    return M


def wedge_square(M):
    n = M.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    W = np.zeros((len(pairs), len(pairs)))
    for b, (k, l) in enumerate(pairs):
        for a, (i, j) in enumerate(pairs):
            W[a, b] = M[i, k] * M[j, l] - M[i, l] * M[j, k]
    return W


def main():
    perm, sign = full_cycle()
    assert len(np.unique(perm)) == DIM, "composite not bijective"
    assert np.all(np.abs(sign) == 1)
    print(f"composite cycle on {NM} modes: signed permutation OK "
          f"({DIM} states)")

    rng = np.random.default_rng(4)
    for trial in range(4):
        env0 = [int(b) for b in rng.integers(0, 2, LX)]
        M1_ref, env_out = single_particle_reference(env0)
        M1 = sector_matrix(perm, sign, env0, env_out, 1)
        ok1 = np.allclose(M1, M1_ref, atol=1e-12)
        M2 = sector_matrix(perm, sign, env0, env_out, 2)
        ok2 = np.allclose(M2, wedge_square(M1_ref), atol=1e-12)
        print(f"env={env0}: 1p == instrument rule: {ok1};  "
              f"2p == Lambda^2(1p): {ok2}")
        assert ok1 and ok2
    print("\ncomposite sign coherence: CERTIFIED "
          "(2-particle sector is the exact antisymmetrised square)")


if __name__ == "__main__":
    main()
