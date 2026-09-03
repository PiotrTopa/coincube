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
import json
import pathlib

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


def apply_l2(perm, sign, s16=None):
    if s16 is None:
        s16 = L2_SIGN16
    for x in range(LX):
        base = 4 * x
        ebit = 1 << (NCAR + x)
        for s in range(DIM):
            t = perm[s]
            if not (t & ebit):
                continue
            loc = (t >> base) & 15
            perm[s] = (t & ~(15 << base)) | (int(L2_PERM16[loc]) << base)
            sign[s] *= int(s16[loc])
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


def full_cycle(schedule="production", s16=None):
    """The composite lifted cycle. schedule='production' (default, the
    committed certificate): one L3 streaming layer per sub-step, origin o.
    schedule='fresh' (F1): three PHASE-CONTINUING streaming layers per
    sub-step (the n-th layer ever applied has origin n mod 2). ``s16``
    optionally overrides the L2 sign table (mutation control)."""
    perm = np.arange(DIM, dtype=np.int64)
    sign = np.ones(DIM, dtype=np.int64)
    phase = 0
    for o in (0, 1):
        perm, sign = apply_l2(perm, sign, s16)
        perm, sign = compose(perm, sign, SH_PERM, SH_SIGN)
        if schedule == "fresh":
            for _ in range(3):
                ep, es = ENV_PERM[phase % 2]
                perm, sign = compose(perm, sign, ep, es)
                phase += 1
        else:
            ep, es = ENV_PERM[o]
            perm, sign = compose(perm, sign, ep, es)
    return perm, sign


def carrier_mode(x, c):
    return 4 * x + c


def single_particle_reference(env_bits, schedule="production"):
    """Instrument rule: per sub-step, signed C at env sites then shift."""
    n1 = 4 * LX                  # 1p states indexed (x, c) -> 4x + c
    M = np.eye(n1)
    env = list(env_bits)
    phase = 0
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
        if schedule == "fresh":
            for _ in range(3):
                x0, x1 = phase % 2, (phase % 2 + 1) % LX
                env[x0], env[x1] = env[x1], env[x0]
                phase += 1
        else:
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


def run_checks(schedule, s16=None, ntrial=4, verbose=True):
    """Bijectivity + 1p-vs-instrument + 2p-vs-Lambda^2 over ntrial env draws.
    Returns (n_1p_mismatch, n_2p_mismatch, n_2p_states_checked)."""
    perm, sign = full_cycle(schedule, s16)
    assert len(np.unique(perm)) == DIM, "composite not bijective"
    assert np.all(np.abs(sign) == 1)
    if verbose:
        print(f"composite cycle on {NM} modes ({schedule}): signed "
              f"permutation OK ({DIM} states)")
    rng = np.random.default_rng(4)
    n1_mis = n2_mis = n2_states = 0
    for trial in range(ntrial):
        env0 = [int(b) for b in rng.integers(0, 2, LX)]
        M1_ref, env_out = single_particle_reference(env0, schedule)
        M1 = sector_matrix(perm, sign, env0, env_out, 1)
        ok1 = np.allclose(M1, M1_ref, atol=1e-12)
        M2 = sector_matrix(perm, sign, env0, env_out, 2)
        W = wedge_square(M1_ref)
        ok2 = np.allclose(M2, W, atol=1e-12)
        n1_mis += int(np.sum(~np.isclose(M1, M1_ref, atol=1e-12)))
        n2_mis += int(np.sum(~np.isclose(M2, W, atol=1e-12)))
        n2_states += M2.size
        if verbose:
            print(f"env={env0}: 1p == instrument rule: {ok1};  "
                  f"2p == Lambda^2(1p): {ok2}")
    return n1_mis, n2_mis, n2_states


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="run the composite certificate under the fresh-tape "
                         "(F1) schedule + a sign-rule mutation control; "
                         "writes results/w3c_composite_check_fresh.json")
    args = ap.parse_args()

    if not args.fresh:                    # committed production certificate
        n1, n2, _ = run_checks("production")
        assert n1 == 0 and n2 == 0
        print("\ncomposite sign coherence: CERTIFIED "
              "(2-particle sector is the exact antisymmetrised square)")
        return

    # --- fresh-tape (F1) rerun -------------------------------------------
    n1, n2, n2s = run_checks("fresh", ntrial=8)
    assert n1 == 0 and n2 == 0
    print("\ncomposite sign coherence under F1: CERTIFIED "
          f"({n2s} 2p sector matrix entries checked over 8 env draws)")

    # mutation control: corrupt ONE sign rule -- flip the L2 sign of the
    # two-carrier local config {c0, c1} (loc = 3). 1p states never populate
    # a two-carrier local config, so the 1p certificate must SURVIVE while
    # the 2p-vs-Lambda^2 coherence must FAIL: the check has teeth exactly
    # where the multi-particle sign structure lives.
    s16_mut = L2_SIGN16.copy()
    assert bin(3).count("1") == 2
    s16_mut[3] *= -1
    m1, m2, _ = run_checks("fresh", s16=s16_mut, verbose=False)
    print(f"mutation control (flip L2 sign at loc=3): 1p mismatches {m1} "
          f"(must be 0), 2p mismatches {m2} (must be > 0)")
    assert m1 == 0, "mutation leaked into the 1p sector (control invalid)"
    assert m2 > 0, "MUTATION CONTROL FAILED: corrupted sign rule not detected"

    out = {"LX": LX, "modes": NM, "dim": DIM, "schedule": "fresh",
           "n_env_draws": 8,
           "n_1p_mismatch": n1, "n_2p_mismatch": n2,
           "n_2p_entries_checked": n2s,
           "mutation_control": {"rule": "L2_SIGN16[3] flipped (config "
                                "{c0,c1}, two-carrier only)",
                                "n_1p_mismatch": m1, "n_2p_mismatch": m2,
                                "passes": bool(m1 == 0 and m2 > 0)},
           "note": "fresh = three phase-continuing streaming layers per "
                   "sub-step (odd-ring single-transposition convention); "
                   "production certificate unchanged and re-runnable "
                   "without --fresh"}
    pathlib.Path("results/w3c_composite_check_fresh.json").write_text(
        json.dumps(out, indent=1))
    print("written: results/w3c_composite_check_fresh.json")


if __name__ == "__main__":
    main()
