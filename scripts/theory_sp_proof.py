#!/usr/bin/env python
"""Proof-carrying certificate: [S, P] = 0 for the lifted coincube at EVERY
size and in EVERY sector, from block-local finite checks.

P is the particle-hole Majorana string P = c_{M-1} ... c_1 c_0 over the
M = 4L^3 carrier modes, c_m = a_m + a_m^dag, in the fixed segregated
(carrier-major, site-major) Jordan-Wigner ordering m = 4*site + channel.

THEOREM (proved in the note, machine-checked here).  Every layer of the
coincube cycle commutes with P, hence so does any product of layers (the
full cycle S, for every env realization, every L, on the full Fock space --
all carrier sectors at once).  The proof rests on:

  LEMMA (string factorization).  Let B be a fermion-parity-EVEN operator
  supported on a mode subset T (T may include env modes; let T_c = T
  intersected with the carrier modes).  Then, with P_S the sub-string of P
  over a carrier-mode set S in the induced order,
      P = sigma * P_{T_c} P_{T_c^complement},   sigma = +-1 a fixed
  reordering sign, and B commutes with every Majorana factor c_j, j not in
  T; hence
      [B, P] = sigma [B, P_{T_c}] P_{T_c^complement}
  and [B, P] = 0  <=>  [B, P_{T_c}] = 0: the block-local check suffices.

  Checked block-locally, once, independent of L:
   (C) L2 conversion: per env=1 site, the certified double Givens on the 4
       same-site carrier modes (+ env control mode).  Its JW spectator
       strings are INTERNAL to the site's contiguous 4-mode block (the
       conversion pairs are same-site channels), so it is parity-even with
       support {4s..4s+3} (+ env mode) -- Lemma applies; [B, P_4] = 0 is a
       16-dim (32-dim controlled) finite check.
   (T) L1 translation: a pure mode permutation pi with the canonical lift
       (inversion-parity sign rule).  For any such lift, U_pi P U_pi^{-1}
       = sign(pi) P; the coin-steered shift is, per channel, L^2 disjoint
       L-cycles, so sign(pi) = (-1)^{(L-1) L^2} = +1 for every L (L even
       => L^2 even; L odd => L-1 even): commutes at every size, wrap
       included (the wrap is part of the cycle structure).
   (E) L3 env streaming: supported on env modes only and parity-even =>
       commutes with every carrier Majorana, hence with P.
   (I) L4 imprint: carrier-parity-controlled env pair swap.  The carrier
       part is a parity projector (parity-even, diagonal); the env part is
       an adjacent-mode swap lift (string-free); [B, P_4] = 0 is a 64-dim
       finite check (both the permutation and the Givens env lift gauges).

  Size independence: (C), (E), (I) are products of DISJOINT copies of the
  same finite blocks at every L (disjoint parity-even blocks commute with
  P independently, by the Lemma, and with each other); (T) is covered by
  the sign formula for every L.  Sector coverage: the identity [S, P] = 0
  is an operator identity on the whole Fock space, so every carrier sector
  (vacuum, all N-carrier sectors, and their particle-hole duals) is covered
  at every size.  scripts/theory_3d_certs.py independently verifies the
  assembled cycle on explicit sectors at L = 2, 3, 4.

All checks below run in exact/1e-12 arithmetic; assertions run BEFORE the
results file is written.  Results -> results/theory_sp_proof.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_sp_proof.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pca3d.models.coincube import COIN_D, PERMS, SIGNS  # noqa: E402

SEED = 20260902
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_sp_proof.json"

PERM = [np.array(p) for p in PERMS]
SGN = [np.array(s) for s in SIGNS]
DD = [np.array(d) for d in COIN_D]

# involution channel pairs of each C_a (as in theory_3d_certs)
PAIRS = []
for a in range(3):
    seen = set()
    pr = []
    for c in range(4):
        c2 = int(PERM[a][c])
        if c not in seen:
            pr.append((min(c, c2), max(c, c2)))
            seen |= {c, c2}
    PAIRS.append(pr)


# ---------------------------------------------------------------- Fock tools

def fermion_ops(n):
    """JW annihilation operators a_0..a_{n-1}, 2^n dim, mode 0 outermost."""
    SZ = np.diag([1.0, -1])
    A = np.array([[0, 1], [0, 0.]])
    ops = []
    for m in range(n):
        mats = [SZ] * m + [A] + [np.eye(2)] * (n - m - 1)
        M = mats[0]
        for x in mats[1:]:
            M = np.kron(M, x)
        ops.append(M)
    return ops


def majorana_string(cs, order):
    """P over the listed mode indices, applied in increasing order:
    P = c_{last} ... c_{first} as a matrix product (theory_3d_certs
    majorana_sign convention)."""
    P = np.eye(cs[0].shape[0])
    for m in order:                       # apply in increasing order
        P = (cs[m] + cs[m].conj().T) @ P
    return P


def parity_op(cs):
    n = len(cs)
    N = sum(c.conj().T @ c for c in cs)
    return np.diag(np.round(np.exp(1j * np.pi * np.diag(N))).real)


def comm(A, B):
    return A @ B - B @ A


# ---------------------------------------------------- 1. string-factor lemma

def check_lemma(rng):
    """n = 8 modes, non-contiguous support T_c = {2, 3, 6}: the exact
    operator identity [B, P] = sigma [B, P_T] P_Tc for parity-even B, and
    its failure for parity-odd B (teeth)."""
    n = 8
    a = fermion_ops(n)
    c = [x + x.conj().T for x in a]
    T = [2, 3, 6]
    Tc = [m for m in range(n) if m not in T]
    P = majorana_string(a, range(n))
    PT = majorana_string(a, T)
    PTc = majorana_string(a, Tc)
    # sigma: inversion count between T and Tc positions in the full order
    inv = sum(1 for i in T for j in Tc if j < i)
    # moving the T factors (in order) to the RIGHT of the Tc factors:
    # P (increasing order) = sigma * PT-applied-after-PTc?  Fix by matrix id.
    sigma = None
    for s in (1, -1):
        if np.abs(P - s * PT @ PTc).max() < 1e-12:
            sigma = s
    res = {"sigma_found": sigma is not None,
           "sigma": sigma, "inversions_mod2": inv % 2}

    def random_even_odd(par):
        """random Hermitian-free operator on modes T with given parity."""
        B = np.zeros((2**n, 2**n), complex)
        monos = []
        for r in (1, 2, 3):
            for combo in itertools.combinations_with_replacement(T, r):
                for dag in itertools.product((0, 1), repeat=r):
                    monos.append((combo, dag))
        for combo, dag in monos:
            if len(combo) % 2 != par:
                continue
            coef = rng.normal() + 1j * rng.normal()
            M = np.eye(2**n)
            for m, d in zip(combo, dag):
                M = (a[m].conj().T if d else a[m]) @ M
            B = B + coef * M
        return B

    dev_id = dev_cj = 0.0
    for _ in range(4):
        B = random_even_odd(0)
        for j in Tc:
            dev_cj = max(dev_cj, np.abs(comm(B, c[j])).max())
        lhs = comm(B, P)
        rhs = sigma * comm(B, PT) @ PTc
        dev_id = max(dev_id, np.abs(lhs - rhs).max() / max(np.abs(lhs).max(), 1))
    res["even_commutes_with_offblock_majoranas"] = float(dev_cj)
    res["factorization_identity_dev"] = float(dev_id)
    # teeth: a parity-ODD B does NOT commute with off-block Majoranas
    Bodd = random_even_odd(1)
    worst = max(np.abs(comm(Bodd, c[j])).max() for j in Tc)
    res["odd_mutation_violates"] = float(worst)
    return res


# ------------------------------------------------- 2. conversion block (L2)

def conv_block_fock(a_axis):
    """16-dim Fock matrix of the env=1 conversion at one site from the
    STORED sign rule (signed pair swaps + spectator JW string), exactly the
    conv_state rule of theory_3d_certs on a 1-site lattice."""
    U = np.zeros((16, 16))
    for idx in range(16):
        # kron convention of fermion_ops: mode m occupied <-> bit (3 - m)
        occ = [m for m in range(4) if (idx >> (3 - m)) & 1]
        sg = 1
        for (c1, c2) in PAIRS[a_axis]:
            o1, o2 = c1 in occ, c2 in occ
            if o1 == o2:
                continue
            src, dst = (c1, c2) if o1 else (c2, c1)
            csrc = c1 if o1 else c2
            btw = sum(1 for m in occ if c1 < m < c2 and m != src)
            sg *= int(SGN[a_axis][csrc]) * (-1 if btw % 2 else 1)
            occ[occ.index(src)] = dst
            occ.sort()
        idx2 = sum(1 << (3 - m) for m in occ)
        U[idx2, idx] = sg
    return U


def check_conversion_controlled():
    """[W, P_carrier] = 0 for the certified env-controlled conversion,
    rebuilt in Fock space from the w3c_lift_check table."""
    from w3c_lift_check import controlled_l2
    a5 = fermion_ops(5)
    P5 = majorana_string(a5, range(4))
    par5 = parity_op(a5)
    # our kron basis: index j has mode m occupied iff bit (4 - m) of j...
    # -> build via explicit occupation decoding to stay convention-safe:
    # basis index in fermion_ops kron order: mode 0 outermost => index
    # j = sum_m n_m * 2^(4 - m).
    def to_our(mask_w3c):
        return sum(((mask_w3c >> m) & 1) << (4 - m) for m in range(5))

    out = {}
    dev_comm = dev_even = dev_perm = 0.0
    for ax in range(3):
        Utab = controlled_l2(ax)
        perm = np.argmax(np.abs(Utab), axis=0)
        sgn = Utab[perm, np.arange(32)]
        W = np.zeros((32, 32))
        for mask in range(32):
            W[to_our(int(perm[mask])), to_our(mask)] = sgn[mask]
        dev_perm = max(dev_perm, np.abs(np.abs(W).sum(0) - 1).max())
        dev_even = max(dev_even, np.abs(comm(W, par5)).max())
        dev_comm = max(dev_comm, np.abs(comm(W, P5)).max())
    out["signed_permutation_dev"] = float(dev_perm)
    out["parity_even_dev"] = float(dev_even)
    out["block_commutator_with_P_carrier"] = float(dev_comm)
    return out


# ------------------------------------------------- 3. translation layer (T)

def perm_parity(p):
    p = list(p)
    seen = [False] * len(p)
    par = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            ln += 1
        par ^= (ln - 1) & 1
    return par


def shift_perm(L, ax):
    """the coin-steered shift as a permutation of the 4L^3 carrier modes"""
    def site(x, y, z):
        return (x % L) + L * ((y % L) + L * (z % L))
    M = 4 * L**3
    p = np.empty(M, dtype=np.int64)
    for s in range(L**3):
        x, y, z = s % L, (s // L) % L, s // (L * L)
        for cch in range(4):
            d = int(DD[ax][cch])
            xyz = [x, y, z]
            xyz[ax] += d
            p[4 * s + cch] = 4 * site(*xyz) + cch
    return p


def check_translation(rng):
    res = {}
    # (a) canonical-lift conjugation law U_pi P U_pi^-1 = sign(pi) P on
    #     n = 6 modes, random permutations (both parities: teeth built in)
    n = 6
    a6 = fermion_ops(n)
    P6 = majorana_string(a6, range(n))
    dev = 0.0
    signs_seen = set()
    for _ in range(12):
        pi = rng.permutation(n)
        U = np.zeros((2**n, 2**n))
        for mask in range(2**n):
            occ = [m for m in range(n) if (mask >> (n - 1 - m)) & 1]
            img = [int(pi[m]) for m in occ]
            invs = sum(1 for i in range(len(img)) for j in range(i + 1, len(img))
                       if img[i] > img[j])
            mask2 = sum(1 << (n - 1 - m) for m in img)
            U[mask2, mask] = -1 if invs % 2 else 1
        s = 1 - 2 * perm_parity(pi)
        dev = max(dev, np.abs(U @ P6 - s * P6 @ U).max())
        signs_seen.add(s)
    res["conjugation_law_dev"] = float(dev)
    res["both_parities_exercised"] = sorted(signs_seen) == [-1, 1]

    # (b) the coin-steered shift permutation is EVEN at L = 2, 3, 4, 5
    pars = {}
    for L in (2, 3, 4, 5):
        pars[L] = [perm_parity(shift_perm(L, ax)) for ax in range(3)]
    res["shift_parity_by_L"] = {str(k): v for k, v in pars.items()}
    # (c) the closed form: per channel L^2 cycles of length L =>
    #     parity = (L-1) L^2 mod 2 = 0 for ALL L
    res["formula_even_all_L_2_to_64"] = all(
        ((L - 1) * L * L) % 2 == 0 for L in range(2, 65))
    # and the constructed permutations match the formula
    res["constructed_matches_formula"] = all(
        v == [0, 0, 0] for v in pars.values())
    return res


# ------------------------------------------------------ 4. env + L4 blocks

def check_env_and_imprint():
    res = {}
    # (E) env streaming block: 2 carrier + 2 env modes; swap lift on env
    a4 = fermion_ops(4)
    Pc = majorana_string(a4, range(2))          # carrier string
    # env swap (modes 2, 3): adjacent, string-free: |01>->|10>, |11>->-|11>
    S = np.zeros((16, 16))
    for mask in range(16):
        occ = [(mask >> (3 - m)) & 1 for m in range(4)]
        n2, n3 = occ[2], occ[3]
        occ2 = occ[:2] + [n3, n2]
        sg = -1 if (n2 and n3) else 1
        mask2 = sum(b << (3 - m) for m, b in enumerate(occ2))
        S[mask2, mask] = sg
    res["env_swap_parity_even"] = float(np.abs(comm(S, parity_op(a4))).max())
    res["env_swap_commutes_with_P_carrier"] = float(np.abs(comm(S, Pc)).max())

    # (I) imprint block: 4 carrier + 2 env modes (64-dim)
    a6 = fermion_ops(6)
    Pc4 = majorana_string(a6, range(4))
    par = parity_op(a6)
    ncar = sum(a6[m].conj().T @ a6[m] for m in range(4))
    podd = np.diag((np.round(np.diag(ncar).real).astype(int) % 2) == 1).astype(float)
    out = {}
    for gauge in ("perm", "givens"):
        W = np.zeros((64, 64))
        for mask in range(64):
            occ = [(mask >> (5 - m)) & 1 for m in range(6)]
            nA, nB = occ[4], occ[5]
            fire = (sum(occ[:4]) % 2) == 1
            occ2 = list(occ)
            sg = 1
            if fire:
                occ2[4], occ2[5] = nB, nA
                if gauge == "perm":
                    sg = -1 if (nA and nB) else 1
                else:                      # Givens: |01> -> -|10>, |11> -> +
                    sg = -1 if (nA == 0 and nB == 1) else 1
            mask2 = sum(b << (5 - m) for m, b in enumerate(occ2))
            W[mask2, mask] = sg
        out[gauge] = dict(
            parity_even=float(np.abs(comm(W, par)).max()),
            commutes_with_P_carrier=float(np.abs(comm(W, Pc4)).max()),
            projector_form=float(np.abs(
                W - (podd @ W @ podd + (np.eye(64) - podd)
                     @ np.eye(64) @ (np.eye(64) - podd))).max()))
    res["imprint"] = out

    # teeth: an OCCUPANCY-controlled (parity-odd control is impossible;
    # instead control on n_0 alone -- still parity-even, still commutes;
    # the true mutation is a WRONG env-swap sign) env swap with the |11>
    # sign dropped is parity-even but FAILS nothing here (it acts on env
    # only) -- the sign matters for the L4 *amplitude law*, not for [S,P].
    # The [S,P]-teeth mutation: a parity-ODD carrier factor:
    Wbad = a6[0] + a6[0].conj().T          # single Majorana: parity-odd
    res["odd_mutation_fails"] = float(np.abs(comm(Wbad, Pc4)).max())
    return res


# ----------------------------------------------------------------------------

def main():
    out = {"seed": SEED}
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out["lemma"] = check_lemma(rng)
    print(f"[{time.time() - t0:5.1f}s] lemma done")
    out["conversion"] = {}
    # plain (classical env = 1) block
    a4 = fermion_ops(4)
    P4 = majorana_string(a4, range(4))
    par4 = parity_op(a4)
    dev_comm = dev_givens = dev_even = 0.0
    for ax in range(3):
        B = conv_block_fock(ax)
        G = np.eye(16)
        for (c1, c2) in PAIRS[ax]:
            s12 = int(SGN[ax][c1])
            K = s12 * (a4[c2].conj().T @ a4[c1] - a4[c1].conj().T @ a4[c2])
            G = expm((np.pi / 2) * K) @ G
        dev_givens = max(dev_givens, np.abs(B - G).max())
        dev_even = max(dev_even, np.abs(comm(B, par4)).max())
        dev_comm = max(dev_comm, np.abs(comm(B, P4)).max())
    out["conversion"]["stored_rule_equals_givens_product"] = float(dev_givens)
    out["conversion"]["parity_even_dev"] = float(dev_even)
    out["conversion"]["block_commutator_with_P4"] = float(dev_comm)
    out["conversion"]["pairs_same_site"] = all(
        0 <= c1 < c2 <= 3 for ax in range(3) for (c1, c2) in PAIRS[ax])
    out["conversion_controlled"] = check_conversion_controlled()
    print(f"[{time.time() - t0:5.1f}s] conversion blocks done")
    out["translation"] = check_translation(rng)
    print(f"[{time.time() - t0:5.1f}s] translation done")
    out["env_imprint"] = check_env_and_imprint()
    print(f"[{time.time() - t0:5.1f}s] env + imprint done")

    # ---------------- assertions BEFORE the results file is written ---------
    lm = out["lemma"]
    assert lm["sigma_found"]
    assert lm["even_commutes_with_offblock_majoranas"] < 1e-12
    assert lm["factorization_identity_dev"] < 1e-10
    assert lm["odd_mutation_violates"] > 0.5          # teeth
    cv = out["conversion"]
    assert cv["stored_rule_equals_givens_product"] < 1e-12
    assert cv["parity_even_dev"] < 1e-12
    assert cv["block_commutator_with_P4"] < 1e-12
    assert cv["pairs_same_site"]                      # JW strings internal
    cc = out["conversion_controlled"]
    assert cc["signed_permutation_dev"] < 1e-12
    assert cc["parity_even_dev"] < 1e-12
    assert cc["block_commutator_with_P_carrier"] < 1e-12
    tr = out["translation"]
    assert tr["conjugation_law_dev"] < 1e-12
    assert tr["both_parities_exercised"]              # teeth
    assert tr["formula_even_all_L_2_to_64"]
    assert tr["constructed_matches_formula"]
    ei = out["env_imprint"]
    assert ei["env_swap_parity_even"] < 1e-12
    assert ei["env_swap_commutes_with_P_carrier"] < 1e-12
    for gauge in ("perm", "givens"):
        g = ei["imprint"][gauge]
        assert g["parity_even"] < 1e-12
        assert g["commutes_with_P_carrier"] < 1e-12
    assert ei["odd_mutation_fails"] > 0.5             # teeth
    print("all headline assertions passed: [S, P] = 0 per layer, "
          "block-locally, size-independently")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
