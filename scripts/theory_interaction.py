#!/usr/bin/env python
"""Phase-I theory: the imprint interaction (L4) of the coincube.

Machine checks for docs/notes/theory-interaction.md; all headline claims
re-asserted at the end of every run. Results -> results/theory_interaction.json.

Model (docs/04-phase-i-plan.md): once per cycle, at sites where an autonomous
Bernoulli(g) field iota = 1 AND the site carrier fermion parity is odd, swap
the env_x and env_y bits at that site (controlled two-mode env Givens).

Theory summary being checked:
  A. Local algebra: per one-sided fired imprint the ensemble amplitude /
     env-overlap factor is EXACTLY 1 - 2q(1-q) (sign-average and orthogonality
     derivations agree); parity control blocks coincident carriers and makes
     the vacuum imprint-transparent.
  B. Closed amplitude law (fresh-read/annealed): U_g(k) = (1-2gq(1-q)) U(k):
     Gamma_int = 2q(1-q) g per cycle, O(g); omega0, v untouched at this order.
     Substrate MC quantifies the recross (read-imprint-reread) residuals.
  B2. GO/NO-GO: the fixed x<->y swap breaks cubic symmetry at O(g) (measured);
     the rotating/randomized swap pair restores it.
  C. Doubled-space superoperator (I3 theory): per-sub-step factorized
     T (x) conj T with r = 0 contact correlation; L4 factor (1-2gq(1-q))^2 off
     the diagonal, exemption at r = 0; trace mode exactly 1; classical sector
     exactly g-independent; amplitude branch damped by (1-2gq(1-q))^2.
  D. Two-carrier parity blocking: coincident carriers never imprint ->
     positive contact contribution to C2, magnitude ~ 2gq(1-q) per
     coincidence-cycle (kinematically suppressed: no two channels co-move).

Run:  PYTHONPATH=src .venv/bin/python scripts/theory_interaction.py
      (--quick shrinks the MC ensembles ~4x)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pca3d.models.coincube import COIN_C, COIN_D, PERMS, SIGNS, annealed_u  # noqa: E402

SEED = 20260901
RESULTS = Path(__file__).resolve().parent.parent / "results" / "theory_interaction.json"

PERM = [np.array(p) for p in PERMS]
SGN = [np.array(s) for s in SIGNS]
DD = [np.array(d) for d in COIN_D]
C4 = [np.array(c) for c in COIN_C]


# ----------------------------------------------------------------------------
# Part A: local imprint algebra (exact)
# ----------------------------------------------------------------------------

def part_A():
    from scipy.linalg import expm
    res = {}
    q = 0.15

    def fermion_ops(nm):
        ops = []
        SZ = np.diag([1.0, -1])
        for m in range(nm):
            mats = [SZ] * m + [np.array([[0, 1], [0, 0.]])] + [np.eye(2)] * (nm - m - 1)
            M = mats[0]
            for x in mats[1:]:
                M = np.kron(M, x)
            ops.append(M)
        return ops

    # 2-env-mode Givens: sign table and the imprint factor, both derivations
    ex, ey = fermion_ops(2)
    G = expm((np.pi / 2) * (ex.T @ ey - ey.T @ ex))
    # basis order (n_x n_y): |00>,|01>,|10>,|11> with kron order (x major)
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    st = {}
    for b in idx:
        v = np.zeros(4); v[idx[b]] = 1
        w = G @ v
        j = int(np.argmax(np.abs(w)))
        st[b] = (list(idx)[j], float(w[j]))
    res["givens_table"] = {str(k): [str(v[0]), v[1]] for k, v in st.items()}
    p = {(0, 0): (1 - q)**2, (0, 1): q * (1 - q), (1, 0): (1 - q) * q, (1, 1): q**2}
    f_sign = sum(p[b] * st[b][1] for b in p)                       # vacuum-bra
    f_ovl = sum(p[b] * st[b][1] * (st[b][0] == b) for b in p)      # doubled
    res["factor_sign_average"] = float(f_sign)
    res["factor_overlap"] = float(f_ovl)
    res["factor_theory"] = float(1 - 2 * q * (1 - q))

    # parity-controlled layer on 4 carrier + 2 env modes (64-dim Fock)
    ops = fermion_ops(6)
    car, (Ex, Ey) = ops[:4], ops[4:]
    N = sum(o.T @ o for o in car)
    Podd = np.zeros((64, 64))
    for s in range(64):
        v = np.zeros(64); v[s] = 1
        if int(np.round(v @ N @ v)) % 2 == 1:
            Podd[s, s] = 1
    W = expm((np.pi / 2) * Podd @ (Ex.T @ Ey - Ey.T @ Ex))
    perm_ok = bool(np.allclose(np.abs(np.abs(W) - np.round(np.abs(W))), 0, atol=1e-12))
    Ntot = N + Ex.T @ Ex + Ey.T @ Ey
    par = expm(1j * np.pi * Ntot).real
    res["W_signed_permutation"] = perm_ok
    res["W_parity_even"] = float(np.abs(W @ par - par @ W).max())
    # sector checks: 0 carriers -> identity; 2 carriers (even) -> identity;
    # 1 carrier -> the swap fires
    P0 = np.eye(64) - Podd
    res["W_identity_on_even_parity"] = float(np.abs(P0 @ (W - np.eye(64)) @ P0).max())
    v1 = car[0].T @ Ex.T @ np.zeros(64)  # placeholder replaced below
    v = np.zeros(64); v[0] = 1
    s1 = car[0].T @ (Ex.T @ v)           # one carrier + env_x occupied
    w1 = W @ s1
    fired = float(np.abs(w1 - car[0].T @ (Ey.T @ v)).max())  # -> env_y (sign +-1)
    res["W_fires_on_one_carrier"] = bool(
        abs(abs(np.vdot(car[0].T @ (Ey.T @ v), w1)) - 1) < 1e-12)
    return res


# ----------------------------------------------------------------------------
# substrate MC machinery
# ----------------------------------------------------------------------------

def _stream(field, L, ax, origin):
    idxA = (np.arange(0, L, 2) + origin) % L
    idxB = (idxA + 1) % L
    slA = [slice(None)] * 4
    slB = [slice(None)] * 4
    slA[1 + ax] = idxA
    slB[1 + ax] = idxB
    tmp = field[tuple(slA)].copy()
    field[tuple(slA)] = field[tuple(slB)]
    field[tuple(slB)] = tmp


def run_1p(E, L, T, q, c0, g, seed, rotate, kvecs):
    r_env = default_rng(seed)
    r_iota = default_rng(seed + 777)
    r_pair = default_rng(seed + 555)
    env = [(r_env.random((E, L, L, L)) < q).astype(np.int8) for _ in range(3)]
    pos = np.zeros((E, 3), np.int64)
    site = np.zeros((E, 3), np.int64)
    coin = np.full(E, c0, np.int64)
    sign = np.ones(E)
    ii = np.arange(E)
    out = np.zeros((T, len(kvecs), 4), complex)
    for t in range(T):
        for a in range(3):
            for sub in range(2):
                b = env[a][ii, site[:, 0], site[:, 1], site[:, 2]]
                cv = b == 1
                old = coin[cv]
                sign[cv] *= SGN[a][old]
                coin[cv] = PERM[a][old]
                step = DD[a][coin]
                pos[:, a] += step
                site[:, a] = (site[:, a] + step) % L
                _stream(env[a], L, (a + 1) % 3, (2 * a + sub) % 2)
        u = r_iota.random(E)
        fire = u < g
        pr = r_pair.integers(0, 3, E) if rotate else np.zeros(E, np.int64)
        if g > 0:
            for p, (A, B) in enumerate(((0, 1), (1, 2), (2, 0))):
                m = fire & (pr == p)
                if m.any():
                    sel = np.where(m)[0]
                    sx, sy, sz = site[sel, 0], site[sel, 1], site[sel, 2]
                    bA = env[A][sel, sx, sy, sz].copy()
                    bB = env[B][sel, sx, sy, sz].copy()
                    sign[sel] *= np.where((bA == 0) & (bB == 1), -1.0, 1.0)
                    env[A][sel, sx, sy, sz] = bB
                    env[B][sel, sx, sy, sz] = bA
        for ik, kv in enumerate(kvecs):
            ph = sign * np.exp(1j * (pos @ kv))
            for cc in range(4):
                out[t, ik, cc] = ph[coin == cc].sum() / E
    return out


def pair_projector(kv, q):
    Uk = annealed_u(np.array(kv, float), q)
    w, V = np.linalg.eig(Uk)
    Vi = np.linalg.inv(V)
    up = [i for i in range(4) if w[i].imag > 0]
    return Vi[up, :], V[:, up]


def mc_series(E, L, T, q, g, rotate, kvecs, nch, seed0):
    """chunked node-pair-projected G series; returns (nch, T, K) complex"""
    projs = [pair_projector(kv, q) for kv in kvecs]
    series = np.zeros((nch, T, len(kvecs)), complex)
    for ch in range(nch):
        Gt = np.zeros((T, len(kvecs), 4, 4), complex)
        for c0 in range(4):
            Gt[:, :, :, c0] = run_1p(E, L, T, q, c0, g, seed0 + ch, rotate, kvecs)
        for tt in range(T):
            for ik, (Pl, Pr) in enumerate(projs):
                series[ch, tt, ik] = np.trace(Pl @ Gt[tt, ik] @ Pr)
    return series


# ----------------------------------------------------------------------------
# Part B: k = 0 damping law + recross drift; Part B2: anisotropy go/no-go
# ----------------------------------------------------------------------------

def part_B(quick):
    res = {}
    L, T, q, g = 10, 3, 0.15, 0.3
    E = 5000 if quick else 20000
    nch = 3
    kvecs = [np.zeros(3)]
    s0 = mc_series(E, L, T, q, 0.0, False, kvecs, nch, 1000)
    sf = mc_series(E, L, T, q, g, False, kvecs, nch, 1000)
    sr = mc_series(E, L, T, q, g, True, kvecs, nch, 1000)
    th = 1 - 2 * g * q * (1 - q)
    res["factor_theory"] = float(th)
    for name, s in (("fixed", sf), ("rotating", sr)):
        ratios = (s / s0)[:, :, 0]                     # (nch, T)
        fpc = np.abs(ratios) ** (1 / np.arange(1, T + 1))
        ph = np.angle(ratios) / np.arange(1, T + 1)
        res[name] = dict(
            damping_per_cycle=[float(x) for x in fpc.mean(0)],
            damping_err=[float(x) for x in (fpc.std(0) / np.sqrt(nch))],
            phase_per_cycle=[float(x) for x in ph.mean(0)],
            phase_err=[float(x) for x in (ph.std(0) / np.sqrt(nch))],
            t1_residual=float(fpc.mean(0)[0] - th),
            drift_t3_minus_t1=float(fpc.mean(0)[-1] - fpc.mean(0)[0]))
    return res


def part_B2(quick):
    res = {}
    L, T, q, g = 10, 3, 0.15, 0.3
    E = 12000 if quick else 20000   # B2 needs statistics for the sigma tests
    nch = 4
    kmag = 0.35
    kvecs = [kmag * np.array(v, float) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    kvecs.append(kmag * np.array((1, 1, 1)) / np.sqrt(3))
    s0 = mc_series(E, L, T, q, 0.0, False, kvecs, nch, 3000)
    sf = mc_series(E, L, T, q, g, False, kvecs, nch, 3000)
    sr = mc_series(E, L, T, q, g, True, kvecs, nch, 3000)
    res["kmag"] = kmag
    for name, s in (("fixed", sf), ("rotating", sr)):
        r3 = (s / s0)[:, T - 1, :]
        mag = np.abs(r3)
        entry = dict(
            r_mod=[float(x) for x in mag.mean(0)],
            r_mod_err=[float(x) for x in (mag.std(0) / np.sqrt(nch))],
            labels=["x", "y", "z", "111"])
        for lab, i in (("xz", 0), ("yz", 1)):
            d = mag[:, i] - mag[:, 2]
            entry[f"aniso_{lab}"] = float(d.mean())
            entry[f"aniso_{lab}_err"] = float(d.std() / np.sqrt(nch))
            entry[f"aniso_{lab}_sigmas"] = float(
                abs(d.mean()) / max(d.std() / np.sqrt(nch), 1e-12))
        res[name] = entry
    return res


# ----------------------------------------------------------------------------
# Part C: the doubled-space superoperator (annealed, exact on a small ring)
# ----------------------------------------------------------------------------

def part_C():
    res = {}
    L, q, g = 4, 0.15, 0.3
    T4 = [(1 - q) * np.eye(4) + q * c for c in C4]
    f1 = 1 - 2 * g * q * (1 - q)

    def apply_cycle(psi, contact, l4, gv):
        f = 1 - 2 * gv * q * (1 - q)
        for a in range(3):
            for _ in range(2):
                new = np.einsum('ij,kl,jl...->ik...', T4[a], T4[a], psi)
                if contact:
                    new[:, :, 0, 0, 0] = (1 - q) * psi[:, :, 0, 0, 0] + q * np.einsum(
                        'ij,kl,jl->ik', C4[a], C4[a], psi[:, :, 0, 0, 0])
                psi = new
                out = np.empty_like(psi)
                for c in range(4):
                    for cp in range(4):
                        s = int(DD[a][c] - DD[a][cp])
                        out[c, cp] = np.roll(psi[c, cp], s, axis=a)
                psi = out
        if l4:
            mask = np.ones((L, L, L)) * f**2
            if contact:
                mask[0, 0, 0] = 1.0
            psi = psi * mask[None, None]
        return psi

    def build(contact, l4, gv):
        dim = 16 * L**3
        M = np.zeros((dim, dim))
        for idx in range(dim):
            e = np.zeros(dim)
            e[idx] = 1
            M[:, idx] = apply_cycle(e.reshape(4, 4, L, L, L), contact, l4, gv).ravel()
        return M

    # gate 1: contact-off operator == factorized f^2 lam lambar exactly
    M_fac = build(False, True, g)
    ks = 2 * np.pi * np.arange(L) / L
    prod = []
    for kx in ks:
        for ky in ks:
            for kz in ks:
                lam = np.linalg.eigvals(annealed_u(np.array([kx, ky, kz]), q))
                prod += [f1**2 * a * np.conj(b) for a in lam for b in lam]
    wa = list(np.linalg.eigvals(M_fac))
    dev = 0.0
    for x in prod:
        i = int(np.argmin(np.abs(np.array(wa) - x)))
        dev = max(dev, abs(wa[i] - x))
        wa.pop(i)
    res["gate_factorization_dev"] = float(dev)

    # full operator: trace mode exactly 1; classical sector g-independent;
    # coherence branch damped by f^2
    M_full = build(True, True, g)
    M_g0 = build(True, False, 0.0)
    tr = np.zeros((4, 4, L, L, L))
    for c in range(4):
        tr[c, c, 0, 0, 0] = 1
    res["trace_mode_dev"] = float(np.abs(M_full.T @ tr.ravel() - tr.ravel()).max())
    w_full = np.sort(np.abs(np.linalg.eigvals(M_full)))[::-1]
    w_g0 = np.sort(np.abs(np.linalg.eigvals(M_g0)))[::-1]
    res["top_eigs_g03"] = [float(x) for x in w_full[:5]]
    res["top_eigs_g0"] = [float(x) for x in w_g0[:5]]
    res["classical_sector_g_independent_dev"] = float(
        np.abs(w_full[:5] - w_g0[:5]).max())
    # amplitude branch: nearest eigenvalue to |lam0|^2 (g=0) and f^2|lam0|^2
    lam = np.linalg.eigvals(annealed_u(np.zeros(3), q))
    l0 = lam[np.argmax(lam.imag)]
    a0 = w_g0[np.argmin(np.abs(w_g0 - abs(l0)**2))]
    ag = w_full[np.argmin(np.abs(w_full - f1**2 * abs(l0)**2))]
    res["amplitude_branch"] = dict(
        g0=float(a0), g03=float(ag), ratio=float(ag / a0), f2=float(f1**2))
    return res


# ----------------------------------------------------------------------------
# Part D: two-carrier parity blocking (paired common-random MC)
# ----------------------------------------------------------------------------

def part_D(quick):
    res = {}
    L, T, q, g = 10, 3, 0.15, 0.3
    E = 20000 if quick else 60000

    def run2(sep, seed):
        r_env = default_rng(seed)
        r_iota = default_rng(seed + 777)
        env = [(r_env.random((E, L, L, L)) < q).astype(np.int8) for _ in range(3)]
        pos = np.zeros((E, 2, 3), np.int64)
        pos[:, 1, 0] = sep
        coin = np.zeros((E, 2), np.int64)
        coin[:, 1] = 2
        ii = np.arange(E)
        nfire = np.zeros(E)
        sprod = np.ones(E)
        ncoin = np.zeros(E)
        for t in range(T):
            for a in range(3):
                for sub in range(2):
                    for i in (0, 1):
                        b = env[a][ii, pos[:, i, 0], pos[:, i, 1], pos[:, i, 2]]
                        other = 1 - i
                        same = np.all(pos[:, i] == pos[:, other], axis=1)
                        blocked = same & (coin[:, other] == PERM[a][coin[:, i]])
                        cv = (b == 1) & ~blocked
                        coin[cv, i] = PERM[a][coin[cv, i]]
                    pos[:, :, a] = (pos[:, :, a] + DD[a][coin]) % L
                    _stream(env[a], L, (a + 1) % 3, (2 * a + sub) % 2)
            same = np.all(pos[:, 0] == pos[:, 1], axis=1)
            ncoin += same
            for i in (0, 1):
                u = r_iota.random(E)
                fire = (u < g) & ~same       # parity even at coincidence
                sel = np.where(fire)[0]
                if len(sel):
                    sx, sy, sz = pos[sel, i, 0], pos[sel, i, 1], pos[sel, i, 2]
                    bA = env[0][sel, sx, sy, sz].copy()
                    bB = env[1][sel, sx, sy, sz].copy()
                    sprod[sel] *= np.where((bA == 0) & (bB == 1), -1.0, 1.0)
                    env[0][sel, sx, sy, sz] = bB
                    env[1][sel, sx, sy, sz] = bA
                    nfire[sel] += 1
        return nfire, sprod, ncoin

    n0, s0, c0 = run2(0, 42)
    n5, s5, c5 = run2(5, 42)          # common random numbers -> paired
    d = n5 - n0
    res["fire_deficit_coincident"] = float(d.mean())
    res["fire_deficit_err"] = float(d.std() / np.sqrt(E))
    res["fire_deficit_sigmas"] = float(d.mean() / max(d.std() / np.sqrt(E), 1e-12))
    res["coincidence_cycles"] = dict(sep0=float(c0.mean()), sep5=float(c5.mean()))
    res["sign_product"] = dict(sep0=float(s0.mean()), sep5=float(s5.mean()))
    res["blocking_prediction"] = float(2 * g * c0.mean())   # ~ fire deficit
    return res


# ----------------------------------------------------------------------------
# Part E: resolution of the I3 damping discrepancy (coherent vs mixture vs
# walker; lift-gauge dependence; the 1D-substrate companion lock)
# ----------------------------------------------------------------------------

def part_E(quick):
    """Exact compact 1p simulator of the i2_connected 1D substrate (ring LX,
    4 channels + 2 env species, certified L2 tables, streaming = transposition
    per sub-step, L4 at cycle end), for BOTH L4 lift gauges:

      permutation lift (i2_connected): |01>->+|10>, |10>->+|01>, |11>->-|11>
      Givens rotation  (the plan):     |01>->-|10>, |10>->+|01>, |11>->+|11>

    and THREE correlators: coherent (sqrt-p env bra), mixture (per-basis-env
    bra = env-return demanded), flat/walker (E[sigma delta], no env bra).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from w3c_lift_check import controlled_l2
    res = {}
    Q = 0.15
    U = controlled_l2(0)
    perm = np.argmax(np.abs(U), axis=0)
    sgn = U[perm, np.arange(32)]
    P1 = np.zeros(4, np.int64)
    S1 = np.zeros(4, np.int64)
    for c in range(4):
        t = int(perm[(1 << c) | 16])
        P1[c] = int(np.log2(t & 15))
        S1[c] = int(sgn[(1 << c) | 16])
    D0 = DD[0]

    def run_basis(L, T, e_init, gauge, iota):
        x, c, sig = 0, 0, 1.0
        env = list(e_init)
        out = []
        for t in range(T):
            for o in (0, 1):
                if env[2 * x] == 1:
                    sig *= S1[c]
                    c = int(P1[c])
                x = (x + int(D0[c])) % L
                for sp in (0, 1):
                    a2, b2 = 2 * (o % L) + sp, 2 * ((o + 1) % L) + sp
                    env[a2], env[b2] = env[b2], env[a2]
            if iota[x] == 1:
                bA, bB = env[2 * x], env[2 * x + 1]
                if gauge == "perm":
                    if bA != bB:
                        env[2 * x], env[2 * x + 1] = bB, bA
                    elif bA == 1:
                        sig *= -1.0
                else:
                    if (bA, bB) == (1, 0):
                        env[2 * x], env[2 * x + 1] = 0, 1
                    elif (bA, bB) == (0, 1):
                        env[2 * x], env[2 * x + 1] = 1, 0
                        sig *= -1.0
            out.append((x, c, sig, tuple(env)))
        return out

    def env_free(L, T, e_init):
        env = list(e_init)
        for t in range(T):
            for o in (0, 1):
                for sp in (0, 1):
                    a2, b2 = 2 * (o % L) + sp, 2 * ((o + 1) % L) + sp
                    env[a2], env[b2] = env[b2], env[a2]
        return tuple(env)

    def p_of(env):
        n1 = sum(env)
        return Q**n1 * (1 - Q)**(len(env) - n1)

    def correlators(L, T, gauge, iota):
        Gc = [np.zeros((L, 4)) for _ in range(T)]
        Gm = [np.zeros((L, 4)) for _ in range(T)]
        Gf = [np.zeros((L, 4)) for _ in range(T)]
        for eidx in range(1 << (2 * L)):
            e = [(eidx >> i) & 1 for i in range(2 * L)]
            pe = p_of(e)
            traj = run_basis(L, T, e, gauge, iota)
            for t in range(T):
                x, c, sig, envt = traj[t]
                if envt == env_free(L, t + 1, e):
                    Gm[t][x, c] += pe * sig
                Gc[t][x, c] += np.sqrt(pe * p_of(envt)) * sig
                Gf[t][x, c] += pe * sig
        return Gc, Gm, Gf

    L, T = 3, 4
    pats = list(itertools.product((0, 1), repeat=L))
    for gauge in ("perm", "givens"):
        data = {p: correlators(L, T, gauge, p) for p in pats}
        base = data[(0,) * L][0]
        entry = {}
        for gval in (0.1, 0.2):
            row = {}
            for obj in range(3):
                avg = None
                for p in pats:
                    w = np.prod([gval if b else 1 - gval for b in p])
                    o = data[p][obj]
                    avg = [w * x for x in o] if avg is None else [
                        a + w * x for a, x in zip(avg, o)]
                pc = []
                for t in range(T):
                    r = float((avg[t] * base[t]).sum()) / float(
                        (base[t] * base[t]).sum())
                    pc.append(float(np.sign(r) * abs(r)**(1 / (t + 1))))
                row[["coherent", "mixture", "flat"][obj]] = pc
            entry[str(gval)] = row
        res[gauge] = entry

    # headline identities and laws
    perm_01 = res["perm"]["0.1"]
    res["identity_flat_equals_coherent_dev"] = float(max(
        abs(a - b) for gauge in ("perm", "givens") for gv in ("0.1", "0.2")
        for a, b in zip(res[gauge][gv]["flat"], res[gauge][gv]["coherent"])))
    res["reproduces_i3_damping_coherent"] = perm_01["coherent"]  # cf. script
    res["mixture_law_1_minus_gq"] = dict(
        g01=dict(measured=perm_01["mixture"][0], law=float(1 - 0.1 * Q)),
        g02=dict(measured=res["perm"]["0.2"]["mixture"][0], law=float(1 - 0.2 * Q)))
    res["coherent_givens_t1_law_1_minus_2gq"] = dict(
        measured=res["givens"]["0.1"]["coherent"][0], law=float(1 - 2 * 0.1 * Q))
    res["ratio_059_explained"] = dict(
        measured=float((1 - perm_01["mixture"][0]) / (2 * 0.1 * Q * (1 - Q))),
        theory_1_over_2_1mq=float(1 / (2 * (1 - Q))))

    # the companion lock at t = 1: P(b0 = 1 at the carrier's site) and
    # P(b0 == last-read bit), exact over all env configs
    def lock_stats(L):
        w1 = mt = 0.0
        for eidx in range(1 << (2 * L)):
            e = [(eidx >> i) & 1 for i in range(2 * L)]
            pe = p_of(e)
            x, c, sig = 0, 0, 1.0
            env = list(e)
            reads = []
            for o in (0, 1):
                reads.append(env[2 * x])
                if env[2 * x] == 1:
                    c = int(P1[c])
                x = (x + int(D0[c])) % L
                for sp in (0, 1):
                    a2, b2 = 2 * (o % L) + sp, 2 * ((o + 1) % L) + sp
                    env[a2], env[b2] = env[b2], env[a2]
            if env[2 * x] == 1:
                w1 += pe
            if env[2 * x] == reads[-1]:
                mt += pe
        return w1, mt

    for LL in (3, 4, 5):
        w1, mt = lock_stats(LL)
        res[f"lock_L{LL}"] = dict(P_b0_1=float(w1), P_b0_eq_lastread=float(mt))

    # 3D walker (rotating pair): both lifts vs their pristine-pair laws
    L3, T3, q3, g3 = 10, 3, 0.15, 0.3
    E = 5000 if quick else 20000
    nch = 3
    U0 = annealed_u(np.zeros(3), q3)
    w, V = np.linalg.eig(U0)
    Vi = np.linalg.inv(V)
    up = [i for i in range(4) if w[i].imag > 0]
    Pl, Pr = Vi[up, :], V[:, up]

    def run3d(E, c0, g, seed, lift):
        r_env = default_rng(seed)
        r_iota = default_rng(seed + 777)
        r_pair = default_rng(seed + 555)
        env = [(r_env.random((E, L3, L3, L3)) < q3).astype(np.int8)
               for _ in range(3)]
        site = np.zeros((E, 3), np.int64)
        coin = np.full(E, c0, np.int64)
        sign = np.ones(E)
        ii = np.arange(E)
        out = np.zeros((T3, 4), complex)
        for t in range(T3):
            for a in range(3):
                for sub in range(2):
                    b = env[a][ii, site[:, 0], site[:, 1], site[:, 2]]
                    cv = b == 1
                    old = coin[cv]
                    sign[cv] *= SGN[a][old]
                    coin[cv] = PERM[a][old]
                    site[:, a] = (site[:, a] + DD[a][coin]) % L3
                    _stream(env[a], L3, (a + 1) % 3, (2 * a + sub) % 2)
            fire = r_iota.random(E) < g
            pr = r_pair.integers(0, 3, E)
            for p, (A, B) in enumerate(((0, 1), (1, 2), (2, 0))):
                m = fire & (pr == p)
                if m.any():
                    sel = np.where(m)[0]
                    sx, sy, sz = site[sel, 0], site[sel, 1], site[sel, 2]
                    bA = env[A][sel, sx, sy, sz].copy()
                    bB = env[B][sel, sx, sy, sz].copy()
                    if lift == "givens":
                        sign[sel] *= np.where((bA == 0) & (bB == 1), -1.0, 1.0)
                    else:
                        sign[sel] *= np.where((bA == 1) & (bB == 1), -1.0, 1.0)
                    env[A][sel, sx, sy, sz] = bB
                    env[B][sel, sx, sy, sz] = bA
            for cc in range(4):
                out[t, cc] = sign[coin == cc].sum() / E
        return out

    series = {}
    for lift, gv in (("none", 0.0), ("givens", g3), ("perm", g3)):
        ser = np.zeros((nch, T3), complex)
        for ch in range(nch):
            Gt = np.zeros((T3, 4, 4), complex)
            for c0 in range(4):
                Gt[:, :, c0] = run3d(E, c0, gv, 1000 + ch,
                                     lift if lift != "none" else "givens")
            for tt in range(T3):
                ser[ch, tt] = np.trace(Pl @ Gt[tt] @ Pr)
        series[lift] = ser
    res["walker_3d_rotating"] = {}
    for lift, law in (("givens", 1 - 2 * g3 * q3 * (1 - q3)),
                      ("perm", 1 - 2 * g3 * q3 * q3)):
        r = series[lift] / series["none"]
        fpc = np.abs(r)**(1 / np.arange(1, T3 + 1))
        res["walker_3d_rotating"][lift] = dict(
            per_cycle=[float(x) for x in fpc.mean(0)],
            err=[float(x) for x in fpc.std(0) / np.sqrt(nch)],
            law=float(law), t1_residual=float(fpc.mean(0)[0] - law))
    return res


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = {"seed": SEED, "quick": args.quick}
    t0 = time.time()
    for name, fn in [("A", part_A), ("B", lambda: part_B(args.quick)),
                     ("B2", lambda: part_B2(args.quick)), ("C", part_C),
                     ("D", lambda: part_D(args.quick)),
                     ("E", lambda: part_E(args.quick))]:
        t = time.time()
        out[name] = fn()
        out[name + "_seconds"] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.1f}s] part {name} done")

    A = out["A"]
    assert abs(A["factor_sign_average"] - A["factor_theory"]) < 1e-14
    assert abs(A["factor_overlap"] - A["factor_theory"]) < 1e-14
    assert A["W_signed_permutation"] and A["W_parity_even"] < 1e-12
    assert A["W_identity_on_even_parity"] < 1e-12 and A["W_fires_on_one_carrier"]
    B = out["B"]
    for name in ("fixed", "rotating"):
        assert abs(B[name]["t1_residual"]) < 0.015          # law to 1.5 %
    assert abs(B["rotating"]["t1_residual"]) < abs(B["fixed"]["t1_residual"])
    assert B["fixed"]["drift_t3_minus_t1"] > 3 * B["fixed"]["damping_err"][-1]
    assert abs(B["rotating"]["phase_per_cycle"][-1]) < abs(
        B["fixed"]["phase_per_cycle"][-1])
    B2 = out["B2"]
    sig_hi, sig_lo = (2.0, 3.5) if args.quick else (4.0, 3.0)
    assert B2["fixed"]["aniso_xz_sigmas"] > sig_hi
    assert B2["fixed"]["aniso_yz_sigmas"] > sig_hi
    assert B2["rotating"]["aniso_xz_sigmas"] < sig_lo
    assert B2["rotating"]["aniso_yz_sigmas"] < sig_lo
    C = out["C"]
    assert C["gate_factorization_dev"] < 1e-12
    assert C["trace_mode_dev"] < 1e-12
    assert C["classical_sector_g_independent_dev"] < 1e-12
    assert abs(C["amplitude_branch"]["ratio"] - C["amplitude_branch"]["f2"]) < 0.02
    Dd = out["D"]
    assert Dd["fire_deficit_sigmas"] > 5
    assert Dd["coincidence_cycles"]["sep5"] < 1e-9
    Ee = out["E"]
    # walker/flat contraction IS the coherent correlator (equal densities)
    assert Ee["identity_flat_equals_coherent_dev"] < 1e-12
    # reproduces the i3_damping.py coherent numbers (perm lift, g = 0.1)
    ref = [0.999997, 0.992268, 0.990815, 0.987668]
    assert max(abs(a - b) for a, b in
               zip(Ee["reproduces_i3_damping_coherent"], ref)) < 2e-6
    # mixture law 1 - g q (the 0.59 resolution) and coherent-Givens 1 - 2 g q
    for k, d in Ee["mixture_law_1_minus_gq"].items():
        assert abs(d["measured"] - d["law"]) < 2e-4
    cg = Ee["coherent_givens_t1_law_1_minus_2gq"]
    assert abs(cg["measured"] - cg["law"]) < 2e-4
    r59 = Ee["ratio_059_explained"]
    assert abs(r59["measured"] - r59["theory_1_over_2_1mq"]) < 0.01
    # the 1D companion lock: P(b0 = 1) = q^3 at L = 3, q^2 at L = 5 (exact)
    assert abs(Ee["lock_L3"]["P_b0_1"] - 0.15**3) < 1e-12
    assert abs(Ee["lock_L5"]["P_b0_1"] - 0.15**2) < 1e-12
    assert Ee["lock_L3"]["P_b0_eq_lastread"] > 0.97
    # 3D walker: each lift matches its own pristine-pair law to < 0.006
    for lift in ("givens", "perm"):
        assert abs(Ee["walker_3d_rotating"][lift]["t1_residual"]) < 0.006
    print("all headline assertions passed")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
