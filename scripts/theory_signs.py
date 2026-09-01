#!/usr/bin/env python
"""Theory strike M3, Target 1: the sign-coherence condition for conversion amplitudes.

Background (J17): rule 891's converted channel G_L is ensemble-noise in the current
Gaussian-mode-map gauge. This script finds out WHY, exactly, and derives the condition
under which a per-transition Z2 sign table sigma(val) (Wetterich's discrete sign
freedom, applied per block transition) makes every single-conversion path contribute
with the same sign.

The machinery, all exact (no floats in any decision):

  (E) EXACT ENUMERATION. On an open chain of L sites we enumerate ALL 2^(2L-1)
      (environment, system) vacuum configurations with the creation site empty, evolve
      the defect and vacuum trajectories through the signed lift, and record every
      surviving single-bit-difference contribution (t, x, sign). This is the amplitude
      instrument run over the *complete* ensemble: G(t, x; q) is then an exact
      polynomial in q. The J17 cancellation must appear here as exact +/- counts.

  (G) THE GAUGE SYSTEM. A per-transition sign table sigma: {0..15} -> {+-1} multiplies
      the lifted block operator for transition `val` by sigma(val). Blocks not
      containing the damage read the same val in both trajectories, so their sigma
      factors cancel; each engagement of the damage contributes
      sigma(val_defect) * sigma(val_vacuum). Demanding ALL surviving contributions
      equal +1 is therefore a linear system over GF(2):

          sum_val [n_d(val) + n_v(val)] x(val)  =  b_rec   (mod 2),

      per record, with x(val) = [sigma(val) == -1], b_rec = [record sign == -1] in the
      base gauge, and n_d/n_v the per-trajectory val-visit counts. Solvable => the
      uniformizing table exists (and is exhibited + re-verified by re-enumeration);
      inconsistent => NO per-transition table exists, with the offending GF(2)
      combination as the certificate.

  (R) THE ROTATION TARGET. For rules with a proven tracer reduction (env autonomous +
      system-blind, cf. theory-linear-law), the tracer path and hence the number of
      L->R conversions N_LR is a function of the environment alone. The second target
      pattern  sign(rec) = (-1)^{N_LR(rec)}  (R->L conversions all +1, L->R all -1) is
      the real-rotation (Dirac-coin) sign structure; same GF(2) machinery.

  (P) PH CHECK. "PH-respecting" is checked operationally: the complement lift
      K = prod_m (a_m + nu_m a_m^dagger) (per-mode nu in {+-1}; per-site global signs
      drop out of conjugation) must satisfy U_B K = +- K U_B at block level, where U_B
      is the 16x16 signed block matrix WITH sigma. Complement flips two modes per
      site, so K_site is parity-even and the global check factorises into this block
      check. A second, weaker check solves for an arbitrary (possibly nonlocal)
      diagonal-sign lift kappa(c) by cycle consistency.

  (T) TRANSFER DISPERSION (Target 3). For gauged rule 891 the proven tracer reduction
      plus sign uniformity closes the amplitude into an exact 2-state Markov transfer
      per class:  T_A(k) = [[(1-q) e^{ik}, s_LR (1-q)], [s_RL q, q e^{-ik}]]  (class B:
      q <-> 1-q).  Verified here against the enumerated G(t, x; q) as exact rationals.
      Eigenvalues give the dressed 2x2 dispersion; the rotation gauge s_RL s_LR = -1
      has det = 2q(1-q) and a mass onset exactly at q* = (2-sqrt2)/4.

Run:  .venv/bin/python scripts/theory_signs.py           (~2-3 min)
      .venv/bin/python scripts/theory_signs.py --big     (adds L=12, t<=5, slow)
      .venv/bin/python scripts/theory_signs.py --scan    (class scan over legal rules)
"""
from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pca3d.fock.signed import SignedBlockCycle  # noqa: E402
from pca3d.models import conditional as C  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RULE_INDEX = 891


# ======================================================================= gauged lift
class GaugedBlockCycle(SignedBlockCycle):
    """SignedBlockCycle with an extra per-transition sign table sigma[val] in {+-1}.

    The parent's per-block application is reproduced with one extra factor
    sigma[val] on the sign of every trajectory whose block reads `val`, and an
    optional GF(2) accumulator of per-trajectory val visits (expo ^= 1 << val).
    """

    def set_sigma(self, sigma) -> None:
        self._sigma = np.asarray(sigma, dtype=np.int64)
        assert self._sigma.shape == (16,) and set(np.abs(self._sigma)) == {1}

    def start_expo(self, n: int) -> None:
        """GF(2) val-visit accumulator: bit v of expo[i] = parity of visits to val v."""
        self._expo = np.zeros(n, dtype=np.uint16)

    @property
    def expo(self) -> np.ndarray:
        return self._expo

    def _apply_blocks(self, configs, signs, origin=0):
        out_c, out_s = configs, signs
        track = getattr(self, "_expo", None) is not None and len(configs) == len(
            getattr(self, "_expo", ())
        )
        for modes in self._blocks_for_origin(origin):
            nib = np.zeros(len(out_c), dtype=np.int64)
            for i, m in enumerate(modes):
                nib |= ((out_c >> m) & 1) << i
            if track:
                self._expo ^= (np.uint16(1) << nib.astype(np.uint16))
            extra = self._sigma[nib]
            for val in range(16):
                target = int(self.block_perm[val])
                sel = nib == val
                if not np.any(sel):
                    continue
                c_sub, s_sub = out_c[sel], out_s[sel]
                pg, phi, pb = self._decompose(modes, val, target)
                for m in sorted(pg, key=lambda m: -self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.annihilate(c_sub, s_sub, m)
                sign_phi = 1
                srcs = sorted((m for m, _ in phi), key=lambda m: self.jw.ranks[m])
                images = [self.jw.ranks[dict(phi)[m]] for m in srcs]
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        if images[i] > images[j]:
                            sign_phi = -sign_phi
                for m, _ in phi:
                    c_sub = c_sub & ~(np.int64(1) << m)
                for m, m2 in phi:
                    c_sub = c_sub | (np.int64(1) << m2)
                s_sub = s_sub * sign_phi
                for m in sorted(pb, key=lambda m: self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.create(c_sub, s_sub, m)
                out_c = out_c.copy()
                out_s = out_s.copy()
                out_c[sel], out_s[sel] = c_sub, s_sub
            out_s = out_s * extra
        return out_c, out_s


def _popcount64(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64).copy()
    c = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        c += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return c


# =============================================================== exact enumeration (E)
def exact_records(perm, sigma, L, Y, T, want_expo=False):
    """All surviving single-bit contributions over the complete (env, sys) ensemble.

    Returns dict with arrays per sub-step t (1..T):
      x, sign, env0 (initial env bits packed), and if want_expo the GF(2) row
      expo_d ^ expo_v (uint16 bitmask over the 16 vals).
    Normalisation: G(t, x) = 2^-(2L-1) * sum over records of sign * q-weight, with the
    creation-site-occupied half of the ensemble contributing 0 (the factor 1/2).
    """
    model = GaugedBlockCycle(n_sites=L, block_perm=perm, boundary="open")
    model.set_sigma(sigma)
    n_modes = 2 * L
    y_mode = 2 * Y
    all_cfg = np.arange(1 << n_modes, dtype=np.int64)
    tau = all_cfg[(all_cfg >> y_mode) & 1 == 0]
    n = len(tau)
    ones = np.ones(n, dtype=np.int64)

    env_mask = np.int64(0)
    for s in range(L):
        env_mask |= np.int64(1) << (2 * s + 1)

    d_cfg, d_sgn = model.jw.create(tau, ones.copy(), y_mode)
    v_cfg, v_sgn = tau.copy(), ones.copy()
    if want_expo:
        model.start_expo(n)
        expo_d = np.zeros(n, dtype=np.uint16)
        expo_v = np.zeros(n, dtype=np.uint16)

    out = {}
    for t in range(1, T + 1):
        if want_expo:
            model._expo = expo_d
            d_cfg, d_sgn = model.substep(d_cfg, d_sgn, (t - 1) % 2)
            expo_d = model._expo
            model._expo = expo_v
            v_cfg, v_sgn = model.substep(v_cfg, v_sgn, (t - 1) % 2)
            expo_v = model._expo
        else:
            d_cfg, d_sgn = model.substep(d_cfg, d_sgn, (t - 1) % 2)
            v_cfg, v_sgn = model.substep(v_cfg, v_sgn, (t - 1) % 2)
        diff = d_cfg ^ v_cfg
        single = (_popcount64(diff) == 1) & (d_sgn != 0) & (v_sgn != 0)
        idx = np.flatnonzero(single)
        dd = diff[idx].astype(np.float64)
        m = np.round(np.log2(dd)).astype(np.int64)  # exact: dd is a power of two
        assert np.all((np.int64(1) << m) == diff[idx])
        below = model.jw.below_mask[m]
        par = _popcount64(v_cfg[idx] & below) & 1
        sign = d_sgn[idx] * v_sgn[idx] * (1 - 2 * par)
        rec = {
            "idx": idx,
            "x": (m // 2),
            "species": (m % 2),
            "sign": sign,
            "env0": (tau[idx] & env_mask),
        }
        if want_expo:
            rec["row"] = expo_d[idx] ^ expo_v[idx]
        out[t] = rec
    out["n_total"] = n
    out["tau_env"] = None
    return out


def env_weight_tables(L, q: Fraction):
    """q-weight per env-bitcount, exact."""
    return [q**e * (1 - q) ** (L - e) for e in range(L + 1)]


def exact_G(records, L, T, q: Fraction, target_signs=None):
    """G(t, x) as exact rationals from the record table (system sector averaged).

    target_signs: optional dict t -> array overriding record signs (for predictions).
    """
    wt = env_weight_tables(L, q)
    norm = Fraction(1, 1 << (2 * L - 1))  # sys ensemble (with the empty-site half)
    # each env config appears with 2^(L-1) system configs; weight = wt[e] / 2^(L-1)
    G = {}
    for t in range(1, T + 1):
        r = records[t]
        e_cnt = _popcount64(r["env0"])
        signs = r["sign"] if target_signs is None else target_signs[t]
        acc = {}
        for x in np.unique(r["x"]):
            sel = r["x"] == x
            tot = Fraction(0)
            for e in np.unique(e_cnt[sel]):
                s = int(signs[sel & (e_cnt == e)].sum())
                if s:
                    tot += Fraction(s) * wt[int(e)]
            acc[int(x)] = tot / Fraction(1 << (L - 1)) / 2
        G[t] = acc
    return G


# ============================================================== GF(2) gauge solve (G)
def gf2_solve(rows_u16, rhs_bits):
    """Solve A x = b over GF(2); rows as uint16 bitmasks (16 unknowns).

    Returns (solution_mask or None, certificate or None). The certificate is the
    reduced augmented row 0 = 1 proving inconsistency.
    """
    aug = np.unique(
        rows_u16.astype(np.uint32) | (rhs_bits.astype(np.uint32) << np.uint32(16))
    )
    pivots: dict[int, int] = {}  # pivot bit -> augmented row (bit 16 = rhs)
    for row in aug.tolist():
        cur = int(row)
        while True:
            vb = cur & 0xFFFF
            if vb == 0:
                break
            hi = vb.bit_length() - 1
            if hi in pivots:
                cur ^= pivots[hi]
            else:
                pivots[hi] = cur
                cur = 0
                break
        if cur == (1 << 16):  # reduced to 0 = 1
            return None, cur
    # particular solution, free variables = 0; each pivot row's other set variable
    # bits are strictly lower, so resolve in ascending pivot order
    x = 0
    for bit in sorted(pivots):
        row = pivots[bit]
        acc = (row >> 16) & 1
        low = row & 0xFFFF & ~(1 << bit)
        acc ^= bin(low & x).count("1") & 1
        if acc:
            x |= 1 << bit
    return x, None


def sigma_from_mask(mask: int) -> np.ndarray:
    return np.array([-1 if (mask >> v) & 1 else 1 for v in range(16)], dtype=np.int64)


def solve_gauge(records, T, target="uniform", nlr=None):
    """GF(2) solve for sigma making sign(rec) * gauge(rec) match the target pattern.

    target 'uniform': all +1.  target 'rotation': (-1)^{N_LR(rec)} (needs nlr arrays).
    Returns (sigma or None, mask or None, n_equations).
    """
    rows, rhs = [], []
    for t in range(1, T + 1):
        r = records[t]
        b = (r["sign"] < 0).astype(np.int64)
        if target == "rotation":
            b = b ^ (nlr[t] & 1)
        rows.append(r["row"])
        rhs.append(b)
    rows = np.concatenate(rows)
    rhs = np.concatenate(rhs)
    mask, cert = gf2_solve(rows, rhs)
    if mask is None:
        return None, None, len(rows)
    return sigma_from_mask(mask), mask, len(rows)


# ================================================= tracer paths for blind rules (R)
def tracer_paths(perm, env0_list, L, Y, T):
    """Direction/stall bookkeeping of a single carrier, per env, by direct simulation
    on the open chain (valid content-independently only for system-blind rules; the
    caller checks blindness). Returns per env: positions, N_LR(t), stalls(t)."""
    out = []
    for env_packed in env0_list:
        env = np.array([(int(env_packed) >> (2 * s + 1)) & 1 for s in range(L)],
                       dtype=np.int64)
        sys_b = np.zeros(L, dtype=np.int64)
        sys_b[Y] = 1
        d = +1
        pos = Y
        nlr, stalls, poss = [], [], []
        n_lr = n_st = 0
        for t in range(1, T + 1):
            o = (t - 1) % 2
            new_sys = sys_b.copy()
            new_env = env.copy()
            if o == 0:
                sites = [(2 * b, 2 * b + 1) for b in range(L // 2)]
            else:
                sites = [(2 * b + 1, 2 * b + 2) for b in range((L - 2) // 2)]
            for i, j in sites:
                cfg = C.encode(int(sys_b[i]), int(env[i]), int(sys_b[j]), int(env[j]))
                o_cfg = int(perm[cfg])
                ps, ph, ps2, ph2 = C.decode(o_cfg)
                new_sys[i], new_env[i], new_sys[j], new_env[j] = ps, ph, ps2, ph2
            sys_b, env = new_sys, new_env
            (where,) = np.nonzero(sys_b)
            p = int(where[0])
            if p == pos:  # stall = reversal
                n_st += 1
                if d == -1:
                    n_lr += 1
                d = -d
            else:
                d = p - pos
            pos = p
            poss.append(pos)
            nlr.append(n_lr)
            stalls.append(n_st)
        out.append((poss, nlr, stalls))
    return out


# ======================================================================= PH check (P)
def _local_K_matrix(nu: tuple[int, int, int, int]) -> np.ndarray:
    """K = (a_0 + nu0 a_0^+)(a_1 + nu1 a_1^+)(a_2 + nu2 a_2^+)(a_3 + nu3 a_3^+) on the
    4-mode block Fock space, JW ranks = the block's global rank order
    (psi, psi', phi, phi') = local modes (0, 2, 1, 3)."""
    ranks = {0: 0, 2: 1, 1: 2, 3: 3}
    dim = 16
    K = np.zeros((dim, dim), dtype=np.int64)
    for c in range(dim):
        cfg, sign = c, 1
        ok = True
        for m in (3, 2, 1, 0):  # rightmost operator acts first
            below = sum(1 for m2 in range(4) if ranks[m2] < ranks[m] and (cfg >> m2) & 1)
            jw = (-1) ** below
            if (cfg >> m) & 1:  # annihilate
                cfg &= ~(1 << m)
                sign *= jw
            else:  # create (times nu)
                cfg |= 1 << m
                sign *= jw * nu[m]
        if ok:
            K[cfg, c] = sign
    return K


def block_matrix(perm, sigma) -> np.ndarray:
    model = GaugedBlockCycle(n_sites=2, block_perm=perm, boundary="open")
    model.set_sigma(sigma)
    return model.dense_substep(0).astype(np.int64)


def ph_check(perm, sigma):
    """Local check: exists nu in {+-1}^4, eps in {+-1} with U K = eps K U.
    Also the abstract (possibly nonlocal) diagonal-sign lift by cycle consistency."""
    U = block_matrix(perm, sigma)
    local = None
    for bits in range(16):
        nu = tuple(1 - 2 * ((bits >> i) & 1) for i in range(4))
        K = _local_K_matrix(nu)
        for eps in (1, -1):
            if np.array_equal(U @ K, eps * K @ U):
                local = {"nu": nu, "eps": eps}
                break
        if local:
            break
    # abstract: kappa(perm(c)) = eps * kappa(c) * s(cbar)/s(c), consistency on cycles
    s = np.zeros(16, dtype=np.int64)
    p = np.zeros(16, dtype=np.int64)
    for c in range(16):
        col = U[:, c]
        (tgt,) = np.nonzero(col)
        p[c] = tgt[0]
        s[c] = col[tgt[0]]
    abstract = []
    for eps in (1, -1):
        kappa = np.zeros(16, dtype=np.int64)
        ok = True
        for start in range(16):
            if kappa[start]:
                continue
            kappa[start] = 1
            c = start
            while True:
                nxt = int(p[c])
                val = eps * kappa[c] * s[c ^ 15] * s[c]
                # kappa(perm(c)) determined by kappa(c) via U K = eps K U on |c>:
                #   s(cbar) kappa(c) = eps kappa(perm(c)) s(c)
                want = eps * kappa[c] * s[c ^ 15] * s[c]
                if kappa[nxt] == 0:
                    kappa[nxt] = want
                elif kappa[nxt] != want:
                    ok = False
                    break
                if nxt == start:
                    break
                c = nxt
            if not ok:
                break
        if ok:
            abstract.append(eps)
    return local, abstract


# =============================================== transfer dispersion, gauged 891 (T)
def transfer_G(L, Y, T, q: Fraction, s_RL: int, s_LR: int):
    """Exact class-resolved signed telegraph walk on the open chain: prediction for
    G(t, x) of gauged rule 891 (per sub-step kinematics: move in direction d with the
    class's co-probability, stall+reverse with the counter-probability)."""
    # class A: companion=b0 matches; move probs: R:(1-q), L:q ; class B: swap.
    G = {t: {} for t in range(1, T + 1)}
    for cls, w_cls, pR, pL in (
        ("A", 1 - q, 1 - q, q),
        ("B", q, q, 1 - q),
    ):
        # amp[dir][x]: dir 0=R, 1=L
        amp = {0: {Y: Fraction(1)}, 1: {}}
        for t in range(1, T + 1):
            new = {0: {}, 1: {}}
            for x, a in amp[0].items():
                new[0][x + 1] = new[0].get(x + 1, Fraction(0)) + a * pR
                new[1][x] = new[1].get(x, Fraction(0)) + a * pL * s_RL
            for x, a in amp[1].items():
                new[1][x - 1] = new[1].get(x - 1, Fraction(0)) + a * pL
                new[0][x] = new[0].get(x, Fraction(0)) + a * pR * s_LR
            amp = new
            for d in (0, 1):
                for x, a in amp[d].items():
                    if a:
                        G[t][x] = G[t].get(x, Fraction(0)) + Fraction(1, 2) * w_cls * a
    return G


def dispersion_2x2(q: float, k: float, s_RL: int, s_LR: int):
    """Eigenvalues of T_A(k); returns (omega_+, gamma_+, omega_-, gamma_-)."""
    TA = np.array(
        [
            [(1 - q) * np.exp(1j * k), s_LR * (1 - q)],
            [s_RL * q, q * np.exp(-1j * k)],
        ]
    )
    lam = np.linalg.eigvals(TA)
    lam = lam[np.argsort(-np.abs(lam))]
    out = []
    for l in lam:
        out += [float(np.angle(l)), float(-np.log(max(abs(l), 1e-300)))]
    return out


# ============================================= the deliverable: table -> best gauge
def best_sign_assignment(table, L=10, Y=4, T=4, q=Fraction(1, 4), verify=True):
    """For a 16-entry block table: does a per-transition sign gauge exist that makes
    every surviving amplitude contribution +1 (ensemble-coherent G in ALL channels)?

    Returns dict:
      solvable        : bool
      sigma           : 16-array in {+-1} (None if not solvable)
      mask            : integer bitmask of the -1 entries
      ph_local        : local complement-lift check result for the gauged block
      coherence_base  : sum_t,x |G(t,x)| at q, base gauge      (exact, as float)
      coherence_gauged: same with sigma applied                (exact, as float)
      conv_base/conv_gauged: the converted-channel share sum_{x: (x-Y-t) odd} |G|
    Contributions are exact over the complete (env, sys) ensemble at chain length L.
    """
    table = np.asarray(table, dtype=np.int64)
    pc = np.array([bin(i).count("1") for i in range(16)])
    if np.any((pc[table] & 1) != (pc & 1)):
        raise ValueError("table does not conserve fermion parity (no lift exists)")
    base = exact_records(table, np.ones(16, dtype=np.int64), L, Y, T, want_expo=True)
    sigma, mask, neq = solve_gauge(base, T, target="uniform")
    out = {"solvable": sigma is not None, "mask": mask, "n_equations": neq}
    out["sigma"] = None if sigma is None else sigma.tolist()

    def coherences(recs):
        G = exact_G(recs, L, T, q)
        tot = sum(abs(v) for t in G for v in G[t].values())
        conv = sum(
            abs(v) for t in G for x, v in G[t].items() if (x - Y - t) % 2 == 1
        )
        return float(tot / T), float(conv / T)

    out["coherence_base"], out["conv_base"] = coherences(base)
    if sigma is not None and verify:
        chk = exact_records(table, sigma, L, Y, T)
        all_plus = all(bool(np.all(chk[t]["sign"] == 1)) for t in range(1, T + 1))
        out["verified_all_plus"] = all_plus
        out["coherence_gauged"], out["conv_gauged"] = coherences(chk)
        local, abstract = ph_check(table, sigma)
        out["ph_local"] = local
        out["ph_abstract_eps"] = abstract
    return out


# ================================================================== structure tests
def is_blind_per_env(perm) -> bool:
    """Motion decision equal in both mixed system sectors for every env state
    (the structural predicate conjectured to control uniformizability)."""
    for e_in in range(4):
        c10 = C.compose(1, e_in)
        c01 = C.compose(2, e_in)
        moved10 = C.system_state(int(perm[c10])) != 1
        moved01 = C.system_state(int(perm[c01])) != 2
        if moved10 != moved01:
            return False
    return True


def legal_rules():
    rules = C.enumerate_conditional_rules()
    pc = np.array([bin(i).count("1") for i in range(16)])
    return [
        (i, r)
        for i, r in enumerate(rules)
        if not np.any((pc[r] & 1) != (pc & 1))
    ]


# ============================================================================= main
def main() -> None:
    big = "--big" in sys.argv
    scan = "--scan" in sys.argv or "--scan-big" in sys.argv
    perm = C.enumerate_conditional_rules()[RULE_INDEX]
    report = {"rule": RULE_INDEX}
    print("=" * 74)
    print("TARGET 1: the sign-coherence condition for conversion amplitudes")
    print("=" * 74)

    # ---------------------------------------------------------------- (a) exhibit
    L, Y, T = 10, 4, 4
    print(f"\n[a] exact enumeration, rule 891, base gauge, L={L} open chain, "
          f"Y={Y}, t<={T} ({1 << (2 * L - 1):,} trajectories)")
    base = exact_records(perm, np.ones(16, dtype=np.int64), L, Y, T, want_expo=True)
    conv_counts = {}
    for t in range(1, T + 1):
        r = base[t]
        conv = (r["x"] - Y - t) % 2 == 1
        for lab, sel in (("co-moving", ~conv), ("converted", conv)):
            s = r["sign"][sel]
            conv_counts[(t, lab)] = (len(s), int((s > 0).sum()), int((s < 0).sum()))
    for (t, lab), (n, p, m) in sorted(conv_counts.items()):
        print(f"    t={t} {lab:>9}: n={n:7d}  +:{p:7d}  -:{m:7d}  net={p - m:7d}")
    conv_net = [conv_counts[(t, "converted")][1] - conv_counts[(t, "converted")][2]
                for t in range(1, T + 1)]
    print(f"    converted-channel nets: {conv_net}  (exact cancellation: "
          f"{all(v == 0 for v in conv_net)})")
    report["a_cancellation"] = {
        "counts": {f"t{t}_{lab}": conv_counts[(t, lab)] for (t, lab) in conv_counts},
        "converted_net_zero": all(v == 0 for v in conv_net),
    }

    # the mechanism: the sign of a single-conversion record is (-1)^{n_partner}
    print("\n    mechanism check: single-stall records against (-1)^(partner bit)")
    # t=1 stall records: x == Y; partner site is Y+1 (origin-0 block (Y, Y+1))
    r1 = base[1]
    stall = r1["x"] == Y
    # partner occupancy = system bit of site Y+1 in the INITIAL vacuum config
    # (recover from record index -> initial tau)
    all_cfg = np.arange(1 << (2 * L), dtype=np.int64)
    tau = all_cfg[(all_cfg >> (2 * Y)) & 1 == 0]
    part = (tau[r1["idx"][stall]] >> (2 * (Y + 1))) & 1
    pred = 1 - 2 * part
    ok = bool(np.all(r1["sign"][stall] == pred))
    print(f"    t=1 stall records: sign == (-1)^(n_partner) for all "
          f"{int(stall.sum())} records: {ok}")
    assert ok
    report["a_mechanism"] = "sign of a stall engagement = (-1)^(vacuum system bit at "\
        "the partner site); Bernoulli(1/2) partner => exact cancellation"

    # ---------------------------------------------------------------- (b) the gauge
    print("\n[b] GF(2) solve for a uniformizing per-transition sign table")
    sigma_u, mask_u, neq = solve_gauge(base, T, target="uniform")
    assert sigma_u is not None, "expected solvable for rule 891"
    minus = [v for v in range(16) if sigma_u[v] < 0]
    print(f"    solvable: YES  ({neq} exact equations); sigma = -1 on vals {minus}")
    chk = exact_records(perm, sigma_u, L, Y, T)
    all_plus = all(bool(np.all(chk[t]["sign"] == 1)) for t in range(1, T + 1))
    print(f"    re-enumeration with sigma: every surviving contribution +1: {all_plus}")
    assert all_plus
    local_u, abstract_u = ph_check(perm, sigma_u)
    local_b, abstract_b = ph_check(perm, np.ones(16, dtype=np.int64))
    print(f"    PH lift, base gauge:  local {local_b}, abstract eps {abstract_b}")
    print(f"    PH lift, sigma gauge: local {local_u}, abstract eps {abstract_u}")
    report["b_uniform"] = {
        "sigma_minus_vals": minus,
        "verified_all_plus": bool(all_plus),
        "ph_local_base": local_b is not None,
        "ph_local_gauged": local_u is not None,
        "ph_abstract_base": abstract_b,
        "ph_abstract_gauged": abstract_u,
    }

    # exact coherent G in the uniform gauge equals half the proven density law
    print("\n    telegraph identity: gauged G(t,x) == (1/2) P[X_t = x] exactly")
    qtest = Fraction(1, 4)
    G_meas = exact_G(chk, L, T, qtest)
    G_pred = transfer_G(L, Y, T, qtest, s_RL=1, s_LR=1)
    same = all(
        G_meas[t].get(x, Fraction(0)) == G_pred[t].get(x, Fraction(0))
        for t in range(1, T + 1)
        for x in set(G_meas[t]) | set(G_pred[t])
    )
    print(f"    q = {qtest}: exact rational equality at all (t, x): {same}")
    assert same
    report["b_telegraph_identity"] = True

    # ------------------------------------------------------- rotation (Dirac) gauge
    print("\n[b'] rotation target: R->L conversions +1, L->R conversions -1")
    # N_LR per record from the tracer path (env determines the path; blind rule)
    assert is_blind_per_env(perm)
    env_vals = np.unique(np.concatenate([base[t]["env0"] for t in range(1, T + 1)]))
    paths = tracer_paths(perm, env_vals, L, Y, T)
    pos_arr = np.array([p[0] for p in paths], dtype=np.int64)  # (n_env, T)
    nlr_arr = np.array([p[1] for p in paths], dtype=np.int64)
    nlr = {}
    for t in range(1, T + 1):
        r = base[t]
        eidx = np.searchsorted(env_vals, r["env0"])
        assert np.array_equal(pos_arr[eidx, t - 1], r["x"]), "path/record mismatch"
        nlr[t] = nlr_arr[eidx, t - 1]
    sigma_r, mask_r, _ = solve_gauge(base, T, target="rotation", nlr=nlr)
    assert sigma_r is not None
    minus_r = [v for v in range(16) if sigma_r[v] < 0]
    print(f"    solvable: YES; sigma = -1 on vals {minus_r}")
    chk_r = exact_records(perm, sigma_r, L, Y, T)
    ok_r = True
    for t in range(1, T + 1):
        want = 1 - 2 * (nlr[t] & 1)
        ok_r &= bool(np.all(chk_r[t]["sign"] == want))
    print(f"    re-enumeration: sign == (-1)^N_LR for every record: {ok_r}")
    assert ok_r
    local_r, abstract_r = ph_check(perm, sigma_r)
    print(f"    PH lift, rotation gauge: local {local_r}, abstract eps {abstract_r}")
    G_meas_r = exact_G(chk_r, L, T, qtest)
    G_pred_r = transfer_G(L, Y, T, qtest, s_RL=1, s_LR=-1)
    same_r = all(
        G_meas_r[t].get(x, Fraction(0)) == G_pred_r[t].get(x, Fraction(0))
        for t in range(1, T + 1)
        for x in set(G_meas_r[t]) | set(G_pred_r[t])
    )
    print(f"    signed-telegraph identity at q = {qtest}: {same_r}")
    assert same_r
    report["b_rotation"] = {
        "sigma_minus_vals": minus_r,
        "verified_signed": bool(ok_r),
        "ph_local": local_r is not None,
        "ph_abstract": abstract_r,
        "transfer_identity": bool(same_r),
    }

    # ------------------------------------------------------------- (T) dispersion
    print("\n[T] dressed 2x2 dispersion from the exact transfer matrix")
    qs = (2 - np.sqrt(2.0)) / 4
    print("    uniform gauge (s_RL s_LR = +1): det T = 0, single branch per class:")
    print("      lambda_A(k) = cos k + i (1-2q) sin k  =>  omega = arctan((1-2q) tan k)")
    print("      => omega ~ (1-2q) k: the chiral linear law at amplitude level, with")
    print("         coherent O(q) off-diagonals sharing the SAME pole (Kac/telegraph).")
    print("    rotation gauge (s_RL s_LR = -1): det T = 2q(1-q), trace c(k):")
    print("      lambda = [c(k) +- sqrt(c(k)^2 - 8q(1-q))]/2")
    print(f"      mass onset at disc(k=0) = 0: 8q^2 - 8q + 1 = 0 => q = (2-sqrt2)/4 "
          f"= {qs:.8f} = q*  (the sqrt2 tuning point)")
    rows = []
    for qq in (0.05, 0.10, float(qs), 0.20, 0.30):
        w0 = dispersion_2x2(qq, 0.0, 1, -1)
        wk = dispersion_2x2(qq, 0.3, 1, -1)
        m = abs(w0[0])
        rows.append((qq, m, wk[0], wk[1]))
        print(f"      q={qq:.4f}: m = |omega(0)| = {m:.4f}, omega(0.3) = {wk[0]:+.4f}, "
              f"Gamma = {wk[1]:.4f}")
    report["T_dispersion"] = {
        "uniform_gauge": "omega = arctan((1-2q) tan k), Gamma(k) = -0.5 ln(1-4q(1-q) sin^2 k)",
        "rotation_gauge": "lambda = [c(k) +- sqrt(c(k)^2 - 8q(1-q))]/2, c = cos k + i(1-2q) sin k",
        "mass_onset_q": "(2-sqrt2)/4 (exact; equals q*)",
        "samples": rows,
    }

    # ------------------------------------------------- the deliverable function demo
    print("\n[fn] best_sign_assignment(table) on rule 891 (the gauge-scan entry point)")
    res = best_sign_assignment(perm, L=8, Y=4, T=3)
    print(f"    solvable={res['solvable']}, sigma -1 on "
          f"{[v for v in range(16) if res['sigma'][v] < 0] if res['sigma'] else None}, "
          f"conv |G| base={res['conv_base']:.4f} -> gauged={res['conv_gauged']:.4f}")
    report["fn_demo"] = {k: v for k, v in res.items() if k != "sigma"} | {
        "sigma": res["sigma"]
    }

    # ---------------------------------------------------------------- (c) class scan
    if scan:
        Ls, Ys, Ts = (10, 4, 4) if "--scan-big" in sys.argv else (8, 4, 3)
        print(f"\n[c] class scan: all legal (parity-conserving) conditional rules "
              f"(L={Ls}, t<={Ts})")
        rules = legal_rules()
        print(f"    legal rules: {len(rules)}")
        tab = []
        for i, r in rules:
            try:
                res = best_sign_assignment(r, L=Ls, Y=Ys, T=Ts, verify=True)
            except Exception as exc:  # pragma: no cover
                tab.append({"rule": i, "error": str(exc)})
                continue
            tab.append(
                {
                    "rule": i,
                    "solvable": res["solvable"],
                    "blind_per_env": is_blind_per_env(r),
                    "conv_base": res["conv_base"],
                    "conv_gauged": res.get("conv_gauged"),
                    "ph_local_gauged": bool(res.get("ph_local")),
                    "sigma_mask": res["mask"],
                }
            )
        n_solv = sum(1 for row in tab if row.get("solvable"))
        n_blind = sum(1 for row in tab if row.get("blind_per_env"))
        agree = sum(
            1 for row in tab if "solvable" in row
            and row["solvable"] == row["blind_per_env"]
        )
        print(f"    solvable: {n_solv}/{len(tab)}; blind-per-env: {n_blind}; "
              f"predicate agrees: {agree}/{len(tab)}")
        (RESULTS / "theory_signs_classscan.json").write_text(
            json.dumps({"geometry": {"L": Ls, "Y": Ys, "T": Ts},
                        "n_rules": len(tab), "n_solvable": n_solv,
                        "n_blind": n_blind, "predicate_agreement": agree,
                        "rows": tab}, indent=2, default=str)
        )
        report["c_scan"] = {"L": Ls, "T": Ts, "n_rules": len(tab),
                           "n_solvable": n_solv, "n_blind": n_blind,
                           "agreement": agree}

    # ---------------------------------------- ADR 0009 follow-up: rules 107 / 133
    if "--kac107" in sys.argv:
        print("\n[kac] rules 107/133 (ADR 0009): exact q->0 conversion coefficients")
        eps = Fraction(1, 10**8)
        for idx in (107, 133):
            perm9 = C.enumerate_conditional_rules()[idx]
            recs9 = exact_records(perm9, np.ones(16, dtype=np.int64), L, Y, T)
            G9 = exact_G(recs9, L, T, eps)
            rows9 = {}
            for t in range(1, T + 1):
                conv = {int(x): float(v / eps) for x, v in G9[t].items()
                        if (x - Y - t) % 2 == 1 and v}
                co = {int(x): float(v) for x, v in G9[t].items()
                      if (x - Y - t) % 2 == 0 and abs(v) > eps}
                rows9[t] = {"dG_conv_dq": conv, "co_moving_q0": co}
                print(f"    rule {idx} t={t}: dG_conv/dq|0 = {conv}; free front {co}")
            report[f"kac_{idx}"] = rows9
        print("    => free chiral front amplitude exactly +1/2 at x = Y+t;")
        print("       conversion coefficient exactly -q/4 (rule 107) / +q/4 (rule 133)")
        print("       per interior site => Kac mass law m(q->0) = q/2 per sub-step,")
        print("       with opposite mass signs for the two rules (the two phases of")
        print("       ADR 0009, derived).")

    # ------------------------------------------------------------------- big rerun
    if big:
        print("\n[big] confirmation at L=12, t<=5")
        Lb, Yb, Tb = 12, 6, 5
        # NOTE: Y must keep t<=T paths off the edges: X in [Y-T, Y+T] = [0, 10] ok
        bigrec = exact_records(perm, sigma_u, Lb, Yb, Tb)
        all_plus_b = all(bool(np.all(bigrec[t]["sign"] == 1)) for t in range(1, Tb + 1))
        print(f"    uniform gauge, all contributions +1 at L=12, t<=5: {all_plus_b}")
        report["big_confirmation"] = bool(all_plus_b)

    RESULTS.mkdir(exist_ok=True)
    # merge with any previous run so optional sections (--scan/--big/--kac107)
    # accumulate instead of clobbering each other
    out_path = RESULTS / "theory_signs.json"
    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text())
        except Exception:
            merged = {}
    merged.update(report)
    out_path.write_text(json.dumps(merged, indent=2, default=str))
    print(f"\nwritten: {RESULTS / 'theory_signs.json'}")
    print("\nVERDICT: J17's converted-channel noise is a GAUGE ARTIFACT of the "
          "Gaussian-mode-map lift\n(sign of a stall = (-1)^(partner occupation)); "
          "a per-transition sign table repairs it\n(two inequivalent coherent gauges: "
          "telegraph and rotation), machine-verified exactly.")


if __name__ == "__main__":
    main()
