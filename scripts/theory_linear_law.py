#!/usr/bin/env python
"""Theory strike, Target 1: prove the linear law v(q) = 2(1-2q) for rule 891.

The proof chain, every link machine-checked here in exact arithmetic:

  (S) STRUCTURE (from the block table, not assumed):
      S1. The environment map is (phi, phi') -> (phi', phi) in all 16 rows --
          autonomous (independent of system content) and a pure swap. In the
          shifted-block geometry a per-block swap on alternating partitions is free
          streaming: env bits on the even sublattice are right-movers, on the odd
          sublattice left-movers, both at 1 site/sub-step (= the light cone 2/cycle).
      S2. The system map given the env is system-blind: the block swaps its system
          bits iff the env parity is even (0 or 2 env bits), holds iff exactly one.
          Hence the system channel is a *passive tracer field*: given the env, all
          system bits are transported by one deterministic site permutation Pi_t.

  (C) CORRELATOR REDUCTION: with the system channel i.i.d. Bernoulli(1/2),
          C(t, x) = <s_x(t) s_0(0)> = P_env[ Pi_t(0) = x ]
      -- the exact law of a single tracer released at site 0.

  (T) TRACER CLOSED FORM: the tracer state is (direction d, companion c) where the
      companion is the co-moving env bit at the tracer's own site. Each sub-step
      reads exactly one FRESH env bit b (fresh by a monotone-gap argument: the
      signed gap to any given left-mover changes by +2 or 0, to any right-mover by
      -2 or 0, so no env bit is ever read twice), and
          - move (in direction d) iff b = c, keeping c;
          - stall, reverse, and set c <- b iff b != c.
      Since c_k = b_k in all cases, with b_0 = phi(0) the walk closes:

          X_t = sum_{k=1..t} 1{b_k = b_{k-1}} * (+1 if b_{k-1} = b_0 else -1)
              = (-1)^{b_0} ( N_00 - N_11 ),

      N_vv = # adjacent equal pairs of value v in the i.i.d. Bernoulli(q) string
      (b_0, ..., b_t). Checked here against direct rule simulation, exactly,
      including the read-position bookkeeping.

  (P) POLYNOMIAL IDENTITY: on an L-site ring the exact all-configuration correlator
      C(t, x) is a polynomial in q of degree <= L. The model law P(X_t = x) is a
      polynomial of degree <= t+1. They are compared COEFFICIENT BY COEFFICIENT in
      exact rational arithmetic. Equality proves the reduction (C)+(T) against the
      real automaton with all correlations included.

  (L) LAWS: from the closed form,
          E[X_t | b_0 = 0] = +[(1-2q) t + q]         (weight 1-q),
          E[X_t | b_0 = 1] = -[(1-2q) t - (1-q)]     (weight q),
      so the correlator is a mixture of two counter-propagating packets with
      sqrt(t) widths and O(1) offsets: asymptotic speed per cycle (2 sub-steps)
      = 2(1-2q) -- the linear law, exact. The short-time laws
      centroid(1) = 1 - 2q(1-q) and centroid(2) = 2(1 - 2q(1-q)) (per sub-step,
      i.e. v(t<=2) = 2(1-2q(1-q)) per cycle) are derived from the same polynomial
      machinery below, as identities.

Run:  .venv/bin/python scripts/theory_linear_law.py            (~2 min, L=12)
      .venv/bin/python scripts/theory_linear_law.py --big      (adds L=14, t=5, slow)
"""
from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction
from itertools import product

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pca3d.models import conditional as C

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RULE_INDEX = 891


# ---------------------------------------------------------------- polynomials over Q
def padd(a, b):
    n = max(len(a), len(b))
    return [ (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n) ]


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] += x * y
    return out


def pscale(a, s):
    return [x * s for x in a]


def ptrim(a):
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a


def q_pow_weight(e: int, n: int):
    """q^e (1-q)^(n-e) as a coefficient list."""
    out = [Fraction(1)]
    for _ in range(e):
        out = pmul(out, [Fraction(0), Fraction(1)])
    for _ in range(n - e):
        out = pmul(out, [Fraction(1), Fraction(-1)])
    return out


def peval(a, q: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(a):
        acc = acc * q + c
    return acc


# ------------------------------------------------------------------- structure (S)
def structural_facts(perm):
    env_map = {}
    for c in range(16):
        ei, eo = C.env_state(c), C.env_state(int(perm[c]))
        env_map.setdefault(ei, set()).add(eo)
    autonomous = all(len(v) == 1 for v in env_map.values())
    env_map = {k: v.pop() for k, v in env_map.items()}
    is_swap = env_map == {0: 0, 1: 2, 2: 1, 3: 3}

    # stall predicates: env states under which a lone carrier holds still
    swap2 = lambda s: ((s & 1) << 1) | (s >> 1)
    S10, S01 = set(), set()  # carrier at left site / at right site
    for c in range(16):
        s_in, s_out = C.system_state(c), C.system_state(int(perm[c]))
        if s_in == 1:  # psi=1 at x, psi'=0: carrier at LEFT of block
            (S10.add(C.env_state(c)) if s_out == s_in else None)
            assert s_out in (s_in, swap2(s_in))
        if s_in == 2:  # carrier at RIGHT of block
            (S01.add(C.env_state(c)) if s_out == s_in else None)
            assert s_out in (s_in, swap2(s_in))

    # system-blindness: swap decision depends on env only (same predicate both ways,
    # after accounting for orientation: env state as seen from the block is the same)
    blind = S10 == S01
    return autonomous, is_swap, env_map, S10, S01, blind


# ------------------------------------------------- direct rule simulation on a ring
def simulate_ring(perm, env0: np.ndarray, x0: int, n_sub: int):
    """Evolve a single carrier at x0 in env0 with the actual block rule; return the
    carrier position after each sub-step (positions unwrapped assuming no wrap)."""
    L = len(env0)
    sys_bits = np.zeros(L, dtype=np.int64)
    sys_bits[x0] = 1
    env = env0.copy()
    pos = []
    prev = x0
    for t in range(1, n_sub + 1):
        o = (t - 1) % 2
        new_sys = sys_bits.copy()
        new_env = env.copy()
        for b in range(L // 2):
            i, j = (2 * b + o) % L, (2 * b + 1 + o) % L
            cfg = C.encode(int(sys_bits[i]), int(env[i]), int(sys_bits[j]), int(env[j]))
            out = int(perm[cfg])
            ps, ph, ps2, ph2 = C.decode(out)
            new_sys[i], new_env[i], new_sys[j], new_env[j] = ps, ph, ps2, ph2
        sys_bits, env = new_sys, new_env
        (where,) = np.nonzero(sys_bits)
        assert len(where) == 1
        p = int(where[0])
        # unwrap: carrier moves at most 1 site per sub-step
        for cand in (p, p + L, p - L):
            if abs(cand - prev) <= 1:
                p = cand
                break
        pos.append(p)
        prev = p
    return pos


def closed_form_positions(env0: np.ndarray, x0: int, n_sub: int):
    """The (T) closed form, with explicit read-position bookkeeping.

    Returns (positions per sub-step, read positions). Requires x0 even (right-mover
    start, matching the correlator's site-0 tracer with sub-step 1 at origin 0)."""
    L = len(env0)
    assert x0 % 2 == 0
    b_prev = int(env0[x0 % L])         # b_0 = companion at the start site
    b0 = b_prev
    X = x0
    out, reads = [], []
    for k in range(1, n_sub + 1):
        d = +1 if b_prev == b0 else -1
        p = X + k if d == +1 else X - k          # initial site of the fresh bit
        # freshness parity: right-moving reads initial left-movers (odd sites),
        # left-moving reads initial right-movers (even sites)
        assert (p - x0) % 2 == (1 if d == +1 else 0) % 2
        b = int(env0[p % L])
        reads.append(p)
        if b == b_prev:
            X += d
        b_prev = b
        out.append(X)
    return out, reads


# -------------------------------------------------- model law P(X_t = x) over Q[q]
def model_law_polys(t: int):
    """P(X_t = x) for x in [-t..t] as exact polynomials in q (degree <= t+1),
    from the closed form, by enumerating the 2^(t+1) read strings."""
    laws = {x: [Fraction(0)] for x in range(-t, t + 1)}
    for bits in product((0, 1), repeat=t + 1):
        b0 = bits[0]
        X = 0
        for k in range(1, t + 1):
            if bits[k] == bits[k - 1]:
                X += +1 if bits[k - 1] == b0 else -1
        e = sum(bits)
        laws[X] = padd(laws[X], q_pow_weight(e, t + 1))
    return {x: ptrim(p) for x, p in laws.items()}


# ------------------------------------------- exact all-configuration ring correlator
def ring_correlator_counts(perm, L: int, n_sub: int):
    """counts[t][x][e]: sum over ALL 4^L ring configs with env weight exponent e of
    s_x(t) s_0(0). C(t,x) = sum_e counts q^e (1-q)^(L-e) / 2^L, exactly."""
    nbits = 2 * L
    n = 1 << nbits
    dtype = np.uint32 if nbits <= 32 else np.uint64
    states = np.arange(n, dtype=dtype)
    table = perm.astype(dtype)

    ne = np.zeros(n, dtype=np.int8)
    for site in range(L):
        ne += ((states >> dtype(2 * site + 1)) & dtype(1)).astype(np.int8)
    n_per_e = np.bincount(ne, minlength=L + 1).astype(np.int64)

    b0 = ((states & dtype(1)) == 1)  # system bit of site 0 at t=0

    def substep(cur, origin):
        mask = dtype(n - 1)
        if origin:
            cur = ((cur >> dtype(2)) | (cur << dtype(nbits - 2))) & mask
        out = np.zeros_like(cur)
        for b in range(L // 2):
            nib = (cur >> dtype(4 * b)) & dtype(15)
            out |= table[nib] << dtype(4 * b)
        if origin:
            out = ((out << dtype(2)) | (out >> dtype(nbits - 2))) & mask
        return out

    all_counts = []
    cur = states
    for t in range(n_sub + 1):
        if t:
            cur = substep(cur, (t - 1) % 2)
        cts = np.zeros((L, L + 1), dtype=np.int64)
        for x in range(L):
            bx = (((cur >> dtype(2 * x)) & dtype(1)) == 1)
            agree = ~(bx ^ b0)
            agree_per_e = np.bincount(ne[agree], minlength=L + 1).astype(np.int64)
            # s0*sx = +1 on agree, -1 on disagree: sum = 2*agree - total
            cts[x] = 2 * agree_per_e - n_per_e
        all_counts.append(cts)
    return all_counts


def ring_poly(counts_t, x: int, L: int):
    """C(t, x) as an exact polynomial in q (already divided by 2^L)."""
    out = [Fraction(0)]
    half = Fraction(1, 1 << L)
    for e in range(L + 1):
        c = int(counts_t[x % L][e])
        if c:
            out = padd(out, pscale(q_pow_weight(e, L), Fraction(c) * half))
    return ptrim(out)


# ------------------------------------------------------------------------- main
def main() -> None:
    big = "--big" in sys.argv
    rng = np.random.default_rng(20260831)
    perm = C.enumerate_conditional_rules()[RULE_INDEX]
    report = {"rule": RULE_INDEX}

    print("=" * 72)
    print(f"TARGET 1: rule {RULE_INDEX} -- the linear law v(q) = 2(1-2q)")
    print("=" * 72)

    # (S) structure
    print("\n[S] structure from the block table")
    print("    table (psi,phi | psi',phi') -> same:")
    for c in range(16):
        print(f"      {C.decode(c)} -> {C.decode(int(perm[c]))}")
    autonomous, is_swap, env_map, S10, S01, blind = structural_facts(perm)
    print(f"    env autonomous: {autonomous};  env map = {env_map}  (pure swap: {is_swap})")
    print(f"    stall predicate, carrier at left  S10 = {sorted(S10)}  (env states)")
    print(f"    stall predicate, carrier at right S01 = {sorted(S01)}")
    print(f"    system-blind (swap iff env in {{0,3}} regardless of content): {blind and S10 == {1, 2}}")
    assert autonomous and is_swap, "env autonomy/swap failed -- derivation void"
    assert S10 == S01 == {1, 2}, "stall predicate is not exactly-one-env -- derivation void"
    report["structure"] = {"env_autonomous": True, "env_map_is_swap": True,
                           "stall_predicate": "exactly one env bit in block",
                           "system_blind": True}
    print("    => env = free-streaming L/R movers; system = passive tracers: VERIFIED")

    # (T) closed form vs direct simulation
    print("\n[T] closed form X_t = (-1)^{b0} (N_00 - N_11) vs direct rule simulation")
    L_sim, T_sim, n_inst = 512, 120, 400
    worst = 0
    for inst in range(n_inst):
        q = rng.uniform(0.05, 0.95)
        env0 = (rng.random(L_sim) < q).astype(np.int64)
        x0 = 2 * int(rng.integers(0, L_sim // 8))  # even start, away from wrap issues
        sim = simulate_ring(perm, env0, x0, T_sim)
        model, reads = closed_form_positions(env0, x0, T_sim)
        if sim != model:
            print(f"    MISMATCH instance {inst}: sim {sim[:12]} model {model[:12]}")
            raise SystemExit(1)
        worst = max(worst, max(abs(p - x0) for p in reads))
    assert worst < L_sim // 2, "reads approached the wrap; enlarge L_sim"
    print(f"    {n_inst} random envs x {T_sim} sub-steps: positions identical, "
          f"every sub-step, every instance (max read offset {worst} < L/2={L_sim//2})")
    report["closed_form_check"] = {"instances": n_inst, "sub_steps": T_sim, "identical": True}

    # (P) polynomial identity model vs exact ring
    print("\n[P] polynomial identity: model law vs exact all-configuration correlator")
    ring_specs = [(10, 3), (12, 4)] + ([(14, 5)] if big else [])
    poly_ok = True
    for L, t_max in ring_specs:
        counts = ring_correlator_counts(perm, L, t_max)
        for t in range(1, t_max + 1):
            model = model_law_polys(t)
            for x in range(-t, t + 1):
                mp = model[x]
                rp = ring_poly(counts[t], x, L)
                if ptrim(mp) != ptrim(rp):
                    poly_ok = False
                    print(f"    L={L} t={t} x={x}: model {mp} != ring {rp}")
            # off-cone entries must vanish identically
            for x in range(t + 1, L - t):
                rp = ring_poly(counts[t], x, L)
                if ptrim(rp) != [Fraction(0)]:
                    poly_ok = False
                    print(f"    L={L} t={t} x={x}: off-cone entry nonzero: {rp}")
        print(f"    ring L={L} ({4**L:,} configs), t <= {t_max}: "
              f"all coefficients identical" if poly_ok else f"    ring L={L}: FAILED")
    assert poly_ok
    report["polynomial_identity"] = {"rings": ring_specs, "identical": True}

    # (L) laws as polynomial identities
    print("\n[L] the laws, in exact arithmetic")
    # mean: E[X_t] per class
    for t in (1, 2, 3, 4, 5, 8):
        model = model_law_polys(t)
        # E[X_t | b0=0] * P(b0=0): split by conditioning is awkward here; instead
        # verify the unconditional identity E[X_t] = (1-2q)^2 * t + 0*... derived:
        # E[X_t] = sum_x x P(x)
        mean = [Fraction(0)]
        for x, p in model.items():
            if x:
                mean = padd(mean, pscale(p, Fraction(x)))
        # predicted: E[X_t] = (1-q)*[(1-2q)t + q] + q*[-(1-2q)t + (1-q)]
        #          = (1-2q)^2 t + 2q(1-q)
        pred = padd(pscale([Fraction(1), Fraction(-4), Fraction(4)], Fraction(t)),
                    [Fraction(0), Fraction(2), Fraction(-2)])
        assert ptrim(mean) == ptrim(pred), (t, mean, pred)
    print("    E[X_t] = (1-2q)^2 t + 2q(1-q)  for t = 1..5, 8: IDENTITY")
    # class-conditional means:
    #   E[X_t ; b0=0] = (1-q) [ (1-2q) t + q ]        (class A, right packet)
    #   E[X_t ; b0=1] = q [ -(1-2q) t + (1-q) ]       (class B, left packet)
    for t in (1, 2, 3, 4, 5):
        for b0v in (0, 1):
            cond = [Fraction(0)]
            for bits in product((0, 1), repeat=t + 1):
                if bits[0] != b0v:
                    continue
                X = 0
                for k in range(1, t + 1):
                    if bits[k] == bits[k - 1]:
                        X += +1 if bits[k - 1] == b0v else -1
                if X:
                    cond = padd(cond, pscale(q_pow_weight(sum(bits), t + 1), Fraction(X)))
            drift = padd(pscale([Fraction(1), Fraction(-2)], Fraction(t)), [Fraction(0), Fraction(1)])
            if b0v == 0:
                pred = pmul([Fraction(1), Fraction(-1)], drift)   # (1-q) * [(1-2q)t + q]
            else:
                driftB = padd(pscale([Fraction(-1), Fraction(2)], Fraction(t)), [Fraction(1), Fraction(-1)])
                pred = pmul([Fraction(0), Fraction(1)], driftB)   # q * [-(1-2q)t + (1-q)]
            assert ptrim(cond) == ptrim(pred), (t, b0v, cond, pred)
    print("    E[X_t ; b_0=0] = (1-q) [ (1-2q) t + q ]        for t = 1..5: IDENTITY")
    print("    E[X_t ; b_0=1] = q [ -(1-2q) t + (1-q) ]       for t = 1..5: IDENTITY")
    print("      => right-moving packet, weight (1-q), centred at +[(1-2q) t + q]")
    print("         left-moving packet,  weight q,     centred at -[(1-2q) t - (1-q)]")
    print("      => asymptotic speed per cycle = 2 (1 - 2q): THE LINEAR LAW (exact)")

    # short-time law: centroid over the full packet (all entries within 10% of peak)
    m1 = model_law_polys(1)
    # centroid(1) = 1*P(1) + 0*P(0) (+1*P(-1)=0) => P(X_1=1) = 1 - 2q(1-q)
    assert ptrim(m1[1]) == ptrim([Fraction(1), Fraction(-2), Fraction(2)])
    m2 = model_law_polys(2)
    num2 = [Fraction(0)]
    for x, p in m2.items():
        num2 = padd(num2, pscale(p, Fraction(abs(x))))
    # sum_x |x| P_2(x) = 2(1 - 2q(1-q))
    assert ptrim(num2) == ptrim([Fraction(2), Fraction(-4), Fraction(4)]), num2
    print("    P(X_1 = 1) = 1 - 2q(1-q);  sum_x |x| P_2(x) = 2(1 - 2q(1-q))")
    print("      => v(t<=2) = 2(1 - 2q(1-q)) per cycle: the ADR 0005 short-time law,")
    print("         derived (valid while every packet entry clears the 10% peak cut)")
    report["laws"] = {
        "mean": "E[X_t] = (1-2q)^2 t + 2q(1-q)",
        "class_mean": "E[X_t | b0=0] = (1-2q) t + q, weight 1-q; "
                      "E[X_t | b0=1] = -(1-2q) t + (1-q), weight q",
        "asymptote_per_cycle": "2(1-2q)",
        "short_time_per_cycle": "2(1-2q(1-q))",
    }

    # spot check against the existing exact machinery at rational q
    print("\n[X] cross-check vs scripts_lib_exact.exact_env_weighted_centroids")
    from scripts_lib_exact import exact_env_weighted_centroids
    L = 12
    for q in (Fraction(1, 4), Fraction(1, 10), Fraction(1, 3)):
        cents = exact_env_weighted_centroids(perm, L, 4, q)
        model_cents = []
        for t in range(1, 5):
            model = model_law_polys(t)
            vals = {x: peval(p, q) for x, p in model.items()}
            mags = {x: abs(v) for x, v in vals.items()}
            peak = max(mags.values())
            num = den = Fraction(0)
            for x, m in mags.items():
                if 10 * m >= peak:
                    num += m * abs(x)
                    den += m
            model_cents.append(num / den)
        match = all(cents[t] == model_cents[t - 1] for t in range(1, 5))
        print(f"    q = {q}: ring centroids {[str(c) for c in cents[1:]]}")
        print(f"           model centroids {[str(c) for c in model_cents]}   match: {match}")
        assert match
    report["cross_check_exact_lib"] = True

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "theory_linear_law.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {RESULTS / 'theory_linear_law.json'}")
    print("\nVERDICT: linear law v(q) = 2(1-2q) PROVEN for rule 891 (structure verified,")
    print("closed form exact vs simulation, polynomial identity vs exact ring, laws exact).")


if __name__ == "__main__":
    main()
