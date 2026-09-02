#!/usr/bin/env python
"""Theory strike, Target 3: a dressed-carrier rationality theorem.

CLAIM PROVED HERE (the autonomous-streaming subclass):

  For every conditional-propagation rule whose environment update is autonomous
  (independent of system content) and equal to the block swap (phi, phi') ->
  (phi', phi), the density dynamics of a lone carrier on the Bernoulli(q)
  environment is a finite Markov-additive process: the pair (direction d,
  companion bit c) is a Markov chain on 4 states whose transition probabilities
  are {q, 1-q}-valued, driven by i.i.d. fresh reads (the freshness/monotone-gap
  argument of theory_linear_law.py, which is geometric and rule-independent
  within this subclass). Consequently every ballistic packet speed is

      v_i(q) = stationary drift of a recurrent class = RATIONAL FUNCTION of q
               with integer coefficients,

  so at every RATIONAL q -- in particular the ADR 0004 symmetric vacuum
  q = 1/2 -- every packet speed is a rational number. The rational-speed
  observation of ADR 0004 is, for this subclass, a theorem; and the continuity
  in q that ADR 0005 measured is the trivial continuity of a rational function.

The chain (derived exactly as for rule 891, per-rule stall predicates S10/S01
read from the block table):

    state (R, c): carrier right-moving, companion = co-moving right-mover c at
                  its own site; reads fresh oncoming left-mover b:
                  block env state = c + 2b; if in S10: stall -> (L, b), X += 0
                  else: move -> (R, c), X += 1
    state (L, c): mirror with env state b + 2c against S01, X -= 1 on move.

Machine checks in this script:
  1. census of the 32 env-count-conserving rules (autonomy, env map, predicates);
  2. for every autonomous-swap rule: the chain vs direct rule simulation,
     exact trajectory equality on random environments;
  3. exact recurrent-class analysis: stationary drifts solved in rational
     arithmetic on a q-grid, reconstructed as rational functions (Cramer degree
     bound), verified on held-out q values;
  4. drift table at q = 1/2: all rational (the theorem's corollary);
  5. confrontation with results/v_of_q.json (measured curves).

NOT covered (stated honestly): rules with frozen (identity) autonomous
environments -- the carrier re-reads the same bits, no freshness, and J5's
localisation is the observed outcome; and non-autonomous rules (the CPA / 3/2
family), for which a conjecture is recorded here (not proven).

Run:  .venv/bin/python scripts/theory_rationality.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pca3d.models import conditional as C

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


# ------------------------------------------------------------------ rule census
def env_conserving_rules():
    rules = C.enumerate_conditional_rules()
    pc = lambda v: bin(v).count("1")
    out = []
    for i, p in enumerate(rules):
        if all(pc(C.env_state(c)) == pc(C.env_state(int(p[c]))) for c in range(16)):
            out.append((i, p))
    return out


SWAP_MAP = {0: 0, 1: 2, 2: 1, 3: 3}
IDENTITY_MAP = {0: 0, 1: 1, 2: 2, 3: 3}


def census_rule(perm):
    env_maps = {}  # sector -> {env_in: env_out}
    for c in range(16):
        sec = C.system_state(c)
        env_maps.setdefault(sec, {})[C.env_state(c)] = C.env_state(int(perm[c]))
    maps = list(env_maps.values())
    autonomous = all(m == maps[0] for m in maps)
    envmap = maps[0] if autonomous else None
    kind = None
    if autonomous:
        if envmap == SWAP_MAP:
            kind = "swap"
        elif envmap == IDENTITY_MAP:
            kind = "identity"
        else:
            kind = f"other:{envmap}"
    vacuum_kind = ("swap" if env_maps[0] == SWAP_MAP
                   else "identity" if env_maps[0] == IDENTITY_MAP
                   else "other")
    S10, S01 = set(), set()
    for c in range(16):
        s_in, s_out = C.system_state(c), C.system_state(int(perm[c]))
        if s_in == 1 and s_out == 1:
            S10.add(C.env_state(c))
        if s_in == 2 and s_out == 2:
            S01.add(C.env_state(c))
    return {"autonomous": autonomous, "env_kind": kind, "vacuum_kind": vacuum_kind,
            "S10": sorted(S10), "S01": sorted(S01), "blind": S10 == S01}


# ------------------------------------------------- the generalized tracer chain
# states 0..3: (d, c) with d in {R, L}, c in {0, 1}: index = (0 if R else 2) + c
#
# c = the env bit at the carrier's OWN site (its co-moving companion). Each
# sub-step reads one FRESH (virgin) env bit b at the other site of the block; the
# block table then decides move/stall AND the two output env bits, of which the
# one at the carrier's new site becomes the new companion. Freshness holds for
# every rule whose carrier-free (zero-sector) env map is the swap: virgin bits
# stream ballistically, the carrier's writes land only on bits whose signed gap
# to the carrier is monotone thereafter (mover type is fixed by slot parity), so
# no written bit is ever read again.
def chain_step_outcome(state: int, b: int, perm):
    """-> (new_state, displacement) for read bit b, from the block table."""
    d, c = ("R" if state < 2 else "L"), state & 1
    if d == "R":
        cfg = C.encode(1, c, 0, b)              # carrier at left, env (c, b)
        ps, ph, ps2, ph2 = C.decode(int(perm[cfg]))
        assert (ps, ps2) in ((1, 0), (0, 1))
        if (ps, ps2) == (0, 1):                 # moved right
            return 0 + ph2, +1                  # (R, phi'_out)
        return 2 + ph, 0                        # stalled -> (L, phi_out)
    cfg = C.encode(0, b, 1, c)                  # carrier at right, env (b, c)
    ps, ph, ps2, ph2 = C.decode(int(perm[cfg]))
    assert (ps, ps2) in ((1, 0), (0, 1))
    if (ps, ps2) == (1, 0):                     # moved left
        return 2 + ph, -1                       # (L, phi_out)
    return 0 + ph2, 0                           # stalled -> (R, phi'_out)


def chain_step_dist(state: int, q: Fraction, perm):
    """-> list of (new_state, displacement, probability)."""
    out = []
    for b in (0, 1):
        pb = q if b == 1 else 1 - q
        s2, dx = chain_step_outcome(state, b, perm)
        out.append((s2, dx, pb))
    return out


def chain_simulate(env0: np.ndarray, x0: int, n_sub: int, perm):
    """Deterministic trajectory of the chain reading the actual (virgin) env bits,
    with the read-position bookkeeping of theory_linear_law.py. x0 must be even."""
    L = len(env0)
    assert x0 % 2 == 0
    state = 0 + int(env0[x0 % L])               # (R, phi(x0))
    X = x0
    out = []
    for k in range(1, n_sub + 1):
        p = X + k if state < 2 else X - k
        b = int(env0[p % L])
        state, dx = chain_step_outcome(state, b, perm)
        X += dx
        out.append(X)
    return out


def simulate_ring(perm, env0: np.ndarray, x0: int, n_sub: int):
    """Single carrier evolved by the actual block rule (same as theory_linear_law)."""
    L = len(env0)
    sys_bits = np.zeros(L, dtype=np.int64)
    sys_bits[x0] = 1
    env = env0.copy()
    pos, prev = [], x0
    for t in range(1, n_sub + 1):
        o = (t - 1) % 2
        new_sys, new_env = sys_bits.copy(), env.copy()
        for b in range(L // 2):
            i, j = (2 * b + o) % L, (2 * b + 1 + o) % L
            cfg = C.encode(int(sys_bits[i]), int(env[i]), int(sys_bits[j]), int(env[j]))
            ps, ph, ps2, ph2 = C.decode(int(perm[cfg]))
            new_sys[i], new_env[i], new_sys[j], new_env[j] = ps, ph, ps2, ph2
        sys_bits, env = new_sys, new_env
        (where,) = np.nonzero(sys_bits)
        assert len(where) == 1
        p = int(where[0])
        for cand in (p, p + L, p - L):
            if abs(cand - prev) <= 1:
                p = cand
                break
        pos.append(p)
        prev = p
    return pos


# ------------------------------------------ exact class analysis of the chain
def transition_matrix(q: Fraction, perm):
    P = [[Fraction(0)] * 4 for _ in range(4)]
    disp = [[Fraction(0)] * 4 for _ in range(4)]
    for s in range(4):
        for s2, dx, p in chain_step_dist(s, q, perm):
            P[s][s2] += p
            disp[s][s2] += Fraction(dx) * p
    return P, disp


def support_graph(perm):
    """Edges that exist for generic q in (0,1)."""
    q = Fraction(1, 3)  # generic
    P, _ = transition_matrix(q, perm)
    return [[i for i in range(4) if P[s][i] != 0] for s in range(4)]


def closed_classes(adj):
    """Communicating classes that are closed (recurrent for generic q)."""
    n = len(adj)
    reach = [set([i]) for i in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            new = set(reach[i])
            for j in adj[i]:
                new |= reach[j]
            if new != reach[i]:
                reach[i] = new
                changed = True
    classes = []
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        comm = {j for j in reach[i] if i in reach[j]}
        if comm & seen:
            seen |= comm
            continue
        seen |= comm
        closed = all(set(adj[j]) <= reach[i] and
                     all(k in comm for k in adj[j] if True) or True for j in comm)
        # a communicating class is closed iff no edge leaves it
        closed = all(all(k in comm for k in adj[j]) for j in comm)
        classes.append((sorted(comm), closed))
    return classes


def solve_linear(A, b):
    """Exact Gaussian elimination over Fractions. A: list of rows."""
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            raise ValueError("singular")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[r][n] for r in range(n)]


def class_drift(states, q, perm):
    """Stationary drift of a closed class at exact rational q (per sub-step)."""
    P, disp = transition_matrix(q, perm)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    if n == 1:
        s = states[0]
        return sum(disp[s][t] for t in range(4))
    # pi (P - I) = 0 with sum pi = 1: replace last equation
    A = [[P[t][s] - (1 if s == t else 0) for t in states] for s in states]
    A[-1] = [Fraction(1)] * n
    b = [Fraction(0)] * (n - 1) + [Fraction(1)]
    pi = solve_linear(A, b)
    return sum(pi[i] * sum(disp[s][t] for t in range(4)) for i, s in enumerate(states))


def absorption_weights(classes, q, perm):
    """P(absorbed into each closed class | start (R, c0), c0 ~ Bernoulli(q))."""
    P, _ = transition_matrix(q, perm)
    closed_states = {s for cls, closed in classes if closed for s in cls}
    weights = []
    for cls, closed in classes:
        if not closed:
            continue
        target = set(cls)
        # h(s) = P(hit this class first | start s): h=1 on class, 0 on other closed,
        # harmonic elsewhere
        h = {}
        unknown = [s for s in range(4) if s not in closed_states]
        for s in closed_states:
            h[s] = Fraction(1) if s in target else Fraction(0)
        if unknown:
            A = [[P[s][t] - (1 if s == t else 0) for t in unknown] for s in unknown]
            b = [-sum(P[s][t] * h[t] for t in range(4) if t not in unknown)
                 for s in unknown]
            sol = solve_linear(A, b)
            for s, v in zip(unknown, sol):
                h[s] = v
        w = (1 - q) * h[0] + q * h[1]   # start (R,0) w.p. 1-q, (R,1) w.p. q
        weights.append(w)
    return weights


# ------------------------------------------ rational function reconstruction
def rational_interpolate(samples, dmax=4):
    """samples: list of (q, value) Fractions. Find P/Q, deg <= dmax, exact."""
    for dn in range(0, dmax + 1):
        for dd in range(0, dmax + 1):
            n_unk = (dn + 1) + (dd + 1)
            if len(samples) < n_unk + 1:
                continue
            rows, rhs = [], []
            for qv, vv in samples[: n_unk + 2]:
                # P(q) - v * Q(q) = 0
                rows.append([qv ** j for j in range(dn + 1)]
                            + [-vv * qv ** j for j in range(dd + 1)])
                rhs.append(Fraction(0))
            # find nullspace vector by fixing leading den coeff candidates
            for pin in range(dd + 1):
                A = [r[:] for r in rows]
                nn = dn + 1 + dd + 1
                # set Q coeff 'pin' = 1
                A2 = [r[: dn + 1 + pin] + r[dn + 2 + pin:] for r in A]
                b2 = [-r[dn + 1 + pin] for r in A]
                # least-squares-free exact solve of the first nn-1 equations
                try:
                    sol = solve_linear([row[: nn - 1] for row in A2[: nn - 1]],
                                       b2[: nn - 1])
                except ValueError:
                    continue
                num = sol[: dn + 1]
                den = sol[dn + 1:]
                den = den[:pin] + [Fraction(1)] + den[pin:]
                def ev(cs, x):
                    a = Fraction(0)
                    for cf in reversed(cs):
                        a = a * x + cf
                    return a
                if all(ev(den, qv) != 0 and ev(num, qv) / ev(den, qv) == vv
                       for qv, vv in samples):
                    return num, den
    raise ValueError("no rational function of degree <= dmax fits")


def poly_str(cs):
    terms = []
    for i, cf in enumerate(cs):
        if cf == 0:
            continue
        t = (f"{cf}" if i == 0 else
             (f"{cf}*q" if i == 1 else f"{cf}*q^{i}"))
        terms.append(t)
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


# ------------------------------------------------------------------------- main
def main() -> None:
    rules32 = env_conserving_rules()
    rng = np.random.default_rng(7)
    report = {"n_rules": len(rules32), "census": {}, "autonomous_swap": {}}
    print("=" * 76)
    print("TARGET 3: rationality of dressed speeds -- census + theorem for the")
    print("autonomous-streaming subclass of the 32 env-conserving rules")
    print("=" * 76)

    groups = {"autonomous-swap": [], "streaming-vacuum": [], "frozen-vacuum": []}
    for idx, perm in rules32:
        info = census_rule(perm)
        report["census"][idx] = info
        if info["env_kind"] == "swap":
            groups["autonomous-swap"].append(idx)
        elif info["vacuum_kind"] == "swap":
            groups["streaming-vacuum"].append(idx)
        else:
            groups["frozen-vacuum"].append(idx)

    print(f"\n[C] census of the 32 rules:")
    print(f"    tier 1, fully autonomous swap env  : {groups['autonomous-swap']}")
    print(f"      (correlator theorem at ANY filling: passive-tracer reduction)")
    print(f"    tier 2, streaming vacuum only      : {groups['streaming-vacuum']}")
    print(f"      (single-carrier theorem: carrier-free blocks stream, freshness holds)")
    print(f"    frozen vacuum (zero-sector identity): {groups['frozen-vacuum']}")
    print(f"      (no freshness -> outside the theorem; J5 localisation regime)")

    vq = json.loads((RESULTS / "v_of_q.json").read_text())
    vq_rows = {r["rule"]: r for r in vq["rows"]}
    names = {0: "(R,0)", 1: "(R,1)", 2: "(L,0)", 3: "(L,1)"}
    qgrid_check = [Fraction(k, 17) for k in range(1, 17)]

    def ev(cs, x):
        a = Fraction(0)
        for cf in reversed(cs):
            a = a * x + cf
        return a

    print("\n[T] streaming-vacuum rules (tiers 1+2): chain vs direct rule simulation,")
    print("    exact class analysis, v(q) rational-function reconstruction")
    for idx in groups["autonomous-swap"] + groups["streaming-vacuum"]:
        perm = dict(rules32)[idx]
        info = report["census"][idx]
        tier = 1 if idx in groups["autonomous-swap"] else 2

        # trajectory identity check: the chain reproduces the automaton exactly
        ok = True
        for _ in range(60):
            qf = rng.uniform(0.05, 0.95)
            env0 = (rng.random(400) < qf).astype(np.int64)
            x0 = 2 * int(rng.integers(0, 30))
            if simulate_ring(perm, env0, x0, 80) != chain_simulate(env0, x0, 80, perm):
                ok = False
                break
        assert ok, f"rule {idx}: chain does not reproduce the automaton"

        adj = support_graph(perm)
        classes = closed_classes(adj)
        closed = [cls for cls, c in classes if c]
        entry = {"tier": tier, "S10": info["S10"], "S01": info["S01"],
                 "blind": info["blind"], "trajectory_check": True, "classes": []}
        print(f"\n    rule {idx} (tier {tier}): S10={info['S10']} S01={info['S01']}"
              f"  closed classes: {[[names[s] for s in cls] for cls in closed]}")
        for ci, cls in enumerate(closed):
            samples = [(qv, class_drift(cls, qv, perm)) for qv in qgrid_check[:12]]
            num, den = rational_interpolate(samples)
            for qv in qgrid_check[12:]:
                assert ev(den, qv) != 0
                assert ev(num, qv) / ev(den, qv) == class_drift(cls, qv, perm)
            wsamples = [(qv, absorption_weights(classes, qv, perm)[ci])
                        for qv in qgrid_check[:12]]
            wnum, wden = rational_interpolate(wsamples)
            d_half = class_drift(cls, Fraction(1, 2), perm)
            w_half = absorption_weights(classes, Fraction(1, 2), perm)[ci]
            cyc_num = [2 * cf for cf in num]  # per-cycle = 2 x per-sub-step
            print(f"      class {[names[s] for s in cls]}: "
                  f"v_cycle(q) = ({poly_str(cyc_num)}) / ({poly_str(den)})"
                  f"   weight(q) = ({poly_str(wnum)}) / ({poly_str(wden)})")
            print(f"        at q=1/2: v_cycle = {2 * d_half}  weight = {w_half}"
                  f"   [RATIONAL]")
            entry["classes"].append({
                "states": [names[s] for s in cls],
                "v_cycle_num": [str(c) for c in cyc_num],
                "v_cycle_den": [str(c) for c in den],
                "weight_num": [str(c) for c in wnum],
                "weight_den": [str(c) for c in wden],
                "v_cycle_at_half": str(2 * d_half),
                "weight_at_half": str(w_half),
            })

        # confront with the measured half-filling correlator v(q)
        if idx in vq_rows:
            row = vq_rows[idx]
            pred = []
            for qm, vm, am in zip(row["q"], row["v"], row["alpha"]):
                qf = Fraction(qm).limit_denominator(100)
                if qf in (0, 1):
                    continue
                ds = [(abs(float(class_drift(cls, qf, perm))) * 2,
                       float(absorption_weights(classes, qf, perm)[ci]))
                      for ci, cls in enumerate(closed)]
                dom = max(ds, key=lambda t: t[1])
                fast = max(d for d, w in ds if w > 1e-12)
                pred.append((qm, dom[0], fast, vm, am))
            line = "  ".join(f"q={qm:.1f}: dom {pv:.3f} fast {pf:.3f} "
                             f"| meas {vm:.3f} (a={am:.2f})"
                             for qm, pv, pf, vm, am in pred[:5])
            print(f"      vs v_of_q (half-filling correlator):\n        {line}")
            entry["measured_check"] = [
                {"q": qm, "pred_dominant_v": pv, "pred_fastest_v": pf,
                 "measured_v": vm, "alpha": am}
                for qm, pv, pf, vm, am in pred]
        report["autonomous_swap"][idx] = entry

    print("\n[F] frozen-vacuum rules: no freshness, theorem does not apply;")
    print("    measured behaviour at q=0.5 (J5 localisation regime unless stall-free):")
    for idx in groups["frozen-vacuum"]:
        if idx in vq_rows:
            row = vq_rows[idx]
            i5 = row["q"].index(0.5)
            info = report["census"][idx]
            print(f"    rule {idx}: S10={info['S10']} S01={info['S01']} "
                  f"env_kind={info['env_kind']} vacuum={info['vacuum_kind']}: "
                  f"v(0.5) = {row['v'][i5]:.3f}, alpha = {row['alpha'][i5]:.2f}")
    report["groups"] = groups

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "theory_rationality.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {RESULTS / 'theory_rationality.json'}")


if __name__ == "__main__":
    main()
