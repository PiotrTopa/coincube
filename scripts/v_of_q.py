#!/usr/bin/env python
"""Phase 2 hinge experiment: does the dressed speed move continuously with the vacuum?

Pre-registered setup (written before the sweep ran):

  - Rules: all 32 conditional-propagation rules that conserve environment count per
    block. For these, the product vacuum Bernoulli(q) on the environment channel
    (system at 1/2) is EXACTLY stationary at every q -- no thermalisation argument --
    because the rule conserves each species' particle number per block. They satisfy
    R4.5 automatically. The finite-ray theorem pins their empty-vacuum one-particle
    sector; the object measured here is the carrier dressed by a finite-density
    environment, which the theorem does not constrain (the FCHC-sound loophole).

  - Knob: q from 0 to 1 in steps of 0.05, multiple seeds per point, GPU.

  - Predictions to check against (from the stall signatures):
      * stall-on-nothing family: v = 2 at every q (control);
      * stall-on-own-site / stall-on-destination families: v(0) = 2 exactly,
        decreasing in q;
      * stall-on-exactly-one-env family: v(0) = v(1) = 2 with a dip between;
      * stall-on-empty-or-full family: v(0) = v(1) = 0 with a maximum between.

  - The hinge: if v(q) takes many distinct values along the sweep (not a step
    function), the speed set is continuous and the rational-speed constraint of ADR
    0004 is a property of the symmetric vacuum, not of the rule class. 3D speed
    equalisation (which needs the irrational ratio 1/sqrt2) then becomes a tuning
    problem. If v(q) is a staircase of the same few rationals, the gate narrows.

    .venv/bin/python scripts/v_of_q.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from pca3d.analysis.correlations import measure_transport
from pca3d.core.lattice import Lattice
from pca3d.models import conditional as C
from pca3d.models.generic import BlockAutomaton, RuleCycle

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

L, T, E = 2048, 150, 64
SEEDS = 3
QS = [round(0.1 * i, 2) for i in range(11)]

#: measurement order: decisive stall signatures first, so the hinge is answered in the
#: first minutes of the sweep rather than after two hours of controls
PRIORITY = [
    ((1, 2), (2, 1)),                          # stall on destination occupied
    ((1, 1), (2, 2)),                          # stall on own site occupied
    ((1, 1), (1, 2), (2, 1), (2, 2)),          # stall on exactly-one-env (dip family)
    ((1, 0), (1, 3), (2, 0), (2, 3)),          # move on exactly-one-env (bump family)
]


def env_conserving_rules():
    rules = C.enumerate_conditional_rules()
    pc = lambda v: bin(v).count("1")
    out = []
    for i, p in enumerate(rules):
        if all(pc(C.env_state(c)) == pc(C.env_state(int(p[c]))) for c in range(16)):
            out.append((i, p))
    return out


def stall_signature(p) -> tuple:
    sig = []
    for c in range(16):
        sv = C.system_state(c)
        if sv in (1, 2) and C.system_state(int(p[c])) == sv:
            sig.append((sv, C.env_state(c)))
    return tuple(sorted(sig))


def cycle(p):
    lat = Lattice(shape=(L,), n_species=2)
    return RuleCycle(steps=(
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=p, origin=(0,)),
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=p, origin=(1,)),
    ))


def main() -> None:
    rules = env_conserving_rules()
    def prio(item):
        sig = stall_signature(item[1])
        return PRIORITY.index(sig) if sig in PRIORITY else len(PRIORITY)
    rules.sort(key=prio)
    print(f"{len(rules)} env-conserving rules; q grid {QS[0]}..{QS[-1]} x {SEEDS} seeds")

    out = []
    for n, (idx, p) in enumerate(rules):
        cyc = cycle(p)
        sig = stall_signature(p)
        row = {"rule": int(idx), "stalls": [list(s) for s in sig], "q": [], "v": [],
               "v_sem": [], "alpha": []}
        for q in QS:
            vs, als = [], []
            for sd in range(SEEDS):
                r = measure_transport(
                    cyc, n_steps=T, ensemble=E, seed=1000 * sd + n, species=0,
                    v_max=2.0, estimator="centroid", use_gpu=True,
                    densities=(0.5, q),
                )
                if np.isfinite(r.velocity):
                    vs.append(r.velocity)
                    als.append(r.exponent)
            if not vs:
                continue
            row["q"].append(q)
            row["v"].append(float(np.mean(vs)))
            row["v_sem"].append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))) if len(vs) > 1 else 0.0)
            row["alpha"].append(float(np.nanmean(als)))
        out.append(row)
        vs = row["v"]
        qq = row["q"]
        prof = " ".join(f"{q:.1f}:{v:.3f}" for q, v in zip(qq, vs))
        print(f"rule {idx:4d} sig={sig}\n    {prof}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "v_of_q.json"
    path.write_text(json.dumps({"L": L, "T": T, "E": E, "seeds": SEEDS, "rows": out}, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
