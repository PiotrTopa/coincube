#!/usr/bin/env python
"""Verification pass for the v(q) winners + the sqrt2 bisection.

Standing rule from J6: no velocity is real until alpha stays at 1 under a >=4x
extension in time and across seeds. Then: locate q* with v(q*) = sqrt2 for rule 891 by
bisection on the measured curve and pin v(q*) with tight errors. sqrt2 is the exact
diagonal/axis ratio needed for FCHC speed equalisation, so hitting it in 1D is the
existence proof that the knob reaches the value 3D needs.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from pca3d.analysis.correlations import measure_transport
from pca3d.core.lattice import Lattice
from pca3d.models import conditional as C
from pca3d.models.generic import BlockAutomaton, RuleCycle

L, T, E, SEEDS = 4096, 600, 64, 6
rules = C.enumerate_conditional_rules()

def cycle(p):
    lat = Lattice(shape=(L,), n_species=2)
    return RuleCycle(steps=(
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=p, origin=(0,)),
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=p, origin=(1,))))

def v_at(idx, q, seeds=SEEDS, T=T):
    cyc = cycle(rules[idx]); vs=[]; als=[]
    for sd in range(seeds):
        r = measure_transport(cyc, n_steps=T, ensemble=E, seed=7000+sd, species=0,
                              v_max=2.0, estimator="centroid", use_gpu=True,
                              densities=(0.5, q))
        vs.append(r.velocity); als.append(r.exponent)
    vs=np.array(vs)
    return float(vs.mean()), float(vs.std(ddof=1)/np.sqrt(len(vs))), float(np.mean(als))

out={}
print("=== 4x-time verification, ballistic branch ===")
print(f"{'rule':>6}{'q':>7}{'v':>10}{'sem':>9}{'alpha':>8}")
for idx in (891, 883):
    out[idx]=[]
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        v, s, a = v_at(idx, q)
        out[idx].append({"q":q,"v":v,"sem":s,"alpha":a})
        print(f"{idx:>6}{q:>7.2f}{v:>10.4f}{s:>9.4f}{a:>8.3f}")

print("\n=== bisection: rule 891, target v = sqrt2 = 1.41421356 ===")
target = float(np.sqrt(2.0))
lo, hi = 0.10, 0.25   # v decreasing in q on this branch
hist=[]
for it in range(7):
    mid = 0.5*(lo+hi)
    v, s, a = v_at(891, mid, seeds=4, T=400)
    hist.append({"q":mid,"v":v,"sem":s,"alpha":a})
    print(f"  q={mid:.5f}  v={v:.5f} +- {s:.5f}  alpha={a:.3f}")
    if v > target: lo = mid
    else: hi = mid
qstar = 0.5*(lo+hi)
v, s, a = v_at(891, qstar, seeds=10, T=600)
print(f"\nq* = {qstar:.5f}:  v = {v:.5f} +- {s:.5f}  alpha = {a:.4f}")
print(f"target sqrt2 = {target:.5f};  |v - sqrt2| = {abs(v-target):.5f} ({abs(v-target)/max(s,1e-9):.1f} sigma)")
out["bisection"]={"target":target,"history":hist,"qstar":qstar,"v":v,"sem":s,"alpha":a}
pathlib.Path("results/verify_vq.json").write_text(json.dumps(out, indent=2))
print("written: results/verify_vq.json")
