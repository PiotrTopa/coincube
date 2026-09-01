import numpy as np, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from i2_connected import DIM, LX, NCAR, QENV, STATES, create, cycle, evolve

TMAX = 3
MODES_OUT = [4*x+c for x in range(LX) for c in range(4)]

def basis_env_state(e):
    v = np.zeros(DIM); v[e << NCAR] = 1.0
    return v

def g1_series_mix(iota):
    perm, sign = cycle(np.asarray(iota, dtype=bool))
    out = np.zeros((TMAX, len(MODES_OUT)))
    for e in range(1 << (2*LX)):
        n1 = bin(e).count("1")
        w = QENV**n1 * (1-QENV)**(2*LX-n1)     # classical probability
        v0 = basis_env_state(e)
        ket = create(0, v0); vac = v0.copy()
        for t in range(TMAX):
            ket = evolve(ket, perm, sign, 1)
            vac = evolve(vac, perm, sign, 1)
            out[t] += w * np.array([create(i, vac) @ ket for i in MODES_OUT])
    return out

pats = [(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)]
data = {p: g1_series_mix(list(p)) for p in pats}
base = data[(0,0,0)]
print(f"MIXTURE (classically sampled env) vacuum, q={QENV}:")
for g in (0.1, 0.2):
    for t in range(1, TMAX+1):
        num = np.zeros_like(base[0])
        for p in pats:
            w = g**sum(p) * (1-g)**(LX-sum(p))
            num += w * data[p][t-1]
        r = float(num @ base[t-1]) / float(base[t-1] @ base[t-1])
        percyc = np.sign(r)*abs(r)**(1/t)
        law = 1 - 2*g*QENV*(1-QENV)
        print(f"  g={g} t={t}: per-cycle {percyc:.6f}  law {law:.6f}  "
              f"rel dev {abs(percyc-law)/(1-law):.3f}")
