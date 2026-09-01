"""The fermionic two-point function of a signed automaton, by damage spreading.

The object
----------
Over the uniform mixed vacuum ``rho = 2^-M sum_tau |tau><tau|`` (exactly stationary for
any signed permutation), the particle propagator is

    G+(t, x; y) = Tr[ rho S^-t a_x S^t a†_y ]
                = E_tau [ s_d(t) * s_v(t) * jw_y(tau) * jw_x(D(t))
                          * 1{ D(t) = V(t) + one extra bit at x } ]

where ``V(t) = S^t applied to tau`` (vacuum trajectory, accumulated sign ``s_v``) and
``D(t) = S^t applied to (tau + bit y)`` (defect trajectory, sign ``s_d``). Because S is
a signed permutation, each term is exactly 0 or +-1: **the fermionic amplitude is a
signed damage-spreading correlator.** The XOR cloud of the pair is the physical
dressing of the excitation; G+ is the amplitude that the dressing has collapsed back to
a single displaced particle.

The hole propagator ``G-(t, x; y) = Tr[rho S^-t a†_x S^t a_y]`` is the same computation
with the defect being a *removed* bit. Their sum at t = 0 obeys the anticommutation sum
rule ``G+(0,x;y) + G-(0,x;y) = delta_xy`` exactly, which the tests enforce.

What this buys over the density correlator: signs. Interference, quasiparticle weight
and (later, in 3D) spinor structure are visible only here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signed import SignedBlockCycle


@dataclass
class PropagatorResult:
    times: np.ndarray  # sub-step indices
    x: np.ndarray  # site displacements (0..L-1, ring)
    g_particle: np.ndarray  # (n_t, L, n_species) real amplitudes
    g_hole: np.ndarray
    survival: np.ndarray  # (n_t,) fraction of samples with single-bit damage (any x)
    ensemble: int

    def light_cone_weight(self, t_index: int) -> float:
        """Total |G+|^2 at a time slice -- the quasiparticle weight proxy."""
        return float((self.g_particle[t_index] ** 2).sum())


def _sample_vacuum(
    rng: np.random.Generator, n_modes: int, ensemble: int,
    densities: tuple[float, float] | None = None,
) -> np.ndarray:
    """Product vacuum; per-species densities allowed (mode m has species m % 2).

    A product measure at density q is exactly stationary only for rules conserving that
    species' number per block -- caller's responsibility (the v(q) rules qualify).
    """
    if densities is None:
        bits = rng.integers(0, 2, size=(ensemble, n_modes), dtype=np.int64)
    else:
        u = rng.random(size=(ensemble, n_modes))
        dens = np.array([densities[m % 2] for m in range(n_modes)])
        bits = (u < dens[None, :]).astype(np.int64)
    out = np.zeros(ensemble, dtype=np.int64)
    for m in range(n_modes):
        out |= bits[:, m] << m
    return out


def propagator(
    model: SignedBlockCycle,
    n_substeps: int,
    y_site: int = 0,
    y_species: int = 0,
    ensemble: int = 4096,
    seed: int = 0,
    densities: tuple[float, float] | None = None,
) -> PropagatorResult:
    """Monte Carlo estimate of G+ and G- over the uniform vacuum.

    Exactness structure: for given tau every contribution is exactly 0 or +-1; the only
    error is the vacuum sampling, so error bars scale 1/sqrt(ensemble) with bounded
    variance. At t = 0 the estimator is exact per sample.
    """
    L = model.n_sites
    M = model.n_modes
    rng = np.random.default_rng(seed)
    y_mode = 2 * y_site + y_species

    tau = _sample_vacuum(rng, M, ensemble, densities)
    ones = np.ones(ensemble, dtype=np.int64)

    # defect trajectories: a†_y for particles, a_y for holes (model's rank gauge)
    d_cfg_p, d_sgn_p = model.jw.create(tau, ones.copy(), y_mode)
    d_cfg_h, d_sgn_h = model.jw.annihilate(tau, ones.copy(), y_mode)
    v_cfg, v_sgn = tau.copy(), ones.copy()

    n_t = n_substeps + 1
    gp = np.zeros((n_t, L, 2))
    gh = np.zeros((n_t, L, 2))
    surv = np.zeros(n_t)

    for t in range(n_t):
        if t:
            v_cfg, v_sgn = model.substep(v_cfg, v_sgn, (t - 1) % 2)
            d_cfg_p, d_sgn_p = model.substep(d_cfg_p, d_sgn_p, (t - 1) % 2)
            d_cfg_h, d_sgn_h = model.substep(d_cfg_h, d_sgn_h, (t - 1) % 2)

        for defect_cfg, defect_sgn, out, extra_particle in (
            (d_cfg_p, d_sgn_p, gp, True),
            (d_cfg_h, d_sgn_h, gh, False),
        ):
            diff = defect_cfg ^ v_cfg
            single = (diff != 0) & ((diff & (diff - 1)) == 0)
            live = single & (defect_sgn != 0)
            if extra_particle:
                surv[t] = live.mean()  # P(single-bit damage), the coherence survival
            if not np.any(live):
                continue
            # locate the single differing bit per sample (vectorised log2)
            dd = diff.copy()
            shift = np.zeros(ensemble, dtype=np.int64)
            while np.any(dd > 1):
                big = dd > 1
                dd[big] >>= 1
                shift[big] += 1
            modes = shift
            # the JW string below (in RANK) the defect mode is identical in V and D
            # (they differ only AT the mode), so it is read off the vacuum trajectory
            # in the model's rank gauge
            for e in np.flatnonzero(live):
                m = int(modes[e])
                below = int(v_cfg[e]) & int(model.jw.below_mask[m])
                jw_e = 1 - 2 * (bin(below).count("1") & 1)
                amp = int(defect_sgn[e]) * int(v_sgn[e]) * jw_e
                out[t, m // 2, m % 2] += amp
    gp /= ensemble
    gh /= ensemble


    return PropagatorResult(
        times=np.arange(n_t),
        x=np.arange(L),
        g_particle=gp,
        g_hole=gh,
        survival=surv,
        ensemble=ensemble,
    )
