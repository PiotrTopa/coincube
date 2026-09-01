"""Real-space transport: how fast, and in what manner, a disturbance spreads.

Why not just use the structure factor
-------------------------------------
``pca3d.analysis.structure_factor`` measures the same physics in Fourier space, and it
is the right tool when there is one clean branch. But reading a velocity off a ridge has
three separate ways to go wrong, all of which bit during development:

  - the Fourier sign convention (reports every velocity backwards);
  - aliasing when ``|v k| > pi`` (manufactures curvature, reports superluminal speeds);
  - branch selection when parity gives mirror ridges at ``+-v|k|`` of equal power
    (``argmax`` hops between them and manufactures curvature again).

None of those exist here. This module computes the space-time correlation

    C(t, x) = < s(t, x0 + x) s(0, x0) >     averaged over x0 and over the ensemble

with ``s = 2n - 1``, and reads the transport off its width directly. A rigid mover at
velocity ``v`` gives ``C(t, x) = delta(x - v t)`` and the width is exactly ``v t``.

Ballistic or not
----------------
Fitting ``rms(t) ~ t**alpha`` separates the two outcomes that matter:

    alpha ~ 1     ballistic -- a light cone, a well-defined propagation speed
    alpha ~ 1/2   diffusive -- no light cone, no quasiparticle, speed is meaningless

This distinction is what "the ridge dissolved" in ADR 0002 was really detecting, stated
as a physical property of the rule rather than as a property of a peak-finder.

Boundaries
----------
The lattice is periodic, so the measurement is only valid until the fastest front wraps.
:func:`spread_velocity` refuses to fit beyond ``L / (2 v_max)`` rather than silently
folding the front back on itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.automaton import Automaton
from .structure_factor import evolve_record


@dataclass
class SpreadResult:
    """Outcome of a transport measurement."""

    velocity: float  # sites per call of step_bits, from a linear fit of rms(t)
    velocity_stderr: float
    exponent: float  # alpha in rms ~ t**alpha
    exponent_stderr: float
    rms: np.ndarray  # front position vs t
    times: np.ndarray
    wrapped: bool  # did the front reach the periodic boundary within the fit window

    @property
    def is_ballistic(self) -> bool:
        """Ballistic within three standard errors of alpha = 1."""
        return abs(self.exponent - 1.0) < max(3.0 * self.exponent_stderr, 0.05)

    @property
    def is_diffusive(self) -> bool:
        return abs(self.exponent - 0.5) < max(3.0 * self.exponent_stderr, 0.05)

    def __str__(self) -> str:
        kind = "ballistic" if self.is_ballistic else ("diffusive" if self.is_diffusive else "neither")
        return (
            f"v = {self.velocity:.4f} +- {self.velocity_stderr:.4f}, "
            f"alpha = {self.exponent:.3f} +- {self.exponent_stderr:.3f} ({kind})"
            + (" [WRAPPED]" if self.wrapped else "")
        )


def correlator_from_series(series: np.ndarray) -> np.ndarray:
    """``C(t, x)`` from an already-projected field series of shape ``(T+1, E, L)``.

    Same FFT cross-correlation core as :func:`space_time_correlation`, factored out so
    that 3D runs can be measured on 1D projections with the calibrated estimators.
    """
    s = np.asarray(series, dtype=np.float32)
    s = s - s.mean(axis=2, keepdims=True)
    L = s.shape[2]
    F = np.fft.fft(s, axis=2)
    C = np.fft.ifft(F * np.conj(F[0:1]), axis=2).real / L
    return C.mean(axis=1)


def space_time_correlation(
    automaton: Automaton,
    n_steps: int,
    ensemble: int,
    seed: int = 0,
    species: int | None = None,
    use_gpu: bool = False,
    densities: tuple[float, ...] | None = None,
) -> np.ndarray:
    """``C(t, x)`` for a 1D lattice, with ``x`` in FFT order (index 0 is displacement 0).

    Computed as a cross-correlation via FFT, which is exact and avoids an O(L^2) loop.
    """
    lat = automaton.lattice
    if lat.dim != 1:
        raise NotImplementedError("space_time_correlation currently handles dim == 1")

    rec = evolve_record(automaton, n_steps, ensemble, np.random.default_rng(seed), use_gpu=use_gpu, densities=densities)
    from ..core.backend import to_numpy as _to_np
    rec = _to_np(rec)
    s = rec.astype(np.float32)
    if species is not None:
        s = s[..., species : species + 1]
    s = s.sum(axis=-1)  # (T+1, E, L)
    s = s - s.mean(axis=2, keepdims=True)

    F = np.fft.fft(s, axis=2)
    F0 = F[0:1]
    C = np.fft.ifft(F * np.conj(F0), axis=2).real / lat.n_sites
    return C.mean(axis=1)  # average over the ensemble -> (T+1, L)


def _centroid_about(absc: np.ndarray, x: np.ndarray, j: int, thresh: float) -> float:
    """Sub-site position of the correlation packet containing index ``j``.

    Walks outward from the peak while the profile stays above ``thresh``, then takes the
    intensity-weighted mean of ``|x|`` over that contiguous run. For a rigid mover the
    run is the single delta and this returns exactly ``v t``; for a broadened packet it
    returns the packet's mean position with resolution far finer than one site.

    Sub-site resolution is the whole point: the integer-valued peak index cannot
    distinguish a velocity of 1.508 from 3/2 = 1.5, which is precisely the question M2c
    has to answer.
    """
    n = len(absc)
    lo = j
    while lo - 1 >= 0 and absc[lo - 1] >= thresh:
        lo -= 1
    hi = j
    while hi + 1 < n and absc[hi + 1] >= thresh:
        hi += 1
    w = absc[lo : hi + 1]
    tot = w.sum()
    if tot <= 0:
        return abs(x[j])
    return float((w * np.abs(x[lo : hi + 1])).sum() / tot)


def spread_velocity(
    C: np.ndarray,
    v_max: float = 2.0,
    t_min: int = 4,
    rel_threshold: float = 0.1,
    estimator: str = "peak",
) -> SpreadResult:
    """Fit the front position of ``C(t, .)`` against time.

    The front is the largest displacement at which the correlation still stands above
    the per-slice noise floor. For a rigid mover it equals ``v t`` exactly; for
    diffusive spreading it grows like ``sqrt(t)``.
    """
    n_t, L = C.shape
    x = np.fft.fftfreq(L, d=1.0 / L)  # signed displacements, minimal image

    # Two estimators were tried and rejected before this one, both documented because
    # both fail in ways that look like physics:
    #
    #   x**2-weighted rms  -- C is zero in expectation away from the front but carries
    #     finite-ensemble noise everywhere, and x**2 weights exactly the far tails where
    #     there is only noise. On free streaming (exact answer v = 1, alpha = 1) it
    #     returned v = 1.32, alpha = 0.27: it was measuring the lattice size.
    #
    #   absolute noise floor -- thresholding at median + 6 MAD of the outer lattice lets
    #     a single tail outlier at large |x| define the "front". On a rigid unit shift it
    #     returned v = 5.59, five times the light cone.
    #
    # What works is a threshold *relative to the peak of the same time slice*. It is
    # scale free, so it cannot be fooled by the overall decay of C, and one stray point
    # cannot pass it. For a rigid mover the correlation is a single delta of height ~1,
    # so only the delta survives and the peak position is exactly v t.
    absC = np.abs(C)
    peak = absC.max(axis=1)
    thresh = np.maximum(rel_threshold * peak, 1e-12)

    front = np.empty(n_t)
    peak_pos = np.empty(n_t)
    centroid = np.empty(n_t)
    for t in range(n_t):
        sig = absC[t] >= thresh[t]
        front[t] = np.max(np.abs(x[sig])) if np.any(sig) else 0.0
        j = int(np.argmax(absC[t]))
        peak_pos[t] = abs(x[j])
        centroid[t] = _centroid_about(absC[t], x, j, thresh[t])

    rms = {"peak": peak_pos, "front": front, "centroid": centroid}[estimator]

    t_wrap = int(L / (2.0 * max(v_max, 1e-9)))
    t_hi = min(n_t - 1, t_wrap)
    wrapped = t_hi < n_t - 1

    times = np.arange(n_t)
    sel = np.flatnonzero((times >= t_min) & (times <= t_hi) & np.isfinite(rms))
    if len(sel) < 4:
        return SpreadResult(np.nan, np.nan, np.nan, np.nan, rms, times, wrapped)

    t_fit, r_fit = times[sel].astype(float), rms[sel]

    # velocity: rms = v t through the origin
    v = float((t_fit @ r_fit) / (t_fit @ t_fit))
    resid = r_fit - v * t_fit
    dof = max(len(t_fit) - 1, 1)
    v_err = float(np.sqrt((resid @ resid) / dof / (t_fit @ t_fit)))

    # exponent: log rms = alpha log t + c
    ok = r_fit > 0
    if ok.sum() >= 4:
        A = np.vstack([np.log(t_fit[ok]), np.ones(ok.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(r_fit[ok]), rcond=None)
        alpha = float(coef[0])
        dof_a = max(ok.sum() - 2, 1)
        s2 = float(res[0] / dof_a) if len(res) else 0.0
        cov = s2 * np.linalg.inv(A.T @ A)
        alpha_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    else:
        alpha, alpha_err = np.nan, np.nan

    return SpreadResult(v, v_err, alpha, alpha_err, rms, times, wrapped)


def measure_transport(
    automaton: Automaton,
    n_steps: int = 48,
    ensemble: int = 32,
    seed: int = 0,
    species: int | None = None,
    v_max: float = 2.0,
    estimator: str = "peak",
    use_gpu: bool = False,
    densities: tuple[float, ...] | None = None,
) -> SpreadResult:
    C = space_time_correlation(automaton, n_steps, ensemble, seed, species, use_gpu=use_gpu, densities=densities)
    return spread_velocity(C, v_max=v_max, estimator=estimator)
