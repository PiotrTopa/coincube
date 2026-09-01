"""Dispersion measured from the dynamic structure factor above a vacuum.

Why this module exists
----------------------
The finite-ray theorem (ADR 0001) pins the *one-particle* dispersion of any
particle-number-conserving automaton to a finite set of straight rays. Once particle
number is not conserved there is no invariant one-particle subspace, so that object
does not exist and the dispersion has to be read off something else: the propagation of
disturbances in a stationary state, i.e. the density-density correlator

    C(t, x) = < dn(t0 + t, x0 + x)  dn(t0, x0) >

whose Fourier transform ``S(omega, k)`` has ridges on the dispersion surface. This is an
*observable*, not a spectral abstraction, and it is what must show ``omega = |k|``
isotropically if the model is to be relativistic.

The vacuum
----------
We use i.i.d. Bernoulli(1/2) on every bit. This is not a modelling choice that needs
justifying by thermalisation: Bernoulli(1/2) per bit *is* the uniform measure on the
2**M configurations, and every bijection preserves the uniform measure. So for any rule
satisfying R2 it is **exactly** stationary, for every lattice size, with no relaxation
time and no equilibration assumption. It is Wetterich's "half-filled random vacuum"
(2203.14081).

Consequence worth stating plainly: ``<n> = 1/2`` exactly and the process is strictly
stationary, so ``S(omega, k)`` estimated by windowed periodogram is unbiased up to
spectral leakage, and the leakage is controlled by the window, not by the dynamics.

What is measured, and what is not
---------------------------------
This measures the propagation of *density* disturbances. For the free 1+1D automaton
``n_R`` streams rigidly, so the correlator is a delta on ``x = t`` and the ridge sits at
``omega = k`` -- the single-particle velocity is visible in the density channel. That is
the validation case in ``tests/test_structure_factor.py``.

It does **not** measure the fermionic two-point function ``<psi psibar>``, which needs
the Grassmann sign structure. Claims about spinor structure cannot be made from this
module; claims about light cones and isotropy can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.automaton import Automaton
from ..core.backend import gpu_available, gpu_status, to_device, to_numpy


def get_backend(use_gpu: bool):
    """Return ``(xp, to_numpy)``. Raises with the concrete reason if GPU is asked
    for but unavailable."""
    if not use_gpu:
        return np, (lambda a: np.asarray(a))
    if not gpu_available():  # pragma: no cover - host dependent
        raise RuntimeError(f"GPU backend requested but {gpu_status()}")
    import cupy as cp

    return cp, (lambda a: cp.asnumpy(a))


@dataclass
class StructureFactor:
    """``S(omega, k)`` on a regular grid, plus the axes."""

    omega: np.ndarray  # (n_omega,) in (-pi, pi]
    k: np.ndarray  # (n_k, dim)
    k_shape: tuple[int, ...]
    S: np.ndarray  # (n_omega, n_k)
    n_steps: int
    ensemble: int

    def _window(self, omega_window: tuple[float, float] | None) -> np.ndarray:
        if omega_window is None:
            return np.ones_like(self.omega, dtype=bool)
        lo, hi = omega_window
        return (self.omega >= lo) & (self.omega <= hi)

    def ridge(self, k_index: int, omega_window: tuple[float, float] | None = None) -> float:
        """The omega of maximum power at a given k -- the measured dispersion point.

        ``omega_window`` restricts the search. **Use it whenever the spectrum has more
        than one branch.** A parity-symmetric model has mirror branches at ``+v|k|`` and
        ``-v|k|`` carrying equal power, and an unrestricted argmax then picks between
        them on noise alone, producing a ridge that hops sign from one k to the next.
        Fitting through that reports a large fictitious curvature -- it did exactly that
        on the first run of the conditional-propagation measurement, alternating
        ``+1.50 k`` and ``-1.50 k`` while the true dispersion was a clean ``|omega| =
        1.50 |k|``. Restricting to ``omega > 0`` selects one branch and the artefact
        disappears.
        """
        mask = self._window(omega_window)
        idx = np.flatnonzero(mask)
        return float(self.omega[idx[int(np.argmax(self.S[mask, k_index]))]])

    def ridge_all(self, omega_window: tuple[float, float] | None = None) -> np.ndarray:
        mask = self._window(omega_window)
        idx = np.flatnonzero(mask)
        return self.omega[idx[np.argmax(self.S[mask], axis=0)]]

    def ridge_interpolated(
        self, k_index: int, omega_window: tuple[float, float] | None = None
    ) -> float:
        """Parabolic refinement of the ridge, to beat the ``2 pi / T`` bin width.

        Without this the measured group velocity is quantised in steps of the frequency
        bin, which at modest ``n_steps`` is comparable to the anisotropies we are trying
        to resolve. See :meth:`ridge` for why ``omega_window`` is usually required.
        """
        col = self.S[:, k_index]
        mask = self._window(omega_window)
        idx = np.flatnonzero(mask)
        j = int(idx[int(np.argmax(col[mask]))])
        if j == 0 or j == len(col) - 1:
            return float(self.omega[j])
        y0, y1, y2 = col[j - 1], col[j], col[j + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
        dw = self.omega[1] - self.omega[0]
        return float(self.omega[j] + delta * dw)

    def n_branches_at(self, k_index: int, rel_height: float = 0.5) -> int:
        """How many distinct peaks sit at this k, above ``rel_height`` of the maximum.

        Diagnostic for whether a single-branch extraction is even meaningful.
        """
        col = self.S[:, k_index]
        thr = rel_height * col.max()
        above = col >= thr
        return int(np.sum(above & ~np.roll(above, 1)))


def evolve_record(
    automaton: Automaton,
    n_steps: int,
    ensemble: int,
    rng: np.random.Generator,
    use_gpu: bool = False,
    densities: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Evolve ``ensemble`` product-measure configurations, recording every step.

    ``densities`` gives the Bernoulli parameter per species (default 1/2 for all).
    A product measure with per-species densities is EXACTLY stationary iff the rule
    conserves each species' particle number per block; callers using ``densities`` are
    responsible for that legality (the v(q) experiment enumerates such rules
    explicitly). Returns Ising spins ``s = 2n - 1``.
    """
    lat = automaton.lattice
    # sample on the CPU so results are bitwise identical across backends for a seed
    if densities is None:
        state_np = rng.integers(0, 2, size=(ensemble, lat.n_sites, lat.n_species)).astype(bool)
    else:
        if len(densities) != lat.n_species:
            raise ValueError(f"need {lat.n_species} densities, got {len(densities)}")
        u = rng.random(size=(ensemble, lat.n_sites, lat.n_species))
        state_np = u < np.asarray(densities)[None, None, :]
    state = to_device(state_np, use_gpu)
    xp, _ = get_backend(use_gpu)

    out = xp.empty((n_steps + 1, ensemble, lat.n_sites, lat.n_species), dtype=xp.int8)
    out[0] = xp.where(state, 1, -1)
    for t in range(n_steps):
        state = automaton.step_bits(state)
        out[t + 1] = xp.where(state, 1, -1)
    return out


def structure_factor(
    automaton: Automaton,
    n_steps: int = 256,
    ensemble: int = 32,
    seed: int = 0,
    species: int | None = None,
    use_gpu: bool = False,
    window: bool = True,
) -> StructureFactor:
    """Windowed space-time periodogram of the density fluctuations.

    ``species=None`` sums the power over all species; an integer selects one channel,
    which is what you want when different species carry different velocities and you do
    not wish to superimpose their ridges.
    """
    lat = automaton.lattice
    rng = np.random.default_rng(seed)
    xp, to_numpy = get_backend(use_gpu)

    rec = evolve_record(automaton, n_steps, ensemble, rng, use_gpu=use_gpu)
    s = xp.asarray(rec, dtype=xp.float32)

    if species is not None:
        s = s[..., species : species + 1]

    # (T, E, *shape, n_species)
    s = s.reshape(n_steps + 1, ensemble, *lat.shape, s.shape[-1])
    s = s - s.mean(axis=0, keepdims=True)

    if window:
        w = xp.asarray(np.hanning(n_steps + 1), dtype=xp.float32)
        s = s * w.reshape(-1, *([1] * (s.ndim - 1)))

    # Physics convention: a mode is exp(i(k.x - omega t)), so the transform must carry
    # exp(-i k.x) in space but exp(+i omega t) in time. Using fftn for both would put a
    # right-mover f(x - t) on the ridge omega = -k, i.e. report every velocity with the
    # wrong sign. ifft supplies the conjugate time kernel; its 1/T normalisation is
    # irrelevant to the ridge locations.
    space_axes = tuple(range(2, 2 + lat.dim))
    F = xp.fft.fftn(s, axes=space_axes)
    F = xp.fft.ifft(F, axis=0)
    P = (xp.abs(F) ** 2).mean(axis=1).sum(axis=-1)  # average ensemble, sum species

    P = xp.fft.fftshift(P, axes=tuple(range(P.ndim)))
    P = to_numpy(P)

    omega = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(n_steps + 1))
    kaxes = [np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(L)) for L in lat.shape]
    kgrid = np.stack(np.meshgrid(*kaxes, indexing="ij"), axis=-1).reshape(-1, lat.dim)

    return StructureFactor(
        omega=omega,
        k=kgrid,
        k_shape=tuple(lat.shape),
        S=P.reshape(len(omega), -1),
        n_steps=n_steps,
        ensemble=ensemble,
    )


def measure_speed_along(
    sf: StructureFactor,
    direction: np.ndarray,
    k_max_frac: float = 0.25,
    omega_window: tuple[float, float] | None = None,
) -> dict:
    """Fit ``omega = v |k|`` along one direction, using the small-|k| ridge points.

    Returns the slope, its standard error, and the points used. Restricting to small
    ``|k|`` matters: a lattice dispersion that is linear near the origin generally bends
    at the zone boundary, and fitting through the bend reports a velocity that is an
    artefact of the fit range rather than a property of the model.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    kmag = np.linalg.norm(sf.k, axis=1)
    kmax = k_max_frac * np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (sf.k @ direction) / np.where(kmag > 0, kmag, np.inf)
    sel = np.flatnonzero((kmag > 1e-9) & (kmag <= kmax) & (cos > 1 - 1e-9))

    if len(sel) < 2:
        return {"velocity": np.nan, "stderr": np.nan, "n_points": len(sel), "k": [], "omega": []}

    ks = kmag[sel]
    ws = np.array([sf.ridge_interpolated(int(i), omega_window) for i in sel])

    # through the origin: omega = v |k|
    v = float((ks @ ws) / (ks @ ks))
    resid = ws - v * ks
    dof = max(len(ks) - 1, 1)
    stderr = float(np.sqrt((resid @ resid) / dof / (ks @ ks)))
    return {
        "velocity": v,
        "stderr": stderr,
        "n_points": int(len(ks)),
        "k": ks.tolist(),
        "omega": ws.tolist(),
    }
