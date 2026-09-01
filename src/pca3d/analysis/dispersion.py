"""One-particle dispersion, and the finite-ray theorem.

The central structural fact about this whole class of models is proved and checked here.

**Claim (finite-ray theorem).** Let the automaton satisfy R1 (locality), R2 (unique
jump) and R3 (homogeneity), and let it conserve particle number in the one-particle
sector. Then in the Bloch basis the one-particle step operator ``S(k)`` is a *monomial*
matrix: an internal permutation ``pi`` of the ``(offset, species)`` basis of the unit
cell, dressed with phases ``exp(-i k . d)`` where ``d`` is the displacement of that
basis state. Consequently, for each cycle ``C`` of ``pi``, of length ``L_C`` and total
displacement ``D_C = sum of d around the cycle``, the eigenvalues obey

    lambda^{L_C} = exp(-i k . D_C)   =>   omega_m(k) = ( k . D_C + 2 pi m ) / L_C

with ``m = 0 ... L_C - 1``. Every branch is **exactly linear in k**, with group velocity

    v_C = D_C / L_C

*independent of k*. The set of achievable group velocities is therefore a **finite set
of fixed rays**, one per cycle, not a sphere.

**Why 1+1D is not a counterexample.** In one space dimension the light cone consists of
exactly two rays, ``v = +1`` and ``v = -1``. A finite ray set reproduces it exactly, and
Wetterich's automaton does: right-movers form cycles with ``D/L = +1``, left-movers
``-1``, giving ``omega = +-k``, the massless Dirac dispersion, with no error at all.

**Why three dimensions is different.** A relativistic light cone in 3D requires a
2-sphere of group velocities. No finite ray set is a 2-sphere. Adding species refines
the ray set but never closes it, and the deviation is scale-invariant -- it does not
shrink as ``k -> 0``, because the branches are exactly linear at every ``k``. Products
of unique-jump matrices are unique-jump, so a Branch B rotated cycle does not escape
this either: the theorem applies to ``S_cycle`` verbatim.

The consequence for the project is not that the programme is dead; it is that isotropy
cannot come from the one-particle dispersion and must instead be emergent from the
interacting, coarse-grained theory -- which is exactly the mechanism by which FCHC
lattice gases recover isotropic hydrodynamics from 24 discrete velocities. That is what
turns "pick a lattice" into a well-posed optimisation, and it is handled in
``pca3d.analysis.isotropy``.

Everything above is *checked*, not assumed: :func:`analyse` derives the branches from
the cycle structure and :func:`numeric_eigenvalues` independently diagonalises ``S(k)``,
and the test suite asserts they agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.automaton import Automaton
from ..core.lattice import Lattice


@dataclass(frozen=True)
class Branch:
    """One dispersion branch, ``omega(k) = (k . D + 2 pi m) / L``."""

    cycle_length: int
    displacement: np.ndarray  # D_C, total displacement around the cycle
    members: tuple[tuple[int, int], ...]  # the (offset_index, species) in the cycle

    @property
    def velocity(self) -> np.ndarray:
        """Group velocity ``D_C / L_C``, constant in k."""
        return self.displacement / self.cycle_length

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def omega(self, k: np.ndarray, m: int = 0) -> np.ndarray:
        k = np.atleast_2d(k)
        return (k @ self.displacement + 2.0 * np.pi * m) / self.cycle_length


@dataclass(frozen=True)
class OneParticleSpectrum:
    lattice: Lattice
    period: np.ndarray  # translation period of the rule, per axis
    branches: tuple[Branch, ...]
    n_substeps: int

    @property
    def velocities(self) -> np.ndarray:
        """``(n_branches, dim)`` array of the distinct achievable group velocities."""
        return np.array([b.velocity for b in self.branches])

    def unique_velocities(self, tol: float = 1e-12) -> np.ndarray:
        v = self.velocities
        if v.size == 0:
            return v
        out: list[np.ndarray] = []
        for row in v:
            if not any(np.allclose(row, o, atol=tol) for o in out):
                out.append(row)
        return np.array(out)

    def omega(self, k: np.ndarray) -> np.ndarray:
        """All branches at momentum ``k``: shape ``(n_k, total_bands)``."""
        k = np.atleast_2d(k)
        cols = [b.omega(k, m) for b in self.branches for m in range(b.cycle_length)]
        return np.stack(cols, axis=-1)

    def __str__(self) -> str:
        uv = self.unique_velocities()
        speeds = sorted({round(float(np.linalg.norm(v)), 10) for v in uv})
        return (
            f"one-particle spectrum: {len(self.branches)} cycle(s), "
            f"{len(uv)} distinct group velocit{'y' if len(uv) == 1 else 'ies'}, "
            f"speeds {speeds}; every branch exactly linear in k"
        )


def detect_period(automaton: Automaton, max_period: int = 8) -> np.ndarray:
    """Smallest translation period under which the one-particle map is covariant.

    A fully homogeneous rule has period 1 in every direction; a Margolus block rule has
    period equal to the block size. Raises if no period up to ``max_period`` works,
    which would mean the rule is not homogeneous at all (violating R3).
    """
    lat = automaton.lattice
    opm = automaton.one_particle_map()
    period = np.ones(lat.dim, dtype=np.int64)

    for axis in range(lat.dim):
        for p in range(1, min(max_period, lat.shape[axis]) + 1):
            if lat.shape[axis] % p:
                continue
            if _covariant_under(lat, opm, axis, p):
                period[axis] = p
                break
        else:
            raise ValueError(
                f"one-particle map is not translation covariant along axis {axis} "
                f"for any period up to {max_period}; rule violates R3 (homogeneity)"
            )
    return period


def _covariant_under(lat: Lattice, opm: np.ndarray, axis: int, p: int) -> bool:
    """Does ``F`` commute with translation by ``p`` along ``axis``?"""
    shift = np.zeros(lat.dim, dtype=np.int64)
    shift[axis] = p
    for i, (x, a) in enumerate(lat.one_particle_states):
        # translate, then step
        x_t = lat.site_index(lat.site_coords(x) + shift)
        j_translate_first = int(opm[x_t * lat.n_species + a])
        # step, then translate
        x_img, a_img = lat.one_particle_states[int(opm[i])]
        x_img_t = lat.site_index(lat.site_coords(x_img) + shift)
        j_step_first = x_img_t * lat.n_species + a_img
        if j_translate_first != j_step_first:
            return False
    return True


def analyse(automaton: Automaton, period: np.ndarray | None = None) -> OneParticleSpectrum:
    """Derive the exact dispersion branches from the cycle structure of ``pi``."""
    lat = automaton.lattice
    opm = automaton.one_particle_map()
    if period is None:
        period = detect_period(automaton)
    period = np.asarray(period, dtype=np.int64)

    # basis of the unit cell: (offset, species)
    offsets = np.array(
        np.meshgrid(*[np.arange(p) for p in period], indexing="ij")
    ).reshape(lat.dim, -1).T
    basis = [(int(o), s) for o in range(len(offsets)) for s in range(lat.n_species)]
    index_of = {b: i for i, b in enumerate(basis)}

    # internal permutation pi and the displacement attached to each basis state
    pi = np.empty(len(basis), dtype=np.int64)
    disp = np.zeros((len(basis), lat.dim), dtype=np.int64)
    for bi, (o, s) in enumerate(basis):
        site = lat.site_index(offsets[o])
        img_site, img_species = lat.one_particle_states[int(opm[site * lat.n_species + s])]
        d = lat.displacement(site, img_site)
        img_offset = tuple(int(v) for v in (lat.site_coords(img_site) % period))
        oi = int(np.flatnonzero((offsets == np.array(img_offset)).all(axis=1))[0])
        pi[bi] = index_of[(oi, img_species)]
        disp[bi] = d

    branches: list[Branch] = []
    seen = np.zeros(len(basis), dtype=bool)
    for start in range(len(basis)):
        if seen[start]:
            continue
        members, total, j = [], np.zeros(lat.dim, dtype=np.int64), start
        while not seen[j]:
            seen[j] = True
            members.append(basis[j])
            total = total + disp[j]
            j = int(pi[j])
        branches.append(
            Branch(cycle_length=len(members), displacement=total, members=tuple(members))
        )

    return OneParticleSpectrum(
        lattice=lat,
        period=period,
        branches=tuple(branches),
        n_substeps=getattr(automaton, "n_substeps", 1),
    )


def bloch_matrix(
    automaton: Automaton, k: np.ndarray, period: np.ndarray | None = None
) -> np.ndarray:
    """``S(k)`` as a dense complex matrix, built directly from the one-particle map.

    Independent of :func:`analyse` -- this is the object used to check the theorem
    rather than to state it.
    """
    lat = automaton.lattice
    opm = automaton.one_particle_map()
    if period is None:
        period = detect_period(automaton)
    period = np.asarray(period, dtype=np.int64)
    k = np.asarray(k, dtype=float)

    offsets = np.array(
        np.meshgrid(*[np.arange(p) for p in period], indexing="ij")
    ).reshape(lat.dim, -1).T
    basis = [(int(o), s) for o in range(len(offsets)) for s in range(lat.n_species)]
    index_of = {b: i for i, b in enumerate(basis)}

    S = np.zeros((len(basis), len(basis)), dtype=complex)
    for bi, (o, s) in enumerate(basis):
        site = lat.site_index(offsets[o])
        img_site, img_species = lat.one_particle_states[int(opm[site * lat.n_species + s])]
        d = lat.displacement(site, img_site)
        img_offset = tuple(int(v) for v in (lat.site_coords(img_site) % period))
        oi = int(np.flatnonzero((offsets == np.array(img_offset)).all(axis=1))[0])
        S[index_of[(oi, img_species)], bi] = np.exp(-1j * float(k @ d))
    return S


def numeric_eigenvalues(
    automaton: Automaton, k: np.ndarray, period: np.ndarray | None = None
) -> np.ndarray:
    """``omega`` at momentum ``k`` by explicit diagonalisation, sorted."""
    S = bloch_matrix(automaton, k, period)
    lam = np.linalg.eigvals(S)
    return np.sort(-np.angle(lam))


def is_monomial(S: np.ndarray, tol: float = 1e-12) -> bool:
    """Exactly one non-zero entry in every row and every column."""
    nz = np.abs(S) > tol
    return bool(np.all(nz.sum(axis=0) == 1) and np.all(nz.sum(axis=1) == 1))


def max_circular_mismatch(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Largest distance between two multisets of frequencies, compared on the circle.

    Frequencies are only defined modulo ``2 pi``, so comparing sorted arrays of angles
    is wrong: a branch sitting on the cut appears at ``-pi`` in one calculation and
    ``+pi - eps`` in another, and a naive comparison reports a discrepancy of ``2 pi``
    for two values that are the same point.

    We therefore map to ``exp(-i omega)`` and take the optimal pairing under chordal
    distance, which has no branch cut. Returns the largest paired distance; identical
    spectra give 0.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.exp(-1j * np.asarray(w_a).ravel())
    b = np.exp(-1j * np.asarray(w_b).ravel())
    if a.size != b.size:
        return float("inf")
    cost = np.abs(a[:, None] - b[None, :])
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].max())
