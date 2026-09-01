"""Adversarial tests against the finite-ray theorem (pca3d.analysis.dispersion).

The theorem says: any automaton satisfying R1-R3 that conserves particle number in the
one-particle sector has a one-particle dispersion consisting of exactly linear branches
``omega = (k . D_C + 2 pi m) / L_C``, with a finite set of k-independent group
velocities ``D_C / L_C``.

The job of this file is to *fail to break it*. We throw block rules, rotated rule
cycles, species-mixing shifts and structureless random bijections at it in one, two and
three dimensions, and check the prediction against an independent eigensolve every time.

The check that carries the weight is :func:`assert_theorem_holds`: the analytic branches
come from the cycle decomposition, the numeric ones from ``np.linalg.eigvals`` on a
densely built ``S(k)``. These share no code path. Agreement at many random ``k``, for a
formula that is exactly linear by construction, is what establishes that the true
dispersion is exactly linear.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.analysis import dispersion
from pca3d.core.automaton import NotOneParticleConserving
from pca3d.core.lattice import Lattice
from pca3d.models.generic import (
    BlockAutomaton,
    RuleCycle,
    SpeciesShift,
    particle_number_conserving_block_perm,
    random_block_perm,
)

CUBIC_6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])




def assert_theorem_holds(auto, n_k: int = 25, seed: int = 0) -> dispersion.OneParticleSpectrum:
    """Monomial structure, unitarity, and analytic == numeric at random momenta."""
    spec = dispersion.analyse(auto)
    rng = np.random.default_rng(seed)
    dim = auto.lattice.dim

    for _ in range(n_k):
        k = rng.uniform(-np.pi, np.pi, size=dim)
        S = dispersion.bloch_matrix(auto, k, spec.period)

        assert dispersion.is_monomial(S), f"S(k) not monomial at k={k}"
        assert np.allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=1e-12)

        analytic = spec.omega(k)[0]
        numeric = dispersion.numeric_eigenvalues(auto, k, spec.period)
        mismatch = dispersion.max_circular_mismatch(analytic, numeric)
        assert mismatch < 1e-9, (
            f"analytic and numeric dispersion disagree at k={k} by {mismatch:.3e}\n"
            f"  analytic {np.sort(analytic)}\n  numeric  {np.sort(numeric)}"
        )

    # every group velocity is k-independent by construction of the branch formula;
    # assert the set really is finite and small
    assert len(spec.unique_velocities()) <= len(spec.branches)
    return spec


# -- one dimension: the case that works ---------------------------------------------


def test_1d_two_rays_reproduce_the_light_cone_exactly():
    """In 1D the light cone IS two rays, so a finite ray set is not an approximation."""
    lat = Lattice(shape=(8,), n_species=2)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[1], [-1]]))
    spec = assert_theorem_holds(auto)
    assert sorted(float(v[0]) for v in spec.unique_velocities()) == [-1.0, 1.0]


# -- three dimensions: the case that does not ---------------------------------------


def test_3d_cubic_streaming_gives_six_rays_not_a_sphere():
    lat = Lattice(shape=(4, 4, 4), n_species=6)
    auto = SpeciesShift(lattice=lat, velocities=CUBIC_6)
    spec = assert_theorem_holds(auto)

    v = spec.unique_velocities()
    assert len(v) == 6
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    # all six rays lie on the coordinate axes -- nothing points along (1,1,1)
    assert np.all(np.count_nonzero(v, axis=1) == 1)


def test_species_mixing_slows_particles_down_it_does_not_curve_them():
    """sigma cycling x -> y -> z averages the velocity over the cycle.

    The mean of three unit axis vectors has norm 1/sqrt(3) < 1, so mixing species buys
    directions at the cost of speed. It never buys curvature.
    """
    lat = Lattice(shape=(6, 6, 6), n_species=3)
    vel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    auto = SpeciesShift(lattice=lat, velocities=vel, sigma=np.array([1, 2, 0]))
    spec = assert_theorem_holds(auto)

    assert len(spec.branches) == 1
    (branch,) = spec.branches
    assert branch.cycle_length == 3
    assert np.array_equal(branch.displacement, [1, 1, 1])
    assert np.allclose(branch.velocity, [1 / 3, 1 / 3, 1 / 3])
    assert branch.speed == pytest.approx(np.sqrt(3) / 3)


@pytest.mark.parametrize("seed", range(6))
def test_random_particle_conserving_block_rules_in_3d(seed):
    """Structureless but legal block bijections. R6 is violated; R1, R2, R3 are not."""
    rng = np.random.default_rng(seed)
    lat = Lattice(shape=(4, 4, 4), n_species=2)
    perm = particle_number_conserving_block_perm(2 * 2 * 2 * 2, rng)  # 2x2x2 block, 2 species
    auto = BlockAutomaton(
        lattice=lat, block_shape=(2, 2, 2), block_perm=perm, origin=(0, 0, 0)
    )
    assert auto.verify_block_bijection()
    assert_theorem_holds(auto, n_k=12, seed=seed)


@pytest.mark.parametrize("seed", range(4))
def test_shifted_block_cycles_in_3d(seed):
    """Wetterich's shifted-cell construction: alternate the partition origin.

    This is what makes a block automaton actually propagate rather than sit still, and
    it is the closest legal analogue of his 4D construction that stays within
    single-fermion carriers. It does not curve the dispersion either.
    """
    rng = np.random.default_rng(100 + seed)
    lat = Lattice(shape=(4, 4, 4), n_species=2)
    perm = particle_number_conserving_block_perm(16, rng)
    cycle = RuleCycle(
        steps=(
            BlockAutomaton(lattice=lat, block_shape=(2, 2, 2), block_perm=perm, origin=(0, 0, 0)),
            BlockAutomaton(lattice=lat, block_shape=(2, 2, 2), block_perm=perm, origin=(1, 1, 1)),
        )
    )
    assert_theorem_holds(cycle, n_k=12, seed=seed)


@pytest.mark.parametrize("seed", range(4))
def test_branch_b_rotated_rule_cycles_do_not_escape(seed):
    """A 3-step cycle of axis-rotated rules -- Wetterich's isotropy trick, Branch B.

    Products of unique-jump matrices are unique-jump, so S_cycle is itself covered by
    the theorem. Rotating between steps changes *which* rays you get; it does not make
    the branch set infinite.
    """
    rng = np.random.default_rng(200 + seed)
    lat = Lattice(shape=(6, 6, 6), n_species=6)
    base = CUBIC_6.copy()
    sigma = rng.permutation(6)

    def rotated(times: int) -> SpeciesShift:
        vel = np.roll(base, times, axis=1)  # cyclic relabelling of the axes
        return SpeciesShift(lattice=lat, velocities=vel, sigma=sigma)

    cycle = RuleCycle(steps=tuple(rotated(t) for t in range(3)))
    spec = assert_theorem_holds(cycle, n_k=12, seed=seed)

    # whatever the cycle does, the reachable velocities remain a finite ray set
    assert len(spec.unique_velocities()) <= lat.n_species


def test_many_species_refine_the_ray_set_but_never_close_it():
    """26 velocities (all cube neighbours) still leave large gaps on the sphere.

    The relevant number is the worst-case angle from an arbitrary direction to the
    nearest available ray. It shrinks with the number of species, but it is *scale
    invariant*: it is the same at k -> 0 as at the zone boundary, because the branches
    are exactly linear. A continuum limit cannot wash it out.
    """
    offsets = np.array([d for d in np.ndindex(3, 3, 3)]) - 1
    vel = offsets[np.any(offsets != 0, axis=1)]
    assert len(vel) == 26

    lat = Lattice(shape=(3, 3, 3), n_species=26)
    auto = SpeciesShift(lattice=lat, velocities=vel)
    spec = assert_theorem_holds(auto, n_k=8)

    rays = spec.unique_velocities()
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    rng = np.random.default_rng(7)
    probe = rng.normal(size=(4000, 3))
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    worst = np.arccos(np.clip((probe @ rays.T).max(axis=1), -1, 1)).max()
    assert worst > 0.3, "26 rays should still leave a gap of order 20 degrees"


# -- the boundary of the theorem's hypotheses ---------------------------------------


def test_non_conserving_rules_are_out_of_scope_and_say_so():
    """A rule that does not preserve the one-particle sector is legal but not covered.

    It must fail loudly rather than return a meaningless spectrum.
    """
    rng = np.random.default_rng(0)
    lat = Lattice(shape=(4, 4), n_species=2)
    auto = BlockAutomaton(
        lattice=lat,
        block_shape=(2, 2),
        block_perm=random_block_perm(8, rng),
        origin=(0, 0),
    )
    assert auto.verify_block_bijection()
    with pytest.raises(NotOneParticleConserving):
        auto.one_particle_map()
