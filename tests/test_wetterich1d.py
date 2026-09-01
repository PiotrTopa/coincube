"""M0: reproduce Wetterich's 1+1D automaton exactly, PRD 105, 074502.

Every property asserted here is one Wetterich states in the paper, with the section or
equation number given. If any of these fail, nothing downstream is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.analysis import dispersion
from pca3d.core.automaton import check_unique_jump
from pca3d.models.wetterich1d import L1, L2, R1, R2, Wetterich1D

VARIANTS = [
    pytest.param(True, id="extended"),
    pytest.param(False, id="base"),
]


@pytest.fixture(params=VARIANTS)
def model(request):
    # 4 sites x 4 species = 16 bits = 65536 configurations, fully enumerable
    return Wetterich1D(n_sites=4, extended=request.param)


@pytest.fixture
def configs(model):
    return model.lattice.all_configs_array()


# -- rule R2, the unique jump property --------------------------------------------


def test_step_is_unique_jump(model):
    """S_hat = S_int S_free is a unique jump matrix (sect. 2, eq. 62)."""
    report = check_unique_jump(model)
    assert report.is_unique_jump, str(report)
    assert report.n_configs == 1 << 16


def test_free_and_interaction_are_separately_unique_jump(model, configs):
    """Both factors are unique jumps, which is what guarantees the product is."""
    for name, fn in [("free", model.free_bits), ("int", model.interact_bits)]:
        img = fn(configs)
        packed = (img.reshape(len(configs), -1) << np.arange(16)).sum(axis=1)
        assert len(np.unique(packed)) == len(configs), f"{name} is not a bijection"


def test_interaction_is_an_involution(model, configs):
    """S_int^2 = 1, eq. 47."""
    assert np.array_equal(model.interact_bits(model.interact_bits(configs)), configs)


# -- sect. 4, the nine listed properties of the automaton ---------------------------


def test_right_and_left_movers_separately_conserved(model, configs):
    """Property 1. Implies conserved total particle number, eq. 117A."""
    out = model.step_bits(configs)
    assert np.array_equal(model.n_right(out), model.n_right(configs))
    assert np.array_equal(model.n_left(out), model.n_left(configs))


def test_doubly_occupied_lines_do_not_scatter(model, configs):
    """Property 3: a red *and* a green particle moving together never scatter."""
    mask = model.interaction_mask(configs)
    doubly_right = configs[..., R1] & configs[..., R2]
    doubly_left = configs[..., L1] & configs[..., L2]
    assert not np.any(mask & doubly_right)
    assert not np.any(mask & doubly_left)


def test_empty_lines_do_not_scatter(model, configs):
    """Property 3, the complementary case: empty lines are inert."""
    mask = model.interaction_mask(configs)
    empty_right = ~configs[..., R1] & ~configs[..., R2]
    empty_left = ~configs[..., L1] & ~configs[..., L2]
    assert not np.any(mask & empty_right)
    assert not np.any(mask & empty_left)


def test_colour_exchange_symmetry(model, configs):
    """Property 8: the dynamics commutes with red <-> green."""
    E = model.exchange_colours
    assert np.array_equal(model.step_bits(E(configs)), E(model.step_bits(configs)))


def test_particle_hole_symmetry(model, configs):
    """Property 9, and rule R5: the dynamics commutes with K, n -> 1 - n.

    This is the involution the complex structure is built on (sect. 6), so without it
    there is no complex Hilbert space and no quantum mechanics.
    """
    K = model.particle_hole
    assert np.array_equal(model.step_bits(K(configs)), K(model.step_bits(configs)))


def test_F_symmetry(model, configs):
    """Property 9: F exchanges a red particle with a green hole. F = K . E."""
    F = lambda n: model.particle_hole(model.exchange_colours(n))
    assert np.array_equal(model.step_bits(F(configs)), F(model.step_bits(configs)))


def test_parity(model, configs):
    """Property 6. P commutes with the interaction and inverts the propagation."""
    P = model.parity
    assert np.array_equal(model.interact_bits(P(configs)), P(model.interact_bits(configs)))
    # P S_free P = S_free^{-1}: reflecting space swaps the two directions of motion
    assert np.array_equal(P(model.free_bits(P(configs))), model.free_bits(configs))


def test_scattering_is_the_documented_transposition(model):
    """The local rule is exactly eq. 44 (and eq. 79 when extended)."""
    lat = model.lattice
    single = np.zeros((1, lat.n_sites, 4), dtype=bool)

    def local(state):
        single[:] = False
        single[0, 0, :] = state
        return tuple(model.interact_bits(single)[0, 0].astype(int))

    # eq. 44: R1 + L2 <-> R2 + L1
    assert local((1, 0, 0, 1)) == (0, 1, 1, 0)
    assert local((0, 1, 1, 0)) == (1, 0, 0, 1)

    if model.extended:
        # eq. 79/80: R1 + L1 <-> R2 + L2
        assert local((1, 0, 1, 0)) == (0, 1, 0, 1)
        assert local((0, 1, 0, 1)) == (1, 0, 1, 0)
    else:
        assert local((1, 0, 1, 0)) == (1, 0, 1, 0)
        assert local((0, 1, 0, 1)) == (0, 1, 0, 1)

    # a third particle present blocks the scattering (sect. 3, "If a third or fourth
    # particle is present, no scattering occurs")
    assert local((1, 1, 0, 1)) == (1, 1, 0, 1)
    assert local((1, 0, 1, 1)) == (1, 0, 1, 1)


# -- the one-particle sector and the finite-ray theorem -----------------------------


def test_one_particle_dispersion_is_exactly_massless_dirac(model):
    """A lone particle streams at c = 1, so omega = +-k with no lattice error at all."""
    spec = dispersion.analyse(model)
    assert np.array_equal(spec.period, [1])
    speeds = sorted(float(abs(v[0])) for v in spec.unique_velocities())
    assert speeds == [1.0, 1.0]
    vs = sorted(float(v[0]) for v in spec.unique_velocities())
    assert vs == [-1.0, 1.0]


def test_bloch_matrix_is_monomial(model):
    """The structural claim the finite-ray theorem rests on."""
    for k in np.linspace(-np.pi, np.pi, 11):
        S = dispersion.bloch_matrix(model, np.array([k]))
        assert dispersion.is_monomial(S)
        # monomial with unit-modulus entries => orthogonal/unitary, i.e. R2 survives
        assert np.allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=1e-12)


def test_analytic_branches_match_numeric_diagonalisation(model):
    """The cycle-structure prediction is checked against an independent eigensolve."""
    spec = dispersion.analyse(model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = rng.uniform(-np.pi, np.pi, size=1)
        analytic = spec.omega(k)[0]
        numeric = dispersion.numeric_eigenvalues(model, k)
        assert dispersion.max_circular_mismatch(analytic, numeric) < 1e-10, f"k={k}"


def test_dispersion_is_exactly_linear_in_k(model):
    """No curvature anywhere in the Brillouin zone: the branches are planes."""
    spec = dispersion.analyse(model)
    k = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    w = spec.omega(k)
    second_difference = w[2:] - 2 * w[1:-1] + w[:-2]
    assert np.allclose(second_difference, 0.0, atol=1e-12)


def test_free_automaton_has_the_same_one_particle_sector(model):
    """Interactions are invisible to a single particle -- there is nothing to scatter off.

    This is why the one-particle dispersion cannot be the place isotropy comes from.
    """
    free = Wetterich1D(n_sites=model.n_sites, extended=model.extended, interact=False)
    assert np.array_equal(model.one_particle_map(), free.one_particle_map())
