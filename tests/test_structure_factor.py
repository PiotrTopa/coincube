"""Validation of the dispersion instrument against cases with known exact answers.

The structure factor is the measuring device for every claim from M2 onward, so it is
calibrated here before it is trusted. Two failure modes it has already caught in
development, both now regression-tested:

  - **Fourier sign convention.** Using ``fftn`` in time as well as space puts a
    right-mover on the ridge ``omega = -k``, reporting every velocity with the wrong
    sign. Caught because a species defined to move in ``+x`` measured ``-1``.
  - **Aliasing.** ``omega`` is periodic, so a ridge with ``|v k| > pi`` wraps and a fit
    through the wrap reports a fictitious bend. Caught because it produced velocities
    faster than the automaton's own light cone.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.analysis.structure_factor import (
    evolve_record,
    measure_speed_along,
    structure_factor,
)
from pca3d.core.lattice import Lattice
from pca3d.models.generic import SpeciesShift
from pca3d.models.wetterich1d import Wetterich1D


def test_uniform_vacuum_is_exactly_stationary():
    """Bernoulli(1/2) per bit is the uniform measure, preserved by every bijection.

    So the mean occupation stays at 1/2 for all time with no relaxation, which is what
    lets the periodogram be interpreted as a stationary spectrum.
    """
    m = Wetterich1D(n_sites=32, extended=True)
    rec = evolve_record(m, n_steps=40, ensemble=200, rng=np.random.default_rng(0))
    per_step_mean = rec.reshape(rec.shape[0], -1).mean(axis=1)
    assert np.abs(per_step_mean).max() < 0.02  # spins average to 0, i.e. n -> 1/2


def test_free_streaming_measures_its_velocity_exactly():
    """A rigid shift must come back as exactly +1 and -1, with the right signs."""
    lat = Lattice(shape=(96,), n_species=2)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[1], [-1]]))

    right = structure_factor(auto, n_steps=191, ensemble=16, seed=0, species=0)
    left = structure_factor(auto, n_steps=191, ensemble=16, seed=0, species=1)

    vr = measure_speed_along(right, np.array([1.0]))["velocity"]
    vl = measure_speed_along(left, np.array([1.0]))["velocity"]
    assert vr == pytest.approx(+1.0, abs=1e-6)
    assert vl == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("velocity", [1, 2, 3])
def test_sign_and_magnitude_over_a_range_of_speeds(velocity):
    """Guards the Fourier convention: a mover at +v must measure +v, not -v."""
    lat = Lattice(shape=(128,), n_species=1)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[velocity]]))
    sf = structure_factor(auto, n_steps=255, ensemble=8, seed=0)
    # stay inside the unaliased window, |v k| < pi
    v = measure_speed_along(sf, np.array([1.0]), k_max_frac=0.8 / velocity)["velocity"]
    assert v == pytest.approx(float(velocity), abs=1e-6)


def test_wetterich_interaction_does_not_move_the_light_cone():
    """Colour exchange scatters the internal state but not the transport channel.

    All four species still travel at exactly c = 1; this is the calibration point for
    "the interaction fired and the ridge did not bend", which is the M2 result.
    """
    m = Wetterich1D(n_sites=128, extended=True, interact=True)
    expected = {0: +1.0, 1: +1.0, 2: -1.0, 3: -1.0}
    for sp, want in expected.items():
        sf = structure_factor(m, n_steps=255, ensemble=24, seed=1, species=sp)
        v = measure_speed_along(sf, np.array([1.0]))["velocity"]
        assert v == pytest.approx(want, abs=1e-3), f"species {sp}"


def test_ridge_interpolation_beats_the_bin_width():
    """Parabolic refinement must do better than snapping to the nearest frequency bin."""
    lat = Lattice(shape=(64,), n_species=1)
    # a velocity that is not commensurate with the frequency grid
    auto = SpeciesShift(lattice=lat, velocities=np.array([[1]]))
    sf = structure_factor(auto, n_steps=100, ensemble=8, seed=0)

    kmag = np.linalg.norm(sf.k, axis=1)
    idx = int(np.flatnonzero((sf.k[:, 0] > 0) & (kmag < 0.5))[0])
    coarse = sf.ridge(idx)
    fine = sf.ridge_interpolated(idx)
    exact = sf.k[idx, 0]  # omega = k for v = 1

    assert abs(fine - exact) <= abs(coarse - exact) + 1e-12
