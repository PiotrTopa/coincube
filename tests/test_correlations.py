"""Calibration of the real-space transport instrument.

Every control here has an exact analytic answer, and the instrument is required to
return it to machine precision. This file exists because two earlier estimators passed
casual inspection while being badly wrong, and both failure modes are regression-tested
below so they cannot come back silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.analysis.correlations import measure_transport, space_time_correlation, spread_velocity
from pca3d.core.lattice import Lattice
from pca3d.models.generic import SpeciesShift


@pytest.mark.parametrize("v", [1, 2, 3, 5])
def test_rigid_shift_returns_its_exact_velocity(v):
    """A rigid shift gives C(t,x) = delta(x - v t); anything but v exactly is a bug.

    The x**2-weighted estimator failed here by returning the lattice size instead of the
    velocity; the absolute-noise-floor estimator failed by letting a single tail outlier
    define the front, reporting v = 5.6 for v = 1.
    """
    lat = Lattice(shape=(512,), n_species=1)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[v]]))
    r = measure_transport(auto, n_steps=50, ensemble=16, v_max=float(v))
    assert r.velocity == pytest.approx(float(v), abs=1e-9)
    assert r.exponent == pytest.approx(1.0, abs=1e-9)
    assert r.is_ballistic


def test_counterpropagating_species_still_give_unit_speed():
    lat = Lattice(shape=(512,), n_species=2)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[1], [-1]]))
    r = measure_transport(auto, n_steps=60, ensemble=32, v_max=1.0)
    assert r.velocity == pytest.approx(1.0, abs=1e-9)
    assert r.is_ballistic


def test_static_rule_reports_zero_and_is_not_ballistic():
    lat = Lattice(shape=(256,), n_species=1)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[0]]))
    r = measure_transport(auto, n_steps=40, ensemble=16, v_max=1.0)
    assert r.velocity == pytest.approx(0.0, abs=1e-12)
    assert not r.is_ballistic


def test_species_selection_separates_channels():
    """A rule where one species moves and another does not must report them separately.

    This caught a bug in a *control*, not the instrument: a 'swap' rule that left the
    environment static produced a dominant stationary correlation at x = 0, and summing
    the channels reported v = 0 for the whole model.
    """
    lat = Lattice(shape=(256,), n_species=2)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[2], [0]]))
    moving = measure_transport(auto, n_steps=40, ensemble=16, species=0, v_max=2.0)
    static = measure_transport(auto, n_steps=40, ensemble=16, species=1, v_max=2.0)
    assert moving.velocity == pytest.approx(2.0, abs=1e-9)
    assert static.velocity == pytest.approx(0.0, abs=1e-12)


def test_correlation_at_t0_is_a_delta_at_the_origin():
    lat = Lattice(shape=(128,), n_species=1)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[1]]))
    C = space_time_correlation(auto, n_steps=3, ensemble=64, seed=0)
    assert np.argmax(np.abs(C[0])) == 0
    assert C[0, 0] > 10 * np.abs(C[0, 1:]).max()


def test_diffusive_input_is_not_called_ballistic():
    """A synthetic sqrt(t) spreading profile must be classified diffusive, not ballistic."""
    L, T = 512, 60
    x = np.fft.fftfreq(L, d=1.0 / L)
    C = np.zeros((T + 1, L))
    C[0, 0] = 1.0
    for t in range(1, T + 1):
        w = np.sqrt(t)
        C[t] = np.exp(-0.5 * (x / w) ** 2)
    # use_peak=False: the *front* estimator. The peak estimator is deliberately blind
    # here -- a symmetric spreading profile has its maximum at x = 0 for all t, so it
    # reports v = 0, which is the correct statement that there is no ballistic mode but
    # says nothing about the spreading exponent. The two estimators answer different
    # questions and the diffusive/ballistic classification belongs to the front one.
    r = spread_velocity(C, v_max=1.0, estimator="front")
    assert r.exponent == pytest.approx(0.5, abs=0.05), str(r)
    assert r.is_diffusive and not r.is_ballistic


def test_peak_estimator_reports_no_ballistic_mode_for_symmetric_spreading():
    L, T = 512, 60
    x = np.fft.fftfreq(L, d=1.0 / L)
    C = np.zeros((T + 1, L))
    C[0, 0] = 1.0
    for t in range(1, T + 1):
        C[t] = np.exp(-0.5 * (x / np.sqrt(t)) ** 2)
    r = spread_velocity(C, v_max=1.0, estimator="peak")
    assert r.velocity == pytest.approx(0.0, abs=1e-12)
    assert not r.is_ballistic


@pytest.mark.parametrize(
    "v_true", [1.0, 4 / 3, 1.37, 1.5, 1.508, np.sqrt(2)]
)
def test_centroid_estimator_resolves_non_integer_velocities(v_true):
    """Sub-site resolution, which is what M2c needs and the peak estimator cannot give.

    The peak estimator returns an integer site index, so it cannot distinguish 1.508
    from 3/2 -- and in ADR 0003 that systematic made Wetterich's rule look 7 sigma away
    from 3/2 when the unbiased value is 1.4984, i.e. 1.7 sigma *from* it. Anything used
    to answer "rational or not" must first pass this test.
    """
    L, T = 2048, 80
    x = np.fft.fftfreq(L, d=1.0 / L)
    C = np.zeros((T + 1, L))
    C[0, 0] = 1.0
    for t in range(1, T + 1):
        C[t] = np.exp(-0.5 * ((np.abs(x) - v_true * t) / 2.0) ** 2)

    r = spread_velocity(C, v_max=2.0, estimator="centroid")
    assert r.velocity == pytest.approx(v_true, abs=1e-3), str(r)


def test_centroid_matches_peak_where_the_packet_is_a_delta():
    """On a rigid mover the two estimators must agree exactly -- no packet to weight."""
    lat = Lattice(shape=(512,), n_species=1)
    auto = SpeciesShift(lattice=lat, velocities=np.array([[2]]))
    peak = measure_transport(auto, n_steps=50, ensemble=16, v_max=2.0, estimator="peak")
    cent = measure_transport(auto, n_steps=50, ensemble=16, v_max=2.0, estimator="centroid")
    assert peak.velocity == pytest.approx(2.0, abs=1e-9)
    assert cent.velocity == pytest.approx(2.0, abs=1e-9)
