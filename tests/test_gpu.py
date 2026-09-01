"""GPU backend: bitwise agreement with the CPU path, on every model class.

Skipped cleanly on hosts without a working CUDA stack. When it runs, agreement is
required to be exact -- the models are pure bit manipulation, so any difference at all
is a bug, not precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.core.backend import gpu_available, to_device, to_numpy
from pca3d.core.lattice import Lattice
from pca3d.models import conditional as C
from pca3d.models.generic import BlockAutomaton, RuleCycle, SpeciesShift
from pca3d.models.wetterich1d import Wetterich1D

pytestmark = pytest.mark.skipif(not gpu_available(), reason="no usable GPU")


def _agree(auto, shape=(8,), n_species=2, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(0, 2, size=(16, auto.lattice.n_sites, auto.lattice.n_species)).astype(bool)
    cpu = n.copy()
    gpu = to_device(n.copy(), True)
    for _ in range(steps):
        cpu = auto.step_bits(cpu)
        gpu = auto.step_bits(gpu)
    assert np.array_equal(cpu, to_numpy(gpu))


def test_species_shift_agrees():
    lat = Lattice(shape=(6, 6, 6), n_species=6)
    vel = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    _agree(SpeciesShift(lattice=lat, velocities=vel))


def test_block_automaton_agrees_1d_and_3d():
    lat1 = Lattice(shape=(16,), n_species=2)
    _agree(BlockAutomaton(lattice=lat1, block_shape=(2,), block_perm=C.wetterich_cpa_perm(), origin=(1,)))
    rng = np.random.default_rng(1)
    from pca3d.models.generic import particle_number_conserving_block_perm
    lat3 = Lattice(shape=(4, 4, 4), n_species=2)
    perm = particle_number_conserving_block_perm(16, rng)
    _agree(BlockAutomaton(lattice=lat3, block_shape=(2, 2, 2), block_perm=perm, origin=(1, 0, 1)))


def test_rule_cycle_agrees():
    lat = Lattice(shape=(32,), n_species=2)
    cyc = RuleCycle(steps=(
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=C.wetterich_cpa_perm(), origin=(0,)),
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=C.wetterich_cpa_perm(), origin=(1,)),
    ))
    _agree(cyc, steps=8)


def test_wetterich1d_agrees():
    _agree(Wetterich1D(n_sites=16, extended=True), steps=8)


def test_evolve_record_bitwise_identical():
    """Same seed => identical record, because sampling happens on the CPU."""
    from pca3d.analysis.structure_factor import evolve_record

    m = Wetterich1D(n_sites=32, extended=True)
    cpu = evolve_record(m, 12, 8, np.random.default_rng(3), use_gpu=False)
    gpu = evolve_record(m, 12, 8, np.random.default_rng(3), use_gpu=True)
    assert np.array_equal(cpu, to_numpy(gpu))


def test_transport_measurement_matches_cpu():
    from pca3d.analysis.correlations import measure_transport

    lat = Lattice(shape=(256,), n_species=2)
    cyc = RuleCycle(steps=(
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=C.wetterich_cpa_perm(), origin=(0,)),
        BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=C.wetterich_cpa_perm(), origin=(1,)),
    ))
    a = measure_transport(cyc, n_steps=30, ensemble=16, seed=5, species=0, v_max=2.0, estimator="centroid")
    b = measure_transport(cyc, n_steps=30, ensemble=16, seed=5, species=0, v_max=2.0, estimator="centroid", use_gpu=True)
    assert a.velocity == pytest.approx(b.velocity, abs=1e-12)
