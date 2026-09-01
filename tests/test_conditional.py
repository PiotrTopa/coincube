"""M2b: Wetterich's conditional-propagation automaton, and the class it belongs to.

Table CPA of arXiv:2203.14081, sect. "Conditional propagation", plus the complete
enumeration of particle-hole symmetric conditional-propagation rules on the same block.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.analysis.correlations import measure_transport
from pca3d.core.lattice import Lattice
from pca3d.models import conditional as C
from pca3d.models.generic import BlockAutomaton, RuleCycle


@pytest.fixture(scope="module")
def cpa():
    return C.wetterich_cpa_perm()


def test_table_cpa_is_a_bijection(cpa):
    """R2. Wetterich gives only 8 of 16 transitions; the rest follow from K."""
    assert len(np.unique(cpa)) == C.N_CFG


def test_table_cpa_properties(cpa):
    assert C.is_particle_hole_symmetric(cpa)  # R5, and how he completes the table
    assert C.is_conditional_propagation(cpa)
    # he states it as motion "in the presence of impurities"; it does not conserve N
    assert not C.conserves_particle_number(cpa)
    # and it is not reflection symmetric -- the mirror case comes from K, not from x<->x'
    assert not C.is_block_reflection_symmetric(cpa)


def test_the_exceptional_no_motion_case(cpa):
    """His one exceptional entry: a lone particle at x with both environment bits set.

    "If the environment-bits take the value one at both positions a single particle at x
    does not change its position."
    """
    c = C.encode(n_psi=1, n_phi=1, n_psi_p=0, n_phi_p=1)
    out = int(cpa[c])
    assert C.system_state(out) == C.system_state(c)  # did not move
    assert C.decode(out) == (1, 0, 0, 0)  # exactly his "10,00"

    # and its particle-hole partner: a lone particle at x' with both env bits zero
    c2 = C.particle_hole(c)
    assert C.system_state(int(cpa[c2])) == C.system_state(c2)


def test_every_other_single_carrier_config_propagates(cpa):
    """Eight single-carrier configurations, exactly two of which stall."""
    stalls = [
        c
        for c in range(C.N_CFG)
        if C.system_state(c) in (1, 2) and C.system_state(int(cpa[c])) == C.system_state(c)
    ]
    assert len(stalls) == 2


def test_enumeration_is_complete_and_contains_his_rule(cpa):
    rules = C.enumerate_conditional_rules()
    assert len(rules) == 9216  # 24 environment permutations x 384 mixed-sector maps
    assert any(np.array_equal(r, cpa) for r in rules)
    for r in rules[:300]:
        assert len(np.unique(r)) == C.N_CFG
        assert C.is_particle_hole_symmetric(r)
        assert C.is_conditional_propagation(r)


# -- transport ---------------------------------------------------------------------


def _cycle(perm, L=512):
    lat = Lattice(shape=(L,), n_species=2)
    return RuleCycle(
        steps=(
            BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=perm, origin=(0,)),
            BlockAutomaton(lattice=lat, block_shape=(2,), block_perm=perm, origin=(1,)),
        )
    )


def test_unconditional_swap_saturates_the_light_cone():
    """Control: with no condition the carrier moves one site per sub-step, so v = 2."""
    swap = np.array(
        [
            C.compose(((C.system_state(c) & 1) << 1) | ((C.system_state(c) >> 1) & 1), C.env_state(c))
            for c in range(C.N_CFG)
        ]
    )
    r = measure_transport(_cycle(swap), n_steps=60, ensemble=32, species=0, v_max=2.0)
    assert r.velocity == pytest.approx(2.0, abs=1e-9)
    assert r.is_ballistic


def test_conditional_propagation_is_ballistic_but_subluminal(cpa):
    """The M2b result: transport stays ballistic while the speed drops off the cone.

    This is what no rule in ADR 0002 could do -- there, every surviving ridge sat at the
    unmodified free-streaming velocity.
    """
    r = measure_transport(_cycle(cpa), n_steps=120, ensemble=64, seed=0, species=0, v_max=2.0)
    assert r.is_ballistic, str(r)
    assert 1.4 < r.velocity < 1.6, str(r)
    assert r.velocity < 2.0 - 0.3  # clearly below the light cone


def test_a_frozen_environment_traps_the_carrier():
    """Necessary condition found in M2b: if the environment cannot move, transport dies.

    Built explicitly rather than sampled. The carrier propagates only where the frozen
    environment reads 00 or 11 (that choice, not "00" alone, is what keeps the rule
    particle-hole symmetric). Since the environment never updates, a carrier reaching a
    blocking block is stuck there forever, and transport localises.

    In the sampled scan this held for 7 of 7 static-environment rules.
    """
    perm = np.empty(C.N_CFG, dtype=np.int64)
    for c in range(C.N_CFG):
        sysv, envv = C.system_state(c), C.env_state(c)
        moves = envv in (0, 3)
        out = (((sysv & 1) << 1) | ((sysv >> 1) & 1)) if moves else sysv
        perm[c] = C.compose(out, envv)  # environment frozen

    assert len(np.unique(perm)) == C.N_CFG, "frozen-environment rule must still be a bijection"
    assert C.is_particle_hole_symmetric(perm)
    assert C.is_conditional_propagation(perm)

    r = measure_transport(_cycle(perm), n_steps=60, ensemble=32, species=0, v_max=2.0)
    assert r.velocity < 0.5, str(r)


def test_tagged_carrier_count_is_conserved_by_the_class():
    """Conditional propagation conserves SYSTEM bits by construction.

    The class is non-conserving only in the environment channel: the system bit either
    moves or stays, never appears or disappears. Verified here for Wetterich's rule on
    every block configuration; it is why a lone tagged carrier stays lone forever
    (256/256 long runs in the M2c side-measurement).
    """
    perm = C.wetterich_cpa_perm()
    for c in range(C.N_CFG):
        n_in = bin(C.system_state(c)).count("1")
        n_out = bin(C.system_state(int(perm[c]))).count("1")
        assert n_in == n_out, f"config {c:04b}"
