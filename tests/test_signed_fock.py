"""Calibration of the signed (fermionic) lift and the amplitude propagator.

The plus-gauge fast path is validated against a brute-force dense constructor that
shares no shortcuts with it, and the propagator is calibrated on cases with exact known
answers before it is allowed to say anything new.
"""

from __future__ import annotations

import numpy as np
import pytest

from pca3d.fock.signed import JW, SignedBlockCycle, site_major_ranks, species_major_ranks
from pca3d.fock.propagator import propagator
from pca3d.models import conditional as C


def diagonal_swap_perm() -> np.ndarray:
    """Unconditional streaming: system and env both hop across the block."""
    perm = np.empty(16, dtype=np.int64)
    for c in range(16):
        sysv, envv = C.system_state(c), C.env_state(c)
        sw = lambda v: ((v & 1) << 1) | ((v >> 1) & 1)
        perm[c] = C.compose(sw(sysv), sw(envv))
    return perm


# -- sign helpers -------------------------------------------------------------------


def test_translation_sign_matches_general_permutation_parity():
    """The closed-form per-species wrap sign against brute-force inversion counting."""
    model = SignedBlockCycle(n_sites=6, block_perm=C.wetterich_cpa_perm())
    pi = model._translation_pi()
    rng = np.random.default_rng(0)
    configs = rng.integers(0, 1 << model.n_modes, size=200, dtype=np.int64)
    fast = model._translation_sign(configs)
    slow = np.array([model.jw.mode_permutation_sign(int(c), pi) for c in configs])
    assert np.array_equal(fast, slow)


def test_parity_violating_block_is_rejected():
    perm = np.arange(16, dtype=np.int64)
    perm[0], perm[1] = 1, 0  # 0 particles <-> 1 particle: parity violation
    with pytest.raises(ValueError, match="parity"):
        SignedBlockCycle(n_sites=4, block_perm=perm)


# -- fast path == dense truth -------------------------------------------------------


@pytest.mark.parametrize("rule", ["cpa", "swap", "rule261", "rule884"])
@pytest.mark.parametrize("origin", [1])
def test_fast_path_matches_dense(rule, origin):
    """The contiguity theorem and the JW boundary factor, against brute force.

    n_sites=4 -> 8 modes -> 256-dim Fock space; every basis state checked.
    """
    perm = {
        "cpa": C.wetterich_cpa_perm(),
        "swap": diagonal_swap_perm(),
        "rule261": C.enumerate_conditional_rules()[261],
        "rule884": C.enumerate_conditional_rules()[884],
    }[rule]
    model = SignedBlockCycle(n_sites=4, block_perm=perm, boundary="ring")

    dense = model.dense_substep(origin)
    dim = 1 << model.n_modes
    configs = np.arange(dim, dtype=np.int64)
    signs = np.ones(dim, dtype=np.int64)
    out_cfg, out_sgn = model.substep(configs, signs, origin)

    fast = np.zeros((dim, dim), dtype=np.int8)
    fast[out_cfg, configs] = out_sgn
    assert np.array_equal(fast, dense), f"rule={rule} origin={origin}"


def test_free_swap_lift_is_gaussian():
    """U a†_m U† = ±a†_pi(m) as a matrix identity, for every mode.

    This is the operator-level statement that the lift of a pure transport rule is a
    genuine free-fermion circuit. It failed for two earlier sign conventions (set-based
    normal ordering; mode-map without config-identity crossings) and each failure
    produced a decaying |G| that could have been mistaken for physics.
    """
    model = SignedBlockCycle(n_sites=4, block_perm=diagonal_swap_perm(), boundary="open")
    dim = 1 << model.n_modes
    cfg = np.arange(dim, dtype=np.int64)
    oc, os_ = model.substep(cfg, np.ones(dim, dtype=np.int64), 0)
    U = np.zeros((dim, dim))
    U[oc, cfg] = os_

    def adag(mode):
        A = np.zeros((dim, dim))
        c2, s2 = model.jw.create(cfg, np.ones(dim, dtype=np.int64), mode)
        ok = s2 != 0
        A[c2[ok], cfg[ok]] = s2[ok]
        return A

    pi = {0: 2, 2: 0, 1: 3, 3: 1, 4: 6, 6: 4, 5: 7, 7: 5}
    for m in range(model.n_modes):
        lhs = U @ adag(m) @ U.T
        tgt = adag(pi[m])
        assert np.allclose(lhs, tgt) or np.allclose(lhs, -tgt), f"mode {m} not Gaussian"


def test_dense_substeps_are_signed_permutations():
    model = SignedBlockCycle(n_sites=4, block_perm=C.wetterich_cpa_perm(), boundary="ring")
    for origin in (0, 1):
        m = model.dense_substep(origin).astype(np.int32)
        assert np.array_equal(m @ m.T, np.eye(m.shape[0], dtype=np.int32))
        assert np.all(np.abs(m).sum(axis=0) == 1)


def test_disjoint_block_operators_commute():
    """Parity-even lifted blocks on disjoint modes must commute: applying the block
    layer in reversed block order must give identical configs AND signs."""
    model = SignedBlockCycle(n_sites=6, block_perm=C.wetterich_cpa_perm(), boundary="open")
    probe = np.random.default_rng(3).integers(0, 1 << model.n_modes, size=500, dtype=np.int64)
    ones = np.ones(len(probe), dtype=np.int64)

    fwd_c, fwd_s = model._apply_blocks(probe.copy(), ones.copy(), origin=0)

    original = model._blocks_for_origin
    model._blocks_for_origin = lambda origin: list(reversed(original(origin)))  # type: ignore
    rev_c, rev_s = model._apply_blocks(probe.copy(), ones.copy(), origin=0)
    model._blocks_for_origin = original  # type: ignore

    assert np.array_equal(fwd_c, rev_c)
    assert np.array_equal(fwd_s, rev_s)


# -- propagator calibration ---------------------------------------------------------


def test_anticommutator_sum_rule_at_t0():
    """G+(0,x;y) + G-(0,x;y) = delta_xy exactly, per sample, any rule."""
    model = SignedBlockCycle(n_sites=8, block_perm=C.wetterich_cpa_perm())
    r = propagator(model, n_substeps=0, ensemble=512, seed=1)
    total = r.g_particle[0] + r.g_hole[0]
    want = np.zeros_like(total)
    want[0, 0] = 1.0
    assert np.allclose(total, want, atol=1e-12)


def test_free_streaming_gives_the_exact_massless_propagator():
    """Unconditional diagonal swap: the defect streams rigidly, so |G+| = 1 on the
    light cone and 0 elsewhere, at every time -- the exact free lattice propagator.
    Any sign error would show as amplitude loss or a sign flip along the cone.
    """
    model = SignedBlockCycle(n_sites=12, block_perm=diagonal_swap_perm())
    r = propagator(model, n_substeps=8, ensemble=256, seed=0)

    # Exact-per-sample statements (immune to the binomial fluctuation of how many
    # vacuum samples have site y empty, which is the only stochastic element):
    #   - all amplitude sits on exactly ONE site,
    #   - with magnitude exactly equal to the surviving-sample fraction (all
    #     contributions coherent, same sign -- any sign error would cancel them),
    #   - and that fraction is CONSTANT in time (free streaming never decoheres).
    for t in range(9):
        g = r.g_particle[t, :, 0]  # system species
        assert np.count_nonzero(np.abs(g) > 1e-12) == 1, f"t={t}: {g}"
        assert np.max(np.abs(g)) == pytest.approx(r.survival[t], abs=1e-12)
        assert r.survival[t] == pytest.approx(r.survival[0], abs=1e-12)
    assert r.survival[0] == pytest.approx(0.5, abs=5 / np.sqrt(r.ensemble))

    # the amplitude-carrying site advances one site per sub-step (velocity 1/substep)
    pos = [int(np.argmax(np.abs(r.g_particle[t, :, 0]))) for t in range(9)]
    steps = [(pos[t + 1] - pos[t]) % 12 for t in range(8)]
    assert all(s == 1 for s in steps), steps


def test_conditional_propagation_amplitude_decays_but_stays_causal():
    """CPA: the dressing grows, so single-bit coherence decays; nothing may appear
    outside the light cone (1 site per sub-step), and survival must be < 1."""
    model = SignedBlockCycle(n_sites=16, block_perm=C.wetterich_cpa_perm())
    r = propagator(model, n_substeps=6, ensemble=2048, seed=2)

    assert r.survival[0] == pytest.approx(0.5, abs=5 / np.sqrt(r.ensemble))
    assert r.survival[6] < 0.9 * r.survival[0]  # dressing is real: coherence decays
    for t in range(7):
        g = np.abs(r.g_particle[t]).sum(axis=1)
        for x in range(16):
            d = min(x, 16 - x)
            if d > t:
                assert g[x] == pytest.approx(0.0, abs=1e-12), f"causality violated t={t} x={x}"
