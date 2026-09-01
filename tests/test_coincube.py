"""CA-legality and exactness tests for the layered coincube model (W3c)."""
import numpy as np
import pytest

from pca3d.models.coincube import (COIN_C, COIN_D, PERMS, SIGNS, annealed_u,
                                   evolve_field_cc, layer_convert, layer_env,
                                   layer_shift, step_bits)


def test_quaternion_tables():
    for i in range(3):
        assert np.allclose(COIN_C[i] @ COIN_C[i], -np.eye(4))
        for j in range(i + 1, 3):
            assert np.allclose(COIN_C[i] @ COIN_C[j] + COIN_C[j] @ COIN_C[i], 0)
    # conversion pairs channels of opposite direction on the active axis
    for a in range(3):
        for c in range(4):
            assert COIN_D[a][int(PERMS[a][c])] == -COIN_D[a][c]


def test_layer_bijectivity_and_conservation():
    rng = np.random.default_rng(5)
    L = 4
    n = rng.integers(0, 2, size=(4, L, L, L))
    env = rng.integers(0, 2, size=(3, L, L, L)).astype(bool)
    for a in range(3):
        # L2 is an involution at fixed env (perm parts are involutions)
        n2 = layer_convert(layer_convert(n, env[a], a), env[a], a)
        assert np.array_equal(n2, n)
        # L2 conserves per-site particle count (R4-1/2: 0 or 2 bit flips)
        assert np.array_equal(layer_convert(n, env[a], a).sum(axis=0),
                              n.sum(axis=0))
        # L1 inverse = opposite shift
        s = layer_shift(n, a)
        back = np.stack([np.roll(s[c], -int(COIN_D[a][c]), axis=a)
                         for c in range(4)])
        assert np.array_equal(back, n)
        # L3 is an involution at fixed origin
        for o in (0, 1):
            assert np.array_equal(layer_env(layer_env(env[a], a, o), a, o),
                                  env[a])
    # full cycle conserves particle number (permutation of configurations)
    n3, env3 = step_bits(n.copy(), env.copy())
    assert n3.sum() == n.sum()
    assert env3.sum() == env.sum()


def test_deterministic_exactness_q0_and_q1():
    """q=0 (free streaming) and q=1 (always convert) are deterministic:
    the field must match the annealed Bloch operator exactly."""
    L, T = 8, 3
    for q in (0.0, 1.0):
        G = evolve_field_cc(L, 1, T, q, seed=2, annealed=False)
        Z = np.fft.fftn(G, axes=(2, 3, 4))
        kp = (3, 1, 5)
        kvec = np.array(kp) * 2 * np.pi / L
        z0 = Z[0, :, kp[0], kp[1], kp[2]]
        for t in (1, 3):
            pred = np.linalg.matrix_power(annealed_u(-kvec, q), t) @ z0
            got = Z[t, :, kp[0], kp[1], kp[2]]
            assert np.abs(pred - got).max() < 1e-10, (q, t)


def test_field_norm_conserved_per_realization():
    L, T = 8, 4
    G = evolve_field_cc(L, 1, T, 0.3, seed=7)
    for t in range(T + 1):
        assert abs(np.abs(G[t]).sum() - 1.0) < 1e-12


def test_signs_active():
    """At q=1 two consecutive conversions on the same axis give C^2 = -1:
    the signed field must differ from the unsigned one."""
    L, T = 8, 2
    G = evolve_field_cc(L, 1, T, 1.0, seed=1)
    assert G.min() < -1e-12


def test_m8_deterministic_exactness():
    """q=qm=0 (free) and q=0/qm=1 (pure mass rotation) are deterministic:
    field must match the annealed Bloch operator exactly."""
    from pca3d.models.coincube import annealed_u8, evolve_field_m8
    L, T = 8, 3
    for q, qm in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)):
        G = evolve_field_m8(L, 1, T, q, qm, seed=2)
        Z = np.fft.fftn(G, axes=(2, 3, 4))
        kp = (3, 1, 5)
        kvec = np.array(kp) * 2 * np.pi / L
        z0 = Z[0, :, kp[0], kp[1], kp[2]]
        for t in (1, 3):
            pred = np.linalg.matrix_power(annealed_u8(-kvec, q, qm), t) @ z0
            got = Z[t, :, kp[0], kp[1], kp[2]]
            assert np.abs(pred - got).max() < 1e-10, (q, qm, t)


def test_m8_norm_and_signs():
    from pca3d.models.coincube import evolve_field_m8
    L, T = 8, 4
    G = evolve_field_m8(L, 1, T, 0.3, 0.2, seed=7)
    for t in range(T + 1):
        assert abs(np.abs(G[t]).sum() - 1.0) < 1e-12
    assert G.min() < -1e-12


def test_imprint_layer_legality():
    """L4: bijective (involution at fixed control), PH-equivariant control,
    conserves particle number, identity at g = 0."""
    from pca3d.models.coincube import layer_imprint
    rng = np.random.default_rng(9)
    L = 4
    n = rng.integers(0, 2, size=(8, L, L, L))
    env = rng.integers(0, 2, size=(3, L, L, L)).astype(bool)
    iota = rng.integers(0, 4, size=(L, L, L))
    e1 = layer_imprint(n, env, iota)
    e2 = layer_imprint(n, e1, iota)
    assert np.array_equal(e2, env)                       # involution
    assert e1.sum() == env.sum()                         # number conserved
    # PH equivariance of the control: complementing carriers (8 channels)
    # leaves the site parity unchanged, so the SAME env update results
    e1c = layer_imprint(1 - n, env, iota)
    assert np.array_equal(e1c, e1)
    # g = 0: identity
    assert np.array_equal(layer_imprint(n, env, np.zeros_like(iota)), env)
