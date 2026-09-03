"""The layered quaternionic coin CA ("coincube") — the W3c legal architecture.

Per axis-a sub-step, three sequential layers, each a manifestly legal CA layer
(local, homogeneous, bijective, unique-jump, fermion-parity even):

  L2 (conversion): at every site, IF the axis-a env bit at that site is 1,
      swap the occupations of channel pairs (c, perm_a(c)) — the permutation
      part of C_a, an involution pairing channels of opposite d_a. One-site
      block over (4 species bits, 1 env bit); occupied-channel swap = two bit
      flips (R4-1/2 even). Fermionic lift: a +-90 deg Givens rotation
      G = exp((pi/2)(a_c^dag a_c' - a_c'^dag a_c)) of two same-site modes,
      controlled by the env occupation — no Jordan-Wigner strings. G's
      single-particle matrix IS the signed swap block of C_a (that is where
      C^2 = -I comes from); |11> -> +|11>.
  L1 (motion): channel c translates by d_a(c) along axis a, unconditionally.
      Species-dependent translations are legal CA layers.
  L3 (env): axis-a env field pair-swaps along the NEXT axis (a+1 mod 3),
      origin alternating per sub-step — autonomous streaming; Bernoulli(q)
      product measure is stationary. Cross-streaming is the legal
      decorrelation knob: co-streaming (env_a along axis a) lets a
      non-converting carrier co-move with its env bit and re-read it, so
      conversions arrive in near-scalar bursts (C^2 = -I) — measured to
      PT-split the Weyl pair at the X point into a real+oscillating pair
      (node destroyed). Cross-streaming decorrelates every read.

Annealed Bloch operator per sub-step: T_a(k) = E_a(k_a) [(1-q) I + q C_a],
E_a = diag(e^{i k_a d_a}); cycle U = Tz^2 Ty^2 Tx^2. Same quaternion tables
as ADR 0011; carries the isotropic Weyl cone at all 8 BZ corners.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:  # pragma: no cover
    import numpy as xp
    GPU = False

_I2 = np.eye(2)
_X = np.array([[0.0, 1], [1, 0]])
_Z = np.diag([1.0, -1])
_XZ = _X @ _Z

#: quaternion conversion triple and move-direction diagonals (ADR 0011)
COIN_C = [np.kron(_XZ, _I2), np.kron(_Z, _XZ), -np.kron(_X, _XZ)]
COIN_D = [np.diag(np.kron(_Z, _I2)).astype(np.int64),
          np.diag(np.kron(_Z, _Z)).astype(np.int64),
          np.diag(np.kron(_I2, _Z)).astype(np.int64)]


def perm_sign(c: np.ndarray):
    """Signed permutation matrix -> (perm[j] = i, sign j->i)."""
    perm = np.argmax(np.abs(c), axis=0)
    sign = c[perm, np.arange(c.shape[1])].astype(np.int64)
    return perm, sign


PERMS, SIGNS = zip(*(perm_sign(c) for c in COIN_C))


def annealed_u(kvec, q: float) -> np.ndarray:
    """Exact annealed one-cycle Bloch operator (2 sub-steps per axis)."""
    u = np.eye(4, dtype=complex)
    for a in range(3):
        t = np.diag(np.exp(1j * kvec[a] * COIN_D[a])) @ (
            (1 - q) * np.eye(4) + q * COIN_C[a])
        u = t @ t @ u
    return u


# -- classical multi-particle layers (CA-legality surface) ---------------------


def layer_convert(n: np.ndarray, env_a: np.ndarray, a: int) -> np.ndarray:
    """L2: swap channel pairs (c, perm_a(c)) wherever env_a == 1. Bijective."""
    out = n.copy()
    done = set()
    for c in range(4):
        cp = int(PERMS[a][c])
        if (c, cp) in done or (cp, c) in done:
            continue
        done.add((c, cp))
        out[c] = np.where(env_a, n[cp], n[c])
        out[cp] = np.where(env_a, n[c], n[cp])
    return out


def layer_shift(n: np.ndarray, a: int) -> np.ndarray:
    """L1: channel c translates by d_a(c) along axis a. Bijective."""
    out = np.empty_like(n)
    for c in range(4):
        out[c] = np.roll(n[c], int(COIN_D[a][c]), axis=a)
    return out


def layer_env(env_a: np.ndarray, a: int, o: int,
              stream_axis: int | None = None) -> np.ndarray:
    """L3: pair-swap streaming of the axis-a env field, origin o. Bijective.

    stream_axis defaults to the cross-stream choice (a+1) mod 3.
    """
    sa = (a + 1) % 3 if stream_axis is None else stream_axis
    m = np.moveaxis(env_a, sa, -1)
    if o:
        m = np.roll(m, -1, axis=-1)
    m = m.copy()
    m0 = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = m0
    if o:
        m = np.roll(m, 1, axis=-1)
    return np.moveaxis(m, -1, sa)


def step_bits(n: np.ndarray, env: np.ndarray):
    """One full cycle of the classical CA: n (4, L, L, L), env (3, L, L, L)."""
    for a in range(3):
        for o in (0, 1):
            n = layer_convert(n, env[a], a)
            n = layer_shift(n, a)
            env[a] = layer_env(env[a], a, o)
    return n, env


# -- exact single-particle signed field (the instrument) -----------------------


def _swap_once(field, sa, o):
    """One pair-swap streaming application along axis sa, origin o."""
    m = xp.moveaxis(field, sa, -1)
    if o:
        m = xp.roll(m, -1, axis=-1)
    m0 = m[..., 0::2].copy()
    m[..., 0::2] = m[..., 1::2]
    m[..., 1::2] = m0
    if o:
        m = xp.roll(m, 1, axis=-1)
    return xp.moveaxis(m, -1, sa)


def evolve_field_cc(L: int, R: int, TCYC: int, q: float, seed: int = 3,
                    annealed: bool = False, launch: int = 0,
                    co_stream: bool = False, n_blocks: int = 1,
                    streaming: str = "production") -> np.ndarray:
    """Signed single-particle field of the layered CA, mean over R media.

    Per medium the field evolution is the exact fermionic single-particle
    sector: conversion applies the SIGNED C_a where env = 1 (Givens signs),
    then the coin-steered shift. Returns G(t, 4, L, L, L).

    streaming = 'fresh' (F1) is the manuscript's model: three
    PHASE-CONTINUING swaps per axis substep (bits at +-6 sites/cycle); no
    path ever re-reads a bit, so the quenched ensemble propagator equals
    the annealed operator exactly for TCYC <= ceil(L/8) (freshtape
    theorem; sharp torus horizon).
    streaming = 'production' (default, kept as the byte-identical baseline
    of the generic-schedule campaign): one swap per axis substep (bits
    counter-propagate at +-2 sites/cycle; re-reads and their corrections
    exist).
    """
    if L % 2:
        raise ValueError("coincube requires even L (pair-swap env streaming)")
    if streaming not in ("production", "fresh"):
        raise ValueError("streaming must be 'production' or 'fresh'")
    perms = [xp.asarray(p) for p in PERMS]
    signs = [xp.asarray(s) for s in SIGNS]
    fresh = streaming == "fresh"
    r = np.random.default_rng(seed)
    acc = np.zeros((max(1, n_blocks), TCYC + 1, 4, L, L, L))
    per_block = max(1, R // max(1, n_blocks))

    for rep in range(R):
        env = xp.asarray(r.random((3, L, L, L)) < q)
        phase = [0, 0, 0]
        g = xp.zeros((4, L, L, L))
        g[launch, L // 2, L // 2, L // 2] = 1.0
        out = [g.copy()]

        def substep(a: int, o: int):
            nonlocal g
            if annealed:
                mask = xp.asarray(r.random((L, L, L)) < q)
            else:
                mask = env[a]
            new = xp.empty_like(g)
            for c in range(4):
                cp = int(perms[a][c])
                new[cp] = xp.where(mask, float(signs[a][c]) * g[c], g[cp])
            g = new
            for c in range(4):
                g[c] = xp.roll(g[c], int(COIN_D[a][c]), axis=a)
            if not annealed:
                sa = a if co_stream else (a + 1) % 3
                if fresh:
                    e = env[a]
                    for _s in range(3):
                        e = _swap_once(e, sa, phase[a] % 2)
                        phase[a] += 1
                    env[a] = e
                else:
                    env[a] = _swap_once(env[a], sa, o)

        for _ in range(TCYC):
            for a in (0, 1, 2):
                for o in (0, 1):
                    substep(a, o)
            out.append(g.copy())
        stack = xp.stack(out)
        b = min(rep // per_block, max(1, n_blocks) - 1)
        acc[b] += stack.get() if GPU else stack
    acc /= (R / max(1, n_blocks))
    return acc[0] if n_blocks <= 1 else acc


# -- M8: the massive (inversion-doubled) coincube ------------------------------
#
# 8 channels: index = 2*coin + b_m. The b_m = 1 sector is the spatial-inversion
# copy (directions flipped) -- same node, same speed, opposite chirality; the
# mass layer C_m = 1_4 (x) XZ (a controlled Givens flipping b_m, same certified
# layer type, controlled by a 4th env field of density q_m) couples the two
# chiralities: exact Dirac mass m = arctan(q_m / (1 - q_m)) at the node
# (scripts/m8_mass_exact.py). The C (x) Z doubling fails: -C is the other
# corner speed family (different omega0) -- no shared node.

COIN_C8 = [np.kron(c, _I2) for c in COIN_C]
COIN_D8 = [np.kron(d, np.array([1, -1], dtype=np.int64)) for d in COIN_D]
MASS_C = np.kron(np.eye(4), _XZ)
PERMS8, SIGNS8 = zip(*(perm_sign(c) for c in COIN_C8))
MASS_PERM, MASS_SIGN = perm_sign(MASS_C)


def annealed_u8(kvec, q: float, qm: float) -> np.ndarray:
    """Annealed one-cycle Bloch operator of the massive coincube."""
    u = np.eye(8, dtype=complex)
    for a in range(3):
        t = np.diag(np.exp(1j * kvec[a] * COIN_D8[a])) @ (
            (1 - q) * np.eye(8) + q * COIN_C8[a])
        u = t @ t @ u
    return ((1 - qm) * np.eye(8) + qm * MASS_C) @ u


def evolve_field_m8(L: int, R: int, TCYC: int, q: float, qm: float,
                    seed: int = 3, annealed: bool = False,
                    launch: int = 0, n_blocks: int = 1,
                    streaming: str = "production") -> np.ndarray:
    """Signed single-particle field of the massive coincube, mean over media.

    Axis layers as in evolve_field_cc (conversion, coin-steered shift,
    cross-streamed env); one mass layer per cycle (site-controlled Givens on
    the b_m pairs, 4th env field of density q_m). streaming = 'fresh' (F1)
    is the manuscript's model: carrier fields three phase-continuing swaps
    per substep; mass field three phase-continuing swaps per cycle along
    axis 0 -- all reads fresh for TCYC <= ceil(L/8).
    streaming = 'production' (default, generic-schedule baseline): carrier
    fields one swap per substep, mass field streamed along axis 0 with
    per-cycle alternating origin (re-reads at Delta t = 2 explain the
    quenched mass drifts). Returns G(t, 8, L, L, L).
    """
    if L % 2:
        raise ValueError("coincube requires even L (pair-swap env streaming)")
    if streaming not in ("production", "fresh"):
        raise ValueError("streaming must be 'production' or 'fresh'")
    perms = [xp.asarray(p) for p in PERMS8]
    signs = [xp.asarray(s) for s in SIGNS8]
    mperm = np.asarray(MASS_PERM)
    msign = np.asarray(MASS_SIGN)
    fresh = streaming == "fresh"
    r = np.random.default_rng(seed)
    acc = np.zeros((max(1, n_blocks), TCYC + 1, 8, L, L, L))
    per_block = max(1, R // max(1, n_blocks))

    for rep in range(R):
        env = xp.asarray(r.random((3, L, L, L)) < q)
        envm = xp.asarray(r.random((L, L, L)) < qm)
        phase = [0, 0, 0]
        mphase = 0
        g = xp.zeros((8, L, L, L))
        g[launch, L // 2, L // 2, L // 2] = 1.0
        out = [g.copy()]

        def substep(a: int, o: int):
            nonlocal g
            mask = (xp.asarray(r.random((L, L, L)) < q) if annealed
                    else env[a])
            new = xp.empty_like(g)
            for c in range(8):
                cp = int(perms[a][c])
                new[cp] = xp.where(mask, float(signs[a][c]) * g[c], g[cp])
            g = new
            for c in range(8):
                g[c] = xp.roll(g[c], int(COIN_D8[a][c]), axis=a)
            if not annealed:
                sa = (a + 1) % 3
                if fresh:
                    e = env[a]
                    for _s in range(3):
                        e = _swap_once(e, sa, phase[a] % 2)
                        phase[a] += 1
                    env[a] = e
                else:
                    env[a] = _swap_once(env[a], sa, o)

        for t in range(TCYC):
            for a in (0, 1, 2):
                for o in (0, 1):
                    substep(a, o)
            mask = (xp.asarray(r.random((L, L, L)) < qm) if annealed
                    else envm)
            new = xp.empty_like(g)
            for c in range(8):
                cp = int(mperm[c])
                new[cp] = xp.where(mask, float(msign[c]) * g[c], g[cp])
            g = new
            if not annealed:
                if fresh:
                    nonloc = envm
                    for _s in range(3):
                        nonloc = _swap_once(nonloc, 0, mphase % 2)
                        mphase += 1
                    envm = nonloc
                else:
                    envm = _swap_once(envm, 0, t % 2)
            out.append(g.copy())
        stack = xp.stack(out)
        b = min(rep // per_block, max(1, n_blocks) - 1)
        acc[b] += stack.get() if GPU else stack
    acc /= (R / max(1, n_blocks))
    return acc[0] if n_blocks <= 1 else acc


# -- Phase I: the imprint (back-reaction) layer L4 -----------------------------
#
# Once per cycle, at sites where the autonomous imprint field fires AND the
# site's carrier fermion parity is odd, two env species are swapped at that
# site. Parity control (not occupancy) keeps the layer particle-hole
# equivariant, so the E1 complex structure survives at g > 0. Fermionic lift:
# a controlled swap of two same-site env modes (adjacent in the site-major env
# ordering -- no strings; |11> -> -|11> standard fermionic swap). g = 0 is
# exactly the v1.0-dirac model.
#
# SYMMETRY (theory-interaction.md, binding): a FIXED swap pair breaks cubic
# symmetry at O(g) (anisotropic node corrections at 10 sigma, parasitic
# frequency drift). The imprint field is therefore 4-valued:
# iota(x) in {0, (xy), (yz), (zx)} with probability g/3 for each pair --
# C3-covariant, cannot generate anisotropic node corrections at any order.
# One-sided amplitude law (annealed order, LIFT-GAUGE DEPENDENT): with the
# production PERMUTATION lift (ADR 0014), U_g(k) = (1 - 2 g q^2) U(k); the
# Givens lift would give (1 - 2 g q (1-q)). Pure isotropic damping either
# way; cone, residues, mass untouched.


PAIRS = ((0, 1), (1, 2), (2, 0))


def layer_imprint(n: np.ndarray, env: np.ndarray, iota: np.ndarray,
                  pair: tuple[int, int] | None = None):
    """L4 classical layer: swap an env pair where iota fires & parity odd.

    iota is 4-valued: 0 = no imprint; p = 1..3 selects PAIRS[p-1]. A fixed
    pair (legacy/tests) can be forced via ``pair``. Rotating (C3-covariant)
    selection is the production form -- see the symmetry note above.
    """
    par = (n.sum(axis=0) % 2).astype(bool)
    env = env.copy()
    if pair is not None:
        fire = (iota.astype(bool)) & par
        a, b = pair
        ea = np.where(fire, env[b], env[a])
        eb = np.where(fire, env[a], env[b])
        env[a], env[b] = ea, eb
        return env
    for p, (a, b) in enumerate(PAIRS, start=1):
        fire = (iota == p) & par
        ea = np.where(fire, env[b], env[a])
        eb = np.where(fire, env[a], env[b])
        env[a], env[b] = ea, eb
    return env


def sample_iota(rng, L: int, g: float) -> np.ndarray:
    """4-valued C3-covariant imprint field: P(pair p) = g/3 each."""
    u = rng.random((L, L, L))
    out = np.zeros((L, L, L), dtype=np.int64)
    for p in range(1, 4):
        out[(u >= (p - 1) * g / 3) & (u < p * g / 3)] = p
    return out
