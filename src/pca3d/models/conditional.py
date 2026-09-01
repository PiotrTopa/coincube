"""Conditional propagation: rules where *where* a carrier moves depends on its neighbours.

ADR 0002 found that non-conserving rules which leave propagation alone cannot bend the
dispersion -- the ridge either stays exactly at the free-streaming velocity or dissolves.
Conditional propagation is the one mechanism in Wetterich's toolbox that makes transport
itself configuration-dependent (2203.14081, sect. "Updating by shifted blocks" ->
"Conditional propagation").

The setup, following him exactly: a two-site block ``(x, x+eps)`` carrying one **system**
bit ``psi`` and one **environment** bit ``phi`` per site. For most environment
configurations the system bit propagates along the block diagonal, ``psi <-> psi'``. For
selected environment configurations it does not move -- "the motion of particles in the
presence of impurities". The environment updates too, and its update depends on the
system bit, so the coupling runs both ways.

Bit layout matches ``Lattice.bit_index`` with ``n_species = 2``:

    bit 0 = psi(x)      bit 1 = phi(x)      bit 2 = psi(x')     bit 3 = phi(x')

which coincides with Wetterich's tuple order ``(n_psi, n_phi, n_psi', n_phi')``, written
in his tables as the pair-of-pairs ``ab,cd``.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

N_BITS = 4
N_CFG = 1 << N_BITS
FULL = N_CFG - 1


def encode(n_psi: int, n_phi: int, n_psi_p: int, n_phi_p: int) -> int:
    """Wetterich's ``(n_psi, n_phi, n_psi', n_phi')`` -> configuration integer."""
    return n_psi | (n_phi << 1) | (n_psi_p << 2) | (n_phi_p << 3)


def decode(c: int) -> tuple[int, int, int, int]:
    return (c & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1)


def particle_hole(c: int) -> int:
    return c ^ FULL


def system_state(c: int) -> int:
    """``(psi, psi')`` packed as ``psi + 2 psi'``."""
    return (c & 1) | (((c >> 2) & 1) << 1)


def env_state(c: int) -> int:
    """``(phi, phi')`` packed as ``phi + 2 phi'``."""
    return ((c >> 1) & 1) | (((c >> 3) & 1) << 1)


def compose(system: int, env: int) -> int:
    return (system & 1) | ((env & 1) << 1) | (((system >> 1) & 1) << 2) | (((env >> 1) & 1) << 3)


# -- Wetterich's Table CPA, transcribed -------------------------------------------

#: The eight transitions written out in table CPA of arXiv:2203.14081. The other eight
#: follow by particle-hole symmetry, which is how he states the rule ("We specify only
#: the updating for eight out of the sixteen configurations").
#:
#: Read as ``"before": "after"`` in his ``ab,cd`` notation.
CPA_TABLE: dict[str, str] = {
    "11,11": "11,11",
    "11,10": "10,11",
    "10,11": "11,10",
    "10,10": "10,10",
    "10,01": "00,11",
    "11,00": "01,10",
    "01,11": "11,01",
    "11,01": "10,00",
}


def _parse(s: str) -> int:
    left, right = s.split(",")
    return encode(int(left[0]), int(left[1]), int(right[0]), int(right[1]))


def wetterich_cpa_perm() -> np.ndarray:
    """The block permutation of table CPA, completed by particle-hole symmetry."""
    perm = np.full(N_CFG, -1, dtype=np.int64)
    for before, after in CPA_TABLE.items():
        b, a = _parse(before), _parse(after)
        perm[b] = a
        perm[particle_hole(b)] = particle_hole(a)

    if np.any(perm < 0):
        missing = [format(int(c), "04b") for c in np.flatnonzero(perm < 0)]
        raise ValueError(f"table CPA plus its PH images does not cover {missing}")
    if len(np.unique(perm)) != N_CFG:
        raise ValueError("table CPA plus its PH images is not a bijection")
    return perm


# -- the class that rule belongs to ------------------------------------------------


def is_conditional_propagation(perm: np.ndarray) -> bool:
    """Does every configuration either propagate the system bits or hold them still?

    The system output must be either ``(psi, psi')`` (stay) or ``(psi', psi)`` (move
    along the diagonal). This is what distinguishes conditional *propagation* from an
    arbitrary block bijection: the carrier is never created, destroyed or teleported --
    only delayed.
    """
    for c in range(N_CFG):
        s_in, s_out = system_state(c), system_state(int(perm[c]))
        swapped = ((s_in & 1) << 1) | ((s_in >> 1) & 1)
        if s_out not in (s_in, swapped):
            return False
    return True


def is_particle_hole_symmetric(perm: np.ndarray) -> bool:
    return all(int(perm[particle_hole(c)]) == particle_hole(int(perm[c])) for c in range(N_CFG))


def conserves_particle_number(perm: np.ndarray) -> bool:
    return all(bin(c).count("1") == bin(int(perm[c])).count("1") for c in range(N_CFG))


def reflect_block(c: int) -> int:
    """Exchange the two sites of the block: ``(psi, phi, psi', phi') -> (psi', phi', psi, phi)``."""
    psi, phi, psi_p, phi_p = decode(c)
    return encode(psi_p, phi_p, psi, phi)


def is_block_reflection_symmetric(perm: np.ndarray) -> bool:
    """Invariance under exchanging the two sites of the block, ``x <-> x'``.

    Wetterich's table CPA is *not* reflection symmetric: the exceptional no-motion case
    is stated for a particle at ``x`` with both environment bits set, and the mirror case
    at ``x'`` arises by particle-hole conjugation rather than by reflection. So this is
    tracked as a property, not required.
    """
    return all(int(perm[reflect_block(c)]) == reflect_block(int(perm[c])) for c in range(N_CFG))


def enumerate_conditional_rules() -> list[np.ndarray]:
    """Every particle-hole symmetric conditional-propagation rule on this block.

    The enumeration is complete, and it is complete because the structure forces it:

      - System state ``(0,0)``: swapping is trivial, so these four configurations (one
        per environment state) map into themselves. The environment map there is any
        permutation of the four environment states -- 24 choices.
      - System state ``(1,1)``: likewise trivial to swap, and its map is *forced* by
        particle-hole conjugation of the ``(0,0)`` sector, since K exchanges the two.
      - Mixed system states ``(1,0)`` and ``(0,1)``: eight configurations that map among
        themselves. Choosing the images of the four ``(1,0)`` ones fixes the rest by K;
        we enumerate all ``8**4`` choices and keep the bijective ones, which leaves 384.

    Total: ``24 * 384 = 9216``. Wetterich's table CPA is one of them, which the test
    suite checks.
    """
    sys_of = np.array([system_state(c) for c in range(N_CFG)])
    env_of = np.array([env_state(c) for c in range(N_CFG)])

    zero_sector = [c for c in range(N_CFG) if sys_of[c] == 0]  # (0,0), 4 configs
    mixed = [c for c in range(N_CFG) if sys_of[c] in (1, 2)]  # 8 configs
    mixed_10 = [c for c in mixed if sys_of[c] == 1]  # 4 configs

    # -- the 384 legal maps of the mixed sector
    mixed_maps: list[dict[int, int]] = []
    for images in product(mixed, repeat=len(mixed_10)):
        m: dict[int, int] = {}
        ok = True
        for c, img in zip(mixed_10, images):
            m[c] = img
            m[particle_hole(c)] = particle_hole(img)
        if len(set(m.values())) != len(mixed):
            ok = False
        if ok and set(m.values()) == set(mixed):
            mixed_maps.append(m)

    rules: list[np.ndarray] = []
    for env_perm in permutations(range(4)):
        for m in mixed_maps:
            perm = np.full(N_CFG, -1, dtype=np.int64)
            for c in zero_sector:
                img = compose(0, env_perm[env_of[c]])
                perm[c] = img
                perm[particle_hole(c)] = particle_hole(img)
            for c, img in m.items():
                perm[c] = img
            if np.any(perm < 0) or len(np.unique(perm)) != N_CFG:
                continue
            rules.append(perm)
    return rules
