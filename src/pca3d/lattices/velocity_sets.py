"""Candidate velocity sets, and the two things that decide whether one can work.

Given the finite-ray theorem, a single-fermion automaton's light cone is the finite set
of rays ``{D_C / L_C}``. So the choice of velocity set is no longer a detail -- it *is*
the approximation to the light cone. Two numbers matter:

1. **Angular gap.** The worst-case angle from an arbitrary direction on the sphere to
   the nearest available ray. This is the irreducible anisotropy of ballistic transport
   and it does *not* shrink as ``k -> 0``, because the branches are exactly linear.

2. **Lattice tensor isotropy.** The classical lattice-gas criterion (d'Humieres,
   Lallemand & Frisch 1986): the second- and fourth-rank tensors

       T2^ab   = sum_i c_i^a c_i^b
       T4^abcd = sum_i c_i^a c_i^b c_i^c c_i^d

   must be isotropic, i.e. ``T2 ~ delta^ab`` and
   ``T4 ~ (delta^ab delta^cd + delta^ac delta^bd + delta^ad delta^bc)``. No 3D Bravais
   lattice achieves this for T4 with a single speed; FCHC in 4D does, which is the
   entire reason it was invented. This governs whether isotropy can be recovered
   *statistically* after coarse-graining and interaction, which -- given the theorem --
   is the only route left.

A velocity set with a small angular gap but anisotropic T4 will not give isotropic
hydrodynamics; one with isotropic T4 but a large angular gap will not give ballistic
isotropy. We report both and never conflate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np


@dataclass(frozen=True)
class VelocitySet:
    name: str
    vectors: np.ndarray  # (n, dim) integer displacements per time step
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.vectors)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def single_speed(self) -> bool:
        """All carriers move at the same speed -- required if all are to be 'light'."""
        s = self.speeds
        return bool(np.allclose(s, s[0]))

    def directions(self) -> np.ndarray:
        v = self.vectors[self.speeds > 0]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # -- criterion 1: angular gap ---------------------------------------------

    def angular_gap(self, n_probe: int = 200_000, seed: int = 0) -> float:
        """Worst-case angle (radians) from a direction on the sphere to the nearest ray.

        Monte Carlo over the sphere. Deterministic given ``seed``.
        """
        rays = self.directions()
        rng = np.random.default_rng(seed)
        probe = rng.normal(size=(n_probe, self.dim))
        probe /= np.linalg.norm(probe, axis=1, keepdims=True)
        best = (probe @ rays.T).max(axis=1)
        return float(np.arccos(np.clip(best, -1.0, 1.0)).max())

    # -- criterion 2: lattice tensor isotropy ---------------------------------

    def tensor_2(self) -> np.ndarray:
        return np.einsum("ia,ib->ab", self.vectors, self.vectors).astype(float)

    def tensor_4(self) -> np.ndarray:
        v = self.vectors.astype(float)
        return np.einsum("ia,ib,ic,id->abcd", v, v, v, v)

    def t2_anisotropy(self) -> float:
        """Relative deviation of T2 from a multiple of the identity."""
        T = self.tensor_2()
        d = self.dim
        iso = np.trace(T) / d * np.eye(d)
        return float(np.linalg.norm(T - iso) / max(np.linalg.norm(iso), 1e-300))

    def t4_anisotropy(self) -> float:
        """Relative deviation of T4 from the isotropic rank-4 tensor.

        The isotropic form is ``A (d^ab d^cd + d^ac d^bd + d^ad d^bc)``; ``A`` is fixed
        by least squares, so this measures the part of T4 that no choice of coefficient
        can absorb. Zero means fully isotropic to fourth order.
        """
        T = self.tensor_4()
        d = self.dim
        I = np.eye(d)
        iso = (
            np.einsum("ab,cd->abcd", I, I)
            + np.einsum("ac,bd->abcd", I, I)
            + np.einsum("ad,bc->abcd", I, I)
        )
        A = float((T * iso).sum() / (iso * iso).sum())
        return float(np.linalg.norm(T - A * iso) / max(np.linalg.norm(T), 1e-300))


def _perms_with_signs(pattern: tuple[int, ...]) -> np.ndarray:
    """All distinct signed permutations of a pattern, e.g. (1,1,0) -> the fcc 12."""
    out = set()
    for p in set(permutations(pattern)):
        for signs in product(*[(1, -1) if v else (1,) for v in p]):
            out.add(tuple(s * v for s, v in zip(signs, p)))
    return np.array(sorted(out), dtype=np.int64)


# -- the candidates ------------------------------------------------------------

CUBIC_6 = VelocitySet(
    "cubic-6", _perms_with_signs((1, 0, 0)), "simple cubic nearest neighbours, |c| = 1"
)
FCC_12 = VelocitySet(
    "fcc-12", _perms_with_signs((1, 1, 0)), "face centres, |c| = sqrt(2), single speed"
)
BCC_8 = VelocitySet(
    "bcc-8", _perms_with_signs((1, 1, 1)), "body diagonals, |c| = sqrt(3), single speed"
)
CUBIC_FCC_18 = VelocitySet(
    "cubic+fcc-18",
    np.vstack([CUBIC_6.vectors, FCC_12.vectors]),
    "the D3Q19 stencil (minus rest particle); two speeds",
)
ALL_26 = VelocitySet(
    "all-26",
    np.vstack([CUBIC_6.vectors, FCC_12.vectors, BCC_8.vectors]),
    "the D3Q27 stencil (minus rest particle); three speeds",
)

#: FCHC: the 24 vectors (+-1, +-1, 0, 0) and permutations -- the D4 root system, the
#: 24-cell. Single speed sqrt(2), and its point group makes T4 isotropic in 4D. This is
#: the lattice that rescued 3D lattice-gas hydrodynamics.
FCHC_24 = VelocitySet(
    "fchc-24", _perms_with_signs((1, 1, 0, 0)), "D4 root system / 24-cell, |c| = sqrt(2)"
)


def project_fchc_to_3d() -> VelocitySet:
    """Drop the 4th coordinate, as in the standard one-cell-thick FCHC slab.

    The 24 four-dimensional vectors split: 12 have ``c_3 = 0`` and project to the fcc
    12; the other 12 have ``c_3 = +-1`` and project onto the 6 cube axes with
    multiplicity two. So the projected *direction* set is the 18 of D3Q19 -- but the
    multiplicities, and the fact that all 24 carriers move at one speed in 4D, are what
    the isotropy actually rests on.
    """
    v4 = FCHC_24.vectors
    return VelocitySet(
        "fchc-24->3d",
        v4[:, :3].copy(),
        "FCHC projected along the 4th axis; 18 distinct directions, axes doubled",
    )


ALL_SETS = [
    CUBIC_6,
    FCC_12,
    BCC_8,
    CUBIC_FCC_18,
    ALL_26,
    project_fchc_to_3d(),
    FCHC_24,
]
