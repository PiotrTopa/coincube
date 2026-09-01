"""Generic legal automata: species shifts, Margolus block rules, and rule cycles.

These are the constructions the search will draw from, and they are also the ammunition
used to attack the finite-ray theorem in ``tests/test_finite_ray_theorem.py``. Each one
satisfies R1-R3 by construction, and each carries a method that *verifies* rather than
asserts the unique-jump property.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.automaton import Automaton
from ..core.backend import array_module
from ..core.lattice import Lattice


def _as_grid(lat: Lattice, n: np.ndarray) -> np.ndarray:
    """``(..., n_sites, n_species)`` -> ``(..., *shape, n_species)``."""
    return n.reshape(*n.shape[:-2], *lat.shape, lat.n_species)


def _as_flat(lat: Lattice, g: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_as_grid`."""
    return g.reshape(*g.shape[: -(lat.dim + 1)], lat.n_sites, lat.n_species)


@dataclass
class SpeciesShift(Automaton):
    """Species ``alpha`` moves by ``velocities[alpha]`` and becomes ``sigma[alpha]``.

    This is the most general *purely propagating* homogeneous rule: it is a unique jump
    for any velocity set and any species permutation sigma, since it is a relabelling of
    sites and species with no dependence on occupation. It is the free part ``S_free``
    of every model here, and Wetterich's eq. 41 is the case ``d=1``, ``sigma=id``,
    ``velocities = (+1, +1, -1, -1)``.
    """

    lattice: Lattice
    velocities: np.ndarray  # (n_species, dim), integer lattice displacements
    sigma: np.ndarray | None = None  # species permutation, default identity

    def __post_init__(self) -> None:
        self.velocities = np.asarray(self.velocities, dtype=np.int64)
        if self.velocities.shape != (self.lattice.n_species, self.lattice.dim):
            raise ValueError(
                f"velocities must be {(self.lattice.n_species, self.lattice.dim)}, "
                f"got {self.velocities.shape}"
            )
        if self.sigma is None:
            self.sigma = np.arange(self.lattice.n_species, dtype=np.int64)
        self.sigma = np.asarray(self.sigma, dtype=np.int64)
        if sorted(self.sigma.tolist()) != list(range(self.lattice.n_species)):
            raise ValueError(f"sigma must be a permutation, got {self.sigma!r}")

    def step_bits(self, n: np.ndarray) -> np.ndarray:
        xp = array_module(n)
        lat = self.lattice
        g = _as_grid(lat, n)
        out = xp.empty_like(g)
        batch = g.ndim - lat.dim - 1
        axes = tuple(range(batch, batch + lat.dim))
        for a in range(lat.n_species):
            shifted = xp.roll(g[..., a], tuple(int(v) for v in self.velocities[a]), axis=axes)
            out[..., int(self.sigma[a])] = shifted
        return _as_flat(lat, out)


@dataclass
class BlockAutomaton(Automaton):
    """Margolus-style block rule: partition into disjoint blocks, permute within each.

    ``block_perm`` is a permutation of the ``2 ** (prod(block_shape) * n_species)``
    states of one block; because the blocks are disjoint and the map is a bijection on
    each, the global step is a bijection -- this is the standard way to guarantee R2
    without enumerating the whole configuration space.

    ``origin`` shifts the partition, which is how consecutive steps are coupled
    (2211.09002, "Updating with shifted cells"; 2203.14081, "Updating by shifted
    blocks"). With a fixed origin the automaton decouples into independent blocks and
    goes nowhere, which is exactly Wetterich's "somewhat boring automaton".
    """

    lattice: Lattice
    block_shape: tuple[int, ...]
    block_perm: np.ndarray
    origin: tuple[int, ...] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        lat = self.lattice
        if len(self.block_shape) != lat.dim:
            raise ValueError("block_shape must have one entry per space dimension")
        for s, b in zip(lat.shape, self.block_shape):
            if s % b:
                raise ValueError(f"lattice shape {lat.shape} not divisible by block {self.block_shape}")
        if self.origin is None:
            self.origin = tuple(0 for _ in lat.shape)
        self.block_perm = np.asarray(self.block_perm, dtype=np.int64)
        if self.block_perm.size != 1 << self.block_bits:
            raise ValueError(
                f"block_perm has {self.block_perm.size} entries, expected {1 << self.block_bits}"
            )

    @property
    def block_sites(self) -> int:
        return int(np.prod(self.block_shape))

    @property
    def block_bits(self) -> int:
        return self.block_sites * self.lattice.n_species

    def verify_block_bijection(self) -> bool:
        """R2 for one block; disjointness then gives R2 globally."""
        return len(np.unique(self.block_perm)) == self.block_perm.size

    def step_bits(self, n: np.ndarray) -> np.ndarray:
        xp = array_module(n)
        lat = self.lattice
        g = _as_grid(lat, n)
        batch_shape = g.shape[: g.ndim - lat.dim - 1]
        nb = g.ndim - lat.dim - 1
        axes = tuple(range(nb, nb + lat.dim))

        # align the partition so blocks start at index 0
        g = xp.roll(g, tuple(-int(o) for o in self.origin), axis=axes)

        # (..., nblk_0, b_0, nblk_1, b_1, ..., n_species)
        inter: list[int] = []
        for s, b in zip(lat.shape, self.block_shape):
            inter += [s // b, b]
        g = g.reshape(*batch_shape, *inter, lat.n_species)

        # move all the block-cell axes to the front of the trailing group
        n_blk_axes = list(range(nb, nb + 2 * lat.dim, 2))
        in_blk_axes = list(range(nb + 1, nb + 2 * lat.dim, 2))
        perm = list(range(nb)) + n_blk_axes + in_blk_axes + [g.ndim - 1]
        g = xp.transpose(g, perm)

        blk_grid_shape = g.shape[nb : nb + lat.dim]
        flat = g.reshape(-1, self.block_bits)

        weights = xp.asarray(np.arange(self.block_bits))
        idx = (flat.astype(xp.int64) << weights).sum(axis=1)
        idx = xp.asarray(self.block_perm)[idx]
        flat = ((idx[:, None] >> weights[None, :]) & 1).astype(bool)

        g = flat.reshape(*batch_shape, *blk_grid_shape, *self.block_shape, lat.n_species)
        g = xp.transpose(g, _invert_perm(perm))
        g = g.reshape(*batch_shape, *lat.shape, lat.n_species)
        g = xp.roll(g, tuple(int(o) for o in self.origin), axis=axes)
        return _as_flat(lat, g)


def _invert_perm(perm: list[int]) -> list[int]:
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return out


@dataclass
class RuleCycle(Automaton):
    """Branch B: a finite periodic sequence of legal rules, applied in order.

    ``step_bits`` applies ``steps[0]`` first, so as a matrix
    ``S_cycle = S_{n-1} ... S_1 S_0``. A product of unique-jump matrices is unique-jump,
    so the cycle satisfies R2 whenever its factors do -- which is precisely why the
    finite-ray theorem is not evaded by rotating the rule between steps.
    """

    steps: tuple[Automaton, ...]
    lattice: Lattice = field(init=False)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a cycle needs at least one step")
        lats = {(s.lattice.shape, s.lattice.n_species) for s in self.steps}
        if len(lats) != 1:
            raise ValueError(f"all steps must share a lattice, got {lats}")
        self.lattice = self.steps[0].lattice
        self.n_substeps = len(self.steps)

    def step_bits(self, n: np.ndarray) -> np.ndarray:
        for s in self.steps:
            n = s.step_bits(n)
        return n


# -- helpers for building block permutations ---------------------------------------


def identity_block_perm(n_bits: int) -> np.ndarray:
    return np.arange(1 << n_bits, dtype=np.int64)


def random_block_perm(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random bijection on the block. Legal R2, wildly illegal R6.

    Useful only as an adversarial probe: if even a maximally structureless block rule
    cannot curve the dispersion, the theorem is not an artefact of the rules we like.
    """
    return rng.permutation(1 << n_bits).astype(np.int64)


def particle_number_conserving_block_perm(
    n_bits: int, rng: np.random.Generator
) -> np.ndarray:
    """A random bijection that permutes states only within fixed-population sectors."""
    popcount = np.array([bin(i).count("1") for i in range(1 << n_bits)])
    perm = np.empty(1 << n_bits, dtype=np.int64)
    for p in range(n_bits + 1):
        idx = np.flatnonzero(popcount == p)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm
