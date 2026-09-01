"""Building legal block rules out of elementary processes.

A block rule is specified the way Wetterich specifies one: as a small set of *elementary
processes*, each a transposition of two block configurations, rather than as an opaque
permutation of ``2**block_bits`` states. Everything that makes such a rule legal is then
checkable combinatorially:

  - **R2 (unique jump).** A set of *disjoint* transpositions is an involution, hence a
    bijection, with no enumeration needed. Disjointness is the whole content of
    Wetterich's "Restrictions for cellular automata" (2203.14081): the failure mode he
    describes -- one incoming configuration being multiplied by a *sum* of outgoing
    Grassmann elements -- is exactly two processes sharing a configuration.
  - **R5 (particle-hole).** The process set must be closed under ``K: n -> 1 - n``,
    which on a configuration integer is ``c XOR (2**n_bits - 1)``.
  - **R6 (point group).** The process set must be closed under the symmetries of the
    block, which act on configurations by permuting bits.

The generator below therefore takes a seed process, closes it under the required
symmetries, and *rejects* the candidate if the closure is not disjoint. Rejection is the
common case and is not a bug: it is the constraint doing its job.

Particle number conservation is tracked but not required -- deliberately. It is the
hypothesis of the finite-ray theorem (ADR 0001), and rules that violate it are the point
of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def particle_hole(config: int, n_bits: int) -> int:
    """K: flip every occupation number."""
    return config ^ ((1 << n_bits) - 1)


def popcount(config: int) -> int:
    return bin(config).count("1")


def permute_bits(config: int, bit_map: np.ndarray) -> int:
    """Relabel bits: the bit at position ``i`` moves to position ``bit_map[i]``."""
    out = 0
    for i, j in enumerate(bit_map):
        if (config >> i) & 1:
            out |= 1 << int(j)
    return out


@dataclass(frozen=True)
class Process:
    """One elementary process: the transposition ``a <-> b``."""

    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("a process must exchange two distinct configurations")
        if self.a > self.b:  # canonical order so that sets deduplicate correctly
            lo, hi = self.b, self.a  # capture before mutating; assigning a first would
            object.__setattr__(self, "a", lo)  # make the read of self.a see the new value
            object.__setattr__(self, "b", hi)

    @property
    def configs(self) -> tuple[int, int]:
        return (self.a, self.b)

    def conserves_particle_number(self) -> bool:
        return popcount(self.a) == popcount(self.b)

    def describe(self, n_bits: int) -> str:
        fmt = lambda c: format(c, f"0{n_bits}b")[::-1]  # bit 0 leftmost
        return (
            f"{fmt(self.a)} <-> {fmt(self.b)}  "
            f"(N: {popcount(self.a)} -> {popcount(self.b)})"
        )


class NotDisjoint(ValueError):
    """The closed process set reuses a configuration, so it is not a bijection."""


@dataclass(frozen=True)
class BlockRule:
    """A validated set of disjoint processes on a block."""

    n_bits: int
    processes: tuple[Process, ...]

    @property
    def conserves_particle_number(self) -> bool:
        return all(p.conserves_particle_number() for p in self.processes)

    @property
    def is_particle_hole_symmetric(self) -> bool:
        want = {
            Process(particle_hole(p.a, self.n_bits), particle_hole(p.b, self.n_bits))
            for p in self.processes
        }
        return want == set(self.processes)

    def to_perm(self) -> np.ndarray:
        """The block permutation: identity except on the listed transpositions."""
        perm = np.arange(1 << self.n_bits, dtype=np.int64)
        for p in self.processes:
            perm[p.a], perm[p.b] = p.b, p.a
        return perm

    def describe(self) -> str:
        head = (
            f"BlockRule on {self.n_bits} bits, {len(self.processes)} process(es), "
            f"{'conserving' if self.conserves_particle_number else 'NON-conserving'}, "
            f"{'PH-symmetric' if self.is_particle_hole_symmetric else 'NOT PH-symmetric'}"
        )
        return "\n".join([head] + ["  " + p.describe(self.n_bits) for p in self.processes])


def close(
    seeds: list[Process],
    n_bits: int,
    bit_maps: list[np.ndarray] | None = None,
    particle_hole_closure: bool = True,
) -> BlockRule:
    """Close a set of seed processes under K and the given bit relabellings.

    Raises :class:`NotDisjoint` if the closure touches any configuration twice, which is
    the R2 violation described above.
    """
    frontier = list(seeds)
    found: set[Process] = set()

    while frontier:
        p = frontier.pop()
        if p in found:
            continue
        found.add(p)
        if particle_hole_closure:
            q = Process(particle_hole(p.a, n_bits), particle_hole(p.b, n_bits))
            if q not in found:
                frontier.append(q)
        for m in bit_maps or []:
            q = Process(permute_bits(p.a, m), permute_bits(p.b, m))
            if q not in found:
                frontier.append(q)

    touched: dict[int, Process] = {}
    for p in sorted(found, key=lambda p: (p.a, p.b)):
        for c in p.configs:
            if c in touched:
                raise NotDisjoint(
                    f"configuration {format(c, f'0{n_bits}b')} is used by both "
                    f"[{touched[c].describe(n_bits)}] and [{p.describe(n_bits)}]; "
                    "the closed process set is not a bijection (R2)"
                )
            touched[c] = p

    return BlockRule(n_bits=n_bits, processes=tuple(sorted(found, key=lambda p: (p.a, p.b))))


def enumerate_rules(
    n_bits: int,
    max_seeds: int = 1,
    bit_maps: list[np.ndarray] | None = None,
    require_non_conserving: bool = False,
    require_nontrivial: bool = True,
) -> list[BlockRule]:
    """Every rule obtainable by closing up to ``max_seeds`` seed transpositions.

    This is a *complete* enumeration of the sparse-process rules on the block, which is
    the class Wetterich works in ("only four out of the 28 possible bilinears for eight
    Grassmann variables are used"). It is not a complete enumeration of all block
    bijections -- that is ``(2**n_bits)!`` and is not a reachable target for any block
    of interest.
    """
    n_cfg = 1 << n_bits
    all_pairs = [Process(a, b) for a, b in combinations(range(n_cfg), 2)]

    rules: dict[tuple[Process, ...], BlockRule] = {}
    for n_seed in range(1, max_seeds + 1):
        for seeds in combinations(all_pairs, n_seed):
            try:
                rule = close(list(seeds), n_bits, bit_maps)
            except NotDisjoint:
                continue
            if require_nontrivial and not rule.processes:
                continue
            if require_non_conserving and rule.conserves_particle_number:
                continue
            rules[rule.processes] = rule
    return list(rules.values())


def block_bit_map_from_site_permutation(
    site_perm: np.ndarray, n_species: int, species_perm: np.ndarray | None = None
) -> np.ndarray:
    """Build the bit relabelling induced by permuting sites (and optionally species).

    Bit layout matches ``Lattice.bit_index``: ``bit = site * n_species + species``.
    """
    n_sites = len(site_perm)
    if species_perm is None:
        species_perm = np.arange(n_species)
    out = np.empty(n_sites * n_species, dtype=np.int64)
    for s in range(n_sites):
        for a in range(n_species):
            out[s * n_species + a] = int(site_perm[s]) * n_species + int(species_perm[a])
    return out
