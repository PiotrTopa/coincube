"""The Automaton base class and the unique-jump verification (rule R2).

An automaton is defined by its action on occupation-number arrays. Everything else --
the step evolution operator as a permutation of configurations, the one-particle sector,
the checks -- is derived from that single method, so a model cannot accidentally declare
a property it does not have.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .lattice import Lattice


class Automaton(abc.ABC):
    """Base class. Subclasses implement :meth:`step_bits` and nothing else is required.

    ``step_bits`` must be:
      - vectorised over a leading batch axis (R2 verification applies it to every
        configuration at once);
      - pure, i.e. it must not mutate its argument.
    """

    lattice: Lattice

    #: Number of elementary sub-steps combined into one call of :meth:`step_bits`.
    #: Branch B cycles report the cycle length here so that dispersion is reported per
    #: elementary step rather than per cycle.
    n_substeps: int = 1

    @abc.abstractmethod
    def step_bits(self, n: np.ndarray) -> np.ndarray:
        """``(..., n_sites, n_species)`` bool array -> the updated array."""

    # -- derived: the step evolution operator ----------------------------------

    def permutation(self) -> np.ndarray:
        """``S_hat`` as a permutation array: ``perm[rho]`` is the successor of ``rho``.

        This is the full step evolution operator, stripped of the signs (which are a
        gauge choice, cf. 2211.09002 sect. 5 "Signs of Grassmann basis elements"). It
        exists only for lattices small enough to enumerate.
        """
        lat = self.lattice
        configs = lat.all_configs_array()
        out = self.step_bits(configs)
        if out.shape != configs.shape:
            raise ValueError(f"step_bits changed shape: {configs.shape} -> {out.shape}")
        idx = np.arange(lat.n_bits)
        # array_to_config, but vectorised and safe because n_bits <= 24 here
        return (out.reshape(-1, lat.n_bits) << idx).sum(axis=1).astype(np.int64)

    def one_particle_map(self) -> np.ndarray:
        """The induced map on one-particle states, as an index array.

        ``opm[i]`` is the index of the successor of one-particle state ``i`` in
        ``lattice.one_particle_states`` order.

        Raises if the automaton does not map one-particle states to one-particle states,
        i.e. if particle number is not conserved in this sector. That is a legal
        automaton (see docs/01-ca-rules.md, "deliberately NOT required") but it needs a
        different analyser.
        """
        lat = self.lattice
        n = np.zeros((lat.n_bits, lat.n_sites, lat.n_species), dtype=bool)
        for i, (x, a) in enumerate(lat.one_particle_states):
            n[i, x, a] = True
        out = self.step_bits(n)
        counts = out.reshape(lat.n_bits, -1).sum(axis=1)
        if not np.all(counts == 1):
            bad = int(np.flatnonzero(counts != 1)[0])
            raise NotOneParticleConserving(
                f"one-particle state {lat.one_particle_states[bad]} mapped to a state "
                f"with {int(counts[bad])} particles; particle number is not conserved "
                "in the one-particle sector"
            )
        flat = out.reshape(lat.n_bits, -1)
        return np.argmax(flat, axis=1).astype(np.int64)


class NotOneParticleConserving(RuntimeError):
    """Raised when the one-particle sector is not invariant under the update."""


@dataclass(frozen=True)
class UniqueJumpReport:
    """Result of checking rule R2."""

    is_unique_jump: bool
    n_configs: int
    n_images: int
    collisions: list[tuple[int, int, int]]  # (rho_a, rho_b, shared image)
    cycle_lengths: dict[int, int]  # cycle length -> how many cycles of that length

    @property
    def order(self) -> int:
        """Smallest ``k`` with ``S_hat**k = 1``; ``S_hat`` has order lcm(cycle lengths)."""
        out = 1
        for L in self.cycle_lengths:
            out = np.lcm(out, L)
        return int(out)

    def __str__(self) -> str:
        if not self.is_unique_jump:
            ex = self.collisions[:3]
            return (
                f"NOT a unique jump matrix: {self.n_configs} configurations map onto "
                f"only {self.n_images} images. Examples of collisions: {ex}"
            )
        spectrum = ", ".join(
            f"{n} cycle(s) of length {L}" for L, n in sorted(self.cycle_lengths.items())
        )
        return (
            f"unique jump matrix on {self.n_configs} configurations; "
            f"order {self.order}; cycle structure: {spectrum}"
        )


def check_unique_jump(automaton: Automaton) -> UniqueJumpReport:
    """Verify rule R2 by explicit bijectivity check on the configuration permutation.

    This is deliberately brute force. The unique-jump property is the one thing the
    whole construction rests on, so it is established by enumeration and never by
    argument.
    """
    perm = automaton.permutation()
    n_configs = perm.size
    uniq, first_idx, counts = np.unique(perm, return_index=True, return_counts=True)

    collisions: list[tuple[int, int, int]] = []
    if uniq.size != n_configs:
        for img in uniq[counts > 1][:8]:
            pre = np.flatnonzero(perm == img)
            collisions.append((int(pre[0]), int(pre[1]), int(img)))
        return UniqueJumpReport(False, n_configs, int(uniq.size), collisions, {})

    return UniqueJumpReport(
        is_unique_jump=True,
        n_configs=n_configs,
        n_images=int(uniq.size),
        collisions=[],
        cycle_lengths=_cycle_structure(perm),
    )


def _cycle_structure(perm: np.ndarray) -> dict[int, int]:
    """Histogram of cycle lengths of a permutation.

    The cycle structure is not cosmetic: the eigenvalues of a permutation matrix are
    exactly the roots of unity ``exp(2 pi i m / L)`` over its cycle lengths ``L``, so
    this histogram *is* the energy spectrum of the automaton
    (2203.14081, sect. "Energy spectrum").
    """
    seen = np.zeros(perm.size, dtype=bool)
    hist: dict[int, int] = {}
    for start in range(perm.size):
        if seen[start]:
            continue
        L, j = 0, start
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
            L += 1
        hist[L] = hist.get(L, 0) + 1
    return hist
