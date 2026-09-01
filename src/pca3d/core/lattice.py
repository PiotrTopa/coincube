"""Lattice geometry and the bit <-> configuration encoding.

A state of the automaton is a set of occupation numbers

    n_alpha(x) in {0, 1},   x a lattice site,   alpha a species index

which we hold either as a boolean array of shape ``(..., n_sites, n_species)`` (the
working representation, vectorised over a leading batch axis) or as a single Python
integer whose bit ``site * n_species + species`` is ``n_alpha(x)`` (the representation
used when we need to talk about configurations tau, rho as indices of the step
evolution operator).

Rule R4 of docs/01-ca-rules.md: one bit per species per site, single fermions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """A periodic rectangular lattice carrying ``n_species`` bits per site.

    ``shape`` is the extent in each spatial direction, so ``len(shape)`` is the number
    of *space* dimensions; time is the update index and is not part of the lattice.
    """

    shape: tuple[int, ...]
    n_species: int

    def __post_init__(self) -> None:
        if not self.shape or any(s < 1 for s in self.shape):
            raise ValueError(f"shape must be non-empty and positive, got {self.shape!r}")
        if self.n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {self.n_species}")

    # -- basic counts ----------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.shape)

    @cached_property
    def n_sites(self) -> int:
        return int(np.prod(self.shape))

    @cached_property
    def n_bits(self) -> int:
        """Total bits M. The configuration space has 2**M elements."""
        return self.n_sites * self.n_species

    @cached_property
    def n_configs(self) -> int:
        return 1 << self.n_bits

    # -- site indexing ---------------------------------------------------------

    def site_index(self, coords) -> int:
        """Flatten lattice coordinates to a site index (C order, wrapping)."""
        c = np.asarray(coords, dtype=np.int64) % np.asarray(self.shape, dtype=np.int64)
        return int(np.ravel_multi_index(tuple(c), self.shape))

    def site_coords(self, index: int) -> np.ndarray:
        return np.array(np.unravel_index(int(index), self.shape), dtype=np.int64)

    @cached_property
    def all_coords(self) -> np.ndarray:
        """``(n_sites, dim)`` array of the coordinates of every site, in index order."""
        grids = np.indices(self.shape).reshape(self.dim, -1)
        return grids.T.astype(np.int64)

    def displacement(self, frm: int, to: int) -> np.ndarray:
        """Minimal-image displacement ``to - frm``.

        Components are folded into ``(-L/2, L/2]``. This is the physically meaningful
        displacement for a *local* rule (R1); it is meaningless for a rule whose
        neighbourhood is comparable to the lattice, which is why callers that depend on
        it must also assert locality.
        """
        d = self.site_coords(to) - self.site_coords(frm)
        L = np.asarray(self.shape, dtype=np.int64)
        d = (d + L // 2) % L - L // 2
        return d

    # -- bit <-> array encoding ------------------------------------------------

    def bit_index(self, site: int, species: int) -> int:
        return int(site) * self.n_species + int(species)

    def config_to_array(self, config) -> np.ndarray:
        """Integer configuration(s) -> bool array ``(..., n_sites, n_species)``."""
        cfg = np.asarray(config, dtype=object)
        flat = np.array(
            [[(int(c) >> b) & 1 for b in range(self.n_bits)] for c in cfg.reshape(-1)],
            dtype=bool,
        )
        return flat.reshape(*cfg.shape, self.n_sites, self.n_species)

    def array_to_config(self, arr: np.ndarray) -> np.ndarray:
        """Bool array ``(..., n_sites, n_species)`` -> integer configuration(s).

        Returns an object-dtype array of Python ints so that ``n_bits > 63`` does not
        silently overflow.
        """
        a = np.asarray(arr, dtype=bool)
        if a.shape[-2:] != (self.n_sites, self.n_species):
            raise ValueError(
                f"expected trailing shape {(self.n_sites, self.n_species)}, got {a.shape[-2:]}"
            )
        batch = a.shape[:-2]
        flat = a.reshape(-1, self.n_bits)
        out = np.empty(flat.shape[0], dtype=object)
        for i, row in enumerate(flat):
            v = 0
            for b in np.flatnonzero(row):
                v |= 1 << int(b)
            out[i] = v
        return out.reshape(batch)

    def all_configs_array(self) -> np.ndarray:
        """Every configuration as a bool array ``(2**M, n_sites, n_species)``.

        Only sane for small M; the caller is responsible for not asking for 2**64.
        """
        if self.n_bits > 24:
            raise ValueError(
                f"refusing to enumerate 2**{self.n_bits} configurations; "
                "use the one-particle sector or Monte Carlo over initial conditions"
            )
        idx = np.arange(1 << self.n_bits, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(self.n_bits)[None, :]) & 1).astype(bool)
        return bits.reshape(-1, self.n_sites, self.n_species)

    # -- one-particle sector ---------------------------------------------------

    @cached_property
    def one_particle_states(self) -> list[tuple[int, int]]:
        """The ``(site, species)`` pairs, in a fixed canonical order."""
        return [(x, a) for x in range(self.n_sites) for a in range(self.n_species)]

    @property
    def n_one_particle(self) -> int:
        return self.n_bits
