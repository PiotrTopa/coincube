"""Wetterich's 1+1-dimensional Thirring-type automaton, PRD 105, 074502.

Reference implementation. This is the model we must reproduce exactly before any
three-dimensional claim is taken seriously; it is the only fully worked example in the
literature where the fermionic quantum field theory and the automaton are known to be
identical.

Species order is Wetterich's, ``gamma = (R1, R2, L1, L2)``, so a configuration at one
site is written ``(n_R1, n_R2, n_L1, n_L2)`` (2111.06728 eq. 44 ff.).

The update is ``S_hat = S_int S_free`` (eq. 62):

  ``S_free``  right-movers R move one site in +x, left-movers L one site in -x (eq. 41).
  ``S_int``   at each site independently, if precisely one right-mover and precisely one
              left-mover are present, their colours are exchanged; otherwise nothing
              happens (sect. 4).

Two variants of the interaction, both of which Wetterich discusses:

  ``base``      only the exchange (1,0,0,1) <-> (0,1,1,0), eqs. 44-46. This is the
                scattering R1+L2 <-> R2+L1.
  ``extended``  additionally (1,0,1,0) <-> (0,1,0,1), eqs. 79-81, the crossing-related
                process R1+L1 <-> R2+L2. The two together give the Thirring/Gross-Neveu
                coupling of eq. 105D; either one alone halves it (eq. 929 discussion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.automaton import Automaton
from ..core.backend import array_module
from ..core.lattice import Lattice

R1, R2, L1, L2 = 0, 1, 2, 3
SPECIES_NAMES = ("R1", "R2", "L1", "L2")


@dataclass
class Wetterich1D(Automaton):
    """The 1+1D Thirring automaton on a periodic chain of ``n_sites`` sites."""

    n_sites: int
    extended: bool = True
    interact: bool = True

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("need at least 2 sites")
        self.lattice = Lattice(shape=(self.n_sites,), n_species=4)

    # -- the two factors of S_hat ---------------------------------------------

    def free_bits(self, n: np.ndarray) -> np.ndarray:
        """``S_free``: right-movers to +x, left-movers to -x."""
        xp = array_module(n)
        out = xp.empty_like(n)
        out[..., R1] = xp.roll(n[..., R1], +1, axis=-1)
        out[..., R2] = xp.roll(n[..., R2], +1, axis=-1)
        out[..., L1] = xp.roll(n[..., L1], -1, axis=-1)
        out[..., L2] = xp.roll(n[..., L2], -1, axis=-1)
        return out

    def interaction_mask(self, n: np.ndarray) -> np.ndarray:
        """Sites where the colour exchange fires. Shape ``(..., n_sites)``."""
        n_right = n[..., R1].astype(np.int8) + n[..., R2]
        n_left = n[..., L1].astype(np.int8) + n[..., L2]
        mask = (n_right == 1) & (n_left == 1)
        if not self.extended:
            # keep only the case where the right- and left-mover carry *different*
            # colours, i.e. (1001) and (0110) but not (1010) and (0101)
            mask &= n[..., R1] != n[..., L1]
        return mask

    def interact_bits(self, n: np.ndarray) -> np.ndarray:
        """``S_int``: exchange colours at the sites selected by the mask."""
        xp = array_module(n)
        out = n.copy()
        m = self.interaction_mask(n)
        out[..., R1] = xp.where(m, n[..., R2], n[..., R1])
        out[..., R2] = xp.where(m, n[..., R1], n[..., R2])
        out[..., L1] = xp.where(m, n[..., L2], n[..., L1])
        out[..., L2] = xp.where(m, n[..., L1], n[..., L2])
        return out

    # -- the full step ---------------------------------------------------------

    def step_bits(self, n: np.ndarray) -> np.ndarray:
        moved = self.free_bits(n)
        if not self.interact:
            return moved
        return self.interact_bits(moved)

    # -- observables used by the property tests --------------------------------

    @staticmethod
    def n_right(n: np.ndarray) -> np.ndarray:
        """Total number of right-movers. Conserved (property 1 of sect. 4)."""
        return n[..., R1].sum(axis=-1) + n[..., R2].sum(axis=-1)

    @staticmethod
    def n_left(n: np.ndarray) -> np.ndarray:
        return n[..., L1].sum(axis=-1) + n[..., L2].sum(axis=-1)

    @staticmethod
    def particle_hole(n: np.ndarray) -> np.ndarray:
        """The involution K: ``n -> 1 - n`` for every species (property 9)."""
        return ~n

    @staticmethod
    def exchange_colours(n: np.ndarray) -> np.ndarray:
        """The symmetry E: red <-> green everywhere (property 8)."""
        return n[..., [R2, R1, L2, L1]]

    @staticmethod
    def parity(n: np.ndarray) -> np.ndarray:
        """Reflection in x, which also exchanges right- and left-movers (property 6)."""
        return n[..., ::-1, :][..., [L1, L2, R1, R2]]
