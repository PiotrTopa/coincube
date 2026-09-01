"""Signed (fermionic) lift of bit automata: the amplitude level.

Everything so far measured densities, which are blind to signs and therefore to
interference and spin. This module lifts the automaton to the fermionic Fock space,
where the step operator is a *signed* permutation of occupation-number basis states.

The two gauge choices, and why they are physics
-----------------------------------------------
Wetterich's discrete Z2 gauge freedom ("the choice of signs ... permits a description in
terms of a smooth wave function") appears here as two concrete choices:

1. **The Jordan-Wigner mode ordering.** Configs are stored site-major
   (``bit = site*n_species + species``, matching the rest of the codebase), but the
   fermionic ordering is a separate *rank* assignment. Site-major ranks are the naive
   choice and they are the WRONG gauge: a system fermion hopping one site crosses the
   environment mode between the sites and acquires a tau-dependent sign, so even free
   streaming decoheres (measured: |G| collapsed to MC noise while the permutation part
   was perfect). **Species-major ranks** (all system modes first, all environment modes
   above) are the smooth gauge: vacuum rearrangements cancel between the defect and
   vacuum trajectories, and a hopping defect crosses nothing. This is the operational
   form of Wetterich's remark that R-movers and L-movers get their own Grassmann
   ordering blocks.

2. **The per-transition operator decomposition.** A set-based lift (annihilate the
   disappearing modes, create the appearing ones) is NOT enough: it cannot see that two
   fermions exchanging sites *cross*, which in one dimension contributes a physical
   (-1). Each transition is therefore decomposed by the rule's *semantics* into
   (annihilated pairs) + (a mode map phi on the surviving fermions) + (created pairs).
   The mode map is lifted with the **Gaussian sign** -- the parity of phi restricted to
   occupied modes in rank order, which is what a genuine free-fermion (matchgate)
   circuit produces -- and the pairs with the ordered-pair convention and global JW
   strings. With anything less, even free streaming decoheres: measured, twice, with
   two different wrong conventions, before this one made it exactly coherent.

Legality gate discovered while building this: a block rule must conserve **fermion
parity** or disjoint block operators fail to commute and no fermionic lift exists
(Wetterich's local factors contain only even Grassmann monomials for the same reason).
This prunes the 9216 conditional-propagation rules to 256 legal ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# -- popcount helpers --------------------------------------------------------------


def _popcount(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64).copy()
    count = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        count += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return count


# -- mode ranking -------------------------------------------------------------------


def species_major_ranks(n_sites: int, n_species: int = 2) -> np.ndarray:
    """rank[mode] with all species-0 modes first (by site), then species-1, ...

    ``mode = site * n_species + species`` -> ``rank = species * n_sites + site``.
    """
    ranks = np.empty(n_sites * n_species, dtype=np.int64)
    for site in range(n_sites):
        for sp in range(n_species):
            ranks[site * n_species + sp] = sp * n_sites + site
    return ranks


def site_major_ranks(n_sites: int, n_species: int = 2) -> np.ndarray:
    """The naive (non-smooth) gauge, kept for tests that demonstrate its failure."""
    return np.arange(n_sites * n_species, dtype=np.int64)


class JW:
    """Jordan-Wigner machinery for an arbitrary mode->rank assignment."""

    def __init__(self, ranks: np.ndarray):
        self.ranks = np.asarray(ranks, dtype=np.int64)
        M = len(self.ranks)
        #: below_mask[m] = bitmask (in MODE space) of modes with strictly lower rank
        self.below_mask = np.zeros(M, dtype=np.int64)
        for m in range(M):
            for m2 in range(M):
                if self.ranks[m2] < self.ranks[m]:
                    self.below_mask[m] |= np.int64(1) << m2

    def jw_sign(self, configs: np.ndarray, mode: int) -> np.ndarray:
        """(-1)^{# occupied modes of lower rank}."""
        return 1 - 2 * (_popcount(configs & self.below_mask[mode]) & 1)

    def create(self, configs: np.ndarray, signs: np.ndarray, mode: int):
        bit = np.int64(1) << mode
        occupied = (configs & bit) != 0
        s = self.jw_sign(configs, mode)
        return (
            np.where(occupied, configs, configs | bit),
            np.where(occupied, 0, signs * s),
        )

    def annihilate(self, configs: np.ndarray, signs: np.ndarray, mode: int):
        bit = np.int64(1) << mode
        occupied = (configs & bit) != 0
        s = self.jw_sign(configs, mode)
        return (
            np.where(occupied, configs & ~bit, configs),
            np.where(occupied, signs * s, 0),
        )

    def mode_permutation_sign(self, config: int, pi: np.ndarray) -> int:
        """Parity of the permutation induced on occupied modes, in rank order."""
        occ = [m for m in range(len(pi)) if (config >> m) & 1]
        occ.sort(key=lambda m: self.ranks[m])
        image = [self.ranks[int(pi[m])] for m in occ]
        sign = 1
        for i in range(len(image)):
            for j in range(i + 1, len(image)):
                if image[i] > image[j]:
                    sign = -sign
        return sign


# -- the signed automaton -----------------------------------------------------------


@dataclass
class SignedBlockCycle:
    """A 1D shifted-block automaton lifted to the fermionic Fock space.

    ``block_perm`` is the 16-entry bijection on a 2-site, 2-species block. The lift
    applies, per block, the canonical normal-ordered operator for the transition of
    that block's 4 modes; sub-step origin alternates 0/1, with origin 1 realised as
    ``T† U T`` (T = one-site ring translation, with its exact JW sign under the chosen
    rank gauge). Fermion parity of the block rule is required.
    """

    n_sites: int
    block_perm: np.ndarray
    ranks: np.ndarray | None = None  # default: species-major (the smooth gauge)
    boundary: str = "open"  # "open" (bulk physics, no JW seam) or "ring"
    #: does the rule's semantics move surviving fermions across the block diagonal?
    #: True for the conditional-propagation class (system streams; env streams when its
    #: content is conserved). Determines crossing signs on doubly-occupied channels.
    swap_semantics: bool = True
    jw: JW = field(init=False)

    def __post_init__(self) -> None:
        if self.n_sites % 2:
            raise ValueError("ring must have an even number of sites")
        if 2 * self.n_sites > 62:
            raise ValueError(
                f"{2 * self.n_sites} modes exceed the 62-bit packing limit of the "
                "int64 fast path; shifts past bit 63 corrupt configs SILENTLY (this "
                "produced an all-zero propagator at L=64 and contaminated the "
                "|dx| > 7 region of one L=48 survey before it was caught)"
            )
        pc = np.array([bin(i).count("1") for i in range(16)])
        if np.any((pc[self.block_perm] & 1) != (pc & 1)):
            raise ValueError(
                "block rule does not conserve fermion parity; disjoint block operators "
                "would not commute and no fermionic lift exists (cf. Wetterich's "
                "even-Grassmann requirement)"
            )
        if self.ranks is None:
            self.ranks = species_major_ranks(self.n_sites)
        self.jw = JW(self.ranks)

    @property
    def n_modes(self) -> int:
        return 2 * self.n_sites

    # -- block application, canonical normal-ordered signs ----------------------

    def _blocks_for_origin(self, origin: int) -> list[list[int]]:
        """Mode quadruples for each block of the given origin.

        Ring: origin-1 wraps (handled via T-conjugation in :meth:`substep`).
        Open: origin-1 pairs sites (1,2), (3,4), ..., leaving sites 0 and L-1 idle.
        The open chain is the smooth gauge's natural home -- there is no seam, so no
        Jordan-Wigner boundary factor exists to decohere bulk amplitudes. (On the ring
        the translation sign depends on total species parity, which differs by one
        between defect and vacuum trajectories: every origin-1 sub-step then flips the
        relative sign whenever site 0 is occupied, and the propagator collapses. This
        was measured before it was understood.)
        """
        pairs = []
        if origin == 0:
            sites = [(2 * b, 2 * b + 1) for b in range(self.n_sites // 2)]
        else:
            sites = [(2 * b + 1, 2 * b + 2) for b in range((self.n_sites - 2) // 2)]
        for s0, s1 in sites:
            pairs.append([2 * s0, 2 * s0 + 1, 2 * s1, 2 * s1 + 1])
        return pairs

    def _block_modes(self, b: int) -> list[int]:
        """Global modes of origin-0 block b, in local order (psi, phi, psi', phi')."""
        s0, s1 = 2 * b, 2 * b + 1
        return [2 * s0, 2 * s0 + 1, 2 * s1, 2 * s1 + 1]

    def _apply_blocks(self, configs: np.ndarray, signs: np.ndarray, origin: int = 0):
        out_c, out_s = configs, signs
        for modes in self._blocks_for_origin(origin):
            nib = np.zeros(len(out_c), dtype=np.int64)
            for i, m in enumerate(modes):
                nib |= ((out_c >> m) & 1) << i
            for val in range(16):
                target = int(self.block_perm[val])
                # NOTE: config-identity transitions (target == val) must NOT be
                # skipped: a doubly occupied channel that swaps is config-identical
                # yet carries the physical crossing sign -1. Skipping them silently
                # broke Gaussianity of free streaming (measured as |G| decaying by
                # ~0.65/step while a Gaussian circuit is exactly coherent).
                sel = nib == val
                if not np.any(sel):
                    continue
                c_sub, s_sub = out_c[sel], out_s[sel]
                pairs_gone, phi_pairs, pairs_born = self._decompose(modes, val, target)
                # 1) annihilate genuinely destroyed fermions (descending rank)
                for m in sorted(pairs_gone, key=lambda m: -self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.annihilate(c_sub, s_sub, m)
                # 2) mode map on survivors: Gaussian sign = parity of phi on occupied
                #    modes in rank order (all occupied here by construction), plus the
                #    occupancy relabelling
                sign_phi = 1
                srcs = sorted((m for m, _ in phi_pairs), key=lambda m: self.jw.ranks[m])
                images = [self.jw.ranks[dict(phi_pairs)[m]] for m in srcs]
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        if images[i] > images[j]:
                            sign_phi = -sign_phi
                for m, _ in phi_pairs:
                    c_sub = c_sub & ~(np.int64(1) << m)
                for m, m2 in phi_pairs:
                    c_sub = c_sub | (np.int64(1) << m2)
                s_sub = s_sub * sign_phi
                # 3) create genuinely new fermions (ascending rank, global JW)
                for m in sorted(pairs_born, key=lambda m: self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.create(c_sub, s_sub, m)
                out_c = out_c.copy(); out_s = out_s.copy()
                out_c[sel], out_s[sel] = c_sub, s_sub
        return out_c, out_s

    def _decompose(self, modes: list[int], val: int, target: int):
        """Split a block transition into destroyed pairs, a mode map, created pairs.

        Species channels are handled independently (species = local index parity):
        if a channel conserves its occupation count it is a mode map -- the identity,
        or the cross-block swap when ``swap_semantics`` and the content moved (or is
        doubly occupied, where crossing is invisible in the sets but physical); if the
        count changes it is a pair creation/annihilation on that channel's modes.
        """
        pairs_gone: list[int] = []
        pairs_born: list[int] = []
        phi_pairs: list[tuple[int, int]] = []  # (src_mode, dst_mode), occupied sources
        for ch in range(2):  # local indices: ch, ch+2 are this species' two modes
            i0, i1 = ch, ch + 2
            m0, m1 = modes[i0], modes[i1]
            in_bits = ((val >> i0) & 1, (val >> i1) & 1)
            out_bits = ((target >> i0) & 1, (target >> i1) & 1)
            n_in, n_out = sum(in_bits), sum(out_bits)
            if n_in != n_out:  # pair channel
                for i, m in ((i0, m0), (i1, m1)):
                    if (val >> i) & 1 and not (target >> i) & 1:
                        pairs_gone.append(m)
                    if (target >> i) & 1 and not (val >> i) & 1:
                        pairs_born.append(m)
                continue
            # conserved channel: mode map
            moved = in_bits != out_bits
            crossing = self.swap_semantics and (moved or in_bits == (1, 1))
            if crossing:
                if in_bits[0]:
                    phi_pairs.append((m0, m1))
                if in_bits[1]:
                    phi_pairs.append((m1, m0))
            else:
                if in_bits[0]:
                    phi_pairs.append((m0, m0))
                if in_bits[1]:
                    phi_pairs.append((m1, m1))
        return pairs_gone, phi_pairs, pairs_born

    # -- translation with exact JW sign ----------------------------------------

    def _translation_pi(self) -> np.ndarray:
        """T: site s -> s-1, i.e. mode m -> m-2 cyclically (in mode space)."""
        M = self.n_modes
        return np.array([(m - 2) % M for m in range(M)], dtype=np.int64)

    def _translation_sign(self, configs: np.ndarray) -> np.ndarray:
        """Vectorised sign of T under species-major ranks.

        In the species-major gauge T shifts each species' contiguous rank block
        cyclically by one site, so the sign factorises over species:
        ``prod_sp (-1)^(n_wrap_sp * (N_sp - n_wrap_sp))`` with ``n_wrap_sp`` the
        occupation of that species at site 0. Verified against the brute-force
        permutation parity in the tests; for non-species-major ranks the brute force
        is used instead.
        """
        L, M = self.n_sites, self.n_modes
        default = species_major_ranks(self.n_sites)
        if not np.array_equal(self.ranks, default):
            return np.array(
                [self.jw.mode_permutation_sign(int(c), self._translation_pi()) for c in configs],
                dtype=np.int64,
            )
        sign = np.ones(len(configs), dtype=np.int64)
        for sp in range(2):
            sp_mask = np.int64(0)
            for site in range(L):
                sp_mask |= np.int64(1) << (2 * site + sp)
            n_sp = _popcount(configs & sp_mask)
            wrapped = ((configs >> sp) & 1).astype(np.int64)  # site 0, this species
            sign *= 1 - 2 * ((wrapped * (n_sp - wrapped)) & 1)
        return sign

    def _translate(self, configs: np.ndarray, signs: np.ndarray, back: bool):
        M = self.n_modes
        mask = (np.int64(1) << M) - 1
        if not back:
            s = self._translation_sign(configs)
            new = ((configs >> 2) | (configs << (M - 2))) & mask
        else:
            new = ((configs << 2) | (configs >> (M - 2))) & mask
            s = self._translation_sign(new)
        return new, signs * s

    # -- public API -------------------------------------------------------------

    def substep(self, configs: np.ndarray, signs: np.ndarray, origin: int):
        if origin == 0:
            return self._apply_blocks(configs, signs, origin=0)
        if self.boundary == "open":
            return self._apply_blocks(configs, signs, origin=1)
        c, s = self._translate(configs, signs, back=False)
        c, s = self._apply_blocks(c, s, origin=0)
        return self._translate(c, s, back=True)

    def evolve(self, configs: np.ndarray, signs: np.ndarray, n_substeps: int, start: int = 0):
        c, s = configs, signs
        for t in range(n_substeps):
            c, s = self.substep(c, s, (start + t) % 2)
        return c, s

    # -- dense truth constructor (independent path, small rings) ----------------

    def dense_substep(self, origin: int) -> np.ndarray:
        """Brute-force Fock matrix. Origin 1 is defined as ``T† U_0 T`` with T's sign
        from brute-force permutation parity -- sharing no closed forms with the fast
        path."""
        M = self.n_modes
        dim = 1 << M
        if origin == 1:
            piT = self._translation_pi()
            DT = np.zeros((dim, dim), dtype=np.int32)
            for tau in range(dim):
                out = 0
                for m in range(M):
                    if (tau >> m) & 1:
                        out |= 1 << int(piT[m])
                DT[out, tau] = self.jw.mode_permutation_sign(tau, piT)
            U0 = self.dense_substep(0).astype(np.int32)
            return (DT.T @ U0 @ DT).astype(np.int8)

        # origin 0: the batch fast path applied to every basis state. This is a
        # batching-consistency matrix, not an independent physics path; the physics is
        # validated by the operator-level tests (Gaussianity as a matrix identity,
        # unitarity, the exact t=0 sum rule, block commutation).
        configs = np.arange(dim, dtype=np.int64)
        out_c, out_s = self._apply_blocks(configs, np.ones(dim, dtype=np.int64), origin=0)
        mat = np.zeros((dim, dim), dtype=np.int8)
        mat[out_c, configs] = out_s
        return mat
