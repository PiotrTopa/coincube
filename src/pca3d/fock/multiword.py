"""Multiword configuration packing: the fermionic lift beyond 62 modes.

Same physics and gauge as ``fock.signed`` (species-major Jordan-Wigner ranks, Gaussian
mode-map lift, open chain), with configurations stored as ``(batch, n_words)`` uint64
arrays instead of single int64s. Motivated by threat class #4 (silent int64 overflow,
J14): here the mode count is bounded only by memory, and the word arithmetic is
explicit rather than implicit.

Popcounts use ``np.bitwise_count`` (numpy >= 2.0), so the hot paths stay vectorised.
Cross-validated bitwise against the int64 path on <= 62 modes (tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import conditional as C  # block-table semantics live there


def n_words(n_modes: int) -> int:
    return (n_modes + 63) // 64


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """(batch, M) bool -> (batch, W) uint64."""
    batch, M = bits.shape
    W = n_words(M)
    out = np.zeros((batch, W), dtype=np.uint64)
    for m in range(M):
        w, b = divmod(m, 64)
        out[:, w] |= bits[:, m].astype(np.uint64) << np.uint64(b)
    return out


def get_bit(cfg: np.ndarray, mode: int) -> np.ndarray:
    w, b = divmod(mode, 64)
    return (cfg[:, w] >> np.uint64(b)) & np.uint64(1)


def _flip(cfg: np.ndarray, mode: int, on: np.ndarray) -> None:
    """In place: set bit where on, clear where ~on (callers guarantee validity)."""
    w, b = divmod(mode, 64)
    bit = np.uint64(1) << np.uint64(b)
    cfg[:, w] = np.where(on, cfg[:, w] | bit, cfg[:, w] & ~bit)


def masked_parity(cfg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Parity of popcount(cfg & mask) per row; mask shape (W,)."""
    return np.bitwise_count(cfg & mask[None, :]).sum(axis=1) & 1


class MWJW:
    """Jordan-Wigner machinery on multiword configs, species-major ranks."""

    def __init__(self, n_sites: int, n_species: int = 2):
        self.M = n_sites * n_species
        self.W = n_words(self.M)
        ranks = np.empty(self.M, dtype=np.int64)
        for site in range(n_sites):
            for sp in range(n_species):
                ranks[site * n_species + sp] = sp * n_sites + site
        self.ranks = ranks
        self.below = np.zeros((self.M, self.W), dtype=np.uint64)
        for m in range(self.M):
            for m2 in range(self.M):
                if ranks[m2] < ranks[m]:
                    w, b = divmod(m2, 64)
                    self.below[m, w] |= np.uint64(1) << np.uint64(b)

    def jw_sign(self, cfg: np.ndarray, mode: int) -> np.ndarray:
        return 1 - 2 * masked_parity(cfg, self.below[mode]).astype(np.int64)

    def create(self, cfg: np.ndarray, signs: np.ndarray, mode: int):
        occupied = get_bit(cfg, mode) != 0
        s = self.jw_sign(cfg, mode)
        out = cfg.copy()
        w, b = divmod(mode, 64)
        bit = np.uint64(1) << np.uint64(b)
        out[:, w] = np.where(occupied, cfg[:, w], cfg[:, w] | bit)
        return out, np.where(occupied, 0, signs * s)

    def annihilate(self, cfg: np.ndarray, signs: np.ndarray, mode: int):
        occupied = get_bit(cfg, mode) != 0
        s = self.jw_sign(cfg, mode)
        w, b = divmod(mode, 64)
        bit = np.uint64(1) << np.uint64(b)
        out = cfg.copy()
        out[:, w] = np.where(occupied, cfg[:, w] & ~bit, cfg[:, w])
        return out, np.where(occupied, signs * s, 0)


@dataclass
class MWSignedBlockCycle:
    """Open-chain signed block cycle on multiword configs.

    Mirrors ``fock.signed.SignedBlockCycle`` (open boundary only): the same Gaussian
    mode-map lift with crossing signs, the same per-transition decomposition, valid for
    any number of sites.
    """

    n_sites: int
    block_perm: np.ndarray
    swap_semantics: bool = True
    jw: MWJW = field(init=False)

    def __post_init__(self) -> None:
        pc = np.array([bin(i).count("1") for i in range(16)])
        if np.any((pc[self.block_perm] & 1) != (pc & 1)):
            raise ValueError("block rule does not conserve fermion parity")
        self.jw = MWJW(self.n_sites)

    @property
    def n_modes(self) -> int:
        return 2 * self.n_sites

    def _blocks_for_origin(self, origin: int):
        if origin == 0:
            sites = [(2 * b, 2 * b + 1) for b in range(self.n_sites // 2)]
        else:
            sites = [(2 * b + 1, 2 * b + 2) for b in range((self.n_sites - 2) // 2)]
        return [[2 * s0, 2 * s0 + 1, 2 * s1, 2 * s1 + 1] for s0, s1 in sites]

    def _decompose(self, modes, val, target):
        pairs_gone, pairs_born, phi_pairs = [], [], []
        for ch in range(2):
            i0, i1 = ch, ch + 2
            m0, m1 = modes[i0], modes[i1]
            in_bits = ((val >> i0) & 1, (val >> i1) & 1)
            out_bits = ((target >> i0) & 1, (target >> i1) & 1)
            if sum(in_bits) != sum(out_bits):
                for i, m in ((i0, m0), (i1, m1)):
                    if (val >> i) & 1 and not (target >> i) & 1:
                        pairs_gone.append(m)
                    if (target >> i) & 1 and not (val >> i) & 1:
                        pairs_born.append(m)
                continue
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

    def substep(self, cfg: np.ndarray, signs: np.ndarray, origin: int):
        out_c, out_s = cfg, signs
        for modes in self._blocks_for_origin(origin):
            nib = (get_bit(out_c, modes[0]).astype(np.int64)
                   | (get_bit(out_c, modes[1]).astype(np.int64) << 1)
                   | (get_bit(out_c, modes[2]).astype(np.int64) << 2)
                   | (get_bit(out_c, modes[3]).astype(np.int64) << 3))
            for val in range(16):
                target = int(self.block_perm[val])
                sel = nib == val
                if not np.any(sel):
                    continue
                c_sub, s_sub = out_c[sel], out_s[sel]
                pg, phi, pb = self._decompose(modes, val, target)
                for m in sorted(pg, key=lambda m: -self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.annihilate(c_sub, s_sub, m)
                sign_phi = 1
                srcs = sorted((m for m, _ in phi), key=lambda m: self.jw.ranks[m])
                images = [self.jw.ranks[dict(phi)[m]] for m in srcs]
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        if images[i] > images[j]:
                            sign_phi = -sign_phi
                c_sub = c_sub.copy()
                for m, _ in phi:
                    w, b = divmod(m, 64)
                    c_sub[:, w] &= ~(np.uint64(1) << np.uint64(b))
                for _, m2 in phi:
                    w, b = divmod(m2, 64)
                    c_sub[:, w] |= np.uint64(1) << np.uint64(b)
                s_sub = s_sub * sign_phi
                for m in sorted(pb, key=lambda m: self.jw.ranks[m]):
                    c_sub, s_sub = self.jw.create(c_sub, s_sub, m)
                out_c = out_c.copy(); out_s = out_s.copy()
                out_c[sel], out_s[sel] = c_sub, s_sub
        return out_c, out_s


def propagator_mw(
    model: MWSignedBlockCycle,
    n_substeps: int,
    y_site: int,
    ensemble: int = 8192,
    seed: int = 0,
    densities: tuple[float, float] = (0.5, 0.5),
):
    """G+ over the product vacuum, multiword. Returns (g[(t, site, species)], survival)."""
    rng = np.random.default_rng(seed)
    M = model.n_modes
    dens = np.array([densities[m % 2] for m in range(M)])
    bits = rng.random(size=(ensemble, M)) < dens[None, :]
    tau = pack_bits(bits)
    ones = np.ones(ensemble, dtype=np.int64)
    y_mode = 2 * y_site

    d_cfg, d_sgn = model.jw.create(tau, ones.copy(), y_mode)
    v_cfg, v_sgn = tau.copy(), ones.copy()

    L = model.n_sites
    g = np.zeros((n_substeps + 1, L, 2))
    surv = np.zeros(n_substeps + 1)
    for t in range(n_substeps + 1):
        if t:
            d_cfg, d_sgn = model.substep(d_cfg, d_sgn, (t - 1) % 2)
            v_cfg, v_sgn = model.substep(v_cfg, v_sgn, (t - 1) % 2)
        diff = d_cfg ^ v_cfg
        counts = np.bitwise_count(diff).sum(axis=1)
        live = (counts == 1) & (d_sgn != 0)
        surv[t] = live.mean()
        if not np.any(live):
            continue
        idx = np.flatnonzero(live)
        for e in idx:
            row = diff[e]
            w = int(np.flatnonzero(row)[0])
            m = w * 64 + int(row[w]).bit_length() - 1
            par = int(masked_parity(v_cfg[e : e + 1], model.jw.below[m])[0])
            amp = int(d_sgn[e]) * int(v_sgn[e]) * (1 - 2 * par)
            g[t, m // 2, m % 2] += amp
    return g / ensemble, surv
