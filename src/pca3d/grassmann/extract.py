"""Table -> local factor -> action: the mechanical Wetterich pipeline (Phase E2).

Given a block permutation ``perm`` on ``2^M`` states (M = block modes; 16-state blocks
have M = 4, the 6-bit tier has M = 6) and an optional per-transition sign table, this
module constructs the *local factor*

    K = sum_v  sign(v) * g_{perm(v)}(psi') * g'_v(psibar)         (2111.06728 eq. 10,
                                                                   2203.14081 eq. GU5)

and the action element

    L = -log K,        K = exp(-L)                                 (GU15)

as an exact Grassmann polynomial over rational coefficients. The log is the
terminating nilpotent series; this *is* the "systematic expansion" of 2203.14081
(CS3-CS5) performed to all orders at once, since the exact log agrees with the
order-by-order construction of L_int whenever the latter converges.

Conventions (2111.06728 appendix A, matching this project's bit layout):

  - Generator index alpha = block bit index + 1; the project packs block configs
    site-major with species fastest (``bit = site*n_species + species``), which is
    exactly Wetterich's ``alpha = n_species*m + gamma`` ordering (his eq. A2).
  - ``g_tau`` carries a factor ``psi_alpha`` for every EMPTY bit ``n_alpha = 0``,
    ordered by increasing alpha, overall sign +1 (his eq. A8). The fully occupied
    state has ``g = 1``.
  - ``g'_tau = eps_tau g_tau`` with ``eps_tau = (-1)^(m(m-1)/2)``, m the number of
    psi factors (his eq. 07); this absorbs the reordering signs so that the identity
    table gives exactly ``K = exp(psi'_alpha psibar_alpha)`` (his eqs. 08, 32).
  - Output-time variables psi' are generators ``0 .. M-1``; input-time variables
    psibar are generators ``M .. 2M-1``. Every canonical monomial therefore reads
    "primed factors left of barred factors", as written in the papers.

Sign gauge: Wetterich's automaton tables carry all-plus signs. The project's
fermionic lift (``fock/signed.py``: species-major Jordan-Wigner ranks + Gaussian
mode-map crossing signs) assigns each block transition a sign; ``block_sign_table``
reproduces that gauge locally so the extracted action is the action of the *actual*
signed quantum model. For a table that is a pure permutation of modes (free
transport) that gauge is precisely the one in which L is exactly bilinear,
``L = -F_ab psi'_a psibar_b`` (2203.14081 eq. GU17).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .algebra import G, exp, log


def n_block_modes(perm: np.ndarray) -> int:
    n = len(perm)
    m = n.bit_length() - 1
    if 1 << m != n:
        raise ValueError(f"table length {n} is not a power of two")
    return m


def basis_monomial(config: int, M: int, barred: bool) -> tuple[int, ...]:
    """Generator tuple of ``g_config``: one factor per empty bit, ascending alpha."""
    off = M if barred else 0
    return tuple(off + a for a in range(M) if not (config >> a) & 1)


def eps_tau(config: int, M: int) -> int:
    """Reordering sign ``eps_tau = (-1)^(m(m-1)/2)``, m = number of psi factors."""
    m = M - bin(config).count("1")
    return -1 if (m * (m - 1) // 2) % 2 else 1


# -- the pipeline -------------------------------------------------------------------


def local_factor(perm: np.ndarray, signs: np.ndarray | None = None) -> G:
    """``K = sum_v sign(v) g_{perm(v)}(psi') g'_v(psibar)`` as an exact element."""
    M = n_block_modes(perm)
    terms: dict[tuple[int, ...], Fraction] = {}
    for v in range(len(perm)):
        s = int(signs[v]) if signs is not None else 1
        if s not in (-1, 1):
            raise ValueError(f"sign table entry {s} at {v}; must be +-1")
        out_mono = basis_monomial(int(perm[v]), M, barred=False)
        in_mono = basis_monomial(v, M, barred=True)
        # primed generators all precede barred ones, so concatenation is canonical
        # and carries no interleaving sign
        terms[out_mono + in_mono] = Fraction(s * eps_tau(v, M))
    return G(terms)


def extract_action(perm: np.ndarray, signs: np.ndarray | None = None) -> G:
    """``L`` with ``K = exp(-L)``, verified exactly before returning.

    Requires the table to fix the fully occupied state with sign +1 (K then has
    scalar part 1, which is Wetterich's normalization of the step evolution
    operator); otherwise no exponential form ``exp(-L)`` with polynomial ``L``
    exists and a ``ValueError`` is raised.
    """
    K = local_factor(perm, signs)
    L = -log(K)
    if exp(-L) != K:
        raise AssertionError("exp(-L) != K: exp/log roundtrip broken (internal bug)")
    return L


def step_operator_from_factor(K: G, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse direction (2111.06728 eqs. 30-31): recover ``(perm, signs)`` from K.

    Every monomial must split as (primed block)(barred block) with coefficient
    ``sign * eps``; each input state must appear exactly once. Raises if K is not a
    signed unique-jump factor.
    """
    N = 1 << M
    perm = np.full(N, -1, dtype=np.int64)
    signs = np.zeros(N, dtype=np.int64)
    for mono, coeff in K.terms.items():
        tau = N - 1
        rho = N - 1
        for g in mono:
            if g < M:
                tau &= ~(1 << g)
            else:
                rho &= ~(1 << (g - M))
        s = coeff / eps_tau(rho, M)
        if s not in (1, -1):
            raise ValueError(f"coefficient {coeff} at monomial {mono} is not +-eps")
        if perm[rho] != -1:
            raise ValueError(f"input state {rho} multiplied by two output elements")
        perm[rho] = tau
        signs[rho] = int(s)
    if np.any(perm < 0) or len(np.unique(perm)) != N:
        raise ValueError("K is not a unique-jump local factor")
    return perm, signs


# -- the project's sign gauge --------------------------------------------------------


def block_sign_table(
    perm: np.ndarray, n_species: int, swap_semantics: bool = True
) -> np.ndarray:
    """Per-transition signs of the project's fermionic lift, restricted to one block.

    Reproduces ``fock/signed.py`` exactly (verified against its dense matrix in the
    tests) and generalizes it to ``n_species`` channels per site for the 6-bit tier:
    two-site block, ``bit = site*n_species + species``, species-major Jordan-Wigner
    ranks ``rank = species*2 + site``. Each transition is decomposed per species
    channel into destroyed pairs + a mode map on survivors (with the Gaussian
    crossing sign) + created pairs; parity-even tables make these signs independent
    of the environment outside the block, so the restriction is exact.
    """
    M = 2 * n_species
    if len(perm) != 1 << M:
        raise ValueError(f"table has {len(perm)} entries, expected {1 << M}")
    pc = np.array([bin(i).count("1") for i in range(1 << M)])
    if np.any((pc[perm] & 1) != (pc & 1)):
        raise ValueError(
            "block rule does not conserve fermion parity; no fermionic lift exists "
            "(R4.5) and no per-block sign gauge is defined"
        )
    ranks = np.empty(M, dtype=np.int64)
    for site in range(2):
        for sp in range(n_species):
            ranks[site * n_species + sp] = sp * 2 + site

    def jw_sign(cfg: int, mode: int) -> int:
        below = sum(
            1 for m2 in range(M) if ranks[m2] < ranks[mode] and (cfg >> m2) & 1
        )
        return -1 if below % 2 else 1

    signs = np.empty(1 << M, dtype=np.int64)
    for val in range(1 << M):
        target = int(perm[val])
        pairs_gone: list[int] = []
        pairs_born: list[int] = []
        phi_pairs: list[tuple[int, int]] = []
        for ch in range(n_species):
            i0, i1 = ch, ch + n_species
            in_bits = ((val >> i0) & 1, (val >> i1) & 1)
            out_bits = ((target >> i0) & 1, (target >> i1) & 1)
            if sum(in_bits) != sum(out_bits):
                for i in (i0, i1):
                    if (val >> i) & 1 and not (target >> i) & 1:
                        pairs_gone.append(i)
                    if (target >> i) & 1 and not (val >> i) & 1:
                        pairs_born.append(i)
                continue
            moved = in_bits != out_bits
            crossing = swap_semantics and (moved or in_bits == (1, 1))
            if crossing:
                if in_bits[0]:
                    phi_pairs.append((i0, i1))
                if in_bits[1]:
                    phi_pairs.append((i1, i0))
            else:
                if in_bits[0]:
                    phi_pairs.append((i0, i0))
                if in_bits[1]:
                    phi_pairs.append((i1, i1))

        cfg, sign = val, 1
        for m in sorted(pairs_gone, key=lambda m: -ranks[m]):
            sign *= jw_sign(cfg, m)
            cfg &= ~(1 << m)
        srcs = sorted((m for m, _ in phi_pairs), key=lambda m: ranks[m])
        dst_of = dict(phi_pairs)
        images = [ranks[dst_of[m]] for m in srcs]
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if images[i] > images[j]:
                    sign = -sign
        for m, _ in phi_pairs:
            cfg &= ~(1 << m)
        for _, m2 in phi_pairs:
            cfg |= 1 << m2
        for m in sorted(pairs_born, key=lambda m: ranks[m]):
            sign *= jw_sign(cfg, m)
            cfg |= 1 << m
        if cfg != target:
            raise AssertionError(
                f"decomposition of transition {val}->{target} landed on {cfg}"
            )
        signs[val] = sign
    return signs


def clock_sign_table(mode_perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(state perm, signs) of the clock/transport gauge for a pure mode permutation.

    This is the gauge of Wetterich's transport operators (2111.06728 eqs. 32-33,
    2203.14081 eq. GU17): the local factor is exactly
    ``K = exp{psi'_{F(b)} psibar_b}`` and the action exactly the bilinear
    ``L = -F_ab psi'_a psibar_b``. Expanding that exponential multiplies each state's
    transition by the parity of the mode permutation restricted to the HOLES of the
    state (the psibar factors), which is the sign computed here. It generally differs
    from the particle-picture Gaussian gauge of ``block_sign_table`` by a Z2 state
    gauge transformation.
    """
    M = len(mode_perm)
    N = 1 << M
    perm = np.empty(N, dtype=np.int64)
    signs = np.empty(N, dtype=np.int64)
    for v in range(N):
        out = 0
        for b in range(M):
            if (v >> b) & 1:
                out |= 1 << int(mode_perm[b])
        holes = [b for b in range(M) if not (v >> b) & 1]
        images = [int(mode_perm[b]) for b in holes]  # holes already ascending
        sign = 1
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if images[i] > images[j]:
                    sign = -sign
        perm[v] = out
        signs[v] = sign
    return perm, signs


def sequential_swap_signs(
    n_modes: int, swaps_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Fermionic lift of a rule defined as a sequence of conditional transpositions.

    ``swaps_fn(v)`` returns the ordered list of mode transpositions ``(m0, m1)`` the
    rule actually fires on input ``v`` (decisions may depend on the evolving state,
    exactly as in the sequential table builders). Each fired swap is lifted as the
    Gaussian mode transposition, whose sign in the species-major rank gauge (channel
    modes at adjacent ranks) is -1 iff both modes are occupied at that moment. This
    is the semantically faithful counterpart of ``block_sign_table`` for tables whose
    doubly-occupied channels do NOT always cross (e.g. the 6-bit tier's conditional
    imprint): the product of the individual signed swap operators is itself a signed
    permutation, so the gauge is consistent by construction.
    """
    N = 1 << n_modes
    perm = np.empty(N, dtype=np.int64)
    signs = np.empty(N, dtype=np.int64)
    for v in range(N):
        cfg, sign = v, 1
        for m0, m1 in swaps_fn(v):
            b0, b1 = (cfg >> m0) & 1, (cfg >> m1) & 1
            if b0 and b1:
                sign = -sign
            cfg = cfg & ~((1 << m0) | (1 << m1)) | (b0 << m1) | (b1 << m0)
        perm[v] = cfg
        signs[v] = sign
    return perm, signs
# -- presentation --------------------------------------------------------------------


def split_action(L: G) -> tuple[G, G]:
    """(quadratic part, everything of higher degree). The quadratic part is the
    free/kinetic sector; for parity-even tables L has no odd-degree terms."""
    quad = L.degree_part(2)
    rest = G()
    rest.terms = {m: c for m, c in L.terms.items() if len(m) > 2}
    return quad, rest


def format_element(x: G, names: list[str], barred_style: str = "bar") -> str:
    """Human-readable rendering. ``names[mode]`` names block mode ``mode``; primed
    (output-time) generators print as ``name'``, barred (input-time) ones as
    ``name~`` (or ``name`` with ``barred_style='plain'``)."""
    M = len(names)
    if not x.terms:
        return "0"
    lines = []
    for mono in sorted(x.terms, key=lambda m: (len(m), m)):
        c = x.terms[mono]
        factors = []
        for g in mono:
            if g < M:
                factors.append(names[g] + "'")
            else:
                suffix = "~" if barred_style == "bar" else ""
                factors.append(names[g - M] + suffix)
        body = " ".join(factors) if factors else "1"
        if c == 1:
            lines.append(f"+ {body}")
        elif c == -1:
            lines.append(f"- {body}")
        else:
            sgn = "+" if c > 0 else "-"
            lines.append(f"{sgn} {abs(c)} {body}")
    return "\n".join(lines)
