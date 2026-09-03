# The rules a candidate is not allowed to break

This file is the contract for the whole search. Every candidate automaton produced by
`pca3d.search` is checked against R1-R7 mechanically, and any candidate that violates
one is discarded rather than "explained". If we ever want to relax one of these, that is
a paradigm change and gets its own ADR first -- it does not happen quietly inside a
search loop.

Sources: Wetterich, PRD 105 074502 (arXiv:2111.06728) sects. 2-4;
arXiv:2203.14081 sects. 3-4; arXiv:2211.09002 sect. 1.

---

## R1 -- Locality

The updated state of a cell depends only on a bounded neighbourhood of that cell at the
previous time step. Equivalently, we permit block-partitioned (Margolus) updating: space
is partitioned into disjoint finite blocks, each block is updated by a bijection acting
only on that block's own bits, and the partition may be shifted between consecutive time
steps.

The neighbourhood radius must not grow with lattice size. This is what generates the
light-cone / causal structure (2211.09002 sect. 1).

## R2 -- Determinism and invertibility (the unique-jump property)

The step evolution operator `S_hat` must be a **unique jump matrix**: exactly one entry
equal to +/-1 in each row and each column, all others zero.

    S_hat[tau, rho] = delta(tau, taubar(rho)) = delta(rhobar(tau), rho)

Consequences we rely on, and therefore must never silently lose:

  - `S_hat` is orthogonal, `S_hat S_hat^T = 1`. No information is destroyed.
  - Every configuration maps to exactly one successor, and has exactly one predecessor.
  - In the presence of a compatible complex structure (R5) this is a unitary evolution.

This is the single most important constraint. `pca3d.core.automaton` (`check_unique_jump`) verifies it by
explicit bijectivity check on the configuration permutation, never by argument.

## R3 -- Homogeneity

The same rule is applied in every cell. Two branches of the search are permitted, and
they are tracked and reported separately -- results from one are never quoted as results
for the other:

  - **Branch A (strict).** One fixed rule, identical at every time step:
    `S(t) = S` for all `t`. Any isotropy must come from the lattice point group alone.

  - **Branch B (sequence).** A finite periodic sequence of rules
    `S_cycle = S_n ... S_2 S_1`, where the `S_k` are lattice-rotated or reflected copies
    of one another and each `S_k` independently satisfies R1 and R2. This is Wetterich's
    6- and 24-step construction (2211.09002 sect. 5). Note the product of unique-jump
    matrices is unique-jump, so `S_cycle` itself satisfies R2.

Position-dependent rules ("disorder") are outside both branches for now. They are how
Wetterich generates a mass term (2203.14081 sect. "Disorder and mass term"); if we need
them, that is an ADR.

## R4 -- Bit / fermion encoding, single fermions only

The state is a configuration of occupation numbers

    n_alpha(x) in {0, 1},   alpha = (spin/chirality, flavour),   x a lattice site

one bit per species per site. Configurations are identified with Grassmann basis
elements via the bit-fermion map, so that a valid automaton is simultaneously a
discretised fermionic quantum field theory.

**Single fermions only.** The propagating object is a single-fermion occupation number.
Transporting Lorentz-invariant composites `chi^A = psi_1^A psi_2^A psi_3^A psi_4^A`
(the 2211.09002 four-dimensional route) is *excluded*, deliberately and permanently, by
project decision. This forecloses the one construction already known to reach four
dimensions; that is accepted. The contract is restated in the manuscript's introduction.

## R4half -- Block fermion parity

Every local update block must conserve fermion parity: the number of occupied bits in a
block may change only by even amounts. Discovered as a *derived* requirement while
building the fermionic lift (Phase 1): an odd-parity block operator anticommutes with
operators on disjoint blocks, so no fermionic lift with commuting disjoint blocks
exists, and the automaton -- although perfectly legal as a classical CA -- is not a
fermionic quantum field theory in Wetterich's sense. This is the operator-level face of
his requirement that local factors contain only even powers of Grassmann variables
(2111.06728 eq. 4).

Checked mechanically at construction of any signed lift (`pca3d.fock.signed`), and to
be checked by every rule generator from now on. Effect measured: prunes the
conditional-propagation class from 9216 rules to 256, and the confirmed-ballistic set
from 133 to 9.

## R5 -- Particle-hole symmetry and a compatible complex structure

There must exist an involution `K` (particle-hole conjugation, `n -> 1-n`) and a map `I`
with

    K^2 = +1,    I^2 = -1,    {K, I} = 0

such that `S_hat` is compatible with the resulting complex structure, i.e. in the basis
`q = (q', q^c)`

    S_hat = [[S', S~], [-S~, S']]   =>   U = S' + i S~,   U^dagger U = 1

Without this there is a real orthogonal evolution but no complex Hilbert space, and the
system is not quantum mechanics in the usual sense (2111.06728 sect. 6).

## R6 -- Discrete space-time symmetries

The rule must be invariant under the point group of the chosen lattice, possibly
combined with flavour rotations, and under reflection in time (time reversal) and in
space (parity). Under Branch B, invariance is required of the full cycle `S_cycle`, not
of each `S_k` -- but the deficit of each individual step is recorded, because it is
exactly the quantity that has to die out in the true continuum limit.

## R7 -- No hidden continuum input

The rule is specified entirely by combinatorics on bits. No metric, vierbein, gamma
matrix, or real-valued coupling may appear in the definition of the update. Continuum
objects may only *emerge* in the analysis of the resulting automaton.

---

## What is deliberately NOT required

  - **Particle number conservation.** Wetterich's models happen to conserve it, and the
    one-particle analysis in `pca3d.analysis.dispersion` assumes it, but the search is
    allowed to propose non-conserving rules (2203.14081 sect. "Automata without particle
    number conservation"). Such candidates are routed to a different analyser.

  - **Exact Lorentz symmetry at the discrete level.** Nobody expects it; a discrete
    lattice admits only a discrete subgroup. What is required is that it emerge in the
    continuum limit, and quantifying the failure to do so is the point of the project.

  - **Smoothness of the wave function.** This is a property of the *state*, not of the
    rule. No continuum limit exists for the sharp wave function of a deterministic
    automaton (2211.09002 sect. 6); it is the probabilistic initial conditions that
    make the limit possible.
