# coincube

**A 3D probabilistic cellular automaton that is exactly a fermionic
quantum field theory, whose single-fermion excitation is an isotropic
Weyl fermion — exact by theorem, confirmed by gated measurement —
verification code and manuscript.**

The *coincube* is a probabilistic cellular automaton on the 3D cubic
lattice — classical bits, local invertible homogeneous update layers, a
Bernoulli product vacuum — whose dynamics is *exactly* unitary quantum
mechanics (explicit complex structure, mechanically extracted Grassmann
action). Its environment streaming outruns the carrier ("fresh tape"):
no bit is ever consulted twice, so the ensemble propagator equals the
Bloch operator theory identically at every time, and every closed form
is a statement about the automaton itself. The single-fermion
excitation is an isotropic Weyl fermion, exact by the fresh-tape
theorem and confirmed by gated measurement. A dense-sweep census with
computed topological charges charts the full gapless spectrum;
inversion doubling gives the resonant node quartet a gap of exactly
`2·arctan[q_m/(1−q_m)]`; a media-mediated interaction with a continuous
coupling carries the exact damping law `U_g = (1−2gq²)·U` at every
cycle.

This repository contains the complete verification code behind the paper

> P. S. Topa, *Fermionic quantum field theories as probabilistic cellular
> automata in three dimensions* (2026) — `paper/main.pdf`

Every claim tagged [proven] / [machine-checked] / [measured] in the paper
maps to a runnable, self-asserting script here.

## In plain terms, for the software engineer

**The machine.** Picture a 3D grid, say 48×48×48 cells. Each cell holds a
few bits: one *carrier* particle bit with a 2-bit internal register (the
"coin" — four channels), plus three independent boolean *environment*
fields, one per axis, each bit set to 1 with probability `q` at
initialization. That initialization is the **only randomness in the entire
system**. From t=0 on, everything is a deterministic, reversible, local
update rule — a massively parallel bijective state machine. Every step is
a permutation of global states, so nothing is ever lost: you could run it
backwards.

One update cycle, for each axis (twice per axis), applies three
sub-layers:

1. **Convert** — at every cell, *if* that cell's env bit for the current
   axis is 1, apply a fixed 4×4 signed swap table to the coin register. A
   conditional lookup-table op: branch-free, local.
2. **Move** — every carrier shifts ±1 cell along the current axis,
   direction read off its coin register. Unconditional.
3. **Stream** — the env field itself scrolls, by three consecutive
   pair-swap layers *along the next axis over* (cross-streaming), with
   swap origins that keep alternating across the whole run
   (phase-continuing). Two load-bearing details. If the env scrolled
   along its own axis, a carrier would keep re-reading its own bit and
   the construction breaks. And the *speed* matters: three swaps per
   substep move every env bit ±6 cells per cycle while the carrier
   manages at most ±2, so every bit the particle could revisit has
   already left its light cone — **no bit is ever read twice** (the
   "fresh-tape" theorem, proven and verified as an exact sum over all
   2²⁴ read histories).

So the environment is a pre-rolled random tape the particle reads as it
walks, the tape scrolls deterministically, and the scrolling is fast
enough that every read hits an untouched cell. Quantum behavior is not
injected anywhere. The claim (Wetterich's construction, extended here to
3D) is that the *statistics* of this classical bit machine are exactly a
fermionic quantum field theory: signed permutations are orthogonal
matrices, and an explicitly constructed complex structure turns
orthogonal into unitary. No approximation, no hidden floats.

**Why it's hard.** A relativistic particle needs an isotropic light cone —
same speed in every direction, ω = v|k|. We prove a no-go ("finite
rays"): any such deterministic rule, viewed in momentum space, is a
*monomial* matrix (one nonzero entry per row), and monomial matrices only
ever produce a finite menu of fixed velocity vectors. On a cubic grid the
speed surface is a diamond — diagonal moves come out √3 "faster" — and no
continuum limit fixes it; the error is scale-invariant. Software analogy:
a character on a square grid moving one cell per tick never has true
circular range. A\*-on-a-grid always has that diagonal artifact.

**The fix, in two ingredients.**

*Dressing.* Conversions fire randomly (density `q`), so the
*ensemble-averaged* transfer per step is a mixture, (1−q)·I + q·C — no
longer monomial. Averaging over the random tape is the loophole in the
no-go: each individual run is still a signed permutation (we test exactly
this), but the measured, averaged propagator isn't, so continuous
velocities become possible. Because of the fresh tape this isn't an
approximation: every read is an independent Bernoulli draw *exactly*, so
the averaged operator IS the automaton's propagator, not an idealization
of it. That alone, however, still gives you the diamond.

*The quaternion coin.* The fix is *which* three conversion tables you
use. C_x, C_y, C_z are chosen to multiply like the quaternion units
**i, j, k**: each squares to −1, and applying two different ones in
opposite orders gives opposite signs. (They're still just fixed signed
bit-swap tables — the machine stays a bit machine.) That sign-flip on
reordering makes the three axis updates *interfere* instead of composing
independently, and near the operator's degenerate point the cross-terms
between axes cancel in exactly the pattern of a Clifford algebra,
`{h_a, h_b} = 2v²·δ_ab` — the algebraic definition of "the splitting
depends only on |k|, not on direction". That's the Weyl equation's
algebra, proven symbolically for the averaged operator and then measured
on the running automaton: direction-ratio deviations from the diamond
prediction are excluded at about ten error bars. The one-line intuition:
you can't make a cube isotropic by moving cleverly, but you can by
*rotating internally* — the particle carries a tiny 4-state register
whose axis updates multiply like quaternions, the smallest real algebra
in which x-ness, y-ness and z-ness combine into one rotationally
symmetric object.

**Bonus.** The same trick extends: doubling the register to 8 states plus
a fourth env field gives an exact, tunable mass (a Dirac fermion), and
one extra rule letting the carrier *write back* into the environment
gives interactions.

**The honest fine print** (stated carefully in the paper): the
fresh-tape identity covers observables *linear* in the evolved field —
amplitudes, propagators, spectra; higher-order statistics still know the
difference between one tape and an average over tapes. On a finite torus
it holds up to a sharp horizon, T ≤ ⌈L/8⌉ cycles, before the fast tape
laps the lattice. The cone comes with lattice partner species elsewhere
in the zone, charted by a computed census, and the mass gaps only the
resonant quartet of them. And the in-state quasiparticle is damped —
provably unavoidably so within this event class, with an exact bound
tying phase advance to visibility loss (the two-boundary propagator of
the same formalism is the unitary completion).

**Why the repo looks the way it does.** Every `[proven]` /
`[machine-checked]` / `[measured]` tag in the paper maps to a script here
that *asserts* the claim and crashes if it fails — proofs as tests.
Measurements run behind known-answer gates (the pipeline must first
reproduce an exactly solvable case) and positive controls (it must *not*
report isotropy when fed an anisotropic truth). Think CI for physics
claims: `pytest` runs the theorem suite, and `results/` holds the
committed evidence behind every number in the paper.

## Layout

    paper/          manuscript (REVTeX source, figures, PDF)
    docs/           the R1-R7 rules contract every construction obeys
    src/pca3d/
      core/         lattice, bit encoding, unique-jump verification
      models/       wetterich1d (1+1D reference), conditional (CPA class),
                    coincube (the model: massless, massive, interacting)
      lattices/     velocity sets (finite-ray theorem support)
      analysis/     dispersion and correlator instruments
      fock/         signed fermionic lifts (Jordan-Wigner, propagators)
      grassmann/    exact Grassmann algebra and action extraction
      search/       legal-rule enumeration from elementary processes
    scripts/        certificates, proofs and measurements, by paper section:
                    freshtape_*  the fresh-tape theorem (exact path sums)
                                 and the exact interaction law
                    reread_kinematics.py  re-read census, generic vs fresh
                    w3*/w3c*  cone scan, verification, legality, lift coherence
                    *_fresh   the gated campaign on the fresh-tape model
                              (cone, helicity, mass, interaction, controls,
                              in-out) and schedule-parameterized certificates
                    w4*       helicity residues
                    e1/e2     complex structure; Grassmann actions
                    m8*       massive Dirac
                    i2*/i3*   interaction certificates and survival
                    a_inout*  two-boundary (in-out) propagator
                    theory_*  machine-checked proof packs
                    census_sweep.py  dense-sweep census + Chern charges
                    spectrum_census.py  structural loci + factorization
                    orbit_flow.py  spectator-orbit continuation in q
                    bridge_check.py  complex-structure/spectral bridge
                    m8_pulls.py  massive dispersion vs exact branches
                    w3c_positive_control.py  estimator positive controls
                    paper_figs.py  regenerates the manuscript figures
    tests/          137 tests: reproduction, theorem suite, legality,
                    fresh-tape schedule, exactness anchors

## Reproduce

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    PYTHONPATH=src .venv/bin/python -m pytest -q          # 137 tests (GPU-marked tests skip on CPU-only boxes)
    PYTHONPATH=src .venv/bin/python scripts/m8_mass_exact.py       # example
    PYTHONPATH=src:scripts .venv/bin/python scripts/e1_complex_structure.py

Scripts regenerate their outputs under `results/`. GPU (CuPy) is optional;
all certificates run on CPU.

## License

MIT (see `LICENSE`). The manuscript in `paper/` is the author's preprint
version and is not covered by the code license.
