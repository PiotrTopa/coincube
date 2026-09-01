# coincube

**A 3D probabilistic cellular automaton that is exactly a quantum field
theory of Weyl and Dirac fermions — verification code and manuscript.**

The *coincube* is a probabilistic cellular automaton on the 3D cubic
lattice — classical bits, local invertible homogeneous update layers, a
Bernoulli product vacuum — whose dynamics is *exactly* unitary quantum
mechanics (explicit complex structure, mechanically extracted Grassmann
action) and whose single-fermion excitation is an isotropic Weyl fermion,
extendable to a massive Dirac fermion with gap exactly
`2·arctan[q_m/(1−q_m)]` and to a certified media-mediated interaction.

This repository contains the complete verification code behind the paper

> P. Topa, *Fermionic quantum field theories as probabilistic cellular
> automata in three dimensions* (2026) — `paper/main.pdf`

Every claim tagged [proven] / [machine-checked] / [measured] in the paper
maps to a runnable, self-asserting script here.

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
                    w3*/w3c*  cone scan, verification, legality, lift coherence
                    w4*       helicity residues
                    e1/e2     complex structure; Grassmann actions
                    m8*       massive Dirac
                    i2*/i3*   interaction certificates and survival
                    a_inout*  two-boundary (in-out) propagator
                    theory_*  machine-checked proof packs
                    paper_figs.py  regenerates the manuscript figures
    tests/          131 tests: reproduction, theorem suite, legality,
                    exactness anchors

## Reproduce

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    PYTHONPATH=src .venv/bin/python -m pytest -q          # 131 tests
    PYTHONPATH=src .venv/bin/python scripts/m8_mass_exact.py       # example
    PYTHONPATH=src:scripts .venv/bin/python scripts/e1_complex_structure.py

Scripts regenerate their outputs under `results/`. GPU (CuPy) is optional;
all certificates run on CPU.

## License

MIT (see `LICENSE`). The manuscript in `paper/` is the author's preprint
version and is not covered by the code license.
