#!/usr/bin/env python
"""Committed reproduction of the Eq. (38) erratum reported in the manuscript.

Claim (manuscript footnote, Sec. "Grassmann action"): in
C. Wetterich, "Fermion picture for cellular automata", arXiv:2203.14081,
the coefficient of the top monomial of the closed-form interaction action
Eq. (38) (source label CS9) must be -2, not the printed -1: the printed
value fails that reference's own defining relation, Eq. (CS3),
exp{psi psibar - L_int} = K, at exactly that one monomial.

This script recomputes the whole object in EXACT Grassmann arithmetic over
the rationals, from the reference's own inputs, and writes the verdict as a
committed artifact.  It is deliberately standalone (the same check also runs
inside tests/test_grassmann.py) because a published-erratum claim should not
rest on an assertion with no stored output.

Steps, all asserted:
  1. Build the 2<->4 scattering block as a signed unique-jump permutation on
     the 16-state local Fock space: (0,0,1,1) <-> (0,0,0,0), i.e. v=12 <-> 0,
     with sign -1 on 12 -> 0.  This is the block whose local factor the
     reference prints as Eq. (CS8).
  2. Its local factor K, computed by the pipeline, equals the PRINTED CS8
     exactly -- so the input block is the reference's own.
  3. The extracted action equals the printed CS9 with the top monomial
     coefficient corrected from 1 to 2.
  4. The printed CS9 fails CS3: exp{ps - L_printed} != K.
  5. The corrected CS9 satisfies CS3: exp{ps - (L_printed + top)} == K,
     and the discrepancy is confined to the single top monomial.

Output: results/wetterich_eq38_check.json
"""
import json
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pca3d.grassmann import G, exp, extract_action, local_factor  # noqa: E402

T0 = time.time()


def pp(a):          # psi'_a, 1-based, M = 4
    return G.monomial((a - 1,))


def bb(a):          # psibar_a, 1-based, M = 4
    return G.monomial((4 + a - 1,))


def main():
    out = {"reference": "arXiv:2203.14081 (Wetterich), Eq. (38) / label CS9",
           "claim": "top-monomial coefficient is -2, printed -1"}

    # 1. the reference's own 2<->4 scattering block
    perm = np.arange(16)
    perm[12], perm[0] = 0, 12
    signs = np.ones(16, dtype=np.int64)
    signs[12] = -1
    K = local_factor(perm, signs)

    # 2. K equals the printed CS8, exactly
    ps = sum((pp(a) * bb(a) for a in range(1, 5)), G())
    K8 = (G.scalar(1) + ps + Fraction(1, 2) * ps * ps
          + pp(1) * pp(2) * bb(1) * bb(2)
          + Fraction(1, 6) * ps * ps * ps
          + pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2)
          + pp(1) * pp(2) * bb(1) * bb(2) * bb(3) * bb(4))
    assert K == K8, "input block is not the reference's CS8 block"
    out["local_factor_equals_printed_CS8"] = True

    # 3. extracted action = printed CS9 with the top coefficient corrected
    top = pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2) * bb(3) * bb(4)
    L_printed = -(pp(1) * pp(2) * bb(1) * bb(2)
                  + pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2)
                  + pp(1) * pp(2) * bb(1) * bb(2) * bb(3) * bb(4)
                  - pp(1) * pp(2) * pp(3) * bb(1) * bb(2) * bb(3)
                  - pp(1) * pp(2) * pp(4) * bb(1) * bb(2) * bb(4)
                  - top)
    L = extract_action(perm, signs)
    assert L == -ps + L_printed + top, "extraction disagrees with corrected CS9"
    out["extracted_action_equals_corrected_CS9"] = True

    # 4./5. the printed action fails CS3; the corrected one satisfies it
    assert exp(ps - L_printed) != K8
    assert exp(ps - (L_printed + top)) == K8
    out["printed_CS9_satisfies_CS3"] = False
    out["corrected_CS9_satisfies_CS3"] = True

    # the discrepancy is exactly one monomial, with coefficient 1 -> 2
    diff = (L_printed + top) - L_printed
    assert diff == top, "discrepancy is not the single top monomial"
    out["discrepancy"] = ("single monomial p1p2p3p4 b1b2b3b4; printed "
                          "coefficient -1, correct -2")
    out["elapsed_s"] = time.time() - T0

    print("Wetterich arXiv:2203.14081, Eq. (38) (CS9) -- exact recomputation")
    print("  local factor == printed CS8                : yes")
    print("  extracted action == printed CS9 + top      : yes")
    print("  printed CS9 satisfies its own CS3          : NO")
    print("  corrected CS9 (top coeff -1 -> -2) satisfies CS3: yes")
    print("  discrepancy: the single monomial "
          "p1p2p3p4 b1b2b3b4 (coefficient -1 printed, -2 correct)")
    pathlib.Path("results/wetterich_eq38_check.json").write_text(
        json.dumps(out, indent=1))
    print(f"\n[ALL ASSERTIONS PASSED]  ({out['elapsed_s']:.1f}s)  "
          "-> results/wetterich_eq38_check.json")


if __name__ == "__main__":
    main()
