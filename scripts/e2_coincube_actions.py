#!/usr/bin/env python
"""E2 applied to the coincube: mechanical Grassmann actions for the ADR 0012
conversion blocks (the literal Wetterich eq. 51-58 certificate for the model
that carries the Weyl cone).

Blocks extracted (exact rational arithmetic, exp(-L) = K verified by the
extractor before anything is reported):
  - conversion block per axis a: one site, modes (c0, c1, c2, c3, e) with
    e = the axis-a env bit; table = identity at e = 0, the certified double
    Givens (channel pairs of C_a with its signs) at e = 1. M = 5, 32 configs.
    Extracted in TWO gauges: the physical Givens-lift gauge (the signs of the
    certified fermionic operator) and the bare all-plus gauge (Wetterich's
    table convention), per the E2 note's standing practice.
  - env transport block (two sites of one env field, pair swap): the standard
    streaming bilinear, extracted as a control.

The motion layer L1 is a pure species translation whose action is the
established transport bilinear L = -sum psi'_{x+d(c)} psibar_x (2203.14081
GU17; validated in the E2 module tests as "pure transport"); it needs no new
extraction. A full coincube sub-step is the Grassmann composition of the
three layers as consecutive factors.

Appends the results to docs/notes/e2-extracted-actions.md.
"""
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, "scripts")
from w3c_lift_check import controlled_l2  # noqa: E402

from pca3d.grassmann.extract import (extract_action, local_factor,  # noqa: E402
                                     step_operator_from_factor)

MODE = ["c0", "c1", "c2", "c3", "e"]


def table_of(a):
    U = controlled_l2(a)
    perm = np.argmax(np.abs(U), axis=0).astype(np.int64)
    sign = U[perm, np.arange(32)].astype(np.int64)
    assert np.all(np.abs(sign) == 1)
    full = 31  # all five modes occupied
    assert perm[full] == full and sign[full] == 1
    return perm, sign


def fmt_mono(mono, M=5):
    parts = []
    for g in mono:
        if g < M:
            parts.append(MODE[g] + "'")
        else:
            parts.append(MODE[g - M] + "~")
    return " ".join(parts)


def fmt_action(L, M=5):
    by_deg = {}
    for mono, c in sorted(L.terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        by_deg.setdefault(len(mono), []).append((mono, c))
    lines = []
    for deg in sorted(by_deg):
        lines.append(f"  degree {deg} ({len(by_deg[deg])} terms):")
        for mono, c in by_deg[deg]:
            cs = str(c) if c != 1 else "+1"
            if c == -1:
                cs = "-1"
            lines.append(f"    {cs:>5}  {fmt_mono(mono, M)}")
    return lines


def main():
    out = ["", "## Coincube conversion blocks (E2 applied to ADR 0012)", "",
           "*Axis-a conversion layer: one site, modes `(c0, c1, c2, c3, e)`,",
           "identity at `e = 0`, the certified double Givens at `e = 1`",
           "(channel pairs of `C_a` with its lift signs). `exp(-L) = K` exact,",
           "round-trip `K -> (perm, signs)` verified. Physical gauge = the",
           "Givens lift of `scripts/w3c_lift_check.py`; bare = all-plus.*", ""]
    for a, name in enumerate(["x", "y", "z"]):
        perm, sign = table_of(a)
        # round trip on the physical gauge
        K = local_factor(perm, sign)
        p2, s2 = step_operator_from_factor(K, 5)
        assert np.array_equal(p2, perm) and np.array_equal(s2, sign)
        L_phys = extract_action(perm, sign)
        L_bare = extract_action(perm, None)

        def hist(L):
            h = {}
            for m in L.terms:
                h[len(m)] = h.get(len(m), 0) + 1
            return ", ".join(f"deg {d}: {h[d]}" for d in sorted(h))

        print(f"axis {name}: physical gauge {len(L_phys.terms)} terms "
              f"({hist(L_phys)}); bare {len(L_bare.terms)} terms "
              f"({hist(L_bare)})")
        out.append(f"### C_{name} conversion block")
        out.append("")
        out.append(f"Physical (Givens-lift) gauge — {len(L_phys.terms)} terms "
                   f"({hist(L_phys)}):")
        out.append("```")
        out.extend(fmt_action(L_phys))
        out.append("```")
        out.append(f"Bare gauge — {len(L_bare.terms)} terms ({hist(L_bare)}):")
        out.append("```")
        out.extend(fmt_action(L_bare))
        out.append("```")
        out.append("")

    # env transport control block (2 sites, pair swap)
    perm_sw = np.array([0, 2, 1, 3], dtype=np.int64)
    L_sw = extract_action(perm_sw, None)
    print(f"env swap block: {len(L_sw.terms)} terms")
    out.append("### Env transport block (pair swap, control)")
    out.append("")
    out.append("```")
    for mono, c in sorted(L_sw.terms.items()):
        parts = []
        for g in mono:
            nm = f"e({'x' if (g % 2) == 0 else 'x+1'})"
            parts.append(nm + ("'" if g < 2 else "~"))
        out.append(f"  {'+1' if c == 1 else str(c):>5}  {' '.join(parts)}")
    out.append("```")
    out.append("")
    out.append("*The motion layer is the established transport bilinear "
               "`L = -sum_c psi'_{c, x+d_a(c)} psibar_{c, x}` (GU17-validated); "
               "one coincube sub-step = conversion factor x transport factor x "
               "env factor as consecutive Grassmann layers.*")

    with open("docs/notes/e2-extracted-actions.md", "a") as f:
        f.write("\n".join(out) + "\n")
    print("appended: docs/notes/e2-extracted-actions.md")


if __name__ == "__main__":
    main()
