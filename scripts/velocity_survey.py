#!/usr/bin/env python
"""Regenerate the velocity-set comparison table in docs/adr/0001-finite-ray-theorem.md.

    .venv/bin/python scripts/velocity_survey.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from pca3d.lattices.velocity_sets import CUBIC_6, FCC_12, ALL_SETS, VelocitySet

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def survey() -> list[dict]:
    rows = []
    for vs in ALL_SETS:
        rows.append(
            {
                "name": vs.name,
                "n": vs.n,
                "dim": vs.dim,
                "speeds": sorted({round(float(s), 6) for s in vs.speeds}),
                "single_speed": vs.single_speed,
                "angular_gap_deg": float(np.degrees(vs.angular_gap(n_probe=400_000))),
                "t2_anisotropy": vs.t2_anisotropy(),
                "t4_anisotropy": vs.t4_anisotropy(),
                "note": vs.note,
            }
        )
    return rows


def axis_weight_scan() -> dict:
    """The unique axis multiplicity that makes T4 isotropic in 3D."""
    I = np.eye(3)
    iso = (
        np.einsum("ab,cd->abcd", I, I)
        + np.einsum("ac,bd->abcd", I, I)
        + np.einsum("ad,bc->abcd", I, I)
    )
    ws = np.linspace(0.0, 4.0, 40_001)
    aniso = []
    for w in ws:
        T4 = w * CUBIC_6.tensor_4() + FCC_12.tensor_4()
        A = (T4 * iso).sum() / (iso * iso).sum()
        aniso.append(float(np.linalg.norm(T4 - A * iso) / np.linalg.norm(T4)))
    aniso = np.array(aniso)
    return {
        "argmin_weight": float(ws[aniso.argmin()]),
        "min_anisotropy": float(aniso.min()),
        "fchc_projection_weight": 2.0,
    }


def main() -> None:
    rows = survey()
    scan = axis_weight_scan()

    head = f"{'set':<14}{'n':>4}{'dim':>5}{'gap(deg)':>10}{'T2 aniso':>11}{'T4 aniso':>11}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['name']:<14}{r['n']:>4}{r['dim']:>5}"
            f"{r['angular_gap_deg']:>10.2f}{r['t2_anisotropy']:>11.2e}{r['t4_anisotropy']:>11.2e}"
        )
    print()
    print(
        f"axis-weight scan: T4 isotropic at w = {scan['argmin_weight']:.4f} "
        f"(anisotropy {scan['min_anisotropy']:.2e}); "
        f"FCHC projection supplies w = {scan['fchc_projection_weight']}"
    )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "velocity_survey.json"
    out.write_text(json.dumps({"sets": rows, "axis_weight_scan": scan}, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
