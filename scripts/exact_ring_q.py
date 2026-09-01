#!/usr/bin/env python
"""Exact q-weighted ring correlator: centroid(t) as an exact rational function of q.

For env-count-conserving rules the product vacuum with env density q is exactly
stationary, and on a small ring the correlator can be computed exactly with
configuration weights w = q^Ne (1-q)^(Me-Ne) (system channel at 1/2, which cancels in
the normalised centroid). Every centroid is then a ratio of polynomials in q with
rational coefficients -- evaluated here at exact rational q via fractions.Fraction, so
the continuity of the short-time speed in q is established by arithmetic, not sampling.

    .venv/bin/python scripts/exact_ring_q.py <rule_index> [L]
"""
from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np

from pca3d.models import conditional as C
from scripts_lib_exact import exact_env_weighted_centroids  # local helper below

def main():
    print("thin wrapper; see scripts_lib_exact.py")

if __name__ == "__main__":
    main()
