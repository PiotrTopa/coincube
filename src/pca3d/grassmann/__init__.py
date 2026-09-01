"""Mechanical Grassmann action extraction (Phase E2).

Turns a block permutation table into the fermionic action of the equivalent quantum
field theory: table -> local factor K -> L = -log K, exactly, over rational
coefficients. This is the literal "what Wetterich did" certificate
(arXiv:2111.06728 sects. 2-3, eqs. 51-58; arXiv:2203.14081 eqs. GU5, GU15, CS3-CS9).
"""

from .algebra import G, exp, log
from .extract import (
    block_sign_table,
    extract_action,
    format_element,
    local_factor,
    split_action,
    step_operator_from_factor,
)

__all__ = [
    "G",
    "exp",
    "log",
    "block_sign_table",
    "extract_action",
    "format_element",
    "local_factor",
    "split_action",
    "step_operator_from_factor",
]
