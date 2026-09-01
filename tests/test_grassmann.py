"""Phase E2: the mechanical Grassmann action extractor, validated against Wetterich.

Gold standards (exact, term-for-term, rational coefficients):

  - identity table    -> L = -sum_a psi'_a psibar_a          (2111.06728 eq. 32)
  - pure transport    -> L = -F_ab psi'_a psibar_b           (eq. 33 / 2203.14081 GU17)
  - his interaction block (eqs. 44-47, the (1001)<->(0110) exchange)
                      -> L_i = (-psi'_g psibar_g + D)(1 + D) (eq. 58, D from eq. 52)
  - the FPCA 2<->4 scattering example -> K matches eq. CS8 exactly; L matches eq. CS9
    term-for-term except the top monomial, where the exact log gives coefficient 2
    against the printed 1 -- and the printed version demonstrably fails to reproduce
    CS8 under exp, so this is a typo in the paper, not in the extractor.

Plus algebra unit tests and parity of the sign gauge with the fock/signed.py lift.
"""

from fractions import Fraction

import numpy as np
import pytest

from pca3d.fock.signed import SignedBlockCycle
from pca3d.grassmann import (
    G,
    block_sign_table,
    exp,
    extract_action,
    local_factor,
    log,
    split_action,
    step_operator_from_factor,
)
from pca3d.grassmann.extract import (
    clock_sign_table,
    sequential_swap_signs,
)
from pca3d.models.conditional import enumerate_conditional_rules


def mono(*gens, coeff=1):
    return G.monomial(gens, coeff)


def pp(a):  # psi'_a, 1-based, M=4
    return G.monomial((a - 1,))


def bb(a):  # psibar_a, 1-based, M=4
    return G.monomial((4 + a - 1,))


# -- algebra ------------------------------------------------------------------------


class TestAlgebra:
    def test_anticommutation(self):
        t0, t1 = mono(0), mono(1)
        assert t0 * t1 == -(t1 * t0)
        assert t0 * t0 == G()
        assert mono(1, 0) == -mono(0, 1)
        assert mono(0, 1, 0) == G()

    def test_reordering_sign_three_generators(self):
        # 2,0,1 -> 0,1,2 is an even permutation; 1,0,2 odd
        assert mono(2, 0, 1) == mono(0, 1, 2)
        assert mono(1, 0, 2) == -mono(0, 1, 2)

    def test_ring_axioms_spot(self):
        x = mono(0, 1, coeff=Fraction(2, 3)) + mono(2, 3, coeff=-1) + G.scalar(5)
        y = mono(1, 2) + G.scalar(Fraction(-1, 7))
        z = mono(0, 3) + mono(4, 5, coeff=3)
        assert (x * y) * z == x * (y * z)
        assert x * (y + z) == x * y + x * z
        assert (x + y) * z == x * z + y * z

    def test_even_elements_commute(self):
        x = mono(0, 1) + mono(2, 3, coeff=Fraction(1, 2))
        y = mono(1, 2) + mono(0, 3, coeff=-2)
        assert x * y == y * x

    def test_exp_log_roundtrip_nilpotent(self):
        x = (
            mono(0, 1, coeff=Fraction(3, 2))
            + mono(2, 3, coeff=-1)
            + mono(0, 1, 2, 3, coeff=Fraction(1, 5))
            + mono(1, 4, coeff=2)
            + mono(2, 5)
        )
        assert x.scalar_part == 0 and x.is_even()
        assert log(exp(x)) == x
        n = x + mono(0, 2, 4, 5, coeff=7)
        assert exp(log(G.scalar(1) + n)) == G.scalar(1) + n

    def test_exp_additive_for_even(self):
        x = mono(0, 1, coeff=2) + mono(2, 3)
        y = mono(1, 2, coeff=Fraction(-1, 3)) + mono(4, 5)
        assert exp(x) * exp(y) == exp(x + y)

    def test_exp_terminates_and_is_product_form(self):
        # exp(psi'_a psibar_a) = prod_a (1 + psi'_a psibar_a)   (2111.06728 eq. 08)
        x = sum((pp(a) * bb(a) for a in range(1, 5)), G())
        prod = G.scalar(1)
        for a in range(1, 5):
            prod = prod * (G.scalar(1) + pp(a) * bb(a))
        assert exp(x) == prod

    def test_log_requires_unit_scalar(self):
        with pytest.raises(ValueError):
            log(G.scalar(-1) + mono(0, 1))


# -- extractor gold standards -------------------------------------------------------


class TestWetterichGoldStandards:
    def test_identity_block_eq32(self):
        L = extract_action(np.arange(16))
        expected = -sum((pp(a) * bb(a) for a in range(1, 5)), G())
        assert L == expected

    def test_pure_transport_eq33_form(self):
        # both species swap sites, clock (hole) gauge; modes (psi, phi, psi', phi')
        F = {0: 2, 1: 3, 2: 0, 3: 1}
        mode_perm = np.array([F[b] for b in range(4)])
        perm, signs = clock_sign_table(mode_perm)
        L = extract_action(perm, signs)
        expected = -sum(
            (G.monomial((al, 4 + be)) for be, al in F.items()), G()
        )
        assert L == expected  # L = -F_ab psi'_a psibar_b, exactly bilinear (GU17)
        # a shift F = (0 2 1 3)-style cycle works too: use the 4-cycle 0->1->2->3->0
        perm4, signs4 = clock_sign_table(np.array([1, 2, 3, 0]))
        L4 = extract_action(perm4, signs4)
        assert L4 == -sum((G.monomial(((b + 1) % 4, 4 + b)) for b in range(4)), G())

    def test_gauges_differ_but_describe_same_automaton(self):
        # particle-Gaussian gauge (fock/signed.py) vs clock gauge for the same swap:
        # same permutation, different signs, and the Gaussian-gauge action picks up
        # a quartic dressing -- the sign gauge is physics-per-basis, not per-table
        F = np.array([2, 3, 0, 1])
        perm_c, signs_c = clock_sign_table(F)
        perm_g, signs_g = sequential_swap_signs(4, lambda v: [(0, 2), (1, 3)])
        assert np.array_equal(perm_c, perm_g)
        assert not np.array_equal(signs_c, signs_g)
        L_g = extract_action(perm_g, signs_g)
        assert L_g.max_degree() == 4  # not bilinear in this gauge

    def test_interaction_block_eq58(self):
        # (1001) <-> (0110), identity elsewhere (eqs. 44-47); config bit a = n_{a+1}
        perm = np.arange(16)
        perm[0b1001], perm[0b0110] = 0b0110, 0b1001
        L = extract_action(perm)
        D = -(pp(1) * pp(4) - pp(2) * pp(3)) * (bb(1) * bb(4) - bb(2) * bb(3))
        kin = sum((pp(a) * bb(a) for a in range(1, 5)), G())
        assert L == (-kin + D) * (G.scalar(1) + D)  # eq. 58, term for term

    def test_eq58_building_blocks(self):
        # the identities of eq. 53 hold for our symbolic D
        D1 = pp(1) * pp(4) * bb(2) * bb(3) + pp(2) * pp(3) * bb(1) * bb(4)
        D2 = pp(1) * pp(4) * bb(1) * bb(4) + pp(2) * pp(3) * bb(2) * bb(3)
        D = D1 - D2
        assert D1 * D2 == G()
        assert D * D == 2 * (D1 * D1)
        assert D * D * D == G()
        top = pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2) * bb(3) * bb(4)
        assert D * D == 4 * top

    def test_fpca_cs8_cs9(self):
        # 2<->4 scattering: (0,0,1,1) <-> (0,0,0,0), i.e. v=12 <-> v=0. The signed
        # unique jump reproducing the printed CS8 has sign -1 on 12 -> 0.
        perm = np.arange(16)
        perm[12], perm[0] = 0, 12
        signs = np.ones(16, dtype=np.int64)
        signs[12] = -1
        K = local_factor(perm, signs)
        ps = sum((pp(a) * bb(a) for a in range(1, 5)), G())
        K8 = (
            G.scalar(1)
            + ps
            + Fraction(1, 2) * ps * ps
            + pp(1) * pp(2) * bb(1) * bb(2)
            + Fraction(1, 6) * ps * ps * ps
            + pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2)
            + pp(1) * pp(2) * bb(1) * bb(2) * bb(3) * bb(4)
        )
        assert K == K8  # eq. CS8 exactly

        top = pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2) * bb(3) * bb(4)
        L_int_printed = -(
            pp(1) * pp(2) * bb(1) * bb(2)
            + pp(1) * pp(2) * pp(3) * pp(4) * bb(1) * bb(2)
            + pp(1) * pp(2) * bb(1) * bb(2) * bb(3) * bb(4)
            - pp(1) * pp(2) * pp(3) * bb(1) * bb(2) * bb(3)
            - pp(1) * pp(2) * pp(4) * bb(1) * bb(2) * bb(4)
            - top
        )
        L = extract_action(perm, signs)
        # exact result: printed CS9 with the top coefficient corrected from 1 to 2
        assert L == -ps + L_int_printed + top
        # and the printed version provably does NOT satisfy its own defining
        # relation CS3, exp{psi psibar - L_int} = K -- a typo in the paper
        assert exp(ps - L_int_printed) != K8
        assert exp(ps - (L_int_printed + top)) == K8


# -- pipeline consistency ------------------------------------------------------------


class TestPipeline:
    def test_step_operator_roundtrip_rule891(self):
        perm = enumerate_conditional_rules()[891]
        signs = block_sign_table(perm, n_species=2)
        p2, s2 = step_operator_from_factor(local_factor(perm, signs), 4)
        assert np.array_equal(p2, perm) and np.array_equal(s2, signs)

    def test_exp_of_minus_action_is_local_factor(self):
        for idx in (891, 109):
            perm = enumerate_conditional_rules()[idx]
            signs = block_sign_table(perm, n_species=2)
            L = extract_action(perm, signs)
            assert exp(-L) == local_factor(perm, signs)
            assert L.is_even()  # R4.5's Grassmann face (2111.06728 eq. 4)

    def test_sign_gauge_matches_fock_signed_lift(self):
        # block_sign_table must reproduce fock/signed.py's dense matrix exactly
        for perm in (
            enumerate_conditional_rules()[891],
            enumerate_conditional_rules()[109],
            np.array([((v & 1) << 2) | ((v >> 2) & 1) | (((v >> 1) & 1) << 3)
                      | (((v >> 3) & 1) << 1) for v in range(16)]),
        ):
            signs = block_sign_table(perm, n_species=2)
            dense = SignedBlockCycle(
                n_sites=2, block_perm=perm, boundary="open"
            ).dense_substep(0)
            for v in range(16):
                assert dense[int(perm[v]), v] == signs[v]

    def test_parity_violation_rejected(self):
        perm = np.arange(16)
        perm[0], perm[1] = 1, 0  # creates/destroys a single fermion
        with pytest.raises(ValueError, match="parity"):
            block_sign_table(perm, n_species=2)

