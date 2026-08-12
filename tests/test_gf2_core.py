from __future__ import annotations

from dataclasses import fields
import unittest

from mathpsg.gf2 import (
    GF2AffineSolution,
    GF2Inconsistency,
    GF2Quotient,
    GF2Reduction,
    MatrixGF2,
    quotient_basis,
    rref,
    solve_affine,
)


class GF2CoreTests(unittest.TestCase):
    def test_rref_exposes_only_the_numerical_reduction(self) -> None:
        reduction = rref(((1, 1, 0), (1, 0, 1)))

        self.assertEqual(
            reduction.reduced,
            MatrixGF2(((1, 0, 1), (0, 1, 1))),
        )
        self.assertEqual(reduction.pivots, (0, 1))
        self.assertEqual(tuple(field.name for field in fields(GF2Reduction)), (
            "reduced",
            "pivots",
        ))

    def test_affine_solve_returns_the_canonical_solution_space(self) -> None:
        solution = solve_affine(((1, 1, 0), (0, 1, 1)), (1, 0))

        self.assertIsInstance(solution, GF2AffineSolution)
        assert isinstance(solution, GF2AffineSolution)
        self.assertEqual(solution.basepoint, (1, 0, 0))
        self.assertEqual(solution.kernel_basis, ((1, 1, 1),))

    def test_affine_inconsistency_is_a_marker_without_a_witness_payload(self) -> None:
        solution = solve_affine(((1,), (1,)), (0, 1))

        self.assertIsInstance(solution, GF2Inconsistency)
        self.assertEqual(fields(GF2Inconsistency), ())

    def test_quotient_returns_only_the_decomposition_used_by_compute(self) -> None:
        cycles = MatrixGF2(((1, 0), (0, 1), (0, 0)), column_count=2)
        boundaries = MatrixGF2(((1,), (0,), (0,)), column_count=1)

        quotient = quotient_basis(cycles, boundaries)

        self.assertEqual(quotient.ambient_dimension, 3)
        self.assertEqual(quotient.boundary_basis, ((1, 0, 0),))
        self.assertEqual(quotient.representatives, ((0, 1, 0),))
        self.assertEqual(
            tuple(field.name for field in fields(GF2Quotient)),
            ("ambient_dimension", "boundary_basis", "representatives"),
        )

    def test_empty_equation_system_preserves_its_unknown_dimension(self) -> None:
        solution = solve_affine(MatrixGF2((), column_count=2), ())

        self.assertEqual(
            solution,
            GF2AffineSolution((0, 0), ((1, 0), (0, 1))),
        )


if __name__ == "__main__":
    unittest.main()
