from __future__ import annotations

from fractions import Fraction
import unittest
from unittest.mock import patch

import mathpsg.integer_linalg as integer_linalg
import mathpsg.torus as torus
from mathpsg.integer_linalg import MatrixZ, zero_matrix
from mathpsg.torus import Phase, TorusObstruction, TorusSolution


class TorusCoreTests(unittest.TestCase):
    def test_mixed_compact_group_and_raw_coordinates(self) -> None:
        solution = torus.solve_torus_quotient(
            MatrixZ(((2, 0),)),
            zero_matrix(2, 0),
            (Phase(Fraction(0)),),
        )

        self.assertIsInstance(solution, TorusSolution)
        assert isinstance(solution, TorusSolution)
        self.assertEqual(solution.group.free_rank, 1)
        self.assertEqual(solution.group.torsion_orders, (2,))
        self.assertEqual(solution.group.dual_generators, ((0, 1), (1, 0)))
        self.assertEqual(
            torus.raw_torsor_point(
                solution,
                (Phase(Fraction(1, 3)),),
                (1,),
            ),
            (Phase(Fraction(1, 2)), Phase(Fraction(1, 3))),
        )

    def test_nonzero_zero_row_offset_is_an_obstruction(self) -> None:
        result = torus.solve_torus_quotient(
            zero_matrix(1, 1),
            zero_matrix(1, 0),
            (Phase(Fraction(1, 2)),),
        )

        self.assertIsInstance(result, TorusObstruction)

    def test_solver_runs_each_essential_smith_reduction_once(self) -> None:
        torus_smith_form = torus.smith_form
        kernel_smith_form = integer_linalg.smith_form
        with (
            patch.object(torus, "smith_form", wraps=torus_smith_form) as torus_smith,
            patch.object(
                integer_linalg,
                "smith_form",
                wraps=kernel_smith_form,
            ) as kernel_smith,
        ):
            result = torus.solve_torus_quotient(
                MatrixZ(((2, 0),)),
                zero_matrix(2, 0),
                (Phase(Fraction(0)),),
            )

        self.assertIsInstance(result, TorusSolution)
        self.assertEqual(torus_smith.call_count, 2)
        self.assertEqual(kernel_smith.call_count, 1)


if __name__ == "__main__":
    unittest.main()
