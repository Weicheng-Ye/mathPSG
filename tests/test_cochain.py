from __future__ import annotations

from fractions import Fraction
import unittest

from mathpsg.cochain import (
    RelativeCochainCoordinates,
    u1_basis_presentation,
    z2_basis_presentation,
)
from mathpsg.compute import (
    PhysicalClassification,
    PhysicalQuotient,
    Z2PhysicalStratum,
)
from mathpsg.integer_linalg import MatrixZ
from mathpsg.live_classify import _details, _z2_summary
from mathpsg.torus import (
    CompactGroupPresentation,
    Phase,
    PrimalTorsorChart,
    TorusSolution,
)


class Z2CochainBasisTests(unittest.TestCase):
    def test_zero_dimensional_quotient_has_no_generators(self) -> None:
        result = z2_basis_presentation(
            basepoint=(),
            quotient_basis=(),
            residual_shifts=(),
            coordinates=RelativeCochainCoordinates((), ()),
            labels=(),
        )

        self.assertEqual(result["basepoint"], ())
        self.assertEqual(result["basis"], ())

    def test_residual_shift_is_removed_before_generators_are_reported(self) -> None:
        coordinates = RelativeCochainCoordinates(
            ("a2:0", "a2:1"),
            (("l1:0",),),
        )

        result = z2_basis_presentation(
            basepoint=(1, 0, 0),
            quotient_basis=(
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ),
            residual_shifts=((1, 1, 0),),
            coordinates=coordinates,
            labels=("a",),
        )

        self.assertEqual(
            result,
            {
                "coordinate_blocks": {
                    "ambient_degree_2": ("a2:0", "a2:1"),
                    "local_degree_1": (
                        {
                            "orbit_index": 0,
                            "wp": "a",
                            "basis": ("l1:0",),
                        },
                    ),
                },
                "basepoint": (1, 0, 0),
                "basis": (
                    {
                        "kind": "torsion",
                        "order": 2,
                        "quotient_coordinates": (1, 0, 0),
                        "direction": (1, 0, 0),
                        "representative": (0, 0, 0),
                    },
                    {
                        "kind": "torsion",
                        "order": 2,
                        "quotient_coordinates": (0, 0, 1),
                        "direction": (0, 0, 1),
                        "representative": (1, 0, 1),
                    },
                ),
            },
        )


class U1CochainBasisTests(unittest.TestCase):
    def test_generator_free_torsor_has_an_empty_basis(self) -> None:
        solution = TorusSolution(
            basepoint=(Phase(Fraction(1, 3)),),
            group=CompactGroupPresentation(
                free_rank=0,
                torsion_orders=(),
                dual_generators=MatrixZ(((),), column_count=0),
            ),
            primal_chart=PrimalTorsorChart(
                raw_dimension=1,
                free_lifts=MatrixZ(((),), column_count=0),
                torsion_lifts=((),),
            ),
        )

        result = u1_basis_presentation(
            solution=solution,
            weyl_shift=(Phase(Fraction(0)),),
            coordinates=RelativeCochainCoordinates(("a2:0",), ()),
            labels=(),
        )

        self.assertEqual(result["basepoint_phases"], ("1/3",))
        self.assertEqual(result["basis"], ())

    def test_free_and_torsion_lifts_are_reported_exactly(self) -> None:
        solution = TorusSolution(
            basepoint=(Phase(Fraction(1, 4)), Phase(Fraction(1, 2))),
            group=CompactGroupPresentation(
                free_rank=1,
                torsion_orders=(2,),
                dual_generators=MatrixZ(((1, 0), (0, 1))),
            ),
            primal_chart=PrimalTorsorChart(
                raw_dimension=2,
                free_lifts=MatrixZ(((1,), (-2,))),
                torsion_lifts=(
                    (Phase(Fraction(1, 2)),),
                    (Phase(Fraction(0)),),
                ),
            ),
        )
        coordinates = RelativeCochainCoordinates(("a2:0",), (("l1:0",),))

        result = u1_basis_presentation(
            solution=solution,
            weyl_shift=(Phase(Fraction(0)), Phase(Fraction(1, 2))),
            coordinates=coordinates,
            labels=("a",),
        )

        self.assertEqual(
            result,
            {
                "coordinate_blocks": {
                    "ambient_degree_2": ("a2:0",),
                    "local_degree_1": (
                        {
                            "orbit_index": 0,
                            "wp": "a",
                            "basis": ("l1:0",),
                        },
                    ),
                },
                "basepoint_phases": ("1/4", "1/2"),
                "weyl_shift_phases": ("0", "1/2"),
                "basis": (
                    {
                        "kind": "free",
                        "parameter": "phi0",
                        "coefficients": (1, -2),
                    },
                    {
                        "kind": "torsion",
                        "order": 2,
                        "torsion_coordinates": (1,),
                        "direction_phases": ("1/2", "0"),
                        "representative_phases": ("3/4", "1/2"),
                    },
                ),
            },
        )


class CochainCoordinateTests(unittest.TestCase):
    def test_repeated_labels_remain_distinct_by_orbit_index(self) -> None:
        coordinates = RelativeCochainCoordinates(
            ("a2:0",),
            (("l1:0",), ("l1:1",)),
        )

        blocks = coordinates.mapping(("a", "a"))["local_degree_1"]

        self.assertEqual(tuple(block["wp"] for block in blocks), ("a", "a"))
        self.assertEqual(tuple(block["orbit_index"] for block in blocks), (0, 1))


class CochainDetailSerializationTests(unittest.TestCase):
    def test_cochain_basis_is_added_only_when_requested(self) -> None:
        coordinates = RelativeCochainCoordinates(("a2:0",), (("l1:0",),))
        stratum = Z2PhysicalStratum(
            basepoint=(1, 0),
            quotient_basis=((1, 0), (0, 1)),
            residual_shifts=((1, 1),),
            coordinates=coordinates,
            unframed_class_count=2,
        )
        physical = PhysicalClassification(
            (stratum,),
            PhysicalQuotient(2, False, 4),
        )
        summaries = (_z2_summary(stratum),)

        ordinary = _details(
            physical,
            summaries,
            labels=("a",),
            cochain=False,
        )
        explicit = _details(
            physical,
            summaries,
            labels=("a",),
            cochain=True,
        )

        self.assertNotIn("cochain", ordinary["strata"][0])
        self.assertEqual(
            explicit["strata"][0]["cochain"]["basepoint"],
            (1, 0),
        )
        self.assertEqual(len(explicit["strata"][0]["cochain"]["basis"]), 1)


if __name__ == "__main__":
    unittest.main()
