from __future__ import annotations

from unittest import mock
import unittest

from mathpsg import compute


class Z2OrbitCountTests(unittest.TestCase):
    def test_no_shifts_leave_every_point_in_its_own_orbit(self) -> None:
        self.assertEqual(compute._z2_orbit_count(4, ()), 16)

    def test_dependent_shifts_reduce_by_their_rank(self) -> None:
        shifts = (
            (1, 0, 1, 0),
            (1, 0, 1, 0),
            (0, 1, 1, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
        )

        self.assertEqual(compute._z2_orbit_count(4, shifts), 4)

    def test_large_dimension_does_not_enumerate_points(self) -> None:
        dimension = 100_000
        with mock.patch.object(
            compute.itertools,
            "product",
            side_effect=AssertionError("Z2 quotient counting must not enumerate"),
        ):
            count = compute._z2_orbit_count(dimension, ())

        self.assertEqual(count, 1 << dimension)


if __name__ == "__main__":
    unittest.main()
