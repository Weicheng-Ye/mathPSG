from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import inspect
import json
from pathlib import Path
import tomllib
import unittest

import mathpsg

from tests.support.physical_results import (
    RESULT_FIELDS,
    canonical_result,
    forbidden_public_paths,
    load_physical_cases,
    result_field_names,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "physical_results.json"
MINIMAL_SIGNATURE = (
    "it_number",
    "wps",
    "igg",
    "time_reversal",
    "setting",
    "details",
    "gap",
    "timeout",
)
class MinimalPackageSurfaceTests(unittest.TestCase):
    def test_classify_is_the_only_exported_function(self) -> None:
        self.assertEqual(
            tuple(mathpsg.__all__),
            ("classify", "ClassificationResult", "ClassificationError"),
        )
        exported_functions = tuple(
            name for name in mathpsg.__all__ if inspect.isfunction(getattr(mathpsg, name))
        )
        self.assertEqual(exported_functions, ("classify",))

    def test_classify_signature_has_no_cache_or_validation_controls(self) -> None:
        signature = inspect.signature(mathpsg.classify)
        self.assertEqual(tuple(signature.parameters), MINIMAL_SIGNATURE)
        with self.assertRaises(TypeError):
            signature.bind(1, ["a"], cache=ROOT / "cache")
        with self.assertRaises(TypeError):
            signature.bind(1, ["a"], validate=False)

    def test_distribution_has_no_cli(self) -> None:
        self.assertIsNone(importlib.util.find_spec("mathpsg.cli"))
        self.assertIsNone(importlib.util.find_spec("mathpsg.__main__"))
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertNotIn("scripts", metadata["project"])


class PhysicalOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_physical_cases(FIXTURE)
        cls.results = []
        for case in cls.cases:
            cls.results.append(
                mathpsg.classify(*case["arguments"], **case["keywords"], timeout=900)
            )

    def test_representative_matrix_matches_pre_refactor_physics(self) -> None:
        for case, result in zip(self.cases, self.results, strict=True):
            with self.subTest(case=case["name"]):
                self.assertEqual(canonical_result(result), case["expected"])

    def test_result_has_exactly_five_immutable_fields(self) -> None:
        for case, result in zip(self.cases, self.results, strict=True):
            with self.subTest(case=case["name"]):
                self.assertEqual(result_field_names(result), RESULT_FIELDS)
                for removed in (
                    "certification_status",
                    "runtime",
                    "cache",
                    "backend",
                ):
                    self.assertFalse(hasattr(result, removed))
                with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
                    result.class_count = 0

    def test_public_result_contains_physical_values_only(self) -> None:
        for case, result in zip(self.cases, self.results, strict=True):
            with self.subTest(case=case["name"]):
                plain = canonical_result(result)
                self.assertEqual(forbidden_public_paths(plain), ())
                json.dumps(plain, allow_nan=False)

    def test_details_default_to_none(self) -> None:
        result = mathpsg.classify(1, ["a"], igg="Z2", timeout=900)
        self.assertIsNone(result.details)


class ResidualQuotientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.z2 = mathpsg.classify(
            2,
            ["a", "b"],
            igg="Z2",
            time_reversal=True,
            details=True,
            timeout=900,
        )
        cls.u1 = mathpsg.classify(
            221,
            ["a"],
            igg="U1",
            details=True,
            timeout=900,
        )

    def test_z2_residual_action_reduces_the_framed_count(self) -> None:
        self.assertEqual(self.z2.class_count, 2176)
        self.assertFalse(self.z2.continuous)
        self.assertEqual(
            dict(self.z2.details["quotient"]),
            {
                "framed_finite_cardinality": 3328,
                "unframed_finite_cardinality": 2176,
                "continuous_family_count": 0,
            },
        )
        self.assertEqual(len(self.z2.summaries), 13)
        self.assertEqual(
            sum(
                summary["unframed_finite_cardinality"] == 128
                for summary in self.z2.summaries
            ),
            9,
        )

    def test_finite_u1_result_runs_the_weyl_quotient(self) -> None:
        self.assertEqual(self.u1.class_count, 40)
        self.assertFalse(self.u1.continuous)
        self.assertEqual(
            dict(self.u1.details["quotient"]),
            {
                "framed_finite_cardinality": 48,
                "unframed_finite_cardinality": 40,
                "continuous_family_count": 0,
            },
        )
        self.assertEqual(len(self.u1.summaries), 4)
        for summary in self.u1.summaries:
            self.assertEqual(summary["free_rank"], 0)
        self.assertEqual(
            sorted(summary["finite_class_count"] for summary in self.u1.summaries),
            [8, 8, 8, 24],
        )
        self.assertEqual(
            sum(
                summary["torsion_orders"] == (2, 2, 6)
                for summary in self.u1.summaries
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
