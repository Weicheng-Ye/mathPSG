from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psgmath.catalogue_loader import CatalogueIndex, load_ndjson
from psgmath.certified_classifier import (
    BackendIdentity,
    ClassifierBackendAuthority,
    classify_request,
)
from psgmath.classification_schema import ClassificationRequest, OrbitInstance
from psgmath.classifier_cache import ClassifierCache
from psgmath.query import (
    make_diagnostic_verified_catalogue,
    resolve_request_orbits,
    verify_verified_catalogue,
)
from psgmath.live_catalogue import CatalogueError, LiveCatalogue
from psgmath.live_classify import resolve_occupancy_request
from psgmath.local_gap import probe_gap


ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(suffix: str) -> BackendIdentity:
    return BackendIdentity(
        gap_environment_digest=_digest("gap-env:" + suffix),
        affine_pcp_conversion_digest=_digest("pcp-conversion:" + suffix),
        affine_pcp_transport_digest=_digest("pcp-transport:" + suffix),
        target_model_digest=_digest("target-model:" + suffix),
        local_library_digest=_digest("local-library:" + suffix),
        ambient_algorithm_digest=_digest("ambient:" + suffix),
        local_algorithm_digest=_digest("local:" + suffix),
        inclusion_algorithm_digest=_digest("inclusion:" + suffix),
        relative_algorithm_digest=_digest("relative:" + suffix),
    )


class CapturingBackend(ClassifierBackendAuthority):
    """Stop after recording the real orchestrator's joint ambient request."""

    def __init__(self) -> None:
        self.identity = _identity("capture")
        self.ambient_calls: list[tuple[str, ...]] = []

    def ambient_resolution_plan(self, request, resolved_orbits, timeout_seconds):
        self.ambient_calls.append(tuple(item.instance_id for item in resolved_orbits))
        raise RuntimeError("intentional ambient stop")


class UnsupportedReleaseBackend:
    def __init__(self) -> None:
        self.identity = _identity("unsupported-release")
        self.task5_release_store = object()


class RestoredOrchestrationTests(unittest.TestCase):
    """Catch loss of the live-record to joint-classifier authority bridge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.live = LiveCatalogue(
            probe_gap(),
            cache_root=Path(cls.temporary.name),
            repository_root=ROOT,
        )

    def test_live_records_form_a_diagnostic_verified_catalogue(self) -> None:
        records = self.live.records(1)

        catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(records),
            backend=None,
        )

        self.assertFalse(catalogue.release_complete)
        self.assertEqual(tuple(catalogue.index), records)
        self.assertEqual(catalogue.candidate_ids(1, "1"), (records[0].wyckoff_id,))

    def test_copied_catalogue_does_not_inherit_factory_authority(self) -> None:
        catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.live.records(1)),
            backend=None,
        )

        with self.assertRaisesRegex(TypeError, "factory-issued|authority"):
            verify_verified_catalogue(copy.copy(catalogue))

    def test_backend_identity_drift_invalidates_catalogue(self) -> None:
        backend = CapturingBackend()
        catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.live.records(1)),
            backend=backend,
        )
        backend.identity = replace(
            backend.identity,
            relative_algorithm_digest=_digest("relative:mutated"),
        )

        with self.assertRaisesRegex(ValueError, "backend identity"):
            verify_verified_catalogue(catalogue)

    def test_repeated_family_instances_keep_distinct_symbol_names(self) -> None:
        records = self.live.records(70)
        generic = max(records, key=lambda item: item.orbit["parameter_dimension"])
        request = ClassificationRequest(
            1,
            70,
            str(generic.space_group["setting"]),
            "Z2",
            False,
            (
                OrbitInstance("atom-0", generic.wyckoff_id, "family"),
                OrbitInstance("atom-1", generic.wyckoff_id, "family"),
            ),
        )
        catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(records),
            backend=None,
        )

        resolved = resolve_request_orbits(request, catalogue)

        self.assertEqual(tuple(item.instance_id for item in resolved), ("atom-0", "atom-1"))
        self.assertTrue(resolved[0].symbolic_parameters)
        self.assertTrue(set(resolved[0].symbolic_parameters).isdisjoint(
            resolved[1].symbolic_parameters
        ))

    def test_repeated_occupancy_reaches_one_joint_backend_call(self) -> None:
        records = self.live.records(1)
        backend = CapturingBackend()
        catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(records),
            backend=backend,
        )
        request = ClassificationRequest(
            1,
            1,
            "1",
            "Z2",
            False,
            (
                OrbitInstance("atom-0", records[0].wyckoff_id, "family"),
                OrbitInstance("atom-1", records[0].wyckoff_id, "family"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = classify_request(
                request,
                catalogue,
                cache=ClassifierCache(Path(directory).resolve()),
            )

        self.assertEqual(backend.ambient_calls, [("atom-0", "atom-1")])
        self.assertEqual(result.record.layer.status, "failed")
        self.assertEqual(result.record.layer.failures[0].stage, "ambient_resolution")

    def test_release_store_path_fails_with_standalone_domain_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "release.*unavailable"):
            make_diagnostic_verified_catalogue(
                CatalogueIndex(self.live.records(1)),
                backend=UnsupportedReleaseBackend(),
            )

    def test_release_catalogue_path_fails_with_standalone_domain_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = root / "wyckoff.ndjson"
            geometry.write_bytes(b"")
            geometry.with_name("manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "release.*unavailable"):
                tuple(load_ndjson(geometry))


class OccupancyRequestTests(unittest.TestCase):
    """Catch splitting one occupied-WP list into independent calculations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.catalogue = LiveCatalogue(
            probe_gap(),
            cache_root=Path(cls.temporary.name),
            repository_root=ROOT,
        )

    def test_a_and_b_are_one_ordered_joint_request(self) -> None:
        request = resolve_occupancy_request(
            99,
            ["a", "b"],
            igg="Z2",
            time_reversal=False,
            setting=None,
            catalogue=self.catalogue,
        )

        self.assertEqual(request.space_group, 99)
        self.assertEqual(request.setting_id, "1")
        self.assertEqual(len(request.orbits), 2)
        self.assertEqual(
            tuple(item.wyckoff_id for item in request.orbits),
            (
                self.catalogue.resolve(99, "a", "1").wyckoff_id,
                self.catalogue.resolve(99, "b", "1").wyckoff_id,
            ),
        )
        self.assertEqual(
            tuple(item.instance_id for item in request.orbits),
            ("atom-0000", "atom-0001"),
        )

    def test_repeated_a_retains_two_distinct_instances(self) -> None:
        request = resolve_occupancy_request(
            1,
            ["a", "a"],
            igg="Z2",
            time_reversal=False,
            setting=None,
            catalogue=self.catalogue,
        )

        self.assertEqual(len(request.orbits), 2)
        self.assertNotEqual(request.orbits[0].instance_id, request.orbits[1].instance_id)
        self.assertEqual(request.orbits[0].wyckoff_id, request.orbits[1].wyckoff_id)

    def test_one_string_is_one_occupied_position(self) -> None:
        request = resolve_occupancy_request(
            1,
            "a",
            igg="U1",
            time_reversal=True,
            setting="1",
            catalogue=self.catalogue,
        )

        self.assertEqual(len(request.orbits), 1)
        self.assertEqual(request.igg, "U1")
        self.assertTrue(request.time_reversal)

    def test_full_conventional_label_is_accepted(self) -> None:
        request = resolve_occupancy_request(
            1,
            "1a",
            igg="Z2",
            time_reversal=False,
            setting="1",
            catalogue=self.catalogue,
        )

        self.assertEqual(len(request.orbits), 1)
        self.assertEqual(
            request.orbits[0].wyckoff_id,
            self.catalogue.resolve(1, "a", "1").wyckoff_id,
        )

    def test_structurally_invalid_inputs_do_not_access_catalogue(self) -> None:
        cases = (
            ({"wps": [""], "igg": "Z2", "setting": None}, ValueError),
            ({"wps": ["aa"], "igg": "Z2", "setting": None}, ValueError),
            ({"wps": ["0a"], "igg": "Z2", "setting": None}, ValueError),
            ({"wps": ["a"], "igg": 1, "setting": None}, TypeError),
            ({"wps": ["a"], "igg": "Z2", "setting": " 1"}, ValueError),
            ({"wps": ["a"], "igg": "Z2", "setting": "setting/1"}, ValueError),
        )
        for values, error in cases:
            with self.subTest(values=values), patch.object(
                self.catalogue,
                "records",
                side_effect=AssertionError("invalid input reached catalogue"),
            ), self.assertRaises(error):
                resolve_occupancy_request(
                    1,
                    time_reversal=False,
                    catalogue=self.catalogue,
                    **values,
                )

    def test_explicit_mismatched_setting_is_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogueError, "no match"):
            resolve_occupancy_request(
                1,
                ["a"],
                igg="Z2",
                time_reversal=False,
                setting="2",
                catalogue=self.catalogue,
            )

    def test_invalid_public_arguments_fail_before_resolution(self) -> None:
        cases = (
            {"it_number": True, "wps": ["a"], "igg": "Z2", "time_reversal": False},
            {"it_number": 0, "wps": ["a"], "igg": "Z2", "time_reversal": False},
            {"it_number": 1, "wps": [], "igg": "Z2", "time_reversal": False},
            {"it_number": 1, "wps": ["a"], "igg": "SU2", "time_reversal": False},
            {"it_number": 1, "wps": ["a"], "igg": "Z2", "time_reversal": 1},
            {"it_number": 1, "wps": ["missing"], "igg": "Z2", "time_reversal": False},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(
                (TypeError, ValueError, RuntimeError)
            ):
                resolve_occupancy_request(
                    setting=None,
                    catalogue=self.catalogue,
                    **values,
                )


if __name__ == "__main__":
    unittest.main()
