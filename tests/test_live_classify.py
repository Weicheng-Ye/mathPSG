from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from psgmath.catalogue_loader import CatalogueIndex
from psgmath.certified_classifier import (
    ArtifactPlan,
    classify_request,
    verify_u1_sector_coverage,
)
from psgmath.classifier_cache import ClassifierCache
from psgmath.host_classifier_backend import HostNativeClassifierBackend
from psgmath.live_catalogue import LiveCatalogue
from psgmath.live_classify import resolve_occupancy_request
from psgmath.local_gap import probe_gap
from psgmath.query import make_diagnostic_verified_catalogue


ROOT = Path(__file__).resolve().parents[1]


class LiveJointZ2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = probe_gap()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        cls.catalogue = LiveCatalogue(
            cls.runtime,
            cache_root=cls.root / "catalogue-cache",
            repository_root=ROOT,
        )
        cls.backend = HostNativeClassifierBackend(
            runtime=cls.runtime,
            repository_root=ROOT,
        )
        cls.verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(cls.catalogue.records(1)), backend=cls.backend
        )

    def _classify(self, labels: list[str], *, time_reversal: bool = False):
        request = resolve_occupancy_request(
            1,
            labels,
            igg="Z2",
            time_reversal=time_reversal,
            setting=None,
            catalogue=self.catalogue,
        )
        cache = ClassifierCache(
            self.root
            / ("cache-" + "-".join(labels) + ("-graded" if time_reversal else ""))
        )
        return request, classify_request(
            request,
            self.verified,
            cache=cache,
            timeout_seconds=300,
        )

    def test_sg1_a_z2_returns_one_complete_joint_layer(self) -> None:
        request, result = self._classify(["a"])

        self.assertEqual(result.record.layer.status, "complete")
        self.assertFalse(result.record.layer.failures)
        self.assertTrue(result.framed_strata)
        self.assertTrue(all(len(item.skeleton_ids) == 1 for item in result.framed_strata))
        self.assertEqual(len(request.orbits), 1)

    def test_repeated_a_is_one_joint_solve_with_two_instances(self) -> None:
        request, result = self._classify(["a", "a"])

        self.assertEqual(result.record.layer.status, "complete")
        self.assertEqual(
            tuple(item.instance_id for item in request.orbits),
            ("atom-0000", "atom-0001"),
        )
        self.assertTrue(
            all(len(item.skeleton_ids) == 2 for item in result.framed_strata)
        )

    def test_sg1_onsite_time_z2_applies_exact_centralizer_actions(self) -> None:
        _, result = self._classify(["a"], time_reversal=True)

        self.assertEqual(result.record.layer.status, "complete")
        self.assertTrue(result.framed_strata)
        self.assertTrue(
            all(item.certificate.provenance == "diagnostic" for item in result.framed_strata)
        )
        self.assertTrue(
            any(item.certificate.centralizer_actions for item in result.framed_strata)
        )

    def test_warm_cache_replays_without_launching_gap_again(self) -> None:
        request = resolve_occupancy_request(
            1, ["a"], igg="Z2", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        cache_root = self.root / "cache-replay-without-gap"
        first = classify_request(
            request,
            self.verified,
            cache=ClassifierCache(cache_root),
            timeout_seconds=300,
        )
        self.assertEqual(first.record.layer.status, "complete")

        replay_backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        replay_catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=replay_backend
        )
        with patch(
            "psgmath.host_classifier_backend.build_host_source_evidence",
            side_effect=AssertionError("cache replay must not execute GAP"),
        ) as launcher:
            second = classify_request(
                request,
                replay_catalogue,
                cache=ClassifierCache(cache_root),
                timeout_seconds=300,
            )

        launcher.assert_not_called()
        self.assertEqual(second.record.layer.status, "complete")
        self.assertEqual(first.record, second.record)

    def test_cache_rejects_self_rehashed_foreign_launcher_identity(self) -> None:
        request = resolve_occupancy_request(
            1, ["a"], igg="Z2", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        cache_root = self.root / "cache-forged-launcher"
        first = classify_request(
            request,
            self.verified,
            cache=ClassifierCache(cache_root),
            timeout_seconds=300,
        )
        self.assertEqual(first.record.layer.status, "complete")
        ambient_files = tuple((cache_root / "ambient-resolution").glob("*.json"))
        self.assertEqual(len(ambient_files), 1)
        artifact_path = ambient_files[0]
        wrapper = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload = wrapper["payload"]
        attestation = payload["source"]["member_attestations"][0]
        attestation["resolved_launcher_digest"] = "sha256:" + "1" * 64
        core = {
            key: value
            for key, value in attestation.items()
            if key not in {"attestation_id", "record_type", "schema_version"}
        }
        attestation["attestation_id"] = "sha256:" + hashlib.sha256(
            b"mathpsg-task5-launcher-execution-attestation-v1|"
            + json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        payload_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        wrapper["artifact_digest"] = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
        artifact_path.write_text(
            json.dumps(
                wrapper,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        replay_backend = HostNativeClassifierBackend(
            runtime=self.runtime, repository_root=ROOT
        )
        replay_catalogue = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=replay_backend
        )
        with patch(
            "psgmath.host_classifier_backend.build_host_source_evidence",
            side_effect=AssertionError("forged cache must not execute GAP"),
        ):
            attacked = classify_request(
                request,
                replay_catalogue,
                cache=ClassifierCache(cache_root),
                timeout_seconds=300,
            )
        self.assertEqual(attacked.record.layer.status, "failed")
        self.assertEqual(
            attacked.record.layer.failures[0].stage, "ambient_resolution"
        )

    def test_distinct_a_b_and_b_a_each_use_one_ordered_joint_solve(self) -> None:
        records = self.catalogue.records(2)
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(records), backend=self.backend
        )
        observed: list[tuple[str, ...]] = []
        original = self.backend.relative_layer_plan

        def capture(backend, request, resolved, ambient, locals_, inclusions, timeout):
            observed.append(tuple(item.instance_id for item in inclusions))
            return original(
                request, resolved, ambient, locals_, inclusions, timeout
            )

        for labels in (["a", "b"], ["b", "a"]):
            with self.subTest(labels=labels), patch.object(
                HostNativeClassifierBackend,
                "relative_layer_plan",
                autospec=True,
                side_effect=capture,
            ) as relative:
                request = resolve_occupancy_request(
                    2,
                    labels,
                    igg="Z2",
                    time_reversal=False,
                    setting=None,
                    catalogue=self.catalogue,
                )
                result = classify_request(
                    request,
                    verified,
                    cache=ClassifierCache(
                        self.root / ("cache-sg2-" + "-".join(labels))
                    ),
                    timeout_seconds=300,
                )
                self.assertEqual(relative.call_count, 1)
                self.assertEqual(result.record.layer.status, "complete")
                self.assertTrue(
                    all(len(item.skeleton_ids) == 2 for item in result.framed_strata)
                )
                self.assertEqual(
                    tuple(item.instance_id for item in request.orbits),
                    ("atom-0000", "atom-0001"),
                )
                self.assertNotEqual(
                    request.orbits[0].wyckoff_id,
                    request.orbits[1].wyckoff_id,
                )
        self.assertEqual(
            observed,
            [("atom-0000", "atom-0001"), ("atom-0000", "atom-0001")],
        )

    def test_sg1_a_u1_returns_exhaustive_complete_sector_coverage(self) -> None:
        request = resolve_occupancy_request(
            1, ["a"], igg="U1", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        result = classify_request(
            request,
            self.verified,
            cache=ClassifierCache(self.root / "cache-sg1-u1"),
            timeout_seconds=300,
        )

        self.assertEqual(result.record.layer.status, "complete")
        self.assertFalse(result.record.layer.failures)
        self.assertTrue(result.framed_strata)
        self.assertTrue(
            all(len(item.skeleton_ids) == 1 for item in result.framed_strata)
        )
        global_weyl = tuple(
            item
            for item in result.residual_groupoid.arrows
            if item.kind == "global_weyl"
        )
        self.assertTrue(global_weyl)
        self.assertTrue(
            all(item.diagnostic for item in global_weyl)
        )

    def _assert_u1_joint_orbits(self, it_number: int, labels: list[str]) -> None:
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(it_number)), backend=self.backend
        )
        request = resolve_occupancy_request(
            it_number, labels, igg="U1", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        materials = []
        original = self.backend.relative_layer_plan

        def capture(
            backend, requested, resolved, ambient, local_rows, inclusions, timeout
        ):
            plan = original(
                requested, resolved, ambient, local_rows, inclusions, timeout
            )

            def verify(payload):
                material = plan.verify(payload)
                materials.append(material)
                return material

            return ArtifactPlan(
                build=plan.build, verify=verify, plan_digest=plan.plan_digest
            )

        with patch.object(
            HostNativeClassifierBackend,
            "relative_layer_plan",
            autospec=True,
            side_effect=capture,
        ) as relative:
            result = classify_request(
                request,
                verified,
                cache=ClassifierCache(self.root / f"cache-u1-joint-sg{it_number}"),
                timeout_seconds=300,
            )

        self.assertEqual(relative.call_count, 1)
        self.assertEqual(len(materials), 1)
        self.assertEqual(result.record.layer.status, "complete")
        material = materials[0]
        self.assertIsNotNone(material.u1_sector_coverage)
        expected_instances = ("atom-0000", "atom-0001")
        for outcome in material.u1_sector_coverage.outcomes:
            self.assertIsNotNone(outcome.problem)
            problem = outcome.problem
            self.assertEqual(len(outcome.skeleton_ids), 2)
            self.assertEqual(
                tuple(item.instance_id for item in problem.local_data),
                expected_instances,
            )
            self.assertEqual(
                tuple(item.instance_id for item in problem.bindings),
                expected_instances,
            )
            self.assertEqual(
                tuple(
                    item.instance_id
                    for item in problem.relative_problem.restrictions
                ),
                expected_instances,
            )
        self.assertTrue(material.global_weyl_data)
        for _, row in material.global_weyl_data:
            self.assertEqual(
                tuple(item.instance_id for item in row), expected_instances
            )
            self.assertTrue(all(item.evaluator.diagnostic for item in row))
            self.assertTrue(
                all(
                    item.evaluator.authority is None
                    and item.evaluator.equivalence is None
                    for item in row
                )
            )
        with self.assertRaisesRegex(ValueError, "not release authority"):
            verify_u1_sector_coverage(
                material.u1_sector_coverage, allow_diagnostic=False
            )
        self.assertEqual(
            tuple(item.instance_id for item in request.orbits), expected_instances
        )
        self.assertEqual(
            tuple(
                item.inclusion.inclusion_id
                for item in material.u1_sector_coverage.outcomes[0].problem.local_data
            ),
            tuple(item.wyckoff_id for item in request.orbits),
        )

    def test_u1_repeated_orbits_stay_in_one_ordered_joint_problem(self) -> None:
        self._assert_u1_joint_orbits(1, ["a", "a"])

    def test_u1_distinct_orbits_use_one_ordered_joint_problem(self) -> None:
        self._assert_u1_joint_orbits(2, ["a", "b"])

    def test_u1_reverse_distinct_request_preserves_caller_order(self) -> None:
        request = resolve_occupancy_request(
            2, ["b", "a"], igg="U1", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        self.assertEqual(
            tuple(item.wyckoff_id for item in request.orbits),
            tuple(
                item.wyckoff_id
                for item in reversed(
                    resolve_occupancy_request(
                        2, ["a", "b"], igg="U1", time_reversal=False,
                        setting=None, catalogue=self.catalogue,
                    ).orbits
                )
            ),
        )
        self.assertEqual(
            tuple(item.instance_id for item in request.orbits),
            ("atom-0000", "atom-0001"),
        )

    def test_sg1_onsite_time_u1_returns_complete_sector_coverage(self) -> None:
        request = resolve_occupancy_request(
            1, ["a"], igg="U1", time_reversal=True, setting=None,
            catalogue=self.catalogue,
        )
        result = classify_request(
            request,
            self.verified,
            cache=ClassifierCache(self.root / "cache-sg1-graded-u1"),
            timeout_seconds=300,
        )

        self.assertEqual(result.record.layer.status, "complete")
        self.assertFalse(result.record.layer.failures)
        self.assertTrue(
            result.framed_strata or result.record.layer.obstructed_branches
        )


if __name__ == "__main__":
    unittest.main()
