from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psgmath.bar_evaluator import (
    replay_gap_inclusion_batch_artifact,
    verify_gap_batch_launcher_execution,
)
from psgmath.catalogue import catalogue_record_order_key
from psgmath.catalogue_loader import CatalogueIndex
from psgmath.certified_classifier import make_z2_local_skeleton_evidence
from psgmath.host_classifier_backend import (
    HostNativeClassifierBackend,
    HostNativeSourceEvidence,
    assemble_host_ambient_artifact,
    build_host_source_evidence,
    verify_host_source_evidence,
)
from psgmath.live_catalogue import LiveCatalogue
from psgmath.live_classify import resolve_occupancy_request
from psgmath.local_gap import GapRuntimeError, probe_gap
from psgmath.query import (
    make_diagnostic_verified_catalogue,
    resolve_request_orbits,
)


ROOT = Path(__file__).resolve().parents[1]


class HostSourceEvidenceTests(unittest.TestCase):
    """Catch collapsing repeated atom instances or skipping Task5 replay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = probe_gap()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.catalogue = LiveCatalogue(
            cls.runtime,
            cache_root=Path(cls.temporary.name),
            repository_root=ROOT,
        )
        cls.sg1 = cls.catalogue.records(1)[0]
        cls.evidence = build_host_source_evidence(
            (cls.sg1, cls.sg1),
            runtime=cls.runtime,
            time_reversal=False,
            timeout=300,
            repository_root=ROOT,
        )
        cls.graded_evidence = build_host_source_evidence(
            (cls.sg1,),
            runtime=cls.runtime,
            time_reversal=True,
            timeout=300,
            repository_root=ROOT,
        )

    def test_repeated_instances_share_one_exact_grouped_inclusion(self) -> None:
        evidence = self.evidence

        self.assertEqual(
            evidence.instance_wyckoff_ids,
            (self.sg1.wyckoff_id, self.sg1.wyckoff_id),
        )
        self.assertEqual(evidence.unique_inclusion_ids, (self.sg1.wyckoff_id,))
        self.assertEqual(evidence.certification_status, "host-native")
        self.assertFalse(evidence.time_reversal)
        self.assertEqual(evidence.task4_response.status, "conversion_only")
        self.assertEqual(len(evidence.task5_execution.member_executions), 1)
        self.assertFalse(
            evidence.task5_execution.member_executions[0].attestation.release_certified
        )
        self.assertEqual(
            tuple(member.inclusion_id for member in evidence.task5_replay.members),
            (self.sg1.wyckoff_id,),
        )

    def test_two_unique_inclusions_keep_instance_order_and_canonical_gap_order(self) -> None:
        first, second = self.catalogue.records(2)[:2]
        supplied = (second, first, second)
        evidence = build_host_source_evidence(
            supplied,
            runtime=self.runtime,
            time_reversal=False,
            timeout=300,
            repository_root=ROOT,
        )

        self.assertEqual(
            evidence.instance_wyckoff_ids,
            tuple(item.wyckoff_id for item in supplied),
        )
        self.assertEqual(
            evidence.unique_inclusion_ids,
            tuple(
                item.wyckoff_id
                for item in sorted((first, second), key=catalogue_record_order_key)
            ),
        )

    def test_copied_batch_execution_does_not_inherit_issuer_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "issued|registry"):
            verify_gap_batch_launcher_execution(
                copy.copy(self.evidence.task5_execution),
                require_release=False,
            )

    def test_source_evidence_requires_factory_issuance(self) -> None:
        evidence = self.evidence
        with self.assertRaisesRegex(ValueError, "factory|issued|seal"):
            HostNativeSourceEvidence(
                evidence.instance_wyckoff_ids,
                evidence.unique_inclusion_ids,
                evidence.time_reversal,
                evidence.task4_request,
                evidence.task4_response,
                evidence.task5_execution,
                evidence.task5_replay,
                evidence.provenance,
            )

        with self.assertRaisesRegex(ValueError, "factory|issued|registry"):
            verify_host_source_evidence(copy.copy(evidence))

    def test_source_provenance_is_recursively_immutable(self) -> None:
        with self.assertRaises(TypeError):
            self.evidence.provenance["gap"]["version"] = "forged"  # type: ignore[index]

    def test_nested_task4_request_mutation_fails_evidence_replay(self) -> None:
        evidence = self.evidence
        original = evidence.task4_request.time_reversal
        object.__setattr__(evidence.task4_request, "time_reversal", not original)
        self.addCleanup(
            object.__setattr__, evidence.task4_request, "time_reversal", original
        )

        with self.assertRaisesRegex(
            ValueError, "Task4 request|mutated|replay|registry"
        ):
            verify_host_source_evidence(evidence)

    def test_forged_runtime_is_rejected_before_gap_evidence_execution(self) -> None:
        forged = replace(self.runtime, gap_version="0.0")
        with patch(
            "psgmath.host_classifier_backend.build_evidence",
            side_effect=AssertionError("Task4 must not run"),
        ) as task4:
            with self.assertRaises(GapRuntimeError):
                build_host_source_evidence(
                    (self.sg1,),
                    runtime=forged,
                    time_reversal=False,
                    timeout=300,
                    repository_root=ROOT,
                )
        task4.assert_not_called()

    def test_mutated_batch_bytes_fail_pure_replay(self) -> None:
        value = json.loads(self.evidence.task5_execution.raw_output)
        value["batch_input_digest"] = "sha256:" + "0" * 64
        mutated = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "identity|request|batch"):
            replay_gap_inclusion_batch_artifact(
                self.evidence.task5_execution.spec,
                mutated,
            )

    def test_invalid_arguments_fail_before_external_execution(self) -> None:
        cases = (
            {"records": (), "time_reversal": False, "timeout": 300},
            {"records": (self.sg1,), "time_reversal": 0, "timeout": 300},
            {"records": (self.sg1,), "time_reversal": False, "timeout": 0},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(
                (TypeError, ValueError)
            ):
                build_host_source_evidence(
                    runtime=self.runtime,
                    repository_root=ROOT,
                    **values,
                )

    def test_spatial_z2_local_plan_is_exact_and_exhaustive(self) -> None:
        ambient = assemble_host_ambient_artifact(self.evidence)
        backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        request = resolve_occupancy_request(
            1,
            ["a"],
            igg="Z2",
            time_reversal=False,
            setting=None,
            catalogue=self.catalogue,
        )
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)),
            backend=backend,
        )
        resolved = resolve_request_orbits(request, verified)[0]

        plans = tuple(
            backend.local_skeleton_plans(request, resolved, ambient, 300)
        )

        self.assertEqual(len(plans), 1)
        evidence = plans[0].plan.verify(plans[0].plan.build())
        self.assertEqual(evidence.coefficient_kind, "Z2")
        self.assertTrue(evidence.skeletons)
        self.assertEqual(
            evidence.skeleton_ids,
            tuple(item.skeleton_id for item in evidence.skeletons),
        )
        with self.assertRaisesRegex(ValueError, "exhaustive|differs"):
            make_z2_local_skeleton_evidence(
                instance_id=evidence.instance_id,
                source_table=evidence.source_table,
                skeletons=evidence.skeletons[:-1],
                restricted_grade=evidence.restricted_grade,
                graded=False,
            )
        attacked = bytearray(plans[0].plan.build())
        attacked[-2] ^= 1
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            plans[0].plan.verify(bytes(attacked))

        with self.assertRaisesRegex(ValueError, "factory-issued"):
            from psgmath.host_classifier_backend import verify_host_ambient_artifact

            verify_host_ambient_artifact(copy.copy(ambient))

    def test_spatial_u1_local_plans_cover_every_ambient_rho(self) -> None:
        ambient = assemble_host_ambient_artifact(self.evidence)
        backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        request = resolve_occupancy_request(
            1,
            ["a"],
            igg="U1",
            time_reversal=False,
            setting=None,
            catalogue=self.catalogue,
        )
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=backend
        )
        resolved = resolve_request_orbits(request, verified)[0]

        plans = tuple(backend.local_skeleton_plans(request, resolved, ambient, 300))
        evidence = tuple(plan.plan.verify(plan.plan.build()) for plan in plans)

        self.assertTrue(evidence)
        self.assertEqual(
            len({item.ambient_rho.bits for item in evidence}), len(evidence)
        )
        self.assertTrue(all(item.coefficient_kind == "U1" for item in evidence))
        self.assertTrue(all(len(item.skeletons) == 1 for item in evidence))

    def test_graded_z2_local_plan_enumerates_full_spatial_cross_c2_library(self) -> None:
        ambient = assemble_host_ambient_artifact(
            self.graded_evidence,
            spatial_parent=assemble_host_ambient_artifact(self.evidence),
        )
        backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        request = resolve_occupancy_request(
            1,
            ["a"],
            igg="Z2",
            time_reversal=True,
            setting=None,
            catalogue=self.catalogue,
        )
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=backend
        )
        resolved = resolve_request_orbits(request, verified)[0]

        plan = tuple(backend.local_skeleton_plans(request, resolved, ambient, 300))[0]
        evidence = plan.plan.verify(plan.plan.build())

        self.assertTrue(evidence.graded)
        self.assertTrue(evidence.skeletons)
        self.assertTrue(all(item.time_orbit is not None for item in evidence.skeletons))

    def test_local_plan_rejects_ambient_from_opposite_time_reversal_mode(self) -> None:
        backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=backend
        )
        spatial_request = resolve_occupancy_request(
            1, ["a"], igg="Z2", time_reversal=False, setting=None,
            catalogue=self.catalogue,
        )
        graded_request = resolve_occupancy_request(
            1, ["a"], igg="Z2", time_reversal=True, setting=None,
            catalogue=self.catalogue,
        )
        spatial_resolved = resolve_request_orbits(spatial_request, verified)[0]
        graded_resolved = resolve_request_orbits(graded_request, verified)[0]
        spatial_ambient = assemble_host_ambient_artifact(self.evidence)
        graded_ambient = assemble_host_ambient_artifact(
            self.graded_evidence,
            spatial_parent=spatial_ambient,
        )

        with self.assertRaisesRegex(ValueError, "time-reversal|time reversal"):
            backend.local_skeleton_plans(
                graded_request, graded_resolved, spatial_ambient, 300
            )
        with self.assertRaisesRegex(ValueError, "time-reversal|time reversal"):
            backend.local_skeleton_plans(
                spatial_request, spatial_resolved, graded_ambient, 300
            )

    def test_graded_u1_local_plans_cover_every_ambient_rho(self) -> None:
        ambient = assemble_host_ambient_artifact(
            self.graded_evidence,
            spatial_parent=assemble_host_ambient_artifact(self.evidence),
        )
        backend = HostNativeClassifierBackend(
            runtime=self.runtime,
            repository_root=ROOT,
        )
        request = resolve_occupancy_request(
            1,
            ["a"],
            igg="U1",
            time_reversal=True,
            setting=None,
            catalogue=self.catalogue,
        )
        verified = make_diagnostic_verified_catalogue(
            CatalogueIndex(self.catalogue.records(1)), backend=backend
        )
        resolved = resolve_request_orbits(request, verified)[0]

        plans = tuple(backend.local_skeleton_plans(request, resolved, ambient, 300))
        evidence = tuple(plan.plan.verify(plan.plan.build()) for plan in plans)

        self.assertTrue(evidence)
        self.assertEqual(
            len({item.ambient_rho.bits for item in evidence}), len(evidence)
        )
        self.assertTrue(all(item.coefficient_kind == "U1" for item in evidence))
        self.assertTrue(all(item.graded for item in evidence))
        self.assertTrue(all(len(item.skeletons) == 1 for item in evidence))
        self.assertEqual(
            tuple(item.ambient_rho.bits for item in evidence),
            tuple(sorted(item.ambient_rho.bits for item in evidence)),
        )
        self.assertEqual(len({plan.plan.plan_digest for plan in plans}), len(plans))
        attacked = bytearray(plans[-1].plan.build())
        attacked[-2] ^= 1
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            plans[-1].plan.verify(bytes(attacked))


if __name__ == "__main__":
    unittest.main()
