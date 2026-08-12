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
from psgmath.host_classifier_backend import (
    HostNativeSourceEvidence,
    build_host_source_evidence,
    verify_host_source_evidence,
)
from psgmath.live_catalogue import LiveCatalogue
from psgmath.local_gap import GapRuntimeError, probe_gap


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


if __name__ == "__main__":
    unittest.main()
