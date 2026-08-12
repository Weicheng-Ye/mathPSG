from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from mathpsg.live_catalogue import LiveCatalogue
from mathpsg.live_evidence import build_evidence
from mathpsg.local_gap import probe_gap


ROOT = Path(__file__).resolve().parents[1]


class LiveEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = probe_gap()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.catalogue = LiveCatalogue(
            cls.runtime,
            cache_root=Path(cls.temporary.name),
            repository_root=ROOT,
        )
        cls.records = cls.catalogue.records(1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_spatial_evidence_is_host_native_and_replayed(self) -> None:
        batch = build_evidence(
            self.records,
            runtime=self.runtime,
            repository_root=ROOT,
            time_reversal=False,
        )
        self.assertEqual(batch.certification_status, "host-native")
        self.assertFalse(batch.time_reversal)
        self.assertEqual(batch.member_ids, tuple(r.wyckoff_id for r in self.records))
        self.assertEqual(batch.response.status, "conversion_only")
        self.assertIsNotNone(batch.affine_certificate)
        encoded = json.loads(batch.canonical_data)
        environment = encoded["environment"]
        self.assertEqual(environment["execution_mode"], "diagnostic_local")
        self.assertNotIn("oci_image_digest", environment)
        self.assertNotIn("lock_digest", environment)
        self.assertTrue(
            all(
                set(package) == {"name", "version"}
                for package in environment["packages"]
            )
        )

    def test_onsite_time_preserves_the_member_universe(self) -> None:
        batch = build_evidence(
            self.records,
            runtime=self.runtime,
            repository_root=ROOT,
            time_reversal=True,
        )
        self.assertTrue(batch.time_reversal)
        self.assertEqual(batch.member_ids, tuple(r.wyckoff_id for r in self.records))


if __name__ == "__main__":
    unittest.main()
