from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from mathpsg.live_catalogue import CatalogueError, LiveCatalogue
from mathpsg.local_gap import probe_gap


ROOT = Path(__file__).resolve().parents[1] / "mathpsg" / "_assets"


class LiveCatalogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalogue = LiveCatalogue(probe_gap(), repository_root=ROOT)

    def test_fresh_sg1_geometry_is_labelled_without_certification_data(self) -> None:
        records = self.catalogue.records(1)
        self.assertEqual(len(records), 1)
        self.assertEqual(self.catalogue.resolve(1, "a"), records[0])
        self.assertEqual(self.catalogue.resolve(1, "1A"), records[0])
        payload = repr(records[0]).lower()
        for forbidden in ("sha256", "certificate", "digest", "version"):
            self.assertNotIn(forbidden, payload)

    def test_unknown_label_is_not_guessed(self) -> None:
        with self.assertRaisesRegex(CatalogueError, "Wyckoff label"):
            self.catalogue.resolve(1, "not-a-label")

    def test_malformed_gap_json_is_rejected(self) -> None:
        runtime = probe_gap()

        def malformed(arguments, **keywords):
            output = Path(arguments[arguments.index("--json-output") + 1])
            output.write_text("not-json", encoding="utf-8")
            return subprocess.CompletedProcess(arguments, 0, b"", b"")

        with mock.patch("mathpsg.live_catalogue.subprocess.run", malformed):
            with self.assertRaisesRegex(CatalogueError, "computation data"):
                LiveCatalogue(runtime, repository_root=ROOT).records(1)

    def test_runtime_exporter_contains_no_exact_version_or_hash_gate(self) -> None:
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "gap" / "catalogue" / "export_one.g",
                ROOT / "gap" / "catalogue" / "lib" / "normalize_affine.g",
            )
        ).lower()
        self.assertNotIn("sha256", sources)
        self.assertNotIn("packageinfo", sources)
        self.assertNotIn("gapinfo.version", sources)
        self.assertNotIn("=4.", sources)

    def test_letter_table_is_unhashed_and_covers_all_space_groups(self) -> None:
        value = json.loads(
            (ROOT / "resources" / "wyckoff-labels.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(value["groups"]), 230)
        self.assertNotIn("sha256", json.dumps(value).lower())


if __name__ == "__main__":
    unittest.main()
