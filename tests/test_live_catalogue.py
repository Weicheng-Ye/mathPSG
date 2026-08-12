from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mathpsg.live_catalogue import CatalogueError, LiveCatalogue
from mathpsg.local_gap import probe_gap


ROOT = Path(__file__).resolve().parents[1]


class LiveCatalogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.catalogue = LiveCatalogue(
            probe_gap(), cache_root=self.cache, repository_root=ROOT
        )

    def test_sg1_is_generated_labelled_and_reused(self) -> None:
        first = self.catalogue.records(1)
        second = self.catalogue.records(1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(self.catalogue.resolve(1, "a"), first[0])
        self.assertEqual(self.catalogue.resolve(1, "1a"), first[0])
        metadata = tuple(self.cache.rglob("record.json"))
        self.assertEqual(len(metadata), 1)
        self.assertIn('"certification_status":"host-native"', metadata[0].read_text())

    def test_invalid_it_number_fails_before_gap(self) -> None:
        for value in (0, 231, True, "1"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.catalogue.records(value)  # type: ignore[arg-type]

    def test_unknown_label_is_not_guessed(self) -> None:
        self.catalogue.records(1)
        with self.assertRaisesRegex(CatalogueError, "Wyckoff label"):
            self.catalogue.resolve(1, "not-a-label")

    def test_cache_cannot_live_in_the_runtime_tree(self) -> None:
        with self.assertRaisesRegex(CatalogueError, "cache"):
            LiveCatalogue(
                self.catalogue.runtime,
                cache_root=ROOT / "cache",
                repository_root=ROOT,
            )

    def test_self_rehashed_cross_group_cache_is_rejected(self) -> None:
        sg1 = self.catalogue.records(1)
        self.catalogue.records(70)
        sg70_directory = next((self.cache / "catalogue").glob("sg70-*"))
        geometry = b"".join(
            json.dumps(
                record.to_mapping(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for record in sg1
        )
        metadata_path = sg70_directory / "record.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["geometry_sha256"] = "sha256:" + hashlib.sha256(geometry).hexdigest()
        metadata["record_count"] = len(sg1)
        (sg70_directory / "wyckoff.ndjson").write_bytes(geometry)
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        fresh = LiveCatalogue(
            self.catalogue.runtime,
            cache_root=self.cache,
            repository_root=ROOT,
        )
        with self.assertRaisesRegex(CatalogueError, "cache|group|coverage"):
            fresh.records(70)

    def test_self_rehashed_action_mutation_is_rejected(self) -> None:
        self.catalogue.records(1)
        directory = next((self.cache / "catalogue").glob("sg1-*"))
        geometry_path = directory / "wyckoff.ndjson"
        record = json.loads(geometry_path.read_text(encoding="utf-8"))
        generators = record["space_group_action"]["source_generators"]
        generators.append(generators[0])
        geometry = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        metadata_path = directory / "record.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["geometry_sha256"] = "sha256:" + hashlib.sha256(geometry).hexdigest()
        geometry_path.write_bytes(geometry)
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        fresh = LiveCatalogue(
            self.catalogue.runtime,
            cache_root=self.cache,
            repository_root=ROOT,
        )
        with self.assertRaisesRegex(CatalogueError, "action|provenance|binding"):
            fresh.records(1)

    def test_self_rehashed_provenance_digest_mutation_is_rejected(self) -> None:
        self.catalogue.records(1)
        directory = next((self.cache / "catalogue").glob("sg1-*"))
        geometry_path = directory / "wyckoff.ndjson"
        record = json.loads(geometry_path.read_text(encoding="utf-8"))
        record["action_provenance_digest"] = "sha256:" + "0" * 64
        record["provenance"]["generator_input_digest"] = "sha256:" + "1" * 64
        geometry = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        metadata_path = directory / "record.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["geometry_sha256"] = "sha256:" + hashlib.sha256(geometry).hexdigest()
        geometry_path.write_bytes(geometry)
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        fresh = LiveCatalogue(
            self.catalogue.runtime,
            cache_root=self.cache,
            repository_root=ROOT,
        )
        with self.assertRaisesRegex(CatalogueError, "provenance|binding"):
            fresh.records(1)


if __name__ == "__main__":
    unittest.main()
