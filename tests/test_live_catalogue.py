from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from psgmath.live_catalogue import CatalogueError, LiveCatalogue
from psgmath.local_gap import probe_gap


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


if __name__ == "__main__":
    unittest.main()
