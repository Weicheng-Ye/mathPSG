from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExtractionBoundaryTests(unittest.TestCase):
    def test_required_scaffold_exists(self) -> None:
        for relative in ("LICENSE", "README.md", "pyproject.toml", "psgmath"):
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_forbidden_trees_and_dependencies_are_absent(self) -> None:
        for relative in (
            "containers",
            "release",
            "psgmath/benchmarks",
            "psgmath/audits",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("pyxtal", pyproject.lower())

    def test_source_inventory_replays(self) -> None:
        inventory = json.loads(
            (ROOT / "EXTRACTED_SOURCES.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            inventory["record_type"], "mathpsg-standalone-source-inventory"
        )
        for relative, expected in inventory["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected["standalone_sha256"], relative)


if __name__ == "__main__":
    unittest.main()
