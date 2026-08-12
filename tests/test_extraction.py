from __future__ import annotations

import hashlib
import importlib
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
        self.assertFalse((ROOT / "docs" / "superpowers").exists())

    def test_standalone_tools_do_not_name_the_source_worktree(self) -> None:
        forbidden = "/".join(
            ("", "Users", "victor", "Downloads", "mathPSG", "mathPSG")
        )
        for path in (ROOT / "tools").rglob("*.py"):
            self.assertNotIn(forbidden, path.read_text(encoding="utf-8"), path.name)

    def test_public_gap_launcher_is_host_native(self) -> None:
        exporter = (ROOT / "gap/classifier/export_problem.g").read_text(
            encoding="utf-8"
        )
        classifier = (ROOT / "psgmath/gap_classifier.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("/opt/mathpsg", exporter)
        self.assertNotIn("/opt/mathpsg", classifier)

    def test_every_shipped_python_module_imports(self) -> None:
        for path in sorted((ROOT / "psgmath").glob("*.py")):
            if path.stem == "__main__":
                continue
            with self.subTest(module=path.stem):
                importlib.import_module(f"psgmath.{path.stem}")

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
