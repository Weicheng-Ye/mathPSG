from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from psgmath.live_classify import (
    HostNativeClassificationResult,
    HostRuntimeProvenance,
)
from psgmath.classification_schema import FrozenJSONObject
from psgmath import cli
from psgmath.cli import build_parser, main


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_has_only_implemented_commands(self) -> None:
        help_text = build_parser().format_help()
        for command in (
            "doctor", "catalogue", "evidence", "classify", "capabilities"
        ):
            self.assertIn(command, help_text)

    def test_classify_cli_matches_public_python_result(self) -> None:
        expected = HostNativeClassificationResult(
            request=FrozenJSONObject((("space_group", 1),)),
            class_count=3,
            continuous=False,
            summaries=(),
            details=None,
            certification_status="host-native",
            runtime=HostRuntimeProvenance(
                certification_status="host-native",
                gap_version="4.15.1",
                gap_packages=(("cryst", "4.1.30"),),
                gap_executable_sha256="sha256:" + "0" * 64,
                python_version="3.14.6",
                package_version="0.1.0",
                source_inventory_digest="sha256:" + "1" * 64,
            ),
        )
        with mock.patch("psgmath.cli.classify", return_value=expected) as calculate:
            code, stdout, stderr = self.run_cli(
                "classify", "--it-number", "1", "--wps", "a", "--igg", "Z2"
            )

        self.assertEqual((code, stderr), (0, ""))
        output = json.loads(stdout)
        self.assertEqual(output["class_count"], 3)
        self.assertEqual(output["request"]["space_group"], 1)
        self.assertEqual(output["runtime"]["gap_packages"], {"cryst": "4.1.30"})
        calculate.assert_called_once()
        self.assertEqual(calculate.call_args.args, (1, ["a"]))

    def test_cli_runtime_files_are_packaged_below_psgmath(self) -> None:
        package = Path(cli.__file__).resolve().parent
        self.assertTrue(cli.RUNTIME_ROOT.is_relative_to(package))
        self.assertTrue((cli.RUNTIME_ROOT / "gap/catalogue/export_one.g").is_file())
        self.assertTrue(
            (cli.RUNTIME_ROOT / "gap/classifier/export_problem.g").is_file()
        )
        self.assertTrue(
            (cli.RUNTIME_ROOT / "resources/display-crosswalk.ndjson").is_file()
        )
        self.assertTrue(
            (cli.RUNTIME_ROOT / "resources/action-bindings.json").is_file()
        )

    def test_doctor_records_all_observed_versions(self) -> None:
        code, stdout, stderr = self.run_cli("doctor")
        self.assertEqual((code, stderr), (0, ""))
        value = json.loads(stdout)
        self.assertEqual(value["certification_status"], "host-native")
        self.assertEqual(
            set(value["gap"]["packages"]),
            {"cryst", "hap", "hapcryst", "json", "io"},
        )

    def test_catalogue_summarizes_one_generated_group(self) -> None:
        with tempfile.TemporaryDirectory() as cache:
            code, stdout, stderr = self.run_cli(
                "catalogue", "--it-number", "1", "--cache", cache
            )
        self.assertEqual((code, stderr), (0, ""))
        value = json.loads(stdout)
        self.assertEqual(value["it_number"], 1)
        self.assertEqual(value["record_count"], 1)
        self.assertEqual(value["wyckoff_positions"][0]["label"], "1a")

    def test_backend_errors_return_one_without_traceback(self) -> None:
        with mock.patch("psgmath.cli.probe_gap", side_effect=RuntimeError("no GAP")):
            code, stdout, stderr = self.run_cli("doctor")
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("no GAP", stderr)


if __name__ == "__main__":
    unittest.main()
