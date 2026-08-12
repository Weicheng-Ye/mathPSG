from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock

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
        for command in ("doctor", "catalogue", "evidence", "capabilities"):
            self.assertIn(command, help_text)
        self.assertNotIn("classify", help_text)

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
