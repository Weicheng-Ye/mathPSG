from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from mathpsg.local_gap import GapRuntimeError, parse_gap_probe, probe_gap


class LocalGapTests(unittest.TestCase):
    def _executable(self, body: str) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fake-gap"
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return os.fspath(path)

    def test_real_gap_exists_terminates_and_returns_json(self) -> None:
        runtime = probe_gap()
        self.assertTrue(Path(runtime.executable).is_file())

    def test_probe_rejects_unsuccessful_gap(self) -> None:
        executable = self._executable("exit 7\n")
        with self.assertRaisesRegex(GapRuntimeError, "unsuccessfully"):
            probe_gap(executable)

    def test_probe_rejects_non_json_output(self) -> None:
        executable = self._executable("printf 'not-json\\n'\n")
        with self.assertRaisesRegex(GapRuntimeError, "valid JSON"):
            probe_gap(executable)

    def test_probe_payload_has_no_version_contract(self) -> None:
        runtime = parse_gap_probe(
            '{"ok": true}', executable=shutil.which("gap") or "gap"
        )
        self.assertEqual(tuple(runtime.__dataclass_fields__), ("executable",))


if __name__ == "__main__":
    unittest.main()
