from __future__ import annotations

import re
import shutil
import unittest

from psgmath.local_gap import (
    GapRuntimeError,
    host_provenance,
    parse_gap_probe,
    probe_gap,
    source_inventory_digest,
)


class LocalGapTests(unittest.TestCase):
    def test_parse_probe_requires_exact_package_set(self) -> None:
        transcript = "\n".join(
            (
                "GAP=4.15.1",
                "Cryst=4.1.30",
                "HAP=1.70",
                "HAPcryst=0.1.15",
                "json=2.2.3",
                "io=4.9.3",
            )
        )
        executable = shutil.which("gap")
        self.assertIsNotNone(executable)
        runtime = parse_gap_probe(transcript, executable=executable or "gap")
        self.assertEqual(runtime.packages["json"], "2.2.3")
        self.assertEqual(runtime.execution_mode, "host-native")

    def test_parse_probe_rejects_a_missing_package(self) -> None:
        with self.assertRaisesRegex(GapRuntimeError, "io"):
            parse_gap_probe(
                "GAP=4.15.1\nCryst=4.1.30\nHAP=1.70\n"
                "HAPcryst=0.1.15\njson=2.2.3\n",
                executable=shutil.which("gap") or "gap",
            )

    def test_provenance_binds_runtime_and_source_inventory(self) -> None:
        runtime = probe_gap()
        record = host_provenance(runtime)
        self.assertEqual(record["certification_status"], "host-native")
        self.assertEqual(record["execution_mode"], "host-native")
        self.assertEqual(record["gap"]["version"], "4.15.1")
        self.assertRegex(
            record["source_inventory_digest"], r"^sha256:[0-9a-f]{64}$"
        )
        self.assertEqual(
            record["source_inventory_digest"], source_inventory_digest()
        )

    def test_real_probe_records_required_packages(self) -> None:
        runtime = probe_gap()
        self.assertEqual(
            set(runtime.packages), {"cryst", "hap", "hapcryst", "json", "io"}
        )
        self.assertTrue(re.fullmatch(r"sha256:[0-9a-f]{64}", runtime.executable_sha256))


if __name__ == "__main__":
    unittest.main()
