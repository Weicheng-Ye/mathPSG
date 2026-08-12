from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackageNameTests(unittest.TestCase):
    def test_mathpsg_is_the_only_import_name(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                "-c",
                (
                    "import importlib.util; import mathpsg; "
                    "assert mathpsg.__name__ == 'mathpsg'; "
                    "assert callable(mathpsg.classify); "
                    "assert importlib.util.find_spec('psgmath') is None"
                ),
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
