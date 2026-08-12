from __future__ import annotations

from dataclasses import fields
import unittest

from mathpsg.live_classify import HostNativeClassificationResult


class PublicResultShapeTests(unittest.TestCase):
    def test_classification_result_does_not_record_runtime(self) -> None:
        names = tuple(field.name for field in fields(HostNativeClassificationResult))

        self.assertNotIn("runtime", names)


if __name__ == "__main__":
    unittest.main()
