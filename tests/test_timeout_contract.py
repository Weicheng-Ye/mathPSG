from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import mathpsg
from mathpsg.live_catalogue import CatalogueRecord
from mathpsg.local_gap import GapRuntime


_IDENTITY = {
    "matrix": (
        ("q(1,1)", "q(0,1)", "q(0,1)"),
        ("q(0,1)", "q(1,1)", "q(0,1)"),
        ("q(0,1)", "q(0,1)", "q(1,1)"),
    ),
    "translation": ("q(0,1)", "q(0,1)", "q(0,1)"),
}


class TimeoutContractTests(unittest.TestCase):
    def test_classify_converts_gap_timeout_to_classification_error(self) -> None:
        record = CatalogueRecord(
            space_group={"setting": "a"},
            wyckoff_id="sg1:a:a",
            letter="a",
            multiplicity=1,
            stabilizer={"embedded_elements": (_IDENTITY,)},
            space_group_action={
                "source_generators": (),
                "translation_basis": (),
            },
        )
        timeout = subprocess.TimeoutExpired(cmd=("gap", "-q"), timeout=1)

        with (
            mock.patch(
                "mathpsg.live_classify.probe_gap",
                return_value=GapRuntime("/fake-gap"),
            ),
            mock.patch(
                "mathpsg.live_catalogue.LiveCatalogue._generate",
                return_value=(record,),
            ),
            mock.patch(
                "mathpsg.host_classifier_backend.subprocess.run",
                side_effect=timeout,
            ),
        ):
            with self.assertRaises(mathpsg.ClassificationError) as raised:
                mathpsg.classify(1, ["a"], timeout=1)

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIs(raised.exception.__cause__.__cause__, timeout)


if __name__ == "__main__":
    unittest.main()
