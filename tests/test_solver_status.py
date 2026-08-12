from __future__ import annotations

import unittest

import psgmath
from psgmath.solver_status import solver_capabilities


class SolverBoundaryTests(unittest.TestCase):
    def test_live_final_classification_is_not_falsely_advertised(self) -> None:
        status = solver_capabilities()
        self.assertTrue(status["generic_z2_solver_source_present"])
        self.assertTrue(status["generic_u1_solver_source_present"])
        self.assertFalse(status["live_evidence_bridge_present"])
        self.assertFalse(status["bundled_stabilizer_skeletons_present"])
        self.assertNotIn("classify", psgmath.__all__)
        self.assertFalse(hasattr(psgmath, "classify"))


if __name__ == "__main__":
    unittest.main()
