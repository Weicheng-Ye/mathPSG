from __future__ import annotations

import unittest

import psgmath
from psgmath.solver_status import solver_capabilities


class SolverBoundaryTests(unittest.TestCase):
    def test_live_classifier_and_unbundled_skeleton_boundary_is_truthful(self) -> None:
        status = solver_capabilities()
        self.assertTrue(status["generic_z2_solver_source_present"])
        self.assertTrue(status["generic_u1_solver_source_present"])
        self.assertTrue(status["live_evidence_bridge_present"])
        self.assertFalse(status["bundled_stabilizer_skeletons_present"])
        self.assertTrue(status["public_classify_api_present"])
        self.assertIn("classify", psgmath.__all__)
        self.assertTrue(callable(psgmath.classify))
        self.assertIn("host-native", status["reason"])
        self.assertIn("rather than release-certified", status["reason"])


if __name__ == "__main__":
    unittest.main()
