"""Host-native GAP evidence and exact algebraic PSG tools."""

from __future__ import annotations

__version__ = "0.1.0"

from .antiunitary import GradedSU2Element
from .affine import AffineMap
from .lattice import PeriodicLattice
from .live_catalogue import LiveCatalogue
from .live_evidence import HostNativeEvidenceBatch, build_evidence
from .live_classify import (
    ClassificationError,
    HostNativeClassificationResult,
    HostRuntimeProvenance,
    classify,
)
from .local_gap import GapRuntime, probe_gap
from .solver_status import solver_capabilities


__all__ = [
    "AffineMap",
    "GapRuntime",
    "GradedSU2Element",
    "HostNativeEvidenceBatch",
    "HostNativeClassificationResult",
    "HostRuntimeProvenance",
    "LiveCatalogue",
    "PeriodicLattice",
    "build_evidence",
    "classify",
    "ClassificationError",
    "probe_gap",
    "solver_capabilities",
]
