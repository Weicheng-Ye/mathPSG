"""Truthful capability report for the extracted mathematical modules."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


_CAPABILITIES = MappingProxyType(
    {
        "generic_z2_solver_source_present": True,
        "generic_u1_solver_source_present": True,
        "ordered_multi_orbit_algorithms_present": True,
        "live_evidence_bridge_present": True,
        "bundled_stabilizer_skeletons_present": False,
        "public_classify_api_present": True,
        "reason": (
            "the public calculator uses replay-verified local GAP evidence and "
            "reports host-native rather than release-certified authority"
        ),
    }
)


def solver_capabilities() -> Mapping[str, object]:
    """Describe implemented code without advertising an unavailable query API."""

    return _CAPABILITIES


__all__ = ["solver_capabilities"]
