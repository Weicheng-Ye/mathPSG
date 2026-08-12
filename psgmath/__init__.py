"""Lattice-aware algebraic PSG tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

__version__ = "0.1.0"

from .antiunitary import GradedSU2Element
from .affine import AffineMap
from .lattice import PeriodicLattice

if TYPE_CHECKING:
    from .public_api import PublicQueryResult


def classify(
    it_number: int,
    wps: str | Sequence[str],
    *,
    igg: str = "Z2",
    time_reversal: bool = False,
    setting: str | None = None,
    details: bool = False,
) -> PublicQueryResult:
    """Return the certified class count and optional exact class details."""

    from .public_api import classify as _classify

    return _classify(
        it_number,
        wps,
        igg=igg,
        time_reversal=time_reversal,
        setting=setting,
        details=details,
    )


__all__ = ["AffineMap", "GradedSU2Element", "PeriodicLattice", "classify"]
