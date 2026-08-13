"""Coordinate blocks for the relative degree-two cochain complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RelativeCochainCoordinates:
    """Basis labels for ``ambient C^2`` followed by local ``C^1`` blocks."""

    ambient_degree_2: tuple[str, ...]
    local_degree_1: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ambient_degree_2",
            tuple(str(value) for value in self.ambient_degree_2),
        )
        object.__setattr__(
            self,
            "local_degree_1",
            tuple(
                tuple(str(value) for value in block)
                for block in self.local_degree_1
            ),
        )

    @property
    def dimension(self) -> int:
        return len(self.ambient_degree_2) + sum(
            len(block) for block in self.local_degree_1
        )

    def mapping(self, labels: Sequence[str]) -> dict[str, object]:
        normalized = tuple(str(label) for label in labels)
        if len(normalized) != len(self.local_degree_1):
            raise ValueError("one Wyckoff label is required per local cochain block")
        return {
            "ambient_degree_2": self.ambient_degree_2,
            "local_degree_1": tuple(
                {
                    "orbit_index": index,
                    "wp": label,
                    "basis": basis,
                }
                for index, (label, basis) in enumerate(
                    zip(normalized, self.local_degree_1, strict=True)
                )
            ),
        }
