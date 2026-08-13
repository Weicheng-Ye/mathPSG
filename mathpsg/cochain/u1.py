"""Compact explicit representatives for affine compact-U1 cochain solutions."""

from __future__ import annotations

from typing import Sequence

from ..torus import Phase, TorusSolution, ZERO_PHASE, raw_torsor_point
from .coordinates import RelativeCochainCoordinates


def u1_basis_presentation(
    *,
    solution: TorusSolution,
    weyl_shift: Sequence[Phase],
    coordinates: RelativeCochainCoordinates,
    labels: Sequence[str],
) -> dict[str, object]:
    """Return the basepoint and one cochain direction per U1 group generator."""

    if len(solution.basepoint) != coordinates.dimension:
        raise ValueError("U1 basepoint and cochain coordinate dimensions differ")
    shift = tuple(weyl_shift)
    if len(shift) != coordinates.dimension:
        raise ValueError("U1 Weyl shift and cochain coordinate dimensions differ")

    group = solution.group
    chart = solution.primal_chart
    generators: list[dict[str, object]] = []
    for column in range(group.free_rank):
        generators.append(
            {
                "kind": "free",
                "parameter": f"phi{column}",
                "coefficients": tuple(
                    chart.free_lifts[row][column]
                    for row in range(chart.raw_dimension)
                ),
            }
        )
    zero_free = (ZERO_PHASE,) * group.free_rank
    for column, order in enumerate(group.torsion_orders):
        torsion_coordinates = tuple(
            int(index == column) for index in range(len(group.torsion_orders))
        )
        direction = tuple(
            chart.torsion_lifts[row][column]
            for row in range(chart.raw_dimension)
        )
        representative = raw_torsor_point(
            solution,
            zero_free,
            torsion_coordinates,
        )
        generators.append(
            {
                "kind": "torsion",
                "order": order,
                "torsion_coordinates": torsion_coordinates,
                "direction_phases": tuple(str(value) for value in direction),
                "representative_phases": tuple(
                    str(value) for value in representative
                ),
            }
        )
    return {
        "coordinate_blocks": coordinates.mapping(labels),
        "basepoint_phases": tuple(str(value) for value in solution.basepoint),
        "weyl_shift_phases": tuple(str(value) for value in shift),
        "basis": tuple(generators),
    }
