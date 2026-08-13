"""Compact explicit representatives for affine Z2 cochain solutions."""

from __future__ import annotations

from typing import Sequence

from ..gf2 import MatrixGF2, quotient_basis as gf2_quotient_basis
from .coordinates import RelativeCochainCoordinates


def _matrix_from_columns(
    columns: Sequence[Sequence[int]], row_count: int
) -> MatrixGF2:
    values = tuple(tuple(column) for column in columns)
    if any(len(column) != row_count for column in values):
        raise ValueError("cochain vector has the wrong dimension")
    return MatrixGF2(
        tuple(tuple(column[row] for column in values) for row in range(row_count)),
        column_count=len(values),
    )


def _linear_combination(
    basis: tuple[tuple[int, ...], ...], coordinates: tuple[int, ...]
) -> tuple[int, ...]:
    dimension = len(basis[0]) if basis else 0
    return tuple(
        sum(
            coefficient * vector[row]
            for coefficient, vector in zip(coordinates, basis, strict=True)
        )
        & 1
        for row in range(dimension)
    )


def z2_basis_presentation(
    *,
    basepoint: Sequence[int],
    quotient_basis: Sequence[Sequence[int]],
    residual_shifts: Sequence[Sequence[int]],
    coordinates: RelativeCochainCoordinates,
    labels: Sequence[str],
) -> dict[str, object]:
    """Return one affine cochain representative per unframed Z2 basis vector."""

    base = tuple(int(value) & 1 for value in basepoint)
    basis = tuple(tuple(int(value) & 1 for value in vector) for vector in quotient_basis)
    shifts = tuple(tuple(int(value) & 1 for value in vector) for vector in residual_shifts)
    if len(base) != coordinates.dimension:
        raise ValueError("Z2 basepoint and cochain coordinate dimensions differ")
    if any(len(vector) != len(base) for vector in basis):
        raise ValueError("Z2 quotient basis has the wrong cochain dimension")
    quotient_dimension = len(basis)
    if any(len(vector) != quotient_dimension for vector in shifts):
        raise ValueError("Z2 residual shift has the wrong quotient dimension")

    identity = MatrixGF2(
        tuple(
            tuple(int(row == column) for column in range(quotient_dimension))
            for row in range(quotient_dimension)
        ),
        column_count=quotient_dimension,
    )
    residual = _matrix_from_columns(shifts, quotient_dimension)
    unframed = gf2_quotient_basis(identity, residual).representatives

    generators = []
    for quotient_coordinates in unframed:
        direction = _linear_combination(basis, quotient_coordinates)
        representative = tuple(
            left ^ right for left, right in zip(base, direction, strict=True)
        )
        generators.append(
            {
                "kind": "torsion",
                "order": 2,
                "quotient_coordinates": quotient_coordinates,
                "direction": direction,
                "representative": representative,
            }
        )
    return {
        "coordinate_blocks": coordinates.mapping(labels),
        "basepoint": base,
        "basis": tuple(generators),
    }
