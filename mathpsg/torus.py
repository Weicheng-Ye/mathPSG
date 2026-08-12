r"""Exact affine solving and quotients of compact tori.

For integer matrices ``D`` and ``B`` with ``D @ B == 0``, this module solves

``{z in (R/Z)^N : D z = offset} / im(B)``.

The homogeneous Pontryagin dual is reduced as
``ker(B.T) / im(D.T)``.  The same transformations construct the primal lifts
needed to evaluate raw torsor points.  Positive-dimensional groups are kept
symbolic; certificate and replay payloads are not built on the live path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from mathpsg.integer_linalg import (
    MatrixInput,
    MatrixZ,
    as_matrix,
    integer_kernel,
    inverse_unimodular,
    matmul,
    smith_form,
    transpose,
)


@dataclass(frozen=True, slots=True, order=True)
class Phase:
    """An exact element of ``R/Z``, represented canonically in ``[0, 1)``."""

    value: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value % 1)

    def __str__(self) -> str:
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"{self.value.numerator}/{self.value.denominator}"


ZERO_PHASE = Phase(Fraction(0))


def _phase(value: Fraction) -> Phase:
    return Phase(value)


def _phase_dot(coefficients: Sequence[int], phases: Sequence[Phase]) -> Phase:
    return _phase(
        sum(
            (
                coefficient * phase.value
                for coefficient, phase in zip(coefficients, phases)
            ),
            Fraction(0),
        )
    )


def _phase_matvec(matrix: MatrixZ, vector: Sequence[Phase]) -> tuple[Phase, ...]:
    phases = tuple(vector)
    return tuple(_phase_dot(row, phases) for row in matrix)


def _matrix_columns(matrix: MatrixZ, indices: Sequence[int]) -> MatrixZ:
    selected = tuple(indices)
    return MatrixZ(
        tuple(tuple(matrix[row][column] for column in selected) for row in range(matrix.row_count)),
        column_count=len(selected),
    )


@dataclass(frozen=True, slots=True)
class CompactGroupPresentation:
    free_rank: int
    torsion_orders: tuple[int, ...]
    dual_generators: MatrixZ

@dataclass(frozen=True, slots=True)
class PrimalTorsorChart:
    raw_dimension: int
    free_lifts: MatrixZ
    torsion_lifts: tuple[tuple[Phase, ...], ...]

@dataclass(frozen=True, slots=True)
class TorusObstruction:
    """Marker returned when the affine torus equation has no solution."""


@dataclass(frozen=True, slots=True)
class TorusSolution:
    basepoint: tuple[Phase, ...]
    group: CompactGroupPresentation
    primal_chart: PrimalTorsorChart


def _problem(
    D: MatrixInput,
    B: MatrixInput,
    offset: Sequence[Phase],
) -> tuple[MatrixZ, MatrixZ, tuple[Phase, ...]]:
    equation = as_matrix(D, "$solve_torus_quotient.D")
    quotient = as_matrix(B, "$solve_torus_quotient.B")
    return equation, quotient, tuple(offset)


def _solve_torus_map(
    matrix: MatrixZ,
    target: tuple[Phase, ...],
) -> tuple[Phase, ...] | TorusObstruction:
    smith = smith_form(matrix)
    transformed = _phase_matvec(smith.left, target)
    coordinates = [ZERO_PHASE for _ in range(matrix.column_count)]
    for index, factor in enumerate(smith.invariant_factors):
        coordinates[index] = _phase(transformed[index].value / factor)
    for row in range(smith.rank, matrix.row_count):
        if transformed[row] != ZERO_PHASE:
            return TorusObstruction()
    return _phase_matvec(smith.right, tuple(coordinates))


def _homogeneous_presentation(
    equation: MatrixZ,
    quotient: MatrixZ,
) -> tuple[CompactGroupPresentation, PrimalTorsorChart]:
    raw_dimension = equation.column_count
    quotient_kernel = integer_kernel(transpose(quotient))
    kernel_basis = quotient_kernel.basis
    kernel_rank = quotient_kernel.nullity
    relation_coordinates = matmul(
        quotient_kernel.coordinate_projection,
        transpose(equation),
    )
    presentation_smith = smith_form(relation_coordinates)
    presentation_rank = presentation_smith.rank
    smith_left_inverse = inverse_unimodular(presentation_smith.left)
    all_dual_generators = matmul(kernel_basis, smith_left_inverse)

    free_indices = tuple(range(presentation_rank, kernel_rank))
    torsion_indices = tuple(
        index
        for index, order in enumerate(presentation_smith.invariant_factors)
        if order > 1
    )
    torsion_orders = tuple(
        presentation_smith.invariant_factors[index]
        for index in torsion_indices
    )
    selected_indices = free_indices + torsion_indices
    dual_generators = _matrix_columns(all_dual_generators, selected_indices)
    group = CompactGroupPresentation(
        len(free_indices),
        torsion_orders,
        dual_generators,
    )

    # If L K = I for the saturated kernel basis K, then
    # X = L.T U.T is an integer right inverse of (K U^-1).T.
    right_inverse = matmul(
        transpose(quotient_kernel.coordinate_projection),
        transpose(presentation_smith.left),
    )

    free_lifts = _matrix_columns(right_inverse, free_indices)
    torsion_integer_lifts = _matrix_columns(right_inverse, torsion_indices)
    torsion_lifts = tuple(
        tuple(
            _phase(Fraction(torsion_integer_lifts[row][column], torsion_orders[column]))
            for column in range(len(torsion_orders))
        )
        for row in range(raw_dimension)
    )
    chart = PrimalTorsorChart(
        raw_dimension,
        free_lifts,
        torsion_lifts,
    )
    return group, chart


def solve_torus_quotient(
    D: MatrixInput,
    B: MatrixInput,
    offset: Sequence[Phase],
) -> TorusSolution | TorusObstruction:
    """Solve an affine compact-torus quotient exactly."""

    equation, quotient, normalized_offset = _problem(D, B, offset)
    affine = _solve_torus_map(equation, normalized_offset)
    if isinstance(affine, TorusObstruction):
        return affine
    basepoint = affine
    group, chart = _homogeneous_presentation(equation, quotient)
    return TorusSolution(
        basepoint,
        group,
        chart,
    )


def raw_torsor_point(
    solution: TorusSolution,
    free: Sequence[Phase],
    torsion: Sequence[int],
) -> tuple[Phase, ...]:
    """Lift canonical symbolic torsor coordinates to raw ``z`` phases."""

    free_coordinates = tuple(free)
    torsion_coordinates = tuple(torsion)
    raw = list(solution.basepoint)
    for row in range(solution.primal_chart.raw_dimension):
        value = raw[row].value
        value += sum(
            (
                solution.primal_chart.free_lifts[row][column]
                * free_coordinates[column].value
                for column in range(solution.group.free_rank)
            ),
            Fraction(0),
        )
        value += sum(
            (
                torsion_coordinates[column]
                * solution.primal_chart.torsion_lifts[row][column].value
                for column in range(len(solution.group.torsion_orders))
            ),
            Fraction(0),
        )
        raw[row] = _phase(value)
    return tuple(raw)
