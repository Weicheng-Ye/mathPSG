"""Fresh physical PSG computation with no cache or certificate orchestration.

The public entry point in this module consumes only ``request.igg``,
``request.time_reversal`` and resolved-orbit ``record`` attributes.  GAP output
is turned into the matrices needed by the two exact solvers; provenance, replay
receipts, response schemas and cache identities do not enter this call graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
from pathlib import Path
from typing import Sequence, cast

from .direct_algebra import (
    bar_chain_cochain_value,
    twist_group_ring_matrix,
    word_character,
)
from .gf2 import (
    GF2Character,
    GF2Inconsistency,
    MatrixGF2,
    quotient_basis,
    rref,
    solve_affine,
)
from .host_classifier_backend import (
    DirectHostAmbient,
    DirectInclusion,
    build_direct_host_ambient,
    direct_character_context,
    enumerate_direct_local_branches,
    resolve_direct_inclusions,
)
from .integer_linalg import MatrixZ, transpose
from .local_gap import GapRuntime
from .torus import (
    Phase,
    TorusObstruction,
    TorusSolution,
    ZERO_PHASE,
    raw_torsor_point,
    solve_torus_quotient,
)


@dataclass(frozen=True, slots=True)
class Z2PhysicalStratum:
    basepoint: tuple[int, ...]
    quotient_basis: tuple[tuple[int, ...], ...]
    unframed_class_count: int

    @property
    def quotient_dimension(self) -> int:
        return len(self.quotient_basis)

    @property
    def framed_class_count(self) -> int:
        return 1 << self.quotient_dimension


@dataclass(frozen=True, slots=True)
class U1PhysicalStratum:
    rho_bits: tuple[int, ...]
    solution: TorusSolution
    weyl_shift: tuple[Phase, ...]

    @property
    def continuous(self) -> bool:
        return self.solution.group.free_rank > 0

    @property
    def framed_class_count(self) -> int | None:
        if self.continuous:
            return None
        result = 1
        for order in self.solution.group.torsion_orders:
            result *= order
        return result


PhysicalStratum = Z2PhysicalStratum | U1PhysicalStratum


@dataclass(frozen=True, slots=True)
class PhysicalQuotient:
    class_count: int | None
    continuous: bool
    framed_class_count: int | None


@dataclass(frozen=True, slots=True)
class PhysicalClassification:
    framed_strata: tuple[PhysicalStratum, ...]
    quotient: PhysicalQuotient

    @property
    def class_count(self) -> int | None:
        return self.quotient.class_count

    @property
    def continuous(self) -> bool:
        return self.quotient.continuous


@dataclass(frozen=True, slots=True)
class _Complex:
    dimensions: tuple[int, ...]
    differentials: tuple[MatrixGF2 | MatrixZ, ...]


@dataclass(frozen=True, slots=True)
class _RelativeMatrices:
    B: MatrixGF2 | MatrixZ
    D: MatrixGF2 | MatrixZ
    offset: tuple[int, ...] | tuple[Phase, ...]
    local_slices: tuple[tuple[int, int], ...]


def _gf2_transpose(matrix) -> MatrixGF2:
    dense = [[0] * matrix.column_count for _ in range(matrix.row_count)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] = sum(
            term.coefficient for term in entry.terms
        ) & 1
    return MatrixGF2(
        tuple(
            tuple(dense[row][column] for row in range(matrix.row_count))
            for column in range(matrix.column_count)
        ),
        column_count=matrix.row_count,
    )


def _z2_complex(resolution) -> _Complex:
    return _Complex(
        tuple(len(degree) for degree in resolution.basis),
        tuple(_gf2_transpose(boundary) for boundary in resolution.boundaries),
    )


def _u1_complex(resolution, character: GF2Character) -> _Complex:
    return _Complex(
        tuple(len(degree) for degree in resolution.basis),
        tuple(
            transpose(twist_group_ring_matrix(boundary, resolution, character.bits))
            for boundary in resolution.boundaries
        ),
    )


def _restriction(
    inclusion: DirectInclusion,
    *,
    ring: str,
    ambient_character: GF2Character | None = None,
) -> tuple[MatrixGF2 | MatrixZ, ...]:
    if ring == "gf2":
        maps = tuple(_gf2_transpose(matrix) for matrix in inclusion.maps)
    else:
        assert ambient_character is not None
        maps = tuple(
            transpose(
                twist_group_ring_matrix(
                    matrix,
                    inclusion.target_resolution,
                    ambient_character.bits,
                )
            )
            for matrix in inclusion.maps
        )
    return maps


def _relative_differential(
    ring: str,
    ambient: _Complex,
    locals_: tuple[_Complex, ...],
    restrictions: tuple[tuple[MatrixGF2 | MatrixZ, ...], ...],
    degree: int,
) -> tuple[MatrixGF2 | MatrixZ, tuple[tuple[int, int], ...]]:
    source_local_slices: list[tuple[int, int]] = []
    cursor = ambient.dimensions[degree]
    for local in locals_:
        stop = cursor + local.dimensions[degree - 1]
        source_local_slices.append((cursor, stop))
        cursor = stop
    source_dimension = cursor
    target_slices: list[tuple[int, int]] = []
    cursor = ambient.dimensions[degree + 1]
    for local in locals_:
        target_slices.append((cursor, cursor + local.dimensions[degree]))
        cursor += local.dimensions[degree]
    dense = [[0] * source_dimension for _ in range(cursor)]
    ambient_d = ambient.differentials[degree]
    for row in range(ambient_d.row_count):
        for column in range(ambient_d.column_count):
            dense[row][column] = ambient_d[row][column]
    for index, (local, restriction) in enumerate(
        zip(locals_, restrictions)
    ):
        target_start, _ = target_slices[index]
        restricted = restriction[degree]
        for row in range(restricted.row_count):
            for column in range(restricted.column_count):
                dense[target_start + row][column] = restricted[row][column]
        local_d = local.differentials[degree - 1]
        source_start, _ = source_local_slices[index]
        for row in range(local_d.row_count):
            for column in range(local_d.column_count):
                value = -local_d[row][column]
                dense[target_start + row][source_start + column] = (
                    value & 1 if ring == "gf2" else value
                )
    rows = tuple(tuple(row) for row in dense)
    constructor = MatrixGF2 if ring == "gf2" else MatrixZ
    # Residual gauge/Weyl translations act on the source coordinates of this
    # differential (relative degree ``degree``), not its equation rows.
    return constructor(rows, column_count=source_dimension), tuple(source_local_slices)


def _relative_matrices(
    ring: str,
    ambient: _Complex,
    locals_: tuple[_Complex, ...],
    restrictions: tuple[tuple[MatrixGF2 | MatrixZ, ...], ...],
    defects: tuple[tuple[int, ...] | tuple[Phase, ...], ...],
) -> _RelativeMatrices:
    B, _ = _relative_differential(ring, ambient, locals_, restrictions, 1)
    D, local_slices = _relative_differential(
        ring, ambient, locals_, restrictions, 2
    )
    zero = 0 if ring == "gf2" else ZERO_PHASE
    offset = (zero,) * ambient.dimensions[3] + tuple(
        value for defect in defects for value in defect
    )
    return _RelativeMatrices(B, D, offset, local_slices)


def _matrix_from_columns(
    columns: Sequence[Sequence[int]], row_count: int
) -> MatrixGF2:
    values = tuple(tuple(column) for column in columns)
    return MatrixGF2(
        tuple(tuple(column[row] for column in values) for row in range(row_count)),
        column_count=len(values),
    )


def _quotient_shift(
    quotient,
    raw_shift: tuple[int, ...],
) -> tuple[int, ...]:
    decomposition = quotient.boundary_basis + quotient.representatives
    coordinates = solve_affine(
        _matrix_from_columns(decomposition, quotient.ambient_dimension), raw_shift
    )
    return coordinates.basepoint[len(quotient.boundary_basis) :]  # type: ignore[union-attr]


def _pullback_mod2(equivalence, *, degree: int, value) -> tuple[int, ...]:
    """Evaluate a finite-group cochain on the Task-5 comparison map."""

    traces = {
        item.basis_id: item.image
        for item in equivalence.psi_on_basis
        if item.degree == degree
    }
    element_index = {
        element: index
        for index, element in enumerate(equivalence.finite_group.element_order)
    }
    coordinates: list[int] = []
    for basis_id in equivalence.resolution.basis[degree]:
        coordinate = 0
        for term in traces[basis_id].terms:
            indices = tuple(element_index[element] for element in term.group_tuple)
            cochain_value = (
                value[indices[0]]
                if degree == 1
                else value[indices[0]][indices[1]]
            )
            coordinate ^= (term.coefficient & 1) & cochain_value
        coordinates.append(coordinate)
    return tuple(coordinates)


def _z2_defect(skeleton, equivalence) -> tuple[int, ...]:
    return _pullback_mod2(
        equivalence, degree=2, value=skeleton.defect_bits
    )


def _orbit_count(points, actions) -> int:
    unseen = set(points)
    count = 0
    while unseen:
        count += 1
        pending = [unseen.pop()]
        while pending:
            point = pending.pop()
            for action in actions:
                image = action(point)
                if image in unseen:
                    unseen.remove(image)
                    pending.append(image)
    return count


def _z2_orbit_count(
    dimension: int, shifts: tuple[tuple[int, ...], ...]
) -> int:
    shift_matrix = MatrixGF2(shifts, column_count=dimension)
    shift_rank = len(rref(shift_matrix).pivots)
    return 1 << (dimension - shift_rank)


def _z2_strata(
    ambient: DirectHostAmbient,
    local_rows: tuple[tuple[object, ...], ...],
    inclusions: tuple[DirectInclusion, ...],
) -> tuple[Z2PhysicalStratum, ...]:
    ambient_complex = _z2_complex(ambient.resolution)
    local_complexes = tuple(
        _z2_complex(inclusion.source_resolution) for inclusion in inclusions
    )
    restrictions = tuple(
        _restriction(inclusion, ring="gf2") for inclusion in inclusions
    )
    strata: list[Z2PhysicalStratum] = []
    for skeletons in itertools.product(*local_rows):
        defects = tuple(
            _z2_defect(skeleton, inclusion.bar_equivalence)
            for skeleton, inclusion in zip(skeletons, inclusions)
        )
        matrices = _relative_matrices(
            "gf2", ambient_complex, local_complexes, restrictions, defects
        )
        differential = cast(MatrixGF2, matrices.D)
        boundaries = cast(MatrixGF2, matrices.B)
        offset = cast(tuple[int, ...], matrices.offset)
        solution = solve_affine(differential, offset)
        if isinstance(solution, GF2Inconsistency):
            continue
        quotient = quotient_basis(
            _matrix_from_columns(solution.kernel_basis, differential.column_count),
            boundaries,
        )
        shifts: list[tuple[int, ...]] = []
        for index, (skeleton, inclusion) in enumerate(
            zip(skeletons, inclusions)
        ):
            start, stop = matrices.local_slices[index]
            for marking_shift in skeleton.marking_shifts:
                local = _pullback_mod2(
                    inclusion.bar_equivalence,
                    degree=1,
                    value=marking_shift,
                )
                raw = [0] * differential.column_count
                raw[start:stop] = local
                shifts.append(_quotient_shift(quotient, tuple(raw)))
        residual_shifts = tuple(shifts)
        strata.append(
            Z2PhysicalStratum(
                solution.basepoint,
                quotient.representatives,
                _z2_orbit_count(len(quotient.representatives), residual_shifts),
            )
        )
    return tuple(strata)


def _u1_defect(skeleton, equivalence) -> tuple[Phase, ...]:
    index = {
        element: position for position, element in enumerate(skeleton.element_order)
    }
    cocycle = {
        pair: skeleton.normalized_bar_defect[index[pair[0]]][index[pair[1]]].value
        for pair in equivalence.normalized_tuples(2)
    }
    character = GF2Character(skeleton.rho_values)
    psi = {
        item.basis_id: item.image
        for item in equivalence.psi_on_basis
        if item.degree == 2
    }
    return tuple(
        Phase(
            bar_chain_cochain_value(
                equivalence, cocycle, psi[basis_id], character.bits
            )
        )
        for basis_id in equivalence.resolution.basis[2]
    )


def _solve_rational_system(
    rows: Sequence[Sequence[int]], right: Sequence[Fraction], columns: int
) -> tuple[Fraction, ...]:
    matrix = [
        [Fraction(value) for value in row] + [Fraction(target)]
        for row, target in zip(rows, right)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            coefficient = matrix[row][column]
            if coefficient:
                matrix[row] = [
                    left - coefficient * value
                    for left, value in zip(
                        matrix[row], matrix[pivot_row]
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
    if any(not any(row[:columns]) and row[columns] for row in matrix):
        raise ArithmeticError("Weyl comparison cochain has no solution")
    result = [Fraction(0)] * columns
    for row, column in enumerate(pivot_columns):
        result[column] = matrix[row][columns]
    return tuple(result)


def _weyl_coordinates(skeleton, equivalence) -> tuple[Phase, ...]:
    table = equivalence.finite_group
    element_index = {element: index for index, element in enumerate(table.element_order)}
    phi = {item.group_tuple: item.image for item in equivalence.phi_on_queries}
    dimension = len(equivalence.resolution.basis[1])
    rows: list[tuple[int, ...]] = []
    values: list[Fraction] = []
    for element in table.element_order[1:]:
        weights = [0] * dimension
        for term in phi[(element,)].terms:
            basis_index = int(term.basis_id.split(":", 1)[1])
            sign = -1 if skeleton.rho_values[element_index[term.element]] else 1
            weights[basis_index] += term.coefficient * sign
        rows.append(tuple(weights))
        values.append(
            Fraction(skeleton.grade_values[element_index[element]], 2)
        )
    return tuple(Phase(value) for value in _solve_rational_system(rows, values, dimension))


def _u1_strata(
    ambient: DirectHostAmbient,
    local_rows: tuple[tuple[object, ...], ...],
    inclusions: tuple[DirectInclusion, ...],
) -> tuple[U1PhysicalStratum, ...]:
    basis, _ = direct_character_context(ambient)
    strata: list[U1PhysicalStratum] = []
    for sector_index, rho in enumerate(basis):
        skeletons = tuple(row[sector_index] for row in local_rows)
        ambient_complex = _u1_complex(ambient.resolution, rho)
        local_characters = tuple(
            GF2Character(
                tuple(
                    word_character(ambient.resolution, rho.bits, image)
                    for image in inclusion.source_element_images
                )
            )
            for inclusion in inclusions
        )
        local_complexes = tuple(
            _u1_complex(inclusion.source_resolution, local_character)
            for inclusion, local_character in zip(
                inclusions, local_characters
            )
        )
        restrictions = tuple(
            _restriction(
                inclusion,
                ring="torus",
                ambient_character=rho,
            )
            for inclusion in inclusions
        )
        defects = tuple(
            _u1_defect(skeleton, inclusion.bar_equivalence)
            for skeleton, inclusion in zip(skeletons, inclusions)
        )
        matrices = _relative_matrices(
            "torus", ambient_complex, local_complexes, restrictions, defects
        )
        differential = cast(MatrixZ, matrices.D)
        boundaries = cast(MatrixZ, matrices.B)
        offset = cast(tuple[Phase, ...], matrices.offset)
        solution = solve_torus_quotient(
            differential, boundaries, offset
        )
        if isinstance(solution, TorusObstruction):
            continue
        shift = [ZERO_PHASE] * differential.column_count
        for index, (skeleton, inclusion) in enumerate(
            zip(skeletons, inclusions)
        ):
            start, stop = matrices.local_slices[index]
            local_shift = _weyl_coordinates(
                skeleton, inclusion.bar_equivalence
            )
            shift[start:stop] = local_shift
        strata.append(
            U1PhysicalStratum(
                tuple(rho.bits), solution, tuple(shift)
            )
        )
    return tuple(strata)


def _z2_quotient(strata: tuple[Z2PhysicalStratum, ...]) -> PhysicalQuotient:
    framed = sum(stratum.framed_class_count for stratum in strata)
    unframed = sum(stratum.unframed_class_count for stratum in strata)
    return PhysicalQuotient(unframed, False, framed)


def _u1_weyl_image(
    stratum: U1PhysicalStratum, coordinates: tuple[int, ...]
) -> tuple[int, ...]:
    raw = raw_torsor_point(stratum.solution, (), coordinates)
    image = tuple(
        Phase(-value.value + shift.value)
        for value, shift in zip(raw, stratum.weyl_shift)
    )
    homogeneous = tuple(
        Phase(value.value - base.value)
        for value, base in zip(image, stratum.solution.basepoint)
    )
    result: list[int] = []
    free_rank = stratum.solution.group.free_rank
    for index, order in enumerate(stratum.solution.group.torsion_orders):
        column = free_rank + index
        evaluation = sum(
            (
                stratum.solution.group.dual_generators[row][column]
                * homogeneous[row].value
                for row in range(len(homogeneous))
            ),
            Fraction(0),
        ) % 1
        scaled = evaluation * order
        if scaled.denominator != 1:
            raise ArithmeticError("Weyl image has a nonintegral torsion coordinate")
        result.append(scaled.numerator % order)
    return tuple(result)


def _u1_quotient(strata: tuple[U1PhysicalStratum, ...]) -> PhysicalQuotient:
    if any(stratum.continuous for stratum in strata):
        return PhysicalQuotient(None, True, None)
    framed = sum(stratum.framed_class_count or 0 for stratum in strata)
    unframed = 0
    for stratum in strata:
        points = tuple(
            itertools.product(
                *(range(order) for order in stratum.solution.group.torsion_orders)
            )
        )
        unframed += _orbit_count(
            points,
            (lambda point, current=stratum: _u1_weyl_image(current, point),),
        )
    return PhysicalQuotient(unframed, False, framed)


def compute_classification(
    request,
    resolved_orbits: Sequence[object],
    *,
    runtime: GapRuntime,
    repository_root: Path,
    timeout_seconds: int = 300,
) -> PhysicalClassification:
    """Compute the exhaustive Z2 or U1 physical answer from fresh GAP data."""

    resolved = tuple(resolved_orbits)
    ambient = build_direct_host_ambient(
        tuple(item.record for item in resolved),
        runtime=runtime,
        time_reversal=request.time_reversal,
        timeout_seconds=timeout_seconds,
        repository_root=repository_root,
    )
    local_rows = tuple(
        enumerate_direct_local_branches(request, item, ambient) for item in resolved
    )
    inclusions = resolve_direct_inclusions(resolved, ambient)
    if request.igg == "Z2":
        strata = _z2_strata(ambient, local_rows, inclusions)
        quotient = _z2_quotient(strata)
    elif request.igg == "U1":
        strata = _u1_strata(ambient, local_rows, inclusions)
        quotient = _u1_quotient(strata)
    else:
        raise ValueError("classification supports only Z2 or U1")
    return PhysicalClassification(strata, quotient)


__all__ = [
    "PhysicalClassification",
    "PhysicalQuotient",
    "U1PhysicalStratum",
    "Z2PhysicalStratum",
    "compute_classification",
]
