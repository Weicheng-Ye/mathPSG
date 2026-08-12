r"""Exact affine solving and certified quotients of compact tori.

For integer matrices ``D`` and ``B`` with ``D @ B == 0``, this module solves

``{z in (R/Z)^N : D z = offset} / im(B)``.

The homogeneous Pontryagin dual is reduced as
``ker(B.T) / im(D.T)``.  The same unimodular transformations construct a
primal chart in the original ``z`` coordinates: integer free lifts, exact
phase-valued torsion lifts, and characters proving their exact quotient
orders.  Positive-dimensional groups are always kept symbolic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import prod

from psgmath.integer_linalg import (
    MatrixInput,
    MatrixZ,
    SmithForm,
    as_matrix,
    identity_matrix,
    integer_kernel,
    inverse_unimodular,
    matmul,
    matrix_from_columns,
    smith_form,
    transpose,
    zero_matrix,
)


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path}: expected integer")
    return value


@dataclass(frozen=True, slots=True, order=True)
class Phase:
    """An exact element of ``R/Z``, represented canonically in ``[0, 1)``."""

    value: Fraction

    def __post_init__(self) -> None:
        if type(self.value) is not Fraction:
            raise TypeError("$Phase.value: expected fractions.Fraction")
        object.__setattr__(self, "value", self.value % 1)

    def __str__(self) -> str:
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"{self.value.numerator}/{self.value.denominator}"


ZERO_PHASE = Phase(Fraction(0))


def _phase(value: Fraction) -> Phase:
    return Phase(value)


def _phase_vector(value: Sequence[Phase], path: str, length: int | None = None) -> tuple[Phase, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected phase vector")
    normalized: list[Phase] = []
    for index, entry in enumerate(value):
        if type(entry) is not Phase:
            raise TypeError(f"{path}[{index}]: expected Phase")
        normalized.append(entry)
    if length is not None and len(normalized) != length:
        raise ValueError(f"{path}: expected length {length}")
    return tuple(normalized)


def _integer_vector(value: Sequence[int], path: str, length: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected integer vector")
    normalized = tuple(_integer(entry, f"{path}[{index}]") for index, entry in enumerate(value))
    if length is not None and len(normalized) != length:
        raise ValueError(f"{path}: expected length {length}")
    return normalized


def _phase_matrix(
    value: Sequence[Sequence[Phase]],
    path: str,
    *,
    rows: int,
    columns: int,
) -> tuple[tuple[Phase, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected phase matrix")
    if len(value) != rows:
        raise ValueError(f"{path}: expected {rows} rows")
    return tuple(
        _phase_vector(row, f"{path}[{index}]", columns)
        for index, row in enumerate(value)
    )


def _phase_dot(coefficients: Sequence[int], phases: Sequence[Phase]) -> Phase:
    if len(coefficients) != len(phases):
        raise ValueError("phase dot-product dimensions differ")
    return _phase(
        sum(
            (
                coefficient * phase.value
                for coefficient, phase in zip(coefficients, phases, strict=True)
            ),
            Fraction(0),
        )
    )


def _phase_matvec(matrix: MatrixZ, vector: Sequence[Phase]) -> tuple[Phase, ...]:
    phases = _phase_vector(vector, "$phase_matvec.vector", matrix.column_count)
    return tuple(_phase_dot(row, phases) for row in matrix)


def _integer_phase_matvec(matrix: MatrixZ, vector: Sequence[Phase]) -> tuple[Phase, ...]:
    return _phase_matvec(matrix, vector)


def _phase_add(left: Sequence[Phase], right: Sequence[Phase]) -> tuple[Phase, ...]:
    if len(left) != len(right):
        raise ValueError("phase-vector dimensions differ")
    return tuple(
        _phase(a.value + b.value)
        for a, b in zip(left, right, strict=True)
    )


def _phase_subtract(left: Sequence[Phase], right: Sequence[Phase]) -> tuple[Phase, ...]:
    if len(left) != len(right):
        raise ValueError("phase-vector dimensions differ")
    return tuple(
        _phase(a.value - b.value)
        for a, b in zip(left, right, strict=True)
    )


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
    dual_relations: MatrixZ

    def __post_init__(self) -> None:
        free_rank = _integer(self.free_rank, "$CompactGroupPresentation.free_rank")
        if free_rank < 0:
            raise ValueError("$CompactGroupPresentation.free_rank: expected nonnegative integer")
        if isinstance(self.torsion_orders, (str, bytes)) or not isinstance(self.torsion_orders, Sequence):
            raise TypeError("$CompactGroupPresentation.torsion_orders: expected integer tuple")
        orders = tuple(
            _integer(order, f"$CompactGroupPresentation.torsion_orders[{index}]")
            for index, order in enumerate(self.torsion_orders)
        )
        if any(order <= 1 for order in orders):
            raise ValueError("torsion orders must exceed one")
        if any(next_order % order for order, next_order in zip(orders, orders[1:])):
            raise ValueError("torsion orders must divide their successors")
        generators = as_matrix(self.dual_generators, "$CompactGroupPresentation.dual_generators")
        relations = as_matrix(self.dual_relations, "$CompactGroupPresentation.dual_relations")
        generator_count = free_rank + len(orders)
        if generators.column_count != generator_count:
            raise ValueError("dual-generator count differs from invariant count")
        if smith_form(generators).rank != generator_count:
            raise ValueError("dual generators must be integer-linearly independent")
        if relations.shape != (generator_count, generator_count):
            raise ValueError("dual-relation matrix has incompatible shape")
        expected = MatrixZ(
            tuple(
                tuple(
                    (
                        orders[row - free_rank]
                        if row == column and row >= free_rank
                        else 0
                    )
                    for column in range(generator_count)
                )
                for row in range(generator_count)
            ),
            column_count=generator_count,
        )
        if relations != expected:
            raise ValueError("dual relations must serialize free zeros then torsion orders")
        object.__setattr__(self, "free_rank", free_rank)
        object.__setattr__(self, "torsion_orders", orders)
        object.__setattr__(self, "dual_generators", generators)
        object.__setattr__(self, "dual_relations", relations)

    @property
    def serialized_invariants(self) -> tuple[int, ...]:
        return (0,) * self.free_rank + self.torsion_orders

    @property
    def finite_cardinality(self) -> int | None:
        return None if self.free_rank else prod(self.torsion_orders, start=1)


@dataclass(frozen=True, slots=True)
class PrimalTorsorChart:
    raw_dimension: int
    free_lifts: MatrixZ
    torsion_lifts: tuple[tuple[Phase, ...], ...]
    free_character_pairing: MatrixZ
    torsion_pairing: tuple[tuple[Phase, ...], ...]
    quotient_witnesses: tuple[MatrixZ, ...]

    def __post_init__(self) -> None:
        raw_dimension = _integer(self.raw_dimension, "$PrimalTorsorChart.raw_dimension")
        if raw_dimension < 0:
            raise ValueError("$PrimalTorsorChart.raw_dimension: expected nonnegative integer")
        free_lifts = as_matrix(self.free_lifts, "$PrimalTorsorChart.free_lifts")
        if free_lifts.row_count != raw_dimension:
            raise ValueError("$PrimalTorsorChart.free_lifts: raw dimension mismatch")
        free_rank = free_lifts.column_count
        if free_rank > raw_dimension or smith_form(free_lifts).rank != free_rank:
            raise ValueError("free lifts must be independent and fit the raw dimension")
        free_pairing = as_matrix(self.free_character_pairing, "$PrimalTorsorChart.free_character_pairing")
        if free_pairing != identity_matrix(free_rank):
            raise ValueError("free-character pairing must be the integer identity")

        if isinstance(self.torsion_pairing, (str, bytes)) or not isinstance(self.torsion_pairing, Sequence):
            raise TypeError("$PrimalTorsorChart.torsion_pairing: expected phase matrix")
        torsion_rank = len(self.torsion_pairing)
        torsion_pairing = _phase_matrix(
            self.torsion_pairing,
            "$PrimalTorsorChart.torsion_pairing",
            rows=torsion_rank,
            columns=torsion_rank,
        )
        for row in range(torsion_rank):
            diagonal = torsion_pairing[row][row].value
            if diagonal.numerator != 1 or diagonal.denominator <= 1:
                raise ValueError("torsion-pairing diagonal must be the canonical phase 1/n")
            if any(torsion_pairing[row][column] != ZERO_PHASE for column in range(torsion_rank) if column != row):
                raise ValueError("torsion-pairing off-diagonal entries must vanish")
        torsion_lifts = _phase_matrix(
            self.torsion_lifts,
            "$PrimalTorsorChart.torsion_lifts",
            rows=raw_dimension,
            columns=torsion_rank,
        )
        if isinstance(self.quotient_witnesses, (str, bytes)) or not isinstance(self.quotient_witnesses, Sequence):
            raise TypeError("$PrimalTorsorChart.quotient_witnesses: expected matrix tuple")
        witnesses = tuple(
            as_matrix(witness, f"$PrimalTorsorChart.quotient_witnesses[{index}]")
            for index, witness in enumerate(self.quotient_witnesses)
        )
        if len(witnesses) != torsion_rank:
            raise ValueError("one quotient witness is required per torsion lift")
        if any(witness.shape != (2, raw_dimension) for witness in witnesses):
            raise ValueError("each quotient witness must contain character and numerator rows")
        torsion_orders = tuple(
            torsion_pairing[index][index].value.denominator
            for index in range(torsion_rank)
        )
        if any(
            next_order % order
            for order, next_order in zip(torsion_orders, torsion_orders[1:])
        ):
            raise ValueError("torsion invariant orders must divide their successors")
        for index, (witness, order) in enumerate(zip(witnesses, torsion_orders, strict=True)):
            character = witness[0]
            numerator = witness[1]
            expected_numerator: list[int] = []
            for row in range(raw_dimension):
                scaled = torsion_lifts[row][index].value * order
                if scaled.denominator != 1:
                    raise ValueError("torsion lift denominator does not divide its certified order")
                expected_numerator.append(scaled.numerator % order)
            if any(not 0 <= value < order for value in numerator) or numerator != tuple(expected_numerator):
                raise ValueError("quotient-witness numerator does not encode the torsion lift")
            for column in range(torsion_rank):
                lift = tuple(torsion_lifts[row][column] for row in range(raw_dimension))
                if _phase_dot(character, lift) != torsion_pairing[index][column]:
                    raise ValueError("quotient-witness character does not reproduce torsion pairing")
            for column in range(free_rank):
                evaluation = sum(
                    character[row] * free_lifts[row][column]
                    for row in range(raw_dimension)
                )
                if evaluation != 0:
                    raise ValueError("quotient-witness character has nonzero free mixed pairing")
        object.__setattr__(self, "raw_dimension", raw_dimension)
        object.__setattr__(self, "free_lifts", free_lifts)
        object.__setattr__(self, "torsion_lifts", torsion_lifts)
        object.__setattr__(self, "free_character_pairing", free_pairing)
        object.__setattr__(self, "torsion_pairing", torsion_pairing)
        object.__setattr__(self, "quotient_witnesses", witnesses)

    @property
    def free_rank(self) -> int:
        return self.free_lifts.column_count

    @property
    def torsion_rank(self) -> int:
        return len(self.torsion_pairing)


@dataclass(frozen=True, slots=True)
class TorusSolvabilityWitness:
    smith: SmithForm
    transformed_offset: tuple[Phase, ...]
    smith_coordinates: tuple[Phase, ...]
    zero_row_characters: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.smith) is not SmithForm:
            raise TypeError("$TorusSolvabilityWitness.smith: expected SmithForm")
        transformed = _phase_vector(
            self.transformed_offset,
            "$TorusSolvabilityWitness.transformed_offset",
            self.smith.diagonal.row_count,
        )
        coordinates = _phase_vector(
            self.smith_coordinates,
            "$TorusSolvabilityWitness.smith_coordinates",
            self.smith.diagonal.column_count,
        )
        if isinstance(self.zero_row_characters, (str, bytes)) or not isinstance(self.zero_row_characters, Sequence):
            raise TypeError("$TorusSolvabilityWitness.zero_row_characters: expected vectors")
        characters = tuple(
            _integer_vector(
                character,
                f"$TorusSolvabilityWitness.zero_row_characters[{index}]",
                self.smith.diagonal.row_count,
            )
            for index, character in enumerate(self.zero_row_characters)
        )
        rank = self.smith.rank
        if len(characters) != self.smith.diagonal.row_count - rank:
            raise ValueError("zero-row character count differs from Smith corank")
        for index in range(rank):
            order = self.smith.invariant_factors[index]
            if _phase(order * coordinates[index].value) != transformed[index]:
                raise ValueError("Smith coordinate does not solve transformed equation")
        if any(value != ZERO_PHASE for value in coordinates[rank:]):
            raise ValueError("nonpivot Smith coordinates must use the canonical zero lift")
        if any(value != ZERO_PHASE for value in transformed[rank:]):
            raise ValueError("solvability witness contains a nonzero zero-row phase")
        object.__setattr__(self, "transformed_offset", transformed)
        object.__setattr__(self, "smith_coordinates", coordinates)
        object.__setattr__(self, "zero_row_characters", characters)


@dataclass(frozen=True, slots=True)
class TorusObstruction:
    equation_matrix: MatrixZ
    offset: tuple[Phase, ...]
    character: tuple[int, ...]
    phase: Phase
    smith: SmithForm

    def __post_init__(self) -> None:
        equation = as_matrix(self.equation_matrix, "$TorusObstruction.equation_matrix")
        offset = _phase_vector(self.offset, "$TorusObstruction.offset", equation.row_count)
        character = _integer_vector(self.character, "$TorusObstruction.character", equation.row_count)
        if type(self.phase) is not Phase:
            raise TypeError("$TorusObstruction.phase: expected Phase")
        if type(self.smith) is not SmithForm:
            raise TypeError("$TorusObstruction.smith: expected SmithForm")
        if matmul(matmul(self.smith.left, equation), self.smith.right) != self.smith.diagonal:
            raise ValueError("obstruction Smith witness does not replay")
        annihilation = matmul(
            transpose(equation),
            MatrixZ(tuple((entry,) for entry in character), column_count=1),
        )
        if annihilation != zero_matrix(equation.column_count, 1):
            raise ValueError("obstruction character does not annihilate D")
        if _phase_dot(character, offset) != self.phase or self.phase == ZERO_PHASE:
            raise ValueError("obstruction phase must be the nonzero character evaluation")
        object.__setattr__(self, "equation_matrix", equation)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "character", character)


@dataclass(frozen=True, slots=True)
class TorsorCoordinates:
    free: tuple[Phase, ...]
    torsion: tuple[int, ...]
    quotient_preimage: tuple[Phase, ...]

    def __post_init__(self) -> None:
        free = _phase_vector(self.free, "$TorsorCoordinates.free")
        torsion = _integer_vector(self.torsion, "$TorsorCoordinates.torsion")
        preimage = _phase_vector(self.quotient_preimage, "$TorsorCoordinates.quotient_preimage")
        object.__setattr__(self, "free", free)
        object.__setattr__(self, "torsion", torsion)
        object.__setattr__(self, "quotient_preimage", preimage)


@dataclass(frozen=True, slots=True)
class TorusSolution:
    basepoint: tuple[Phase, ...]
    group: CompactGroupPresentation
    primal_chart: PrimalTorsorChart
    solvability_witness: TorusSolvabilityWitness
    equation_matrix: MatrixZ
    quotient_matrix: MatrixZ
    offset: tuple[Phase, ...]

    def __post_init__(self) -> None:
        equation, quotient, offset = _problem(
            self.equation_matrix,
            self.quotient_matrix,
            self.offset,
        )
        basepoint = _phase_vector(self.basepoint, "$TorusSolution.basepoint", equation.column_count)
        if type(self.group) is not CompactGroupPresentation:
            raise TypeError("$TorusSolution.group: expected CompactGroupPresentation")
        if type(self.primal_chart) is not PrimalTorsorChart:
            raise TypeError("$TorusSolution.primal_chart: expected PrimalTorsorChart")
        if type(self.solvability_witness) is not TorusSolvabilityWitness:
            raise TypeError("$TorusSolution.solvability_witness: expected TorusSolvabilityWitness")
        if _phase_matvec(equation, basepoint) != offset:
            raise ValueError("basepoint does not satisfy affine torus equation")
        smith = self.solvability_witness.smith
        if matmul(matmul(smith.left, equation), smith.right) != smith.diagonal:
            raise ValueError("solvability Smith witness does not replay")
        if _phase_matvec(smith.left, offset) != self.solvability_witness.transformed_offset:
            raise ValueError("transformed offset differs from solvability witness")
        expected_basepoint = _phase_matvec(smith.right, self.solvability_witness.smith_coordinates)
        if expected_basepoint != basepoint:
            raise ValueError("basepoint differs from Smith-coordinate witness")
        expected_characters = tuple(
            smith.left[row]
            for row in range(smith.rank, equation.row_count)
        )
        if self.solvability_witness.zero_row_characters != expected_characters:
            raise ValueError("zero-row characters differ from Smith left witness")
        for character in expected_characters:
            annihilation = matmul(
                transpose(equation),
                MatrixZ(tuple((entry,) for entry in character), column_count=1),
            )
            if annihilation != zero_matrix(equation.column_count, 1):
                raise ValueError("zero-row character does not annihilate the equation matrix")

        expected_group, expected_chart = _homogeneous_presentation(equation, quotient)
        if self.group != expected_group:
            raise ValueError("compact-group presentation does not match D and B")
        if self.primal_chart != expected_chart:
            raise ValueError("primal chart does not match certified dual reduction")
        object.__setattr__(self, "basepoint", basepoint)
        object.__setattr__(self, "equation_matrix", equation)
        object.__setattr__(self, "quotient_matrix", quotient)
        object.__setattr__(self, "offset", offset)


def _problem(
    D: MatrixInput,
    B: MatrixInput,
    offset: Sequence[Phase],
) -> tuple[MatrixZ, MatrixZ, tuple[Phase, ...]]:
    equation = as_matrix(D, "$solve_torus_quotient.D")
    quotient = as_matrix(B, "$solve_torus_quotient.B")
    if quotient.row_count != equation.column_count:
        raise ValueError("B row count must equal the raw torus dimension")
    normalized_offset = _phase_vector(offset, "$solve_torus_quotient.offset", equation.row_count)
    composite = matmul(equation, quotient)
    if composite != zero_matrix(equation.row_count, quotient.column_count):
        raise ValueError("D @ B must vanish over the integers")
    return equation, quotient, normalized_offset


def _solve_torus_map(
    matrix: MatrixZ,
    target: tuple[Phase, ...],
) -> tuple[tuple[Phase, ...], TorusSolvabilityWitness] | TorusObstruction:
    smith = smith_form(matrix)
    transformed = _phase_matvec(smith.left, target)
    coordinates = [ZERO_PHASE for _ in range(matrix.column_count)]
    for index, factor in enumerate(smith.invariant_factors):
        coordinates[index] = _phase(transformed[index].value / factor)
    for row in range(smith.rank, matrix.row_count):
        if transformed[row] != ZERO_PHASE:
            character = smith.left[row]
            return TorusObstruction(matrix, target, character, transformed[row], smith)
    characters = tuple(smith.left[row] for row in range(smith.rank, matrix.row_count))
    witness = TorusSolvabilityWitness(
        smith,
        transformed,
        tuple(coordinates),
        characters,
    )
    raw = _phase_matvec(smith.right, tuple(coordinates))
    if _phase_matvec(matrix, raw) != target:
        raise ArithmeticError("constructed torus preimage failed direct verification")
    return raw, witness


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
    if matmul(kernel_basis, relation_coordinates) != transpose(equation):
        raise ValueError("im(D.T) is not contained in the saturated kernel of B.T")
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
    generator_count = len(selected_indices)
    relations = MatrixZ(
        tuple(
            tuple(
                (
                    torsion_orders[row - len(free_indices)]
                    if row == column and row >= len(free_indices)
                    else 0
                )
                for column in range(generator_count)
            )
            for row in range(generator_count)
        ),
        column_count=generator_count,
    )
    group = CompactGroupPresentation(
        len(free_indices),
        torsion_orders,
        dual_generators,
        relations,
    )

    # If L K = I for the saturated kernel basis K, then
    # X = L.T U.T is an integer right inverse of (K U^-1).T.
    right_inverse = matmul(
        transpose(quotient_kernel.coordinate_projection),
        transpose(presentation_smith.left),
    )
    if matmul(transpose(all_dual_generators), right_inverse) != identity_matrix(kernel_rank):
        raise ArithmeticError("dual-character right inverse failed")

    free_lifts = _matrix_columns(right_inverse, free_indices)
    torsion_integer_lifts = _matrix_columns(right_inverse, torsion_indices)
    torsion_lifts = tuple(
        tuple(
            _phase(Fraction(torsion_integer_lifts[row][column], torsion_orders[column]))
            for column in range(len(torsion_orders))
        )
        for row in range(raw_dimension)
    )
    free_pairing = identity_matrix(len(free_indices))
    torsion_pairing = tuple(
        tuple(
            _phase(Fraction(1, torsion_orders[row]))
            if row == column
            else ZERO_PHASE
            for column in range(len(torsion_orders))
        )
        for row in range(len(torsion_orders))
    )
    quotient_witnesses = tuple(
        MatrixZ(
            (
                tuple(dual_generators[row][len(free_indices) + index] for row in range(raw_dimension)),
                tuple(torsion_integer_lifts[row][index] % order for row in range(raw_dimension)),
            ),
            column_count=raw_dimension,
        )
        for index, order in enumerate(torsion_orders)
    )
    chart = PrimalTorsorChart(
        raw_dimension,
        free_lifts,
        torsion_lifts,
        free_pairing,
        torsion_pairing,
        quotient_witnesses,
    )

    # Replay the two separately typed pairing blocks and all mixed blocks.
    free_dual = _matrix_columns(dual_generators, range(len(free_indices)))
    torsion_dual = _matrix_columns(
        dual_generators,
        range(len(free_indices), generator_count),
    )
    if matmul(transpose(free_dual), free_lifts) != free_pairing:
        raise ArithmeticError("integer free-character pairing is not identity")
    if matmul(transpose(torsion_dual), free_lifts) != zero_matrix(len(torsion_orders), len(free_indices)):
        raise ArithmeticError("torsion/free mixed pairing does not vanish integrally")
    for row in range(len(free_indices)):
        character = tuple(free_dual[index][row] for index in range(raw_dimension))
        for column in range(len(torsion_orders)):
            lift = tuple(torsion_lifts[index][column] for index in range(raw_dimension))
            if _phase_dot(character, lift) != ZERO_PHASE:
                raise ArithmeticError("free/torsion mixed phase pairing does not vanish")
    for row, order in enumerate(torsion_orders):
        character = tuple(torsion_dual[index][row] for index in range(raw_dimension))
        for column in range(len(torsion_orders)):
            lift = tuple(torsion_lifts[index][column] for index in range(raw_dimension))
            expected = _phase(Fraction(1, order)) if row == column else ZERO_PHASE
            if _phase_dot(character, lift) != expected:
                raise ArithmeticError("torsion pairing differs from Smith diagonal")

    # Free lifts solve D l = 0 over Z; torsion lifts solve it in R/Z.
    if matmul(equation, free_lifts) != zero_matrix(equation.row_count, len(free_indices)):
        raise ArithmeticError("free primal lift is not an exact homogeneous lift")
    for column in range(len(torsion_orders)):
        lift = tuple(torsion_lifts[row][column] for row in range(raw_dimension))
        if _phase_matvec(equation, lift) != (ZERO_PHASE,) * equation.row_count:
            raise ArithmeticError("torsion primal lift is not homogeneous")
    return group, chart


def solve_torus_quotient(
    D: MatrixInput,
    B: MatrixInput,
    offset: Sequence[Phase],
) -> TorusSolution | TorusObstruction:
    """Solve an affine compact-torus quotient with replayable certificates."""

    equation, quotient, normalized_offset = _problem(D, B, offset)
    affine = _solve_torus_map(equation, normalized_offset)
    if isinstance(affine, TorusObstruction):
        return affine
    basepoint, witness = affine
    group, chart = _homogeneous_presentation(equation, quotient)
    return TorusSolution(
        basepoint,
        group,
        chart,
        witness,
        equation,
        quotient,
        normalized_offset,
    )


def raw_torsor_point(
    solution: TorusSolution,
    free: Sequence[Phase],
    torsion: Sequence[int],
) -> tuple[Phase, ...]:
    """Lift canonical symbolic torsor coordinates to raw ``z`` phases."""

    if type(solution) is not TorusSolution:
        raise TypeError("$raw_torsor_point.solution: expected TorusSolution")
    free_coordinates = _phase_vector(free, "$raw_torsor_point.free", solution.group.free_rank)
    torsion_coordinates = _integer_vector(
        torsion,
        "$raw_torsor_point.torsion",
        len(solution.group.torsion_orders),
    )
    for index, (coordinate, order) in enumerate(zip(torsion_coordinates, solution.group.torsion_orders, strict=True)):
        if not 0 <= coordinate < order:
            raise ValueError(f"$raw_torsor_point.torsion[{index}]: expected canonical residue modulo {order}")
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
    result = tuple(raw)
    if _phase_matvec(solution.equation_matrix, result) != solution.offset:
        raise ArithmeticError("lifted raw point does not satisfy affine equation")
    return result


def torsor_coordinates(
    solution: TorusSolution,
    raw: Sequence[Phase],
) -> TorsorCoordinates:
    """Decode a raw affine point and certify its boundary-torus remainder."""

    if type(solution) is not TorusSolution:
        raise TypeError("$torsor_coordinates.solution: expected TorusSolution")
    point = _phase_vector(raw, "$torsor_coordinates.raw", solution.equation_matrix.column_count)
    if _phase_matvec(solution.equation_matrix, point) != solution.offset:
        raise ValueError("raw point does not satisfy the affine torus equation")
    homogeneous = _phase_subtract(point, solution.basepoint)
    free_values: list[Phase] = []
    torsion_values: list[int] = []
    for column in range(solution.group.free_rank):
        character = tuple(solution.group.dual_generators[row][column] for row in range(solution.equation_matrix.column_count))
        free_values.append(_phase_dot(character, homogeneous))
    for index, order in enumerate(solution.group.torsion_orders):
        column = solution.group.free_rank + index
        character = tuple(solution.group.dual_generators[row][column] for row in range(solution.equation_matrix.column_count))
        evaluation = _phase_dot(character, homogeneous)
        scaled = evaluation.value * order
        if scaled.denominator != 1:
            raise ValueError("raw point has a nonintegral torsion-character coordinate")
        torsion_values.append(scaled.numerator % order)
    free_tuple = tuple(free_values)
    torsion_tuple = tuple(torsion_values)
    reconstructed = raw_torsor_point(solution, free_tuple, torsion_tuple)
    remainder = _phase_subtract(point, reconstructed)
    quotient_solution = _solve_torus_map(solution.quotient_matrix, remainder)
    if isinstance(quotient_solution, TorusObstruction):
        raise ValueError("raw point does not agree with decoded coordinates modulo im(B)")
    quotient_preimage, _ = quotient_solution
    if _phase_matvec(solution.quotient_matrix, quotient_preimage) != remainder:
        raise ArithmeticError("quotient preimage failed direct verification")
    return TorsorCoordinates(free_tuple, torsion_tuple, quotient_preimage)
