r"""Deterministic linear algebra over :math:`\mathbb F_2`.

Matrices are immutable shaped row sequences and act on column vectors.  The
explicit shape distinguishes ``0 x n`` systems from ``0 x 0``.  Public vectors
remain ordinary bit tuples while packed integers are used only inside row
reduction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload


VectorGF2 = tuple[int, ...]


def _bit(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path}: expected GF(2) bit")
    if value not in (0, 1):
        raise ValueError(f"{path}: expected GF(2) bit")
    return value


def _vector(value: Sequence[int], path: str) -> VectorGF2:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected bit vector")
    return tuple(_bit(bit, f"{path}[{index}]") for index, bit in enumerate(value))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class MatrixGF2(Sequence[VectorGF2]):
    """Immutable matrix whose shape remains defined when it has zero rows."""

    rows: tuple[VectorGF2, ...]
    column_count: int

    def __init__(
        self,
        rows: Sequence[Sequence[int]],
        column_count: int | None = None,
    ) -> None:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError("$MatrixGF2.rows: expected matrix rows")
        preserved_column_count = (
            rows.column_count if isinstance(rows, MatrixGF2) else None
        )
        normalized = tuple(
            _vector(row, f"$MatrixGF2.rows[{index}]")
            for index, row in enumerate(rows)
        )
        inferred = len(normalized[0]) if normalized else 0
        if column_count is None:
            column_count = (
                preserved_column_count
                if preserved_column_count is not None
                else inferred
            )
        if type(column_count) is not int:
            raise TypeError("$MatrixGF2.column_count: expected integer")
        if column_count < 0:
            raise ValueError("$MatrixGF2.column_count: expected nonnegative integer")
        if (
            preserved_column_count is not None
            and column_count != preserved_column_count
        ):
            raise ValueError(
                "$MatrixGF2.column_count: explicit shape differs from shaped input"
            )
        if normalized and inferred != column_count:
            raise ValueError("$MatrixGF2.rows: row width differs from column_count")
        if any(len(row) != column_count for row in normalized):
            raise ValueError("$MatrixGF2.rows: matrix rows must have equal width")
        object.__setattr__(self, "rows", normalized)
        object.__setattr__(self, "column_count", column_count)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, self.column_count)

    def __len__(self) -> int:
        return self.row_count

    @overload
    def __getitem__(self, index: int) -> VectorGF2: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[VectorGF2, ...]: ...

    def __getitem__(self, index: int | slice) -> VectorGF2 | tuple[VectorGF2, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[VectorGF2]:
        return iter(self.rows)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MatrixGF2):
            return self.shape == other.shape and self.rows == other.rows
        if type(other) is not tuple:
            return False
        if any(type(row) is not tuple for row in other):
            return False
        if any(type(bit) is not int for row in other for bit in row):
            return False
        try:
            unshaped = MatrixGF2(other)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self.shape == unshaped.shape and self.rows == unshaped.rows

    def __hash__(self) -> int:
        return hash(self.rows)


MatrixInput = MatrixGF2 | Sequence[Sequence[int]]


def _matrix(value: MatrixInput, path: str = "$matrix") -> MatrixGF2:
    if isinstance(value, MatrixGF2):
        return value
    try:
        return MatrixGF2(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from error


def _width(matrix: MatrixGF2) -> int:
    return matrix.column_count


def _pack(row: VectorGF2) -> int:
    packed = 0
    for column, bit in enumerate(row):
        packed |= bit << column
    return packed


def _unpack(row: int, width: int) -> VectorGF2:
    return tuple((row >> column) & 1 for column in range(width))


def _column(matrix: MatrixGF2, column: int) -> VectorGF2:
    return tuple(row[column] for row in matrix)


def _matrix_from_columns(vectors: Sequence[VectorGF2], ambient: int) -> MatrixGF2:
    vectors = tuple(vectors)
    if any(len(vector) != ambient for vector in vectors):
        raise ValueError("column vector has wrong ambient dimension")
    return MatrixGF2(
        tuple(tuple(vector[row] for vector in vectors) for row in range(ambient)),
        column_count=len(vectors),
    )


@dataclass(frozen=True, slots=True)
class GF2Reduction:
    reduced: MatrixGF2
    pivots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GF2AffineSolution:
    basepoint: VectorGF2
    kernel_basis: tuple[VectorGF2, ...]


@dataclass(frozen=True, slots=True)
class GF2Inconsistency:
    """Marker returned when an affine system has no solution."""


@dataclass(frozen=True, slots=True)
class GF2Character:
    bits: VectorGF2

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _vector(self.bits, "$GF2Character.bits"))


@dataclass(frozen=True, slots=True)
class GF2Quotient:
    ambient_dimension: int
    boundary_basis: tuple[VectorGF2, ...]
    representatives: tuple[VectorGF2, ...]


def rref(matrix: MatrixInput) -> GF2Reduction:
    """Return deterministic reduced row echelon form and pivot columns."""

    original = _matrix(matrix)
    row_count = len(original)
    column_count = _width(original)
    rows = [_pack(row) for row in original]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (row for row in range(pivot_row, row_count) if (rows[row] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        if selected != pivot_row:
            rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        for row in range(row_count):
            if row != pivot_row and ((rows[row] >> column) & 1):
                rows[row] ^= rows[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return GF2Reduction(
        reduced=MatrixGF2(
            tuple(_unpack(row, column_count) for row in rows),
            column_count=column_count,
        ),
        pivots=tuple(pivots),
    )


def kernel_basis(matrix: MatrixInput) -> tuple[VectorGF2, ...]:
    """Return the canonical free-column basis of the right kernel."""

    original = _matrix(matrix)
    column_count = _width(original)
    reduction = rref(original)
    pivot_set = set(reduction.pivots)
    basis: list[VectorGF2] = []
    for free_column in range(column_count):
        if free_column in pivot_set:
            continue
        vector = [0] * column_count
        vector[free_column] = 1
        for row, pivot in enumerate(reduction.pivots):
            vector[pivot] = reduction.reduced[row][free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def image_basis(matrix: MatrixInput) -> tuple[VectorGF2, ...]:
    """Return independent original columns in deterministic pivot order."""

    original = _matrix(matrix)
    if original.column_count == 0:
        return ()
    pivots = rref(original).pivots
    return tuple(_column(original, column) for column in pivots)


def solve_affine(
    matrix: MatrixInput,
    rhs: VectorGF2,
) -> GF2AffineSolution | GF2Inconsistency:
    """Solve ``matrix*x=rhs`` or return an inconsistency marker."""

    original = _matrix(matrix)
    right = _vector(rhs, "$rhs")
    column_count = _width(original)
    augmented = MatrixGF2(
        tuple(row + (value,) for row, value in zip(original, right)),
        column_count=column_count + 1,
    )
    reduction = rref(augmented)
    for reduced_row in reduction.reduced:
        if not any(reduced_row[:column_count]) and reduced_row[column_count]:
            return GF2Inconsistency()
    basepoint = [0] * _width(original)
    for row, pivot in enumerate(reduction.pivots):
        if pivot < column_count:
            basepoint[pivot] = reduction.reduced[row][column_count]
    return GF2AffineSolution(tuple(basepoint), kernel_basis(original))


def _coordinates_in_basis(
    basis: Sequence[VectorGF2],
    vector: VectorGF2,
    ambient: int,
) -> VectorGF2 | None:
    matrix = _matrix_from_columns(tuple(basis), ambient)
    result = solve_affine(matrix, vector)
    if isinstance(result, GF2Inconsistency):
        return None
    return result.basepoint


def quotient_basis(cycles: MatrixInput, boundaries: MatrixInput) -> GF2Quotient:
    """Return a deterministic basis for ``span(cycles)/span(boundaries)``."""

    cycle_matrix = _matrix(cycles, "$cycles")
    boundary_matrix = _matrix(boundaries, "$boundaries")
    ambient = cycle_matrix.row_count
    cycle_vectors = image_basis(cycle_matrix)
    boundary_vectors = image_basis(boundary_matrix)

    span_basis: list[VectorGF2] = list(boundary_vectors)
    representatives: list[VectorGF2] = []
    for vector in cycle_vectors:
        if _coordinates_in_basis(span_basis, vector, ambient) is None:
            representatives.append(vector)
            span_basis.append(vector)

    return GF2Quotient(
        ambient_dimension=ambient,
        boundary_basis=boundary_vectors,
        representatives=tuple(representatives),
    )


__all__ = [
    "GF2AffineSolution",
    "GF2Character",
    "GF2Inconsistency",
    "GF2Quotient",
    "GF2Reduction",
    "MatrixGF2",
    "VectorGF2",
    "image_basis",
    "kernel_basis",
    "quotient_basis",
    "rref",
    "solve_affine",
]
