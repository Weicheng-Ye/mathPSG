r"""Deterministic, witness-producing linear algebra over :math:`\mathbb F_2`.

Matrices are immutable shaped row sequences and act on column vectors.  The
explicit shape distinguishes ``0 x n`` systems from ``0 x 0``.  Public vectors
remain ordinary bit tuples so every reduction, solution, obstruction, and
quotient claim can be replayed without depending on the packed-bit
implementation used internally.
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


def _identity(size: int) -> MatrixGF2:
    return MatrixGF2(
        tuple(
            tuple(int(row == column) for column in range(size))
            for row in range(size)
        ),
        column_count=size,
    )


def _pack(row: VectorGF2) -> int:
    packed = 0
    for column, bit in enumerate(row):
        packed |= bit << column
    return packed


def _unpack(row: int, width: int) -> VectorGF2:
    return tuple((row >> column) & 1 for column in range(width))


def _dot(left: VectorGF2, right: VectorGF2) -> int:
    if len(left) != len(right):
        raise ValueError("dot-product dimensions differ")
    return sum((a & b for a, b in zip(left, right, strict=True)), 0) & 1


def _matvec(matrix: MatrixGF2, vector: VectorGF2) -> VectorGF2:
    if _width(matrix) != len(vector):
        raise ValueError("matrix/vector dimensions differ")
    return tuple(_dot(row, vector) for row in matrix)


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


def _rank(matrix: MatrixGF2) -> int:
    rows = [_pack(row) for row in matrix]
    pivot_row = 0
    for column in range(matrix.column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, matrix.row_count)
                if (rows[row] >> column) & 1
            ),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        for row in range(pivot_row + 1, matrix.row_count):
            if (rows[row] >> column) & 1:
                rows[row] ^= rows[pivot_row]
        pivot_row += 1
        if pivot_row == matrix.row_count:
            break
    return pivot_row


def _vector_family(
    value: Sequence[Sequence[int]],
    path: str,
    ambient: int,
    *,
    independent: bool,
) -> tuple[VectorGF2, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected vector family")
    vectors = tuple(
        _vector(vector, f"{path}[{index}]")
        for index, vector in enumerate(value)
    )
    if any(len(vector) != ambient for vector in vectors):
        raise ValueError(f"{path}: vector has wrong ambient dimension")
    if independent and _rank(_matrix_from_columns(vectors, ambient)) != len(vectors):
        raise ValueError(f"{path}: vectors must be linearly independent")
    return vectors


def _linear_combination(
    vectors: tuple[VectorGF2, ...],
    coefficients: VectorGF2,
    ambient: int,
) -> VectorGF2:
    if len(vectors) != len(coefficients):
        raise ValueError("coefficient dimension differs from basis dimension")
    return tuple(
        sum(
            (
                coefficient & vector[row]
                for coefficient, vector in zip(coefficients, vectors, strict=True)
            ),
            0,
        )
        & 1
        for row in range(ambient)
    )


@dataclass(frozen=True, slots=True)
class GF2Reduction:
    reduced: MatrixGF2
    left_witness: MatrixGF2
    pivots: tuple[int, ...]

    def __post_init__(self) -> None:
        reduced = _matrix(self.reduced, "$GF2Reduction.reduced")
        witness = _matrix(self.left_witness, "$GF2Reduction.left_witness")
        if isinstance(self.pivots, (str, bytes)) or not isinstance(
            self.pivots, Sequence
        ):
            raise TypeError("$GF2Reduction.pivots: expected integer sequence")
        pivots = tuple(self.pivots)
        if any(
            type(pivot) is not int
            for pivot in pivots
        ):
            raise TypeError("$GF2Reduction.pivots: expected integers")
        if witness.shape != (reduced.row_count, reduced.row_count):
            raise ValueError("$GF2Reduction.left_witness: expected square row witness")
        if _rank(witness) != reduced.row_count:
            raise ValueError("$GF2Reduction.left_witness: witness must be invertible")

        leading_columns: list[int] = []
        saw_zero_row = False
        for row in reduced:
            leading = next((column for column, bit in enumerate(row) if bit), None)
            if leading is None:
                saw_zero_row = True
                continue
            if saw_zero_row:
                raise ValueError("$GF2Reduction.reduced: nonzero row follows zero row")
            leading_columns.append(leading)
        if tuple(leading_columns) != pivots:
            raise ValueError("$GF2Reduction.pivots: do not match reduced leading columns")
        if any(left >= right for left, right in zip(pivots, pivots[1:])):
            raise ValueError("$GF2Reduction.pivots: expected strictly increasing pivots")
        for row, pivot in enumerate(pivots):
            if pivot < 0 or pivot >= reduced.column_count:
                raise ValueError("$GF2Reduction.pivots: pivot outside matrix")
            if any(
                reduced[other][pivot]
                for other in range(reduced.row_count)
                if other != row
            ):
                raise ValueError("$GF2Reduction.reduced: pivot column is not reduced")
        object.__setattr__(self, "reduced", reduced)
        object.__setattr__(self, "left_witness", witness)
        object.__setattr__(self, "pivots", pivots)


@dataclass(frozen=True, slots=True)
class GF2AffineSolution:
    basepoint: VectorGF2
    kernel_basis: tuple[VectorGF2, ...]

    def __post_init__(self) -> None:
        basepoint = _vector(self.basepoint, "$GF2AffineSolution.basepoint")
        basis = _vector_family(
            self.kernel_basis,
            "$GF2AffineSolution.kernel_basis",
            len(basepoint),
            independent=True,
        )
        object.__setattr__(self, "basepoint", basepoint)
        object.__setattr__(self, "kernel_basis", basis)


@dataclass(frozen=True, slots=True)
class GF2Inconsistency:
    left_null_vector: VectorGF2

    def __post_init__(self) -> None:
        vector = _vector(
            self.left_null_vector,
            "$GF2Inconsistency.left_null_vector",
        )
        if not any(vector):
            raise ValueError(
                "$GF2Inconsistency.left_null_vector: witness must be nonzero"
            )
        object.__setattr__(self, "left_null_vector", vector)


@dataclass(frozen=True, slots=True)
class GF2Character:
    bits: VectorGF2

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _vector(self.bits, "$GF2Character.bits"))


@dataclass(frozen=True, slots=True)
class GF2AffineArrow:
    linear: MatrixGF2
    shift: VectorGF2

    def __post_init__(self) -> None:
        matrix = _matrix(self.linear, "$GF2AffineArrow.linear")
        shift = _vector(self.shift, "$GF2AffineArrow.shift")
        if len(shift) != len(matrix):
            raise ValueError("$GF2AffineArrow.shift: target dimension mismatch")
        object.__setattr__(self, "linear", matrix)
        object.__setattr__(self, "shift", shift)

    @property
    def source_dimension(self) -> int:
        return _width(self.linear)

    @property
    def target_dimension(self) -> int:
        return len(self.linear)

    def apply(self, vector: Sequence[int]) -> VectorGF2:
        point = _vector(vector, "$GF2AffineArrow.input")
        if len(point) != self.source_dimension:
            raise ValueError("$GF2AffineArrow.input: source dimension mismatch")
        image = _matvec(self.linear, point)
        return tuple(value ^ shift for value, shift in zip(image, self.shift, strict=True))


@dataclass(frozen=True, slots=True)
class GF2Quotient:
    ambient_dimension: int
    cycle_basis: tuple[VectorGF2, ...]
    boundary_basis: tuple[VectorGF2, ...]
    representatives: tuple[VectorGF2, ...]
    boundary_coordinates: tuple[VectorGF2, ...]

    def __post_init__(self) -> None:
        if type(self.ambient_dimension) is not int:
            raise TypeError("$GF2Quotient.ambient_dimension: expected integer")
        if self.ambient_dimension < 0:
            raise ValueError(
                "$GF2Quotient.ambient_dimension: expected nonnegative integer"
            )
        ambient = self.ambient_dimension
        cycles = _vector_family(
            self.cycle_basis,
            "$GF2Quotient.cycle_basis",
            ambient,
            independent=True,
        )
        boundaries = _vector_family(
            self.boundary_basis,
            "$GF2Quotient.boundary_basis",
            ambient,
            independent=True,
        )
        representatives = _vector_family(
            self.representatives,
            "$GF2Quotient.representatives",
            ambient,
            independent=True,
        )
        coordinates = _vector_family(
            self.boundary_coordinates,
            "$GF2Quotient.boundary_coordinates",
            len(cycles),
            independent=True,
        )
        if len(coordinates) != len(boundaries):
            raise ValueError(
                "$GF2Quotient.boundary_coordinates: expected one row per boundary"
            )
        for vector, coefficients in zip(boundaries, coordinates, strict=True):
            if _linear_combination(cycles, coefficients, ambient) != vector:
                raise ValueError(
                    "$GF2Quotient.boundary_coordinates: coordinate witness does not replay"
                )
        decomposition = boundaries + representatives
        if _rank(_matrix_from_columns(decomposition, ambient)) != len(decomposition):
            raise ValueError(
                "$GF2Quotient.representatives: not independent modulo boundaries"
            )
        if len(decomposition) != len(cycles):
            raise ValueError(
                "$GF2Quotient: boundary and representative spans do not exhaust cycles"
            )
        if _rank(_matrix_from_columns(decomposition + cycles, ambient)) != len(
            decomposition
        ):
            raise ValueError("$GF2Quotient: decomposition does not span cycle basis")
        object.__setattr__(self, "cycle_basis", cycles)
        object.__setattr__(self, "boundary_basis", boundaries)
        object.__setattr__(self, "representatives", representatives)
        object.__setattr__(self, "boundary_coordinates", coordinates)
        object.__setattr__(self, "ambient_dimension", ambient)

    @property
    def dimension(self) -> int:
        return len(self.representatives)


def rref(matrix: MatrixInput) -> GF2Reduction:
    """Return deterministic reduced row echelon form and its left witness."""

    original = _matrix(matrix)
    row_count = len(original)
    column_count = _width(original)
    rows = [_pack(row) for row in original]
    witness = [1 << row for row in range(row_count)]
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
            witness[pivot_row], witness[selected] = witness[selected], witness[pivot_row]
        for row in range(row_count):
            if row != pivot_row and ((rows[row] >> column) & 1):
                rows[row] ^= rows[pivot_row]
                witness[row] ^= witness[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return GF2Reduction(
        reduced=MatrixGF2(
            tuple(_unpack(row, column_count) for row in rows),
            column_count=column_count,
        ),
        left_witness=MatrixGF2(
            tuple(_unpack(row, row_count) for row in witness),
            column_count=row_count,
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
    """Solve ``matrix*x=rhs`` or return a left-null inconsistency witness."""

    original = _matrix(matrix)
    right = _vector(rhs, "$rhs")
    if len(right) != len(original):
        raise ValueError("$rhs: row dimension mismatch")
    reduction = rref(original)
    transformed = tuple(_dot(row, right) for row in reduction.left_witness)
    for row, (reduced_row, value) in enumerate(
        zip(reduction.reduced, transformed, strict=True)
    ):
        if not any(reduced_row) and value:
            return GF2Inconsistency(reduction.left_witness[row])
    basepoint = [0] * _width(original)
    for row, pivot in enumerate(reduction.pivots):
        basepoint[pivot] = transformed[row]
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
    """Certify ``span(cycles)/span(boundaries)`` from column-generator matrices."""

    cycle_matrix = _matrix(cycles, "$cycles")
    boundary_matrix = _matrix(boundaries, "$boundaries")
    if cycle_matrix.row_count != boundary_matrix.row_count:
        raise ValueError("cycle and boundary ambient dimensions differ")
    ambient = cycle_matrix.row_count
    cycle_vectors = image_basis(cycle_matrix)
    boundary_vectors = image_basis(boundary_matrix)
    boundary_coordinates: list[VectorGF2] = []
    for vector in boundary_vectors:
        coordinates = _coordinates_in_basis(cycle_vectors, vector, ambient)
        if coordinates is None:
            raise ValueError("boundary space is not contained in cycle space")
        boundary_coordinates.append(coordinates)

    span_basis: list[VectorGF2] = list(boundary_vectors)
    representatives: list[VectorGF2] = []
    for vector in cycle_vectors:
        if _coordinates_in_basis(span_basis, vector, ambient) is None:
            representatives.append(vector)
            span_basis.append(vector)

    return GF2Quotient(
        ambient_dimension=ambient,
        cycle_basis=cycle_vectors,
        boundary_basis=boundary_vectors,
        representatives=tuple(representatives),
        boundary_coordinates=tuple(boundary_coordinates),
    )


__all__ = [
    "GF2AffineArrow",
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
