r"""Certified exact linear algebra over :math:`\mathbb Z`.

Matrices act on column vectors.  ``MatrixZ`` retains its column count even
when it has no rows, so the public boundary distinguishes ``0 x n`` from
``0 x 0``.  Normal forms carry the unimodular transformations needed to
replay every claimed equality; no optional computer-algebra dependency is
used at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import overload


VectorZ = tuple[int, ...]


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path}: expected integer")
    return value


def _vector(value: Sequence[int], path: str) -> VectorZ:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected integer vector")
    return tuple(_integer(entry, f"{path}[{index}]") for index, entry in enumerate(value))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class MatrixZ(Sequence[VectorZ]):
    """Immutable shaped integer matrix."""

    rows: tuple[VectorZ, ...]
    column_count: int

    def __init__(
        self,
        rows: Sequence[Sequence[int]],
        column_count: int | None = None,
    ) -> None:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError("$MatrixZ.rows: expected matrix rows")
        preserved = rows.column_count if isinstance(rows, MatrixZ) else None
        normalized = tuple(_vector(row, f"$MatrixZ.rows[{index}]") for index, row in enumerate(rows))
        inferred = len(normalized[0]) if normalized else 0
        if column_count is None:
            column_count = preserved if preserved is not None else inferred
        column_count = _integer(column_count, "$MatrixZ.column_count")
        if column_count < 0:
            raise ValueError("$MatrixZ.column_count: expected nonnegative integer")
        if preserved is not None and preserved != column_count:
            raise ValueError("$MatrixZ.column_count: explicit shape differs from shaped input")
        if any(len(row) != column_count for row in normalized):
            raise ValueError("$MatrixZ.rows: matrix rows must have equal width")
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
    def __getitem__(self, index: int) -> VectorZ: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[VectorZ, ...]: ...

    def __getitem__(self, index: int | slice) -> VectorZ | tuple[VectorZ, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[VectorZ]:
        return iter(self.rows)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MatrixZ):
            return self.shape == other.shape and self.rows == other.rows
        if type(other) is not tuple or any(type(row) is not tuple for row in other):
            return False
        if any(type(entry) is not int for row in other for entry in row):
            return False
        try:
            unshaped = MatrixZ(other)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self.shape == unshaped.shape and self.rows == unshaped.rows

    def __hash__(self) -> int:
        # Backward-compatible tuple equality requires the same hash.
        return hash(self.rows)


MatrixInput = MatrixZ | Sequence[Sequence[int]]


def as_matrix(value: MatrixInput, path: str = "$matrix") -> MatrixZ:
    if isinstance(value, MatrixZ):
        return value
    try:
        return MatrixZ(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from error


def zero_matrix(rows: int, columns: int) -> MatrixZ:
    rows = _integer(rows, "$zero_matrix.rows")
    columns = _integer(columns, "$zero_matrix.columns")
    if rows < 0 or columns < 0:
        raise ValueError("zero-matrix dimensions must be nonnegative")
    return MatrixZ(tuple((0,) * columns for _ in range(rows)), column_count=columns)


def identity_matrix(size: int) -> MatrixZ:
    size = _integer(size, "$identity_matrix.size")
    if size < 0:
        raise ValueError("identity-matrix dimension must be nonnegative")
    return MatrixZ(
        tuple(tuple(int(row == column) for column in range(size)) for row in range(size)),
        column_count=size,
    )


def transpose(matrix: MatrixInput) -> MatrixZ:
    source = as_matrix(matrix)
    return MatrixZ(
        tuple(
            tuple(source[row][column] for row in range(source.row_count))
            for column in range(source.column_count)
        ),
        column_count=source.row_count,
    )


def matmul(left: MatrixInput, right: MatrixInput) -> MatrixZ:
    a = as_matrix(left, "$matmul.left")
    b = as_matrix(right, "$matmul.right")
    if a.column_count != b.row_count:
        raise ValueError("matrix dimensions differ")
    return MatrixZ(
        tuple(
            tuple(
                sum((a[row][index] * b[index][column] for index in range(a.column_count)), 0)
                for column in range(b.column_count)
            )
            for row in range(a.row_count)
        ),
        column_count=b.column_count,
    )


def matvec(matrix: MatrixInput, vector: Sequence[int]) -> VectorZ:
    source = as_matrix(matrix)
    normalized = _vector(vector, "$matvec.vector")
    if source.column_count != len(normalized):
        raise ValueError("matrix/vector dimensions differ")
    return tuple(sum((entry * coefficient for entry, coefficient in zip(row, normalized, strict=True)), 0) for row in source)


def matrix_from_columns(vectors: Sequence[Sequence[int]], ambient: int) -> MatrixZ:
    ambient = _integer(ambient, "$matrix_from_columns.ambient")
    if ambient < 0:
        raise ValueError("ambient dimension must be nonnegative")
    normalized = tuple(_vector(vector, f"$matrix_from_columns.vectors[{index}]") for index, vector in enumerate(vectors))
    if any(len(vector) != ambient for vector in normalized):
        raise ValueError("column vector has wrong ambient dimension")
    return MatrixZ(
        tuple(tuple(vector[row] for vector in normalized) for row in range(ambient)),
        column_count=len(normalized),
    )


def determinant(matrix: MatrixInput) -> int:
    source = as_matrix(matrix)
    if source.row_count != source.column_count:
        raise ValueError("determinant requires a square matrix")
    size = source.row_count
    if size == 0:
        return 1
    work = [list(row) for row in source]
    sign = 1
    previous = 1
    for pivot_index in range(size - 1):
        selected = next((row for row in range(pivot_index, size) if work[row][pivot_index]), None)
        if selected is None:
            return 0
        if selected != pivot_index:
            work[pivot_index], work[selected] = work[selected], work[pivot_index]
            sign *= -1
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = work[row][column] * pivot - work[row][pivot_index] * work[pivot_index][column]
                if numerator % previous:
                    raise ArithmeticError("Bareiss division was not exact")
                work[row][column] = numerator // previous
            work[row][pivot_index] = 0
        previous = pivot
    return sign * work[-1][-1]


def inverse_unimodular(matrix: MatrixInput) -> MatrixZ:
    source = as_matrix(matrix)
    if source.row_count != source.column_count:
        raise ValueError("inverse requires a square matrix")
    size = source.row_count
    augmented = [
        [Fraction(entry) for entry in source[row]]
        + [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        selected = next((row for row in range(column, size) if augmented[row][column]), None)
        if selected is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[selected] = augmented[selected], augmented[column]
        pivot = augmented[column][column]
        augmented[column] = [entry / pivot for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if coefficient:
                augmented[row] = [
                    entry - coefficient * pivot_entry
                    for entry, pivot_entry in zip(augmented[row], augmented[column], strict=True)
                ]
    inverse_rows: list[VectorZ] = []
    for row in augmented:
        entries = row[size:]
        if any(entry.denominator != 1 for entry in entries):
            raise ValueError("matrix is not unimodular")
        inverse_rows.append(tuple(entry.numerator for entry in entries))
    result = MatrixZ(tuple(inverse_rows), column_count=size)
    if matmul(source, result) != identity_matrix(size):
        raise ArithmeticError("integer inverse witness failed")
    return result


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, remainder = a, b
    old_s, coefficient_s = 1, 0
    old_t, coefficient_t = 0, 1
    while remainder:
        quotient = old_r // remainder
        old_r, remainder = remainder, old_r - quotient * remainder
        old_s, coefficient_s = coefficient_s, old_s - quotient * coefficient_s
        old_t, coefficient_t = coefficient_t, old_t - quotient * coefficient_t
    if old_r < 0:
        return (-old_r, -old_s, -old_t)
    return (old_r, old_s, old_t)


def _swap_rows(matrix: list[list[int]], first: int, second: int) -> None:
    matrix[first], matrix[second] = matrix[second], matrix[first]


def _swap_columns(matrix: list[list[int]], first: int, second: int) -> None:
    for row in matrix:
        row[first], row[second] = row[second], row[first]


def _combine_rows(
    matrix: list[list[int]],
    first: int,
    second: int,
    a: int,
    b: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    divisor, x, y = _extended_gcd(a, b)
    if divisor == 0:
        raise ValueError("cannot combine two zero pivots")
    transform = ((x, y), (-b // divisor, a // divisor))
    first_row = matrix[first][:]
    second_row = matrix[second][:]
    matrix[first] = [transform[0][0] * left + transform[0][1] * right for left, right in zip(first_row, second_row, strict=True)]
    matrix[second] = [transform[1][0] * left + transform[1][1] * right for left, right in zip(first_row, second_row, strict=True)]
    return transform


def _apply_row_pair(
    witness: list[list[int]],
    first: int,
    second: int,
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    first_row = witness[first][:]
    second_row = witness[second][:]
    witness[first] = [transform[0][0] * left + transform[0][1] * right for left, right in zip(first_row, second_row, strict=True)]
    witness[second] = [transform[1][0] * left + transform[1][1] * right for left, right in zip(first_row, second_row, strict=True)]


def _combine_columns(
    matrix: list[list[int]],
    first: int,
    second: int,
    a: int,
    b: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    divisor, x, y = _extended_gcd(a, b)
    if divisor == 0:
        raise ValueError("cannot combine two zero pivots")
    # Right multiplication by [[x, -b/g], [y, a/g]].
    transform = ((x, -b // divisor), (y, a // divisor))
    first_column = [row[first] for row in matrix]
    second_column = [row[second] for row in matrix]
    for row_index, row in enumerate(matrix):
        row[first] = transform[0][0] * first_column[row_index] + transform[1][0] * second_column[row_index]
        row[second] = transform[0][1] * first_column[row_index] + transform[1][1] * second_column[row_index]
    return transform


def _apply_column_pair(
    witness: list[list[int]],
    first: int,
    second: int,
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    first_column = [row[first] for row in witness]
    second_column = [row[second] for row in witness]
    for row_index, row in enumerate(witness):
        row[first] = transform[0][0] * first_column[row_index] + transform[1][0] * second_column[row_index]
        row[second] = transform[0][1] * first_column[row_index] + transform[1][1] * second_column[row_index]


def _add_row(matrix: list[list[int]], target: int, source: int, coefficient: int) -> None:
    matrix[target] = [left + coefficient * right for left, right in zip(matrix[target], matrix[source], strict=True)]


def _add_column(
    matrix: list[list[int]], target: int, source: int, coefficient: int
) -> None:
    for row in matrix:
        row[target] += coefficient * row[source]


@dataclass(frozen=True, slots=True)
class SmithForm:
    diagonal: MatrixZ
    left: MatrixZ
    right: MatrixZ
    invariant_factors: tuple[int, ...]

    def __post_init__(self) -> None:
        diagonal = as_matrix(self.diagonal, "$SmithForm.diagonal")
        left = as_matrix(self.left, "$SmithForm.left")
        right = as_matrix(self.right, "$SmithForm.right")
        factors = tuple(_integer(value, f"$SmithForm.invariant_factors[{index}]") for index, value in enumerate(self.invariant_factors))
        if left.shape != (diagonal.row_count, diagonal.row_count):
            raise ValueError("$SmithForm.left: incompatible shape")
        if right.shape != (diagonal.column_count, diagonal.column_count):
            raise ValueError("$SmithForm.right: incompatible shape")
        if abs(determinant(left)) != 1 or abs(determinant(right)) != 1:
            raise ValueError("Smith witnesses must be unimodular")
        if any(diagonal[row][column] for row in range(diagonal.row_count) for column in range(diagonal.column_count) if row != column):
            raise ValueError("$SmithForm.diagonal: expected diagonal matrix")
        diagonal_entries = tuple(
            diagonal[index][index] for index in range(min(diagonal.shape))
        )
        rank = len(factors)
        if any(value == 0 for value in diagonal_entries[:rank]) or any(
            value != 0 for value in diagonal_entries[rank:]
        ):
            raise ValueError(
                "Smith diagonal must have one contiguous positive prefix followed by zeros"
            )
        observed = diagonal_entries[:rank]
        if any(value <= 0 for value in observed):
            raise ValueError("Smith diagonal entries must be positive")
        if any(next_value % value for value, next_value in zip(observed, observed[1:])):
            raise ValueError("Smith invariant factors must divide their successors")
        if factors != observed:
            raise ValueError("Smith invariant-factor list differs from diagonal")
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "invariant_factors", factors)

    @property
    def rank(self) -> int:
        return len(self.invariant_factors)


def smith_form(matrix: MatrixInput) -> SmithForm:
    """Return deterministic Smith form and both unimodular witnesses."""

    source = as_matrix(matrix)
    row_count, column_count = source.shape
    work = [list(row) for row in source]
    left = [list(row) for row in identity_matrix(row_count)]
    right = [list(row) for row in identity_matrix(column_count)]
    diagonal_index = 0
    limit = min(row_count, column_count)
    while diagonal_index < limit:
        candidates = [
            (abs(work[row][column]), row, column)
            for row in range(diagonal_index, row_count)
            for column in range(diagonal_index, column_count)
            if work[row][column]
        ]
        if not candidates:
            break
        _, selected_row, selected_column = min(candidates)
        if selected_row != diagonal_index:
            _swap_rows(work, diagonal_index, selected_row)
            _swap_rows(left, diagonal_index, selected_row)
        if selected_column != diagonal_index:
            _swap_columns(work, diagonal_index, selected_column)
            _swap_columns(right, diagonal_index, selected_column)

        reduction_steps = 0
        while True:
            reduction_steps += 1
            if reduction_steps > 10_000:
                raise ArithmeticError(
                    f"Smith reduction did not converge at pivot {diagonal_index}: "
                    f"source={source.rows!r}, work={tuple(tuple(row) for row in work)!r}"
                )
            for row in range(diagonal_index + 1, row_count):
                if work[row][diagonal_index]:
                    pivot = work[diagonal_index][diagonal_index]
                    entry = work[row][diagonal_index]
                    if entry % pivot == 0:
                        quotient = entry // pivot
                        _add_row(work, row, diagonal_index, -quotient)
                        _add_row(left, row, diagonal_index, -quotient)
                    else:
                        transform = _combine_rows(
                            work,
                            diagonal_index,
                            row,
                            pivot,
                            entry,
                        )
                        _apply_row_pair(left, diagonal_index, row, transform)
            for column in range(diagonal_index + 1, column_count):
                if work[diagonal_index][column]:
                    pivot = work[diagonal_index][diagonal_index]
                    entry = work[diagonal_index][column]
                    if entry % pivot == 0:
                        quotient = entry // pivot
                        _add_column(work, column, diagonal_index, -quotient)
                        _add_column(right, column, diagonal_index, -quotient)
                    else:
                        transform = _combine_columns(
                            work,
                            diagonal_index,
                            column,
                            pivot,
                            entry,
                        )
                        _apply_column_pair(right, diagonal_index, column, transform)

            if any(work[row][diagonal_index] for row in range(diagonal_index + 1, row_count)) or any(work[diagonal_index][column] for column in range(diagonal_index + 1, column_count)):
                continue
            pivot = work[diagonal_index][diagonal_index]
            offender = next(
                (
                    (row, column)
                    for row in range(diagonal_index + 1, row_count)
                    for column in range(diagonal_index + 1, column_count)
                    if work[row][column] % pivot
                ),
                None,
            )
            if offender is None:
                break
            offender_row, _ = offender
            _add_row(work, diagonal_index, offender_row, 1)
            _add_row(left, diagonal_index, offender_row, 1)

        if work[diagonal_index][diagonal_index] < 0:
            work[diagonal_index] = [-entry for entry in work[diagonal_index]]
            left[diagonal_index] = [-entry for entry in left[diagonal_index]]
        diagonal_index += 1

    diagonal = MatrixZ(tuple(tuple(row) for row in work), column_count=column_count)
    left_matrix = MatrixZ(tuple(tuple(row) for row in left), column_count=row_count)
    right_matrix = MatrixZ(tuple(tuple(row) for row in right), column_count=column_count)
    factors = tuple(
        diagonal[index][index]
        for index in range(limit)
        if diagonal[index][index]
    )
    result = SmithForm(diagonal, left_matrix, right_matrix, factors)
    if matmul(matmul(result.left, source), result.right) != result.diagonal:
        raise ArithmeticError("Smith witness identity failed")
    return result


@dataclass(frozen=True, slots=True)
class HermiteForm:
    source: MatrixZ
    basis: MatrixZ
    forward_witness: MatrixZ
    backward_witness: MatrixZ
    rank: int

    def __post_init__(self) -> None:
        source = as_matrix(self.source, "$HermiteForm.source")
        basis = as_matrix(self.basis, "$HermiteForm.basis")
        forward = as_matrix(self.forward_witness, "$HermiteForm.forward_witness")
        backward = as_matrix(self.backward_witness, "$HermiteForm.backward_witness")
        rank = _integer(self.rank, "$HermiteForm.rank")
        if not 0 <= rank <= min(source.shape):
            raise ValueError("$HermiteForm.rank: incompatible rank")
        if basis.shape != (source.row_count, rank):
            raise ValueError("$HermiteForm.basis: incompatible shape")
        if forward.shape != (source.column_count, rank):
            raise ValueError("$HermiteForm.forward_witness: incompatible shape")
        if backward.shape != (rank, source.column_count):
            raise ValueError("$HermiteForm.backward_witness: incompatible shape")
        if matmul(source, forward) != basis or matmul(basis, backward) != source:
            raise ValueError("Hermite span witnesses do not replay")
        source_rank = smith_form(source).rank
        basis_rank = smith_form(basis).rank
        if rank != source_rank or rank != basis_rank:
            raise ValueError("Hermite rank differs from the exact source or basis rank")
        pivot_rows: list[int] = []
        for column in range(rank):
            nonzero_rows = [row for row in range(basis.row_count) if basis[row][column]]
            if not nonzero_rows:
                raise ValueError("Hermite basis column has no pivot")
            pivot = max(nonzero_rows)
            pivot_rows.append(pivot)
            pivot_value = basis[pivot][column]
            if pivot_value <= 0:
                raise ValueError("Hermite pivots must be positive")
            if any(basis[row][column] for row in range(pivot + 1, basis.row_count)):
                raise ValueError("Hermite basis has a nonzero entry below a pivot")
            for later_column in range(column + 1, rank):
                residue = basis[pivot][later_column]
                if not 0 <= residue < pivot_value:
                    raise ValueError("Hermite basis has a noncanonical pivot-row residue")
        if pivot_rows != sorted(pivot_rows) or len(set(pivot_rows)) != rank:
            raise ValueError("Hermite pivot rows must be strictly increasing")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "forward_witness", forward)
        object.__setattr__(self, "backward_witness", backward)


def column_hermite_form(matrix: MatrixInput) -> HermiteForm:
    """Return the canonical column-Hermite basis and span witnesses."""

    source = as_matrix(matrix)
    row_source = transpose(source)
    work = [list(row) for row in row_source]
    witness = [list(row) for row in identity_matrix(row_source.row_count)]
    pivot_row = 0
    # Right-to-left row Hermite, followed by a reversal of the nonzero rows,
    # transposes to the usual upper-triangular column convention.
    for column in range(row_source.column_count - 1, -1, -1):
        if pivot_row == row_source.row_count:
            break
        selected = next((row for row in range(pivot_row, row_source.row_count) if work[row][column]), None)
        if selected is None:
            continue
        if selected != pivot_row:
            _swap_rows(work, pivot_row, selected)
            _swap_rows(witness, pivot_row, selected)
        for row in range(pivot_row + 1, row_source.row_count):
            if work[row][column]:
                pivot = work[pivot_row][column]
                entry = work[row][column]
                if entry % pivot == 0:
                    quotient = entry // pivot
                    _add_row(work, row, pivot_row, -quotient)
                    _add_row(witness, row, pivot_row, -quotient)
                else:
                    transform = _combine_rows(
                        work,
                        pivot_row,
                        row,
                        pivot,
                        entry,
                    )
                    _apply_row_pair(witness, pivot_row, row, transform)
        if work[pivot_row][column] < 0:
            work[pivot_row] = [-entry for entry in work[pivot_row]]
            witness[pivot_row] = [-entry for entry in witness[pivot_row]]
        pivot = work[pivot_row][column]
        for row in range(pivot_row):
            quotient = work[row][column] // pivot
            if quotient:
                _add_row(work, row, pivot_row, -quotient)
                _add_row(witness, row, pivot_row, -quotient)
        pivot_row += 1

    rank = pivot_row
    for first in range(rank // 2):
        second = rank - first - 1
        _swap_rows(work, first, second)
        _swap_rows(witness, first, second)
    row_basis = MatrixZ(tuple(tuple(work[row]) for row in range(rank)), column_count=source.row_count)
    basis = transpose(row_basis)
    witness_matrix = MatrixZ(tuple(tuple(row) for row in witness), column_count=source.column_count)
    forward = transpose(
        MatrixZ(tuple(witness_matrix[row] for row in range(rank)), column_count=source.column_count)
    )
    witness_inverse = inverse_unimodular(witness_matrix)
    backward = transpose(
        MatrixZ(
            tuple(tuple(witness_inverse[row][column] for column in range(rank)) for row in range(source.column_count)),
            column_count=rank,
        )
    )
    return HermiteForm(source, basis, forward, backward, rank)


@dataclass(frozen=True, slots=True)
class IntegerKernel:
    source: MatrixZ
    basis: MatrixZ
    completion: MatrixZ
    completion_inverse: MatrixZ
    coordinate_projection: MatrixZ
    rank: int

    def __post_init__(self) -> None:
        source = as_matrix(self.source, "$IntegerKernel.source")
        basis = as_matrix(self.basis, "$IntegerKernel.basis")
        completion = as_matrix(self.completion, "$IntegerKernel.completion")
        completion_inverse = as_matrix(self.completion_inverse, "$IntegerKernel.completion_inverse")
        projection = as_matrix(self.coordinate_projection, "$IntegerKernel.coordinate_projection")
        rank = _integer(self.rank, "$IntegerKernel.rank")
        if not 0 <= rank <= source.column_count:
            raise ValueError("$IntegerKernel.rank: incompatible rank")
        if rank != smith_form(source).rank:
            raise ValueError("integer-kernel rank is incomplete or overdeclared")
        nullity = source.column_count - rank
        if basis.shape != (source.column_count, nullity):
            raise ValueError("$IntegerKernel.basis: incompatible shape")
        if completion.shape != (source.column_count, source.column_count) or completion_inverse.shape != completion.shape:
            raise ValueError("integer-kernel completion has incompatible shape")
        if abs(determinant(completion)) != 1:
            raise ValueError("integer-kernel completion must be unimodular")
        if matmul(completion_inverse, completion) != identity_matrix(source.column_count):
            raise ValueError("integer-kernel inverse witness failed")
        expected_basis = MatrixZ(
            tuple(tuple(completion[row][column] for column in range(rank, source.column_count)) for row in range(source.column_count)),
            column_count=nullity,
        )
        if basis != expected_basis:
            raise ValueError("integer-kernel basis differs from completion tail")
        expected_projection = MatrixZ(tuple(completion_inverse[row] for row in range(rank, source.column_count)), column_count=source.column_count)
        if projection != expected_projection:
            raise ValueError("integer-kernel coordinate projection differs from inverse witness")
        if matmul(source, basis) != zero_matrix(source.row_count, nullity):
            raise ValueError("integer-kernel basis does not lie in kernel")
        if matmul(projection, basis) != identity_matrix(nullity):
            raise ValueError("integer-kernel coordinate projection is not a left inverse")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "completion", completion)
        object.__setattr__(self, "completion_inverse", completion_inverse)
        object.__setattr__(self, "coordinate_projection", projection)

    @property
    def nullity(self) -> int:
        return self.source.column_count - self.rank


def integer_kernel(matrix: MatrixInput) -> IntegerKernel:
    """Return a saturated integer-kernel basis with completion certificate."""

    source = as_matrix(matrix)
    smith = smith_form(source)
    rank = smith.rank
    completion = smith.right
    completion_inverse = inverse_unimodular(completion)
    nullity = source.column_count - rank
    basis = MatrixZ(
        tuple(tuple(completion[row][column] for column in range(rank, source.column_count)) for row in range(source.column_count)),
        column_count=nullity,
    )
    projection = MatrixZ(
        tuple(completion_inverse[row] for row in range(rank, source.column_count)),
        column_count=source.column_count,
    )
    return IntegerKernel(source, basis, completion, completion_inverse, projection, rank)
