r"""Exact linear algebra over :math:`\mathbb Z`.

Matrices act on column vectors.  ``MatrixZ`` retains its column count even
when it has no rows, so the public boundary distinguishes ``0 x n`` from
``0 x 0``.  Normal-form transformations are retained when later calculations
need them; results are not replayed as independent certificates.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
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
                    for entry, pivot_entry in zip(augmented[row], augmented[column])
                ]
    inverse_rows: list[VectorZ] = []
    for row in augmented:
        entries = row[size:]
        if any(entry.denominator != 1 for entry in entries):
            raise ValueError("matrix is not unimodular")
        inverse_rows.append(tuple(entry.numerator for entry in entries))
    return MatrixZ(tuple(inverse_rows), column_count=size)


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
    matrix[first] = [transform[0][0] * left + transform[0][1] * right for left, right in zip(first_row, second_row)]
    matrix[second] = [transform[1][0] * left + transform[1][1] * right for left, right in zip(first_row, second_row)]
    return transform


def _apply_row_pair(
    witness: list[list[int]],
    first: int,
    second: int,
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    first_row = witness[first][:]
    second_row = witness[second][:]
    witness[first] = [transform[0][0] * left + transform[0][1] * right for left, right in zip(first_row, second_row)]
    witness[second] = [transform[1][0] * left + transform[1][1] * right for left, right in zip(first_row, second_row)]


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
    matrix[target] = [left + coefficient * right for left, right in zip(matrix[target], matrix[source])]


def _add_column(
    matrix: list[list[int]], target: int, source: int, coefficient: int
) -> None:
    for row in matrix:
        row[target] += coefficient * row[source]


@dataclass(frozen=True, slots=True)
class SmithForm:
    left: MatrixZ
    right: MatrixZ
    invariant_factors: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.invariant_factors)


def smith_form(matrix: MatrixInput) -> SmithForm:
    """Return a deterministic Smith form and the transformations it needs."""

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

    left_matrix = MatrixZ(tuple(tuple(row) for row in left), column_count=row_count)
    right_matrix = MatrixZ(tuple(tuple(row) for row in right), column_count=column_count)
    factors = tuple(
        work[index][index]
        for index in range(limit)
        if work[index][index]
    )
    return SmithForm(left_matrix, right_matrix, factors)


@dataclass(frozen=True, slots=True)
class IntegerKernel:
    basis: MatrixZ
    coordinate_projection: MatrixZ

    @property
    def nullity(self) -> int:
        return self.basis.column_count


def integer_kernel(matrix: MatrixInput) -> IntegerKernel:
    """Return a saturated integer-kernel basis and its coordinate map."""

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
    return IntegerKernel(basis, projection)
