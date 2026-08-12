"""Exact affine transformations in three dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Sequence

Scalar = Fraction
Vector = tuple[Scalar, Scalar, Scalar]
Matrix = tuple[Vector, Vector, Vector]


def rational(value: int | str | Fraction) -> Fraction:
    """Return an exact rational number."""

    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def vector(values: Iterable[int | str | Fraction]) -> Vector:
    result = tuple(rational(value) for value in values)
    if len(result) != 3:
        raise ValueError("a three-dimensional vector must have three entries")
    return result  # type: ignore[return-value]


def matrix(rows: Sequence[Sequence[int | str | Fraction]]) -> Matrix:
    result = tuple(vector(row) for row in rows)
    if len(result) != 3:
        raise ValueError("a three-dimensional matrix must have three rows")
    return result  # type: ignore[return-value]


ZERO_VECTOR: Vector = vector((0, 0, 0))
IDENTITY_MATRIX: Matrix = matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def add_vectors(left: Vector, right: Vector) -> Vector:
    return vector(left[index] + right[index] for index in range(3))


def subtract_vectors(left: Vector, right: Vector) -> Vector:
    return vector(left[index] - right[index] for index in range(3))


def multiply_matrix_vector(left: Matrix, right: Vector) -> Vector:
    return vector(
        sum(left[row][column] * right[column] for column in range(3))
        for row in range(3)
    )


def multiply_matrices(left: Matrix, right: Matrix) -> Matrix:
    return matrix(
        (
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def invert_matrix(value: Matrix) -> Matrix:
    """Invert a nonsingular 3-by-3 rational matrix."""

    augmented = [
        list(value[row]) + list(IDENTITY_MATRIX[row])
        for row in range(3)
    ]
    for column in range(3):
        pivot = next(
            (row for row in range(column, 3) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [entry / pivot_value for entry in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if coefficient == 0:
                continue
            augmented[row] = [
                augmented[row][index] - coefficient * augmented[column][index]
                for index in range(6)
            ]
    return matrix(row[3:] for row in augmented)


@dataclass(frozen=True)
class AffineMap:
    """The affine map ``x -> linear*x + shift``.

    Composition uses the mathematical order ``left @ right = left after right``.
    """

    linear: Matrix
    shift: Vector = ZERO_VECTOR

    @classmethod
    def identity(cls) -> "AffineMap":
        return cls(IDENTITY_MATRIX, ZERO_VECTOR)

    @classmethod
    def translation(
        cls, displacement: Iterable[int | str | Fraction]
    ) -> "AffineMap":
        return cls(IDENTITY_MATRIX, vector(displacement))

    def act(self, point: Vector) -> Vector:
        return add_vectors(multiply_matrix_vector(self.linear, point), self.shift)

    def compose(self, other: "AffineMap") -> "AffineMap":
        return AffineMap(
            multiply_matrices(self.linear, other.linear),
            add_vectors(multiply_matrix_vector(self.linear, other.shift), self.shift),
        )

    def __matmul__(self, other: "AffineMap") -> "AffineMap":
        return self.compose(other)

    @lru_cache(maxsize=None)
    def inverse(self) -> "AffineMap":
        inverse_linear = invert_matrix(self.linear)
        inverse_shift = vector(
            -entry
            for entry in multiply_matrix_vector(inverse_linear, self.shift)
        )
        return AffineMap(inverse_linear, inverse_shift)

    def power(self, exponent: int) -> "AffineMap":
        if exponent < 0:
            return self.inverse().power(-exponent)
        result = AffineMap.identity()
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result @ factor
            factor = factor @ factor
            remaining >>= 1
        return result
