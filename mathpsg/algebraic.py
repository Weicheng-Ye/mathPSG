"""Exact arithmetic for crystallographic rotations.

The target field is the biquadratic field
``Q(sqrt(2), sqrt(3))`` in its fixed basis ``(1, sqrt2, sqrt3,
sqrt6)``.  Public values never use floating point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def _solve_rational(matrix: tuple[tuple[Fraction, ...], ...], rhs: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    size = len(matrix)
    if len(rhs) != size or any(len(row) != size for row in matrix):
        raise ValueError("rational solve requires a square matrix")
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ZeroDivisionError("field element is zero")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [entry / scale for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if coefficient:
                augmented[row] = [
                    left - coefficient * right
                    for left, right in zip(augmented[row], augmented[column])
                ]
    return tuple(augmented[row][-1] for row in range(size))


@dataclass(frozen=True, order=True)
class Q23:
    """One exact element of ``Q(sqrt(2), sqrt(3))``."""

    one: Fraction = Fraction(0)
    sqrt2: Fraction = Fraction(0)
    sqrt3: Fraction = Fraction(0)
    sqrt6: Fraction = Fraction(0)

    @classmethod
    def zero(cls) -> Q23:
        return cls()

    @classmethod
    def from_rational(cls, value: int | Fraction) -> Q23:
        if type(value) is int:
            return cls(Fraction(value))
        if type(value) is Fraction:
            return cls(value)
        raise TypeError("rational value must be an exact int or Fraction")

    @property
    def coefficients(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.one, self.sqrt2, self.sqrt3, self.sqrt6)

    def to_fraction(self) -> Fraction:
        if self.sqrt2 or self.sqrt3 or self.sqrt6:
            raise ValueError("field element is not rational")
        return self.one

    def __add__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        return Q23(*(left + right for left, right in zip(self.coefficients, other.coefficients)))

    def __sub__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        return Q23(*(left - right for left, right in zip(self.coefficients, other.coefficients)))

    def __neg__(self) -> Q23:
        return Q23(*(-entry for entry in self.coefficients))

    def __mul__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        a, b, c, d = self.coefficients
        e, f, g, h = other.coefficients
        return Q23(
            a * e + 2 * b * f + 3 * c * g + 6 * d * h,
            a * f + b * e + 3 * c * h + 3 * d * g,
            a * g + c * e + 2 * b * h + 2 * d * f,
            a * h + d * e + b * g + c * f,
        )

    def __truediv__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        return self * other.inverse()

    def inverse(self) -> Q23:
        if not self:
            raise ZeroDivisionError("zero has no inverse")
        basis = (ONE_Q23, SQRT2, SQRT3, SQRT6)
        columns = tuple((self * element).coefficients for element in basis)
        multiplication = tuple(
            tuple(columns[column][row] for column in range(4)) for row in range(4)
        )
        solution = _solve_rational(
            multiplication,
            (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        )
        return Q23(*solution)

    def __bool__(self) -> bool:
        return any(self.coefficients)

ZERO_Q23 = Q23.zero()
ONE_Q23 = Q23.from_rational(1)
SQRT2 = Q23(Fraction(0), Fraction(1))
SQRT3 = Q23(Fraction(0), Fraction(0), Fraction(1))
SQRT6 = Q23(Fraction(0), Fraction(0), Fraction(0), Fraction(1))


def _q(value: int | Fraction) -> Q23:
    return Q23.from_rational(value)


@dataclass(frozen=True)
class ExactSO3:
    rows: tuple[tuple[Q23, Q23, Q23], tuple[Q23, Q23, Q23], tuple[Q23, Q23, Q23]]

    @classmethod
    def diagonal(cls, entries: tuple[int | Fraction, int | Fraction, int | Fraction]) -> ExactSO3:
        return cls(
            tuple(
                tuple(_q(entries[row]) if row == column else ZERO_Q23 for column in range(3))
                for row in range(3)
            )  # type: ignore[arg-type]
        )

    def __matmul__(self, other: object) -> ExactSO3:
        if not isinstance(other, ExactSO3):
            return NotImplemented
        return ExactSO3(
            tuple(
                tuple(
                    sum(
                        (self.rows[row][inner] * other.rows[inner][column] for inner in range(3)),
                        ZERO_Q23,
                    )
                    for column in range(3)
                )
                for row in range(3)
            )  # type: ignore[arg-type]
        )

    def transpose(self) -> ExactSO3:
        return ExactSO3(
            tuple(tuple(self.rows[column][row] for column in range(3)) for row in range(3))  # type: ignore[arg-type]
        )

    @property
    def canonical_key(self) -> tuple[Fraction, ...]:
        return tuple(
            coefficient
            for row in self.rows
            for entry in row
            for coefficient in entry.coefficients
        )

def identity_so3() -> ExactSO3:
    return ExactSO3.diagonal((1, 1, 1))


@dataclass(frozen=True)
class ExactQuaternion:
    scalar: Q23
    x: Q23 = ZERO_Q23
    y: Q23 = ZERO_Q23
    z: Q23 = ZERO_Q23

    @property
    def coefficients(self) -> tuple[Q23, Q23, Q23, Q23]:
        return (self.scalar, self.x, self.y, self.z)

    def __neg__(self) -> ExactQuaternion:
        return ExactQuaternion(*(-entry for entry in self.coefficients))

    def __mul__(self, other: object) -> ExactQuaternion:
        if not isinstance(other, ExactQuaternion):
            return NotImplemented
        a, x, y, z = self.coefficients
        b, u, v, w = other.coefficients
        return ExactQuaternion(
            a * b - x * u - y * v - z * w,
            a * u + x * b + y * w - z * v,
            a * v - x * w + y * b + z * u,
            a * w + x * v - y * u + z * b,
        )

    def has_canonical_sign(self) -> bool:
        for entry in self.coefficients:
            for coefficient in entry.coefficients:
                if coefficient:
                    return coefficient > 0
        return True

    def canonicalized(self) -> ExactQuaternion:
        return self if self.has_canonical_sign() else -self

    def to_so3(self) -> ExactSO3:
        w, x, y, z = self.coefficients
        two = _q(2)
        return ExactSO3(
            (
                (
                    ONE_Q23 - two * (y * y + z * z),
                    two * (x * y - w * z),
                    two * (x * z + w * y),
                ),
                (
                    two * (x * y + w * z),
                    ONE_Q23 - two * (x * x + z * z),
                    two * (y * z - w * x),
                ),
                (
                    two * (x * z - w * y),
                    two * (y * z + w * x),
                    ONE_Q23 - two * (x * x + y * y),
                ),
            )
        )

ONE_QUATERNION = ExactQuaternion(ONE_Q23)
