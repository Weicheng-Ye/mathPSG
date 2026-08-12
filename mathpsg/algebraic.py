"""Exact arithmetic for crystallographic rotations.

The target field is the biquadratic field
``Q(sqrt(2), sqrt(3))`` in its fixed basis ``(1, sqrt2, sqrt3,
sqrt6)``.  Public values never use floating point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


def _fraction(value: object, context: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{context} must be an exact Fraction")
    return value


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


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
                    for left, right in zip(augmented[row], augmented[column], strict=True)
                ]
    return tuple(augmented[row][-1] for row in range(size))


@dataclass(frozen=True, order=True)
class Q23:
    """One exact element of ``Q(sqrt(2), sqrt(3))``."""

    one: Fraction = Fraction(0)
    sqrt2: Fraction = Fraction(0)
    sqrt3: Fraction = Fraction(0)
    sqrt6: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        _fraction(self.one, "one coefficient")
        _fraction(self.sqrt2, "sqrt2 coefficient")
        _fraction(self.sqrt3, "sqrt3 coefficient")
        _fraction(self.sqrt6, "sqrt6 coefficient")

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

    def to_json(self) -> dict[str, str]:
        return {
            "one": _fraction_text(self.one),
            "sqrt2": _fraction_text(self.sqrt2),
            "sqrt3": _fraction_text(self.sqrt3),
            "sqrt6": _fraction_text(self.sqrt6),
        }

    def to_fraction(self) -> Fraction:
        if self.sqrt2 or self.sqrt3 or self.sqrt6:
            raise ValueError("field element is not rational")
        return self.one

    def __add__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        return Q23(*(left + right for left, right in zip(self.coefficients, other.coefficients, strict=True)))

    def __sub__(self, other: object) -> Q23:
        if not isinstance(other, Q23):
            return NotImplemented
        return Q23(*(left - right for left, right in zip(self.coefficients, other.coefficients, strict=True)))

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

    def conjugate_sqrt2(self) -> Q23:
        return Q23(self.one, -self.sqrt2, self.sqrt3, -self.sqrt6)

    def conjugate_sqrt3(self) -> Q23:
        return Q23(self.one, self.sqrt2, -self.sqrt3, -self.sqrt6)

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
        result = Q23(*solution)
        if self * result != ONE_Q23:
            raise ArithmeticError("invalid exact field inverse witness")
        return result

    def __bool__(self) -> bool:
        return any(self.coefficients)

    def __str__(self) -> str:
        return str(self.to_json())


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

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or len(self.rows) != 3:
            raise ValueError("ExactSO3 must have exactly three rows")
        for row in self.rows:
            if type(row) is not tuple or len(row) != 3:
                raise ValueError("ExactSO3 rows must have exactly three entries")
            if any(type(entry) is not Q23 for entry in row):
                raise TypeError("ExactSO3 entries must be exact Q23 values")

    @classmethod
    def diagonal(cls, entries: tuple[int | Fraction, int | Fraction, int | Fraction]) -> ExactSO3:
        if type(entries) is not tuple or len(entries) != 3:
            raise TypeError("diagonal entries must be an exact length-three tuple")
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

    def determinant(self) -> Q23:
        a = self.rows
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    def is_rotation(self) -> bool:
        return self @ self.transpose() == identity_so3() and self.determinant() == ONE_Q23

    def order(self, maximum: int = 120) -> int:
        if type(maximum) is not int or maximum < 1:
            raise TypeError("maximum must be a positive exact int")
        product = identity_so3()
        for exponent in range(1, maximum + 1):
            product = product @ self
            if product == identity_so3():
                return exponent
        raise ValueError("matrix order exceeds the certified finite bound")

    @property
    def canonical_key(self) -> tuple[Fraction, ...]:
        return tuple(
            coefficient
            for row in self.rows
            for entry in row
            for coefficient in entry.coefficients
        )

    def to_json(self) -> list[list[dict[str, str]]]:
        return [[entry.to_json() for entry in row] for row in self.rows]


def identity_so3() -> ExactSO3:
    return ExactSO3.diagonal((1, 1, 1))


@dataclass(frozen=True)
class ExactQuaternion:
    scalar: Q23
    x: Q23 = ZERO_Q23
    y: Q23 = ZERO_Q23
    z: Q23 = ZERO_Q23

    def __post_init__(self) -> None:
        if any(type(entry) is not Q23 for entry in (self.scalar, self.x, self.y, self.z)):
            raise TypeError("ExactQuaternion entries must be exact Q23 values")

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

    def norm_squared(self) -> Q23:
        return sum((entry * entry for entry in self.coefficients), ZERO_Q23)

    def has_canonical_sign(self) -> bool:
        for entry in self.coefficients:
            for coefficient in entry.coefficients:
                if coefficient:
                    return coefficient > 0
        return True

    def canonicalized(self) -> ExactQuaternion:
        return self if self.has_canonical_sign() else -self

    def to_so3(self) -> ExactSO3:
        if self.norm_squared() != ONE_Q23:
            raise ValueError("quaternion must have exact unit norm")
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

    def to_json(self) -> dict[str, Any]:
        return {
            "scalar": self.scalar.to_json(),
            "x": self.x.to_json(),
            "y": self.y.to_json(),
            "z": self.z.to_json(),
        }


ONE_QUATERNION = ExactQuaternion(ONE_Q23)
