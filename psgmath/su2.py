"""Small exact models for the SU(2) elements used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Qsqrt3:
    """An element ``rational + radical*sqrt(3)``."""

    rational: Fraction = Fraction(0)
    radical: Fraction = Fraction(0)

    @classmethod
    def from_value(cls, value: int | Fraction) -> "Qsqrt3":
        return cls(Fraction(value), Fraction(0))

    def __add__(self, other: object) -> "Qsqrt3":
        if not isinstance(other, Qsqrt3):
            return NotImplemented
        return Qsqrt3(
            self.rational + other.rational,
            self.radical + other.radical,
        )

    def __sub__(self, other: object) -> "Qsqrt3":
        if not isinstance(other, Qsqrt3):
            return NotImplemented
        return Qsqrt3(
            self.rational - other.rational,
            self.radical - other.radical,
        )

    def __neg__(self) -> "Qsqrt3":
        return Qsqrt3(-self.rational, -self.radical)

    def __mul__(self, other: object) -> "Qsqrt3":
        if not isinstance(other, Qsqrt3):
            return NotImplemented
        return Qsqrt3(
            self.rational * other.rational
            + 3 * self.radical * other.radical,
            self.rational * other.radical
            + self.radical * other.rational,
        )


ZERO = Qsqrt3()
ONE_Q = Qsqrt3.from_value(1)
HALF = Qsqrt3.from_value(Fraction(1, 2))
ROOT3_HALF = Qsqrt3(Fraction(0), Fraction(1, 2))


@dataclass(frozen=True)
class QuaternionSU2:
    """Exact ``a0*1 + sum_k ak*(i sigma_k)`` arithmetic."""

    scalar: Qsqrt3
    vector: tuple[Qsqrt3, Qsqrt3, Qsqrt3] = (ZERO, ZERO, ZERO)

    def __neg__(self) -> "QuaternionSU2":
        return QuaternionSU2(-self.scalar, tuple(-x for x in self.vector))

    def __mul__(self, other: object) -> "QuaternionSU2":
        if not isinstance(other, QuaternionSU2):
            return NotImplemented
        a = self.vector
        b = other.vector
        scalar = self.scalar * other.scalar
        for index in range(3):
            scalar = scalar - a[index] * b[index]
        cross = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        vector = tuple(
            self.scalar * b[index]
            + other.scalar * a[index]
            - cross[index]
            for index in range(3)
        )
        return QuaternionSU2(scalar, vector)  # type: ignore[arg-type]

    def inverse(self) -> "QuaternionSU2":
        """Return the inverse of a unit quaternion."""

        return QuaternionSU2(self.scalar, tuple(-x for x in self.vector))

    def complex_conjugate(self) -> "QuaternionSU2":
        """Return the entrywise complex conjugate of the SU(2) matrix.

        In the quaternion basis used here,

        ``(i*sigma_1)^* = -i*sigma_1``,
        ``(i*sigma_2)^* = +i*sigma_2``, and
        ``(i*sigma_3)^* = -i*sigma_3``.

        This is deliberately distinct from :meth:`inverse`, which is
        quaternion/Hermitian conjugation for a unit quaternion.
        """

        a1, a2, a3 = self.vector
        return QuaternionSU2(self.scalar, (-a1, a2, -a3))

    @property
    def is_central(self) -> bool:
        return self.vector == (ZERO, ZERO, ZERO)

    @property
    def central_sign(self) -> int:
        if not self.is_central:
            raise ValueError("the SU(2) element is not central")
        if self.scalar == ONE_Q:
            return 1
        if self.scalar == -ONE_Q:
            return -1
        raise ValueError("the central element is not in {+1,-1}")


ONE_SU2 = QuaternionSU2(ONE_Q)
E1 = QuaternionSU2(ZERO, (ONE_Q, ZERO, ZERO))
E2 = QuaternionSU2(ZERO, (ZERO, ONE_Q, ZERO))
E3 = QuaternionSU2(ZERO, (ZERO, ZERO, ONE_Q))


_COSINES = (
    ONE_Q,
    ROOT3_HALF,
    HALF,
    ZERO,
    -HALF,
    -ROOT3_HALF,
    -ONE_Q,
    -ROOT3_HALF,
    -HALF,
    ZERO,
    HALF,
    ROOT3_HALF,
)
_SINES = (
    ZERO,
    HALF,
    ROOT3_HALF,
    ONE_Q,
    ROOT3_HALF,
    HALF,
    ZERO,
    -HALF,
    -ROOT3_HALF,
    -ONE_Q,
    -ROOT3_HALF,
    -HALF,
)


def exp_pauli(axis: int, angle_over_pi: Fraction | int) -> QuaternionSU2:
    """Return ``exp(i*pi*angle_over_pi*sigma_axis)`` exactly.

    The benchmark only needs angles in integer multiples of ``pi/6``.
    """

    sixths = Fraction(angle_over_pi) * 6
    if sixths.denominator != 1:
        raise ValueError("angle must be an integer multiple of pi/6")
    index = sixths.numerator % 12
    vector = [ZERO, ZERO, ZERO]
    if axis not in (1, 2, 3):
        raise ValueError("Pauli axis must be 1, 2, or 3")
    vector[axis - 1] = _SINES[index]
    return QuaternionSU2(_COSINES[index], tuple(vector))  # type: ignore[arg-type]


@dataclass(frozen=True)
class PinElement:
    """An exact element ``j**parity * exp(i*pi*phase*sigma_3)`` of Pin(2).

    Here ``j=i*sigma_1``, so ``j**2=-1``.  The phase is stored modulo two.
    """

    parity: int
    phase: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parity", self.parity % 2)
        object.__setattr__(self, "phase", self.phase % 2)

    def __mul__(self, other: object) -> "PinElement":
        if not isinstance(other, PinElement):
            return NotImplemented
        square = int(self.parity == 1 and other.parity == 1)
        phase = (
            ((-1) ** other.parity) * self.phase
            + other.phase
            + square
        )
        return PinElement(self.parity + other.parity, phase)

    def inverse(self) -> "PinElement":
        if self.parity == 0:
            return PinElement(0, -self.phase)
        return PinElement(1, self.phase + 1)


ONE_PIN = PinElement(0, Fraction(0))
