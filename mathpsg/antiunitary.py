"""Exact primitives for graded (unitary/antiunitary) PSG calculations.

An antiunitary symmetry is recorded by a grade in ``Z2``.  If ``kappa`` is
the involution induced by complex conjugation on the gauge group, the
semidirect-product law is

``(u, a) * (v, b) = (u * kappa**a(v), a + b)``.

Keeping this action explicit is important even in a pseudoreal convention
where a redefinition of the time-reversal gauge matrix makes the resulting
involution equal to the identity on ``SU(2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from mathpsg.su2 import E2, ONE_SU2, QuaternionSU2


SU2Involution = Callable[[QuaternionSU2], QuaternionSU2]


def raw_complex_conjugation(value: QuaternionSU2) -> QuaternionSU2:
    """The raw antiunitary involution ``u -> u*``."""

    return value.complex_conjugate()


def pseudoreal_involution(value: QuaternionSU2) -> QuaternionSU2:
    """The involution after extracting ``J=i*sigma_2`` from time reversal.

    For the fundamental representation of ``SU(2)``,

    ``J u* J^-1 = u``.

    The implementation evaluates the left-hand side instead of returning the
    input directly, so tests continue to exercise entrywise conjugation.
    """

    return E2 * value.complex_conjugate() * E2.inverse()


@dataclass(frozen=True)
class GradedSU2Element:
    """An element of ``SU(2) semidirect Z2^anti``."""

    matrix: QuaternionSU2
    grade: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "grade", self.grade % 2)


GRADED_IDENTITY = GradedSU2Element(ONE_SU2, 0)


def graded_multiply(
    left: GradedSU2Element,
    right: GradedSU2Element,
    involution: SU2Involution = raw_complex_conjugation,
) -> GradedSU2Element:
    """Multiply two graded gauge transformations."""

    acted = involution(right.matrix) if left.grade else right.matrix
    return GradedSU2Element(
        left.matrix * acted,
        left.grade + right.grade,
    )


def graded_inverse(
    value: GradedSU2Element,
    involution: SU2Involution = raw_complex_conjugation,
) -> GradedSU2Element:
    """Return the inverse for the graded semidirect-product law."""

    inverse = value.matrix.inverse()
    if value.grade:
        inverse = involution(inverse)
    return GradedSU2Element(inverse, value.grade)


def u1_action_bit(
    antiunitary_grade: int,
    normalizer_parity: int,
    *,
    involution_inverts_torus: bool = True,
) -> int:
    """Return whether the induced action on ``U(1)`` is inversion.

    In the raw convention, complex conjugation inverts the canonical torus,
    so the effective coefficient character is ``a + q``.  After the standard
    pseudoreal redefinition the involution fixes the torus and the same
    invariant character is carried by the shifted normalizer parity.
    """

    antiunitary_contribution = (
        antiunitary_grade % 2 if involution_inverts_torus else 0
    )
    return (antiunitary_contribution + normalizer_parity) % 2


def act_on_u1_phase(phase: Fraction, action_bit: int) -> Fraction:
    """Act on a phase measured in units of ``pi``, modulo two."""

    sign = -1 if action_bit % 2 else 1
    return (sign * phase) % 2


def direct_product_h2_z2_dimension(
    spatial_h1_dimension: int,
    spatial_h2_dimension: int,
) -> int:
    """Return ``dim H^2(G x C2, Z2)`` from the field Kunneth formula."""

    if spatial_h1_dimension < 0 or spatial_h2_dimension < 0:
        raise ValueError("cohomology dimensions must be nonnegative")
    return spatial_h2_dimension + spatial_h1_dimension + 1
