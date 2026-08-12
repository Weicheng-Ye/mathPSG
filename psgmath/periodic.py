"""Dimension-independent exact actions on periodic cell/sublattice labels."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import Iterable, Mapping, Sequence

from .presentation import GradedPresentation, evaluate_word


Cell = tuple[int, ...]
Site = tuple[Cell, int]
IntegerMatrix = tuple[tuple[int, ...], ...]


def integer_matrix(rows: Iterable[Iterable[int]]) -> IntegerMatrix:
    result = tuple(tuple(int(entry) for entry in row) for row in rows)
    if not result:
        raise ValueError("an integer matrix must be nonempty")
    dimension = len(result)
    if any(len(row) != dimension for row in result):
        raise ValueError("an integer matrix must be square")
    return result


def identity_matrix(dimension: int) -> IntegerMatrix:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return integer_matrix(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )


def multiply_matrix_vector(
    matrix: IntegerMatrix,
    vector: Cell,
) -> Cell:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(matrix))
    )


def multiply_matrices(
    left: IntegerMatrix,
    right: IntegerMatrix,
) -> IntegerMatrix:
    dimension = len(left)
    if len(right) != dimension:
        raise ValueError("matrix dimensions do not agree")
    return integer_matrix(
        (
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(dimension)
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _invert_unimodular(value: IntegerMatrix) -> IntegerMatrix:
    dimension = len(value)
    augmented = [
        [Fraction(entry) for entry in value[row]]
        + [Fraction(int(row == column)) for column in range(dimension)]
        for row in range(dimension)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column]
            ),
            None,
        )
        if pivot is None:
            raise ValueError("cell matrix is singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        pivot_value = augmented[column][column]
        augmented[column] = [
            entry / pivot_value for entry in augmented[column]
        ]
        for row in range(dimension):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if not coefficient:
                continue
            augmented[row] = [
                augmented[row][index]
                - coefficient * augmented[column][index]
                for index in range(2 * dimension)
            ]
    inverse = tuple(
        tuple(augmented[row][dimension:])
        for row in range(dimension)
    )
    if any(entry.denominator != 1 for row in inverse for entry in row):
        raise ValueError("cell matrix is not unimodular")
    return integer_matrix(
        tuple(entry.numerator for entry in row)
        for row in inverse
    )


@dataclass(frozen=True)
class CellSymmetry:
    """An action ``(n,s) -> (M n + d_s, permutation(s))``."""

    linear: IntegerMatrix
    permutation: tuple[int, ...]
    shifts: tuple[Cell, ...]

    def __post_init__(self) -> None:
        dimension = len(self.linear)
        if self.linear != integer_matrix(self.linear):
            raise ValueError("linear cell action must be an integer matrix")
        sublattices = len(self.permutation)
        if sorted(self.permutation) != list(range(sublattices)):
            raise ValueError("sublattice action must be a permutation")
        if len(self.shifts) != sublattices:
            raise ValueError("one cell shift is required per sublattice")
        if any(len(shift) != dimension for shift in self.shifts):
            raise ValueError("cell shifts have the wrong dimension")
        _invert_unimodular(self.linear)

    @classmethod
    def identity(
        cls,
        dimension: int,
        sublattices: int,
    ) -> "CellSymmetry":
        return cls(
            identity_matrix(dimension),
            tuple(range(sublattices)),
            tuple((0,) * dimension for _ in range(sublattices)),
        )

    @property
    def dimension(self) -> int:
        return len(self.linear)

    @property
    def sublattice_count(self) -> int:
        return len(self.permutation)

    def act(self, site: Site) -> Site:
        cell, sublattice = site
        if len(cell) != self.dimension:
            raise ValueError("site cell has the wrong dimension")
        transformed = multiply_matrix_vector(self.linear, cell)
        shift = self.shifts[sublattice]
        return (
            tuple(
                transformed[index] + shift[index]
                for index in range(self.dimension)
            ),
            self.permutation[sublattice],
        )

    def compose(self, other: "CellSymmetry") -> "CellSymmetry":
        """Return ``self`` after ``other``."""

        if (
            self.dimension != other.dimension
            or self.sublattice_count != other.sublattice_count
        ):
            raise ValueError("cell actions have incompatible dimensions")
        linear = multiply_matrices(self.linear, other.linear)
        permutation = tuple(
            self.permutation[other.permutation[source]]
            for source in range(self.sublattice_count)
        )
        shifts = []
        for source in range(self.sublattice_count):
            transformed = multiply_matrix_vector(
                self.linear,
                other.shifts[source],
            )
            outer_shift = self.shifts[other.permutation[source]]
            shifts.append(
                tuple(
                    transformed[index] + outer_shift[index]
                    for index in range(self.dimension)
                )
            )
        return CellSymmetry(linear, permutation, tuple(shifts))

    def __matmul__(self, other: "CellSymmetry") -> "CellSymmetry":
        return self.compose(other)

    def inverse(self) -> "CellSymmetry":
        inverse_linear = _invert_unimodular(self.linear)
        inverse_permutation = [0] * self.sublattice_count
        for source, target in enumerate(self.permutation):
            inverse_permutation[target] = source
        inverse_shifts: list[Cell] = []
        for target in range(self.sublattice_count):
            source = inverse_permutation[target]
            transformed = multiply_matrix_vector(
                inverse_linear,
                self.shifts[source],
            )
            inverse_shifts.append(tuple(-entry for entry in transformed))
        return CellSymmetry(
            inverse_linear,
            tuple(inverse_permutation),
            tuple(inverse_shifts),
        )

    def power(self, exponent: int) -> "CellSymmetry":
        if exponent < 0:
            return self.inverse().power(-exponent)
        result = CellSymmetry.identity(
            self.dimension,
            self.sublattice_count,
        )
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result @ factor
            factor = factor @ factor
            remaining >>= 1
        return result


@dataclass(frozen=True)
class PeriodicAction:
    """A presentation together with its exact periodic-site action."""

    presentation: GradedPresentation
    generators: Mapping[str, CellSymmetry]
    reference_sublattices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if set(self.generators) != set(self.presentation.generators):
            raise ValueError(
                "presentation and action must use the same generators"
            )
        actions = tuple(self.generators.values())
        first = actions[0]
        if any(
            action.dimension != first.dimension
            or action.sublattice_count != first.sublattice_count
            for action in actions
        ):
            raise ValueError("all generator actions must share one lattice")
        if any(
            sublattice not in range(first.sublattice_count)
            for sublattice in self.reference_sublattices
        ):
            raise ValueError("a reference sublattice is out of range")

    @cached_property
    def identity(self) -> CellSymmetry:
        first = next(iter(self.generators.values()))
        return CellSymmetry.identity(
            first.dimension,
            first.sublattice_count,
        )

    def evaluate(self, word: tuple[tuple[str, int], ...]) -> CellSymmetry:
        return evaluate_word(
            word,
            self.generators,
            self.identity,
            multiply=lambda left, right: left @ right,
            inverse=lambda value: value.inverse(),
        )

    def relator_values(self) -> dict[str, CellSymmetry]:
        return {
            label: self.evaluate(word)
            for label, word in self.presentation.relators.items()
        }

    def validate_relators(self) -> None:
        failures = [
            label
            for label, value in self.relator_values().items()
            if value != self.identity
        ]
        if failures:
            raise ValueError(
                "nonidentity lattice relators: "
                + ", ".join(failures)
            )

    def sublattice_orbit(
        self,
        start: int,
        *,
        unitary_only: bool = True,
    ) -> frozenset[int]:
        names = tuple(
            generator
            for generator in self.presentation.generators
            if not unitary_only
            or self.presentation.grades[generator] == 0
        )
        orbit = {start}
        frontier = [start]
        while frontier:
            source = frontier.pop()
            for generator in names:
                target = self.generators[generator].permutation[source]
                if target not in orbit:
                    orbit.add(target)
                    frontier.append(target)
        return frozenset(orbit)
