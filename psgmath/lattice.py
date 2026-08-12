"""Exact actions on periodic lattices with a finite sublattice basis."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from typing import Iterable

from .affine import (
    AffineMap,
    Matrix,
    Vector,
    invert_matrix,
    multiply_matrices,
    multiply_matrix_vector,
    subtract_vectors,
    vector,
)

Cell = tuple[int, int, int]
Site = tuple[Cell, int]


def _as_integer(value: Fraction) -> int:
    if value.denominator != 1:
        raise ValueError(f"expected an integer, obtained {value}")
    return value.numerator


def _integer_vector(value: Vector) -> Cell:
    return tuple(_as_integer(entry) for entry in value)  # type: ignore[return-value]


def _integer_matrix(value: Matrix) -> tuple[Cell, Cell, Cell]:
    return tuple(_integer_vector(row) for row in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class PeriodicLattice:
    """A lattice ``B Z^3 + {q_s}`` with exact affine symmetry actions."""

    basis: Matrix
    sublattices: tuple[Vector, ...]

    @cached_property
    def basis_inverse(self) -> Matrix:
        return invert_matrix(self.basis)

    def cell_linear(self, symmetry: AffineMap) -> tuple[Cell, Cell, Cell]:
        transformed = multiply_matrices(
            self.basis_inverse,
            multiply_matrices(symmetry.linear, self.basis),
        )
        return _integer_matrix(transformed)

    def decompose_point(self, point: Vector) -> Site:
        for sublattice, representative in enumerate(self.sublattices):
            coordinates = multiply_matrix_vector(
                self.basis_inverse,
                subtract_vectors(point, representative),
            )
            try:
                cell = _integer_vector(coordinates)
            except ValueError:
                continue
            return cell, sublattice
        raise ValueError(f"point {point!r} does not lie on this periodic lattice")

    @lru_cache(maxsize=None)
    def generator_data(
        self, symmetry: AffineMap
    ) -> tuple[tuple[Cell, Cell, Cell], tuple[int, ...], tuple[Cell, ...]]:
        cell_linear = self.cell_linear(symmetry)
        permutation: list[int] = []
        shifts: list[Cell] = []
        for representative in self.sublattices:
            shift, target = self.decompose_point(symmetry.act(representative))
            permutation.append(target)
            shifts.append(shift)
        return cell_linear, tuple(permutation), tuple(shifts)

    def act_site(self, symmetry: AffineMap, site: Site) -> Site:
        cell, sublattice = site
        cell_linear, permutation, shifts = self.generator_data(symmetry)
        transformed_cell = tuple(
            sum(cell_linear[row][column] * cell[column] for column in range(3))
            + shifts[sublattice][row]
            for row in range(3)
        )
        return transformed_cell, permutation[sublattice]  # type: ignore[return-value]

    def point(self, site: Site) -> Vector:
        cell, sublattice = site
        displacement = multiply_matrix_vector(self.basis, vector(cell))
        return vector(
            displacement[index] + self.sublattices[sublattice][index]
            for index in range(3)
        )

    def validate_symmetry(self, symmetry: AffineMap) -> None:
        self.generator_data(symmetry)

    def orbit_of_sublattice(
        self, generators: Iterable[AffineMap], start: int = 0
    ) -> frozenset[int]:
        orbit = {start}
        frontier = [start]
        generator_list = tuple(generators)
        while frontier:
            current = frontier.pop()
            for generator in generator_list:
                _, permutation, _ = self.generator_data(generator)
                target = permutation[current]
                if target not in orbit:
                    orbit.add(target)
                    frontier.append(target)
        return frozenset(orbit)
