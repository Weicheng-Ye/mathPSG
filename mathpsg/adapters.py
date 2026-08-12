"""Adapters from the original three-dimensional lattice implementation.

Classifier v0.1 uses :mod:`mathpsg.periodic`, whose cell actions work in an
arbitrary dimension.  The completed benchmarks predate that module and store
their crystallographic actions as exact rational affine maps.  This adapter
keeps those certified inputs authoritative while exposing them through the
new common interface.
"""

from __future__ import annotations

from typing import Mapping

from mathpsg.affine import AffineMap
from mathpsg.lattice import PeriodicLattice
from mathpsg.periodic import CellSymmetry, PeriodicAction
from mathpsg.presentation import GradedPresentation, Word


def cell_symmetry_from_affine(
    lattice: PeriodicLattice,
    symmetry: AffineMap,
) -> CellSymmetry:
    """Convert one exact legacy affine action to integer cell data."""

    linear, permutation, shifts = lattice.generator_data(symmetry)
    return CellSymmetry(linear, permutation, shifts)


def periodic_action_from_affine(
    *,
    lattice: PeriodicLattice,
    generators: Mapping[str, AffineMap],
    relators: Mapping[str, Word],
    grades: Mapping[str, int] | None = None,
    reference_sublattices: tuple[int, ...] = (0,),
) -> PeriodicAction:
    """Expose a legacy benchmark as a dimension-independent action."""

    names = tuple(generators)
    generator_grades = (
        {name: 0 for name in names}
        if grades is None
        else dict(grades)
    )
    presentation = GradedPresentation(
        names,
        dict(relators),
        generator_grades,
    )
    return PeriodicAction(
        presentation,
        {
            name: cell_symmetry_from_affine(lattice, symmetry)
            for name, symmetry in generators.items()
        },
        reference_sublattices,
    )
