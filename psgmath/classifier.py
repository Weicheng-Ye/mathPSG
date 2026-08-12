"""Generic finite-fiber classifier implementing the stabilizer theorem.

The cohomology backend supplies marked ambient extension classes and their
restrictions.  The local backend supplies every inequivalent stabilizer lift
over each restricted class.  This module forms the orbitwise products and
then applies the global extension-unmarking and coefficient-normalizer
actions.  It deliberately contains no lattice-specific PSG formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Hashable, Mapping


RestrictionKey = Hashable


@dataclass(frozen=True)
class ExtensionDatum:
    """One marked ambient extension and its orbitwise restrictions."""

    class_id: str
    sector: str
    coordinates: tuple[Hashable, ...]
    restrictions: tuple[RestrictionKey, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalLiftDatum:
    """One framed local lift over a restricted extension class."""

    lift_id: str
    restricted_class: RestrictionKey
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrbitDatum:
    """The lift table for one occupied Wyckoff orbit."""

    orbit_id: str
    stabilizer: str
    local_lifts: tuple[LocalLiftDatum, ...]


@dataclass(frozen=True, order=True)
class MarkedPSGClass:
    """A marked extension with one compatible lift on every orbit."""

    extension_id: str
    lift_ids: tuple[str, ...]

    @property
    def class_id(self) -> str:
        lift_part = "::".join(self.lift_ids)
        return f"{self.extension_id}::{lift_part}"


ClassActionMap = Callable[[MarkedPSGClass], MarkedPSGClass]


@dataclass(frozen=True)
class ClassAction:
    """A generator of the global unmarking or Weyl action."""

    name: str
    apply: ClassActionMap


@dataclass(frozen=True)
class PSGOrbit:
    """One quotient class and all marked representatives in its orbit."""

    representative: MarkedPSGClass
    members: tuple[MarkedPSGClass, ...]


@dataclass(frozen=True)
class ClassificationProblem:
    """Finite data required by the extension/fiber master formula."""

    problem_id: str
    igg: str
    extensions: tuple[ExtensionDatum, ...]
    orbits: tuple[OrbitDatum, ...]
    actions: tuple[ClassAction, ...] = ()
    framed: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        extension_ids = [extension.class_id for extension in self.extensions]
        if len(extension_ids) != len(set(extension_ids)):
            raise ValueError("ambient extension identifiers must be unique")
        orbit_ids = [orbit.orbit_id for orbit in self.orbits]
        if len(orbit_ids) != len(set(orbit_ids)):
            raise ValueError("occupied orbit identifiers must be unique")
        for extension in self.extensions:
            if len(extension.restrictions) != len(self.orbits):
                raise ValueError(
                    f"extension {extension.class_id!r} has "
                    "the wrong number of restrictions"
                )
        for orbit in self.orbits:
            lift_ids = [lift.lift_id for lift in orbit.local_lifts]
            if len(lift_ids) != len(set(lift_ids)):
                raise ValueError(
                    f"local lift identifiers for {orbit.orbit_id!r} "
                    "must be unique"
                )


def enumerate_marked_classes(
    problem: ClassificationProblem,
) -> tuple[MarkedPSGClass, ...]:
    """Form every compatible product of local stabilizer lifts."""

    results: list[MarkedPSGClass] = []
    for extension in problem.extensions:
        compatible = []
        for index, orbit in enumerate(problem.orbits):
            restriction = extension.restrictions[index]
            lifts = tuple(
                lift
                for lift in orbit.local_lifts
                if lift.restricted_class == restriction
            )
            if not lifts:
                break
            compatible.append(lifts)
        else:
            for choices in product(*compatible):
                results.append(
                    MarkedPSGClass(
                        extension.class_id,
                        tuple(choice.lift_id for choice in choices),
                    )
                )
    return tuple(sorted(results))


def quotient_classes(
    classes: tuple[MarkedPSGClass, ...],
    actions: tuple[ClassAction, ...],
) -> tuple[PSGOrbit, ...]:
    """Return the orbit set under global unmarking/Weyl generators."""

    universe = set(classes)
    remaining = set(classes)
    result: list[PSGOrbit] = []
    while remaining:
        seed = min(remaining)
        orbit = {seed}
        frontier = [seed]
        while frontier:
            value = frontier.pop()
            for action in actions:
                image = action.apply(value)
                if image not in universe:
                    raise ValueError(
                        f"class action {action.name!r} left the "
                        "compatible marked-class set"
                    )
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        members = tuple(sorted(orbit))
        result.append(PSGOrbit(members[0], members))
        remaining -= orbit
    return tuple(sorted(result, key=lambda value: value.representative))


def classify(
    problem: ClassificationProblem,
) -> tuple[PSGOrbit, ...]:
    """Apply the master formula to a finite certified problem."""

    marked = enumerate_marked_classes(problem)
    return quotient_classes(marked, problem.actions)


def classification_record(
    problem: ClassificationProblem,
) -> dict[str, object]:
    """Return a machine-readable summary."""

    marked = enumerate_marked_classes(problem)
    quotient = quotient_classes(marked, problem.actions)
    return {
        "problem_id": problem.problem_id,
        "igg": problem.igg,
        "framed": problem.framed,
        "ambient_extensions": len(problem.extensions),
        "occupied_orbits": [
            {
                "orbit_id": orbit.orbit_id,
                "stabilizer": orbit.stabilizer,
                "local_lifts": len(orbit.local_lifts),
            }
            for orbit in problem.orbits
        ],
        "marked_realizations": len(marked),
        "quotient_classes": len(quotient),
        "quotient_actions": [
            action.name for action in problem.actions
        ],
        "metadata": dict(problem.metadata),
    }
