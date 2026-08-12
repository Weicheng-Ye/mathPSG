"""Exact, immutable certificates shared by Task 13.

This module deliberately distinguishes diagnostic coordinate traces from
Task-5-authorized bar evaluators.  A diagnostic trace is useful for focused
algebra tests, but the flag is carried into every reconstruction certificate
and cannot be mistaken for release authority by downstream code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Literal, Sequence

from .algebraic import ExactQuaternion, ONE_QUATERNION
from .bar_evaluator import (
    BarResolutionEquivalence,
    evaluate_bar_cochain,
    verify_bar_resolution_equivalence,
)
from .cochains import FiniteGroupTable, Task5VerificationAuthority
from .gf2 import GF2Character
from .integer_linalg import MatrixZ
from .relative_complex import (
    RelativeMatrices,
    RelativeProblem,
    verify_relative_certificate,
)
from .torus import Phase
from .u1_local import U1LocalSkeleton, verify_u1_local_skeleton
from .z2_local import Z2LocalSkeleton, verify_z2_local_skeleton


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROTOCOL = b"mathpsg-task13-certificates-v1|"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: str, value: object) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL + domain.encode("ascii") + b"|" + _canonical_json(value)
    ).hexdigest()


def _require_digest(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256 digest")
    return value


def _phase_text(value: Phase) -> str:
    return str(value)


@dataclass(frozen=True, slots=True)
class FormalPhase:
    """An exact affine expression ``coefficients . parameters + constant``.

    The constant is an element of ``R/Z`` while the coefficients are exact
    integers.  Parameter values are never sampled during relation replay.
    """

    parameter_names: tuple[str, ...]
    coefficients: tuple[int, ...]
    constant: Phase

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        coefficients = tuple(self.coefficients)
        if len(names) != len(coefficients):
            raise ValueError("$FormalPhase: parameter and coefficient counts differ")
        if len(set(names)) != len(names) or any(
            type(name) is not str or not name for name in names
        ):
            raise ValueError("$FormalPhase.parameter_names: expected unique names")
        if any(type(value) is not int for value in coefficients):
            raise TypeError("$FormalPhase.coefficients: expected exact integers")
        if type(self.constant) is not Phase:
            raise TypeError("$FormalPhase.constant: expected Phase")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def zero(cls, parameter_names: Sequence[str] = ()) -> "FormalPhase":
        names = tuple(parameter_names)
        return cls(names, (0,) * len(names), Phase(Fraction(0)))

    def scale(self, coefficient: int) -> "FormalPhase":
        if type(coefficient) is not int:
            raise TypeError("formal-phase scale must be an exact integer")
        return FormalPhase(
            self.parameter_names,
            tuple(coefficient * value for value in self.coefficients),
            Phase(coefficient * self.constant.value),
        )

    def __add__(self, other: object) -> "FormalPhase":
        if not isinstance(other, FormalPhase):
            return NotImplemented
        if self.parameter_names != other.parameter_names:
            raise ValueError("formal phases use different parameter bases")
        return FormalPhase(
            self.parameter_names,
            tuple(
                left + right
                for left, right in zip(
                    self.coefficients, other.coefficients, strict=True
                )
            ),
            Phase(self.constant.value + other.constant.value),
        )

    def __neg__(self) -> "FormalPhase":
        return self.scale(-1)

    def __sub__(self, other: object) -> "FormalPhase":
        if not isinstance(other, FormalPhase):
            return NotImplemented
        return self + (-other)

    @property
    def is_zero(self) -> bool:
        return not any(self.coefficients) and self.constant.value == 0

    def mapping(self) -> dict[str, object]:
        return {
            "constant": _phase_text(self.constant),
            "coefficients": list(self.coefficients),
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True, slots=True)
class BarCoordinateTrace:
    """One finite, on-demand bar query written in resolution coordinates."""

    trace_id: str
    resolution_id: str
    degree: int
    group_tuple: tuple[str, ...]
    coordinate_weights: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_digest(self.trace_id, "$BarCoordinateTrace.trace_id")
        _require_digest(self.resolution_id, "$BarCoordinateTrace.resolution_id")
        if type(self.degree) is not int or self.degree not in (1, 2):
            raise ValueError("$BarCoordinateTrace.degree: expected one or two")
        group_tuple = tuple(self.group_tuple)
        weights = tuple(self.coordinate_weights)
        if len(group_tuple) != self.degree or any(
            type(value) is not str or not value for value in group_tuple
        ):
            raise ValueError("$BarCoordinateTrace.group_tuple: wrong degree")
        if any(type(value) is not int for value in weights):
            raise TypeError("$BarCoordinateTrace.coordinate_weights: expected integers")
        core = {
            "coordinate_weights": list(weights),
            "degree": self.degree,
            "group_tuple": list(group_tuple),
            "resolution_id": self.resolution_id,
        }
        if self.trace_id != _digest("bar-coordinate-trace", core):
            raise ValueError("$BarCoordinateTrace.trace_id: payload digest differs")
        object.__setattr__(self, "group_tuple", group_tuple)
        object.__setattr__(self, "coordinate_weights", weights)

    @classmethod
    def make(
        cls,
        resolution_id: str,
        degree: int,
        group_tuple: Sequence[str],
        coordinate_weights: Sequence[int],
    ) -> "BarCoordinateTrace":
        group = tuple(group_tuple)
        weights = tuple(coordinate_weights)
        core = {
            "coordinate_weights": list(weights),
            "degree": degree,
            "group_tuple": list(group),
            "resolution_id": resolution_id,
        }
        return cls(
            _digest("bar-coordinate-trace", core),
            resolution_id,
            degree,
            group,
            weights,
        )


@dataclass(frozen=True, slots=True)
class BarEvaluatorCertificate:
    """A finite query domain for an exact normalized-bar evaluator."""

    evaluator_id: str
    resolution_id: str
    finite_group: FiniteGroupTable
    coordinate_dimensions: tuple[int, ...]
    traces: tuple[BarCoordinateTrace, ...]
    equivalence: BarResolutionEquivalence | None
    authority: Task5VerificationAuthority | None
    diagnostic: bool

    def __post_init__(self) -> None:
        _require_digest(self.evaluator_id, "$BarEvaluatorCertificate.evaluator_id")
        _require_digest(self.resolution_id, "$BarEvaluatorCertificate.resolution_id")
        if type(self.finite_group) is not FiniteGroupTable:
            raise TypeError("$BarEvaluatorCertificate.finite_group: invalid table")
        dimensions = tuple(self.coordinate_dimensions)
        if len(dimensions) < 3 or any(
            type(value) is not int or value < 0 for value in dimensions
        ):
            raise ValueError("$BarEvaluatorCertificate.coordinate_dimensions: invalid")
        traces = tuple(self.traces)
        if any(type(trace) is not BarCoordinateTrace for trace in traces):
            raise TypeError("$BarEvaluatorCertificate.traces: invalid trace")
        keys = tuple((trace.degree, trace.group_tuple) for trace in traces)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("$BarEvaluatorCertificate.traces: expected canonical queries")
        elements = set(self.finite_group.element_order)
        for trace in traces:
            if trace.resolution_id != self.resolution_id:
                raise ValueError("bar trace binds another resolution")
            if len(trace.coordinate_weights) != dimensions[trace.degree]:
                raise ValueError("bar trace coordinate dimension differs")
            if any(element not in elements for element in trace.group_tuple):
                raise ValueError("bar trace leaves the finite presentation")
        if type(self.diagnostic) is not bool:
            raise TypeError("$BarEvaluatorCertificate.diagnostic: expected boolean")
        if self.diagnostic:
            if self.equivalence is not None or self.authority is not None:
                raise ValueError("diagnostic bar evaluator cannot claim Task-5 authority")
        else:
            if not isinstance(self.equivalence, BarResolutionEquivalence) or not isinstance(
                self.authority,
                Task5VerificationAuthority,
            ):
                raise TypeError(
                    "release bar evaluator requires a Task-5 equivalence and authority"
                )
            if traces:
                raise ValueError("release bar evaluator derives traces on demand")
            if (
                self.equivalence.resolution_id != self.resolution_id
                or self.equivalence.finite_group != self.finite_group
                or dimensions
                != tuple(len(basis) for basis in self.equivalence.resolution.basis)
            ):
                raise ValueError("release bar evaluator differs from Task-5 equivalence")
            report = verify_bar_resolution_equivalence(
                self.equivalence,
                self.authority,
            )
            if not report.valid:
                raise ValueError(
                    "Task-5 bar equivalence replay failed: "
                    + ", ".join(issue.code for issue in report.issues)
                )
        core = {
            "bar_equivalence_id": (
                None if self.equivalence is None else self.equivalence.equivalence_id
            ),
            "coordinate_dimensions": list(dimensions),
            "diagnostic": self.diagnostic,
            "finite_group_table_digest": self.finite_group.table_digest,
            "resolution_id": self.resolution_id,
            "task5_authority": (
                None if self.authority is None else asdict(self.authority)
            ),
            "trace_ids": [trace.trace_id for trace in traces],
        }
        if self.evaluator_id != _digest("bar-evaluator-certificate", core):
            raise ValueError("$BarEvaluatorCertificate.evaluator_id: payload digest differs")
        object.__setattr__(self, "coordinate_dimensions", dimensions)
        object.__setattr__(self, "traces", traces)

    def trace(self, group_tuple: Sequence[str]) -> BarCoordinateTrace | None:
        query = tuple(group_tuple)
        elements = set(self.finite_group.element_order)
        if len(query) not in (1, 2) or any(element not in elements for element in query):
            raise KeyError(f"bar evaluator query is outside certified domain: {query!r}")
        if any(
            element == self.finite_group.element_order[self.finite_group.identity_index]
            for element in query
        ):
            return None
        if not self.diagnostic:
            raise TypeError(
                "release bar evaluator requires coordinate_weights and a coefficient character"
            )
        for trace in self.traces:
            if trace.degree == len(query) and trace.group_tuple == query:
                return trace
        raise KeyError(f"bar evaluator query is outside certified domain: {query!r}")

    def coordinate_weights(
        self,
        group_tuple: Sequence[str],
        coefficient_character: Sequence[int],
    ) -> tuple[int, ...] | None:
        """Return the exact finite-resolution functional for one bar query."""

        query = tuple(group_tuple)
        elements = set(self.finite_group.element_order)
        if len(query) not in (1, 2) or any(element not in elements for element in query):
            raise KeyError(f"bar evaluator query is outside certified domain: {query!r}")
        bits = tuple(coefficient_character)
        if len(bits) != len(self.finite_group.element_order) or any(
            type(bit) is not int or bit not in (0, 1) for bit in bits
        ):
            raise ValueError("bar evaluator coefficient character has wrong domain")
        identity = self.finite_group.element_order[self.finite_group.identity_index]
        if any(element == identity for element in query):
            return None
        if self.diagnostic:
            trace = self.trace(query)
            assert trace is not None
            return trace.coordinate_weights
        assert self.equivalence is not None
        if query not in self.equivalence.queried_bar_tuples:
            raise KeyError(f"bar evaluator query is outside certified domain: {query!r}")
        character = GF2Character(bits)
        dimension = self.coordinate_dimensions[len(query)]
        weights: list[int] = []
        for column in range(dimension):
            coordinates = [0] * dimension
            coordinates[column] = 1
            value = evaluate_bar_cochain(
                self.equivalence,
                coordinates,
                query,
                coefficient_character=character,
            )
            if value.denominator != 1:
                raise ArithmeticError("bar comparison produced a nonintegral functional")
            weights.append(value.numerator)
        return tuple(weights)


def make_diagnostic_bar_evaluator(
    *,
    resolution_id: str,
    finite_group: FiniteGroupTable,
    coordinate_dimensions: Sequence[int],
    traces: Sequence[BarCoordinateTrace],
) -> BarEvaluatorCertificate:
    """Construct an explicitly diagnostic evaluator for focused tests."""

    dimensions = tuple(coordinate_dimensions)
    ordered = tuple(sorted(tuple(traces), key=lambda item: (item.degree, item.group_tuple)))
    core = {
        "bar_equivalence_id": None,
        "coordinate_dimensions": list(dimensions),
        "diagnostic": True,
        "finite_group_table_digest": finite_group.table_digest,
        "resolution_id": resolution_id,
        "task5_authority": None,
        "trace_ids": [trace.trace_id for trace in ordered],
    }
    return BarEvaluatorCertificate(
        _digest("bar-evaluator-certificate", core),
        resolution_id,
        finite_group,
        dimensions,
        ordered,
        None,
        None,
        True,
    )


def make_bar_evaluator_certificate(
    *,
    equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
) -> BarEvaluatorCertificate:
    """Bind a release evaluator to a replayed Task-5 bar equivalence."""

    if not isinstance(equivalence, BarResolutionEquivalence):
        raise TypeError("bar evaluator requires BarResolutionEquivalence")
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("bar evaluator requires Task5VerificationAuthority")
    dimensions = tuple(len(basis) for basis in equivalence.resolution.basis)
    core = {
        "bar_equivalence_id": equivalence.equivalence_id,
        "coordinate_dimensions": list(dimensions),
        "diagnostic": False,
        "finite_group_table_digest": equivalence.finite_group.table_digest,
        "resolution_id": equivalence.resolution_id,
        "task5_authority": asdict(authority),
        "trace_ids": [],
    }
    return BarEvaluatorCertificate(
        _digest("bar-evaluator-certificate", core),
        equivalence.resolution_id,
        equivalence.finite_group,
        dimensions,
        (),
        equivalence,
        authority,
        False,
    )


@dataclass(frozen=True, slots=True)
class ContinuousOrbitPresentation:
    """Finite, deeply immutable presentation of one continuous orbit component."""

    representative_stratum_id: str
    framed_stratum_ids: tuple[str, ...]
    arrow_ids: tuple[str, ...]
    free_rank: int
    torsion_orders: tuple[tuple[int, ...], ...]
    global_weyl_generator_count: int

    def __post_init__(self) -> None:
        _require_digest(
            self.representative_stratum_id,
            "$ContinuousOrbitPresentation.representative_stratum_id",
        )
        framed = tuple(self.framed_stratum_ids)
        arrows = tuple(self.arrow_ids)
        torsion = tuple(tuple(row) for row in self.torsion_orders)
        if not framed or framed != tuple(sorted(set(framed))):
            raise ValueError("continuous presentation framed strata are not canonical")
        if self.representative_stratum_id not in framed:
            raise ValueError("continuous representative is outside its framed component")
        if arrows != tuple(sorted(set(arrows))):
            raise ValueError("continuous presentation arrow IDs are not canonical")
        for index, arrow_id in enumerate(arrows):
            _require_digest(arrow_id, f"$ContinuousOrbitPresentation.arrow_ids[{index}]")
        if type(self.free_rank) is not int or self.free_rank < 0:
            raise ValueError("continuous presentation free rank is invalid")
        if any(
            type(order) is not int or order <= 1
            for row in torsion
            for order in row
        ):
            raise ValueError("continuous presentation torsion orders are invalid")
        if (
            type(self.global_weyl_generator_count) is not int
            or self.global_weyl_generator_count < 0
        ):
            raise ValueError("continuous presentation Weyl count is invalid")
        object.__setattr__(self, "framed_stratum_ids", framed)
        object.__setattr__(self, "arrow_ids", arrows)
        object.__setattr__(self, "torsion_orders", torsion)

    def mapping(self) -> dict[str, object]:
        return {
            "arrow_ids": list(self.arrow_ids),
            "framed_stratum_ids": list(self.framed_stratum_ids),
            "free_rank": self.free_rank,
            "global_weyl_generator_count": self.global_weyl_generator_count,
            "representative_stratum_id": self.representative_stratum_id,
            "torsion_orders": [list(row) for row in self.torsion_orders],
        }


@dataclass(frozen=True, slots=True, order=True)
class FiniteOrbitRepresentative:
    """Canonical typed point representing one finite residual-groupoid orbit."""

    representative_id: str
    stratum_id: str
    coordinates: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_digest(
            self.representative_id,
            "$FiniteOrbitRepresentative.representative_id",
        )
        if type(self.stratum_id) is not str or not self.stratum_id:
            raise ValueError("$FiniteOrbitRepresentative.stratum_id: invalid identifier")
        coordinates = tuple(self.coordinates)
        if any(type(value) is not int or value < 0 for value in coordinates):
            raise ValueError(
                "$FiniteOrbitRepresentative.coordinates: expected nonnegative integers"
            )
        core = {
            "coordinates": list(coordinates),
            "stratum_id": self.stratum_id,
        }
        if self.representative_id != _digest("finite-orbit-representative", core):
            raise ValueError(
                "$FiniteOrbitRepresentative.representative_id: payload differs"
            )
        object.__setattr__(self, "coordinates", coordinates)

    def mapping(self) -> dict[str, object]:
        return {
            "coordinates": list(self.coordinates),
            "representative_id": self.representative_id,
            "stratum_id": self.stratum_id,
        }


def make_finite_orbit_representative(
    stratum_id: str,
    coordinates: Sequence[int],
) -> FiniteOrbitRepresentative:
    """Bind one exact finite stratum point to an unambiguous semantic ID."""

    values = tuple(coordinates)
    core = {
        "coordinates": list(values),
        "stratum_id": stratum_id,
    }
    return FiniteOrbitRepresentative(
        _digest("finite-orbit-representative", core),
        stratum_id,
        values,
    )


@dataclass(frozen=True, slots=True, order=True)
class FiniteOrbitPathCertificate:
    """Canonical residual-arrow path between two finite framed points."""

    path_id: str
    source: FiniteOrbitRepresentative
    target: FiniteOrbitRepresentative
    arrow_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest(self.path_id, "$FiniteOrbitPathCertificate.path_id")
        if (
            type(self.source) is not FiniteOrbitRepresentative
            or type(self.target) is not FiniteOrbitRepresentative
        ):
            raise TypeError(
                "$FiniteOrbitPathCertificate: expected typed finite endpoints"
            )
        arrows = tuple(self.arrow_ids)
        for index, arrow_id in enumerate(arrows):
            _require_digest(
                arrow_id,
                f"$FiniteOrbitPathCertificate.arrow_ids[{index}]",
            )
        if (self.source == self.target) != (not arrows):
            raise ValueError(
                "$FiniteOrbitPathCertificate: only the representative has an empty path"
            )
        core = {
            "arrow_ids": list(arrows),
            "source": self.source.mapping(),
            "target": self.target.mapping(),
        }
        if self.path_id != _digest("finite-orbit-path", core):
            raise ValueError("$FiniteOrbitPathCertificate.path_id: payload differs")
        object.__setattr__(self, "arrow_ids", arrows)

    def mapping(self) -> dict[str, object]:
        return {
            "arrow_ids": list(self.arrow_ids),
            "path_id": self.path_id,
            "source": self.source.mapping(),
            "target": self.target.mapping(),
        }


def make_finite_orbit_path(
    source: FiniteOrbitRepresentative,
    target: FiniteOrbitRepresentative,
    arrow_ids: Sequence[str],
) -> FiniteOrbitPathCertificate:
    arrows = tuple(arrow_ids)
    core = {
        "arrow_ids": list(arrows),
        "source": source.mapping(),
        "target": target.mapping(),
    }
    return FiniteOrbitPathCertificate(
        _digest("finite-orbit-path", core),
        source,
        target,
        arrows,
    )


@dataclass(frozen=True, slots=True, order=True)
class FiniteOrbitMembershipCertificate:
    """Complete finite point orbit with one replayable path per member."""

    membership_id: str
    groupoid_digest: str
    representative: FiniteOrbitRepresentative
    members: tuple[FiniteOrbitRepresentative, ...]
    paths: tuple[FiniteOrbitPathCertificate, ...]
    arrow_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest(
            self.membership_id,
            "$FiniteOrbitMembershipCertificate.membership_id",
        )
        _require_digest(
            self.groupoid_digest,
            "$FiniteOrbitMembershipCertificate.groupoid_digest",
        )
        if type(self.representative) is not FiniteOrbitRepresentative:
            raise TypeError(
                "$FiniteOrbitMembershipCertificate.representative: expected typed point"
            )
        members = tuple(self.members)
        paths = tuple(self.paths)
        arrows = tuple(self.arrow_ids)
        if (
            not members
            or any(type(item) is not FiniteOrbitRepresentative for item in members)
            or members != tuple(sorted(set(members)))
            or members[0] != self.representative
        ):
            raise ValueError(
                "$FiniteOrbitMembershipCertificate.members: expected canonical complete orbit"
            )
        if (
            any(type(item) is not FiniteOrbitPathCertificate for item in paths)
            or tuple(path.target for path in paths) != members
            or any(path.source != self.representative for path in paths)
        ):
            raise ValueError(
                "$FiniteOrbitMembershipCertificate.paths: expected one canonical "
                "representative path per member"
            )
        if not arrows or arrows != tuple(sorted(set(arrows))):
            raise ValueError(
                "$FiniteOrbitMembershipCertificate.arrow_ids: expected complete canonical arrows"
            )
        for index, arrow_id in enumerate(arrows):
            _require_digest(
                arrow_id,
                f"$FiniteOrbitMembershipCertificate.arrow_ids[{index}]",
            )
        if any(
            arrow_id not in arrows
            for path in paths
            for arrow_id in path.arrow_ids
        ):
            raise ValueError(
                "$FiniteOrbitMembershipCertificate.arrow_ids: path arrow is absent"
            )
        core = {
            "arrow_ids": list(arrows),
            "groupoid_digest": self.groupoid_digest,
            "members": [member.mapping() for member in members],
            "paths": [path.mapping() for path in paths],
            "representative": self.representative.mapping(),
        }
        if self.membership_id != _digest("finite-orbit-membership", core):
            raise ValueError(
                "$FiniteOrbitMembershipCertificate.membership_id: payload differs"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "arrow_ids", arrows)

    def mapping(self) -> dict[str, object]:
        return {
            "arrow_ids": list(self.arrow_ids),
            "groupoid_digest": self.groupoid_digest,
            "members": [member.mapping() for member in self.members],
            "membership_id": self.membership_id,
            "paths": [path.mapping() for path in self.paths],
            "representative": self.representative.mapping(),
        }


def make_finite_orbit_membership(
    groupoid_digest: str,
    representative: FiniteOrbitRepresentative,
    members: Sequence[FiniteOrbitRepresentative],
    paths: Sequence[FiniteOrbitPathCertificate],
    arrow_ids: Sequence[str],
) -> FiniteOrbitMembershipCertificate:
    normalized_members = tuple(members)
    normalized_paths = tuple(paths)
    normalized_arrows = tuple(arrow_ids)
    core = {
        "arrow_ids": list(normalized_arrows),
        "groupoid_digest": groupoid_digest,
        "members": [member.mapping() for member in normalized_members],
        "paths": [path.mapping() for path in normalized_paths],
        "representative": representative.mapping(),
    }
    return FiniteOrbitMembershipCertificate(
        _digest("finite-orbit-membership", core),
        groupoid_digest,
        representative,
        normalized_members,
        normalized_paths,
        normalized_arrows,
    )


@dataclass(frozen=True, slots=True)
class UnframedQuotientCertificate:
    certificate_id: str
    framed_stratum_ids: tuple[str, ...]
    groupoid_digest: str
    orbit_representatives: tuple[FiniteOrbitRepresentative, ...]
    framed_finite_cardinality: int | None
    unframed_finite_cardinality: int | None
    continuous_orbit_presentations: tuple[ContinuousOrbitPresentation, ...]
    finite_orbit_memberships: tuple[FiniteOrbitMembershipCertificate, ...] = ()

    def __post_init__(self) -> None:
        _require_digest(self.certificate_id, "$UnframedQuotientCertificate.certificate_id")
        _require_digest(self.groupoid_digest, "$UnframedQuotientCertificate.groupoid_digest")
        framed = tuple(self.framed_stratum_ids)
        representatives = tuple(self.orbit_representatives)
        presentations = tuple(self.continuous_orbit_presentations)
        memberships = tuple(self.finite_orbit_memberships)
        if any(type(item) is not ContinuousOrbitPresentation for item in presentations):
            raise TypeError("continuous quotient presentations must be frozen typed records")
        if framed != tuple(sorted(set(framed))):
            raise ValueError("framed stratum IDs must be canonical and complete")
        if any(type(value) is not FiniteOrbitRepresentative for value in representatives):
            raise TypeError("finite orbit representatives must be typed point records")
        if any(
            type(value) is not FiniteOrbitMembershipCertificate
            for value in memberships
        ):
            raise TypeError("finite orbit memberships must be typed path records")
        for value in (self.framed_finite_cardinality, self.unframed_finite_cardinality):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("finite cardinalities must be nonnegative integers")
        if not framed:
            if (
                representatives
                or presentations
                or memberships
                or self.framed_finite_cardinality != 0
                or self.unframed_finite_cardinality != 0
            ):
                raise ValueError(
                    "empty quotient requires zero cardinalities and no "
                    "representatives/presentations"
                )
        elif presentations and (
            self.framed_finite_cardinality is not None
            or self.unframed_finite_cardinality is not None
        ):
            raise ValueError("continuous quotient cannot claim point cardinalities")
        if not framed:
            pass
        elif presentations:
            if representatives or memberships:
                raise ValueError("continuous quotient cannot carry finite point representatives")
            covered = tuple(
                sorted(
                    instance
                    for presentation in presentations
                    for instance in presentation.framed_stratum_ids
                )
            )
            if covered != framed:
                raise ValueError("continuous presentations do not partition framed strata")
        else:
            if (
                self.framed_finite_cardinality is None
                or self.unframed_finite_cardinality is None
            ):
                raise ValueError("finite quotient requires both exact cardinalities")
            if len(set(representatives)) != len(representatives):
                raise ValueError("finite quotient representatives are not unique")
            if self.unframed_finite_cardinality != len(representatives):
                raise ValueError(
                    "unframed finite cardinality differs from unique representative count"
                )
            if self.unframed_finite_cardinality > self.framed_finite_cardinality:
                raise ValueError("unframed finite cardinality exceeds framed cardinality")
            if (
                len(memberships) != self.unframed_finite_cardinality
                or tuple(item.representative for item in memberships)
                != representatives
                or any(
                    item.groupoid_digest != self.groupoid_digest
                    for item in memberships
                )
                or sum(len(item.members) for item in memberships)
                != self.framed_finite_cardinality
                or len(
                    {
                        member
                        for membership in memberships
                        for member in membership.members
                    }
                )
                != self.framed_finite_cardinality
            ):
                raise ValueError(
                    "finite orbit memberships do not partition the framed point set"
                )
        core = {
            "continuous_orbit_presentations": [
                presentation.mapping() for presentation in presentations
            ],
            "finite_orbit_memberships": [
                membership.mapping() for membership in memberships
            ],
            "framed_finite_cardinality": self.framed_finite_cardinality,
            "framed_stratum_ids": list(framed),
            "groupoid_digest": self.groupoid_digest,
            "orbit_representatives": [
                representative.mapping() for representative in representatives
            ],
            "unframed_finite_cardinality": self.unframed_finite_cardinality,
        }
        if self.certificate_id != _digest("unframed-quotient-certificate", core):
            raise ValueError("$UnframedQuotientCertificate.certificate_id: payload differs")
        object.__setattr__(self, "framed_stratum_ids", framed)
        object.__setattr__(self, "orbit_representatives", representatives)
        object.__setattr__(self, "continuous_orbit_presentations", presentations)
        object.__setattr__(self, "finite_orbit_memberships", memberships)


@dataclass(frozen=True, slots=True, order=True)
class PeriodicSite:
    """One certified site in the finite reconstruction query domain."""

    instance_id: str
    cell: tuple[int, int, int]
    branch_index: int

    def __post_init__(self) -> None:
        if type(self.instance_id) is not str or not self.instance_id:
            raise ValueError("$PeriodicSite.instance_id: invalid identifier")
        cell = tuple(self.cell)
        if len(cell) != 3 or any(type(value) is not int for value in cell):
            raise ValueError("$PeriodicSite.cell: expected three exact integers")
        if type(self.branch_index) is not int or self.branch_index < 0:
            raise ValueError("$PeriodicSite.branch_index: expected nonnegative integer")
        object.__setattr__(self, "cell", cell)

    def mapping(self) -> dict[str, object]:
        return {
            "branch_index": self.branch_index,
            "cell": list(self.cell),
            "instance_id": self.instance_id,
        }


@dataclass(frozen=True, slots=True)
class SiteTransport:
    transport_id: str
    site: PeriodicSite
    ambient_element: str
    catalogue_transport_digest: str
    z2_transport_lift: ExactQuaternion | None = None

    def __post_init__(self) -> None:
        _require_digest(self.transport_id, "$SiteTransport.transport_id")
        if type(self.site) is not PeriodicSite:
            raise TypeError("$SiteTransport.site: expected PeriodicSite")
        if type(self.ambient_element) is not str or not self.ambient_element:
            raise ValueError("$SiteTransport.ambient_element: invalid element")
        _require_digest(
            self.catalogue_transport_digest,
            "$SiteTransport.catalogue_transport_digest",
        )
        if self.z2_transport_lift is not None and (
            type(self.z2_transport_lift) is not ExactQuaternion
            or self.z2_transport_lift.norm_squared().coefficients
            != (
                Fraction(1),
                Fraction(0),
                Fraction(0),
                Fraction(0),
            )
        ):
            raise ValueError("$SiteTransport.z2_transport_lift: expected exact SU2 lift")
        core = {
            "ambient_element": self.ambient_element,
            "catalogue_transport_digest": self.catalogue_transport_digest,
            "site": self.site.mapping(),
            "z2_transport_lift": (
                None
                if self.z2_transport_lift is None
                else self.z2_transport_lift.to_json()
            ),
        }
        if self.transport_id != _digest("site-transport", core):
            raise ValueError("$SiteTransport.transport_id: payload digest differs")


@dataclass(frozen=True, slots=True)
class GeneratorAction:
    action_id: str
    name: str
    ambient_element: str
    antiunitary_grade: int
    inverse_site_images: tuple[tuple[PeriodicSite, PeriodicSite], ...]
    action_provenance_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.action_id, "$GeneratorAction.action_id")
        if type(self.name) is not str or not self.name or self.name.endswith("^-1"):
            raise ValueError("$GeneratorAction.name: invalid generator name")
        if type(self.ambient_element) is not str or not self.ambient_element:
            raise ValueError("$GeneratorAction.ambient_element: invalid element")
        if type(self.antiunitary_grade) is not int or self.antiunitary_grade not in (0, 1):
            raise ValueError("$GeneratorAction.antiunitary_grade: expected bit")
        images = tuple(tuple(pair) for pair in self.inverse_site_images)
        if not images or any(
            len(pair) != 2
            or type(pair[0]) is not PeriodicSite
            or type(pair[1]) is not PeriodicSite
            for pair in images
        ):
            raise ValueError("$GeneratorAction.inverse_site_images: invalid site map")
        sources = tuple(pair[0] for pair in images)
        if sources != tuple(sorted(set(sources))):
            raise ValueError("generator action source domain must be canonical and unique")
        _require_digest(
            self.action_provenance_digest,
            "$GeneratorAction.action_provenance_digest",
        )
        core = {
            "action_provenance_digest": self.action_provenance_digest,
            "ambient_element": self.ambient_element,
            "antiunitary_grade": self.antiunitary_grade,
            "inverse_site_images": [
                [source.mapping(), target.mapping()] for source, target in images
            ],
            "name": self.name,
        }
        if self.action_id != _digest("generator-action", core):
            raise ValueError("$GeneratorAction.action_id: payload digest differs")
        object.__setattr__(self, "inverse_site_images", images)


@dataclass(frozen=True, slots=True)
class Relator:
    relator_id: str
    name: str
    kind: Literal["spatial", "time_square", "mixed_time_space"]
    word: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest(self.relator_id, "$Relator.relator_id")
        if type(self.name) is not str or not self.name:
            raise ValueError("$Relator.name: invalid name")
        if self.kind not in ("spatial", "time_square", "mixed_time_space"):
            raise ValueError("$Relator.kind: invalid relation kind")
        word = tuple(self.word)
        if not word or any(type(token) is not str or not token for token in word):
            raise ValueError("$Relator.word: expected a nonempty word")
        core = {"kind": self.kind, "name": self.name, "word": list(word)}
        if self.relator_id != _digest("relator", core):
            raise ValueError("$Relator.relator_id: payload digest differs")
        object.__setattr__(self, "word", word)


LocalSkeleton = U1LocalSkeleton | Z2LocalSkeleton


@dataclass(frozen=True, slots=True)
class OrbitReconstructionData:
    orbit_id: str
    instance_id: str
    local_evaluator: BarEvaluatorCertificate
    skeleton: LocalSkeleton
    stabilizer_element_map: tuple[tuple[str, str], ...]
    local_grade_values: tuple[int, ...]
    local_rho_values: tuple[int, ...]
    catalogue_record_digest: str
    literal_stabilizer_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.orbit_id, "$OrbitReconstructionData.orbit_id")
        if type(self.instance_id) is not str or not self.instance_id:
            raise ValueError("$OrbitReconstructionData.instance_id: invalid identifier")
        if type(self.local_evaluator) is not BarEvaluatorCertificate:
            raise TypeError("$OrbitReconstructionData.local_evaluator: invalid evaluator")
        if not isinstance(self.skeleton, (U1LocalSkeleton, Z2LocalSkeleton)):
            raise TypeError("$OrbitReconstructionData.skeleton: invalid local skeleton")
        mapping = tuple(tuple(pair) for pair in self.stabilizer_element_map)
        if not mapping or any(
            len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            for pair in mapping
        ):
            raise ValueError("$OrbitReconstructionData.stabilizer_element_map: invalid")
        if tuple(pair[0] for pair in mapping) != tuple(sorted(set(pair[0] for pair in mapping))):
            raise ValueError("stabilizer ambient keys must be canonical and unique")
        local_elements = set(self.local_evaluator.finite_group.element_order)
        if any(local not in local_elements for _, local in mapping):
            raise ValueError("stabilizer map leaves the certified local table")
        grades = tuple(self.local_grade_values)
        rhos = tuple(self.local_rho_values)
        order = len(self.local_evaluator.finite_group.element_order)
        if len(grades) != order or len(rhos) != order or any(
            type(bit) is not int or bit not in (0, 1) for bit in grades + rhos
        ):
            raise ValueError("local grade/rho assignments have wrong finite-table domain")
        if type(self.skeleton) is U1LocalSkeleton:
            verify_u1_local_skeleton(self.skeleton, self.local_evaluator.finite_group)
            if grades != self.skeleton.grade_values or rhos != self.skeleton.rho_values:
                raise ValueError("local U1 grade/rho assignments differ from skeleton")
        else:
            verify_z2_local_skeleton(
                self.skeleton,
                self.local_evaluator.finite_group,
            )
            expected_order = (
                len(self.skeleton.full_graded_su2_lifts)
                if self.skeleton.full_graded_su2_lifts
                else len(self.skeleton.su2_lifts)
            )
            if expected_order != order or any(rhos):
                raise ValueError("local Z2 evaluator differs from exact skeleton domain")
        _require_digest(
            self.catalogue_record_digest,
            "$OrbitReconstructionData.catalogue_record_digest",
        )
        _require_digest(
            self.literal_stabilizer_digest,
            "$OrbitReconstructionData.literal_stabilizer_digest",
        )
        core = {
            "catalogue_record_digest": self.catalogue_record_digest,
            "instance_id": self.instance_id,
            "literal_stabilizer_digest": self.literal_stabilizer_digest,
            "local_evaluator_id": self.local_evaluator.evaluator_id,
            "local_grade_values": list(grades),
            "local_rho_values": list(rhos),
            "skeleton_id": self.skeleton.skeleton_id,
            "stabilizer_element_map": [list(pair) for pair in mapping],
        }
        if self.orbit_id != _digest("orbit-reconstruction-data", core):
            raise ValueError("$OrbitReconstructionData.orbit_id: payload digest differs")
        object.__setattr__(self, "stabilizer_element_map", mapping)
        object.__setattr__(self, "local_grade_values", grades)
        object.__setattr__(self, "local_rho_values", rhos)


@dataclass(frozen=True, slots=True)
class ReconstructionProblem:
    problem_id: str
    relative_certificate_id: str
    relative_problem: RelativeProblem
    relative_matrices: RelativeMatrices
    igg: Literal["Z2", "U1"]
    ambient_evaluator: BarEvaluatorCertificate
    ambient_grade_values: tuple[int, ...]
    ambient_rho_values: tuple[int, ...]
    generators: tuple[GeneratorAction, ...]
    site_transports: tuple[SiteTransport, ...]
    orbits: tuple[OrbitReconstructionData, ...]
    relators: tuple[Relator, ...]
    diagnostic: bool

    def __post_init__(self) -> None:
        _require_digest(self.problem_id, "$ReconstructionProblem.problem_id")
        _require_digest(
            self.relative_certificate_id,
            "$ReconstructionProblem.relative_certificate_id",
        )
        if type(self.relative_problem) is not RelativeProblem:
            raise TypeError("$ReconstructionProblem.relative_problem: invalid source problem")
        if type(self.relative_matrices) is not RelativeMatrices:
            raise TypeError("$ReconstructionProblem.relative_matrices: invalid matrices")
        verify_relative_certificate(self.relative_matrices, self.relative_problem)
        if self.relative_matrices.certificate.certificate_id != self.relative_certificate_id:
            raise ValueError("reconstruction relative certificate differs from source replay")
        if self.igg not in ("Z2", "U1"):
            raise ValueError("$ReconstructionProblem.igg: expected Z2 or U1")
        expected_ring = "torus" if self.igg == "U1" else "gf2"
        if self.relative_problem.ring != expected_ring:
            raise ValueError("reconstruction IGG differs from relative coefficient ring")
        if type(self.ambient_evaluator) is not BarEvaluatorCertificate:
            raise TypeError("$ReconstructionProblem.ambient_evaluator: invalid ambient evaluator")
        if self.ambient_evaluator.resolution_id != self.relative_problem.ambient.authority_id:
            raise ValueError(
                "ambient evaluator resolution authority differs from relative problem"
            )
        order = len(self.ambient_evaluator.finite_group.element_order)
        grades = tuple(self.ambient_grade_values)
        rhos = tuple(self.ambient_rho_values)
        if len(grades) != order or len(rhos) != order or any(
            type(bit) is not int or bit not in (0, 1) for bit in grades + rhos
        ):
            raise ValueError("ambient grade/rho assignments have wrong finite-table domain")
        table = self.ambient_evaluator.finite_group
        for values, name in ((grades, "grade"), (rhos, "rho")):
            if values[table.identity_index] != 0 or any(
                values[table.multiplication_table[left][right]]
                != (values[left] ^ values[right])
                for left in range(order)
                for right in range(order)
            ):
                raise ValueError(f"ambient {name} assignment is not a character")
        if self.igg == "Z2" and any(rhos):
            raise ValueError("Z2 coefficients have trivial automorphism character")
        generators = tuple(self.generators)
        transports = tuple(self.site_transports)
        orbits = tuple(self.orbits)
        relators = tuple(self.relators)
        if not generators or any(type(item) is not GeneratorAction for item in generators):
            raise ValueError("reconstruction problem requires generator actions")
        if not transports or any(type(item) is not SiteTransport for item in transports):
            raise ValueError("reconstruction problem requires certified site transports")
        if not orbits or any(type(item) is not OrbitReconstructionData for item in orbits):
            raise ValueError("reconstruction problem requires local orbit data")
        if not relators or any(type(item) is not Relator for item in relators):
            raise ValueError("reconstruction problem requires defining relators")
        names = tuple(item.name for item in generators)
        if len(set(names)) != len(names):
            raise ValueError("reconstruction generator names must be unique")
        element_index = {
            element: index for index, element in enumerate(table.element_order)
        }
        if any(item.ambient_element not in element_index for item in generators):
            raise ValueError("generator action leaves the ambient finite presentation")
        for generator in generators:
            if grades[element_index[generator.ambient_element]] != generator.antiunitary_grade:
                raise ValueError("generator antiunitary grade differs from ambient character")
        sites = tuple(item.site for item in transports)
        if sites != tuple(sorted(set(sites))):
            raise ValueError("site transport domain must be canonical and unique")
        if any(item.ambient_element not in element_index for item in transports):
            raise ValueError("site transport leaves the ambient finite presentation")
        ambient_identity = table.element_order[table.identity_index]
        if self.igg == "U1" and any(
            item.z2_transport_lift is not None for item in transports
        ):
            raise ValueError("U1 reconstruction cannot carry a Z2 transport lift")
        if self.igg == "Z2":
            for transport in transports:
                if grades[element_index[transport.ambient_element]]:
                    raise ValueError("Z2 site transport must be spatial, not antiunitary")
                if (
                    transport.ambient_element != ambient_identity
                    and transport.z2_transport_lift is None
                ):
                    raise ValueError("nonidentity Z2 site transport requires an exact lift")
        site_set = set(sites)
        for generator in generators:
            source_set = {source for source, _ in generator.inverse_site_images}
            target_set = {target for _, target in generator.inverse_site_images}
            if source_set != site_set or target_set != site_set:
                raise ValueError("generator site action does not permute the complete domain")
            if any(
                source.instance_id != target.instance_id
                for source, target in generator.inverse_site_images
            ):
                raise ValueError("generator site action crosses orbit instances")
        instance_ids = tuple(item.instance_id for item in orbits)
        if instance_ids != tuple(sorted(set(instance_ids))):
            raise ValueError("orbit reconstruction data must use canonical instance order")
        if {site.instance_id for site in sites} != set(instance_ids):
            raise ValueError("site transports and orbit instances differ")
        local_authority_by_instance = {
            restriction.instance_id: local.authority_id
            for local, restriction in zip(
                self.relative_problem.locals,
                self.relative_problem.restrictions,
                strict=True,
            )
        }
        if set(local_authority_by_instance) != set(instance_ids):
            raise ValueError("relative local authorities differ from orbit instances")
        for orbit in orbits:
            if (
                orbit.local_evaluator.resolution_id
                != local_authority_by_instance[orbit.instance_id]
            ):
                raise ValueError(
                    f"local evaluator resolution authority differs for {orbit.instance_id}"
                )
        for orbit in orbits:
            stabilizer = dict(orbit.stabilizer_element_map)
            if ambient_identity not in stabilizer:
                raise ValueError("stabilizer subgroup omits the ambient identity")
            local_table = orbit.local_evaluator.finite_group
            local_identity = local_table.element_order[local_table.identity_index]
            if stabilizer[ambient_identity] != local_identity:
                raise ValueError("stabilizer homomorphism does not preserve identity")
            for left in stabilizer:
                left_index = element_index.get(left)
                if left_index is None:
                    raise ValueError("stabilizer key leaves the ambient finite presentation")
                local_left = local_table.element_order.index(stabilizer[left])
                if (
                    grades[left_index] != orbit.local_grade_values[local_left]
                    or rhos[left_index] != orbit.local_rho_values[local_left]
                ):
                    raise ValueError("stabilizer characters do not restrict from ambient data")
                for right in stabilizer:
                    right_index = element_index[right]
                    product = table.element_order[
                        table.multiplication_table[left_index][right_index]
                    ]
                    if product not in stabilizer:
                        raise ValueError("stabilizer ambient keys are not a subgroup")
                    local_right = local_table.element_order.index(stabilizer[right])
                    local_product = local_table.element_order[
                        local_table.multiplication_table[local_left][local_right]
                    ]
                    if stabilizer[product] != local_product:
                        raise ValueError("stabilizer map is not a finite-table homomorphism")
        generator_set = set(names)
        for relator in relators:
            for token in relator.word:
                base = token[:-3] if token.endswith("^-1") else token
                if base not in generator_set:
                    raise ValueError("relator uses an unknown generator token")
        expected_diagnostic = self.ambient_evaluator.diagnostic or any(
            orbit.local_evaluator.diagnostic for orbit in orbits
        )
        if type(self.diagnostic) is not bool or self.diagnostic != expected_diagnostic:
            raise ValueError("reconstruction diagnostic provenance differs from evaluators")
        core = {
            "ambient_evaluator_id": self.ambient_evaluator.evaluator_id,
            "ambient_grade_values": list(grades),
            "ambient_rho_values": list(rhos),
            "diagnostic": self.diagnostic,
            "generator_action_ids": [item.action_id for item in generators],
            "igg": self.igg,
            "orbit_ids": [item.orbit_id for item in orbits],
            "relative_certificate_id": self.relative_certificate_id,
            "relator_ids": [item.relator_id for item in relators],
            "site_transport_ids": [item.transport_id for item in transports],
        }
        if self.problem_id != _digest("reconstruction-problem", core):
            raise ValueError("$ReconstructionProblem.problem_id: payload digest differs")
        object.__setattr__(self, "ambient_grade_values", grades)
        object.__setattr__(self, "ambient_rho_values", rhos)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "site_transports", transports)
        object.__setattr__(self, "orbits", orbits)
        object.__setattr__(self, "relators", relators)


@dataclass(frozen=True, slots=True)
class ExactGaugeElement:
    """An exact element of the Z2 or Pin-minus U1 graded normalizer."""

    igg: Literal["Z2", "U1"]
    phase: FormalPhase
    weyl_parity: int
    antiunitary_grade: int
    rotation: ExactQuaternion | None = None

    def __post_init__(self) -> None:
        if self.igg not in ("Z2", "U1"):
            raise ValueError("$ExactGaugeElement.igg: invalid IGG")
        if type(self.phase) is not FormalPhase:
            raise TypeError("$ExactGaugeElement.phase: expected FormalPhase")
        if self.weyl_parity not in (0, 1) or self.antiunitary_grade not in (0, 1):
            raise ValueError("$ExactGaugeElement: parity and grade must be bits")
        if self.igg == "Z2":
            if self.weyl_parity != 0 or type(self.rotation) is not ExactQuaternion:
                raise ValueError("Z2 gauge elements require an exact SU2 lift and q=0")
            if any(self.phase.coefficients) or self.phase.constant.value not in (
                Fraction(0),
                Fraction(1, 2),
            ):
                raise ValueError("Z2 central phase must be exactly 0 or 1/2")
            if self.rotation.norm_squared().coefficients != (
                Fraction(1),
                Fraction(0),
                Fraction(0),
                Fraction(0),
            ):
                raise ValueError("Z2 gauge rotation is not an exact unit quaternion")
        elif self.rotation is not None:
            raise ValueError("U1 normalizer elements do not carry an SU2 quaternion field")

    @classmethod
    def identity(
        cls, igg: Literal["Z2", "U1"], parameter_names: Sequence[str] = ()
    ) -> "ExactGaugeElement":
        return cls(
            igg,
            FormalPhase.zero(parameter_names),
            0,
            0,
            ONE_QUATERNION if igg == "Z2" else None,
        )

    def __mul__(self, other: object) -> "ExactGaugeElement":
        if not isinstance(other, ExactGaugeElement):
            return NotImplemented
        if self.igg != other.igg or self.phase.parameter_names != other.phase.parameter_names:
            raise ValueError("gauge elements use different coefficient sectors")
        if self.igg == "Z2":
            assert self.rotation is not None and other.rotation is not None
            return ExactGaugeElement(
                "Z2",
                self.phase + other.phase,
                0,
                self.antiunitary_grade ^ other.antiunitary_grade,
                self.rotation * other.rotation,
            )
        rho = self.weyl_parity ^ self.antiunitary_grade
        defect = FormalPhase(
            self.phase.parameter_names,
            (0,) * len(self.phase.parameter_names),
            Phase(
                Fraction(
                    self.weyl_parity * other.weyl_parity
                    + self.antiunitary_grade * other.weyl_parity,
                    2,
                )
            ),
        )
        return ExactGaugeElement(
            "U1",
            self.phase + other.phase.scale(-1 if rho else 1) + defect,
            self.weyl_parity ^ other.weyl_parity,
            self.antiunitary_grade ^ other.antiunitary_grade,
            None,
        )

    def inverse(self) -> "ExactGaugeElement":
        if self.igg == "Z2":
            assert self.rotation is not None
            rotation_inverse = ExactQuaternion(
                self.rotation.scalar,
                -self.rotation.x,
                -self.rotation.y,
                -self.rotation.z,
            )
            return ExactGaugeElement(
                "Z2",
                -self.phase,
                0,
                self.antiunitary_grade,
                rotation_inverse,
            )
        rho = self.weyl_parity ^ self.antiunitary_grade
        defect = FormalPhase(
            self.phase.parameter_names,
            (0,) * len(self.phase.parameter_names),
            Phase(
                Fraction(
                    self.weyl_parity * self.weyl_parity
                    + self.antiunitary_grade * self.weyl_parity,
                    2,
                )
            ),
        )
        inverse_phase = (-(self.phase + defect)).scale(-1 if rho else 1)
        return ExactGaugeElement(
            "U1",
            inverse_phase,
            self.weyl_parity,
            self.antiunitary_grade,
            None,
        )

    @property
    def is_central(self) -> bool:
        if self.weyl_parity or self.antiunitary_grade:
            return False
        if self.igg == "U1":
            return True
        assert self.rotation is not None
        return self.rotation in (ONE_QUATERNION, -ONE_QUATERNION)

    def central_phase(self) -> FormalPhase:
        if not self.is_central:
            raise ValueError("gauge element is not in the diagonal IGG")
        if self.igg == "U1":
            return self.phase
        assert self.rotation is not None
        rotation_phase = FormalPhase(
            self.phase.parameter_names,
            (0,) * len(self.phase.parameter_names),
            Phase(Fraction(0) if self.rotation == ONE_QUATERNION else Fraction(1, 2)),
        )
        return self.phase + rotation_phase

    @property
    def is_identity(self) -> bool:
        return self.is_central and self.central_phase().is_zero


@dataclass(frozen=True, slots=True)
class PSGEvaluatorCertificate:
    certificate_id: str
    problem: ReconstructionProblem
    stratum_id: str
    point_id: str
    relative_certificate_id: str
    parameter_names: tuple[str, ...]
    raw_constant: tuple[Phase, ...]
    raw_free_coefficients: MatrixZ
    ambient_degree_two_slice: tuple[int, int]
    local_degree_one_slices: tuple[tuple[str, tuple[int, int]], ...]
    coordinate_digest: str
    diagnostic: bool

    def __post_init__(self) -> None:
        _require_digest(self.certificate_id, "$PSGEvaluatorCertificate.certificate_id")
        if type(self.problem) is not ReconstructionProblem:
            raise TypeError("$PSGEvaluatorCertificate.problem: invalid envelope")
        for name in ("stratum_id", "point_id", "relative_certificate_id", "coordinate_digest"):
            _require_digest(getattr(self, name), f"$PSGEvaluatorCertificate.{name}")
        names = tuple(self.parameter_names)
        constant = tuple(self.raw_constant)
        coefficients = MatrixZ(self.raw_free_coefficients)
        if any(type(value) is not Phase for value in constant):
            raise TypeError("$PSGEvaluatorCertificate.raw_constant: expected phases")
        if coefficients.shape != (len(constant), len(names)):
            raise ValueError("raw free-coordinate matrix has incompatible shape")
        ambient_slice = tuple(self.ambient_degree_two_slice)
        if (
            len(ambient_slice) != 2
            or any(type(value) is not int for value in ambient_slice)
            or ambient_slice[0] != 0
            or not 0 <= ambient_slice[0] <= ambient_slice[1] <= len(constant)
        ):
            raise ValueError("ambient degree-two coordinate slice is invalid")
        local_slices = tuple(
            (instance_id, tuple(coordinate_slice))
            for instance_id, coordinate_slice in self.local_degree_one_slices
        )
        if tuple(instance_id for instance_id, _ in local_slices) != tuple(
            orbit.instance_id for orbit in self.problem.orbits
        ):
            raise ValueError("local coordinate slices differ from problem orbit order")
        cursor = ambient_slice[1]
        for instance_id, coordinate_slice in local_slices:
            if (
                len(coordinate_slice) != 2
                or any(type(value) is not int for value in coordinate_slice)
                or coordinate_slice[0] != cursor
                or coordinate_slice[1] < coordinate_slice[0]
                or coordinate_slice[1] > len(constant)
            ):
                raise ValueError(
                    f"local degree-one coordinate slice is invalid for {instance_id}"
                )
            cursor = coordinate_slice[1]
        if cursor != len(constant):
            raise ValueError("degree-two coordinate slices do not cover raw coordinates")
        if (
            self.problem.ambient_evaluator.coordinate_dimensions[2]
            != ambient_slice[1] - ambient_slice[0]
        ):
            raise ValueError("ambient evaluator and stored coordinate slice differ")
        for orbit, (_, coordinate_slice) in zip(
            self.problem.orbits,
            local_slices,
            strict=True,
        ):
            if (
                orbit.local_evaluator.coordinate_dimensions[1]
                != coordinate_slice[1] - coordinate_slice[0]
            ):
                raise ValueError("local evaluator and stored coordinate slice differ")
        if self.relative_certificate_id != self.problem.relative_certificate_id:
            raise ValueError("evaluator and problem relative certificates differ")
        if self.diagnostic != self.problem.diagnostic:
            raise ValueError("evaluator diagnostic provenance differs from problem")
        core = {
            "coordinate_digest": self.coordinate_digest,
            "diagnostic": self.diagnostic,
            "parameter_names": list(names),
            "point_id": self.point_id,
            "problem_id": self.problem.problem_id,
            "raw_constant": [_phase_text(value) for value in constant],
            "raw_free_coefficients": {
                "column_count": coefficients.column_count,
                "rows": [list(row) for row in coefficients],
            },
            "ambient_degree_two_slice": list(ambient_slice),
            "local_degree_one_slices": [
                [instance_id, list(coordinate_slice)]
                for instance_id, coordinate_slice in local_slices
            ],
            "relative_certificate_id": self.relative_certificate_id,
            "stratum_id": self.stratum_id,
        }
        if self.certificate_id != _digest("psg-evaluator-certificate", core):
            raise ValueError("$PSGEvaluatorCertificate.certificate_id: payload differs")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "raw_constant", constant)
        object.__setattr__(self, "raw_free_coefficients", coefficients)
        object.__setattr__(self, "ambient_degree_two_slice", ambient_slice)
        object.__setattr__(self, "local_degree_one_slices", local_slices)


@dataclass(frozen=True, slots=True)
class RelationEvaluation:
    relation_id: str
    relator_id: str
    relation_name: str
    kind: Literal["spatial", "time_square", "mixed_time_space"]
    site: PeriodicSite
    actual_phase: FormalPhase
    expected_phase: FormalPhase
    residual: FormalPhase

    def __post_init__(self) -> None:
        _require_digest(self.relation_id, "$RelationEvaluation.relation_id")
        _require_digest(self.relator_id, "$RelationEvaluation.relator_id")
        if type(self.relation_name) is not str or not self.relation_name:
            raise ValueError("$RelationEvaluation.relation_name: invalid name")
        if self.kind not in ("spatial", "time_square", "mixed_time_space"):
            raise ValueError("$RelationEvaluation.kind: invalid relation kind")
        if type(self.site) is not PeriodicSite:
            raise TypeError("$RelationEvaluation.site: expected PeriodicSite")
        if any(
            type(value) is not FormalPhase
            for value in (self.actual_phase, self.expected_phase, self.residual)
        ):
            raise TypeError("relation phases must be exact formal expressions")
        if self.residual != self.actual_phase - self.expected_phase:
            raise ValueError("relation residual does not replay exact phases")
        core = {
            "actual_phase": self.actual_phase.mapping(),
            "expected_phase": self.expected_phase.mapping(),
            "kind": self.kind,
            "relation_name": self.relation_name,
            "relator_id": self.relator_id,
            "residual": self.residual.mapping(),
            "site": self.site.mapping(),
        }
        if self.relation_id != _digest("relation-evaluation", core):
            raise ValueError("$RelationEvaluation.relation_id: payload differs")


@dataclass(frozen=True, slots=True)
class RelationCertificate:
    certificate_id: str
    evaluator: PSGEvaluatorCertificate
    evaluator_certificate_id: str
    query_sites: tuple[PeriodicSite, ...]
    diagnostic: bool
    results: tuple[RelationEvaluation, ...]
    checked_relations: int
    verified: bool

    def __post_init__(self) -> None:
        _require_digest(self.certificate_id, "$RelationCertificate.certificate_id")
        if type(self.evaluator) is not PSGEvaluatorCertificate:
            raise TypeError("$RelationCertificate.evaluator: invalid evaluator")
        _require_digest(
            self.evaluator_certificate_id,
            "$RelationCertificate.evaluator_certificate_id",
        )
        if self.evaluator_certificate_id != self.evaluator.certificate_id:
            raise ValueError("relation certificate binds another evaluator")
        sites = tuple(self.query_sites)
        expected_sites = tuple(
            sorted(transport.site for transport in self.evaluator.problem.site_transports)
        )
        if sites != expected_sites:
            raise ValueError("relation certificate query domain is not the complete site domain")
        if type(self.diagnostic) is not bool or self.diagnostic != self.evaluator.diagnostic:
            raise ValueError("relation certificate diagnostic provenance differs")
        results = tuple(self.results)
        if any(type(value) is not RelationEvaluation for value in results):
            raise TypeError("$RelationCertificate.results: invalid relation result")
        if type(self.checked_relations) is not int or self.checked_relations != len(results):
            raise ValueError("relation certificate count differs from results")
        expected_queries = tuple(
            (relator, site)
            for relator in self.evaluator.problem.relators
            for site in sites
        )
        if len(results) != len(expected_queries):
            raise ValueError("relation certificate lacks complete relator-site coverage")
        for result, (relator, site) in zip(results, expected_queries, strict=True):
            if (
                result.relator_id != relator.relator_id
                or result.relation_name != relator.name
                or result.kind != relator.kind
                or result.site != site
            ):
                raise ValueError("relation certificate query order or coverage differs")
        if type(self.verified) is not bool or self.verified != all(
            result.residual.is_zero for result in results
        ):
            raise ValueError("relation certificate verified flag differs from replay")
        core = {
            "diagnostic": self.diagnostic,
            "evaluator_certificate_id": self.evaluator_certificate_id,
            "query_sites": [site.mapping() for site in sites],
            "relation_ids": [item.relation_id for item in results],
            "verified": self.verified,
        }
        if self.certificate_id != _digest("relation-certificate", core):
            raise ValueError("$RelationCertificate.certificate_id: payload differs")
        object.__setattr__(self, "query_sites", sites)
        object.__setattr__(self, "results", results)


__all__ = [
    "BarCoordinateTrace",
    "BarEvaluatorCertificate",
    "ContinuousOrbitPresentation",
    "FiniteOrbitMembershipCertificate",
    "FiniteOrbitPathCertificate",
    "FiniteOrbitRepresentative",
    "ExactGaugeElement",
    "FormalPhase",
    "GeneratorAction",
    "OrbitReconstructionData",
    "PSGEvaluatorCertificate",
    "PeriodicSite",
    "ReconstructionProblem",
    "RelationCertificate",
    "RelationEvaluation",
    "Relator",
    "SiteTransport",
    "UnframedQuotientCertificate",
    "make_bar_evaluator_certificate",
    "make_diagnostic_bar_evaluator",
    "make_finite_orbit_membership",
    "make_finite_orbit_path",
    "make_finite_orbit_representative",
]
