r"""Exhaustive spatial :math:`\mathbb Z_2` local lifts for Wyckoff stabilizers.

Homomorphisms are enumerated into the two exact crystallographic rotation
hosts, checked against every multiplication-table entry, quotiented by exact
marked ``SO(3)`` conjugacy, and lifted canonically to ``SU(2)``.  There is no
search-radius or numerical tolerance in this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
import math
from pathlib import Path
import re
from typing import Any, Literal

from .algebraic import (
    ONE_Q23,
    ONE_QUATERNION,
    SQRT2,
    SQRT3,
    SQRT6,
    ZERO_Q23,
    ExactQuaternion,
    ExactSO3,
    Q23,
    identity_so3,
)
from .catalogue_schema import canonical_json
from .cochains import FiniteGroupTable
from .gf2 import MatrixGF2, rref
from .stabilizer_types import (
    STABILIZER_TYPE_IDS,
    _manifest_mapping,
    _read_manifest,
    _safe_read_regular,
    _strict_json,
    _types_bytes,
    _write_atomic,
    canonical_generators,
    canonical_stabilizer_table,
    identify_stabilizer_type,
    load_stabilizer_type_library,
)
from .z2_targets import (
    CertifiedConjugator,
    CertifiedHostElement,
    FiniteRotationGroup,
    certify_conjugator,
    dihedral_six_rotation_group,
    lift_certified_rotation,
    octahedral_rotation_group,
)


HostID = Literal["O", "D6"]
_HOST_COMPLETENESS = "finite-crystallographic-rotation-subgroup-theorem-v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROJECTED_IMAGE_TYPES = frozenset(
    {"C1", "C2", "C3", "C4", "C6", "C2xC2", "D3", "D4", "D6", "A4", "S4"}
)


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _index_tuple(
    value: object,
    context: str,
    *,
    nonempty: bool = False,
    sorted_unique: bool = False,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{context} must be a tuple")
    result = value
    if nonempty and not result:
        raise ValueError(f"{context} must be nonempty")
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{context} contains an invalid index")
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{context} must be sorted and unique")
    return result


def _sha256(payload: object, domain: str) -> str:
    return "sha256:" + hashlib.sha256(
        domain.encode("ascii") + b"|" + canonical_json(payload)
    ).hexdigest()


def _trace(matrix: ExactSO3) -> Q23:
    return matrix.rows[0][0] + matrix.rows[1][1] + matrix.rows[2][2]


def _character_key(matrices: tuple[ExactSO3, ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(_trace(matrix).coefficients for matrix in matrices)


def _element_orders(table: FiniteGroupTable) -> tuple[int, ...]:
    result: list[int] = []
    for element in range(len(table.element_order)):
        value = table.identity_index
        for exponent in range(1, len(table.element_order) + 1):
            value = table.multiplication_table[value][element]
            if value == table.identity_index:
                result.append(exponent)
                break
        else:
            raise ArithmeticError("finite source element has no order")
    return tuple(result)


def _generator_words(table: FiniteGroupTable, generators: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    words: list[tuple[int, ...] | None] = [None] * len(table.element_order)
    words[table.identity_index] = ()
    queue = deque((table.identity_index,))
    while queue:
        element = queue.popleft()
        word = words[element]
        assert word is not None
        for position, generator in enumerate(generators):
            target = table.multiplication_table[element][generator]
            if words[target] is None:
                words[target] = word + (position,)
                queue.append(target)
    if any(word is None for word in words):
        raise ArithmeticError("canonical source generators are incomplete")
    return tuple(word for word in words if word is not None)


def _host(host_id: HostID) -> FiniteRotationGroup:
    return octahedral_rotation_group() if host_id == "O" else dihedral_six_rotation_group()


def _homomorphisms(table: FiniteGroupTable, generators: tuple[int, ...], host: FiniteRotationGroup) -> tuple[tuple[int, ...], ...]:
    if not generators:
        return ((host.identity_index,),)
    source_orders = _element_orders(table)
    words = _generator_words(table, generators)
    candidate_sets = tuple(
        tuple(
            element
            for element in range(len(host.elements))
            if source_orders[generator] % host.element_orders[element] == 0
        )
        for generator in generators
    )
    homomorphisms: set[tuple[int, ...]] = set()
    for images in product(*candidate_sets):
        mapping: list[int] = []
        for word in words:
            image = host.identity_index
            for generator_position in word:
                image = host.multiplication_table[image][images[generator_position]]
            mapping.append(image)
        candidate = tuple(mapping)
        if all(
            candidate[table.multiplication_table[left][right]]
            == host.multiplication_table[candidate[left]][candidate[right]]
            for left in range(len(table.element_order))
            for right in range(len(table.element_order))
        ):
            homomorphisms.add(candidate)
    return tuple(sorted(homomorphisms))


def _conjugate_mapping(mapping: tuple[int, ...], conjugator: int, host: FiniteRotationGroup) -> tuple[int, ...]:
    inverse = host.inverse_indices[conjugator]
    return tuple(
        host.multiplication_table[host.multiplication_table[conjugator][image]][inverse]
        for image in mapping
    )


@dataclass(frozen=True, slots=True)
class HostConjugacyWitness:
    mapping: tuple[int, ...]
    conjugator_index: int

    def __post_init__(self) -> None:
        _index_tuple(self.mapping, "host conjugacy mapping", nonempty=True)
        if type(self.conjugator_index) is not int or self.conjugator_index < 0:
            raise ValueError("host conjugator index must be nonnegative")


@dataclass(frozen=True, slots=True)
class HostHomomorphismOrbit:
    host_id: HostID
    representative_mapping: tuple[int, ...]
    orbit_size: int
    kernel_elements: tuple[int, ...]
    skeleton_id: str
    conjugacy_witnesses: tuple[HostConjugacyWitness, ...]

    def __post_init__(self) -> None:
        if self.host_id not in {"O", "D6"}:
            raise ValueError("unknown homomorphism host")
        host = _host(self.host_id)
        mapping = _index_tuple(
            self.representative_mapping, "host representative mapping", nonempty=True
        )
        if any(item >= len(host.elements) for item in mapping):
            raise ValueError("host representative mapping leaves the certified host")
        if type(self.orbit_size) is not int or self.orbit_size <= 0:
            raise ValueError("host orbit size must be positive")
        kernel = _index_tuple(
            self.kernel_elements, "host orbit kernel", nonempty=True, sorted_unique=True
        )
        expected_kernel = tuple(
            index for index, image in enumerate(mapping) if image == host.identity_index
        )
        if kernel != expected_kernel:
            raise ValueError("host orbit kernel does not replay its representative")
        _digest(self.skeleton_id, "host orbit skeleton ID")
        if type(self.conjugacy_witnesses) is not tuple or any(
            not isinstance(item, HostConjugacyWitness)
            for item in self.conjugacy_witnesses
        ):
            raise TypeError("host orbit conjugacy witnesses must be a tuple of certificates")
        if len(self.conjugacy_witnesses) != self.orbit_size:
            raise ValueError("host orbit size does not equal its conjugacy witnesses")
        witnessed_mappings: list[tuple[int, ...]] = []
        for witness in self.conjugacy_witnesses:
            if (
                len(witness.mapping) != len(mapping)
                or any(item >= len(host.elements) for item in witness.mapping)
                or witness.conjugator_index >= len(host.elements)
            ):
                raise ValueError("host orbit witness leaves its certified universe")
            if witness.mapping != _conjugate_mapping(
                mapping, witness.conjugator_index, host
            ):
                raise ValueError("host orbit conjugacy witness does not replay")
            witnessed_mappings.append(witness.mapping)
        if len(set(witnessed_mappings)) != self.orbit_size or mapping not in witnessed_mappings:
            raise ValueError("host orbit witnesses do not enumerate one exact orbit")


@dataclass(frozen=True, slots=True)
class NormalSubgroupImageCertificate:
    kernel_elements: tuple[int, ...]
    quotient_order: int
    image_type_ids: tuple[str, ...]
    homomorphism_count_o: int
    homomorphism_count_d6: int

    def __post_init__(self) -> None:
        kernel = _index_tuple(
            self.kernel_elements,
            "normal-subgroup image kernel",
            nonempty=True,
            sorted_unique=True,
        )
        if kernel[0] != 0:
            raise ValueError("normal-subgroup image kernel must contain identity 0")
        if type(self.quotient_order) is not int or self.quotient_order <= 0:
            raise ValueError("normal-subgroup quotient order must be positive")
        if type(self.image_type_ids) is not tuple or any(
            item not in _PROJECTED_IMAGE_TYPES for item in self.image_type_ids
        ):
            raise ValueError("normal-subgroup image types are invalid")
        expected_order = tuple(
            type_id for type_id in STABILIZER_TYPE_IDS if type_id in self.image_type_ids
        )
        if self.image_type_ids != expected_order:
            raise ValueError("normal-subgroup image types are not canonical")
        for value in (self.homomorphism_count_o, self.homomorphism_count_d6):
            if type(value) is not int or value < 0:
                raise ValueError("normal-subgroup homomorphism counts must be nonnegative")
        if bool(self.image_type_ids) != bool(
            self.homomorphism_count_o + self.homomorphism_count_d6
        ):
            raise ValueError("normal-subgroup image types and homomorphism counts disagree")


@dataclass(frozen=True, slots=True)
class SkeletonExhaustivenessCertificate:
    source_table_digest: str
    host_ids: tuple[HostID, HostID]
    host_completeness_basis: str
    normal_subgroups: tuple[tuple[int, ...], ...]
    realized_normal_subgroups: tuple[tuple[int, ...], ...]
    normal_subgroup_images: tuple[NormalSubgroupImageCertificate, ...]
    homomorphism_count_o: int
    homomorphism_count_d6: int
    host_orbits: tuple[HostHomomorphismOrbit, ...]
    final_skeleton_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.source_table_digest, "exhaustiveness source table digest")
        if self.host_ids != ("O", "D6"):
            raise ValueError("exhaustiveness hosts must be exactly ordered as O,D6")
        if self.host_completeness_basis != _HOST_COMPLETENESS:
            raise ValueError("exhaustiveness host-completeness basis is not authoritative")
        if type(self.normal_subgroups) is not tuple or not self.normal_subgroups:
            raise ValueError("exhaustiveness requires normal subgroups")
        for subgroup in self.normal_subgroups:
            indices = _index_tuple(
                subgroup, "exhaustiveness normal subgroup", nonempty=True, sorted_unique=True
            )
            if indices[0] != 0:
                raise ValueError("exhaustiveness normal subgroup omits identity 0")
        if len(set(self.normal_subgroups)) != len(self.normal_subgroups):
            raise ValueError("exhaustiveness normal subgroups contain duplicates")
        if type(self.realized_normal_subgroups) is not tuple or any(
            item not in self.normal_subgroups for item in self.realized_normal_subgroups
        ):
            raise ValueError("realized normal subgroups are not a subset of all normals")
        if type(self.normal_subgroup_images) is not tuple or any(
            not isinstance(item, NormalSubgroupImageCertificate)
            for item in self.normal_subgroup_images
        ):
            raise TypeError("normal-subgroup images must be certificate tuples")
        if tuple(item.kernel_elements for item in self.normal_subgroup_images) != self.normal_subgroups:
            raise ValueError("normal-subgroup image certificates do not cover all normals in order")
        expected_realized = tuple(
            item.kernel_elements for item in self.normal_subgroup_images if item.image_type_ids
        )
        if self.realized_normal_subgroups != expected_realized:
            raise ValueError("realized normal-subgroup list does not replay images")
        for count in (self.homomorphism_count_o, self.homomorphism_count_d6):
            if type(count) is not int or count <= 0:
                raise ValueError("exhaustiveness homomorphism counts must be positive")
        if type(self.host_orbits) is not tuple or not self.host_orbits or any(
            not isinstance(item, HostHomomorphismOrbit) for item in self.host_orbits
        ):
            raise TypeError("exhaustiveness host orbits must be a nonempty certificate tuple")
        if sum(item.orbit_size for item in self.host_orbits if item.host_id == "O") != self.homomorphism_count_o:
            raise ValueError("O orbit sizes do not exhaust the marked homomorphisms")
        if sum(item.orbit_size for item in self.host_orbits if item.host_id == "D6") != self.homomorphism_count_d6:
            raise ValueError("D6 orbit sizes do not exhaust the marked homomorphisms")
        if type(self.final_skeleton_ids) is not tuple or not self.final_skeleton_ids:
            raise ValueError("exhaustiveness requires final skeleton IDs")
        if len(set(self.final_skeleton_ids)) != len(self.final_skeleton_ids):
            raise ValueError("exhaustiveness final skeleton IDs contain duplicates")
        for skeleton_id in self.final_skeleton_ids:
            _digest(skeleton_id, "exhaustiveness final skeleton ID")
        if any(item.skeleton_id not in self.final_skeleton_ids for item in self.host_orbits):
            raise ValueError("host orbit refers to an unknown final skeleton")


@dataclass(frozen=True, slots=True)
class CentralizerCertificate:
    image_type_id: str
    connected_model: str
    component_group_order: int
    image_digest: str

    def __post_init__(self) -> None:
        if self.image_type_id not in _PROJECTED_IMAGE_TYPES:
            raise ValueError("centralizer projected image type is invalid")
        expected = {
            "C1": ("SO3", 1),
            "C2": ("SO2-axis", 2),
            "C3": ("SO2-axis", 1),
            "C4": ("SO2-axis", 1),
            "C6": ("SO2-axis", 1),
            "C2xC2": ("trivial", 4),
            "D3": ("trivial", 2),
            "D4": ("trivial", 2),
            "D6": ("trivial", 2),
            "A4": ("trivial", 1),
            "S4": ("trivial", 1),
        }[self.image_type_id]
        if (self.connected_model, self.component_group_order) != expected:
            raise ValueError("centralizer connected/component model is not canonical")
        _digest(self.image_digest, "centralizer image digest")


RotationCertificate = CertifiedHostElement | CertifiedConjugator


@dataclass(frozen=True, slots=True)
class TimeConjugacyWitness:
    """Analytic orbit certificate inside one exact spatial centralizer."""

    witness_id: str
    image_type_id: str
    spatial_image_digest: str
    spatial_image: tuple[ExactSO3, ...]
    centralizer_model_digest: str
    analytic_family: str
    representative_matrix: ExactSO3
    centralizer_action_model: str
    invariant: tuple[int, ...]

    def __post_init__(self) -> None:
        _digest(self.witness_id, "time conjugacy witness ID")
        if self.image_type_id not in _PROJECTED_IMAGE_TYPES:
            raise ValueError("time conjugacy witness image type is invalid")
        _digest(self.spatial_image_digest, "time witness spatial image digest")
        _digest(self.centralizer_model_digest, "time witness centralizer model digest")
        if self.analytic_family not in {
            "identity",
            "all-half-turn-axes",
            "parallel-half-turn",
            "perpendicular-half-turn-axes",
            "discrete-centralizer-element",
        }:
            raise ValueError("time conjugacy witness has an unknown analytic family")
        if type(self.representative_matrix) is not ExactSO3 or not self.representative_matrix.is_rotation():
            raise TypeError("time conjugacy witness requires an exact rotation")
        if self.representative_matrix @ self.representative_matrix != identity_so3():
            raise ValueError("time conjugacy witness representative is not an involution")
        if self.centralizer_action_model not in {
            "fixed-identity",
            "SO3-transitive-unoriented-axes",
            "O2-transitive-perpendicular-unoriented-axes",
            "central-fixed-axis",
            "finite-abelian-singleton",
        }:
            raise ValueError("time conjugacy witness has an unknown action model")
        _index_tuple(self.invariant, "time conjugacy invariant", nonempty=True)
        expected_action = {
            "identity": "fixed-identity",
            "all-half-turn-axes": "SO3-transitive-unoriented-axes",
            "parallel-half-turn": "central-fixed-axis",
            "perpendicular-half-turn-axes": "O2-transitive-perpendicular-unoriented-axes",
            "discrete-centralizer-element": "finite-abelian-singleton",
        }[self.analytic_family]
        if self.centralizer_action_model != expected_action:
            raise ValueError("time witness analytic family/action model do not replay")
        allowed_families = {
            "C1": {"identity", "all-half-turn-axes"},
            "C2": {"identity", "parallel-half-turn", "perpendicular-half-turn-axes"},
            "C3": {"identity", "parallel-half-turn"},
            "C4": {"identity", "parallel-half-turn"},
            "C6": {"identity", "parallel-half-turn"},
            "C2xC2": {"identity", "discrete-centralizer-element"},
            "D3": {"identity", "parallel-half-turn"},
            "D4": {"identity", "parallel-half-turn"},
            "D6": {"identity", "parallel-half-turn"},
            "A4": {"identity"},
            "S4": {"identity"},
        }[self.image_type_id]
        if self.analytic_family not in allowed_families:
            raise ValueError("time witness analytic family is impossible for its centralizer")
        canonical_image = _canonical_rotation_image(self.spatial_image)
        derived_image_type = _image_type_from_matrices(canonical_image)
        if self.image_type_id != derived_image_type:
            raise ValueError("time witness image type does not replay exact spatial image")
        if _centralizer(self.image_type_id, canonical_image).image_digest != self.spatial_image_digest:
            raise ValueError("time witness spatial image digest does not replay exact matrices")
        expected_model_digest = _time_centralizer_model_digest(
            self.image_type_id,
            canonical_image,
        )
        if self.centralizer_model_digest != expected_model_digest:
            raise ValueError("time witness centralizer model digest does not replay exact image")
        _, expected_family, expected_action, expected_invariant = (
            _derived_time_orbit_semantics(
                self.image_type_id,
                canonical_image,
                self.representative_matrix,
            )
        )
        if (
            self.analytic_family != expected_family
            or self.centralizer_action_model != expected_action
            or self.invariant != expected_invariant
        ):
            raise ValueError(
                "time witness family, action, or invariant does not replay exact representative"
            )
        expected_id = _sha256(
            {
                "action_model": self.centralizer_action_model,
                "analytic_family": self.analytic_family,
                "centralizer_model_digest": self.centralizer_model_digest,
                "image_type_id": self.image_type_id,
                "invariant": list(self.invariant),
                "representative": self.representative_matrix.to_json(),
                "spatial_image_digest": self.spatial_image_digest,
            },
            "mathpsg-z2-time-conjugacy-witness-v1",
        )
        if self.witness_id != expected_id:
            raise ValueError("time conjugacy witness ID does not replay every semantic field")

    def replays(self, orbit: TimeInvolutionOrbit, spatial_image: object) -> bool:
        try:
            image_type, images, image_digest = _spatial_image_data(spatial_image)
        except (TypeError, ValueError):
            return False
        return (
            type(orbit) is TimeInvolutionOrbit
            and self.image_type_id == image_type
            and self.spatial_image_digest == image_digest
            and self.centralizer_model_digest == orbit.centralizer_model_digest
            and orbit.spatial_image_digest == image_digest
            and orbit.centralizer_model_digest
            == _time_centralizer_model_digest(image_type, images)
            and self.representative_matrix == orbit.representative.matrix
            and self.invariant == orbit.nonconjugacy_invariant
            and all(
                self.representative_matrix @ image == image @ self.representative_matrix
                for image in images
            )
        )


@dataclass(frozen=True, slots=True)
class TimeInvolutionOrbit:
    orbit_id: str
    spatial_image_digest: str
    representative: RotationCertificate
    centralizer_model_digest: str
    conjugacy_witnesses: tuple[TimeConjugacyWitness, ...]
    nonconjugacy_invariant: tuple[int, ...]
    time_label: str
    exhaustive_invariants: tuple[tuple[int, ...], ...]
    exhaustiveness_digest: str

    def __post_init__(self) -> None:
        _digest(self.orbit_id, "time involution orbit ID")
        _digest(self.spatial_image_digest, "time involution spatial image digest")
        _digest(self.centralizer_model_digest, "time involution centralizer model digest")
        _digest(self.exhaustiveness_digest, "time involution exhaustiveness digest")
        if type(self.representative) not in {CertifiedHostElement, CertifiedConjugator}:
            raise TypeError("time involution representative must be a certified exact rotation")
        lift_certified_rotation(self.representative)
        if self.representative.matrix @ self.representative.matrix != identity_so3():
            raise ValueError("time involution representative does not square to identity")
        if type(self.conjugacy_witnesses) is not tuple or not self.conjugacy_witnesses or any(
            type(item) is not TimeConjugacyWitness for item in self.conjugacy_witnesses
        ):
            raise TypeError("time involution orbit requires exact conjugacy witnesses")
        invariant = _index_tuple(
            self.nonconjugacy_invariant,
            "time involution nonconjugacy invariant",
            nonempty=True,
        )
        if any(item.invariant != invariant for item in self.conjugacy_witnesses):
            raise ValueError("time conjugacy witnesses do not bind the orbit invariant")
        if any(item.representative_matrix != self.representative.matrix for item in self.conjugacy_witnesses):
            raise ValueError("time conjugacy witnesses do not bind the orbit representative")
        if type(self.time_label) is not str or not self.time_label:
            raise ValueError("time involution orbit requires a stable label")
        if (
            type(self.exhaustive_invariants) is not tuple
            or not self.exhaustive_invariants
            or any(type(item) is not tuple for item in self.exhaustive_invariants)
        ):
            raise TypeError("time orbit exhaustiveness requires an exact invariant partition")
        exhaustive = tuple(
            _index_tuple(item, "time exhaustive invariant", nonempty=True)
            for item in self.exhaustive_invariants
        )
        if len(set(exhaustive)) != len(exhaustive) or invariant not in exhaustive:
            raise ValueError("time orbit invariant partition is incomplete or duplicated")
        image_type = self.conjugacy_witnesses[0].image_type_id
        witness_image = self.conjugacy_witnesses[0].spatial_image
        expected_model_digest = _time_centralizer_model_digest(image_type, witness_image)
        expected_label, _, _, expected_invariant = _derived_time_orbit_semantics(
            image_type,
            witness_image,
            self.representative.matrix,
        )
        if self.centralizer_model_digest != expected_model_digest:
            raise ValueError("time orbit centralizer model does not replay exact image")
        if invariant != expected_invariant or self.time_label != expected_label:
            raise ValueError("time orbit invariant or label does not replay exact representative")
        expected_partition = {
            "C1": ((0,), (1,)),
            "C2": ((0,), (1,), (2,)),
            "C3": ((0,), (1,)),
            "C4": ((0,), (1,)),
            "C6": ((0,), (1,)),
            "C2xC2": ((0,), (1,), (2,), (3,)),
            "D3": ((0,), (1,)),
            "D4": ((0,), (1,)),
            "D6": ((0,), (1,)),
            "A4": ((0,),),
            "S4": ((0,),),
        }[image_type]
        if exhaustive != expected_partition:
            raise ValueError("time orbit exhaustive invariant partition is not canonical")
        if any(
            witness.image_type_id != image_type
            or witness.spatial_image_digest != self.spatial_image_digest
            or witness.centralizer_model_digest != self.centralizer_model_digest
            for witness in self.conjugacy_witnesses
        ):
            raise ValueError("time witnesses do not bind the orbit centralizer")
        expected_exhaustiveness = _sha256(
            {
                "analytic_partition": [list(item) for item in exhaustive],
                "centralizer_model_digest": self.centralizer_model_digest,
                "image_type_id": image_type,
                "method": "analytic-centralizer-involution-partition-v1",
                "spatial_image_digest": self.spatial_image_digest,
            },
            "mathpsg-z2-time-orbit-exhaustiveness-v1",
        )
        if self.exhaustiveness_digest != expected_exhaustiveness:
            raise ValueError("time orbit exhaustiveness digest does not replay")
        expected_id = _sha256(
            {
                "centralizer_model_digest": self.centralizer_model_digest,
                "nonconjugacy_invariant": list(invariant),
                "representative": self.representative.matrix.to_json(),
                "spatial_image_digest": self.spatial_image_digest,
                "time_label": self.time_label,
                "exhaustiveness_digest": self.exhaustiveness_digest,
            },
            "mathpsg-z2-time-involution-orbit-v1",
        )
        if self.orbit_id != expected_id:
            raise ValueError("time involution orbit ID does not replay its exact data")


@dataclass(frozen=True, slots=True)
class CentralizerComponent:
    component_id: str
    full_graded_image_digest: str
    representative: RotationCertificate
    marking_shift: tuple[int, ...]
    domain_digest: str
    domain_dimension: int

    def __post_init__(self) -> None:
        _digest(self.component_id, "full-image centralizer component ID")
        _digest(self.full_graded_image_digest, "full graded image digest")
        _digest(self.domain_digest, "centralizer component domain digest")
        if type(self.domain_dimension) is not int or self.domain_dimension <= 0:
            raise ValueError("centralizer component domain dimension must be positive")
        if type(self.representative) not in {CertifiedHostElement, CertifiedConjugator}:
            raise TypeError("centralizer component representative must be certified")
        lift_certified_rotation(self.representative)
        if type(self.marking_shift) is not tuple or not self.marking_shift or any(
            type(bit) is not int or bit not in (0, 1) for bit in self.marking_shift
        ):
            raise ValueError("centralizer component marking shift must be a nonempty GF(2) tuple")
        if len(self.marking_shift) != self.domain_dimension:
            raise ValueError("centralizer component marking shift has the wrong domain dimension")
        expected_id = _sha256(
            {
                "full_graded_image_digest": self.full_graded_image_digest,
                "marking_shift": list(self.marking_shift),
                "representative": self.representative.matrix.to_json(),
                "domain_digest": self.domain_digest,
                "domain_dimension": self.domain_dimension,
            },
            "mathpsg-z2-full-centralizer-component-v1",
        )
        if self.component_id != expected_id:
            raise ValueError("centralizer component ID does not replay its exact data")


@dataclass(frozen=True, slots=True)
class MarkedConjugator:
    source_host_id: HostID
    target_host_id: HostID
    matrix: ExactSO3
    source_marked_images: tuple[ExactSO3, ...]
    target_marked_images: tuple[ExactSO3, ...]

    def __post_init__(self) -> None:
        if self.source_host_id not in {"O", "D6"} or self.target_host_id not in {"O", "D6"}:
            raise ValueError("marked conjugator host is invalid")
        if type(self.matrix) is not ExactSO3:
            raise TypeError("marked conjugator matrix must be ExactSO3")
        if not self.matrix.is_rotation():
            raise ValueError("marked conjugator must be an exact proper rotation")
        if (
            type(self.source_marked_images) is not tuple
            or type(self.target_marked_images) is not tuple
            or not self.source_marked_images
            or any(
                type(item) is not ExactSO3 or not item.is_rotation()
                for item in self.source_marked_images
            )
            or any(
                type(item) is not ExactSO3 or not item.is_rotation()
                for item in self.target_marked_images
            )
        ):
            raise TypeError("marked conjugator images must be nonempty ExactSO3 tuples")
        if len(self.source_marked_images) != len(self.target_marked_images):
            raise ValueError("marked conjugator images have different dimensions")
        for source, target in zip(self.source_marked_images, self.target_marked_images, strict=True):
            if self.matrix @ source @ self.matrix.transpose() != target:
                raise ValueError("marked conjugator does not replay elementwise")


@dataclass(frozen=True, slots=True)
class Z2LocalSkeleton:
    skeleton_id: str
    stabilizer_type_id: str
    so3_images: tuple[ExactSO3, ...]
    su2_lifts: tuple[ExactQuaternion, ...]
    defect_bits: MatrixGF2
    kernel_elements: tuple[int, ...]
    projected_image_order: int
    projected_image_type_id: str
    source_hosts: tuple[HostID, ...]
    cross_host_conjugators: tuple[MarkedConjugator, ...]
    centralizer: CentralizerCertificate
    exhaustiveness: SkeletonExhaustivenessCertificate
    source_multiplication_table: tuple[tuple[int, ...], ...]
    spatial_skeleton_id: str | None = None
    classification_label: str | None = None
    time_orbit: TimeInvolutionOrbit | None = None
    time_reversal_lift: ExactQuaternion | None = None
    time_square_bit: int | None = None
    kramers_tag: Literal["kramers", "non-kramers"] | None = None
    full_graded_so3_images: tuple[ExactSO3, ...] = ()
    full_graded_su2_lifts: tuple[ExactQuaternion, ...] = ()
    full_graded_multiplication_table: tuple[tuple[int, ...], ...] = ()
    full_graded_defect_bits: MatrixGF2 | None = None
    full_graded_image_order: int | None = None
    full_graded_image_digest: str | None = None
    full_graded_image_type_id: str | None = None
    defect_cohomology_label: str | None = None
    centralizer_components: tuple[CentralizerComponent, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.skeleton_id, "spatial skeleton ID")
        if self.stabilizer_type_id not in STABILIZER_TYPE_IDS:
            raise ValueError("spatial skeleton stabilizer type is invalid")
        if type(self.so3_images) is not tuple or not self.so3_images or any(
            type(item) is not ExactSO3 or not item.is_rotation() for item in self.so3_images
        ):
            raise ValueError("spatial skeleton SO(3) images must be exact rotations")
        if type(self.su2_lifts) is not tuple or len(self.su2_lifts) != len(self.so3_images) or any(
            type(item) is not ExactQuaternion or item.norm_squared() != ONE_Q23
            for item in self.su2_lifts
        ):
            raise ValueError("spatial skeleton SU(2) lifts must be exact unit quaternions")
        if any(
            lift.to_so3() != image
            for lift, image in zip(self.su2_lifts, self.so3_images, strict=True)
        ):
            raise ValueError("spatial skeleton SU(2) lifts do not project to its SO(3) images")
        if type(self.defect_bits) is not MatrixGF2 or self.defect_bits.shape != (
            len(self.so3_images), len(self.so3_images)
        ):
            raise ValueError("spatial skeleton defect matrix has the wrong shape")
        kernel = _index_tuple(
            self.kernel_elements, "spatial skeleton kernel", nonempty=True, sorted_unique=True
        )
        expected_kernel = tuple(
            index for index, image in enumerate(self.so3_images) if image == identity_so3()
        )
        if kernel != expected_kernel:
            raise ValueError("spatial skeleton kernel indices are invalid")
        if (
            type(self.projected_image_order) is not int
            or self.projected_image_order != len(set(self.so3_images))
        ):
            raise ValueError("spatial skeleton projected image order is invalid")
        if self.projected_image_type_id not in _PROJECTED_IMAGE_TYPES:
            raise ValueError("spatial skeleton projected image type is invalid")
        if type(self.source_hosts) is not tuple or not self.source_hosts:
            raise ValueError("spatial skeleton requires source hosts")
        expected_hosts = tuple(item for item in ("O", "D6") if item in self.source_hosts)
        if self.source_hosts != expected_hosts:
            raise ValueError("spatial skeleton source hosts are not canonical")
        if type(self.cross_host_conjugators) is not tuple or any(
            not isinstance(item, MarkedConjugator) for item in self.cross_host_conjugators
        ):
            raise TypeError("spatial skeleton marked conjugators must be a tuple")
        if any(
            item.source_host_id not in self.source_hosts
            or item.target_host_id not in self.source_hosts
            for item in self.cross_host_conjugators
        ):
            raise ValueError("marked conjugator host is absent from skeleton sources")
        if not isinstance(self.centralizer, CentralizerCertificate):
            raise TypeError("spatial skeleton centralizer certificate is invalid")
        if self.centralizer.image_type_id != self.projected_image_type_id:
            raise ValueError("spatial skeleton centralizer image type differs")
        if self.centralizer != _centralizer(
            self.projected_image_type_id, self.so3_images
        ):
            raise ValueError("spatial skeleton centralizer certificate does not replay image")
        if not isinstance(self.exhaustiveness, SkeletonExhaustivenessCertificate):
            raise TypeError("spatial skeleton exhaustiveness certificate is invalid")
        if self.skeleton_id not in self.exhaustiveness.final_skeleton_ids:
            if self.spatial_skeleton_id is None:
                raise ValueError("spatial skeleton ID is absent from exhaustiveness certificate")
        source_order = len(self.so3_images)
        if (
            type(self.source_multiplication_table) is not tuple
            or len(self.source_multiplication_table) != source_order
            or any(
                type(row) is not tuple or len(row) != source_order
                for row in self.source_multiplication_table
            )
            or any(
                type(value) is not int or not 0 <= value < source_order
                for row in self.source_multiplication_table
                for value in row
            )
        ):
            raise ValueError("local skeleton source multiplication table is invalid")
        if any(
            self.so3_images[self.source_multiplication_table[left][right]]
            != self.so3_images[left] @ self.so3_images[right]
            for left in range(source_order)
            for right in range(source_order)
        ):
            raise ValueError("local skeleton source table does not replay its SO(3) homomorphism")
        source_type, source_table_digest, _ = _source_table_binding(
            self.source_multiplication_table
        )
        if source_type != self.stabilizer_type_id:
            raise ValueError("local skeleton source table has the wrong abstract type")
        if self.exhaustiveness.source_table_digest != source_table_digest:
            raise ValueError("local skeleton source table differs from its exhaustive parent")
        expected_spatial_defect = _defect_from_multiplication(
            self.source_multiplication_table,
            self.su2_lifts,
        )
        if self.defect_bits != expected_spatial_defect:
            raise ValueError("local skeleton spatial defect does not replay its exact lifts")
        graded_values = (
            self.spatial_skeleton_id,
            self.classification_label,
            self.time_orbit,
            self.time_reversal_lift,
            self.time_square_bit,
            self.kramers_tag,
            self.full_graded_defect_bits,
            self.full_graded_image_order,
            self.full_graded_image_digest,
            self.full_graded_image_type_id,
            self.defect_cohomology_label,
        )
        if all(value is None for value in graded_values):
            if (
                self.full_graded_so3_images
                or self.full_graded_su2_lifts
                or self.full_graded_multiplication_table
                or self.centralizer_components
            ):
                raise ValueError("spatial skeleton contains partial graded metadata")
            return
        if any(value is None for value in graded_values):
            raise ValueError("graded skeleton metadata must be complete")
        assert self.spatial_skeleton_id is not None
        assert self.classification_label is not None
        assert self.time_orbit is not None
        assert self.time_reversal_lift is not None
        assert self.time_square_bit is not None
        assert self.kramers_tag is not None
        assert self.full_graded_defect_bits is not None
        assert self.full_graded_image_order is not None
        assert self.full_graded_image_digest is not None
        assert self.full_graded_image_type_id is not None
        assert self.defect_cohomology_label is not None
        _digest(self.spatial_skeleton_id, "graded parent spatial skeleton ID")
        if self.spatial_skeleton_id not in self.exhaustiveness.final_skeleton_ids:
            raise ValueError("graded parent is absent from spatial exhaustiveness certificate")
        if type(self.classification_label) is not str or "__time_" not in self.classification_label:
            raise ValueError("graded skeleton classification label is invalid")
        if type(self.time_orbit) is not TimeInvolutionOrbit:
            raise TypeError("graded skeleton time choice is not a TimeInvolutionOrbit")
        parent_image = _canonical_rotation_image(self.so3_images)
        parent_model_digest = _time_centralizer_model_digest(
            self.projected_image_type_id,
            parent_image,
        )
        if (
            self.time_orbit.spatial_image_digest != self.centralizer.image_digest
            or self.time_orbit.centralizer_model_digest != parent_model_digest
            or any(
                witness.image_type_id != self.projected_image_type_id
                or witness.spatial_image_digest != self.centralizer.image_digest
                or witness.centralizer_model_digest != parent_model_digest
                or _canonical_rotation_image(witness.spatial_image) != parent_image
                for witness in self.time_orbit.conjugacy_witnesses
            )
        ):
            raise ValueError("graded time orbit does not bind the inherited spatial parent image")
        if self.time_reversal_lift.norm_squared() != ONE_Q23 or (
            self.time_reversal_lift.to_so3() != self.time_orbit.representative.matrix
        ):
            raise ValueError("graded time lift does not project to its certified involution")
        time_square = self.time_reversal_lift * self.time_reversal_lift
        expected_square_bit = 0 if time_square == ONE_QUATERNION else 1
        if time_square not in (ONE_QUATERNION, -ONE_QUATERNION) or self.time_square_bit != expected_square_bit:
            raise ValueError("graded time-square bit does not replay its exact lift")
        if self.kramers_tag != ("kramers" if self.time_square_bit else "non-kramers"):
            raise ValueError("graded Kramers tag differs from the time-square sign")
        full_order = len(self.full_graded_so3_images)
        if full_order != 2 * len(self.so3_images) or len(self.full_graded_su2_lifts) != full_order:
            raise ValueError("graded direct-product image has the wrong source order")
        if any(type(item) is not ExactSO3 or not item.is_rotation() for item in self.full_graded_so3_images):
            raise ValueError("graded full image contains a non-exact rotation")
        if any(
            type(item) is not ExactQuaternion
            or item.norm_squared() != ONE_Q23
            or item.to_so3() != image
            for item, image in zip(
                self.full_graded_su2_lifts,
                self.full_graded_so3_images,
                strict=True,
            )
        ):
            raise ValueError("graded full lifts do not project exactly")
        if (
            len(self.full_graded_multiplication_table) != full_order
            or any(len(row) != full_order for row in self.full_graded_multiplication_table)
            or any(
                type(value) is not int or not 0 <= value < full_order
                for row in self.full_graded_multiplication_table
                for value in row
            )
        ):
            raise ValueError("graded multiplication table has the wrong shape")
        if self.full_graded_defect_bits.shape != (full_order, full_order):
            raise ValueError("graded defect matrix has the wrong shape")
        if self.full_graded_image_order != len(set(self.full_graded_so3_images)):
            raise ValueError("graded projected image order does not replay")
        _digest(self.full_graded_image_digest, "graded full image digest")
        _digest(self.defect_cohomology_label, "graded defect cohomology label")
        if self.full_graded_image_type_id not in _PROJECTED_IMAGE_TYPES:
            raise ValueError("graded full image type is invalid")
        if type(self.centralizer_components) is not tuple or not self.centralizer_components or any(
            type(item) is not CentralizerComponent for item in self.centralizer_components
        ):
            raise TypeError("graded skeleton requires separate full-image components")
        if any(
            item.full_graded_image_digest != self.full_graded_image_digest
            or len(item.marking_shift) != full_order
            for item in self.centralizer_components
        ):
            raise ValueError("graded centralizer component is not bound to the full image")
        expected_table = _direct_product_table(self.source_multiplication_table)
        if self.full_graded_multiplication_table != expected_table:
            raise ValueError("graded multiplication table is not the source direct product")
        if self.full_graded_so3_images[:source_order] != self.so3_images:
            raise ValueError("graded first SO(3) half does not equal the inherited spatial image")
        if self.full_graded_su2_lifts[:source_order] != self.su2_lifts:
            raise ValueError("graded first lift half does not equal the inherited spatial lifts")
        time_matrix = self.time_orbit.representative.matrix
        expected_second_images = tuple(image @ time_matrix for image in self.so3_images)
        expected_second_lifts = tuple(
            lift * self.time_reversal_lift for lift in self.su2_lifts
        )
        if self.full_graded_so3_images[source_order:] != expected_second_images:
            raise ValueError("graded time SO(3) half does not replay the spatial/time images")
        if self.full_graded_su2_lifts[source_order:] != expected_second_lifts:
            raise ValueError("graded time lift half does not replay the spatial/time lifts")
        if self.full_graded_defect_bits != _defect_from_multiplication(
            expected_table,
            self.full_graded_su2_lifts,
        ):
            raise ValueError("graded full defect does not replay its exact lifts")
        if any(
            self.full_graded_defect_bits[left][right] != self.defect_bits[left][right]
            for left in range(source_order)
            for right in range(source_order)
        ):
            raise ValueError("graded full defect does not contain the inherited spatial defect")
        if self.full_graded_image_digest != _full_image_digest(self.full_graded_so3_images):
            raise ValueError("graded full image digest does not replay")
        if self.full_graded_image_type_id != _image_type_from_matrices(
            self.full_graded_so3_images
        ):
            raise ValueError("graded full image type does not replay")
        expected_component_count = _centralizer(
            self.full_graded_image_type_id,
            _canonical_rotation_image(self.full_graded_so3_images),
        ).component_group_order
        if (
            len(self.centralizer_components) != expected_component_count
            or len({item.component_id for item in self.centralizer_components})
            != expected_component_count
        ):
            raise ValueError("graded full-image component enumeration is not exhaustive")
        expected_domain_digest = _component_domain_digest(
            self.full_graded_so3_images,
            self.full_graded_su2_lifts,
            self.full_graded_multiplication_table,
        )
        if any(
            item.domain_digest != expected_domain_digest
            or item.domain_dimension != full_order
            for item in self.centralizer_components
        ):
            raise ValueError("graded centralizer components do not bind the actual source domain")
        for component in self.centralizer_components:
            matrix = component.representative.matrix
            if not all(
                matrix @ image == image @ matrix
                for image in self.full_graded_so3_images
            ):
                raise ValueError(
                    "graded centralizer component representative does not centralize the full image"
                )
            if component.marking_shift != _component_marking_shift(
                component.representative,
                self.full_graded_su2_lifts,
            ):
                raise ValueError(
                    "graded centralizer component shift does not replay exact SU(2) conjugation"
                )
            if any(
                component.marking_shift[target]
                != component.marking_shift[left] ^ component.marking_shift[right]
                for left, row in enumerate(self.full_graded_multiplication_table)
                for right, target in enumerate(row)
            ):
                raise ValueError(
                    "graded centralizer component marking shift is not an exact character"
                )


@dataclass(frozen=True, slots=True)
class _HostOrbitPrototype:
    host_id: HostID
    representative_mapping: tuple[int, ...]
    kernel_elements: tuple[int, ...]
    witnesses: tuple[HostConjugacyWitness, ...]

    @property
    def orbit_size(self) -> int:
        return len(self.witnesses)


def _host_orbits(homomorphisms: tuple[tuple[int, ...], ...], host: FiniteRotationGroup) -> tuple[_HostOrbitPrototype, ...]:
    remaining = set(homomorphisms)
    result: list[_HostOrbitPrototype] = []
    while remaining:
        seed = min(remaining)
        seed_orbit = {_conjugate_mapping(seed, conjugator, host) for conjugator in range(len(host.elements))}
        representative = min(seed_orbit)
        members: dict[tuple[int, ...], int] = {}
        for conjugator in range(len(host.elements)):
            member = _conjugate_mapping(representative, conjugator, host)
            members.setdefault(member, conjugator)
        if not set(members).issubset(set(homomorphisms)):
            raise ArithmeticError("host conjugation escaped the exhaustive homomorphism set")
        remaining.difference_update(members)
        kernel = tuple(index for index, image in enumerate(representative) if image == host.identity_index)
        result.append(
            _HostOrbitPrototype(
                host_id=host.host_id,
                representative_mapping=representative,
                kernel_elements=kernel,
                witnesses=tuple(
                    HostConjugacyWitness(mapping, conjugator)
                    for mapping, conjugator in sorted(members.items())
                ),
            )
        )
    return tuple(sorted(result, key=lambda orbit: orbit.representative_mapping))


def _subgroup_generated(table: FiniteGroupTable, seed: set[int]) -> frozenset[int]:
    members = {table.identity_index, *seed}
    changed = True
    while changed:
        changed = False
        for left in tuple(members):
            for right in tuple(members):
                for value in (table.multiplication_table[left][right], table.inverse_indices[left]):
                    if value not in members:
                        members.add(value)
                        changed = True
    return frozenset(members)


def _normal_subgroups(table: FiniteGroupTable) -> tuple[tuple[int, ...], ...]:
    subgroups = {frozenset((table.identity_index,))}
    frontier = list(subgroups)
    while frontier:
        subgroup = frontier.pop()
        for element in range(len(table.element_order)):
            if element in subgroup:
                continue
            candidate = _subgroup_generated(table, set(subgroup) | {element})
            if candidate not in subgroups:
                subgroups.add(candidate)
                frontier.append(candidate)
    normal = tuple(
        tuple(sorted(subgroup))
        for subgroup in subgroups
        if all(
            table.multiplication_table[
                table.multiplication_table[group][member]
            ][table.inverse_indices[group]]
            in subgroup
            for group in range(len(table.element_order))
            for member in subgroup
        )
    )
    return tuple(sorted(normal, key=lambda subgroup: (len(subgroup), subgroup)))


def _image_type(host: FiniteRotationGroup, mapping: tuple[int, ...]) -> str:
    values_set = set(mapping)
    values = (host.identity_index,) + tuple(sorted(values_set - {host.identity_index}))
    index = {value: position for position, value in enumerate(values)}
    table = tuple(
        tuple(index[host.multiplication_table[left][right]] for right in values)
        for left in values
    )
    return identify_stabilizer_type(table).type_id


def _centralizer(image_type_id: str, images: tuple[ExactSO3, ...]) -> CentralizerCertificate:
    if image_type_id == "C1":
        connected, components = "SO3", 1
    elif image_type_id == "C2":
        connected, components = "SO2-axis", 2
    elif image_type_id in {"C3", "C4", "C6"}:
        connected, components = "SO2-axis", 1
    elif image_type_id == "C2xC2":
        connected, components = "trivial", 4
    elif image_type_id in {"D3", "D4", "D6"}:
        connected, components = "trivial", 2
    elif image_type_id in {"A4", "S4"}:
        connected, components = "trivial", 1
    else:
        raise ArithmeticError("noncrystallographic projected image type")
    image = tuple(sorted(set(images), key=lambda matrix: matrix.canonical_key))
    return CentralizerCertificate(
        image_type_id,
        connected,
        components,
        _sha256([matrix.to_json() for matrix in image], "mathpsg-z2-spatial-centralizer-v1"),
    )


Vector = tuple[Q23, Q23, Q23]


def _sqrt_fraction_field(value: Q23) -> Q23:
    rational = value.to_fraction()
    if rational < 0:
        raise ArithmeticError("negative squared norm")
    if rational == 0:
        return ZERO_Q23
    for square_free, radical in ((1, ONE_Q23), (2, SQRT2), (3, SQRT3), (6, SQRT6)):
        reduced = rational / square_free
        numerator = math.isqrt(reduced.numerator)
        denominator = math.isqrt(reduced.denominator)
        if numerator * numerator == reduced.numerator and denominator * denominator == reduced.denominator:
            return Q23.from_rational(Fraction(numerator, denominator)) * radical
    raise ArithmeticError("required exact square root lies outside Q(sqrt2,sqrt3)")


def _dot(left: Vector, right: Vector) -> Q23:
    return sum((a * b for a, b in zip(left, right, strict=True)), ZERO_Q23)


def _scale(value: Q23, vector: Vector) -> Vector:
    return tuple(value * entry for entry in vector)  # type: ignore[return-value]


def _subtract(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _cross(left: Vector, right: Vector) -> Vector:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Vector) -> Vector:
    norm = _sqrt_fraction_field(_dot(vector, vector))
    if not norm:
        raise ValueError("cannot normalize the zero vector")
    inverse = norm.inverse()
    return _scale(inverse, vector)


def _canonical_rotation_image(images: Sequence[ExactSO3]) -> tuple[ExactSO3, ...]:
    result = tuple(sorted(set(images), key=lambda matrix: matrix.canonical_key))
    if not result or identity_so3() not in result:
        raise ValueError("exact rotation image must contain identity")
    if any(type(item) is not ExactSO3 or not item.is_rotation() for item in result):
        raise TypeError("exact rotation image contains an invalid matrix")
    if any(left @ right not in result for left in result for right in result):
        raise ValueError("exact rotation image is not closed")
    return result


def _image_type_from_matrices(images: Sequence[ExactSO3]) -> str:
    image = _canonical_rotation_image(images)
    index = {matrix: position for position, matrix in enumerate(image)}
    table = tuple(tuple(index[left @ right] for right in image) for left in image)
    return identify_stabilizer_type(table).type_id


def _spatial_image_data(
    spatial_image: object,
) -> tuple[str, tuple[ExactSO3, ...], str]:
    if type(spatial_image) is FiniteRotationGroup:
        images = tuple(element.matrix for element in spatial_image.elements)
        image_type = identify_stabilizer_type(spatial_image.multiplication_table).type_id
        digest = _sha256(
            [matrix.to_json() for matrix in _canonical_rotation_image(images)],
            "mathpsg-z2-spatial-centralizer-v1",
        )
        return image_type, images, digest
    if type(spatial_image) is Z2LocalSkeleton:
        if spatial_image.time_orbit is not None:
            raise TypeError("time involutions require a spatial image, not a graded skeleton")
        return (
            spatial_image.projected_image_type_id,
            spatial_image.so3_images,
            spatial_image.centralizer.image_digest,
        )
    raise TypeError("time involutions require a FiniteRotationGroup or spatial Z2LocalSkeleton")


def _time_centralizer_model_digest(
    image_type: str,
    images: Sequence[ExactSO3],
) -> str:
    certificate = _centralizer(image_type, tuple(images))
    return _sha256(
        {
            "component_group_order": certificate.component_group_order,
            "connected_model": certificate.connected_model,
            "image_digest": certificate.image_digest,
            "image_type_id": certificate.image_type_id,
            "involution_orbit_method": "analytic-centralizer-classification-v1",
        },
        "mathpsg-z2-time-centralizer-model-v1",
    )


def _known_rotation_certificate(matrix: ExactSO3) -> CertifiedHostElement | None:
    for host in (octahedral_rotation_group(), dihedral_six_rotation_group()):
        for element in host.elements:
            if element.matrix == matrix:
                return element
    return None


def _quaternion_inverse(value: ExactQuaternion) -> ExactQuaternion:
    if value.norm_squared() != ONE_Q23:
        raise ValueError("quaternion inverse requires exact unit norm")
    return ExactQuaternion(value.scalar, -value.x, -value.y, -value.z)


def _certificate_for_rotation(
    matrix: ExactSO3,
    image: Sequence[ExactSO3],
    *,
    lift_witness: ExactQuaternion | None = None,
) -> RotationCertificate:
    known = _known_rotation_certificate(matrix)
    if known is not None:
        return known
    canonical = _canonical_rotation_image(image)
    if lift_witness is None:
        raise ArithmeticError("non-host exact rotation requires an explicit quaternion lift")
    return certify_conjugator(
        matrix,
        canonical,
        canonical,
        lift_witness=lift_witness,
    )


def _matrix_order(matrix: ExactSO3, bound: int = 24) -> int:
    value = identity_so3()
    for exponent in range(1, bound + 1):
        value = value @ matrix
        if value == identity_so3():
            return exponent
    raise ArithmeticError("exact crystallographic rotation order exceeds the analytic bound")


def _axis_from_matrix(matrix: ExactSO3) -> Vector:
    skew = (
        matrix.rows[2][1] - matrix.rows[1][2],
        matrix.rows[0][2] - matrix.rows[2][0],
        matrix.rows[1][0] - matrix.rows[0][1],
    )
    if _dot(skew, skew):
        return _normalize(skew)
    rows = tuple(
        tuple(
            matrix.rows[row][column] - (ONE_Q23 if row == column else ZERO_Q23)
            for column in range(3)
        )
        for row in range(3)
    )
    for first in range(3):
        for second in range(first + 1, 3):
            candidate = _cross(rows[first], rows[second])  # type: ignore[arg-type]
            if _dot(candidate, candidate):
                return _normalize(candidate)
    raise ArithmeticError("nonidentity exact rotation has no certified fixed axis")


def _axis_and_half_turn(images: Sequence[ExactSO3]) -> tuple[Vector, ExactSO3, ExactQuaternion]:
    nonidentity = tuple(matrix for matrix in set(images) if matrix != identity_so3())
    if not nonidentity:
        axis = _COORDINATE_AXES[2]
    else:
        selected = max(nonidentity, key=lambda matrix: (_matrix_order(matrix), matrix.canonical_key))
        axis = _axis_from_matrix(selected)
    half_turn_lift = ExactQuaternion(ZERO_Q23, *axis).canonicalized()
    return axis, half_turn_lift.to_so3(), half_turn_lift


def _perpendicular_half_turn(axis: Vector) -> tuple[ExactSO3, ExactQuaternion]:
    perpendicular = next(
        _subtract(candidate, _scale(_dot(candidate, axis), axis))
        for candidate in _COORDINATE_AXES
        if _dot(_subtract(candidate, _scale(_dot(candidate, axis), axis)), _subtract(candidate, _scale(_dot(candidate, axis), axis)))
    )
    normalized = _normalize(perpendicular)
    lift = ExactQuaternion(ZERO_Q23, *normalized).canonicalized()
    return lift.to_so3(), lift


def _derived_time_orbit_semantics(
    image_type: str,
    images: Sequence[ExactSO3],
    representative: ExactSO3,
) -> tuple[str, str, str, tuple[int, ...]]:
    """Derive the analytic orbit row from exact image and representative alone."""

    canonical_image = _canonical_rotation_image(images)
    if representative @ representative != identity_so3():
        raise ValueError("time representative is not an exact involution")
    if not all(
        representative @ image == image @ representative
        for image in canonical_image
    ):
        raise ValueError("time representative does not centralize its exact spatial image")
    identity = identity_so3()
    if image_type == "C2xC2":
        if representative not in canonical_image:
            raise ValueError("time representative is outside the exact D2 centralizer")
        position = canonical_image.index(representative)
        if representative == identity:
            return "identity", "identity", "fixed-identity", (position,)
        return (
            f"component_{position}",
            "discrete-centralizer-element",
            "finite-abelian-singleton",
            (position,),
        )
    if representative == identity:
        return "identity", "identity", "fixed-identity", (0,)
    _, parallel, _ = _axis_and_half_turn(canonical_image)
    if image_type == "C1":
        return (
            "half_turn",
            "all-half-turn-axes",
            "SO3-transitive-unoriented-axes",
            (1,),
        )
    if image_type == "C2":
        if representative == parallel:
            return "parallel", "parallel-half-turn", "central-fixed-axis", (1,)
        return (
            "perpendicular",
            "perpendicular-half-turn-axes",
            "O2-transitive-perpendicular-unoriented-axes",
            (2,),
        )
    if image_type in {"C3", "C4", "C6", "D3", "D4", "D6"}:
        if representative != parallel:
            raise ValueError("time representative is outside the exact cyclic centralizer orbit")
        return "parallel", "parallel-half-turn", "central-fixed-axis", (1,)
    if image_type in {"A4", "S4"}:
        raise ValueError("nonidentity time representative is impossible for this centralizer")
    raise ValueError(f"unsupported crystallographic spatial image {image_type}")


def _time_orbit(
    *,
    image_type: str,
    images: tuple[ExactSO3, ...],
    image_digest: str,
    centralizer_digest: str,
    matrix: ExactSO3,
    lift: ExactQuaternion,
    time_label: str,
    family: str,
    action_model: str,
    invariant: tuple[int, ...],
    exhaustive_invariants: tuple[tuple[int, ...], ...],
) -> TimeInvolutionOrbit:
    representative = _certificate_for_rotation(matrix, images, lift_witness=lift)
    witness_id = _sha256(
        {
            "action_model": action_model,
            "analytic_family": family,
            "centralizer_model_digest": centralizer_digest,
            "image_type_id": image_type,
            "invariant": list(invariant),
            "representative": matrix.to_json(),
            "spatial_image_digest": image_digest,
        },
        "mathpsg-z2-time-conjugacy-witness-v1",
    )
    witness = TimeConjugacyWitness(
        witness_id=witness_id,
        image_type_id=image_type,
        spatial_image_digest=image_digest,
        spatial_image=images,
        centralizer_model_digest=centralizer_digest,
        analytic_family=family,
        representative_matrix=matrix,
        centralizer_action_model=action_model,
        invariant=invariant,
    )
    exhaustiveness_digest = _sha256(
        {
            "analytic_partition": [list(item) for item in exhaustive_invariants],
            "centralizer_model_digest": centralizer_digest,
            "image_type_id": image_type,
            "method": "analytic-centralizer-involution-partition-v1",
            "spatial_image_digest": image_digest,
        },
        "mathpsg-z2-time-orbit-exhaustiveness-v1",
    )
    orbit_id = _sha256(
        {
            "centralizer_model_digest": centralizer_digest,
            "nonconjugacy_invariant": list(invariant),
            "representative": matrix.to_json(),
            "spatial_image_digest": image_digest,
            "time_label": time_label,
            "exhaustiveness_digest": exhaustiveness_digest,
        },
        "mathpsg-z2-time-involution-orbit-v1",
    )
    return TimeInvolutionOrbit(
        orbit_id=orbit_id,
        spatial_image_digest=image_digest,
        representative=representative,
        centralizer_model_digest=centralizer_digest,
        conjugacy_witnesses=(witness,),
        nonconjugacy_invariant=invariant,
        time_label=time_label,
        exhaustive_invariants=exhaustive_invariants,
        exhaustiveness_digest=exhaustiveness_digest,
    )


def time_involution_orbits(
    spatial_image: FiniteRotationGroup | Z2LocalSkeleton,
) -> tuple[TimeInvolutionOrbit, ...]:
    """Classify involutions analytically in the spatial-image centralizer."""

    image_type, marked_images, image_digest = _spatial_image_data(spatial_image)
    images = _canonical_rotation_image(marked_images)
    centralizer_digest = _time_centralizer_model_digest(image_type, images)
    identity = identity_so3()
    identity_lift = ONE_QUATERNION
    axis, parallel, parallel_lift = _axis_and_half_turn(images)
    perpendicular, perpendicular_lift = _perpendicular_half_turn(axis)
    candidates: list[
        tuple[ExactSO3, ExactQuaternion, str, str, str, tuple[int, ...]]
    ] = [
        (identity, identity_lift, "identity", "identity", "fixed-identity", (0,))
    ]
    if image_type == "C1":
        candidates.append(
            (
                parallel,
                parallel_lift,
                "half_turn",
                "all-half-turn-axes",
                "SO3-transitive-unoriented-axes",
                (1,),
            )
        )
    elif image_type == "C2":
        candidates.append(
            (
                parallel,
                parallel_lift,
                "parallel",
                "parallel-half-turn",
                "central-fixed-axis",
                (1,),
            )
        )
        candidates.append(
            (
                perpendicular,
                perpendicular_lift,
                "perpendicular",
                "perpendicular-half-turn-axes",
                "O2-transitive-perpendicular-unoriented-axes",
                (2,),
            )
        )
    elif image_type in {"C3", "C4", "C6", "D3", "D4", "D6"}:
        candidates.append(
            (
                parallel,
                parallel_lift,
                "parallel",
                "parallel-half-turn",
                "central-fixed-axis",
                (1,),
            )
        )
    elif image_type == "C2xC2":
        candidates = []
        for position, matrix in enumerate(images):
            certificate = _known_rotation_certificate(matrix)
            if certificate is None:
                raise ArithmeticError("D2 centralizer element escaped exact hosts")
            lift = lift_certified_rotation(certificate)[0]
            candidates.append(
                (
                    matrix,
                    lift,
                    "identity" if matrix == identity else f"component_{position}",
                    "identity" if matrix == identity else "discrete-centralizer-element",
                    "fixed-identity" if matrix == identity else "finite-abelian-singleton",
                    (position,),
                )
            )
    elif image_type not in {"A4", "S4"}:
        raise ValueError(f"unsupported crystallographic spatial image {image_type}")
    exhaustive_invariants = tuple(item[5] for item in candidates)
    if len(set(exhaustive_invariants)) != len(exhaustive_invariants):
        raise ArithmeticError("analytic time partition contains duplicate invariants")
    result = tuple(
        _time_orbit(
            image_type=image_type,
            images=images,
            image_digest=image_digest,
            centralizer_digest=centralizer_digest,
            matrix=matrix,
            lift=lift,
            time_label=label,
            family=family,
            action_model=action_model,
            invariant=invariant,
            exhaustive_invariants=exhaustive_invariants,
        )
        for matrix, lift, label, family, action_model, invariant in candidates
    )
    if tuple(item.nonconjugacy_invariant for item in result) != exhaustive_invariants:
        raise ArithmeticError("analytic time-involution partition does not replay")
    for orbit in result:
        if not all(orbit.representative.matrix @ image == image @ orbit.representative.matrix for image in images):
            raise ArithmeticError("analytic time representative does not centralize the spatial image")
        if not all(witness.replays(orbit, spatial_image) for witness in orbit.conjugacy_witnesses):
            raise ArithmeticError("analytic time-involution witness does not replay")
    return result


def _axis_options(host: FiniteRotationGroup, element: int) -> tuple[Vector, ...]:
    if element == host.identity_index:
        return ()
    lift = lift_certified_rotation(host.elements[element])[0]
    axis = _normalize((lift.x, lift.y, lift.z))
    return (axis, _scale(Q23.from_rational(-1), axis)) if host.element_orders[element] == 2 else (axis,)


def _frame(primary: Vector, secondary: Vector) -> ExactSO3 | None:
    perpendicular = _subtract(secondary, _scale(_dot(primary, secondary), primary))
    if not _dot(perpendicular, perpendicular):
        return None
    second = _normalize(perpendicular)
    third = _cross(primary, second)
    matrix = ExactSO3(
        tuple(
            tuple((primary, second, third)[column][row] for column in range(3))
            for row in range(3)
        )  # type: ignore[arg-type]
    )
    if not matrix.is_rotation():
        raise ArithmeticError("axis frame is not an exact proper rotation")
    return matrix


_COORDINATE_AXES: tuple[Vector, ...] = (
    (ONE_Q23, ZERO_Q23, ZERO_Q23),
    (ZERO_Q23, ONE_Q23, ZERO_Q23),
    (ZERO_Q23, ZERO_Q23, ONE_Q23),
)


def _marked_matrices(host: FiniteRotationGroup, mapping: tuple[int, ...]) -> tuple[ExactSO3, ...]:
    return tuple(host.elements[index].matrix for index in mapping)


def _find_marked_conjugator(
    source_host: FiniteRotationGroup,
    source_mapping: tuple[int, ...],
    target_host: FiniteRotationGroup,
    target_mapping: tuple[int, ...],
) -> MarkedConjugator:
    source_images = _marked_matrices(source_host, source_mapping)
    target_images = _marked_matrices(target_host, target_mapping)

    def accepts(matrix: ExactSO3) -> bool:
        return all(
            matrix @ source @ matrix.transpose() == target
            for source, target in zip(source_images, target_images, strict=True)
        )

    identity = identity_so3()
    if accepts(identity):
        return MarkedConjugator(source_host.host_id, target_host.host_id, identity, source_images, target_images)
    nonidentity = tuple(
        index
        for index, (source, target) in enumerate(zip(source_mapping, target_mapping, strict=True))
        if source != source_host.identity_index and target != target_host.identity_index
    )
    for primary_index in nonidentity:
        for source_primary in _axis_options(source_host, source_mapping[primary_index]):
            for target_primary in _axis_options(target_host, target_mapping[primary_index]):
                paired_secondaries: list[tuple[Vector, Vector]] = []
                for secondary_index in nonidentity:
                    for source_secondary in _axis_options(source_host, source_mapping[secondary_index]):
                        for target_secondary in _axis_options(target_host, target_mapping[secondary_index]):
                            paired_secondaries.append((source_secondary, target_secondary))
                paired_secondaries.extend(
                    (source_secondary, target_secondary)
                    for source_secondary in _COORDINATE_AXES
                    for target_secondary in _COORDINATE_AXES
                )
                for source_secondary, target_secondary in paired_secondaries:
                    source_frame = _frame(source_primary, source_secondary)
                    target_frame = _frame(target_primary, target_secondary)
                    if source_frame is None or target_frame is None:
                        continue
                    matrix = target_frame @ source_frame.transpose()
                    if accepts(matrix):
                        return MarkedConjugator(
                            source_host.host_id,
                            target_host.host_id,
                            matrix,
                            source_images,
                            target_images,
                        )
    raise ArithmeticError("exact marked SO(3) conjugator was not constructed")


def _canonical_lifts(host: FiniteRotationGroup, mapping: tuple[int, ...]) -> tuple[ExactQuaternion, ...]:
    return tuple(lift_certified_rotation(host.elements[index])[0] for index in mapping)


def _defect(table: FiniteGroupTable, lifts: tuple[ExactQuaternion, ...]) -> MatrixGF2:
    rows: list[tuple[int, ...]] = []
    for left in range(len(table.element_order)):
        row: list[int] = []
        for right in range(len(table.element_order)):
            product_lift = lifts[left] * lifts[right]
            expected = lifts[table.multiplication_table[left][right]]
            if product_lift == expected:
                row.append(0)
            elif product_lift == -expected:
                row.append(1)
            else:
                raise ArithmeticError("canonical lifts have a noncentral multiplication defect")
        rows.append(tuple(row))
    result = MatrixGF2(tuple(rows), column_count=len(table.element_order))
    identity = table.identity_index
    if any(result[identity][right] or result[right][identity] for right in range(len(table.element_order))):
        raise ArithmeticError("canonical lift defect is not normalized")
    for first, second, third in product(range(len(table.element_order)), repeat=3):
        if (
            result[first][second] ^ result[table.multiplication_table[first][second]][third]
            != result[second][third] ^ result[first][table.multiplication_table[second][third]]
        ):
            raise ArithmeticError("canonical lift defect fails an exact cocycle triple")
    return result


@dataclass(frozen=True, slots=True)
class _SkeletonPrototype:
    skeleton_id: str
    host: FiniteRotationGroup
    mapping: tuple[int, ...]
    image_type_id: str
    kernel_elements: tuple[int, ...]
    source_hosts: tuple[HostID, ...]
    conjugators: tuple[MarkedConjugator, ...]
    character_key: tuple[tuple[Fraction, ...], ...]


def _skeleton_id(type_id: str, table: FiniteGroupTable, matrices: tuple[ExactSO3, ...]) -> str:
    return _sha256(
        {
            "abstract_table_digest": table.table_digest,
            "marked_so3_character": [
                [str(value) for value in coefficients]
                for coefficients in _character_key(matrices)
            ],
            "stabilizer_type_id": type_id,
        },
        "mathpsg-z2-spatial-skeleton-v1",
    )


@lru_cache(maxsize=18)
def _enumerate_type(type_id: str) -> tuple[Z2LocalSkeleton, ...]:
    table = canonical_stabilizer_table(type_id)
    generators = canonical_generators(type_id)
    all_homomorphisms: dict[HostID, tuple[tuple[int, ...], ...]] = {}
    orbit_prototypes: list[_HostOrbitPrototype] = []
    for host_id in ("O", "D6"):
        host = _host(host_id)
        homomorphisms = _homomorphisms(table, generators, host)
        all_homomorphisms[host_id] = homomorphisms
        orbit_prototypes.extend(_host_orbits(homomorphisms, host))

    by_character: dict[tuple[tuple[Fraction, ...], ...], list[_HostOrbitPrototype]] = {}
    for orbit in orbit_prototypes:
        matrices = _marked_matrices(_host(orbit.host_id), orbit.representative_mapping)
        by_character.setdefault(_character_key(matrices), []).append(orbit)

    prototypes: list[_SkeletonPrototype] = []
    character_to_id: dict[tuple[tuple[Fraction, ...], ...], str] = {}
    for character, equivalent_orbits in by_character.items():
        selected = min(
            equivalent_orbits,
            key=lambda orbit: ((0 if orbit.host_id == "O" else 1), orbit.representative_mapping),
        )
        host = _host(selected.host_id)
        matrices = _marked_matrices(host, selected.representative_mapping)
        skeleton_id = _skeleton_id(type_id, table, matrices)
        character_to_id[character] = skeleton_id
        conjugators = tuple(
            _find_marked_conjugator(
                _host(orbit.host_id),
                orbit.representative_mapping,
                host,
                selected.representative_mapping,
            )
            for orbit in sorted(equivalent_orbits, key=lambda item: (item.host_id, item.representative_mapping))
            if orbit != selected
        )
        prototypes.append(
            _SkeletonPrototype(
                skeleton_id=skeleton_id,
                host=host,
                mapping=selected.representative_mapping,
                image_type_id=_image_type(host, selected.representative_mapping),
                kernel_elements=selected.kernel_elements,
                source_hosts=tuple(host_id for host_id in ("O", "D6") if any(orbit.host_id == host_id for orbit in equivalent_orbits)),
                conjugators=conjugators,
                character_key=character,
            )
        )
    prototypes.sort(key=lambda item: (len(set(item.mapping)), item.kernel_elements, item.skeleton_id))

    host_orbits = tuple(
        HostHomomorphismOrbit(
            host_id=orbit.host_id,
            representative_mapping=orbit.representative_mapping,
            orbit_size=orbit.orbit_size,
            kernel_elements=orbit.kernel_elements,
            skeleton_id=character_to_id[
                _character_key(_marked_matrices(_host(orbit.host_id), orbit.representative_mapping))
            ],
            conjugacy_witnesses=orbit.witnesses,
        )
        for orbit in sorted(orbit_prototypes, key=lambda item: (item.host_id, item.representative_mapping))
    )
    normals = _normal_subgroups(table)
    normal_images: list[NormalSubgroupImageCertificate] = []
    for normal in normals:
        image_types: set[str] = set()
        counts: dict[HostID, int] = {"O": 0, "D6": 0}
        for host_id in ("O", "D6"):
            host = _host(host_id)
            for mapping in all_homomorphisms[host_id]:
                kernel = tuple(index for index, image in enumerate(mapping) if image == host.identity_index)
                if kernel == normal:
                    counts[host_id] += 1
                    image_types.add(_image_type(host, mapping))
        normal_images.append(
            NormalSubgroupImageCertificate(
                kernel_elements=normal,
                quotient_order=len(table.element_order) // len(normal),
                image_type_ids=tuple(type_id for type_id in STABILIZER_TYPE_IDS if type_id in image_types),
                homomorphism_count_o=counts["O"],
                homomorphism_count_d6=counts["D6"],
            )
        )
    realized = tuple(certificate.kernel_elements for certificate in normal_images if certificate.image_type_ids)
    certificate = SkeletonExhaustivenessCertificate(
        source_table_digest=str(table.table_digest),
        host_ids=("O", "D6"),
        host_completeness_basis=_HOST_COMPLETENESS,
        normal_subgroups=normals,
        realized_normal_subgroups=realized,
        normal_subgroup_images=tuple(normal_images),
        homomorphism_count_o=len(all_homomorphisms["O"]),
        homomorphism_count_d6=len(all_homomorphisms["D6"]),
        host_orbits=host_orbits,
        final_skeleton_ids=tuple(item.skeleton_id for item in prototypes),
    )

    skeletons: list[Z2LocalSkeleton] = []
    for prototype in prototypes:
        matrices = _marked_matrices(prototype.host, prototype.mapping)
        lifts = _canonical_lifts(prototype.host, prototype.mapping)
        if lifts[table.identity_index] != ONE_QUATERNION:
            raise ArithmeticError("canonical identity lift is not +1")
        defect = _defect(table, lifts)
        skeletons.append(
            Z2LocalSkeleton(
                skeleton_id=prototype.skeleton_id,
                stabilizer_type_id=type_id,
                so3_images=matrices,
                su2_lifts=lifts,
                defect_bits=defect,
                kernel_elements=prototype.kernel_elements,
                projected_image_order=len(set(prototype.mapping)),
                projected_image_type_id=prototype.image_type_id,
                source_hosts=prototype.source_hosts,
                cross_host_conjugators=prototype.conjugators,
                centralizer=_centralizer(prototype.image_type_id, matrices),
                exhaustiveness=certificate,
                source_multiplication_table=table.multiplication_table,
            )
        )
    return tuple(skeletons)


def _normalize_source_table(group: object) -> tuple[str, FiniteGroupTable, tuple[int, ...]]:
    try:
        raw_table = tuple(tuple(row) for row in group.multiplication_table)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError("spatial skeleton enumeration requires a finite multiplication table") from error
    identified = identify_stabilizer_type(raw_table)
    canonical = canonical_stabilizer_table(identified.type_id)
    return identified.type_id, canonical, identified.canonical_to_literal


@lru_cache(maxsize=64)
def _source_table_binding(
    source_table: tuple[tuple[int, ...], ...],
) -> tuple[str, str, tuple[int, ...]]:
    identified = identify_stabilizer_type(source_table)
    canonical = canonical_stabilizer_table(identified.type_id)
    return (
        identified.type_id,
        str(canonical.table_digest),
        identified.canonical_to_literal,
    )


def _transport_skeleton(
    skeleton: Z2LocalSkeleton,
    canonical_to_literal: tuple[int, ...],
) -> Z2LocalSkeleton:
    if canonical_to_literal == tuple(range(len(canonical_to_literal))):
        return skeleton
    literal_to_canonical = tuple(
        next(canonical for canonical, literal in enumerate(canonical_to_literal) if literal == index)
        for index in range(len(canonical_to_literal))
    )
    images = tuple(skeleton.so3_images[literal_to_canonical[index]] for index in range(len(literal_to_canonical)))
    lifts = tuple(skeleton.su2_lifts[literal_to_canonical[index]] for index in range(len(literal_to_canonical)))
    defect = MatrixGF2(
        tuple(
            tuple(skeleton.defect_bits[literal_to_canonical[left]][literal_to_canonical[right]] for right in range(len(literal_to_canonical)))
            for left in range(len(literal_to_canonical))
        ),
        column_count=len(literal_to_canonical),
    )
    kernel = tuple(sorted(canonical_to_literal[index] for index in skeleton.kernel_elements))
    literal_table = tuple(
        tuple(
            canonical_to_literal[
                skeleton.source_multiplication_table[
                    literal_to_canonical[left]
                ][literal_to_canonical[right]]
            ]
            for right in range(len(literal_to_canonical))
        )
        for left in range(len(literal_to_canonical))
    )
    return replace(
        skeleton,
        so3_images=images,
        su2_lifts=lifts,
        defect_bits=defect,
        kernel_elements=kernel,
        source_multiplication_table=literal_table,
    )


def enumerate_spatial_z2_skeletons(group: object) -> tuple[Z2LocalSkeleton, ...]:
    type_id, _, canonical_to_literal = _normalize_source_table(group)
    return tuple(_transport_skeleton(skeleton, canonical_to_literal) for skeleton in _enumerate_type(type_id))


def _direct_product_table(
    source: FiniteGroupTable | tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    table = source.multiplication_table if type(source) is FiniteGroupTable else source
    order = len(table)
    return tuple(
        tuple(
            ((left // order) ^ (right // order)) * order
            + table[left % order][right % order]
            for right in range(2 * order)
        )
        for left in range(2 * order)
    )


@lru_cache(maxsize=2048)
def _defect_from_multiplication(
    multiplication: tuple[tuple[int, ...], ...],
    lifts: tuple[ExactQuaternion, ...],
) -> MatrixGF2:
    order = len(multiplication)
    rows: list[tuple[int, ...]] = []
    for left in range(order):
        row: list[int] = []
        for right in range(order):
            product_lift = lifts[left] * lifts[right]
            target_lift = lifts[multiplication[left][right]]
            if product_lift == target_lift:
                row.append(0)
            elif product_lift == -target_lift:
                row.append(1)
            else:
                raise ArithmeticError("graded lifts have a noncentral multiplication defect")
        rows.append(tuple(row))
    result = MatrixGF2(tuple(rows), column_count=order)
    identities = tuple(
        candidate
        for candidate in range(order)
        if all(
            multiplication[candidate][value] == value
            and multiplication[value][candidate] == value
            for value in range(order)
        )
    )
    if len(identities) != 1:
        raise ArithmeticError("graded multiplication table has no unique identity")
    identity = identities[0]
    if any(result[identity][value] or result[value][identity] for value in range(order)):
        raise ArithmeticError("graded lift defect is not normalized")
    for first, second, third in product(range(order), repeat=3):
        if (
            result[first][second]
            ^ result[multiplication[first][second]][third]
            != result[second][third]
            ^ result[first][multiplication[second][third]]
        ):
            raise ArithmeticError("graded lift defect fails an exact cocycle triple")
    return result


def _defect_cohomology_label(
    source_multiplication: tuple[tuple[int, ...], ...],
    defect: MatrixGF2,
) -> str:
    """Return the cocycle coset in canonical abstract-table coordinates."""

    source_order = len(source_multiplication)
    type_id, _, canonical_to_literal = _source_table_binding(source_multiplication)
    canonical_source = canonical_stabilizer_table(type_id).multiplication_table
    multiplication = _direct_product_table(canonical_source)
    order = len(multiplication)
    if defect.shape != (order, order):
        raise ValueError("graded defect has the wrong canonicalization dimension")
    full_canonical_to_literal = tuple(
        (index // source_order) * source_order
        + canonical_to_literal[index % source_order]
        for index in range(order)
    )
    canonical_defect = MatrixGF2(
        tuple(
            tuple(
                defect[full_canonical_to_literal[left]][full_canonical_to_literal[right]]
                for right in range(order)
            )
            for left in range(order)
        ),
        column_count=order,
    )
    identities = tuple(
        candidate
        for candidate in range(order)
        if all(
            multiplication[candidate][value] == value
            and multiplication[value][candidate] == value
            for value in range(order)
        )
    )
    if len(identities) != 1:
        raise ArithmeticError("canonical graded table has no unique identity")
    identity = identities[0]
    flattened = [bit for row in canonical_defect for bit in row]
    boundary_rows: list[tuple[int, ...]] = []
    for coordinate in range(order):
        if coordinate == identity:
            continue
        boundary_rows.append(
            tuple(
                int(left == coordinate)
                ^ int(right == coordinate)
                ^ int(multiplication[left][right] == coordinate)
                for left in range(order)
                for right in range(order)
            )
        )
    reduction = rref(MatrixGF2(tuple(boundary_rows), column_count=order * order))
    remainder = list(flattened)
    for row, pivot in enumerate(reduction.pivots):
        if remainder[pivot]:
            remainder = [
                left ^ right
                for left, right in zip(remainder, reduction.reduced[row], strict=True)
            ]
    return _sha256(
        {
            "multiplication_table": [list(row) for row in multiplication],
            "normalized_cocycle_coset": remainder,
        },
        "mathpsg-z2-local-defect-cohomology-v1",
    )


def _spatial_classification_label(spatial: Z2LocalSkeleton) -> str:
    if spatial.stabilizer_type_id == "S4":
        return {
            "C1": "trivial_C1",
            "C2": "sign_C2",
            "D3": "quotient_S3",
            "S4": "faithful_S4",
        }[spatial.projected_image_type_id]
    return (
        f"{spatial.projected_image_type_id}_from_{spatial.stabilizer_type_id}_"
        f"{spatial.skeleton_id.removeprefix('sha256:')[:12]}"
    )


def _full_image_digest(images: Sequence[ExactSO3]) -> str:
    return _sha256(
        [matrix.to_json() for matrix in _canonical_rotation_image(images)],
        "mathpsg-z2-full-graded-image-v1",
    )


def _centralizer_component_matrices(
    image_type: str,
    images: tuple[ExactSO3, ...],
) -> tuple[tuple[ExactSO3, ExactQuaternion], ...]:
    identity = (identity_so3(), ONE_QUATERNION)
    if image_type in {"C1", "C3", "C4", "C6", "A4", "S4"}:
        return (identity,)
    axis, parallel, parallel_lift = _axis_and_half_turn(images)
    if image_type == "C2":
        perpendicular, perpendicular_lift = _perpendicular_half_turn(axis)
        return (identity, (perpendicular, perpendicular_lift))
    if image_type == "C2xC2":
        result: list[tuple[ExactSO3, ExactQuaternion]] = []
        for matrix in _canonical_rotation_image(images):
            certificate = _known_rotation_certificate(matrix)
            if certificate is None:
                raise ArithmeticError("D2 component representative escaped exact hosts")
            result.append((matrix, lift_certified_rotation(certificate)[0]))
        return tuple(result)
    if image_type in {"D3", "D4", "D6"}:
        return (identity, (parallel, parallel_lift))
    raise ValueError(f"unsupported full graded image type {image_type}")


def _component_domain_digest(
    images: tuple[ExactSO3, ...],
    lifts: tuple[ExactQuaternion, ...],
    multiplication: tuple[tuple[int, ...], ...],
) -> str:
    return _sha256(
        {
            "images": [matrix.to_json() for matrix in images],
            "lifts": [lift.to_json() for lift in lifts],
            "multiplication_table": [list(row) for row in multiplication],
        },
        "mathpsg-z2-component-source-domain-v1",
    )


@lru_cache(maxsize=2048)
def _component_marking_shift(
    representative: RotationCertificate,
    lifts: tuple[ExactQuaternion, ...],
) -> tuple[int, ...]:
    conjugator_lift = lift_certified_rotation(representative)[0]
    inverse_lift = _quaternion_inverse(conjugator_lift)
    shift: list[int] = []
    for lift in lifts:
        conjugated = conjugator_lift * lift * inverse_lift
        if conjugated == lift:
            shift.append(0)
        elif conjugated == -lift:
            shift.append(1)
        else:
            raise ArithmeticError("SO(3)-centralizer lift has a noncentral SU(2) action")
    return tuple(shift)


def _centralizer_components_from_data(
    images: tuple[ExactSO3, ...],
    lifts: tuple[ExactQuaternion, ...],
    multiplication: tuple[tuple[int, ...], ...],
) -> tuple[CentralizerComponent, ...]:
    full_image = _canonical_rotation_image(images)
    image_type = _image_type_from_matrices(full_image)
    image_digest = _full_image_digest(full_image)
    domain_digest = _component_domain_digest(images, lifts, multiplication)
    domain_dimension = len(images)
    expected_count = _centralizer(image_type, full_image).component_group_order
    result: list[CentralizerComponent] = []
    for matrix, supplied_lift in _centralizer_component_matrices(image_type, full_image):
        if not all(matrix @ image == image @ matrix for image in full_image):
            raise ArithmeticError("full-image component representative does not centralize")
        representative = _certificate_for_rotation(
            matrix,
            full_image,
            lift_witness=supplied_lift,
        )
        marking_shift = _component_marking_shift(representative, lifts)
        if any(
            marking_shift[target]
            != marking_shift[left] ^ marking_shift[right]
            for left, row in enumerate(multiplication)
            for right, target in enumerate(row)
        ):
            raise ArithmeticError("component marking shift is not an exact local character")
        component_id = _sha256(
            {
                "full_graded_image_digest": image_digest,
                "marking_shift": list(marking_shift),
                "representative": matrix.to_json(),
                "domain_digest": domain_digest,
                "domain_dimension": domain_dimension,
            },
            "mathpsg-z2-full-centralizer-component-v1",
        )
        result.append(
            CentralizerComponent(
                component_id=component_id,
                full_graded_image_digest=image_digest,
                representative=representative,
                marking_shift=marking_shift,
                domain_digest=domain_digest,
                domain_dimension=domain_dimension,
            )
        )
    if len(result) != expected_count or len({item.component_id for item in result}) != expected_count:
        raise ArithmeticError("full-image centralizer components are not exhaustive")
    return tuple(result)


def centralizer_components(
    full_graded_image: FiniteRotationGroup | Z2LocalSkeleton,
) -> tuple[CentralizerComponent, ...]:
    """Compute only residual components of the already formed full graded image."""

    if type(full_graded_image) is FiniteRotationGroup:
        images = tuple(element.matrix for element in full_graded_image.elements)
        lifts = tuple(lift_certified_rotation(element)[0] for element in full_graded_image.elements)
        multiplication = full_graded_image.multiplication_table
        return _centralizer_components_from_data(images, lifts, multiplication)
    if type(full_graded_image) is not Z2LocalSkeleton or full_graded_image.time_orbit is None:
        raise TypeError("centralizer components require a full graded image, never a time orbit")
    replayed = _centralizer_components_from_data(
        full_graded_image.full_graded_so3_images,
        full_graded_image.full_graded_su2_lifts,
        full_graded_image.full_graded_multiplication_table,
    )
    if replayed != full_graded_image.centralizer_components:
        raise ValueError("stored full-image centralizer components do not replay")
    return replayed


@lru_cache(maxsize=None)
def enumerate_graded_z2_skeletons(
    spatial: Z2LocalSkeleton,
) -> tuple[Z2LocalSkeleton, ...]:
    if type(spatial) is not Z2LocalSkeleton or spatial.time_orbit is not None:
        raise TypeError("graded enumeration requires one spatial Z2LocalSkeleton")
    source_table = spatial.source_multiplication_table
    source_order = len(source_table)
    if len(spatial.so3_images) != source_order:
        raise ValueError("spatial skeleton order differs from its canonical source table")
    multiplication = _direct_product_table(source_table)
    spatial_label = _spatial_classification_label(spatial)
    result: list[Z2LocalSkeleton] = []
    for orbit in time_involution_orbits(spatial):
        time_matrix = orbit.representative.matrix
        time_lift = lift_certified_rotation(orbit.representative)[0]
        full_images = spatial.so3_images + tuple(
            image @ time_matrix for image in spatial.so3_images
        )
        full_lifts = spatial.su2_lifts + tuple(
            lift * time_lift for lift in spatial.su2_lifts
        )
        defect = _defect_from_multiplication(multiplication, full_lifts)
        image_type = _image_type_from_matrices(full_images)
        image_digest = _full_image_digest(full_images)
        cohomology_label = _defect_cohomology_label(source_table, defect)
        components = _centralizer_components_from_data(full_images, full_lifts, multiplication)
        classification_label = f"{spatial_label}__time_{orbit.time_label}"
        skeleton_id = _sha256(
            {
                "classification_label": classification_label,
                "defect_cohomology_label": cohomology_label,
                "full_graded_image_digest": image_digest,
                "spatial_skeleton_id": spatial.skeleton_id,
                "time_orbit_id": orbit.orbit_id,
            },
            "mathpsg-z2-graded-skeleton-v1",
        )
        time_square = time_lift * time_lift
        if time_square == ONE_QUATERNION:
            time_square_bit = 0
        elif time_square == -ONE_QUATERNION:
            time_square_bit = 1
        else:
            raise ArithmeticError("time lift square is not the central Z2 IGG")
        result.append(
            replace(
                spatial,
                skeleton_id=skeleton_id,
                spatial_skeleton_id=spatial.skeleton_id,
                classification_label=classification_label,
                time_orbit=orbit,
                time_reversal_lift=time_lift,
                time_square_bit=time_square_bit,
                kramers_tag="kramers" if time_square_bit else "non-kramers",
                full_graded_so3_images=full_images,
                full_graded_su2_lifts=full_lifts,
                full_graded_multiplication_table=multiplication,
                full_graded_defect_bits=defect,
                full_graded_image_order=len(set(full_images)),
                full_graded_image_digest=image_digest,
                full_graded_image_type_id=image_type,
                defect_cohomology_label=cohomology_label,
                centralizer_components=components,
            )
        )
    if len({item.skeleton_id for item in result}) != len(result):
        raise ArithmeticError("graded skeleton enumeration produced duplicate IDs")
    return tuple(result)


def verify_graded_z2_skeleton(
    skeleton: Z2LocalSkeleton,
    group: object,
) -> Z2LocalSkeleton:
    """Replay the complete direct-product table, lifts, labels, and components."""

    if type(skeleton) is not Z2LocalSkeleton or skeleton.time_orbit is None:
        raise TypeError("graded semantic verification requires a graded Z2LocalSkeleton")
    type_id, _, _ = _normalize_source_table(group)
    if skeleton.stabilizer_type_id != type_id:
        raise ValueError("semantic replay: graded stabilizer type differs from source")
    parents = tuple(
        item
        for item in enumerate_spatial_z2_skeletons(group)
        if item.skeleton_id == skeleton.spatial_skeleton_id
    )
    if len(parents) != 1:
        raise ValueError("semantic replay: graded parent spatial skeleton is absent or ambiguous")
    parent = parents[0]
    for name in (
        "stabilizer_type_id",
        "so3_images",
        "su2_lifts",
        "defect_bits",
        "kernel_elements",
        "projected_image_order",
        "projected_image_type_id",
        "source_hosts",
        "cross_host_conjugators",
        "centralizer",
        "exhaustiveness",
        "source_multiplication_table",
    ):
        if getattr(skeleton, name) != getattr(parent, name):
            raise ValueError(f"semantic replay: inherited spatial {name} differs from parent")
    authoritative = tuple(
        item
        for item in enumerate_graded_z2_skeletons(parent)
        if item.skeleton_id == skeleton.skeleton_id
    )
    if len(authoritative) != 1:
        raise ValueError("semantic replay: graded skeleton ID is absent from exact enumeration")
    expected = authoritative[0]
    for name in (
        "classification_label",
        "time_orbit",
        "time_reversal_lift",
        "time_square_bit",
        "kramers_tag",
        "full_graded_so3_images",
        "full_graded_su2_lifts",
        "full_graded_multiplication_table",
        "full_graded_defect_bits",
        "full_graded_image_order",
        "full_graded_image_digest",
        "full_graded_image_type_id",
        "defect_cohomology_label",
        "centralizer_components",
    ):
        if getattr(skeleton, name) != getattr(expected, name):
            raise ValueError(f"semantic replay: graded {name} differs from authoritative enumeration")
    centralizer_components(skeleton)
    return skeleton


def _graded_type_dependency_key(
    type_id: str,
) -> tuple[str, tuple[str, ...], str]:
    table = canonical_stabilizer_table(type_id)
    spatial_ids = tuple(item.skeleton_id for item in _enumerate_type(type_id))
    host_lift_digest = _sha256(
        [
            {
                "host_id": host.host_id,
                "table_witness_digest": host.table_witness_digest,
                "lifts": [
                    lift_certified_rotation(element)[0].to_json()
                    for element in host.elements
                ],
            }
            for host in (octahedral_rotation_group(), dihedral_six_rotation_group())
        ],
        "mathpsg-z2-graded-host-lift-dependencies-v1",
    )
    return str(table.table_digest), spatial_ids, host_lift_digest


@lru_cache(maxsize=18)
def _enumerate_graded_type_cached(
    type_id: str,
    source_table_digest: str,
    spatial_skeleton_ids: tuple[str, ...],
    host_lift_digest: str,
) -> tuple[Z2LocalSkeleton, ...]:
    if (
        source_table_digest,
        spatial_skeleton_ids,
        host_lift_digest,
    ) != _graded_type_dependency_key(type_id):
        raise ValueError("graded type cache dependencies do not replay canonical inputs")
    return tuple(
        graded
        for spatial in _enumerate_type(type_id)
        for graded in enumerate_graded_z2_skeletons(spatial)
    )


def _enumerate_graded_type(type_id: str) -> tuple[Z2LocalSkeleton, ...]:
    return _enumerate_graded_type_cached(
        type_id,
        *_graded_type_dependency_key(type_id),
    )


def change_lift_signs(
    skeleton: Z2LocalSkeleton,
    group: object,
    signs: Sequence[int],
) -> Z2LocalSkeleton:
    if type(skeleton) is not Z2LocalSkeleton or skeleton.time_orbit is not None:
        raise TypeError("lift-sign changes require a spatial, never graded, skeleton")
    _, table, canonical_to_literal = _normalize_source_table(group)
    if canonical_to_literal != tuple(range(len(canonical_to_literal))):
        raise ValueError("lift-sign changes currently require the canonical abstract table order")
    bits = tuple(signs)
    if len(bits) != len(table.element_order) or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("lift signs must be one GF(2) bit per group element")
    if bits[table.identity_index]:
        raise ValueError("normalized lift-sign cochain must vanish at the identity")
    lifts = tuple((-lift if bit else lift) for lift, bit in zip(skeleton.su2_lifts, bits, strict=True))
    defect = _defect(table, lifts)
    return replace(skeleton, su2_lifts=lifts, defect_bits=defect)


def verify_z2_local_skeleton(
    skeleton: Z2LocalSkeleton,
    group: object,
) -> Z2LocalSkeleton:
    """Replay every spatial, lift, cocycle, and exhaustiveness witness exactly."""

    if not isinstance(skeleton, Z2LocalSkeleton):
        raise TypeError("semantic Z2 verification requires a Z2LocalSkeleton")
    type_id, canonical_table, canonical_to_literal = _normalize_source_table(group)
    try:
        multiplication = tuple(tuple(row) for row in group.multiplication_table)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError("semantic Z2 verification requires a finite multiplication table") from error
    order = len(multiplication)
    if skeleton.stabilizer_type_id != type_id:
        raise ValueError("semantic replay: skeleton stabilizer type differs from source table")
    if skeleton.source_multiplication_table != multiplication:
        raise ValueError("semantic replay: skeleton source multiplication table differs")
    if len(skeleton.so3_images) != order or len(skeleton.su2_lifts) != order:
        raise ValueError("semantic replay: marked image dimensions differ from source table")
    literal_identity = canonical_to_literal[canonical_table.identity_index]
    if skeleton.so3_images[literal_identity] != identity_so3():
        raise ValueError("semantic replay: SO(3) identity image is not exact identity")
    for left, right in product(range(order), repeat=2):
        if (
            skeleton.so3_images[multiplication[left][right]]
            != skeleton.so3_images[left] @ skeleton.so3_images[right]
        ):
            raise ValueError("semantic replay: marked SO(3) images are not a full-table homomorphism")
    kernel = tuple(
        index for index, image in enumerate(skeleton.so3_images) if image == identity_so3()
    )
    if skeleton.kernel_elements != kernel:
        raise ValueError("semantic replay: projected kernel does not replay")
    if skeleton.projected_image_order != len(set(skeleton.so3_images)):
        raise ValueError("semantic replay: projected image order does not replay")

    canonical_images = tuple(
        skeleton.so3_images[canonical_to_literal[index]] for index in range(order)
    )
    expected_id = _skeleton_id(type_id, canonical_table, canonical_images)
    if skeleton.skeleton_id != expected_id:
        raise ValueError("semantic replay: stable skeleton ID does not bind the marked image")
    candidates = tuple(
        item
        for item in enumerate_spatial_z2_skeletons(group)
        if item.skeleton_id == skeleton.skeleton_id
    )
    if len(candidates) != 1:
        raise ValueError("semantic replay: stable skeleton ID is absent or ambiguous in exhaustive enumeration")
    authoritative = candidates[0]
    if skeleton.so3_images != authoritative.so3_images:
        raise ValueError("semantic replay: marked SO(3) representative is not canonical")
    for name in (
        "kernel_elements",
        "projected_image_order",
        "projected_image_type_id",
        "source_hosts",
        "cross_host_conjugators",
        "centralizer",
        "exhaustiveness",
    ):
        if getattr(skeleton, name) != getattr(authoritative, name):
            raise ValueError(f"semantic replay: {name} differs from exhaustive host classification")

    if skeleton.su2_lifts[literal_identity] != ONE_QUATERNION:
        raise ValueError("semantic replay: normalized identity lift must be +1")
    sign_bits: list[int] = []
    for lift, image, canonical_lift in zip(
        skeleton.su2_lifts,
        skeleton.so3_images,
        authoritative.su2_lifts,
        strict=True,
    ):
        if lift.norm_squared() != ONE_Q23 or lift.to_so3() != image:
            raise ValueError("semantic replay: SU(2) lift does not project exactly to SO(3)")
        if lift == canonical_lift:
            sign_bits.append(0)
        elif lift == -canonical_lift:
            sign_bits.append(1)
        else:
            raise ValueError("semantic replay: SU(2) lift is not a lift-sign cochain")
    if sign_bits[literal_identity]:
        raise ValueError("semantic replay: lift-sign coboundary is not normalized")
    for left, right in product(range(order), repeat=2):
        target = multiplication[left][right]
        expected_bit = (
            authoritative.defect_bits[left][right]
            ^ sign_bits[left]
            ^ sign_bits[right]
            ^ sign_bits[target]
        )
        product_lift = skeleton.su2_lifts[left] * skeleton.su2_lifts[right]
        target_lift = skeleton.su2_lifts[target]
        if product_lift == target_lift:
            actual_bit = 0
        elif product_lift == -target_lift:
            actual_bit = 1
        else:
            raise ValueError("semantic replay: SU(2) multiplication defect is not central")
        if skeleton.defect_bits[left][right] != actual_bit or actual_bit != expected_bit:
            raise ValueError("semantic replay: defect is not the exact lift-sign coboundary")
    if any(
        skeleton.defect_bits[literal_identity][index]
        or skeleton.defect_bits[index][literal_identity]
        for index in range(order)
    ):
        raise ValueError("semantic replay: defect cocycle is not normalized")
    for first, second, third in product(range(order), repeat=3):
        if (
            skeleton.defect_bits[first][second]
            ^ skeleton.defect_bits[multiplication[first][second]][third]
            != skeleton.defect_bits[second][third]
            ^ skeleton.defect_bits[first][multiplication[second][third]]
        ):
            raise ValueError("semantic replay: defect fails an exact cocycle triple")
    return skeleton


def _conjugator_mapping(value: MarkedConjugator) -> dict[str, Any]:
    return {
        "matrix": value.matrix.to_json(),
        "source_host_id": value.source_host_id,
        "source_marked_images": [matrix.to_json() for matrix in value.source_marked_images],
        "target_host_id": value.target_host_id,
        "target_marked_images": [matrix.to_json() for matrix in value.target_marked_images],
    }


def _rotation_certificate_mapping(value: RotationCertificate) -> dict[str, Any]:
    if type(value) is CertifiedHostElement:
        return {
            "certificate_type": "host-element",
            "element_index": value.element_index,
            "host_id": value.host_id,
            "matrix": value.matrix.to_json(),
            "table_witness_digest": value.table_witness_digest,
        }
    if type(value) is CertifiedConjugator:
        return {
            "certificate_type": "conjugator",
            "conjugacy_witness": list(value.conjugacy_witness),
            "lift_witness": value.lift_witness.to_json(),
            "matrix": value.matrix.to_json(),
            "source_image": [matrix.to_json() for matrix in value.source_image],
            "source_image_digest": value.source_image_digest,
            "target_image": [matrix.to_json() for matrix in value.target_image],
            "target_image_digest": value.target_image_digest,
        }
    raise TypeError("rotation certificate mapping requires an exact certificate")


def _time_orbit_mapping(value: TimeInvolutionOrbit) -> dict[str, Any]:
    return {
        "centralizer_model_digest": value.centralizer_model_digest,
        "conjugacy_witnesses": [
            {
                "analytic_family": witness.analytic_family,
                "centralizer_action_model": witness.centralizer_action_model,
                "centralizer_model_digest": witness.centralizer_model_digest,
                "image_type_id": witness.image_type_id,
                "invariant": list(witness.invariant),
                "representative_matrix": witness.representative_matrix.to_json(),
                "witness_id": witness.witness_id,
                "spatial_image_digest": witness.spatial_image_digest,
                "spatial_image": [matrix.to_json() for matrix in witness.spatial_image],
            }
            for witness in value.conjugacy_witnesses
        ],
        "nonconjugacy_invariant": list(value.nonconjugacy_invariant),
        "exhaustive_invariants": [list(item) for item in value.exhaustive_invariants],
        "exhaustiveness_digest": value.exhaustiveness_digest,
        "orbit_id": value.orbit_id,
        "representative": _rotation_certificate_mapping(value.representative),
        "spatial_image_digest": value.spatial_image_digest,
        "time_label": value.time_label,
    }


def _centralizer_component_mapping(value: CentralizerComponent) -> dict[str, Any]:
    return {
        "component_id": value.component_id,
        "domain_digest": value.domain_digest,
        "domain_dimension": value.domain_dimension,
        "full_graded_image_digest": value.full_graded_image_digest,
        "marking_shift": list(value.marking_shift),
        "representative": _rotation_certificate_mapping(value.representative),
    }


def _graded_skeleton_mapping(value: Z2LocalSkeleton) -> dict[str, Any]:
    if value.time_orbit is None or value.full_graded_defect_bits is None:
        raise TypeError("graded mapping requires a graded Z2LocalSkeleton")
    return {
        "centralizer_components": [
            _centralizer_component_mapping(item) for item in value.centralizer_components
        ],
        "classification_label": value.classification_label,
        "defect_cohomology_label": value.defect_cohomology_label,
        "full_graded_defect_bits": [list(row) for row in value.full_graded_defect_bits],
        "full_graded_image_digest": value.full_graded_image_digest,
        "full_graded_image_order": value.full_graded_image_order,
        "full_graded_image_type_id": value.full_graded_image_type_id,
        "full_graded_multiplication_table": [
            list(row) for row in value.full_graded_multiplication_table
        ],
        "full_graded_so3_images": [
            matrix.to_json() for matrix in value.full_graded_so3_images
        ],
        "full_graded_su2_lifts": [
            lift.to_json() for lift in value.full_graded_su2_lifts
        ],
        "kramers_tag": value.kramers_tag,
        "skeleton_id": value.skeleton_id,
        "spatial_skeleton_id": value.spatial_skeleton_id,
        "time_orbit": _time_orbit_mapping(value.time_orbit),
        "time_reversal_lift": value.time_reversal_lift.to_json()
        if value.time_reversal_lift is not None
        else None,
        "time_square_bit": value.time_square_bit,
    }


def _skeleton_mapping(value: Z2LocalSkeleton) -> dict[str, Any]:
    return {
        "centralizer": {
            "component_group_order": value.centralizer.component_group_order,
            "connected_model": value.centralizer.connected_model,
            "image_digest": value.centralizer.image_digest,
            "image_type_id": value.centralizer.image_type_id,
        },
        "cross_host_conjugators": [_conjugator_mapping(item) for item in value.cross_host_conjugators],
        "defect_bits": [list(row) for row in value.defect_bits],
        "kernel_elements": list(value.kernel_elements),
        "projected_image_order": value.projected_image_order,
        "projected_image_type_id": value.projected_image_type_id,
        "skeleton_id": value.skeleton_id,
        "so3_images": [matrix.to_json() for matrix in value.so3_images],
        "source_hosts": list(value.source_hosts),
        "stabilizer_type_id": value.stabilizer_type_id,
        "su2_lifts": [lift.to_json() for lift in value.su2_lifts],
    }


def _exhaustiveness_summary_mapping(value: SkeletonExhaustivenessCertificate) -> dict[str, Any]:
    return {
        "final_skeleton_ids": list(value.final_skeleton_ids),
        "homomorphism_count_d6": value.homomorphism_count_d6,
        "homomorphism_count_o": value.homomorphism_count_o,
        "host_completeness_basis": value.host_completeness_basis,
        "host_ids": list(value.host_ids),
        "host_orbit_count": len(value.host_orbits),
        "normal_subgroup_images": [
            {
                "homomorphism_count_d6": item.homomorphism_count_d6,
                "homomorphism_count_o": item.homomorphism_count_o,
                "image_type_ids": list(item.image_type_ids),
                "kernel_elements": list(item.kernel_elements),
                "quotient_order": item.quotient_order,
            }
            for item in value.normal_subgroup_images
        ],
        "normal_subgroups": [list(item) for item in value.normal_subgroups],
        "realized_normal_subgroups": [list(item) for item in value.realized_normal_subgroups],
        "source_table_digest": value.source_table_digest,
    }


def _library_bytes(records: Sequence[object] | None = None) -> bytes:
    rows: list[bytes] = []
    inventory = load_stabilizer_type_library() if records is None else tuple(records)
    for record in inventory:
        skeletons = _enumerate_type(record.type_id)
        certificate = skeletons[0].exhaustiveness
        rows.append(
            canonical_json(
                {
                    "exhaustiveness": _exhaustiveness_summary_mapping(certificate),
                    "record_type": "z2-spatial-exhaustiveness-v1",
                    "schema_version": 1,
                    "stabilizer_type_id": record.type_id,
                }
            )
            + b"\n"
        )
        for orbit_index, orbit in enumerate(certificate.host_orbits):
            rows.append(
                canonical_json(
                    {
                        "conjugacy_witnesses": [
                            {
                                "conjugator_index": witness.conjugator_index,
                                "mapping": list(witness.mapping),
                            }
                            for witness in orbit.conjugacy_witnesses
                        ],
                        "host_id": orbit.host_id,
                        "kernel_elements": list(orbit.kernel_elements),
                        "orbit_index": orbit_index,
                        "orbit_size": orbit.orbit_size,
                        "record_type": "z2-spatial-host-orbit-v1",
                        "representative_mapping": list(orbit.representative_mapping),
                        "schema_version": 1,
                        "skeleton_id": orbit.skeleton_id,
                        "stabilizer_type_id": record.type_id,
                    }
                )
                + b"\n"
            )
        for skeleton in skeletons:
            rows.append(
                canonical_json(
                    {
                        "record_type": "z2-spatial-skeleton-v1",
                        "schema_version": 1,
                        "skeleton": _skeleton_mapping(skeleton),
                        "stabilizer_type_id": record.type_id,
                    }
                )
                + b"\n"
            )
    return b"".join(rows)


def _graded_library_bytes(records: Sequence[object] | None = None) -> bytes:
    rows: list[bytes] = []
    inventory = load_stabilizer_type_library() if records is None else tuple(records)
    for record in inventory:
        spatial = _enumerate_type(record.type_id)
        graded = _enumerate_graded_type(record.type_id)
        children_by_parent: dict[str, list[Z2LocalSkeleton]] = {
            skeleton.skeleton_id: [] for skeleton in spatial
        }
        for skeleton in graded:
            assert skeleton.spatial_skeleton_id is not None
            children_by_parent[skeleton.spatial_skeleton_id].append(skeleton)
        graded_by_parent = tuple(
            (skeleton, tuple(children_by_parent[skeleton.skeleton_id]))
            for skeleton in spatial
        )
        rows.append(
            canonical_json(
                {
                    "final_graded_skeleton_ids": [item.skeleton_id for item in graded],
                    "record_type": "z2-graded-exhaustiveness-v1",
                    "schema_version": 1,
                    "spatial_branches": [
                        {
                            "graded_skeleton_ids": [item.skeleton_id for item in children],
                            "spatial_skeleton_id": parent.skeleton_id,
                            "time_orbit_count": len(children),
                        }
                        for parent, children in graded_by_parent
                    ],
                    "stabilizer_type_id": record.type_id,
                }
            )
            + b"\n"
        )
        for skeleton in graded:
            rows.append(
                canonical_json(
                    {
                        "record_type": "z2-graded-skeleton-v1",
                        "schema_version": 1,
                        "skeleton": _graded_skeleton_mapping(skeleton),
                        "stabilizer_type_id": record.type_id,
                    }
                )
                + b"\n"
            )
    return b"".join(rows)


def load_spatial_skeleton_library(
    library: Path | None = None,
    *,
    catalogue_atlas: Path | None = None,
) -> tuple[Z2LocalSkeleton, ...]:
    """Load only canonical, manifest-bound rows that replay the exhaustive engine."""

    source = None if library is None else Path(library)
    records = load_stabilizer_type_library(source, catalogue_atlas=catalogue_atlas)
    _, artifacts = _read_manifest(source)
    if "z2-spatial-skeletons.ndjson" not in artifacts:
        raise ValueError("spatial skeleton library is not bound by the manifest")
    data = artifacts["z2-spatial-skeletons.ndjson"]
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("spatial skeleton library is not canonical NDJSON")
    allowed_fields = {
        "z2-spatial-exhaustiveness-v1": {
            "exhaustiveness",
            "record_type",
            "schema_version",
            "stabilizer_type_id",
        },
        "z2-spatial-host-orbit-v1": {
            "conjugacy_witnesses",
            "host_id",
            "kernel_elements",
            "orbit_index",
            "orbit_size",
            "record_type",
            "representative_mapping",
            "schema_version",
            "skeleton_id",
            "stabilizer_type_id",
        },
        "z2-spatial-skeleton-v1": {
            "record_type",
            "schema_version",
            "skeleton",
            "stabilizer_type_id",
        },
    }
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        value = _strict_json(line[:-1])
        if not isinstance(value, Mapping) or canonical_json(value) + b"\n" != line:
            raise ValueError(
                f"spatial skeleton library row {line_number} is not canonical JSON"
            )
        record_type = value.get("record_type")
        if record_type not in allowed_fields or set(value) != allowed_fields[record_type]:
            raise ValueError(
                f"spatial skeleton library row {line_number} has invalid strict fields"
            )
        if value.get("schema_version") != 1 or value.get("stabilizer_type_id") not in STABILIZER_TYPE_IDS:
            raise ValueError(
                f"spatial skeleton library row {line_number} has invalid protocol bindings"
            )
    expected = _library_bytes(records)
    if data != expected:
        raise ValueError(
            "spatial skeleton library is not byte-identical to authoritative semantic replay"
        )
    skeletons = tuple(
        skeleton
        for record in records
        for skeleton in _enumerate_type(record.type_id)
    )
    for record in records:
        for skeleton in _enumerate_type(record.type_id):
            verify_z2_local_skeleton(skeleton, record.table)
    return skeletons


def load_graded_skeleton_library(
    library: Path | None = None,
    *,
    catalogue_atlas: Path | None = None,
) -> tuple[Z2LocalSkeleton, ...]:
    """Load only manifest-bound graded rows identical to authoritative replay."""

    source = None if library is None else Path(library)
    records = load_stabilizer_type_library(source, catalogue_atlas=catalogue_atlas)
    _, artifacts = _read_manifest(source)
    if "z2-graded-skeletons.ndjson" not in artifacts:
        raise ValueError("graded skeleton library is not bound by manifest coverage")
    data = artifacts["z2-graded-skeletons.ndjson"]
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("graded skeleton library is not canonical NDJSON")
    allowed_fields = {
        "z2-graded-exhaustiveness-v1": {
            "final_graded_skeleton_ids",
            "record_type",
            "schema_version",
            "spatial_branches",
            "stabilizer_type_id",
        },
        "z2-graded-skeleton-v1": {
            "record_type",
            "schema_version",
            "skeleton",
            "stabilizer_type_id",
        },
    }
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        value = _strict_json(line[:-1])
        if not isinstance(value, Mapping) or canonical_json(value) + b"\n" != line:
            raise ValueError(f"graded skeleton library row {line_number} is not canonical JSON")
        record_type = value.get("record_type")
        if record_type not in allowed_fields or set(value) != allowed_fields[record_type]:
            raise ValueError(f"graded skeleton library row {line_number} has invalid strict fields")
        if value.get("schema_version") != 1 or value.get("stabilizer_type_id") not in STABILIZER_TYPE_IDS:
            raise ValueError(f"graded skeleton library row {line_number} has invalid protocol bindings")
    expected = _graded_library_bytes(records)
    if data != expected:
        raise ValueError(
            "graded skeleton library is not byte-identical to authoritative semantic replay"
        )
    skeletons = tuple(
        skeleton
        for record in records
        for skeleton in _enumerate_graded_type(record.type_id)
    )
    for record in records:
        for skeleton in _enumerate_graded_type(record.type_id):
            verify_graded_z2_skeleton(skeleton, record.table)
    return skeletons


def write_spatial_skeleton_library(output: Path) -> None:
    manifest_target = output / "manifest.json"
    types_target = output / "types.ndjson"
    if manifest_target.exists() and types_target.exists():
        records = load_stabilizer_type_library(output)
    else:
        records = load_stabilizer_type_library()
    types_bytes = _types_bytes(records)
    skeleton_bytes = _library_bytes(records)
    graded_target = output / "z2-graded-skeletons.ndjson"
    graded_skeleton_bytes = (
        _safe_read_regular(graded_target, "graded skeleton artifact")
        if graded_target.exists() and graded_target.is_file() and not graded_target.is_symlink()
        else None
    )
    manifest = _manifest_mapping(
        types_bytes,
        records,
        skeleton_bytes=skeleton_bytes,
        graded_skeleton_bytes=graded_skeleton_bytes,
    )
    _write_atomic(types_target, types_bytes)
    _write_atomic(output / "z2-spatial-skeletons.ndjson", skeleton_bytes)
    _write_atomic(output / "manifest.json", canonical_json(manifest) + b"\n")


def write_graded_skeleton_library(output: Path) -> None:
    """Regenerate the complete graded artifact and its strict manifest binding."""

    manifest_target = output / "manifest.json"
    types_target = output / "types.ndjson"
    spatial_target = output / "z2-spatial-skeletons.ndjson"
    if manifest_target.exists() and types_target.exists() and spatial_target.exists():
        records = load_stabilizer_type_library(output)
        types_bytes = _safe_read_regular(types_target, "stabilizer type artifact")
        spatial_bytes = _safe_read_regular(spatial_target, "spatial skeleton artifact")
    else:
        records = load_stabilizer_type_library()
        types_bytes = _types_bytes(records)
        spatial_bytes = _library_bytes(records)
    graded_bytes = _graded_library_bytes(records)
    manifest = _manifest_mapping(
        types_bytes,
        records,
        skeleton_bytes=spatial_bytes,
        graded_skeleton_bytes=graded_bytes,
    )
    _write_atomic(types_target, types_bytes)
    _write_atomic(spatial_target, spatial_bytes)
    _write_atomic(output / "z2-graded-skeletons.ndjson", graded_bytes)
    _write_atomic(manifest_target, canonical_json(manifest) + b"\n")


__all__ = [
    "CentralizerCertificate",
    "CentralizerComponent",
    "HostConjugacyWitness",
    "HostHomomorphismOrbit",
    "MarkedConjugator",
    "NormalSubgroupImageCertificate",
    "SkeletonExhaustivenessCertificate",
    "TimeConjugacyWitness",
    "TimeInvolutionOrbit",
    "Z2LocalSkeleton",
    "change_lift_signs",
    "centralizer_components",
    "enumerate_graded_z2_skeletons",
    "enumerate_spatial_z2_skeletons",
    "load_graded_skeleton_library",
    "load_spatial_skeleton_library",
    "time_involution_orbits",
    "verify_z2_local_skeleton",
    "verify_graded_z2_skeleton",
    "write_graded_skeleton_library",
    "write_spatial_skeleton_library",
]
