r"""Certified finite affine :math:`\mathbb Z_2` relative classification.

``CertifiedCochainProblem`` intentionally contains only the Task-5 resolution
and inclusion certificates.  It does not contain local target choices or a
relative-cone source binding.  Consequently the production entry point in
this module accepts a Task-12-owned envelope assembled by
``make_certified_z2_problem``; accepting the Task-5 value alone would invent
authority that the serialized object does not carry.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import itertools
import json
import re
from typing import Sequence
import weakref

from .cochains import (
    _word_character,
    CertifiedCochainProblem,
    CharacterBasisCertificate,
    CochainComplex,
    CochainMap,
    FreeResolutionCertificate,
    InclusionChainMapCertificate,
    LauncherExecutionAttestation,
    SparseGroupRingMatrix,
    Task5VerificationAuthority,
    VerificationIssue,
    VerificationReport,
    make_cochain_complex,
    make_cochain_map,
    restrict_coefficient_character,
    verify_character_basis,
    verify_inclusion_chain_map,
    verify_resolution,
)
from .bar_evaluator import (
    BarResolutionEquivalence,
    SparseBarChain,
    verify_bar_resolution_equivalence,
)
from .classification_schema import (
    FrozenJSONArray,
    FrozenJSONObject,
    ObstructedBranch,
)
from .gf2 import (
    GF2AffineArrow,
    GF2AffineSolution,
    GF2Character,
    GF2Inconsistency,
    MatrixGF2,
    image_basis,
    kernel_basis,
    quotient_basis,
    solve_affine,
)
from .relative_complex import (
    RelativeMatrices,
    RelativeProblem,
    verify_relative_certificate,
)
from .z2_local import (
    CentralizerComponent,
    Z2LocalSkeleton,
    verify_graded_z2_skeleton,
    verify_z2_local_skeleton,
)


_PROTOCOL = b"mathpsg-z2-classifier-v1|"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")


class CertificateInvalidError(ValueError):
    """A typed hard failure at a Task-12 certificate boundary."""

    code = "certificate_invalid"


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


def _bits(value: Sequence[int], path: str, *, length: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected GF(2) vector")
    result = tuple(value)
    if any(type(bit) is not int or bit not in (0, 1) for bit in result):
        raise ValueError(f"{path}: expected GF(2) bits")
    if length is not None and len(result) != length:
        raise ValueError(f"{path}: vector has wrong dimension")
    return result


def _skeleton_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("skeleton_ids must be an ordered sequence")
    result = tuple(value)
    if not result:
        raise ValueError("skeleton_ids must be nonempty")
    for index, item in enumerate(result):
        if type(item) is not str or _IDENTIFIER_RE.fullmatch(item) is None:
            raise ValueError(f"skeleton_ids[{index}]: invalid identifier")
    return result


def _matvec(matrix: MatrixGF2, vector: Sequence[int]) -> tuple[int, ...]:
    point = _bits(vector, "vector", length=matrix.column_count)
    return tuple(
        sum(entry * bit for entry, bit in zip(row, point, strict=True)) & 1
        for row in matrix
    )


def _columns_matrix(vectors: Sequence[Sequence[int]], ambient: int) -> MatrixGF2:
    columns = tuple(_bits(vector, "column", length=ambient) for vector in vectors)
    return MatrixGF2(
        tuple(tuple(column[row] for column in columns) for row in range(ambient)),
        column_count=len(columns),
    )


def _xor(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("GF(2) vector dimensions differ")
    return tuple(a ^ b for a, b in zip(left, right, strict=True))


def _linear_combination(
    basis: Sequence[Sequence[int]], coefficients: Sequence[int], ambient: int
) -> tuple[int, ...]:
    vectors = tuple(_bits(vector, "basis vector", length=ambient) for vector in basis)
    scalars = _bits(coefficients, "basis coefficients", length=len(vectors))
    return tuple(
        sum(scalar * vector[row] for scalar, vector in zip(scalars, vectors, strict=True))
        & 1
        for row in range(ambient)
    )


def _arrow_mapping(arrow: GF2AffineArrow) -> dict[str, object]:
    return {
        "linear": [list(row) for row in arrow.linear],
        "linear_column_count": arrow.linear.column_count,
        "shift": list(arrow.shift),
    }


def _arrow_id(arrow: GF2AffineArrow) -> str:
    return _digest("residual-affine-arrow", _arrow_mapping(arrow))


def _sparse_mod2_transpose(matrix: SparseGroupRingMatrix) -> MatrixGF2:
    """Evaluate the trivial Z2 module and transpose chains to cochains."""

    dense = [[0] * matrix.column_count for _ in range(matrix.row_count)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] = sum(
            term.coefficient for term in entry.terms
        ) & 1
    return MatrixGF2(
        tuple(
            tuple(dense[row][column] for row in range(matrix.row_count))
            for column in range(matrix.column_count)
        ),
        column_count=matrix.row_count,
    )


def _gf2_character_size(resolution: FreeResolutionCertificate) -> int:
    if resolution.finite_group is not None:
        return len(resolution.finite_group.element_order)
    return len(
        resolution.affine_pcp_certificate.pcp_normal_form.relative_orders
    ) + int(resolution.group_id.endswith("+onsite-T"))


def _certified_gf2_complex(
    resolution: FreeResolutionCertificate,
) -> CochainComplex:
    return make_cochain_complex(
        authority_id=resolution.resolution_id,
        dimensions=tuple(len(degree) for degree in resolution.basis),
        differentials=tuple(
            _sparse_mod2_transpose(boundary)
            for boundary in resolution.boundaries
        ),
        coefficient_character=GF2Character(
            (0,) * _gf2_character_size(resolution)
        ),
    )


def _certified_gf2_restriction(
    inclusion: InclusionChainMapCertificate,
    *,
    instance_id: str,
    ambient: CochainComplex,
    local: CochainComplex,
) -> CochainMap:
    return make_cochain_map(
        instance_id=instance_id,
        source=ambient,
        target=local,
        maps=tuple(_sparse_mod2_transpose(matrix) for matrix in inclusion.maps),
    )


def _verified_release_bundle_context(
    release_bundle: object,
) -> tuple[object, Task5VerificationAuthority]:
    """Replay one exact Task-5 bundle without creating a serializable proxy."""

    from . import task5_release

    verified = task5_release.verify_task5_release_bundle(release_bundle)
    if verified is not release_bundle:
        raise CertificateInvalidError(
            "certificate_invalid: release verifier returned another bundle capability"
        )
    authority = task5_release.verify_task5_release_authority(
        verified.release_authority
    )
    if authority is not verified.release_authority.verification_authority:
        raise CertificateInvalidError(
            "certificate_invalid: bundle lost its exact Task-5 authority capability"
        )
    return verified, authority


def _make_release_gf2_complex(
    resolution: FreeResolutionCertificate,
    *,
    release_bundle: object,
    role: str,
) -> CochainComplex:
    if type(resolution) is not FreeResolutionCertificate:
        raise TypeError("resolution must be an exact FreeResolutionCertificate")
    bundle, authority = _verified_release_bundle_context(release_bundle)
    inclusion = bundle.inclusion
    if type(inclusion) is not InclusionChainMapCertificate:
        raise CertificateInvalidError(
            "certificate_invalid: release bundle lacks a typed inclusion"
        )
    if role == "ambient":
        expected = inclusion.target_resolution
    elif role == "local":
        expected = inclusion.source_resolution
    else:  # pragma: no cover - only the two public wrappers call this helper.
        raise ValueError("unsupported GF(2) resolution role")
    if resolution is not expected:
        raise CertificateInvalidError(
            f"certificate_invalid: {role} role requires the exact bundle "
            + ("target" if role == "ambient" else "source")
            + " resolution"
        )
    report = verify_resolution(resolution, authority)
    if not report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: {role} resolution: {report.issues[0].code}"
        )
    if resolution.group_id.endswith("+onsite-T") and (
        resolution.construction != "onsite-c2-direct-product-resolution"
        or resolution.parent_spatial_resolution_id is None
    ):
        raise CertificateInvalidError(
            "certificate_invalid: graded resolution lacks its exact parent spatial resolution"
        )
    return _certified_gf2_complex(resolution)


def make_certified_gf2_ambient_complex(
    resolution: FreeResolutionCertificate,
    *,
    release_bundle: object,
) -> CochainComplex:
    """Convert the exact signed target resolution to trivial GF(2) cochains."""

    return _make_release_gf2_complex(
        resolution,
        release_bundle=release_bundle,
        role="ambient",
    )


def make_certified_gf2_local_complex(
    resolution: FreeResolutionCertificate,
    *,
    release_bundle: object,
) -> CochainComplex:
    """Convert the exact signed inclusion source to trivial GF(2) cochains."""

    return _make_release_gf2_complex(
        resolution,
        release_bundle=release_bundle,
        role="local",
    )


def make_certified_gf2_restriction(
    inclusion: InclusionChainMapCertificate,
    *,
    instance_id: str,
    ambient: CochainComplex,
    local: CochainComplex,
    release_bundle: object,
) -> CochainMap:
    """Transpose one exact signed Task-5 chain map over the trivial GF(2) module."""

    if type(inclusion) is not InclusionChainMapCertificate:
        raise TypeError("inclusion must be an exact InclusionChainMapCertificate")
    if type(ambient) is not CochainComplex or type(local) is not CochainComplex:
        raise TypeError("ambient and local must be exact CochainComplex values")
    bundle, authority = _verified_release_bundle_context(release_bundle)
    if bundle.inclusion is not inclusion:
        raise CertificateInvalidError(
            "certificate_invalid: release bundle does not bind the exact inclusion"
        )
    report = verify_inclusion_chain_map(
        inclusion,
        authority,
        require_release=True,
        trusted_release_attestation=inclusion.launcher_attestation,
    )
    if not report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: inclusion chain map: {report.issues[0].code}"
        )
    expected_ambient = _certified_gf2_complex(inclusion.target_resolution)
    expected_local = _certified_gf2_complex(inclusion.source_resolution)
    if ambient != expected_ambient:
        raise CertificateInvalidError(
            "certificate_invalid: ambient coefficient complex is not the exact signed GF(2) conversion"
        )
    if local != expected_local:
        raise CertificateInvalidError(
            "certificate_invalid: local coefficient complex is not the exact signed GF(2) conversion"
        )
    restricted = restrict_coefficient_character(
        inclusion,
        ambient.coefficient_character,
        release_bundle=release_bundle,
    )
    if restricted != local.coefficient_character:
        raise CertificateInvalidError(
            "certificate_invalid: local coefficient character is not the exact restriction"
        )
    return _certified_gf2_restriction(
        inclusion,
        instance_id=instance_id,
        ambient=ambient,
        local=local,
    )


@dataclass(frozen=True, slots=True)
class CertifiedCentralizerAction:
    action_id: str
    relative_certificate_id: str
    instance_id: str
    skeleton_id: str
    component_id: str
    full_graded_image_digest: str
    component_domain_digest: str
    marking_shift: tuple[int, ...]
    local_coordinates: tuple[int, ...]
    raw_translation: tuple[int, ...]
    quotient_action: GF2AffineArrow
    coordinate_certificate_id: str
    diagnostic: bool

    def __post_init__(self) -> None:
        for name in (
            "action_id",
            "relative_certificate_id",
            "skeleton_id",
            "component_id",
            "full_graded_image_digest",
            "component_domain_digest",
            "coordinate_certificate_id",
        ):
            _require_digest(getattr(self, name), f"$CertifiedCentralizerAction.{name}")
        if type(self.instance_id) is not str or _IDENTIFIER_RE.fullmatch(self.instance_id) is None:
            raise ValueError("$CertifiedCentralizerAction.instance_id: invalid identifier")
        marking = _bits(self.marking_shift, "$CertifiedCentralizerAction.marking_shift")
        local = _bits(self.local_coordinates, "$CertifiedCentralizerAction.local_coordinates")
        raw = _bits(self.raw_translation, "$CertifiedCentralizerAction.raw_translation")
        if type(self.quotient_action) is not GF2AffineArrow:
            raise TypeError("$CertifiedCentralizerAction.quotient_action: invalid arrow")
        if type(self.diagnostic) is not bool:
            raise TypeError("$CertifiedCentralizerAction.diagnostic: expected boolean")
        core = {
            "component_domain_digest": self.component_domain_digest,
            "component_id": self.component_id,
            "coordinate_certificate_id": self.coordinate_certificate_id,
            "diagnostic": self.diagnostic,
            "full_graded_image_digest": self.full_graded_image_digest,
            "instance_id": self.instance_id,
            "local_coordinates": list(local),
            "marking_shift": list(marking),
            "quotient_action": _arrow_mapping(self.quotient_action),
            "raw_translation": list(raw),
            "relative_certificate_id": self.relative_certificate_id,
            "skeleton_id": self.skeleton_id,
        }
        if self.action_id != _digest("certified-centralizer-action", core):
            raise ValueError("$CertifiedCentralizerAction.action_id: payload digest differs")
        object.__setattr__(self, "marking_shift", marking)
        object.__setattr__(self, "local_coordinates", local)
        object.__setattr__(self, "raw_translation", raw)


@dataclass(frozen=True, slots=True)
class Z2CrossSkeletonArrow:
    arrow_id: str
    source_skeleton_ids: tuple[str, ...]
    target_skeleton_ids: tuple[str, ...]
    quotient_action: GF2AffineArrow
    conjugacy_witness_id: str

    def __post_init__(self) -> None:
        _require_digest(self.arrow_id, "$Z2CrossSkeletonArrow.arrow_id")
        source = _skeleton_ids(self.source_skeleton_ids)
        target = _skeleton_ids(self.target_skeleton_ids)
        if source == target:
            raise CertificateInvalidError(
                "certificate_invalid: a within-skeleton action is not a cross-skeleton boundary"
            )
        if len(source) != len(target):
            raise ValueError("$Z2CrossSkeletonArrow: source/target orbit counts differ")
        if type(self.quotient_action) is not GF2AffineArrow:
            raise TypeError("$Z2CrossSkeletonArrow.quotient_action: invalid arrow")
        _require_digest(
            self.conjugacy_witness_id,
            "$Z2CrossSkeletonArrow.conjugacy_witness_id",
        )
        core = {
            "conjugacy_witness_id": self.conjugacy_witness_id,
            "quotient_action": _arrow_mapping(self.quotient_action),
            "source_skeleton_ids": list(source),
            "target_skeleton_ids": list(target),
        }
        if self.arrow_id != _digest("cross-skeleton-arrow", core):
            raise ValueError("$Z2CrossSkeletonArrow.arrow_id: payload digest differs")
        object.__setattr__(self, "source_skeleton_ids", source)
        object.__setattr__(self, "target_skeleton_ids", target)


def make_cross_skeleton_arrow(
    *,
    source_skeleton_ids: Sequence[str],
    target_skeleton_ids: Sequence[str],
    quotient_action: GF2AffineArrow,
    conjugacy_witness_id: str,
) -> Z2CrossSkeletonArrow:
    source = _skeleton_ids(source_skeleton_ids)
    target = _skeleton_ids(target_skeleton_ids)
    if type(quotient_action) is not GF2AffineArrow:
        raise TypeError("quotient_action must be GF2AffineArrow")
    core = {
        "conjugacy_witness_id": conjugacy_witness_id,
        "quotient_action": _arrow_mapping(quotient_action),
        "source_skeleton_ids": list(source),
        "target_skeleton_ids": list(target),
    }
    return Z2CrossSkeletonArrow(
        _digest("cross-skeleton-arrow", core),
        source,
        target,
        quotient_action,
        conjugacy_witness_id,
    )


@dataclass(frozen=True, slots=True)
class Z2DefectCoordinateCertificate:
    certificate_id: str
    resolution_id: str
    bar_equivalence_id: str
    skeleton_id: str
    source_defect_digest: str
    coordinates: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "resolution_id",
            "bar_equivalence_id",
            "skeleton_id",
            "source_defect_digest",
        ):
            _require_digest(getattr(self, name), f"$Z2DefectCoordinateCertificate.{name}")
        coordinates = _bits(
            self.coordinates,
            "$Z2DefectCoordinateCertificate.coordinates",
        )
        core = {
            "bar_equivalence_id": self.bar_equivalence_id,
            "coordinates": list(coordinates),
            "resolution_id": self.resolution_id,
            "skeleton_id": self.skeleton_id,
            "source_defect_digest": self.source_defect_digest,
        }
        if self.certificate_id != _digest("z2-defect-coordinate", core):
            raise ValueError(
                "$Z2DefectCoordinateCertificate.certificate_id: coordinate payload digest differs"
            )
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, slots=True)
class Z2MarkingCoordinateCertificate:
    certificate_id: str
    resolution_id: str
    bar_equivalence_id: str
    finite_group_table_digest: str
    source_marking_shift: tuple[int, ...]
    coordinates: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "resolution_id",
            "bar_equivalence_id",
            "finite_group_table_digest",
        ):
            _require_digest(getattr(self, name), f"$Z2MarkingCoordinateCertificate.{name}")
        source = _bits(
            self.source_marking_shift,
            "$Z2MarkingCoordinateCertificate.source_marking_shift",
        )
        coordinates = _bits(
            self.coordinates,
            "$Z2MarkingCoordinateCertificate.coordinates",
        )
        core = {
            "bar_equivalence_id": self.bar_equivalence_id,
            "coordinates": list(coordinates),
            "finite_group_table_digest": self.finite_group_table_digest,
            "resolution_id": self.resolution_id,
            "source_marking_shift": list(source),
        }
        if self.certificate_id != _digest("z2-marking-coordinate", core):
            raise ValueError(
                "$Z2MarkingCoordinateCertificate.certificate_id: coordinate payload digest differs"
            )
        object.__setattr__(self, "source_marking_shift", source)
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, slots=True)
class _MultiplicationTableView:
    multiplication_table: tuple[tuple[int, ...], ...]


def _skeleton_defect_data(
    skeleton: Z2LocalSkeleton,
) -> tuple[tuple[tuple[int, ...], ...], MatrixGF2]:
    if type(skeleton) is not Z2LocalSkeleton:
        raise TypeError("Z2 coordinate conversion requires Z2LocalSkeleton")
    if skeleton.time_orbit is None:
        return skeleton.source_multiplication_table, skeleton.defect_bits
    if skeleton.full_graded_defect_bits is None:
        raise CertificateInvalidError(
            "certificate_invalid: graded skeleton lacks its full defect"
        )
    return skeleton.full_graded_multiplication_table, skeleton.full_graded_defect_bits


def _verify_skeleton_against_equivalence(
    skeleton: Z2LocalSkeleton,
    equivalence: BarResolutionEquivalence,
) -> None:
    table, _ = _skeleton_defect_data(skeleton)
    if equivalence.finite_group.multiplication_table != table:
        raise CertificateInvalidError(
            "certificate_invalid: bar finite table differs from the skeleton domain"
        )
    if skeleton.time_orbit is None:
        verify_z2_local_skeleton(skeleton, equivalence.finite_group)
    else:
        verify_graded_z2_skeleton(
            skeleton,
            _MultiplicationTableView(skeleton.source_multiplication_table),
        )


def _pullback_mod2(
    equivalence: BarResolutionEquivalence,
    *,
    degree: int,
    value: object,
) -> tuple[int, ...]:
    traces = {
        item.basis_id: item.image
        for item in equivalence.psi_on_basis
        if item.degree == degree
    }
    expected_basis = equivalence.resolution.basis[degree]
    if set(traces) != set(expected_basis):
        raise CertificateInvalidError(
            "certificate_invalid: verified bar psi lacks the complete requested degree"
        )
    element_index = {
        element: index
        for index, element in enumerate(equivalence.finite_group.element_order)
    }
    coordinates: list[int] = []
    for basis_id in expected_basis:
        chain = traces[basis_id]
        if not isinstance(chain, SparseBarChain) or chain.degree != degree:
            raise CertificateInvalidError(
                "certificate_invalid: bar psi trace has the wrong degree"
            )
        coordinate = 0
        for term in chain.terms:
            indices = tuple(element_index[element] for element in term.group_tuple)
            if degree == 1:
                cochain_value = value[indices[0]]  # type: ignore[index]
            elif degree == 2:
                cochain_value = value[indices[0]][indices[1]]  # type: ignore[index]
            else:
                raise ValueError("Z2 pullback supports degrees one and two")
            coordinate ^= (term.coefficient & 1) & cochain_value
        coordinates.append(coordinate)
    return tuple(coordinates)


def coordinate_z2_defect(
    skeleton: Z2LocalSkeleton,
    equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
) -> Z2DefectCoordinateCertificate:
    if type(equivalence) is not BarResolutionEquivalence:
        raise TypeError("coordinate_z2_defect requires BarResolutionEquivalence")
    report = verify_bar_resolution_equivalence(equivalence, authority)
    if not report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: bar equivalence: {report.issues[0].code}"
        )
    _verify_skeleton_against_equivalence(skeleton, equivalence)
    table, defect = _skeleton_defect_data(skeleton)
    source_digest = _digest(
        "z2-normalized-defect",
        {
            "defect": [list(row) for row in defect],
            "multiplication_table": [list(row) for row in table],
        },
    )
    coordinates = _pullback_mod2(
        equivalence,
        degree=2,
        value=defect,
    )
    core = {
        "bar_equivalence_id": equivalence.equivalence_id,
        "coordinates": list(coordinates),
        "resolution_id": equivalence.resolution_id,
        "skeleton_id": skeleton.skeleton_id,
        "source_defect_digest": source_digest,
    }
    return Z2DefectCoordinateCertificate(
        _digest("z2-defect-coordinate", core),
        equivalence.resolution_id,
        equivalence.equivalence_id,
        skeleton.skeleton_id,
        source_digest,
        coordinates,
    )


def verify_z2_defect_coordinates(
    certificate: Z2DefectCoordinateCertificate,
    skeleton: Z2LocalSkeleton,
    equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
) -> VerificationReport:
    issues: list[VerificationIssue] = []
    try:
        expected = coordinate_z2_defect(skeleton, equivalence, authority)
        if type(certificate) is not Z2DefectCoordinateCertificate or certificate != expected:
            issues.append(
                VerificationIssue(
                    "z2_defect_coordinate_mismatch",
                    "stored coordinates do not replay the verified skeleton and bar psi",
                )
            )
    except (TypeError, ValueError) as error:
        issues.append(VerificationIssue("z2_defect_coordinate_invalid", str(error)))
    return VerificationReport(not issues, tuple(issues), 1)


def coordinate_z2_marking_shift(
    marking_shift: Sequence[int],
    equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
) -> Z2MarkingCoordinateCertificate:
    if type(equivalence) is not BarResolutionEquivalence:
        raise TypeError("coordinate_z2_marking_shift requires BarResolutionEquivalence")
    report = verify_bar_resolution_equivalence(equivalence, authority)
    if not report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: bar equivalence: {report.issues[0].code}"
        )
    shift = _bits(
        marking_shift,
        "marking_shift",
        length=len(equivalence.finite_group.element_order),
    )
    table = equivalence.finite_group
    if shift[table.identity_index] or any(
        shift[table.multiplication_table[left][right]]
        != shift[left] ^ shift[right]
        for left in range(len(shift))
        for right in range(len(shift))
    ):
        raise CertificateInvalidError(
            "certificate_invalid: marking shift is not an exact table character"
        )
    coordinates = _pullback_mod2(
        equivalence,
        degree=1,
        value=shift,
    )
    core = {
        "bar_equivalence_id": equivalence.equivalence_id,
        "coordinates": list(coordinates),
        "finite_group_table_digest": table.table_digest,
        "resolution_id": equivalence.resolution_id,
        "source_marking_shift": list(shift),
    }
    return Z2MarkingCoordinateCertificate(
        _digest("z2-marking-coordinate", core),
        equivalence.resolution_id,
        equivalence.equivalence_id,
        str(table.table_digest),
        shift,
        coordinates,
    )


@dataclass(frozen=True, slots=True)
class H1UnmarkingCertificate:
    certificate_id: str
    relative_certificate_id: str
    character_basis_certificate_id: str
    ambient_c1_columns: tuple[int, ...]
    ambient_h1_coordinates: tuple[tuple[int, ...], ...]
    boundary_columns: tuple[tuple[int, ...], ...]
    diagnostic_boundary_columns: tuple[tuple[int, ...], ...]
    application_count: int

    def __post_init__(self) -> None:
        _require_digest(self.certificate_id, "$H1UnmarkingCertificate.certificate_id")
        _require_digest(
            self.relative_certificate_id,
            "$H1UnmarkingCertificate.relative_certificate_id",
        )
        _require_digest(
            self.character_basis_certificate_id,
            "$H1UnmarkingCertificate.character_basis_certificate_id",
        )
        columns = tuple(self.ambient_c1_columns)
        if any(type(column) is not int or column < 0 for column in columns):
            raise ValueError("$H1UnmarkingCertificate.ambient_c1_columns: invalid index")
        expected_columns = tuple(range(columns[0], columns[-1] + 1)) if columns else ()
        if columns != expected_columns:
            raise ValueError("$H1UnmarkingCertificate.ambient_c1_columns: expected contiguous columns")
        h1_coordinates = tuple(
            _bits(
                vector,
                "$H1UnmarkingCertificate.ambient_h1_coordinates",
                length=len(columns),
            )
            for vector in self.ambient_h1_coordinates
        )
        boundaries = tuple(
            _bits(vector, "$H1UnmarkingCertificate.boundary_columns")
            for vector in self.boundary_columns
        )
        diagnostic = tuple(
            _bits(vector, "$H1UnmarkingCertificate.diagnostic_boundary_columns")
            for vector in self.diagnostic_boundary_columns
        )
        if len(boundaries) != len(h1_coordinates) or diagnostic != boundaries:
            raise CertificateInvalidError(
                "certificate_invalid: H1 diagnostic columns differ from the global C_G^1 columns"
            )
        widths = {len(vector) for vector in boundaries + diagnostic}
        if len(widths) > 1:
            raise ValueError("$H1UnmarkingCertificate: boundary vector widths differ")
        if type(self.application_count) is not int or self.application_count != 1:
            raise CertificateInvalidError(
                "certificate_invalid: H1 unmarking must be applied exactly once"
            )
        core = {
            "ambient_c1_columns": list(columns),
            "ambient_h1_coordinates": [list(vector) for vector in h1_coordinates],
            "application_count": self.application_count,
            "boundary_columns": [list(vector) for vector in boundaries],
            "character_basis_certificate_id": self.character_basis_certificate_id,
            "diagnostic_boundary_columns": [list(vector) for vector in diagnostic],
            "relative_certificate_id": self.relative_certificate_id,
        }
        if self.certificate_id != _digest("h1-unmarking-certificate", core):
            raise ValueError("$H1UnmarkingCertificate.certificate_id: payload digest differs")
        object.__setattr__(self, "ambient_c1_columns", columns)
        object.__setattr__(self, "ambient_h1_coordinates", h1_coordinates)
        object.__setattr__(self, "boundary_columns", boundaries)
        object.__setattr__(self, "diagnostic_boundary_columns", diagnostic)


def certify_h1_unmarking(
    matrices: RelativeMatrices,
    *,
    character_basis_id: str,
    diagnostic_boundary_columns: Sequence[Sequence[int]],
    ambient_h1_coordinates: Sequence[Sequence[int]] | None = None,
) -> H1UnmarkingCertificate:
    if type(matrices) is not RelativeMatrices or type(matrices.B) is not MatrixGF2:
        raise TypeError("H1 unmarking requires certified GF(2) relative matrices")
    _require_digest(character_basis_id, "character_basis_id")
    start, stop = matrices.coordinate_blocks.ambient_slices[0]
    columns = tuple(range(start, stop))
    if ambient_h1_coordinates is None:
        h1_coordinates = tuple(
            tuple(int(row == column) for row in range(len(columns)))
            for column in range(len(columns))
        )
    else:
        h1_coordinates = tuple(
            _bits(vector, "ambient_h1_coordinates", length=len(columns))
            for vector in ambient_h1_coordinates
        )
    boundaries = tuple(
        tuple(
            sum(
                coefficient * matrices.B[row][column]
                for coefficient, column in zip(
                    coordinates,
                    columns,
                    strict=True,
                )
            )
            & 1
            for row in range(matrices.B.row_count)
        )
        for coordinates in h1_coordinates
    )
    diagnostic = tuple(tuple(vector) for vector in diagnostic_boundary_columns)
    core = {
        "ambient_c1_columns": list(columns),
        "ambient_h1_coordinates": [list(vector) for vector in h1_coordinates],
        "application_count": 1,
        "boundary_columns": [list(vector) for vector in boundaries],
        "character_basis_certificate_id": character_basis_id,
        "diagnostic_boundary_columns": [list(vector) for vector in diagnostic],
        "relative_certificate_id": matrices.certificate.certificate_id,
    }
    return H1UnmarkingCertificate(
        _digest("h1-unmarking-certificate", core),
        matrices.certificate.certificate_id,
        character_basis_id,
        columns,
        h1_coordinates,
        boundaries,
        diagnostic,
        1,
    )


def _resolution_h1_coordinates(
    character_basis: CharacterBasisCertificate,
    inclusion: InclusionChainMapCertificate,
) -> tuple[tuple[int, ...], ...]:
    traces = {
        item.basis_id: item.psi
        for item in inclusion.target_bar_equivalence.basis_traces
        if item.degree == 1
    }
    basis = inclusion.target_resolution.basis[1]
    first_boundary = inclusion.target_resolution.boundaries[0]
    boundary_by_column = {
        entry.column: entry.terms
        for entry in first_boundary.entries
        if entry.row == 0
    }
    if (
        not set(traces) <= set(basis)
        or first_boundary.row_count != 1
        or first_boundary.column_count != len(basis)
        or set(boundary_by_column) != set(range(len(basis)))
    ):
        raise CertificateInvalidError(
            "certificate_invalid: Task-5 H1 coordinates differ from the ambient degree-one basis"
        )
    result = tuple(
        tuple(
            sum(
                (term.coefficient & 1)
                * _word_character(
                    inclusion.target_resolution,
                    character,
                    term.element,
                )
                for term in boundary_by_column[column]
            )
            & 1
            for column in range(len(basis))
        )
        for character in character_basis.hom_basis
    )
    for character, coordinates in zip(
        character_basis.hom_basis,
        result,
        strict=True,
    ):
        for basis_id, trace in traces.items():
            traced = (
                sum(
                    (term.coefficient & 1)
                    * _word_character(
                        inclusion.target_resolution,
                        character,
                        term.group_tuple[0],
                    )
                    for term in trace
                )
                & 1
            )
            if traced != coordinates[basis.index(basis_id)]:
                raise CertificateInvalidError(
                    "certificate_invalid: Task-5 H1 basis disagrees with target psi"
                )
    return result


def _require_spatial_character_parent(
    character_basis: CharacterBasisCertificate,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
) -> None:
    graded = (
        character_basis.presentation_kind
        == "graded-direct-product-presentation"
    )
    if graded:
        if type(spatial_character_basis) is not CharacterBasisCertificate:
            raise CertificateInvalidError(
                "certificate_invalid: graded character basis lacks its exact spatial certificate"
            )
        if type(spatial_resolution) is not FreeResolutionCertificate:
            raise CertificateInvalidError(
                "certificate_invalid: graded character basis lacks its exact spatial resolution"
            )
    elif spatial_character_basis is not None or spatial_resolution is not None:
        raise CertificateInvalidError(
            "certificate_invalid: spatial parent authority is reserved to graded characters"
        )


def certify_task5_h1_unmarking(
    matrices: RelativeMatrices,
    *,
    character_basis: CharacterBasisCertificate,
    inclusions: Sequence[InclusionChainMapCertificate],
    orbit_bindings: Sequence[Z2OrbitBinding],
    authority: Task5VerificationAuthority,
    spatial_character_basis: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
) -> H1UnmarkingCertificate:
    """Replay the independent Task-5 H1 restriction map against ``B`` once."""

    if type(matrices) is not RelativeMatrices or type(matrices.B) is not MatrixGF2:
        raise TypeError("Task-5 H1 unmarking requires GF(2) relative matrices")
    if type(character_basis) is not CharacterBasisCertificate:
        raise TypeError("Task-5 H1 unmarking requires CharacterBasisCertificate")
    if type(authority) is not Task5VerificationAuthority:
        raise TypeError("Task-5 H1 unmarking requires Task5VerificationAuthority")
    _require_spatial_character_parent(
        character_basis,
        spatial_character_basis,
        spatial_resolution,
    )
    certified_inclusions = tuple(inclusions)
    bindings = tuple(orbit_bindings)
    if (
        not certified_inclusions
        or any(type(item) is not InclusionChainMapCertificate for item in certified_inclusions)
        or any(type(item) is not Z2OrbitBinding for item in bindings)
    ):
        raise TypeError("Task-5 H1 unmarking requires typed inclusions and orbit bindings")
    by_id = {item.inclusion_id: item for item in certified_inclusions}
    if len(by_id) != len(certified_inclusions):
        raise CertificateInvalidError("certificate_invalid: duplicate H1 inclusion ID")
    binding_instance_ids = tuple(binding.instance_id for binding in bindings)
    if (
        len(set(binding_instance_ids)) != len(binding_instance_ids)
        or set(binding_instance_ids)
        != set(matrices.coordinate_blocks.instance_ids)
    ):
        raise CertificateInvalidError(
            "certificate_invalid: H1 bindings do not cover one ordered orbit-instance tuple"
        )
    first = certified_inclusions[0]
    if any(item.target_resolution != first.target_resolution for item in certified_inclusions):
        raise CertificateInvalidError(
            "certificate_invalid: H1 inclusions do not share one ambient resolution"
        )
    character_report = verify_character_basis(
        character_basis,
        first.target_resolution,
        authority,
        spatial_certificate=spatial_character_basis,
        spatial_resolution=spatial_resolution,
    )
    if not character_report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: H1 character basis: {character_report.issues[0].code}"
        )
    ambient_coordinates = _resolution_h1_coordinates(character_basis, first)
    diagnostic = [[0] * matrices.B.row_count for _ in ambient_coordinates]
    for binding in bindings:
        inclusion = by_id.get(binding.inclusion_id)
        if inclusion is None:
            raise CertificateInvalidError(
                "certificate_invalid: H1 orbit binding lacks a Task-5 inclusion"
            )
        report = verify_inclusion_chain_map(
            inclusion,
            authority,
            require_release=False,
        )
        if not report.valid:
            raise CertificateInvalidError(
                f"certificate_invalid: H1 inclusion: {report.issues[0].code}"
            )
        if (
            binding.bar_equivalence.resolution_id
            != inclusion.source_resolution_id
            or binding.bar_equivalence.equivalence_id
            != inclusion.source_bar_equivalence_id
        ):
            raise CertificateInvalidError(
                "certificate_invalid: H1 local bar trace differs from its inclusion"
            )
        bar_report = verify_bar_resolution_equivalence(
            binding.bar_equivalence,
            authority,
        )
        if not bar_report.valid:
            raise CertificateInvalidError(
                f"certificate_invalid: H1 local bar trace: {bar_report.issues[0].code}"
            )
        orbit_index = matrices.coordinate_blocks.instance_ids.index(binding.instance_id)
        start, stop = matrices.coordinate_blocks.local_slices[1][orbit_index]
        for character_index, character in enumerate(character_basis.hom_basis):
            restricted_values = tuple(
                _word_character(
                    inclusion.target_resolution,
                    character,
                    image,
                )
                for image in inclusion.source_element_images
            )
            local_coordinates = _pullback_mod2(
                binding.bar_equivalence,
                degree=1,
                value=restricted_values,
            )
            if len(local_coordinates) != stop - start:
                raise CertificateInvalidError(
                    "certificate_invalid: H1 local coordinates differ from the relative slice"
                )
            diagnostic[character_index][start:stop] = local_coordinates
    return certify_h1_unmarking(
        matrices,
        character_basis_id=character_basis.certificate_id,
        ambient_h1_coordinates=ambient_coordinates,
        diagnostic_boundary_columns=tuple(tuple(vector) for vector in diagnostic),
    )


def _identity(size: int) -> MatrixGF2:
    return MatrixGF2(
        tuple(
            tuple(int(row == column) for column in range(size))
            for row in range(size)
        ),
        column_count=size,
    )


def _quotient_translation(
    matrices: RelativeMatrices,
    raw_translation: tuple[int, ...],
) -> GF2AffineArrow:
    solved = solve_affine(matrices.D, tuple(matrices.offset))
    if not isinstance(solved, GF2AffineSolution):
        raise CertificateInvalidError(
            "certificate_invalid: an obstructed branch has no centralizer action"
        )
    if any(_matvec(matrices.D, raw_translation)):
        raise CertificateInvalidError(
            "certificate_invalid: component marking translation is not a D-cycle"
        )
    ambient = matrices.D.column_count
    boundaries = image_basis(matrices.B)
    quotient = quotient_basis(
        _columns_matrix(solved.kernel_basis, ambient),
        matrices.B,
    )
    decomposition = boundaries + quotient.representatives
    coordinates = solve_affine(
        _columns_matrix(decomposition, ambient),
        raw_translation,
    )
    if not isinstance(coordinates, GF2AffineSolution):
        raise CertificateInvalidError(
            "certificate_invalid: component translation leaves the homogeneous solution space"
        )
    shift = coordinates.basepoint[len(boundaries) :]
    return GF2AffineArrow(_identity(quotient.dimension), shift)


def _certify_centralizer_action_diagnostic(
    matrices: RelativeMatrices,
    *,
    skeleton_ids: Sequence[str],
    instance_id: str,
    skeleton: Z2LocalSkeleton,
    component: CentralizerComponent,
    local_coordinates: Sequence[int],
) -> CertifiedCentralizerAction:
    """Explicit normalized-bar diagnostic; production uses a Task-5 trace."""

    if type(matrices) is not RelativeMatrices or type(matrices.D) is not MatrixGF2:
        raise TypeError("centralizer action requires GF(2) relative matrices")
    skeletons = _skeleton_ids(skeleton_ids)
    if type(skeleton) is not Z2LocalSkeleton or skeleton.time_orbit is None:
        raise TypeError("centralizer action requires a graded Z2LocalSkeleton")
    if type(component) is not CentralizerComponent:
        raise TypeError("centralizer action requires CentralizerComponent")
    try:
        orbit_index = matrices.coordinate_blocks.instance_ids.index(instance_id)
    except ValueError as error:
        raise CertificateInvalidError(
            "certificate_invalid: centralizer instance is absent from the relative problem"
        ) from error
    if orbit_index >= len(skeletons) or skeletons[orbit_index] != skeleton.skeleton_id:
        raise CertificateInvalidError(
            "certificate_invalid: component action does not fix the selected skeleton"
        )
    if (
        component not in skeleton.centralizer_components
        or component.full_graded_image_digest != skeleton.full_graded_image_digest
    ):
        raise CertificateInvalidError(
            "certificate_invalid: component certificate does not fix this full graded skeleton"
        )
    local = _bits(local_coordinates, "local_coordinates")
    if local != component.marking_shift:
        raise CertificateInvalidError(
            "certificate_invalid: diagnostic marking coordinates differ from the exact component shift"
        )
    start, stop = matrices.coordinate_blocks.local_slices[1][orbit_index]
    if stop - start != len(local):
        raise CertificateInvalidError(
            "certificate_invalid: local marking coordinate dimension differs from the relative slice"
        )
    raw = (0,) * start + local + (0,) * (matrices.D.column_count - stop)
    quotient_action = _quotient_translation(matrices, raw)
    coordinate_certificate_id = _digest(
        "diagnostic-component-coordinate",
        {
            "component_id": component.component_id,
            "local_coordinates": list(local),
            "relative_certificate_id": matrices.certificate.certificate_id,
            "skeleton_id": skeleton.skeleton_id,
        },
    )
    core = {
        "component_domain_digest": component.domain_digest,
        "component_id": component.component_id,
        "coordinate_certificate_id": coordinate_certificate_id,
        "diagnostic": True,
        "full_graded_image_digest": component.full_graded_image_digest,
        "instance_id": instance_id,
        "local_coordinates": list(local),
        "marking_shift": list(component.marking_shift),
        "quotient_action": _arrow_mapping(quotient_action),
        "raw_translation": list(raw),
        "relative_certificate_id": matrices.certificate.certificate_id,
        "skeleton_id": skeleton.skeleton_id,
    }
    return CertifiedCentralizerAction(
        _digest("certified-centralizer-action", core),
        matrices.certificate.certificate_id,
        instance_id,
        skeleton.skeleton_id,
        component.component_id,
        component.full_graded_image_digest,
        component.domain_digest,
        component.marking_shift,
        local,
        raw,
        quotient_action,
        coordinate_certificate_id,
        True,
    )


def certify_centralizer_action(
    matrices: RelativeMatrices,
    *,
    skeleton_ids: Sequence[str],
    instance_id: str,
    skeleton: Z2LocalSkeleton,
    component: CentralizerComponent,
    marking_coordinates: Z2MarkingCoordinateCertificate,
    bar_equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
) -> CertifiedCentralizerAction:
    """Convert one exact Task-11 component shift through verified Task-5 psi."""

    if type(matrices) is not RelativeMatrices or type(matrices.D) is not MatrixGF2:
        raise TypeError("centralizer action requires GF(2) relative matrices")
    if type(skeleton) is not Z2LocalSkeleton or skeleton.time_orbit is None:
        raise TypeError("centralizer action requires a graded Z2LocalSkeleton")
    if type(component) is not CentralizerComponent:
        raise TypeError("centralizer action requires CentralizerComponent")
    if type(marking_coordinates) is not Z2MarkingCoordinateCertificate:
        raise TypeError("centralizer action requires Z2MarkingCoordinateCertificate")
    skeletons = _skeleton_ids(skeleton_ids)
    try:
        orbit_index = matrices.coordinate_blocks.instance_ids.index(instance_id)
    except ValueError as error:
        raise CertificateInvalidError(
            "certificate_invalid: centralizer instance is absent from the relative problem"
        ) from error
    if orbit_index >= len(skeletons) or skeletons[orbit_index] != skeleton.skeleton_id:
        raise CertificateInvalidError(
            "certificate_invalid: component action does not fix the selected skeleton"
        )
    if (
        component not in skeleton.centralizer_components
        or component.full_graded_image_digest != skeleton.full_graded_image_digest
    ):
        raise CertificateInvalidError(
            "certificate_invalid: component certificate does not fix this full graded skeleton"
        )
    _verify_skeleton_against_equivalence(skeleton, bar_equivalence)
    expected = coordinate_z2_marking_shift(
        component.marking_shift,
        bar_equivalence,
        authority,
    )
    if marking_coordinates != expected:
        raise CertificateInvalidError(
            "certificate_invalid: component marking coordinates do not replay verified bar psi"
        )
    start, stop = matrices.coordinate_blocks.local_slices[1][orbit_index]
    if (
        stop - start != len(expected.coordinates)
        or matrices.certificate.local_complex_ids[orbit_index]
        != _certified_gf2_complex(bar_equivalence.resolution).complex_id
    ):
        raise CertificateInvalidError(
            "certificate_invalid: marking coordinates bind another local cochain slice"
        )
    raw = (
        (0,) * start
        + expected.coordinates
        + (0,) * (matrices.D.column_count - stop)
    )
    quotient_action = _quotient_translation(matrices, raw)
    core = {
        "component_domain_digest": component.domain_digest,
        "component_id": component.component_id,
        "coordinate_certificate_id": expected.certificate_id,
        "diagnostic": False,
        "full_graded_image_digest": component.full_graded_image_digest,
        "instance_id": instance_id,
        "local_coordinates": list(expected.coordinates),
        "marking_shift": list(component.marking_shift),
        "quotient_action": _arrow_mapping(quotient_action),
        "raw_translation": list(raw),
        "relative_certificate_id": matrices.certificate.certificate_id,
        "skeleton_id": skeleton.skeleton_id,
    }
    return CertifiedCentralizerAction(
        _digest("certified-centralizer-action", core),
        matrices.certificate.certificate_id,
        instance_id,
        skeleton.skeleton_id,
        component.component_id,
        component.full_graded_image_digest,
        component.domain_digest,
        component.marking_shift,
        expected.coordinates,
        raw,
        quotient_action,
        expected.certificate_id,
        False,
    )


_Z2_STRATUM_RELEASE_SEAL = object()


@dataclass(frozen=True, slots=True)
class _ReleaseZ2StratumSnapshot:
    certificate_id: str
    relative_certificate_id: str
    skeleton_ids: tuple[str, ...]
    problem_id: str
    source_verification_digest: str
    source_snapshot: _CertifiedZ2ProblemSnapshot
    _factory_token: object

    def __post_init__(self) -> None:
        if self._factory_token is not _Z2_STRATUM_RELEASE_SEAL:
            raise CertificateInvalidError(
                "certificate_invalid: release stratum snapshot requires the classifier factory"
            )
        _require_digest(self.certificate_id, "$_ReleaseZ2StratumSnapshot.certificate_id")
        _require_digest(
            self.relative_certificate_id,
            "$_ReleaseZ2StratumSnapshot.relative_certificate_id",
        )
        skeletons = _skeleton_ids(self.skeleton_ids)
        _require_digest(self.problem_id, "$_ReleaseZ2StratumSnapshot.problem_id")
        _require_digest(
            self.source_verification_digest,
            "$_ReleaseZ2StratumSnapshot.source_verification_digest",
        )
        if (
            type(self.source_snapshot) is not _CertifiedZ2ProblemSnapshot
            or self.source_snapshot._factory_token is not _CERTIFIED_PROBLEM_SEAL
            or self.problem_id != self.source_snapshot.problem_id
            or self.source_verification_digest
            != self.source_snapshot.source_verification_digest
            or sum(
                branch.matrices.certificate.certificate_id
                == self.relative_certificate_id
                and branch.skeleton_ids == skeletons
                for branch in self.source_snapshot.branches
            )
            != 1
        ):
            raise CertificateInvalidError(
                "certificate_invalid: release stratum snapshot differs from its certified source"
            )
        object.__setattr__(self, "skeleton_ids", skeletons)


@dataclass(frozen=True, slots=True)
class Z2StratumCertificate:
    certificate_id: str
    relative_certificate_id: str
    relative_problem_digest: str
    provenance: str
    problem_id: str | None
    source_verification_digest: str | None
    skeleton_ids: tuple[str, ...]
    basepoint: tuple[int, ...]
    kernel_basis: tuple[tuple[int, ...], ...]
    boundary_basis: tuple[tuple[int, ...], ...]
    quotient_basis: tuple[tuple[int, ...], ...]
    h1_unmarking_passes: int
    h1_diagnostic_id: str
    h1_unmarking: H1UnmarkingCertificate
    residual_action_ids: tuple[str, ...]
    centralizer_action_ids: tuple[str, ...]
    cross_skeleton_arrow_ids: tuple[str, ...]
    centralizer_actions: tuple[CertifiedCentralizerAction, ...]
    cross_skeleton_arrows: tuple[Z2CrossSkeletonArrow, ...]
    matrices: RelativeMatrices
    _release_snapshot: _ReleaseZ2StratumSnapshot | None = None

    def __post_init__(self) -> None:
        _require_digest(self.certificate_id, "$Z2StratumCertificate.certificate_id")
        _require_digest(
            self.relative_certificate_id,
            "$Z2StratumCertificate.relative_certificate_id",
        )
        _require_digest(
            self.relative_problem_digest,
            "$Z2StratumCertificate.relative_problem_digest",
        )
        if self.provenance not in {"diagnostic-direct", "diagnostic", "release"}:
            raise ValueError("$Z2StratumCertificate.provenance: invalid provenance")
        if self.provenance == "diagnostic-direct":
            if self.problem_id is not None or self.source_verification_digest is not None:
                raise CertificateInvalidError(
                    "certificate_invalid: direct diagnostic provenance cannot claim an envelope"
                )
        elif self.provenance == "diagnostic":
            _require_digest(self.problem_id, "$Z2StratumCertificate.problem_id")
            if self.source_verification_digest is not None:
                raise CertificateInvalidError(
                    "certificate_invalid: diagnostic provenance cannot claim a release source"
                )
        else:
            _require_digest(self.problem_id, "$Z2StratumCertificate.problem_id")
            _require_digest(
                self.source_verification_digest,
                "$Z2StratumCertificate.source_verification_digest",
            )
        skeletons = _skeleton_ids(self.skeleton_ids)
        if type(self.matrices) is not RelativeMatrices:
            raise TypeError("$Z2StratumCertificate.matrices: expected RelativeMatrices")
        matrices = RelativeMatrices(
            self.matrices.B,
            self.matrices.D,
            self.matrices.E,
            self.matrices.offset,
            self.matrices.coordinate_blocks,
            self.matrices.certificate,
        )
        if type(matrices.B) is not MatrixGF2:
            raise TypeError("$Z2StratumCertificate.matrices: expected GF(2) cone")
        if self.relative_certificate_id != matrices.certificate.certificate_id:
            raise ValueError("$Z2StratumCertificate: relative certificate differs")
        if self.relative_problem_digest != matrices.certificate.problem_digest:
            raise ValueError("$Z2StratumCertificate: relative problem digest differs")
        ambient = matrices.D.column_count
        basepoint = _bits(self.basepoint, "$Z2StratumCertificate.basepoint", length=ambient)
        kernel = tuple(
            _bits(vector, "$Z2StratumCertificate.kernel_basis", length=ambient)
            for vector in self.kernel_basis
        )
        boundaries = tuple(
            _bits(vector, "$Z2StratumCertificate.boundary_basis", length=ambient)
            for vector in self.boundary_basis
        )
        representatives = tuple(
            _bits(vector, "$Z2StratumCertificate.quotient_basis", length=ambient)
            for vector in self.quotient_basis
        )
        solved = solve_affine(matrices.D, tuple(matrices.offset))
        if not isinstance(solved, GF2AffineSolution):
            raise ValueError("$Z2StratumCertificate: stored affine branch is obstructed")
        if solved.basepoint != basepoint or solved.kernel_basis != kernel:
            raise ValueError("$Z2StratumCertificate: affine solution does not replay")
        expected_boundaries = image_basis(matrices.B)
        if boundaries != expected_boundaries:
            raise ValueError("$Z2StratumCertificate: boundary basis does not replay B")
        expected_quotient = quotient_basis(
            _columns_matrix(kernel, ambient),
            matrices.B,
        )
        if representatives != expected_quotient.representatives:
            raise ValueError("$Z2StratumCertificate: quotient basis does not replay")
        if type(self.h1_unmarking_passes) is not int or self.h1_unmarking_passes != 1:
            raise CertificateInvalidError("certificate_invalid: H1 unmarking must occur exactly once")
        _require_digest(self.h1_diagnostic_id, "$Z2StratumCertificate.h1_diagnostic_id")
        if type(self.h1_unmarking) is not H1UnmarkingCertificate:
            raise TypeError("$Z2StratumCertificate.h1_unmarking: invalid certificate")
        if (
            self.h1_unmarking.certificate_id != self.h1_diagnostic_id
            or self.h1_unmarking.relative_certificate_id != self.relative_certificate_id
            or self.h1_unmarking.application_count != 1
        ):
            raise CertificateInvalidError(
                "certificate_invalid: H1 unmarking does not replay this stratum"
            )
        residual_ids = tuple(self.residual_action_ids)
        centralizer_ids = tuple(self.centralizer_action_ids)
        cross_ids = tuple(self.cross_skeleton_arrow_ids)
        for index, item in enumerate(residual_ids):
            _require_digest(item, f"$Z2StratumCertificate.residual_action_ids[{index}]")
        for index, item in enumerate(cross_ids):
            _require_digest(item, f"$Z2StratumCertificate.cross_skeleton_arrow_ids[{index}]")
        for index, item in enumerate(centralizer_ids):
            _require_digest(item, f"$Z2StratumCertificate.centralizer_action_ids[{index}]")
        if len(set(residual_ids)) != len(residual_ids) or residual_ids != tuple(sorted(residual_ids)):
            raise ValueError("$Z2StratumCertificate.residual_action_ids: expected canonical IDs")
        if len(set(cross_ids)) != len(cross_ids) or cross_ids != tuple(sorted(cross_ids)):
            raise ValueError("$Z2StratumCertificate.cross_skeleton_arrow_ids: expected canonical IDs")
        if (
            len(set(centralizer_ids)) != len(centralizer_ids)
            or centralizer_ids != tuple(sorted(centralizer_ids))
        ):
            raise ValueError("$Z2StratumCertificate.centralizer_action_ids: expected canonical IDs")
        centralizers = tuple(self.centralizer_actions)
        crosses = tuple(self.cross_skeleton_arrows)
        if any(type(item) is not CertifiedCentralizerAction for item in centralizers):
            raise TypeError("$Z2StratumCertificate.centralizer_actions: invalid action")
        if any(type(item) is not Z2CrossSkeletonArrow for item in crosses):
            raise TypeError("$Z2StratumCertificate.cross_skeleton_arrows: invalid arrow")
        if tuple(sorted(item.action_id for item in centralizers)) != centralizer_ids:
            raise ValueError("$Z2StratumCertificate.centralizer_actions: IDs differ")
        if tuple(sorted(item.arrow_id for item in crosses)) != cross_ids:
            raise ValueError("$Z2StratumCertificate.cross_skeleton_arrows: IDs differ")
        if any(
            item.relative_certificate_id != self.relative_certificate_id
            or item.skeleton_id not in skeletons
            or _arrow_id(item.quotient_action) not in residual_ids
            for item in centralizers
        ):
            raise CertificateInvalidError(
                "certificate_invalid: centralizer action does not replay this stratum"
            )
        if any(item.source_skeleton_ids != skeletons for item in crosses):
            raise CertificateInvalidError(
                "certificate_invalid: cross-skeleton arrow does not replay this stratum"
            )
        core = {
            "basepoint": list(basepoint),
            "boundary_basis": [list(vector) for vector in boundaries],
            "centralizer_action_ids": list(centralizer_ids),
            "cross_skeleton_arrow_ids": list(cross_ids),
            "h1_diagnostic_id": self.h1_diagnostic_id,
            "h1_unmarking_passes": self.h1_unmarking_passes,
            "kernel_basis": [list(vector) for vector in kernel],
            "quotient_basis": [list(vector) for vector in representatives],
            "relative_certificate_id": self.relative_certificate_id,
            "relative_problem_digest": self.relative_problem_digest,
            "problem_id": self.problem_id,
            "provenance": self.provenance,
            "residual_action_ids": list(residual_ids),
            "skeleton_ids": list(skeletons),
            "source_verification_digest": self.source_verification_digest,
        }
        if self.certificate_id != _digest("z2-stratum-certificate", core):
            raise ValueError("$Z2StratumCertificate.certificate_id: payload digest differs")
        snapshot = self._release_snapshot
        if self.provenance == "release":
            if (
                type(snapshot) is not _ReleaseZ2StratumSnapshot
                or snapshot.certificate_id != self.certificate_id
                or snapshot.relative_certificate_id != self.relative_certificate_id
                or snapshot.skeleton_ids != skeletons
                or snapshot.problem_id != self.problem_id
                or snapshot.source_verification_digest
                != self.source_verification_digest
            ):
                raise CertificateInvalidError(
                    "certificate_invalid: release stratum lacks its certified source snapshot"
                )
        elif snapshot is not None:
            raise CertificateInvalidError(
                "certificate_invalid: diagnostic stratum cannot carry release provenance"
            )
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "basepoint", basepoint)
        object.__setattr__(self, "kernel_basis", kernel)
        object.__setattr__(self, "boundary_basis", boundaries)
        object.__setattr__(self, "quotient_basis", representatives)
        object.__setattr__(self, "residual_action_ids", residual_ids)
        object.__setattr__(self, "centralizer_action_ids", centralizer_ids)
        object.__setattr__(self, "cross_skeleton_arrow_ids", cross_ids)
        object.__setattr__(self, "centralizer_actions", centralizers)
        object.__setattr__(self, "cross_skeleton_arrows", crosses)
        object.__setattr__(self, "matrices", matrices)


@dataclass(frozen=True, slots=True)
class FiniteAffineStratum:
    stratum_id: str
    skeleton_ids: tuple[str, ...]
    basepoint: tuple[int, ...]
    homogeneous_basis: tuple[tuple[int, ...], ...]
    quotient_dimension: int
    residual_actions: tuple[GF2AffineArrow, ...]
    certificate: Z2StratumCertificate

    def __post_init__(self) -> None:
        _require_digest(self.stratum_id, "$FiniteAffineStratum.stratum_id")
        skeletons = _skeleton_ids(self.skeleton_ids)
        if type(self.certificate) is not Z2StratumCertificate:
            raise TypeError("$FiniteAffineStratum.certificate: expected Z2StratumCertificate")
        if skeletons != self.certificate.skeleton_ids:
            raise ValueError("$FiniteAffineStratum.skeleton_ids: certificate differs")
        basepoint = _bits(
            self.basepoint,
            "$FiniteAffineStratum.basepoint",
            length=self.certificate.matrices.D.column_count,
        )
        basis = tuple(
            _bits(vector, "$FiniteAffineStratum.homogeneous_basis", length=len(basepoint))
            for vector in self.homogeneous_basis
        )
        if basepoint != self.certificate.basepoint or basis != self.certificate.quotient_basis:
            raise ValueError("$FiniteAffineStratum: affine coordinates differ from certificate")
        if type(self.quotient_dimension) is not int or self.quotient_dimension != len(basis):
            raise ValueError("$FiniteAffineStratum.quotient_dimension: basis differs")
        actions = tuple(self.residual_actions)
        if any(type(action) is not GF2AffineArrow for action in actions):
            raise TypeError("$FiniteAffineStratum.residual_actions: expected affine arrows")
        if any(
            action.source_dimension != len(basis)
            or action.target_dimension != len(basis)
            or kernel_basis(action.linear)
            for action in actions
        ):
            raise CertificateInvalidError(
                "certificate_invalid: residual action must be an invertible quotient endomorphism"
            )
        if tuple(sorted(_arrow_id(action) for action in actions)) != self.certificate.residual_action_ids:
            raise ValueError("$FiniteAffineStratum.residual_actions: certificate IDs differ")
        expected_id = _digest(
            "finite-affine-stratum",
            {
                "certificate_id": self.certificate.certificate_id,
                "skeleton_ids": list(skeletons),
            },
        )
        if self.stratum_id != expected_id:
            raise ValueError("$FiniteAffineStratum.stratum_id: payload digest differs")
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "basepoint", basepoint)
        object.__setattr__(self, "homogeneous_basis", basis)
        object.__setattr__(self, "residual_actions", actions)


_Z2_POINT_BATCH_SEAL = object()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _CertifiedZ2PointBatchAuthority:
    stratum: FiniteAffineStratum
    state_snapshot: tuple[object, ...]
    _factory_token: object

    def __post_init__(self) -> None:
        if self._factory_token is not _Z2_POINT_BATCH_SEAL:
            raise CertificateInvalidError(
                "certificate_invalid: point batch authority requires verified enumeration"
            )


_Z2_POINT_BATCH_AUTHORITIES: dict[
    int,
    tuple[
        weakref.ReferenceType[_CertifiedZ2PointBatchAuthority],
        FiniteAffineStratum,
        tuple[object, ...],
    ],
] = {}


def _z2_point_batch_state(
    stratum: FiniteAffineStratum,
) -> tuple[object, ...]:
    """Ephemeral structural snapshot; never serialized or used as math authority."""

    certificate = stratum.certificate
    release_snapshot = certificate._release_snapshot
    source_snapshot = (
        None if release_snapshot is None else release_snapshot.source_snapshot
    )
    return (
        id(stratum),
        id(certificate),
        id(release_snapshot),
        id(source_snapshot),
        stratum.stratum_id,
        certificate.certificate_id,
        certificate.relative_certificate_id,
        certificate.problem_id,
        certificate.source_verification_digest,
        repr(stratum),
    )


def _register_z2_point_batch_authority(
    authority: _CertifiedZ2PointBatchAuthority,
) -> None:
    key = id(authority)

    def discard(
        reference: weakref.ReferenceType[_CertifiedZ2PointBatchAuthority],
    ) -> None:
        current = _Z2_POINT_BATCH_AUTHORITIES.get(key)
        if current is not None and current[0] is reference:
            _Z2_POINT_BATCH_AUTHORITIES.pop(key, None)

    identity = weakref.ref(authority, discard)
    _Z2_POINT_BATCH_AUTHORITIES[key] = (
        identity,
        authority.stratum,
        authority.state_snapshot,
    )


def _verified_z2_point_batch_stratum(
    authority: object,
    stratum: FiniteAffineStratum | None,
) -> FiniteAffineStratum:
    if type(authority) is not _CertifiedZ2PointBatchAuthority:
        raise CertificateInvalidError(
            "certificate_invalid: point batch authority is not factory issued"
        )
    issued = _Z2_POINT_BATCH_AUTHORITIES.get(id(authority))
    if (
        issued is None
        or issued[0]() is not authority
        or authority._factory_token is not _Z2_POINT_BATCH_SEAL
        or authority.stratum is not issued[1]
        or authority.state_snapshot != issued[2]
        or stratum is not issued[1]
        or _z2_point_batch_state(issued[1]) != issued[2]
    ):
        raise CertificateInvalidError(
            "certificate_invalid: point batch authority or stratum changed after verification"
        )
    if (
        stratum.certificate.provenance == "release"
        and _certified_z2_snapshot_record(
            stratum.certificate._release_snapshot.source_snapshot
        )
        is None
    ):
        raise CertificateInvalidError(
            "certificate_invalid: point batch lost its exact release source authority"
        )
    return stratum


@dataclass(frozen=True, slots=True)
class Z2OrbitMembershipWitness:
    quotient_coordinates: tuple[int, ...]
    raw_representative: tuple[int, ...]
    orbit_members: tuple[tuple[int, ...], ...]
    witness_id: str

    def __post_init__(self) -> None:
        coordinates = _bits(self.quotient_coordinates, "$Z2OrbitMembershipWitness.quotient_coordinates")
        raw = _bits(self.raw_representative, "$Z2OrbitMembershipWitness.raw_representative")
        members = tuple(
            _bits(member, "$Z2OrbitMembershipWitness.orbit_members", length=len(coordinates))
            for member in self.orbit_members
        )
        if not members or members != tuple(sorted(set(members))) or coordinates != members[0]:
            raise ValueError("$Z2OrbitMembershipWitness.orbit_members: expected canonical orbit")
        expected = _digest(
            "z2-orbit-membership",
            {
                "orbit_members": [list(member) for member in members],
                "quotient_coordinates": list(coordinates),
                "raw_representative": list(raw),
            },
        )
        if self.witness_id != expected:
            raise ValueError("$Z2OrbitMembershipWitness.witness_id: payload digest differs")
        object.__setattr__(self, "quotient_coordinates", coordinates)
        object.__setattr__(self, "raw_representative", raw)
        object.__setattr__(self, "orbit_members", members)


@dataclass(frozen=True, slots=True)
class CertifiedZ2Point:
    point_id: str
    stratum_id: str
    quotient_coordinates: tuple[int, ...]
    representative: tuple[int, ...]
    orbit_members: tuple[tuple[int, ...], ...]
    membership_witness: Z2OrbitMembershipWitness
    stratum: FiniteAffineStratum | None = None
    _batch_authority: InitVar[object | None] = None

    def __post_init__(self, _batch_authority: object | None) -> None:
        _require_digest(self.point_id, "$CertifiedZ2Point.point_id")
        _require_digest(self.stratum_id, "$CertifiedZ2Point.stratum_id")
        if type(self.membership_witness) is not Z2OrbitMembershipWitness:
            raise TypeError("$CertifiedZ2Point.membership_witness: invalid witness")
        try:
            checked_witness = Z2OrbitMembershipWitness(
                quotient_coordinates=self.membership_witness.quotient_coordinates,
                raw_representative=self.membership_witness.raw_representative,
                orbit_members=self.membership_witness.orbit_members,
                witness_id=self.membership_witness.witness_id,
            )
        except (TypeError, ValueError) as error:
            raise CertificateInvalidError(
                f"certificate_invalid: point membership witness replay failed: {error}"
            ) from error
        if checked_witness != self.membership_witness:
            raise CertificateInvalidError(
                "certificate_invalid: point membership witness differs after replay"
            )
        if self.stratum is None:
            raise CertificateInvalidError(
                "certificate_invalid: a certified point requires its complete stratum"
            )
        if _batch_authority is None:
            try:
                checked_stratum = _replay_finite_affine_stratum(self.stratum)
            except CertificateInvalidError:
                raise
            except (TypeError, ValueError) as error:
                raise CertificateInvalidError(
                    f"certificate_invalid: point stratum replay failed: {error}"
                ) from error
        else:
            checked_stratum = _verified_z2_point_batch_stratum(
                _batch_authority,
                self.stratum,
            )
        if checked_stratum != self.stratum or self.stratum_id != checked_stratum.stratum_id:
            raise CertificateInvalidError(
                "certificate_invalid: point binds another or mutated stratum"
            )
        coordinates = _bits(
            self.quotient_coordinates,
            "$CertifiedZ2Point.quotient_coordinates",
            length=checked_stratum.quotient_dimension,
        )
        raw = _bits(
            self.representative,
            "$CertifiedZ2Point.representative",
            length=len(checked_stratum.basepoint),
        )
        members = tuple(
            _bits(
                member,
                "$CertifiedZ2Point.orbit_members",
                length=checked_stratum.quotient_dimension,
            )
            for member in self.orbit_members
        )
        complete_orbit = {coordinates}
        frontier = [coordinates]
        while frontier:
            current = frontier.pop()
            for action in checked_stratum.residual_actions:
                image = action.apply(current)
                if image not in complete_orbit:
                    complete_orbit.add(image)
                    frontier.append(image)
        expected_members = tuple(sorted(complete_orbit))
        expected_raw = _xor(
            checked_stratum.basepoint,
            _linear_combination(
                checked_stratum.homogeneous_basis,
                coordinates,
                len(checked_stratum.basepoint),
            ),
        )
        if members != expected_members or raw != expected_raw:
            raise CertificateInvalidError(
                "certificate_invalid: point orbit or raw representative is incomplete"
            )
        if _matvec(checked_stratum.certificate.matrices.D, raw) != tuple(
            checked_stratum.certificate.matrices.offset
        ):
            raise CertificateInvalidError(
                "certificate_invalid: point representative does not solve Dz=b"
            )
        if (
            coordinates != checked_witness.quotient_coordinates
            or raw != checked_witness.raw_representative
            or members != checked_witness.orbit_members
        ):
            raise ValueError("$CertifiedZ2Point: membership witness differs")
        expected = _digest(
            "certified-z2-point",
            {
                "membership_witness_id": checked_witness.witness_id,
                "stratum_id": self.stratum_id,
            },
        )
        if self.point_id != expected:
            raise ValueError("$CertifiedZ2Point.point_id: payload digest differs")
        object.__setattr__(self, "quotient_coordinates", coordinates)
        object.__setattr__(self, "representative", raw)
        object.__setattr__(self, "orbit_members", members)
        object.__setattr__(self, "membership_witness", checked_witness)
        object.__setattr__(self, "stratum", checked_stratum)


def _canonical_complete_centralizer_actions(
    skeleton: Z2LocalSkeleton,
    actions: Sequence[CertifiedCentralizerAction],
) -> tuple[CertifiedCentralizerAction, ...]:
    """Match actions exactly to a skeleton's exhaustive component list."""

    if type(skeleton) is not Z2LocalSkeleton:
        raise TypeError("centralizer completeness requires Z2LocalSkeleton")
    normalized = tuple(actions)
    if any(type(action) is not CertifiedCentralizerAction for action in normalized):
        raise TypeError("centralizer actions must be certified actions")
    components = tuple(skeleton.centralizer_components)
    component_ids = tuple(component.component_id for component in components)
    action_ids = tuple(action.component_id for action in normalized)
    if (
        len(set(component_ids)) != len(component_ids)
        or len(set(action_ids)) != len(action_ids)
        or set(action_ids) != set(component_ids)
    ):
        raise CertificateInvalidError(
            "certificate_invalid: actions must completely cover the exhaustive centralizer components"
        )
    by_component = {action.component_id: action for action in normalized}
    canonical = tuple(by_component[component_id] for component_id in component_ids)
    for component, action in zip(components, canonical, strict=True):
        if (
            action.skeleton_id != skeleton.skeleton_id
            or action.full_graded_image_digest
            != component.full_graded_image_digest
            or action.component_domain_digest != component.domain_digest
            or action.marking_shift != component.marking_shift
        ):
            raise CertificateInvalidError(
                "certificate_invalid: centralizer action does not bind its exhaustive component"
            )
    return canonical


@dataclass(frozen=True, slots=True)
class Z2OrbitBinding:
    binding_id: str
    instance_id: str
    inclusion_id: str
    skeleton: Z2LocalSkeleton
    bar_equivalence: BarResolutionEquivalence
    defect_coordinates: Z2DefectCoordinateCertificate
    centralizer_actions: tuple[CertifiedCentralizerAction, ...] = ()

    def __post_init__(self) -> None:
        _require_digest(self.binding_id, "$Z2OrbitBinding.binding_id")
        for name in ("instance_id", "inclusion_id"):
            value = getattr(self, name)
            if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
                raise ValueError(f"$Z2OrbitBinding.{name}: invalid identifier")
        if type(self.skeleton) is not Z2LocalSkeleton:
            raise TypeError("$Z2OrbitBinding.skeleton: invalid skeleton")
        if type(self.bar_equivalence) is not BarResolutionEquivalence:
            raise TypeError("$Z2OrbitBinding.bar_equivalence: invalid equivalence")
        if type(self.defect_coordinates) is not Z2DefectCoordinateCertificate:
            raise TypeError("$Z2OrbitBinding.defect_coordinates: invalid certificate")
        if (
            self.defect_coordinates.skeleton_id != self.skeleton.skeleton_id
            or self.defect_coordinates.resolution_id
            != self.bar_equivalence.resolution_id
            or self.defect_coordinates.bar_equivalence_id
            != self.bar_equivalence.equivalence_id
        ):
            raise CertificateInvalidError(
                "certificate_invalid: orbit defect coordinates do not bind its skeleton and bar trace"
            )
        actions = _canonical_complete_centralizer_actions(
            self.skeleton,
            self.centralizer_actions,
        )
        if any(
            action.instance_id != self.instance_id
            or action.skeleton_id != self.skeleton.skeleton_id
            for action in actions
        ):
            raise CertificateInvalidError(
                "certificate_invalid: centralizer action does not bind the orbit"
            )
        core = {
            "bar_equivalence_id": self.bar_equivalence.equivalence_id,
            "centralizer_action_ids": sorted(action.action_id for action in actions),
            "defect_coordinate_certificate_id": self.defect_coordinates.certificate_id,
            "inclusion_id": self.inclusion_id,
            "instance_id": self.instance_id,
            "skeleton_id": self.skeleton.skeleton_id,
        }
        if self.binding_id != _digest("z2-orbit-binding", core):
            raise ValueError("$Z2OrbitBinding.binding_id: payload digest differs")
        object.__setattr__(self, "centralizer_actions", actions)


def make_z2_orbit_binding(
    *,
    instance_id: str,
    inclusion_id: str,
    skeleton: Z2LocalSkeleton,
    bar_equivalence: BarResolutionEquivalence,
    defect_coordinates: Z2DefectCoordinateCertificate,
    centralizer_actions: Sequence[CertifiedCentralizerAction] = (),
) -> Z2OrbitBinding:
    actions = _canonical_complete_centralizer_actions(
        skeleton,
        centralizer_actions,
    )
    core = {
        "bar_equivalence_id": bar_equivalence.equivalence_id,
        "centralizer_action_ids": sorted(action.action_id for action in actions),
        "defect_coordinate_certificate_id": defect_coordinates.certificate_id,
        "inclusion_id": inclusion_id,
        "instance_id": instance_id,
        "skeleton_id": skeleton.skeleton_id,
    }
    return Z2OrbitBinding(
        _digest("z2-orbit-binding", core),
        instance_id,
        inclusion_id,
        skeleton,
        bar_equivalence,
        defect_coordinates,
        actions,
    )


@dataclass(frozen=True, slots=True)
class Z2Branch:
    branch_id: str
    source_problem: RelativeProblem
    matrices: RelativeMatrices
    skeleton_ids: tuple[str, ...]
    inclusion_ids: tuple[str, ...]
    defect_coordinate_certificate_ids: tuple[str, ...]
    orbit_bindings: tuple[Z2OrbitBinding, ...]
    h1_unmarking: H1UnmarkingCertificate
    centralizer_actions: tuple[CertifiedCentralizerAction, ...]
    cross_skeleton_arrows: tuple[Z2CrossSkeletonArrow, ...]
    diagnostic: bool

    def __post_init__(self) -> None:
        _require_digest(self.branch_id, "$Z2Branch.branch_id")
        if type(self.source_problem) is not RelativeProblem or self.source_problem.ring != "gf2":
            raise TypeError("$Z2Branch.source_problem: expected GF(2) RelativeProblem")
        if type(self.matrices) is not RelativeMatrices or type(self.matrices.D) is not MatrixGF2:
            raise TypeError("$Z2Branch.matrices: expected GF(2) RelativeMatrices")
        verify_relative_certificate(self.matrices, self.source_problem)
        skeletons = _skeleton_ids(self.skeleton_ids)
        if len(skeletons) != len(self.matrices.coordinate_blocks.instance_ids):
            raise ValueError("$Z2Branch.skeleton_ids: expected one ID per orbit instance")
        inclusion_ids = tuple(self.inclusion_ids)
        defect_ids = tuple(self.defect_coordinate_certificate_ids)
        bindings = tuple(self.orbit_bindings)
        if inclusion_ids and len(inclusion_ids) != len(skeletons):
            raise ValueError("$Z2Branch.inclusion_ids: expected one ID per orbit")
        if defect_ids and len(defect_ids) != len(skeletons):
            raise ValueError(
                "$Z2Branch.defect_coordinate_certificate_ids: expected one ID per orbit"
            )
        for index, inclusion_id in enumerate(inclusion_ids):
            if type(inclusion_id) is not str or _IDENTIFIER_RE.fullmatch(inclusion_id) is None:
                raise ValueError(f"$Z2Branch.inclusion_ids[{index}]: invalid identifier")
        for index, item in enumerate(defect_ids):
            _require_digest(item, f"$Z2Branch.defect_coordinate_certificate_ids[{index}]")
        if any(type(binding) is not Z2OrbitBinding for binding in bindings):
            raise TypeError("$Z2Branch.orbit_bindings: invalid binding")
        if bindings:
            if len(bindings) != len(skeletons):
                raise ValueError("$Z2Branch.orbit_bindings: expected one binding per orbit")
            binding_instance_ids = tuple(
                binding.instance_id for binding in bindings
            )
            if (
                len(set(binding_instance_ids)) != len(binding_instance_ids)
                or set(binding_instance_ids)
                != set(self.matrices.coordinate_blocks.instance_ids)
            ):
                raise ValueError(
                    "$Z2Branch.orbit_bindings: instances do not cover the relative coordinates"
                )
            if tuple(binding.skeleton.skeleton_id for binding in bindings) != skeletons:
                raise ValueError("$Z2Branch.orbit_bindings: skeleton IDs differ")
            if tuple(binding.inclusion_id for binding in bindings) != inclusion_ids:
                raise ValueError("$Z2Branch.orbit_bindings: inclusion IDs differ")
            if tuple(
                binding.defect_coordinates.certificate_id for binding in bindings
            ) != defect_ids:
                raise ValueError("$Z2Branch.orbit_bindings: defect IDs differ")
        if type(self.h1_unmarking) is not H1UnmarkingCertificate:
            raise TypeError("$Z2Branch.h1_unmarking: invalid certificate")
        if self.h1_unmarking.relative_certificate_id != self.matrices.certificate.certificate_id:
            raise CertificateInvalidError(
                "certificate_invalid: H1 certificate binds another branch"
            )
        centralizers = tuple(self.centralizer_actions)
        crosses = tuple(self.cross_skeleton_arrows)
        if any(type(item) is not CertifiedCentralizerAction for item in centralizers):
            raise TypeError("$Z2Branch.centralizer_actions: invalid certificate")
        if any(type(item) is not Z2CrossSkeletonArrow for item in crosses):
            raise TypeError("$Z2Branch.cross_skeleton_arrows: invalid boundary")
        if any(
            item.relative_certificate_id != self.matrices.certificate.certificate_id
            or item.skeleton_id not in skeletons
            for item in centralizers
        ):
            raise CertificateInvalidError(
                "certificate_invalid: centralizer action binds another branch"
            )
        if any(item.source_skeleton_ids != skeletons for item in crosses):
            raise CertificateInvalidError(
                "certificate_invalid: cross-skeleton boundary binds another source"
            )
        if type(self.diagnostic) is not bool:
            raise TypeError("$Z2Branch.diagnostic: expected boolean")
        if not self.diagnostic and (
            not inclusion_ids
            or not defect_ids
            or not bindings
            or any(item.diagnostic for item in centralizers)
        ):
            raise CertificateInvalidError(
                "certificate_invalid: production branch lacks release source bindings"
            )
        core = {
            "centralizer_action_ids": sorted(item.action_id for item in centralizers),
            "cross_skeleton_arrow_ids": sorted(item.arrow_id for item in crosses),
            "defect_coordinate_certificate_ids": list(defect_ids),
            "diagnostic": self.diagnostic,
            "h1_unmarking_certificate_id": self.h1_unmarking.certificate_id,
            "inclusion_ids": list(inclusion_ids),
            "orbit_binding_ids": [binding.binding_id for binding in bindings],
            "relative_certificate_id": self.matrices.certificate.certificate_id,
            "skeleton_ids": list(skeletons),
        }
        if self.branch_id != _digest("z2-branch", core):
            raise ValueError("$Z2Branch.branch_id: payload digest differs")
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "inclusion_ids", inclusion_ids)
        object.__setattr__(self, "defect_coordinate_certificate_ids", defect_ids)
        object.__setattr__(self, "orbit_bindings", bindings)
        object.__setattr__(self, "centralizer_actions", centralizers)
        object.__setattr__(self, "cross_skeleton_arrows", crosses)


def _branch_core(
    *,
    matrices: RelativeMatrices,
    skeleton_ids: tuple[str, ...],
    inclusion_ids: tuple[str, ...],
    defect_coordinate_certificate_ids: tuple[str, ...],
    orbit_bindings: tuple[Z2OrbitBinding, ...],
    h1_unmarking: H1UnmarkingCertificate,
    centralizer_actions: tuple[CertifiedCentralizerAction, ...],
    cross_skeleton_arrows: tuple[Z2CrossSkeletonArrow, ...],
    diagnostic: bool,
) -> dict[str, object]:
    return {
        "centralizer_action_ids": sorted(item.action_id for item in centralizer_actions),
        "cross_skeleton_arrow_ids": sorted(item.arrow_id for item in cross_skeleton_arrows),
        "defect_coordinate_certificate_ids": list(defect_coordinate_certificate_ids),
        "diagnostic": diagnostic,
        "h1_unmarking_certificate_id": h1_unmarking.certificate_id,
        "inclusion_ids": list(inclusion_ids),
        "orbit_binding_ids": [binding.binding_id for binding in orbit_bindings],
        "relative_certificate_id": matrices.certificate.certificate_id,
        "skeleton_ids": list(skeleton_ids),
    }


def make_diagnostic_z2_branch(
    *,
    source_problem: RelativeProblem,
    matrices: RelativeMatrices,
    skeleton_ids: Sequence[str],
    h1_unmarking: H1UnmarkingCertificate | None = None,
    centralizer_actions: Sequence[CertifiedCentralizerAction] = (),
    cross_skeleton_arrows: Sequence[Z2CrossSkeletonArrow] = (),
) -> Z2Branch:
    if type(source_problem) is not RelativeProblem or type(matrices) is not RelativeMatrices:
        raise TypeError("diagnostic branch requires relative source and matrices")
    verify_relative_certificate(matrices, source_problem)
    skeletons = _skeleton_ids(skeleton_ids)
    if h1_unmarking is None:
        start, stop = matrices.coordinate_blocks.ambient_slices[0]
        diagnostic_columns = tuple(
            tuple(matrices.B[row][column] for row in range(matrices.B.row_count))
            for column in range(start, stop)
        )
        h1_unmarking = certify_h1_unmarking(
            matrices,
            character_basis_id=_digest(
                "diagnostic-character-basis",
                {"ambient_complex_id": matrices.certificate.ambient_complex_id},
            ),
            diagnostic_boundary_columns=diagnostic_columns,
        )
    centralizers = tuple(centralizer_actions)
    crosses = tuple(cross_skeleton_arrows)
    core = _branch_core(
        matrices=matrices,
        skeleton_ids=skeletons,
        inclusion_ids=(),
        defect_coordinate_certificate_ids=(),
        orbit_bindings=(),
        h1_unmarking=h1_unmarking,
        centralizer_actions=centralizers,
        cross_skeleton_arrows=crosses,
        diagnostic=True,
    )
    return Z2Branch(
        _digest("z2-branch", core),
        source_problem,
        matrices,
        skeletons,
        (),
        (),
        (),
        h1_unmarking,
        centralizers,
        crosses,
        True,
    )


def make_certified_z2_branch(
    *,
    source_problem: RelativeProblem,
    matrices: RelativeMatrices,
    orbit_bindings: Sequence[Z2OrbitBinding],
    h1_unmarking: H1UnmarkingCertificate,
    cross_skeleton_arrows: Sequence[Z2CrossSkeletonArrow] = (),
) -> Z2Branch:
    """Bind verified local target coordinates to one whole-tuple cone branch."""

    if type(source_problem) is not RelativeProblem or source_problem.ring != "gf2":
        raise TypeError("certified branch requires a GF(2) RelativeProblem")
    if type(matrices) is not RelativeMatrices or type(matrices.D) is not MatrixGF2:
        raise TypeError("certified branch requires GF(2) RelativeMatrices")
    verify_relative_certificate(matrices, source_problem)
    bindings = tuple(orbit_bindings)
    if (
        not bindings
        or any(type(binding) is not Z2OrbitBinding for binding in bindings)
    ):
        raise CertificateInvalidError(
            "certificate_invalid: orbit bindings must be a typed whole-tuple"
        )
    binding_instance_ids = tuple(binding.instance_id for binding in bindings)
    if (
        len(set(binding_instance_ids)) != len(binding_instance_ids)
        or set(binding_instance_ids)
        != set(matrices.coordinate_blocks.instance_ids)
    ):
        raise CertificateInvalidError(
            "certificate_invalid: orbit bindings must cover one ordered whole-tuple"
        )
    source_by_instance = {
        restriction.instance_id: (local, tuple(defect))
        for local, restriction, defect in zip(
            source_problem.locals,
            source_problem.restrictions,
            source_problem.local_defects,
            strict=True,
        )
    }
    if set(source_by_instance) != set(matrices.coordinate_blocks.instance_ids):
        raise CertificateInvalidError(
            "certificate_invalid: relative source instances differ from coordinate blocks"
        )
    for binding in bindings:
        local, defect = source_by_instance[binding.instance_id]
        if local.authority_id != binding.defect_coordinates.resolution_id:
            raise CertificateInvalidError(
                "certificate_invalid: defect coordinate resolution differs from the local complex"
            )
        if defect != binding.defect_coordinates.coordinates:
            raise CertificateInvalidError(
                "certificate_invalid: relative offset differs from the verified local defect coordinates"
            )
    if type(h1_unmarking) is not H1UnmarkingCertificate:
        raise TypeError("certified branch requires H1UnmarkingCertificate")
    if h1_unmarking.relative_certificate_id != matrices.certificate.certificate_id:
        raise CertificateInvalidError(
            "certificate_invalid: H1 unmarking certificate binds another branch"
        )
    centralizers = tuple(
        action for binding in bindings for action in binding.centralizer_actions
    )
    if any(action.diagnostic for action in centralizers):
        raise CertificateInvalidError(
            "certificate_invalid: diagnostic centralizer action cannot enter a release branch"
        )
    crosses = tuple(cross_skeleton_arrows)
    skeletons = tuple(binding.skeleton.skeleton_id for binding in bindings)
    inclusion_ids = tuple(binding.inclusion_id for binding in bindings)
    defect_ids = tuple(
        binding.defect_coordinates.certificate_id for binding in bindings
    )
    core = _branch_core(
        matrices=matrices,
        skeleton_ids=skeletons,
        inclusion_ids=inclusion_ids,
        defect_coordinate_certificate_ids=defect_ids,
        orbit_bindings=bindings,
        h1_unmarking=h1_unmarking,
        centralizer_actions=centralizers,
        cross_skeleton_arrows=crosses,
        diagnostic=False,
    )
    return Z2Branch(
        _digest("z2-branch", core),
        source_problem,
        matrices,
        skeletons,
        inclusion_ids,
        defect_ids,
        bindings,
        h1_unmarking,
        centralizers,
        crosses,
        False,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticZ2Problem:
    problem_id: str
    branches: tuple[Z2Branch, ...]

    def __post_init__(self) -> None:
        _require_digest(self.problem_id, "$DiagnosticZ2Problem.problem_id")
        branches = tuple(self.branches)
        if not branches or any(type(branch) is not Z2Branch or not branch.diagnostic for branch in branches):
            raise ValueError("$DiagnosticZ2Problem.branches: expected diagnostic branches")
        if len({branch.branch_id for branch in branches}) != len(branches):
            raise ValueError("$DiagnosticZ2Problem.branches: duplicate branch")
        if self.problem_id != _digest(
            "diagnostic-z2-problem",
            {"branch_ids": sorted(branch.branch_id for branch in branches)},
        ):
            raise ValueError("$DiagnosticZ2Problem.problem_id: payload digest differs")
        object.__setattr__(self, "branches", branches)


def make_diagnostic_z2_problem(branches: Sequence[Z2Branch]) -> DiagnosticZ2Problem:
    normalized = tuple(branches)
    return DiagnosticZ2Problem(
        _digest(
            "diagnostic-z2-problem",
            {"branch_ids": sorted(branch.branch_id for branch in normalized)},
        ),
        normalized,
    )


def _replay_h1_unmarking(
    certificate: H1UnmarkingCertificate,
) -> H1UnmarkingCertificate:
    if type(certificate) is not H1UnmarkingCertificate:
        raise TypeError("expected H1UnmarkingCertificate")
    return H1UnmarkingCertificate(
        certificate_id=certificate.certificate_id,
        relative_certificate_id=certificate.relative_certificate_id,
        character_basis_certificate_id=certificate.character_basis_certificate_id,
        ambient_c1_columns=certificate.ambient_c1_columns,
        ambient_h1_coordinates=certificate.ambient_h1_coordinates,
        boundary_columns=certificate.boundary_columns,
        diagnostic_boundary_columns=certificate.diagnostic_boundary_columns,
        application_count=certificate.application_count,
    )


def _replay_centralizer_action(
    action: CertifiedCentralizerAction,
) -> CertifiedCentralizerAction:
    if type(action) is not CertifiedCentralizerAction:
        raise TypeError("expected CertifiedCentralizerAction")
    return CertifiedCentralizerAction(
        action_id=action.action_id,
        relative_certificate_id=action.relative_certificate_id,
        instance_id=action.instance_id,
        skeleton_id=action.skeleton_id,
        component_id=action.component_id,
        full_graded_image_digest=action.full_graded_image_digest,
        component_domain_digest=action.component_domain_digest,
        marking_shift=action.marking_shift,
        local_coordinates=action.local_coordinates,
        raw_translation=action.raw_translation,
        quotient_action=_replay_affine_arrow(action.quotient_action),
        coordinate_certificate_id=action.coordinate_certificate_id,
        diagnostic=action.diagnostic,
    )


def _replay_cross_skeleton_arrow(
    arrow: Z2CrossSkeletonArrow,
) -> Z2CrossSkeletonArrow:
    if type(arrow) is not Z2CrossSkeletonArrow:
        raise TypeError("expected Z2CrossSkeletonArrow")
    return Z2CrossSkeletonArrow(
        arrow_id=arrow.arrow_id,
        source_skeleton_ids=arrow.source_skeleton_ids,
        target_skeleton_ids=arrow.target_skeleton_ids,
        quotient_action=_replay_affine_arrow(arrow.quotient_action),
        conjugacy_witness_id=arrow.conjugacy_witness_id,
    )


def _replay_z2_branch(branch: Z2Branch) -> Z2Branch:
    """Re-run every Task-12 dataclass boundary below the release envelope."""

    if type(branch) is not Z2Branch:
        raise TypeError("expected Z2Branch")
    h1 = _replay_h1_unmarking(branch.h1_unmarking)
    centralizers = tuple(
        _replay_centralizer_action(action)
        for action in branch.centralizer_actions
    )
    crosses = tuple(
        _replay_cross_skeleton_arrow(arrow)
        for arrow in branch.cross_skeleton_arrows
    )
    bindings = []
    for binding in branch.orbit_bindings:
        if type(binding) is not Z2OrbitBinding:
            raise TypeError("expected Z2OrbitBinding")
        defect = Z2DefectCoordinateCertificate(
            certificate_id=binding.defect_coordinates.certificate_id,
            resolution_id=binding.defect_coordinates.resolution_id,
            bar_equivalence_id=binding.defect_coordinates.bar_equivalence_id,
            skeleton_id=binding.defect_coordinates.skeleton_id,
            source_defect_digest=binding.defect_coordinates.source_defect_digest,
            coordinates=binding.defect_coordinates.coordinates,
        )
        binding_actions = tuple(
            _replay_centralizer_action(action)
            for action in binding.centralizer_actions
        )
        bindings.append(
            Z2OrbitBinding(
                binding_id=binding.binding_id,
                instance_id=binding.instance_id,
                inclusion_id=binding.inclusion_id,
                skeleton=binding.skeleton,
                bar_equivalence=binding.bar_equivalence,
                defect_coordinates=defect,
                centralizer_actions=binding_actions,
            )
        )
    return Z2Branch(
        branch_id=branch.branch_id,
        source_problem=branch.source_problem,
        matrices=branch.matrices,
        skeleton_ids=branch.skeleton_ids,
        inclusion_ids=branch.inclusion_ids,
        defect_coordinate_certificate_ids=(
            branch.defect_coordinate_certificate_ids
        ),
        orbit_bindings=tuple(bindings),
        h1_unmarking=h1,
        centralizer_actions=centralizers,
        cross_skeleton_arrows=crosses,
        diagnostic=branch.diagnostic,
    )


_CERTIFIED_PROBLEM_SEAL = object()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _CertifiedZ2ProblemSnapshot:
    cochain_problem: CertifiedCochainProblem
    authority: Task5VerificationAuthority
    branches: tuple[Z2Branch, ...]
    trusted_release_attestations: tuple[LauncherExecutionAttestation, ...]
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    source_verification_digest: str
    problem_id: str
    _factory_token: object

    def __post_init__(self) -> None:
        if self._factory_token is not _CERTIFIED_PROBLEM_SEAL:
            raise CertificateInvalidError(
                "certificate_invalid: certified snapshot requires the Task-12 factory"
            )
        _require_digest(
            self.source_verification_digest,
            "$_CertifiedZ2ProblemSnapshot.source_verification_digest",
        )
        _require_digest(self.problem_id, "$_CertifiedZ2ProblemSnapshot.problem_id")


@dataclass(frozen=True, slots=True)
class _CertifiedZ2ProblemFactoryRecord:
    cochain_problem: CertifiedCochainProblem
    authority: Task5VerificationAuthority
    branches: tuple[Z2Branch, ...]
    trusted_release_attestations: tuple[LauncherExecutionAttestation, ...]
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    source_verification_digest: str
    problem_id: str


_CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS: dict[
    int,
    tuple[
        weakref.ReferenceType[_CertifiedZ2ProblemSnapshot],
        _CertifiedZ2ProblemFactoryRecord,
    ],
] = {}


def _register_certified_z2_snapshot(
    snapshot: _CertifiedZ2ProblemSnapshot,
) -> None:
    key = id(snapshot)

    def discard(reference: weakref.ReferenceType[_CertifiedZ2ProblemSnapshot]) -> None:
        current = _CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS.get(key)
        if current is not None and current[0] is reference:
            _CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS.pop(key, None)

    identity = weakref.ref(snapshot, discard)
    _CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS[key] = (
        identity,
        _CertifiedZ2ProblemFactoryRecord(
            snapshot.cochain_problem,
            snapshot.authority,
            snapshot.branches,
            snapshot.trusted_release_attestations,
            snapshot.spatial_character_basis,
            snapshot.spatial_resolution,
            snapshot.source_verification_digest,
            snapshot.problem_id,
        ),
    )


def _certified_z2_snapshot_record(
    snapshot: _CertifiedZ2ProblemSnapshot,
) -> _CertifiedZ2ProblemFactoryRecord | None:
    issued = _CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS.get(id(snapshot))
    if issued is None or issued[0]() is not snapshot:
        return None
    record = issued[1]
    if (
        snapshot._factory_token is not _CERTIFIED_PROBLEM_SEAL
        or snapshot.cochain_problem is not record.cochain_problem
        or snapshot.authority is not record.authority
        or snapshot.branches is not record.branches
        or snapshot.trusted_release_attestations
        is not record.trusted_release_attestations
        or snapshot.spatial_character_basis is not record.spatial_character_basis
        or snapshot.spatial_resolution is not record.spatial_resolution
        or snapshot.source_verification_digest
        != record.source_verification_digest
        or snapshot.problem_id != record.problem_id
    ):
        return None
    return record


@dataclass(frozen=True, slots=True)
class CertifiedZ2Problem:
    problem_id: str
    cochain_problem: CertifiedCochainProblem
    authority: Task5VerificationAuthority
    branches: tuple[Z2Branch, ...]
    source_verification_digest: str
    _seal: object

    def __post_init__(self) -> None:
        if type(self._seal) is not _CertifiedZ2ProblemSnapshot:
            raise CertificateInvalidError(
                "certificate_invalid: CertifiedZ2Problem must come from the Task-12 factory"
            )
        snapshot = self._seal
        record = _certified_z2_snapshot_record(snapshot)
        if (
            record is None
            or self.cochain_problem is not snapshot.cochain_problem
            or self.authority is not snapshot.authority
            or self.branches is not snapshot.branches
            or self.source_verification_digest
            != snapshot.source_verification_digest
            or self.problem_id != snapshot.problem_id
        ):
            raise CertificateInvalidError(
                "certificate_invalid: certified problem differs from its source snapshot"
            )
        _require_digest(self.problem_id, "$CertifiedZ2Problem.problem_id")
        _require_digest(
            self.source_verification_digest,
            "$CertifiedZ2Problem.source_verification_digest",
        )
        if type(self.cochain_problem) is not CertifiedCochainProblem:
            raise TypeError("$CertifiedZ2Problem.cochain_problem: invalid Task-5 source")
        if type(self.authority) is not Task5VerificationAuthority:
            raise TypeError("$CertifiedZ2Problem.authority: invalid Task-5 authority")
        branches = self.branches
        if not branches or any(type(branch) is not Z2Branch or branch.diagnostic for branch in branches):
            raise ValueError("$CertifiedZ2Problem.branches: expected release-certified branches")
        object.__setattr__(self, "branches", branches)


def _task5_authority_mapping(
    authority: Task5VerificationAuthority,
) -> dict[str, object]:
    return {
        "affine_pcp_certificate_digest": authority.affine_pcp_certificate_digest,
        "backend_environment_id": authority.backend_environment_id,
        "backend_lock_digest": authority.backend_lock_digest,
        "catalogue_action_digest": authority.catalogue_action_digest,
        "catalogue_record_digest": authority.catalogue_record_digest,
        "inclusions": [
            {
                "diagnostic_backend": item.diagnostic_backend,
                "diagnostic_failure_degrees": list(item.diagnostic_failure_degrees),
                "diagnostic_outcome": item.diagnostic_outcome,
                "diagnostic_residue_digests": list(item.diagnostic_residue_digests),
                "gap_inclusion_projection_digest": item.gap_inclusion_projection_digest,
                "inclusion_id": item.inclusion_id,
                "launcher_attestation_id": item.launcher_attestation_id,
                "literal_element_digest": item.literal_element_digest,
                "literal_stabilizer_digest": item.literal_stabilizer_digest,
                "source_bar_equivalence_id": item.source_bar_equivalence_id,
                "target_bar_equivalence_id": item.target_bar_equivalence_id,
                "transported_inclusion_digest": item.transported_inclusion_digest,
            }
            for item in authority.inclusions
        ],
        "runtime_provenance_digest": authority.runtime_provenance_digest,
    }


def _canonical_release_branch_index(
    branches: Sequence[Z2Branch],
    verified_inclusion_ids: Sequence[str] | frozenset[str] | set[str],
) -> tuple[Z2Branch, ...]:
    """Validate the canonical whole-tuple branch index and its Task-5 coverage."""

    normalized = tuple(branches)
    expected_inclusions = frozenset(verified_inclusion_ids)
    if (
        len({branch.branch_id for branch in normalized}) != len(normalized)
        or len({branch.skeleton_ids for branch in normalized}) != len(normalized)
    ):
        raise CertificateInvalidError(
            "certificate_invalid: release branches and skeleton tuples must be unique"
        )
    canonical = tuple(
        sorted(normalized, key=lambda branch: (branch.skeleton_ids, branch.branch_id))
    )
    if normalized != canonical:
        raise CertificateInvalidError(
            "certificate_invalid: release branch tuples are not in canonical order"
        )
    for branch in normalized:
        branch_inclusions = tuple(branch.inclusion_ids)
        if frozenset(branch_inclusions) != expected_inclusions:
            raise CertificateInvalidError(
                "certificate_invalid: every release branch must cover all unique Task-5 capabilities"
            )
    return normalized


def _validate_certified_z2_source(
    cochain_problem: CertifiedCochainProblem,
    authority: Task5VerificationAuthority,
    branches: Sequence[Z2Branch],
    *,
    trusted_release_attestations: Sequence[LauncherExecutionAttestation],
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
) -> tuple[
    tuple[Z2Branch, ...],
    tuple[LauncherExecutionAttestation, ...],
    str,
    str,
]:
    """Replay the complete nonserialized Task-5 and relative source boundary."""

    if type(cochain_problem) is not CertifiedCochainProblem:
        raise TypeError("make_certified_z2_problem requires CertifiedCochainProblem")
    if type(authority) is not Task5VerificationAuthority:
        raise TypeError("make_certified_z2_problem requires Task5VerificationAuthority")
    resolution_report = verify_resolution(cochain_problem.ambient, authority)
    if not resolution_report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: ambient resolution: {resolution_report.issues[0].code}"
        )
    _require_spatial_character_parent(
        cochain_problem.character_basis,
        spatial_character_basis,
        spatial_resolution,
    )
    character_report = verify_character_basis(
        cochain_problem.character_basis,
        cochain_problem.ambient,
        authority,
        spatial_certificate=spatial_character_basis,
        spatial_resolution=spatial_resolution,
    )
    if not character_report.valid:
        raise CertificateInvalidError(
            f"certificate_invalid: character basis: {character_report.issues[0].code}"
        )
    attestations = tuple(trusted_release_attestations)
    if any(type(item) is not LauncherExecutionAttestation for item in attestations):
        raise TypeError("trusted release attestations must be typed Task-5 results")
    by_id = {attestation.attestation_id: attestation for attestation in attestations}
    if len(by_id) != len(attestations):
        raise CertificateInvalidError("certificate_invalid: duplicate release attestation")
    if set(by_id) != {
        inclusion.launcher_attestation.attestation_id
        for inclusion in cochain_problem.inclusions
    }:
        raise CertificateInvalidError(
            "certificate_invalid: release attestations must exactly cover the Task-5 inclusions"
        )
    inclusion_ids: set[str] = set()
    for inclusion in cochain_problem.inclusions:
        trusted = by_id.get(inclusion.launcher_attestation.attestation_id)
        report = verify_inclusion_chain_map(
            inclusion,
            authority,
            require_release=True,
            trusted_release_attestation=trusted,
        )
        if not report.valid:
            raise CertificateInvalidError(
                f"certificate_invalid: inclusion {inclusion.inclusion_id}: {report.issues[0].code}"
            )
        inclusion_ids.add(inclusion.inclusion_id)
    if len(inclusion_ids) != len(cochain_problem.inclusions):
        raise CertificateInvalidError("certificate_invalid: duplicate Task-5 inclusion ID")
    normalized = tuple(branches)
    if not normalized or any(type(branch) is not Z2Branch or branch.diagnostic for branch in normalized):
        raise CertificateInvalidError(
            "certificate_invalid: production envelope requires release-certified branches"
        )
    normalized = _canonical_release_branch_index(normalized, inclusion_ids)
    inclusions_by_id = {item.inclusion_id: item for item in cochain_problem.inclusions}
    expected_ambient = _certified_gf2_complex(cochain_problem.ambient)
    for untrusted_branch in normalized:
        try:
            branch = _replay_z2_branch(untrusted_branch)
        except CertificateInvalidError:
            raise
        except (TypeError, ValueError) as error:
            raise CertificateInvalidError(
                f"certificate_invalid: branch replay failed: {error}"
            ) from error
        if branch != untrusted_branch:
            raise CertificateInvalidError(
                "certificate_invalid: branch replay differs from the stored branch"
            )
        if branch.cross_skeleton_arrows:
            raise CertificateInvalidError(
                "certificate_invalid: release cross-skeleton arrows require Task-13 LocalConjugacy replay"
            )
        if branch.h1_unmarking.character_basis_certificate_id != cochain_problem.character_basis.certificate_id:
            raise CertificateInvalidError(
                "certificate_invalid: H1 diagnostic is not bound to the Task-5 character basis"
            )
        if branch.source_problem.ambient != expected_ambient:
            raise CertificateInvalidError(
                "certificate_invalid: relative ambient is not the literal Task-5 GF(2) complex"
            )
        source_by_instance = {
            restriction.instance_id: (local, restriction)
            for local, restriction in zip(
                branch.source_problem.locals,
                branch.source_problem.restrictions,
                strict=True,
            )
        }
        if set(source_by_instance) != set(branch.matrices.coordinate_blocks.instance_ids):
            raise CertificateInvalidError(
                "certificate_invalid: relative source instances are not canonical"
            )
        replayed_centralizers: list[CertifiedCentralizerAction] = []
        for inclusion_id, binding in zip(
            branch.inclusion_ids,
            branch.orbit_bindings,
            strict=True,
        ):
            inclusion = inclusions_by_id[inclusion_id]
            if binding.inclusion_id != inclusion_id:
                raise CertificateInvalidError(
                    "certificate_invalid: orbit binding refers to another Task-5 inclusion"
                )
            local, restriction = source_by_instance[binding.instance_id]
            expected_local = _certified_gf2_complex(inclusion.source_resolution)
            expected_restriction = _certified_gf2_restriction(
                inclusion,
                instance_id=binding.instance_id,
                ambient=expected_ambient,
                local=expected_local,
            )
            if local != expected_local or restriction != expected_restriction:
                raise CertificateInvalidError(
                    "certificate_invalid: branch orbit does not replay its literal Task-5 inclusion"
                )
            if binding.bar_equivalence.resolution_id != inclusion.source_resolution_id:
                raise CertificateInvalidError(
                    "certificate_invalid: local bar trace differs from the literal inclusion source"
                )
            defect_report = verify_z2_defect_coordinates(
                binding.defect_coordinates,
                binding.skeleton,
                binding.bar_equivalence,
                authority,
            )
            if not defect_report.valid:
                raise CertificateInvalidError(
                    "certificate_invalid: local defect coordinate certificate does not replay"
                )
            complete_actions = _canonical_complete_centralizer_actions(
                binding.skeleton,
                binding.centralizer_actions,
            )
            for component, action in zip(
                binding.skeleton.centralizer_components,
                complete_actions,
                strict=True,
            ):
                if action.diagnostic:
                    raise CertificateInvalidError(
                        "certificate_invalid: diagnostic component action entered a release branch"
                    )
                expected_coordinates = coordinate_z2_marking_shift(
                    component.marking_shift,
                    binding.bar_equivalence,
                    authority,
                )
                if (
                    action.coordinate_certificate_id
                    != expected_coordinates.certificate_id
                    or action.local_coordinates != expected_coordinates.coordinates
                ):
                    raise CertificateInvalidError(
                        "certificate_invalid: component coordinate trace does not replay"
                    )
                expected_action = certify_centralizer_action(
                    branch.matrices,
                    skeleton_ids=branch.skeleton_ids,
                    instance_id=binding.instance_id,
                    skeleton=binding.skeleton,
                    component=component,
                    marking_coordinates=expected_coordinates,
                    bar_equivalence=binding.bar_equivalence,
                    authority=authority,
                )
                if action != expected_action:
                    raise CertificateInvalidError(
                        "certificate_invalid: component quotient action does not replay"
                    )
                replayed_centralizers.append(expected_action)
        expected_h1 = certify_task5_h1_unmarking(
            branch.matrices,
            character_basis=cochain_problem.character_basis,
            inclusions=cochain_problem.inclusions,
            orbit_bindings=branch.orbit_bindings,
            authority=authority,
            spatial_character_basis=spatial_character_basis,
            spatial_resolution=spatial_resolution,
        )
        if branch.h1_unmarking != expected_h1:
            raise CertificateInvalidError(
                "certificate_invalid: H1 unmarking differs from the independent Task-5 restriction map"
            )
        if tuple(replayed_centralizers) != branch.centralizer_actions:
            raise CertificateInvalidError(
                "certificate_invalid: branch component-action order differs from its orbit bindings"
            )
        expected_branch = make_certified_z2_branch(
            source_problem=branch.source_problem,
            matrices=branch.matrices,
            orbit_bindings=branch.orbit_bindings,
            h1_unmarking=branch.h1_unmarking,
            cross_skeleton_arrows=branch.cross_skeleton_arrows,
        )
        if branch != expected_branch:
            raise CertificateInvalidError(
                "certificate_invalid: release branch does not replay its certified inputs"
            )
    source_verification_digest = _digest(
        "task5-source-verification",
        {
            "ambient_resolution_id": cochain_problem.ambient.resolution_id,
            "authority": _task5_authority_mapping(authority),
            "branch_ids": sorted(branch.branch_id for branch in normalized),
            "character_basis_certificate_id": cochain_problem.character_basis.certificate_id,
            "inclusion_certificate_ids": sorted(
                inclusion.certificate_id for inclusion in cochain_problem.inclusions
            ),
            "release_attestation_ids": sorted(by_id),
            "spatial_character_basis_id": (
                None
                if spatial_character_basis is None
                else spatial_character_basis.certificate_id
            ),
            "spatial_resolution_id": (
                None
                if spatial_resolution is None
                else spatial_resolution.resolution_id
            ),
        },
    )
    problem_id = _digest(
        "certified-z2-problem",
        {
            "branch_ids": sorted(branch.branch_id for branch in normalized),
            "source_verification_digest": source_verification_digest,
        },
    )
    return normalized, attestations, source_verification_digest, problem_id


def _replay_release_stratum_snapshot(
    stored: _ReleaseZ2StratumSnapshot,
    *,
    certificate_id: str,
    relative_certificate_id: str,
    skeleton_ids: tuple[str, ...],
    problem_id: str,
    source_verification_digest: str,
) -> tuple[_ReleaseZ2StratumSnapshot, Z2Branch]:
    """Rebuild release provenance from independently reverified Task-5 inputs."""

    if (
        type(stored) is not _ReleaseZ2StratumSnapshot
        or stored._factory_token is not _Z2_STRATUM_RELEASE_SEAL
        or type(stored.source_snapshot) is not _CertifiedZ2ProblemSnapshot
        or _certified_z2_snapshot_record(stored.source_snapshot) is None
    ):
        raise CertificateInvalidError(
            "certificate_invalid: release stratum source snapshot is not factory certified"
        )
    source = stored.source_snapshot
    try:
        normalized, attestations, computed_source_digest, computed_problem_id = (
            _validate_certified_z2_source(
                source.cochain_problem,
                source.authority,
                source.branches,
                trusted_release_attestations=(
                    source.trusted_release_attestations
                ),
                spatial_character_basis=source.spatial_character_basis,
                spatial_resolution=source.spatial_resolution,
            )
        )
        replayed_branches = tuple(
            _replay_z2_branch(branch) for branch in normalized
        )
    except CertificateInvalidError:
        raise
    except (TypeError, ValueError, KeyError, IndexError) as error:
        raise CertificateInvalidError(
            f"certificate_invalid: release stratum source replay failed: {error}"
        ) from error
    if (
        normalized != source.branches
        or replayed_branches != normalized
        or attestations != source.trusted_release_attestations
        or computed_source_digest != source.source_verification_digest
        or computed_problem_id != source.problem_id
        or stored.certificate_id != certificate_id
        or stored.relative_certificate_id != relative_certificate_id
        or stored.skeleton_ids != skeleton_ids
        or stored.problem_id != problem_id
        or stored.source_verification_digest != source_verification_digest
        or source.problem_id != problem_id
        or source.source_verification_digest != source_verification_digest
    ):
        raise CertificateInvalidError(
            "certificate_invalid: release stratum source differs after independent replay"
        )
    matching = tuple(
        branch
        for branch in replayed_branches
        if branch.matrices.certificate.certificate_id
        == relative_certificate_id
        and branch.skeleton_ids == skeleton_ids
    )
    if len(matching) != 1:
        raise CertificateInvalidError(
            "certificate_invalid: release stratum must match exactly one certified branch"
        )
    replayed_source = _CertifiedZ2ProblemSnapshot(
        cochain_problem=source.cochain_problem,
        authority=source.authority,
        branches=replayed_branches,
        trusted_release_attestations=attestations,
        spatial_character_basis=source.spatial_character_basis,
        spatial_resolution=source.spatial_resolution,
        source_verification_digest=computed_source_digest,
        problem_id=computed_problem_id,
        _factory_token=_CERTIFIED_PROBLEM_SEAL,
    )
    _register_certified_z2_snapshot(replayed_source)
    return (
        _ReleaseZ2StratumSnapshot(
            certificate_id=certificate_id,
            relative_certificate_id=relative_certificate_id,
            skeleton_ids=skeleton_ids,
            problem_id=problem_id,
            source_verification_digest=source_verification_digest,
            source_snapshot=replayed_source,
            _factory_token=_Z2_STRATUM_RELEASE_SEAL,
        ),
        matching[0],
    )


def make_certified_z2_problem(
    cochain_problem: CertifiedCochainProblem,
    authority: Task5VerificationAuthority,
    branches: Sequence[Z2Branch],
    *,
    trusted_release_attestations: Sequence[LauncherExecutionAttestation],
    spatial_character_basis: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
) -> CertifiedZ2Problem:
    """Bind release authority to exact Task-5, relative, and local sources."""

    try:
        normalized, attestations, source_digest, problem_id = (
            _validate_certified_z2_source(
                cochain_problem,
                authority,
                branches,
                trusted_release_attestations=trusted_release_attestations,
                spatial_character_basis=spatial_character_basis,
                spatial_resolution=spatial_resolution,
            )
        )
    except CertificateInvalidError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise CertificateInvalidError(
            f"certificate_invalid: Task-5 source replay failed: {error}"
        ) from error
    snapshot = _CertifiedZ2ProblemSnapshot(
        cochain_problem=cochain_problem,
        authority=authority,
        branches=normalized,
        trusted_release_attestations=attestations,
        spatial_character_basis=spatial_character_basis,
        spatial_resolution=spatial_resolution,
        source_verification_digest=source_digest,
        problem_id=problem_id,
        _factory_token=_CERTIFIED_PROBLEM_SEAL,
    )
    _register_certified_z2_snapshot(snapshot)
    try:
        return CertifiedZ2Problem(
            problem_id=problem_id,
            cochain_problem=cochain_problem,
            authority=authority,
            branches=normalized,
            source_verification_digest=source_digest,
            _seal=snapshot,
        )
    except BaseException:
        _CERTIFIED_Z2_PROBLEM_FACTORY_RECORDS.pop(id(snapshot), None)
        raise


def _classify_branches(
    branches: Sequence[Z2Branch],
    *,
    provenance: str,
    problem_id: str,
    source_verification_digest: str | None,
    release_source_snapshot: _CertifiedZ2ProblemSnapshot | None,
) -> tuple[FiniteAffineStratum | ObstructedBranch, ...]:
    results = tuple(
        _solve_z2_branch_diagnostic(
            branch.source_problem,
            branch.matrices,
            branch.skeleton_ids,
            centralizer_actions=branch.centralizer_actions,
            cross_skeleton_arrows=branch.cross_skeleton_arrows,
            h1_unmarking=branch.h1_unmarking,
            provenance=provenance,
            problem_id=problem_id,
            source_verification_digest=source_verification_digest,
            _release_source_snapshot=release_source_snapshot,
        )
        for branch in branches
    )
    return tuple(sorted(results, key=lambda item: (item.skeleton_ids, item.stratum_id)))


def classify_z2_diagnostic(
    problem: DiagnosticZ2Problem,
) -> tuple[FiniteAffineStratum | ObstructedBranch, ...]:
    if type(problem) is not DiagnosticZ2Problem:
        raise TypeError("classify_z2_diagnostic requires DiagnosticZ2Problem")
    return _classify_branches(
        problem.branches,
        provenance="diagnostic",
        problem_id=problem.problem_id,
        source_verification_digest=None,
        release_source_snapshot=None,
    )


def classify_z2(
    problem: CertifiedZ2Problem | CertifiedCochainProblem,
) -> tuple[FiniteAffineStratum | ObstructedBranch, ...]:
    if type(problem) is DiagnosticZ2Problem:
        raise CertificateInvalidError(
            "certificate_invalid: diagnostic Z2 envelope is not release authority"
        )
    if type(problem) is not CertifiedZ2Problem or type(problem._seal) is not _CertifiedZ2ProblemSnapshot:
        raise CertificateInvalidError(
            "certificate_invalid: classify_z2 requires the Task-12 certified envelope"
        )
    snapshot = problem._seal
    if _certified_z2_snapshot_record(snapshot) is None:
        raise CertificateInvalidError(
            "certificate_invalid: certified source snapshot differs from its factory authority"
        )
    try:
        CertifiedZ2Problem(
            problem_id=problem.problem_id,
            cochain_problem=problem.cochain_problem,
            authority=problem.authority,
            branches=problem.branches,
            source_verification_digest=problem.source_verification_digest,
            _seal=snapshot,
        )
        normalized, attestations, source_digest, problem_id = (
            _validate_certified_z2_source(
                problem.cochain_problem,
                problem.authority,
                problem.branches,
                trusted_release_attestations=(
                    snapshot.trusted_release_attestations
                ),
                spatial_character_basis=snapshot.spatial_character_basis,
                spatial_resolution=snapshot.spatial_resolution,
            )
        )
    except CertificateInvalidError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise CertificateInvalidError(
            f"certificate_invalid: certified source snapshot replay failed: {error}"
        ) from error
    if (
        normalized is not snapshot.branches
        or attestations != snapshot.trusted_release_attestations
        or source_digest != problem.source_verification_digest
        or problem_id != problem.problem_id
    ):
        raise CertificateInvalidError(
            "certificate_invalid: certified source snapshot differs after replay"
        )
    return _classify_branches(
        problem.branches,
        provenance="release",
        problem_id=problem.problem_id,
        source_verification_digest=problem.source_verification_digest,
        release_source_snapshot=snapshot,
    )


def _certificate_core(
    *,
    matrices: RelativeMatrices,
    skeleton_ids: tuple[str, ...],
    solution: GF2AffineSolution,
    boundaries: tuple[tuple[int, ...], ...],
    representatives: tuple[tuple[int, ...], ...],
    h1_diagnostic_id: str,
    residual_action_ids: tuple[str, ...],
    centralizer_action_ids: tuple[str, ...],
    cross_skeleton_arrow_ids: tuple[str, ...],
    provenance: str,
    problem_id: str | None,
    source_verification_digest: str | None,
) -> dict[str, object]:
    return {
        "basepoint": list(solution.basepoint),
        "boundary_basis": [list(vector) for vector in boundaries],
        "centralizer_action_ids": list(centralizer_action_ids),
        "cross_skeleton_arrow_ids": list(cross_skeleton_arrow_ids),
        "h1_diagnostic_id": h1_diagnostic_id,
        "h1_unmarking_passes": 1,
        "kernel_basis": [list(vector) for vector in solution.kernel_basis],
        "quotient_basis": [list(vector) for vector in representatives],
        "relative_certificate_id": matrices.certificate.certificate_id,
        "relative_problem_digest": matrices.certificate.problem_digest,
        "problem_id": problem_id,
        "provenance": provenance,
        "residual_action_ids": list(residual_action_ids),
        "skeleton_ids": list(skeleton_ids),
        "source_verification_digest": source_verification_digest,
    }


def _solve_z2_branch_diagnostic(
    source_problem: RelativeProblem,
    matrices: RelativeMatrices,
    skeleton_ids: Sequence[str],
    *,
    residual_actions: Sequence[GF2AffineArrow] = (),
    cross_skeleton_arrow_ids: Sequence[str] = (),
    centralizer_actions: Sequence[CertifiedCentralizerAction] = (),
    cross_skeleton_arrows: Sequence[Z2CrossSkeletonArrow] = (),
    h1_unmarking: H1UnmarkingCertificate | None = None,
    provenance: str = "diagnostic-direct",
    problem_id: str | None = None,
    source_verification_digest: str | None = None,
    _release_source_snapshot: _CertifiedZ2ProblemSnapshot | None = None,
) -> FiniteAffineStratum | ObstructedBranch:
    """Diagnostic algebra boundary; production callers must use ``classify_z2``."""

    if type(source_problem) is not RelativeProblem or source_problem.ring != "gf2":
        raise TypeError("diagnostic Z2 solve requires a GF(2) RelativeProblem")
    if type(matrices) is not RelativeMatrices or type(matrices.D) is not MatrixGF2:
        raise TypeError("diagnostic Z2 solve requires certified GF(2) relative matrices")
    verify_relative_certificate(matrices, source_problem)
    skeletons = _skeleton_ids(skeleton_ids)
    if provenance == "release":
        if (
            type(_release_source_snapshot) is not _CertifiedZ2ProblemSnapshot
            or _release_source_snapshot._factory_token is not _CERTIFIED_PROBLEM_SEAL
            or problem_id != _release_source_snapshot.problem_id
            or source_verification_digest
            != _release_source_snapshot.source_verification_digest
        ):
            raise CertificateInvalidError(
                "certificate_invalid: release solve lacks its certified source snapshot"
            )
    elif _release_source_snapshot is not None:
        raise CertificateInvalidError(
            "certificate_invalid: diagnostic solve cannot carry a release source snapshot"
        )
    solution = solve_affine(matrices.D, tuple(matrices.offset))
    if isinstance(solution, GF2Inconsistency):
        witness = FrozenJSONObject(
            (
                ("kind", "gf2-left-null"),
                ("left_null_vector", FrozenJSONArray(solution.left_null_vector)),
                ("offset_pairing", 1),
                ("problem_id", problem_id),
                ("provenance", provenance),
                ("relative_certificate_id", matrices.certificate.certificate_id),
                ("relative_problem_digest", matrices.certificate.problem_digest),
                ("source_verification_digest", source_verification_digest),
            )
        )
        return ObstructedBranch(
            _digest(
                "obstructed-z2-branch",
                {
                    "left_null_vector": list(solution.left_null_vector),
                    "problem_id": problem_id,
                    "provenance": provenance,
                    "relative_certificate_id": matrices.certificate.certificate_id,
                    "skeleton_ids": list(skeletons),
                    "source_verification_digest": source_verification_digest,
                },
            ),
            skeletons,
            witness,
        )
    ambient = matrices.D.column_count
    boundaries = image_basis(matrices.B)
    quotient = quotient_basis(
        _columns_matrix(solution.kernel_basis, ambient),
        matrices.B,
    )
    certified_actions = tuple(centralizer_actions)
    for action in certified_actions:
        if type(action) is not CertifiedCentralizerAction:
            raise TypeError("centralizer_actions must be CertifiedCentralizerAction values")
        if (
            action.relative_certificate_id != matrices.certificate.certificate_id
            or action.skeleton_id not in skeletons
        ):
            raise CertificateInvalidError(
                "certificate_invalid: centralizer action is not bound to this stratum"
            )
    supplied_actions = tuple(residual_actions) + tuple(
        action.quotient_action for action in certified_actions
    )
    for action in supplied_actions:
        if type(action) is not GF2AffineArrow:
            raise TypeError("residual actions must be GF2AffineArrow values")
        if (
            action.source_dimension != quotient.dimension
            or action.target_dimension != quotient.dimension
            or kernel_basis(action.linear)
        ):
            raise CertificateInvalidError(
                "certificate_invalid: residual action must be an invertible quotient endomorphism"
            )
    actions_by_id = {_arrow_id(action): action for action in supplied_actions}
    action_ids = tuple(sorted(actions_by_id))
    actions = tuple(actions_by_id[identifier] for identifier in action_ids)
    cross_boundaries = tuple(cross_skeleton_arrows)
    for boundary in cross_boundaries:
        if type(boundary) is not Z2CrossSkeletonArrow:
            raise TypeError("cross_skeleton_arrows must be Z2CrossSkeletonArrow values")
        if boundary.source_skeleton_ids != skeletons:
            raise CertificateInvalidError(
                "certificate_invalid: cross-skeleton boundary has the wrong source"
            )
        if (
            boundary.quotient_action.source_dimension != quotient.dimension
            or boundary.quotient_action.target_dimension != quotient.dimension
            or kernel_basis(boundary.quotient_action.linear)
        ):
            raise CertificateInvalidError(
                "certificate_invalid: cross-skeleton boundary must be an invertible quotient arrow"
            )
    if tuple(cross_skeleton_arrow_ids):
        raise CertificateInvalidError(
            "certificate_invalid: cross-skeleton IDs require typed groupoid arrows"
        )
    centralizer_ids = tuple(sorted(item.action_id for item in certified_actions))
    cross_ids = tuple(sorted(item.arrow_id for item in cross_boundaries))
    for item in cross_ids:
        _require_digest(item, "cross_skeleton_arrow_ids")
    if h1_unmarking is None:
        start, stop = matrices.coordinate_blocks.ambient_slices[0]
        diagnostic_columns = tuple(
            tuple(matrices.B[row][column] for row in range(matrices.B.row_count))
            for column in range(start, stop)
        )
        h1_unmarking = certify_h1_unmarking(
            matrices,
            character_basis_id=_digest(
                "diagnostic-character-basis",
                {"ambient_complex_id": matrices.certificate.ambient_complex_id},
            ),
            diagnostic_boundary_columns=diagnostic_columns,
        )
    if type(h1_unmarking) is not H1UnmarkingCertificate:
        raise TypeError("h1_unmarking must be an H1UnmarkingCertificate")
    if h1_unmarking.relative_certificate_id != matrices.certificate.certificate_id:
        raise CertificateInvalidError(
            "certificate_invalid: H1 diagnostic binds another relative problem"
        )
    h1_diagnostic_id = h1_unmarking.certificate_id
    core = _certificate_core(
        matrices=matrices,
        skeleton_ids=skeletons,
        solution=solution,
        boundaries=boundaries,
        representatives=quotient.representatives,
        h1_diagnostic_id=h1_diagnostic_id,
        residual_action_ids=action_ids,
        centralizer_action_ids=centralizer_ids,
        cross_skeleton_arrow_ids=cross_ids,
        provenance=provenance,
        problem_id=problem_id,
        source_verification_digest=source_verification_digest,
    )
    certificate_id = _digest("z2-stratum-certificate", core)
    release_snapshot = None
    if provenance == "release":
        assert problem_id is not None
        assert source_verification_digest is not None
        assert _release_source_snapshot is not None
        release_snapshot = _ReleaseZ2StratumSnapshot(
            certificate_id=certificate_id,
            relative_certificate_id=matrices.certificate.certificate_id,
            skeleton_ids=skeletons,
            problem_id=problem_id,
            source_verification_digest=source_verification_digest,
            source_snapshot=_release_source_snapshot,
            _factory_token=_Z2_STRATUM_RELEASE_SEAL,
        )
    certificate = Z2StratumCertificate(
        certificate_id=certificate_id,
        relative_certificate_id=matrices.certificate.certificate_id,
        relative_problem_digest=matrices.certificate.problem_digest,
        provenance=provenance,
        problem_id=problem_id,
        source_verification_digest=source_verification_digest,
        skeleton_ids=skeletons,
        basepoint=solution.basepoint,
        kernel_basis=solution.kernel_basis,
        boundary_basis=boundaries,
        quotient_basis=quotient.representatives,
        h1_unmarking_passes=1,
        h1_diagnostic_id=h1_diagnostic_id,
        h1_unmarking=h1_unmarking,
        residual_action_ids=action_ids,
        centralizer_action_ids=centralizer_ids,
        cross_skeleton_arrow_ids=cross_ids,
        centralizer_actions=certified_actions,
        cross_skeleton_arrows=cross_boundaries,
        matrices=matrices,
        _release_snapshot=release_snapshot,
    )
    stratum_id = _digest(
        "finite-affine-stratum",
        {
            "certificate_id": certificate.certificate_id,
            "skeleton_ids": list(skeletons),
        },
    )
    return FiniteAffineStratum(
        stratum_id,
        skeletons,
        solution.basepoint,
        quotient.representatives,
        quotient.dimension,
        actions,
        certificate,
    )


def apply_h1_unmarking(
    stratum: FiniteAffineStratum,
    certificate: H1UnmarkingCertificate,
) -> FiniteAffineStratum:
    """Reject a second quotient: every public stratum is already unmarked."""

    if type(stratum) is not FiniteAffineStratum:
        raise TypeError("H1 unmarking requires FiniteAffineStratum")
    if type(certificate) is not H1UnmarkingCertificate:
        raise TypeError("H1 unmarking requires H1UnmarkingCertificate")
    if stratum.certificate.h1_diagnostic_id != certificate.certificate_id:
        raise CertificateInvalidError(
            "certificate_invalid: H1 diagnostic differs from the stored quotient"
        )
    raise CertificateInvalidError(
        "certificate_invalid: H1 unmarking is already applied exactly once"
    )


def _replay_affine_arrow(arrow: GF2AffineArrow) -> GF2AffineArrow:
    if type(arrow) is not GF2AffineArrow:
        raise TypeError("expected GF2AffineArrow")
    return GF2AffineArrow(
        MatrixGF2(arrow.linear.rows, column_count=arrow.linear.column_count),
        arrow.shift,
    )


def _replay_finite_affine_stratum(
    stratum: FiniteAffineStratum,
) -> FiniteAffineStratum:
    if type(stratum) is not FiniteAffineStratum:
        raise TypeError("expected FiniteAffineStratum")
    stored = stratum.certificate
    if type(stored) is not Z2StratumCertificate:
        raise TypeError("expected Z2StratumCertificate")
    release_snapshot = None
    matching_branch = None
    if stored.provenance == "release":
        if (
            type(stored.problem_id) is not str
            or type(stored.source_verification_digest) is not str
        ):
            raise CertificateInvalidError(
                "certificate_invalid: release stratum lacks source provenance"
            )
        release_snapshot, matching_branch = (
            _replay_release_stratum_snapshot(
                stored._release_snapshot,
                certificate_id=stored.certificate_id,
                relative_certificate_id=stored.relative_certificate_id,
                skeleton_ids=stored.skeleton_ids,
                problem_id=stored.problem_id,
                source_verification_digest=(
                    stored.source_verification_digest
                ),
            )
        )
    certificate = Z2StratumCertificate(
        certificate_id=stored.certificate_id,
        relative_certificate_id=stored.relative_certificate_id,
        relative_problem_digest=stored.relative_problem_digest,
        provenance=stored.provenance,
        problem_id=stored.problem_id,
        source_verification_digest=stored.source_verification_digest,
        skeleton_ids=stored.skeleton_ids,
        basepoint=stored.basepoint,
        kernel_basis=stored.kernel_basis,
        boundary_basis=stored.boundary_basis,
        quotient_basis=stored.quotient_basis,
        h1_unmarking_passes=stored.h1_unmarking_passes,
        h1_diagnostic_id=stored.h1_diagnostic_id,
        h1_unmarking=_replay_h1_unmarking(stored.h1_unmarking),
        residual_action_ids=stored.residual_action_ids,
        centralizer_action_ids=stored.centralizer_action_ids,
        cross_skeleton_arrow_ids=stored.cross_skeleton_arrow_ids,
        centralizer_actions=tuple(
            _replay_centralizer_action(action)
            for action in stored.centralizer_actions
        ),
        cross_skeleton_arrows=tuple(
            _replay_cross_skeleton_arrow(arrow)
            for arrow in stored.cross_skeleton_arrows
        ),
        matrices=stored.matrices,
        _release_snapshot=release_snapshot,
    )
    replayed = FiniteAffineStratum(
        stratum_id=stratum.stratum_id,
        skeleton_ids=stratum.skeleton_ids,
        basepoint=stratum.basepoint,
        homogeneous_basis=stratum.homogeneous_basis,
        quotient_dimension=stratum.quotient_dimension,
        residual_actions=tuple(
            _replay_affine_arrow(action) for action in stratum.residual_actions
        ),
        certificate=certificate,
    )
    if matching_branch is not None:
        assert release_snapshot is not None
        expected = _solve_z2_branch_diagnostic(
            matching_branch.source_problem,
            matching_branch.matrices,
            matching_branch.skeleton_ids,
            centralizer_actions=matching_branch.centralizer_actions,
            cross_skeleton_arrows=matching_branch.cross_skeleton_arrows,
            h1_unmarking=matching_branch.h1_unmarking,
            provenance="release",
            problem_id=release_snapshot.problem_id,
            source_verification_digest=(
                release_snapshot.source_verification_digest
            ),
            _release_source_snapshot=release_snapshot.source_snapshot,
        )
        if type(expected) is not FiniteAffineStratum or replayed != expected:
            raise CertificateInvalidError(
                "certificate_invalid: release stratum content differs from its uniquely matched branch"
            )
    return replayed


def _verify_and_issue_z2_point_batch(
    stratum: FiniteAffineStratum,
) -> tuple[FiniteAffineStratum, _CertifiedZ2PointBatchAuthority]:
    """Verify one complete stratum, then issue an ephemeral enumeration session."""

    try:
        checked = _replay_finite_affine_stratum(stratum)
    except CertificateInvalidError:
        raise
    except (TypeError, ValueError) as error:
        raise CertificateInvalidError(
            f"certificate_invalid: stratum certificate replay failed: {error}"
        ) from error
    if checked != stratum:
        raise CertificateInvalidError(
            "certificate_invalid: stratum differs after certificate replay"
        )
    state = _z2_point_batch_state(checked)
    authority = _CertifiedZ2PointBatchAuthority(
        checked,
        state,
        _Z2_POINT_BATCH_SEAL,
    )
    _register_z2_point_batch_authority(authority)
    return checked, authority


def enumerate_finite_stratum(
    stratum: FiniteAffineStratum,
) -> tuple[CertifiedZ2Point, ...]:
    if type(stratum) is not FiniteAffineStratum:
        raise TypeError("enumerate_finite_stratum requires FiniteAffineStratum")
    stratum, batch_authority = _verify_and_issue_z2_point_batch(stratum)
    dimension = stratum.quotient_dimension
    universe = tuple(itertools.product((0, 1), repeat=dimension))
    unvisited = set(universe)
    result: list[CertifiedZ2Point] = []
    while unvisited:
        seed = min(unvisited)
        orbit = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for action in stratum.residual_actions:
                image = action.apply(current)
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        members = tuple(sorted(orbit))
        representative_coordinates = members[0]
        displacement = _linear_combination(
            stratum.homogeneous_basis,
            representative_coordinates,
            len(stratum.basepoint),
        )
        raw = _xor(stratum.basepoint, displacement)
        witness_core = {
            "orbit_members": [list(member) for member in members],
            "quotient_coordinates": list(representative_coordinates),
            "raw_representative": list(raw),
        }
        witness = Z2OrbitMembershipWitness(
            representative_coordinates,
            raw,
            members,
            _digest("z2-orbit-membership", witness_core),
        )
        point = CertifiedZ2Point(
            _digest(
                "certified-z2-point",
                {
                    "membership_witness_id": witness.witness_id,
                    "stratum_id": stratum.stratum_id,
                },
            ),
            stratum.stratum_id,
            representative_coordinates,
            raw,
            members,
            witness,
            stratum,
            batch_authority,
        )
        result.append(point)
        unvisited -= orbit
    return tuple(result)


__all__ = [
    "CertificateInvalidError",
    "CertifiedCentralizerAction",
    "CertifiedZ2Problem",
    "CertifiedZ2Point",
    "DiagnosticZ2Problem",
    "FiniteAffineStratum",
    "H1UnmarkingCertificate",
    "Z2OrbitMembershipWitness",
    "Z2CrossSkeletonArrow",
    "Z2Branch",
    "Z2StratumCertificate",
    "apply_h1_unmarking",
    "certify_h1_unmarking",
    "classify_z2",
    "classify_z2_diagnostic",
    "enumerate_finite_stratum",
    "make_cross_skeleton_arrow",
    "make_certified_gf2_ambient_complex",
    "make_certified_gf2_local_complex",
    "make_certified_gf2_restriction",
    "make_certified_z2_problem",
    "make_diagnostic_z2_branch",
    "make_diagnostic_z2_problem",
]
