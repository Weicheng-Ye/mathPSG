"""Joint, certificate-first classification orchestration.

The backend boundary in this module is deliberately narrow.  It transports
canonical bytes through exact dependency-keyed caches and requires the backend
authority to replay those bytes into certified Task-5/8/12 objects.  Task 14
then performs routing, whole-orbit-tuple binding, residual-groupoid formation,
and schema aggregation itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
import weakref
from typing import Any

from .catalogue_schema import CatalogueRecord, canonical_json as canonical_catalogue_json
from .classification_schema import (
    ClassificationRecord,
    ClassificationRequest,
    InstanceParameterRoute,
    LayerRecord,
    ObstructedBranch,
    ParameterRoutingResult,
    StructuredFailure,
    canonical_classification_json,
    loads_classification_query_result,
)
from .classifier_cache import (
    CacheCorruptError,
    CacheKey,
    ClassifierCache,
    make_local_skeleton_cache_key,
)
from .query import (
    ResolvedOrbit,
    RoutingVerification,
    VerifiedCatalogue,
    classification_request_digest,
    parameter_routes,
    resolve_request_orbits,
    same_stratum_routing_verification_cache_key,
    verify_parameter_routing,
    verify_same_stratum_routes,
    verify_verified_catalogue,
)
from .residual_groupoid import (
    LocalConjugacy,
    ResidualGroupoid,
    WeylOrbitData,
    _replay_local_conjugacy,
    _replay_nonempty_stratum,
    _replay_weyl_orbit_data,
    build_residual_groupoid,
    certify_unframed_quotient,
    make_global_weyl_conjugacy,
    make_local_conjugacy,
)
from .certificates import UnframedQuotientCertificate
from .cochains import (
    CertifiedCochainProblem,
    CharacterBasisCertificate,
    FiniteGroupTable,
    FreeResolutionCertificate,
    Task5VerificationAuthority,
    character_certificate_digest,
    enumerate_coefficient_characters,
)
from .gf2 import GF2Character
from .u1_classifier import (
    TorsorStratum,
    U1SectorProblem,
    classify_u1_sector,
    verify_u1_sector_problem,
)
from .u1_local import U1LocalSkeleton, verify_u1_local_skeleton
from .z2_local import (
    Z2LocalSkeleton,
    enumerate_graded_z2_skeletons,
    enumerate_spatial_z2_skeletons,
    verify_graded_z2_skeleton,
    verify_z2_local_skeleton,
)
from .z2_classifier import (
    CertificateInvalidError,
    FiniteAffineStratum,
    enumerate_finite_stratum,
)


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ARTIFACT_NAME_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_PROTOCOL = b"mathpsg-certified-classifier-v1|"
_CERTIFIED_CLASSIFICATION_CONSTRUCTION_SEAL = object()
_CERTIFICATION_AUTHORITIES: dict[
    int, tuple[weakref.ReferenceType[object], bytes]
] = {}


class BackendProcessError(RuntimeError):
    """The external algebra backend exited without a usable artifact."""


class MalformedCertificateError(ValueError):
    """A backend artifact did not decode as its claimed certificate kind."""


class ChainIdentityError(ValueError):
    """A decoded inclusion/resolution failed an exact chain identity."""


class LocalLibraryIncompleteError(ValueError):
    """The certified local skeleton library omitted a required branch."""


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: str, value: object) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL + domain.encode("ascii") + b"|" + _canonical_json(value)
    ).hexdigest()


def _raw_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256 digest")
    return value


def _bits(value: Sequence[int], path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{path}: expected bit sequence")
    result = tuple(value)
    if not result or any(type(bit) is not int or bit not in (0, 1) for bit in result):
        raise ValueError(f"{path}: expected nonempty exact bits")
    return result


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    gap_environment_digest: str
    affine_pcp_conversion_digest: str
    affine_pcp_transport_digest: str
    target_model_digest: str
    local_library_digest: str
    ambient_algorithm_digest: str
    local_algorithm_digest: str
    inclusion_algorithm_digest: str
    relative_algorithm_digest: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_digest(getattr(self, name), f"$BackendIdentity.{name}")


def _backend_identity_digest(value: BackendIdentity) -> str:
    return _digest(
        "classifier-backend-identity",
        {
            name: getattr(value, name)
            for name in value.__dataclass_fields__
        },
    )


def _backend_identity_snapshot(
    backend: "ClassifierBackendAuthority",
) -> BackendIdentity:
    current = getattr(backend, "identity", None)
    if type(current) is not BackendIdentity:
        raise MalformedCertificateError(
            "classifier backend lacks an exact BackendIdentity"
        )
    return BackendIdentity(
        **{
            name: getattr(current, name)
            for name in current.__dataclass_fields__
        }
    )


def _require_backend_identity(
    backend: "ClassifierBackendAuthority",
    expected: BackendIdentity,
) -> None:
    try:
        current = _backend_identity_snapshot(backend)
    except (TypeError, ValueError) as error:
        raise MalformedCertificateError(
            "classifier backend identity changed during classification"
        ) from error
    if current != expected:
        raise MalformedCertificateError(
            "classifier backend identity changed during classification"
        )


@dataclass(frozen=True, slots=True)
class _BackendExternalArtifactSnapshot:
    bindings: tuple[tuple[str, str], ...]
    task5_release_store: object | None


def _external_artifact_provenance_digest(
    value: _BackendExternalArtifactSnapshot,
) -> str:
    if type(value) is not _BackendExternalArtifactSnapshot:
        raise TypeError("external artifact provenance requires a verified snapshot")
    return _digest(
        "backend-external-artifact-provenance",
        {
            "bindings": [list(item) for item in value.bindings],
            "task5_store_bound": value.task5_release_store is not None,
        },
    )


_RESERVED_EXTERNAL_ARTIFACT_NAMES = frozenset(
    {
        "ambient",
        "backend-identity",
        "character-basis",
        "relative",
        "routing-verification",
    }
)


def _canonical_external_artifact_bindings(
    value: object,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError("external artifact bindings must be an exact immutable tuple")
    rows: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(
                f"external artifact binding {index} must be an exact name/digest pair"
            )
        name, digest = item
        if type(name) is not str or _ARTIFACT_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                f"external artifact binding {index} has a noncanonical name"
            )
        _require_digest(digest, f"external artifact binding {name}")
        if (
            name in _RESERVED_EXTERNAL_ARTIFACT_NAMES
            or name.startswith("local-")
            or name.startswith("inclusion-")
        ):
            raise ValueError(f"external artifact binding name {name!r} is reserved")
        rows.append((name, digest))
    result = tuple(rows)
    if result != tuple(sorted(result)) or len({name for name, _ in result}) != len(
        result
    ):
        raise ValueError("external artifact bindings must be unique and canonical")
    return result


def _backend_external_artifact_snapshot(
    backend: "ClassifierBackendAuthority",
    identity: BackendIdentity,
) -> _BackendExternalArtifactSnapshot:
    _require_backend_identity(backend, identity)
    store_before = getattr(backend, "task5_release_store", None)
    bindings = _canonical_external_artifact_bindings(
        backend.external_artifact_bindings()
    )
    store_after = getattr(backend, "task5_release_store", None)
    _require_backend_identity(backend, identity)
    if store_after is not store_before:
        raise MalformedCertificateError(
            "classifier backend external artifact authority changed during replay"
        )

    task5_rows = tuple(
        item for item in bindings if item[0] == "task5-release-store"
    )
    if store_before is None:
        if task5_rows:
            raise MalformedCertificateError(
                "Task5 release-store binding lacks an exact issued store authority"
            )
    else:
        raise TypeError(
            "release Task5 stores are unavailable in the standalone host-native package"
        )
    return _BackendExternalArtifactSnapshot(bindings, store_before)


def _require_backend_external_artifacts(
    backend: "ClassifierBackendAuthority",
    identity: BackendIdentity,
    expected: _BackendExternalArtifactSnapshot,
) -> None:
    current = _backend_external_artifact_snapshot(backend, identity)
    if (
        current.bindings != expected.bindings
        or current.task5_release_store is not expected.task5_release_store
    ):
        raise MalformedCertificateError(
            "classifier backend external artifact authority changed during classification"
        )


def _backend_call(
    backend: "ClassifierBackendAuthority",
    identity: BackendIdentity,
    operation: Callable[[], object],
    external_artifacts: _BackendExternalArtifactSnapshot | None = None,
) -> object:
    _require_backend_identity(backend, identity)
    if external_artifacts is not None:
        _require_backend_external_artifacts(backend, identity, external_artifacts)
    try:
        result = operation()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _require_backend_identity(backend, identity)
        if external_artifacts is not None:
            _require_backend_external_artifacts(backend, identity, external_artifacts)
        raise
    _require_backend_identity(backend, identity)
    if external_artifacts is not None:
        _require_backend_external_artifacts(backend, identity, external_artifacts)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactPlan:
    """One input-bound canonical-byte build plus independent output replay."""

    build: Callable[[], bytes]
    verify: Callable[[bytes], object]
    plan_digest: str

    def __post_init__(self) -> None:
        if not callable(self.build) or not callable(self.verify):
            raise TypeError("artifact plan requires build and verify callables")
        _require_digest(self.plan_digest, "artifact_plan.plan_digest")


_LOCAL_SKELETON_EVIDENCE_SEAL = object()


def _local_evidence_core(
    *,
    instance_id: str,
    coefficient_kind: str,
    skeleton_ids: Sequence[str],
    restricted_grade: Sequence[int],
    restricted_rho: Sequence[int] | None,
    derived_q: Sequence[int] | None,
    ambient_rho: GF2Character | None,
    graded: bool,
    source_table: FiniteGroupTable | None,
    diagnostic: bool,
) -> dict[str, object]:
    return {
        "ambient_rho": None if ambient_rho is None else list(ambient_rho.bits),
        "coefficient_kind": coefficient_kind,
        "derived_q": None if derived_q is None else list(derived_q),
        "diagnostic": diagnostic,
        "graded": graded,
        "instance_id": instance_id,
        "restricted_grade": list(restricted_grade),
        "restricted_rho": (
            None if restricted_rho is None else list(restricted_rho)
        ),
        "skeleton_ids": list(skeleton_ids),
        "source_table_digest": (
            None if source_table is None else source_table.table_digest
        ),
    }


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LocalSkeletonEvidence:
    """Verified complete local-library payload for one orbit/cache sector."""

    evidence_id: str
    instance_id: str
    coefficient_kind: str
    skeleton_ids: tuple[str, ...]
    restricted_grade: tuple[int, ...]
    restricted_rho: tuple[int, ...] | None
    derived_q: tuple[int, ...] | None
    ambient_rho: GF2Character | None
    graded: bool
    source_table: FiniteGroupTable | None
    skeletons: tuple[Z2LocalSkeleton | U1LocalSkeleton, ...]
    diagnostic: bool
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        if _construction_seal is not _LOCAL_SKELETON_EVIDENCE_SEAL:
            raise ValueError(
                "LocalSkeletonEvidence construction is reserved to verified factories"
            )
        _require_digest(self.evidence_id, "local_skeleton_evidence.evidence_id")
        if type(self.instance_id) is not str or not self.instance_id:
            raise ValueError("local skeleton evidence instance ID is invalid")
        if self.coefficient_kind not in ("Z2", "U1"):
            raise ValueError("local skeleton evidence coefficient kind is invalid")
        identifiers = tuple(self.skeleton_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("local skeleton evidence IDs must be nonempty and unique")
        if any(type(item) is not str or not item for item in identifiers):
            raise ValueError("local skeleton evidence contains an invalid skeleton ID")
        grade = _bits(self.restricted_grade, "local_evidence.restricted_grade")
        rho = (
            None
            if self.restricted_rho is None
            else _bits(self.restricted_rho, "local_evidence.restricted_rho")
        )
        q = (
            None
            if self.derived_q is None
            else _bits(self.derived_q, "local_evidence.derived_q")
        )
        if type(self.graded) is not bool or type(self.diagnostic) is not bool:
            raise TypeError("local skeleton evidence flags must be boolean")
        skeletons = tuple(self.skeletons)
        if self.diagnostic:
            if self.source_table is not None or skeletons:
                raise ValueError("diagnostic local evidence cannot claim release objects")
        elif type(self.source_table) is not FiniteGroupTable:
            raise TypeError("release local evidence requires a finite group table")
        if self.coefficient_kind == "Z2":
            if rho is not None or q is not None or self.ambient_rho is not None:
                raise ValueError("Z2 local evidence cannot carry rho/q")
            if not self.diagnostic and any(
                type(item) is not Z2LocalSkeleton for item in skeletons
            ):
                raise TypeError("Z2 local evidence requires Z2LocalSkeleton values")
        else:
            if rho is None or q is None or type(self.ambient_rho) is not GF2Character:
                raise ValueError("U1 local evidence requires ambient/restricted rho and q")
            if len(grade) != len(rho) or q != tuple(
                left ^ right for left, right in zip(grade, rho, strict=True)
            ):
                raise ValueError("U1 local evidence q differs from a + rho")
            if len(identifiers) != 1:
                raise ValueError("one U1 local skeleton is required per ambient rho sector")
            if not self.diagnostic and (
                len(skeletons) != 1 or type(skeletons[0]) is not U1LocalSkeleton
            ):
                raise TypeError("U1 local evidence requires one U1LocalSkeleton")
        core = _local_evidence_core(
            instance_id=self.instance_id,
            coefficient_kind=self.coefficient_kind,
            skeleton_ids=identifiers,
            restricted_grade=grade,
            restricted_rho=rho,
            derived_q=q,
            ambient_rho=self.ambient_rho,
            graded=self.graded,
            source_table=self.source_table,
            diagnostic=self.diagnostic,
        )
        if self.evidence_id != _digest("local-skeleton-evidence", core):
            raise ValueError("local skeleton evidence digest differs")
        object.__setattr__(self, "skeleton_ids", identifiers)
        object.__setattr__(self, "restricted_grade", grade)
        object.__setattr__(self, "restricted_rho", rho)
        object.__setattr__(self, "derived_q", q)
        object.__setattr__(self, "skeletons", skeletons)


def _make_local_skeleton_evidence(
    *,
    instance_id: str,
    coefficient_kind: str,
    skeleton_ids: Sequence[str],
    restricted_grade: Sequence[int],
    restricted_rho: Sequence[int] | None,
    derived_q: Sequence[int] | None,
    ambient_rho: GF2Character | None,
    graded: bool,
    source_table: FiniteGroupTable | None,
    skeletons: Sequence[Z2LocalSkeleton | U1LocalSkeleton],
    diagnostic: bool,
) -> LocalSkeletonEvidence:
    core = _local_evidence_core(
        instance_id=instance_id,
        coefficient_kind=coefficient_kind,
        skeleton_ids=tuple(skeleton_ids),
        restricted_grade=tuple(restricted_grade),
        restricted_rho=(None if restricted_rho is None else tuple(restricted_rho)),
        derived_q=None if derived_q is None else tuple(derived_q),
        ambient_rho=ambient_rho,
        graded=graded,
        source_table=source_table,
        diagnostic=diagnostic,
    )
    return LocalSkeletonEvidence(
        _digest("local-skeleton-evidence", core),
        instance_id,
        coefficient_kind,
        tuple(skeleton_ids),
        tuple(restricted_grade),
        None if restricted_rho is None else tuple(restricted_rho),
        None if derived_q is None else tuple(derived_q),
        ambient_rho,
        graded,
        source_table,
        tuple(skeletons),
        diagnostic,
        _LOCAL_SKELETON_EVIDENCE_SEAL,
    )


def make_diagnostic_local_skeleton_evidence(
    *,
    instance_id: str,
    coefficient_kind: str,
    skeleton_ids: Sequence[str],
    restricted_grade: Sequence[int],
    restricted_rho: Sequence[int] | None = None,
    derived_q: Sequence[int] | None = None,
    ambient_rho: GF2Character | None = None,
    graded: bool = False,
) -> LocalSkeletonEvidence:
    return _make_local_skeleton_evidence(
        instance_id=instance_id,
        coefficient_kind=coefficient_kind,
        skeleton_ids=skeleton_ids,
        restricted_grade=restricted_grade,
        restricted_rho=restricted_rho,
        derived_q=derived_q,
        ambient_rho=ambient_rho,
        graded=graded,
        source_table=None,
        skeletons=(),
        diagnostic=True,
    )


def make_z2_local_skeleton_evidence(
    *,
    instance_id: str,
    source_table: FiniteGroupTable,
    skeletons: Sequence[Z2LocalSkeleton],
    restricted_grade: Sequence[int],
    graded: bool,
) -> LocalSkeletonEvidence:
    expected_spatial = enumerate_spatial_z2_skeletons(source_table)
    expected = (
        tuple(
            child
            for spatial in expected_spatial
            for child in enumerate_graded_z2_skeletons(spatial)
        )
        if graded
        else expected_spatial
    )
    supplied = tuple(skeletons)
    for item in supplied:
        if graded:
            verify_graded_z2_skeleton(item, source_table)
        else:
            verify_z2_local_skeleton(item, source_table)
    if supplied != expected:
        raise ValueError("Z2 local library differs from exhaustive enumeration")
    return _make_local_skeleton_evidence(
        instance_id=instance_id,
        coefficient_kind="Z2",
        skeleton_ids=tuple(item.skeleton_id for item in expected),
        restricted_grade=restricted_grade,
        restricted_rho=None,
        derived_q=None,
        ambient_rho=None,
        graded=graded,
        source_table=source_table,
        skeletons=expected,
        diagnostic=False,
    )


def make_u1_local_skeleton_evidence(
    *,
    instance_id: str,
    source_table: FiniteGroupTable,
    skeleton: U1LocalSkeleton,
    ambient_rho: GF2Character,
    graded: bool,
) -> LocalSkeletonEvidence:
    checked = verify_u1_local_skeleton(skeleton, source_table)
    return _make_local_skeleton_evidence(
        instance_id=instance_id,
        coefficient_kind="U1",
        skeleton_ids=(checked.skeleton_id,),
        restricted_grade=checked.grade_values,
        restricted_rho=checked.rho_values,
        derived_q=checked.q_values,
        ambient_rho=ambient_rho,
        graded=graded,
        source_table=source_table,
        skeletons=(checked,),
        diagnostic=False,
    )


def verify_local_skeleton_evidence(
    value: LocalSkeletonEvidence,
    *,
    allow_diagnostic: bool,
) -> LocalSkeletonEvidence:
    if type(value) is not LocalSkeletonEvidence:
        raise TypeError("local artifact lacks LocalSkeletonEvidence")
    checked = _make_local_skeleton_evidence(
        instance_id=value.instance_id,
        coefficient_kind=value.coefficient_kind,
        skeleton_ids=value.skeleton_ids,
        restricted_grade=value.restricted_grade,
        restricted_rho=value.restricted_rho,
        derived_q=value.derived_q,
        ambient_rho=value.ambient_rho,
        graded=value.graded,
        source_table=value.source_table,
        skeletons=value.skeletons,
        diagnostic=value.diagnostic,
    )
    if checked != value:
        raise ValueError("local skeleton evidence was mutated after construction")
    if value.diagnostic:
        if not allow_diagnostic:
            raise ValueError("diagnostic local skeleton evidence is not release authority")
        return value
    assert value.source_table is not None
    if value.coefficient_kind == "Z2":
        replayed = make_z2_local_skeleton_evidence(
            instance_id=value.instance_id,
            source_table=value.source_table,
            skeletons=tuple(
                item for item in value.skeletons if type(item) is Z2LocalSkeleton
            ),
            restricted_grade=value.restricted_grade,
            graded=value.graded,
        )
    else:
        assert value.ambient_rho is not None
        replayed = make_u1_local_skeleton_evidence(
            instance_id=value.instance_id,
            source_table=value.source_table,
            skeleton=value.skeletons[0],  # type: ignore[arg-type]
            ambient_rho=value.ambient_rho,
            graded=value.graded,
        )
    if replayed != value:
        raise ValueError("local skeleton evidence differs under authority replay")
    return value


@dataclass(frozen=True, slots=True)
class U1SectorOutcome:
    rho: GF2Character
    sector_id: str
    skeleton_ids: tuple[str, ...]
    result: TorsorStratum | ObstructedBranch | None
    failure: StructuredFailure | None
    problem: U1SectorProblem | None

    def __post_init__(self) -> None:
        if type(self.rho) is not GF2Character:
            raise TypeError("U1 sector outcome requires GF2Character rho")
        _require_digest(self.sector_id, "u1_sector_outcome.sector_id")
        skeletons = tuple(self.skeleton_ids)
        if not skeletons or any(type(item) is not str or not item for item in skeletons):
            raise ValueError("U1 sector outcome requires one skeleton per orbit")
        if (self.result is None) == (self.failure is None):
            raise ValueError("U1 sector outcome requires exactly one result or hard failure")
        if self.result is not None and type(self.result) not in (
            TorsorStratum,
            ObstructedBranch,
        ):
            raise TypeError("U1 sector outcome result is invalid")
        if self.failure is not None and type(self.failure) is not StructuredFailure:
            raise TypeError("U1 sector outcome failure is invalid")
        if self.problem is not None and type(self.problem) is not U1SectorProblem:
            raise TypeError("U1 sector outcome problem is invalid")
        if self.problem is not None and (
            self.problem.rho != self.rho or self.problem.sector_id != self.sector_id
        ):
            raise ValueError("U1 sector outcome differs from its certified problem")
        if self.result is not None and self.result.skeleton_ids != skeletons:
            raise ValueError("U1 sector outcome skeleton tuple differs from result")
        if type(self.result) is TorsorStratum and (
            self.result.rho_bits != self.rho.bits
            or self.result.certificate.sector_id != self.sector_id
        ):
            raise ValueError("U1 sector outcome rho/sector differs from torsor")
        object.__setattr__(self, "skeleton_ids", skeletons)


_U1_SECTOR_COVERAGE_SEAL = object()


def _task5_authority_payload(
    value: Task5VerificationAuthority | None,
) -> object:
    if value is None:
        return None
    return {
        name: (
            [
                {
                    field_name: _plain(getattr(item, field_name))
                    for field_name in item.__dataclass_fields__
                }
                for item in value.inclusions
            ]
            if name == "inclusions"
            else getattr(value, name)
        )
        for name in value.__dataclass_fields__
    }


def _u1_outcome_payload(value: U1SectorOutcome) -> dict[str, object]:
    if type(value.result) is TorsorStratum:
        result: object = {
            "certificate_id": value.result.certificate.certificate_id,
            "kind": "nonempty",
            "stratum_id": value.result.stratum_id,
        }
    elif type(value.result) is ObstructedBranch:
        result = {
            "branch": _protocol_value(value.result),
            "kind": "obstructed",
        }
    else:
        result = {
            "failure": _protocol_value(value.failure),
            "kind": "failed",
        }
    return {
        "problem_sector_id": (
            None if value.problem is None else value.problem.sector_id
        ),
        "problem_snapshot_digest": (
            None
            if value.problem is None
            else value.problem.source_snapshot_digest
        ),
        "result": result,
        "rho_bits": list(value.rho.bits),
        "sector_id": value.sector_id,
        "skeleton_ids": list(value.skeleton_ids),
    }


def _u1_coverage_core(
    *,
    source: CertifiedCochainProblem | None,
    authority: Task5VerificationAuthority | None,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
    character_certificate_id: str,
    grade: GF2Character,
    outcomes: Sequence[U1SectorOutcome],
    diagnostic: bool,
) -> dict[str, object]:
    return {
        "authority": _task5_authority_payload(authority),
        "character_certificate_id": character_certificate_id,
        "character_recomputed_digest": (
            None
            if source is None
            else character_certificate_digest(source.character_basis)
        ),
        "diagnostic": diagnostic,
        "grade_bits": list(grade.bits),
        "outcomes": [_u1_outcome_payload(item) for item in outcomes],
        "source": (
            None
            if source is None
            else {
                "ambient_resolution_id": source.ambient.resolution_id,
                "inclusion_ids": [
                    item.certificate_id for item in source.inclusions
                ],
            }
        ),
        "spatial_character_certificate_id": (
            None
            if spatial_character_basis is None
            else spatial_character_basis.certificate_id
        ),
        "spatial_resolution_id": (
            None
            if spatial_resolution is None
            else spatial_resolution.resolution_id
        ),
    }


@dataclass(frozen=True, slots=True)
class U1SectorCoverage:
    """Task-5-backed exhaustive Hom(Gamma,C2) sector partition."""

    coverage_id: str
    source: CertifiedCochainProblem | None
    authority: Task5VerificationAuthority | None
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    character_certificate_id: str
    grade: GF2Character
    outcomes: tuple[U1SectorOutcome, ...]
    diagnostic: bool
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        if _construction_seal is not _U1_SECTOR_COVERAGE_SEAL:
            raise ValueError(
                "U1SectorCoverage construction is reserved to verified factories"
            )
        _require_digest(self.coverage_id, "u1_sector_coverage.coverage_id")
        _require_digest(
            self.character_certificate_id,
            "u1_sector_coverage.character_certificate_id",
        )
        if type(self.grade) is not GF2Character:
            raise TypeError("U1 sector coverage requires a GF2Character grade")
        outcomes = tuple(self.outcomes)
        if not outcomes or any(type(item) is not U1SectorOutcome for item in outcomes):
            raise ValueError("U1 sector coverage requires nonempty typed outcomes")
        if len({item.rho.bits for item in outcomes}) != len(outcomes):
            raise ValueError("U1 sector coverage contains duplicate rho")
        if type(self.diagnostic) is not bool:
            raise TypeError("U1 sector coverage diagnostic flag is invalid")
        if self.source is None:
            if (
                self.authority is not None
                or self.spatial_character_basis is not None
                or self.spatial_resolution is not None
                or not self.diagnostic
                or any(item.problem is not None for item in outcomes)
            ):
                raise ValueError("lightweight U1 coverage must be diagnostic-only")
        elif (
            type(self.source) is not CertifiedCochainProblem
            or type(self.authority) is not Task5VerificationAuthority
            or self.source.character_basis.certificate_id
            != self.character_certificate_id
        ):
            raise TypeError("release U1 coverage lacks Task-5 character authority")
        else:
            graded = (
                self.source.character_basis.presentation_kind
                == "graded-direct-product-presentation"
            )
            if graded and (
                type(self.spatial_character_basis)
                is not CharacterBasisCertificate
                or type(self.spatial_resolution)
                is not FreeResolutionCertificate
            ):
                raise TypeError(
                    "graded U1 coverage lacks its exact spatial character/resolution authority"
                )
            if not graded and (
                self.spatial_character_basis is not None
                or self.spatial_resolution is not None
            ):
                raise ValueError(
                    "spatial parent authority is reserved to graded U1 coverage"
                )
        core = _u1_coverage_core(
            source=self.source,
            authority=self.authority,
            spatial_character_basis=self.spatial_character_basis,
            spatial_resolution=self.spatial_resolution,
            character_certificate_id=self.character_certificate_id,
            grade=self.grade,
            outcomes=outcomes,
            diagnostic=self.diagnostic,
        )
        if self.coverage_id != _digest("u1-sector-coverage", core):
            raise ValueError("U1 sector coverage digest differs")
        object.__setattr__(self, "outcomes", outcomes)


def _make_u1_sector_coverage(
    *,
    source: CertifiedCochainProblem | None,
    authority: Task5VerificationAuthority | None,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
    character_certificate_id: str,
    grade: GF2Character,
    outcomes: Sequence[U1SectorOutcome],
    diagnostic: bool,
) -> U1SectorCoverage:
    rows = tuple(outcomes)
    core = _u1_coverage_core(
        source=source,
        authority=authority,
        spatial_character_basis=spatial_character_basis,
        spatial_resolution=spatial_resolution,
        character_certificate_id=character_certificate_id,
        grade=grade,
        outcomes=rows,
        diagnostic=diagnostic,
    )
    return U1SectorCoverage(
        _digest("u1-sector-coverage", core),
        source,
        authority,
        spatial_character_basis,
        spatial_resolution,
        character_certificate_id,
        grade,
        rows,
        diagnostic,
        _U1_SECTOR_COVERAGE_SEAL,
    )


def make_u1_sector_coverage(
    *,
    source: CertifiedCochainProblem,
    authority: Task5VerificationAuthority,
    grade: GF2Character,
    problems: Sequence[U1SectorProblem],
    spatial_character_basis: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
    failed_sectors: Mapping[tuple[int, ...], StructuredFailure] | None = None,
    allow_diagnostic: bool = False,
) -> U1SectorCoverage:
    characters = enumerate_coefficient_characters(
        source.character_basis,
        source.ambient,
        authority,
        spatial_certificate=spatial_character_basis,
        spatial_resolution=spatial_resolution,
    )
    problem_rows = tuple(problems)
    if tuple(item.rho for item in problem_rows) != characters:
        raise ValueError(
            "U1 sector problems omit, duplicate, or reorder certified rho characters"
        )
    failures = {} if failed_sectors is None else dict(failed_sectors)
    if set(failures) - {item.bits for item in characters}:
        raise ValueError("U1 failed-sector map names an uncertified rho")
    outcomes: list[U1SectorOutcome] = []
    for rho, problem in zip(characters, problem_rows, strict=True):
        if (
            problem.source != source
            or problem.authority != authority
            or problem.spatial_character_basis is not spatial_character_basis
            or problem.spatial_resolution is not spatial_resolution
            or problem.grade != grade
        ):
            raise ValueError("U1 sector problem authority differs across rho universe")
        report = verify_u1_sector_problem(
            problem,
            allow_diagnostic=allow_diagnostic,
        )
        if not report.valid:
            raise ValueError("U1 sector problem failed Task-5/8 replay")
        result = classify_u1_sector(
            problem,
            rho,
            allow_diagnostic=allow_diagnostic,
        )[0]
        failure = failures.get(rho.bits)
        outcomes.append(
            U1SectorOutcome(
                rho,
                problem.sector_id,
                tuple(item.skeleton.skeleton_id for item in problem.bindings),
                None if failure is not None else result,
                failure,
                problem,
            )
        )
    return _make_u1_sector_coverage(
        source=source,
        authority=authority,
        spatial_character_basis=spatial_character_basis,
        spatial_resolution=spatial_resolution,
        character_certificate_id=source.character_basis.certificate_id,
        grade=grade,
        outcomes=outcomes,
        diagnostic=allow_diagnostic,
    )


def make_diagnostic_u1_sector_coverage(
    *,
    character_certificate_id: str,
    grade: GF2Character,
    outcomes: Sequence[U1SectorOutcome],
) -> U1SectorCoverage:
    return _make_u1_sector_coverage(
        source=None,
        authority=None,
        spatial_character_basis=None,
        spatial_resolution=None,
        character_certificate_id=character_certificate_id,
        grade=grade,
        outcomes=outcomes,
        diagnostic=True,
    )


def verify_u1_sector_coverage(
    value: U1SectorCoverage,
    *,
    allow_diagnostic: bool,
) -> U1SectorCoverage:
    if type(value) is not U1SectorCoverage:
        raise TypeError("U1 classification lacks U1SectorCoverage")
    checked = _make_u1_sector_coverage(
        source=value.source,
        authority=value.authority,
        spatial_character_basis=value.spatial_character_basis,
        spatial_resolution=value.spatial_resolution,
        character_certificate_id=value.character_certificate_id,
        grade=value.grade,
        outcomes=value.outcomes,
        diagnostic=value.diagnostic,
    )
    if checked != value:
        raise ValueError("U1 sector coverage was mutated after construction")
    if value.diagnostic and not allow_diagnostic:
        raise ValueError("diagnostic U1 sector coverage is not release authority")
    if value.source is None:
        return value
    assert value.authority is not None
    failures = {
        item.rho.bits: item.failure
        for item in value.outcomes
        if item.failure is not None
    }
    replayed = make_u1_sector_coverage(
        source=value.source,
        authority=value.authority,
        grade=value.grade,
        problems=tuple(item.problem for item in value.outcomes),  # type: ignore[arg-type]
        spatial_character_basis=value.spatial_character_basis,
        spatial_resolution=value.spatial_resolution,
        failed_sectors=failures,  # type: ignore[arg-type]
        allow_diagnostic=value.diagnostic,
    )
    if replayed != value:
        raise ValueError("U1 sector coverage differs under Task-5 replay")
    return value


@dataclass(frozen=True, slots=True)
class LocalSkeletonPlan:
    plan: ArtifactPlan
    stabilizer_table_digest: str
    stabilizer_normalization_digest: str
    restricted_grade: tuple[int, ...]
    restricted_rho: tuple[int, ...] | None
    derived_q: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if type(self.plan) is not ArtifactPlan:
            raise TypeError("local skeleton plan requires ArtifactPlan")
        _require_digest(self.stabilizer_table_digest, "stabilizer_table_digest")
        _require_digest(
            self.stabilizer_normalization_digest,
            "stabilizer_normalization_digest",
        )
        grade = _bits(self.restricted_grade, "restricted_grade")
        rho = None if self.restricted_rho is None else _bits(self.restricted_rho, "restricted_rho")
        q = None if self.derived_q is None else _bits(self.derived_q, "derived_q")
        if rho is not None and len(rho) != len(grade):
            raise ValueError("local grade and rho lengths differ")
        if q is not None and len(q) != len(grade):
            raise ValueError("local grade and q lengths differ")
        object.__setattr__(self, "restricted_grade", grade)
        object.__setattr__(self, "restricted_rho", rho)
        object.__setattr__(self, "derived_q", q)


class ClassifierBackendAuthority:
    """Explicit adapter from Task 14 to the evolving Task-5/8/12 factories."""

    identity: BackendIdentity

    def external_artifact_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return immutable backend provenance derived from exact capabilities.

        The production Task-5 binding is computed here from the exact
        factory-issued ``task5_release_store`` object.  A backend cannot
        promote a caller-supplied digest string into that authority.
        """

        store = getattr(self, "task5_release_store", None)
        if store is None:
            return ()
        raise TypeError(
            "release Task5 stores are unavailable in the standalone host-native package"
        )

    def ambient_resolution_plan(
        self,
        request: ClassificationRequest,
        resolved_orbits: tuple[ResolvedOrbit, ...],
        timeout_seconds: int,
    ) -> ArtifactPlan:
        raise NotImplementedError

    def local_skeleton_plans(
        self,
        request: ClassificationRequest,
        resolved_orbit: ResolvedOrbit,
        ambient: object,
        timeout_seconds: int,
    ) -> Sequence[LocalSkeletonPlan]:
        raise NotImplementedError

    def inclusion_plan(
        self,
        request: ClassificationRequest,
        resolved_orbit: ResolvedOrbit,
        ambient: object,
        timeout_seconds: int,
    ) -> ArtifactPlan:
        raise NotImplementedError

    def relative_layer_plan(
        self,
        request: ClassificationRequest,
        resolved_orbits: tuple[ResolvedOrbit, ...],
        ambient: object,
        local_skeletons: tuple[tuple[object, ...], ...],
        inclusions: tuple[object, ...],
        timeout_seconds: int,
    ) -> ArtifactPlan:
        raise NotImplementedError


NonemptyStratum = FiniteAffineStratum | TorsorStratum


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JointLayerMaterial:
    """Verified result of exactly one joint relative-layer solve."""

    branch_ids: tuple[str, ...]
    framed_strata: tuple[NonemptyStratum, ...]
    local_arrows: tuple[LocalConjugacy, ...]
    global_weyl_data: tuple[tuple[str, tuple[WeylOrbitData, ...]], ...]
    obstructed_branches: tuple[ObstructedBranch, ...]
    failures: tuple[StructuredFailure, ...]
    source_artifact_digests: tuple[tuple[str, str], ...]
    u1_sector_coverage: U1SectorCoverage | None = None

    def __post_init__(self) -> None:
        branches = tuple(self.branch_ids)
        strata = tuple(self.framed_strata)
        arrows = tuple(self.local_arrows)
        weyl_rows = tuple(
            (identifier, tuple(bindings))
            for identifier, bindings in self.global_weyl_data
        )
        obstructions = tuple(self.obstructed_branches)
        failures = tuple(self.failures)
        dependencies = tuple(tuple(item) for item in self.source_artifact_digests)
        coverage = self.u1_sector_coverage
        if not branches or branches != tuple(sorted(set(branches))) or any(
            type(item) is not str or not item for item in branches
        ):
            raise ValueError("joint material branch IDs must be nonempty and canonical")
        if any(type(item) not in (FiniteAffineStratum, TorsorStratum) for item in strata):
            raise TypeError("joint material contains an uncertified stratum")
        if strata != tuple(sorted(strata, key=lambda item: item.stratum_id)):
            raise ValueError("joint material strata must use canonical ID order")
        if len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("joint material contains duplicate strata")
        if any(type(item) is not LocalConjugacy for item in arrows):
            raise TypeError("joint material local arrows are invalid")
        if any(type(item) is not ObstructedBranch for item in obstructions):
            raise TypeError("joint material obstruction witnesses are invalid")
        if obstructions != tuple(sorted(obstructions, key=lambda item: item.stratum_id)):
            raise ValueError("joint material obstructions must use canonical ID order")
        if any(type(item) is not StructuredFailure for item in failures):
            raise TypeError("joint material failures are invalid")
        represented = tuple(
            sorted(
                (item.stratum_id for item in strata),
            )
        ) + tuple(sorted(item.stratum_id for item in obstructions))
        if tuple(sorted(represented)) != branches or len(represented) != len(set(represented)):
            raise ValueError("joint material does not completely partition enumerated branches")
        if tuple(identifier for identifier, _ in weyl_rows) != tuple(
            sorted(set(identifier for identifier, _ in weyl_rows))
        ):
            raise ValueError("joint material global Weyl rows must be unique and canonical")
        if any(
            type(binding) is not WeylOrbitData
            for _, bindings in weyl_rows
            for binding in bindings
        ):
            raise TypeError("joint material global Weyl bindings are invalid")
        if dependencies != tuple(sorted(dependencies)) or len(
            {name for name, _ in dependencies}
        ) != len(dependencies):
            raise ValueError("joint material source digests must be named and canonical")
        if tuple(name for name, _ in dependencies) != ("relative",):
            raise ValueError(
                "joint material requires exactly the verified relative certificate digest"
            )
        for name, digest in dependencies:
            if type(name) is not str or not name:
                raise ValueError("joint material source digest name is invalid")
            _require_digest(digest, f"source_artifact_digests.{name}")
        if coverage is not None and type(coverage) is not U1SectorCoverage:
            raise TypeError("joint material U1 coverage is invalid")
        object.__setattr__(self, "branch_ids", branches)
        object.__setattr__(self, "framed_strata", strata)
        object.__setattr__(self, "local_arrows", arrows)
        object.__setattr__(self, "global_weyl_data", weyl_rows)
        object.__setattr__(self, "obstructed_branches", obstructions)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "source_artifact_digests", dependencies)


def _semantic_value_payload(value: object) -> object:
    if type(value) is LocalSkeletonEvidence:
        verify_local_skeleton_evidence(
            value,
            allow_diagnostic=value.diagnostic,
        )
        return {
            "evidence_id": value.evidence_id,
            "semantic_core": _local_evidence_core(
                instance_id=value.instance_id,
                coefficient_kind=value.coefficient_kind,
                skeleton_ids=value.skeleton_ids,
                restricted_grade=value.restricted_grade,
                restricted_rho=value.restricted_rho,
                derived_q=value.derived_q,
                ambient_rho=value.ambient_rho,
                graded=value.graded,
                source_table=value.source_table,
                diagnostic=value.diagnostic,
            ),
        }
    if type(value) is JointLayerMaterial:
        strata = tuple(_replay_nonempty_stratum(item) for item in value.framed_strata)
        by_stratum = {item.stratum_id: item for item in strata}
        arrows = tuple(
            _replay_local_conjugacy(item, by_stratum)
            for item in value.local_arrows
        )
        weyl_rows = tuple(
            (
                identifier,
                tuple(_replay_weyl_orbit_data(binding) for binding in bindings),
            )
            for identifier, bindings in value.global_weyl_data
        )
        return {
            "branch_ids": list(value.branch_ids),
            "failures": [_protocol_value(item) for item in value.failures],
            "framed_strata": [
                {
                    "certificate_id": item.certificate.certificate_id,
                    "kind": type(item).__name__,
                    "skeleton_ids": list(item.skeleton_ids),
                    "stratum_id": item.stratum_id,
                }
                for item in strata
            ],
            "global_weyl_data": [
                [
                    identifier,
                    [
                        {
                            "evaluator_id": binding.evaluator.evaluator_id,
                            "instance_id": binding.instance_id,
                            "skeleton_id": binding.skeleton.skeleton_id,
                        }
                        for binding in bindings
                    ],
                ]
                for identifier, bindings in weyl_rows
            ],
            "local_arrow_ids": [item.conjugacy_id for item in arrows],
            "obstructed_branches": [
                _protocol_value(item) for item in value.obstructed_branches
            ],
            "source_artifact_digests": [
                list(item) for item in value.source_artifact_digests
            ],
            "u1_sector_coverage_id": (
                None
                if value.u1_sector_coverage is None
                else value.u1_sector_coverage.coverage_id
            ),
        }
    if isinstance(value, Mapping):
        return _plain(value)
    if isinstance(value, (tuple, list)):
        return [_semantic_value_payload(item) for item in value]
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is CertifiedCochainProblem:
        return {
            "ambient_resolution_id": value.ambient.resolution_id,
            "character_certificate_id": value.character_basis.certificate_id,
            "inclusion_certificate_ids": [
                item.certificate_id for item in value.inclusions
            ],
        }
    certificate_id = getattr(value, "certificate_id", None)
    if type(certificate_id) is str and _DIGEST_RE.fullmatch(certificate_id):
        return {
            "certificate_id": certificate_id,
            "kind": type(value).__name__,
        }
    resolution_id = getattr(value, "resolution_id", None)
    if type(resolution_id) is str and _DIGEST_RE.fullmatch(resolution_id):
        return {
            "kind": type(value).__name__,
            "resolution_id": resolution_id,
        }
    raise TypeError(
        "artifact verifier returned a value without Task14 semantic identity"
    )


def artifact_semantic_digest(value: object) -> str:
    """Return Task14's canonical semantic identity for a verified artifact."""

    return _digest("verified-artifact-semantic-value", _semantic_value_payload(value))


@dataclass(frozen=True, slots=True)
class _LoadedArtifact:
    key: CacheKey
    payload_digest: str
    semantic_digest: str
    value: object


def _loaded_artifact_binding(domain: str, value: _LoadedArtifact) -> str:
    return _digest(
        domain,
        {
            "cache_key": value.key.digest,
            "payload": value.payload_digest,
            "semantic": value.semantic_digest,
        },
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CertifiedClassification:
    """Schema result paired with the typed objects that justify it."""

    record: ClassificationRecord
    framed_strata: tuple[NonemptyStratum, ...]
    residual_groupoid: ResidualGroupoid | None
    unframed_quotient: UnframedQuotientCertificate | None
    routing_verification: RoutingVerification | None
    artifact_digests: tuple[tuple[str, str], ...]
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        if _construction_seal is not _CERTIFIED_CLASSIFICATION_CONSTRUCTION_SEAL:
            raise ValueError(
                "CertifiedClassification construction is reserved to the verified factory"
            )
        if type(self.record) is not ClassificationRecord:
            raise TypeError("CertifiedClassification requires ClassificationRecord")
        strata = tuple(self.framed_strata)
        artifacts = tuple(tuple(item) for item in self.artifact_digests)
        if any(type(item) not in (FiniteAffineStratum, TorsorStratum) for item in strata):
            raise TypeError("CertifiedClassification contains an uncertified stratum")
        if strata != tuple(sorted(strata, key=lambda item: item.stratum_id)):
            raise ValueError("CertifiedClassification strata are not canonical")
        if len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("CertifiedClassification contains duplicate strata")
        if artifacts != tuple(sorted(artifacts)) or len({name for name, _ in artifacts}) != len(artifacts):
            raise ValueError("CertifiedClassification artifact digests are not canonical")
        for name, digest in artifacts:
            if type(name) is not str or not name:
                raise ValueError("CertifiedClassification artifact name is invalid")
            _require_digest(digest, f"artifact_digests.{name}")
        complete = self.record.layer.status == "complete"
        if complete:
            if type(self.residual_groupoid) is not ResidualGroupoid:
                raise TypeError("complete classification requires residual groupoid")
            if type(self.unframed_quotient) is not UnframedQuotientCertificate:
                raise TypeError("complete classification requires typed quotient")
        if not complete and (
            self.residual_groupoid is not None or self.unframed_quotient is not None
        ):
            raise ValueError("failed classification cannot claim typed realizations")
        if self.record.point_routes:
            if type(self.routing_verification) is not RoutingVerification:
                raise TypeError("point routes require routing verification")
            if self.record.routing_verification_digest != self.routing_verification.certificate_digest:
                raise ValueError("routing verification digest differs from record")
        elif self.routing_verification is not None:
            raise ValueError("family-only classification cannot carry routing verification")
        object.__setattr__(self, "framed_strata", strata)
        object.__setattr__(self, "artifact_digests", artifacts)

    @property
    def classification_digest(self) -> str:
        verify_certified_classification(self)
        return _digest(
            "certified-classification",
            {
                "artifact_digests": [list(item) for item in self.artifact_digests],
                "catalogue_manifest_digest": self.record.catalogue_manifest_digest,
                "layer_id": self.record.layer.layer_id,
                "request_digest": self.record.request_digest,
            },
        )


def _record_digest(record: CatalogueRecord) -> str:
    return _raw_digest(canonical_catalogue_json(record))


def _candidate_records(
    catalogue: VerifiedCatalogue,
    request: ClassificationRequest,
) -> tuple[CatalogueRecord, ...]:
    result = catalogue.candidate_records(request.space_group, request.setting_id)
    if not result:
        raise ValueError("verified catalogue has no candidate records for group/setting")
    return result


def _group_action_digest(records: Sequence[CatalogueRecord]) -> str:
    unique = {
        record.action_provenance_digest: {
            "action": record.space_group_action,
            "action_provenance_digest": record.action_provenance_digest,
        }
        for record in records
    }
    return _digest(
        "group-action-authority",
        [unique[key] for key in sorted(unique)],
    )


def _ambient_key(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
    identity: BackendIdentity,
    plan: ArtifactPlan,
    external_artifact_provenance_digest: str,
) -> CacheKey:
    records = _candidate_records(catalogue, request)
    return CacheKey(
        "ambient-resolution",
        1,
        identity.ambient_algorithm_digest,
        tuple(
            sorted(
                {
                    "affine_pcp_conversion": identity.affine_pcp_conversion_digest,
                    "catalogue_normalization": catalogue.normalization_digest,
                    "external_artifacts": _require_digest(
                        external_artifact_provenance_digest,
                        "external_artifact_provenance_digest",
                    ),
                    "gap_environment": identity.gap_environment_digest,
                    "group_action": _group_action_digest(records),
                    "artifact_plan": plan.plan_digest,
                    "time_reversal": _digest(
                        "time-reversal-extension",
                        {"enabled": request.time_reversal},
                    ),
                }.items()
            )
        ),
    )


def _inclusion_key(
    resolved: ResolvedOrbit,
    catalogue: VerifiedCatalogue,
    identity: BackendIdentity,
    ambient: _LoadedArtifact,
    plan: ArtifactPlan,
    external_artifact_provenance_digest: str,
) -> CacheKey:
    record = resolved.record
    return CacheKey(
        "inclusion",
        1,
        identity.inclusion_algorithm_digest,
        tuple(
            sorted(
                {
                    "affine_pcp_transport": identity.affine_pcp_transport_digest,
                    "ambient_resolution": _loaded_artifact_binding(
                        "loaded-ambient-resolution", ambient
                    ),
                    "catalogue_manifest": catalogue.catalogue_manifest_digest,
                    "catalogue_normalization": catalogue.normalization_digest,
                    "catalogue_record": _record_digest(record),
                    "external_artifacts": _require_digest(
                        external_artifact_provenance_digest,
                        "external_artifact_provenance_digest",
                    ),
                    "group_action": _digest(
                        "record-group-action",
                        {
                            "action": record.space_group_action,
                            "action_provenance_digest": record.action_provenance_digest,
                        },
                    ),
                    "literal_stabilizer": _digest(
                        "record-literal-stabilizer", record.stabilizer
                    ),
                    "artifact_plan": plan.plan_digest,
                }.items()
            )
        ),
    )


def _symbolic_orbit_tuple_digest(resolved: Sequence[ResolvedOrbit]) -> str:
    return _digest(
        "full-symbolic-orbit-tuple",
        [
            {
                "instance_id": item.instance_id,
                "parameter_values": [str(value) for value in item.parameter_values],
                "record_digest": _record_digest(item.record),
                "symbolic_parameters": list(item.symbolic_parameters),
                "wyckoff_id": item.record.wyckoff_id,
            }
            for item in resolved
        ],
    )


def _relative_key(
    request: ClassificationRequest,
    resolved: Sequence[ResolvedOrbit],
    catalogue: VerifiedCatalogue,
    identity: BackendIdentity,
    ambient: _LoadedArtifact,
    local_rows: Sequence[Sequence[_LoadedArtifact]],
    local_plans: Sequence[Sequence[LocalSkeletonPlan]],
    inclusions: Sequence[_LoadedArtifact],
    plan: ArtifactPlan,
    external_artifact_provenance_digest: str,
) -> CacheKey:
    selected_records = tuple(item.record for item in resolved)
    inclusion_set = _digest(
        "ordered-inclusion-artifacts",
        [
            [
                item.instance_id,
                _loaded_artifact_binding("loaded-inclusion", artifact),
            ]
            for item, artifact in zip(resolved, inclusions, strict=True)
        ],
    )
    rho_rows = [
        [
            [] if plan.restricted_rho is None else list(plan.restricted_rho)
            for plan in plans
        ]
        for plans in local_plans
    ]
    local_payloads = [
        [
            _loaded_artifact_binding("loaded-local-skeleton", artifact)
            for artifact in artifacts
        ]
        for artifacts in local_rows
    ]
    return CacheKey(
        "relative-layer",
        1,
        identity.relative_algorithm_digest,
        tuple(
            sorted(
                {
                    "affine_pcp_transport": identity.affine_pcp_transport_digest,
                    "ambient_resolution": _loaded_artifact_binding(
                        "loaded-ambient-resolution", ambient
                    ),
                    "catalogue_manifest": catalogue.catalogue_manifest_digest,
                    "catalogue_normalization": catalogue.normalization_digest,
                    "catalogue_record_set": _digest(
                        "ordered-selected-record-set",
                        [[item.wyckoff_id, _record_digest(item)] for item in selected_records],
                    ),
                    "external_artifacts": _require_digest(
                        external_artifact_provenance_digest,
                        "external_artifact_provenance_digest",
                    ),
                    "group_action": _group_action_digest(
                        _candidate_records(catalogue, request)
                    ),
                    "igg": _digest("igg-target", {"igg": request.igg}),
                    "inclusion_set": inclusion_set,
                    "local_library": _digest(
                        "loaded-local-library",
                        {
                            "authority": identity.local_library_digest,
                            "payloads": local_payloads,
                        },
                    ),
                    "artifact_plan": plan.plan_digest,
                    "rho": _digest("enumerated-rho-sectors", rho_rows),
                    "symbolic_orbit_tuple": _symbolic_orbit_tuple_digest(resolved),
                    "target_model": identity.target_model_digest,
                }.items()
            )
        ),
    )


def _load_artifact(
    cache: ClassifierCache,
    key: CacheKey,
    plan: ArtifactPlan,
    backend: ClassifierBackendAuthority,
    identity: BackendIdentity,
    external_artifacts: _BackendExternalArtifactSnapshot | None = None,
) -> _LoadedArtifact:
    if type(plan) is not ArtifactPlan:
        raise TypeError("backend stage must return ArtifactPlan")

    def guarded_build() -> bytes:
        built = _backend_call(
            backend,
            identity,
            plan.build,
            external_artifacts,
        )
        if type(built) is not bytes:
            raise TypeError("artifact plan builder must return canonical bytes")
        return built

    _require_backend_identity(backend, identity)
    if external_artifacts is not None:
        _require_backend_external_artifacts(backend, identity, external_artifacts)
    payload = cache.get_or_build(key, guarded_build)
    _require_backend_identity(backend, identity)
    if external_artifacts is not None:
        _require_backend_external_artifacts(backend, identity, external_artifacts)
    try:
        payload_mapping = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(
            "cache_corrupt: artifact payload is not canonical semantic JSON"
        ) from error
    if not isinstance(payload_mapping, Mapping):
        raise CacheCorruptError(
            "cache_corrupt: artifact payload semantic binding differs"
        )
    if payload_mapping.get("record_type") == "mathpsg-artifact-recipe":
        # Production plans cache the reviewed capability-free recipe itself.
        # Its result digest is content-hashed with every dependency and is
        # independently checked by ``plan.verify`` against the caller-trusted
        # recipe digest before a fresh typed value is accepted.
        from .backend_artifacts import loads_artifact_recipe

        try:
            recipe = loads_artifact_recipe(payload)
        except (TypeError, ValueError) as error:
            raise CacheCorruptError(
                "cache_corrupt: artifact recipe semantic binding differs"
            ) from error
        declared_semantic_digest = dict(recipe.result_summary).get(
            "result_digest"
        )
    else:
        declared_semantic_digest = payload_mapping.get("semantic_digest")
    if (
        type(declared_semantic_digest) is not str
        or _DIGEST_RE.fullmatch(declared_semantic_digest) is None
    ):
        raise CacheCorruptError(
            "cache_corrupt: artifact payload semantic binding differs"
        )
    value = _backend_call(
        backend,
        identity,
        lambda: plan.verify(payload),
        external_artifacts,
    )
    actual_semantic_digest = artifact_semantic_digest(value)
    if actual_semantic_digest != declared_semantic_digest:
        raise CacheCorruptError(
            "cache_corrupt: artifact replay changed semantic material"
        )
    return _LoadedArtifact(key, _raw_digest(payload), actual_semantic_digest, value)


def _verification_mapping(value: RoutingVerification) -> dict[str, object]:
    return {
        "candidate_ids": list(value.candidate_ids),
        "candidate_set_digest": value.candidate_set_digest,
        "catalogue_manifest_digest": value.catalogue_manifest_digest,
        "certificate_digest": value.certificate_digest,
        "comparison_evidence_digest": value.comparison_evidence_digest,
        "point_instance_ids": list(value.point_instance_ids),
        "request_digest": value.request_digest,
        "result_digest": value.result_digest,
        "setting_id": value.setting_id,
        "space_group": value.space_group,
    }


def _cached_same_stratum_verification(
    routes: tuple[InstanceParameterRoute, ...],
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
    cache: ClassifierCache,
) -> RoutingVerification:
    initial = verify_same_stratum_routes(routes, request, catalogue)
    key = same_stratum_routing_verification_cache_key(routes, request, catalogue)
    expected = _canonical_json(_verification_mapping(initial))
    cached = cache.get_or_build(key, lambda: expected)
    # A cache hit is only a candidate certificate.  Replay all geometry,
    # stabilizer, inclusion, order, and request bindings against current input.
    replayed = verify_same_stratum_routes(routes, request, catalogue)
    if cached != _canonical_json(_verification_mapping(replayed)):
        raise CacheCorruptError(
            "cache_corrupt: routing verification payload differs after semantic replay"
        )
    return replayed


def _material(value: object) -> JointLayerMaterial:
    if type(value) is not JointLayerMaterial:
        raise MalformedCertificateError("relative layer did not yield JointLayerMaterial")
    try:
        return JointLayerMaterial(
            value.branch_ids,
            value.framed_strata,
            value.local_arrows,
            value.global_weyl_data,
            value.obstructed_branches,
            value.failures,
            value.source_artifact_digests,
            value.u1_sector_coverage,
        )
    except (TypeError, ValueError) as error:
        raise MalformedCertificateError(
            f"relative layer material failed replay: {error}"
        ) from error


def _verify_z2_local_branch_universe(
    local_rows: Sequence[Sequence[_LoadedArtifact]],
    material: JointLayerMaterial,
) -> None:
    libraries: list[tuple[str, ...]] = []
    for row in local_rows:
        if len(row) != 1 or type(row[0].value) is not LocalSkeletonEvidence:
            raise LocalLibraryIncompleteError(
                "Z2 classification requires one typed exhaustive local library per orbit"
            )
        libraries.append(row[0].value.skeleton_ids)
    expected = tuple(itertools.product(*libraries))
    represented = tuple(
        item.skeleton_ids
        for item in material.framed_strata + material.obstructed_branches
    )
    if (
        len(represented) != len(expected)
        or len(set(represented)) != len(represented)
        or set(represented) != set(expected)
    ):
        raise LocalLibraryIncompleteError(
            "joint output does not partition the exhaustive Cartesian local-branch universe"
        )


def _verify_u1_sector_universe(
    local_rows: Sequence[Sequence[_LoadedArtifact]],
    material: JointLayerMaterial,
    *,
    allow_diagnostic: bool,
) -> U1SectorCoverage:
    try:
        coverage = verify_u1_sector_coverage(
            material.u1_sector_coverage,  # type: ignore[arg-type]
            allow_diagnostic=allow_diagnostic,
        )
    except (TypeError, ValueError) as error:
        raise LocalLibraryIncompleteError(
            "U1 classification lacks exhaustive Task-5 rho-sector authority"
        ) from error
    outcomes = coverage.outcomes
    for orbit_index, row in enumerate(local_rows):
        if len(row) != len(outcomes):
            raise LocalLibraryIncompleteError(
                "U1 local artifacts omit or duplicate a certified rho sector"
            )
        for outcome, artifact in zip(outcomes, row, strict=True):
            if type(artifact.value) is not LocalSkeletonEvidence:
                raise LocalLibraryIncompleteError(
                    "U1 local artifact lacks typed skeleton evidence"
                )
            evidence = artifact.value
            if (
                evidence.ambient_rho != outcome.rho
                or evidence.skeleton_ids != (outcome.skeleton_ids[orbit_index],)
            ):
                raise LocalLibraryIncompleteError(
                    "U1 local artifact order/restriction differs from rho coverage"
                )

    expected_strata = tuple(
        sorted(
            (
                item.result
                for item in outcomes
                if type(item.result) is TorsorStratum
            ),
            key=lambda item: item.stratum_id,
        )
    )
    expected_obstructions = tuple(
        sorted(
            (
                item.result
                for item in outcomes
                if type(item.result) is ObstructedBranch
            ),
            key=lambda item: item.stratum_id,
        )
    )
    expected_failures = tuple(
        item.failure for item in outcomes if item.failure is not None
    )
    if (
        material.framed_strata != expected_strata
        or material.obstructed_branches != expected_obstructions
        or material.failures != expected_failures
    ):
        raise LocalLibraryIncompleteError(
            "joint U1 output does not exactly partition every certified rho sector"
        )
    return coverage


def _stratum_instance_ids(
    stratum: NonemptyStratum,
    *,
    u1_weyl_data: Sequence[WeylOrbitData] = (),
) -> tuple[str, ...]:
    if type(stratum) is FiniteAffineStratum:
        snapshot = stratum.certificate._release_snapshot
        if snapshot is not None:
            matches = tuple(
                branch
                for branch in snapshot.source_snapshot.branches
                if branch.matrices.certificate.certificate_id
                == stratum.certificate.relative_certificate_id
                and branch.skeleton_ids == stratum.skeleton_ids
            )
            if len(matches) != 1:
                raise MalformedCertificateError(
                    "release Z2 stratum lacks one exact ordered source branch"
                )
            return tuple(
                binding.instance_id for binding in matches[0].orbit_bindings
            )
        return stratum.certificate.matrices.coordinate_blocks.instance_ids
    rows = tuple(u1_weyl_data)
    if rows:
        return tuple(binding.instance_id for binding in rows)
    return stratum.matrices.coordinate_blocks.instance_ids


def _phase_text(value: object) -> str:
    return str(value)


def _continuous_digests(
    quotient: UnframedQuotientCertificate,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for presentation in quotient.continuous_orbit_presentations:
        digest = _digest("continuous-orbit-presentation", presentation.mapping())
        for identifier in presentation.framed_stratum_ids:
            if identifier in result:
                raise MalformedCertificateError(
                    "continuous quotient presentations overlap"
                )
            result[identifier] = digest
    return result


def _z2_mapping(stratum: FiniteAffineStratum) -> dict[str, object]:
    return {
        "basepoint": list(stratum.basepoint),
        "dimension": stratum.quotient_dimension,
        "framed_finite_cardinality": 2**stratum.quotient_dimension,
        "kind": "finite-affine-z2",
        "quotient_basis": [list(row) for row in stratum.homogeneous_basis],
        "residual_orbit_certificate": stratum.certificate.certificate_id,
        "skeleton_ids": list(stratum.skeleton_ids),
        "stratum_id": stratum.stratum_id,
        "unframed_finite_cardinality": len(enumerate_finite_stratum(stratum)),
    }


def _u1_mapping(
    stratum: TorsorStratum,
    groupoid: ResidualGroupoid,
    continuous_digests: Mapping[str, str],
) -> dict[str, object]:
    group = stratum.homogeneous_group
    presentation_digest = continuous_digests.get(
        stratum.stratum_id,
        _digest(
            "rank-zero-u1-orbit-presentation",
            {
                "groupoid_digest": groupoid.groupoid_digest,
                "stratum_id": stratum.stratum_id,
            },
        ),
    )
    result: dict[str, object] = {
        "affine_arrow_ids": sorted(
            arrow.conjugacy_id
            for arrow in groupoid.arrows
            if arrow.conjugacy_id in groupoid.generator_ids
            and (
                arrow.source_stratum_id == stratum.stratum_id
                or arrow.target_stratum_id == stratum.stratum_id
            )
        ),
        "basepoint_phases": [_phase_text(value) for value in stratum.basepoint],
        "formal_parameters": list(stratum.free_parameters),
        "framed_torsor_summary": {
            "free_rank": group.free_rank,
            "torsion_orders": list(group.torsion_orders),
        },
        "free_rank": group.free_rank,
        "kind": "compact-u1-torsor",
        "primal_chart_digest": stratum.certificate.primal_chart_digest,
        "rho_bits": list(stratum.rho_bits),
        "skeleton_ids": list(stratum.skeleton_ids),
        "stratum_id": stratum.stratum_id,
        "torsion_orders": list(group.torsion_orders),
        "unframed_torsor_orbit_summary": {
            "presentation_digest": presentation_digest,
        },
    }
    if group.free_rank == 0:
        result["finite_class_count"] = math.prod(group.torsion_orders)
    return result


def _quotient_mapping(
    quotient: UnframedQuotientCertificate,
    strata: Sequence[NonemptyStratum],
    continuous_digests: Mapping[str, str],
) -> dict[str, object]:
    return {
        "certificate_digest": quotient.certificate_id,
        "continuous_orbit_presentations": [
            {
                "presentation_digest": continuous_digests[stratum.stratum_id],
                "stratum_id": stratum.stratum_id,
            }
            for stratum in strata
            if type(stratum) is TorsorStratum
            and stratum.homogeneous_group.free_rank > 0
        ],
        "framed_finite_cardinality": quotient.framed_finite_cardinality,
        "framed_stratum_ids": list(quotient.framed_stratum_ids),
        "unframed_finite_cardinality": quotient.unframed_finite_cardinality,
    }


def _artifact_tuple(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(values.items()))


def _protocol_value(value: object) -> object:
    return json.loads(canonical_classification_json(value).decode("utf-8"))


def _complete_layer_id(
    request_digest: str,
    branch_ids: Sequence[str],
    strata_mappings: Sequence[Mapping[str, object]],
    quotient_mapping: Mapping[str, object],
    obstructions: Sequence[ObstructedBranch],
    groupoid: ResidualGroupoid,
    quotient: UnframedQuotientCertificate,
    artifacts: tuple[tuple[str, str], ...],
) -> str:
    return _digest(
        "complete-classification-layer",
        {
            "artifact_digests": [list(item) for item in artifacts],
            "branch_ids": list(branch_ids),
            "framed_strata": list(strata_mappings),
            "groupoid_digest": groupoid.groupoid_digest,
            "obstructed_branches": [
                _protocol_value(item) for item in obstructions
            ],
            "quotient": quotient_mapping,
            "quotient_certificate": quotient.certificate_id,
            "request_digest": request_digest,
        },
    )


def _failed_layer_id(
    request_digest: str,
    failures: Sequence[StructuredFailure],
    artifacts: tuple[tuple[str, str], ...],
) -> str:
    return _digest(
        "failed-classification-layer",
        {
            "artifact_digests": [list(item) for item in artifacts],
            "failures": [_protocol_value(item) for item in failures],
            "request_digest": request_digest,
        },
    )


def _routing_replay(value: RoutingVerification) -> RoutingVerification:
    return RoutingVerification(
        value.result_digest,
        value.request_digest,
        value.catalogue_manifest_digest,
        value.space_group,
        value.setting_id,
        value.point_instance_ids,
        value.candidate_ids,
        value.candidate_set_digest,
        value.comparison_evidence_digest,
        value.certificate_digest,
    )


def _verify_complete_artifact_schema(
    layer: LayerRecord,
    artifacts: tuple[tuple[str, str], ...],
) -> None:
    names = tuple(name for name, _ in artifacts)
    allowed_fixed = {
        "ambient",
        "backend-identity",
        "character-basis",
        "relative",
        "routing-verification",
    }
    external = tuple(
        item
        for item in artifacts
        if item[0] not in allowed_fixed
        and re.fullmatch(
            r"(?:inclusion|local)-[0-9]{4}(?:-[0-9]{4})?",
            item[0],
        )
        is None
    )
    _canonical_external_artifact_bindings(external)
    skeleton_rows = tuple(
        tuple(item["skeleton_ids"]) for item in layer.framed_strata
    ) + tuple(item.skeleton_ids for item in layer.obstructed_branches)
    if not skeleton_rows or any(
        not row or len(row) != len(skeleton_rows[0]) for row in skeleton_rows
    ):
        raise ValueError("certification branch evidence has no common orbit arity")
    orbit_count = len(skeleton_rows[0])
    expected_inclusions = {
        f"inclusion-{index:04d}" for index in range(orbit_count)
    }
    actual_inclusions = {name for name in names if name.startswith("inclusion-")}
    if actual_inclusions != expected_inclusions:
        raise ValueError("certification inclusion provenance is incomplete")
    local_names = tuple(name for name in names if name.startswith("local-"))
    expected_local_names: set[str] = set()
    for orbit_index in range(orbit_count):
        prefix = f"local-{orbit_index:04d}-"
        row = tuple(name for name in local_names if name.startswith(prefix))
        expected_row = tuple(
            f"{prefix}{plan_index:04d}" for plan_index in range(len(row))
        )
        if not row or row != expected_row:
            raise ValueError("certification local-library provenance is incomplete")
        expected_local_names.update(row)
    if set(local_names) != expected_local_names:
        raise ValueError("certification local-library provenance names differ")
    if any(item["kind"] == "compact-u1-torsor" for item in layer.framed_strata):
        if "character-basis" not in names:
            raise ValueError("U1 certification lacks character-basis provenance")


def _replay_certified_classification(
    value: CertifiedClassification,
    *,
    replay_typed_authorities: bool = True,
) -> bytes:
    if type(value.record) is not ClassificationRecord:
        raise TypeError("certification record type differs")
    record_bytes = canonical_classification_json(value.record)
    parsed_record = loads_classification_query_result(record_bytes)
    if type(parsed_record) is not ClassificationRecord or parsed_record != value.record:
        raise ValueError("certification record does not survive schema replay")

    strata = tuple(value.framed_strata)
    if any(type(item) not in (FiniteAffineStratum, TorsorStratum) for item in strata):
        raise TypeError("certification contains an uncertified stratum")
    if strata != tuple(sorted(strata, key=lambda item: item.stratum_id)) or len(
        {item.stratum_id for item in strata}
    ) != len(strata):
        raise ValueError("certification stratum order/identity differs")

    artifacts = tuple(tuple(item) for item in value.artifact_digests)
    if artifacts != tuple(sorted(artifacts)) or len(
        {name for name, _ in artifacts}
    ) != len(artifacts):
        raise ValueError("certification artifact evidence is not canonical")
    for name, digest in artifacts:
        if type(name) is not str or not name:
            raise ValueError("certification artifact name is invalid")
        _require_digest(digest, f"artifact_digests.{name}")
    artifact_map = dict(artifacts)

    routes = value.record.point_routes
    routing = value.routing_verification
    if routes:
        if type(routing) is not RoutingVerification or _routing_replay(routing) != routing:
            raise ValueError("certification routing authority does not replay")
        if (
            routing.certificate_digest
            != value.record.routing_verification_digest
            or routing.request_digest != value.record.request_digest
            or routing.catalogue_manifest_digest
            != value.record.catalogue_manifest_digest
            or routing.point_instance_ids
            != tuple(route.instance_id for route in routes)
            or artifact_map.get("routing-verification")
            != routing.certificate_digest
        ):
            raise ValueError("certification routing binding differs from record/artifacts")
    elif routing is not None:
        raise ValueError("family-only certification carries routing authority")

    layer = value.record.layer
    if layer.status == "complete":
        if type(value.residual_groupoid) is not ResidualGroupoid:
            raise TypeError("complete certification lacks typed residual groupoid")
        if type(value.unframed_quotient) is not UnframedQuotientCertificate:
            raise TypeError("complete certification lacks typed unframed quotient")
        if not {"ambient", "backend-identity", "relative"}.issubset(
            artifact_map
        ) or not any(
            name.startswith("local-") for name in artifact_map
        ) or not any(name.startswith("inclusion-") for name in artifact_map):
            raise ValueError("complete certification lacks required artifact evidence")
        _verify_complete_artifact_schema(layer, artifacts)

        groupoid = value.residual_groupoid
        if replay_typed_authorities:
            generator_set = set(groupoid.generator_ids)
            generators = tuple(
                arrow
                for arrow in groupoid.arrows
                if arrow.conjugacy_id in generator_set
            )
            if {arrow.conjugacy_id for arrow in generators} != generator_set:
                raise ValueError("certification residual generator evidence is incomplete")
            rebuilt_groupoid = build_residual_groupoid(strata, generators)
            if rebuilt_groupoid != groupoid:
                raise ValueError("certification residual groupoid does not replay")
            rebuilt_quotient = certify_unframed_quotient(strata, rebuilt_groupoid)
            if rebuilt_quotient != value.unframed_quotient:
                raise ValueError("certification unframed quotient does not replay")
        else:
            rebuilt_groupoid = groupoid
            rebuilt_quotient = value.unframed_quotient
            if (
                rebuilt_groupoid.object_ids
                != tuple(sorted(stratum.stratum_id for stratum in strata))
                or rebuilt_quotient.groupoid_digest
                != rebuilt_groupoid.groupoid_digest
                or rebuilt_quotient.framed_stratum_ids
                != tuple(stratum.stratum_id for stratum in strata)
            ):
                raise ValueError("factory typed groupoid/quotient binding differs")

        continuous = _continuous_digests(rebuilt_quotient)
        expected_strata = tuple(
            _z2_mapping(stratum)
            if type(stratum) is FiniteAffineStratum
            else _u1_mapping(stratum, rebuilt_groupoid, continuous)
            for stratum in strata
        )
        expected_quotient = _quotient_mapping(
            rebuilt_quotient, strata, continuous
        )
        layer_payload = _protocol_value(layer)
        assert isinstance(layer_payload, dict)
        if _canonical_json(expected_strata) != _canonical_json(
            layer_payload["framed_strata"]
        ):
            raise ValueError("certification framed-stratum summary differs from typed strata")
        if _canonical_json(expected_quotient) != _canonical_json(
            layer_payload["unframed_quotient"]
        ):
            raise ValueError("certification quotient summary differs from typed quotient")
        branch_ids = tuple(
            sorted(
                tuple(stratum.stratum_id for stratum in strata)
                + tuple(item.stratum_id for item in layer.obstructed_branches)
            )
        )
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("certification branch summaries overlap")
        expected_layer_id = _complete_layer_id(
            value.record.request_digest,
            branch_ids,
            expected_strata,
            expected_quotient,
            layer.obstructed_branches,
            rebuilt_groupoid,
            rebuilt_quotient,
            artifacts,
        )
        if layer.layer_id != expected_layer_id:
            raise ValueError("certification layer binding digest differs")
    else:
        if strata or value.residual_groupoid is not None or value.unframed_quotient is not None:
            raise ValueError("failed certification retains typed partial realizations")
        if layer.framed_strata or layer.unframed_quotient is not None or layer.obstructed_branches:
            raise ValueError("failed certification retains schema partial realizations")
        expected_layer_id = _failed_layer_id(
            value.record.request_digest,
            layer.failures,
            artifacts,
        )
        if layer.layer_id != expected_layer_id:
            raise ValueError("failed certification layer binding digest differs")

    snapshot = {
        "artifact_digests": [list(item) for item in artifacts],
        "groupoid_digest": (
            None
            if value.residual_groupoid is None
            else value.residual_groupoid.groupoid_digest
        ),
        "quotient_certificate": (
            None
            if value.unframed_quotient is None
            else value.unframed_quotient.certificate_id
        ),
        "record": json.loads(record_bytes.decode("utf-8")),
        "routing": None if routing is None else _verification_mapping(routing),
        "typed_strata": [
            {
                "certificate_id": (
                    stratum.certificate.certificate_id
                    if type(stratum) is FiniteAffineStratum
                    else stratum.certificate.certificate_id
                ),
                "kind": type(stratum).__name__,
                "stratum_id": stratum.stratum_id,
            }
            for stratum in strata
        ],
    }
    return _canonical_json(snapshot)


def _make_certified_classification(
    record: ClassificationRecord,
    framed_strata: Sequence[NonemptyStratum],
    residual_groupoid: ResidualGroupoid | None,
    unframed_quotient: UnframedQuotientCertificate | None,
    routing_verification: RoutingVerification | None,
    artifact_digests: tuple[tuple[str, str], ...],
) -> CertifiedClassification:
    value = CertifiedClassification(
        record,
        tuple(framed_strata),
        residual_groupoid,
        unframed_quotient,
        routing_verification,
        artifact_digests,
        _CERTIFIED_CLASSIFICATION_CONSTRUCTION_SEAL,
    )
    snapshot = _replay_certified_classification(
        value,
        replay_typed_authorities=False,
    )
    identifier = id(value)

    def remove_authority(reference: weakref.ReferenceType[object]) -> None:
        current = _CERTIFICATION_AUTHORITIES.get(identifier)
        if current is not None and current[0] is reference:
            _CERTIFICATION_AUTHORITIES.pop(identifier, None)

    reference = weakref.ref(value, remove_authority)
    _CERTIFICATION_AUTHORITIES[identifier] = (reference, snapshot)
    return value


def verify_certified_classification(
    value: CertifiedClassification,
) -> CertifiedClassification:
    """Replay the complete in-memory certification and its factory snapshot."""

    if type(value) is not CertifiedClassification:
        raise TypeError("certification replay requires CertifiedClassification")
    authority = _CERTIFICATION_AUTHORITIES.get(id(value))
    if authority is None or authority[0]() is not value:
        raise ValueError("certification lacks verified factory authority")
    try:
        current = _replay_certified_classification(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ValueError(f"certification replay failed: {error}") from error
    if current != authority[1]:
        raise ValueError("certification snapshot binding differs")
    return value


def _failed_result(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
    *,
    routes: tuple[InstanceParameterRoute, ...],
    routing: RoutingVerification | None,
    failures: tuple[StructuredFailure, ...],
    artifacts: Mapping[str, str],
) -> CertifiedClassification:
    request_digest = classification_request_digest(request)
    artifact_tuple = _artifact_tuple(artifacts)
    layer_id = _failed_layer_id(request_digest, failures, artifact_tuple)
    layer = LayerRecord(layer_id, "failed", (), None, (), failures)
    record = ClassificationRecord(
        request_digest,
        catalogue.catalogue_manifest_digest,
        layer,
        routes,
        None if routing is None else routing.certificate_digest,
    )
    return _make_certified_classification(
        record,
        (),
        None,
        None,
        routing,
        artifact_tuple,
    )


def _failure_for(error: Exception, stage: str) -> StructuredFailure:
    if isinstance(error, TimeoutError):
        code = "backend_timeout"
    elif isinstance(error, BackendProcessError):
        code = "backend_failed"
    elif isinstance(error, ChainIdentityError):
        code = "chain_identity_failed"
    elif isinstance(error, LocalLibraryIncompleteError):
        code = "local_library_incomplete"
    elif isinstance(error, CacheCorruptError):
        code = "cache_corrupt"
    elif isinstance(error, (MalformedCertificateError, CertificateInvalidError)):
        code = "certificate_invalid"
    else:
        code = "certificate_invalid"
    return StructuredFailure(
        code,
        stage,
        f"{code} during {stage}",
        {"error_type": type(error).__name__},
    )


def classify_request(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
    *,
    cache: ClassifierCache,
    timeout_seconds: int = 300,
) -> CertifiedClassification | ParameterRoutingResult:
    """Classify the complete occupied-orbit tuple or return aggregate routing."""

    if type(request) is not ClassificationRequest:
        raise TypeError("classify_request requires ClassificationRequest")
    if type(catalogue) is not VerifiedCatalogue:
        raise TypeError("classify_request requires verified catalogue authority")
    if type(cache) is not ClassifierCache:
        raise TypeError("classify_request requires ClassifierCache")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    catalogue = verify_verified_catalogue(catalogue)
    backend = catalogue.backend
    if not isinstance(backend, ClassifierBackendAuthority):
        raise TypeError("verified catalogue lacks a classifier backend authority")
    identity = _backend_identity_snapshot(backend)
    external_artifacts = _backend_external_artifact_snapshot(backend, identity)
    external_artifact_provenance = _external_artifact_provenance_digest(
        external_artifacts
    )

    resolved = resolve_request_orbits(request, catalogue)
    routes: tuple[InstanceParameterRoute, ...] = ()
    routing: RoutingVerification | None = None
    artifacts: dict[str, str] = dict(external_artifacts.bindings)
    artifacts["backend-identity"] = _backend_identity_digest(identity)
    if any(orbit.parameter_mode == "point" for orbit in request.orbits):
        routes = parameter_routes(request, catalogue)
        if any(route.outcome != "same_stratum" for route in routes):
            aggregate = ParameterRoutingResult(
                "parameter_specialization",
                classification_request_digest(request),
                catalogue.catalogue_manifest_digest,
                request.space_group,
                request.setting_id,
                routes,
            )
            verify_parameter_routing(aggregate, request, catalogue)
            _require_backend_identity(backend, identity)
            _require_backend_external_artifacts(
                backend, identity, external_artifacts
            )
            return aggregate
        try:
            routing = _cached_same_stratum_verification(
                routes, request, catalogue, cache
            )
            artifacts["routing-verification"] = routing.certificate_digest
            _require_backend_external_artifacts(
                backend, identity, external_artifacts
            )
        except Exception as error:
            if not isinstance(error, CacheCorruptError):
                raise
            try:
                _require_backend_external_artifacts(
                    backend, identity, external_artifacts
                )
            except Exception as binding_error:
                error = binding_error
            return _failed_result(
                request,
                catalogue,
                # The cache entry failed before semantic route authority was
                # recovered, so the failed record must not retain unverified
                # same-stratum route claims.
                routes=(),
                routing=None,
                failures=(_failure_for(error, "routing_verification"),),
                artifacts=artifacts,
            )

    stage = "ambient_resolution"
    try:
        ambient_plan = _backend_call(
            backend,
            identity,
            lambda: backend.ambient_resolution_plan(
                request, resolved, timeout_seconds
            ),
            external_artifacts,
        )
        if type(ambient_plan) is not ArtifactPlan:
            raise TypeError("backend ambient stage returned invalid plan")
        ambient = _load_artifact(
            cache,
            _ambient_key(
                request,
                catalogue,
                identity,
                ambient_plan,
                external_artifact_provenance,
            ),
            ambient_plan,
            backend,
            identity,
            external_artifacts,
        )
        artifacts["ambient"] = _loaded_artifact_binding(
            "certified-ambient-artifact", ambient
        )

        local_rows: list[tuple[_LoadedArtifact, ...]] = []
        local_plan_rows: list[tuple[LocalSkeletonPlan, ...]] = []
        for orbit_index, item in enumerate(resolved):
            stage = "local_skeleton"
            plans = _backend_call(
                backend,
                identity,
                lambda item=item: tuple(
                    backend.local_skeleton_plans(
                        request, item, ambient.value, timeout_seconds
                    )
                ),
                external_artifacts,
            )
            if type(plans) is not tuple:
                raise TypeError("backend local stage returned an invalid plan sequence")
            if not plans:
                raise LocalLibraryIncompleteError(
                    f"no local skeleton branch for {item.instance_id}"
                )
            if request.igg == "Z2" and len(plans) != 1:
                raise LocalLibraryIncompleteError(
                    f"{item.instance_id}: Z2 local library must be one exhaustive artifact"
                )
            loaded = []
            for plan_index, plan in enumerate(plans):
                if type(plan) is not LocalSkeletonPlan:
                    raise TypeError("backend local stage returned invalid plan")
                if request.igg == "Z2" and (
                    plan.restricted_rho is not None or plan.derived_q is not None
                ):
                    raise MalformedCertificateError(
                        "Z2 local plan unexpectedly carries rho/q"
                    )
                key = make_local_skeleton_cache_key(
                    request.igg,
                    algorithm_digest=identity.local_algorithm_digest,
                    target_model_digest=identity.target_model_digest,
                    stabilizer_table_digest=plan.stabilizer_table_digest,
                    stabilizer_normalization_digest=plan.stabilizer_normalization_digest,
                    local_library_digest=identity.local_library_digest,
                    plan_digest=plan.plan.plan_digest,
                    external_artifact_provenance_digest=(
                        external_artifact_provenance
                    ),
                    restricted_grade=plan.restricted_grade,
                    restricted_rho=plan.restricted_rho,
                    derived_q=plan.derived_q,
                )
                artifact = _load_artifact(
                    cache,
                    key,
                    plan.plan,
                    backend,
                    identity,
                    external_artifacts,
                )
                try:
                    evidence = verify_local_skeleton_evidence(
                        artifact.value,
                        allow_diagnostic=not catalogue.release_complete,
                    )
                except (TypeError, ValueError) as error:
                    raise LocalLibraryIncompleteError(
                        f"{item.instance_id}: local skeleton evidence failed replay"
                    ) from error
                if (
                    evidence.instance_id != item.instance_id
                    or evidence.coefficient_kind != request.igg
                    or evidence.restricted_grade != plan.restricted_grade
                    or evidence.restricted_rho != plan.restricted_rho
                    or evidence.derived_q != plan.derived_q
                    or evidence.graded != request.time_reversal
                ):
                    raise LocalLibraryIncompleteError(
                        f"{item.instance_id}: local artifact differs from its plan/request"
                    )
                loaded.append(artifact)
                artifacts[
                    f"local-{orbit_index:04d}-{plan_index:04d}"
                ] = _loaded_artifact_binding(
                    "certified-local-artifact", artifact
                )
            local_plan_rows.append(plans)
            local_rows.append(tuple(loaded))

        inclusions: list[_LoadedArtifact] = []
        for orbit_index, item in enumerate(resolved):
            stage = "inclusion"
            plan = _backend_call(
                backend,
                identity,
                lambda item=item: backend.inclusion_plan(
                    request, item, ambient.value, timeout_seconds
                ),
                external_artifacts,
            )
            if type(plan) is not ArtifactPlan:
                raise TypeError("backend inclusion stage returned invalid plan")
            artifact = _load_artifact(
                cache,
                _inclusion_key(
                    item,
                    catalogue,
                    identity,
                    ambient,
                    plan,
                    external_artifact_provenance,
                ),
                plan,
                backend,
                identity,
                external_artifacts,
            )
            inclusions.append(artifact)
            artifacts[f"inclusion-{orbit_index:04d}"] = _loaded_artifact_binding(
                "certified-inclusion-artifact", artifact
            )

        stage = "relative_layer"
        relative_plan = _backend_call(
            backend,
            identity,
            lambda: backend.relative_layer_plan(
                request,
                resolved,
                ambient.value,
                tuple(tuple(item.value for item in row) for row in local_rows),
                tuple(item.value for item in inclusions),
                timeout_seconds,
            ),
            external_artifacts,
        )
        if type(relative_plan) is not ArtifactPlan:
            raise TypeError("backend relative stage returned invalid plan")
        relative = _load_artifact(
            cache,
            _relative_key(
                request,
                resolved,
                catalogue,
                identity,
                ambient,
                local_rows,
                local_plan_rows,
                inclusions,
                relative_plan,
                external_artifact_provenance,
            ),
            relative_plan,
            backend,
            identity,
            external_artifacts,
        )
        artifacts["relative"] = _loaded_artifact_binding(
            "certified-relative-artifact", relative
        )
        material = _material(relative.value)
        if request.igg == "Z2":
            if material.u1_sector_coverage is not None:
                raise LocalLibraryIncompleteError(
                    "Z2 joint material cannot carry U1 rho coverage"
                )
            _verify_z2_local_branch_universe(local_rows, material)
        else:
            coverage = _verify_u1_sector_universe(
                local_rows,
                material,
                allow_diagnostic=not catalogue.release_complete,
            )
            artifacts["character-basis"] = coverage.character_certificate_id
        _require_backend_identity(backend, identity)
        _require_backend_external_artifacts(backend, identity, external_artifacts)
    except Exception as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return _failed_result(
            request,
            catalogue,
            routes=routes,
            routing=routing,
            failures=(_failure_for(error, stage),),
            artifacts=artifacts,
        )

    if material.failures:
        try:
            _require_backend_identity(backend, identity)
            _require_backend_external_artifacts(
                backend, identity, external_artifacts
            )
        except Exception as error:
            return _failed_result(
                request,
                catalogue,
                routes=routes,
                routing=routing,
                failures=(_failure_for(error, "relative_layer"),),
                artifacts=artifacts,
            )
        return _failed_result(
            request,
            catalogue,
            routes=routes,
            routing=routing,
            failures=material.failures,
            artifacts=artifacts,
        )

    request_digest = classification_request_digest(request)
    if not material.framed_strata:
        try:
            empty_groupoid = build_residual_groupoid((), ())
            empty_quotient = certify_unframed_quotient((), empty_groupoid)
            quotient_mapping = _quotient_mapping(empty_quotient, (), {})
            artifact_tuple = _artifact_tuple(artifacts)
            layer_id = _complete_layer_id(
                request_digest,
                material.branch_ids,
                (),
                quotient_mapping,
                material.obstructed_branches,
                empty_groupoid,
                empty_quotient,
                artifact_tuple,
            )
            layer = LayerRecord(
                layer_id,
                "complete",
                (),
                quotient_mapping,
                material.obstructed_branches,
                (),
            )
            record = ClassificationRecord(
                request_digest,
                catalogue.catalogue_manifest_digest,
                layer,
                routes,
                None if routing is None else routing.certificate_digest,
            )
            _require_backend_identity(backend, identity)
            _require_backend_external_artifacts(
                backend, identity, external_artifacts
            )
            result = _make_certified_classification(
                record,
                (),
                empty_groupoid,
                empty_quotient,
                routing,
                artifact_tuple,
            )
            _require_backend_identity(backend, identity)
            _require_backend_external_artifacts(
                backend, identity, external_artifacts
            )
            return result
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return _failed_result(
                request,
                catalogue,
                routes=routes,
                routing=routing,
                failures=(_failure_for(error, "residual_groupoid"),),
                artifacts=artifacts,
            )

    try:
        expected_instances = tuple(item.instance_id for item in resolved)
        weyl_by_id = dict(material.global_weyl_data)
        for stratum in material.framed_strata:
            if _stratum_instance_ids(
                stratum,
                u1_weyl_data=weyl_by_id.get(stratum.stratum_id, ()),
            ) != expected_instances:
                raise MalformedCertificateError(
                    "joint stratum bindings differ from complete orbit tuple"
                )
            if request.igg == "Z2" and type(stratum) is not FiniteAffineStratum:
                raise MalformedCertificateError("Z2 layer returned a U1 torsor")
            if request.igg == "U1" and type(stratum) is not TorsorStratum:
                raise MalformedCertificateError("U1 layer returned a GF2 stratum")

        arrows = list(material.local_arrows)
        for stratum in material.framed_strata:
            if type(stratum) is FiniteAffineStratum:
                arrows.extend(
                    make_local_conjugacy(
                        stratum,
                        stratum,
                        action,
                        witness_digest=stratum.certificate.certificate_id,
                        orbit_instance_ids=expected_instances,
                        acted_instance_ids=expected_instances,
                        diagnostic=stratum.certificate.provenance == "diagnostic",
                    )
                    for action in stratum.residual_actions
                )

        u1_ids = tuple(
            stratum.stratum_id
            for stratum in material.framed_strata
            if type(stratum) is TorsorStratum
        )
        if tuple(sorted(weyl_by_id)) != tuple(sorted(u1_ids)):
            raise MalformedCertificateError(
                "global Weyl data must cover every and only U1 stratum"
            )
        for stratum in material.framed_strata:
            if type(stratum) is TorsorStratum:
                arrows.append(
                    make_global_weyl_conjugacy(
                        stratum,
                        weyl_by_id[stratum.stratum_id],
                        acted_instance_ids=expected_instances,
                    )
                )

        groupoid = build_residual_groupoid(material.framed_strata, tuple(arrows))
        quotient = certify_unframed_quotient(material.framed_strata, groupoid)
        continuous = _continuous_digests(quotient)
        strata_mappings = tuple(
            _z2_mapping(stratum)
            if type(stratum) is FiniteAffineStratum
            else _u1_mapping(stratum, groupoid, continuous)
            for stratum in material.framed_strata
        )
        quotient_mapping = _quotient_mapping(
            quotient, material.framed_strata, continuous
        )
        artifact_tuple = _artifact_tuple(artifacts)
        layer_id = _complete_layer_id(
            request_digest,
            material.branch_ids,
            strata_mappings,
            quotient_mapping,
            material.obstructed_branches,
            groupoid,
            quotient,
            artifact_tuple,
        )
        layer = LayerRecord(
            layer_id,
            "complete",
            strata_mappings,
            quotient_mapping,
            material.obstructed_branches,
            (),
        )
        record = ClassificationRecord(
            request_digest,
            catalogue.catalogue_manifest_digest,
            layer,
            routes,
            None if routing is None else routing.certificate_digest,
        )
        _require_backend_identity(backend, identity)
        _require_backend_external_artifacts(backend, identity, external_artifacts)
        result = _make_certified_classification(
            record,
            material.framed_strata,
            groupoid,
            quotient,
            routing,
            artifact_tuple,
        )
        _require_backend_identity(backend, identity)
        _require_backend_external_artifacts(backend, identity, external_artifacts)
        return result
    except Exception as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return _failed_result(
            request,
            catalogue,
            routes=routes,
            routing=routing,
            failures=(_failure_for(error, "residual_groupoid"),),
            artifacts=artifacts,
        )


__all__ = [
    "ArtifactPlan",
    "BackendIdentity",
    "BackendProcessError",
    "CertifiedClassification",
    "ChainIdentityError",
    "ClassifierBackendAuthority",
    "JointLayerMaterial",
    "LocalSkeletonEvidence",
    "LocalLibraryIncompleteError",
    "LocalSkeletonPlan",
    "MalformedCertificateError",
    "U1SectorCoverage",
    "U1SectorOutcome",
    "artifact_semantic_digest",
    "classify_request",
    "make_diagnostic_local_skeleton_evidence",
    "make_diagnostic_u1_sector_coverage",
    "make_u1_local_skeleton_evidence",
    "make_u1_sector_coverage",
    "make_z2_local_skeleton_evidence",
    "verify_certified_classification",
    "verify_local_skeleton_evidence",
    "verify_u1_sector_coverage",
]
