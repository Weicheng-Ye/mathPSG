"""Host-native GAP source evidence for the joint PSG classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from types import MappingProxyType
import weakref

from . import bar_evaluator as _bar
from .bar_evaluator import (
    BarResolutionEquivalence,
    GapBatchArtifactReplay,
    GapBatchLauncherExecution,
    assemble_gap_inclusion_fixture,
    export_gap_inclusion_batch_raw,
    make_gap_inclusion_batch_spec,
    replay_gap_inclusion_batch_artifact,
    restore_diagnostic_gap_batch_execution,
    verify_gap_batch_execution_provenance,
    verify_gap_batch_launcher_execution,
    verify_gap_batch_member_execution,
)
from .catalogue import catalogue_record_order_key, validate_catalogue_record_identity
from .catalogue_schema import CatalogueRecord
from .certified_classifier import (
    ArtifactPlan,
    BackendIdentity,
    ClassifierBackendAuthority,
    JointLayerMaterial,
    LocalSkeletonEvidence,
    LocalSkeletonPlan,
    artifact_semantic_digest,
    make_u1_local_skeleton_evidence,
    make_u1_sector_coverage,
    make_z2_local_skeleton_evidence,
)
from . import cochains as _cochains
from .cochains import (
    CertifiedCochainProblem,
    FiniteGroupTable,
    InclusionChainMapCertificate,
    SparseGroupRingMatrix,
    Task5InclusionAuthority,
    Task5VerificationAuthority,
    assemble_gap_free_resolution_certificate,
    character_basis_certificate,
    diagnostic_residue_digests,
    dumps_inclusion_chain_map_certificate,
    make_inclusion_chain_map_certificate,
    launcher_execution_attestation_mapping,
    pcp_presentation_relators,
    task5_diagnostic_observed_outcome,
)
from .certificates import (
    BarCoordinateTrace,
    make_diagnostic_bar_evaluator,
)
from .gf2 import GF2Character
from .gap_classifier import (
    GAPClassifierRequest,
    GAPClassifierResponse,
    canonical_gap_classifier_json,
    catalogue_record_authority_digest,
    loads_gap_classifier_request,
    loads_gap_classifier_response,
    verify_affine_pcp_certificate,
    transported_inclusion_authority_digest,
)
from .live_evidence import build_evidence
from .local_gap import (
    GapRuntime,
    GapRuntimeError,
    host_provenance,
    probe_gap,
    source_inventory_digest,
)
from .query import ResolvedOrbit, classification_request_digest
from .residual_groupoid import WeylOrbitData
from .z2_local import (
    enumerate_graded_z2_skeletons,
    enumerate_spatial_z2_skeletons,
)
from .u1_local import u1_local_skeleton


_PINNED_GAP_VERSION = "4.15.1"
_PINNED_PACKAGES = MappingProxyType(
    {
        "cryst": "4.1.30",
        "hap": "1.70",
        "hapcryst": "0.1.15",
        "io": "4.9.3",
        "json": "2.2.3",
    }
)


def _diagnostic_bar_evaluator(
    equivalence: BarResolutionEquivalence,
    rho: GF2Character,
):
    """Materialize the host-only bar evaluator without retaining Task5 authority."""

    dimensions = tuple(len(basis) for basis in equivalence.resolution.basis)
    traces: list[BarCoordinateTrace] = []
    for query in (
        equivalence.normalized_tuples(1) + equivalence.normalized_tuples(2)
    ):
        dimension = dimensions[len(query)]
        weights: list[int] = []
        for column in range(dimension):
            coordinates = [0] * dimension
            coordinates[column] = 1
            value = _bar.evaluate_bar_cochain(
                equivalence,
                coordinates,
                query,
                coefficient_character=rho,
            )
            if value.denominator != 1:
                raise ArithmeticError(
                    "host bar comparison produced a nonintegral functional"
                )
            weights.append(value.numerator)
        traces.append(
            BarCoordinateTrace.make(
                equivalence.resolution_id,
                len(query),
                query,
                weights,
            )
        )
    return make_diagnostic_bar_evaluator(
        resolution_id=equivalence.resolution_id,
        finite_group=equivalence.finite_group,
        coordinate_dimensions=dimensions,
        traces=traces,
    )
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_HOST_SOURCE_EVIDENCE_SEAL = object()
_HOST_SOURCE_EVIDENCE_REGISTRY: dict[
    int, tuple[weakref.ReferenceType["HostNativeSourceEvidence"], tuple[object, ...]]
] = {}
_HOST_AMBIENT_SEAL = object()
_HOST_AMBIENT_REGISTRY: dict[
    int, tuple[weakref.ReferenceType["HostNativeAmbientArtifact"], tuple[object, ...]]
] = {}


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
        b"mathpsg-host-classifier-v1|"
        + domain.encode("ascii")
        + b"|"
        + _canonical_json(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class HostNativeSourceEvidence:
    """One grouped, replayed Task4/Task5 execution for occupied inclusions."""

    instance_wyckoff_ids: tuple[str, ...]
    unique_inclusion_ids: tuple[str, ...]
    time_reversal: bool
    task4_request: GAPClassifierRequest
    task4_response: GAPClassifierResponse
    task5_execution: GapBatchLauncherExecution
    task5_replay: GapBatchArtifactReplay
    provenance: Mapping[str, object]
    certification_status: str = "host-native"
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        if _construction_seal is not _HOST_SOURCE_EVIDENCE_SEAL:
            raise ValueError(
                "HostNativeSourceEvidence construction is reserved to the verified factory"
            )
        instances = tuple(self.instance_wyckoff_ids)
        unique = tuple(self.unique_inclusion_ids)
        if not instances or any(type(item) is not str or not item for item in instances):
            raise ValueError("host source evidence requires nonempty instance IDs")
        if (
            not unique
            or len(set(unique)) != len(unique)
            or any(type(item) is not str or not item for item in unique)
        ):
            raise ValueError("host source evidence requires unique inclusion IDs")
        if any(item not in set(unique) for item in instances):
            raise ValueError("host source evidence instance IDs leave the unique inclusion set")
        if type(self.time_reversal) is not bool:
            raise TypeError("host source evidence time-reversal flag must be boolean")
        if type(self.task4_request) is not GAPClassifierRequest:
            raise TypeError("host source evidence requires an exact Task4 request")
        if type(self.task4_response) is not GAPClassifierResponse:
            raise TypeError("host source evidence requires an exact Task4 response")
        if type(self.task5_execution) is not GapBatchLauncherExecution:
            raise TypeError("host source evidence requires an issued Task5 execution")
        if type(self.task5_replay) is not GapBatchArtifactReplay:
            raise TypeError("host source evidence requires an exact Task5 replay")
        if type(self.provenance) is not _MAPPING_PROXY_TYPE:
            raise TypeError("host source evidence provenance must be recursively immutable")
        if self.certification_status != "host-native":
            raise ValueError("host source evidence cannot claim release certification")
        object.__setattr__(self, "instance_wyckoff_ids", instances)
        object.__setattr__(self, "unique_inclusion_ids", unique)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("host provenance keys must be strings")
            rows[key] = _freeze_json(item)
        return MappingProxyType(dict(sorted(rows.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("host provenance contains a non-JSON value")


def _runtime_snapshot(runtime: GapRuntime, *, timeout: int) -> GapRuntime:
    if type(runtime) is not GapRuntime:
        raise TypeError("runtime must be a probed GapRuntime")
    probe_timeout = min(timeout, 30)
    observed = probe_gap(runtime.executable, timeout_seconds=probe_timeout)
    if observed != runtime:
        raise GapRuntimeError("supplied GAP runtime differs from a fresh executable probe")
    if (
        observed.execution_mode != "host-native"
        or observed.gap_version != _PINNED_GAP_VERSION
        or dict(observed.packages) != dict(_PINNED_PACKAGES)
    ):
        raise GapRuntimeError(
            "local classifier requires the exact recorded GAP and package versions"
        )
    return observed


def _evidence_snapshot(value: HostNativeSourceEvidence) -> tuple[object, ...]:
    return (
        value.instance_wyckoff_ids,
        value.unique_inclusion_ids,
        value.time_reversal,
        id(value.task4_request),
        canonical_gap_classifier_json(value.task4_request),
        id(value.task4_response),
        canonical_gap_classifier_json(value.task4_response),
        id(value.task5_execution),
        id(value.task5_replay),
        id(value.provenance),
        value.certification_status,
    )


def _register_host_source_evidence(value: HostNativeSourceEvidence) -> None:
    key = id(value)

    def discard(reference: weakref.ReferenceType[HostNativeSourceEvidence]) -> None:
        current = _HOST_SOURCE_EVIDENCE_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _HOST_SOURCE_EVIDENCE_REGISTRY.pop(key, None)

    reference = weakref.ref(value, discard)
    _HOST_SOURCE_EVIDENCE_REGISTRY[key] = (reference, _evidence_snapshot(value))


def verify_host_source_evidence(
    value: HostNativeSourceEvidence,
) -> HostNativeSourceEvidence:
    """Replay an exact factory-issued host evidence capability."""

    if type(value) is not HostNativeSourceEvidence:
        raise TypeError("host source evidence has the wrong type")
    registered = _HOST_SOURCE_EVIDENCE_REGISTRY.get(id(value))
    if (
        registered is None
        or registered[0]() is not value
        or registered[1] != _evidence_snapshot(value)
    ):
        raise ValueError("host source evidence is not an exact factory-issued registry value")
    request = value.task4_request
    response = value.task4_response
    replayed_request = loads_gap_classifier_request(
        canonical_gap_classifier_json(request)
    )
    replayed_response = loads_gap_classifier_response(
        canonical_gap_classifier_json(response)
    )
    if (
        replayed_request != request
        or request.time_reversal is not value.time_reversal
        or replayed_response != response
        or response.status != "conversion_only"
        or response.request_digest != request.request_digest
        or response.affine_pcp_certificate is None
    ):
        raise ValueError("Task4 response differs from its exact request or replay")
    verify_affine_pcp_certificate(request.action, response.affine_pcp_certificate)
    unique_ids = tuple(item.inclusion_id for item in request.inclusions)
    certificate_inclusions = tuple(
        (
            item.inclusion_id,
            item.literal_stabilizer_digest,
            item.literal_element_digest,
            item.literal_elements,
        )
        for item in response.affine_pcp_certificate.transported_stabilizers
    )
    request_inclusions = tuple(
        (
            item.inclusion_id,
            item.literal_stabilizer_digest,
            item.literal_element_digest,
            item.literal_elements,
        )
        for item in request.inclusions
    )
    if (
        unique_ids != value.unique_inclusion_ids
        or certificate_inclusions != request_inclusions
    ):
        raise ValueError("Task4 inclusion IDs differ from host evidence")

    execution = value.task5_execution
    verify_gap_batch_launcher_execution(execution, require_release=False)
    execution_provenance = execution.provenance
    if execution_provenance is None:
        raise ValueError("issued Task5 batch lacks provenance")
    verify_gap_batch_execution_provenance(execution_provenance)
    replay = replay_gap_inclusion_batch_artifact(execution.spec, execution.raw_output)
    task5_ids = tuple(
        member.inclusion.inclusion_id for member in execution.spec.members
    )
    if (
        execution.spec.action != request.action
        or execution.spec.time_reversal is not value.time_reversal
        or set(task5_ids) != set(unique_ids)
        or replay != value.task5_replay
        or tuple(member.inclusion_id for member in replay.members) != task5_ids
    ):
        raise ValueError("Task5 replay differs from host evidence")
    for child, member in zip(
        execution.member_executions,
        execution.spec.members,
        strict=True,
    ):
        verify_gap_batch_member_execution(
            child,
            action=request.action,
            inclusion=member.inclusion,
            element_labels=member.element_labels,
            time_reversal=value.time_reversal,
            finite_group_id=member.finite_group_id,
            require_release=False,
        )
        if child.attestation.release_certified:
            raise ValueError("host evidence contains a release-certified child")
    return value


@dataclass(frozen=True, slots=True)
class HostNativeInclusionArtifact:
    inclusion_id: str
    bar_equivalence: BarResolutionEquivalence
    inclusion: InclusionChainMapCertificate

    def __post_init__(self) -> None:
        if type(self.inclusion_id) is not str or not self.inclusion_id:
            raise ValueError("host inclusion artifact requires an inclusion ID")
        if type(self.bar_equivalence) is not BarResolutionEquivalence:
            raise TypeError("host inclusion artifact requires bar equivalence")
        if type(self.inclusion) is not InclusionChainMapCertificate:
            raise TypeError("host inclusion artifact requires a chain map")
        if (
            self.inclusion.inclusion_id != self.inclusion_id
            or self.inclusion.source_bar_equivalence_id
            != self.bar_equivalence.equivalence_id
            or self.inclusion.source_resolution != self.bar_equivalence.resolution
        ):
            raise ValueError("host inclusion artifact has inconsistent bindings")


@dataclass(frozen=True, slots=True)
class HostNativeResolvedInclusion:
    """One inclusion rebound to a distinct occupied atom-orbit instance."""

    instance_id: str
    source: HostNativeInclusionArtifact

    def __post_init__(self) -> None:
        if type(self.instance_id) is not str or not self.instance_id:
            raise ValueError("resolved host inclusion requires an instance ID")
        if type(self.source) is not HostNativeInclusionArtifact:
            raise TypeError("resolved host inclusion requires a typed source")

    @property
    def certificate_id(self) -> str:
        return _digest(
            "resolved-host-inclusion",
            {
                "inclusion_certificate_id": self.source.inclusion.certificate_id,
                "instance_id": self.instance_id,
            },
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class HostNativeAmbientArtifact:
    resolution: object
    authority: Task5VerificationAuthority
    inclusions: tuple[HostNativeInclusionArtifact, ...]
    source_evidence: HostNativeSourceEvidence
    spatial_parent: HostNativeAmbientArtifact | None = None
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        from .cochains import FreeResolutionCertificate

        if _construction_seal is not _HOST_AMBIENT_SEAL:
            raise ValueError(
                "HostNativeAmbientArtifact construction is reserved to the verified factory"
            )
        if type(self.resolution) is not FreeResolutionCertificate:
            raise TypeError("host ambient artifact requires a free resolution")
        if type(self.authority) is not Task5VerificationAuthority:
            raise TypeError("host ambient artifact requires Task5 replay authority")
        rows = tuple(self.inclusions)
        if not rows or any(type(item) is not HostNativeInclusionArtifact for item in rows):
            raise TypeError("host ambient artifact requires typed inclusions")
        identifiers = tuple(item.inclusion_id for item in rows)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("host ambient inclusion order must be canonical")
        if any(item.inclusion.target_resolution != self.resolution for item in rows):
            raise ValueError("host inclusions do not share the ambient resolution")
        if tuple(item.inclusion_id for item in self.authority.inclusions) != identifiers:
            raise ValueError("host authority inclusion coverage differs")
        if type(self.source_evidence) is not HostNativeSourceEvidence:
            raise TypeError("host ambient artifact requires source evidence")
        parent = self.spatial_parent
        if parent is not None:
            if type(parent) is not HostNativeAmbientArtifact:
                raise TypeError("host ambient spatial parent has the wrong type")
            verify_host_ambient_artifact(parent)
            if (
                not self.source_evidence.time_reversal
                or parent.source_evidence.time_reversal
                or self.source_evidence.unique_inclusion_ids
                != parent.source_evidence.unique_inclusion_ids
                or self.source_evidence.task4_response.affine_pcp_certificate
                != parent.source_evidence.task4_response.affine_pcp_certificate
                or self.resolution.group_id
                != parent.resolution.group_id + "+onsite-T"
            ):
                raise ValueError(
                    "host ambient spatial parent is not the exact onsite-time sibling"
                )
        elif self.source_evidence.time_reversal and not self.resolution.group_id.endswith(
            "+onsite-T"
        ):
            raise ValueError("host graded ambient lacks the onsite-time group suffix")
        object.__setattr__(self, "inclusions", rows)

    @property
    def resolution_id(self) -> str:
        return self.resolution.resolution_id  # type: ignore[no-any-return]

    def inclusion_for(self, inclusion_id: str) -> HostNativeInclusionArtifact:
        matches = tuple(item for item in self.inclusions if item.inclusion_id == inclusion_id)
        if len(matches) != 1:
            raise ValueError("host ambient lacks one exact requested inclusion")
        return matches[0]


def _ambient_snapshot(value: HostNativeAmbientArtifact) -> tuple[object, ...]:
    return (
        value.resolution_id,
        tuple(
            (
                item.inclusion_id,
                _bar.dumps_bar_resolution_equivalence(item.bar_equivalence),
                dumps_inclusion_chain_map_certificate(item.inclusion),
            )
            for item in value.inclusions
        ),
        id(value.source_evidence),
        None if value.spatial_parent is None else id(value.spatial_parent),
    )


def _register_host_ambient(value: HostNativeAmbientArtifact) -> None:
    key = id(value)

    def discard(reference: weakref.ReferenceType[HostNativeAmbientArtifact]) -> None:
        current = _HOST_AMBIENT_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _HOST_AMBIENT_REGISTRY.pop(key, None)

    reference = weakref.ref(value, discard)
    _HOST_AMBIENT_REGISTRY[key] = (reference, _ambient_snapshot(value))


def verify_host_ambient_artifact(
    value: HostNativeAmbientArtifact,
) -> HostNativeAmbientArtifact:
    if type(value) is not HostNativeAmbientArtifact:
        raise TypeError("host ambient artifact has the wrong type")
    snapshot = _HOST_AMBIENT_REGISTRY.get(id(value))
    if (
        snapshot is None
        or snapshot[0]() is not value
        or snapshot[1] != _ambient_snapshot(value)
    ):
        raise ValueError("host ambient artifact is not exactly factory-issued")
    verify_host_source_evidence(value.source_evidence)
    for item in value.inclusions:
        report = _bar.verify_bar_resolution_equivalence(
            item.bar_equivalence, value.authority
        )
        if not report.valid:
            raise ValueError("host source bar equivalence failed replay")
        inclusion_report = _bar.verify_inclusion_chain_map(
            item.inclusion,
            value.authority,
            require_release=False,
        )
        if not inclusion_report.valid:
            raise ValueError(
                "host inclusion chain map failed replay: "
                + "; ".join(
                    f"{issue.code}: {issue.detail}"
                    for issue in inclusion_report.issues
                )
            )
    return value


def _raw_member_components(
    *,
    raw: Mapping[str, object],
    inclusion_id: str,
    source_group_id: str,
    target_group_id: str,
    certificate: object,
    catalogue_record_digest: str,
    diagnostic_digest: str,
    literal_stabilizer_digest: str,
    literal_element_digest: str,
    transported_inclusion_digest: str,
    launcher_attestation: object,
    target_construction: str,
    target_parent_spatial_resolution_id: str | None,
):
    from .cochains import LauncherExecutionAttestation
    from .gap_classifier import AffinePCPIsomorphismCertificate

    if type(certificate) is not AffinePCPIsomorphismCertificate:
        raise TypeError("host Task4 certificate has the wrong type")
    if type(launcher_attestation) is not LauncherExecutionAttestation:
        raise TypeError("host Task5 member lacks a launcher attestation")
    raw_table = raw.get("finite_group")
    if not isinstance(raw_table, Mapping):
        raise TypeError("host Task5 member lacks a finite group table")
    table = FiniteGroupTable(
        raw_table["group_id"],
        tuple(raw_table["element_order"]),
        raw_table["identity_index"],
        tuple(tuple(row) for row in raw_table["multiplication_table"]),
        tuple(raw_table["inverse_indices"]),
    )
    source = assemble_gap_free_resolution_certificate(
        raw["source"],
        group_id=source_group_id,
        affine_pcp_certificate=certificate,
        catalogue_record_digest=catalogue_record_digest,
        finite_group=table,
        construction="hap-resolution-finite-group-4-lookahead5",
        backend_lock_digest=diagnostic_digest,
        backend_environment_id=diagnostic_digest,
        runtime_provenance_digest=diagnostic_digest,
    )
    target = assemble_gap_free_resolution_certificate(
        raw["target"],
        group_id=target_group_id,
        affine_pcp_certificate=certificate,
        catalogue_record_digest=catalogue_record_digest,
        finite_group=None,
        construction=target_construction,
        backend_lock_digest=diagnostic_digest,
        backend_environment_id=diagnostic_digest,
        runtime_provenance_digest=diagnostic_digest,
        parent_spatial_resolution_id=target_parent_spatial_resolution_id,
    )
    equivalence = _bar.assemble_gap_bar_resolution_equivalence(
        raw["bar_equivalence"],
        resolution=source,
        benchmark_coordinates=(),
        benchmark_tuple=(),
    )
    traces = _bar._parse_gap_bar_comparison_traces(
        raw["bar_comparison_traces"],
        equivalence=equivalence,
        target_resolution=target,
        source_element_images=raw["source_element_images"],
    )
    target_equivalence = _bar.assemble_gap_target_bar_resolution_equivalence(
        raw["target_bar_equivalence"],
        target_resolution=target,
    )
    zero_maps = tuple(
        SparseGroupRingMatrix(
            len(target.basis[degree]), len(source.basis[degree]), ()
        )
        for degree in range(5)
    )
    diagnostic_maps = tuple(
        _bar._parse_matrix(item, f"$host.diagnostic_maps[{index}]")
        for index, item in enumerate(raw["diagnostic_maps"])
    )
    placeholder = ("sha256:" + "0" * 64,) * 4
    provisional = make_inclusion_chain_map_certificate(
        inclusion_id=inclusion_id,
        literal_stabilizer_digest=literal_stabilizer_digest,
        literal_element_digest=literal_element_digest,
        transported_inclusion_digest=transported_inclusion_digest,
        source_resolution=source,
        target_resolution=target,
        source_element_images=raw["source_element_images"],
        maps=zero_maps,
        source_bar_equivalence_id=equivalence.equivalence_id,
        target_bar_equivalence=target_equivalence,
        launcher_attestation=launcher_attestation,
        bar_comparison_traces=traces,
        diagnostic_backend=raw["diagnostic_backend"],
        diagnostic_maps=diagnostic_maps,
        diagnostic_outcome="commuting",
        diagnostic_residue_digests=placeholder,
    )
    maps = _bar._reconstruct_comparison_maps(provisional)
    provisional = make_inclusion_chain_map_certificate(
        inclusion_id=inclusion_id,
        literal_stabilizer_digest=provisional.literal_stabilizer_digest,
        literal_element_digest=provisional.literal_element_digest,
        transported_inclusion_digest=provisional.transported_inclusion_digest,
        source_resolution=source,
        target_resolution=target,
        source_element_images=raw["source_element_images"],
        maps=maps,
        source_bar_equivalence_id=equivalence.equivalence_id,
        target_bar_equivalence=target_equivalence,
        launcher_attestation=launcher_attestation,
        bar_comparison_traces=traces,
        diagnostic_backend=raw["diagnostic_backend"],
        diagnostic_maps=diagnostic_maps,
        diagnostic_outcome="commuting",
        diagnostic_residue_digests=placeholder,
    )
    failures = tuple(
        degree
        for degree in range(1, 5)
        if _bar._inclusion_left(provisional, degree, diagnostic_maps)
        != _bar._inclusion_right(provisional, degree, diagnostic_maps)
    )
    residues = diagnostic_residue_digests(provisional)
    return source, target, equivalence, target_equivalence, failures, residues


def assemble_host_ambient_artifact(
    evidence: HostNativeSourceEvidence,
    *,
    spatial_parent: HostNativeAmbientArtifact | None = None,
) -> HostNativeAmbientArtifact:
    """Project one issued local batch into typed diagnostic algebra objects."""

    checked = verify_host_source_evidence(evidence)
    if checked.time_reversal:
        if spatial_parent is None:
            raise ValueError("graded host ambient requires its exact spatial sibling")
        verify_host_ambient_artifact(spatial_parent)
        target_construction = "onsite-c2-direct-product-resolution"
        target_parent_id = spatial_parent.resolution_id
    else:
        if spatial_parent is not None:
            raise ValueError("spatial host ambient cannot carry a spatial parent")
        target_construction = "hap-resolution-almost-crystal-group-4-lookahead5"
        target_parent_id = None
    request = checked.task4_request
    response = checked.task4_response
    certificate = response.affine_pcp_certificate
    assert certificate is not None
    authority_group_id = (
        "ambient:" + request.action.action_digest.removeprefix("sha256:")
    )
    target_group_id = authority_group_id + ":spatial"
    target_group_id += "+onsite-T" if checked.time_reversal else ""
    catalogue_digest = catalogue_record_authority_digest(
        group_id=authority_group_id,
        catalogue_action_digest=request.action.action_digest,
        inclusions=tuple(
            sorted(
                certificate.transported_stabilizers,
                key=lambda item: item.inclusion_id,
            )
        ),
    )
    transported_by_id = {
        item.inclusion_id: item for item in certificate.transported_stabilizers
    }
    literal_by_id = {item.inclusion_id: item for item in request.inclusions}
    execution = checked.task5_execution
    preliminary: list[tuple[object, ...]] = []
    authorities: list[Task5InclusionAuthority] = []
    diagnostic_digests: set[str] = set()
    for replay_member, spec_member, child in zip(
        checked.task5_replay.members,
        execution.spec.members,
        execution.member_executions,
        strict=True,
    ):
        inclusion_id = spec_member.inclusion.inclusion_id
        if replay_member.inclusion_id != inclusion_id:
            raise ValueError("host Task5 replay member order differs")
        raw = json.loads(replay_member.raw_output)
        if not isinstance(raw, Mapping):
            raise TypeError("host Task5 member output must be an object")
        backend = raw.get("backend_environment")
        if not isinstance(backend, Mapping):
            raise TypeError("host Task5 member lacks backend observation")
        diagnostic_digest = _bar._task5_domain_digest(
            "task5-diagnostic-backend-observation-v1", backend
        )
        diagnostic_digests.add(diagnostic_digest)
        transported = transported_by_id.get(inclusion_id)
        literal = literal_by_id.get(inclusion_id)
        if transported is None or literal is None:
            raise ValueError("host Task4 and Task5 inclusion coverage differs")
        transported_digest = transported_inclusion_authority_digest(transported)
        raw_table = raw.get("finite_group")
        if not isinstance(raw_table, Mapping) or type(raw_table.get("group_id")) is not str:
            raise TypeError("host Task5 member finite group ID is unavailable")
        source_group_id = raw_table["group_id"]
        components = _raw_member_components(
            raw=raw,
            inclusion_id=inclusion_id,
            source_group_id=source_group_id,
            target_group_id=target_group_id,
            certificate=certificate,
            catalogue_record_digest=catalogue_digest,
            diagnostic_digest=diagnostic_digest,
            literal_stabilizer_digest=literal.literal_stabilizer_digest,
            literal_element_digest=literal.literal_element_digest,
            transported_inclusion_digest=transported_digest,
            launcher_attestation=child.attestation,
            target_construction=target_construction,
            target_parent_spatial_resolution_id=target_parent_id,
        )
        _, _, equivalence, target_equivalence, failures, residues = components
        outcome = (
            "commuting"
            if not failures
            else task5_diagnostic_observed_outcome(
                raw["diagnostic_backend"], failures, residues
            )
        )
        authority_row = Task5InclusionAuthority(
            inclusion_id,
            literal.literal_stabilizer_digest,
            literal.literal_element_digest,
            transported_digest,
            equivalence.equivalence_id,
            target_equivalence.equivalence_id,
            child.attestation.attestation_id,
            _bar.gap_inclusion_projection_digest(raw),
            raw["diagnostic_backend"],
            outcome,
            failures,
            residues,
        )
        authorities.append(authority_row)
        preliminary.append(
            (
                replay_member.raw_output,
                child.attestation,
                source_group_id,
                literal,
                transported_digest,
            )
        )
    if len(diagnostic_digests) != 1:
        raise ValueError("host Task5 members report different backend observations")
    diagnostic_digest = next(iter(diagnostic_digests))
    authority = Task5VerificationAuthority(
        request.action.action_digest,
        catalogue_digest,
        certificate.certificate_digest,
        diagnostic_digest,
        diagnostic_digest,
        diagnostic_digest,
        tuple(sorted(authorities, key=lambda item: item.inclusion_id)),
    )
    rows: list[HostNativeInclusionArtifact] = []
    ambient = None
    for raw_output, attestation, source_group_id, literal, transported_digest in preliminary:
        equivalence, inclusion = assemble_gap_inclusion_fixture(
            raw_output,
            attestation,
            authority=authority,
            affine_pcp_certificate=certificate,
            catalogue_record_digest=catalogue_digest,
            source_group_id=source_group_id,
            target_group_id=target_group_id,
            source_construction="hap-resolution-finite-group-4-lookahead5",
            target_construction=target_construction,
            target_parent_spatial_resolution_id=target_parent_id,
            inclusion_id=literal.inclusion_id,
            literal_stabilizer_digest=literal.literal_stabilizer_digest,
            literal_element_digest=literal.literal_element_digest,
            transported_inclusion_digest=transported_digest,
            benchmark_coordinates=(),
            benchmark_tuple=(),
            allow_diagnostic=True,
        )
        if ambient is None:
            ambient = inclusion.target_resolution
        elif inclusion.target_resolution != ambient:
            raise ValueError("host Task5 members produced different ambient resolutions")
        rows.append(
            HostNativeInclusionArtifact(
                literal.inclusion_id,
                equivalence,
                inclusion,
            )
        )
    assert ambient is not None
    result = HostNativeAmbientArtifact(
        ambient,
        authority,
        tuple(sorted(rows, key=lambda item: item.inclusion_id)),
        checked,
        spatial_parent,
        _HOST_AMBIENT_SEAL,
    )
    _register_host_ambient(result)
    return verify_host_ambient_artifact(result)


def _records(
    records: Sequence[CatalogueRecord],
) -> tuple[tuple[CatalogueRecord, ...], tuple[CatalogueRecord, ...]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a finite CatalogueRecord sequence")
    supplied = tuple(records)
    if not supplied:
        raise ValueError("records must contain at least one occupied orbit")
    if any(type(record) is not CatalogueRecord for record in supplied):
        raise TypeError("records must contain only CatalogueRecord values")
    replayed = tuple(validate_catalogue_record_identity(record) for record in supplied)
    unique_by_id: dict[str, CatalogueRecord] = {}
    for record in replayed:
        prior = unique_by_id.get(record.wyckoff_id)
        if prior is not None and prior != record:
            raise ValueError("duplicate Wyckoff ID has inconsistent catalogue records")
        unique_by_id[record.wyckoff_id] = record
    unique = tuple(sorted(unique_by_id.values(), key=catalogue_record_order_key))
    return replayed, unique


def build_host_source_evidence(
    records: Sequence[CatalogueRecord],
    *,
    runtime: GapRuntime,
    time_reversal: bool,
    timeout: int,
    repository_root: Path,
) -> HostNativeSourceEvidence:
    """Run and replay one local GAP batch for all unique occupied inclusions."""

    if type(time_reversal) is not bool:
        raise TypeError("time_reversal must be a boolean")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    verified_runtime = _runtime_snapshot(runtime, timeout=timeout)
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository_root must be a directory")
    supplied, unique = _records(records)

    task4 = build_evidence(
        unique,
        runtime=verified_runtime,
        repository_root=root,
        time_reversal=time_reversal,
        timeout_seconds=timeout,
    )
    request = task4.request
    response = task4.response
    unique_ids = tuple(inclusion.inclusion_id for inclusion in request.inclusions)
    if unique_ids != tuple(record.wyckoff_id for record in unique):
        raise ValueError("Task4 inclusion order differs from unique catalogue order")

    element_labels = tuple(
        ("1",)
        + tuple(f"g{index}" for index in range(1, len(inclusion.literal_elements)))
        for inclusion in request.inclusions
    )
    execution = export_gap_inclusion_batch_raw(
        request.action,
        request.inclusions,
        element_label_sequences=element_labels,
        time_reversal=time_reversal,
        cwd=root,
        command=(verified_runtime.executable, "-q"),
        finite_group_ids=unique_ids,
        timeout_seconds=timeout,
    )
    verify_gap_batch_launcher_execution(execution, require_release=False)
    provenance = execution.provenance
    assert provenance is not None
    verify_gap_batch_execution_provenance(provenance)
    replay = replay_gap_inclusion_batch_artifact(execution.spec, execution.raw_output)
    task5_ids = tuple(member.inclusion.inclusion_id for member in execution.spec.members)
    if (
        replay.batch_input_digest != provenance.batch_input_digest
        or replay.batch_raw_output_digest != provenance.batch_raw_output_digest
        or replay.request_input_digest != provenance.request_input_digest
        or set(task5_ids) != set(unique_ids)
        or tuple(member.inclusion_id for member in replay.members) != task5_ids
    ):
        raise ValueError("Task5 pure replay differs from issued batch provenance")
    for child, spec_member in zip(
        execution.member_executions,
        execution.spec.members,
        strict=True,
    ):
        verify_gap_batch_member_execution(
            child,
            action=request.action,
            inclusion=spec_member.inclusion,
            element_labels=spec_member.element_labels,
            time_reversal=time_reversal,
            finite_group_id=spec_member.finite_group_id,
            require_release=False,
        )
        if (
            child.attestation.release_certified
            or child.attestation.runtime_manifest_digest is not None
        ):
            raise ValueError("local GAP execution cannot claim release authority")

    after = _runtime_snapshot(verified_runtime, timeout=timeout)
    if after.executable_sha256 != verified_runtime.executable_sha256:
        raise GapRuntimeError("GAP executable changed during host evidence execution")
    frozen_provenance = _freeze_json(host_provenance(verified_runtime))
    assert isinstance(frozen_provenance, Mapping)
    result = HostNativeSourceEvidence(
        instance_wyckoff_ids=tuple(record.wyckoff_id for record in supplied),
        unique_inclusion_ids=unique_ids,
        time_reversal=time_reversal,
        task4_request=request,
        task4_response=response,
        task5_execution=execution,
        task5_replay=replay,
        provenance=frozen_provenance,
        _construction_seal=_HOST_SOURCE_EVIDENCE_SEAL,
    )
    _register_host_source_evidence(result)
    return verify_host_source_evidence(result)


def _host_source_evidence_mapping(
    evidence: HostNativeSourceEvidence,
) -> dict[str, object]:
    checked = verify_host_source_evidence(evidence)
    return {
        "instance_wyckoff_ids": list(checked.instance_wyckoff_ids),
        "member_attestations": [
            launcher_execution_attestation_mapping(child.attestation)
            for child in checked.task5_execution.member_executions
        ],
        "provenance": _plain_frozen(checked.provenance),
        "record_type": "host-native-source-evidence",
        "schema_version": 1,
        "task4_request": json.loads(
            canonical_gap_classifier_json(checked.task4_request)
        ),
        "task4_response": json.loads(
            canonical_gap_classifier_json(checked.task4_response)
        ),
        "task5_raw_output": json.loads(
            checked.task5_execution.raw_output.decode("utf-8")
        ),
        "time_reversal": checked.time_reversal,
        "unique_inclusion_ids": list(checked.unique_inclusion_ids),
    }


def _plain_frozen(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_frozen(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_frozen(item) for item in value]
    return value


def _restore_host_source_evidence(
    mapping: Mapping[str, object],
    *,
    runtime: GapRuntime,
    timeout: int,
) -> HostNativeSourceEvidence:
    expected_fields = {
        "instance_wyckoff_ids",
        "member_attestations",
        "provenance",
        "record_type",
        "schema_version",
        "task4_request",
        "task4_response",
        "task5_raw_output",
        "time_reversal",
        "unique_inclusion_ids",
    }
    if set(mapping) != expected_fields:
        raise ValueError("cached host source evidence fields differ")
    if (
        mapping["record_type"] != "host-native-source-evidence"
        or mapping["schema_version"] != 1
        or type(mapping["time_reversal"]) is not bool
    ):
        raise ValueError("cached host source evidence schema differs")
    verified_runtime = _runtime_snapshot(runtime, timeout=timeout)
    if mapping["provenance"] != host_provenance(verified_runtime):
        raise ValueError("cached host source provenance differs from the runtime")
    request = loads_gap_classifier_request(_canonical_json(mapping["task4_request"]))
    response = loads_gap_classifier_response(_canonical_json(mapping["task4_response"]))
    unique_ids = tuple(mapping["unique_inclusion_ids"])
    instances = tuple(mapping["instance_wyckoff_ids"])
    if (
        any(type(item) is not str for item in unique_ids + instances)
        or tuple(item.inclusion_id for item in request.inclusions) != unique_ids
    ):
        raise ValueError("cached host source inclusion bindings differ")
    element_labels = tuple(
        ("1",)
        + tuple(f"g{index}" for index in range(1, len(inclusion.literal_elements)))
        for inclusion in request.inclusions
    )
    spec = make_gap_inclusion_batch_spec(
        request.action,
        request.inclusions,
        element_label_sequences=element_labels,
        time_reversal=mapping["time_reversal"],
        finite_group_ids=unique_ids,
    )
    raw_output = _canonical_json(mapping["task5_raw_output"])
    attestations = mapping["member_attestations"]
    if not isinstance(attestations, list) or any(
        not isinstance(item, Mapping) for item in attestations
    ):
        raise TypeError("cached host source attestations are invalid")
    execution = restore_diagnostic_gap_batch_execution(
        spec,
        raw_output,
        tuple(attestations),
        expected_resolved_launcher_digest=_bar._resolved_launcher_digest(
            (verified_runtime.executable, "-q"),
            Path(verified_runtime.executable).resolve(strict=True),
        ),
    )
    replay = replay_gap_inclusion_batch_artifact(spec, raw_output)
    frozen = _freeze_json(mapping["provenance"])
    assert isinstance(frozen, Mapping)
    result = HostNativeSourceEvidence(
        instance_wyckoff_ids=instances,
        unique_inclusion_ids=unique_ids,
        time_reversal=mapping["time_reversal"],
        task4_request=request,
        task4_response=response,
        task5_execution=execution,
        task5_replay=replay,
        provenance=frozen,
        _construction_seal=_HOST_SOURCE_EVIDENCE_SEAL,
    )
    _register_host_source_evidence(result)
    return verify_host_source_evidence(result)


def _spatial_source_table(table: FiniteGroupTable) -> FiniteGroupTable:
    if not table.group_id.endswith("+onsite-T"):
        return table
    if len(table.element_order) % 2:
        raise ValueError("graded local finite table has odd order")
    size = len(table.element_order) // 2
    spatial_order = table.element_order[:size]
    expected = spatial_order + ("T",) + tuple(
        f"{item}+T" for item in spatial_order[1:]
    )
    if table.element_order != expected:
        raise ValueError("graded local finite table is not the canonical spatial x C2 order")
    spatial_multiplication = tuple(
        tuple(table.multiplication_table[left][right] for right in range(size))
        for left in range(size)
    )
    if any(value >= size for row in spatial_multiplication for value in row):
        raise ValueError("graded local table does not close on its spatial half")
    return FiniteGroupTable(
        table.group_id.removesuffix("+onsite-T"),
        spatial_order,
        0,
        spatial_multiplication,
        table.inverse_indices[:size],
    )


def _replaying_artifact_plan(
    *,
    domain: str,
    inputs: object,
    produce,
) -> ArtifactPlan:
    """Cache only a semantic receipt; reconstruct and replay the typed value."""

    holder: dict[str, object] = {}

    def build() -> bytes:
        value = produce()
        payload = _canonical_json(
            {
                "record_type": "host-native-artifact-receipt",
                "schema_version": 1,
                "semantic_digest": artifact_semantic_digest(value),
            }
        )
        holder["payload"] = payload
        holder["value"] = value
        return payload

    def verify(data: bytes):
        if type(data) is not bytes:
            raise TypeError("host artifact receipt must be bytes")
        try:
            mapping = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("host artifact receipt is not canonical JSON") from error
        if (
            not isinstance(mapping, Mapping)
            or set(mapping)
            != {"record_type", "schema_version", "semantic_digest"}
            or mapping.get("record_type") != "host-native-artifact-receipt"
            or mapping.get("schema_version") != 1
            or _canonical_json(mapping) != data
        ):
            raise ValueError("host artifact receipt bytes differ")
        if holder.get("payload") == data:
            value = holder["value"]
        else:
            value = produce()
        if artifact_semantic_digest(value) != mapping.get("semantic_digest"):
            raise ValueError("host artifact semantic replay differs")
        return value

    return ArtifactPlan(
        build=build,
        verify=verify,
        plan_digest=_digest(domain, inputs),
    )


def _host_character_context(
    ambient: HostNativeAmbientArtifact,
):
    checked = verify_host_ambient_artifact(ambient)
    graded = checked.source_evidence.time_reversal
    spatial = checked.spatial_parent if graded else checked
    if spatial is None:
        raise ValueError("graded host character context lacks a spatial parent")
    verify_host_ambient_artifact(spatial)
    spatial_resolution = spatial.resolution
    generator_count = len(
        spatial_resolution.affine_pcp_certificate.pcp_normal_form.relative_orders
    )
    spatial_basis = character_basis_certificate(
        tuple(f"p{index + 1}" for index in range(generator_count)),
        pcp_presentation_relators(spatial_resolution),
        group_id=spatial_resolution.group_id,
        resolution_id=spatial_resolution.resolution_id,
        presentation_kind="pcp-presentation",
        finite_group_table_digest=None,
    )
    report = _cochains.verify_character_basis(
        spatial_basis,
        spatial_resolution,
        spatial.authority,
    )
    if not report.valid:
        raise ValueError("host spatial character basis failed replay")
    if graded:
        basis = _cochains.adjoin_onsite_time_reversal_character(
            spatial_basis,
            spatial_resolution,
            checked.authority,
            graded_resolution=checked.resolution,
        )
        grade = GF2Character((0,) * len(spatial_basis.generator_order) + (1,))
        return basis, grade, spatial_basis, spatial_resolution
    grade = GF2Character((0,) * len(spatial_basis.generator_order))
    return spatial_basis, grade, None, None


@dataclass(frozen=True, slots=True)
class HostNativeClassifierBackend(ClassifierBackendAuthority):
    """Host-only classifier backend; all returned results remain diagnostic."""

    runtime: GapRuntime
    repository_root: Path
    identity: BackendIdentity = field(init=False)

    def __post_init__(self) -> None:
        runtime = _runtime_snapshot(self.runtime, timeout=30)
        root = Path(self.repository_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository_root must be a directory")
        inventory = source_inventory_digest()
        environment = host_provenance(runtime)
        identity = BackendIdentity(
            gap_environment_digest=_digest("gap-environment", environment),
            affine_pcp_conversion_digest=_digest(
                "algorithm", "task4-host-affine-pcp-v1"
            ),
            affine_pcp_transport_digest=_digest(
                "algorithm", "task5-host-literal-transport-v1"
            ),
            target_model_digest=_digest("algorithm", "host-relative-target-v1"),
            local_library_digest=_digest(
                "local-library", {"inventory": inventory, "version": 1}
            ),
            ambient_algorithm_digest=_digest(
                "ambient-algorithm", {"inventory": inventory, "version": 1}
            ),
            local_algorithm_digest=_digest(
                "local-algorithm", {"inventory": inventory, "version": 1}
            ),
            inclusion_algorithm_digest=_digest(
                "inclusion-algorithm", {"inventory": inventory, "version": 1}
            ),
            relative_algorithm_digest=_digest(
                "relative-algorithm", {"inventory": inventory, "version": 1}
            ),
        )
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "repository_root", root)
        object.__setattr__(self, "identity", identity)

    def ambient_resolution_plan(
        self,
        request,
        resolved_orbits: tuple[ResolvedOrbit, ...],
        timeout_seconds: int,
    ) -> ArtifactPlan:
        resolved = tuple(resolved_orbits)
        if not resolved or any(type(item) is not ResolvedOrbit for item in resolved):
            raise TypeError("host ambient stage requires resolved occupied orbits")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        records = tuple(item.record for item in resolved)

        def produce_sources():
            if request.time_reversal:
                spatial_source = build_host_source_evidence(
                    records,
                    runtime=self.runtime,
                    time_reversal=False,
                    timeout=timeout_seconds,
                    repository_root=self.repository_root,
                )
                spatial = assemble_host_ambient_artifact(spatial_source)
            else:
                spatial_source = None
                spatial = None
            source = build_host_source_evidence(
                records,
                runtime=self.runtime,
                time_reversal=request.time_reversal,
                timeout=timeout_seconds,
                repository_root=self.repository_root,
            )
            value = assemble_host_ambient_artifact(
                source,
                spatial_parent=spatial,
            )
            return spatial_source, source, value

        holder: dict[str, object] = {}

        def build() -> bytes:
            spatial_source, source, value = produce_sources()
            mapping = {
                "record_type": "host-native-ambient-evidence",
                "schema_version": 1,
                "semantic_digest": artifact_semantic_digest(value),
                "source": _host_source_evidence_mapping(source),
                "spatial_source": (
                    None
                    if spatial_source is None
                    else _host_source_evidence_mapping(spatial_source)
                ),
            }
            payload = _canonical_json(mapping)
            holder["payload"] = payload
            holder["value"] = value
            return payload

        def verify(data: bytes):
            try:
                mapping = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("cached host ambient evidence is not JSON") from error
            if (
                not isinstance(mapping, Mapping)
                or set(mapping)
                != {
                    "record_type",
                    "schema_version",
                    "semantic_digest",
                    "source",
                    "spatial_source",
                }
                or mapping.get("record_type") != "host-native-ambient-evidence"
                or mapping.get("schema_version") != 1
                or _canonical_json(mapping) != data
            ):
                raise ValueError("cached host ambient evidence bytes differ")
            if holder.get("payload") == data:
                value = holder["value"]
            else:
                raw_source = mapping["source"]
                raw_spatial = mapping["spatial_source"]
                if not isinstance(raw_source, Mapping):
                    raise TypeError("cached host ambient source is invalid")
                if raw_spatial is not None and not isinstance(raw_spatial, Mapping):
                    raise TypeError("cached spatial ambient source is invalid")
                spatial = (
                    None
                    if raw_spatial is None
                    else assemble_host_ambient_artifact(
                        _restore_host_source_evidence(
                            raw_spatial,
                            runtime=self.runtime,
                            timeout=timeout_seconds,
                        )
                    )
                )
                source = _restore_host_source_evidence(
                    raw_source,
                    runtime=self.runtime,
                    timeout=timeout_seconds,
                )
                value = assemble_host_ambient_artifact(
                    source,
                    spatial_parent=spatial,
                )
            if artifact_semantic_digest(value) != mapping.get("semantic_digest"):
                raise ValueError("cached host ambient semantic replay differs")
            return value

        plan_inputs = {
                "request": classification_request_digest(request),
                "records": [item.record.wyckoff_id for item in resolved],
                "runtime": self.identity.gap_environment_digest,
            }
        return ArtifactPlan(
            build=build,
            verify=verify,
            plan_digest=_digest("host-ambient-plan", plan_inputs),
        )

    def local_skeleton_plans(
        self,
        request,
        resolved_orbit: ResolvedOrbit,
        ambient: object,
        timeout_seconds: int,
    ) -> Sequence[LocalSkeletonPlan]:
        if type(resolved_orbit) is not ResolvedOrbit:
            raise TypeError("host local stage requires an exact ResolvedOrbit")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        checked = verify_host_ambient_artifact(ambient)  # type: ignore[arg-type]
        if checked.source_evidence.time_reversal is not request.time_reversal:
            raise ValueError(
                "host ambient time-reversal mode differs from the classification request"
            )
        row = checked.inclusion_for(resolved_orbit.record.wyckoff_id)
        table = row.inclusion.source_resolution.finite_group
        if table is None:
            raise ValueError("host local inclusion lacks a finite group table")
        spatial_table = _spatial_source_table(table)
        graded = request.time_reversal

        normalization_digest = _digest(
            "host-local-normalization",
            {
                "element_order": list(spatial_table.element_order),
                "inclusion_id": row.inclusion_id,
                "table_digest": spatial_table.table_digest,
            },
        )

        if request.igg == "U1":
            resolution = checked.resolution
            basis, grade, _, _ = _host_character_context(checked)
            plans: list[LocalSkeletonPlan] = []
            for sector_index, rho in enumerate(basis.characters):
                local_rho = GF2Character(
                    tuple(
                        _cochains._word_character(resolution, rho, image)
                        for image in row.inclusion.source_element_images
                    )
                )
                local_grade = GF2Character(
                    tuple(
                        _cochains._word_character(resolution, grade, image)
                        for image in row.inclusion.source_element_images
                    )
                )

                def replay_u1(
                    rho: GF2Character = rho,
                    local_rho: GF2Character = local_rho,
                    local_grade: GF2Character = local_grade,
                ):
                    skeleton = u1_local_skeleton(table, local_grade, local_rho)
                    return make_u1_local_skeleton_evidence(
                        instance_id=resolved_orbit.instance_id,
                        source_table=table,
                        skeleton=skeleton,
                        ambient_rho=rho,
                        graded=graded,
                    )

                expected = replay_u1()
                semantic_digest = artifact_semantic_digest(expected)
                payload = _canonical_json(
                    {
                        "record_type": "host-native-local-skeleton",
                        "schema_version": 1,
                        "semantic_digest": semantic_digest,
                    }
                )

                def verify_u1(
                    data: bytes,
                    *,
                    payload: bytes = payload,
                    replay_u1=replay_u1,
                    semantic_digest: str = semantic_digest,
                ):
                    if data != payload:
                        raise ValueError("host U1 local skeleton artifact bytes differ")
                    value = replay_u1()
                    if artifact_semantic_digest(value) != semantic_digest:
                        raise ValueError("host U1 local skeleton replay differs")
                    return value

                plan = ArtifactPlan(
                    build=lambda payload=payload: payload,
                    verify=verify_u1,
                    plan_digest=_digest(
                        "u1-local-skeleton-plan",
                        {
                            "ambient_rho": list(rho.bits),
                            "graded": graded,
                            "inclusion_id": row.inclusion_id,
                            "instance_id": resolved_orbit.instance_id,
                            "request": classification_request_digest(request),
                            "sector_index": sector_index,
                            "source_table": table.table_digest,
                        },
                    ),
                )
                plans.append(
                    LocalSkeletonPlan(
                        plan=plan,
                        stabilizer_table_digest=str(table.table_digest),
                        stabilizer_normalization_digest=normalization_digest,
                        restricted_grade=expected.restricted_grade,
                        restricted_rho=expected.restricted_rho,
                        derived_q=expected.derived_q,
                    )
                )
            return tuple(plans)

        if request.igg != "Z2":
            raise ValueError("host local stage supports only Z2 or U1")

        def replay():
            spatial = enumerate_spatial_z2_skeletons(spatial_table)
            skeletons = (
                tuple(
                    child
                    for item in spatial
                    for child in enumerate_graded_z2_skeletons(item)
                )
                if graded
                else spatial
            )
            restricted_grade = (
                (0,) * len(spatial_table.element_order)
                + (1,) * len(spatial_table.element_order)
                if graded
                else (0,) * len(spatial_table.element_order)
            )
            return make_z2_local_skeleton_evidence(
                instance_id=resolved_orbit.instance_id,
                source_table=spatial_table,
                skeletons=skeletons,
                restricted_grade=restricted_grade,
                graded=graded,
            )

        expected = replay()
        semantic_digest = artifact_semantic_digest(expected)
        payload = _canonical_json(
            {
                "record_type": "host-native-local-skeleton",
                "schema_version": 1,
                "semantic_digest": semantic_digest,
            }
        )

        def verify(data: bytes):
            if data != payload:
                raise ValueError("host local skeleton artifact bytes differ")
            value = replay()
            if artifact_semantic_digest(value) != semantic_digest:
                raise ValueError("host local skeleton replay differs")
            return value

        plan_digest = _digest(
            "local-skeleton-plan",
            {
                "graded": graded,
                "inclusion_id": row.inclusion_id,
                "instance_id": resolved_orbit.instance_id,
                "request": classification_request_digest(request),
                "source_table": spatial_table.table_digest,
            },
        )
        plan = ArtifactPlan(
            build=lambda: payload,
            verify=verify,
            plan_digest=plan_digest,
        )
        return (
            LocalSkeletonPlan(
                plan=plan,
                stabilizer_table_digest=str(spatial_table.table_digest),
                stabilizer_normalization_digest=normalization_digest,
                restricted_grade=expected.restricted_grade,
                restricted_rho=None,
                derived_q=None,
            ),
        )

    def inclusion_plan(
        self,
        request,
        resolved_orbit: ResolvedOrbit,
        ambient: object,
        timeout_seconds: int,
    ) -> ArtifactPlan:
        if type(resolved_orbit) is not ResolvedOrbit:
            raise TypeError("host inclusion stage requires an exact ResolvedOrbit")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        def produce():
            checked = verify_host_ambient_artifact(ambient)  # type: ignore[arg-type]
            if checked.source_evidence.time_reversal is not request.time_reversal:
                raise ValueError("host inclusion time-reversal mode differs")
            return HostNativeResolvedInclusion(
                resolved_orbit.instance_id,
                checked.inclusion_for(resolved_orbit.record.wyckoff_id),
            )

        return _replaying_artifact_plan(
            domain="host-inclusion-plan",
            inputs={
                "ambient": artifact_semantic_digest(ambient),
                "inclusion_id": resolved_orbit.record.wyckoff_id,
                "instance_id": resolved_orbit.instance_id,
                "request": classification_request_digest(request),
            },
            produce=produce,
        )

    def relative_layer_plan(
        self,
        request,
        resolved_orbits: tuple[ResolvedOrbit, ...],
        ambient: object,
        local_skeletons: tuple[tuple[object, ...], ...],
        inclusions: tuple[object, ...],
        timeout_seconds: int,
    ) -> ArtifactPlan:
        if request.igg not in ("Z2", "U1"):
            raise ValueError("host relative classification supports Z2 or U1")
        resolved = tuple(resolved_orbits)
        local_rows = tuple(tuple(row) for row in local_skeletons)
        inclusion_rows = tuple(inclusions)
        if not (
            resolved
            and len(resolved) == len(local_rows) == len(inclusion_rows)
            and (
                request.igg == "U1"
                or all(len(row) == 1 for row in local_rows)
            )
        ):
            raise ValueError("host relative stage dependencies differ from occupied orbits")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        def produce():
            from .classification_schema import ObstructedBranch
            from .relative_complex import RelativeProblem, assemble_relative_problem
            from .z2_classifier import (
                FiniteAffineStratum,
                _certified_gf2_complex,
                _certified_gf2_restriction,
                certify_centralizer_action,
                classify_z2_diagnostic,
                coordinate_z2_defect,
                coordinate_z2_marking_shift,
                make_diagnostic_z2_branch,
                make_diagnostic_z2_problem,
            )

            checked = verify_host_ambient_artifact(ambient)  # type: ignore[arg-type]
            if checked.source_evidence.time_reversal is not request.time_reversal:
                raise ValueError("host relative time-reversal mode differs")
            bound = tuple(inclusion_rows)
            if any(type(item) is not HostNativeResolvedInclusion for item in bound):
                raise TypeError("host relative stage requires resolved inclusions")
            if tuple(item.instance_id for item in bound) != tuple(
                item.instance_id for item in resolved
            ):
                raise ValueError("host relative inclusion instance order differs")

            if request.igg == "U1":
                from .classification_schema import ObstructedBranch
                from .u1_classifier import LocalU1Data, TorsorStratum, make_u1_sector_problem

                basis, grade, spatial_basis, spatial_resolution = (
                    _host_character_context(checked)
                )
                if any(len(row) != len(basis.characters) for row in local_rows):
                    raise ValueError("host U1 local rows omit or duplicate rho sectors")
                for orbit_index, row in enumerate(local_rows):
                    for rho, value in zip(basis.characters, row, strict=True):
                        if (
                            type(value) is not LocalSkeletonEvidence
                            or value.coefficient_kind != "U1"
                            or value.diagnostic
                            or value.instance_id != resolved[orbit_index].instance_id
                            or value.ambient_rho != rho
                            or value.graded is not request.time_reversal
                            or len(value.skeletons) != 1
                        ):
                            raise ValueError(
                                "host U1 local evidence differs from the exhaustive rho order"
                            )
                source = CertifiedCochainProblem(
                    checked.resolution,
                    tuple(item.inclusion for item in checked.inclusions),
                    basis,
                )
                problems = []
                for sector_index, rho in enumerate(basis.characters):
                    local_data = tuple(
                        LocalU1Data(
                            inclusion.instance_id,
                            inclusion.source.inclusion,
                            inclusion.source.bar_equivalence,
                            local_rows[orbit_index][sector_index].skeletons[0],
                        )
                        for orbit_index, inclusion in enumerate(bound)
                    )
                    problems.append(
                        make_u1_sector_problem(
                            source,
                            rho,
                            grade=grade,
                            authority=checked.authority,
                            local_data=local_data,
                            spatial_character_basis=spatial_basis,
                            spatial_resolution=spatial_resolution,
                            allow_diagnostic=True,
                        )
                    )
                coverage = make_u1_sector_coverage(
                    source=source,
                    authority=checked.authority,
                    grade=grade,
                    problems=tuple(problems),
                    spatial_character_basis=spatial_basis,
                    spatial_resolution=spatial_resolution,
                    allow_diagnostic=True,
                )
                strata = tuple(
                    sorted(
                        (
                            outcome.result
                            for outcome in coverage.outcomes
                            if type(outcome.result) is TorsorStratum
                        ),
                        key=lambda item: item.stratum_id,
                    )
                )
                obstructions = tuple(
                    sorted(
                        (
                            outcome.result
                            for outcome in coverage.outcomes
                            if type(outcome.result) is ObstructedBranch
                        ),
                        key=lambda item: item.stratum_id,
                    )
                )
                failures = tuple(
                    outcome.failure
                    for outcome in coverage.outcomes
                    if outcome.failure is not None
                )
                problem_by_sector = {item.sector_id: item for item in problems}
                global_weyl = tuple(
                    sorted(
                        (
                            stratum.stratum_id,
                            tuple(
                                WeylOrbitData(
                                    binding.instance_id,
                                    binding.skeleton,
                                    _diagnostic_bar_evaluator(
                                        binding.bar_equivalence,
                                        GF2Character(binding.skeleton.rho_values),
                                    ),
                                )
                                for binding in problem_by_sector[
                                    stratum.certificate.sector_id
                                ].bindings
                            ),
                        )
                        for stratum in strata
                    )
                )
                relative_digest = _digest(
                    "joint-host-u1-relative-source",
                    {
                        "character_certificate_id": basis.certificate_id,
                        "coverage_id": coverage.coverage_id,
                        "inclusion_ids": [item.certificate_id for item in bound],
                        "instances": [item.instance_id for item in resolved],
                        "sector_ids": [item.sector_id for item in problems],
                    },
                )
                return JointLayerMaterial(
                    branch_ids=tuple(
                        sorted(item.stratum_id for item in strata + obstructions)
                    ),
                    framed_strata=strata,
                    local_arrows=(),
                    global_weyl_data=global_weyl,
                    obstructed_branches=obstructions,
                    failures=failures,
                    source_artifact_digests=(("relative", relative_digest),),
                    u1_sector_coverage=coverage,
                )

            evidences = tuple(row[0] for row in local_rows)
            if any(
                type(item) is not LocalSkeletonEvidence
                or item.coefficient_kind != "Z2"
                or item.diagnostic
                or item.graded is not request.time_reversal
                for item in evidences
            ):
                raise ValueError("host relative stage received invalid Z2 local evidence")
            if tuple(item.instance_id for item in evidences) != tuple(
                item.instance_id for item in resolved
            ):
                raise ValueError("host relative local instance order differs")

            ambient_complex = _certified_gf2_complex(checked.resolution)
            locals_ = tuple(
                _certified_gf2_complex(item.source.inclusion.source_resolution)
                for item in bound
            )
            restrictions = tuple(
                _certified_gf2_restriction(
                    item.source.inclusion,
                    instance_id=item.instance_id,
                    ambient=ambient_complex,
                    local=local,
                )
                for item, local in zip(bound, locals_, strict=True)
            )
            branches = []
            for skeleton_tuple in itertools.product(
                *(item.skeletons for item in evidences)
            ):
                defects = tuple(
                    coordinate_z2_defect(
                        skeleton,
                        item.source.bar_equivalence,
                        checked.authority,
                    )
                    for skeleton, item in zip(
                        skeleton_tuple, bound, strict=True
                    )
                )
                source = RelativeProblem(
                    "gf2",
                    ambient_complex,
                    locals_,
                    restrictions,
                    tuple(item.coordinates for item in defects),
                )
                matrices = assemble_relative_problem(source)
                skeleton_ids = tuple(item.skeleton_id for item in skeleton_tuple)
                actions = tuple(
                    certify_centralizer_action(
                        matrices,
                        skeleton_ids=skeleton_ids,
                        instance_id=inclusion.instance_id,
                        skeleton=skeleton,
                        component=component,
                        marking_coordinates=coordinate_z2_marking_shift(
                            component.marking_shift,
                            inclusion.source.bar_equivalence,
                            checked.authority,
                        ),
                        bar_equivalence=inclusion.source.bar_equivalence,
                        authority=checked.authority,
                    )
                    for inclusion, skeleton in zip(bound, skeleton_tuple, strict=True)
                    for component in skeleton.centralizer_components
                )
                branches.append(
                    make_diagnostic_z2_branch(
                        source_problem=source,
                        matrices=matrices,
                        skeleton_ids=skeleton_ids,
                        centralizer_actions=actions,
                    )
                )
            branches.sort(key=lambda item: (item.skeleton_ids, item.branch_id))
            results = classify_z2_diagnostic(make_diagnostic_z2_problem(branches))
            strata = tuple(
                sorted(
                    (item for item in results if type(item) is FiniteAffineStratum),
                    key=lambda item: item.stratum_id,
                )
            )
            obstructions = tuple(
                sorted(
                    (item for item in results if type(item) is ObstructedBranch),
                    key=lambda item: item.stratum_id,
                )
            )
            relative_digest = _digest(
                "joint-host-z2-relative-source",
                {
                    "branch_ids": [item.branch_id for item in branches],
                    "inclusion_ids": [item.certificate_id for item in bound],
                    "instances": [item.instance_id for item in resolved],
                },
            )
            return JointLayerMaterial(
                branch_ids=tuple(
                    sorted(item.stratum_id for item in strata + obstructions)
                ),
                framed_strata=strata,
                local_arrows=(),
                global_weyl_data=(),
                obstructed_branches=obstructions,
                failures=(),
                source_artifact_digests=(("relative", relative_digest),),
            )

        return _replaying_artifact_plan(
            domain=f"host-relative-{request.igg.lower()}-plan",
            inputs={
                "ambient": artifact_semantic_digest(ambient),
                "inclusions": [artifact_semantic_digest(item) for item in inclusions],
                "locals": [
                    [artifact_semantic_digest(item) for item in row]
                    for row in local_skeletons
                ],
                "request": classification_request_digest(request),
            },
            produce=produce,
        )


__all__ = [
    "HostNativeAmbientArtifact",
    "HostNativeClassifierBackend",
    "HostNativeInclusionArtifact",
    "HostNativeSourceEvidence",
    "assemble_host_ambient_artifact",
    "build_host_source_evidence",
    "verify_host_ambient_artifact",
    "verify_host_source_evidence",
]
