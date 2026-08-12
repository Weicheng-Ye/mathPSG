"""Host-native GAP source evidence for the joint PSG classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
import weakref

from .bar_evaluator import (
    GapBatchArtifactReplay,
    GapBatchLauncherExecution,
    export_gap_inclusion_batch_raw,
    replay_gap_inclusion_batch_artifact,
    verify_gap_batch_execution_provenance,
    verify_gap_batch_launcher_execution,
    verify_gap_batch_member_execution,
)
from .catalogue import catalogue_record_order_key, validate_catalogue_record_identity
from .catalogue_schema import CatalogueRecord
from .gap_classifier import (
    GAPClassifierRequest,
    GAPClassifierResponse,
    canonical_gap_classifier_json,
    loads_gap_classifier_request,
    loads_gap_classifier_response,
    verify_affine_pcp_certificate,
)
from .live_evidence import build_evidence
from .local_gap import GapRuntime, GapRuntimeError, host_provenance, probe_gap


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
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_HOST_SOURCE_EVIDENCE_SEAL = object()
_HOST_SOURCE_EVIDENCE_REGISTRY: dict[
    int, tuple[weakref.ReferenceType["HostNativeSourceEvidence"], tuple[object, ...]]
] = {}


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


__all__ = [
    "HostNativeSourceEvidence",
    "build_host_source_evidence",
    "verify_host_source_evidence",
]
