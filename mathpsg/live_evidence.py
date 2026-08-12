"""Verified host-native affine/PCP evidence from the copied GAP backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

from .catalogue import catalogue_record_order_key, validate_catalogue_record_identity
from .catalogue_schema import CatalogueRecord
from .gap_classifier import (
    AffinePCPIsomorphismCertificate,
    AffineTransformation,
    GAPClassifierRequest,
    GAPClassifierResponse,
    LiteralStabilizerInclusion,
    canonical_gap_classifier_json,
    literal_element_authority_digest,
    make_certified_space_group_action,
    make_gap_classifier_request,
    run_gap_classifier,
)
from .local_gap import GapRuntime, host_provenance


_Q0 = "q(0,1)"
_Q1 = "q(1,1)"
_IDENTITY = AffineTransformation(
    ((_Q1, _Q0, _Q0), (_Q0, _Q1, _Q0), (_Q0, _Q0, _Q1)),
    (_Q0, _Q0, _Q0),
)


class EvidenceError(RuntimeError):
    """The local GAP process could not produce replayable evidence."""


def _string_matrix(value: object, path: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path}: expected matrix")
    rows = tuple(tuple(row) for row in value)
    if any(any(type(item) is not str for item in row) for row in rows):
        raise TypeError(f"{path}: expected exact rational strings")
    return rows


def _string_vector(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path}: expected vector")
    result = tuple(value)
    if any(type(item) is not str for item in result):
        raise TypeError(f"{path}: expected exact rational strings")
    return result


def _affine(value: object, path: str) -> AffineTransformation:
    if not isinstance(value, Mapping) or set(value) != {"matrix", "translation"}:
        raise TypeError(f"{path}: expected canonical affine transformation")
    return AffineTransformation(
        _string_matrix(value["matrix"], f"{path}.matrix"),
        _string_vector(value["translation"], f"{path}.translation"),
    )


def catalogue_records_gap_request(
    records: Sequence[CatalogueRecord], *, time_reversal: bool
) -> GAPClassifierRequest:
    """Copy the reviewed generic catalogue-to-Task4 adapter."""

    if type(time_reversal) is not bool:
        raise TypeError("time_reversal must be boolean")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("catalogue records must be a finite sequence")
    supplied = tuple(records)
    if not supplied:
        raise ValueError("catalogue records must be nonempty")
    if any(type(record) is not CatalogueRecord for record in supplied):
        raise TypeError("catalogue adapter requires CatalogueRecord values")
    replayed = tuple(validate_catalogue_record_identity(record) for record in supplied)
    unique_by_id: dict[str, CatalogueRecord] = {}
    for record in replayed:
        prior = unique_by_id.get(record.wyckoff_id)
        if prior is not None and prior != record:
            raise ValueError("catalogue records disagree for one duplicate Wyckoff ID")
        unique_by_id[record.wyckoff_id] = record
    ordered = tuple(sorted(unique_by_id.values(), key=catalogue_record_order_key))
    setting_keys = {
        (record.space_group["international_number"], record.space_group["setting"])
        for record in ordered
    }
    if len(setting_keys) != 1 or any(
        record.space_group_action != ordered[0].space_group_action
        for record in ordered[1:]
    ):
        raise ValueError("catalogue records must share one space-group setting/action")

    action_mapping = ordered[0].space_group_action
    generators_value = action_mapping["source_generators"]
    if not isinstance(generators_value, Sequence):
        raise TypeError("catalogue source_generators must be a sequence")
    action = make_certified_space_group_action(
        tuple(
            _affine(value, f"space_group_action.source_generators[{index}]")
            for index, value in enumerate(generators_value)
        ),
        _string_matrix(
            action_mapping["translation_basis"],
            "space_group_action.translation_basis",
        ),
    )

    inclusions: list[LiteralStabilizerInclusion] = []
    for record in ordered:
        stabilizer_value = record.stabilizer["embedded_elements"]
        if not isinstance(stabilizer_value, Sequence):
            raise TypeError("catalogue stabilizer elements must be a sequence")
        literal_elements = tuple(
            _affine(value, f"stabilizer.embedded_elements[{index}]")
            for index, value in enumerate(stabilizer_value)
        )
        identity_indices = tuple(
            index for index, element in enumerate(literal_elements) if element == _IDENTITY
        )
        if len(identity_indices) != 1:
            raise ValueError("catalogue literal stabilizer requires one exact identity")
        identity_index = identity_indices[0]
        literal_elements = (
            literal_elements[identity_index],
            *literal_elements[:identity_index],
            *literal_elements[identity_index + 1 :],
        )
        inclusions.append(
            LiteralStabilizerInclusion(
                inclusion_id=record.wyckoff_id,
                literal_stabilizer_digest=record.embedding_digest,
                literal_element_digest=literal_element_authority_digest(
                    literal_elements
                ),
                literal_elements=literal_elements,
            )
        )
    return make_gap_classifier_request(
        action, tuple(inclusions), time_reversal=time_reversal
    )


@dataclass(frozen=True, slots=True)
class HostNativeEvidenceBatch:
    """Replay-verified conversion evidence, explicitly not release authority."""

    member_ids: tuple[str, ...]
    time_reversal: bool
    request: GAPClassifierRequest
    response: GAPClassifierResponse
    affine_certificate: AffinePCPIsomorphismCertificate
    canonical_data: bytes
    provenance: Mapping[str, object]
    certification_status: str = "host-native"


def build_evidence(
    records: Sequence[CatalogueRecord],
    *,
    runtime: GapRuntime,
    repository_root: Path,
    time_reversal: bool,
    timeout_seconds: int = 300,
) -> HostNativeEvidenceBatch:
    """Run the existing conversion-only GAP protocol with the host executable."""

    if type(runtime) is not GapRuntime:
        raise TypeError("build_evidence requires GapRuntime")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("evidence timeout must be a positive integer")
    root = Path(repository_root).resolve(strict=True)
    exporter = root / "gap" / "classifier" / "export_problem.g"
    if not exporter.is_file():
        raise EvidenceError("GAP classifier exporter is unavailable")
    request = catalogue_records_gap_request(records, time_reversal=time_reversal)
    response = run_gap_classifier(
        request,
        timeout_seconds=timeout_seconds,
        command=(runtime.executable, "-q", os.fspath(exporter), "--"),
        cwd=root,
    )
    if response.status != "conversion_only" or response.affine_pcp_certificate is None:
        failure = response.failures[0].message if response.failures else "unknown failure"
        raise EvidenceError(f"host GAP evidence failed: {failure}")
    member_ids = tuple(inclusion.inclusion_id for inclusion in request.inclusions)
    provenance = MappingProxyType(host_provenance(runtime))
    return HostNativeEvidenceBatch(
        member_ids=member_ids,
        time_reversal=time_reversal,
        request=request,
        response=response,
        affine_certificate=response.affine_pcp_certificate,
        canonical_data=canonical_gap_classifier_json(response),
        provenance=provenance,
    )


__all__ = [
    "EvidenceError",
    "HostNativeEvidenceBatch",
    "build_evidence",
    "catalogue_records_gap_request",
]
