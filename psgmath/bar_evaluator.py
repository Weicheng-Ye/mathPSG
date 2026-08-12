r"""Certified comparison with the normalized inhomogeneous bar resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any
import weakref

from .cochains import (
    BarComparisonBasisTrace,
    BarComparisonTerm,
    FiniteGroupTable,
    FreeResolutionCertificate,
    InclusionChainMapCertificate,
    LauncherExecutionAttestation,
    SparseGroupRingMatrix,
    ResolutionComparisonTerm,
    TargetBarResolutionEquivalence,
    TargetResolutionBasisTrace,
    TargetBarPhiTrace,
    Task5VerificationAuthority,
    VerificationIssue,
    VerificationReport,
    _canonical_json,
    _compose,
    _digest,
    _domain_digest,
    _fields,
    _inclusion_left,
    _inclusion_right,
    _normal_key,
    _pcp_action_and_decoder,
    _parse_finite,
    _parse_matrix,
    _parse_resolution,
    _reconstruct_comparison_maps,
    _strict_json,
    _task5_backend_binding,
    _task5_domain_digest,
    _make_launcher_execution_attestation,
    assemble_gap_free_resolution_certificate,
    diagnostic_residue_digests,
    free_resolution_mapping,
    gap_inclusion_projection_digest,
    inclusion_chain_map_mapping,
    launcher_execution_attestation_digest,
    launcher_execution_attestation_mapping,
    make_inclusion_chain_map_certificate,
    target_bar_equivalence_digest,
    verify_inclusion_chain_map,
    verify_resolution,
)
from .gap_classifier import (
    AffineTransformation,
    CertifiedSpaceGroupAction,
    AffinePCPIsomorphismCertificate,
    LiteralStabilizerInclusion,
    _action_mapping,
    _classifier_source_bytes,
    _compose_affine,
    _domain_digest as _classifier_domain_digest,
    _evaluate_pcp_word,
    _inclusion_mapping,
    _inverse_affine,
    _locked_environment_core,
    literal_element_authority_digest,
    make_certified_space_group_action,
)
from .gf2 import GF2Character
from .torus import Phase


_CANONICAL_FRACTION_RE = re.compile(r"(?:0|-?[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z")
_EXPORT_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAP_LAUNCHER_EXECUTION_FACTORY_SEAL = object()
_GAP_BATCH_LAUNCHER_EXECUTION_FACTORY_SEAL = object()
_GAP_BATCH_EXECUTION_PROVENANCE_FACTORY_SEAL = object()
_GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY: dict[int, tuple[Any, ...]] = {}
_GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY: dict[int, tuple[Any, ...]] = {}
_GAP_BATCH_EXECUTION_PROVENANCE_ISSUER_REGISTRY: dict[
    int, tuple[Any, ...]
] = {}
_SIGNED_RELEASE_CORPUS_ATTESTATION_ISSUER_REGISTRY: dict[
    int, tuple[Any, ...]
] = {}
_SIGNED_RELEASE_CORPUS_PROVENANCE_ISSUER_REGISTRY: dict[
    int, tuple[Any, ...]
] = {}
_SIGNED_RELEASE_CORPUS_BATCH_ISSUER_REGISTRY: dict[int, tuple[Any, ...]] = {}
_SIGNED_RELEASE_CORPUS_DOMAIN = "signed-release-corpus-v1"


@dataclass(frozen=True, slots=True)
class GapInclusionBatchMember:
    """One validated literal-inclusion member of a canonical GAP batch."""

    inclusion: LiteralStabilizerInclusion
    element_labels: tuple[str, ...]
    finite_group_id: str
    input_digest: str


@dataclass(frozen=True, slots=True)
class GapInclusionBatchSpec:
    """Canonical identity for one ambient action and its inclusion members."""

    action: CertifiedSpaceGroupAction
    members: tuple[GapInclusionBatchMember, ...]
    time_reversal: bool
    input_digest: str


@dataclass(frozen=True, slots=True)
class GapBatchMemberProvenance:
    """Canonical identity of one projection in an issued GAP batch."""

    member_index: int
    inclusion_id: str
    member_input_digest: str
    raw_output_digest: str
    projection_digest: str
    attestation_id: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class GapBatchExecutionProvenance:
    """Factory-issued, replayable parent provenance shared by batch children."""

    batch_id: str
    batch_input_digest: str
    batch_raw_output_digest: str
    request_input_digest: str
    batch_input: bytes
    request_input: bytes
    raw_output: bytes
    spec: GapInclusionBatchSpec
    members: tuple[GapBatchMemberProvenance, ...]
    _factory_seal: InitVar[object | None] = None

    def __post_init__(self, _factory_seal: object | None) -> None:
        if _factory_seal is not _GAP_BATCH_EXECUTION_PROVENANCE_FACTORY_SEAL:
            raise TypeError(
                "GapBatchExecutionProvenance construction requires the GAP batch factory"
            )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class GapLauncherExecution:
    """Exact bytes and process envelope produced by the GAP launcher."""

    raw_output: bytes
    attestation: LauncherExecutionAttestation
    _factory_seal: InitVar[object | None] = None
    batch_input_digest: str | None = None
    batch_raw_output_digest: str | None = None
    batch_member_input_digest: str | None = None
    batch_member_index: int | None = None
    batch_provenance: GapBatchExecutionProvenance | None = None

    def __post_init__(self, _factory_seal: object | None) -> None:
        if _factory_seal is not _GAP_LAUNCHER_EXECUTION_FACTORY_SEAL:
            raise TypeError(
                "GapLauncherExecution construction requires the GAP launcher factory"
            )
        if type(self.raw_output) is not bytes:
            raise TypeError("GAP launcher output must be exact bytes")
        if type(self.attestation) is not LauncherExecutionAttestation:
            raise TypeError("GAP launcher result requires a typed attestation")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class GapBatchLauncherExecution:
    """Exact one-process batch envelope and its factory-issued projections."""

    raw_output: bytes
    spec: GapInclusionBatchSpec
    request_input_digest: str
    member_executions: tuple[GapLauncherExecution, ...]
    provenance: GapBatchExecutionProvenance | None = None
    _factory_seal: InitVar[object | None] = None

    def __post_init__(self, _factory_seal: object | None) -> None:
        if _factory_seal is not _GAP_BATCH_LAUNCHER_EXECUTION_FACTORY_SEAL:
            raise TypeError(
                "GapBatchLauncherExecution construction requires the GAP batch factory"
            )
        if type(self.raw_output) is not bytes:
            raise TypeError("GAP batch launcher output must be exact bytes")
        if type(self.spec) is not GapInclusionBatchSpec:
            raise TypeError("GAP batch launcher requires a canonical specification")
        if _DIGEST_RE.fullmatch(self.request_input_digest) is None:
            raise ValueError("GAP batch request identity must be a sha256 digest")
        if any(type(item) is not GapLauncherExecution for item in self.member_executions):
            raise TypeError("GAP batch members must be exact launcher executions")
        if type(self.provenance) is not GapBatchExecutionProvenance:
            raise TypeError("GAP batch launcher requires issued batch provenance")


@dataclass(frozen=True, slots=True)
class GapBatchArtifactMemberReplay:
    """Pure replay result for one serialized batch member; never authority."""

    member_index: int
    inclusion_id: str
    member_input_digest: str
    raw_output: bytes
    raw_output_digest: str
    projection_digest: str


@dataclass(frozen=True, slots=True)
class GapBatchArtifactReplay:
    """Pure canonical batch replay with no issuer-registry side effects."""

    batch_input: bytes
    batch_input_digest: str
    request_input: bytes
    request_input_digest: str
    raw_output: bytes
    batch_raw_output_digest: str
    members: tuple[GapBatchArtifactMemberReplay, ...]


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _batch_member_provenance_mapping(
    value: GapBatchMemberProvenance,
) -> dict[str, object]:
    return {
        "attestation_id": value.attestation_id,
        "inclusion_id": value.inclusion_id,
        "member_index": value.member_index,
        "member_input_digest": value.member_input_digest,
        "projection_digest": value.projection_digest,
        "raw_output_digest": value.raw_output_digest,
    }


def _gap_batch_provenance_identity_payload(
    value: GapBatchExecutionProvenance,
) -> dict[str, object]:
    return {
        "batch_input_digest": value.batch_input_digest,
        "batch_raw_output_digest": value.batch_raw_output_digest,
        "members": [
            _batch_member_provenance_mapping(member) for member in value.members
        ],
        "request_input_digest": value.request_input_digest,
    }


def gap_batch_execution_provenance_mapping(
    value: GapBatchExecutionProvenance,
) -> dict[str, object]:
    """Return the canonical public identity of issued batch provenance."""

    verify_gap_batch_execution_provenance(value)
    return {
        **_gap_batch_provenance_identity_payload(value),
        "batch_id": value.batch_id,
        "record_type": "task5-gap-batch-execution-provenance",
        "schema_version": 1,
    }


def _gap_batch_request_bytes(spec: GapInclusionBatchSpec) -> bytes:
    return build_gap_inclusion_batch_export_program(spec).replace(
        "{output_path}",
        json.dumps("mathpsg-task5-gap-output.json", ensure_ascii=True),
    ).encode("utf-8")


def verify_gap_batch_execution_provenance(
    value: GapBatchExecutionProvenance,
) -> GapBatchExecutionProvenance:
    """Replay factory issuance, exact request bytes, envelope, and member universe."""

    if type(value) is not GapBatchExecutionProvenance:
        raise TypeError("expected exact GAP batch execution provenance")
    snapshot = _GAP_BATCH_EXECUTION_PROVENANCE_ISSUER_REGISTRY.get(id(value))
    if snapshot is None or snapshot[0]() is not value:
        raise ValueError("GAP batch provenance was not issued by its registry")
    try:
        identity_bytes = _canonical_json(_gap_batch_provenance_identity_payload(value))
        spec_bytes = _canonical_json(_gap_inclusion_batch_payload(value.spec))
        changed = (
            identity_bytes != snapshot[1]
            or value.raw_output != snapshot[2]
            or spec_bytes != snapshot[3]
            or value.batch_input != snapshot[4]
            or value.request_input != snapshot[5]
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("GAP batch provenance changed from its issued snapshot") from error
    if changed:
        raise ValueError("GAP batch provenance changed from its issued snapshot")
    verify_gap_inclusion_batch_spec(value.spec)
    envelope = _validate_gap_batch_output_envelope(value.raw_output, value.spec)
    if (
        value.batch_input_digest != value.spec.input_digest
        or value.batch_raw_output_digest != _bytes_digest(value.raw_output)
        or type(value.batch_input) is not bytes
        or value.batch_input != _canonical_json(_gap_inclusion_batch_payload(value.spec))
        or type(value.request_input) is not bytes
        or value.request_input != _gap_batch_request_bytes(value.spec)
        or value.request_input_digest != _bytes_digest(value.request_input)
        or len(value.members) != len(value.spec.members)
    ):
        raise ValueError("GAP batch provenance differs from its exact parent request")
    for index, (record, member_row, member) in enumerate(
        zip(value.members, envelope["members"], value.spec.members, strict=True)
    ):
        if type(record) is not GapBatchMemberProvenance or not isinstance(
            member_row, Mapping
        ):
            raise ValueError("GAP batch provenance has a malformed member universe")
        raw_mapping = member_row["raw_output"]
        assert isinstance(raw_mapping, Mapping)
        raw_bytes = _canonical_json(raw_mapping)
        if (
            record.member_index != index
            or record.inclusion_id != member.inclusion.inclusion_id
            or record.member_input_digest != member.input_digest
            or record.raw_output_digest != _bytes_digest(raw_bytes)
            or record.projection_digest
            != gap_inclusion_projection_digest(raw_mapping)
            or _DIGEST_RE.fullmatch(record.attestation_id) is None
        ):
            raise ValueError("GAP batch provenance member universe differs from its raw parent")
    expected_id = _task5_domain_digest(
        "task5-gap-batch-execution-provenance-v1",
        _gap_batch_provenance_identity_payload(value),
    )
    if value.batch_id != expected_id:
        raise ValueError("GAP batch provenance ID differs from its canonical universe")
    return value


def _verify_gap_launcher_execution(
    value: GapLauncherExecution,
    *,
    require_release: bool,
    replay_live_parent: bool,
) -> GapLauncherExecution:
    if type(value) is not GapLauncherExecution:
        raise TypeError("expected an exact GapLauncherExecution")
    snapshot = _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(id(value))
    if snapshot is None or snapshot[0]() is not value:
        raise ValueError("GAP launcher execution was not issued by its registry")
    try:
        attestation_bytes = _canonical_json(
            launcher_execution_attestation_mapping(value.attestation)
        )
        changed = (
            type(value.raw_output) is not bytes
            or value.raw_output != snapshot[1]
            or value.attestation is not snapshot[2]
            or attestation_bytes != snapshot[3]
            or value.batch_input_digest != snapshot[4]
            or value.batch_raw_output_digest != snapshot[5]
            or value.batch_member_input_digest != snapshot[6]
            or value.batch_member_index != snapshot[7]
            or _bytes_digest(value.raw_output)
            != value.attestation.raw_output_digest
            or launcher_execution_attestation_digest(value.attestation)
            != value.attestation.attestation_id
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "GAP launcher execution changed from its issued snapshot"
        ) from error
    if changed:
        raise ValueError(
            "GAP launcher execution identity or immutable snapshot changed"
        )
    provenance_reference = snapshot[9]
    provenance = (
        provenance_reference() if provenance_reference is not None else None
    )
    if value.batch_input_digest is None:
        if value.batch_provenance is not None or provenance is not None:
            raise ValueError("non-batch GAP execution acquired batch provenance")
    else:
        if (
            provenance is None
            or value.batch_provenance is not provenance
            or id(provenance) != snapshot[10]
        ):
            raise ValueError("GAP batch child has stale issued provenance")
        verify_gap_batch_execution_provenance(provenance)
        index = value.batch_member_index
        if type(index) is not int or index < 0 or index >= len(provenance.members):
            raise ValueError("GAP batch child member index is outside its issued universe")
        member = provenance.members[index]
        if (
            value.batch_input_digest != provenance.batch_input_digest
            or value.batch_raw_output_digest != provenance.batch_raw_output_digest
            or value.batch_member_input_digest != member.member_input_digest
            or _bytes_digest(value.raw_output) != member.raw_output_digest
            or value.attestation.attestation_id != member.attestation_id
            or value.attestation.gap_inclusion_projection_digest
            != member.projection_digest
        ):
            raise ValueError("GAP batch child differs from its issued member identity")
        parent_reference = snapshot[8]
        parent = parent_reference() if parent_reference is not None else None
        if replay_live_parent and parent is not None:
            _verify_gap_batch_launcher_execution(
                parent,
                require_release=require_release,
                verify_children=False,
            )
            if (
                index >= len(parent.member_executions)
                or parent.member_executions[index] is not value
            ):
                raise ValueError("GAP batch child is not at its issued parent index")
    if require_release and (
        not value.attestation.release_certified
        or value.attestation.runtime_manifest_digest is None
        or value.attestation.exit_status != 0
    ):
        raise ValueError(
            "diagnostic-only GAP launcher execution is not release-certified"
        )
    return value


def verify_gap_launcher_execution(
    value: GapLauncherExecution,
    *,
    require_release: bool = False,
) -> GapLauncherExecution:
    """Verify exact launcher issuance, attestation identity, and immutable bytes."""

    if type(require_release) is not bool:
        raise TypeError("require_release must be an exact boolean")
    return _verify_gap_launcher_execution(
        value, require_release=require_release, replay_live_parent=True
    )


def verify_gap_batch_member_execution(
    value: GapLauncherExecution,
    *,
    action: CertifiedSpaceGroupAction,
    inclusion: LiteralStabilizerInclusion,
    element_labels: Sequence[str],
    time_reversal: bool,
    finite_group_id: str | None = None,
    require_release: bool = False,
) -> GapLauncherExecution:
    """Verify that an issued child is the exact requested member of its batch."""

    verify_gap_launcher_execution(value, require_release=require_release)
    if value.batch_input_digest is None or value.batch_member_index is None:
        raise ValueError("GAP launcher execution was not issued as a batch member")
    provenance = value.batch_provenance
    if provenance is None:
        raise ValueError("GAP batch member lacks its issued parent provenance")
    verify_gap_batch_execution_provenance(provenance)
    spec = provenance.spec
    if spec.action != action or spec.time_reversal is not time_reversal:
        raise ValueError("GAP batch child action or grading differs from its request")
    payload = _validated_gap_inclusion_export_input(
        action,
        inclusion,
        element_labels,
        time_reversal=time_reversal,
        finite_group_id=finite_group_id,
    )
    try:
        member = spec.members[value.batch_member_index]
    except IndexError as error:
        raise ValueError("GAP batch child member index is outside its request") from error
    if (
        member.inclusion != inclusion
        or member.element_labels != tuple(element_labels)
        or member.finite_group_id != payload["finite_group_id"]
        or member.input_digest != payload["input_digest"]
        or value.batch_member_input_digest != member.input_digest
        or value.batch_input_digest != spec.input_digest
    ):
        raise ValueError("GAP batch child does not bind the exact requested member")
    return value


def _verify_gap_batch_launcher_execution(
    value: GapBatchLauncherExecution,
    *,
    require_release: bool,
    verify_children: bool,
) -> GapBatchLauncherExecution:
    if type(value) is not GapBatchLauncherExecution:
        raise TypeError("expected an exact GapBatchLauncherExecution")
    snapshot = _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(id(value))
    if snapshot is None or snapshot[0]() is not value:
        raise ValueError("GAP batch launcher execution was not issued by its registry")
    if (
        value.raw_output != snapshot[1]
        or _canonical_json(_gap_inclusion_batch_payload(value.spec)) != snapshot[2]
        or value.request_input_digest != snapshot[3]
        or len(value.member_executions) != len(snapshot[4])
        or any(
            expected() is not actual
            for actual, expected in zip(
                value.member_executions, snapshot[4], strict=True
            )
        )
        or _bytes_digest(value.raw_output) != snapshot[5]
        or snapshot[6]() is not value.provenance
        or id(value.provenance) != snapshot[7]
    ):
        raise ValueError("GAP batch launcher execution changed from its issued snapshot")
    provenance = value.provenance
    assert provenance is not None
    verify_gap_batch_execution_provenance(provenance)
    if (
        value.raw_output != provenance.raw_output
        or value.spec != provenance.spec
        or value.request_input_digest != provenance.request_input_digest
    ):
        raise ValueError("GAP batch parent differs from its issued provenance")
    for index, child in enumerate(value.member_executions):
        if (
            child.batch_provenance is not provenance
            or child.batch_member_index != index
        ):
            raise ValueError("GAP batch child tuple differs from its issued indices")
        if verify_children:
            _verify_gap_launcher_execution(
                child,
                require_release=require_release,
                replay_live_parent=False,
            )
    return value


def verify_gap_batch_launcher_execution(
    value: GapBatchLauncherExecution,
    *,
    require_release: bool = False,
) -> GapBatchLauncherExecution:
    """Replay the exact batch envelope and every issued child projection."""

    if type(require_release) is not bool:
        raise TypeError("require_release must be an exact boolean")
    return _verify_gap_batch_launcher_execution(
        value, require_release=require_release, verify_children=True
    )


def _resolved_launcher_digest(command: Sequence[str], executable: Path) -> str:
    return _task5_domain_digest(
        "task5-resolved-gap-launcher-v1",
        {
            "logical_argv": [executable.name, *command[1:]],
            "protocol": "task5-stdin-fixed-relative-output-v1",
            "executable_sha256": _bytes_digest(executable.read_bytes()),
        },
    )


def _runtime_file_entry(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    common: dict[str, object] = {
        "gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "path": os.fspath(path),
        "uid": metadata.st_uid,
    }
    if stat.S_ISREG(metadata.st_mode):
        return {
            **common,
            "kind": "file",
            "sha256": _bytes_digest(path.read_bytes()),
            "size": metadata.st_size,
        }
    if stat.S_ISDIR(metadata.st_mode):
        return {**common, "kind": "directory"}
    if stat.S_ISLNK(metadata.st_mode):
        return {**common, "kind": "symlink", "target": os.readlink(path)}
    raise ValueError("runtime manifest contains an unsupported file type")


def _runtime_tree_entries(
    root: Path,
    exclusions: frozenset[Path],
) -> tuple[dict[str, object], ...]:
    result = [_runtime_file_entry(root)]
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        directory_names.sort()
        file_names.sort()
        retained_directories: list[str] = []
        for name in directory_names:
            child = base / name
            if child in exclusions:
                continue
            result.append(_runtime_file_entry(child))
            if not child.is_symlink():
                retained_directories.append(name)
        directory_names[:] = retained_directories
        result.extend(_runtime_file_entry(base / name) for name in file_names)
    return tuple(result)


def _verify_runtime_file_manifest(
    value: object,
    *,
    allowed_roots: Sequence[Path] = (
        Path("/opt/mathpsg"),
        Path("/opt/gap"),
        Path("/workspace"),
    ),
    allowed_exclusions: Sequence[Path] = (),
) -> bool:
    """Replay the exact file-set Merkle seal without following symlink leaves."""

    if not isinstance(value, Mapping) or set(value) != {
        "entries",
        "exclusions",
        "merkle_digest",
        "roots",
        "schema_version",
    } or value.get("schema_version") != 1:
        return False
    roots = value.get("roots")
    entries = value.get("entries")
    exclusions = value.get("exclusions")
    if (
        not isinstance(roots, list)
        or not roots
        or not isinstance(entries, list)
        or not isinstance(exclusions, list)
    ):
        return False
    normalized_allowed = tuple(Path(item) for item in allowed_roots)
    if not normalized_allowed or any(not item.is_absolute() for item in normalized_allowed):
        return False
    normalized_exclusions = tuple(Path(item) for item in exclusions)
    expected_allowed_exclusions = tuple(
        sorted((Path(item) for item in allowed_exclusions), key=os.fspath)
    )
    if (
        tuple(os.fspath(path) for path in normalized_exclusions) != tuple(exclusions)
        or normalized_exclusions
        != tuple(sorted(normalized_exclusions, key=os.fspath))
        or len(set(normalized_exclusions)) != len(normalized_exclusions)
        or normalized_exclusions != expected_allowed_exclusions
        or any(
            not path.is_absolute()
            or not any(
                path == allowed or path.is_relative_to(allowed)
                for allowed in normalized_allowed
            )
            for path in normalized_exclusions
        )
    ):
        return False
    expected_entries: list[dict[str, object]] = []
    previous_root: tuple[str, str] | None = None
    root_paths: set[str] = set()
    tree_paths: list[Path] = []
    try:
        for row in roots:
            if not isinstance(row, Mapping) or set(row) != {"kind", "path"}:
                return False
            kind = row.get("kind")
            encoded = row.get("path")
            if kind not in ("file", "tree") or type(encoded) is not str:
                return False
            path = Path(encoded)
            if (
                not path.is_absolute()
                or os.fspath(path) != encoded
                or encoded in root_paths
                or not any(
                    path == allowed or path.is_relative_to(allowed)
                    for allowed in normalized_allowed
                )
            ):
                return False
            current_root = (encoded, kind)
            if previous_root is not None and current_root <= previous_root:
                return False
            previous_root = current_root
            root_paths.add(encoded)
            metadata = path.lstat()
            if kind == "file":
                if not stat.S_ISREG(metadata.st_mode):
                    return False
                expected_entries.append(_runtime_file_entry(path))
            else:
                if not stat.S_ISDIR(metadata.st_mode):
                    return False
                tree_paths.append(path)
                expected_entries.extend(_runtime_tree_entries(
                    path, frozenset(normalized_exclusions)
                ))
        for index, exclusion in enumerate(normalized_exclusions):
            metadata = exclusion.lstat()
            containing = tuple(
                tree for tree in tree_paths if exclusion.is_relative_to(tree)
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or len(containing) != 1
                or exclusion == containing[0]
                or any(
                    exclusion.is_relative_to(other)
                    or other.is_relative_to(exclusion)
                    for other in normalized_exclusions[:index]
                )
            ):
                return False
    except (OSError, ValueError):
        return False
    expected_entries.sort(key=lambda item: str(item["path"]))
    if len({str(item["path"]) for item in expected_entries}) != len(expected_entries):
        return False
    core = {
        "entries": expected_entries,
        "exclusions": list(exclusions),
        "roots": [dict(item) for item in roots],
        "schema_version": 1,
    }
    digest = "sha256:" + hashlib.sha256(
        b"mathpsg-classifier-runtime-files-v1|" + _canonical_json(core)
    ).hexdigest()
    return entries == expected_entries and value.get("merkle_digest") == digest


_LOCKED_RUNTIME_STATIC_ROOTS = frozenset(
    {
        ("file", "/opt/mathpsg/bin/locked-gap"),
        ("file", "/opt/mathpsg/classifier-gap.lock.json"),
        ("file", "/opt/mathpsg/seal-runtime"),
        ("file", "/workspace/environments/classifier-gap.lock.json"),
        (
            "file",
            "/workspace/psgmath/_assets/environments/classifier-gap.lock.json",
        ),
        *(
            ("file", f"/workspace/psgmath/{name}")
            for name in (
                "__init__.py",
                "_resources.py",
                "affine.py",
                "antiunitary.py",
                "bar_evaluator.py",
                "classification_schema.py",
                "cochains.py",
                "gap_classifier.py",
                "gf2.py",
                "integer_linalg.py",
                "lattice.py",
                "su2.py",
                "torus.py",
            )
        ),
        ("tree", "/opt/mathpsg/classifier-gap/pkg"),
        ("tree", "/opt/mathpsg/python"),
        ("tree", "/opt/mathpsg/python-standalone-licenses"),
        ("tree", "/workspace/gap/classifier"),
        ("tree", "/workspace/psgmath/_assets/gap/classifier"),
    }
)


def _runtime_seal_roots_are_exact(
    roots: object,
    gap_executable: str,
) -> bool:
    """Require exactly the reviewed producer roots and no consumer bytes."""

    if type(gap_executable) is not str or not gap_executable.startswith("/"):
        return False
    return roots == _LOCKED_RUNTIME_STATIC_ROOTS | {
        ("file", gap_executable)
    }


def _locked_runtime_manifest_digest(command: tuple[str, ...], executable: Path) -> str | None:
    if command[0] != "/opt/mathpsg/bin/locked-gap":
        return None
    runtime_root = Path("/opt/mathpsg/classifier-gap")
    manifest_path = runtime_root / "runtime-provenance.json"
    lock_path = Path("/opt/mathpsg/classifier-gap.lock.json")
    if not executable.is_relative_to(Path("/opt/mathpsg")):
        return None
    try:
        manifest_bytes = manifest_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
    except OSError:
        return None
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, Mapping)
        or manifest_bytes != _canonical_json(manifest)
        or manifest.get("schema_version") != 1
    ):
        return None
    supplied_digest = manifest.get("runtime_provenance_digest")
    if type(supplied_digest) is not str:
        return None
    core = {key: manifest[key] for key in sorted(set(manifest) - {"runtime_provenance_digest"})}
    expected_digest = "sha256:" + hashlib.sha256(
        b"mathpsg-classifier-runtime-provenance-v1|" + _canonical_json(core)
    ).hexdigest()
    try:
        locked = _locked_environment_core()
    except ValueError:
        return None
    base_image = manifest.get("base_image")
    runtime_files = manifest.get("runtime_files")
    if (
        supplied_digest != expected_digest
        or _bytes_digest(lock_bytes) != locked["lock_digest"]
        or not isinstance(base_image, Mapping)
        or base_image.get("index_digest") != locked["oci_image_digest"]
        or manifest.get("lock_digest") != locked["lock_digest"]
        or not _verify_runtime_file_manifest(runtime_files)
    ):
        return None
    assert isinstance(runtime_files, Mapping)
    roots = runtime_files.get("roots")
    entries = runtime_files.get("entries")
    if not isinstance(roots, list) or not isinstance(entries, list):
        return None
    root_set = {
        (row.get("kind"), row.get("path"))
        for row in roots
        if isinstance(row, Mapping)
    }
    try:
        wrapper = executable.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = re.fullmatch(r'#!/bin/sh\nexec "([^"]+)" "\$@"\n', wrapper)
    if match is None:
        return None
    gap_executable = match.group(1)
    if not _runtime_seal_roots_are_exact(root_set, gap_executable):
        return None
    for row in entries:
        if not isinstance(row, Mapping) or row.get("uid") != 0:
            return None
        if row.get("kind") != "symlink" and int(row.get("mode", 0)) & 0o022:
            return None
    return _bytes_digest(manifest_bytes)


def run_gap_inclusion_export(
    command: Sequence[str],
    program_template: str,
    *,
    cwd: str | os.PathLike[str],
    timeout_seconds: int = 180,
) -> GapLauncherExecution:
    """Run GAP once and attest its exact input, output, executable, and runtime.

    ``program_template`` must contain exactly one ``{output_path}`` marker.  A
    diagnostic local process can produce an attestation but never release
    authority.  Release status is derived only from the fixed locked-launcher
    path plus the independently verified runtime manifest inside the image.
    """

    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("GAP launcher command must be an argument sequence")
    argv = tuple(command)
    if not argv or any(type(item) is not str or not item for item in argv):
        raise ValueError("GAP launcher command must contain nonempty strings")
    if any(
        item in (".", "..") or "/" in item or "\\" in item
        for item in argv[1:]
    ):
        raise ValueError(
            "GAP launcher arguments must use the canonical path-free stdin protocol"
        )
    if type(program_template) is not str or program_template.count("{output_path}") != 1:
        raise ValueError("GAP program must contain exactly one {output_path} marker")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("GAP launcher timeout must be a positive integer")
    executable = Path(argv[0]).resolve(strict=True)
    if not executable.is_file():
        raise ValueError("GAP launcher executable is not a regular file")
    logical_output_name = "mathpsg-task5-gap-output.json"
    logical_cwd = Path(cwd).resolve(strict=True)
    if not logical_cwd.is_dir():
        raise ValueError("GAP launcher cwd must be a directory")
    if (logical_cwd / logical_output_name).exists():
        raise ValueError("GAP launcher logical output name collides with the workspace")
    with tempfile.TemporaryDirectory(prefix="mathpsg-task5-launch-") as directory:
        execution_root = Path(directory)
        for item in logical_cwd.iterdir():
            (execution_root / item.name).symlink_to(item, target_is_directory=item.is_dir())
        output_path = execution_root / logical_output_name
        quoted_output = json.dumps(logical_output_name, ensure_ascii=True)
        request_bytes = program_template.replace("{output_path}", quoted_output).encode("utf-8")
        process = subprocess.run(
            argv,
            cwd=os.fspath(execution_root),
            input=request_bytes,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if output_path.exists():
            raw_output = output_path.read_bytes()
        else:
            raw_output = b""
    if process.returncode != 0:
        raise RuntimeError(
            f"GAP launcher exited with status {process.returncode}; "
            "no certificate was assembled"
        )
    raw_mapping = _strict_json(raw_output)
    backend = raw_mapping.get("backend_environment")
    if not isinstance(backend, Mapping):
        raise ValueError("GAP launcher output lacks an observed backend environment")
    runtime_manifest_digest = _locked_runtime_manifest_digest(argv, executable)
    release_certified = runtime_manifest_digest is not None
    attestation = _make_launcher_execution_attestation(
        request_input_digest=_bytes_digest(request_bytes),
        raw_output_digest=_bytes_digest(raw_output),
        gap_inclusion_projection_digest=gap_inclusion_projection_digest(
            raw_mapping
        ),
        process_stdout_digest=_bytes_digest(process.stdout),
        process_stderr_digest=_bytes_digest(process.stderr),
        resolved_launcher_digest=_resolved_launcher_digest(argv, executable),
        backend_observation_digest=_task5_domain_digest(
            "task5-launcher-backend-observation-v1", backend
        ),
        runtime_manifest_digest=runtime_manifest_digest,
        exit_status=process.returncode,
        release_certified=release_certified,
    )
    result = GapLauncherExecution(
        raw_output,
        attestation,
        _GAP_LAUNCHER_EXECUTION_FACTORY_SEAL,
    )
    key = id(result)

    def discard(reference: weakref.ReferenceType[GapLauncherExecution]) -> None:
        current = _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY.pop(key, None)

    identity = weakref.ref(result, discard)
    _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY[key] = (
        identity,
        result.raw_output,
        result.attestation,
        _canonical_json(launcher_execution_attestation_mapping(result.attestation)),
        result.batch_input_digest,
        result.batch_raw_output_digest,
        result.batch_member_input_digest,
        result.batch_member_index,
        None,
        None,
        None,
    )
    return result


def _validated_gap_inclusion_export_input(
    action: CertifiedSpaceGroupAction,
    inclusion: LiteralStabilizerInclusion,
    element_labels: Sequence[str],
    *,
    time_reversal: bool,
    finite_group_id: str | None,
) -> dict[str, Any]:
    if not isinstance(action, CertifiedSpaceGroupAction):
        raise TypeError("action must be a CertifiedSpaceGroupAction")
    expected_action = make_certified_space_group_action(
        action.affine_generators, action.translation_basis
    )
    if action != expected_action:
        raise ValueError("action digest does not bind the exact affine generators")
    if not isinstance(inclusion, LiteralStabilizerInclusion):
        raise TypeError("inclusion must be a LiteralStabilizerInclusion")
    if (
        _EXPORT_IDENTIFIER_RE.fullmatch(inclusion.inclusion_id) is None
        or _DIGEST_RE.fullmatch(inclusion.literal_stabilizer_digest) is None
        or inclusion.literal_element_digest
        != literal_element_authority_digest(inclusion.literal_elements)
    ):
        raise ValueError("literal element digest does not bind the exact inclusion")
    if type(time_reversal) is not bool:
        raise TypeError("time_reversal must be a boolean")
    if isinstance(element_labels, (str, bytes)) or not isinstance(
        element_labels, Sequence
    ):
        raise TypeError("element_labels must be a canonical sequence")
    labels = tuple(element_labels)
    if (
        len(labels) != len(inclusion.literal_elements)
        or not labels
        or labels[0] != "1"
        or len(set(labels)) != len(labels)
        or any(
            type(label) is not str
            or _EXPORT_IDENTIFIER_RE.fullmatch(label) is None
            or label == "T"
            for label in labels
        )
    ):
        raise ValueError(
            "element_labels must uniquely name the identity-first literal tuple"
        )
    if time_reversal:
        graded_labels = tuple(
            "T" if label == "1" else f"{label}+T" for label in labels
        )
        if len(set(labels + graded_labels)) != 2 * len(labels):
            raise ValueError("graded element labels collide with spatial labels")
    group_id = inclusion.inclusion_id if finite_group_id is None else finite_group_id
    if type(group_id) is not str or _EXPORT_IDENTIFIER_RE.fullmatch(group_id) is None:
        raise ValueError("finite_group_id must be a canonical identifier")

    elements = inclusion.literal_elements
    identity = AffineTransformation(
        (
            ("q(1,1)", "q(0,1)", "q(0,1)"),
            ("q(0,1)", "q(1,1)", "q(0,1)"),
            ("q(0,1)", "q(0,1)", "q(1,1)"),
        ),
        ("q(0,1)", "q(0,1)", "q(0,1)"),
    )
    if elements[0] != identity or len(set(elements)) != len(elements):
        raise ValueError("literal inclusion must be unique and identity-first")
    element_set = set(elements)
    for left in elements:
        if _inverse_affine(left) not in element_set:
            raise ValueError("literal inclusion lacks an exact inverse")
        for right in elements:
            # Cryst's right-action source convention is C(x*y)=C(y) o C(x).
            if _compose_affine(right, left) not in element_set:
                raise ValueError("literal inclusion is not exactly closed")

    core = {
        "action": _action_mapping(action),
        "element_labels": list(labels),
        "finite_group_id": group_id,
        "inclusion": _inclusion_mapping(inclusion),
        "time_reversal": time_reversal,
    }
    return {
        **core,
        "input_digest": _classifier_domain_digest(
            "task5-literal-inclusion-export-input-v1", core
        ),
        "record_type": "task5-literal-inclusion-export-input",
        "schema_version": 1,
    }


def make_gap_inclusion_batch_spec(
    action: CertifiedSpaceGroupAction,
    inclusions: Sequence[LiteralStabilizerInclusion],
    *,
    element_label_sequences: Sequence[Sequence[str]],
    time_reversal: bool,
    finite_group_ids: Sequence[str | None] | None = None,
) -> GapInclusionBatchSpec:
    """Validate and canonicalize a nonempty literal-inclusion export batch."""

    if isinstance(inclusions, (str, bytes)) or not isinstance(inclusions, Sequence):
        raise TypeError("batch inclusions must be a canonical sequence")
    if isinstance(element_label_sequences, (str, bytes)) or not isinstance(
        element_label_sequences, Sequence
    ):
        raise TypeError("batch element-label sequences must be canonical")
    copied_inclusions = tuple(inclusions)
    copied_labels = tuple(element_label_sequences)
    if not copied_inclusions:
        raise ValueError("batch inclusion members must be nonempty")
    if len(copied_labels) != len(copied_inclusions):
        raise ValueError("batch inclusions and element-label sequences differ")
    if finite_group_ids is None:
        copied_group_ids: tuple[str | None, ...] = (None,) * len(copied_inclusions)
    else:
        if isinstance(finite_group_ids, (str, bytes)) or not isinstance(
            finite_group_ids, Sequence
        ):
            raise TypeError("batch finite-group IDs must be a canonical sequence")
        copied_group_ids = tuple(finite_group_ids)
        if len(copied_group_ids) != len(copied_inclusions):
            raise ValueError("batch inclusions and finite-group IDs differ")

    payloads = tuple(
        _validated_gap_inclusion_export_input(
            action,
            inclusion,
            labels,
            time_reversal=time_reversal,
            finite_group_id=group_id,
        )
        for inclusion, labels, group_id in zip(
            copied_inclusions, copied_labels, copied_group_ids, strict=True
        )
    )
    ordered = tuple(
        sorted(payloads, key=lambda item: item["inclusion"]["inclusion_id"])
    )
    inclusion_ids = tuple(item["inclusion"]["inclusion_id"] for item in ordered)
    if len(set(inclusion_ids)) != len(inclusion_ids):
        raise ValueError("batch inclusion IDs must be unique")
    members = tuple(
        GapInclusionBatchMember(
            next(
                inclusion
                for inclusion in copied_inclusions
                if inclusion.inclusion_id == payload["inclusion"]["inclusion_id"]
            ),
            tuple(payload["element_labels"]),
            payload["finite_group_id"],
            payload["input_digest"],
        )
        for payload in ordered
    )
    core = {
        "action": _action_mapping(action),
        "members": [
            {
                "element_labels": list(member.element_labels),
                "finite_group_id": member.finite_group_id,
                "inclusion": _inclusion_mapping(member.inclusion),
                "input_digest": member.input_digest,
            }
            for member in members
        ],
        "time_reversal": time_reversal,
    }
    return GapInclusionBatchSpec(
        action,
        members,
        time_reversal,
        _classifier_domain_digest("task5-literal-inclusion-batch-input-v1", core),
    )


def verify_gap_inclusion_batch_spec(
    spec: GapInclusionBatchSpec,
) -> GapInclusionBatchSpec:
    """Reconstruct every field of a canonical batch specification."""

    if type(spec) is not GapInclusionBatchSpec:
        raise TypeError("expected an exact GapInclusionBatchSpec")
    expected = make_gap_inclusion_batch_spec(
        spec.action,
        tuple(member.inclusion for member in spec.members),
        element_label_sequences=tuple(
            member.element_labels for member in spec.members
        ),
        time_reversal=spec.time_reversal,
        finite_group_ids=tuple(member.finite_group_id for member in spec.members),
    )
    if spec != expected:
        raise ValueError("GAP inclusion batch specification is not canonical")
    return spec


def _gap_inclusion_batch_payload(spec: GapInclusionBatchSpec) -> dict[str, Any]:
    core = {
        "action": _action_mapping(spec.action),
        "members": [
            {
                "element_labels": list(member.element_labels),
                "finite_group_id": member.finite_group_id,
                "inclusion": _inclusion_mapping(member.inclusion),
                "input_digest": member.input_digest,
            }
            for member in spec.members
        ],
        "time_reversal": spec.time_reversal,
    }
    return {
        **core,
        "input_digest": spec.input_digest,
        "record_type": "task5-literal-inclusion-batch-input",
        "schema_version": 1,
    }


def build_gap_inclusion_batch_export_program(spec: GapInclusionBatchSpec) -> str:
    """Build one canonical, fully inlined stdin GAP program for a batch."""

    if type(spec) is not GapInclusionBatchSpec:
        raise TypeError("batch export requires an exact GapInclusionBatchSpec")
    verify_gap_inclusion_batch_spec(spec)
    _locked_environment_core()
    encoded = _canonical_json(_gap_inclusion_batch_payload(spec)).decode("ascii")
    gap_string = json.dumps(encoded, ensure_ascii=True)
    tracked_source = "\n".join(
        _classifier_source_bytes(name).decode("utf-8", errors="strict").rstrip("\n")
        for name in (
            "protocol.g",
            "affine_pcp.g",
            "resolutions.g",
            "restrictions.g",
            "bar_equivalence.g",
        )
    )
    return (
        'if LoadPackage("json", "=2.2.3", false : OnlyNeeded) <> true then '
        'Error("locked JSON package is unavailable"); fi;\n'
        'if LoadPackage("cryst", "=4.1.30", false : OnlyNeeded) <> true then '
        'Error("locked Cryst package is unavailable"); fi;\n'
        'if LoadPackage("hap", "=1.70", false : OnlyNeeded) <> true then '
        'Error("locked HAP package is unavailable"); fi;\n'
        'if LoadPackage("hapcryst", "=0.1.15", false : OnlyNeeded) <> true then '
        'Error("locked HAPcryst package is unavailable"); fi;\n'
        + tracked_source
        + "\n"
        f"MathPSGClassifierTask5BatchInputBytes := {gap_string};\n"
        "MathPSGClassifierTask5BatchInput := "
        "JsonStringToGap(MathPSGClassifierTask5BatchInputBytes);\n"
        "MathPSGClassifierTask5BatchResult := CALL_WITH_CATCH(\n"
        "    MathPSGClassifierTask5LiteralInclusionBatchRaw,\n"
        "    [MathPSGClassifierTask5BatchInput, "
        "MathPSGClassifierTask5BatchInputBytes]\n"
        ");\n"
        "if MathPSGClassifierTask5BatchResult[1] <> true then QUIT_GAP(2); fi;\n"
        "if FileString({output_path}, MathPSGClassifierJson(\n"
        "    MathPSGClassifierTask5BatchResult[2]\n"
        ")) = fail then QUIT_GAP(2); fi;\n"
        "QUIT_GAP(0);\n"
    )


def _register_gap_launcher_execution(
    value: GapLauncherExecution,
    parent: GapBatchLauncherExecution,
) -> None:
    key = id(value)

    def discard(reference: weakref.ReferenceType[GapLauncherExecution]) -> None:
        current = _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY.pop(key, None)

    identity = weakref.ref(value, discard)
    _GAP_LAUNCHER_EXECUTION_ISSUER_REGISTRY[key] = (
        identity,
        value.raw_output,
        value.attestation,
        _canonical_json(launcher_execution_attestation_mapping(value.attestation)),
        value.batch_input_digest,
        value.batch_raw_output_digest,
        value.batch_member_input_digest,
        value.batch_member_index,
        weakref.ref(parent),
        weakref.ref(parent.provenance),
        id(parent.provenance),
    )


def _register_gap_batch_execution_provenance(
    value: GapBatchExecutionProvenance,
) -> None:
    key = id(value)

    def discard(
        reference: weakref.ReferenceType[GapBatchExecutionProvenance],
    ) -> None:
        current = _GAP_BATCH_EXECUTION_PROVENANCE_ISSUER_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _GAP_BATCH_EXECUTION_PROVENANCE_ISSUER_REGISTRY.pop(key, None)

    identity = weakref.ref(value, discard)
    _GAP_BATCH_EXECUTION_PROVENANCE_ISSUER_REGISTRY[key] = (
        identity,
        _canonical_json(_gap_batch_provenance_identity_payload(value)),
        value.raw_output,
        _canonical_json(_gap_inclusion_batch_payload(value.spec)),
        value.batch_input,
        value.request_input,
    )


def _register_gap_batch_launcher_execution(
    value: GapBatchLauncherExecution,
) -> None:
    key = id(value)

    def discard(reference: weakref.ReferenceType[GapBatchLauncherExecution]) -> None:
        current = _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY.pop(key, None)

    provenance = value.provenance
    assert provenance is not None
    identity = weakref.ref(value, discard)
    _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY[key] = (
        identity,
        value.raw_output,
        _canonical_json(_gap_inclusion_batch_payload(value.spec)),
        value.request_input_digest,
        tuple(weakref.ref(child) for child in value.member_executions),
        _bytes_digest(value.raw_output),
        weakref.ref(provenance),
        id(provenance),
    )


def _register_signed_release_corpus_value(
    registry: dict[int, tuple[Any, ...]],
    value: object,
    corpus: object,
    *snapshot: object,
) -> None:
    key = id(value)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        current = registry.get(key)
        if current is not None and current[0] is reference:
            registry.pop(key, None)

    registry[key] = (
        weakref.ref(value, discard),
        weakref.ref(corpus),
        id(corpus),
        _SIGNED_RELEASE_CORPUS_DOMAIN,
        *snapshot,
    )


def _restore_authenticated_task5_release_batch(
    corpus: object,
    shard_index: int,
) -> GapBatchLauncherExecution:
    """Privately issue one batch solely from an exact signed corpus graph."""

    if type(shard_index) is not int or shard_index < 0:
        raise ValueError("signed release corpus shard index must be nonnegative")
    from .task5_release import _authenticated_task5_release_batch_material

    shard, replayed = _authenticated_task5_release_batch_material(
        corpus,
        shard_index,
    )
    spec = replayed.batch_spec
    replay = replayed.batch_replay
    if (
        type(spec) is not GapInclusionBatchSpec
        or type(replay) is not GapBatchArtifactReplay
        or replay.batch_input != _canonical_json(_gap_inclusion_batch_payload(spec))
        or replay.batch_input_digest != shard.batch.batch_input_digest
        or replay.batch_raw_output_digest != shard.batch.batch_raw_output_digest
        or replay.request_input_digest != shard.batch.request_input_digest
        or len(replay.members) != len(shard.batch.members)
    ):
        raise ValueError("signed release corpus batch material differs")

    attestations: list[LauncherExecutionAttestation] = []
    children: list[GapLauncherExecution] = []
    provenance_members: list[GapBatchMemberProvenance] = []
    for member_index, (signed_member, replay_member, spec_member) in enumerate(
        zip(shard.batch.members, replay.members, spec.members, strict=True)
    ):
        mapping = _strict_json(signed_member.attestation.mapping_bytes)
        if (
            _canonical_json(mapping) != signed_member.attestation.mapping_bytes
            or mapping.get("record_type")
            != "task5-launcher-execution-attestation"
            or type(mapping.get("schema_version")) is not int
            or mapping["schema_version"] != 1
            or type(mapping.get("release_certified")) is not bool
            or mapping["release_certified"] is not True
            or type(mapping.get("exit_status")) is not int
            or mapping["exit_status"] != 0
        ):
            raise ValueError("signed release corpus attestation schema differs")
        attestation = _make_launcher_execution_attestation(
            request_input_digest=mapping["request_input_digest"],
            raw_output_digest=mapping["raw_output_digest"],
            gap_inclusion_projection_digest=mapping[
                "gap_inclusion_projection_digest"
            ],
            process_stdout_digest=mapping["process_stdout_digest"],
            process_stderr_digest=mapping["process_stderr_digest"],
            resolved_launcher_digest=mapping["resolved_launcher_digest"],
            backend_observation_digest=mapping["backend_observation_digest"],
            runtime_manifest_digest=mapping["runtime_manifest_digest"],
            exit_status=mapping["exit_status"],
            release_certified=mapping["release_certified"],
        )
        if (
            launcher_execution_attestation_mapping(attestation) != mapping
            or attestation.attestation_id != signed_member.attestation.attestation_id
            or signed_member.member_index != member_index
            or replay_member.member_index != member_index
            or signed_member.inclusion_id != spec_member.inclusion.inclusion_id
            or replay_member.inclusion_id != signed_member.inclusion_id
            or signed_member.member_input_digest != spec_member.input_digest
            or replay_member.member_input_digest != signed_member.member_input_digest
            or replay_member.raw_output_digest != signed_member.raw_output_digest
            or replay_member.projection_digest != signed_member.projection_digest
            or attestation.raw_output_digest != replay_member.raw_output_digest
            or attestation.gap_inclusion_projection_digest
            != replay_member.projection_digest
            or attestation.request_input_digest != replay.request_input_digest
        ):
            raise ValueError("signed release corpus attestation/member differs")
        child = GapLauncherExecution(
            replay_member.raw_output,
            attestation,
            _GAP_LAUNCHER_EXECUTION_FACTORY_SEAL,
            batch_input_digest=replay.batch_input_digest,
            batch_raw_output_digest=replay.batch_raw_output_digest,
            batch_member_input_digest=replay_member.member_input_digest,
            batch_member_index=member_index,
        )
        provenance_member = GapBatchMemberProvenance(
            member_index=member_index,
            inclusion_id=replay_member.inclusion_id,
            member_input_digest=replay_member.member_input_digest,
            raw_output_digest=replay_member.raw_output_digest,
            projection_digest=replay_member.projection_digest,
            attestation_id=attestation.attestation_id,
        )
        attestations.append(attestation)
        children.append(child)
        provenance_members.append(provenance_member)

    provenance = GapBatchExecutionProvenance(
        batch_id=shard.batch.batch_id,
        batch_input_digest=replay.batch_input_digest,
        batch_raw_output_digest=replay.batch_raw_output_digest,
        request_input_digest=replay.request_input_digest,
        batch_input=replay.batch_input,
        request_input=replay.request_input,
        raw_output=replay.raw_output,
        spec=spec,
        members=tuple(provenance_members),
        _factory_seal=_GAP_BATCH_EXECUTION_PROVENANCE_FACTORY_SEAL,
    )
    for child in children:
        object.__setattr__(child, "batch_provenance", provenance)
    parent = GapBatchLauncherExecution(
        raw_output=replay.raw_output,
        spec=spec,
        request_input_digest=replay.request_input_digest,
        member_executions=tuple(children),
        provenance=provenance,
        _factory_seal=_GAP_BATCH_LAUNCHER_EXECUTION_FACTORY_SEAL,
    )
    _register_gap_batch_execution_provenance(provenance)
    for member_index, (attestation, child) in enumerate(
        zip(attestations, children, strict=True)
    ):
        _register_gap_launcher_execution(child, parent)
        _register_signed_release_corpus_value(
            _SIGNED_RELEASE_CORPUS_ATTESTATION_ISSUER_REGISTRY,
            attestation,
            corpus,
            shard_index,
            member_index,
            _canonical_json(launcher_execution_attestation_mapping(attestation)),
        )
    _register_gap_batch_launcher_execution(parent)
    _register_signed_release_corpus_value(
        _SIGNED_RELEASE_CORPUS_PROVENANCE_ISSUER_REGISTRY,
        provenance,
        corpus,
        shard_index,
        _canonical_json(_gap_batch_provenance_identity_payload(provenance)),
    )
    _register_signed_release_corpus_value(
        _SIGNED_RELEASE_CORPUS_BATCH_ISSUER_REGISTRY,
        parent,
        corpus,
        shard_index,
        weakref.ref(provenance),
        tuple(weakref.ref(child) for child in children),
    )
    return _verify_signed_release_corpus_batch(parent, corpus, shard_index)


def _verify_signed_release_corpus_attestation(
    value: LauncherExecutionAttestation,
    corpus: object,
    shard_index: int,
    member_index: int,
) -> LauncherExecutionAttestation:
    if type(value) is not LauncherExecutionAttestation:
        raise TypeError("expected a signed release corpus attestation")
    snapshot = _SIGNED_RELEASE_CORPUS_ATTESTATION_ISSUER_REGISTRY.get(id(value))
    if (
        snapshot is None
        or snapshot[0]() is not value
        or snapshot[1]() is not corpus
        or snapshot[2] != id(corpus)
        or snapshot[3] != _SIGNED_RELEASE_CORPUS_DOMAIN
        or snapshot[4] != shard_index
        or snapshot[5] != member_index
        or snapshot[6]
        != _canonical_json(launcher_execution_attestation_mapping(value))
    ):
        raise ValueError("signed release corpus attestation was not exactly issued")
    return value


def _verify_signed_release_corpus_batch(
    value: GapBatchLauncherExecution,
    corpus: object,
    shard_index: int,
) -> GapBatchLauncherExecution:
    if type(value) is not GapBatchLauncherExecution:
        raise TypeError("expected a signed release corpus batch")
    snapshot = _SIGNED_RELEASE_CORPUS_BATCH_ISSUER_REGISTRY.get(id(value))
    if (
        snapshot is None
        or snapshot[0]() is not value
        or snapshot[1]() is not corpus
        or snapshot[2] != id(corpus)
        or snapshot[3] != _SIGNED_RELEASE_CORPUS_DOMAIN
        or snapshot[4] != shard_index
        or snapshot[5]() is not value.provenance
        or len(snapshot[6]) != len(value.member_executions)
        or any(
            reference() is not child
            for reference, child in zip(
                snapshot[6], value.member_executions, strict=True
            )
        )
    ):
        raise ValueError("signed release corpus batch was not exactly issued")
    provenance = value.provenance
    assert provenance is not None
    provenance_snapshot = _SIGNED_RELEASE_CORPUS_PROVENANCE_ISSUER_REGISTRY.get(
        id(provenance)
    )
    if (
        provenance_snapshot is None
        or provenance_snapshot[0]() is not provenance
        or provenance_snapshot[1]() is not corpus
        or provenance_snapshot[2] != id(corpus)
        or provenance_snapshot[3] != _SIGNED_RELEASE_CORPUS_DOMAIN
        or provenance_snapshot[4] != shard_index
        or provenance_snapshot[5]
        != _canonical_json(_gap_batch_provenance_identity_payload(provenance))
    ):
        raise ValueError("signed release corpus provenance was not exactly issued")
    verify_gap_batch_launcher_execution(value, require_release=True)
    for member_index, child in enumerate(value.member_executions):
        _verify_signed_release_corpus_attestation(
            child.attestation,
            corpus,
            shard_index,
            member_index,
        )
    return value


def _strict_batch_json(data: bytes) -> Mapping[str, Any]:
    if type(data) is not bytes:
        raise TypeError("GAP batch output requires exact bytes")
    if len(data) > 512 * 1024 * 1024:
        raise ValueError("GAP batch output exceeds the canonical size bound")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"GAP batch output has duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_float=lambda token: (_ for _ in ()).throw(
                ValueError("GAP batch output forbids floating-point JSON")
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("GAP batch output forbids non-finite JSON")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GAP batch output is invalid strict JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("GAP batch output must be an object")
    if _canonical_json(value) != data:
        raise ValueError("GAP batch output bytes are not canonical JSON")
    return value


def _validate_gap_batch_output_envelope(
    raw_output: bytes,
    spec: GapInclusionBatchSpec,
) -> Mapping[str, Any]:
    """Replay the strict shared and ordered-member batch response contract."""

    if type(spec) is not GapInclusionBatchSpec:
        raise TypeError("GAP batch output requires a canonical batch specification")
    verify_gap_inclusion_batch_spec(spec)
    envelope = _strict_batch_json(raw_output)
    expected_fields = {
        "action_digest",
        "ambient_construction_count",
        "backend_environment",
        "batch_input_digest",
        "members",
        "record_type",
        "schema_version",
        "time_reversal",
    }
    if set(envelope) != expected_fields:
        raise ValueError("GAP batch output has a noncanonical field universe")
    expected_child_fields = {
        "backend_environment",
        "bar_comparison_traces",
        "bar_equivalence",
        "chain_map_algorithm",
        "diagnostic_backend",
        "diagnostic_maps",
        "finite_group",
        "lookahead_boundary",
        "source",
        "source_element_images",
        "target",
        "target_bar_equivalence",
    }
    members = envelope["members"]
    if (
        type(envelope["record_type"]) is not str
        or envelope["record_type"] != "task5-literal-inclusion-batch-output"
        or type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or type(envelope["ambient_construction_count"]) is not int
        or envelope["ambient_construction_count"] != 1
        or envelope["action_digest"] != spec.action.action_digest
        or envelope["batch_input_digest"] != spec.input_digest
        or envelope["time_reversal"] is not spec.time_reversal
        or not isinstance(envelope["backend_environment"], Mapping)
        or not isinstance(members, list)
        or len(members) != len(spec.members)
    ):
        raise ValueError("GAP batch output differs from its exact request identity")
    for member_row, member in zip(members, spec.members, strict=True):
        if not isinstance(member_row, Mapping) or set(member_row) != {
            "inclusion_id",
            "member_input_digest",
            "raw_output",
        }:
            raise ValueError("GAP batch output contains a malformed member record")
        raw_member = member_row.get("raw_output")
        if (
            member_row["inclusion_id"] != member.inclusion.inclusion_id
            or member_row["member_input_digest"] != member.input_digest
            or not isinstance(raw_member, Mapping)
            or set(raw_member) != expected_child_fields
            or raw_member.get("backend_environment")
            != envelope["backend_environment"]
        ):
            raise ValueError("GAP batch output member order or identity differs")
        assert isinstance(raw_member, Mapping)
        finite = raw_member.get("finite_group")
        graded_labels = tuple(
            "T" if label == "1" else f"{label}+T"
            for label in member.element_labels
        )
        expected_labels = member.element_labels + (
            graded_labels if spec.time_reversal else ()
        )
        expected_group_id = member.finite_group_id + (
            "+onsite-T" if spec.time_reversal else ""
        )
        if (
            not isinstance(finite, Mapping)
            or finite.get("group_id") != expected_group_id
            or finite.get("element_order") != list(expected_labels)
        ):
            raise ValueError("GAP batch member raw identity or labels differ")
    return envelope


def replay_gap_inclusion_batch_artifact(
    spec: GapInclusionBatchSpec,
    raw_output: bytes,
) -> GapBatchArtifactReplay:
    """Replay canonical batch bytes without issuing launcher capabilities."""

    envelope = _validate_gap_batch_output_envelope(raw_output, spec)
    raw_members = envelope["members"]
    assert isinstance(raw_members, list)
    members: list[GapBatchArtifactMemberReplay] = []
    for member_index, (member_row, member) in enumerate(
        zip(raw_members, spec.members, strict=True)
    ):
        assert isinstance(member_row, Mapping)
        child_mapping = member_row["raw_output"]
        assert isinstance(child_mapping, Mapping)
        child_raw = _canonical_json(child_mapping)
        members.append(
            GapBatchArtifactMemberReplay(
                member_index=member_index,
                inclusion_id=member.inclusion.inclusion_id,
                member_input_digest=member.input_digest,
                raw_output=child_raw,
                raw_output_digest=_bytes_digest(child_raw),
                projection_digest=gap_inclusion_projection_digest(
                    child_mapping
                ),
            )
        )
    batch_input = _canonical_json(_gap_inclusion_batch_payload(spec))
    request_input = _gap_batch_request_bytes(spec)
    return GapBatchArtifactReplay(
        batch_input=batch_input,
        batch_input_digest=spec.input_digest,
        request_input=request_input,
        request_input_digest=_bytes_digest(request_input),
        raw_output=raw_output,
        batch_raw_output_digest=_bytes_digest(raw_output),
        members=tuple(members),
    )


def restore_diagnostic_gap_batch_execution(
    spec: GapInclusionBatchSpec,
    raw_output: bytes,
    attestation_mappings: Sequence[Mapping[str, object]],
    *,
    expected_resolved_launcher_digest: str,
) -> GapBatchLauncherExecution:
    """Restore a diagnostic batch capability from canonical cached evidence.

    This performs the same pure batch/member replay as the live launcher but
    never executes GAP and can never restore release certification.
    """

    verify_gap_inclusion_batch_spec(spec)
    if (
        type(expected_resolved_launcher_digest) is not str
        or _DIGEST_RE.fullmatch(expected_resolved_launcher_digest) is None
    ):
        raise ValueError("expected restored launcher digest must be sha256")
    replay = replay_gap_inclusion_batch_artifact(spec, raw_output)
    mappings = tuple(attestation_mappings)
    if len(mappings) != len(replay.members):
        raise ValueError("cached diagnostic attestations do not cover the batch")
    attestations: list[LauncherExecutionAttestation] = []
    for index, (mapping, member) in enumerate(
        zip(mappings, replay.members, strict=True)
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError("cached diagnostic attestation must be an object")
        if (
            mapping.get("record_type") != "task5-launcher-execution-attestation"
            or mapping.get("schema_version") != 1
            or mapping.get("release_certified") is not False
            or mapping.get("runtime_manifest_digest") is not None
            or mapping.get("exit_status") != 0
            or mapping.get("resolved_launcher_digest")
            != expected_resolved_launcher_digest
        ):
            raise ValueError("cached batch cannot restore release or failed authority")
        attestation = _make_launcher_execution_attestation(
            request_input_digest=mapping["request_input_digest"],
            raw_output_digest=mapping["raw_output_digest"],
            gap_inclusion_projection_digest=mapping[
                "gap_inclusion_projection_digest"
            ],
            process_stdout_digest=mapping["process_stdout_digest"],
            process_stderr_digest=mapping["process_stderr_digest"],
            resolved_launcher_digest=mapping["resolved_launcher_digest"],
            backend_observation_digest=mapping["backend_observation_digest"],
            runtime_manifest_digest=None,
            exit_status=0,
            release_certified=False,
        )
        if (
            launcher_execution_attestation_mapping(attestation) != dict(mapping)
            or attestation.raw_output_digest != member.raw_output_digest
            or attestation.gap_inclusion_projection_digest
            != member.projection_digest
            or attestation.request_input_digest != replay.request_input_digest
        ):
            raise ValueError(
                f"cached diagnostic attestation {index} differs from pure replay"
            )
        attestations.append(attestation)

    provenance_members = tuple(
        GapBatchMemberProvenance(
            member_index=index,
            inclusion_id=member.inclusion_id,
            member_input_digest=member.member_input_digest,
            raw_output_digest=member.raw_output_digest,
            projection_digest=member.projection_digest,
            attestation_id=attestation.attestation_id,
        )
        for index, (member, attestation) in enumerate(
            zip(replay.members, attestations, strict=True)
        )
    )
    provenance_identity = {
        "batch_input_digest": replay.batch_input_digest,
        "batch_raw_output_digest": replay.batch_raw_output_digest,
        "members": [
            _batch_member_provenance_mapping(member)
            for member in provenance_members
        ],
        "request_input_digest": replay.request_input_digest,
    }
    provenance = GapBatchExecutionProvenance(
        batch_id=_task5_domain_digest(
            "task5-gap-batch-execution-provenance-v1", provenance_identity
        ),
        batch_input_digest=replay.batch_input_digest,
        batch_raw_output_digest=replay.batch_raw_output_digest,
        request_input_digest=replay.request_input_digest,
        batch_input=replay.batch_input,
        request_input=replay.request_input,
        raw_output=replay.raw_output,
        spec=spec,
        members=provenance_members,
        _factory_seal=_GAP_BATCH_EXECUTION_PROVENANCE_FACTORY_SEAL,
    )
    children = tuple(
        GapLauncherExecution(
            member.raw_output,
            attestation,
            _GAP_LAUNCHER_EXECUTION_FACTORY_SEAL,
            batch_input_digest=replay.batch_input_digest,
            batch_raw_output_digest=replay.batch_raw_output_digest,
            batch_member_input_digest=member.member_input_digest,
            batch_member_index=index,
            batch_provenance=provenance,
        )
        for index, (member, attestation) in enumerate(
            zip(replay.members, attestations, strict=True)
        )
    )
    result = GapBatchLauncherExecution(
        raw_output=replay.raw_output,
        spec=spec,
        request_input_digest=replay.request_input_digest,
        member_executions=children,
        provenance=provenance,
        _factory_seal=_GAP_BATCH_LAUNCHER_EXECUTION_FACTORY_SEAL,
    )
    _register_gap_batch_execution_provenance(provenance)
    for child in children:
        _register_gap_launcher_execution(child, result)
    _register_gap_batch_launcher_execution(result)
    return verify_gap_batch_launcher_execution(result, require_release=False)


def export_gap_inclusion_batch_raw(
    action: CertifiedSpaceGroupAction,
    inclusions: Sequence[LiteralStabilizerInclusion],
    *,
    element_label_sequences: Sequence[Sequence[str]],
    time_reversal: bool,
    cwd: str | os.PathLike[str],
    command: Sequence[str] | None = None,
    finite_group_ids: Sequence[str | None] | None = None,
    timeout_seconds: int = 180,
) -> GapBatchLauncherExecution:
    """Execute one validated GAP process and issue one child per batch member."""

    spec = make_gap_inclusion_batch_spec(
        action,
        inclusions,
        element_label_sequences=element_label_sequences,
        time_reversal=time_reversal,
        finite_group_ids=finite_group_ids,
    )
    program_template = build_gap_inclusion_batch_export_program(spec)
    command_value: Sequence[str] = (
        ("/opt/mathpsg/bin/locked-gap", "-q") if command is None else command
    )
    if isinstance(command_value, (str, bytes)) or not isinstance(
        command_value, Sequence
    ):
        raise TypeError("GAP launcher command must be an argument sequence")
    argv = tuple(command_value)
    if not argv or any(type(item) is not str or not item for item in argv):
        raise ValueError("GAP launcher command must contain nonempty strings")
    if any(item in (".", "..") or "/" in item or "\\" in item for item in argv[1:]):
        raise ValueError(
            "GAP launcher arguments must use the canonical path-free stdin protocol"
        )
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("GAP launcher timeout must be a positive integer")
    executable = Path(argv[0]).resolve(strict=True)
    if not executable.is_file():
        raise ValueError("GAP launcher executable is not a regular file")
    logical_output_name = "mathpsg-task5-gap-output.json"
    logical_cwd = Path(cwd).resolve(strict=True)
    if not logical_cwd.is_dir():
        raise ValueError("GAP launcher cwd must be a directory")
    if (logical_cwd / logical_output_name).exists():
        raise ValueError("GAP launcher logical output name collides with the workspace")
    with tempfile.TemporaryDirectory(prefix="mathpsg-task5-batch-launch-") as directory:
        execution_root = Path(directory)
        for item in logical_cwd.iterdir():
            (execution_root / item.name).symlink_to(
                item, target_is_directory=item.is_dir()
            )
        output_path = execution_root / logical_output_name
        request_bytes = program_template.replace(
            "{output_path}", json.dumps(logical_output_name, ensure_ascii=True)
        ).encode("utf-8")
        process = subprocess.run(
            argv,
            cwd=os.fspath(execution_root),
            input=request_bytes,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        raw_output = output_path.read_bytes() if output_path.exists() else b""
    if process.returncode != 0:
        raise RuntimeError(
            f"GAP batch launcher exited with status {process.returncode}; "
            "no member certificates were assembled"
        )
    envelope = _validate_gap_batch_output_envelope(raw_output, spec)
    batch_raw_digest = _bytes_digest(raw_output)
    request_input_digest = _bytes_digest(request_bytes)
    runtime_manifest_digest = _locked_runtime_manifest_digest(argv, executable)
    release_certified = runtime_manifest_digest is not None
    children: list[GapLauncherExecution] = []
    for index, (member_row, member) in enumerate(
        zip(envelope["members"], spec.members, strict=True)
    ):
        assert isinstance(member_row, Mapping)
        child_raw = _canonical_json(member_row["raw_output"])
        child_mapping = _strict_json(child_raw)
        attestation = _make_launcher_execution_attestation(
            request_input_digest=request_input_digest,
            raw_output_digest=_bytes_digest(child_raw),
            gap_inclusion_projection_digest=gap_inclusion_projection_digest(
                child_mapping
            ),
            process_stdout_digest=_bytes_digest(process.stdout),
            process_stderr_digest=_bytes_digest(process.stderr),
            resolved_launcher_digest=_resolved_launcher_digest(argv, executable),
            backend_observation_digest=_task5_domain_digest(
                "task5-launcher-backend-observation-v1",
                envelope["backend_environment"],
            ),
            runtime_manifest_digest=runtime_manifest_digest,
            exit_status=process.returncode,
            release_certified=release_certified,
        )
        child = GapLauncherExecution(
            child_raw,
            attestation,
            _GAP_LAUNCHER_EXECUTION_FACTORY_SEAL,
            batch_input_digest=spec.input_digest,
            batch_raw_output_digest=batch_raw_digest,
            batch_member_input_digest=member.input_digest,
            batch_member_index=index,
        )
        children.append(child)
    member_provenance = tuple(
        GapBatchMemberProvenance(
            member_index=index,
            inclusion_id=member.inclusion.inclusion_id,
            member_input_digest=member.input_digest,
            raw_output_digest=child.attestation.raw_output_digest,
            projection_digest=child.attestation.gap_inclusion_projection_digest,
            attestation_id=child.attestation.attestation_id,
        )
        for index, (member, child) in enumerate(
            zip(spec.members, children, strict=True)
        )
    )
    provenance_identity = {
        "batch_input_digest": spec.input_digest,
        "batch_raw_output_digest": batch_raw_digest,
        "members": [
            _batch_member_provenance_mapping(member)
            for member in member_provenance
        ],
        "request_input_digest": request_input_digest,
    }
    provenance = GapBatchExecutionProvenance(
        batch_id=_task5_domain_digest(
            "task5-gap-batch-execution-provenance-v1", provenance_identity
        ),
        batch_input_digest=spec.input_digest,
        batch_raw_output_digest=batch_raw_digest,
        request_input_digest=request_input_digest,
        batch_input=_canonical_json(_gap_inclusion_batch_payload(spec)),
        request_input=request_bytes,
        raw_output=raw_output,
        spec=spec,
        members=member_provenance,
        _factory_seal=_GAP_BATCH_EXECUTION_PROVENANCE_FACTORY_SEAL,
    )
    for child in children:
        object.__setattr__(child, "batch_provenance", provenance)
    result = GapBatchLauncherExecution(
        raw_output=raw_output,
        spec=spec,
        request_input_digest=request_input_digest,
        member_executions=tuple(children),
        provenance=provenance,
        _factory_seal=_GAP_BATCH_LAUNCHER_EXECUTION_FACTORY_SEAL,
    )
    _register_gap_batch_execution_provenance(provenance)
    for child in result.member_executions:
        _register_gap_launcher_execution(child, result)
    key = id(result)

    def discard(reference: weakref.ReferenceType[GapBatchLauncherExecution]) -> None:
        current = _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY.pop(key, None)

    identity = weakref.ref(result, discard)
    _GAP_BATCH_LAUNCHER_EXECUTION_ISSUER_REGISTRY[key] = (
        identity,
        result.raw_output,
        _canonical_json(_gap_inclusion_batch_payload(result.spec)),
        result.request_input_digest,
        tuple(weakref.ref(child) for child in result.member_executions),
        batch_raw_digest,
        weakref.ref(provenance),
        id(provenance),
    )
    return result


def build_gap_inclusion_export_program(
    action: CertifiedSpaceGroupAction,
    inclusion: LiteralStabilizerInclusion,
    *,
    element_labels: Sequence[str],
    time_reversal: bool,
    finite_group_id: str | None = None,
) -> str:
    """Build the canonical stdin-only GAP program for one literal inclusion."""

    # Fail before emitting executable text if the packaged program bytes do
    # not match the authenticated classifier source closure.
    _locked_environment_core()
    payload = _validated_gap_inclusion_export_input(
        action,
        inclusion,
        element_labels,
        time_reversal=time_reversal,
        finite_group_id=finite_group_id,
    )
    encoded = _canonical_json(payload).decode("ascii")
    gap_string = json.dumps(encoded, ensure_ascii=True)
    tracked_source = "\n".join(
        _classifier_source_bytes(name).decode("utf-8", errors="strict").rstrip("\n")
        for name in (
            "protocol.g",
            "affine_pcp.g",
            "resolutions.g",
            "restrictions.g",
            "bar_equivalence.g",
        )
    )
    return (
        'if LoadPackage("json", "=2.2.3", false : OnlyNeeded) <> true then '
        'Error("locked JSON package is unavailable"); fi;\n'
        'if LoadPackage("cryst", "=4.1.30", false : OnlyNeeded) <> true then '
        'Error("locked Cryst package is unavailable"); fi;\n'
        'if LoadPackage("hap", "=1.70", false : OnlyNeeded) <> true then '
        'Error("locked HAP package is unavailable"); fi;\n'
        'if LoadPackage("hapcryst", "=0.1.15", false : OnlyNeeded) <> true then '
        'Error("locked HAPcryst package is unavailable"); fi;\n'
        + tracked_source
        + "\n"
        f"MathPSGClassifierTask5ExportInputBytes := {gap_string};\n"
        'MathPSGClassifierTask5ExportInput := '
        'JsonStringToGap(MathPSGClassifierTask5ExportInputBytes);\n'
        'MathPSGClassifierTask5ExportResult := CALL_WITH_CATCH(\n'
        '    MathPSGClassifierTask5LiteralInclusionRaw,\n'
        '    [MathPSGClassifierTask5ExportInput, '
        'MathPSGClassifierTask5ExportInputBytes]\n'
        ');\n'
        'if MathPSGClassifierTask5ExportResult[1] <> true then QUIT_GAP(2); fi;\n'
        'if FileString({output_path}, MathPSGClassifierJson(\n'
        '    MathPSGClassifierTask5ExportResult[2]\n'
        ')) = fail then QUIT_GAP(2); fi;\n'
        'QUIT_GAP(0);\n'
    )


def export_gap_inclusion_raw(
    action: CertifiedSpaceGroupAction,
    inclusion: LiteralStabilizerInclusion,
    *,
    element_labels: Sequence[str],
    time_reversal: bool,
    cwd: str | os.PathLike[str],
    command: Sequence[str] | None = None,
    finite_group_id: str | None = None,
    timeout_seconds: int = 180,
) -> GapLauncherExecution:
    """Generate and execute one validated raw Task5 literal-inclusion export."""

    batch = export_gap_inclusion_batch_raw(
        action,
        (inclusion,),
        element_label_sequences=(tuple(element_labels),),
        time_reversal=time_reversal,
        cwd=cwd,
        command=command,
        finite_group_ids=(finite_group_id,),
        timeout_seconds=timeout_seconds,
    )
    return batch.member_executions[0]


def _parse_fraction(value: Any, path: str) -> Fraction:
    if type(value) is not str or _CANONICAL_FRACTION_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected a canonical rational")
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError(f"{path}: rational must use reduced canonical rational spelling")
    return parsed


@dataclass(frozen=True, slots=True, order=True)
class SparseResolutionChainTerm:
    basis_id: str
    element: str
    coefficient: int

    def __post_init__(self) -> None:
        if type(self.basis_id) is not str or type(self.element) is not str:
            raise TypeError("resolution-chain labels must be strings")
        if type(self.coefficient) is not int or self.coefficient == 0:
            raise ValueError("resolution-chain coefficient must be a nonzero integer")


@dataclass(frozen=True, slots=True)
class SparseResolutionChain:
    degree: int
    terms: tuple[SparseResolutionChainTerm, ...]

    def __post_init__(self) -> None:
        if type(self.degree) is not int or self.degree < 0:
            raise ValueError("resolution-chain degree must be nonnegative")
        terms = tuple(self.terms)
        if any(not isinstance(term, SparseResolutionChainTerm) for term in terms):
            raise TypeError("resolution chain requires sparse terms")
        keys = tuple((term.basis_id, term.element) for term in terms)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("resolution-chain terms must use canonical sparse order")
        object.__setattr__(self, "terms", terms)


@dataclass(frozen=True, slots=True, order=True)
class SparseBarTerm:
    left_element: str
    group_tuple: tuple[str, ...]
    coefficient: int

    def __post_init__(self) -> None:
        if type(self.left_element) is not str:
            raise TypeError("bar left element must be a string")
        group_tuple = tuple(self.group_tuple)
        if any(type(element) is not str for element in group_tuple):
            raise TypeError("bar tuple elements must be strings")
        if type(self.coefficient) is not int or self.coefficient == 0:
            raise ValueError("bar coefficient must be a nonzero integer")
        object.__setattr__(self, "group_tuple", group_tuple)


@dataclass(frozen=True, slots=True)
class SparseBarChain:
    degree: int
    terms: tuple[SparseBarTerm, ...]

    def __post_init__(self) -> None:
        if type(self.degree) is not int or self.degree < 0:
            raise ValueError("bar-chain degree must be nonnegative")
        terms = tuple(self.terms)
        if any(not isinstance(term, SparseBarTerm) for term in terms):
            raise TypeError("bar chain requires sparse bar terms")
        if any(len(term.group_tuple) != self.degree for term in terms):
            raise ValueError("bar term tuple length differs from chain degree")
        keys = tuple((term.left_element, term.group_tuple) for term in terms)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("bar-chain terms must use canonical sparse order")
        object.__setattr__(self, "terms", terms)


@dataclass(frozen=True, slots=True)
class ResolutionBasisImage:
    degree: int
    basis_id: str
    image: SparseBarChain | SparseResolutionChain


@dataclass(frozen=True, slots=True)
class BarPhiValue:
    group_tuple: tuple[str, ...]
    image: SparseResolutionChain
    bar_homotopy: SparseBarChain | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_tuple", tuple(self.group_tuple))


@dataclass(frozen=True, slots=True)
class BarResolutionEquivalence:
    resolution_id: str
    resolution: FreeResolutionCertificate
    finite_group: FiniteGroupTable
    phi_algorithm: str
    psi_on_basis: tuple[ResolutionBasisImage, ...]
    resolution_homotopy_on_basis: tuple[ResolutionBasisImage, ...]
    bar_homotopy_algorithm: str
    phi_on_queries: tuple[BarPhiValue, ...]
    queried_bar_tuples: tuple[tuple[str, ...], ...]
    lookahead_boundary: SparseGroupRingMatrix
    benchmark_coordinates: tuple[Fraction, ...]
    benchmark_tuple: tuple[str, ...]
    equivalence_id: str

    def __post_init__(self) -> None:
        if self.resolution_id != self.resolution.resolution_id:
            raise ValueError("bar equivalence resolution ID differs from embedded resolution")
        if self.resolution.finite_group != self.finite_group:
            raise ValueError("bar equivalence finite table differs from resolution authority")
        psi = tuple(self.psi_on_basis)
        homotopy = tuple(self.resolution_homotopy_on_basis)
        phi = tuple(self.phi_on_queries)
        queried = tuple(tuple(item) for item in self.queried_bar_tuples)
        coordinates = tuple(Fraction(item) for item in self.benchmark_coordinates)
        benchmark = tuple(self.benchmark_tuple)
        if any(not isinstance(item, ResolutionBasisImage) for item in psi + homotopy):
            raise TypeError("bar equivalence basis data have invalid type")
        if any(not isinstance(item, BarPhiValue) for item in phi):
            raise TypeError("bar equivalence phi data have invalid type")
        if len({item.group_tuple for item in phi}) != len(phi):
            raise ValueError("bar equivalence contains duplicate phi queries")
        object.__setattr__(self, "psi_on_basis", psi)
        object.__setattr__(self, "resolution_homotopy_on_basis", homotopy)
        object.__setattr__(self, "phi_on_queries", phi)
        object.__setattr__(self, "queried_bar_tuples", queried)
        object.__setattr__(self, "benchmark_coordinates", coordinates)
        object.__setattr__(self, "benchmark_tuple", benchmark)

    def normalized_tuples(self, degree: int) -> tuple[tuple[str, ...], ...]:
        if type(degree) is not int or degree < 0:
            raise ValueError("bar degree must be nonnegative")
        nonidentity = self.finite_group.element_order[1:]
        return tuple(itertools.product(nonidentity, repeat=degree))


@dataclass(frozen=True, slots=True)
class CochainCoordinateCertificate:
    resolution_id: str
    degree: int
    coordinates: tuple[Fraction, ...]
    coboundary_1cochain: tuple[Fraction, ...]
    coefficient_character: GF2Character
    mod_one: bool
    source_cocycle_digest: str
    certificate_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(Fraction(item) for item in self.coordinates))
        object.__setattr__(self, "coboundary_1cochain", tuple(Fraction(item) for item in self.coboundary_1cochain))
        if not isinstance(self.coefficient_character, GF2Character):
            raise TypeError("coordinate certificate requires a GF2Character")
        if type(self.mod_one) is not bool:
            raise TypeError("coordinate certificate mod_one flag must be boolean")


def _resolution_chain_mapping(chain: SparseResolutionChain) -> dict[str, Any]:
    return {
        "degree": chain.degree,
        "terms": [
            {
                "basis_id": term.basis_id,
                "coefficient": term.coefficient,
                "element": term.element,
            }
            for term in chain.terms
        ],
    }


def _bar_chain_mapping(chain: SparseBarChain) -> dict[str, Any]:
    return {
        "degree": chain.degree,
        "terms": [
            {
                "coefficient": term.coefficient,
                "group_tuple": list(term.group_tuple),
                "left_element": term.left_element,
            }
            for term in chain.terms
        ],
    }


def _parse_resolution_chain(value: Any, path: str) -> SparseResolutionChain:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(value, {"degree", "terms"}, path)
    if not isinstance(value["terms"], list):
        raise TypeError(f"{path}.terms: expected array")
    terms = []
    for index, item in enumerate(value["terms"]):
        item_path = f"{path}.terms[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"basis_id", "coefficient", "element"}, item_path)
        terms.append(SparseResolutionChainTerm(item["basis_id"], item["element"], item["coefficient"]))
    return SparseResolutionChain(value["degree"], tuple(terms))


def _parse_bar_chain(value: Any, path: str) -> SparseBarChain:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(value, {"degree", "terms"}, path)
    if not isinstance(value["terms"], list):
        raise TypeError(f"{path}.terms: expected array")
    terms = []
    for index, item in enumerate(value["terms"]):
        item_path = f"{path}.terms[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"coefficient", "group_tuple", "left_element"}, item_path)
        terms.append(SparseBarTerm(item["left_element"], tuple(item["group_tuple"]), item["coefficient"]))
    return SparseBarChain(value["degree"], tuple(terms))


def _equivalence_core(value: BarResolutionEquivalence) -> dict[str, Any]:
    return {
        "bar_homotopy_algorithm": value.bar_homotopy_algorithm,
        "benchmark_coordinates": [str(item) for item in value.benchmark_coordinates],
        "benchmark_tuple": list(value.benchmark_tuple),
        "finite_group": {
            "element_order": list(value.finite_group.element_order),
            "group_id": value.finite_group.group_id,
            "identity_index": value.finite_group.identity_index,
            "inverse_indices": list(value.finite_group.inverse_indices),
            "multiplication_table": [list(row) for row in value.finite_group.multiplication_table],
            "table_digest": value.finite_group.table_digest,
        },
        "lookahead_boundary": {
            "column_count": value.lookahead_boundary.column_count,
            "entries": [
                {
                    "column": entry.column,
                    "row": entry.row,
                    "terms": [[term.coefficient, term.element] for term in entry.terms],
                }
                for entry in value.lookahead_boundary.entries
            ],
            "row_count": value.lookahead_boundary.row_count,
        },
        "phi_algorithm": value.phi_algorithm,
        "phi_on_queries": [
            {
                "bar_homotopy": None if item.bar_homotopy is None else _bar_chain_mapping(item.bar_homotopy),
                "group_tuple": list(item.group_tuple),
                "image": _resolution_chain_mapping(item.image),
            }
            for item in value.phi_on_queries
        ],
        "psi_on_basis": [
            {
                "basis_id": item.basis_id,
                "degree": item.degree,
                "image": _bar_chain_mapping(item.image),
            }
            for item in value.psi_on_basis
        ],
        "queried_bar_tuples": [list(item) for item in value.queried_bar_tuples],
        "resolution": free_resolution_mapping(value.resolution),
        "resolution_homotopy_on_basis": [
            {
                "basis_id": item.basis_id,
                "degree": item.degree,
                "image": _resolution_chain_mapping(item.image),
            }
            for item in value.resolution_homotopy_on_basis
        ],
        "resolution_id": value.resolution_id,
    }


def bar_equivalence_digest(value: BarResolutionEquivalence) -> str:
    from .cochains import _task5_domain_digest

    return _task5_domain_digest(
        "task5-bar-resolution-equivalence-v1", _equivalence_core(value)
    )


def bar_equivalence_mapping(value: BarResolutionEquivalence) -> dict[str, Any]:
    return {
        **_equivalence_core(value),
        "equivalence_id": value.equivalence_id,
        "record_type": "bar-resolution-equivalence",
        "schema_version": 1,
    }


def make_bar_resolution_equivalence(
    *,
    resolution: FreeResolutionCertificate,
    phi_algorithm: str,
    psi_on_basis: Sequence[ResolutionBasisImage],
    resolution_homotopy_on_basis: Sequence[ResolutionBasisImage],
    bar_homotopy_algorithm: str,
    phi_on_queries: Sequence[BarPhiValue],
    queried_bar_tuples: Sequence[Sequence[str]],
    lookahead_boundary: SparseGroupRingMatrix,
    benchmark_coordinates: Sequence[Fraction | int],
    benchmark_tuple: Sequence[str],
) -> BarResolutionEquivalence:
    if resolution.finite_group is None:
        raise ValueError("bar equivalence requires a finite-group resolution")
    provisional = BarResolutionEquivalence(
        resolution.resolution_id,
        resolution,
        resolution.finite_group,
        phi_algorithm,
        tuple(psi_on_basis),
        tuple(resolution_homotopy_on_basis),
        bar_homotopy_algorithm,
        tuple(phi_on_queries),
        tuple(tuple(item) for item in queried_bar_tuples),
        lookahead_boundary,
        tuple(Fraction(item) for item in benchmark_coordinates),
        tuple(benchmark_tuple),
        "sha256:" + "0" * 64,
    )
    return BarResolutionEquivalence(
        provisional.resolution_id,
        provisional.resolution,
        provisional.finite_group,
        provisional.phi_algorithm,
        provisional.psi_on_basis,
        provisional.resolution_homotopy_on_basis,
        provisional.bar_homotopy_algorithm,
        provisional.phi_on_queries,
        provisional.queried_bar_tuples,
        provisional.lookahead_boundary,
        provisional.benchmark_coordinates,
        provisional.benchmark_tuple,
        bar_equivalence_digest(provisional),
    )


def assemble_gap_bar_resolution_equivalence(
    raw_export: Mapping[str, Any],
    *,
    resolution: FreeResolutionCertificate,
    benchmark_coordinates: Sequence[Fraction | int],
    benchmark_tuple: Sequence[str],
) -> BarResolutionEquivalence:
    """Normalize a raw GAP/HAP trace into the sole public bar schema."""

    if not isinstance(raw_export, Mapping):
        raise TypeError("raw GAP bar export must be an object")
    _fields(
        raw_export,
        {
            "bar_homotopy_algorithm",
            "finite_group",
            "phi_algorithm",
            "phi_on_queries",
            "psi_on_basis",
            "queried_bar_tuples",
            "resolution_homotopy_on_basis",
        },
        "$gap_bar_export",
    )
    if resolution.finite_group is None:
        raise ValueError("raw GAP bar export requires a finite resolution")
    raw_table = dict(raw_export["finite_group"])
    raw_table["table_digest"] = resolution.finite_group.table_digest
    if _parse_finite(raw_table, "$gap_bar_export.finite_group") != resolution.finite_group:
        raise ValueError("raw GAP finite table differs from the resolution authority")
    psi = []
    raw_psi = raw_export["psi_on_basis"]
    raw_homotopy = raw_export["resolution_homotopy_on_basis"]
    if not isinstance(raw_psi, list) or not isinstance(raw_homotopy, list):
        raise TypeError("raw GAP basis traces must be degree arrays")
    if len(raw_psi) != 5 or len(raw_homotopy) != 5:
        raise ValueError("raw GAP basis traces must cover degrees zero through four")
    homotopy = []
    for degree in range(5):
        if not isinstance(raw_psi[degree], list) or not isinstance(raw_homotopy[degree], list):
            raise TypeError("raw GAP basis degree must be an array")
        for item in raw_psi[degree]:
            _fields(item, {"basis_id", "image"}, "$gap_bar_export.psi_on_basis[]")
            psi.append(
                ResolutionBasisImage(
                    degree,
                    item["basis_id"],
                    _parse_bar_chain(
                        {"degree": degree, "terms": item["image"]},
                        "$gap_bar_export.psi_on_basis[].image",
                    ),
                )
            )
        for item in raw_homotopy[degree]:
            _fields(item, {"basis_id", "image"}, "$gap_bar_export.resolution_homotopy_on_basis[]")
            homotopy.append(
                ResolutionBasisImage(
                    degree,
                    item["basis_id"],
                    _parse_resolution_chain(
                        item["image"],
                        "$gap_bar_export.resolution_homotopy_on_basis[].image",
                    ),
                )
            )
    phi = []
    if not isinstance(raw_export["phi_on_queries"], list):
        raise TypeError("raw GAP phi queries must be an array")
    for item in raw_export["phi_on_queries"]:
        _fields(item, {"bar_homotopy", "group_tuple", "image"}, "$gap_bar_export.phi_on_queries[]")
        group_tuple = tuple(item["group_tuple"])
        raw_bar_homotopy = item["bar_homotopy"]
        phi.append(
            BarPhiValue(
                group_tuple,
                _parse_resolution_chain(
                    item["image"], "$gap_bar_export.phi_on_queries[].image"
                ),
                None
                if raw_bar_homotopy is None
                else _parse_bar_chain(
                    {"degree": len(group_tuple) + 1, "terms": raw_bar_homotopy},
                    "$gap_bar_export.phi_on_queries[].bar_homotopy",
                ),
            )
        )
    return make_bar_resolution_equivalence(
        resolution=resolution,
        phi_algorithm=raw_export["phi_algorithm"],
        psi_on_basis=psi,
        resolution_homotopy_on_basis=homotopy,
        bar_homotopy_algorithm=raw_export["bar_homotopy_algorithm"],
        phi_on_queries=phi,
        queried_bar_tuples=raw_export["queried_bar_tuples"],
        lookahead_boundary=resolution.lookahead_boundary,
        benchmark_coordinates=benchmark_coordinates,
        benchmark_tuple=benchmark_tuple,
    )


def make_target_bar_resolution_equivalence(
    *,
    target_resolution: FreeResolutionCertificate,
    phi_algorithm: str,
    bar_homotopy_algorithm: str,
    basis_traces: Sequence[TargetResolutionBasisTrace],
    phi_traces: Sequence[TargetBarPhiTrace],
    queried_bar_tuples: Sequence[Sequence[str]],
) -> TargetBarResolutionEquivalence:
    provisional = TargetBarResolutionEquivalence(
        target_resolution.resolution_id,
        phi_algorithm,
        bar_homotopy_algorithm,
        tuple(basis_traces),
        tuple(phi_traces),
        tuple(tuple(item) for item in queried_bar_tuples),
        target_resolution.lookahead_boundary,
        "sha256:" + "0" * 64,
    )
    return TargetBarResolutionEquivalence(
        provisional.target_resolution_id,
        provisional.phi_algorithm,
        provisional.bar_homotopy_algorithm,
        provisional.basis_traces,
        provisional.phi_traces,
        provisional.queried_bar_tuples,
        provisional.lookahead_boundary,
        target_bar_equivalence_digest(provisional),
    )


def assemble_gap_target_bar_resolution_equivalence(
    raw_export: Mapping[str, Any],
    *,
    target_resolution: FreeResolutionCertificate,
) -> TargetBarResolutionEquivalence:
    if not isinstance(raw_export, Mapping):
        raise TypeError("raw GAP target bar export must be an object")
    _fields(
        raw_export,
        {
            "bar_homotopy_algorithm",
            "phi_algorithm",
            "phi_on_queries",
            "psi_on_basis",
            "queried_bar_tuples",
            "resolution_homotopy_on_basis",
        },
        "$gap_target_bar_export",
    )
    raw_psi = raw_export["psi_on_basis"]
    raw_homotopy = raw_export["resolution_homotopy_on_basis"]
    if (
        not isinstance(raw_psi, list)
        or not isinstance(raw_homotopy, list)
        or len(raw_psi) != 5
        or len(raw_homotopy) != 5
    ):
        raise ValueError("raw target basis traces must cover degree arrays zero through four")
    basis = []
    for degree in range(5):
        if not isinstance(raw_psi[degree], list) or not isinstance(
            raw_homotopy[degree], list
        ):
            raise TypeError("raw target basis degree must be an array")
        psi_by_basis = {}
        for item in raw_psi[degree]:
            _fields(item, {"basis_id", "image"}, "$gap_target_bar_export.psi_on_basis[]")
            image = _parse_bar_chain(
                {"degree": degree, "terms": item["image"]},
                "$gap_target_bar_export.psi_on_basis[].image",
            )
            psi_by_basis[item["basis_id"]] = tuple(
                BarComparisonTerm(
                    term.left_element, term.group_tuple, term.coefficient
                )
                for term in image.terms
            )
        homotopy_by_basis = {}
        for item in raw_homotopy[degree]:
            _fields(
                item,
                {"basis_id", "image"},
                "$gap_target_bar_export.resolution_homotopy_on_basis[]",
            )
            image = _parse_resolution_chain(
                item["image"],
                "$gap_target_bar_export.resolution_homotopy_on_basis[].image",
            )
            homotopy_by_basis[item["basis_id"]] = tuple(
                ResolutionComparisonTerm(
                    term.basis_id, term.element, term.coefficient
                )
                for term in image.terms
            )
        if set(psi_by_basis) != set(homotopy_by_basis):
            raise ValueError("target psi and homotopy basis domains differ")
        for basis_id in sorted(
            psi_by_basis,
            key=lambda item: int(item.split(":", 1)[1]),
        ):
            basis.append(
                TargetResolutionBasisTrace(
                    degree,
                    basis_id,
                    psi_by_basis[basis_id],
                    homotopy_by_basis[basis_id],
                )
            )
    if not isinstance(raw_export["phi_on_queries"], list) or not isinstance(
        raw_export["queried_bar_tuples"], list
    ):
        raise TypeError("raw target phi/query domains must be arrays")
    raw_phi = {}
    for item in raw_export["phi_on_queries"]:
        _fields(
            item,
            {"bar_homotopy", "group_tuple", "image"},
            "$gap_target_bar_export.phi_on_queries[]",
        )
        group_tuple = tuple(item["group_tuple"])
        if item["bar_homotopy"] is None:
            raise ValueError("target phi query lacks its bar homotopy")
        image = _parse_resolution_chain(
            item["image"], "$gap_target_bar_export.phi_on_queries[].image"
        )
        homotopy = _parse_bar_chain(
            {
                "degree": len(group_tuple) + 1,
                "terms": item["bar_homotopy"],
            },
            "$gap_target_bar_export.phi_on_queries[].bar_homotopy",
        )
        if group_tuple in raw_phi:
            raise ValueError("raw target phi contains a duplicate query")
        raw_phi[group_tuple] = TargetBarPhiTrace(
            group_tuple,
            tuple(
                ResolutionComparisonTerm(
                    term.basis_id, term.element, term.coefficient
                )
                for term in image.terms
            ),
            tuple(
                BarComparisonTerm(
                    term.left_element, term.group_tuple, term.coefficient
                )
                for term in homotopy.terms
            ),
        )
    queries = tuple(
        sorted(
            (tuple(item) for item in raw_export["queried_bar_tuples"]),
            key=lambda item: (len(item), item),
        )
    )
    if set(queries) != set(raw_phi):
        raise ValueError("target queried domain differs from target phi traces")
    return make_target_bar_resolution_equivalence(
        target_resolution=target_resolution,
        phi_algorithm=raw_export["phi_algorithm"],
        bar_homotopy_algorithm=raw_export["bar_homotopy_algorithm"],
        basis_traces=basis,
        phi_traces=tuple(raw_phi[item] for item in queries),
        queried_bar_tuples=queries,
    )


def _backend_observation_binding(
    raw: Any,
    raw_output: bytes,
    attestation: LauncherExecutionAttestation,
    authority: Task5VerificationAuthority,
    *,
    inclusion_id: str,
    allow_diagnostic: bool,
) -> tuple[str, str, str]:
    path = "$gap_inclusion_export.backend_environment"
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path}: expected observed backend object")
    if type(raw_output) is not bytes:
        raise TypeError("launcher binding requires exact raw output bytes")
    if not isinstance(attestation, LauncherExecutionAttestation):
        raise TypeError("launcher binding requires a typed execution attestation")
    expected_fields = {
        "backend_lock_digest",
        "execution_mode",
        "gap_version",
        "packages",
        "runtime_manifest_digest",
        "schema_version",
    }
    _fields(raw, expected_fields, path)
    trusted_inclusion = next(
        (item for item in authority.inclusions if item.inclusion_id == inclusion_id),
        None,
    )
    if (
        launcher_execution_attestation_digest(attestation)
        != attestation.attestation_id
        or attestation.raw_output_digest != _bytes_digest(raw_output)
        or attestation.backend_observation_digest
        != _task5_domain_digest("task5-launcher-backend-observation-v1", raw)
        or attestation.exit_status != 0
        or trusted_inclusion is None
        or trusted_inclusion.launcher_attestation_id != attestation.attestation_id
    ):
        raise ValueError(
            "launcher execution attestation does not bind the exact output and external authority"
        )
    if raw["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported backend observation schema")
    if type(raw["gap_version"]) is not str or not raw["gap_version"]:
        raise ValueError(f"{path}.gap_version: expected observed version")
    if not isinstance(raw["packages"], list):
        raise TypeError(f"{path}.packages: expected array")
    packages = []
    for index, item in enumerate(raw["packages"]):
        item_path = f"{path}.packages[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"name", "version"}, item_path)
        if any(type(item[field]) is not str or not item[field] for field in ("name", "version")):
            raise ValueError(f"{item_path}: expected observed package name and version")
        packages.append((item["name"], item["version"]))
    if tuple(packages) != tuple(sorted(packages)) or len(set(packages)) != len(packages):
        raise ValueError(f"{path}.packages: expected canonical unique package order")
    if raw["execution_mode"] == "diagnostic-local":
        if any(
            raw[field] is not None
            for field in ("backend_lock_digest", "runtime_manifest_digest")
        ):
            raise ValueError("diagnostic GAP observation cannot carry release authority")
        if not allow_diagnostic:
            raise ValueError(
                "local GAP observation is diagnostic-only and cannot mint a release certificate"
            )
        if attestation.release_certified or attestation.runtime_manifest_digest is not None:
            raise ValueError("diagnostic GAP output cannot carry a release launcher attestation")
        diagnostic = _task5_domain_digest(
            "task5-diagnostic-backend-observation-v1", raw
        )
        return diagnostic, diagnostic, diagnostic
    if raw["execution_mode"] != "locked-oci":
        raise ValueError(f"{path}: invalid release execution mode")
    if not attestation.release_certified or attestation.runtime_manifest_digest is None:
        raise ValueError(
            "release GAP output requires a locked-launcher runtime-manifest attestation"
        )
    for field in ("backend_lock_digest", "runtime_manifest_digest"):
        _digest(raw[field], f"{path}.{field}")
    backend_lock, backend_environment, runtime_provenance = _task5_backend_binding()
    if (
        raw["backend_lock_digest"] != backend_lock
        or raw["runtime_manifest_digest"] != attestation.runtime_manifest_digest
        or raw["backend_lock_digest"] != authority.backend_lock_digest
        or backend_environment != authority.backend_environment_id
        or runtime_provenance != authority.runtime_provenance_digest
    ):
        raise ValueError("release backend observation differs from trusted Task 5 authority")
    locked = _locked_environment_core()
    expected_packages = tuple(
        sorted((item["name"], item["version"]) for item in locked["packages"])
    )
    if tuple(packages) != expected_packages:
        raise ValueError("release backend observation has unpinned package versions")
    gap_version = next(version for name, version in packages if name == "GAP")
    if raw["gap_version"] != gap_version:
        raise ValueError("release backend observation GAP version is inconsistent")
    return backend_lock, backend_environment, runtime_provenance


def _parse_gap_bar_comparison_traces(
    raw: Any,
    *,
    equivalence: BarResolutionEquivalence,
    target_resolution: FreeResolutionCertificate,
    source_element_images: Sequence[str],
) -> tuple[BarComparisonBasisTrace, ...]:
    if not isinstance(raw, list):
        raise TypeError("$gap_inclusion_export.bar_comparison_traces: expected array")
    table = equivalence.finite_group
    images = tuple(source_element_images)
    if len(images) != len(table.element_order):
        raise ValueError("source inclusion must map every finite-group element")
    transported = dict(zip(table.element_order, images, strict=True))
    psi = {(item.degree, item.basis_id): item.image for item in equivalence.psi_on_basis}
    expected_order = tuple(
        (degree, basis_id)
        for degree, basis in enumerate(equivalence.resolution.basis)
        for basis_id in basis
    )
    traces = []
    for index, item in enumerate(raw):
        path = f"$gap_inclusion_export.bar_comparison_traces[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{path}: expected object")
        _fields(
            item,
            {
                "degree",
                "source_basis_id",
                "source_psi",
                "target_phi_input",
                "target_phi_output",
            },
            path,
        )
        degree = item["degree"]
        basis_id = item["source_basis_id"]
        source = _parse_bar_chain(item["source_psi"], f"{path}.source_psi")
        target_input = _parse_bar_chain(
            item["target_phi_input"], f"{path}.target_phi_input"
        )
        target_output = _parse_resolution_chain(
            item["target_phi_output"], f"{path}.target_phi_output"
        )
        expected_source = psi.get((degree, basis_id))
        if source != expected_source:
            raise ValueError(f"{path}: source psi differs from verified source equivalence")
        expected_transport = SparseBarChain(
            degree,
            tuple(
                sorted(
                    SparseBarTerm(
                        transported[term.left_element],
                        tuple(transported[element] for element in term.group_tuple),
                        term.coefficient,
                    )
                    for term in source.terms
                )
            ),
        )
        if target_input != expected_transport:
            raise ValueError(f"{path}: target-phi input differs from literal bar inclusion")
        if target_output.degree != degree:
            raise ValueError(f"{path}: target-phi output has the wrong degree")
        for term in target_input.terms:
            _normal_key(target_resolution, term.left_element)
            for element in term.group_tuple:
                _normal_key(target_resolution, element)
        for term in target_output.terms:
            _normal_key(target_resolution, term.element)
        traces.append(
            BarComparisonBasisTrace(
                degree,
                basis_id,
                tuple(
                    BarComparisonTerm(
                        term.left_element, term.group_tuple, term.coefficient
                    )
                    for term in source.terms
                ),
                tuple(
                    BarComparisonTerm(
                        term.left_element, term.group_tuple, term.coefficient
                    )
                    for term in target_input.terms
                ),
                tuple(
                    ResolutionComparisonTerm(
                        term.basis_id, term.element, term.coefficient
                    )
                    for term in target_output.terms
                ),
            )
        )
    if tuple((trace.degree, trace.source_basis_id) for trace in traces) != expected_order:
        raise ValueError("independent target-phi traces do not cover canonical source basis")
    return tuple(traces)


def assemble_gap_inclusion_fixture(
    raw_export: bytes | Mapping[str, Any],
    execution_attestation: LauncherExecutionAttestation | None = None,
    *,
    authority: Task5VerificationAuthority,
    affine_pcp_certificate: AffinePCPIsomorphismCertificate,
    catalogue_record_digest: str,
    source_group_id: str,
    target_group_id: str,
    source_construction: str,
    target_construction: str,
    inclusion_id: str,
    literal_stabilizer_digest: str,
    literal_element_digest: str,
    transported_inclusion_digest: str,
    benchmark_coordinates: Sequence[Fraction | int],
    benchmark_tuple: Sequence[str],
    allow_diagnostic: bool = False,
    authenticated_backend_environment: Mapping[str, Any] | None = None,
) -> tuple[BarResolutionEquivalence, InclusionChainMapCertificate]:
    """Assemble the canonical Python certificates from one raw GAP export."""

    if type(allow_diagnostic) is not bool:
        raise TypeError("allow_diagnostic must be a boolean")
    if execution_attestation is None:
        raise ValueError(
            "raw GAP assembly requires a launcher execution attestation bound to exact output bytes"
        )
    if authenticated_backend_environment is not None:
        raise ValueError(
            "caller-supplied backend mappings are not launcher execution attestations"
        )
    if type(raw_export) is not bytes:
        raise TypeError("raw GAP inclusion export must be exact canonical JSON bytes")
    raw_output = raw_export
    raw_export = _strict_json(raw_output)
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("raw GAP assembler requires external Task 5 authority")
    if "maps" in raw_export:
        raise ValueError(
            "raw GAP export cannot supply final maps; independent target-phi traces are required"
        )
    _fields(
        raw_export,
        {
            "backend_environment",
            "bar_equivalence",
            "bar_comparison_traces",
            "chain_map_algorithm",
            "diagnostic_backend",
            "diagnostic_maps",
            "finite_group",
            "lookahead_boundary",
            "source",
            "source_element_images",
            "target",
            "target_bar_equivalence",
        },
        "$gap_inclusion_export",
    )
    raw_projection_digest = gap_inclusion_projection_digest(raw_export)
    trusted_inclusion = next(
        (
            item
            for item in authority.inclusions
            if item.inclusion_id == inclusion_id
        ),
        None,
    )
    if trusted_inclusion is None:
        raise ValueError("inclusion is absent from external Task 5 authority")
    if (
        execution_attestation.gap_inclusion_projection_digest
        != raw_projection_digest
        or trusted_inclusion.gap_inclusion_projection_digest
        != raw_projection_digest
    ):
        raise ValueError(
            "raw GAP inclusion projection differs from the launcher record "
            "or external inclusion authority"
        )
    raw_table = raw_export["finite_group"]
    if not isinstance(raw_table, Mapping):
        raise TypeError("$gap_inclusion_export.finite_group: expected object")
    _fields(
        raw_table,
        {
            "element_order",
            "group_id",
            "identity_index",
            "inverse_indices",
            "multiplication_table",
            "table_digest",
        },
        "$gap_inclusion_export.finite_group",
    )
    if raw_table["table_digest"] is not None:
        raise ValueError("raw GAP finite table must not self-declare Python authority")
    table = FiniteGroupTable(
        raw_table["group_id"],
        tuple(raw_table["element_order"]),
        raw_table["identity_index"],
        tuple(tuple(row) for row in raw_table["multiplication_table"]),
        tuple(raw_table["inverse_indices"]),
    )
    backend_lock, backend_environment, runtime_provenance = _backend_observation_binding(
        raw_export["backend_environment"],
        raw_output,
        execution_attestation,
        authority,
        inclusion_id=inclusion_id,
        allow_diagnostic=allow_diagnostic,
    )
    source = assemble_gap_free_resolution_certificate(
        raw_export["source"],
        group_id=source_group_id,
        affine_pcp_certificate=affine_pcp_certificate,
        catalogue_record_digest=catalogue_record_digest,
        finite_group=table,
        construction=source_construction,
        backend_lock_digest=backend_lock,
        backend_environment_id=backend_environment,
        runtime_provenance_digest=runtime_provenance,
    )
    target = assemble_gap_free_resolution_certificate(
        raw_export["target"],
        group_id=target_group_id,
        affine_pcp_certificate=affine_pcp_certificate,
        catalogue_record_digest=catalogue_record_digest,
        finite_group=None,
        construction=target_construction,
        backend_lock_digest=backend_lock,
        backend_environment_id=backend_environment,
        runtime_provenance_digest=runtime_provenance,
    )
    if _parse_matrix(
        raw_export["lookahead_boundary"],
        "$gap_inclusion_export.lookahead_boundary",
    ) != source.lookahead_boundary:
        raise ValueError("raw GAP lookahead differs from the source degree five")
    equivalence = assemble_gap_bar_resolution_equivalence(
        raw_export["bar_equivalence"],
        resolution=source,
        benchmark_coordinates=benchmark_coordinates,
        benchmark_tuple=benchmark_tuple,
    )
    if raw_export["chain_map_algorithm"] != (
        "hap-1.70-bar-phi-target-inclusion-psi-source"
    ):
        raise ValueError("raw GAP inclusion used a noncanonical chain-map algorithm")
    diagnostic_maps = tuple(
        _parse_matrix(
            matrix, f"$gap_inclusion_export.diagnostic_maps[{index}]"
        )
        for index, matrix in enumerate(raw_export["diagnostic_maps"])
    )
    traces = _parse_gap_bar_comparison_traces(
        raw_export["bar_comparison_traces"],
        equivalence=equivalence,
        target_resolution=target,
        source_element_images=raw_export["source_element_images"],
    )
    target_equivalence = assemble_gap_target_bar_resolution_equivalence(
        raw_export["target_bar_equivalence"],
        target_resolution=target,
    )
    zero_maps = tuple(
        SparseGroupRingMatrix(
            len(target.basis[degree]), len(source.basis[degree]), ()
        )
        for degree in range(5)
    )
    placeholder = ("sha256:" + "0" * 64,) * 4
    provisional = make_inclusion_chain_map_certificate(
        inclusion_id=inclusion_id,
        literal_stabilizer_digest=literal_stabilizer_digest,
        literal_element_digest=literal_element_digest,
        transported_inclusion_digest=transported_inclusion_digest,
        source_resolution=source,
        target_resolution=target,
        source_element_images=raw_export["source_element_images"],
        maps=zero_maps,
        source_bar_equivalence_id=equivalence.equivalence_id,
        target_bar_equivalence=target_equivalence,
        launcher_attestation=execution_attestation,
        bar_comparison_traces=traces,
        diagnostic_backend=raw_export["diagnostic_backend"],
        diagnostic_maps=diagnostic_maps,
        diagnostic_outcome="commuting",
        diagnostic_residue_digests=placeholder,
    )
    maps = _reconstruct_comparison_maps(provisional)
    provisional = make_inclusion_chain_map_certificate(
        inclusion_id=inclusion_id,
        literal_stabilizer_digest=literal_stabilizer_digest,
        literal_element_digest=literal_element_digest,
        transported_inclusion_digest=transported_inclusion_digest,
        source_resolution=source,
        target_resolution=target,
        source_element_images=raw_export["source_element_images"],
        maps=maps,
        source_bar_equivalence_id=equivalence.equivalence_id,
        target_bar_equivalence=target_equivalence,
        launcher_attestation=execution_attestation,
        bar_comparison_traces=traces,
        diagnostic_backend=raw_export["diagnostic_backend"],
        diagnostic_maps=diagnostic_maps,
        diagnostic_outcome="commuting",
        diagnostic_residue_digests=placeholder,
    )
    failures = tuple(
        degree
        for degree in range(1, 5)
        if _inclusion_left(provisional, degree, diagnostic_maps)
        != _inclusion_right(provisional, degree, diagnostic_maps)
    )
    residues = diagnostic_residue_digests(provisional)
    if (
        trusted_inclusion.literal_stabilizer_digest
        != literal_stabilizer_digest
        or trusted_inclusion.literal_element_digest != literal_element_digest
        or trusted_inclusion.transported_inclusion_digest
        != transported_inclusion_digest
        or trusted_inclusion.source_bar_equivalence_id
        != equivalence.equivalence_id
        or trusted_inclusion.target_bar_equivalence_id
        != target_equivalence.equivalence_id
        or trusted_inclusion.launcher_attestation_id
        != execution_attestation.attestation_id
        or trusted_inclusion.gap_inclusion_projection_digest
        != raw_projection_digest
        or trusted_inclusion.diagnostic_backend
        != raw_export["diagnostic_backend"]
        or trusted_inclusion.diagnostic_failure_degrees != failures
        or trusted_inclusion.diagnostic_residue_digests != residues
    ):
        raise ValueError("inclusion differs from external Task 5 authority")
    diagnostic_outcome = trusted_inclusion.diagnostic_outcome
    inclusion = make_inclusion_chain_map_certificate(
        inclusion_id=inclusion_id,
        literal_stabilizer_digest=literal_stabilizer_digest,
        literal_element_digest=literal_element_digest,
        transported_inclusion_digest=transported_inclusion_digest,
        source_resolution=source,
        target_resolution=target,
        source_element_images=raw_export["source_element_images"],
        maps=maps,
        source_bar_equivalence_id=equivalence.equivalence_id,
        target_bar_equivalence=target_equivalence,
        launcher_attestation=execution_attestation,
        bar_comparison_traces=traces,
        diagnostic_backend=raw_export["diagnostic_backend"],
        diagnostic_maps=diagnostic_maps,
        diagnostic_outcome=diagnostic_outcome,
        diagnostic_residue_digests=residues,
    )
    if (
        inclusion.gap_inclusion_projection_digest != raw_projection_digest
        or gap_inclusion_projection_digest(inclusion) != raw_projection_digest
    ):
        raise ValueError(
            "typed inclusion certificate does not reconstruct the raw GAP "
            "inclusion projection"
        )
    if not allow_diagnostic:
        equivalence_report = verify_bar_resolution_equivalence(equivalence, authority)
        if not equivalence_report.valid:
            raise ValueError("assembled source bar equivalence failed independent replay")
        inclusion_report = verify_inclusion_chain_map(
            inclusion,
            authority,
            require_release=True,
            trusted_release_attestation=execution_attestation,
        )
        if not inclusion_report.valid:
            raise ValueError("assembled inclusion failed independent replay")
    return equivalence, inclusion


def dumps_gap_inclusion_fixture(
    equivalence: BarResolutionEquivalence,
    inclusion: InclusionChainMapCertificate,
) -> bytes:
    if inclusion.source_bar_equivalence_id != equivalence.equivalence_id:
        raise ValueError("inclusion does not bind the embedded bar equivalence")
    return _canonical_json(
        {
            "bar_equivalence": bar_equivalence_mapping(equivalence),
            "inclusion": inclusion_chain_map_mapping(inclusion),
            "record_type": "task5-p4mm-fixture",
            "schema_version": 1,
        }
    )


def _parse_equivalence(value: Mapping[str, Any], path: str) -> BarResolutionEquivalence:
    _fields(
        value,
        {
            "bar_homotopy_algorithm", "benchmark_coordinates", "benchmark_tuple",
            "equivalence_id", "finite_group", "lookahead_boundary", "phi_algorithm",
            "phi_on_queries", "psi_on_basis", "queried_bar_tuples", "record_type",
            "resolution", "resolution_homotopy_on_basis", "resolution_id", "schema_version",
        },
        path,
    )
    if value["record_type"] != "bar-resolution-equivalence" or value["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported bar-equivalence schema")
    resolution = _parse_resolution(value["resolution"], f"{path}.resolution")
    for key in (
        "psi_on_basis",
        "resolution_homotopy_on_basis",
        "phi_on_queries",
        "queried_bar_tuples",
        "benchmark_coordinates",
        "benchmark_tuple",
    ):
        if not isinstance(value[key], list):
            raise TypeError(f"{path}.{key}: expected array")
    psi_values = []
    for index, item in enumerate(value["psi_on_basis"]):
        item_path = f"{path}.psi_on_basis[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"basis_id", "degree", "image"}, item_path)
        psi_values.append(
            ResolutionBasisImage(
                item["degree"], item["basis_id"],
                _parse_bar_chain(item["image"], f"{item_path}.image"),
            )
        )
    psi = tuple(psi_values)
    homotopy_values = []
    for index, item in enumerate(value["resolution_homotopy_on_basis"]):
        item_path = f"{path}.resolution_homotopy_on_basis[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"basis_id", "degree", "image"}, item_path)
        homotopy_values.append(
            ResolutionBasisImage(
                item["degree"], item["basis_id"],
                _parse_resolution_chain(item["image"], f"{item_path}.image"),
            )
        )
    homotopy = tuple(homotopy_values)
    phi_values = []
    for index, item in enumerate(value["phi_on_queries"]):
        item_path = f"{path}.phi_on_queries[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"bar_homotopy", "group_tuple", "image"}, item_path)
        if not isinstance(item["group_tuple"], list):
            raise TypeError(f"{item_path}.group_tuple: expected array")
        phi_values.append(
            BarPhiValue(
                tuple(item["group_tuple"]),
                _parse_resolution_chain(item["image"], f"{item_path}.image"),
                None if item["bar_homotopy"] is None else _parse_bar_chain(
                    item["bar_homotopy"], f"{item_path}.bar_homotopy"
                ),
            )
        )
    phi = tuple(phi_values)
    result = BarResolutionEquivalence(
        value["resolution_id"],
        resolution,
        _parse_finite(value["finite_group"], f"{path}.finite_group"),
        value["phi_algorithm"],
        psi,
        homotopy,
        value["bar_homotopy_algorithm"],
        phi,
        tuple(tuple(item) for item in value["queried_bar_tuples"]),
        _parse_matrix(value["lookahead_boundary"], f"{path}.lookahead_boundary"),
        tuple(
            _parse_fraction(item, f"{path}.benchmark_coordinates[{index}]")
            for index, item in enumerate(value["benchmark_coordinates"])
        ),
        tuple(value["benchmark_tuple"]),
        value["equivalence_id"],
    )
    if bar_equivalence_digest(result) != result.equivalence_id:
        raise ValueError(f"{path}.equivalence_id: does not bind payload")
    return result


def loads_bar_resolution_equivalence(data: bytes) -> BarResolutionEquivalence:
    value = _strict_json(data)
    if value.get("record_type") == "task5-p4mm-fixture":
        _fields(value, {"bar_equivalence", "inclusion", "record_type", "schema_version"}, "$fixture")
        if value["schema_version"] != 1:
            raise ValueError("$fixture: unsupported fixture schema")
        value = value["bar_equivalence"]
    if not isinstance(value, Mapping):
        raise TypeError("bar-equivalence record must be an object")
    return _parse_equivalence(value, "$bar_equivalence")


def dumps_bar_resolution_equivalence(value: BarResolutionEquivalence) -> bytes:
    return _canonical_json(bar_equivalence_mapping(value))


def _index(table: FiniteGroupTable, element: str) -> int:
    try:
        return table.element_order.index(element)
    except ValueError as error:
        raise ValueError(f"unknown finite-group element {element!r}") from error


def _multiply(table: FiniteGroupTable, left: str, right: str) -> str:
    return table.element_order[table.multiplication_table[_index(table, left)][_index(table, right)]]


def _resolution_collected(
    degree: int, terms: Sequence[SparseResolutionChainTerm]
) -> SparseResolutionChain:
    values: dict[tuple[str, str], int] = {}
    for term in terms:
        key = (term.basis_id, term.element)
        values[key] = values.get(key, 0) + term.coefficient
    return SparseResolutionChain(
        degree,
        tuple(
            SparseResolutionChainTerm(basis, element, coefficient)
            for (basis, element), coefficient in sorted(values.items())
            if coefficient
        ),
    )


def _bar_collected(degree: int, terms: Sequence[SparseBarTerm]) -> SparseBarChain:
    values: dict[tuple[str, tuple[str, ...]], int] = {}
    for term in terms:
        key = (term.left_element, term.group_tuple)
        values[key] = values.get(key, 0) + term.coefficient
    return SparseBarChain(
        degree,
        tuple(
            SparseBarTerm(left, group_tuple, coefficient)
            for (left, group_tuple), coefficient in sorted(values.items())
            if coefficient
        ),
    )


def _resolution_scale(chain: SparseResolutionChain, coefficient: int) -> SparseResolutionChain:
    return _resolution_collected(
        chain.degree,
        tuple(SparseResolutionChainTerm(term.basis_id, term.element, coefficient * term.coefficient) for term in chain.terms),
    )


def _bar_scale(chain: SparseBarChain, coefficient: int) -> SparseBarChain:
    return _bar_collected(
        chain.degree,
        tuple(SparseBarTerm(term.left_element, term.group_tuple, coefficient * term.coefficient) for term in chain.terms),
    )


def _resolution_add(*chains: SparseResolutionChain) -> SparseResolutionChain:
    degree = chains[0].degree
    return _resolution_collected(degree, tuple(term for chain in chains for term in chain.terms))


def _bar_add(*chains: SparseBarChain) -> SparseBarChain:
    degree = chains[0].degree
    return _bar_collected(degree, tuple(term for chain in chains for term in chain.terms))


def _resolution_action(table: FiniteGroupTable, left: str, chain: SparseResolutionChain) -> SparseResolutionChain:
    return _resolution_collected(
        chain.degree,
        tuple(SparseResolutionChainTerm(term.basis_id, _multiply(table, left, term.element), term.coefficient) for term in chain.terms),
    )


def _bar_action(table: FiniteGroupTable, left: str, chain: SparseBarChain) -> SparseBarChain:
    return _bar_collected(
        chain.degree,
        tuple(SparseBarTerm(_multiply(table, left, term.left_element), term.group_tuple, term.coefficient) for term in chain.terms),
    )


def _bar_boundary(table: FiniteGroupTable, chain: SparseBarChain) -> SparseBarChain:
    if chain.degree == 0:
        return SparseBarChain(0, ())
    terms: list[SparseBarTerm] = []
    for term in chain.terms:
        values = term.group_tuple
        first_tuple = values[1:]
        terms.append(SparseBarTerm(_multiply(table, term.left_element, values[0]), first_tuple, term.coefficient))
        for index in range(len(values) - 1):
            product = _multiply(table, values[index], values[index + 1])
            merged = values[:index] + (product,) + values[index + 2 :]
            if "1" not in merged:
                terms.append(SparseBarTerm(term.left_element, merged, term.coefficient * (-1) ** (index + 1)))
        trailing = values[:-1]
        if "1" not in trailing:
            terms.append(SparseBarTerm(term.left_element, trailing, term.coefficient * (-1) ** len(values)))
    return _bar_collected(chain.degree - 1, terms)


def _resolution_boundary(
    equivalence: BarResolutionEquivalence, chain: SparseResolutionChain
) -> SparseResolutionChain:
    if chain.degree == 0:
        return SparseResolutionChain(0, ())
    matrix = equivalence.lookahead_boundary if chain.degree == 5 else equivalence.resolution.boundaries[chain.degree - 1]
    entries = {(entry.row, entry.column): entry.terms for entry in matrix.entries}
    terms: list[SparseResolutionChainTerm] = []
    for term in chain.terms:
        column = int(term.basis_id.split(":", 1)[1])
        for row in range(matrix.row_count):
            for coefficient in entries.get((row, column), ()):
                terms.append(
                    SparseResolutionChainTerm(
                        f"c{chain.degree - 1}:{row}",
                        _multiply(equivalence.finite_group, term.element, coefficient.element),
                        term.coefficient * coefficient.coefficient,
                    )
                )
    return _resolution_collected(chain.degree - 1, terms)


def _phi_lookup(equivalence: BarResolutionEquivalence) -> dict[tuple[str, ...], SparseResolutionChain]:
    return {item.group_tuple: item.image for item in equivalence.phi_on_queries}


def _apply_phi(equivalence: BarResolutionEquivalence, chain: SparseBarChain) -> SparseResolutionChain:
    lookup = _phi_lookup(equivalence)
    terms: list[SparseResolutionChainTerm] = []
    for term in chain.terms:
        if term.group_tuple not in lookup:
            raise ValueError(f"phi query is absent for {term.group_tuple!r}")
        image = _resolution_action(equivalence.finite_group, term.left_element, lookup[term.group_tuple])
        terms.extend(
            SparseResolutionChainTerm(item.basis_id, item.element, term.coefficient * item.coefficient)
            for item in image.terms
        )
    return _resolution_collected(chain.degree, terms)


def _basis_lookup(values: Sequence[ResolutionBasisImage]) -> dict[str, SparseBarChain | SparseResolutionChain]:
    return {item.basis_id: item.image for item in values}


def _apply_basis_map(
    equivalence: BarResolutionEquivalence,
    chain: SparseResolutionChain,
    values: Sequence[ResolutionBasisImage],
):
    lookup = _basis_lookup(values)
    output_bar = bool(values) and isinstance(values[0].image, SparseBarChain)
    if output_bar:
        terms_bar: list[SparseBarTerm] = []
        for term in chain.terms:
            image = lookup.get(term.basis_id)
            if not isinstance(image, SparseBarChain):
                raise ValueError(f"basis image is absent for {term.basis_id}")
            acted = _bar_action(equivalence.finite_group, term.element, image)
            terms_bar.extend(SparseBarTerm(item.left_element, item.group_tuple, term.coefficient * item.coefficient) for item in acted.terms)
        target_degree = chain.degree
        return _bar_collected(target_degree, terms_bar)
    terms_resolution: list[SparseResolutionChainTerm] = []
    for term in chain.terms:
        image = lookup.get(term.basis_id)
        if not isinstance(image, SparseResolutionChain):
            raise ValueError(f"basis image is absent for {term.basis_id}")
        acted = _resolution_action(equivalence.finite_group, term.element, image)
        terms_resolution.extend(SparseResolutionChainTerm(item.basis_id, item.element, term.coefficient * item.coefficient) for item in acted.terms)
    return _resolution_collected(chain.degree + 1, terms_resolution)


def _apply_bar_homotopy(equivalence: BarResolutionEquivalence, chain: SparseBarChain) -> SparseBarChain:
    lookup = {item.group_tuple: item.bar_homotopy for item in equivalence.phi_on_queries}
    terms: list[SparseBarTerm] = []
    for term in chain.terms:
        image = lookup.get(term.group_tuple)
        if not isinstance(image, SparseBarChain):
            raise ValueError(f"bar homotopy query is absent for {term.group_tuple!r}")
        acted = _bar_action(equivalence.finite_group, term.left_element, image)
        terms.extend(SparseBarTerm(item.left_element, item.group_tuple, term.coefficient * item.coefficient) for item in acted.terms)
    return _bar_collected(chain.degree + 1, terms)


def _target_group_multiplier(resolution: FreeResolutionCertificate):
    if resolution.finite_group is not None:
        raise ValueError("target bar equivalence requires the ambient affine-PCP group")
    normal_form, decoder = _pcp_action_and_decoder(resolution)
    cache: dict[tuple[str, str], str] = {}

    def word(coordinates: Sequence[int]) -> str:
        pieces = []
        for index, exponent in enumerate(coordinates, start=1):
            if exponent:
                pieces.append(
                    f"p{index}" if exponent == 1 else f"p{index}^{exponent}"
                )
        return "*".join(pieces) if pieces else "1"

    def multiply(left: str, right: str) -> str:
        cached = cache.get((left, right))
        if cached is not None:
            return cached
        _normal_key(resolution, left)
        _normal_key(resolution, right)
        graded = resolution.group_id.endswith("+onsite-T")

        def split(value: str) -> tuple[str, int]:
            if not graded:
                return value, 0
            if value == "T":
                return "1", 1
            if value.endswith("+T"):
                spatial = value[:-2]
                if not spatial or spatial == "1" or "T" in spatial:
                    raise ValueError("invalid onsite-time-reversal normal form")
                return spatial, 1
            if "T" in value:
                raise ValueError("invalid onsite-time-reversal normal form")
            return value, 0

        left_spatial, left_time = split(left)
        right_spatial, right_time = split(right)
        left_affine = _evaluate_pcp_word(left_spatial, normal_form)
        right_affine = _evaluate_pcp_word(right_spatial, normal_form)
        product = _compose_affine(right_affine, left_affine)
        result = word(decoder(product))
        if graded and left_time ^ right_time:
            result = "T" if result == "1" else result + "+T"
        cache[(left, right)] = result
        return result

    return multiply


def _target_bar_action(
    multiply, left: str, chain: SparseBarChain
) -> SparseBarChain:
    return _bar_collected(
        chain.degree,
        tuple(
            SparseBarTerm(
                multiply(left, term.left_element),
                term.group_tuple,
                term.coefficient,
            )
            for term in chain.terms
        ),
    )


def _target_resolution_action(
    multiply, left: str, chain: SparseResolutionChain
) -> SparseResolutionChain:
    return _resolution_collected(
        chain.degree,
        tuple(
            SparseResolutionChainTerm(
                term.basis_id,
                multiply(left, term.element),
                term.coefficient,
            )
            for term in chain.terms
        ),
    )


def _target_bar_boundary(multiply, chain: SparseBarChain) -> SparseBarChain:
    if chain.degree == 0:
        return SparseBarChain(0, ())
    terms: list[SparseBarTerm] = []
    for term in chain.terms:
        values = term.group_tuple
        terms.append(
            SparseBarTerm(
                multiply(term.left_element, values[0]),
                values[1:],
                term.coefficient,
            )
        )
        for index in range(len(values) - 1):
            merged = (
                values[:index]
                + (multiply(values[index], values[index + 1]),)
                + values[index + 2 :]
            )
            if "1" not in merged:
                terms.append(
                    SparseBarTerm(
                        term.left_element,
                        merged,
                        term.coefficient * (-1) ** (index + 1),
                    )
                )
        trailing = values[:-1]
        if "1" not in trailing:
            terms.append(
                SparseBarTerm(
                    term.left_element,
                    trailing,
                    term.coefficient * (-1) ** len(values),
                )
            )
    return _bar_collected(chain.degree - 1, terms)


def _target_resolution_boundary(
    equivalence: TargetBarResolutionEquivalence,
    resolution: FreeResolutionCertificate,
    multiply,
    chain: SparseResolutionChain,
) -> SparseResolutionChain:
    if chain.degree == 0:
        return SparseResolutionChain(0, ())
    matrix = (
        equivalence.lookahead_boundary
        if chain.degree == 5
        else resolution.boundaries[chain.degree - 1]
    )
    entries = {(entry.row, entry.column): entry.terms for entry in matrix.entries}
    terms = []
    for term in chain.terms:
        column = int(term.basis_id.split(":", 1)[1])
        for row in range(matrix.row_count):
            for coefficient in entries.get((row, column), ()):
                terms.append(
                    SparseResolutionChainTerm(
                        f"c{chain.degree - 1}:{row}",
                        multiply(term.element, coefficient.element),
                        term.coefficient * coefficient.coefficient,
                    )
                )
    return _resolution_collected(chain.degree - 1, terms)


def _target_psi_chain(trace: TargetResolutionBasisTrace) -> SparseBarChain:
    return SparseBarChain(
        trace.degree,
        tuple(
            SparseBarTerm(
                term.left_element, term.group_tuple, term.coefficient
            )
            for term in trace.psi
        ),
    )


def _target_homotopy_chain(
    trace: TargetResolutionBasisTrace,
) -> SparseResolutionChain:
    return SparseResolutionChain(
        trace.degree + 1,
        tuple(
            SparseResolutionChainTerm(
                term.basis_id, term.element, term.coefficient
            )
            for term in trace.resolution_homotopy
        ),
    )


def _target_phi_chain(trace: TargetBarPhiTrace) -> SparseResolutionChain:
    return SparseResolutionChain(
        len(trace.group_tuple),
        tuple(
            SparseResolutionChainTerm(
                term.basis_id, term.element, term.coefficient
            )
            for term in trace.image
        ),
    )


def _target_bar_homotopy_chain(trace: TargetBarPhiTrace) -> SparseBarChain:
    return SparseBarChain(
        len(trace.group_tuple) + 1,
        tuple(
            SparseBarTerm(
                term.left_element, term.group_tuple, term.coefficient
            )
            for term in trace.bar_homotopy
        ),
    )


def _target_apply_phi(
    equivalence: TargetBarResolutionEquivalence,
    multiply,
    chain: SparseBarChain,
) -> SparseResolutionChain:
    lookup = {
        trace.group_tuple: _target_phi_chain(trace)
        for trace in equivalence.phi_traces
    }
    terms = []
    for term in chain.terms:
        image = lookup.get(term.group_tuple)
        if image is None:
            raise ValueError(f"target phi query is absent for {term.group_tuple!r}")
        acted = _target_resolution_action(multiply, term.left_element, image)
        terms.extend(
            SparseResolutionChainTerm(
                item.basis_id,
                item.element,
                term.coefficient * item.coefficient,
            )
            for item in acted.terms
        )
    return _resolution_collected(chain.degree, terms)


def _target_apply_basis_map(
    equivalence: TargetBarResolutionEquivalence,
    multiply,
    chain: SparseResolutionChain,
    *,
    homotopy: bool,
):
    lookup = {
        trace.basis_id: (
            _target_homotopy_chain(trace)
            if homotopy
            else _target_psi_chain(trace)
        )
        for trace in equivalence.basis_traces
    }
    if homotopy:
        terms_resolution = []
        for term in chain.terms:
            image = lookup.get(term.basis_id)
            if not isinstance(image, SparseResolutionChain):
                raise ValueError(f"target homotopy basis is absent for {term.basis_id}")
            acted = _target_resolution_action(multiply, term.element, image)
            terms_resolution.extend(
                SparseResolutionChainTerm(
                    item.basis_id,
                    item.element,
                    term.coefficient * item.coefficient,
                )
                for item in acted.terms
            )
        return _resolution_collected(chain.degree + 1, terms_resolution)
    terms_bar = []
    for term in chain.terms:
        image = lookup.get(term.basis_id)
        if not isinstance(image, SparseBarChain):
            raise ValueError(f"target psi basis is absent for {term.basis_id}")
        acted = _target_bar_action(multiply, term.element, image)
        terms_bar.extend(
            SparseBarTerm(
                item.left_element,
                item.group_tuple,
                term.coefficient * item.coefficient,
            )
            for item in acted.terms
        )
    return _bar_collected(chain.degree, terms_bar)


def _target_apply_bar_homotopy(
    equivalence: TargetBarResolutionEquivalence,
    multiply,
    chain: SparseBarChain,
) -> SparseBarChain:
    lookup = {
        trace.group_tuple: _target_bar_homotopy_chain(trace)
        for trace in equivalence.phi_traces
    }
    terms = []
    for term in chain.terms:
        image = lookup.get(term.group_tuple)
        if image is None:
            raise ValueError(
                f"target bar homotopy query is absent for {term.group_tuple!r}"
            )
        acted = _target_bar_action(multiply, term.left_element, image)
        terms.extend(
            SparseBarTerm(
                item.left_element,
                item.group_tuple,
                term.coefficient * item.coefficient,
            )
            for item in acted.terms
        )
    return _bar_collected(chain.degree + 1, terms)


def _target_query_closure(
    resolution: FreeResolutionCertificate,
    multiply,
    seeds: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    queries: set[tuple[str, ...]] = {()}
    pending = [tuple(seed) for seed in seeds]
    while pending:
        query = pending.pop()
        if "1" in query:
            continue
        for element in query:
            _normal_key(resolution, element)
        if query in queries:
            continue
        queries.add(query)
        if not query:
            continue
        pending.append(query[1:])
        pending.append(query[:-1])
        for index in range(len(query) - 1):
            merged = (
                query[:index]
                + (multiply(query[index], query[index + 1]),)
                + query[index + 2 :]
            )
            if "1" not in merged:
                pending.append(merged)
    return tuple(sorted(queries, key=lambda item: (len(item), item)))


def verify_target_bar_resolution_equivalence(
    equivalence: TargetBarResolutionEquivalence,
    resolution: FreeResolutionCertificate,
    inclusion_traces: Sequence[BarComparisonBasisTrace],
    authority: Task5VerificationAuthority,
) -> VerificationReport:
    if not isinstance(equivalence, TargetBarResolutionEquivalence):
        raise TypeError("target equivalence has the wrong type")
    if not isinstance(resolution, FreeResolutionCertificate):
        raise TypeError("target equivalence requires a target resolution")
    issues: list[VerificationIssue] = []
    report = verify_resolution(resolution, authority)
    issues.extend(
        VerificationIssue("target_resolution_invalid", issue.detail)
        for issue in report.issues
    )
    if equivalence.target_resolution_id != resolution.resolution_id:
        issues.append(
            VerificationIssue(
                "target_resolution_binding_mismatch",
                "target equivalence resolution ID differs from the embedded target",
            )
        )
    if (
        equivalence.phi_algorithm != "hap-1.70-bar-resolution-equivalence-phi"
        or equivalence.bar_homotopy_algorithm
        != "hap-1.70-bar-resolution-equivalence-equiv"
    ):
        issues.append(
            VerificationIssue(
                "target_bar_algorithm_mismatch",
                "target comparison must use pinned HAP 1.70 phi/psi/equiv",
            )
        )
    required_basis = {
        (
            trace.degree,
            int(term.basis_id.split(":", 1)[1]),
        )
        for trace in inclusion_traces
        for term in trace.target_phi
    }
    pending_basis = list(required_basis)
    while pending_basis:
        degree, column = pending_basis.pop()
        if degree == 0:
            continue
        boundary = resolution.boundaries[degree - 1]
        for entry in boundary.entries:
            candidate = (degree - 1, entry.row)
            if entry.column == column and candidate not in required_basis:
                required_basis.add(candidate)
                pending_basis.append(candidate)
    expected_basis = tuple(
        (degree, f"c{degree}:{basis}")
        for degree, basis in sorted(required_basis)
    )
    actual_basis = tuple(
        (trace.degree, trace.basis_id) for trace in equivalence.basis_traces
    )
    if actual_basis != expected_basis:
        issues.append(
            VerificationIssue(
                "target_bar_basis_coverage_mismatch",
                "target psi and resolution homotopy must cover every target basis",
            )
        )
    if equivalence.lookahead_boundary != resolution.lookahead_boundary:
        issues.append(
            VerificationIssue(
                "target_bar_lookahead_mismatch",
                "target equivalence does not bind the degree-five lookahead",
            )
        )
    from .cochains import target_bar_equivalence_digest

    if target_bar_equivalence_digest(equivalence) != equivalence.equivalence_id:
        issues.append(
            VerificationIssue(
                "target_bar_digest_mismatch",
                "target equivalence ID does not bind its full phi/psi/homotopy payload",
            )
        )
    try:
        multiply = _target_group_multiplier(resolution)
        seeds = [
            term.group_tuple
            for trace in inclusion_traces
            for term in trace.transported_bar
        ] + [
            term.group_tuple
            for trace in equivalence.basis_traces
            for term in trace.psi
        ]
        expected_queries = _target_query_closure(resolution, multiply, seeds)
    except (TypeError, ValueError) as error:
        issues.append(VerificationIssue("target_bar_query_invalid", str(error)))
        return VerificationReport(False, tuple(issues), 0)
    phi_queries = tuple(trace.group_tuple for trace in equivalence.phi_traces)
    if (
        equivalence.queried_bar_tuples != expected_queries
        or phi_queries != expected_queries
        or len(set(phi_queries)) != len(phi_queries)
    ):
        issues.append(
            VerificationIssue(
                "target_bar_query_coverage_mismatch",
                "target phi/homotopy domain must equal the exact inclusion-and-psi boundary closure",
            )
        )
    checked = 0
    identity = "1"
    try:
        for trace in equivalence.basis_traces:
            basis = SparseResolutionChain(
                trace.degree,
                (SparseResolutionChainTerm(trace.basis_id, identity, 1),),
            )
            psi = _target_psi_chain(trace)
            if trace.degree > 0:
                left = _target_bar_boundary(multiply, psi)
                right = _target_apply_basis_map(
                    equivalence,
                    multiply,
                    _target_resolution_boundary(
                        equivalence, resolution, multiply, basis
                    ),
                    homotopy=False,
                )
                checked += 1
                if left != right:
                    issues.append(
                        VerificationIssue(
                            "target_psi_not_chain_map", trace.basis_id
                        )
                    )
            phi_psi = _target_apply_phi(equivalence, multiply, psi)
            left = _resolution_add(basis, _resolution_scale(phi_psi, -1))
            homotopy = _target_homotopy_chain(trace)
            d_k = _target_resolution_boundary(
                equivalence, resolution, multiply, homotopy
            )
            if trace.degree:
                k_d = _target_apply_basis_map(
                    equivalence,
                    multiply,
                    _target_resolution_boundary(
                        equivalence, resolution, multiply, basis
                    ),
                    homotopy=True,
                )
            else:
                k_d = SparseResolutionChain(0, ())
            checked += 1
            if left != _resolution_add(d_k, k_d):
                issues.append(
                    VerificationIssue(
                        "target_resolution_homotopy_invalid", trace.basis_id
                    )
                )
        for query in equivalence.queried_bar_tuples:
            basis = SparseBarChain(
                len(query), (SparseBarTerm(identity, query, 1),)
            )
            phi = _target_apply_phi(equivalence, multiply, basis)
            if query:
                left = _target_resolution_boundary(
                    equivalence, resolution, multiply, phi
                )
                right = _target_apply_phi(
                    equivalence,
                    multiply,
                    _target_bar_boundary(multiply, basis),
                )
                checked += 1
                if left != right:
                    issues.append(
                        VerificationIssue("target_phi_not_chain_map", repr(query))
                    )
            else:
                checked += 1
            psi_phi = _target_apply_basis_map(
                equivalence, multiply, phi, homotopy=False
            )
            homotopy = _target_apply_bar_homotopy(
                equivalence, multiply, basis
            )
            d_h = _target_bar_boundary(multiply, homotopy)
            h_d = (
                _target_apply_bar_homotopy(
                    equivalence,
                    multiply,
                    _target_bar_boundary(multiply, basis),
                )
                if query
                else SparseBarChain(0, ())
            )
            checked += 1
            if _bar_add(basis, _bar_scale(psi_phi, -1)) != _bar_add(d_h, h_d):
                issues.append(
                    VerificationIssue("target_bar_homotopy_invalid", repr(query))
                )
        for trace in inclusion_traces:
            target_input = SparseBarChain(
                trace.degree,
                tuple(
                    SparseBarTerm(
                        term.left_element,
                        term.group_tuple,
                        term.coefficient,
                    )
                    for term in trace.transported_bar
                ),
            )
            expected_output = SparseResolutionChain(
                trace.degree,
                tuple(
                    SparseResolutionChainTerm(
                        term.basis_id, term.element, term.coefficient
                    )
                    for term in trace.target_phi
                ),
            )
            checked += 1
            if _target_apply_phi(equivalence, multiply, target_input) != expected_output:
                issues.append(
                    VerificationIssue(
                        "target_phi_trace_mismatch", trace.source_basis_id
                    )
                )
    except (TypeError, ValueError) as error:
        issues.append(VerificationIssue("target_bar_replay_failed", str(error)))
    expected_checked = (
        sum(trace.degree > 0 for trace in equivalence.basis_traces)
        + len(equivalence.basis_traces)
        + 2 * len(equivalence.queried_bar_tuples)
        + len(tuple(inclusion_traces))
    )
    if checked == 0 or checked != expected_checked:
        issues.append(
            VerificationIssue(
                "target_bar_check_count_mismatch",
                "target verifier did not replay every required identity",
            )
        )
    return VerificationReport(not issues, tuple(issues), checked)


def verify_bar_resolution_equivalence(
    equivalence: BarResolutionEquivalence,
    authority: Task5VerificationAuthority,
    *,
    queries: Sequence[Sequence[str]] | None = None,
) -> VerificationReport:
    if not isinstance(equivalence, BarResolutionEquivalence):
        raise TypeError("equivalence must be a BarResolutionEquivalence")
    issues: list[VerificationIssue] = []
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("authority must be a caller-supplied Task5VerificationAuthority")
    resolution_report = verify_resolution(equivalence.resolution, authority)
    issues.extend(VerificationIssue("resolution_invalid", issue.detail) for issue in resolution_report.issues)
    if (
        equivalence.phi_algorithm
        != "hap-1.70-bar-resolution-equivalence-phi"
        or equivalence.bar_homotopy_algorithm
        != "hap-1.70-bar-resolution-equivalence-equiv"
    ):
        issues.append(VerificationIssue(
            "bar_algorithm_mismatch",
            "bar comparison must use the pinned HAP 1.70 phi/psi/equiv algorithms",
        ))
    expected_basis = tuple(
        (degree, basis_id)
        for degree, basis in enumerate(equivalence.resolution.basis)
        for basis_id in basis
    )
    psi_basis = tuple((item.degree, item.basis_id) for item in equivalence.psi_on_basis)
    homotopy_basis = tuple(
        (item.degree, item.basis_id)
        for item in equivalence.resolution_homotopy_on_basis
    )
    if psi_basis != expected_basis or homotopy_basis != expected_basis:
        issues.append(VerificationIssue(
            "bar_basis_coverage_mismatch",
            "psi and the resolution homotopy must cover each canonical basis exactly once",
        ))
    for item in equivalence.psi_on_basis:
        if not isinstance(item.image, SparseBarChain) or item.image.degree != item.degree:
            issues.append(VerificationIssue(
                "bar_basis_type_mismatch", f"psi image {item.basis_id} has the wrong degree or type"
            ))
    for item in equivalence.resolution_homotopy_on_basis:
        if not isinstance(item.image, SparseResolutionChain) or item.image.degree != item.degree + 1:
            issues.append(VerificationIssue(
                "bar_basis_type_mismatch",
                f"resolution homotopy image {item.basis_id} has the wrong degree or type",
            ))
    phi_queries = tuple(item.group_tuple for item in equivalence.phi_on_queries)
    queried = equivalence.queried_bar_tuples
    required_query_prefix = (
        ((),)
        + equivalence.normalized_tuples(1)
        + equivalence.normalized_tuples(2)
    )
    table = equivalence.finite_group
    witness: tuple[tuple[str, ...], ...] = ()
    for left in range(len(table.element_order)):
        for right in range(len(table.element_order)):
            if table.multiplication_table[left][right] != table.multiplication_table[right][left]:
                witness = ((table.element_order[right], table.element_order[left], table.element_order[right]),)
                break
        if witness:
            break
    expected_queries = required_query_prefix + witness
    required_phi = list(expected_queries)
    for image in equivalence.psi_on_basis:
        if not isinstance(image.image, SparseBarChain):
            continue
        for term in image.image.terms:
            if term.group_tuple not in required_phi:
                required_phi.append(term.group_tuple)
    finite_elements = set(table.element_order)
    query_elements_are_finite = all(
        element in finite_elements
        for query in queried + phi_queries
        for element in query
    )
    if (
        not queried
        or len(set(queried)) != len(queried)
        or queried != expected_queries
        or len(phi_queries) != len(required_phi)
        or set(phi_queries) != set(required_phi)
        or not query_elements_are_finite
        or any(
            next(item for item in equivalence.phi_on_queries if item.group_tuple == query).bar_homotopy
            is None
            for query in queried
        )
    ):
        issues.append(VerificationIssue(
            "bar_query_domain_mismatch",
            "queries must be the exact finite-group domain and phi must add exactly the psi tuples",
        ))
        issues.append(VerificationIssue(
            "bar_query_coverage_mismatch",
            "certified nonempty queries must have exact phi and bar-homotopy traces",
        ))
    for item in equivalence.phi_on_queries:
        if not isinstance(item.image, SparseResolutionChain) or item.image.degree != len(
            item.group_tuple
        ):
            issues.append(VerificationIssue(
                "bar_query_type_mismatch",
                f"phi image {item.group_tuple!r} has the wrong degree or type",
            ))
        if item.bar_homotopy is not None and (
            not isinstance(item.bar_homotopy, SparseBarChain)
            or item.bar_homotopy.degree != len(item.group_tuple) + 1
        ):
            issues.append(VerificationIssue(
                "bar_query_type_mismatch",
                f"bar homotopy {item.group_tuple!r} has the wrong degree or type",
            ))
    expected_lookahead_shape = (
        len(equivalence.resolution.basis[4]),
        len(equivalence.resolution.degree_five_basis),
    )
    if (
        equivalence.lookahead_boundary.row_count,
        equivalence.lookahead_boundary.column_count,
    ) != expected_lookahead_shape or (
        equivalence.lookahead_boundary.row_count > 0
        and equivalence.lookahead_boundary.column_count == 0
    ):
        issues.append(VerificationIssue(
            "lookahead_shape_mismatch",
            "degree-five lookahead must start at the complete degree-four basis",
        ))
    elif equivalence.lookahead_boundary != equivalence.resolution.lookahead_boundary:
        issues.append(VerificationIssue(
            "lookahead_binding_mismatch",
            "bar comparison lookahead differs from its bound resolution degree five",
        ))
    else:
        try:
            if _compose(
                equivalence.resolution.boundaries[3],
                equivalence.lookahead_boundary,
                equivalence.resolution,
            ):
                issues.append(VerificationIssue(
                    "boundary_not_square_zero", "degree 5: nonzero sparse residue"
                ))
        except (TypeError, ValueError) as error:
            issues.append(VerificationIssue("boundary_not_square_zero", f"degree 5: {error}"))
    if bar_equivalence_digest(equivalence) != equivalence.equivalence_id:
        issues.append(VerificationIssue("bar_equivalence_digest_mismatch", "equivalence ID does not bind payload"))
    checked = 0
    identity = equivalence.finite_group.element_order[0]
    # psi is a chain map on every stored finite-resolution basis.
    for item in equivalence.psi_on_basis:
        if item.degree == 0:
            continue
        basis = SparseResolutionChain(item.degree, (SparseResolutionChainTerm(item.basis_id, identity, 1),))
        try:
            left = _bar_boundary(equivalence.finite_group, item.image)
            right = _apply_basis_map(equivalence, _resolution_boundary(equivalence, basis), equivalence.psi_on_basis)
            checked += 1
            if left != right:
                issues.append(VerificationIssue("psi_not_chain_map", f"basis {item.basis_id} fails"))
        except ValueError as error:
            issues.append(VerificationIssue("psi_not_chain_map", str(error)))
    # id - phi psi = dK + Kd on every stored basis.
    for item in equivalence.resolution_homotopy_on_basis:
        basis = SparseResolutionChain(item.degree, (SparseResolutionChainTerm(item.basis_id, identity, 1),))
        try:
            psi_basis = _apply_basis_map(equivalence, basis, equivalence.psi_on_basis)
            phi_psi = _apply_phi(equivalence, psi_basis)
            left = _resolution_add(basis, _resolution_scale(phi_psi, -1))
            d_k = _resolution_boundary(equivalence, item.image)
            if item.degree == 0:
                k_d = SparseResolutionChain(0, ())
            else:
                k_d = _apply_basis_map(equivalence, _resolution_boundary(equivalence, basis), equivalence.resolution_homotopy_on_basis)
            right = _resolution_add(d_k, k_d)
            checked += 1
            if left != right:
                issues.append(VerificationIssue("resolution_homotopy_invalid", f"basis {item.basis_id} fails"))
        except ValueError as error:
            issues.append(VerificationIssue("resolution_homotopy_invalid", str(error)))
    selected = tuple(
        tuple(query)
        for query in (
            equivalence.queried_bar_tuples if queries is None else queries
        )
    )
    if selected != equivalence.queried_bar_tuples:
        issues.append(VerificationIssue(
            "bar_query_coverage_mismatch",
            "verification must replay the exact certified query domain",
        ))
    for group_tuple in selected:
        if any(element == identity for element in group_tuple):
            issues.append(VerificationIssue("bar_tuple_not_normalized", repr(group_tuple)))
            continue
        basis = SparseBarChain(len(group_tuple), (SparseBarTerm(identity, group_tuple, 1),))
        try:
            phi_basis = _apply_phi(equivalence, basis)
            if group_tuple:
                left = _resolution_boundary(equivalence, phi_basis)
                right = _apply_phi(equivalence, _bar_boundary(equivalence.finite_group, basis))
                checked += 1
                if left != right:
                    issues.append(VerificationIssue("phi_not_chain_map", repr(group_tuple)))
            else:
                checked += 1
            psi_phi = _apply_basis_map(equivalence, phi_basis, equivalence.psi_on_basis)
            homotopy = _apply_bar_homotopy(equivalence, basis)
            d_h = _bar_boundary(equivalence.finite_group, homotopy)
            if group_tuple:
                h_d = _apply_bar_homotopy(equivalence, _bar_boundary(equivalence.finite_group, basis))
            else:
                h_d = SparseBarChain(0, ())
            checked += 1
            if _bar_add(basis, _bar_scale(psi_phi, -1)) != _bar_add(d_h, h_d):
                issues.append(VerificationIssue("bar_homotopy_invalid", repr(group_tuple)))
        except ValueError as error:
            code = "phi_not_chain_map" if "phi" in str(error) else "bar_homotopy_invalid"
            issues.append(VerificationIssue(code, str(error)))
    expected_checked = (
        sum(item.degree > 0 for item in equivalence.psi_on_basis)
        + len(equivalence.resolution_homotopy_on_basis)
        + 2 * len(selected)
    )
    if checked == 0 or checked != expected_checked:
        issues.append(VerificationIssue(
            "bar_check_count_mismatch",
            "the verifier did not replay every expected nonvacuous identity",
        ))
    return VerificationReport(not issues, tuple(issues), checked)


def _coefficient(value: object) -> Fraction:
    if isinstance(value, Phase):
        return value.value
    if type(value) is int or isinstance(value, Fraction):
        return Fraction(value)
    raise TypeError("bar cochain coefficients must be exact integers, fractions, or phases")


def _coefficient_character(
    equivalence: BarResolutionEquivalence,
    character: GF2Character | None,
) -> GF2Character:
    table = equivalence.finite_group
    result = GF2Character((0,) * len(table.element_order)) if character is None else character
    if not isinstance(result, GF2Character) or len(result.bits) != len(table.element_order):
        raise ValueError("coefficient character must be in finite-table element order")
    for left, row in enumerate(table.multiplication_table):
        for right, product in enumerate(row):
            if result.bits[product] != (result.bits[left] ^ result.bits[right]):
                raise ValueError("coefficient character is not a finite-group homomorphism")
    return result


def _action_sign(
    equivalence: BarResolutionEquivalence,
    character: GF2Character,
    element: str,
) -> int:
    return -1 if character.bits[_index(equivalence.finite_group, element)] else 1


def evaluate_bar_cochain(
    equivalence: BarResolutionEquivalence,
    coordinates: Sequence[object],
    group_tuple: Sequence[str],
    *,
    coefficient_character: GF2Character | None = None,
) -> Fraction:
    group_tuple = tuple(group_tuple)
    chain = _apply_phi(
        equivalence,
        SparseBarChain(len(group_tuple), (SparseBarTerm("1", group_tuple, 1),)),
    )
    values = tuple(_coefficient(item) for item in coordinates)
    expected = len(equivalence.resolution.basis[len(group_tuple)])
    if len(values) != expected:
        raise ValueError("cochain coordinate length differs from resolution rank")
    character = _coefficient_character(equivalence, coefficient_character)
    return sum(
        (
            Fraction(term.coefficient)
            * _action_sign(equivalence, character, term.element)
            * values[int(term.basis_id.split(":", 1)[1])]
            for term in chain.terms
        ),
        Fraction(0),
    )


def _cocycle_digest(cocycle: Mapping[tuple[str, ...], Fraction]) -> str:
    return _domain_digest(
        "normalized-bar-cocycle",
        [[list(key), str(value)] for key, value in sorted(cocycle.items())],
    )


def _bar_chain_cochain_value(
    equivalence: BarResolutionEquivalence,
    cocycle: Mapping[tuple[str, ...], Fraction],
    chain: SparseBarChain,
    character: GF2Character,
) -> Fraction:
    return sum(
        (
            Fraction(term.coefficient)
            * _action_sign(equivalence, character, term.left_element)
            * cocycle.get(term.group_tuple, Fraction(0))
            for term in chain.terms
        ),
        Fraction(0),
    )


def _one_cochain_value(
    equivalence: BarResolutionEquivalence,
    values: Sequence[Fraction],
    element: str,
) -> Fraction:
    if element == equivalence.finite_group.element_order[0]:
        return Fraction(0)
    return values[_index(equivalence.finite_group, element) - 1]


def _coboundary_one(
    equivalence: BarResolutionEquivalence,
    values: Sequence[Fraction],
    pair: tuple[str, str],
    character: GF2Character,
) -> Fraction:
    left, right = pair
    return (
        _action_sign(equivalence, character, left)
        * _one_cochain_value(equivalence, values, right)
        - _one_cochain_value(equivalence, values, _multiply(equivalence.finite_group, left, right))
        + _one_cochain_value(equivalence, values, left)
    )


def _equal_coefficient(left: Fraction, right: Fraction, mod_one: bool) -> bool:
    return Phase(left) == Phase(right) if mod_one else left == right


def _verify_bar_two_cocycle(
    equivalence: BarResolutionEquivalence,
    cocycle: Mapping[tuple[str, ...], Fraction],
    character: GF2Character,
    mod_one: bool,
) -> bool:
    table = equivalence.finite_group
    for left, middle, right in equivalence.normalized_tuples(3):
        value = (
            _action_sign(equivalence, character, left)
            * cocycle.get((middle, right), Fraction(0))
            - cocycle.get((_multiply(table, left, middle), right), Fraction(0))
            + cocycle.get((left, _multiply(table, middle, right)), Fraction(0))
            - cocycle.get((left, middle), Fraction(0))
        )
        if not _equal_coefficient(value, Fraction(0), mod_one):
            return False
    return True


def coordinate_bar_cocycle(
    equivalence: BarResolutionEquivalence,
    cocycle: Mapping[Sequence[str], object],
    *,
    coefficient_character: GF2Character | None = None,
) -> CochainCoordinateCertificate:
    mod_one = any(isinstance(value, Phase) for value in cocycle.values())
    normalized = {tuple(key): _coefficient(value) for key, value in cocycle.items()}
    degrees = {len(key) for key in normalized}
    if len(degrees) != 1:
        raise ValueError("bar cocycle must have one homogeneous degree")
    degree = degrees.pop()
    if degree != 2:
        raise ValueError("coordinate conversion v1 requires a normalized bar 2-cocycle")
    expected_tuples = equivalence.normalized_tuples(degree)
    if set(normalized) != set(expected_tuples):
        raise ValueError("bar cocycle must supply every normalized tuple exactly once")
    character = _coefficient_character(equivalence, coefficient_character)
    if not _verify_bar_two_cocycle(equivalence, normalized, character, mod_one):
        raise ValueError("normalized bar input is not a 2-cocycle")
    psi_degree_two = {
        item.basis_id: item.image
        for item in equivalence.psi_on_basis
        if item.degree == 2 and isinstance(item.image, SparseBarChain)
    }
    expected_basis = equivalence.resolution.basis[2]
    if set(psi_degree_two) != set(expected_basis):
        raise ValueError("psi is absent on a degree-two resolution basis")
    # Pull the bar cocycle back along psi.  This is the canonical finite-
    # resolution coordinate map; it does not infer coordinates from a rank
    # or cohomology summary.
    coordinates = tuple(
        _bar_chain_cochain_value(
            equivalence, normalized, psi_degree_two[basis_id], character
        )
        for basis_id in expected_basis
    )
    phi_values = {item.group_tuple: item for item in equivalence.phi_on_queries}
    coboundary = []
    for element in equivalence.finite_group.element_order[1:]:
        trace = phi_values.get((element,))
        if trace is None or trace.bar_homotopy is None:
            raise ValueError(f"bar homotopy query is absent for {(element,)!r}")
        # With id-psi*phi=dH+Hd and dc=0, c-eval(phi^*psi^*c)=d(cH).
        coboundary.append(
            _bar_chain_cochain_value(
                equivalence, normalized, trace.bar_homotopy, character
            )
        )
    coboundary_1cochain = tuple(coboundary)
    source_digest = _cocycle_digest(normalized)
    core = {
        "coboundary_1cochain": [str(item) for item in coboundary_1cochain],
        "coefficient_character": list(character.bits),
        "coordinates": [str(item) for item in coordinates],
        "degree": degree,
        "mod_one": mod_one,
        "resolution_id": equivalence.resolution_id,
        "source_cocycle_digest": source_digest,
    }
    return CochainCoordinateCertificate(
        equivalence.resolution_id,
        degree,
        coordinates,
        coboundary_1cochain,
        character,
        mod_one,
        source_digest,
        _domain_digest("cochain-coordinate-certificate", core),
    )


def verify_cochain_coordinate_certificate(
    equivalence: BarResolutionEquivalence,
    cocycle: Mapping[Sequence[str], object],
    certificate: CochainCoordinateCertificate,
) -> VerificationReport:
    issues: list[VerificationIssue] = []
    mod_one = any(isinstance(value, Phase) for value in cocycle.values())
    normalized = {tuple(key): _coefficient(value) for key, value in cocycle.items()}
    if certificate.resolution_id != equivalence.resolution_id or certificate.source_cocycle_digest != _cocycle_digest(normalized):
        issues.append(VerificationIssue("coordinate_binding_mismatch", "coordinate certificate does not bind resolution and cocycle"))
    core = {
        "coboundary_1cochain": [str(item) for item in certificate.coboundary_1cochain],
        "coefficient_character": list(certificate.coefficient_character.bits),
        "coordinates": [str(item) for item in certificate.coordinates],
        "degree": certificate.degree,
        "mod_one": certificate.mod_one,
        "resolution_id": certificate.resolution_id,
        "source_cocycle_digest": certificate.source_cocycle_digest,
    }
    if _domain_digest("cochain-coordinate-certificate", core) != certificate.certificate_id:
        issues.append(VerificationIssue("coordinate_digest_mismatch", "coordinate certificate ID fails"))
    try:
        character = _coefficient_character(
            equivalence, certificate.coefficient_character
        )
    except ValueError as error:
        issues.append(VerificationIssue("coordinate_roundtrip_failed", str(error)))
        character = GF2Character((0,) * len(equivalence.finite_group.element_order))
    if certificate.degree != 2 or certificate.mod_one != mod_one:
        issues.append(VerificationIssue("coordinate_roundtrip_failed", "degree or coefficient quotient differs"))
    if len(certificate.coboundary_1cochain) != len(equivalence.finite_group.element_order) - 1:
        issues.append(VerificationIssue("coordinate_roundtrip_failed", "1-cochain has wrong dimension"))
    if set(normalized) != set(equivalence.normalized_tuples(2)):
        issues.append(VerificationIssue("coordinate_roundtrip_failed", "normalized pair domain is incomplete"))
    elif not _verify_bar_two_cocycle(equivalence, normalized, character, mod_one):
        issues.append(VerificationIssue("coordinate_roundtrip_failed", "source is not a 2-cocycle"))
    for group_tuple, expected in normalized.items():
        try:
            actual = evaluate_bar_cochain(
                equivalence,
                certificate.coordinates,
                group_tuple,
                coefficient_character=character,
            )
            actual += _coboundary_one(
                equivalence,
                certificate.coboundary_1cochain,
                group_tuple,
                character,
            )
        except ValueError as error:
            issues.append(VerificationIssue("coordinate_roundtrip_failed", str(error)))
            break
        if not _equal_coefficient(actual, expected, mod_one):
            issues.append(VerificationIssue("coordinate_roundtrip_failed", repr(group_tuple)))
            break
    return VerificationReport(not issues, tuple(issues), len(normalized))


__all__ = [
    "BarPhiValue",
    "BarResolutionEquivalence",
    "CochainCoordinateCertificate",
    "GapBatchExecutionProvenance",
    "GapBatchLauncherExecution",
    "GapBatchMemberProvenance",
    "GapInclusionBatchMember",
    "GapInclusionBatchSpec",
    "GapLauncherExecution",
    "ResolutionBasisImage",
    "SparseBarChain",
    "SparseBarTerm",
    "SparseResolutionChain",
    "SparseResolutionChainTerm",
    "assemble_gap_bar_resolution_equivalence",
    "assemble_gap_inclusion_fixture",
    "assemble_gap_target_bar_resolution_equivalence",
    "bar_equivalence_digest",
    "bar_equivalence_mapping",
    "build_gap_inclusion_batch_export_program",
    "build_gap_inclusion_export_program",
    "coordinate_bar_cocycle",
    "dumps_bar_resolution_equivalence",
    "dumps_gap_inclusion_fixture",
    "evaluate_bar_cochain",
    "export_gap_inclusion_batch_raw",
    "export_gap_inclusion_raw",
    "gap_batch_execution_provenance_mapping",
    "loads_bar_resolution_equivalence",
    "make_gap_inclusion_batch_spec",
    "make_bar_resolution_equivalence",
    "make_target_bar_resolution_equivalence",
    "run_gap_inclusion_export",
    "restore_diagnostic_gap_batch_execution",
    "verify_gap_batch_launcher_execution",
    "verify_gap_batch_member_execution",
    "verify_gap_batch_execution_provenance",
    "verify_gap_inclusion_batch_spec",
    "verify_gap_launcher_execution",
    "verify_bar_resolution_equivalence",
    "verify_target_bar_resolution_equivalence",
    "verify_cochain_coordinate_certificate",
]
