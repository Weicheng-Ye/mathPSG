r"""Replayable low-degree group-ring and character certificates.

The GAP worker is an untrusted producer.  This module parses its sparse
matrices, replays group multiplication using either a finite multiplication
table or the exact affine--PCP authority certified by :mod:`gap_classifier`,
and checks every displayed chain identity before exposing integer cochains.
No cohomology-group summary is accepted as a substitute for these matrices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, replace
import hashlib
import itertools
import json
import re
from typing import Any

from ._resources import asset_bytes
from .gap_classifier import (
    AffinePCPIsomorphismCertificate,
    _certificate_mapping,
    _affine_exact,
    _compose_affine,
    _evaluate_pcp_word,
    _normal_form_decoder,
    _locked_environment_core,
    _parse_certificate,
    _pcp_word_coordinates,
    _power_affine,
    affine_pcp_certificate_digest,
    literal_element_authority_digest,
    make_certified_space_group_action,
    transported_inclusion_authority_digest,
)
from .gf2 import GF2Character, MatrixGF2, kernel_basis, solve_affine
from .integer_linalg import MatrixZ


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_BASIS_RE = re.compile(r"c([0-9]+):([0-9]+)\Z")
_PROTOCOL_PREFIX = b"mathpsg-cochains-v1|"
_TASK5_PROTOCOL_PREFIX = b"mathpsg-gap-classifier-v1|"
_MAX_CERTIFICATE_BYTES = 4 * 1024 * 1024
_PATH_LEAK_RE = re.compile(r"(?:/(?:Users|home|tmp|private|var)/|[A-Za-z]:[\\/])")
_OBSERVED_NONCOMMUTING_OUTCOME_PREFIX = "observed-noncommuting:"
_FROZEN_P4MM_DIAGNOSTIC_OUTCOME = "known-noncommuting-p4mm-hap-1.70"
_INCLUSION_CERTIFICATE_CACHE: dict[
    tuple[bytes, "Task5VerificationAuthority", bool],
    "InclusionChainMapCertificate",
] = {}


def _looks_like_path(value: str) -> bool:
    """Reject host/path syntax without mistaking canonical rationals for paths."""

    if value.startswith(("/", "\\")) or _PATH_LEAK_RE.search(value):
        return True
    if "\\" in value:
        return True
    return any(component in (".", "..") for component in value.split("/"))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_digest(domain: str, payload: object) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL_PREFIX + domain.encode("ascii") + b"|" + _canonical_json(payload)
    ).hexdigest()


def _task5_domain_digest(domain: str, payload: object) -> str:
    return "sha256:" + hashlib.sha256(
        _TASK5_PROTOCOL_PREFIX
        + domain.encode("ascii")
        + b"|"
        + _canonical_json(payload)
    ).hexdigest()


_TASK5_GAP_SOURCE_NAMES = (
    "bar_equivalence.g",
    "characters.g",
    "resolutions.g",
    "restrictions.g",
    "u1_relative.g",
)


def _task5_backend_binding() -> tuple[str, str, str]:
    locked = _locked_environment_core()
    lock_digest = locked["lock_digest"]
    source_hashes = [
        {
            "path": name,
            "sha256": hashlib.sha256(
                asset_bytes("gap/classifier/lib/" + name)
            ).hexdigest(),
        }
        for name in _TASK5_GAP_SOURCE_NAMES
    ]
    declared = locked.get("task5_source_closure")
    if declared != source_hashes:
        raise ValueError("tracked Task 5 GAP source closure differs from the lock")
    required_apis = (
        "BarResolutionEquivalence",
        "EquivariantChainMap",
        "ResolutionAlmostCrystalGroup",
        "ResolutionDirectProduct",
        "ResolutionFiniteGroup",
    )
    if tuple(locked.get("api_closure", ())) != required_apis:
        raise ValueError("tracked Task 5 HAP API closure differs from the lock")
    runtime_provenance_digest = _task5_domain_digest(
        "task5-runtime-provenance-v1",
        {
            "api_closure": list(required_apis),
            "lock_digest": lock_digest,
            "source_closure": source_hashes,
        },
    )
    environment_id = _task5_domain_digest(
        "task5-backend-environment-v1",
        {
            "algorithms": list(required_apis),
            "lock_digest": lock_digest,
            "oci_image_digest": locked["oci_image_digest"],
            "packages": locked["packages"],
            "runtime_provenance_digest": runtime_provenance_digest,
        },
    )
    return lock_digest, environment_id, runtime_provenance_digest


def _strict_json(data: bytes) -> Mapping[str, Any]:
    if type(data) is not bytes:
        raise TypeError("certificate loader requires bytes")
    if len(data) > _MAX_CERTIFICATE_BYTES:
        raise ValueError("certificate exceeds the canonical size bound")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_float=lambda token: (_ for _ in ()).throw(
                ValueError("floating-point JSON is forbidden")
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON is forbidden")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict certificate JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("certificate JSON must be an object")
    encoded = _canonical_json(value)
    if data != encoded:
        raise ValueError("certificate bytes are not canonical JSON")

    def reject_paths(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if _looks_like_path(key):
                    raise ValueError(f"{path}: host path leak is forbidden")
                reject_paths(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                reject_paths(nested, f"{path}[{index}]")
        elif isinstance(item, str) and _looks_like_path(item):
            raise ValueError(f"{path}: host path leak is forbidden")

    reject_paths(value, "$")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{path}: missing field {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"{path}: unexpected field {sorted(extra)[0]}")


def _identifier(value: object, path: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: invalid identifier")
    return value


def _digest(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256 digest")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path}: expected integer")
    return value


def _canonical_diagnostic_failure_degrees(
    values: Sequence[int], path: str
) -> tuple[int, ...]:
    failures = tuple(values)
    if (
        failures != tuple(sorted(set(failures)))
        or any(type(degree) is not int or degree not in range(1, 5) for degree in failures)
    ):
        raise ValueError(
            f"{path}: expected canonical subset of degrees one through four"
        )
    return failures


def _canonical_diagnostic_residue_digests(
    values: Sequence[str], path: str
) -> tuple[str, ...]:
    residues = tuple(values)
    if len(residues) != 4:
        raise ValueError(f"{path}: expected four digests")
    for index, digest in enumerate(residues):
        _digest(digest, f"{path}[{index}]")
    return residues


def task5_diagnostic_observed_outcome(
    diagnostic_backend: str,
    diagnostic_failure_degrees: Sequence[int],
    diagnostic_residue_digests: Sequence[str],
) -> str:
    """Name one independently replayable noncommuting diagnostic observation."""

    backend = _identifier(
        diagnostic_backend, "$task5_diagnostic_observed_outcome.diagnostic_backend"
    )
    failures = _canonical_diagnostic_failure_degrees(
        diagnostic_failure_degrees,
        "$task5_diagnostic_observed_outcome.diagnostic_failure_degrees",
    )
    if not failures:
        raise ValueError(
            "$task5_diagnostic_observed_outcome: noncommuting outcome requires failure degrees"
        )
    residues = _canonical_diagnostic_residue_digests(
        diagnostic_residue_digests,
        "$task5_diagnostic_observed_outcome.diagnostic_residue_digests",
    )
    return _OBSERVED_NONCOMMUTING_OUTCOME_PREFIX + _task5_domain_digest(
        "task5-diagnostic-observation-v1",
        {
            "diagnostic_backend": backend,
            "diagnostic_failure_degrees": list(failures),
            "diagnostic_residue_digests": list(residues),
        },
    )


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    detail: str

    def __post_init__(self) -> None:
        _identifier(self.code, "$VerificationIssue.code")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("$VerificationIssue.detail: expected nonempty text")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    issues: tuple[VerificationIssue, ...] = ()
    checked_identities: int = 0

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise TypeError("$VerificationReport.valid: expected boolean")
        issues = tuple(self.issues)
        if any(not isinstance(issue, VerificationIssue) for issue in issues):
            raise TypeError("$VerificationReport.issues: expected issue tuple")
        checked = _integer(self.checked_identities, "$VerificationReport.checked_identities")
        if checked < 0:
            raise ValueError("$VerificationReport.checked_identities: expected nonnegative")
        if self.valid != (not issues):
            raise ValueError("$VerificationReport.valid: inconsistent with issues")
        object.__setattr__(self, "issues", issues)


@dataclass(frozen=True, slots=True)
class FiniteGroupTable:
    group_id: str
    element_order: tuple[str, ...]
    identity_index: int
    multiplication_table: tuple[tuple[int, ...], ...]
    inverse_indices: tuple[int, ...]
    table_digest: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.group_id, "$FiniteGroupTable.group_id")
        elements = tuple(self.element_order)
        if not elements or len(set(elements)) != len(elements):
            raise ValueError("$FiniteGroupTable.element_order: expected unique nonempty tuple")
        for index, element in enumerate(elements):
            _identifier(element, f"$FiniteGroupTable.element_order[{index}]")
        identity = _integer(self.identity_index, "$FiniteGroupTable.identity_index")
        if identity != 0 or elements[0] != "1":
            raise ValueError("$FiniteGroupTable: canonical identity must be element 0 named 1")
        table = tuple(tuple(row) for row in self.multiplication_table)
        inverses = tuple(self.inverse_indices)
        order = len(elements)
        if len(table) != order or any(len(row) != order for row in table):
            raise ValueError("$FiniteGroupTable.multiplication_table: expected full square table")
        if len(inverses) != order:
            raise ValueError("$FiniteGroupTable.inverse_indices: expected one inverse per element")
        for row in table:
            for product in row:
                if type(product) is not int or not 0 <= product < order:
                    raise ValueError("$FiniteGroupTable.multiplication_table: invalid product index")
        if any(table[identity][i] != i or table[i][identity] != i for i in range(order)):
            raise ValueError("$FiniteGroupTable: identity row or column is invalid")
        for left in range(order):
            inverse = inverses[left]
            if type(inverse) is not int or not 0 <= inverse < order:
                raise ValueError("$FiniteGroupTable.inverse_indices: invalid index")
            if table[left][inverse] != identity or table[inverse][left] != identity:
                raise ValueError("$FiniteGroupTable.inverse_indices: inverse witness fails")
            for middle in range(order):
                for right in range(order):
                    if table[table[left][middle]][right] != table[left][table[middle][right]]:
                        raise ValueError("$FiniteGroupTable.multiplication_table: associativity fails")
        core = {
            "element_order": list(elements),
            "group_id": self.group_id,
            "identity_index": identity,
            "inverse_indices": list(inverses),
            "multiplication_table": [list(row) for row in table],
        }
        expected = _domain_digest("finite-group-table", core)
        if self.table_digest is not None and self.table_digest != expected:
            raise ValueError("$FiniteGroupTable.table_digest: does not bind table")
        object.__setattr__(self, "element_order", elements)
        object.__setattr__(self, "multiplication_table", table)
        object.__setattr__(self, "inverse_indices", inverses)
        object.__setattr__(self, "table_digest", expected)


@dataclass(frozen=True, slots=True, order=True)
class SparseGroupRingTerm:
    element: str
    coefficient: int

    def __post_init__(self) -> None:
        if type(self.element) is not str or not self.element:
            raise ValueError("$SparseGroupRingTerm.element: expected nonempty normal form")
        coefficient = _integer(self.coefficient, "$SparseGroupRingTerm.coefficient")
        if coefficient == 0:
            raise ValueError("$SparseGroupRingTerm.coefficient: zero terms must be omitted")


@dataclass(frozen=True, slots=True)
class SparseGroupRingEntry:
    row: int
    column: int
    terms: tuple[SparseGroupRingTerm, ...]

    def __post_init__(self) -> None:
        row = _integer(self.row, "$SparseGroupRingEntry.row")
        column = _integer(self.column, "$SparseGroupRingEntry.column")
        if row < 0 or column < 0:
            raise ValueError("$SparseGroupRingEntry: indices must be nonnegative")
        terms = tuple(self.terms)
        if not terms or any(not isinstance(term, SparseGroupRingTerm) for term in terms):
            raise ValueError("$SparseGroupRingEntry.terms: expected nonempty term tuple")
        if tuple(term.element for term in terms) != tuple(
            sorted(term.element for term in terms)
        ) or len({term.element for term in terms}) != len(terms):
            raise ValueError("$SparseGroupRingEntry.terms: expected canonical term order")
        object.__setattr__(self, "terms", terms)


@dataclass(frozen=True, slots=True)
class SparseGroupRingMatrix:
    row_count: int
    column_count: int
    entries: tuple[SparseGroupRingEntry, ...]

    def __post_init__(self) -> None:
        rows = _integer(self.row_count, "$SparseGroupRingMatrix.row_count")
        columns = _integer(self.column_count, "$SparseGroupRingMatrix.column_count")
        if rows < 0 or columns < 0:
            raise ValueError("$SparseGroupRingMatrix: dimensions must be nonnegative")
        entries = tuple(self.entries)
        if any(not isinstance(entry, SparseGroupRingEntry) for entry in entries):
            raise TypeError("$SparseGroupRingMatrix.entries: expected entry tuple")
        positions = tuple((entry.row, entry.column) for entry in entries)
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
            raise ValueError("$SparseGroupRingMatrix.entries: expected canonical sparse order")
        if any(entry.row >= rows or entry.column >= columns for entry in entries):
            raise ValueError("$SparseGroupRingMatrix.entries: entry outside matrix shape")
        object.__setattr__(self, "entries", entries)


_LOCKED_LAUNCHER_ATTESTATION_SEAL = object()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LauncherExecutionAttestation:
    request_input_digest: str
    raw_output_digest: str
    gap_inclusion_projection_digest: str
    process_stdout_digest: str
    process_stderr_digest: str
    resolved_launcher_digest: str
    backend_observation_digest: str
    runtime_manifest_digest: str | None
    exit_status: int
    release_certified: bool
    attestation_id: str
    _release_seal: InitVar[object | None] = None

    def __post_init__(self, _release_seal: object | None) -> None:
        for field_name in (
            "request_input_digest",
            "raw_output_digest",
            "gap_inclusion_projection_digest",
            "process_stdout_digest",
            "process_stderr_digest",
            "resolved_launcher_digest",
            "backend_observation_digest",
        ):
            _digest(
                getattr(self, field_name),
                f"$LauncherExecutionAttestation.{field_name}",
            )
        if self.runtime_manifest_digest is not None:
            _digest(
                self.runtime_manifest_digest,
                "$LauncherExecutionAttestation.runtime_manifest_digest",
            )
        if type(self.exit_status) is not int:
            raise TypeError("$LauncherExecutionAttestation.exit_status: expected integer")
        if type(self.release_certified) is not bool:
            raise TypeError(
                "$LauncherExecutionAttestation.release_certified: expected boolean"
            )
        if self.release_certified != (self.runtime_manifest_digest is not None):
            raise ValueError(
                "$LauncherExecutionAttestation: release flag and runtime manifest differ"
            )
        if (
            self.release_certified
            and _release_seal is not _LOCKED_LAUNCHER_ATTESTATION_SEAL
        ):
            raise ValueError(
                "$LauncherExecutionAttestation: release authority requires the locked launcher"
            )
        _digest(self.attestation_id, "$LauncherExecutionAttestation.attestation_id")


def _launcher_attestation_core(
    value: LauncherExecutionAttestation,
) -> dict[str, Any]:
    return {
        "backend_observation_digest": value.backend_observation_digest,
        "exit_status": value.exit_status,
        "gap_inclusion_projection_digest": (
            value.gap_inclusion_projection_digest
        ),
        "raw_output_digest": value.raw_output_digest,
        "process_stderr_digest": value.process_stderr_digest,
        "process_stdout_digest": value.process_stdout_digest,
        "release_certified": value.release_certified,
        "request_input_digest": value.request_input_digest,
        "resolved_launcher_digest": value.resolved_launcher_digest,
        "runtime_manifest_digest": value.runtime_manifest_digest,
    }


def launcher_execution_attestation_digest(
    value: LauncherExecutionAttestation,
) -> str:
    if not isinstance(value, LauncherExecutionAttestation):
        raise TypeError("launcher attestation digest requires a typed attestation")
    return _task5_domain_digest(
        "task5-launcher-execution-attestation-v1",
        _launcher_attestation_core(value),
    )


def make_launcher_execution_attestation(
    *,
    request_input_digest: str,
    raw_output_digest: str,
    gap_inclusion_projection_digest: str,
    process_stdout_digest: str,
    process_stderr_digest: str,
    resolved_launcher_digest: str,
    backend_observation_digest: str,
    runtime_manifest_digest: str | None,
    exit_status: int,
    release_certified: bool,
) -> LauncherExecutionAttestation:
    if release_certified or runtime_manifest_digest is not None:
        raise ValueError(
            "public launcher-attestation constructor is diagnostic-only; "
            "release attestations must come from the locked launcher"
        )
    return _make_launcher_execution_attestation(
        request_input_digest=request_input_digest,
        raw_output_digest=raw_output_digest,
        gap_inclusion_projection_digest=gap_inclusion_projection_digest,
        process_stdout_digest=process_stdout_digest,
        process_stderr_digest=process_stderr_digest,
        resolved_launcher_digest=resolved_launcher_digest,
        backend_observation_digest=backend_observation_digest,
        runtime_manifest_digest=runtime_manifest_digest,
        exit_status=exit_status,
        release_certified=release_certified,
    )


def _make_launcher_execution_attestation(
    *,
    request_input_digest: str,
    raw_output_digest: str,
    gap_inclusion_projection_digest: str,
    process_stdout_digest: str,
    process_stderr_digest: str,
    resolved_launcher_digest: str,
    backend_observation_digest: str,
    runtime_manifest_digest: str | None,
    exit_status: int,
    release_certified: bool,
) -> LauncherExecutionAttestation:
    provisional = LauncherExecutionAttestation(
        request_input_digest,
        raw_output_digest,
        gap_inclusion_projection_digest,
        process_stdout_digest,
        process_stderr_digest,
        resolved_launcher_digest,
        backend_observation_digest,
        runtime_manifest_digest,
        exit_status,
        release_certified,
        "sha256:" + "0" * 64,
        _LOCKED_LAUNCHER_ATTESTATION_SEAL if release_certified else None,
    )
    return LauncherExecutionAttestation(
        provisional.request_input_digest,
        provisional.raw_output_digest,
        provisional.gap_inclusion_projection_digest,
        provisional.process_stdout_digest,
        provisional.process_stderr_digest,
        provisional.resolved_launcher_digest,
        provisional.backend_observation_digest,
        provisional.runtime_manifest_digest,
        provisional.exit_status,
        provisional.release_certified,
        launcher_execution_attestation_digest(provisional),
        _LOCKED_LAUNCHER_ATTESTATION_SEAL
        if provisional.release_certified
        else None,
    )


def launcher_execution_attestation_mapping(
    value: LauncherExecutionAttestation,
) -> dict[str, Any]:
    return {
        **_launcher_attestation_core(value),
        "attestation_id": value.attestation_id,
        "record_type": "task5-launcher-execution-attestation",
        "schema_version": 1,
    }


def _parse_launcher_execution_attestation(
    value: Any,
    path: str,
    *,
    allow_diagnostic: bool,
    trusted_release_attestation: LauncherExecutionAttestation | None,
) -> LauncherExecutionAttestation:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(
        value,
        {
            "attestation_id",
            "backend_observation_digest",
            "exit_status",
            "gap_inclusion_projection_digest",
            "raw_output_digest",
            "process_stderr_digest",
            "process_stdout_digest",
            "record_type",
            "release_certified",
            "request_input_digest",
            "resolved_launcher_digest",
            "runtime_manifest_digest",
            "schema_version",
        },
        path,
    )
    if (
        value["record_type"] != "task5-launcher-execution-attestation"
        or value["schema_version"] != 1
    ):
        raise ValueError(f"{path}: unsupported launcher-attestation schema")
    if value["release_certified"] is True:
        if (
            trusted_release_attestation is None
            or not isinstance(
                trusted_release_attestation, LauncherExecutionAttestation
            )
            or not trusted_release_attestation.release_certified
            or launcher_execution_attestation_mapping(
                trusted_release_attestation
            )
            != value
        ):
            raise ValueError(
                f"{path}: serialized release claims require external "
                "locked-launcher authority"
            )
        # Reuse the actual nonserialized launcher result.  Never restore the
        # private release-construction seal from attacker-controlled bytes.
        return trusted_release_attestation
    if not allow_diagnostic:
        raise ValueError(
            f"{path}: diagnostic attestation requires explicit diagnostic opt-in; "
            "release loading requires an external locked-launcher attestation authority"
        )
    result = LauncherExecutionAttestation(
        value["request_input_digest"],
        value["raw_output_digest"],
        value["gap_inclusion_projection_digest"],
        value["process_stdout_digest"],
        value["process_stderr_digest"],
        value["resolved_launcher_digest"],
        value["backend_observation_digest"],
        value["runtime_manifest_digest"],
        value["exit_status"],
        value["release_certified"],
        value["attestation_id"],
        None,
    )
    if launcher_execution_attestation_digest(result) != result.attestation_id:
        raise ValueError(f"{path}.attestation_id: does not bind payload")
    return result


@dataclass(frozen=True, slots=True)
class Task5InclusionAuthority:
    inclusion_id: str
    literal_stabilizer_digest: str
    literal_element_digest: str
    transported_inclusion_digest: str
    source_bar_equivalence_id: str
    target_bar_equivalence_id: str
    launcher_attestation_id: str
    gap_inclusion_projection_digest: str
    diagnostic_backend: str
    diagnostic_outcome: str
    diagnostic_failure_degrees: tuple[int, ...]
    diagnostic_residue_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.inclusion_id, "$Task5InclusionAuthority.inclusion_id")
        _digest(
            self.literal_stabilizer_digest,
            "$Task5InclusionAuthority.literal_stabilizer_digest",
        )
        _digest(
            self.literal_element_digest,
            "$Task5InclusionAuthority.literal_element_digest",
        )
        _digest(
            self.transported_inclusion_digest,
            "$Task5InclusionAuthority.transported_inclusion_digest",
        )
        _digest(
            self.source_bar_equivalence_id,
            "$Task5InclusionAuthority.source_bar_equivalence_id",
        )
        _digest(
            self.target_bar_equivalence_id,
            "$Task5InclusionAuthority.target_bar_equivalence_id",
        )
        _digest(
            self.launcher_attestation_id,
            "$Task5InclusionAuthority.launcher_attestation_id",
        )
        _digest(
            self.gap_inclusion_projection_digest,
            "$Task5InclusionAuthority.gap_inclusion_projection_digest",
        )
        backend = _identifier(
            self.diagnostic_backend,
            "$Task5InclusionAuthority.diagnostic_backend",
        )
        failures = _canonical_diagnostic_failure_degrees(
            self.diagnostic_failure_degrees,
            "$Task5InclusionAuthority.diagnostic_failure_degrees",
        )
        if (self.diagnostic_outcome == "commuting") != (not failures):
            raise ValueError(
                "$Task5InclusionAuthority: diagnostic outcome and failure degrees differ"
            )
        residues = _canonical_diagnostic_residue_digests(
            self.diagnostic_residue_digests,
            "$Task5InclusionAuthority.diagnostic_residue_digests",
        )
        if self.diagnostic_outcome == "commuting":
            pass
        elif self.diagnostic_outcome == _FROZEN_P4MM_DIAGNOSTIC_OUTCOME:
            if (
                self.inclusion_id != "p4mm-1a"
                or backend != "HAP-1.70-EquivariantChainMap"
            ):
                raise ValueError(
                    "$Task5InclusionAuthority.diagnostic_outcome: "
                    "frozen p4mm outcome has the wrong inclusion or backend"
                )
        elif self.diagnostic_outcome.startswith(
            _OBSERVED_NONCOMMUTING_OUTCOME_PREFIX
        ):
            expected = task5_diagnostic_observed_outcome(
                backend, failures, residues
            )
            if self.diagnostic_outcome != expected:
                raise ValueError(
                    "$Task5InclusionAuthority.diagnostic_outcome: "
                    "observed outcome does not bind failures and residues"
                )
        else:
            raise ValueError(
                "$Task5InclusionAuthority.diagnostic_outcome: unsupported outcome"
            )
        object.__setattr__(self, "diagnostic_failure_degrees", failures)
        object.__setattr__(self, "diagnostic_residue_digests", residues)


@dataclass(frozen=True, slots=True)
class Task5VerificationAuthority:
    catalogue_action_digest: str
    catalogue_record_digest: str
    affine_pcp_certificate_digest: str
    backend_lock_digest: str
    backend_environment_id: str
    runtime_provenance_digest: str
    inclusions: tuple[Task5InclusionAuthority, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "catalogue_action_digest",
            "catalogue_record_digest",
            "affine_pcp_certificate_digest",
            "backend_lock_digest",
            "backend_environment_id",
            "runtime_provenance_digest",
        ):
            _digest(getattr(self, field_name), f"$Task5VerificationAuthority.{field_name}")
        inclusions = tuple(self.inclusions)
        if any(not isinstance(item, Task5InclusionAuthority) for item in inclusions):
            raise TypeError("$Task5VerificationAuthority.inclusions: expected authority tuple")
        if len({item.inclusion_id for item in inclusions}) != len(inclusions):
            raise ValueError("$Task5VerificationAuthority.inclusions: duplicate inclusion")
        if tuple(item.inclusion_id for item in inclusions) != tuple(
            sorted(item.inclusion_id for item in inclusions)
        ):
            raise ValueError("$Task5VerificationAuthority.inclusions: expected canonical order")
        object.__setattr__(self, "inclusions", inclusions)


@dataclass(frozen=True, slots=True)
class FreeResolutionCertificate:
    group_id: str
    max_degree: int
    basis: tuple[tuple[str, ...], ...]
    boundaries: tuple[SparseGroupRingMatrix, ...]
    degree_five_basis: tuple[str, ...]
    lookahead_boundary: SparseGroupRingMatrix
    affine_pcp_certificate_digest: str
    affine_pcp_certificate: AffinePCPIsomorphismCertificate
    catalogue_record_digest: str
    finite_group: FiniteGroupTable | None
    backend_lock_digest: str
    backend_environment_id: str
    runtime_provenance_digest: str
    resolution_id: str
    construction: str
    parent_spatial_resolution_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.group_id, "$FreeResolutionCertificate.group_id")
        maximum = _integer(self.max_degree, "$FreeResolutionCertificate.max_degree")
        if maximum != 4:
            raise ValueError("$FreeResolutionCertificate.max_degree: v1 requires degree 4")
        basis = tuple(tuple(degree) for degree in self.basis)
        boundaries = tuple(self.boundaries)
        degree_five_basis = tuple(self.degree_five_basis)
        if len(basis) != maximum + 1 or len(boundaries) != maximum:
            raise ValueError("$FreeResolutionCertificate: expected all degrees through four")
        if any(not isinstance(matrix, SparseGroupRingMatrix) for matrix in boundaries):
            raise TypeError("$FreeResolutionCertificate.boundaries: expected sparse matrices")
        if not isinstance(self.lookahead_boundary, SparseGroupRingMatrix):
            raise TypeError("$FreeResolutionCertificate.lookahead_boundary: expected sparse matrix")
        _digest(self.affine_pcp_certificate_digest, "$FreeResolutionCertificate.affine_pcp_certificate_digest")
        if not isinstance(self.affine_pcp_certificate, AffinePCPIsomorphismCertificate):
            raise TypeError("$FreeResolutionCertificate.affine_pcp_certificate: expected Task 4 certificate")
        if self.finite_group is not None and not isinstance(self.finite_group, FiniteGroupTable):
            raise TypeError("$FreeResolutionCertificate.finite_group: expected finite table or None")
        _digest(self.catalogue_record_digest, "$FreeResolutionCertificate.catalogue_record_digest")
        _digest(self.backend_lock_digest, "$FreeResolutionCertificate.backend_lock_digest")
        _digest(self.backend_environment_id, "$FreeResolutionCertificate.backend_environment_id")
        _digest(self.runtime_provenance_digest, "$FreeResolutionCertificate.runtime_provenance_digest")
        _digest(self.resolution_id, "$FreeResolutionCertificate.resolution_id")
        _identifier(self.construction, "$FreeResolutionCertificate.construction")
        if self.construction == "onsite-c2-direct-product-resolution":
            if self.parent_spatial_resolution_id is None:
                raise ValueError(
                    "$FreeResolutionCertificate.parent_spatial_resolution_id: "
                    "onsite direct product requires its exact parent"
                )
            _digest(
                self.parent_spatial_resolution_id,
                "$FreeResolutionCertificate.parent_spatial_resolution_id",
            )
        elif self.parent_spatial_resolution_id is not None:
            raise ValueError(
                "$FreeResolutionCertificate.parent_spatial_resolution_id: "
                "only onsite direct products have a parent spatial resolution"
            )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "degree_five_basis", degree_five_basis)


@dataclass(frozen=True, slots=True, order=True)
class BarComparisonTerm:
    left_element: str
    group_tuple: tuple[str, ...]
    coefficient: int

    def __post_init__(self) -> None:
        if type(self.left_element) is not str or not self.left_element:
            raise ValueError("$BarComparisonTerm.left_element: expected group normal form")
        group_tuple = tuple(self.group_tuple)
        if any(type(item) is not str or not item for item in group_tuple):
            raise ValueError("$BarComparisonTerm.group_tuple: expected group normal forms")
        if type(self.coefficient) is not int or self.coefficient == 0:
            raise ValueError("$BarComparisonTerm.coefficient: expected nonzero integer")
        object.__setattr__(self, "group_tuple", group_tuple)


@dataclass(frozen=True, slots=True, order=True)
class ResolutionComparisonTerm:
    basis_id: str
    element: str
    coefficient: int

    def __post_init__(self) -> None:
        if _BASIS_RE.fullmatch(self.basis_id) is None:
            raise ValueError("$ResolutionComparisonTerm.basis_id: expected canonical basis ID")
        if type(self.element) is not str or not self.element:
            raise ValueError("$ResolutionComparisonTerm.element: expected group normal form")
        if type(self.coefficient) is not int or self.coefficient == 0:
            raise ValueError("$ResolutionComparisonTerm.coefficient: expected nonzero integer")


@dataclass(frozen=True, slots=True)
class BarComparisonBasisTrace:
    degree: int
    source_basis_id: str
    source_psi: tuple[BarComparisonTerm, ...]
    transported_bar: tuple[BarComparisonTerm, ...]
    target_phi: tuple[ResolutionComparisonTerm, ...]

    def __post_init__(self) -> None:
        if type(self.degree) is not int or not 0 <= self.degree <= 4:
            raise ValueError("$BarComparisonBasisTrace.degree: expected degree zero through four")
        if self.source_basis_id != f"c{self.degree}:{self.source_basis_id.split(':')[-1]}":
            raise ValueError("$BarComparisonBasisTrace.source_basis_id: degree mismatch")
        source = tuple(self.source_psi)
        transported = tuple(self.transported_bar)
        target = tuple(self.target_phi)
        if any(not isinstance(item, BarComparisonTerm) for item in source + transported):
            raise TypeError("$BarComparisonBasisTrace: expected bar terms")
        if any(not isinstance(item, ResolutionComparisonTerm) for item in target):
            raise TypeError("$BarComparisonBasisTrace.target_phi: expected resolution terms")
        for label, terms in (("source_psi", source), ("transported_bar", transported)):
            if any(len(term.group_tuple) != self.degree for term in terms):
                raise ValueError(f"$BarComparisonBasisTrace.{label}: tuple degree mismatch")
            if terms != tuple(sorted(terms)):
                raise ValueError(f"$BarComparisonBasisTrace.{label}: expected canonical order")
        if target != tuple(sorted(target)):
            raise ValueError("$BarComparisonBasisTrace.target_phi: expected canonical order")
        object.__setattr__(self, "source_psi", source)
        object.__setattr__(self, "transported_bar", transported)
        object.__setattr__(self, "target_phi", target)


@dataclass(frozen=True, slots=True)
class TargetResolutionBasisTrace:
    degree: int
    basis_id: str
    psi: tuple[BarComparisonTerm, ...]
    resolution_homotopy: tuple[ResolutionComparisonTerm, ...]

    def __post_init__(self) -> None:
        if type(self.degree) is not int or not 0 <= self.degree <= 4:
            raise ValueError("$TargetResolutionBasisTrace.degree: expected zero through four")
        if self.basis_id != f"c{self.degree}:{self.basis_id.split(':')[-1]}":
            raise ValueError("$TargetResolutionBasisTrace.basis_id: degree mismatch")
        psi = tuple(self.psi)
        homotopy = tuple(self.resolution_homotopy)
        if any(not isinstance(term, BarComparisonTerm) for term in psi):
            raise TypeError("$TargetResolutionBasisTrace.psi: expected bar terms")
        if any(len(term.group_tuple) != self.degree for term in psi):
            raise ValueError("$TargetResolutionBasisTrace.psi: tuple degree mismatch")
        if any(not isinstance(term, ResolutionComparisonTerm) for term in homotopy):
            raise TypeError(
                "$TargetResolutionBasisTrace.resolution_homotopy: expected resolution terms"
            )
        if any(
            int(term.basis_id.split(":", 1)[0][1:]) != self.degree + 1
            for term in homotopy
        ):
            raise ValueError(
                "$TargetResolutionBasisTrace.resolution_homotopy: degree mismatch"
            )
        if psi != tuple(sorted(psi)) or homotopy != tuple(sorted(homotopy)):
            raise ValueError("$TargetResolutionBasisTrace: expected canonical term order")
        object.__setattr__(self, "psi", psi)
        object.__setattr__(self, "resolution_homotopy", homotopy)


@dataclass(frozen=True, slots=True)
class TargetBarPhiTrace:
    group_tuple: tuple[str, ...]
    image: tuple[ResolutionComparisonTerm, ...]
    bar_homotopy: tuple[BarComparisonTerm, ...]

    def __post_init__(self) -> None:
        group_tuple = tuple(self.group_tuple)
        image = tuple(self.image)
        homotopy = tuple(self.bar_homotopy)
        degree = len(group_tuple)
        if any(type(item) is not str or not item for item in group_tuple):
            raise ValueError("$TargetBarPhiTrace.group_tuple: expected group normal forms")
        if any(not isinstance(term, ResolutionComparisonTerm) for term in image):
            raise TypeError("$TargetBarPhiTrace.image: expected resolution terms")
        if any(
            int(term.basis_id.split(":", 1)[0][1:]) != degree for term in image
        ):
            raise ValueError("$TargetBarPhiTrace.image: degree mismatch")
        if any(not isinstance(term, BarComparisonTerm) for term in homotopy):
            raise TypeError("$TargetBarPhiTrace.bar_homotopy: expected bar terms")
        if any(len(term.group_tuple) != degree + 1 for term in homotopy):
            raise ValueError("$TargetBarPhiTrace.bar_homotopy: degree mismatch")
        if image != tuple(sorted(image)) or homotopy != tuple(sorted(homotopy)):
            raise ValueError("$TargetBarPhiTrace: expected canonical term order")
        object.__setattr__(self, "group_tuple", group_tuple)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "bar_homotopy", homotopy)


@dataclass(frozen=True, slots=True)
class TargetBarResolutionEquivalence:
    target_resolution_id: str
    phi_algorithm: str
    bar_homotopy_algorithm: str
    basis_traces: tuple[TargetResolutionBasisTrace, ...]
    phi_traces: tuple[TargetBarPhiTrace, ...]
    queried_bar_tuples: tuple[tuple[str, ...], ...]
    lookahead_boundary: SparseGroupRingMatrix
    equivalence_id: str

    def __post_init__(self) -> None:
        _digest(
            self.target_resolution_id,
            "$TargetBarResolutionEquivalence.target_resolution_id",
        )
        _identifier(self.phi_algorithm, "$TargetBarResolutionEquivalence.phi_algorithm")
        _identifier(
            self.bar_homotopy_algorithm,
            "$TargetBarResolutionEquivalence.bar_homotopy_algorithm",
        )
        basis = tuple(self.basis_traces)
        phi = tuple(self.phi_traces)
        queried = tuple(tuple(item) for item in self.queried_bar_tuples)
        if any(not isinstance(item, TargetResolutionBasisTrace) for item in basis):
            raise TypeError("$TargetBarResolutionEquivalence.basis_traces: invalid trace")
        if any(not isinstance(item, TargetBarPhiTrace) for item in phi):
            raise TypeError("$TargetBarResolutionEquivalence.phi_traces: invalid trace")
        if not isinstance(self.lookahead_boundary, SparseGroupRingMatrix):
            raise TypeError("$TargetBarResolutionEquivalence.lookahead_boundary: invalid matrix")
        _digest(self.equivalence_id, "$TargetBarResolutionEquivalence.equivalence_id")
        object.__setattr__(self, "basis_traces", basis)
        object.__setattr__(self, "phi_traces", phi)
        object.__setattr__(self, "queried_bar_tuples", queried)


@dataclass(frozen=True, slots=True)
class InclusionChainMapCertificate:
    inclusion_id: str
    affine_pcp_certificate_digest: str
    literal_stabilizer_digest: str
    literal_element_digest: str
    transported_inclusion_digest: str
    source_resolution_id: str
    target_resolution_id: str
    source_resolution: FreeResolutionCertificate
    target_resolution: FreeResolutionCertificate
    source_element_images: tuple[str, ...]
    maps: tuple[SparseGroupRingMatrix, ...]
    chain_map_algorithm: str
    source_bar_equivalence_id: str
    target_bar_equivalence_id: str
    target_bar_equivalence: TargetBarResolutionEquivalence
    launcher_attestation: LauncherExecutionAttestation
    gap_inclusion_projection_digest: str
    bar_comparison_traces: tuple[BarComparisonBasisTrace, ...]
    diagnostic_backend: str
    diagnostic_maps: tuple[SparseGroupRingMatrix, ...]
    diagnostic_outcome: str
    diagnostic_residue_digests: tuple[str, ...]
    certificate_id: str

    def __post_init__(self) -> None:
        _identifier(self.inclusion_id, "$InclusionChainMapCertificate.inclusion_id")
        _digest(self.affine_pcp_certificate_digest, "$InclusionChainMapCertificate.affine_pcp_certificate_digest")
        _digest(self.literal_stabilizer_digest, "$InclusionChainMapCertificate.literal_stabilizer_digest")
        _digest(self.literal_element_digest, "$InclusionChainMapCertificate.literal_element_digest")
        _digest(self.transported_inclusion_digest, "$InclusionChainMapCertificate.transported_inclusion_digest")
        _digest(self.source_resolution_id, "$InclusionChainMapCertificate.source_resolution_id")
        _digest(self.target_resolution_id, "$InclusionChainMapCertificate.target_resolution_id")
        if not isinstance(self.source_resolution, FreeResolutionCertificate) or not isinstance(
            self.target_resolution, FreeResolutionCertificate
        ):
            raise TypeError("$InclusionChainMapCertificate: expected embedded resolutions")
        images = tuple(self.source_element_images)
        maps = tuple(self.maps)
        traces = tuple(self.bar_comparison_traces)
        diagnostic = tuple(self.diagnostic_maps)
        if len(maps) != 5 or any(not isinstance(matrix, SparseGroupRingMatrix) for matrix in maps):
            raise ValueError("$InclusionChainMapCertificate.maps: expected degree zero through four")
        if any(not isinstance(item, BarComparisonBasisTrace) for item in traces):
            raise TypeError("$InclusionChainMapCertificate.bar_comparison_traces: expected trace tuple")
        if len(diagnostic) != 5 or any(not isinstance(matrix, SparseGroupRingMatrix) for matrix in diagnostic):
            raise ValueError("$InclusionChainMapCertificate.diagnostic_maps: expected degree zero through four")
        _identifier(self.chain_map_algorithm, "$InclusionChainMapCertificate.chain_map_algorithm")
        _digest(self.source_bar_equivalence_id, "$InclusionChainMapCertificate.source_bar_equivalence_id")
        _digest(self.target_bar_equivalence_id, "$InclusionChainMapCertificate.target_bar_equivalence_id")
        if not isinstance(self.target_bar_equivalence, TargetBarResolutionEquivalence):
            raise TypeError(
                "$InclusionChainMapCertificate.target_bar_equivalence: expected target equivalence"
            )
        if not isinstance(self.launcher_attestation, LauncherExecutionAttestation):
            raise TypeError(
                "$InclusionChainMapCertificate.launcher_attestation: expected typed attestation"
            )
        _digest(
            self.gap_inclusion_projection_digest,
            "$InclusionChainMapCertificate.gap_inclusion_projection_digest",
        )
        backend = _identifier(
            self.diagnostic_backend,
            "$InclusionChainMapCertificate.diagnostic_backend",
        )
        residues = _canonical_diagnostic_residue_digests(
            self.diagnostic_residue_digests,
            "$InclusionChainMapCertificate.diagnostic_residue_digests",
        )
        if self.diagnostic_outcome == "commuting":
            pass
        elif self.diagnostic_outcome == _FROZEN_P4MM_DIAGNOSTIC_OUTCOME:
            if (
                self.inclusion_id != "p4mm-1a"
                or backend != "HAP-1.70-EquivariantChainMap"
            ):
                raise ValueError(
                    "$InclusionChainMapCertificate.diagnostic_outcome: "
                    "frozen p4mm outcome has the wrong inclusion or backend"
                )
        elif self.diagnostic_outcome.startswith(
            _OBSERVED_NONCOMMUTING_OUTCOME_PREFIX
        ):
            if _DIGEST_RE.fullmatch(
                self.diagnostic_outcome[
                    len(_OBSERVED_NONCOMMUTING_OUTCOME_PREFIX) :
                ]
            ) is None:
                raise ValueError(
                    "$InclusionChainMapCertificate.diagnostic_outcome: "
                    "observed outcome is not digest-bound"
                )
        else:
            raise ValueError(
                "$InclusionChainMapCertificate.diagnostic_outcome: unsupported outcome"
            )
        _digest(self.certificate_id, "$InclusionChainMapCertificate.certificate_id")
        object.__setattr__(self, "source_element_images", images)
        object.__setattr__(self, "maps", maps)
        object.__setattr__(self, "bar_comparison_traces", traces)
        object.__setattr__(self, "diagnostic_maps", diagnostic)
        object.__setattr__(self, "diagnostic_residue_digests", residues)


@dataclass(frozen=True, slots=True)
class CharacterBasisCertificate:
    group_id: str
    resolution_id: str
    presentation_kind: str
    presentation_digest: str
    generator_order: tuple[str, ...]
    relator_words: tuple[tuple[tuple[int, int], ...], ...]
    relator_matrix_mod2: MatrixGF2
    row_change: MatrixGF2
    column_change: MatrixGF2
    normal_form: MatrixGF2
    abelianization_basis: tuple[tuple[int, ...], ...]
    generator_to_abelianization: MatrixGF2
    abelianization_to_generators: MatrixGF2
    hom_basis: tuple[GF2Character, ...]
    characters: tuple[GF2Character, ...]
    spatial_certificate_id: str | None
    onsite_time_reversal_generator: str | None
    certificate_id: str

    def __post_init__(self) -> None:
        _identifier(self.group_id, "$CharacterBasisCertificate.group_id")
        _digest(self.resolution_id, "$CharacterBasisCertificate.resolution_id")
        if self.presentation_kind not in (
            "finite-table-presentation",
            "graded-direct-product-presentation",
            "pcp-presentation",
        ):
            raise ValueError("$CharacterBasisCertificate.presentation_kind: unsupported presentation")
        _digest(self.presentation_digest, "$CharacterBasisCertificate.presentation_digest")
        generators = tuple(self.generator_order)
        if not generators or len(set(generators)) != len(generators):
            raise ValueError("$CharacterBasisCertificate.generator_order: expected unique generators")
        for index, generator in enumerate(generators):
            _identifier(generator, f"$CharacterBasisCertificate.generator_order[{index}]")
        matrices = (
            self.relator_matrix_mod2,
            self.row_change,
            self.column_change,
            self.normal_form,
            self.generator_to_abelianization,
            self.abelianization_to_generators,
        )
        if any(not isinstance(matrix, MatrixGF2) for matrix in matrices):
            raise TypeError("$CharacterBasisCertificate: expected shaped GF(2) matrices")
        basis = tuple(tuple(vector) for vector in self.abelianization_basis)
        hom = tuple(self.hom_basis)
        characters = tuple(self.characters)
        if any(not isinstance(character, GF2Character) for character in hom + characters):
            raise TypeError("$CharacterBasisCertificate: expected GF2Character tuples")
        relators = tuple(tuple(tuple(step) for step in word) for word in self.relator_words)
        for word_index, word in enumerate(relators):
            if not word:
                raise ValueError(f"$CharacterBasisCertificate.relator_words[{word_index}]: empty relator")
            for step_index, step in enumerate(word):
                if (
                    len(step) != 2
                    or type(step[0]) is not int
                    or not 0 <= step[0] < len(generators)
                    or type(step[1]) is not int
                    or step[1] == 0
                ):
                    raise ValueError(
                        f"$CharacterBasisCertificate.relator_words[{word_index}][{step_index}]: invalid generator/exponent"
                    )
        if self.spatial_certificate_id is not None:
            _digest(
                self.spatial_certificate_id,
                "$CharacterBasisCertificate.spatial_certificate_id",
            )
        if self.onsite_time_reversal_generator is not None:
            _identifier(
                self.onsite_time_reversal_generator,
                "$CharacterBasisCertificate.onsite_time_reversal_generator",
            )
        _digest(self.certificate_id, "$CharacterBasisCertificate.certificate_id")
        object.__setattr__(self, "generator_order", generators)
        object.__setattr__(self, "relator_words", relators)
        object.__setattr__(self, "abelianization_basis", basis)
        object.__setattr__(self, "hom_basis", hom)
        object.__setattr__(self, "characters", characters)


def _matrix_product_is_zero(
    left: MatrixZ | MatrixGF2, right: MatrixZ | MatrixGF2
) -> bool:
    if type(left) is not type(right):
        raise TypeError("matrix coefficient rings differ")
    if left.column_count != right.row_count:
        raise ValueError("matrix dimensions differ")
    modulus = 2 if isinstance(left, MatrixGF2) else None
    for row in range(left.row_count):
        for column in range(right.column_count):
            value = sum(
                left[row][middle] * right[middle][column]
                for middle in range(left.column_count)
            )
            if (value & 1 if modulus else value) != 0:
                return False
    return True


def _matrix_products_equal(
    left: MatrixZ | MatrixGF2,
    right: MatrixZ | MatrixGF2,
    other_left: MatrixZ | MatrixGF2,
    other_right: MatrixZ | MatrixGF2,
) -> bool:
    if not (
        type(left) is type(right)
        and type(left) is type(other_left)
        and type(left) is type(other_right)
    ):
        return False
    if left.column_count != right.row_count or other_left.column_count != other_right.row_count:
        return False
    if (left.row_count, right.column_count) != (
        other_left.row_count,
        other_right.column_count,
    ):
        return False
    modulus = 2 if isinstance(left, MatrixGF2) else None
    for row in range(left.row_count):
        for column in range(right.column_count):
            value = sum(
                left[row][middle] * right[middle][column]
                for middle in range(left.column_count)
            )
            other = sum(
                other_left[row][middle] * other_right[middle][column]
                for middle in range(other_left.column_count)
            )
            if modulus:
                value &= 1
                other &= 1
            if value != other:
                return False
    return True


@dataclass(frozen=True, slots=True)
class CochainComplex:
    complex_id: str
    authority_id: str
    dimensions: tuple[int, ...]
    differentials: tuple[MatrixZ | MatrixGF2, ...]
    coefficient_character: GF2Character

    def __post_init__(self) -> None:
        _digest(self.complex_id, "$CochainComplex.complex_id")
        _digest(self.authority_id, "$CochainComplex.authority_id")
        dimensions = tuple(self.dimensions)
        differentials = tuple(self.differentials)
        if len(dimensions) != 5 or any(type(value) is not int or value < 0 for value in dimensions):
            raise ValueError("$CochainComplex.dimensions: expected five nonnegative ranks")
        if len(differentials) != 4 or any(
            not isinstance(matrix, (MatrixZ, MatrixGF2)) for matrix in differentials
        ):
            raise ValueError("$CochainComplex.differentials: expected degree zero through three")
        matrix_type = type(differentials[0]) if differentials else MatrixZ
        if any(type(matrix) is not matrix_type for matrix in differentials):
            raise TypeError("$CochainComplex.differentials: coefficient rings differ")
        for degree, matrix in enumerate(differentials):
            if matrix.shape != (dimensions[degree + 1], dimensions[degree]):
                raise ValueError(
                    f"$CochainComplex.differentials[{degree}]: shape differs from dimensions"
                )
        for degree in range(3):
            if not _matrix_product_is_zero(
                differentials[degree + 1], differentials[degree]
            ):
                raise ValueError(
                    f"$CochainComplex.differentials[{degree}]: D squared is nonzero"
                )
        if not isinstance(self.coefficient_character, GF2Character):
            raise TypeError("$CochainComplex.coefficient_character: expected GF2Character")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "differentials", differentials)
        if self.complex_id != _cochain_complex_digest(
            self.authority_id,
            dimensions,
            differentials,
            self.coefficient_character,
        ):
            raise ValueError("$CochainComplex.complex_id: does not bind complex")


@dataclass(frozen=True, slots=True)
class CochainMap:
    instance_id: str
    source_id: str
    target_id: str
    maps: tuple[MatrixZ | MatrixGF2, ...]

    def __post_init__(self) -> None:
        _identifier(self.instance_id, "$CochainMap.instance_id")
        _digest(self.source_id, "$CochainMap.source_id")
        _digest(self.target_id, "$CochainMap.target_id")
        maps = tuple(self.maps)
        if len(maps) != 5 or any(
            not isinstance(matrix, (MatrixZ, MatrixGF2)) for matrix in maps
        ):
            raise ValueError("$CochainMap.maps: expected degree zero through four")
        if any(type(matrix) is not type(maps[0]) for matrix in maps):
            raise TypeError("$CochainMap.maps: coefficient rings differ")
        object.__setattr__(self, "maps", maps)


@dataclass(frozen=True, slots=True)
class CertifiedCochainProblem:
    ambient: FreeResolutionCertificate
    inclusions: tuple[InclusionChainMapCertificate, ...]
    character_basis: CharacterBasisCertificate


def _exact_matrix_mapping(matrix: MatrixZ | MatrixGF2) -> dict[str, Any]:
    return {
        "column_count": matrix.column_count,
        "ring": "gf2" if isinstance(matrix, MatrixGF2) else "z",
        "rows": [list(row) for row in matrix.rows],
    }


def _cochain_complex_digest(
    authority_id: str,
    dimensions: Sequence[int],
    differentials: Sequence[MatrixZ | MatrixGF2],
    coefficient_character: GF2Character,
) -> str:
    return _domain_digest(
        "cochain-complex",
        {
            "authority_id": authority_id,
            "coefficient_character": list(coefficient_character.bits),
            "differentials": [
                _exact_matrix_mapping(matrix) for matrix in differentials
            ],
            "dimensions": list(dimensions),
        },
    )


def make_cochain_complex(
    *,
    authority_id: str,
    dimensions: Sequence[int],
    differentials: Sequence[MatrixZ | MatrixGF2],
    coefficient_character: GF2Character,
) -> CochainComplex:
    normalized_dimensions = tuple(dimensions)
    normalized_differentials = tuple(differentials)
    return CochainComplex(
        _cochain_complex_digest(
            authority_id,
            normalized_dimensions,
            normalized_differentials,
            coefficient_character,
        ),
        authority_id,
        normalized_dimensions,
        normalized_differentials,
        coefficient_character,
    )


def _matrix_mapping(matrix: SparseGroupRingMatrix) -> dict[str, Any]:
    return {
        "column_count": matrix.column_count,
        "entries": [
            {
                "column": entry.column,
                "row": entry.row,
                "terms": [[term.coefficient, term.element] for term in entry.terms],
            }
            for entry in matrix.entries
        ],
        "row_count": matrix.row_count,
    }


def _parse_matrix(value: Any, path: str) -> SparseGroupRingMatrix:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(value, {"column_count", "entries", "row_count"}, path)
    if not isinstance(value["entries"], list):
        raise TypeError(f"{path}.entries: expected array")
    entries: list[SparseGroupRingEntry] = []
    for index, item in enumerate(value["entries"]):
        item_path = f"{path}.entries[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"column", "row", "terms"}, item_path)
        if not isinstance(item["terms"], list):
            raise TypeError(f"{item_path}.terms: expected array")
        terms: list[SparseGroupRingTerm] = []
        for term_index, term in enumerate(item["terms"]):
            if not isinstance(term, list) or len(term) != 2:
                raise ValueError(f"{item_path}.terms[{term_index}]: expected [coefficient,element]")
            terms.append(SparseGroupRingTerm(term[1], term[0]))
        entries.append(SparseGroupRingEntry(item["row"], item["column"], tuple(terms)))
    return SparseGroupRingMatrix(value["row_count"], value["column_count"], tuple(entries))


def _finite_mapping(table: FiniteGroupTable) -> dict[str, Any]:
    return {
        "element_order": list(table.element_order),
        "group_id": table.group_id,
        "identity_index": table.identity_index,
        "inverse_indices": list(table.inverse_indices),
        "multiplication_table": [list(row) for row in table.multiplication_table],
        "table_digest": table.table_digest,
    }


def _parse_finite(value: Any, path: str) -> FiniteGroupTable:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(
        value,
        {"element_order", "group_id", "identity_index", "inverse_indices", "multiplication_table", "table_digest"},
        path,
    )
    return FiniteGroupTable(
        value["group_id"],
        tuple(value["element_order"]),
        value["identity_index"],
        tuple(tuple(row) for row in value["multiplication_table"]),
        tuple(value["inverse_indices"]),
        value["table_digest"],
    )


def _resolution_core(certificate: FreeResolutionCertificate) -> dict[str, Any]:
    core = {
        "affine_pcp_certificate": _certificate_mapping(certificate.affine_pcp_certificate),
        "affine_pcp_certificate_digest": certificate.affine_pcp_certificate_digest,
        "backend_environment_id": certificate.backend_environment_id,
        "backend_lock_digest": certificate.backend_lock_digest,
        "basis": [list(degree) for degree in certificate.basis],
        "boundaries": [_matrix_mapping(matrix) for matrix in certificate.boundaries],
        "catalogue_record_digest": certificate.catalogue_record_digest,
        "construction": certificate.construction,
        "degree_five_basis": list(certificate.degree_five_basis),
        "finite_group": None if certificate.finite_group is None else _finite_mapping(certificate.finite_group),
        "group_id": certificate.group_id,
        "lookahead_boundary": _matrix_mapping(certificate.lookahead_boundary),
        "max_degree": certificate.max_degree,
        "runtime_provenance_digest": certificate.runtime_provenance_digest,
    }
    if certificate.parent_spatial_resolution_id is not None:
        core["parent_spatial_resolution_id"] = (
            certificate.parent_spatial_resolution_id
        )
    return core


def free_resolution_digest(certificate: FreeResolutionCertificate) -> str:
    return _task5_domain_digest(
        "task5-free-resolution-certificate-v1", _resolution_core(certificate)
    )


def free_resolution_mapping(certificate: FreeResolutionCertificate) -> dict[str, Any]:
    return {
        **_resolution_core(certificate),
        "record_type": "free-resolution-certificate",
        "resolution_id": certificate.resolution_id,
        "schema_version": 1,
    }


def _parse_resolution(value: Any, path: str) -> FreeResolutionCertificate:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    expected_fields = {
        "affine_pcp_certificate",
        "affine_pcp_certificate_digest",
        "backend_environment_id",
        "backend_lock_digest",
        "basis",
        "boundaries",
        "catalogue_record_digest",
        "construction",
        "degree_five_basis",
        "finite_group",
        "group_id",
        "lookahead_boundary",
        "max_degree",
        "record_type",
        "resolution_id",
        "runtime_provenance_digest",
        "schema_version",
    }
    if value.get("construction") == "onsite-c2-direct-product-resolution":
        expected_fields.add("parent_spatial_resolution_id")
    _fields(value, expected_fields, path)
    if value["record_type"] != "free-resolution-certificate" or value["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported resolution schema")
    certificate = FreeResolutionCertificate(
        value["group_id"],
        value["max_degree"],
        tuple(tuple(degree) for degree in value["basis"]),
        tuple(_parse_matrix(matrix, f"{path}.boundaries[{index}]") for index, matrix in enumerate(value["boundaries"])),
        tuple(value["degree_five_basis"]),
        _parse_matrix(value["lookahead_boundary"], f"{path}.lookahead_boundary"),
        value["affine_pcp_certificate_digest"],
        _parse_certificate(value["affine_pcp_certificate"], f"{path}.affine_pcp_certificate"),
        value["catalogue_record_digest"],
        None if value["finite_group"] is None else _parse_finite(value["finite_group"], f"{path}.finite_group"),
        value["backend_lock_digest"],
        value["backend_environment_id"],
        value["runtime_provenance_digest"],
        value["resolution_id"],
        value["construction"],
        value.get("parent_spatial_resolution_id"),
    )
    if free_resolution_digest(certificate) != certificate.resolution_id:
        raise ValueError(f"{path}.resolution_id: does not bind certificate")
    return certificate


def loads_free_resolution_certificate(data: bytes) -> FreeResolutionCertificate:
    return _parse_resolution(_strict_json(data), "$resolution")


def dumps_free_resolution_certificate(certificate: FreeResolutionCertificate) -> bytes:
    if not isinstance(certificate, FreeResolutionCertificate):
        raise TypeError("expected FreeResolutionCertificate")
    return _canonical_json(free_resolution_mapping(certificate))


def make_free_resolution_certificate(
    *,
    group_id: str,
    basis: Sequence[Sequence[str]],
    boundaries: Sequence[SparseGroupRingMatrix],
    degree_five_basis: Sequence[str],
    lookahead_boundary: SparseGroupRingMatrix,
    affine_pcp_certificate: AffinePCPIsomorphismCertificate,
    catalogue_record_digest: str,
    finite_group: FiniteGroupTable | None = None,
    construction: str,
    parent_spatial_resolution_id: str | None = None,
) -> FreeResolutionCertificate:
    backend_lock_digest, backend_environment_id, runtime_provenance_digest = _task5_backend_binding()
    provisional = FreeResolutionCertificate(
        group_id,
        4,
        tuple(tuple(degree) for degree in basis),
        tuple(boundaries),
        tuple(degree_five_basis),
        lookahead_boundary,
        affine_pcp_certificate.certificate_digest,
        affine_pcp_certificate,
        catalogue_record_digest,
        finite_group,
        backend_lock_digest,
        backend_environment_id,
        runtime_provenance_digest,
        "sha256:" + "0" * 64,
        construction,
        parent_spatial_resolution_id,
    )
    return replace_resolution_id(provisional)


def assemble_gap_free_resolution_certificate(
    raw_export: Mapping[str, Any],
    *,
    group_id: str,
    affine_pcp_certificate: AffinePCPIsomorphismCertificate,
    catalogue_record_digest: str,
    finite_group: FiniteGroupTable | None,
    construction: str,
    backend_lock_digest: str,
    backend_environment_id: str,
    runtime_provenance_digest: str,
    parent_spatial_resolution_id: str | None = None,
) -> FreeResolutionCertificate:
    """Assemble the sole public certificate schema from an untrusted GAP export."""

    if not isinstance(raw_export, Mapping):
        raise TypeError("raw GAP resolution export must be an object")
    _fields(
        raw_export,
        {"basis", "boundaries", "degree_five_basis", "lookahead_boundary"},
        "$gap_resolution_export",
    )
    if not isinstance(raw_export["basis"], list) or not isinstance(
        raw_export["boundaries"], list
    ) or not isinstance(raw_export["degree_five_basis"], list):
        raise TypeError("raw GAP resolution basis and boundaries must be arrays")
    provisional = FreeResolutionCertificate(
        group_id,
        4,
        tuple(tuple(degree) for degree in raw_export["basis"]),
        tuple(
            _parse_matrix(matrix, f"$gap_resolution_export.boundaries[{index}]")
            for index, matrix in enumerate(raw_export["boundaries"])
        ),
        tuple(raw_export["degree_five_basis"]),
        _parse_matrix(
            raw_export["lookahead_boundary"],
            "$gap_resolution_export.lookahead_boundary",
        ),
        affine_pcp_certificate.certificate_digest,
        affine_pcp_certificate,
        catalogue_record_digest,
        finite_group,
        backend_lock_digest,
        backend_environment_id,
        runtime_provenance_digest,
        "sha256:" + "0" * 64,
        construction,
        parent_spatial_resolution_id,
    )
    return replace_resolution_id(provisional)


def replace_resolution_id(certificate: FreeResolutionCertificate) -> FreeResolutionCertificate:
    return FreeResolutionCertificate(
        certificate.group_id,
        certificate.max_degree,
        certificate.basis,
        certificate.boundaries,
        certificate.degree_five_basis,
        certificate.lookahead_boundary,
        certificate.affine_pcp_certificate_digest,
        certificate.affine_pcp_certificate,
        certificate.catalogue_record_digest,
        certificate.finite_group,
        certificate.backend_lock_digest,
        certificate.backend_environment_id,
        certificate.runtime_provenance_digest,
        free_resolution_digest(certificate),
        certificate.construction,
        certificate.parent_spatial_resolution_id,
    )


def _target_bar_equivalence_core(
    value: TargetBarResolutionEquivalence,
) -> dict[str, Any]:
    return {
        "bar_homotopy_algorithm": value.bar_homotopy_algorithm,
        "basis_traces": [
            {
                "basis_id": trace.basis_id,
                "degree": trace.degree,
                "psi": [
                    {
                        "coefficient": term.coefficient,
                        "group_tuple": list(term.group_tuple),
                        "left_element": term.left_element,
                    }
                    for term in trace.psi
                ],
                "resolution_homotopy": [
                    {
                        "basis_id": term.basis_id,
                        "coefficient": term.coefficient,
                        "element": term.element,
                    }
                    for term in trace.resolution_homotopy
                ],
            }
            for trace in value.basis_traces
        ],
        "lookahead_boundary": _matrix_mapping(value.lookahead_boundary),
        "phi_algorithm": value.phi_algorithm,
        "phi_traces": [
            {
                "bar_homotopy": [
                    {
                        "coefficient": term.coefficient,
                        "group_tuple": list(term.group_tuple),
                        "left_element": term.left_element,
                    }
                    for term in trace.bar_homotopy
                ],
                "group_tuple": list(trace.group_tuple),
                "image": [
                    {
                        "basis_id": term.basis_id,
                        "coefficient": term.coefficient,
                        "element": term.element,
                    }
                    for term in trace.image
                ],
            }
            for trace in value.phi_traces
        ],
        "queried_bar_tuples": [list(item) for item in value.queried_bar_tuples],
        "target_resolution_id": value.target_resolution_id,
    }


def target_bar_equivalence_digest(
    value: TargetBarResolutionEquivalence,
) -> str:
    if not isinstance(value, TargetBarResolutionEquivalence):
        raise TypeError("target bar digest requires a target equivalence")
    return _task5_domain_digest(
        "task5-target-bar-resolution-equivalence-v1",
        _target_bar_equivalence_core(value),
    )


def target_bar_equivalence_mapping(
    value: TargetBarResolutionEquivalence,
) -> dict[str, Any]:
    return {
        **_target_bar_equivalence_core(value),
        "equivalence_id": value.equivalence_id,
        "record_type": "target-bar-resolution-equivalence",
        "schema_version": 1,
    }


_GAP_INCLUSION_PROJECTION_FIELDS = (
    "bar_comparison_traces",
    "chain_map_algorithm",
    "diagnostic_backend",
    "diagnostic_maps",
    "source_element_images",
    "target_bar_equivalence",
)


def _bar_comparison_term_mapping(term: BarComparisonTerm) -> dict[str, Any]:
    return {
        "coefficient": term.coefficient,
        "group_tuple": list(term.group_tuple),
        "left_element": term.left_element,
    }


def _resolution_comparison_term_mapping(
    term: ResolutionComparisonTerm,
) -> dict[str, Any]:
    return {
        "basis_id": term.basis_id,
        "coefficient": term.coefficient,
        "element": term.element,
    }


def _gap_target_bar_equivalence_projection(
    value: TargetBarResolutionEquivalence,
) -> dict[str, Any]:
    psi_on_basis: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    resolution_homotopy_on_basis: list[list[dict[str, Any]]] = [
        [] for _ in range(5)
    ]
    for trace in value.basis_traces:
        psi_on_basis[trace.degree].append(
            {
                "basis_id": trace.basis_id,
                "image": [
                    _bar_comparison_term_mapping(term) for term in trace.psi
                ],
            }
        )
        resolution_homotopy_on_basis[trace.degree].append(
            {
                "basis_id": trace.basis_id,
                "image": {
                    "degree": trace.degree + 1,
                    "terms": [
                        _resolution_comparison_term_mapping(term)
                        for term in trace.resolution_homotopy
                    ],
                },
            }
        )
    return {
        "bar_homotopy_algorithm": value.bar_homotopy_algorithm,
        "phi_algorithm": value.phi_algorithm,
        "phi_on_queries": [
            {
                "bar_homotopy": [
                    _bar_comparison_term_mapping(term)
                    for term in trace.bar_homotopy
                ],
                "group_tuple": list(trace.group_tuple),
                "image": {
                    "degree": len(trace.group_tuple),
                    "terms": [
                        _resolution_comparison_term_mapping(term)
                        for term in trace.image
                    ],
                },
            }
            for trace in value.phi_traces
        ],
        "psi_on_basis": psi_on_basis,
        "queried_bar_tuples": [
            list(group_tuple) for group_tuple in value.queried_bar_tuples
        ],
        "resolution_homotopy_on_basis": resolution_homotopy_on_basis,
    }


def _gap_inclusion_projection_from_certificate(
    certificate: InclusionChainMapCertificate,
) -> dict[str, Any]:
    return {
        "bar_comparison_traces": [
            {
                "degree": trace.degree,
                "source_basis_id": trace.source_basis_id,
                "source_psi": {
                    "degree": trace.degree,
                    "terms": [
                        _bar_comparison_term_mapping(term)
                        for term in trace.source_psi
                    ],
                },
                "target_phi_input": {
                    "degree": trace.degree,
                    "terms": [
                        _bar_comparison_term_mapping(term)
                        for term in trace.transported_bar
                    ],
                },
                "target_phi_output": {
                    "degree": trace.degree,
                    "terms": [
                        _resolution_comparison_term_mapping(term)
                        for term in trace.target_phi
                    ],
                },
            }
            for trace in certificate.bar_comparison_traces
        ],
        "chain_map_algorithm": certificate.chain_map_algorithm,
        "diagnostic_backend": certificate.diagnostic_backend,
        "diagnostic_maps": [
            _matrix_mapping(matrix) for matrix in certificate.diagnostic_maps
        ],
        "source_element_images": list(certificate.source_element_images),
        "target_bar_equivalence": _gap_target_bar_equivalence_projection(
            certificate.target_bar_equivalence
        ),
    }


def gap_inclusion_projection_digest(
    value: Mapping[str, Any] | InclusionChainMapCertificate,
) -> str:
    """Digest exactly the raw GAP fields that determine an inclusion map."""

    if isinstance(value, InclusionChainMapCertificate):
        projection = _gap_inclusion_projection_from_certificate(value)
    elif isinstance(value, Mapping):
        missing = [
            field for field in _GAP_INCLUSION_PROJECTION_FIELDS if field not in value
        ]
        if missing:
            raise ValueError(
                "raw GAP inclusion projection is missing fields: "
                + ", ".join(missing)
            )
        projection = {
            field: value[field] for field in _GAP_INCLUSION_PROJECTION_FIELDS
        }
    else:
        raise TypeError(
            "GAP inclusion projection digest requires a mapping or typed certificate"
        )
    return _task5_domain_digest(
        "task5-gap-inclusion-projection-v1", projection
    )


def _inclusion_core(certificate: InclusionChainMapCertificate) -> dict[str, Any]:
    return {
        "affine_pcp_certificate_digest": certificate.affine_pcp_certificate_digest,
        "bar_comparison_traces": [
            {
                "degree": trace.degree,
                "source_basis_id": trace.source_basis_id,
                "source_psi": [
                    {
                        "coefficient": term.coefficient,
                        "group_tuple": list(term.group_tuple),
                        "left_element": term.left_element,
                    }
                    for term in trace.source_psi
                ],
                "target_phi": [
                    {
                        "basis_id": term.basis_id,
                        "coefficient": term.coefficient,
                        "element": term.element,
                    }
                    for term in trace.target_phi
                ],
                "transported_bar": [
                    {
                        "coefficient": term.coefficient,
                        "group_tuple": list(term.group_tuple),
                        "left_element": term.left_element,
                    }
                    for term in trace.transported_bar
                ],
            }
            for trace in certificate.bar_comparison_traces
        ],
        "chain_map_algorithm": certificate.chain_map_algorithm,
        "diagnostic_backend": certificate.diagnostic_backend,
        "diagnostic_maps": [
            _matrix_mapping(matrix) for matrix in certificate.diagnostic_maps
        ],
        "diagnostic_outcome": certificate.diagnostic_outcome,
        "diagnostic_residue_digests": list(certificate.diagnostic_residue_digests),
        "inclusion_id": certificate.inclusion_id,
        "gap_inclusion_projection_digest": (
            certificate.gap_inclusion_projection_digest
        ),
        "launcher_attestation": launcher_execution_attestation_mapping(
            certificate.launcher_attestation
        ),
        "literal_element_digest": certificate.literal_element_digest,
        "literal_stabilizer_digest": certificate.literal_stabilizer_digest,
        "maps": [_matrix_mapping(matrix) for matrix in certificate.maps],
        "source_bar_equivalence_id": certificate.source_bar_equivalence_id,
        "source_element_images": list(certificate.source_element_images),
        "source_resolution": free_resolution_mapping(certificate.source_resolution),
        "source_resolution_id": certificate.source_resolution_id,
        "target_resolution": free_resolution_mapping(certificate.target_resolution),
        "target_resolution_id": certificate.target_resolution_id,
        "target_bar_equivalence": target_bar_equivalence_mapping(
            certificate.target_bar_equivalence
        ),
        "target_bar_equivalence_id": certificate.target_bar_equivalence_id,
        "transported_inclusion_digest": certificate.transported_inclusion_digest,
    }


def inclusion_chain_map_digest(certificate: InclusionChainMapCertificate) -> str:
    return _task5_domain_digest(
        "task5-inclusion-chain-map-certificate-v1", _inclusion_core(certificate)
    )


def inclusion_chain_map_mapping(certificate: InclusionChainMapCertificate) -> dict[str, Any]:
    return {
        **_inclusion_core(certificate),
        "certificate_id": certificate.certificate_id,
        "record_type": "inclusion-chain-map-certificate",
        "schema_version": 1,
    }


def _parse_target_bar_equivalence(
    value: Any, path: str
) -> TargetBarResolutionEquivalence:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _fields(
        value,
        {
            "bar_homotopy_algorithm",
            "basis_traces",
            "equivalence_id",
            "lookahead_boundary",
            "phi_algorithm",
            "phi_traces",
            "queried_bar_tuples",
            "record_type",
            "schema_version",
            "target_resolution_id",
        },
        path,
    )
    if (
        value["record_type"] != "target-bar-resolution-equivalence"
        or value["schema_version"] != 1
    ):
        raise ValueError(f"{path}: unsupported target bar-equivalence schema")
    for field in ("basis_traces", "phi_traces", "queried_bar_tuples"):
        if not isinstance(value[field], list):
            raise TypeError(f"{path}.{field}: expected array")

    def parse_bar_terms(raw: Any, term_path: str) -> tuple[BarComparisonTerm, ...]:
        if not isinstance(raw, list):
            raise TypeError(f"{term_path}: expected array")
        terms = []
        for index, item in enumerate(raw):
            item_path = f"{term_path}[{index}]"
            if not isinstance(item, Mapping):
                raise TypeError(f"{item_path}: expected object")
            _fields(
                item,
                {"coefficient", "group_tuple", "left_element"},
                item_path,
            )
            if not isinstance(item["group_tuple"], list):
                raise TypeError(f"{item_path}.group_tuple: expected array")
            terms.append(
                BarComparisonTerm(
                    item["left_element"],
                    tuple(item["group_tuple"]),
                    item["coefficient"],
                )
            )
        return tuple(terms)

    def parse_resolution_terms(
        raw: Any, term_path: str
    ) -> tuple[ResolutionComparisonTerm, ...]:
        if not isinstance(raw, list):
            raise TypeError(f"{term_path}: expected array")
        terms = []
        for index, item in enumerate(raw):
            item_path = f"{term_path}[{index}]"
            if not isinstance(item, Mapping):
                raise TypeError(f"{item_path}: expected object")
            _fields(item, {"basis_id", "coefficient", "element"}, item_path)
            terms.append(
                ResolutionComparisonTerm(
                    item["basis_id"], item["element"], item["coefficient"]
                )
            )
        return tuple(terms)

    basis = []
    for index, item in enumerate(value["basis_traces"]):
        item_path = f"{path}.basis_traces[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(
            item,
            {"basis_id", "degree", "psi", "resolution_homotopy"},
            item_path,
        )
        basis.append(
            TargetResolutionBasisTrace(
                item["degree"],
                item["basis_id"],
                parse_bar_terms(item["psi"], f"{item_path}.psi"),
                parse_resolution_terms(
                    item["resolution_homotopy"],
                    f"{item_path}.resolution_homotopy",
                ),
            )
        )
    phi = []
    for index, item in enumerate(value["phi_traces"]):
        item_path = f"{path}.phi_traces[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(item, {"bar_homotopy", "group_tuple", "image"}, item_path)
        if not isinstance(item["group_tuple"], list):
            raise TypeError(f"{item_path}.group_tuple: expected array")
        phi.append(
            TargetBarPhiTrace(
                tuple(item["group_tuple"]),
                parse_resolution_terms(item["image"], f"{item_path}.image"),
                parse_bar_terms(
                    item["bar_homotopy"], f"{item_path}.bar_homotopy"
                ),
            )
        )
    if any(not isinstance(item, list) for item in value["queried_bar_tuples"]):
        raise TypeError(f"{path}.queried_bar_tuples: expected arrays")
    result = TargetBarResolutionEquivalence(
        value["target_resolution_id"],
        value["phi_algorithm"],
        value["bar_homotopy_algorithm"],
        tuple(basis),
        tuple(phi),
        tuple(tuple(item) for item in value["queried_bar_tuples"]),
        _parse_matrix(value["lookahead_boundary"], f"{path}.lookahead_boundary"),
        value["equivalence_id"],
    )
    if target_bar_equivalence_digest(result) != result.equivalence_id:
        raise ValueError(f"{path}.equivalence_id: does not bind payload")
    return result


def loads_inclusion_chain_map_certificate(
    data: bytes,
    authority: Task5VerificationAuthority,
    *,
    source_equivalence: object | None = None,
    allow_diagnostic: bool = False,
    trusted_release_attestation: LauncherExecutionAttestation | None = None,
) -> InclusionChainMapCertificate:
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError(
            "inclusion loader requires a caller-supplied Task5VerificationAuthority"
        )
    if type(allow_diagnostic) is not bool:
        raise TypeError("allow_diagnostic must be a boolean")
    cache_key = None
    if (
        source_equivalence is None
        and trusted_release_attestation is None
        and type(data) is bytes
    ):
        cache_key = (data, authority, allow_diagnostic)
        cached = _INCLUSION_CERTIFICATE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    value = _strict_json(data)
    wrapper_bar_equivalence: Mapping[str, Any] | None = None
    if value.get("record_type") == "task5-p4mm-fixture":
        _fields(value, {"bar_equivalence", "inclusion", "record_type", "schema_version"}, "$fixture")
        if value["schema_version"] != 1:
            raise ValueError("$fixture: unsupported fixture schema")
        if not isinstance(value["inclusion"], Mapping):
            raise TypeError("$fixture.inclusion: expected object")
        if not isinstance(value["bar_equivalence"], Mapping):
            raise TypeError("$fixture.bar_equivalence: expected object")
        if source_equivalence is not None:
            raise ValueError(
                "inclusion loader received both embedded and separately verified bar equivalence"
            )
        wrapper_bar_equivalence = value["bar_equivalence"]
        value = value["inclusion"]
    elif source_equivalence is None:
        raise ValueError(
            "inclusion loader requires an embedded or separately verified bar equivalence"
        )
    _fields(
        value,
        {
            "affine_pcp_certificate_digest",
            "bar_comparison_traces",
            "certificate_id",
            "chain_map_algorithm",
            "diagnostic_backend",
            "diagnostic_maps",
            "diagnostic_outcome",
            "diagnostic_residue_digests",
            "gap_inclusion_projection_digest",
            "inclusion_id",
            "launcher_attestation",
            "literal_element_digest",
            "literal_stabilizer_digest",
            "maps",
            "record_type",
            "schema_version",
            "source_bar_equivalence_id",
            "source_element_images",
            "source_resolution",
            "source_resolution_id",
            "target_resolution",
            "target_resolution_id",
            "target_bar_equivalence",
            "target_bar_equivalence_id",
            "transported_inclusion_digest",
        },
        "$inclusion",
    )
    if value["record_type"] != "inclusion-chain-map-certificate" or value["schema_version"] != 1:
        raise ValueError("$inclusion: unsupported inclusion schema")
    traces = []
    if not isinstance(value["bar_comparison_traces"], list):
        raise TypeError("$inclusion.bar_comparison_traces: expected array")
    for index, item in enumerate(value["bar_comparison_traces"]):
        item_path = f"$inclusion.bar_comparison_traces[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _fields(
            item,
            {"degree", "source_basis_id", "source_psi", "target_phi", "transported_bar"},
            item_path,
        )

        def parse_bar_terms(raw: Any, field: str) -> tuple[BarComparisonTerm, ...]:
            if not isinstance(raw, list):
                raise TypeError(f"{item_path}.{field}: expected array")
            parsed = []
            for term_index, term in enumerate(raw):
                term_path = f"{item_path}.{field}[{term_index}]"
                if not isinstance(term, Mapping):
                    raise TypeError(f"{term_path}: expected object")
                _fields(term, {"coefficient", "group_tuple", "left_element"}, term_path)
                if not isinstance(term["group_tuple"], list):
                    raise TypeError(f"{term_path}.group_tuple: expected array")
                parsed.append(
                    BarComparisonTerm(
                        term["left_element"],
                        tuple(term["group_tuple"]),
                        term["coefficient"],
                    )
                )
            return tuple(parsed)

        if not isinstance(item["target_phi"], list):
            raise TypeError(f"{item_path}.target_phi: expected array")
        target_terms = []
        for term_index, term in enumerate(item["target_phi"]):
            term_path = f"{item_path}.target_phi[{term_index}]"
            if not isinstance(term, Mapping):
                raise TypeError(f"{term_path}: expected object")
            _fields(term, {"basis_id", "coefficient", "element"}, term_path)
            target_terms.append(
                ResolutionComparisonTerm(
                    term["basis_id"], term["element"], term["coefficient"]
                )
            )
        traces.append(
            BarComparisonBasisTrace(
                item["degree"],
                item["source_basis_id"],
                parse_bar_terms(item["source_psi"], "source_psi"),
                parse_bar_terms(item["transported_bar"], "transported_bar"),
                tuple(target_terms),
            )
        )
    certificate = InclusionChainMapCertificate(
        value["inclusion_id"],
        value["affine_pcp_certificate_digest"],
        value["literal_stabilizer_digest"],
        value["literal_element_digest"],
        value["transported_inclusion_digest"],
        value["source_resolution_id"],
        value["target_resolution_id"],
        _parse_resolution(value["source_resolution"], "$inclusion.source_resolution"),
        _parse_resolution(value["target_resolution"], "$inclusion.target_resolution"),
        tuple(value["source_element_images"]),
        tuple(_parse_matrix(matrix, f"$inclusion.maps[{index}]") for index, matrix in enumerate(value["maps"])),
        value["chain_map_algorithm"],
        value["source_bar_equivalence_id"],
        value["target_bar_equivalence_id"],
        _parse_target_bar_equivalence(
            value["target_bar_equivalence"], "$inclusion.target_bar_equivalence"
        ),
        _parse_launcher_execution_attestation(
            value["launcher_attestation"],
            "$inclusion.launcher_attestation",
            allow_diagnostic=allow_diagnostic,
            trusted_release_attestation=trusted_release_attestation,
        ),
        value["gap_inclusion_projection_digest"],
        tuple(traces),
        value["diagnostic_backend"],
        tuple(
            _parse_matrix(matrix, f"$inclusion.diagnostic_maps[{index}]")
            for index, matrix in enumerate(value["diagnostic_maps"])
        ),
        value["diagnostic_outcome"],
        tuple(value["diagnostic_residue_digests"]),
        value["certificate_id"],
    )
    if wrapper_bar_equivalence is not None or source_equivalence is not None:
        # Import locally to avoid the module-level bar_evaluator -> cochains
        # dependency cycle.  The complete nested record is strict-parsed and
        # every phi/psi/homotopy identity is replayed before inclusion use.
        from .bar_evaluator import _parse_equivalence, verify_bar_resolution_equivalence

        equivalence = (
            _parse_equivalence(
                wrapper_bar_equivalence, "$fixture.bar_equivalence"
            )
            if wrapper_bar_equivalence is not None
            else source_equivalence
        )
        equivalence_report = verify_bar_resolution_equivalence(
            equivalence, authority
        )
        if not equivalence_report.valid:
            raise ValueError(
                "$fixture: embedded source bar equivalence failed replay: "
                + "; ".join(
                    f"{issue.code}: {issue.detail}"
                    for issue in equivalence_report.issues
                )
            )
        if (
            equivalence.equivalence_id != certificate.source_bar_equivalence_id
            or equivalence.resolution != certificate.source_resolution
        ):
            raise ValueError(
                "$fixture: inclusion does not bind the embedded source bar equivalence"
            )
        expected_psi = [
            (
                item.degree,
                item.basis_id,
                tuple(
                    BarComparisonTerm(
                        term.left_element,
                        term.group_tuple,
                        term.coefficient,
                    )
                    for term in item.image.terms
                ),
            )
            for item in equivalence.psi_on_basis
        ]
        actual_psi = [
            (trace.degree, trace.source_basis_id, trace.source_psi)
            for trace in certificate.bar_comparison_traces
        ]
        if expected_psi != actual_psi:
            raise ValueError("$fixture: inclusion traces differ from embedded source psi")
    if inclusion_chain_map_digest(certificate) != certificate.certificate_id:
        raise ValueError("$inclusion.certificate_id: does not bind certificate")
    report = verify_inclusion_chain_map(
        certificate,
        authority,
        require_release=not allow_diagnostic,
        trusted_release_attestation=trusted_release_attestation,
    )
    if not report.valid:
        raise ValueError(
            "$inclusion: certificate failed independent replay: "
            + "; ".join(f"{issue.code}: {issue.detail}" for issue in report.issues)
        )
    if cache_key is not None:
        _INCLUSION_CERTIFICATE_CACHE[cache_key] = certificate
    return certificate


def dumps_inclusion_chain_map_certificate(certificate: InclusionChainMapCertificate) -> bytes:
    return _canonical_json(inclusion_chain_map_mapping(certificate))


def make_inclusion_chain_map_certificate(
    *,
    inclusion_id: str,
    literal_stabilizer_digest: str,
    literal_element_digest: str,
    transported_inclusion_digest: str,
    source_resolution: FreeResolutionCertificate,
    target_resolution: FreeResolutionCertificate,
    source_element_images: Sequence[str],
    maps: Sequence[SparseGroupRingMatrix],
    source_bar_equivalence_id: str,
    target_bar_equivalence: TargetBarResolutionEquivalence,
    launcher_attestation: LauncherExecutionAttestation,
    bar_comparison_traces: Sequence[BarComparisonBasisTrace],
    diagnostic_backend: str,
    diagnostic_maps: Sequence[SparseGroupRingMatrix],
    diagnostic_outcome: str,
    diagnostic_residue_digests: Sequence[str],
) -> InclusionChainMapCertificate:
    provisional = InclusionChainMapCertificate(
        inclusion_id,
        target_resolution.affine_pcp_certificate_digest,
        literal_stabilizer_digest,
        literal_element_digest,
        transported_inclusion_digest,
        source_resolution.resolution_id,
        target_resolution.resolution_id,
        source_resolution,
        target_resolution,
        tuple(source_element_images),
        tuple(maps),
        "hap-1.70-bar-phi-target-inclusion-psi-source",
        source_bar_equivalence_id,
        target_bar_equivalence.equivalence_id,
        target_bar_equivalence,
        launcher_attestation,
        launcher_attestation.gap_inclusion_projection_digest,
        tuple(bar_comparison_traces),
        diagnostic_backend,
        tuple(diagnostic_maps),
        diagnostic_outcome,
        tuple(diagnostic_residue_digests),
        "sha256:" + "0" * 64,
    )
    return InclusionChainMapCertificate(
        provisional.inclusion_id,
        provisional.affine_pcp_certificate_digest,
        provisional.literal_stabilizer_digest,
        provisional.literal_element_digest,
        provisional.transported_inclusion_digest,
        provisional.source_resolution_id,
        provisional.target_resolution_id,
        provisional.source_resolution,
        provisional.target_resolution,
        provisional.source_element_images,
        provisional.maps,
        provisional.chain_map_algorithm,
        provisional.source_bar_equivalence_id,
        provisional.target_bar_equivalence_id,
        provisional.target_bar_equivalence,
        provisional.launcher_attestation,
        provisional.gap_inclusion_projection_digest,
        provisional.bar_comparison_traces,
        provisional.diagnostic_backend,
        provisional.diagnostic_maps,
        provisional.diagnostic_outcome,
        provisional.diagnostic_residue_digests,
        inclusion_chain_map_digest(provisional),
    )


def _entries(matrix: SparseGroupRingMatrix) -> dict[tuple[int, int], tuple[SparseGroupRingTerm, ...]]:
    return {(entry.row, entry.column): entry.terms for entry in matrix.entries}


def _onsite_time_reversal_normal_form(
    resolution: FreeResolutionCertificate,
    element: str,
) -> tuple[str, int]:
    """Split a canonical ``G x C2^T`` word into its two exact factors."""

    if not resolution.group_id.endswith("+onsite-T"):
        return element, 0
    if element == "T":
        return "1", 1
    if element.endswith("+T"):
        spatial = element[:-2]
        if not spatial or spatial == "1" or "T" in spatial:
            raise ValueError("invalid canonical onsite-time-reversal normal form")
        return spatial, 1
    if "T" in element:
        raise ValueError("invalid canonical onsite-time-reversal normal form")
    return element, 0


def _normal_key(resolution: FreeResolutionCertificate, element: str) -> object:
    if resolution.finite_group is not None:
        try:
            return resolution.finite_group.element_order.index(element)
        except ValueError as error:
            raise ValueError(f"finite group element {element!r} is absent") from error
    graded = resolution.group_id.endswith("+onsite-T")
    spatial, time_bit = _onsite_time_reversal_normal_form(resolution, element)
    _pcp_word_coordinates(spatial, resolution.affine_pcp_certificate.pcp_normal_form)
    affine = _evaluate_pcp_word(
        spatial, resolution.affine_pcp_certificate.pcp_normal_form
    )
    if graded:
        return (affine.matrix, affine.translation, time_bit)
    return (affine.matrix, affine.translation)


def _multiply_keys(resolution: FreeResolutionCertificate, left: str, right: str) -> object:
    if resolution.finite_group is not None:
        table = resolution.finite_group
        li = table.element_order.index(left)
        ri = table.element_order.index(right)
        return table.multiplication_table[li][ri]
    authority = resolution.affine_pcp_certificate.pcp_normal_form
    graded = resolution.group_id.endswith("+onsite-T")
    left_spatial, left_time = _onsite_time_reversal_normal_form(resolution, left)
    right_spatial, right_time = _onsite_time_reversal_normal_form(
        resolution, right
    )
    left_affine = _evaluate_pcp_word(left_spatial, authority)
    right_affine = _evaluate_pcp_word(right_spatial, authority)
    # Cryst's right-action matrices give C(left*right)=C(right) o C(left).
    product = _compose_affine(right_affine, left_affine)
    if graded:
        return (product.matrix, product.translation, left_time ^ right_time)
    return (product.matrix, product.translation)


def _compose(
    lower: SparseGroupRingMatrix,
    upper: SparseGroupRingMatrix,
    resolution: FreeResolutionCertificate,
) -> dict[tuple[int, int, object], int]:
    if lower.column_count != upper.row_count:
        raise ValueError("group-ring matrix dimensions differ")
    a = _entries(lower)
    b = _entries(upper)
    result: dict[tuple[int, int, object], int] = {}
    for row in range(lower.row_count):
        for middle in range(lower.column_count):
            lower_terms = a.get((row, middle), ())
            if not lower_terms:
                continue
            for column in range(upper.column_count):
                upper_terms = b.get((middle, column), ())
                for upper_term in upper_terms:
                    for lower_term in lower_terms:
                        key = (
                            row,
                            column,
                            _multiply_keys(resolution, upper_term.element, lower_term.element),
                        )
                        result[key] = result.get(key, 0) + upper_term.coefficient * lower_term.coefficient
    return {key: value for key, value in result.items() if value}


def _issue(code: str, detail: str) -> VerificationIssue:
    return VerificationIssue(code, detail)


def verify_resolution(
    certificate: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
) -> VerificationReport:
    if not isinstance(certificate, FreeResolutionCertificate):
        raise TypeError("certificate must be a FreeResolutionCertificate")
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("authority must be a caller-supplied Task5VerificationAuthority")
    issues: list[VerificationIssue] = []
    try:
        expected_lock, expected_environment, expected_runtime = _task5_backend_binding()
        authority_claims_release = (
            authority.backend_lock_digest == expected_lock
            and authority.backend_environment_id == expected_environment
            and authority.runtime_provenance_digest == expected_runtime
        )
        if authority_claims_release and (
            certificate.backend_lock_digest != expected_lock
            or certificate.backend_environment_id != expected_environment
            or certificate.runtime_provenance_digest != expected_runtime
        ):
            issues.append(_issue(
                "backend_environment_mismatch",
                "resolution does not bind the tracked GAP/HAP lock and Task 5 API closure",
            ))
    except (TypeError, ValueError) as error:
        issues.append(_issue("backend_environment_mismatch", str(error)))
    if (
        certificate.backend_lock_digest != authority.backend_lock_digest
        or certificate.backend_environment_id != authority.backend_environment_id
        or certificate.runtime_provenance_digest != authority.runtime_provenance_digest
    ):
        issues.append(_issue(
            "trusted_backend_authority_mismatch",
            "resolution differs from the caller-supplied runtime authority",
        ))
    if (
        certificate.affine_pcp_certificate.catalogue_action_digest
        != authority.catalogue_action_digest
        or certificate.catalogue_record_digest != authority.catalogue_record_digest
        or certificate.affine_pcp_certificate_digest
        != authority.affine_pcp_certificate_digest
    ):
        issues.append(_issue(
            "trusted_catalogue_authority_mismatch",
            "resolution differs from the caller-supplied Task 4/catalogue authority",
        ))
    if certificate.affine_pcp_certificate_digest != certificate.affine_pcp_certificate.certificate_digest:
        issues.append(_issue("affine_pcp_digest_mismatch", "resolution does not bind its embedded Task 4 certificate"))
    try:
        if affine_pcp_certificate_digest(certificate.affine_pcp_certificate) != certificate.affine_pcp_certificate.certificate_digest:
            issues.append(_issue("affine_pcp_digest_mismatch", "embedded Task 4 certificate content hash fails"))
    except (TypeError, ValueError) as error:
        issues.append(_issue("affine_pcp_digest_mismatch", str(error)))
    if free_resolution_digest(certificate) != certificate.resolution_id:
        issues.append(_issue("resolution_digest_mismatch", "resolution ID does not bind the payload"))
    for degree, basis in enumerate(certificate.basis):
        expected = tuple(f"c{degree}:{index}" for index in range(len(basis)))
        if basis != expected or any(_BASIS_RE.fullmatch(item) is None for item in basis):
            issues.append(_issue("noncanonical_basis_id", f"degree {degree} basis IDs are not canonical"))
    expected_degree_five = tuple(
        f"c5:{index}" for index in range(len(certificate.degree_five_basis))
    )
    if certificate.degree_five_basis != expected_degree_five:
        issues.append(_issue("noncanonical_basis_id", "degree 5 basis IDs are not canonical"))
    for degree, boundary in enumerate(certificate.boundaries, start=1):
        expected_shape = (len(certificate.basis[degree - 1]), len(certificate.basis[degree]))
        if (boundary.row_count, boundary.column_count) != expected_shape:
            issues.append(_issue("boundary_shape_mismatch", f"degree {degree} boundary has wrong shape"))
        for entry in boundary.entries:
            for term in entry.terms:
                try:
                    _normal_key(certificate, term.element)
                except (TypeError, ValueError) as error:
                    issues.append(_issue("noncanonical_group_normal_form", f"degree {degree}: {error}"))
    lookahead = certificate.lookahead_boundary
    if (lookahead.row_count, lookahead.column_count) != (
        len(certificate.basis[4]),
        len(certificate.degree_five_basis),
    ):
        issues.append(_issue("boundary_shape_mismatch", "degree 5 boundary has wrong shape"))
    for entry in lookahead.entries:
        for term in entry.terms:
            try:
                _normal_key(certificate, term.element)
            except (TypeError, ValueError) as error:
                issues.append(_issue("noncanonical_group_normal_form", f"degree 5: {error}"))
    checked = 0
    all_boundaries = certificate.boundaries + (certificate.lookahead_boundary,)
    for degree in range(2, certificate.max_degree + 2):
        checked += 1
        try:
            residue = _compose(
                all_boundaries[degree - 2],
                all_boundaries[degree - 1],
                certificate,
            )
        except (TypeError, ValueError) as error:
            issues.append(_issue("boundary_not_square_zero", f"degree {degree}: {error}"))
            continue
        if residue:
            issues.append(_issue("boundary_not_square_zero", f"degree {degree}: nonzero sparse residue"))
    return VerificationReport(not issues, tuple(issues), checked)


def _map_source_element(certificate: InclusionChainMapCertificate, element: str) -> str:
    table = certificate.source_resolution.finite_group
    if table is None:
        raise ValueError("inclusion source must carry a finite group table")
    try:
        return certificate.source_element_images[table.element_order.index(element)]
    except (ValueError, IndexError) as error:
        raise ValueError("source element lacks a transported target image") from error


def _inclusion_left(
    certificate: InclusionChainMapCertificate,
    degree: int,
    maps: Sequence[SparseGroupRingMatrix] | None = None,
) -> dict[tuple[int, int, object], int]:
    source_boundary = certificate.source_resolution.boundaries[degree - 1]
    selected = certificate.maps if maps is None else tuple(maps)
    lower_map = selected[degree - 1]
    source_entries = _entries(source_boundary)
    map_entries = _entries(lower_map)
    target = certificate.target_resolution
    result: dict[tuple[int, int, object], int] = {}
    for row in range(lower_map.row_count):
        for middle in range(lower_map.column_count):
            mapped = map_entries.get((row, middle), ())
            for column in range(source_boundary.column_count):
                source = source_entries.get((middle, column), ())
                for source_term in source:
                    image = _map_source_element(certificate, source_term.element)
                    for map_term in mapped:
                        key = (row, column, _multiply_keys(target, image, map_term.element))
                        result[key] = result.get(key, 0) + source_term.coefficient * map_term.coefficient
    return {key: value for key, value in result.items() if value}


def _inclusion_right(
    certificate: InclusionChainMapCertificate,
    degree: int,
    maps: Sequence[SparseGroupRingMatrix] | None = None,
) -> dict[tuple[int, int, object], int]:
    target_boundary = certificate.target_resolution.boundaries[degree - 1]
    selected = certificate.maps if maps is None else tuple(maps)
    upper_map = selected[degree]
    return _compose(target_boundary, upper_map, certificate.target_resolution)


def _transport_bar_term(
    certificate: InclusionChainMapCertificate, term: BarComparisonTerm
) -> BarComparisonTerm:
    return BarComparisonTerm(
        _map_source_element(certificate, term.left_element),
        tuple(_map_source_element(certificate, item) for item in term.group_tuple),
        term.coefficient,
    )


def _reconstruct_comparison_maps(
    certificate: InclusionChainMapCertificate,
) -> tuple[SparseGroupRingMatrix, ...]:
    expected_trace_order = tuple(
        (degree, basis_id)
        for degree, basis in enumerate(certificate.source_resolution.basis)
        for basis_id in basis
    )
    actual_trace_order = tuple(
        (trace.degree, trace.source_basis_id)
        for trace in certificate.bar_comparison_traces
    )
    if actual_trace_order != expected_trace_order:
        raise ValueError("bar-comparison trace coverage differs from the source basis")
    by_degree: list[list[SparseGroupRingEntry]] = [[] for _ in range(5)]
    source_column = {basis_id: index for basis in certificate.source_resolution.basis for index, basis_id in enumerate(basis)}
    for trace in certificate.bar_comparison_traces:
        transported = tuple(sorted(_transport_bar_term(certificate, term) for term in trace.source_psi))
        if transported != trace.transported_bar:
            raise ValueError("transported bar trace differs from the literal inclusion")
        collected: dict[tuple[int, str], int] = {}
        for term in trace.target_phi:
            match = _BASIS_RE.fullmatch(term.basis_id)
            if match is None or int(match.group(1)) != trace.degree:
                raise ValueError("target phi trace has a wrong-degree basis term")
            row = int(match.group(2))
            if row >= len(certificate.target_resolution.basis[trace.degree]):
                raise ValueError("target phi trace basis is outside the target resolution")
            _normal_key(certificate.target_resolution, term.element)
            key = (row, term.element)
            collected[key] = collected.get(key, 0) + term.coefficient
        column = source_column[trace.source_basis_id]
        rows: dict[int, list[SparseGroupRingTerm]] = {}
        for (row, element), coefficient in sorted(collected.items()):
            if coefficient:
                rows.setdefault(row, []).append(SparseGroupRingTerm(element, coefficient))
        for row, terms in sorted(rows.items()):
            by_degree[trace.degree].append(
                SparseGroupRingEntry(row, column, tuple(sorted(terms)))
            )
    return tuple(
        SparseGroupRingMatrix(
            len(certificate.target_resolution.basis[degree]),
            len(certificate.source_resolution.basis[degree]),
            tuple(sorted(entries, key=lambda item: (item.row, item.column))),
        )
        for degree, entries in enumerate(by_degree)
    )


def _residue_key_mapping(
    certificate: InclusionChainMapCertificate, key: object
) -> object:
    target = certificate.target_resolution
    if target.finite_group is not None:
        return target.finite_group.element_order[int(key)]
    if target.group_id.endswith("+onsite-T"):
        matrix, translation, time_bit = key
    else:
        matrix, translation = key
        time_bit = None
    result = {
        "matrix": [list(row) for row in matrix],
        "translation": list(translation),
    }
    if time_bit is not None:
        result["onsite_time_reversal"] = time_bit
    return result


def _diagnostic_residue_digest(
    certificate: InclusionChainMapCertificate, degree: int
) -> str:
    left = _inclusion_left(certificate, degree, certificate.diagnostic_maps)
    right = _inclusion_right(certificate, degree, certificate.diagnostic_maps)

    def mapping(value: Mapping[tuple[int, int, object], int]) -> list[dict[str, Any]]:
        rows = [
            {
                "coefficient": coefficient,
                "column": column,
                "element": _residue_key_mapping(certificate, element),
                "row": row,
            }
            for (row, column, element), coefficient in value.items()
        ]
        return sorted(rows, key=_canonical_json)

    return _task5_domain_digest(
        "task5-diagnostic-chain-residue-v1",
        {"degree": degree, "left": mapping(left), "right": mapping(right)},
    )


def diagnostic_residue_digests(
    certificate: InclusionChainMapCertificate,
) -> tuple[str, ...]:
    return tuple(_diagnostic_residue_digest(certificate, degree) for degree in range(1, 5))


def verify_inclusion_chain_map(
    certificate: InclusionChainMapCertificate,
    authority: Task5VerificationAuthority,
    *,
    require_release: bool = True,
    trusted_release_attestation: LauncherExecutionAttestation | None = None,
) -> VerificationReport:
    if not isinstance(certificate, InclusionChainMapCertificate):
        raise TypeError("certificate must be an InclusionChainMapCertificate")
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("authority must be a caller-supplied Task5VerificationAuthority")
    if type(require_release) is not bool:
        raise TypeError("require_release must be a boolean")
    if (
        trusted_release_attestation is not None
        and not isinstance(
            trusted_release_attestation, LauncherExecutionAttestation
        )
    ):
        raise TypeError(
            "trusted_release_attestation must be a launcher attestation or None"
        )
    issues: list[VerificationIssue] = []
    source_report = verify_resolution(certificate.source_resolution, authority)
    target_report = verify_resolution(certificate.target_resolution, authority)
    issues.extend(_issue("source_resolution_invalid", issue.detail) for issue in source_report.issues)
    issues.extend(_issue("target_resolution_invalid", issue.detail) for issue in target_report.issues)
    if certificate.source_resolution_id != certificate.source_resolution.resolution_id or certificate.target_resolution_id != certificate.target_resolution.resolution_id:
        issues.append(_issue("resolution_binding_mismatch", "chain map resolution IDs do not bind embedded resolutions"))
    target_affine = certificate.target_resolution.affine_pcp_certificate
    if certificate.affine_pcp_certificate_digest != target_affine.certificate_digest:
        issues.append(_issue("affine_pcp_digest_mismatch", "chain map does not bind target affine-PCP certificate"))
    trusted_inclusion = next(
        (item for item in authority.inclusions if item.inclusion_id == certificate.inclusion_id),
        None,
    )
    attestation = certificate.launcher_attestation
    if (
        launcher_execution_attestation_digest(attestation)
        != attestation.attestation_id
        or attestation.exit_status != 0
        or trusted_inclusion is None
        or trusted_inclusion.launcher_attestation_id != attestation.attestation_id
    ):
        issues.append(_issue(
            "launcher_attestation_mismatch",
            "launcher execution attestation is invalid or absent from external authority",
        ))
    if require_release and (
        trusted_release_attestation is None
        or attestation is not trusted_release_attestation
        or not trusted_release_attestation.release_certified
        or trusted_release_attestation.runtime_manifest_digest is None
        or launcher_execution_attestation_digest(trusted_release_attestation)
        != trusted_release_attestation.attestation_id
    ):
        issues.append(_issue(
            "release_launcher_attestation_required",
            "release verification requires the external nonserialized "
            "locked-launcher attestation object",
        ))
    if (
        trusted_inclusion is None
        or trusted_inclusion.literal_stabilizer_digest != certificate.literal_stabilizer_digest
        or trusted_inclusion.literal_element_digest != certificate.literal_element_digest
        or trusted_inclusion.transported_inclusion_digest
        != certificate.transported_inclusion_digest
        or trusted_inclusion.source_bar_equivalence_id
        != certificate.source_bar_equivalence_id
        or trusted_inclusion.target_bar_equivalence_id
        != certificate.target_bar_equivalence_id
    ):
        issues.append(_issue(
            "trusted_inclusion_authority_mismatch",
            "inclusion differs from the caller-supplied catalogue authority",
        ))
    try:
        reconstructed_projection_digest = gap_inclusion_projection_digest(
            certificate
        )
    except (TypeError, ValueError) as error:
        reconstructed_projection_digest = None
        issues.append(
            _issue("gap_inclusion_projection_mismatch", str(error))
        )
    if reconstructed_projection_digest is not None and (
        trusted_inclusion is None
        or certificate.gap_inclusion_projection_digest
        != reconstructed_projection_digest
        or attestation.gap_inclusion_projection_digest
        != reconstructed_projection_digest
        or trusted_inclusion.gap_inclusion_projection_digest
        != reconstructed_projection_digest
    ):
        issues.append(
            _issue(
                "gap_inclusion_projection_mismatch",
                "raw GAP projection differs among the typed certificate, "
                "launcher record, or external inclusion authority",
            )
        )
    if trusted_inclusion is not None and (
        trusted_inclusion.diagnostic_backend != certificate.diagnostic_backend
        or trusted_inclusion.diagnostic_outcome != certificate.diagnostic_outcome
        or trusted_inclusion.diagnostic_residue_digests
        != certificate.diagnostic_residue_digests
    ):
        issues.append(_issue(
            "trusted_diagnostic_authority_mismatch",
            "diagnostic backend, outcome, or residues differ from external inclusion authority",
        ))
    transported = next(
        (item for item in target_affine.transported_stabilizers if item.inclusion_id == certificate.inclusion_id),
        None,
    )
    if (
        transported is None
        or transported.literal_stabilizer_digest
        != certificate.literal_stabilizer_digest
        or transported.literal_element_digest != certificate.literal_element_digest
    ):
        issues.append(_issue("literal_stabilizer_binding_mismatch", "literal stabilizer digest is not the Task 4 transported inclusion"))
    else:
        if (
            literal_element_authority_digest(transported.literal_elements)
            != certificate.literal_element_digest
        ):
            issues.append(_issue(
                "literal_stabilizer_binding_mismatch",
                "literal element digest does not bind the transported affine elements",
            ))
        if (
            transported_inclusion_authority_digest(transported)
            != certificate.transported_inclusion_digest
        ):
            issues.append(_issue(
                "transported_inclusion_digest_mismatch",
                "transported inclusion digest does not bind the embedded Task 4 inclusion",
            ))
        source_table = certificate.source_resolution.finite_group
        if source_table is None or source_table.multiplication_table != transported.multiplication_table or certificate.source_element_images != transported.pcp_images:
            issues.append(_issue("literal_stabilizer_binding_mismatch", "finite table or transported element images differ from Task 4"))
        else:
            try:
                pcp_authority = target_affine.pcp_normal_form
                for literal, word in zip(
                    transported.literal_elements,
                    transported.pcp_images,
                    strict=True,
                ):
                    if _evaluate_pcp_word(word, pcp_authority) != literal:
                        raise ValueError("a transported PCP word does not replay its literal affine")
                for left, row in enumerate(transported.multiplication_table):
                    for right, product in enumerate(row):
                        # Catalogue source words follow Cryst's right-action
                        # convention C(x*y)=C(y) o C(x).
                        actual = _compose_affine(
                            transported.literal_elements[right],
                            transported.literal_elements[left],
                        )
                        if actual != transported.literal_elements[product]:
                            raise ValueError("literal affine multiplication table does not replay")
            except (TypeError, ValueError) as error:
                issues.append(_issue("literal_stabilizer_binding_mismatch", str(error)))
    if certificate.chain_map_algorithm != "hap-1.70-bar-phi-target-inclusion-psi-source":
        issues.append(_issue(
            "chain_map_algorithm_mismatch",
            "authoritative inclusion must use the canonical bar-comparison algorithm",
        ))
    if (
        certificate.target_bar_equivalence_id
        != certificate.target_bar_equivalence.equivalence_id
        or certificate.target_bar_equivalence.target_resolution_id
        != certificate.target_resolution_id
        or target_bar_equivalence_digest(certificate.target_bar_equivalence)
        != certificate.target_bar_equivalence_id
    ):
        issues.append(_issue(
            "target_bar_equivalence_mismatch",
            "target equivalence does not bind the target resolution and certificate ID",
        ))
    else:
        from .bar_evaluator import verify_target_bar_resolution_equivalence

        target_equivalence_report = verify_target_bar_resolution_equivalence(
            certificate.target_bar_equivalence,
            certificate.target_resolution,
            certificate.bar_comparison_traces,
            authority,
        )
        issues.extend(
            _issue("target_bar_equivalence_not_verified", issue.detail)
            for issue in target_equivalence_report.issues
        )
    try:
        reconstructed = _reconstruct_comparison_maps(certificate)
        if reconstructed != certificate.maps:
            issues.append(_issue(
                "bar_comparison_reconstruction_mismatch",
                "maps are not phi_target composed with inclusion_bar and psi_source",
            ))
    except (TypeError, ValueError) as error:
        issues.append(_issue("bar_comparison_reconstruction_mismatch", str(error)))
    augmentation = certificate.maps[0]
    if augmentation != SparseGroupRingMatrix(
        1,
        1,
        (SparseGroupRingEntry(0, 0, (SparseGroupRingTerm("1", 1),)),),
    ):
        issues.append(_issue(
            "degree_zero_augmentation_mismatch",
            "degree-zero map must preserve the canonical augmentation generator",
        ))
    for degree, matrix in enumerate(certificate.maps):
        expected = (len(certificate.target_resolution.basis[degree]), len(certificate.source_resolution.basis[degree]))
        if (matrix.row_count, matrix.column_count) != expected:
            issues.append(_issue("chain_map_shape_mismatch", f"degree {degree} map has wrong shape"))
    for degree, matrix in enumerate(certificate.diagnostic_maps):
        expected = (len(certificate.target_resolution.basis[degree]), len(certificate.source_resolution.basis[degree]))
        if (matrix.row_count, matrix.column_count) != expected:
            issues.append(_issue("diagnostic_chain_map_shape_mismatch", f"degree {degree} diagnostic map has wrong shape"))
    checked = 0
    for degree in range(1, 5):
        checked += 1
        try:
            left = _inclusion_left(certificate, degree)
            right = _inclusion_right(certificate, degree)
        except (TypeError, ValueError) as error:
            issues.append(_issue("chain_map_not_commuting", f"degree {degree}: {error}"))
            continue
        if left != right:
            issues.append(_issue("chain_map_not_commuting", f"degree {degree}: sparse residues differ"))
    diagnostic_failures = []
    for degree in range(1, 5):
        try:
            left = _inclusion_left(
                certificate, degree, certificate.diagnostic_maps
            )
            right = _inclusion_right(
                certificate, degree, certificate.diagnostic_maps
            )
        except (TypeError, ValueError):
            diagnostic_failures.append(degree)
            continue
        if left != right:
            diagnostic_failures.append(degree)
    if (
        trusted_inclusion is not None
        and tuple(diagnostic_failures)
        != trusted_inclusion.diagnostic_failure_degrees
    ):
        issues.append(_issue(
            "trusted_diagnostic_authority_mismatch",
            "diagnostic failure degrees differ from external inclusion authority",
        ))
    expected_residues = None
    try:
        expected_residues = diagnostic_residue_digests(certificate)
        if certificate.diagnostic_residue_digests != expected_residues:
            issues.append(_issue(
                "backend_diagnostic_not_reproduced",
                "stored diagnostic residues differ from independent replay",
            ))
    except (TypeError, ValueError) as error:
        issues.append(_issue("backend_diagnostic_not_reproduced", str(error)))
    if diagnostic_failures:
        if certificate.diagnostic_outcome == "commuting":
            issues.append(_issue(
                "backend_diagnostic_not_reproduced",
                "a noncommuting diagnostic cannot be recorded as commuting",
            ))
        elif certificate.diagnostic_outcome.startswith(
            _OBSERVED_NONCOMMUTING_OUTCOME_PREFIX
        ) and expected_residues is not None:
            expected_outcome = task5_diagnostic_observed_outcome(
                certificate.diagnostic_backend,
                diagnostic_failures,
                expected_residues,
            )
            if certificate.diagnostic_outcome != expected_outcome:
                issues.append(_issue(
                    "backend_diagnostic_not_reproduced",
                    "observed outcome does not bind the replayed failures and residues",
                ))
    elif not diagnostic_failures and certificate.diagnostic_outcome != "commuting":
        issues.append(_issue(
            "backend_diagnostic_not_reproduced",
            "a commuting diagnostic must be recorded as commuting",
        ))
    if inclusion_chain_map_digest(certificate) != certificate.certificate_id:
        issues.append(_issue("chain_map_digest_mismatch", "chain-map ID does not bind payload"))
    return VerificationReport(not issues, tuple(issues), checked)


def _gf2_identity(size: int) -> MatrixGF2:
    return MatrixGF2(
        tuple(tuple(int(row == column) for column in range(size)) for row in range(size)),
        column_count=size,
    )


def _gf2_matmul(left: MatrixGF2, right: MatrixGF2) -> MatrixGF2:
    if left.column_count != right.row_count:
        raise ValueError("GF(2) matrix dimensions differ")
    return MatrixGF2(
        tuple(
            tuple(sum(left[row][k] * right[k][column] for k in range(left.column_count)) & 1 for column in range(right.column_count))
            for row in range(left.row_count)
        ),
        column_count=right.column_count,
    )


def _gf2_rank(matrix: MatrixGF2) -> int:
    work = [list(row) for row in matrix]
    rank = 0
    for column in range(matrix.column_count):
        pivot = next((row for row in range(rank, matrix.row_count) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for row in range(matrix.row_count):
            if row != rank and work[row][column]:
                work[row] = [a ^ b for a, b in zip(work[row], work[rank], strict=True)]
        rank += 1
    return rank


def _diagonalize(matrix: MatrixGF2) -> tuple[MatrixGF2, MatrixGF2, MatrixGF2, int]:
    m, n = matrix.shape
    work = [list(row) for row in matrix]
    row_change = [list(row) for row in _gf2_identity(m)]
    column_change = [list(row) for row in _gf2_identity(n)]
    rank = 0
    while rank < min(m, n):
        pivot = next(
            ((row, column) for row in range(rank, m) for column in range(rank, n) if work[row][column]),
            None,
        )
        if pivot is None:
            break
        row, column = pivot
        work[rank], work[row] = work[row], work[rank]
        row_change[rank], row_change[row] = row_change[row], row_change[rank]
        if column != rank:
            for current in range(m):
                work[current][rank], work[current][column] = work[current][column], work[current][rank]
            for current in range(n):
                column_change[current][rank], column_change[current][column] = column_change[current][column], column_change[current][rank]
        for current in range(m):
            if current != rank and work[current][rank]:
                work[current] = [a ^ b for a, b in zip(work[current], work[rank], strict=True)]
                row_change[current] = [a ^ b for a, b in zip(row_change[current], row_change[rank], strict=True)]
        for current in range(n):
            if current != rank and work[rank][current]:
                for row_index in range(m):
                    work[row_index][current] ^= work[row_index][rank]
                for row_index in range(n):
                    column_change[row_index][current] ^= column_change[row_index][rank]
        rank += 1
    return (
        MatrixGF2(tuple(tuple(row) for row in work), column_count=n),
        MatrixGF2(tuple(tuple(row) for row in row_change), column_count=m),
        MatrixGF2(tuple(tuple(row) for row in column_change), column_count=n),
        rank,
    )


def _columns(matrix: MatrixGF2) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(matrix[row][column] for row in range(matrix.row_count)) for column in range(matrix.column_count))


def _characters_from_basis(basis: tuple[GF2Character, ...], ambient: int) -> tuple[GF2Character, ...]:
    result = []
    for coefficients in itertools.product((0, 1), repeat=len(basis)):
        result.append(
            GF2Character(
                tuple(
                    sum(coefficient * character.bits[index] for coefficient, character in zip(coefficients, basis, strict=True)) & 1
                    for index in range(ambient)
                )
            )
        )
    return tuple(result)


def _relator_matrix(
    relator_words: Sequence[Sequence[Sequence[int]]], generator_count: int
) -> MatrixGF2:
    rows = []
    for word in relator_words:
        row = [0] * generator_count
        for generator, exponent in word:
            if type(generator) is not int or not 0 <= generator < generator_count:
                raise ValueError("relator generator index is outside the presentation")
            if type(exponent) is not int or exponent == 0:
                raise ValueError("relator exponent must be a nonzero integer")
            row[generator] ^= exponent & 1
        rows.append(tuple(row))
    return MatrixGF2(tuple(rows), column_count=generator_count)


def finite_table_presentation_relators(
    table: FiniteGroupTable,
    generator_order: Sequence[str],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return a canonical Cayley presentation for an exact finite-group table.

    The breadth-first normal word for every table element is fixed by the
    supplied generator order.  One transition relator is then emitted for
    every table element and generator.  These relators present the table,
    rather than merely listing equations which happen to hold in it.
    """

    if not isinstance(table, FiniteGroupTable):
        raise TypeError("table must be a FiniteGroupTable")
    generators = tuple(generator_order)
    if not generators or len(set(generators)) != len(generators):
        raise ValueError("finite presentation requires unique generators")
    if any(
        type(generator) is not str
        or generator not in table.element_order
        or table.element_order.index(generator) == table.identity_index
        for generator in generators
    ):
        raise ValueError("finite presentation generator is absent or trivial")
    generator_indices = tuple(table.element_order.index(item) for item in generators)
    words: dict[int, tuple[tuple[int, int], ...]] = {
        table.identity_index: ()
    }
    pending = [table.identity_index]
    cursor = 0
    while cursor < len(pending):
        current = pending[cursor]
        cursor += 1
        for generator_index, element_index in enumerate(generator_indices):
            product = table.multiplication_table[current][element_index]
            if product not in words:
                words[product] = words[current] + ((generator_index, 1),)
                pending.append(product)
    if len(words) != len(table.element_order):
        raise ValueError("declared generators do not generate the exact table")

    relators = []
    for element_index in range(len(table.element_order)):
        for generator_index, generator_element in enumerate(generator_indices):
            product = table.multiplication_table[element_index][generator_element]
            inverse_product_word = tuple(
                (word_generator, -exponent)
                for word_generator, exponent in reversed(words[product])
            )
            relators.append(
                words[element_index]
                + ((generator_index, 1),)
                + inverse_product_word
            )
    return tuple(relators)


def _pcp_action_and_decoder(
    resolution: FreeResolutionCertificate,
):
    certificate = resolution.affine_pcp_certificate
    normal_form = certificate.pcp_normal_form
    affine_generators = tuple(
        _evaluate_pcp_word(word, normal_form)
        for word in certificate.affine_generator_images
    )
    translation_columns = []
    for word in certificate.translation_basis_images:
        affine = _evaluate_pcp_word(word, normal_form)
        matrix, _ = _affine_exact(affine)
        if matrix != tuple(
            tuple(int(row == column) for column in range(3))
            for row in range(3)
        ):
            raise ValueError("Task 4 translation-basis image is not a translation")
        translation_columns.append(affine.translation)
    if len(translation_columns) != 3:
        raise ValueError("Task 4 certificate lacks three translation-basis images")
    translation_basis = tuple(
        tuple(translation_columns[column][row] for column in range(3))
        for row in range(3)
    )
    action = make_certified_space_group_action(
        affine_generators, translation_basis
    )
    if action.action_digest != certificate.catalogue_action_digest:
        raise ValueError("Task 4 action cannot be reconstructed from its PCP images")
    return normal_form, _normal_form_decoder(action, normal_form)


def _pcp_coordinate_word(
    coordinates: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (index, exponent)
        for index, exponent in enumerate(coordinates)
        if exponent
    )


def _inverse_presentation_word(
    word: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (generator, -exponent)
        for generator, exponent in reversed(tuple(tuple(step) for step in word))
    )


def _evaluate_presentation_word(
    normal_form: Any,
    word: Sequence[Sequence[int]],
):
    value = _evaluate_pcp_word("1", normal_form)
    for generator, exponent in word:
        factor = _power_affine(
            normal_form.generator_affines[generator], exponent
        )
        value = _compose_affine(factor, value)
    return value


def pcp_presentation_relators(
    resolution: FreeResolutionCertificate,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Derive the complete canonical collection presentation from Task 4.

    Finite relative powers and every out-of-order generator pair are reduced
    with the independently replayed affine normal-form decoder.  The result
    is therefore bound to the exact PCP generator order, not a caller-chosen
    matrix of exponent sums.
    """

    if not isinstance(resolution, FreeResolutionCertificate):
        raise TypeError("resolution must be a FreeResolutionCertificate")
    if resolution.finite_group is not None:
        raise ValueError("PCP presentation requires an ambient PCP resolution")
    normal_form, decode = _pcp_action_and_decoder(resolution)
    relators: list[tuple[tuple[int, int], ...]] = []
    for generator, relative_order in enumerate(normal_form.relative_orders):
        if relative_order == 0:
            continue
        leading = ((generator, relative_order),)
        reduced = _pcp_coordinate_word(
            decode(_evaluate_presentation_word(normal_form, leading))
        )
        relators.append(leading + _inverse_presentation_word(reduced))
    for later in range(len(normal_form.relative_orders)):
        for earlier in range(later):
            leading = ((later, 1), (earlier, 1))
            reduced = _pcp_coordinate_word(
                decode(_evaluate_presentation_word(normal_form, leading))
            )
            relators.append(leading + _inverse_presentation_word(reduced))
    if not relators:
        raise ValueError("canonical PCP presentation unexpectedly has no relators")
    return tuple(relators)


def _graded_direct_product_relators(
    spatial: CharacterBasisCertificate,
    time_generator: str,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    time_index = len(spatial.generator_order)
    relators = list(spatial.relator_words)
    relators.append(((time_index, 2),))
    for spatial_index in range(time_index):
        relators.append(
            (
                (spatial_index, 1),
                (time_index, 1),
                (spatial_index, -1),
                (time_index, -1),
            )
        )
    return tuple(relators)


def character_presentation_digest(
    *,
    group_id: str,
    resolution_id: str,
    presentation_kind: str,
    generator_order: Sequence[str],
    relator_words: Sequence[Sequence[Sequence[int]]],
    finite_group_table_digest: str | None,
    spatial_certificate_id: str | None = None,
    onsite_time_reversal_generator: str | None = None,
) -> str:
    return _task5_domain_digest(
        "task5-character-presentation-v1",
        {
            "finite_group_table_digest": finite_group_table_digest,
            "generator_order": list(generator_order),
            "group_id": group_id,
            "onsite_time_reversal_generator": onsite_time_reversal_generator,
            "presentation_kind": presentation_kind,
            "relator_words": [
                [list(step) for step in word] for word in relator_words
            ],
            "resolution_id": resolution_id,
            "spatial_certificate_id": spatial_certificate_id,
        },
    )


def _character_certificate_core(
    certificate: CharacterBasisCertificate,
) -> dict[str, Any]:
    return {
        "abelianization_basis": [list(vector) for vector in certificate.abelianization_basis],
        "abelianization_to_generators": _exact_matrix_mapping(certificate.abelianization_to_generators),
        "characters": [list(item.bits) for item in certificate.characters],
        "column_change": _exact_matrix_mapping(certificate.column_change),
        "generator_order": list(certificate.generator_order),
        "generator_to_abelianization": _exact_matrix_mapping(certificate.generator_to_abelianization),
        "group_id": certificate.group_id,
        "hom_basis": [list(item.bits) for item in certificate.hom_basis],
        "normal_form": _exact_matrix_mapping(certificate.normal_form),
        "onsite_time_reversal_generator": certificate.onsite_time_reversal_generator,
        "presentation_digest": certificate.presentation_digest,
        "presentation_kind": certificate.presentation_kind,
        "relator_matrix_mod2": _exact_matrix_mapping(certificate.relator_matrix_mod2),
        "relator_words": [[list(step) for step in word] for word in certificate.relator_words],
        "resolution_id": certificate.resolution_id,
        "row_change": _exact_matrix_mapping(certificate.row_change),
        "spatial_certificate_id": certificate.spatial_certificate_id,
    }


def character_certificate_digest(certificate: CharacterBasisCertificate) -> str:
    return _task5_domain_digest(
        "task5-character-basis-certificate-v1",
        _character_certificate_core(certificate),
    )


def character_basis_certificate(
    generator_order: Sequence[str],
    relator_words: Sequence[Sequence[Sequence[int]]],
    *,
    group_id: str,
    resolution_id: str,
    presentation_kind: str,
    finite_group_table_digest: str | None,
    spatial_certificate_id: str | None = None,
    onsite_time_reversal_generator: str | None = None,
) -> CharacterBasisCertificate:
    generators = tuple(generator_order)
    normalized_relators = tuple(
        tuple(tuple(step) for step in word) for word in relator_words
    )
    relator_matrix_mod2 = _relator_matrix(normalized_relators, len(generators))
    presentation_digest = character_presentation_digest(
        group_id=group_id,
        resolution_id=resolution_id,
        presentation_kind=presentation_kind,
        generator_order=generators,
        relator_words=normalized_relators,
        finite_group_table_digest=finite_group_table_digest,
        spatial_certificate_id=spatial_certificate_id,
        onsite_time_reversal_generator=onsite_time_reversal_generator,
    )
    normal, rows, columns, _ = _diagonalize(relator_matrix_mod2)
    hom_vectors = kernel_basis(relator_matrix_mod2)
    hom = tuple(GF2Character(vector) for vector in hom_vectors)
    generator_to = MatrixGF2(tuple(character.bits for character in hom), column_count=len(generators))
    lifts: list[tuple[int, ...]] = []
    for index in range(len(hom)):
        target = tuple(int(index == row) for row in range(len(hom)))
        solution = solve_affine(generator_to, target)
        if not hasattr(solution, "basepoint"):
            raise ArithmeticError("perfect quotient pairing unexpectedly failed")
        lifts.append(solution.basepoint)
    abelianization_to = MatrixGF2(
        tuple(tuple(lifts[column][row] for column in range(len(lifts))) for row in range(len(generators))),
        column_count=len(lifts),
    )
    provisional = CharacterBasisCertificate(
        group_id,
        resolution_id,
        presentation_kind,
        presentation_digest,
        generators,
        normalized_relators,
        relator_matrix_mod2,
        rows,
        columns,
        normal,
        tuple(lifts),
        generator_to,
        abelianization_to,
        hom,
        _characters_from_basis(hom, len(generators)),
        spatial_certificate_id,
        onsite_time_reversal_generator,
        "sha256:" + "0" * 64,
    )
    return replace(
        provisional,
        certificate_id=character_certificate_digest(provisional),
    )


def enumerate_coefficient_characters(
    certificate: CharacterBasisCertificate,
    resolution: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
    *,
    spatial_certificate: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
    graded_table: FiniteGroupTable | None = None,
) -> tuple[GF2Character, ...]:
    report = verify_character_basis(
        certificate,
        resolution,
        authority,
        spatial_certificate=spatial_certificate,
        spatial_resolution=spatial_resolution,
        graded_table=graded_table,
    )
    if not report.valid:
        raise ValueError("character basis certificate is invalid")
    return certificate.characters


def extend_finite_table_character(
    certificate: CharacterBasisCertificate,
    character: GF2Character,
    resolution: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
) -> GF2Character:
    """Extend a certified generator character to exact table-element order."""

    report = verify_character_basis(certificate, resolution, authority)
    if not report.valid:
        raise ValueError("character basis certificate is invalid")
    if certificate.presentation_kind != "finite-table-presentation":
        raise ValueError("character extension requires a finite-table presentation")
    if not isinstance(character, GF2Character) or character not in certificate.characters:
        raise ValueError("character is not one of the certified Hom(G,C2) values")
    table = resolution.finite_group
    if table is None:
        raise ValueError("finite-table presentation lacks its exact group table")
    generator_indices = tuple(
        table.element_order.index(generator)
        for generator in certificate.generator_order
    )
    values = {table.identity_index: 0}
    pending = [table.identity_index]
    cursor = 0
    while cursor < len(pending):
        current = pending[cursor]
        cursor += 1
        for generator_index, element_index in enumerate(generator_indices):
            product = table.multiplication_table[current][element_index]
            value = values[current] ^ character.bits[generator_index]
            if product in values:
                if values[product] != value:
                    raise ValueError("certified generator character has inconsistent table lifts")
            else:
                values[product] = value
                pending.append(product)
    if len(values) != len(table.element_order):
        raise ValueError("certified generators do not cover the exact table")
    result = GF2Character(tuple(values[index] for index in range(len(table.element_order))))
    if any(
        result.bits[table.multiplication_table[left][right]]
        != (result.bits[left] ^ result.bits[right])
        for left in range(len(table.element_order))
        for right in range(len(table.element_order))
    ):
        raise ValueError("extended coefficient character is not a group homomorphism")
    return result


def _evaluate_finite_relator(
    table: FiniteGroupTable,
    generator_order: Sequence[str],
    word: Sequence[Sequence[int]],
) -> int:
    result = table.identity_index
    for generator, exponent in word:
        element = table.element_order.index(generator_order[generator])
        if exponent < 0:
            element = table.inverse_indices[element]
        for _ in range(abs(exponent)):
            result = table.multiplication_table[result][element]
    return result


def _finite_generators_cover_group(
    table: FiniteGroupTable, generator_order: Sequence[str]
) -> bool:
    generator_indices = [table.element_order.index(item) for item in generator_order]
    factors = generator_indices + [table.inverse_indices[item] for item in generator_indices]
    seen = {table.identity_index}
    pending = [table.identity_index]
    while pending:
        current = pending.pop()
        for factor in factors:
            product = table.multiplication_table[current][factor]
            if product not in seen:
                seen.add(product)
                pending.append(product)
    return len(seen) == len(table.element_order)


def _verify_graded_finite_group_table(
    spatial_table: FiniteGroupTable,
    graded_table: FiniteGroupTable,
) -> None:
    if graded_table.group_id != spatial_table.group_id + "+onsite-T":
        raise ValueError("graded finite table has the wrong direct-product group ID")
    spatial_order = spatial_table.element_order
    expected_order = spatial_order + ("T",) + tuple(
        f"{element}+T" for element in spatial_order[1:]
    )
    if graded_table.element_order != expected_order:
        raise ValueError(
            "graded finite table element order must be spatial elements followed by the T coset"
        )
    spatial_size = len(spatial_order)
    for left in range(2 * spatial_size):
        left_spatial = left % spatial_size
        left_time = left // spatial_size
        for right in range(2 * spatial_size):
            right_spatial = right % spatial_size
            right_time = right // spatial_size
            expected = (
                spatial_table.multiplication_table[left_spatial][right_spatial]
                + spatial_size * (left_time ^ right_time)
            )
            if graded_table.multiplication_table[left][right] != expected:
                raise ValueError(
                    "graded finite table multiplication law is not the exact spatial x C2 product"
                )
    time_character = (0,) * spatial_size + (1,) * spatial_size
    if any(
        time_character[graded_table.multiplication_table[left][right]]
        != (time_character[left] ^ time_character[right])
        for left in range(2 * spatial_size)
        for right in range(2 * spatial_size)
    ):
        raise ValueError(
            "graded finite table T-coset membership is not a coefficient character"
        )


def _replay_spatial_character_for_graded_resolution(
    spatial: CharacterBasisCertificate,
    graded_resolution: FreeResolutionCertificate,
    spatial_resolution: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
    graded_table: FiniteGroupTable | None,
) -> None:
    spatial_report = verify_character_basis(spatial, spatial_resolution, authority)
    if not spatial_report.valid:
        raise ValueError(
            "spatial character certificate is not bound to the exact verified spatial resolution"
        )
    if spatial.presentation_kind == "graded-direct-product-presentation":
        raise ValueError("the bound spatial character certificate is itself graded")
    suffix = "+onsite-T"
    if graded_resolution.group_id.endswith(suffix):
        if (
            graded_resolution.group_id != spatial_resolution.group_id + suffix
            or graded_resolution.affine_pcp_certificate
            != spatial_resolution.affine_pcp_certificate
            or graded_resolution.catalogue_record_digest
            != spatial_resolution.catalogue_record_digest
            or graded_resolution.construction
            != "onsite-c2-direct-product-resolution"
            or graded_resolution.parent_spatial_resolution_id
            != spatial_resolution.resolution_id
        ):
            raise ValueError(
                "graded ambient resolution is not bound to the exact verified spatial resolution"
            )
        if (
            graded_resolution.finite_group is not None
            or spatial_resolution.finite_group is not None
            or graded_table is not None
        ):
            raise ValueError(
                "an exported ambient graded direct product must use the affine PCP factor"
            )
    elif graded_resolution != spatial_resolution:
        raise ValueError(
            "synthetic graded presentation requires its exact spatial resolution"
        )
    elif spatial_resolution.finite_group is None:
        if graded_table is not None:
            raise ValueError("an affine graded presentation cannot bind a finite table")
    elif graded_table is None:
        raise ValueError("finite graded verification requires its graded finite group table")
    else:
        _verify_graded_finite_group_table(
            spatial_resolution.finite_group, graded_table
        )


def verify_character_basis(
    certificate: CharacterBasisCertificate,
    resolution: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
    *,
    spatial_certificate: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
    graded_table: FiniteGroupTable | None = None,
) -> VerificationReport:
    if not isinstance(certificate, CharacterBasisCertificate):
        raise TypeError("certificate must be a CharacterBasisCertificate")
    if not isinstance(resolution, FreeResolutionCertificate):
        raise TypeError("resolution must be the bound FreeResolutionCertificate")
    if not isinstance(authority, Task5VerificationAuthority):
        raise TypeError("authority must be a caller-supplied Task5VerificationAuthority")
    issues: list[VerificationIssue] = []
    resolution_report = verify_resolution(resolution, authority)
    issues.extend(
        _issue("resolution_invalid", issue.detail)
        for issue in resolution_report.issues
    )
    if certificate.presentation_kind == "graded-direct-product-presentation":
        if spatial_certificate is None or spatial_resolution is None:
            issues.append(_issue(
                "graded_presentation_binding_mismatch",
                "graded direct product requires its exact spatial character and resolution",
            ))
        elif (
            resolution.group_id.endswith("+onsite-T")
            and certificate.group_id == resolution.group_id
            and certificate.resolution_id == resolution.resolution_id
        ):
            try:
                _replay_spatial_character_for_graded_resolution(
                    spatial_certificate,
                    resolution,
                    spatial_resolution,
                    authority,
                    graded_table,
                )
                if (
                    certificate.spatial_certificate_id
                    != spatial_certificate.certificate_id
                    or certificate.onsite_time_reversal_generator != "T"
                    or certificate.generator_order
                    != spatial_certificate.generator_order
                    + ("T",)
                    or certificate.relator_words
                    != _graded_direct_product_relators(
                        spatial_certificate,
                        certificate.onsite_time_reversal_generator,
                    )
                ):
                    raise ValueError(
                        "onsite T is not the bound direct-product factor"
                    )
            except (TypeError, ValueError) as error:
                issues.append(_issue(
                    "graded_presentation_binding_mismatch", str(error)
                ))
        else:
            spatial_report = verify_character_basis(
                spatial_certificate, spatial_resolution, authority
            )
            try:
                _replay_spatial_character_for_graded_resolution(
                    spatial_certificate,
                    resolution,
                    spatial_resolution,
                    authority,
                    graded_table,
                )
            except (TypeError, ValueError) as error:
                issues.append(_issue(
                    "graded_presentation_binding_mismatch", str(error)
                ))
            expected_resolution_id = _task5_domain_digest(
                "task5-graded-direct-product-resolution-v1",
                {
                    "onsite_factor": "C2",
                    "spatial_resolution_id": spatial_resolution.resolution_id,
                },
            )
            if (
                not spatial_report.valid
                or spatial_certificate.presentation_kind
                == "graded-direct-product-presentation"
                or certificate.spatial_certificate_id
                != spatial_certificate.certificate_id
                or certificate.resolution_id != expected_resolution_id
                or resolution != spatial_resolution
                or certificate.group_id
                != spatial_resolution.group_id + "+onsite-T"
                or certificate.onsite_time_reversal_generator != "T"
                or certificate.generator_order
                != spatial_certificate.generator_order
                + (certificate.onsite_time_reversal_generator,)
                or certificate.relator_words
                != _graded_direct_product_relators(
                    spatial_certificate,
                    certificate.onsite_time_reversal_generator,
                )
            ):
                issues.append(_issue(
                    "graded_presentation_binding_mismatch",
                    "onsite T is not the bound direct-product factor",
                ))
    elif (
        certificate.group_id != resolution.group_id
        or certificate.resolution_id != resolution.resolution_id
        or certificate.spatial_certificate_id is not None
        or certificate.onsite_time_reversal_generator is not None
    ):
        issues.append(_issue(
            "character_resolution_binding_mismatch",
            "character certificate differs from its exact group/resolution",
        ))
    finite_digest = (
        graded_table.table_digest
        if certificate.presentation_kind == "graded-direct-product-presentation"
        and graded_table is not None
        else None if resolution.finite_group is None else resolution.finite_group.table_digest
    )
    expected_presentation_digest = character_presentation_digest(
        group_id=certificate.group_id,
        resolution_id=certificate.resolution_id,
        presentation_kind=certificate.presentation_kind,
        generator_order=certificate.generator_order,
        relator_words=certificate.relator_words,
        finite_group_table_digest=finite_digest,
        spatial_certificate_id=certificate.spatial_certificate_id,
        onsite_time_reversal_generator=certificate.onsite_time_reversal_generator,
    )
    if certificate.presentation_digest != expected_presentation_digest:
        issues.append(_issue(
            "character_presentation_digest_mismatch",
            "presentation digest does not bind the exact group presentation",
        ))
    expected_relator_matrix = _relator_matrix(
        certificate.relator_words, len(certificate.generator_order)
    )
    if certificate.relator_matrix_mod2 != expected_relator_matrix:
        issues.append(_issue(
            "character_relator_matrix_mismatch",
            "mod-two matrix is not the exponent image of the bound relators",
        ))
    if certificate.presentation_kind == "finite-table-presentation":
        table = resolution.finite_group
        try:
            if table is None:
                raise ValueError("finite presentation requires a finite table")
            expected_relators = finite_table_presentation_relators(
                table, certificate.generator_order
            )
            if certificate.relator_words != expected_relators:
                raise ValueError(
                    "relators are not the canonical complete Cayley presentation"
                )
            if any(
                _evaluate_finite_relator(table, certificate.generator_order, word)
                != table.identity_index
                for word in certificate.relator_words
            ):
                raise ValueError("a bound relator is nontrivial in the exact table")
        except (TypeError, ValueError) as error:
            issues.append(_issue("character_presentation_invalid", str(error)))
    elif certificate.presentation_kind == "pcp-presentation":
        expected_generators = tuple(
            f"p{index + 1}"
            for index in range(
                len(resolution.affine_pcp_certificate.pcp_normal_form.relative_orders)
            )
        )
        if certificate.generator_order != expected_generators:
            issues.append(_issue(
                "character_presentation_invalid", "PCP generator order differs from Task 4"
            ))
        else:
            normal_form = resolution.affine_pcp_certificate.pcp_normal_form
            try:
                expected_relators = pcp_presentation_relators(resolution)
                if certificate.relator_words != expected_relators:
                    raise ValueError(
                        "relators are not the complete Task 4 collection presentation"
                    )
                identity = _evaluate_pcp_word("1", normal_form)
                if any(
                    _evaluate_presentation_word(normal_form, word) != identity
                    for word in certificate.relator_words
                ):
                    raise ValueError(
                        "a bound PCP relator is nontrivial in the exact affine realization"
                    )
            except (TypeError, ValueError) as error:
                issues.append(_issue(
                    "character_presentation_invalid",
                    str(error),
                ))
    relation = certificate.relator_matrix_mod2
    m, n = relation.shape
    row = certificate.row_change
    column = certificate.column_change
    if row.shape != (m, m) or column.shape != (n, n) or _gf2_rank(row) != m or _gf2_rank(column) != n:
        issues.append(_issue("invalid_change_witness", "row or column change is not invertible with the required shape"))
    else:
        transformed = _gf2_matmul(_gf2_matmul(row, relation), column)
        if transformed != certificate.normal_form:
            issues.append(_issue("invalid_change_witness", "row/column witnesses do not produce the exported normal form"))
        expected_rank = _gf2_rank(relation)
        expected_normal = MatrixGF2(
            tuple(tuple(int(i == j and i < expected_rank) for j in range(n)) for i in range(m)),
            column_count=n,
        )
        if certificate.normal_form != expected_normal:
            issues.append(_issue("invalid_change_witness", "normal form is not canonical rank diagonal"))
    rank = len(certificate.hom_basis)
    if rank != n - _gf2_rank(relation):
        issues.append(_issue("hom_basis_not_exhaustive", "Hom basis has the wrong rank"))
    for character in certificate.hom_basis:
        if len(character.bits) != n or any(sum(a * b for a, b in zip(relator, character.bits, strict=True)) & 1 for relator in relation):
            issues.append(_issue("hom_basis_not_exhaustive", "Hom basis vector violates a relator"))
    if certificate.generator_to_abelianization.shape != (rank, n) or certificate.abelianization_to_generators.shape != (n, rank):
        issues.append(_issue("abelianization_change_map_invalid", "generator/abelianization map has wrong shape"))
    else:
        if certificate.generator_to_abelianization != MatrixGF2(tuple(character.bits for character in certificate.hom_basis), column_count=n):
            issues.append(_issue("dual_basis_invalid", "generator quotient map is not the Hom pairing"))
        product = _gf2_matmul(certificate.generator_to_abelianization, certificate.abelianization_to_generators)
        if product != _gf2_identity(rank):
            issues.append(_issue("dual_basis_invalid", "abelianization and Hom bases are not mutually dual"))
        if certificate.abelianization_basis != _columns(certificate.abelianization_to_generators):
            issues.append(_issue("abelianization_change_map_invalid", "basis vectors differ from the exported lift map"))
        for relator in relation:
            image = tuple(sum(certificate.generator_to_abelianization[row][column] * relator[column] for column in range(n)) & 1 for row in range(rank))
            if any(image):
                issues.append(_issue("abelianization_change_map_invalid", "a relator survives in the quotient"))
                break
    expected = _characters_from_basis(certificate.hom_basis, n)
    if certificate.characters != expected or len({item.bits for item in certificate.characters}) != len(expected):
        issues.append(_issue("character_enumeration_incomplete", "character list is not the exact lexicographic span"))
    if character_certificate_digest(certificate) != certificate.certificate_id:
        issues.append(_issue(
            "character_certificate_digest_mismatch",
            "character certificate ID does not bind the payload",
        ))
    return VerificationReport(not issues, tuple(issues), 5)


def adjoin_onsite_time_reversal_character(
    certificate: CharacterBasisCertificate,
    resolution: FreeResolutionCertificate,
    authority: Task5VerificationAuthority,
    generator: str = "T",
    *,
    graded_resolution: FreeResolutionCertificate | None = None,
    graded_table: FiniteGroupTable | None = None,
) -> CharacterBasisCertificate:
    report = verify_character_basis(certificate, resolution, authority)
    if not report.valid:
        raise ValueError("spatial character certificate is invalid")
    _identifier(generator, "$time_generator")
    if generator != "T":
        raise ValueError("onsite time-reversal generator must be literal T")
    if generator in certificate.generator_order:
        raise ValueError("time generator duplicates a spatial generator")
    if resolution.finite_group is not None and graded_table is None:
        raise ValueError("finite direct product requires its graded finite group table")
    if graded_table is not None and not isinstance(graded_table, FiniteGroupTable):
        raise TypeError("graded_table must be a FiniteGroupTable")
    if resolution.finite_group is not None:
        _verify_graded_finite_group_table(resolution.finite_group, graded_table)
    elif graded_table is not None:
        raise ValueError("an affine graded presentation cannot bind a finite table")
    if graded_resolution is None:
        graded_group_id = resolution.group_id + "+onsite-T"
        graded_resolution_id = _task5_domain_digest(
            "task5-graded-direct-product-resolution-v1",
            {
                "onsite_factor": "C2",
                "spatial_resolution_id": resolution.resolution_id,
            },
        )
        finite_group_table_digest = (
            None
            if graded_table is None
            else graded_table.table_digest
        )
    else:
        graded_report = verify_resolution(graded_resolution, authority)
        if not graded_report.valid:
            raise ValueError("graded ambient resolution certificate is invalid")
        if (
            graded_resolution.group_id != resolution.group_id + "+onsite-T"
            or graded_resolution.affine_pcp_certificate_digest
            != resolution.affine_pcp_certificate_digest
            or graded_resolution.catalogue_record_digest
            != resolution.catalogue_record_digest
            or graded_resolution.finite_group is not None
            or graded_resolution.parent_spatial_resolution_id
            != resolution.resolution_id
        ):
            raise ValueError(
                "graded ambient resolution has the wrong parent spatial resolution"
            )
        graded_group_id = graded_resolution.group_id
        graded_resolution_id = graded_resolution.resolution_id
        finite_group_table_digest = None
    return character_basis_certificate(
        certificate.generator_order + (generator,),
        _graded_direct_product_relators(certificate, generator),
        group_id=graded_group_id,
        resolution_id=graded_resolution_id,
        presentation_kind="graded-direct-product-presentation",
        finite_group_table_digest=finite_group_table_digest,
        spatial_certificate_id=certificate.certificate_id,
        onsite_time_reversal_generator=generator,
    )


def _word_character(resolution: FreeResolutionCertificate, character: GF2Character, element: str) -> int:
    if resolution.finite_group is not None:
        table = resolution.finite_group
        if len(character.bits) != len(table.element_order):
            raise ValueError("finite coefficient character must be in table-element order")
        return character.bits[table.element_order.index(element)]
    spatial, time_bit = _onsite_time_reversal_normal_form(resolution, element)
    coordinates = _pcp_word_coordinates(
        spatial, resolution.affine_pcp_certificate.pcp_normal_form
    )
    graded = resolution.group_id.endswith("+onsite-T")
    expected_length = len(coordinates) + int(graded)
    if len(character.bits) != expected_length:
        raise ValueError("coefficient character must assign every PCP generator")
    spatial_value = sum(
        (
            bit * (exponent & 1)
            for bit, exponent in zip(
                character.bits[: len(coordinates)], coordinates, strict=True
            )
        ),
        0,
    )
    return (spatial_value ^ (character.bits[-1] * time_bit if graded else 0)) & 1


def restrict_coefficient_character(
    certificate: InclusionChainMapCertificate,
    ambient_character: GF2Character,
    *,
    release_bundle: object,
) -> GF2Character:
    """Restrict an ambient character through one exact signed Task-5 bundle.

    The caller supplies no local values.  They are evaluated in the literal
    finite-table order from ``source_element_images`` after the bundle,
    release authority, launcher capability, and inclusion certificate have
    all replayed with their original identities.
    """

    if type(certificate) is not InclusionChainMapCertificate:
        raise TypeError("certificate must be an exact InclusionChainMapCertificate")
    if type(ambient_character) is not GF2Character:
        raise TypeError("ambient_character must be an exact GF2Character")

    # Imported at the call boundary because task5_release owns the capability
    # registry and itself imports this lower-level algebra module.
    from . import task5_release

    verified_bundle = task5_release.verify_task5_release_bundle(release_bundle)
    if verified_bundle is not release_bundle:
        raise ValueError("release verifier did not return the exact bundle capability")
    if verified_bundle.inclusion is not certificate:
        raise ValueError("release bundle does not bind the exact inclusion certificate")
    verification_authority = task5_release.verify_task5_release_authority(
        verified_bundle.release_authority
    )
    if (
        verification_authority
        is not verified_bundle.release_authority.verification_authority
    ):
        raise ValueError("release bundle does not retain its exact Task-5 authority")
    attestation = certificate.launcher_attestation
    if (
        not attestation.release_certified
        or attestation.runtime_manifest_digest is None
    ):
        raise ValueError("release restriction requires a locked-launcher attestation")
    report = verify_inclusion_chain_map(
        certificate,
        verification_authority,
        require_release=True,
        trusted_release_attestation=attestation,
    )
    if not report.valid:
        raise ValueError(
            "release inclusion certificate is invalid: " + report.issues[0].code
        )
    table = certificate.source_resolution.finite_group
    if table is None:
        raise ValueError("release inclusion source lacks its finite group authority")
    if len(certificate.source_element_images) != len(table.element_order):
        raise ValueError(
            "release inclusion images do not cover the finite table in exact order"
        )
    source_is_graded = certificate.source_resolution.group_id.endswith(
        "+onsite-T"
    )
    target_is_graded = certificate.target_resolution.group_id.endswith(
        "+onsite-T"
    )
    if source_is_graded != target_is_graded:
        raise ValueError("release inclusion has inconsistent onsite-T grading")
    if source_is_graded:
        order = table.element_order
        if len(order) % 2:
            raise ValueError("graded finite table has odd order")
        spatial_size = len(order) // 2
        spatial_order = order[:spatial_size]
        expected_order = spatial_order + ("T",) + tuple(
            element + "+T" for element in spatial_order[1:]
        )
        images = certificate.source_element_images
        spatial_images = images[:spatial_size]
        expected_images = spatial_images + ("T",) + tuple(
            image + "+T" for image in spatial_images[1:]
        )
        if order != expected_order or images != expected_images:
            raise ValueError(
                "graded restriction requires spatial elements followed by the literal T coset"
            )
    return GF2Character(
        tuple(
            _word_character(
                certificate.target_resolution,
                ambient_character,
                image,
            )
            for image in certificate.source_element_images
        )
    )


def twist_group_ring_matrix(
    matrix: SparseGroupRingMatrix,
    resolution: FreeResolutionCertificate,
    character: GF2Character,
) -> MatrixZ:
    dense = [[0] * matrix.column_count for _ in range(matrix.row_count)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] = sum(
            term.coefficient * (-1 if _word_character(resolution, character, term.element) else 1)
            for term in entry.terms
        )
    return MatrixZ(tuple(tuple(row) for row in dense), column_count=matrix.column_count)


def twist_cochain_complex(
    resolution: FreeResolutionCertificate,
    character: GF2Character,
    authority: Task5VerificationAuthority,
) -> CochainComplex:
    report = verify_resolution(resolution, authority)
    if not report.valid:
        raise ValueError("resolution certificate is invalid")
    if not isinstance(character, GF2Character):
        raise TypeError("character must be a GF2Character")
    differentials = []
    for boundary in resolution.boundaries:
        evaluated = twist_group_ring_matrix(boundary, resolution, character)
        differentials.append(
            MatrixZ(
                tuple(
                    tuple(evaluated[row][column] for row in range(evaluated.row_count))
                    for column in range(evaluated.column_count)
                ),
                column_count=evaluated.row_count,
            )
        )
    return make_cochain_complex(
        authority_id=resolution.resolution_id,
        dimensions=tuple(len(degree) for degree in resolution.basis),
        differentials=tuple(differentials),
        coefficient_character=character,
    )


def verify_cochain_map(
    cochain_map: CochainMap,
    source: CochainComplex,
    target: CochainComplex,
) -> VerificationReport:
    """Verify an ambient-to-local cochain map in every stored degree."""

    if not isinstance(cochain_map, CochainMap):
        raise TypeError("cochain_map must be a CochainMap")
    if not isinstance(source, CochainComplex) or not isinstance(target, CochainComplex):
        raise TypeError("source and target must be CochainComplex values")
    issues: list[VerificationIssue] = []
    if cochain_map.source_id != source.complex_id or cochain_map.target_id != target.complex_id:
        issues.append(_issue("cochain_map_binding_mismatch", "source or target complex ID differs"))
    ring = type(source.differentials[0])
    if type(target.differentials[0]) is not ring or any(
        type(matrix) is not ring for matrix in cochain_map.maps
    ):
        issues.append(_issue("cochain_map_ring_mismatch", "map and complexes use different rings"))
    for degree, matrix in enumerate(cochain_map.maps):
        expected = (target.dimensions[degree], source.dimensions[degree])
        if matrix.shape != expected:
            issues.append(_issue("cochain_map_shape_mismatch", f"degree {degree} has wrong shape"))
    checked = 0
    if not any(issue.code in {"cochain_map_ring_mismatch", "cochain_map_shape_mismatch"} for issue in issues):
        for degree in range(4):
            checked += 1
            if not _matrix_products_equal(
                target.differentials[degree],
                cochain_map.maps[degree],
                cochain_map.maps[degree + 1],
                source.differentials[degree],
            ):
                issues.append(_issue("cochain_map_not_commuting", f"degree {degree} square fails"))
    return VerificationReport(not issues, tuple(issues), checked)


def make_cochain_map(
    *,
    instance_id: str,
    source: CochainComplex,
    target: CochainComplex,
    maps: Sequence[MatrixZ | MatrixGF2],
) -> CochainMap:
    result = CochainMap(
        instance_id,
        source.complex_id,
        target.complex_id,
        tuple(maps),
    )
    report = verify_cochain_map(result, source, target)
    if not report.valid:
        raise ValueError(f"invalid cochain map: {report.issues[0].code}")
    return result


def twist_inclusion_cochain_map(
    certificate: InclusionChainMapCertificate,
    ambient_character: GF2Character,
    local_character: GF2Character,
    authority: Task5VerificationAuthority,
    *,
    instance_id: str,
    allow_diagnostic: bool = False,
    trusted_release_attestation: LauncherExecutionAttestation | None = None,
) -> tuple[CochainComplex, CochainComplex, CochainMap]:
    """Twist a certified chain inclusion and transpose it to ambient -> local."""

    if type(allow_diagnostic) is not bool:
        raise TypeError("allow_diagnostic must be a boolean")
    report = verify_inclusion_chain_map(
        certificate,
        authority,
        require_release=not allow_diagnostic,
        trusted_release_attestation=trusted_release_attestation,
    )
    if not report.valid:
        raise ValueError("inclusion chain-map certificate is invalid")
    source_table = certificate.source_resolution.finite_group
    if source_table is None:
        raise ValueError("inclusion source lacks its finite group authority")
    if len(local_character.bits) != len(source_table.element_order):
        raise ValueError("local character must be in finite-table element order")
    for index, image in enumerate(certificate.source_element_images):
        if _word_character(certificate.target_resolution, ambient_character, image) != local_character.bits[index]:
            raise ValueError("ambient character does not restrict to local character")
    ambient = twist_cochain_complex(
        certificate.target_resolution, ambient_character, authority
    )
    local = twist_cochain_complex(
        certificate.source_resolution, local_character, authority
    )
    maps = []
    for matrix in certificate.maps:
        evaluated = twist_group_ring_matrix(
            matrix,
            certificate.target_resolution,
            ambient_character,
        )
        maps.append(
            MatrixZ(
                tuple(
                    tuple(evaluated[row][column] for row in range(evaluated.row_count))
                    for column in range(evaluated.column_count)
                ),
                column_count=evaluated.row_count,
            )
        )
    restriction = make_cochain_map(
        instance_id=instance_id,
        source=ambient,
        target=local,
        maps=maps,
    )
    return ambient, local, restriction


__all__ = [
    "BarComparisonBasisTrace",
    "BarComparisonTerm",
    "CertifiedCochainProblem",
    "CharacterBasisCertificate",
    "CochainComplex",
    "CochainMap",
    "FiniteGroupTable",
    "FreeResolutionCertificate",
    "InclusionChainMapCertificate",
    "LauncherExecutionAttestation",
    "ResolutionComparisonTerm",
    "SparseGroupRingEntry",
    "SparseGroupRingMatrix",
    "SparseGroupRingTerm",
    "Task5InclusionAuthority",
    "Task5VerificationAuthority",
    "TargetBarPhiTrace",
    "TargetBarResolutionEquivalence",
    "TargetResolutionBasisTrace",
    "VerificationIssue",
    "VerificationReport",
    "adjoin_onsite_time_reversal_character",
    "assemble_gap_free_resolution_certificate",
    "character_basis_certificate",
    "character_certificate_digest",
    "character_presentation_digest",
    "diagnostic_residue_digests",
    "dumps_free_resolution_certificate",
    "dumps_inclusion_chain_map_certificate",
    "enumerate_coefficient_characters",
    "extend_finite_table_character",
    "free_resolution_digest",
    "free_resolution_mapping",
    "gap_inclusion_projection_digest",
    "finite_table_presentation_relators",
    "inclusion_chain_map_digest",
    "inclusion_chain_map_mapping",
    "launcher_execution_attestation_digest",
    "launcher_execution_attestation_mapping",
    "loads_free_resolution_certificate",
    "loads_inclusion_chain_map_certificate",
    "make_free_resolution_certificate",
    "make_inclusion_chain_map_certificate",
    "make_launcher_execution_attestation",
    "make_cochain_complex",
    "make_cochain_map",
    "pcp_presentation_relators",
    "replace_resolution_id",
    "restrict_coefficient_character",
    "target_bar_equivalence_digest",
    "target_bar_equivalence_mapping",
    "task5_diagnostic_observed_outcome",
    "twist_cochain_complex",
    "twist_inclusion_cochain_map",
    "twist_group_ring_matrix",
    "verify_character_basis",
    "verify_cochain_map",
    "verify_inclusion_chain_map",
    "verify_resolution",
]
