"""Strict public schemas for certified PSG classification records.

This module deliberately contains no classifier algebra.  It owns the v1
byte grammar, immutable request/result records, and the semantic distinction
between a certified obstruction and a hard backend failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Literal


SCHEMA_VERSION = 1
MAX_JSON_NESTING = 64
MAX_JSON_NODES = 100_000

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_SETTING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PARAMETER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_RATIONAL_RE = re.compile(r"(?:0|-?[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z")
_WYCKOFF_ID_RE = re.compile(
    r"sg([1-9][0-9]{0,2}):setting-([A-Za-z0-9._-]+):(sha256:[0-9a-f]{64})\Z"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")

_FAILURE_CODES = {
    "invalid_request",
    "catalogue_incomplete",
    "unsupported_schema",
    "backend_timeout",
    "backend_failed",
    "certificate_invalid",
    "chain_identity_failed",
    "local_library_incomplete",
    "cache_corrupt",
    "coverage_incomplete",
}

FailureCode = Literal[
    "invalid_request",
    "catalogue_incomplete",
    "unsupported_schema",
    "backend_timeout",
    "backend_failed",
    "certificate_invalid",
    "chain_identity_failed",
    "local_library_incomplete",
    "cache_corrupt",
    "coverage_incomplete",
]

FrozenJSONScalar = str | int | bool | None


@dataclass(frozen=True, slots=True)
class FrozenJSONArray:
    items: tuple["FrozenJSONValue", ...]

    def __post_init__(self) -> None:
        _validate_json_tree(self.items, "$FrozenJSONArray")
        object.__setattr__(self, "items", tuple(_freeze(item) for item in self.items))

    def __getitem__(self, index: int) -> "FrozenJSONValue":
        return self.items[index]

    def __iter__(self) -> Iterator["FrozenJSONValue"]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class FrozenJSONObject:
    items: tuple[tuple[str, "FrozenJSONValue"], ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.items)
        if any(not isinstance(key, str) for key, _ in pairs):
            raise TypeError("$FrozenJSONObject: keys must be strings")
        keys = tuple(key for key, _ in pairs)
        if len(set(keys)) != len(keys):
            raise ValueError("$FrozenJSONObject: duplicate key")
        _validate_json_tree({key: value for key, value in pairs}, "$FrozenJSONObject")
        frozen = tuple(sorted((key, _freeze(value)) for key, value in pairs))
        object.__setattr__(self, "items", frozen)

    def __getitem__(self, key: str) -> "FrozenJSONValue":
        for candidate, value in self.items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.items)

    def __len__(self) -> int:
        return len(self.items)


FrozenJSONValue = FrozenJSONScalar | FrozenJSONArray | FrozenJSONObject


def _check_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise ValueError(f"JSON nesting exceeds {MAX_JSON_NESTING}")
        elif character in "]}":
            depth -= 1


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json_integer(token: str) -> int:
    if token == "-0":
        raise ValueError("negative zero is not canonical JSON")
    return int(token)


def _reject_json_float(token: str) -> float:
    try:
        if token.startswith("-") and Decimal(token).is_zero():
            raise ValueError("negative zero is not canonical JSON")
    except InvalidOperation as error:  # pragma: no cover - json supplies lexemes
        raise ValueError("invalid floating-point JSON token") from error
    raise ValueError("floating-point JSON tokens are forbidden")


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON token is forbidden: {token}")


def _looks_absolute_path(value: str) -> bool:
    candidate = value.lstrip()
    folded = candidate.casefold()
    return (
        candidate.startswith(("/", "~", "\\"))
        or folded.startswith("file:")
        or _WINDOWS_ABSOLUTE_RE.match(candidate) is not None
    )


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_json_tree(value: Any, path: str = "$") -> None:
    """Validate and bound a JSON-like graph before any recursive conversion."""

    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, False)]
    active: set[int] = set()
    visited = 0
    while stack:
        node, node_path, depth, exiting = stack.pop()
        if exiting:
            active.remove(id(node))
            continue
        visited += 1
        if visited > MAX_JSON_NODES:
            raise ValueError(f"{path}: JSON node limit exceeds {MAX_JSON_NODES}")
        if depth > MAX_JSON_NESTING:
            raise ValueError(f"{node_path}: JSON nesting exceeds {MAX_JSON_NESTING}")
        if isinstance(node, FrozenJSONObject):
            identity = id(node)
            if identity in active:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active.add(identity)
            stack.append((node, node_path, depth, True))
            for key, item in reversed(node.items):
                stack.append((item, f"{node_path}.{key}", depth + 1, False))
        elif isinstance(node, FrozenJSONArray):
            identity = id(node)
            if identity in active:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active.add(identity)
            stack.append((node, node_path, depth, True))
            for index in range(len(node.items) - 1, -1, -1):
                stack.append((node.items[index], f"{node_path}[{index}]", depth + 1, False))
        elif isinstance(node, Mapping):
            identity = id(node)
            if identity in active:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active.add(identity)
            stack.append((node, node_path, depth, True))
            for key, item in reversed(tuple(node.items())):
                if not isinstance(key, str):
                    raise TypeError(f"{node_path}: JSON object keys must be strings")
                if _contains_surrogate(key):
                    raise ValueError(f"{node_path}: surrogate code points are forbidden")
                stack.append((item, f"{node_path}.{key}", depth + 1, False))
        elif isinstance(node, (list, tuple)):
            identity = id(node)
            if identity in active:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active.add(identity)
            stack.append((node, node_path, depth, True))
            for index in range(len(node) - 1, -1, -1):
                stack.append((node[index], f"{node_path}[{index}]", depth + 1, False))
        elif isinstance(node, float):
            raise TypeError(f"{node_path}: float values are forbidden")
        elif isinstance(node, str):
            if _contains_surrogate(node):
                raise ValueError(f"{node_path}: surrogate code points are forbidden")
            if _looks_absolute_path(node):
                raise ValueError(f"{node_path}: absolute path is forbidden in public records")
        elif node is None or isinstance(node, (bool, int)):
            continue
        else:
            raise TypeError(f"{node_path}: unsupported JSON value {type(node).__name__}")


def _freeze(value: Any) -> FrozenJSONValue:
    if isinstance(value, (FrozenJSONObject, FrozenJSONArray)):
        return value
    if isinstance(value, Mapping):
        return FrozenJSONObject(tuple((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return FrozenJSONArray(tuple(_freeze(item) for item in value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any] | FrozenJSONObject, path: str) -> FrozenJSONObject:
    if isinstance(value, FrozenJSONObject):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _validate_json_tree(value, path)
    frozen = _freeze(value)
    assert isinstance(frozen, FrozenJSONObject)
    return frozen


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenJSONObject):
        return {key: _thaw(item) for key, item in value.items}
    if isinstance(value, FrozenJSONArray):
        return [_thaw(item) for item in value.items]
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strict_json_loads(data: bytes) -> Any:
    if not isinstance(data, bytes):
        raise TypeError("classification loaders require bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 byte-order mark is forbidden")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("input is not valid UTF-8") from error
    _check_json_nesting(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_int=_parse_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error
    _validate_json_tree(value)
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if data != canonical:
        raise ValueError("classification JSON bytes are not canonical")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path}: expected array")
    return value


def _fields(
    value: Mapping[str, Any],
    required: set[str],
    path: str,
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path}: missing field {missing[0]}")
    unexpected = sorted(set(value) - required - optional)
    if unexpected:
        raise ValueError(f"{path}: unexpected field {unexpected[0]}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: expected string")
    if not value or value.strip() != value:
        raise ValueError(f"{path}: expected nonempty trimmed string")
    if _looks_absolute_path(value):
        raise ValueError(f"{path}: absolute path is forbidden in public records")
    return value


def _identifier(value: Any, path: str) -> str:
    text = _string(value, path)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise ValueError(f"{path}: invalid identifier")
    return text


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path}: integer must be >= {minimum}")
    return value


def _bit(value: Any, path: str) -> int:
    bit = _integer(value, path)
    if bit not in (0, 1):
        raise ValueError(f"{path}: expected bit")
    return bit


def _digest(value: Any, path: str) -> str:
    text = _string(value, path)
    if _DIGEST_RE.fullmatch(text) is None:
        raise ValueError(f"{path}: expected sha256:<64 lowercase hex digits>")
    return text


def _schema_version(value: Any, path: str) -> int:
    version = _integer(value, path)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {version}")
    return version


def _rational(value: Any, path: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{path}: rational must be a string")
    if _RATIONAL_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: invalid canonical rational")
    if "/" in value:
        numerator_text, denominator_text = value.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if denominator == 1:
            raise ValueError(f"{path}: rational denominator one must be omitted")
        if math.gcd(abs(numerator), denominator) != 1:
            raise ValueError(f"{path}: rational must be reduced")
        return Fraction(numerator, denominator)
    return Fraction(int(value), 1)


def _phase(value: Any, path: str) -> Fraction:
    phase = _rational(value, path)
    if not Fraction(0) <= phase < Fraction(1):
        raise ValueError(f"{path}: phase must lie in [0,1)")
    return phase


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _domain_digest(domain: str, value: Any) -> str:
    """Hash canonical JSON with an explicit v1 protocol domain separator."""

    _identifier(domain, "$digest.domain")
    encoded = json.dumps(
        _thaw(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prefix = f"mathpsg-classifier-v1|{domain}|".encode("ascii")
    return "sha256:" + hashlib.sha256(prefix + encoded).hexdigest()


def _json_object_fields(value: FrozenJSONObject, path: str) -> Mapping[str, Any]:
    result = _thaw(value)
    assert isinstance(result, Mapping)
    return result


def _comparison(value: Any, path: str) -> Literal["match", "mismatch"]:
    comparison = _string(value, path)
    if comparison not in ("match", "mismatch"):
        raise ValueError(f"{path}: expected match or mismatch")
    return comparison  # type: ignore[return-value]


def _nullable_mapping(value: Any, path: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, path)


def _nullable_integer_array(value: Any, path: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(
        _integer(item, f"{path}[{index}]", minimum=0)
        for index, item in enumerate(_array(value, path))
    )


def _validate_geometry_evidence(value: FrozenJSONObject, path: str) -> None:
    row = _json_object_fields(value, path)
    _fields(
        row,
        {
            "comparison",
            "point_orbit_digest",
            "candidate_orbit_digest",
            "branch_bijection",
            "lattice_shifts",
            "mismatch_witness",
        },
        path,
    )
    comparison = _comparison(row["comparison"], f"{path}.comparison")
    _digest(row["point_orbit_digest"], f"{path}.point_orbit_digest")
    _digest(row["candidate_orbit_digest"], f"{path}.candidate_orbit_digest")
    bijection = _nullable_integer_array(row["branch_bijection"], f"{path}.branch_bijection")
    shifts_value = row["lattice_shifts"]
    shifts: tuple[tuple[Fraction, ...], ...] | None = None
    if shifts_value is not None:
        shifts = tuple(
            tuple(
                _rational(coordinate, f"{path}.lattice_shifts[{row_index}][{column}]")
                for column, coordinate in enumerate(
                    _array(shift, f"{path}.lattice_shifts[{row_index}]")
                )
            )
            for row_index, shift in enumerate(_array(shifts_value, f"{path}.lattice_shifts"))
        )
    mismatch = _nullable_mapping(row["mismatch_witness"], f"{path}.mismatch_witness")
    if comparison == "match":
        if bijection is None or shifts is None or mismatch is not None:
            raise ValueError(f"{path}: match requires bijection/shifts and no mismatch witness")
        if len(bijection) != len(shifts) or len(set(bijection)) != len(bijection):
            raise ValueError(f"{path}: branch bijection and lattice shifts are inconsistent")
    elif mismatch is None:
        raise ValueError(f"{path}: mismatch requires a mismatch witness")


def _validate_element_evidence(
    value: FrozenJSONObject,
    path: str,
    *,
    point_field: str,
    candidate_field: str,
) -> None:
    row = _json_object_fields(value, path)
    _fields(
        row,
        {
            "comparison",
            point_field,
            candidate_field,
            "element_bijection",
            "comparison_witness",
            "mismatch_witness",
        },
        path,
    )
    comparison = _comparison(row["comparison"], f"{path}.comparison")
    point_elements = _array(row[point_field], f"{path}.{point_field}")
    candidate_elements = _array(row[candidate_field], f"{path}.{candidate_field}")
    for field, elements in ((point_field, point_elements), (candidate_field, candidate_elements)):
        for index, element in enumerate(elements):
            _mapping(element, f"{path}.{field}[{index}]")
    bijection = _nullable_integer_array(row["element_bijection"], f"{path}.element_bijection")
    witness = _nullable_mapping(row["comparison_witness"], f"{path}.comparison_witness")
    mismatch = _nullable_mapping(row["mismatch_witness"], f"{path}.mismatch_witness")
    if comparison == "match":
        if bijection is None or witness is None or mismatch is not None:
            raise ValueError(f"{path}: match requires bijection/comparison witness only")
        if len(bijection) != len(point_elements) or len(point_elements) != len(candidate_elements):
            raise ValueError(f"{path}: element bijection has the wrong size")
        if tuple(sorted(bijection)) != tuple(range(len(candidate_elements))):
            raise ValueError(f"{path}: element bijection must be a permutation")
    elif mismatch is None:
        raise ValueError(f"{path}: mismatch requires a mismatch witness")


def _parse_wyckoff_id(value: Any, path: str) -> tuple[str, int, str]:
    text = _string(value, path)
    match = _WYCKOFF_ID_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"{path}: invalid canonical Wyckoff identifier")
    group = int(match.group(1))
    if not 1 <= group <= 230:
        raise ValueError(f"{path}: Wyckoff space group must be in 1..230")
    return text, group, match.group(2)


def _identifiers(value: Any, path: str) -> tuple[str, ...]:
    items = _array(value, path)
    result = tuple(_identifier(item, f"{path}[{index}]") for index, item in enumerate(items))
    if len(set(result)) != len(result):
        raise ValueError(f"{path}: identifiers must be unique")
    return result


def _identifier_sequence(value: Any, path: str) -> tuple[str, ...]:
    """Parse nonempty ordered instance evidence; repeated values are meaningful."""

    items = _array(value, path)
    if not items:
        raise ValueError(f"{path}: expected nonempty identifier sequence")
    return tuple(
        _identifier(item, f"{path}[{index}]") for index, item in enumerate(items)
    )


@dataclass(frozen=True, slots=True)
class OrbitInstance:
    instance_id: str
    wyckoff_id: str
    parameter_mode: Literal["family", "point"]
    parameter_values: tuple[Fraction, ...] = ()
    species: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.instance_id, "$OrbitInstance.instance_id")
        _parse_wyckoff_id(self.wyckoff_id, "$OrbitInstance.wyckoff_id")
        if self.parameter_mode not in ("family", "point"):
            raise ValueError("$OrbitInstance.parameter_mode: expected family or point")
        values = tuple(Fraction(value) for value in self.parameter_values)
        object.__setattr__(self, "parameter_values", values)
        if self.parameter_mode == "family" and values:
            raise ValueError("$OrbitInstance: family parameter values must be empty")
        if self.parameter_mode == "point" and not values:
            raise ValueError("$OrbitInstance: point parameter values must be nonempty")
        if self.species is not None:
            _string(self.species, "$OrbitInstance.species")


@dataclass(frozen=True, slots=True)
class ClassificationRequest:
    schema_version: int
    space_group: int
    setting_id: str
    igg: Literal["Z2", "U1"]
    time_reversal: bool
    orbits: tuple[OrbitInstance, ...]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "$ClassificationRequest.schema_version")
        if not 1 <= _integer(self.space_group, "$ClassificationRequest.space_group") <= 230:
            raise ValueError("$ClassificationRequest.space_group: must be in 1..230")
        if _SETTING_RE.fullmatch(_string(self.setting_id, "$ClassificationRequest.setting_id")) is None:
            raise ValueError("$ClassificationRequest.setting_id: invalid setting")
        if self.igg not in ("Z2", "U1"):
            raise ValueError("$ClassificationRequest.igg: expected Z2 or U1")
        if not isinstance(self.time_reversal, bool):
            raise TypeError("$ClassificationRequest.time_reversal: expected boolean")
        object.__setattr__(self, "orbits", tuple(self.orbits))
        if not self.orbits:
            raise ValueError("$ClassificationRequest.orbits: expected a nonempty list")
        if any(not isinstance(orbit, OrbitInstance) for orbit in self.orbits):
            raise TypeError("$ClassificationRequest.orbits: expected OrbitInstance records")
        ids = tuple(orbit.instance_id for orbit in self.orbits)
        if len(set(ids)) != len(ids):
            raise ValueError("$ClassificationRequest.orbits: instance_id values must be unique")
        for orbit in self.orbits:
            _, group, setting = _parse_wyckoff_id(orbit.wyckoff_id, "$ClassificationRequest.orbits")
            if group != self.space_group:
                raise ValueError("$ClassificationRequest.orbits: Wyckoff space group does not match request")
            if setting != self.setting_id:
                raise ValueError("$ClassificationRequest.orbits: Wyckoff setting does not match request")


@dataclass(frozen=True, slots=True)
class StructuredFailure:
    code: FailureCode
    stage: str
    message: str
    context: FrozenJSONObject

    def __post_init__(self) -> None:
        if self.code not in _FAILURE_CODES:
            raise ValueError("$StructuredFailure.code: unknown failure code")
        _identifier(self.stage, "$StructuredFailure.stage")
        _string(self.message, "$StructuredFailure.message")
        object.__setattr__(self, "context", _freeze_mapping(self.context, "$StructuredFailure.context"))


@dataclass(frozen=True, slots=True)
class ObstructedBranch:
    stratum_id: str
    skeleton_ids: tuple[str, ...]
    witness: FrozenJSONObject

    def __post_init__(self) -> None:
        _identifier(self.stratum_id, "$ObstructedBranch.stratum_id")
        object.__setattr__(
            self,
            "skeleton_ids",
            _identifier_sequence(self.skeleton_ids, "$ObstructedBranch.skeleton_ids"),
        )
        object.__setattr__(self, "witness", _freeze_mapping(self.witness, "$ObstructedBranch.witness"))


@dataclass(frozen=True, slots=True)
class LayerRecord:
    layer_id: str
    status: Literal["complete", "failed"]
    framed_strata: tuple[FrozenJSONObject, ...]
    unframed_quotient: FrozenJSONObject | None
    obstructed_branches: tuple[ObstructedBranch, ...]
    failures: tuple[StructuredFailure, ...]

    def __post_init__(self) -> None:
        _identifier(self.layer_id, "$LayerRecord.layer_id")
        if self.status not in ("complete", "failed"):
            raise ValueError("$LayerRecord.status: expected complete or failed")
        frozen_strata = tuple(
            _parse_stratum(_thaw(stratum), f"$LayerRecord.framed_strata[{index}]")
            for index, stratum in enumerate(self.framed_strata)
        )
        object.__setattr__(self, "framed_strata", frozen_strata)
        if self.unframed_quotient is not None:
            object.__setattr__(
                self,
                "unframed_quotient",
                _parse_unframed_quotient(
                    _thaw(self.unframed_quotient),
                    "$LayerRecord.unframed_quotient",
                ),
            )
        object.__setattr__(self, "obstructed_branches", tuple(self.obstructed_branches))
        object.__setattr__(self, "failures", tuple(self.failures))
        if any(
            not isinstance(obstruction, ObstructedBranch)
            for obstruction in self.obstructed_branches
        ):
            raise TypeError("$LayerRecord.obstructed_branches: expected ObstructedBranch records")
        if any(not isinstance(failure, StructuredFailure) for failure in self.failures):
            raise TypeError("$LayerRecord.failures: expected StructuredFailure records")
        if self.status == "complete":
            if self.failures:
                raise ValueError("$LayerRecord: complete layer may not contain hard failures")
            if self.unframed_quotient is None:
                raise ValueError("$LayerRecord: complete layer requires an unframed quotient")
            if not self.framed_strata and not self.obstructed_branches:
                raise ValueError(
                    "$LayerRecord: complete layer requires a nonempty or obstructed branch"
                )
        else:
            if not self.failures:
                raise ValueError("$LayerRecord: failed layer requires a hard failure")
            if self.unframed_quotient is not None:
                raise ValueError("$LayerRecord: failed layer may not claim an unframed quotient")
        _validate_layer_contents(self, "$LayerRecord")


@dataclass(frozen=True, slots=True)
class CandidateGeometryEvidence:
    candidate_wyckoff_id: str
    family_geometry_digest: str
    literal_stabilizer_digest: str
    transported_inclusion_digest: str
    geometry_evidence: FrozenJSONObject
    stabilizer_evidence: FrozenJSONObject
    inclusion_evidence: FrozenJSONObject
    geometry_comparison_digest: str
    stabilizer_comparison_digest: str
    inclusion_comparison_digest: str
    geometry_match: bool
    stabilizer_match: bool
    inclusion_match: bool
    rejection_codes: tuple[Literal["geometry", "stabilizer", "inclusion"], ...]

    def __post_init__(self) -> None:
        _parse_wyckoff_id(self.candidate_wyckoff_id, "$CandidateGeometryEvidence.candidate_wyckoff_id")
        for name in ("geometry_evidence", "stabilizer_evidence", "inclusion_evidence"):
            object.__setattr__(
                self,
                name,
                _freeze_mapping(getattr(self, name), f"$CandidateGeometryEvidence.{name}"),
            )
        for name in (
            "family_geometry_digest",
            "literal_stabilizer_digest",
            "transported_inclusion_digest",
            "geometry_comparison_digest",
            "stabilizer_comparison_digest",
            "inclusion_comparison_digest",
        ):
            _digest(getattr(self, name), f"$CandidateGeometryEvidence.{name}")
        _validate_geometry_evidence(
            self.geometry_evidence,
            "$CandidateGeometryEvidence.geometry_evidence",
        )
        _validate_element_evidence(
            self.stabilizer_evidence,
            "$CandidateGeometryEvidence.stabilizer_evidence",
            point_field="point_elements",
            candidate_field="candidate_elements",
        )
        _validate_element_evidence(
            self.inclusion_evidence,
            "$CandidateGeometryEvidence.inclusion_evidence",
            point_field="point_transported_elements",
            candidate_field="candidate_transported_elements",
        )
        for name in ("geometry_match", "stabilizer_match", "inclusion_match"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"$CandidateGeometryEvidence.{name}: expected boolean")
        for kind in ("geometry", "stabilizer", "inclusion"):
            evidence = getattr(self, f"{kind}_evidence")
            assert isinstance(evidence, FrozenJSONObject)
            comparison = evidence["comparison"]
            if (comparison == "match") != getattr(self, f"{kind}_match"):
                raise ValueError(
                    f"$CandidateGeometryEvidence.{kind}_match: inconsistent with evidence"
                )
        object.__setattr__(self, "rejection_codes", tuple(self.rejection_codes))
        expected = tuple(
            name
            for name, matched in (
                ("geometry", self.geometry_match),
                ("stabilizer", self.stabilizer_match),
                ("inclusion", self.inclusion_match),
            )
            if not matched
        )
        if self.rejection_codes != expected:
            raise ValueError(
                "$CandidateGeometryEvidence.rejection_codes: must exactly name failed comparisons"
            )


def _candidate_tuple_digest(candidates: Sequence[CandidateGeometryEvidence]) -> str:
    return _domain_digest(
        "routing-candidate-ids",
        [candidate.candidate_wyckoff_id for candidate in candidates],
    )


def _comparison_digest(
    route: "InstanceParameterRoute",
    candidate: CandidateGeometryEvidence,
    kind: Literal["geometry", "stabilizer", "inclusion"],
) -> str:
    point_digest = getattr(route, f"point_{kind}_digest")
    source_digest = {
        "geometry": candidate.family_geometry_digest,
        "stabilizer": candidate.literal_stabilizer_digest,
        "inclusion": candidate.transported_inclusion_digest,
    }[kind]
    evidence = getattr(candidate, f"{kind}_evidence")
    return _domain_digest(
        f"routing-{kind}-comparison",
        {
            "candidate_wyckoff_id": candidate.candidate_wyckoff_id,
            "evidence": _thaw(evidence),
            "exact_point": [_rational_text(value) for value in route.exact_point],
            "instance_id": route.instance_id,
            "point_digest": point_digest,
            "requested_wyckoff_id": route.requested_wyckoff_id,
            "source_digest": source_digest,
        },
    )


@dataclass(frozen=True, slots=True)
class InstanceParameterRoute:
    instance_id: str
    requested_wyckoff_id: str
    exact_point: tuple[Fraction, ...]
    outcome: Literal["same_stratum", "resolved_specialization", "unresolved_specialization"]
    resolved_wyckoff_id: str | None
    point_geometry_digest: str
    point_stabilizer_digest: str
    point_inclusion_digest: str
    candidate_set_digest: str
    candidates: tuple[CandidateGeometryEvidence, ...]

    def __post_init__(self) -> None:
        _identifier(self.instance_id, "$InstanceParameterRoute.instance_id")
        _, group, setting = _parse_wyckoff_id(
            self.requested_wyckoff_id, "$InstanceParameterRoute.requested_wyckoff_id"
        )
        object.__setattr__(self, "exact_point", tuple(Fraction(value) for value in self.exact_point))
        if not self.exact_point:
            raise ValueError("$InstanceParameterRoute.exact_point: expected nonempty point")
        if self.outcome not in (
            "same_stratum",
            "resolved_specialization",
            "unresolved_specialization",
        ):
            raise ValueError("$InstanceParameterRoute.outcome: unknown outcome")
        if self.resolved_wyckoff_id is not None:
            _, resolved_group, resolved_setting = _parse_wyckoff_id(
                self.resolved_wyckoff_id, "$InstanceParameterRoute.resolved_wyckoff_id"
            )
            if (resolved_group, resolved_setting) != (group, setting):
                raise ValueError("$InstanceParameterRoute: resolved ID changes group or setting")
        if self.outcome == "same_stratum" and self.resolved_wyckoff_id != self.requested_wyckoff_id:
            raise ValueError("$InstanceParameterRoute: same_stratum requires requested resolved ID")
        if self.outcome == "resolved_specialization" and (
            self.resolved_wyckoff_id is None
            or self.resolved_wyckoff_id == self.requested_wyckoff_id
        ):
            raise ValueError(
                "$InstanceParameterRoute: resolved_specialization requires a distinct resolved_wyckoff_id"
            )
        if self.outcome == "unresolved_specialization" and self.resolved_wyckoff_id is not None:
            raise ValueError(
                "$InstanceParameterRoute: unresolved_specialization requires null resolved_wyckoff_id"
            )
        for name in (
            "point_geometry_digest",
            "point_stabilizer_digest",
            "point_inclusion_digest",
            "candidate_set_digest",
        ):
            _digest(getattr(self, name), f"$InstanceParameterRoute.{name}")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates:
            raise ValueError("$InstanceParameterRoute.candidates: complete candidate evidence is required")
        if any(
            not isinstance(candidate, CandidateGeometryEvidence)
            for candidate in self.candidates
        ):
            raise TypeError(
                "$InstanceParameterRoute.candidates: expected CandidateGeometryEvidence records"
            )
        candidate_ids = tuple(candidate.candidate_wyckoff_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("$InstanceParameterRoute.candidates: candidate IDs must be unique")
        for candidate in self.candidates:
            _, candidate_group, candidate_setting = _parse_wyckoff_id(
                candidate.candidate_wyckoff_id, "$InstanceParameterRoute.candidates"
            )
            if (candidate_group, candidate_setting) != (group, setting):
                raise ValueError("$InstanceParameterRoute.candidates: candidate changes group or setting")
            for kind in ("geometry", "stabilizer", "inclusion"):
                digest_name = f"{kind}_comparison_digest"
                if getattr(candidate, digest_name) != _comparison_digest(self, candidate, kind):
                    raise ValueError(
                        f"$InstanceParameterRoute.candidates.{digest_name}: "
                        "does not bind point/candidate evidence"
                    )
        matches = tuple(
            candidate.candidate_wyckoff_id
            for candidate in self.candidates
            if candidate.geometry_match and candidate.stabilizer_match and candidate.inclusion_match
        )
        if self.outcome == "unresolved_specialization" and len(matches) == 1:
            raise ValueError(
                "$InstanceParameterRoute: unresolved route requires zero or multiple complete matches"
            )
        if self.outcome != "unresolved_specialization" and matches != (self.resolved_wyckoff_id,):
            raise ValueError("$InstanceParameterRoute: resolved route requires one matching candidate")
        if self.candidate_set_digest != _candidate_tuple_digest(self.candidates):
            raise ValueError(
                "$InstanceParameterRoute.candidate_set_digest: does not bind ordered candidates"
            )


@dataclass(frozen=True, slots=True)
class ParameterRoutingResult:
    status: Literal["parameter_specialization"]
    request_digest: str
    catalogue_manifest_digest: str
    space_group: int
    setting_id: str
    routes: tuple[InstanceParameterRoute, ...]

    def __post_init__(self) -> None:
        if self.status != "parameter_specialization":
            raise ValueError("$ParameterRoutingResult.status: invalid status")
        _digest(self.request_digest, "$ParameterRoutingResult.request_digest")
        _digest(
            self.catalogue_manifest_digest,
            "$ParameterRoutingResult.catalogue_manifest_digest",
        )
        if not 1 <= _integer(self.space_group, "$ParameterRoutingResult.space_group") <= 230:
            raise ValueError("$ParameterRoutingResult.space_group: must be in 1..230")
        if _SETTING_RE.fullmatch(
            _string(self.setting_id, "$ParameterRoutingResult.setting_id")
        ) is None:
            raise ValueError("$ParameterRoutingResult.setting_id: invalid setting")
        object.__setattr__(self, "routes", tuple(self.routes))
        if not self.routes:
            raise ValueError("$ParameterRoutingResult.routes: expected nonempty routes")
        if any(not isinstance(route, InstanceParameterRoute) for route in self.routes):
            raise TypeError("$ParameterRoutingResult.routes: expected InstanceParameterRoute records")
        ids = tuple(route.instance_id for route in self.routes)
        if len(set(ids)) != len(ids):
            raise ValueError("$ParameterRoutingResult.routes: instance IDs must be unique")
        if all(route.outcome == "same_stratum" for route in self.routes):
            raise ValueError(
                "$ParameterRoutingResult: all same-stratum points belong in ClassificationRecord"
            )
        for route in self.routes:
            _, group, setting = _parse_wyckoff_id(
                route.requested_wyckoff_id, "$ParameterRoutingResult.routes"
            )
            if (group, setting) != (self.space_group, self.setting_id):
                raise ValueError(
                    "$ParameterRoutingResult.routes: route group/setting does not match result"
                )


@dataclass(frozen=True, slots=True)
class ClassificationRecord:
    request_digest: str
    catalogue_manifest_digest: str
    layer: LayerRecord
    point_routes: tuple[InstanceParameterRoute, ...]
    routing_verification_digest: str | None

    def __post_init__(self) -> None:
        _digest(self.request_digest, "$ClassificationRecord.request_digest")
        _digest(
            self.catalogue_manifest_digest,
            "$ClassificationRecord.catalogue_manifest_digest",
        )
        object.__setattr__(self, "point_routes", tuple(self.point_routes))
        if any(not isinstance(route, InstanceParameterRoute) for route in self.point_routes):
            raise TypeError("$ClassificationRecord.point_routes: expected InstanceParameterRoute records")
        route_ids = tuple(route.instance_id for route in self.point_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("$ClassificationRecord.point_routes: instance IDs must be unique")
        if any(route.outcome != "same_stratum" for route in self.point_routes):
            raise ValueError("$ClassificationRecord.point_routes: only same_stratum routes are allowed")
        if self.point_routes and self.routing_verification_digest is None:
            raise ValueError(
                "$ClassificationRecord.routing_verification_digest: required for point routes"
            )
        if not self.point_routes and self.routing_verification_digest is not None:
            raise ValueError(
                "$ClassificationRecord.routing_verification_digest: family-only record must use null"
            )
        if self.routing_verification_digest is not None:
            _digest(
                self.routing_verification_digest,
                "$ClassificationRecord.routing_verification_digest",
            )
        if not isinstance(self.layer, LayerRecord):
            raise TypeError("$ClassificationRecord.layer: expected LayerRecord")


def _parse_orbit(value: Any, path: str) -> OrbitInstance:
    row = _mapping(value, path)
    _fields(
        row,
        {"instance_id", "wyckoff_id", "parameter_mode", "parameter_values"},
        path,
        {"species"},
    )
    instance_id = _identifier(row["instance_id"], f"{path}.instance_id")
    wyckoff_id, _, _ = _parse_wyckoff_id(row["wyckoff_id"], f"{path}.wyckoff_id")
    mode = _string(row["parameter_mode"], f"{path}.parameter_mode")
    if mode not in ("family", "point"):
        raise ValueError(f"{path}.parameter_mode: expected family or point")
    values = tuple(
        _rational(item, f"{path}.parameter_values[{index}]")
        for index, item in enumerate(_array(row["parameter_values"], f"{path}.parameter_values"))
    )
    species_value = row.get("species")
    if species_value is not None:
        species_value = _string(species_value, f"{path}.species")
    return OrbitInstance(instance_id, wyckoff_id, mode, values, species_value)  # type: ignore[arg-type]


def loads_classification_request(data: bytes) -> ClassificationRequest:
    value = _mapping(_strict_json_loads(data), "$request")
    _fields(
        value,
        {"schema_version", "space_group", "setting_id", "igg", "time_reversal", "orbits"},
        "$request",
    )
    version = _schema_version(value["schema_version"], "$request.schema_version")
    group = _integer(value["space_group"], "$request.space_group", minimum=1)
    if group > 230:
        raise ValueError("$request.space_group: must be in 1..230")
    setting = _string(value["setting_id"], "$request.setting_id")
    if _SETTING_RE.fullmatch(setting) is None:
        raise ValueError("$request.setting_id: invalid setting")
    igg = _string(value["igg"], "$request.igg")
    if igg not in ("Z2", "U1"):
        raise ValueError("$request.igg: expected Z2 or U1")
    time_reversal = value["time_reversal"]
    if not isinstance(time_reversal, bool):
        raise TypeError("$request.time_reversal: expected boolean")
    orbit_rows = _array(value["orbits"], "$request.orbits")
    if not orbit_rows:
        raise ValueError("$request.orbits: expected a nonempty list")
    orbits = tuple(_parse_orbit(row, f"$request.orbits[{index}]") for index, row in enumerate(orbit_rows))
    return ClassificationRequest(version, group, setting, igg, time_reversal, orbits)  # type: ignore[arg-type]


def _integer_matrix(
    value: Any,
    path: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    matrix_rows = _array(value, path)
    if rows is not None and len(matrix_rows) != rows:
        raise ValueError(f"{path}: expected {rows} rows")
    parsed: list[tuple[int, ...]] = []
    width = columns
    for row_index, row in enumerate(matrix_rows):
        parsed_row = tuple(
            _integer(item, f"{path}[{row_index}][{column}]")
            for column, item in enumerate(_array(row, f"{path}[{row_index}]"))
        )
        if width is None:
            width = len(parsed_row)
        if len(parsed_row) != width:
            raise ValueError(f"{path}[{row_index}]: matrix width mismatch")
        parsed.append(parsed_row)
    return tuple(parsed)


def _phase_matrix(
    value: Any,
    path: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> tuple[tuple[Fraction, ...], ...]:
    matrix_rows = _array(value, path)
    if rows is not None and len(matrix_rows) != rows:
        raise ValueError(f"{path}: expected {rows} rows")
    parsed: list[tuple[Fraction, ...]] = []
    width = columns
    for row_index, row in enumerate(matrix_rows):
        parsed_row = tuple(
            _phase(item, f"{path}[{row_index}][{column}]")
            for column, item in enumerate(_array(row, f"{path}[{row_index}]"))
        )
        if width is None:
            width = len(parsed_row)
        if len(parsed_row) != width:
            raise ValueError(f"{path}[{row_index}]: matrix width mismatch")
        parsed.append(parsed_row)
    return tuple(parsed)


def _parse_primal_torsor_chart(value: Any, path: str) -> FrozenJSONObject:
    row = _mapping(value, path)
    _fields(
        row,
        {
            "raw_dimension",
            "free_lifts",
            "torsion_lifts",
            "free_character_pairing",
            "torsion_pairing",
            "quotient_witnesses",
        },
        path,
    )
    raw_dimension = _integer(row["raw_dimension"], f"{path}.raw_dimension", minimum=0)
    free_lifts = _integer_matrix(row["free_lifts"], f"{path}.free_lifts", rows=raw_dimension)
    torsion_lifts = _phase_matrix(
        row["torsion_lifts"], f"{path}.torsion_lifts", rows=raw_dimension
    )
    free_rank = len(free_lifts[0]) if free_lifts else 0
    torsion_rank = len(torsion_lifts[0]) if torsion_lifts else 0
    free_pairing = _integer_matrix(
        row["free_character_pairing"],
        f"{path}.free_character_pairing",
        rows=free_rank,
        columns=free_rank,
    )
    if free_pairing != tuple(
        tuple(1 if row_index == column else 0 for column in range(free_rank))
        for row_index in range(free_rank)
    ):
        raise ValueError(f"{path}.free_character_pairing: expected identity matrix")
    torsion_pairing = _phase_matrix(
        row["torsion_pairing"],
        f"{path}.torsion_pairing",
        rows=torsion_rank,
        columns=torsion_rank,
    )
    if any(
        torsion_pairing[row_index][column] != 0
        for row_index in range(torsion_rank)
        for column in range(torsion_rank)
        if row_index != column
    ):
        raise ValueError(f"{path}.torsion_pairing: off-diagonal phases must vanish")
    witnesses = _array(row["quotient_witnesses"], f"{path}.quotient_witnesses")
    for index, witness in enumerate(witnesses):
        _integer_matrix(witness, f"{path}.quotient_witnesses[{index}]")
    return _freeze_mapping(row, path)


def loads_classifier_certificate(data: bytes) -> FrozenJSONObject:
    value = _mapping(_strict_json_loads(data), "$certificate")
    _fields(
        value,
        {
            "schema_version",
            "certificate_kind",
            "certificate_digest",
            "dependency_digests",
            "payload",
        },
        "$certificate",
    )
    version = _schema_version(value["schema_version"], "$certificate.schema_version")
    kind = _identifier(value["certificate_kind"], "$certificate.certificate_kind")
    if kind != "primal-torsor-chart":
        raise ValueError("$certificate.certificate_kind: unsupported certificate kind")
    dependency_rows = _mapping(value["dependency_digests"], "$certificate.dependency_digests")
    dependencies: dict[str, str] = {}
    for name, digest in dependency_rows.items():
        dependencies[_identifier(name, "$certificate.dependency_digests key")] = _digest(
            digest, f"$certificate.dependency_digests.{name}"
        )
    payload = _parse_primal_torsor_chart(value["payload"], "$certificate.payload")
    claimed = _digest(value["certificate_digest"], "$certificate.certificate_digest")
    expected = _domain_digest(
        "certificate-envelope",
        {
            "certificate_kind": kind,
            "dependency_digests": dependencies,
            "payload": _thaw(payload),
            "schema_version": version,
        },
    )
    if claimed != expected:
        raise ValueError("$certificate.certificate_digest: does not bind certificate payload")
    return _freeze_mapping(value, "$certificate")


def _parse_failure(value: Any, path: str) -> StructuredFailure:
    row = _mapping(value, path)
    _fields(row, {"code", "stage", "message", "context"}, path)
    code = _string(row["code"], f"{path}.code")
    stage = _identifier(row["stage"], f"{path}.stage")
    message = _string(row["message"], f"{path}.message")
    context = _mapping(row["context"], f"{path}.context")
    return StructuredFailure(code, stage, message, context)


def _parse_obstruction(value: Any, path: str) -> ObstructedBranch:
    row = _mapping(value, path)
    _fields(row, {"stratum_id", "skeleton_ids", "witness"}, path)
    stratum_id = _identifier(row["stratum_id"], f"{path}.stratum_id")
    skeletons = _identifier_sequence(row["skeleton_ids"], f"{path}.skeleton_ids")
    witness = _mapping(row["witness"], f"{path}.witness")
    if "character" in witness:
        for index, bit in enumerate(_array(witness["character"], f"{path}.witness.character")):
            _bit(bit, f"{path}.witness.character[{index}]")
    if "phase" in witness:
        _phase(witness["phase"], f"{path}.witness.phase")
    return ObstructedBranch(stratum_id, skeletons, witness)


def _parse_z2_stratum(row: Mapping[str, Any], path: str) -> FrozenJSONObject:
    required = {
        "kind",
        "stratum_id",
        "skeleton_ids",
        "dimension",
        "basepoint",
        "quotient_basis",
        "residual_orbit_certificate",
        "framed_finite_cardinality",
        "unframed_finite_cardinality",
    }
    _fields(row, required, path)
    _identifier(row["stratum_id"], f"{path}.stratum_id")
    _identifier_sequence(row["skeleton_ids"], f"{path}.skeleton_ids")
    dimension = _integer(row["dimension"], f"{path}.dimension", minimum=0)
    basepoint = _array(row["basepoint"], f"{path}.basepoint")
    for index, bit in enumerate(basepoint):
        _bit(bit, f"{path}.basepoint[{index}]")
    basis = _array(row["quotient_basis"], f"{path}.quotient_basis")
    if len(basis) != dimension:
        raise ValueError(f"{path}.quotient_basis: row count must equal dimension")
    for row_index, basis_row in enumerate(basis):
        items = _array(basis_row, f"{path}.quotient_basis[{row_index}]")
        if len(items) != len(basepoint):
            raise ValueError(f"{path}.quotient_basis[{row_index}]: width mismatch")
        for column, bit in enumerate(items):
            _bit(bit, f"{path}.quotient_basis[{row_index}][{column}]")
    _digest(row["residual_orbit_certificate"], f"{path}.residual_orbit_certificate")
    framed = _integer(row["framed_finite_cardinality"], f"{path}.framed_finite_cardinality", minimum=0)
    unframed = _integer(row["unframed_finite_cardinality"], f"{path}.unframed_finite_cardinality", minimum=0)
    if framed != 2**dimension:
        raise ValueError(f"{path}.framed_finite_cardinality: inconsistent with dimension")
    if unframed > framed:
        raise ValueError(f"{path}.unframed_finite_cardinality: cannot exceed framed count")
    return _freeze_mapping(row, path)


def _parse_u1_stratum(row: Mapping[str, Any], path: str) -> FrozenJSONObject:
    required = {
        "kind",
        "stratum_id",
        "rho_bits",
        "skeleton_ids",
        "free_rank",
        "torsion_orders",
        "basepoint_phases",
        "formal_parameters",
        "primal_chart_digest",
        "affine_arrow_ids",
        "framed_torsor_summary",
        "unframed_torsor_orbit_summary",
    }
    _fields(row, required, path, {"finite_class_count"})
    _identifier(row["stratum_id"], f"{path}.stratum_id")
    for index, bit in enumerate(_array(row["rho_bits"], f"{path}.rho_bits")):
        _bit(bit, f"{path}.rho_bits[{index}]")
    _identifier_sequence(row["skeleton_ids"], f"{path}.skeleton_ids")
    free_rank = _integer(row["free_rank"], f"{path}.free_rank", minimum=0)
    torsion = tuple(
        _integer(order, f"{path}.torsion_orders[{index}]", minimum=2)
        for index, order in enumerate(_array(row["torsion_orders"], f"{path}.torsion_orders"))
    )
    if any(right % left != 0 for left, right in zip(torsion, torsion[1:])):
        raise ValueError(f"{path}.torsion_orders: orders must divide the next")
    for index, phase in enumerate(_array(row["basepoint_phases"], f"{path}.basepoint_phases")):
        _phase(phase, f"{path}.basepoint_phases[{index}]")
    parameters = _identifiers(row["formal_parameters"], f"{path}.formal_parameters")
    if len(parameters) != free_rank or any(_PARAMETER_RE.fullmatch(item) is None for item in parameters):
        raise ValueError(f"{path}.formal_parameters: count/names must match free_rank")
    _digest(row["primal_chart_digest"], f"{path}.primal_chart_digest")
    _identifiers(row["affine_arrow_ids"], f"{path}.affine_arrow_ids")
    summary = _mapping(row["framed_torsor_summary"], f"{path}.framed_torsor_summary")
    _fields(summary, {"free_rank", "torsion_orders"}, f"{path}.framed_torsor_summary")
    summary_rank = _integer(
        summary["free_rank"], f"{path}.framed_torsor_summary.free_rank", minimum=0
    )
    summary_torsion = tuple(
        _integer(
            order,
            f"{path}.framed_torsor_summary.torsion_orders[{index}]",
            minimum=2,
        )
        for index, order in enumerate(
            _array(
                summary["torsion_orders"],
                f"{path}.framed_torsor_summary.torsion_orders",
            )
        )
    )
    if any(right % left != 0 for left, right in zip(summary_torsion, summary_torsion[1:])):
        raise ValueError(
            f"{path}.framed_torsor_summary.torsion_orders: orders must divide the next"
        )
    if summary_rank != free_rank or summary_torsion != torsion:
        raise ValueError(f"{path}.framed_torsor_summary: inconsistent torsor invariants")
    orbit_summary = _mapping(
        row["unframed_torsor_orbit_summary"], f"{path}.unframed_torsor_orbit_summary"
    )
    _fields(orbit_summary, {"presentation_digest"}, f"{path}.unframed_torsor_orbit_summary")
    _digest(orbit_summary["presentation_digest"], f"{path}.unframed_torsor_orbit_summary.presentation_digest")
    if free_rank > 0 and "finite_class_count" in row:
        raise ValueError(f"{path}: continuous stratum cannot claim finite_class_count")
    if free_rank == 0:
        if "finite_class_count" not in row:
            raise ValueError(f"{path}: rank-zero torsor requires finite_class_count")
        count = _integer(row["finite_class_count"], f"{path}.finite_class_count", minimum=0)
        expected = math.prod(torsion)
        if count != expected:
            raise ValueError(f"{path}.finite_class_count: expected {expected}")
    return _freeze_mapping(row, path)


def _parse_stratum(value: Any, path: str) -> FrozenJSONObject:
    row = _mapping(value, path)
    kind = _string(row.get("kind"), f"{path}.kind")
    if kind == "finite-affine-z2":
        return _parse_z2_stratum(row, path)
    if kind == "compact-u1-torsor":
        return _parse_u1_stratum(row, path)
    raise ValueError(f"{path}.kind: unknown stratum kind")


def _parse_unframed_quotient(value: Any, path: str) -> FrozenJSONObject:
    row = _mapping(value, path)
    _fields(
        row,
        {
            "certificate_digest",
            "framed_stratum_ids",
            "framed_finite_cardinality",
            "unframed_finite_cardinality",
            "continuous_orbit_presentations",
        },
        path,
    )
    _digest(row["certificate_digest"], f"{path}.certificate_digest")
    _identifiers(row["framed_stratum_ids"], f"{path}.framed_stratum_ids")
    finite_values = (row["framed_finite_cardinality"], row["unframed_finite_cardinality"])
    if (finite_values[0] is None) != (finite_values[1] is None):
        raise ValueError(f"{path}: finite cardinalities must both be null or integers")
    if finite_values[0] is not None:
        framed = _integer(finite_values[0], f"{path}.framed_finite_cardinality", minimum=0)
        unframed = _integer(finite_values[1], f"{path}.unframed_finite_cardinality", minimum=0)
        if unframed > framed:
            raise ValueError(f"{path}: unframed cardinality cannot exceed framed")
    presentations = _array(row["continuous_orbit_presentations"], f"{path}.continuous_orbit_presentations")
    for index, presentation in enumerate(presentations):
        item_path = f"{path}.continuous_orbit_presentations[{index}]"
        item = _mapping(presentation, item_path)
        _fields(item, {"stratum_id", "presentation_digest"}, item_path)
        _identifier(item["stratum_id"], f"{item_path}.stratum_id")
        _digest(item["presentation_digest"], f"{item_path}.presentation_digest")
    if presentations and finite_values[0] is not None:
        raise ValueError(f"{path}: continuous quotient cannot claim finite cardinalities")
    return _freeze_mapping(row, path)


def _validate_layer_contents(layer: LayerRecord, path: str) -> None:
    """Reconcile stratum-level claims with the aggregate unframed quotient."""

    stratum_ids = tuple(_string(stratum["stratum_id"], f"{path}.framed_strata") for stratum in layer.framed_strata)
    if len(set(stratum_ids)) != len(stratum_ids):
        raise ValueError(f"{path}.framed_strata: stratum IDs must be unique")
    obstruction_ids = tuple(branch.stratum_id for branch in layer.obstructed_branches)
    if len(set(obstruction_ids)) != len(obstruction_ids):
        raise ValueError(f"{path}.obstructed_branches: stratum IDs must be unique")
    if set(stratum_ids) & set(obstruction_ids):
        raise ValueError(f"{path}: nonempty and obstructed stratum IDs must be disjoint")
    if layer.unframed_quotient is None:
        return

    quotient = layer.unframed_quotient
    quotient_ids = tuple(quotient["framed_stratum_ids"])
    if quotient_ids != stratum_ids:
        raise ValueError(f"{path}.unframed_quotient: framed stratum IDs/order mismatch")

    continuous: dict[str, str] = {}
    framed_finite_total = 0
    for index, stratum in enumerate(layer.framed_strata):
        kind = stratum["kind"]
        if kind == "finite-affine-z2":
            framed_finite_total += _integer(
                stratum["framed_finite_cardinality"],
                f"{path}.framed_strata[{index}].framed_finite_cardinality",
                minimum=0,
            )
        elif kind == "compact-u1-torsor":
            free_rank = _integer(
                stratum["free_rank"], f"{path}.framed_strata[{index}].free_rank", minimum=0
            )
            if free_rank > 0:
                summary = stratum["unframed_torsor_orbit_summary"]
                assert isinstance(summary, FrozenJSONObject)
                continuous[stratum["stratum_id"]] = _digest(
                    summary["presentation_digest"],
                    f"{path}.framed_strata[{index}].unframed_torsor_orbit_summary.presentation_digest",
                )
            else:
                framed_finite_total += _integer(
                    stratum["finite_class_count"],
                    f"{path}.framed_strata[{index}].finite_class_count",
                    minimum=0,
                )

    presentations_value = quotient["continuous_orbit_presentations"]
    assert isinstance(presentations_value, FrozenJSONArray)
    presentations: list[tuple[str, str]] = []
    for index, item in enumerate(presentations_value):
        assert isinstance(item, FrozenJSONObject)
        presentations.append(
            (
                _identifier(item["stratum_id"], f"{path}.continuous[{index}].stratum_id"),
                _digest(
                    item["presentation_digest"],
                    f"{path}.continuous[{index}].presentation_digest",
                ),
            )
        )
    if tuple(presentations) != tuple(continuous.items()):
        raise ValueError(
            f"{path}.unframed_quotient.continuous_orbit_presentations: "
            "must exactly identify positive-rank U1 strata"
        )

    framed_claim = quotient["framed_finite_cardinality"]
    unframed_claim = quotient["unframed_finite_cardinality"]
    if continuous:
        if framed_claim is not None or unframed_claim is not None:
            raise ValueError(f"{path}.unframed_quotient: continuous layer requires null finite counts")
    else:
        framed_count = _integer(
            framed_claim, f"{path}.unframed_quotient.framed_finite_cardinality", minimum=0
        )
        unframed_count = _integer(
            unframed_claim, f"{path}.unframed_quotient.unframed_finite_cardinality", minimum=0
        )
        if framed_count != framed_finite_total:
            raise ValueError(
                f"{path}.unframed_quotient.framed_finite_cardinality: "
                f"expected aggregate {framed_finite_total}"
            )
        if unframed_count > framed_count:
            raise ValueError(f"{path}.unframed_quotient: unframed count exceeds framed count")


def _parse_layer(value: Any, path: str) -> LayerRecord:
    row = _mapping(value, path)
    _fields(
        row,
        {
            "layer_id",
            "status",
            "framed_strata",
            "unframed_quotient",
            "obstructed_branches",
            "failures",
        },
        path,
    )
    layer_id = _identifier(row["layer_id"], f"{path}.layer_id")
    status = _string(row["status"], f"{path}.status")
    if status not in ("complete", "failed"):
        raise ValueError(f"{path}.status: expected complete or failed")
    strata = tuple(
        _parse_stratum(item, f"{path}.framed_strata[{index}]")
        for index, item in enumerate(_array(row["framed_strata"], f"{path}.framed_strata"))
    )
    quotient = None
    if row["unframed_quotient"] is not None:
        quotient = _parse_unframed_quotient(row["unframed_quotient"], f"{path}.unframed_quotient")
    obstructions = tuple(
        _parse_obstruction(item, f"{path}.obstructed_branches[{index}]")
        for index, item in enumerate(_array(row["obstructed_branches"], f"{path}.obstructed_branches"))
    )
    failures = tuple(
        _parse_failure(item, f"{path}.failures[{index}]")
        for index, item in enumerate(_array(row["failures"], f"{path}.failures"))
    )
    if quotient is not None:
        stratum_ids = tuple(stratum["stratum_id"] for stratum in strata)
        if tuple(quotient["framed_stratum_ids"]) != stratum_ids:
            raise ValueError(f"{path}.unframed_quotient: framed stratum IDs/order mismatch")
    return LayerRecord(layer_id, status, strata, quotient, obstructions, failures)  # type: ignore[arg-type]


def _parse_candidate_evidence(value: Any, path: str) -> CandidateGeometryEvidence:
    row = _mapping(value, path)
    _fields(
        row,
        {
            "candidate_wyckoff_id",
            "geometry_evidence",
            "stabilizer_evidence",
            "inclusion_evidence",
            "family_geometry_digest",
            "literal_stabilizer_digest",
            "transported_inclusion_digest",
            "geometry_comparison_digest",
            "stabilizer_comparison_digest",
            "inclusion_comparison_digest",
            "geometry_match",
            "stabilizer_match",
            "inclusion_match",
            "rejection_codes",
        },
        path,
    )
    candidate_id, _, _ = _parse_wyckoff_id(
        row["candidate_wyckoff_id"], f"{path}.candidate_wyckoff_id"
    )
    evidence = tuple(
        _mapping(row[name], f"{path}.{name}")
        for name in ("geometry_evidence", "stabilizer_evidence", "inclusion_evidence")
    )
    digests = tuple(
        _digest(row[name], f"{path}.{name}")
        for name in (
            "family_geometry_digest",
            "literal_stabilizer_digest",
            "transported_inclusion_digest",
            "geometry_comparison_digest",
            "stabilizer_comparison_digest",
            "inclusion_comparison_digest",
        )
    )
    matches: list[bool] = []
    for name in ("geometry_match", "stabilizer_match", "inclusion_match"):
        match = row[name]
        if not isinstance(match, bool):
            raise TypeError(f"{path}.{name}: expected boolean")
        matches.append(match)
    codes = tuple(
        _string(code, f"{path}.rejection_codes[{index}]")
        for index, code in enumerate(_array(row["rejection_codes"], f"{path}.rejection_codes"))
    )
    if any(code not in ("geometry", "stabilizer", "inclusion") for code in codes):
        raise ValueError(f"{path}.rejection_codes: unknown rejection code")
    return CandidateGeometryEvidence(
        candidate_wyckoff_id=candidate_id,
        family_geometry_digest=digests[0],
        literal_stabilizer_digest=digests[1],
        transported_inclusion_digest=digests[2],
        geometry_evidence=evidence[0],  # type: ignore[arg-type]
        stabilizer_evidence=evidence[1],  # type: ignore[arg-type]
        inclusion_evidence=evidence[2],  # type: ignore[arg-type]
        geometry_comparison_digest=digests[3],
        stabilizer_comparison_digest=digests[4],
        inclusion_comparison_digest=digests[5],
        geometry_match=matches[0],
        stabilizer_match=matches[1],
        inclusion_match=matches[2],
        rejection_codes=codes,  # type: ignore[arg-type]
    )


def _parse_parameter_route(value: Any, path: str) -> InstanceParameterRoute:
    row = _mapping(value, path)
    _fields(
        row,
        {
            "instance_id",
            "requested_wyckoff_id",
            "exact_point",
            "outcome",
            "resolved_wyckoff_id",
            "point_geometry_digest",
            "point_stabilizer_digest",
            "point_inclusion_digest",
            "candidate_set_digest",
            "candidates",
        },
        path,
    )
    instance_id = _identifier(row["instance_id"], f"{path}.instance_id")
    requested, _, _ = _parse_wyckoff_id(
        row["requested_wyckoff_id"], f"{path}.requested_wyckoff_id"
    )
    point = tuple(
        _rational(item, f"{path}.exact_point[{index}]")
        for index, item in enumerate(_array(row["exact_point"], f"{path}.exact_point"))
    )
    outcome = _string(row["outcome"], f"{path}.outcome")
    resolved = None
    if row["resolved_wyckoff_id"] is not None:
        resolved, _, _ = _parse_wyckoff_id(
            row["resolved_wyckoff_id"], f"{path}.resolved_wyckoff_id"
        )
    digests = tuple(
        _digest(row[name], f"{path}.{name}")
        for name in (
            "point_geometry_digest",
            "point_stabilizer_digest",
            "point_inclusion_digest",
            "candidate_set_digest",
        )
    )
    candidates = tuple(
        _parse_candidate_evidence(candidate, f"{path}.candidates[{index}]")
        for index, candidate in enumerate(_array(row["candidates"], f"{path}.candidates"))
    )
    return InstanceParameterRoute(
        instance_id,
        requested,
        point,
        outcome,
        resolved,
        *digests,
        candidates,
    )  # type: ignore[arg-type]


def loads_classification_record(data: bytes) -> ClassificationRecord:
    value = _mapping(_strict_json_loads(data), "$classification")
    _fields(
        value,
        {
            "schema_version",
            "request_digest",
            "catalogue_manifest_digest",
            "point_routes",
            "routing_verification_digest",
            "layer",
        },
        "$classification",
    )
    _schema_version(value["schema_version"], "$classification.schema_version")
    request_digest = _digest(value["request_digest"], "$classification.request_digest")
    catalogue_digest = _digest(
        value["catalogue_manifest_digest"],
        "$classification.catalogue_manifest_digest",
    )
    routes = tuple(
        _parse_parameter_route(route, f"$classification.point_routes[{index}]")
        for index, route in enumerate(
            _array(value["point_routes"], "$classification.point_routes")
        )
    )
    return ClassificationRecord(
        request_digest,
        catalogue_digest,
        _parse_layer(value["layer"], "$classification.layer"),
        routes,
        (
            None
            if value["routing_verification_digest"] is None
            else _digest(
                value["routing_verification_digest"],
                "$classification.routing_verification_digest",
            )
        ),
    )


def _parse_parameter_routing_result(value: Mapping[str, Any]) -> ParameterRoutingResult:
    path = "$parameter_routing"
    _fields(
        value,
        {
            "schema_version",
            "status",
            "request_digest",
            "catalogue_manifest_digest",
            "space_group",
            "setting_id",
            "routes",
        },
        path,
    )
    _schema_version(value["schema_version"], f"{path}.schema_version")
    status = _string(value["status"], f"{path}.status")
    request_digest = _digest(value["request_digest"], f"{path}.request_digest")
    catalogue_digest = _digest(
        value["catalogue_manifest_digest"], f"{path}.catalogue_manifest_digest"
    )
    space_group = _integer(value["space_group"], f"{path}.space_group", minimum=1)
    if space_group > 230:
        raise ValueError(f"{path}.space_group: must be in 1..230")
    setting_id = _string(value["setting_id"], f"{path}.setting_id")
    routes = tuple(
        _parse_parameter_route(route, f"{path}.routes[{index}]")
        for index, route in enumerate(_array(value["routes"], f"{path}.routes"))
    )
    return ParameterRoutingResult(
        status,
        request_digest,
        catalogue_digest,
        space_group,
        setting_id,
        routes,
    )  # type: ignore[arg-type]


def loads_classification_query_result(data: bytes) -> ClassificationRecord | ParameterRoutingResult:
    value = _mapping(_strict_json_loads(data), "$query_result")
    if value.get("status") == "parameter_specialization":
        return _parse_parameter_routing_result(value)
    return loads_classification_record(data)


def _text_to_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("classification text parser requires str")
    if text.startswith("\ufeff"):
        raise ValueError("Unicode byte-order mark is forbidden")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("classification text contains an invalid surrogate") from error


def parse_classification_request_text(text: str) -> ClassificationRequest:
    return loads_classification_request(_text_to_bytes(text))


def parse_classifier_certificate_text(text: str) -> FrozenJSONObject:
    return loads_classifier_certificate(_text_to_bytes(text))


def parse_classification_record_text(text: str) -> ClassificationRecord:
    return loads_classification_record(_text_to_bytes(text))


def parse_classification_query_result_text(
    text: str,
) -> ClassificationRecord | ParameterRoutingResult:
    return loads_classification_query_result(_text_to_bytes(text))


def _orbit_mapping(orbit: OrbitInstance) -> dict[str, Any]:
    result: dict[str, Any] = {
        "instance_id": orbit.instance_id,
        "parameter_mode": orbit.parameter_mode,
        "parameter_values": [_rational_text(value) for value in orbit.parameter_values],
        "wyckoff_id": orbit.wyckoff_id,
    }
    if orbit.species is not None:
        result["species"] = orbit.species
    return result


def _failure_mapping(failure: StructuredFailure) -> dict[str, Any]:
    return {
        "code": failure.code,
        "context": _thaw(failure.context),
        "message": failure.message,
        "stage": failure.stage,
    }


def _obstruction_mapping(obstruction: ObstructedBranch) -> dict[str, Any]:
    return {
        "skeleton_ids": list(obstruction.skeleton_ids),
        "stratum_id": obstruction.stratum_id,
        "witness": _thaw(obstruction.witness),
    }


def _layer_mapping(layer: LayerRecord) -> dict[str, Any]:
    return {
        "failures": [_failure_mapping(failure) for failure in layer.failures],
        "framed_strata": [_thaw(stratum) for stratum in layer.framed_strata],
        "layer_id": layer.layer_id,
        "obstructed_branches": [
            _obstruction_mapping(obstruction) for obstruction in layer.obstructed_branches
        ],
        "status": layer.status,
        "unframed_quotient": (
            None if layer.unframed_quotient is None else _thaw(layer.unframed_quotient)
        ),
    }


def _candidate_evidence_mapping(candidate: CandidateGeometryEvidence) -> dict[str, Any]:
    return {
        "candidate_wyckoff_id": candidate.candidate_wyckoff_id,
        "family_geometry_digest": candidate.family_geometry_digest,
        "geometry_evidence": _thaw(candidate.geometry_evidence),
        "geometry_comparison_digest": candidate.geometry_comparison_digest,
        "geometry_match": candidate.geometry_match,
        "inclusion_evidence": _thaw(candidate.inclusion_evidence),
        "inclusion_comparison_digest": candidate.inclusion_comparison_digest,
        "inclusion_match": candidate.inclusion_match,
        "literal_stabilizer_digest": candidate.literal_stabilizer_digest,
        "rejection_codes": list(candidate.rejection_codes),
        "stabilizer_evidence": _thaw(candidate.stabilizer_evidence),
        "stabilizer_comparison_digest": candidate.stabilizer_comparison_digest,
        "stabilizer_match": candidate.stabilizer_match,
        "transported_inclusion_digest": candidate.transported_inclusion_digest,
    }


def _parameter_route_mapping(route: InstanceParameterRoute) -> dict[str, Any]:
    return {
        "candidate_set_digest": route.candidate_set_digest,
        "candidates": [_candidate_evidence_mapping(candidate) for candidate in route.candidates],
        "exact_point": [_rational_text(item) for item in route.exact_point],
        "instance_id": route.instance_id,
        "outcome": route.outcome,
        "point_geometry_digest": route.point_geometry_digest,
        "point_inclusion_digest": route.point_inclusion_digest,
        "point_stabilizer_digest": route.point_stabilizer_digest,
        "requested_wyckoff_id": route.requested_wyckoff_id,
        "resolved_wyckoff_id": route.resolved_wyckoff_id,
    }


def _protocol_mapping(value: object) -> Any:
    if isinstance(value, ClassificationRequest):
        return {
            "igg": value.igg,
            "orbits": [_orbit_mapping(orbit) for orbit in value.orbits],
            "schema_version": value.schema_version,
            "setting_id": value.setting_id,
            "space_group": value.space_group,
            "time_reversal": value.time_reversal,
        }
    if isinstance(value, ClassificationRecord):
        return {
            "catalogue_manifest_digest": value.catalogue_manifest_digest,
            "layer": _layer_mapping(value.layer),
            "point_routes": [_parameter_route_mapping(route) for route in value.point_routes],
            "request_digest": value.request_digest,
            "routing_verification_digest": value.routing_verification_digest,
            "schema_version": SCHEMA_VERSION,
        }
    if isinstance(value, ParameterRoutingResult):
        return {
            "catalogue_manifest_digest": value.catalogue_manifest_digest,
            "request_digest": value.request_digest,
            "routes": [_parameter_route_mapping(route) for route in value.routes],
            "schema_version": SCHEMA_VERSION,
            "setting_id": value.setting_id,
            "space_group": value.space_group,
            "status": value.status,
        }
    if isinstance(value, InstanceParameterRoute):
        return _parameter_route_mapping(value)
    if isinstance(value, CandidateGeometryEvidence):
        return _candidate_evidence_mapping(value)
    if isinstance(value, LayerRecord):
        return _layer_mapping(value)
    if isinstance(value, StructuredFailure):
        return _failure_mapping(value)
    if isinstance(value, ObstructedBranch):
        return _obstruction_mapping(value)
    if isinstance(value, Mapping):
        return _thaw(value)
    if isinstance(value, (FrozenJSONObject, FrozenJSONArray)):
        return _thaw(value)
    if dataclasses.is_dataclass(value):
        raise TypeError(f"unsupported protocol dataclass {type(value).__name__}")
    return value


def canonical_classification_json(value: object) -> bytes:
    """Serialize a protocol record as deterministic canonical UTF-8 JSON."""

    mapping = _protocol_mapping(value)
    _validate_json_tree(mapping)
    return json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CandidateGeometryEvidence",
    "ClassificationRecord",
    "ClassificationRequest",
    "InstanceParameterRoute",
    "LayerRecord",
    "ObstructedBranch",
    "OrbitInstance",
    "ParameterRoutingResult",
    "StructuredFailure",
    "FrozenJSONArray",
    "FrozenJSONObject",
    "FrozenJSONValue",
    "canonical_classification_json",
    "loads_classifier_certificate",
    "loads_classification_query_result",
    "loads_classification_record",
    "loads_classification_request",
    "parse_classification_record_text",
    "parse_classification_query_result_text",
    "parse_classification_request_text",
    "parse_classifier_certificate_text",
]
