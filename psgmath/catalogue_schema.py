"""Strict, dependency-free protocol validation for catalogue format v1.

This module validates JSON syntax, record shape, and field domains.  Display
rows additionally replay their exact affine, lattice, Smith, and quotient
witnesses.  Catalogue geometry normalization, digest semantics, group closure,
and orbit equivalence belong to later catalogue stages.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1
MAX_JSON_NESTING = 64
MAX_JSON_NODES = 100_000
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_RATIONAL_RE = re.compile(r"q\((-?(?:0|[1-9][0-9]*)),([1-9][0-9]*)\)\Z")
_PARAMETER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_WYCKOFF_ID_RE = re.compile(
    r"sg([1-9][0-9]{0,2}):setting-([A-Za-z0-9._-]+):(sha256:[0-9a-f]{64})\Z"
)
_RAW_ORDINAL_KEYS = {
    "candidateindex",
    "candidateordinal",
    "crystindex",
    "crystlistindex",
    "crystordinal",
    "listindex",
    "listordinal",
    "rawindex",
    "rawordinal",
    "wyckoffindex",
    "wyckoffordinal",
}
_CANONICAL_FILE_PATHS = {
    "geometry": "atlas/catalogue/v1/wyckoff.ndjson",
    "display": "atlas/catalogue/v1/display-crosswalk.ndjson",
    "coverage": "atlas/catalogue/v1/coverage.json",
    "index": "atlas/catalogue/v1/index.json",
    "provenance": "atlas/catalogue/v1/provenance.json",
}
_CATALOGUE_LOCK_PATH = "environments/catalogue-gap.lock.json"


JsonScalar = None | bool | int | str
FrozenJson = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def _freeze(value: Any) -> FrozenJson:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CatalogueRecord:
    schema_version: int
    record_type: str
    space_group: Mapping[str, FrozenJson]
    wyckoff_id: str
    embedding_digest: str
    action_provenance_digest: str
    orbit: Mapping[str, FrozenJson]
    stabilizer: Mapping[str, FrozenJson]
    space_group_action: Mapping[str, FrozenJson]
    provenance: Mapping[str, FrozenJson]
    presentation_conjugation: Mapping[str, FrozenJson] | None = None

    def __post_init__(self) -> None:
        for name in ("space_group", "orbit", "stabilizer", "space_group_action", "provenance"):
            _freeze_mapping_field(self, name)
        if self.presentation_conjugation is not None:
            _freeze_mapping_field(self, "presentation_conjugation")

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "space_group": _thaw(self.space_group),
            "wyckoff_id": self.wyckoff_id,
            "embedding_digest": self.embedding_digest,
            "action_provenance_digest": self.action_provenance_digest,
            "orbit": _thaw(self.orbit),
            "stabilizer": _thaw(self.stabilizer),
            "space_group_action": _thaw(self.space_group_action),
            "provenance": _thaw(self.provenance),
        }
        if self.presentation_conjugation is not None:
            result["presentation_conjugation"] = _thaw(self.presentation_conjugation)
        return result


@dataclass(frozen=True, slots=True)
class DisplayRecord:
    schema_version: int
    wyckoff_id: str
    hall_symbol: str
    hermann_mauguin_symbol: str
    origin_choice: str
    wyckoff_letter: str
    conventional_multiplicity: int
    site_symmetry_symbol: str
    coordinate_crosswalk: Mapping[str, FrozenJson]
    lattice_inclusion: Mapping[str, FrozenJson]
    independent_verification: Mapping[str, FrozenJson]

    def __post_init__(self) -> None:
        for name in (
            "coordinate_crosswalk",
            "lattice_inclusion",
            "independent_verification",
        ):
            _freeze_mapping_field(self, name)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "wyckoff_id": self.wyckoff_id,
            "hall_symbol": self.hall_symbol,
            "hermann_mauguin_symbol": self.hermann_mauguin_symbol,
            "origin_choice": self.origin_choice,
            "wyckoff_letter": self.wyckoff_letter,
            "conventional_multiplicity": self.conventional_multiplicity,
            "site_symmetry_symbol": self.site_symmetry_symbol,
            "coordinate_crosswalk": _thaw(self.coordinate_crosswalk),
            "lattice_inclusion": _thaw(self.lattice_inclusion),
            "independent_verification": _thaw(self.independent_verification),
        }


@dataclass(frozen=True, slots=True)
class CatalogueManifest:
    schema_version: int
    record_type: str
    catalogue_schema_version: int
    display_schema_version: int
    normalization_version: int
    scope: Mapping[str, FrozenJson]
    generator: Mapping[str, FrozenJson]
    environment: Mapping[str, FrozenJson]
    independent_count_reference: Mapping[str, FrozenJson]
    counts: Mapping[str, FrozenJson]
    per_group: tuple[Mapping[str, FrozenJson], ...]
    files: tuple[Mapping[str, FrozenJson], ...]
    timing: Mapping[str, FrozenJson]
    status: Mapping[str, FrozenJson]
    failures: tuple[Mapping[str, FrozenJson], ...]

    def __post_init__(self) -> None:
        for name in (
            "scope",
            "generator",
            "environment",
            "independent_count_reference",
            "counts",
            "timing",
            "status",
        ):
            _freeze_mapping_field(self, name)
        for name in ("per_group", "files", "failures"):
            _freeze_mapping_sequence_field(self, name)

    def to_mapping(self) -> dict[str, Any]:
        return {
            field.name: _thaw(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


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
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_integer(token: str) -> int:
    if token == "-0":
        raise ValueError("negative zero is not canonical JSON")
    return int(token)


def _parse_json_float(token: str) -> float:
    try:
        if token.startswith("-") and Decimal(token).is_zero():
            raise ValueError("negative zero is not canonical JSON")
    except InvalidOperation as error:  # pragma: no cover - json supplies numeric lexemes
        raise ValueError("invalid floating-point token") from error
    raise ValueError("floating-point JSON tokens are forbidden")


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON token is forbidden: {token}")


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.casefold())


def _validate_json_tree(value: Any, path: str = "$") -> None:
    """Iteratively bound and validate one JSON-like object graph.

    Shared acyclic subtrees are permitted.  An object already on the active
    ancestor path is a cycle and fails before any recursive converter runs.
    """

    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, False)]
    active_containers: set[int] = set()
    visited_nodes = 0
    while stack:
        node, node_path, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(node))
            continue
        visited_nodes += 1
        if visited_nodes > MAX_JSON_NODES:
            raise ValueError(f"{path}: JSON node limit exceeds {MAX_JSON_NODES}")
        if depth > MAX_JSON_NESTING:
            raise ValueError(f"{node_path}: JSON nesting exceeds {MAX_JSON_NESTING}")
        if isinstance(node, Mapping):
            identity = id(node)
            if identity in active_containers:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active_containers.add(identity)
            stack.append((node, node_path, depth, True))
            if visited_nodes + len(node) > MAX_JSON_NODES:
                raise ValueError(f"{path}: JSON node limit exceeds {MAX_JSON_NODES}")
            children: list[tuple[Any, str, int, bool]] = []
            for key, item in node.items():
                if not isinstance(key, str):
                    raise TypeError(f"{node_path}: JSON object keys must be strings")
                if _normalized_key(key) in _RAW_ORDINAL_KEYS:
                    raise ValueError(f"{node_path}.{key}: raw ordinal fields are forbidden")
                children.append((item, f"{node_path}.{key}", depth + 1, False))
            stack.extend(reversed(children))
        elif isinstance(node, (list, tuple)):
            identity = id(node)
            if identity in active_containers:
                raise ValueError(f"{node_path}: JSON object graph contains a cycle")
            active_containers.add(identity)
            stack.append((node, node_path, depth, True))
            if visited_nodes + len(node) > MAX_JSON_NODES:
                raise ValueError(f"{path}: JSON node limit exceeds {MAX_JSON_NODES}")
            for index in range(len(node) - 1, -1, -1):
                stack.append((node[index], f"{node_path}[{index}]", depth + 1, False))
        elif isinstance(node, float):
            raise TypeError(f"{node_path}: float values are forbidden")
        elif node is None or isinstance(node, (bool, int, str)):
            continue
        else:
            raise TypeError(f"{node_path}: unsupported JSON value {type(node).__name__}")


def _freeze_mapping_field(record: object, name: str) -> None:
    value = getattr(record, name)
    path = f"${type(record).__name__}.{name}"
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _validate_json_tree(value, path)
    object.__setattr__(record, name, _freeze(value))


def _freeze_mapping_sequence_field(record: object, name: str) -> None:
    value = getattr(record, name)
    path = f"${type(record).__name__}.{name}"
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path}: expected array")
    _validate_json_tree(value, path)
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{path}[{index}]: expected object")
    object.__setattr__(record, name, _freeze(value))


def _validate_protocol_record_fields(
    record: CatalogueRecord | DisplayRecord | CatalogueManifest,
) -> None:
    shallow_fields = {
        field.name: getattr(record, field.name)
        for field in dataclasses.fields(record)
    }
    _validate_json_tree(shallow_fields, f"${type(record).__name__}")


def strict_json_loads(data: bytes | str) -> Any:
    """Decode one strict JSON value used by catalogue protocols."""

    if isinstance(data, bytes):
        if data.startswith(b"\xef\xbb\xbf"):
            raise ValueError("UTF-8 byte-order mark is forbidden")
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("input is not valid UTF-8") from error
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("strict_json_loads expects bytes or str")
    if text.startswith("\ufeff"):
        raise ValueError("Unicode byte-order mark is forbidden")
    _check_json_nesting(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_int=_parse_json_integer,
            parse_float=_parse_json_float,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error
    _validate_json_tree(value)
    return value


def _plain_json(value: Any, path: str = "$") -> Any:
    if isinstance(value, (CatalogueRecord, DisplayRecord, CatalogueManifest)):
        return _plain_json(value.to_mapping(), path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: JSON object keys must be strings")
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float):
        raise TypeError(f"{path}: float values are forbidden")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"{path}: unsupported JSON value {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return deterministic canonical UTF-8 JSON without a trailing newline."""

    if isinstance(value, (CatalogueRecord, DisplayRecord, CatalogueManifest)):
        _validate_protocol_record_fields(value)
        value = value.to_mapping()
    _validate_json_tree(value)
    plain = _plain_json(value)
    return json.dumps(
        plain,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    return value


def _list(value: Any, path: str) -> list[Any] | tuple[Any, ...]:
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


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path}: integer must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path}: integer must be <= {maximum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path}: expected boolean")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: expected string")
    if nonempty and (not value or value.strip() != value):
        raise ValueError(f"{path}: expected a nonempty trimmed string")
    return value


def _schema_version(value: Any, path: str) -> int:
    version = _integer(value, path)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {version}")
    return version


def _normalization_version(value: Any, path: str) -> int:
    version = _integer(value, path)
    if version != NORMALIZATION_VERSION:
        raise ValueError(f"{path}: unsupported normalization_version {version}")
    return version


def _digest(value: Any, path: str) -> str:
    text = _string(value, path)
    if _DIGEST_RE.fullmatch(text) is None:
        raise ValueError(f"{path}: expected sha256:<64 lowercase hex digits>")
    return text


def _wyckoff_id(value: Any, path: str) -> tuple[str, int, str]:
    text = _string(value, path)
    match = _WYCKOFF_ID_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"{path}: invalid v1 identifier shape")
    international_number = int(match.group(1))
    if not 1 <= international_number <= 230:
        raise ValueError(f"{path}: identifier group must be in 1..230")
    return text, international_number, match.group(2)


def _rational(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: expected exact rational q(n,d), not a JSON number")
    match = _RATIONAL_RE.fullmatch(value)
    if match is None or match.group(1) == "-0":
        raise ValueError(f"{path}: invalid exact rational spelling")
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if math.gcd(abs(numerator), denominator) != 1:
        raise ValueError(f"{path}: exact rational must be reduced")
    if numerator == 0 and denominator != 1:
        raise ValueError(f"{path}: exact rational zero must be q(0,1)")
    return value


ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[tuple[Fraction, ...], ...]


def _exact_fraction(value: Any, path: str) -> Fraction:
    text = _rational(value, path)
    match = _RATIONAL_RE.fullmatch(text)
    if match is None:  # pragma: no cover - retained behind _rational's contract
        raise AssertionError("validated rational did not replay")
    return Fraction(int(match.group(1)), int(match.group(2)))


def _exact_vector(value: Any, dimension: int, path: str) -> ExactVector:
    items = _list(value, path)
    if len(items) != dimension:
        raise ValueError(f"{path}: vector dimension must be {dimension}")
    return tuple(_exact_fraction(item, f"{path}[{index}]") for index, item in enumerate(items))


def _exact_matrix(value: Any, rows: int, columns: int, path: str) -> ExactMatrix:
    matrix = _list(value, path)
    if len(matrix) != rows:
        raise ValueError(f"{path}: matrix row dimension must be {rows}")
    result: list[tuple[Fraction, ...]] = []
    for row_index, row in enumerate(matrix):
        row_items = _list(row, f"{path}[{row_index}]")
        if len(row_items) != columns:
            raise ValueError(f"{path}: matrix column dimension must be {columns}")
        result.append(
            tuple(
                _exact_fraction(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row_items)
            )
        )
    return tuple(result)


def _exact_integer_matrix(value: Any, dimension: int, path: str) -> ExactMatrix:
    matrix = _list(value, path)
    if len(matrix) != dimension:
        raise ValueError(f"{path}: matrix row dimension must be {dimension}")
    result: list[tuple[Fraction, ...]] = []
    for row_index, row in enumerate(matrix):
        row_items = _list(row, f"{path}[{row_index}]")
        if len(row_items) != dimension:
            raise ValueError(f"{path}: matrix column dimension must be {dimension}")
        result.append(
            tuple(
                Fraction(_any_integer(item, f"{path}[{row_index}][{column_index}]"))
                for column_index, item in enumerate(row_items)
            )
        )
    return tuple(result)


def _exact_identity(dimension: int) -> ExactMatrix:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )


def _exact_matmul(left: ExactMatrix, right: ExactMatrix) -> ExactMatrix:
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                Fraction(),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _exact_matvec(matrix: ExactMatrix, vector: ExactVector) -> ExactVector:
    return tuple(
        sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction())
        for row in matrix
    )


def _exact_determinant(matrix: ExactMatrix) -> Fraction:
    dimension = len(matrix)
    work = [list(row) for row in matrix]
    determinant = Fraction(1)
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if work[row][column]), None)
        if pivot is None:
            return Fraction()
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, dimension):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for inner in range(column, dimension):
                work[row][inner] -= factor * work[column][inner]
    return determinant


def _exact_inverse(matrix: ExactMatrix, path: str) -> ExactMatrix:
    dimension = len(matrix)
    work = [
        list(row) + list(identity_row)
        for row, identity_row in zip(matrix, _exact_identity(dimension), strict=True)
    ]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if work[row][column]), None)
        if pivot is None:
            raise ValueError(f"{path}: lattice basis must be nonsingular")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(dimension):
            if row == column or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                left - factor * right
                for left, right in zip(work[row], work[column], strict=True)
            ]
    return tuple(tuple(row[dimension:]) for row in work)


def _exact_is_integral(vector: ExactVector) -> bool:
    return all(value.denominator == 1 for value in vector)


def _vector(value: Any, dimension: int, path: str, scalar=_rational) -> None:
    items = _list(value, path)
    if len(items) != dimension:
        raise ValueError(f"{path}: vector dimension must be {dimension}")
    for index, item in enumerate(items):
        scalar(item, f"{path}[{index}]")


def _matrix(value: Any, rows: int, columns: int, path: str, scalar=_rational) -> None:
    matrix = _list(value, path)
    if len(matrix) != rows:
        raise ValueError(f"{path}: matrix row dimension must be {rows}")
    for row_index, row in enumerate(matrix):
        row_items = _list(row, f"{path}[{row_index}]")
        if len(row_items) != columns:
            raise ValueError(f"{path}: matrix column dimension must be {columns}")
        for column_index, item in enumerate(row_items):
            scalar(item, f"{path}[{row_index}][{column_index}]")


def _any_integer(value: Any, path: str) -> int:
    return _integer(value, path)


def _positive_integer(value: Any, path: str) -> int:
    return _integer(value, path, minimum=1)


def _affine_map(value: Any, dimension: int, path: str) -> None:
    mapping = _mapping(value, path)
    _fields(mapping, {"matrix", "translation"}, path)
    _matrix(mapping["matrix"], dimension, dimension, f"{path}.matrix")
    _vector(mapping["translation"], dimension, f"{path}.translation")


def _parameter_names(value: Any, dimension: int, path: str) -> tuple[str, ...]:
    names = _list(value, path)
    if len(names) != dimension:
        raise ValueError(f"{path}: shared parameter name count must be {dimension}")
    result = tuple(_string(name, f"{path}[{index}]") for index, name in enumerate(names))
    if any(_PARAMETER_RE.fullmatch(name) is None for name in result):
        raise ValueError(f"{path}: invalid shared parameter name")
    if len(set(result)) != len(result):
        raise ValueError(f"{path}: shared parameter names must be unique")
    return result


def parse_catalogue_record(mapping: Mapping[str, Any]) -> CatalogueRecord:
    """Validate and freeze one v1 canonical geometry record."""

    value = _mapping(mapping, "$catalogue")
    _validate_json_tree(value)
    required = {
        "schema_version",
        "record_type",
        "space_group",
        "wyckoff_id",
        "embedding_digest",
        "action_provenance_digest",
        "orbit",
        "stabilizer",
        "space_group_action",
        "provenance",
    }
    _fields(value, required, "$catalogue", {"presentation_conjugation"})
    schema_version = _schema_version(value["schema_version"], "$catalogue.schema_version")
    record_type = _string(value["record_type"], "$catalogue.record_type")
    if record_type != "wyckoff-position":
        raise ValueError("$catalogue.record_type: expected wyckoff-position")

    space_group = _mapping(value["space_group"], "$catalogue.space_group")
    _fields(space_group, {"international_number", "setting", "source"}, "$catalogue.space_group")
    international_number = _integer(
        space_group["international_number"],
        "$catalogue.space_group.international_number",
        minimum=1,
        maximum=230,
    )
    setting = _string(space_group["setting"], "$catalogue.space_group.setting")
    if re.fullmatch(r"[A-Za-z0-9._-]+", setting) is None:
        raise ValueError("$catalogue.space_group.setting: invalid setting spelling")
    source = _mapping(space_group["source"], "$catalogue.space_group.source")
    _fields(source, {"gap", "cryst"}, "$catalogue.space_group.source")
    _string(source["gap"], "$catalogue.space_group.source.gap")
    _string(source["cryst"], "$catalogue.space_group.source.cryst")

    wyckoff_id, identifier_group, identifier_setting = _wyckoff_id(
        value["wyckoff_id"], "$catalogue.wyckoff_id"
    )
    if identifier_group != international_number:
        raise ValueError("$catalogue.wyckoff_id: international_number does not match record")
    if identifier_setting != setting:
        raise ValueError("$catalogue.wyckoff_id: setting does not match record")
    embedding = _digest(value["embedding_digest"], "$catalogue.embedding_digest")
    action_digest = _digest(
        value["action_provenance_digest"], "$catalogue.action_provenance_digest"
    )

    orbit = _mapping(value["orbit"], "$catalogue.orbit")
    _fields(
        orbit,
        {
            "primitive_orbit_size",
            "parameter_dimension",
            "parameter_names",
            "branches",
            "reference_branch_digest",
            "branch_transports",
        },
        "$catalogue.orbit",
    )
    primitive_size = _integer(
        orbit["primitive_orbit_size"], "$catalogue.orbit.primitive_orbit_size", minimum=1
    )
    parameter_dimension = _integer(
        orbit["parameter_dimension"],
        "$catalogue.orbit.parameter_dimension",
        minimum=0,
        maximum=3,
    )
    names = _parameter_names(
        orbit["parameter_names"], parameter_dimension, "$catalogue.orbit.parameter_names"
    )
    branches = _list(orbit["branches"], "$catalogue.orbit.branches")
    if len(branches) != primitive_size:
        raise ValueError("$catalogue.orbit.branches: length must equal primitive_orbit_size")
    branch_digests: list[str] = []
    for index, item in enumerate(branches):
        path = f"$catalogue.orbit.branches[{index}]"
        branch = _mapping(item, path)
        _fields(
            branch,
            {
                "branch_digest",
                "parameter_dimension",
                "parameter_names",
                "offset",
                "basis",
            },
            path,
        )
        branch_digests.append(_digest(branch["branch_digest"], f"{path}.branch_digest"))
        branch_dimension = _integer(
            branch["parameter_dimension"], f"{path}.parameter_dimension", minimum=0, maximum=3
        )
        branch_names = _parameter_names(
            branch["parameter_names"], branch_dimension, f"{path}.parameter_names"
        )
        if branch_dimension != parameter_dimension or branch_names != names:
            raise ValueError(f"{path}: branch must use the shared parameter dimension and names")
        _vector(branch["offset"], 3, f"{path}.offset")
        _matrix(branch["basis"], 3, parameter_dimension, f"{path}.basis")

    reference_digest = _digest(
        orbit["reference_branch_digest"], "$catalogue.orbit.reference_branch_digest"
    )
    if reference_digest not in branch_digests:
        raise ValueError("$catalogue.orbit.reference_branch_digest: does not name a branch")

    transports = _list(orbit["branch_transports"], "$catalogue.orbit.branch_transports")
    if len(transports) != len(branches):
        raise ValueError("$catalogue.orbit.branch_transports: length must match branches")
    transport_targets: list[str] = []
    for index, item in enumerate(transports):
        path = f"$catalogue.orbit.branch_transports[{index}]"
        transport = _mapping(item, path)
        _fields(
            transport,
            {
                "target_branch_digest",
                "parameter_dimension",
                "ambient_element",
                "parameter_action",
            },
            path,
        )
        transport_targets.append(
            _digest(transport["target_branch_digest"], f"{path}.target_branch_digest")
        )
        transport_dimension = _integer(
            transport["parameter_dimension"],
            f"{path}.parameter_dimension",
            minimum=0,
            maximum=3,
        )
        if transport_dimension != parameter_dimension:
            raise ValueError(f"{path}: transport must use the shared parameter dimension")
        _affine_map(transport["ambient_element"], 3, f"{path}.ambient_element")
        parameter_action = _mapping(transport["parameter_action"], f"{path}.parameter_action")
        _fields(parameter_action, {"matrix", "translation"}, f"{path}.parameter_action")
        _matrix(
            parameter_action["matrix"],
            parameter_dimension,
            parameter_dimension,
            f"{path}.parameter_action.matrix",
        )
        _vector(
            parameter_action["translation"],
            parameter_dimension,
            f"{path}.parameter_action.translation",
        )
    if transport_targets != branch_digests:
        raise ValueError("$catalogue.orbit.branch_transports: targets must match branches in order")

    stabilizer = _mapping(value["stabilizer"], "$catalogue.stabilizer")
    _fields(stabilizer, {"reference_branch_digest", "order", "embedded_elements"}, "$catalogue.stabilizer")
    stabilizer_reference = _digest(
        stabilizer["reference_branch_digest"], "$catalogue.stabilizer.reference_branch_digest"
    )
    if stabilizer_reference != reference_digest:
        raise ValueError("orbit and stabilizer reference_branch_digest values must be identical")
    stabilizer_order = _integer(stabilizer["order"], "$catalogue.stabilizer.order", minimum=1)
    elements = _list(stabilizer["embedded_elements"], "$catalogue.stabilizer.embedded_elements")
    if len(elements) != stabilizer_order:
        raise ValueError("$catalogue.stabilizer.embedded_elements: length must equal order")
    for index, element in enumerate(elements):
        _affine_map(element, 3, f"$catalogue.stabilizer.embedded_elements[{index}]")

    action = _mapping(value["space_group_action"], "$catalogue.space_group_action")
    _fields(action, {"translation_basis", "source_generators"}, "$catalogue.space_group_action")
    _matrix(action["translation_basis"], 3, 3, "$catalogue.space_group_action.translation_basis")
    generators = _list(action["source_generators"], "$catalogue.space_group_action.source_generators")
    if not generators:
        raise ValueError("$catalogue.space_group_action.source_generators: must not be empty")
    for index, generator in enumerate(generators):
        _affine_map(generator, 3, f"$catalogue.space_group_action.source_generators[{index}]")

    provenance = _mapping(value["provenance"], "$catalogue.provenance")
    _fields(provenance, {"generator_input_digest", "normalization_version"}, "$catalogue.provenance")
    _digest(provenance["generator_input_digest"], "$catalogue.provenance.generator_input_digest")
    _normalization_version(
        provenance["normalization_version"], "$catalogue.provenance.normalization_version"
    )

    presentation = value.get("presentation_conjugation")
    if presentation is not None:
        presentation_mapping = _mapping(presentation, "$catalogue.presentation_conjugation")
        _fields(presentation_mapping, {"forward", "inverse"}, "$catalogue.presentation_conjugation")
        _affine_map(
            presentation_mapping["forward"], 3, "$catalogue.presentation_conjugation.forward"
        )
        _affine_map(
            presentation_mapping["inverse"], 3, "$catalogue.presentation_conjugation.inverse"
        )

    return CatalogueRecord(
        schema_version=schema_version,
        record_type=record_type,
        space_group=space_group,  # type: ignore[arg-type]
        wyckoff_id=wyckoff_id,
        embedding_digest=embedding,
        action_provenance_digest=action_digest,
        orbit=orbit,  # type: ignore[arg-type]
        stabilizer=stabilizer,  # type: ignore[arg-type]
        space_group_action=action,  # type: ignore[arg-type]
        provenance=provenance,  # type: ignore[arg-type]
        presentation_conjugation=(
            None if presentation is None else presentation  # type: ignore[arg-type]
        ),
    )


def parse_display_record(mapping: Mapping[str, Any]) -> DisplayRecord:
    """Validate and freeze one v1 conventional-display crosswalk row."""

    value = _mapping(mapping, "$display")
    _validate_json_tree(value)
    required = {
        "schema_version",
        "wyckoff_id",
        "hall_symbol",
        "hermann_mauguin_symbol",
        "origin_choice",
        "wyckoff_letter",
        "conventional_multiplicity",
        "site_symmetry_symbol",
        "coordinate_crosswalk",
        "lattice_inclusion",
        "independent_verification",
    }
    _fields(value, required, "$display")
    schema_version = _schema_version(value["schema_version"], "$display.schema_version")
    wyckoff_id, _, _ = _wyckoff_id(value["wyckoff_id"], "$display.wyckoff_id")
    hall = _string(value["hall_symbol"], "$display.hall_symbol")
    hm = _string(value["hermann_mauguin_symbol"], "$display.hermann_mauguin_symbol")
    origin = _string(value["origin_choice"], "$display.origin_choice")
    letter = _string(value["wyckoff_letter"], "$display.wyckoff_letter")
    if re.fullmatch(r"[A-Za-z]", letter) is None:
        raise ValueError("$display.wyckoff_letter: expected one single ASCII Wyckoff letter")
    multiplicity = _integer(
        value["conventional_multiplicity"], "$display.conventional_multiplicity", minimum=1
    )
    site_symmetry = _string(value["site_symmetry_symbol"], "$display.site_symmetry_symbol")

    crosswalk = _mapping(value["coordinate_crosswalk"], "$display.coordinate_crosswalk")
    _fields(
        crosswalk,
        {"dimension", "direction", "basis_matrix_P", "origin_shift_o", "inverse"},
        "$display.coordinate_crosswalk",
    )
    dimension = _integer(
        crosswalk["dimension"], "$display.coordinate_crosswalk.dimension", minimum=1, maximum=3
    )
    if dimension != 3:
        raise ValueError("$display.coordinate_crosswalk.dimension: catalogue dimension must be 3")
    direction = _string(crosswalk["direction"], "$display.coordinate_crosswalk.direction")
    if direction != "x_conventional=P*x_primitive+o":
        raise ValueError("$display.coordinate_crosswalk.direction: unsupported direction")
    basis_matrix = _exact_matrix(
        crosswalk["basis_matrix_P"],
        dimension,
        dimension,
        "$display.coordinate_crosswalk.basis_matrix_P",
    )
    origin_shift = _exact_vector(
        crosswalk["origin_shift_o"], dimension, "$display.coordinate_crosswalk.origin_shift_o"
    )
    inverse = _mapping(crosswalk["inverse"], "$display.coordinate_crosswalk.inverse")
    _fields(
        inverse,
        {"basis_matrix_P", "origin_shift_o"},
        "$display.coordinate_crosswalk.inverse",
    )
    inverse_basis_matrix = _exact_matrix(
        inverse["basis_matrix_P"],
        dimension,
        dimension,
        "$display.coordinate_crosswalk.inverse.basis_matrix_P",
    )
    inverse_origin_shift = _exact_vector(
        inverse["origin_shift_o"], dimension, "$display.coordinate_crosswalk.inverse.origin_shift_o"
    )
    identity = _exact_identity(dimension)
    if (
        _exact_matmul(basis_matrix, inverse_basis_matrix) != identity
        or _exact_matmul(inverse_basis_matrix, basis_matrix) != identity
    ):
        raise ValueError(
            "$display.coordinate_crosswalk.inverse: basis matrices are not mutual inverses"
        )
    zero = (Fraction(),) * dimension
    if tuple(
        left + right
        for left, right in zip(
            _exact_matvec(basis_matrix, inverse_origin_shift),
            origin_shift,
            strict=True,
        )
    ) != zero or tuple(
        left + right
        for left, right in zip(
            _exact_matvec(inverse_basis_matrix, origin_shift),
            inverse_origin_shift,
            strict=True,
        )
    ) != zero:
        raise ValueError(
            "$display.coordinate_crosswalk.inverse: affine origin shifts do not compose to zero"
        )

    lattice = _mapping(value["lattice_inclusion"], "$display.lattice_inclusion")
    _fields(
        lattice,
        {
            "dimension",
            "relation",
            "L_conv_basis",
            "L_prim_basis",
            "index",
            "smith_witness",
            "coset_representatives",
        },
        "$display.lattice_inclusion",
    )
    lattice_dimension = _integer(
        lattice["dimension"], "$display.lattice_inclusion.dimension", minimum=1, maximum=3
    )
    if lattice_dimension != dimension:
        raise ValueError("$display.lattice_inclusion.dimension: must match crosswalk dimension")
    relation = _string(lattice["relation"], "$display.lattice_inclusion.relation")
    if relation != "L_conv_subset_L_prim":
        raise ValueError("$display.lattice_inclusion.relation: unsupported lattice relation")
    conventional_basis = _exact_matrix(
        lattice["L_conv_basis"], dimension, dimension, "$display.lattice_inclusion.L_conv_basis"
    )
    primitive_basis = _exact_matrix(
        lattice["L_prim_basis"], dimension, dimension, "$display.lattice_inclusion.L_prim_basis"
    )
    lattice_index = _integer(lattice["index"], "$display.lattice_inclusion.index", minimum=1)
    cosets = _list(
        lattice["coset_representatives"],
        "$display.lattice_inclusion.coset_representatives",
    )
    if len(cosets) != lattice_index:
        raise ValueError(
            "$display.lattice_inclusion.coset_representatives: length must equal lattice index"
        )
    primitive_inverse = _exact_inverse(
        primitive_basis, "$display.lattice_inclusion.L_prim_basis"
    )
    _exact_inverse(conventional_basis, "$display.lattice_inclusion.L_conv_basis")
    inclusion_matrix = _exact_matmul(primitive_inverse, conventional_basis)
    if any(value.denominator != 1 for row in inclusion_matrix for value in row):
        raise ValueError(
            "$display.lattice_inclusion: L_conv_basis does not define a subset of L_prim_basis"
        )
    if _exact_matmul(primitive_basis, inclusion_matrix) != conventional_basis:
        raise ValueError("$display.lattice_inclusion: lattice basis relation does not replay")
    observed_index = abs(_exact_determinant(inclusion_matrix))
    if observed_index != lattice_index:
        raise ValueError(
            "$display.lattice_inclusion.index: differs from the exact lattice-basis index"
        )
    witness = _mapping(lattice["smith_witness"], "$display.lattice_inclusion.smith_witness")
    _fields(
        witness,
        {"left_unimodular", "diagonal", "right_unimodular"},
        "$display.lattice_inclusion.smith_witness",
    )
    smith_left = _exact_integer_matrix(
        witness["left_unimodular"],
        dimension,
        "$display.lattice_inclusion.smith_witness.left_unimodular",
    )
    diagonal_values = _list(
        witness["diagonal"], "$display.lattice_inclusion.smith_witness.diagonal"
    )
    if len(diagonal_values) != dimension:
        raise ValueError(
            "$display.lattice_inclusion.smith_witness.diagonal: vector dimension "
            f"must be {dimension}"
        )
    smith_diagonal = tuple(
        _positive_integer(
            value, f"$display.lattice_inclusion.smith_witness.diagonal[{index}]"
        )
        for index, value in enumerate(diagonal_values)
    )
    smith_right = _exact_integer_matrix(
        witness["right_unimodular"],
        dimension,
        "$display.lattice_inclusion.smith_witness.right_unimodular",
    )
    if abs(_exact_determinant(smith_left)) != 1 or abs(
        _exact_determinant(smith_right)
    ) != 1:
        raise ValueError(
            "$display.lattice_inclusion.smith_witness: left and right matrices must be unimodular"
        )
    if any(
        right % left != 0
        for left, right in zip(smith_diagonal, smith_diagonal[1:], strict=False)
    ):
        raise ValueError(
            "$display.lattice_inclusion.smith_witness.diagonal: expected Smith invariant factors"
        )
    if math.prod(smith_diagonal) != lattice_index:
        raise ValueError(
            "$display.lattice_inclusion.smith_witness.diagonal: product must equal lattice index"
        )
    diagonal_matrix = tuple(
        tuple(
            Fraction(smith_diagonal[row]) if row == column else Fraction()
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    if _exact_matmul(_exact_matmul(smith_left, inclusion_matrix), smith_right) != diagonal_matrix:
        raise ValueError(
            "$display.lattice_inclusion.smith_witness: exact Smith diagonal equation failed"
        )
    exact_cosets = tuple(
        _exact_vector(
            coset,
            dimension,
            f"$display.lattice_inclusion.coset_representatives[{index}]",
        )
        for index, coset in enumerate(cosets)
    )
    if zero not in exact_cosets:
        raise ValueError(
            "$display.lattice_inclusion.coset_representatives: must contain the identity vector"
        )
    for index, coset in enumerate(exact_cosets):
        if not _exact_is_integral(_exact_matvec(primitive_inverse, coset)):
            raise ValueError(
                "$display.lattice_inclusion.coset_representatives"
                f"[{index}]: representative is not in L_prim"
            )
    conventional_inverse = _exact_inverse(
        conventional_basis, "$display.lattice_inclusion.L_conv_basis"
    )
    for left_index, left in enumerate(exact_cosets):
        for right_index in range(left_index + 1, len(exact_cosets)):
            difference = tuple(
                left_value - right_value
                for left_value, right_value in zip(
                    left, exact_cosets[right_index], strict=True
                )
            )
            if _exact_is_integral(_exact_matvec(conventional_inverse, difference)):
                raise ValueError(
                    "$display.lattice_inclusion.coset_representatives: representatives "
                    "must be unique quotient classes"
                )

    verification = _mapping(
        value["independent_verification"], "$display.independent_verification"
    )
    _fields(verification, {"status", "source_digest"}, "$display.independent_verification")
    verification_status = _string(
        verification["status"], "$display.independent_verification.status"
    )
    if verification_status not in {"verified", "unverified", "failed"}:
        raise ValueError("$display.independent_verification.status: unsupported status")
    _digest(
        verification["source_digest"], "$display.independent_verification.source_digest"
    )

    return DisplayRecord(
        schema_version=schema_version,
        wyckoff_id=wyckoff_id,
        hall_symbol=hall,
        hermann_mauguin_symbol=hm,
        origin_choice=origin,
        wyckoff_letter=letter,
        conventional_multiplicity=multiplicity,
        site_symmetry_symbol=site_symmetry,
        coordinate_crosswalk=crosswalk,  # type: ignore[arg-type]
        lattice_inclusion=lattice,  # type: ignore[arg-type]
        independent_verification=verification,  # type: ignore[arg-type]
    )


def _relative_path(value: Any, path: str) -> str:
    text = _string(value, path)
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{path}: control characters are forbidden in public paths")
    if re.match(r"^[A-Za-z]:[/\\]", text):
        raise ValueError(f"{path}: expected repository-relative path, not a Windows absolute path")
    if "\\" in text:
        raise ValueError(f"{path}: expected repository-relative POSIX path")
    if "//" in text:
        raise ValueError(f"{path}: path must use canonical single separators")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/")):
        raise ValueError(f"{path}: expected repository-relative path")
    if str(candidate) != text:
        raise ValueError(f"{path}: path is not in canonical POSIX form")
    return text


def validate_manifest(mapping: Mapping[str, Any]) -> CatalogueManifest:
    """Validate cross-field consistency for one scoped v1 catalogue manifest."""

    value = _mapping(mapping, "$manifest")
    _validate_json_tree(value)
    required = {
        "schema_version",
        "record_type",
        "catalogue_schema_version",
        "display_schema_version",
        "normalization_version",
        "scope",
        "generator",
        "environment",
        "independent_count_reference",
        "counts",
        "per_group",
        "files",
        "timing",
        "status",
        "failures",
    }
    _fields(value, required, "$manifest")
    schema_version = _schema_version(value["schema_version"], "$manifest.schema_version")
    record_type = _string(value["record_type"], "$manifest.record_type")
    if record_type != "catalogue-manifest":
        raise ValueError("$manifest.record_type: expected catalogue-manifest")
    catalogue_version = _schema_version(
        value["catalogue_schema_version"], "$manifest.catalogue_schema_version"
    )
    display_version = _schema_version(
        value["display_schema_version"], "$manifest.display_schema_version"
    )
    normalization_version = _normalization_version(
        value["normalization_version"], "$manifest.normalization_version"
    )

    scope = _mapping(value["scope"], "$manifest.scope")
    _fields(scope, {"international_numbers"}, "$manifest.scope")
    scope_values = _list(scope["international_numbers"], "$manifest.scope.international_numbers")
    groups = tuple(
        _integer(group, f"$manifest.scope.international_numbers[{index}]", minimum=1, maximum=230)
        for index, group in enumerate(scope_values)
    )
    if not groups:
        raise ValueError("$manifest.scope.international_numbers: must not be empty")
    if len(set(groups)) != len(groups):
        raise ValueError("$manifest.scope.international_numbers: duplicate international_number")

    generator = _mapping(value["generator"], "$manifest.generator")
    _fields(generator, {"commit", "input_digest", "executable"}, "$manifest.generator")
    commit = _string(generator["commit"], "$manifest.generator.commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("$manifest.generator.commit: expected 40-64 lowercase hex digits")
    _digest(generator["input_digest"], "$manifest.generator.input_digest")
    executable = _mapping(generator["executable"], "$manifest.generator.executable")
    _fields(
        executable,
        {"id", "sha256", "environment_lock", "environment_lock_sha256"},
        "$manifest.generator.executable",
    )
    _string(executable["id"], "$manifest.generator.executable.id")
    _digest(executable["sha256"], "$manifest.generator.executable.sha256")
    executable_lock_path = _relative_path(
        executable["environment_lock"], "$manifest.generator.executable.environment_lock"
    )
    if executable_lock_path != _CATALOGUE_LOCK_PATH:
        raise ValueError(
            "$manifest.generator.executable.environment_lock: expected canonical catalogue lock"
        )
    executable_lock_digest = _digest(
        executable["environment_lock_sha256"],
        "$manifest.generator.executable.environment_lock_sha256",
    )

    environment = _mapping(value["environment"], "$manifest.environment")
    _fields(environment, {"lock", "components"}, "$manifest.environment")
    environment_lock = _mapping(environment["lock"], "$manifest.environment.lock")
    _fields(environment_lock, {"path", "sha256"}, "$manifest.environment.lock")
    environment_lock_path = _relative_path(
        environment_lock["path"], "$manifest.environment.lock.path"
    )
    if environment_lock_path != _CATALOGUE_LOCK_PATH:
        raise ValueError("$manifest.environment.lock.path: expected canonical catalogue lock")
    environment_lock_digest = _digest(
        environment_lock["sha256"], "$manifest.environment.lock.sha256"
    )
    if (
        executable_lock_path != environment_lock_path
        or executable_lock_digest != environment_lock_digest
    ):
        raise ValueError("$manifest.environment.lock: environment lock path and digest must match executable")
    components = _list(environment["components"], "$manifest.environment.components")
    if not components:
        raise ValueError("$manifest.environment.components: inventory must not be empty")
    component_names: set[str] = set()
    pinned_components: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(components):
        path = f"$manifest.environment.components[{index}]"
        component = _mapping(item, path)
        _fields(component, {"kind", "name", "version", "archive", "license"}, path)
        kind = _string(component["kind"], f"{path}.kind")
        if kind not in {"runtime", "package"}:
            raise ValueError(f"{path}.kind: expected runtime or package")
        name = _string(component["name"], f"{path}.name")
        if re.fullmatch(r"[a-z][a-z0-9._+-]*", name) is None:
            raise ValueError(f"{path}.name: expected canonical lowercase component name")
        if name in component_names:
            raise ValueError(f"{path}.name: duplicate environment component {name}")
        component_names.add(name)
        version = _string(component["version"], f"{path}.version")
        archive = _mapping(component["archive"], f"{path}.archive")
        _fields(archive, {"url", "sha256"}, f"{path}.archive")
        archive_url = _string(archive["url"], f"{path}.archive.url")
        if not archive_url.startswith("https://"):
            raise ValueError(f"{path}.archive.url: expected immutable HTTPS archive URL")
        _digest(archive["sha256"], f"{path}.archive.sha256")
        license_record = _mapping(component["license"], f"{path}.license")
        _fields(license_record, {"spdx", "text_sha256"}, f"{path}.license")
        _string(license_record["spdx"], f"{path}.license.spdx")
        _digest(license_record["text_sha256"], f"{path}.license.text_sha256")
        pinned_components[name] = (kind, version)
    if pinned_components.get("gap") != ("runtime", "4.15.1"):
        raise ValueError("$manifest.environment.components: inventory requires GAP 4.15.1 runtime")
    if pinned_components.get("cryst") != ("package", "4.1.30"):
        raise ValueError("$manifest.environment.components: inventory requires Cryst 4.1.30 package")

    reference = _mapping(
        value["independent_count_reference"], "$manifest.independent_count_reference"
    )
    _fields(
        reference,
        {"vector_digest", "source_digest"},
        "$manifest.independent_count_reference",
    )
    _digest(reference["vector_digest"], "$manifest.independent_count_reference.vector_digest")
    _digest(reference["source_digest"], "$manifest.independent_count_reference.source_digest")

    counts = _mapping(value["counts"], "$manifest.counts")
    count_fields = {
        "expected_space_groups",
        "observed_space_groups",
        "expected_wyckoff_positions",
        "geometry_rows",
        "display_rows",
        "verified_display_rows",
    }
    _fields(counts, count_fields, "$manifest.counts")
    checked_counts = {
        name: _integer(counts[name], f"$manifest.counts.{name}", minimum=0)
        for name in count_fields
    }
    if checked_counts["expected_space_groups"] != len(groups):
        raise ValueError("$manifest.counts.expected_space_groups: must equal scope length")

    per_group_values = _list(value["per_group"], "$manifest.per_group")
    rows: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    per_group_fields = {
        "international_number",
        "reference_count",
        "geometry_rows",
        "display_rows",
        "verified_display_rows",
        "status",
    }
    for index, item in enumerate(per_group_values):
        path = f"$manifest.per_group[{index}]"
        row = _mapping(item, path)
        _fields(row, per_group_fields, path)
        group = _integer(row["international_number"], f"{path}.international_number", minimum=1, maximum=230)
        if group in seen:
            raise ValueError(f"{path}: duplicate international_number {group}")
        seen.add(group)
        reference_count = _integer(row["reference_count"], f"{path}.reference_count", minimum=1)
        geometry_rows = _integer(row["geometry_rows"], f"{path}.geometry_rows", minimum=0)
        display_rows = _integer(row["display_rows"], f"{path}.display_rows", minimum=0)
        verified_rows = _integer(
            row["verified_display_rows"], f"{path}.verified_display_rows", minimum=0
        )
        row_status = _string(row["status"], f"{path}.status")
        if row_status not in {"success", "incomplete", "failure"}:
            raise ValueError(f"{path}.status: unsupported per-group status")
        if geometry_rows > reference_count:
            raise ValueError(f"{path}.geometry_rows: cannot exceed reference_count")
        if display_rows > geometry_rows:
            raise ValueError(f"{path}.display_rows: cannot exceed geometry_rows")
        if verified_rows > display_rows:
            raise ValueError(f"{path}.verified_display_rows: cannot exceed display_rows")
        rows.append(row)
    if seen != set(groups):
        raise ValueError("$manifest.per_group: rows must match scope and include every reference count")

    sums = {
        "expected_wyckoff_positions": sum(int(row["reference_count"]) for row in rows),
        "geometry_rows": sum(int(row["geometry_rows"]) for row in rows),
        "display_rows": sum(int(row["display_rows"]) for row in rows),
        "verified_display_rows": sum(int(row["verified_display_rows"]) for row in rows),
        "observed_space_groups": sum(int(row["geometry_rows"]) > 0 for row in rows),
    }
    for name, total in sums.items():
        if checked_counts[name] != total:
            raise ValueError(f"$manifest.counts.{name}: disagrees with per-group rows")

    files_values = _list(value["files"], "$manifest.files")
    files: list[Mapping[str, Any]] = []
    file_paths: set[str] = set()
    file_kinds: set[str] = set()
    for index, item in enumerate(files_values):
        path = f"$manifest.files[{index}]"
        file_row = _mapping(item, path)
        _fields(file_row, {"kind", "path", "sha256"}, path)
        kind = _string(file_row["kind"], f"{path}.kind")
        if kind not in _CANONICAL_FILE_PATHS:
            raise ValueError(f"{path}.kind: unsupported generated-file kind")
        if kind in file_kinds:
            raise ValueError(f"{path}.kind: duplicate singleton file kind {kind}")
        relative_path = _relative_path(file_row["path"], f"{path}.path")
        if relative_path != _CANONICAL_FILE_PATHS[kind]:
            raise ValueError(f"{path}.path: file kind {kind} requires its canonical logical path")
        if relative_path in file_paths:
            raise ValueError(f"{path}.path: duplicate generated-file path")
        file_paths.add(relative_path)
        file_kinds.add(kind)
        _digest(file_row["sha256"], f"{path}.sha256")
        files.append(file_row)

    timing = _mapping(value["timing"], "$manifest.timing")
    _fields(timing, {"duration_milliseconds"}, "$manifest.timing")
    _integer(timing["duration_milliseconds"], "$manifest.timing.duration_milliseconds", minimum=0)

    failures_values = _list(value["failures"], "$manifest.failures")
    failures: list[Mapping[str, Any]] = []
    for index, item in enumerate(failures_values):
        path = f"$manifest.failures[{index}]"
        failure = _mapping(item, path)
        _fields(failure, {"stage", "code", "message"}, path, {"international_number"})
        _string(failure["stage"], f"{path}.stage")
        _string(failure["code"], f"{path}.code")
        _string(failure["message"], f"{path}.message")
        if "international_number" in failure:
            failure_group = _integer(
                failure["international_number"],
                f"{path}.international_number",
                minimum=1,
                maximum=230,
            )
            if failure_group not in set(groups):
                raise ValueError(f"{path}.international_number: must be inside manifest scope")
        failures.append(failure)

    failure_groups = {
        int(failure["international_number"])
        for failure in failures
        if "international_number" in failure
    }
    for index, row in enumerate(rows):
        group = int(row["international_number"])
        counts_succeed = (
            int(row["geometry_rows"]) == int(row["reference_count"])
            and int(row["display_rows"]) == int(row["reference_count"])
            and int(row["verified_display_rows"]) == int(row["reference_count"])
        )
        expected_row_status = (
            "failure" if group in failure_groups else "success" if counts_succeed else "incomplete"
        )
        if row["status"] != expected_row_status:
            raise ValueError(
                f"$manifest.per_group[{index}].status: expected {expected_row_status} from counts and failures"
            )

    status = _mapping(value["status"], "$manifest.status")
    _fields(status, {"geometry_complete", "display_complete", "release_complete"}, "$manifest.status")
    status_values = {
        name: _boolean(status[name], f"$manifest.status.{name}")
        for name in ("geometry_complete", "display_complete", "release_complete")
    }
    if failures and any(status_values.values()):
        raise ValueError("$manifest.status: complete states cannot coexist with failures")

    geometry_counts_complete = all(
        int(row["geometry_rows"]) == int(row["reference_count"]) for row in rows
    )
    display_counts_complete = all(
        int(row["display_rows"]) == int(row["reference_count"])
        and int(row["verified_display_rows"]) == int(row["reference_count"])
        for row in rows
    )
    expected_geometry = geometry_counts_complete and "geometry" in file_kinds and not failures
    expected_display = (
        expected_geometry and display_counts_complete and "display" in file_kinds and not failures
    )
    global_scope = groups == tuple(range(1, 231))
    global_reference_total = checked_counts["expected_wyckoff_positions"] == 1731
    release_artifacts = set(_CANONICAL_FILE_PATHS).issubset(file_kinds)
    if status_values["display_complete"] and not status_values["geometry_complete"]:
        raise ValueError("$manifest.status.display_complete: implies geometry_complete")
    if status_values["release_complete"] and not (
        status_values["geometry_complete"] and status_values["display_complete"]
    ):
        raise ValueError(
            "$manifest.status.release_complete: implies geometry_complete and display_complete"
        )
    if status_values["release_complete"] and not global_scope:
        raise ValueError("$manifest.status.release_complete: requires global scope 1..230")
    if status_values["release_complete"] and not global_reference_total:
        raise ValueError("$manifest.status.release_complete: global reference total must be 1731")
    if status_values["release_complete"] and not release_artifacts:
        raise ValueError("$manifest.status.release_complete: requires all canonical artifacts")
    expected_release = (
        expected_geometry
        and expected_display
        and global_scope
        and global_reference_total
        and release_artifacts
    )
    expected_status = {
        "geometry_complete": expected_geometry,
        "display_complete": expected_display,
        "release_complete": expected_release,
    }
    for name, expected in expected_status.items():
        if status_values[name] != expected:
            raise ValueError(f"$manifest.status.{name}: inconsistent completion state")

    return CatalogueManifest(
        schema_version=schema_version,
        record_type=record_type,
        catalogue_schema_version=catalogue_version,
        display_schema_version=display_version,
        normalization_version=normalization_version,
        scope=scope,  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
        environment=environment,  # type: ignore[arg-type]
        independent_count_reference=reference,  # type: ignore[arg-type]
        counts=counts,  # type: ignore[arg-type]
        per_group=tuple(rows),  # type: ignore[arg-type]
        files=tuple(files),  # type: ignore[arg-type]
        timing=timing,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        failures=tuple(failures),  # type: ignore[arg-type]
    )


__all__ = [
    "CatalogueManifest",
    "CatalogueRecord",
    "DisplayRecord",
    "canonical_json",
    "parse_catalogue_record",
    "parse_display_record",
    "strict_json_loads",
    "validate_manifest",
]
