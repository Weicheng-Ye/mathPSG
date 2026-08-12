"""Exact symbolic-orbit resolution and semantic parameter-routing authority."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
from types import SimpleNamespace
from typing import Any
import weakref

from .catalogue import catalogue_record_order_key
from .catalogue_loader import CatalogueIndex
from .catalogue_schema import CatalogueManifest, CatalogueRecord, canonical_json
from .classification_schema import (
    CandidateGeometryEvidence,
    ClassificationRequest,
    InstanceParameterRoute,
    OrbitInstance,
    ParameterRoutingResult,
    SCHEMA_VERSION,
    _candidate_tuple_digest,
    _comparison_digest,
    canonical_classification_json,
)
from .classifier_cache import CacheKey


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_Q_RE = re.compile(r"q\((-?(?:0|[1-9][0-9]*)),([1-9][0-9]*)\)\Z")
_PROTOCOL = b"mathpsg-certified-query-v1|"
_VERIFIED_CATALOGUE_FACTORY_TOKEN = object()
_VERIFIED_CATALOGUE_AUTHORITIES: dict[object, object] = {}

ROUTING_ALGORITHM_DIGEST = "sha256:" + hashlib.sha256(
    b"mathpsg-exact-affine-routing-v1"
).hexdigest()
ROUTING_VERIFIER_LIBRARY_DIGEST = "sha256:" + hashlib.sha256(
    b"mathpsg-routing-semantic-verifier-v1"
).hexdigest()


Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Affine = tuple[Matrix, Vector]


def _plain(value: object) -> object:
    """Return a JSON-native snapshot of recursively immutable schema values."""

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


def _fraction(value: object, path: str) -> Fraction:
    if type(value) is not str:
        raise TypeError(f"{path}: expected q(n,d) rational")
    match = _Q_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{path}: invalid q(n,d) rational")
    result = Fraction(int(match.group(1)), int(match.group(2)))
    if f"q({result.numerator},{result.denominator})" != value:
        raise ValueError(f"{path}: rational is not reduced")
    return result


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _matrix(value: object, rows: int, columns: int, path: str) -> Matrix:
    if not isinstance(value, (tuple, list)) or len(value) != rows:
        raise ValueError(f"{path}: expected {rows} rows")
    for row_index, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != columns:
            raise ValueError(f"{path}[{row_index}]: expected {columns} columns")
    return tuple(
        tuple(
            _fraction(item, f"{path}[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        )
        for row_index, row in enumerate(value)
    )


def _vector(value: object, length: int, path: str) -> Vector:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{path}: expected length {length}")
    return tuple(_fraction(item, f"{path}[{index}]") for index, item in enumerate(value))


def _identity(size: int) -> Matrix:
    return tuple(
        tuple(Fraction(row == column) for column in range(size))
        for row in range(size)
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left:
        return ()
    columns = len(right[0]) if right else 0
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(len(right))),
                Fraction(0),
            )
            for column in range(columns)
        )
        for row in range(len(left))
    )


def _matvec(matrix: Matrix, vector: Sequence[Fraction]) -> Vector:
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(len(vector))),
            Fraction(0),
        )
        for row in range(len(matrix))
    )


def _inverse(matrix: Matrix, path: str) -> Matrix:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError(f"{path}: expected square matrix")
    work = [list(row) + list(identity) for row, identity in zip(matrix, _identity(size), strict=True)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            raise ValueError(f"{path}: matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [entry / scale for entry in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(work[row], work[column], strict=True)
                ]
    return tuple(tuple(row[size:]) for row in work)


def _mod_one(value: Fraction) -> Fraction:
    return value - math.floor(value)


def _reduced(vector: Sequence[Fraction]) -> Vector:
    return tuple(_mod_one(value) for value in vector)


def _add(left: Sequence[Fraction], right: Sequence[Fraction]) -> Vector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _negate(value: Sequence[Fraction]) -> Vector:
    return tuple(-item for item in value)


def _affine_from_mapping(value: Mapping[str, Any], path: str) -> Affine:
    return (
        _matrix(value["matrix"], 3, 3, f"{path}.matrix"),
        _vector(value["translation"], 3, f"{path}.translation"),
    )


def _affine_compose(left: Affine, right: Affine) -> Affine:
    """Return ``left`` after ``right`` in primitive-lattice coordinates."""

    return (
        _matmul(left[0], right[0]),
        _reduced(_add(_matvec(left[0], right[1]), left[1])),
    )


def _affine_inverse(value: Affine) -> Affine:
    inverse = _inverse(value[0], "affine")
    return inverse, _reduced(_matvec(inverse, _negate(value[1])))


def _affine_apply(value: Affine, point: Sequence[Fraction]) -> Vector:
    return _reduced(_add(_matvec(value[0], point), value[1]))


def _lattice(record: CatalogueRecord) -> tuple[Matrix, Matrix]:
    basis = _matrix(
        record.space_group_action["translation_basis"],
        3,
        3,
        "$CatalogueRecord.space_group_action.translation_basis",
    )
    return basis, _inverse(basis, "primitive lattice")


def _to_lattice_affine(value: Affine, lattice: Matrix, inverse_lattice: Matrix) -> Affine:
    return (
        _matmul(_matmul(inverse_lattice, value[0]), lattice),
        _reduced(_matvec(inverse_lattice, value[1])),
    )


def _finite_affine_quotient(record: CatalogueRecord) -> tuple[Affine, ...]:
    lattice, inverse_lattice = _lattice(record)
    generators = tuple(
        _to_lattice_affine(
            _affine_from_mapping(item, "$CatalogueRecord.space_group_action.source_generators"),
            lattice,
            inverse_lattice,
        )
        for item in record.space_group_action["source_generators"]
    )
    steps = generators + tuple(_affine_inverse(item) for item in generators)
    identity = (_identity(3), (Fraction(0),) * 3)
    seen = {identity}
    pending = deque((identity,))
    while pending:
        current = pending.popleft()
        for step in steps:
            product = _affine_compose(step, current)
            if product not in seen:
                seen.add(product)
                pending.append(product)
                if len(seen) > 48:
                    raise ValueError("catalogue affine quotient exceeds crystallographic bound")
    return tuple(sorted(seen, key=lambda value: _canonical_json(_affine_mapping(value))))


def _affine_mapping(value: Affine) -> dict[str, object]:
    return {
        "matrix": [[_fraction_text(entry) for entry in row] for row in value[0]],
        "translation": [_fraction_text(entry) for entry in value[1]],
    }


def _affine_digest(value: Affine) -> str:
    return _digest("primitive-affine-element", _affine_mapping(value))


def _branch(record: CatalogueRecord, branch: Mapping[str, Any]) -> tuple[Vector, Matrix]:
    dimension = int(record.orbit["parameter_dimension"])
    return (
        _vector(branch["offset"], 3, "$CatalogueRecord.orbit.branches.offset"),
        _matrix(branch["basis"], 3, dimension, "$CatalogueRecord.orbit.branches.basis"),
    )


def _reference_branch(record: CatalogueRecord) -> tuple[Vector, Matrix]:
    reference = record.orbit["reference_branch_digest"]
    branch = next(
        item for item in record.orbit["branches"] if item["branch_digest"] == reference
    )
    return _branch(record, branch)


def _specialized_reference(record: CatalogueRecord, parameters: Sequence[Fraction]) -> Vector:
    offset, basis = _reference_branch(record)
    return _add(offset, _matvec(basis, parameters))


def _primitive_point(record: CatalogueRecord, euclidean: Sequence[Fraction]) -> Vector:
    _, inverse_lattice = _lattice(record)
    return _reduced(_matvec(inverse_lattice, euclidean))


def _point_geometry(quotient: Sequence[Affine], point: Vector) -> tuple[Vector, ...]:
    return tuple(sorted({_affine_apply(element, point) for element in quotient}))


def _geometry_mapping(points: Sequence[Vector]) -> list[list[str]]:
    return [[_fraction_text(entry) for entry in point] for point in points]


def _geometry_digest(points: Sequence[Vector]) -> str:
    return _digest("exact-reduced-orbit-geometry", _geometry_mapping(points))


def _stabilizer(quotient: Sequence[Affine], point: Vector) -> tuple[Affine, ...]:
    return tuple(element for element in quotient if _affine_apply(element, point) == point)


def _literal_stabilizer(record: CatalogueRecord) -> tuple[Affine, ...]:
    lattice, inverse_lattice = _lattice(record)
    values = tuple(
        _to_lattice_affine(
            _affine_from_mapping(item, "$CatalogueRecord.stabilizer.embedded_elements"),
            lattice,
            inverse_lattice,
        )
        for item in record.stabilizer["embedded_elements"]
    )
    return tuple(sorted(values, key=lambda value: _canonical_json(_affine_mapping(value))))


def _stabilizer_elements(values: Sequence[Affine]) -> list[dict[str, str]]:
    return [{"affine_digest": _affine_digest(value)} for value in values]


def _stabilizer_digest(values: Sequence[Affine]) -> str:
    return _digest("literal-embedded-stabilizer", _stabilizer_elements(values))


def _inclusion_elements(
    values: Sequence[Affine], action_provenance_digest: str
) -> list[dict[str, str]]:
    return [
        {
            "affine_digest": _affine_digest(value),
            "transport_digest": _digest(
                "transported-literal-element",
                {
                    "action_provenance_digest": action_provenance_digest,
                    "affine_digest": _affine_digest(value),
                },
            ),
        }
        for value in values
    ]


def _inclusion_digest(values: Sequence[Affine], action_provenance_digest: str) -> str:
    return _digest(
        "transported-literal-inclusion",
        _inclusion_elements(values, action_provenance_digest),
    )


def _solve_square(matrix: Matrix, right: Vector) -> Vector | None:
    try:
        inverse = _inverse(matrix, "family pivot")
    except ValueError:
        return None
    return _matvec(inverse, right)


def _family_parameter(
    target: Vector,
    offset: Vector,
    basis: Matrix,
) -> Vector | None:
    dimension = 0 if not basis else len(basis[0])
    difference = tuple(target[row] - offset[row] for row in range(3))
    if dimension == 0:
        return () if all(value.denominator == 1 for value in difference) else None
    lower = tuple(sum((min(Fraction(0), entry) for entry in row), Fraction(0)) for row in basis)
    upper = tuple(sum((max(Fraction(0), entry) for entry in row), Fraction(0)) for row in basis)
    ranges = tuple(
        range(
            math.floor(lower[row] - difference[row]) - 1,
            math.ceil(upper[row] - difference[row]) + 2,
        )
        for row in range(3)
    )
    for lattice_shift in itertools.product(*ranges):
        right = tuple(
            difference[row] + lattice_shift[row] for row in range(3)
        )
        for pivot_rows in itertools.combinations(range(3), dimension):
            square = tuple(
                tuple(basis[row][column] for column in range(dimension))
                for row in pivot_rows
            )
            solution = _solve_square(square, tuple(right[row] for row in pivot_rows))
            if solution is None or any(not Fraction(0) <= value < 1 for value in solution):
                continue
            if _matvec(basis, solution) == right:
                return solution
    return None


def _candidate_geometry(
    candidate: CatalogueRecord,
    point_geometry: tuple[Vector, ...],
    quotient: Sequence[Affine],
) -> tuple[bool, tuple[Vector, ...] | None]:
    _, inverse_lattice = _lattice(candidate)
    for branch_row in candidate.orbit["branches"]:
        offset, basis = _branch(candidate, branch_row)
        lattice_offset = _matvec(inverse_lattice, offset)
        lattice_basis = _matmul(inverse_lattice, basis)
        for target in point_geometry:
            parameters = _family_parameter(target, lattice_offset, lattice_basis)
            if parameters is None:
                continue
            candidate_point = _reduced(
                _add(lattice_offset, _matvec(lattice_basis, parameters))
            )
            candidate_geometry = _point_geometry(quotient, candidate_point)
            if candidate_geometry == point_geometry:
                return True, candidate_geometry
    return False, None


def _record_digest(record: CatalogueRecord) -> str:
    return _raw_digest(canonical_json(record))


def _normalization_digest(records: Sequence[CatalogueRecord]) -> str:
    return _digest(
        "catalogue-normalization",
        [
            {
                "normalization_version": record.provenance["normalization_version"],
                "record_digest": _record_digest(record),
            }
            for record in records
        ],
    )


def _backend_identity_digest(backend: object | None) -> str | None:
    if backend is None:
        return None
    identity = getattr(backend, "identity", None)
    if identity is None:
        payload: object = {"object_identity": id(backend)}
    elif dataclasses.is_dataclass(identity):
        payload = {
            field.name: getattr(identity, field.name)
            for field in dataclasses.fields(identity)
        }
    elif hasattr(identity, "__dict__"):
        payload = {
            key: value
            for key, value in sorted(vars(identity).items())
            if not key.startswith("_")
        }
    else:
        payload = {
            "identity_object": id(identity),
            "identity_type": f"{type(identity).__module__}.{type(identity).__qualname__}",
        }
    try:
        return _digest(
            "classifier-backend-identity",
            {
                "backend_type": f"{type(backend).__module__}.{type(backend).__qualname__}",
                "identity": payload,
            },
        )
    except (TypeError, ValueError) as error:
        raise TypeError("classifier backend identity is not canonically serializable") from error


def _backend_task5_release_store(backend: object | None) -> object | None:
    if backend is None:
        return None
    store = getattr(backend, "task5_release_store", None)
    if store is None:
        return None
    raise TypeError(
        "release Task5 stores are unavailable in the standalone host-native package"
    )


@dataclass(frozen=True, slots=True)
class _VerifiedCatalogueAuthority:
    index: CatalogueIndex
    catalogue_manifest_digest: str
    normalization_digest: str
    record_digests: tuple[tuple[str, str], ...]
    release_complete: bool
    backend: object | None
    backend_identity_digest: str | None
    backend_task5_release_store: object | None
    manifest: CatalogueManifest | None
    factory_token: object


def _verified_catalogue_authority(
    value: object,
    *,
    _allow_construction: bool = False,
) -> "_VerifiedCatalogueAuthority | None":
    if type(value) is not VerifiedCatalogue:
        return None
    seal = value._seal
    entry = _VERIFIED_CATALOGUE_AUTHORITIES.get(seal)
    if type(entry) is _VerifiedCatalogueAuthority and _allow_construction:
        # Construction is in progress; the factory replaces this temporary
        # entry with a weak/finalized authority immediately afterwards.
        return entry
    if type(entry) is tuple and len(entry) == 2:
        reference, authority = entry
        if (
            isinstance(reference, weakref.ReferenceType)
            and type(authority) is _VerifiedCatalogueAuthority
            and reference() is value
        ):
            return authority
    return None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class VerifiedCatalogue:
    """Immutable record snapshot carrying the caller's verified manifest binding."""

    index: CatalogueIndex
    catalogue_manifest_digest: str
    normalization_digest: str
    record_digests: tuple[tuple[str, str], ...]
    release_complete: bool
    backend: object | None
    _seal: object

    def __post_init__(self) -> None:
        authority = _verified_catalogue_authority(
            self,
            _allow_construction=True,
        )
        if (
            type(authority) is not _VerifiedCatalogueAuthority
            or authority.factory_token is not _VERIFIED_CATALOGUE_FACTORY_TOKEN
        ):
            raise TypeError("VerifiedCatalogue construction requires a verification factory")
        if type(self.index) is not CatalogueIndex:
            raise TypeError("$VerifiedCatalogue.index: expected CatalogueIndex")
        if (
            self.index is not authority.index
            or self.backend is not authority.backend
            or self.catalogue_manifest_digest != authority.catalogue_manifest_digest
            or self.normalization_digest != authority.normalization_digest
            or tuple(tuple(item) for item in self.record_digests)
            != authority.record_digests
            or self.release_complete is not authority.release_complete
        ):
            raise ValueError("verified catalogue differs from its sealed source snapshot")
        records = tuple(sorted(tuple(self.index), key=catalogue_record_order_key))
        expected = tuple(sorted((record.wyckoff_id, _record_digest(record)) for record in records))
        supplied = tuple(tuple(item) for item in self.record_digests)
        if supplied != expected or supplied != authority.record_digests:
            raise ValueError("$VerifiedCatalogue.record_digests: record snapshot differs")
        _require_digest(
            self.catalogue_manifest_digest,
            "$VerifiedCatalogue.catalogue_manifest_digest",
        )
        _require_digest(self.normalization_digest, "$VerifiedCatalogue.normalization_digest")
        if type(self.release_complete) is not bool:
            raise TypeError("$VerifiedCatalogue.release_complete: expected boolean")
        recomputed_normalization = _normalization_digest(records)
        if (
            self.normalization_digest != recomputed_normalization
            or authority.normalization_digest != recomputed_normalization
        ):
            raise ValueError("verified catalogue normalization binding differs")
        if _backend_identity_digest(self.backend) != authority.backend_identity_digest:
            raise ValueError("verified catalogue backend identity binding differs")
        if (
            _backend_task5_release_store(self.backend)
            is not authority.backend_task5_release_store
        ):
            raise ValueError("verified catalogue backend Task5 store binding differs")
        if self.release_complete:
            if authority.manifest is None:
                raise ValueError("release catalogue lacks retained manifest evidence")
            _validate_release_catalogue(self.index, authority.manifest)
            expected_manifest = _raw_digest(canonical_json(authority.manifest))
        else:
            if authority.manifest is not None:
                raise ValueError("diagnostic catalogue unexpectedly retains release evidence")
            expected_manifest = _digest(
                "diagnostic-catalogue-snapshot",
                [[record.wyckoff_id, _record_digest(record)] for record in records],
            )
        if self.catalogue_manifest_digest != expected_manifest:
            raise ValueError("verified catalogue manifest binding differs")
        object.__setattr__(self, "record_digests", supplied)

    def candidate_records(self, space_group: int, setting_id: str) -> tuple[CatalogueRecord, ...]:
        return tuple(
            record
            for record in self.index
            if record.space_group["international_number"] == space_group
            and record.space_group["setting"] == setting_id
        )

    def candidate_ids(self, space_group: int, setting_id: str) -> tuple[str, ...]:
        return tuple(
            record.wyckoff_id
            for record in self.candidate_records(space_group, setting_id)
        )

    def candidate_record_set_digest(self, space_group: int, setting_id: str) -> str:
        by_id = dict(self.record_digests)
        return _digest(
            "candidate-record-set",
            [
                [identifier, by_id[identifier]]
                for identifier in self.candidate_ids(space_group, setting_id)
            ],
        )


def _make_verified_catalogue(
    index: CatalogueIndex,
    *,
    manifest_digest: str,
    release_complete: bool,
    backend: object | None,
    manifest: CatalogueManifest | None = None,
) -> VerifiedCatalogue:
    records = tuple(sorted(tuple(index), key=catalogue_record_order_key))
    canonical_index = CatalogueIndex(records)
    normalization = _normalization_digest(records)
    record_digests = tuple(
        sorted((record.wyckoff_id, _record_digest(record)) for record in records)
    )
    seal = object()
    authority = _VerifiedCatalogueAuthority(
        canonical_index,
        manifest_digest,
        normalization,
        record_digests,
        release_complete,
        backend,
        _backend_identity_digest(backend),
        _backend_task5_release_store(backend),
        manifest,
        _VERIFIED_CATALOGUE_FACTORY_TOKEN,
    )
    _VERIFIED_CATALOGUE_AUTHORITIES[seal] = authority
    try:
        value = VerifiedCatalogue(
            canonical_index,
            manifest_digest,
            normalization,
            record_digests,
            release_complete,
            backend,
            seal,
        )
    except Exception:
        _VERIFIED_CATALOGUE_AUTHORITIES.pop(seal, None)
        raise
    reference: weakref.ReferenceType[VerifiedCatalogue]

    def finalize(expired: weakref.ReferenceType[VerifiedCatalogue]) -> None:
        entry = _VERIFIED_CATALOGUE_AUTHORITIES.get(seal)
        if type(entry) is tuple and entry[0] is expired:
            _VERIFIED_CATALOGUE_AUTHORITIES.pop(seal, None)

    reference = weakref.ref(value, finalize)
    _VERIFIED_CATALOGUE_AUTHORITIES[seal] = (reference, authority)
    return value


def make_diagnostic_verified_catalogue(
    index: CatalogueIndex,
    *,
    backend: object | None = None,
) -> VerifiedCatalogue:
    """Bind a finite test/diagnostic snapshot without claiming release coverage."""

    if type(index) is not CatalogueIndex:
        raise TypeError("diagnostic catalogue requires CatalogueIndex")
    records = tuple(sorted(tuple(index), key=catalogue_record_order_key))
    manifest_digest = _digest(
        "diagnostic-catalogue-snapshot",
        [[record.wyckoff_id, _record_digest(record)] for record in records],
    )
    return _make_verified_catalogue(
        CatalogueIndex(records),
        manifest_digest=manifest_digest,
        release_complete=False,
        backend=backend,
    )


def _validate_release_catalogue(
    index: CatalogueIndex,
    manifest: CatalogueManifest,
) -> tuple[CatalogueRecord, ...]:
    if type(index) is not CatalogueIndex or type(manifest) is not CatalogueManifest:
        raise TypeError("release catalogue binding requires CatalogueIndex and CatalogueManifest")
    records = tuple(sorted(tuple(index), key=catalogue_record_order_key))
    counts = manifest.counts
    status = manifest.status
    groups = {record.space_group["international_number"] for record in records}
    if (
        status.get("release_complete") is not True
        or status.get("geometry_complete") is not True
        or counts.get("expected_space_groups") != 230
        or counts.get("observed_space_groups") != 230
        or counts.get("expected_wyckoff_positions") != 1731
        or counts.get("geometry_rows") != 1731
        or len(records) != 1731
        or groups != set(range(1, 231))
    ):
        raise ValueError("catalogue_incomplete: release catalogue is not exactly 230/1731")
    if any(
        record.provenance["normalization_version"] != manifest.normalization_version
        for record in records
    ):
        raise ValueError("catalogue manifest normalization differs from records")
    ndjson = b"".join(canonical_json(record) + b"\n" for record in records)
    geometry_rows = tuple(row for row in manifest.files if row["kind"] == "geometry")
    if len(geometry_rows) != 1 or geometry_rows[0]["sha256"] != _raw_digest(ndjson):
        raise ValueError("catalogue manifest geometry digest differs from record snapshot")
    return records


def bind_verified_catalogue(
    index: CatalogueIndex,
    manifest: CatalogueManifest,
    *,
    backend: object | None = None,
) -> VerifiedCatalogue:
    """Bind a full 230/1,731 catalogue to retained verified release evidence."""

    records = _validate_release_catalogue(index, manifest)
    return _make_verified_catalogue(
        CatalogueIndex(records),
        manifest_digest=_raw_digest(canonical_json(manifest)),
        release_complete=True,
        backend=backend,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class ResolvedOrbit:
    instance_id: str
    record: CatalogueRecord
    symbolic_parameters: tuple[str, ...]
    parameter_values: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if type(self.record) is not CatalogueRecord:
            raise TypeError("$ResolvedOrbit.record: expected CatalogueRecord")
        parameters = tuple(self.symbolic_parameters)
        values = tuple(Fraction(value) for value in self.parameter_values)
        dimension = self.record.orbit["parameter_dimension"]
        if (parameters and values) or len(parameters) + len(values) != dimension:
            raise ValueError("$ResolvedOrbit: parameter dimension differs from catalogue")
        if len(set(parameters)) != len(parameters):
            raise ValueError("$ResolvedOrbit.symbolic_parameters: duplicate parameter")
        object.__setattr__(self, "symbolic_parameters", parameters)
        object.__setattr__(self, "parameter_values", values)


def _verified_catalogue(value: object) -> VerifiedCatalogue:
    return verify_verified_catalogue(value)


def verify_verified_catalogue(value: object) -> VerifiedCatalogue:
    """Replay and return one exact factory-issued catalogue capability.

    Equality, a copied seal, or dataclass construction does not transfer the
    factory authority: the weak registry is bound to this object identity.
    """

    if type(value) is not VerifiedCatalogue:
        raise TypeError("query requires an exact verified catalogue authority")
    authority = _verified_catalogue_authority(value)
    if (
        type(authority) is not _VerifiedCatalogueAuthority
        or authority.factory_token is not _VERIFIED_CATALOGUE_FACTORY_TOKEN
    ):
        raise TypeError(
            "query requires the exact factory-issued verified catalogue authority"
        )
    value.__post_init__()
    return value


def _parameter_names(index: int, orbit: OrbitInstance, record: CatalogueRecord) -> tuple[str, ...]:
    safe_instance = re.sub(r"[^A-Za-z0-9_]", "_", orbit.instance_id)
    if not safe_instance or not safe_instance[0].isalpha():
        safe_instance = "x_" + safe_instance
    return tuple(
        f"i{index}_{safe_instance}_{name}"
        for name in record.orbit["parameter_names"]
    )


def resolve_request_orbits(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> tuple[ResolvedOrbit, ...]:
    if type(request) is not ClassificationRequest:
        raise TypeError("request must be ClassificationRequest")
    verified = _verified_catalogue(catalogue)
    result = []
    for index, orbit in enumerate(request.orbits):
        try:
            record = verified.index.find(request.space_group, orbit.wyckoff_id)
        except KeyError as error:
            raise KeyError(f"unknown canonical Wyckoff ID {orbit.wyckoff_id}") from error
        if record.space_group["setting"] != request.setting_id:
            raise ValueError("request and catalogue setting differ")
        dimension = record.orbit["parameter_dimension"]
        if orbit.parameter_mode == "family":
            if orbit.parameter_values:
                raise ValueError("family orbit may not carry point values")
            names = _parameter_names(index, orbit, record)
            values: tuple[Fraction, ...] = ()
        else:
            if len(orbit.parameter_values) != dimension:
                raise ValueError(
                    f"{orbit.instance_id}: point parameter dimension differs from catalogue"
                )
            names = ()
            values = tuple(orbit.parameter_values)
        result.append(ResolvedOrbit(orbit.instance_id, record, names, values))
    return tuple(result)


def classification_request_digest(request: ClassificationRequest) -> str:
    if type(request) is not ClassificationRequest:
        raise TypeError("request digest requires ClassificationRequest")
    return _digest(
        "classification-request",
        json.loads(canonical_classification_json(request)),
    )


def _routing_snapshot(
    routes: Sequence[InstanceParameterRoute],
    *,
    request_digest: str,
    catalogue_manifest_digest: str,
    space_group: int,
    setting_id: str,
) -> dict[str, object]:
    return {
        "catalogue_manifest_digest": catalogue_manifest_digest,
        "request_digest": request_digest,
        "routes": [
            json.loads(canonical_classification_json(route)) for route in routes
        ],
        "schema_version": SCHEMA_VERSION,
        "setting_id": setting_id,
        "space_group": space_group,
        "status": "parameter_specialization",
    }


def routing_result_digest(result: ParameterRoutingResult) -> str:
    if type(result) is not ParameterRoutingResult:
        raise TypeError("routing result digest requires ParameterRoutingResult")
    return _digest(
        "parameter-routing-result",
        _routing_snapshot(
            result.routes,
            request_digest=result.request_digest,
            catalogue_manifest_digest=result.catalogue_manifest_digest,
            space_group=result.space_group,
            setting_id=result.setting_id,
        ),
    )


def _candidate_evidence(
    *,
    instance_id: str,
    requested_wyckoff_id: str,
    exact_point: tuple[Fraction, ...],
    point_geometry: tuple[Vector, ...],
    point_stabilizer: tuple[Affine, ...],
    point_action_digest: str,
    candidate: CatalogueRecord,
    quotient: tuple[Affine, ...],
) -> CandidateGeometryEvidence:
    geometry_match, candidate_geometry = _candidate_geometry(
        candidate, point_geometry, quotient
    )
    candidate_literal = _literal_stabilizer(candidate)
    stabilizer_match = candidate_literal == point_stabilizer
    point_inclusion = _inclusion_elements(point_stabilizer, point_action_digest)
    candidate_inclusion = _inclusion_elements(
        candidate_literal, candidate.action_provenance_digest
    )
    inclusion_match = point_inclusion == candidate_inclusion
    point_geometry_digest = _geometry_digest(point_geometry)
    evaluated_candidate_digest = (
        _geometry_digest(candidate_geometry)
        if candidate_geometry is not None
        else _digest("candidate-family-orbit", candidate.orbit)
    )
    geometry_evidence = {
        "branch_bijection": list(range(len(point_geometry))) if geometry_match else None,
        "candidate_orbit_digest": evaluated_candidate_digest,
        "comparison": "match" if geometry_match else "mismatch",
        "lattice_shifts": (
            [["0", "0", "0"] for _ in point_geometry] if geometry_match else None
        ),
        "mismatch_witness": (
            None if geometry_match else {"reason": "exact-orbit-not-in-candidate-family"}
        ),
        "point_orbit_digest": point_geometry_digest,
    }
    stabilizer_evidence = {
        "candidate_elements": _stabilizer_elements(candidate_literal),
        "comparison": "match" if stabilizer_match else "mismatch",
        "comparison_witness": (
            {"kind": "literal-affine-equality"} if stabilizer_match else None
        ),
        "element_bijection": (
            list(range(len(point_stabilizer))) if stabilizer_match else None
        ),
        "mismatch_witness": (
            None
            if stabilizer_match
            else {"reason": "literal-embedded-stabilizer-differs"}
        ),
        "point_elements": _stabilizer_elements(point_stabilizer),
    }
    inclusion_evidence = {
        "candidate_transported_elements": candidate_inclusion,
        "comparison": "match" if inclusion_match else "mismatch",
        "comparison_witness": (
            {"kind": "transported-literal-equality"} if inclusion_match else None
        ),
        "element_bijection": (
            list(range(len(point_inclusion))) if inclusion_match else None
        ),
        "mismatch_witness": (
            None
            if inclusion_match
            else {"reason": "transported-literal-inclusion-differs"}
        ),
        "point_transported_elements": point_inclusion,
    }
    provisional = CandidateGeometryEvidence(
        candidate_wyckoff_id=candidate.wyckoff_id,
        family_geometry_digest=_digest("catalogue-family-geometry", candidate.orbit),
        literal_stabilizer_digest=_stabilizer_digest(candidate_literal),
        transported_inclusion_digest=_inclusion_digest(
            candidate_literal, candidate.action_provenance_digest
        ),
        geometry_evidence=geometry_evidence,
        stabilizer_evidence=stabilizer_evidence,
        inclusion_evidence=inclusion_evidence,
        geometry_comparison_digest="sha256:" + "0" * 64,
        stabilizer_comparison_digest="sha256:" + "0" * 64,
        inclusion_comparison_digest="sha256:" + "0" * 64,
        geometry_match=geometry_match,
        stabilizer_match=stabilizer_match,
        inclusion_match=inclusion_match,
        rejection_codes=tuple(
            name
            for name, matched in (
                ("geometry", geometry_match),
                ("stabilizer", stabilizer_match),
                ("inclusion", inclusion_match),
            )
            if not matched
        ),
    )
    route_view = SimpleNamespace(
        instance_id=instance_id,
        requested_wyckoff_id=requested_wyckoff_id,
        exact_point=exact_point,
        point_geometry_digest=point_geometry_digest,
        point_stabilizer_digest=_stabilizer_digest(point_stabilizer),
        point_inclusion_digest=_inclusion_digest(point_stabilizer, point_action_digest),
    )
    return replace(
        provisional,
        geometry_comparison_digest=_comparison_digest(
            route_view, provisional, "geometry"
        ),
        stabilizer_comparison_digest=_comparison_digest(
            route_view, provisional, "stabilizer"
        ),
        inclusion_comparison_digest=_comparison_digest(
            route_view, provisional, "inclusion"
        ),
    )


def _route(
    orbit: OrbitInstance,
    resolved: ResolvedOrbit,
    catalogue: VerifiedCatalogue,
) -> InstanceParameterRoute:
    candidates = catalogue.candidate_records(
        resolved.record.space_group["international_number"],
        resolved.record.space_group["setting"],
    )
    if not candidates:
        raise ValueError("catalogue candidate universe is empty")
    quotient = _finite_affine_quotient(resolved.record)
    point = _primitive_point(
        resolved.record,
        _specialized_reference(resolved.record, resolved.parameter_values),
    )
    geometry = _point_geometry(quotient, point)
    stabilizer = _stabilizer(quotient, point)
    evidence = tuple(
        _candidate_evidence(
            instance_id=orbit.instance_id,
            requested_wyckoff_id=orbit.wyckoff_id,
            exact_point=resolved.parameter_values,
            point_geometry=geometry,
            point_stabilizer=stabilizer,
            point_action_digest=resolved.record.action_provenance_digest,
            candidate=candidate,
            quotient=quotient,
        )
        for candidate in candidates
    )
    matches = tuple(
        candidate.candidate_wyckoff_id
        for candidate in evidence
        if candidate.geometry_match
        and candidate.stabilizer_match
        and candidate.inclusion_match
    )
    if matches == (orbit.wyckoff_id,):
        outcome = "same_stratum"
        resolved_id: str | None = orbit.wyckoff_id
    elif len(matches) == 1:
        outcome = "resolved_specialization"
        resolved_id = matches[0]
    else:
        outcome = "unresolved_specialization"
        resolved_id = None
    return InstanceParameterRoute(
        instance_id=orbit.instance_id,
        requested_wyckoff_id=orbit.wyckoff_id,
        exact_point=resolved.parameter_values,
        outcome=outcome,
        resolved_wyckoff_id=resolved_id,
        point_geometry_digest=_geometry_digest(geometry),
        point_stabilizer_digest=_stabilizer_digest(stabilizer),
        point_inclusion_digest=_inclusion_digest(
            stabilizer, resolved.record.action_provenance_digest
        ),
        candidate_set_digest=_candidate_tuple_digest(evidence),
        candidates=evidence,
    )


def parameter_routes(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> tuple[InstanceParameterRoute, ...]:
    verified = _verified_catalogue(catalogue)
    resolved = resolve_request_orbits(request, verified)
    routes = tuple(
        _route(orbit, item, verified)
        for orbit, item in zip(request.orbits, resolved, strict=True)
        if orbit.parameter_mode == "point"
    )
    if not routes:
        raise ValueError("parameter routing requires at least one point-mode orbit")
    return routes


def build_parameter_routing(
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> ParameterRoutingResult:
    verified = _verified_catalogue(catalogue)
    routes = parameter_routes(request, verified)
    if all(route.outcome == "same_stratum" for route in routes):
        raise ValueError(
            "all same-stratum point routes belong in CertifiedClassification"
        )
    return ParameterRoutingResult(
        "parameter_specialization",
        classification_request_digest(request),
        verified.catalogue_manifest_digest,
        request.space_group,
        request.setting_id,
        routes,
    )


def _comparison_evidence_digest(routes: Sequence[InstanceParameterRoute]) -> str:
    return _digest(
        "routing-comparison-evidence",
        [json.loads(canonical_classification_json(route)) for route in routes],
    )


@dataclass(frozen=True, slots=True)
class RoutingVerification:
    result_digest: str
    request_digest: str
    catalogue_manifest_digest: str
    space_group: int
    setting_id: str
    point_instance_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_set_digest: str
    comparison_evidence_digest: str
    certificate_digest: str

    def __post_init__(self) -> None:
        for name in (
            "result_digest",
            "request_digest",
            "catalogue_manifest_digest",
            "candidate_set_digest",
            "comparison_evidence_digest",
            "certificate_digest",
        ):
            _require_digest(getattr(self, name), f"$RoutingVerification.{name}")
        points = tuple(self.point_instance_ids)
        candidates = tuple(self.candidate_ids)
        if len(set(points)) != len(points) or not points:
            raise ValueError("$RoutingVerification.point_instance_ids: expected unique tuple")
        if len(set(candidates)) != len(candidates) or not candidates:
            raise ValueError("$RoutingVerification.candidate_ids: expected unique tuple")
        expected = _digest(
            "routing-verification",
            {
                "candidate_ids": list(candidates),
                "candidate_set_digest": self.candidate_set_digest,
                "catalogue_manifest_digest": self.catalogue_manifest_digest,
                "comparison_evidence_digest": self.comparison_evidence_digest,
                "point_instance_ids": list(points),
                "request_digest": self.request_digest,
                "result_digest": self.result_digest,
                "setting_id": self.setting_id,
                "space_group": self.space_group,
            },
        )
        if self.certificate_digest != expected:
            raise ValueError("$RoutingVerification.certificate_digest: payload differs")
        object.__setattr__(self, "point_instance_ids", points)
        object.__setattr__(self, "candidate_ids", candidates)


def _make_routes_verification(
    routes: Sequence[InstanceParameterRoute],
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
    *,
    result_digest: str,
) -> RoutingVerification:
    candidate_ids = catalogue.candidate_ids(request.space_group, request.setting_id)
    point_ids = tuple(
        orbit.instance_id for orbit in request.orbits if orbit.parameter_mode == "point"
    )
    route_tuple = tuple(routes)
    candidate_set_digest = _candidate_tuple_digest(route_tuple[0].candidates)
    request_digest = classification_request_digest(request)
    evidence_digest = _comparison_evidence_digest(route_tuple)
    core = {
        "candidate_ids": list(candidate_ids),
        "candidate_set_digest": candidate_set_digest,
        "catalogue_manifest_digest": catalogue.catalogue_manifest_digest,
        "comparison_evidence_digest": evidence_digest,
        "point_instance_ids": list(point_ids),
        "request_digest": request_digest,
        "result_digest": result_digest,
        "setting_id": request.setting_id,
        "space_group": request.space_group,
    }
    return RoutingVerification(
        result_digest,
        request_digest,
        catalogue.catalogue_manifest_digest,
        request.space_group,
        request.setting_id,
        point_ids,
        candidate_ids,
        candidate_set_digest,
        evidence_digest,
        _digest("routing-verification", core),
    )


def verify_parameter_routing(
    result: ParameterRoutingResult,
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> RoutingVerification:
    if type(result) is not ParameterRoutingResult:
        raise TypeError("routing verification requires ParameterRoutingResult")
    if type(request) is not ClassificationRequest:
        raise TypeError("routing verification requires ClassificationRequest")
    verified = _verified_catalogue(catalogue)
    # Replay the structural constructor to close in-memory mutation bypasses.
    try:
        checked = ParameterRoutingResult(
            result.status,
            result.request_digest,
            result.catalogue_manifest_digest,
            result.space_group,
            result.setting_id,
            result.routes,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"routing candidate structure is invalid: {error}") from error
    if checked.request_digest != classification_request_digest(request):
        raise ValueError("routing request binding differs")
    if checked.catalogue_manifest_digest != verified.catalogue_manifest_digest:
        raise ValueError("routing catalogue manifest binding differs")
    if checked.space_group != request.space_group:
        raise ValueError("routing space-group binding differs")
    if checked.setting_id != request.setting_id:
        raise ValueError("routing setting binding differs")
    point_ids = tuple(
        orbit.instance_id for orbit in request.orbits if orbit.parameter_mode == "point"
    )
    actual_ids = tuple(route.instance_id for route in checked.routes)
    if actual_ids != point_ids:
        raise ValueError("point route order or completeness differs from request")
    candidate_ids = verified.candidate_ids(request.space_group, request.setting_id)
    for route in checked.routes:
        actual_candidates = tuple(
            candidate.candidate_wyckoff_id for candidate in route.candidates
        )
        if actual_candidates != candidate_ids:
            raise ValueError("candidate sequence is incomplete or reordered")
    expected_routes = parameter_routes(request, verified)
    if checked.routes != expected_routes:
        raise ValueError("routing comparison evidence differs from exact recomputation")
    return _make_routes_verification(
        checked.routes,
        request,
        verified,
        result_digest=routing_result_digest(checked),
    )


def verify_same_stratum_routes(
    routes: Sequence[InstanceParameterRoute],
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> RoutingVerification:
    """Replay all-same point routes before embedding them in a classification.

    Task 1 intentionally forbids serializing an all-same
    :class:`ParameterRoutingResult`.  This verifier uses the identical canonical
    route envelope and semantic recomputation without weakening that boundary.
    """

    if type(request) is not ClassificationRequest:
        raise TypeError("routing verification requires ClassificationRequest")
    verified = _verified_catalogue(catalogue)
    checked_routes: list[InstanceParameterRoute] = []
    for route in tuple(routes):
        if type(route) is not InstanceParameterRoute:
            raise TypeError("same-stratum routing requires InstanceParameterRoute records")
        try:
            checked_routes.append(
                InstanceParameterRoute(
                    route.instance_id,
                    route.requested_wyckoff_id,
                    route.exact_point,
                    route.outcome,
                    route.resolved_wyckoff_id,
                    route.point_geometry_digest,
                    route.point_stabilizer_digest,
                    route.point_inclusion_digest,
                    route.candidate_set_digest,
                    route.candidates,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"routing candidate structure is invalid: {error}") from error
    checked = tuple(checked_routes)
    point_ids = tuple(
        orbit.instance_id for orbit in request.orbits if orbit.parameter_mode == "point"
    )
    if not checked or tuple(route.instance_id for route in checked) != point_ids:
        raise ValueError("point route order or completeness differs from request")
    if any(route.outcome != "same_stratum" for route in checked):
        raise ValueError("embedded point routes must all be same_stratum")
    candidate_ids = verified.candidate_ids(request.space_group, request.setting_id)
    for route in checked:
        if tuple(candidate.candidate_wyckoff_id for candidate in route.candidates) != candidate_ids:
            raise ValueError("candidate sequence is incomplete or reordered")
    expected = parameter_routes(request, verified)
    if checked != expected:
        raise ValueError("routing comparison evidence differs from exact recomputation")
    request_digest = classification_request_digest(request)
    structural_digest = _digest(
        "parameter-routing-result",
        _routing_snapshot(
            checked,
            request_digest=request_digest,
            catalogue_manifest_digest=verified.catalogue_manifest_digest,
            space_group=request.space_group,
            setting_id=request.setting_id,
        ),
    )
    return _make_routes_verification(
        checked,
        request,
        verified,
        result_digest=structural_digest,
    )


def _group_action_digest(records: Sequence[CatalogueRecord]) -> str:
    return _digest(
        "candidate-group-action-set",
        [
            {
                "action": record.space_group_action,
                "action_provenance_digest": record.action_provenance_digest,
                "wyckoff_id": record.wyckoff_id,
            }
            for record in records
        ],
    )


def _routing_verification_cache_key(
    structural_result_digest: str,
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> CacheKey:
    verified = _verified_catalogue(catalogue)
    records = verified.candidate_records(request.space_group, request.setting_id)
    dependencies = {
        "affine_transport": _digest(
            "candidate-affine-transport-set",
            [
                {
                    "action_provenance_digest": record.action_provenance_digest,
                    "stabilizer": record.stabilizer,
                    "wyckoff_id": record.wyckoff_id,
                }
                for record in records
            ],
        ),
        "candidate_record_set": verified.candidate_record_set_digest(
            request.space_group, request.setting_id
        ),
        "catalogue_manifest": verified.catalogue_manifest_digest,
        "comparison_algorithm": ROUTING_ALGORITHM_DIGEST,
        "geometry_normalization": _digest(
            "geometry-normalization-algorithm", {"version": 1}
        ),
        "group_action": _group_action_digest(records),
        "group_setting": _digest(
            "group-setting",
            {"setting_id": request.setting_id, "space_group": request.space_group},
        ),
        "request": classification_request_digest(request),
        "routing_schema": _digest("routing-schema", {"version": SCHEMA_VERSION}),
        "structural_result": structural_result_digest,
        "verifier_library": ROUTING_VERIFIER_LIBRARY_DIGEST,
    }
    return CacheKey(
        "routing-verification",
        1,
        ROUTING_ALGORITHM_DIGEST,
        tuple(sorted(dependencies.items())),
    )


def routing_verification_cache_key(
    result: ParameterRoutingResult,
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> CacheKey:
    return _routing_verification_cache_key(
        routing_result_digest(result), request, catalogue
    )


def same_stratum_routing_verification_cache_key(
    routes: Sequence[InstanceParameterRoute],
    request: ClassificationRequest,
    catalogue: VerifiedCatalogue,
) -> CacheKey:
    verification = verify_same_stratum_routes(routes, request, catalogue)
    return _routing_verification_cache_key(
        verification.result_digest, request, catalogue
    )


__all__ = [
    "ROUTING_ALGORITHM_DIGEST",
    "ROUTING_VERIFIER_LIBRARY_DIGEST",
    "ResolvedOrbit",
    "RoutingVerification",
    "VerifiedCatalogue",
    "bind_verified_catalogue",
    "build_parameter_routing",
    "classification_request_digest",
    "make_diagnostic_verified_catalogue",
    "parameter_routes",
    "resolve_request_orbits",
    "routing_result_digest",
    "same_stratum_routing_verification_cache_key",
    "routing_verification_cache_key",
    "verify_verified_catalogue",
    "verify_parameter_routing",
    "verify_same_stratum_routes",
]
