r"""Certified shifted mapping cones for joint finite-orbit PSG problems.

For an ambient-to-local cochain map ``r`` this module uses the shifted cone

``C_rel^n = C_ambient^n + direct_sum_alpha C_local,alpha^(n-1)``

with differential ``(x, y) -> (d x, r x - d y)``.  All orbit instances are
assembled in one matrix: the ambient columns occur once and every local
restriction meets those same columns.  This is therefore not equivalent to a
Cartesian product of independently solved one-orbit summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Literal, Sequence, TypeAlias

from .cochains import CochainComplex, CochainMap, verify_cochain_map
from .gf2 import MatrixGF2
from .integer_linalg import MatrixZ
from .torus import Phase


Matrix: TypeAlias = MatrixGF2 | MatrixZ
ExactCoefficient: TypeAlias = int | Phase

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_PROTOCOL = b"mathpsg-relative-complex-v1|"


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


def _require_instance_ids(value: object, path: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{path}: expected sorted unique tuple")
    for index, instance_id in enumerate(value):
        if type(instance_id) is not str or _IDENTIFIER_RE.fullmatch(instance_id) is None:
            raise ValueError(f"{path}[{index}]: invalid identifier")
    if value != tuple(sorted(value)) or len(set(value)) != len(value):
        raise ValueError(f"{path}: expected sorted unique tuple")
    return value


def _matrix_mapping(matrix: Matrix) -> dict[str, object]:
    return {
        "column_count": matrix.column_count,
        "ring": "gf2" if type(matrix) is MatrixGF2 else "z",
        "rows": [list(row) for row in matrix],
    }


def _phase_text(value: Phase) -> str:
    fraction = value.value
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _coefficient_mapping(value: ExactCoefficient) -> int | str:
    if type(value) is int:
        return value
    if type(value) is Phase:
        return _phase_text(value)
    raise TypeError("relative offset contains an unsupported coefficient")


def _matrix_digest(matrix: Matrix) -> str:
    return _digest("matrix", _matrix_mapping(matrix))


def _cochain_map_digest(value: CochainMap) -> str:
    return _digest(
        "cochain-map",
        {
            "instance_id": value.instance_id,
            "maps": [_matrix_mapping(matrix) for matrix in value.maps],
            "source_id": value.source_id,
            "target_id": value.target_id,
        },
    )


@dataclass(frozen=True, slots=True)
class RelativeProblem:
    ring: Literal["gf2", "torus"]
    ambient: CochainComplex
    locals: tuple[CochainComplex, ...]
    restrictions: tuple[CochainMap, ...]
    local_defects: tuple[tuple[ExactCoefficient, ...], ...]

    def __post_init__(self) -> None:
        if self.ring not in ("gf2", "torus"):
            raise ValueError("$RelativeProblem.ring: expected gf2 or torus")
        if type(self.ambient) is not CochainComplex:
            raise TypeError("$RelativeProblem.ambient: expected CochainComplex")
        locals_ = tuple(self.locals)
        restrictions = tuple(self.restrictions)
        defects = tuple(tuple(defect) for defect in self.local_defects)
        if not locals_:
            raise ValueError("$RelativeProblem.locals: expected at least one orbit")
        if not (len(locals_) == len(restrictions) == len(defects)):
            raise ValueError("$RelativeProblem: local/map/defect counts differ")
        if any(type(value) is not CochainComplex for value in locals_):
            raise TypeError("$RelativeProblem.locals: expected CochainComplex values")
        if any(type(value) is not CochainMap for value in restrictions):
            raise TypeError("$RelativeProblem.restrictions: expected CochainMap values")
        object.__setattr__(self, "locals", locals_)
        object.__setattr__(self, "restrictions", restrictions)
        object.__setattr__(self, "local_defects", defects)


@dataclass(frozen=True, slots=True)
class RelativeCoordinateBlocks:
    instance_ids: tuple[str, ...]
    ambient_slices: tuple[tuple[int, int], ...]
    local_slices: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        instances = _require_instance_ids(
            self.instance_ids,
            "$RelativeCoordinateBlocks.instance_ids",
        )
        ambient = tuple(tuple(value) for value in self.ambient_slices)
        local = tuple(
            tuple(tuple(value) for value in degree) for degree in self.local_slices
        )
        if len(ambient) != 4 or len(local) != 4:
            raise ValueError("$RelativeCoordinateBlocks: expected relative degrees one through four")
        for degree in range(4):
            if len(ambient[degree]) != 2 or len(local[degree]) != len(instances):
                raise ValueError("$RelativeCoordinateBlocks: malformed degree slices")
            slices = (ambient[degree],) + local[degree]
            cursor = 0
            for start, stop in slices:
                if type(start) is not int or type(stop) is not int or start != cursor or stop < start:
                    raise ValueError("$RelativeCoordinateBlocks: slices are not contiguous")
                cursor = stop
        object.__setattr__(self, "instance_ids", instances)
        object.__setattr__(self, "ambient_slices", ambient)
        object.__setattr__(self, "local_slices", local)


def _blocks_mapping(value: RelativeCoordinateBlocks) -> dict[str, object]:
    return {
        "ambient_slices": [list(item) for item in value.ambient_slices],
        "instance_ids": list(value.instance_ids),
        "local_slices": [
            [list(item) for item in degree] for degree in value.local_slices
        ],
    }


@dataclass(frozen=True, slots=True)
class RelativeIdentityCertificate:
    certificate_id: str
    ring: Literal["gf2", "torus"]
    problem_digest: str
    ambient_complex_id: str
    instance_ids: tuple[str, ...]
    local_complex_ids: tuple[str, ...]
    restriction_digests: tuple[str, ...]
    defect_digests: tuple[str, ...]
    matrix_digest: str
    offset_digest: str
    coordinate_blocks_digest: str
    db_zero_witness_digest: str
    ed_zero_witness_digest: str
    eb_zero_witness_digest: str

    def __post_init__(self) -> None:
        if self.ring not in ("gf2", "torus"):
            raise ValueError("$RelativeIdentityCertificate.ring: invalid ring")
        for name in (
            "certificate_id",
            "problem_digest",
            "ambient_complex_id",
            "matrix_digest",
            "offset_digest",
            "coordinate_blocks_digest",
            "db_zero_witness_digest",
            "ed_zero_witness_digest",
            "eb_zero_witness_digest",
        ):
            _require_digest(getattr(self, name), f"$RelativeIdentityCertificate.{name}")
        instances = _require_instance_ids(
            self.instance_ids,
            "$RelativeIdentityCertificate.instance_ids",
        )
        local_ids = tuple(self.local_complex_ids)
        restriction_ids = tuple(self.restriction_digests)
        defect_ids = tuple(self.defect_digests)
        if not (len(instances) == len(local_ids) == len(restriction_ids) == len(defect_ids)):
            raise ValueError("$RelativeIdentityCertificate: source binding lengths differ")
        for values, name in (
            (local_ids, "local_complex_ids"),
            (restriction_ids, "restriction_digests"),
            (defect_ids, "defect_digests"),
        ):
            for index, value in enumerate(values):
                _require_digest(value, f"$RelativeIdentityCertificate.{name}[{index}]")
        object.__setattr__(self, "instance_ids", instances)
        object.__setattr__(self, "local_complex_ids", local_ids)
        object.__setattr__(self, "restriction_digests", restriction_ids)
        object.__setattr__(self, "defect_digests", defect_ids)
        expected = _digest("relative-certificate", _certificate_core(self))
        if self.certificate_id != expected:
            raise ValueError("$RelativeIdentityCertificate.certificate_id: does not bind certificate")


def _certificate_core(value: RelativeIdentityCertificate) -> dict[str, object]:
    return {
        "ambient_complex_id": value.ambient_complex_id,
        "coordinate_blocks_digest": value.coordinate_blocks_digest,
        "db_zero_witness_digest": value.db_zero_witness_digest,
        "defect_digests": list(value.defect_digests),
        "eb_zero_witness_digest": value.eb_zero_witness_digest,
        "ed_zero_witness_digest": value.ed_zero_witness_digest,
        "instance_ids": list(value.instance_ids),
        "local_complex_ids": list(value.local_complex_ids),
        "matrix_digest": value.matrix_digest,
        "offset_digest": value.offset_digest,
        "problem_digest": value.problem_digest,
        "restriction_digests": list(value.restriction_digests),
        "ring": value.ring,
    }


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    if type(left) is not type(right):
        raise TypeError("relative matrix coefficient rings differ")
    if left.column_count != right.row_count:
        raise ValueError("relative matrix dimensions differ")
    modulus = 2 if type(left) is MatrixGF2 else None
    rows = tuple(
        tuple(
            (
                sum(
                    left[row][middle] * right[middle][column]
                    for middle in range(left.column_count)
                )
                % modulus
                if modulus is not None
                else sum(
                    left[row][middle] * right[middle][column]
                    for middle in range(left.column_count)
                )
            )
            for column in range(right.column_count)
        )
        for row in range(left.row_count)
    )
    constructor = MatrixGF2 if type(left) is MatrixGF2 else MatrixZ
    return constructor(rows, column_count=right.column_count)


def _matvec(matrix: Matrix, vector: Sequence[ExactCoefficient]) -> tuple[ExactCoefficient, ...]:
    if matrix.column_count != len(vector):
        raise ValueError("relative matrix/vector dimensions differ")
    if type(matrix) is MatrixGF2:
        if any(type(value) is not int or value not in (0, 1) for value in vector):
            raise TypeError("GF(2) relative offsets must contain exact bits")
        return tuple(
            sum(matrix[row][column] * vector[column] for column in range(matrix.column_count)) & 1
            for row in range(matrix.row_count)
        )
    if any(type(value) is not Phase for value in vector):
        raise TypeError("torus relative offsets must contain exact Phase values")
    return tuple(
        Phase(
            sum(
                (
                    matrix[row][column] * vector[column].value  # type: ignore[union-attr]
                    for column in range(matrix.column_count)
                ),
                Fraction(0),
            )
        )
        for row in range(matrix.row_count)
    )


def _matrix_is_zero(matrix: Matrix) -> bool:
    return all(entry == 0 for row in matrix for entry in row)


def _vector_is_zero(vector: Sequence[ExactCoefficient]) -> bool:
    return all(
        (value == 0 if type(value) is int else value.value == 0)
        for value in vector
    )


def _zero_matrix_witness(domain: str, left: Matrix, right: Matrix, product: Matrix) -> str:
    if not _matrix_is_zero(product):
        raise ValueError(f"chain_identity_failed: {domain} is nonzero")
    return _digest(
        f"{domain}-zero-witness",
        {
            "left": _matrix_digest(left),
            "product": _matrix_mapping(product),
            "right": _matrix_digest(right),
        },
    )


def _zero_vector_witness(domain: str, matrix: Matrix, offset: Sequence[ExactCoefficient], product: Sequence[ExactCoefficient]) -> str:
    if not _vector_is_zero(product):
        raise ValueError(f"chain_identity_failed: {domain} is nonzero")
    return _digest(
        f"{domain}-zero-witness",
        {
            "matrix": _matrix_digest(matrix),
            "offset": [_coefficient_mapping(value) for value in offset],
            "product": [_coefficient_mapping(value) for value in product],
        },
    )


@dataclass(frozen=True, slots=True)
class RelativeMatrices:
    B: Matrix
    D: Matrix
    E: Matrix
    offset: tuple[ExactCoefficient, ...]
    coordinate_blocks: RelativeCoordinateBlocks
    certificate: RelativeIdentityCertificate

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_blocks, RelativeCoordinateBlocks):
            raise TypeError("$RelativeMatrices.coordinate_blocks: invalid type")
        if not isinstance(self.certificate, RelativeIdentityCertificate):
            raise TypeError("$RelativeMatrices.certificate: invalid type")
        blocks = RelativeCoordinateBlocks(
            self.coordinate_blocks.instance_ids,
            self.coordinate_blocks.ambient_slices,
            self.coordinate_blocks.local_slices,
        )
        certificate = RelativeIdentityCertificate(
            self.certificate.certificate_id,
            self.certificate.ring,
            self.certificate.problem_digest,
            self.certificate.ambient_complex_id,
            self.certificate.instance_ids,
            self.certificate.local_complex_ids,
            self.certificate.restriction_digests,
            self.certificate.defect_digests,
            self.certificate.matrix_digest,
            self.certificate.offset_digest,
            self.certificate.coordinate_blocks_digest,
            self.certificate.db_zero_witness_digest,
            self.certificate.ed_zero_witness_digest,
            self.certificate.eb_zero_witness_digest,
        )
        if certificate.instance_ids != blocks.instance_ids:
            raise ValueError("$RelativeMatrices.certificate: instance IDs differ from blocks")
        if type(self.B) is not type(self.D) or type(self.B) is not type(self.E):
            raise TypeError("$RelativeMatrices: coefficient rings differ")
        if type(self.B) not in (MatrixGF2, MatrixZ):
            raise TypeError("$RelativeMatrices: expected exact shaped matrices")
        offset = tuple(self.offset)
        expected_ring = "gf2" if type(self.B) is MatrixGF2 else "torus"
        if certificate.ring != expected_ring:
            raise ValueError("$RelativeMatrices: certificate ring differs")
        if self.D.column_count != self.B.row_count or self.E.column_count != self.D.row_count:
            raise ValueError("$RelativeMatrices: adjacent matrix dimensions differ")
        if len(offset) != self.D.row_count:
            raise ValueError("$RelativeMatrices.offset: dimension differs from D target")
        if expected_ring == "gf2":
            if any(type(value) is not int or value not in (0, 1) for value in offset):
                raise TypeError("$RelativeMatrices.offset: expected GF(2) bits")
        elif any(type(value) is not Phase for value in offset):
            raise TypeError("$RelativeMatrices.offset: expected Phase values")
        dimensions = (
            self.B.column_count,
            self.B.row_count,
            self.D.row_count,
            self.E.row_count,
        )
        for degree, dimension in enumerate(dimensions):
            final_slice = blocks.local_slices[degree][-1]
            if final_slice[1] != dimension:
                raise ValueError("$RelativeMatrices.coordinate_blocks: dimensions differ")
        matrix_digest = _digest(
            "relative-matrices",
            {"B": _matrix_mapping(self.B), "D": _matrix_mapping(self.D), "E": _matrix_mapping(self.E)},
        )
        offset_digest = _digest(
            "relative-offset", [_coefficient_mapping(value) for value in offset]
        )
        blocks_digest = _digest("relative-coordinate-blocks", _blocks_mapping(blocks))
        if matrix_digest != certificate.matrix_digest:
            raise ValueError("$RelativeMatrices.certificate: matrix digest mismatch")
        if offset_digest != certificate.offset_digest:
            raise ValueError("$RelativeMatrices.certificate: offset digest mismatch")
        if blocks_digest != certificate.coordinate_blocks_digest:
            raise ValueError("$RelativeMatrices.certificate: coordinate-block digest mismatch")
        db = _matmul(self.D, self.B)
        ed = _matmul(self.E, self.D)
        eb = _matvec(self.E, offset)
        if _zero_matrix_witness("DB", self.D, self.B, db) != certificate.db_zero_witness_digest:
            raise ValueError("$RelativeMatrices.certificate: DB witness mismatch")
        if _zero_matrix_witness("ED", self.E, self.D, ed) != certificate.ed_zero_witness_digest:
            raise ValueError("$RelativeMatrices.certificate: ED witness mismatch")
        if _zero_vector_witness("E-offset", self.E, offset, eb) != certificate.eb_zero_witness_digest:
            raise ValueError("$RelativeMatrices.certificate: E@offset witness mismatch")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "coordinate_blocks", blocks)
        object.__setattr__(self, "certificate", certificate)


def _revalidated_complex(value: CochainComplex) -> CochainComplex:
    return CochainComplex(
        value.complex_id,
        value.authority_id,
        value.dimensions,
        value.differentials,
        value.coefficient_character,
    )


def _revalidated_map(value: CochainMap) -> CochainMap:
    return CochainMap(value.instance_id, value.source_id, value.target_id, value.maps)


def _sorted_problem_blocks(problem: RelativeProblem) -> tuple[tuple[CochainComplex, CochainMap, tuple[ExactCoefficient, ...]], ...]:
    ambient = _revalidated_complex(problem.ambient)
    blocks = tuple(
        (
            _revalidated_complex(local),
            _revalidated_map(restriction),
            tuple(defect),
        )
        for local, restriction, defect in zip(
            problem.locals,
            problem.restrictions,
            problem.local_defects,
            strict=True,
        )
    )
    instance_ids = tuple(block[1].instance_id for block in blocks)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("duplicate relative orbit instance_id")
    expected_type = MatrixGF2 if problem.ring == "gf2" else MatrixZ
    if any(type(matrix) is not expected_type for matrix in ambient.differentials):
        raise TypeError("relative problem ring differs from ambient complex")
    checked: list[tuple[CochainComplex, CochainMap, tuple[ExactCoefficient, ...]]] = []
    for local, restriction, defect in blocks:
        if any(type(matrix) is not expected_type for matrix in local.differentials):
            raise TypeError(f"relative problem ring differs for {restriction.instance_id}")
        report = verify_cochain_map(restriction, ambient, local)
        if not report.valid:
            raise ValueError(
                "chain_identity_failed: "
                f"{restriction.instance_id}: {report.issues[0].code}"
            )
        if len(defect) != local.dimensions[2]:
            raise ValueError(f"{restriction.instance_id}: local defect has wrong dimension")
        if problem.ring == "gf2":
            if any(type(value) is not int or value not in (0, 1) for value in defect):
                raise TypeError(f"{restriction.instance_id}: GF(2) defect must contain bits")
        elif any(type(value) is not Phase for value in defect):
            raise TypeError(f"{restriction.instance_id}: torus defect must contain Phase values")
        checked.append((local, restriction, defect))
    return tuple(sorted(checked, key=lambda item: item[1].instance_id))


def _coordinate_blocks(ambient: CochainComplex, blocks: Sequence[tuple[CochainComplex, CochainMap, tuple[ExactCoefficient, ...]]]) -> RelativeCoordinateBlocks:
    ambient_slices: list[tuple[int, int]] = []
    local_slices: list[tuple[tuple[int, int], ...]] = []
    for relative_degree in range(1, 5):
        ambient_stop = ambient.dimensions[relative_degree]
        ambient_slices.append((0, ambient_stop))
        cursor = ambient_stop
        degree_slices: list[tuple[int, int]] = []
        for local, _, _ in blocks:
            stop = cursor + local.dimensions[relative_degree - 1]
            degree_slices.append((cursor, stop))
            cursor = stop
        local_slices.append(tuple(degree_slices))
    return RelativeCoordinateBlocks(
        tuple(item[1].instance_id for item in blocks),
        tuple(ambient_slices),
        tuple(local_slices),
    )


def _relative_differential(
    ring: Literal["gf2", "torus"],
    ambient: CochainComplex,
    blocks: Sequence[tuple[CochainComplex, CochainMap, tuple[ExactCoefficient, ...]]],
    coordinates: RelativeCoordinateBlocks,
    degree: int,
) -> Matrix:
    source_dimension = coordinates.local_slices[degree - 1][-1][1]
    target_dimension = coordinates.local_slices[degree][-1][1]
    dense = [[0] * source_dimension for _ in range(target_dimension)]
    ambient_differential = ambient.differentials[degree]
    for row in range(ambient_differential.row_count):
        for column in range(ambient_differential.column_count):
            dense[row][column] = ambient_differential[row][column]
    for index, (local, restriction, _) in enumerate(blocks):
        target_start, _ = coordinates.local_slices[degree][index]
        source_start, _ = coordinates.local_slices[degree - 1][index]
        restricted = restriction.maps[degree]
        for row in range(restricted.row_count):
            for column in range(restricted.column_count):
                dense[target_start + row][column] = restricted[row][column]
        local_differential = local.differentials[degree - 1]
        for row in range(local_differential.row_count):
            for column in range(local_differential.column_count):
                value = -local_differential[row][column]
                dense[target_start + row][source_start + column] = (
                    value & 1 if ring == "gf2" else value
                )
    rows = tuple(tuple(row) for row in dense)
    constructor = MatrixGF2 if ring == "gf2" else MatrixZ
    return constructor(rows, column_count=source_dimension)


def _problem_bindings(
    problem: RelativeProblem,
    ambient: CochainComplex,
    blocks: Sequence[tuple[CochainComplex, CochainMap, tuple[ExactCoefficient, ...]]],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    local_ids = tuple(item[0].complex_id for item in blocks)
    restriction_ids = tuple(_cochain_map_digest(item[1]) for item in blocks)
    defect_ids = tuple(
        _digest(
            "local-defect",
            [_coefficient_mapping(value) for value in item[2]],
        )
        for item in blocks
    )
    problem_digest = _digest(
        "relative-problem",
        {
            "ambient_complex_id": ambient.complex_id,
            "instances": [
                {
                    "defect_digest": defect_id,
                    "instance_id": block[1].instance_id,
                    "local_complex_id": local_id,
                    "restriction_digest": restriction_id,
                }
                for block, local_id, restriction_id, defect_id in zip(
                    blocks, local_ids, restriction_ids, defect_ids, strict=True
                )
            ],
            "ring": problem.ring,
        },
    )
    return problem_digest, local_ids, restriction_ids, defect_ids


def assemble_relative_problem(problem: RelativeProblem) -> RelativeMatrices:
    if type(problem) is not RelativeProblem:
        raise TypeError("assemble_relative_problem requires RelativeProblem")
    ambient = _revalidated_complex(problem.ambient)
    blocks = _sorted_problem_blocks(problem)
    coordinates = _coordinate_blocks(ambient, blocks)
    B = _relative_differential(problem.ring, ambient, blocks, coordinates, 1)
    D = _relative_differential(problem.ring, ambient, blocks, coordinates, 2)
    E = _relative_differential(problem.ring, ambient, blocks, coordinates, 3)
    zero: ExactCoefficient = 0 if problem.ring == "gf2" else Phase(Fraction(0))
    offset = tuple(zero for _ in range(ambient.dimensions[3])) + tuple(
        value for _, _, defect in blocks for value in defect
    )
    db = _matmul(D, B)
    ed = _matmul(E, D)
    eb = _matvec(E, offset)
    try:
        db_witness = _zero_matrix_witness("DB", D, B, db)
        ed_witness = _zero_matrix_witness("ED", E, D, ed)
    except ValueError as error:
        raise ValueError(f"chain_identity_failed: {error}") from error
    if not _vector_is_zero(eb):
        offending = "ambient"
        for index, (start, stop) in enumerate(coordinates.local_slices[3]):
            if not _vector_is_zero(eb[start:stop]):
                offending = coordinates.instance_ids[index]
                break
        raise ValueError(f"chain_identity_failed: E@offset is nonzero for {offending}")
    eb_witness = _zero_vector_witness("E-offset", E, offset, eb)
    matrix_digest = _digest(
        "relative-matrices",
        {"B": _matrix_mapping(B), "D": _matrix_mapping(D), "E": _matrix_mapping(E)},
    )
    offset_digest = _digest(
        "relative-offset", [_coefficient_mapping(value) for value in offset]
    )
    blocks_digest = _digest("relative-coordinate-blocks", _blocks_mapping(coordinates))
    problem_digest, local_ids, restriction_ids, defect_ids = _problem_bindings(
        problem, ambient, blocks
    )
    core = {
        "ambient_complex_id": ambient.complex_id,
        "coordinate_blocks_digest": blocks_digest,
        "db_zero_witness_digest": db_witness,
        "defect_digests": list(defect_ids),
        "eb_zero_witness_digest": eb_witness,
        "ed_zero_witness_digest": ed_witness,
        "instance_ids": list(coordinates.instance_ids),
        "local_complex_ids": list(local_ids),
        "matrix_digest": matrix_digest,
        "offset_digest": offset_digest,
        "problem_digest": problem_digest,
        "restriction_digests": list(restriction_ids),
        "ring": problem.ring,
    }
    certificate = RelativeIdentityCertificate(
        _digest("relative-certificate", core),
        problem.ring,
        problem_digest,
        ambient.complex_id,
        coordinates.instance_ids,
        local_ids,
        restriction_ids,
        defect_ids,
        matrix_digest,
        offset_digest,
        blocks_digest,
        db_witness,
        ed_witness,
        eb_witness,
    )
    return RelativeMatrices(B, D, E, offset, coordinates, certificate)


def verify_relative_certificate(
    matrices: RelativeMatrices,
    problem: RelativeProblem,
) -> RelativeMatrices:
    if type(matrices) is not RelativeMatrices:
        raise TypeError("verify_relative_certificate requires RelativeMatrices")
    # Reconstructing replays the internal identities; recomputing from the
    # trusted source complexes additionally closes self-consistent rehashes.
    checked = RelativeMatrices(
        matrices.B,
        matrices.D,
        matrices.E,
        matrices.offset,
        matrices.coordinate_blocks,
        matrices.certificate,
    )
    expected = assemble_relative_problem(problem)
    if checked != expected:
        raise ValueError("relative certificate does not bind the source problem")
    return matrices


__all__ = [
    "ExactCoefficient",
    "Matrix",
    "RelativeCoordinateBlocks",
    "RelativeIdentityCertificate",
    "RelativeMatrices",
    "RelativeProblem",
    "assemble_relative_problem",
    "verify_relative_certificate",
]
