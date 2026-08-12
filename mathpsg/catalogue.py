"""Exact semantic normalization for canonical Wyckoff catalogue records."""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any

from .catalogue_schema import CatalogueRecord, canonical_json, parse_catalogue_record


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SETTING_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
_WYCKOFF_ID_DOMAIN = "mathpsg-wyckoff-id-v1"
_NORMALIZATION_VERSION = 1
_RATIONAL_RE = re.compile(r"q\((-?(?:0|[1-9][0-9]*)),([1-9][0-9]*)\)\Z")


Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class _Affine:
    matrix: Matrix
    translation: Vector


@dataclass(frozen=True, slots=True)
class _Branch:
    offset: Vector
    basis: Matrix


@dataclass(frozen=True, slots=True)
class _ParameterAffine:
    matrix: Matrix
    translation: Vector


@dataclass(frozen=True, slots=True)
class _Transport:
    ambient: _Affine
    parameter: _ParameterAffine


@dataclass(frozen=True, slots=True)
class _AffineResidue:
    """An affine coset in lattice coordinates, with translation reduced mod Z^3."""

    matrix: Matrix
    translation: Vector


def embedding_digest(payload: Any) -> str:
    """Hash an identity payload using canonical UTF-8 JSON bytes."""

    return f"sha256:{hashlib.sha256(canonical_json(payload)).hexdigest()}"


def catalogue_record_order_key(
    record: CatalogueRecord | Mapping[str, Any],
) -> tuple[int, bytes]:
    """Return the single v1 canonical ordering key for persisted geometry rows."""

    mapping = record.to_mapping() if isinstance(record, CatalogueRecord) else record
    parsed = parse_catalogue_record(mapping)
    international_number = parsed.space_group["international_number"]
    return int(international_number), canonical_json(parsed)


def catalogue_id(record: CatalogueRecord | Mapping[str, Any]) -> str:
    """Derive a v1 persistent ID from setting identity and embedding digest."""

    if isinstance(record, CatalogueRecord):
        space_group = record.space_group
        digest = record.embedding_digest
        provenance = record.provenance
    elif isinstance(record, Mapping):
        space_group = record.get("space_group")
        digest = record.get("embedding_digest")
        provenance = record.get("provenance")
    else:
        raise TypeError("catalogue_id expects a CatalogueRecord or mapping")
    if not isinstance(space_group, Mapping):
        raise TypeError("catalogue_id: space_group must be an object")
    if not isinstance(provenance, Mapping):
        raise TypeError("catalogue_id: provenance must be an object")
    international_number = space_group.get("international_number")
    setting = space_group.get("setting")
    normalization_version = provenance.get("normalization_version")
    if (
        isinstance(international_number, bool)
        or not isinstance(international_number, int)
        or not 1 <= international_number <= 230
    ):
        raise ValueError("catalogue_id: international_number must be in 1..230")
    if not isinstance(setting, str) or _SETTING_RE.fullmatch(setting) is None:
        raise ValueError("catalogue_id: invalid setting")
    if (
        isinstance(normalization_version, bool)
        or not isinstance(normalization_version, int)
        or normalization_version != _NORMALIZATION_VERSION
    ):
        raise ValueError(
            "catalogue_id: unsupported normalization_version "
            f"{normalization_version!r}; expected {_NORMALIZATION_VERSION}"
        )
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("catalogue_id: invalid embedding_digest")
    id_digest = embedding_digest(
        [
            _WYCKOFF_ID_DOMAIN,
            international_number,
            setting,
            normalization_version,
            digest,
        ]
    )
    return f"sg{international_number}:setting-{setting}:{id_digest}"


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    return value


def _require_list(value: Any, path: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path}: expected array")
    return value


def _sorted_json(values: Any, path: str) -> list[Any]:
    return sorted(_require_list(values, path), key=canonical_json)


def _fraction(value: Any, path: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{path}: expected exact rational q(n,d), not a JSON number")
    match = _RATIONAL_RE.fullmatch(value)
    if match is None or match.group(1) == "-0":
        raise ValueError(f"{path}: invalid exact rational spelling")
    result = Fraction(int(match.group(1)), int(match.group(2)))
    if _rational(result) != value:
        raise ValueError(f"{path}: exact rational must use reduced canonical spelling")
    return result


def _rational(value: Fraction) -> str:
    return f"q({value.numerator},{value.denominator})"


def _vector(value: Any, length: int, path: str) -> Vector:
    items = _require_list(value, path)
    if len(items) != length:
        raise ValueError(f"{path}: vector dimension must be {length}")
    return tuple(_fraction(item, f"{path}[{index}]") for index, item in enumerate(items))


def _matrix(value: Any, rows: int, columns: int, path: str) -> Matrix:
    items = _require_list(value, path)
    if len(items) != rows:
        raise ValueError(f"{path}: matrix row dimension must be {rows}")
    result: list[tuple[Fraction, ...]] = []
    for row_index, row in enumerate(items):
        row_items = _require_list(row, f"{path}[{row_index}]")
        if len(row_items) != columns:
            raise ValueError(f"{path}: matrix column dimension must be {columns}")
        result.append(
            tuple(
                _fraction(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row_items)
            )
        )
    return tuple(result)


def _identity_matrix(dimension: int) -> Matrix:
    return tuple(
        tuple(Fraction(row == column) for column in range(dimension))
        for row in range(dimension)
    )


def _matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    if not left:
        return ()
    inner = len(left[0])
    if inner != len(right):
        raise ValueError("internal matrix dimension mismatch")
    columns = 0 if not right else len(right[0])
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(inner)) for column in range(columns))
        for row in range(len(left))
    )


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix)


def _vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_subtract(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _matrix_inverse(matrix: Matrix, path: str) -> Matrix:
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError(f"{path}: expected square matrix")
    if dimension == 0:
        return ()
    augmented = [list(row) + list(identity_row) for row, identity_row in zip(matrix, _identity_matrix(dimension), strict=True)]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if augmented[row][column]), None)
        if pivot is None:
            raise ValueError(f"{path}: matrix must be invertible")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            multiple = augmented[row][column]
            if multiple:
                augmented[row] = [
                    value - multiple * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
                ]
    return tuple(tuple(row[dimension:]) for row in augmented)


def _parse_affine(value: Any, path: str) -> _Affine:
    mapping = _require_mapping(value, path)
    return _Affine(
        _matrix(mapping.get("matrix"), 3, 3, f"{path}.matrix"),
        _vector(mapping.get("translation"), 3, f"{path}.translation"),
    )


def _affine_mapping(value: _Affine) -> dict[str, Any]:
    return {
        "matrix": [[_rational(item) for item in row] for row in value.matrix],
        "translation": [_rational(item) for item in value.translation],
    }


def _affine_compose(left: _Affine, right: _Affine) -> _Affine:
    """Return ``left after right`` in column-action convention."""

    return _Affine(
        _matrix_multiply(left.matrix, right.matrix),
        _vector_add(_matrix_vector(left.matrix, right.translation), left.translation),
    )


def _affine_inverse(value: _Affine, path: str) -> _Affine:
    inverse_matrix = _matrix_inverse(value.matrix, f"{path}.matrix")
    return _Affine(
        inverse_matrix,
        tuple(-item for item in _matrix_vector(inverse_matrix, value.translation)),
    )


def _conjugate_affine(value: _Affine, forward: _Affine, inverse: _Affine) -> _Affine:
    return _affine_compose(_affine_compose(forward, value), inverse)


def _parse_presentation_conjugation(
    value: Any, path: str = "$export.presentation_conjugation"
) -> tuple[_Affine, _Affine] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, path)
    if set(mapping) != {"forward", "inverse"}:
        raise ValueError(f"{path}: expected exactly forward and inverse")
    forward = _parse_affine(mapping.get("forward"), f"{path}.forward")
    inverse = _parse_affine(mapping.get("inverse"), f"{path}.inverse")
    identity = _Affine(_identity_matrix(3), (Fraction(0),) * 3)
    if (
        _affine_compose(forward, inverse) != identity
        or _affine_compose(inverse, forward) != identity
    ):
        raise ValueError(f"{path}.inverse: not the exact affine inverse of forward")
    return forward, inverse


def _parameter_compose(
    left: _ParameterAffine, right: _ParameterAffine
) -> _ParameterAffine:
    return _ParameterAffine(
        _matrix_multiply(left.matrix, right.matrix),
        _vector_add(_matrix_vector(left.matrix, right.translation), left.translation),
    )


def _parameter_inverse(value: _ParameterAffine, path: str) -> _ParameterAffine:
    inverse_matrix = _matrix_inverse(value.matrix, f"{path}.matrix")
    return _ParameterAffine(
        inverse_matrix,
        tuple(-item for item in _matrix_vector(inverse_matrix, value.translation)),
    )


def _parse_branch(value: Any, dimension: int, names: tuple[str, ...], path: str) -> tuple[str, _Branch]:
    mapping = _require_mapping(value, path)
    branch_dimension = mapping.get("parameter_dimension")
    branch_names = mapping.get("parameter_names")
    if branch_dimension != dimension or tuple(_require_list(branch_names, f"{path}.parameter_names")) != names:
        raise ValueError(f"{path}: branch must use the shared parameter dimension and names")
    branch = _Branch(
        _vector(mapping.get("offset"), 3, f"{path}.offset"),
        _matrix(mapping.get("basis"), 3, dimension, f"{path}.basis"),
    )
    digest = mapping.get("branch_digest")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}.branch_digest: invalid digest")
    source_core = {
        "basis": mapping.get("basis"),
        "offset": mapping.get("offset"),
        "parameter_dimension": dimension,
        "parameter_names": list(names),
    }
    if embedding_digest(source_core) != digest:
        raise ValueError(f"{path}.branch_digest: does not match branch presentation")
    _canonical_parameter_gauge(branch, dimension, path)
    return digest, branch


def _canonical_parameter_gauge(
    reference: _Branch, dimension: int, path: str
) -> tuple[Matrix, Vector]:
    if dimension == 0:
        return (), ()
    for rows in combinations(range(3), dimension):
        minor = tuple(reference.basis[row] for row in rows)
        try:
            inverse = _matrix_inverse(minor, path)
        except ValueError:
            continue
        selected_offset = tuple(reference.offset[row] for row in rows)
        shift = tuple(-value for value in _matrix_vector(inverse, selected_offset))
        return inverse, shift
    raise ValueError(
        f"{path}: basis rank is smaller than parameter_dimension {dimension}"
    )


def _transform_branch(branch: _Branch, linear: Matrix, shift: Vector) -> _Branch:
    return _Branch(
        _vector_add(branch.offset, _matrix_vector(branch.basis, shift)),
        _matrix_multiply(branch.basis, linear),
    )


def _branch_mapping(branch: _Branch, dimension: int) -> dict[str, Any]:
    core: dict[str, Any] = {
        "basis": [[_rational(item) for item in row] for row in branch.basis],
        "offset": [_rational(item) for item in branch.offset],
        "parameter_dimension": dimension,
        "parameter_names": [f"lambda{index}" for index in range(1, dimension + 1)],
    }
    core["branch_digest"] = embedding_digest(core)
    return core


def _transform_parameter_action(
    linear: Matrix,
    shift: Vector,
    action_linear: Matrix,
    action_shift: Vector,
) -> tuple[Matrix, Vector]:
    if not linear:
        return (), ()
    inverse = _matrix_inverse(linear, "$parameter_gauge")
    transformed_linear = _matrix_multiply(
        _matrix_multiply(inverse, action_linear), linear
    )
    moved_shift = _vector_subtract(
        _vector_add(_matrix_vector(action_linear, shift), action_shift), shift
    )
    return transformed_linear, _matrix_vector(inverse, moved_shift)


def _parameter_mapping(matrix: Matrix, translation: Vector) -> dict[str, Any]:
    return {
        "matrix": [[_rational(item) for item in row] for row in matrix],
        "translation": [_rational(item) for item in translation],
    }


def _verify_transport(
    ambient: _Affine,
    reference: _Branch,
    target: _Branch,
    parameter_matrix: Matrix,
    parameter_shift: Vector,
    path: str,
) -> None:
    expected_offset = _vector_add(
        target.offset, _matrix_vector(target.basis, parameter_shift)
    )
    actual_offset = _vector_add(
        _matrix_vector(ambient.matrix, reference.offset), ambient.translation
    )
    expected_basis = _matrix_multiply(target.basis, parameter_matrix)
    actual_basis = _matrix_multiply(ambient.matrix, reference.basis)
    if actual_offset != expected_offset or actual_basis != expected_basis:
        raise ValueError(f"{path}: orbit-closure failure for exact branch transport")


def _verify_stabilizer(
    elements: tuple[_Affine, ...],
    declared_order: Any,
    reference: _Branch,
    dimension: int,
    path: str,
) -> None:
    if (
        isinstance(declared_order, bool)
        or not isinstance(declared_order, int)
        or declared_order < 1
        or declared_order != len(elements)
    ):
        raise ValueError(f"{path}.order: stabilizer-order consistency failure")
    element_set = set(elements)
    if len(element_set) != len(elements):
        raise ValueError(f"{path}: stabilizer-order consistency failure from duplicate elements")
    identity = _Affine(_identity_matrix(3), (Fraction(0),) * 3)
    if identity not in element_set:
        raise ValueError(f"{path}: stabilizer-identity failure")
    gauge_linear, _ = _canonical_parameter_gauge(reference, dimension, f"{path}.reference")
    pivot_rows: tuple[int, ...] = ()
    if dimension:
        for rows in combinations(range(3), dimension):
            minor = tuple(reference.basis[row] for row in rows)
            try:
                _matrix_inverse(minor, f"{path}.reference")
            except ValueError:
                continue
            pivot_rows = rows
            break
    for index, element in enumerate(elements):
        moved_offset = _vector_add(
            _matrix_vector(element.matrix, reference.offset), element.translation
        )
        moved_basis = _matrix_multiply(element.matrix, reference.basis)
        if dimension:
            induced_matrix = _matrix_multiply(
                gauge_linear,
                tuple(moved_basis[row] for row in pivot_rows),
            )
            induced_translation = _matrix_vector(
                gauge_linear,
                tuple(
                    moved_offset[row] - reference.offset[row]
                    for row in pivot_rows
                ),
            )
        else:
            induced_matrix = ()
            induced_translation = ()
        if (
            moved_offset != reference.offset
            or moved_basis != reference.basis
            or induced_matrix != _identity_matrix(dimension)
            or induced_translation != (Fraction(0),) * dimension
        ):
            raise ValueError(f"{path}.embedded_elements[{index}]: exact-fixation failure")
    for left in elements:
        for right in elements:
            if _affine_compose(left, right) not in element_set:
                raise ValueError(f"{path}: stabilizer-closure failure")
    for element in elements:
        if _affine_inverse(element, path) not in element_set:
            raise ValueError(f"{path}: stabilizer-inverse failure")
        power = identity
        for _ in range(1, declared_order + 1):
            power = _affine_compose(element, power)
            if power == identity:
                break
        else:
            raise ValueError(f"{path}: stabilizer-element-order failure")


def _canonical_coset_representative(transport: _Affine, stabilizer: tuple[_Affine, ...]) -> _Affine:
    representatives = [_affine_compose(transport, element) for element in stabilizer]
    return min(representatives, key=lambda item: canonical_json(_affine_mapping(item)))


def _determinant(matrix: Matrix) -> Fraction:
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError("internal determinant requires a square matrix")
    if dimension == 0:
        return Fraction(1)
    total = Fraction(0)
    for column in range(dimension):
        minor = tuple(
            tuple(row[index] for index in range(dimension) if index != column)
            for row in matrix[1:]
        )
        total += ((-1) ** column) * matrix[0][column] * _determinant(minor)
    return total


def _nullspace_rows_for_columns(columns: Matrix, column_count: int) -> tuple[Vector, ...]:
    """Return a rational basis annihilating the supplied column-space matrix."""

    if not columns:
        return _identity_matrix(column_count)
    work = [list(row) for row in columns]
    row_count = len(work)
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(pivot_row, row_count) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            multiple = work[row][column]
            if multiple:
                work[row] = [
                    value - multiple * pivot_value
                    for value, pivot_value in zip(
                        work[row], work[pivot_row], strict=True
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    free_columns = [column for column in range(column_count) if column not in pivot_columns]
    basis: list[Vector] = []
    for free in free_columns:
        vector = [Fraction(0)] * column_count
        vector[free] = Fraction(1)
        for row, pivot in enumerate(pivot_columns):
            vector[pivot] = -work[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def _primitive_integer_row(row: Vector) -> tuple[int, ...]:
    denominator_lcm = 1
    for value in row:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    integers = [int(value * denominator_lcm) for value in row]
    divisor = 0
    for value in integers:
        divisor = math.gcd(divisor, abs(value))
    if divisor == 0:
        raise ValueError("internal zero nullspace row")
    integers = [value // divisor for value in integers]
    first = next(value for value in integers if value)
    if first < 0:
        integers = [-value for value in integers]
    return tuple(integers)


def _column_lattice_index(matrix: tuple[tuple[int, ...], ...]) -> int:
    row_count = len(matrix)
    if row_count == 0:
        return 1
    column_count = len(matrix[0])
    divisor = 0
    for selected in combinations(range(column_count), row_count):
        minor: Matrix = tuple(
            tuple(Fraction(matrix[row][column]) for column in selected)
            for row in range(row_count)
        )
        divisor = math.gcd(divisor, abs(_determinant(minor).numerator))
    return divisor


def _offset_matches_modulo_lattice(
    difference: Vector,
    target_basis: Matrix,
    lattice: Matrix,
) -> bool:
    lattice_inverse = _matrix_inverse(lattice, "$ambient_lattice")
    lattice_difference = _matrix_vector(lattice_inverse, difference)
    lattice_basis = _matrix_multiply(lattice_inverse, target_basis)
    dimension = 0 if not target_basis else len(target_basis[0])
    transpose_basis: Matrix = tuple(
        tuple(lattice_basis[row][column] for row in range(3))
        for column in range(dimension)
    )
    nullspace = _nullspace_rows_for_columns(transpose_basis, 3)
    integer_rows = tuple(_primitive_integer_row(row) for row in nullspace)
    right_hand_side = tuple(
        sum(Fraction(coefficient) * value for coefficient, value in zip(row, lattice_difference, strict=True))
        for row in integer_rows
    )
    if any(value.denominator != 1 for value in right_hand_side):
        return False
    base_index = _column_lattice_index(integer_rows)
    augmented = tuple(
        row + (int(value),)
        for row, value in zip(integer_rows, right_hand_side, strict=True)
    )
    return base_index != 0 and _column_lattice_index(augmented) == base_index


def _families_match_modulo_lattice(
    moved: _Branch,
    target: _Branch,
    dimension: int,
    lattice: Matrix,
) -> bool:
    if dimension:
        pivot_inverse: Matrix | None = None
        pivot_rows: tuple[int, ...] = ()
        for rows in combinations(range(3), dimension):
            try:
                pivot_inverse = _matrix_inverse(
                    tuple(target.basis[row] for row in rows), "$target_family"
                )
            except ValueError:
                continue
            pivot_rows = rows
            break
        if pivot_inverse is None:
            return False
        parameter_matrix = _matrix_multiply(
            pivot_inverse, tuple(moved.basis[row] for row in pivot_rows)
        )
        try:
            _matrix_inverse(parameter_matrix, "$ambient_parameter_action")
        except ValueError:
            return False
        if _matrix_multiply(target.basis, parameter_matrix) != moved.basis:
            return False
    elif moved.basis != target.basis:
        return False
    return _offset_matches_modulo_lattice(
        _vector_subtract(moved.offset, target.offset), target.basis, lattice
    )


def _same_parameter_family_modulo_lattice(
    left: _Branch, right: _Branch, lattice_inverse: Matrix
) -> bool:
    """Compare branches without independently changing their shared parameter."""

    if left.basis != right.basis:
        return False
    lattice_coordinates = _matrix_vector(
        lattice_inverse, _vector_subtract(left.offset, right.offset)
    )
    return all(value.denominator == 1 for value in lattice_coordinates)


def _reduce_mod_one(value: Fraction) -> Fraction:
    return value - math.floor(value)


def _affine_residue(
    value: _Affine,
    lattice: Matrix,
    lattice_inverse: Matrix,
) -> _AffineResidue:
    return _AffineResidue(
        _matrix_multiply(_matrix_multiply(lattice_inverse, value.matrix), lattice),
        tuple(
            _reduce_mod_one(item)
            for item in _matrix_vector(lattice_inverse, value.translation)
        ),
    )


def _residue_compose(
    left: _AffineResidue, right: _AffineResidue
) -> _AffineResidue:
    """Return ``left after right`` in the finite affine quotient."""

    return _AffineResidue(
        _matrix_multiply(left.matrix, right.matrix),
        tuple(
            _reduce_mod_one(item)
            for item in _vector_add(
                _matrix_vector(left.matrix, right.translation), left.translation
            )
        ),
    )


def _residue_inverse(value: _AffineResidue, path: str) -> _AffineResidue:
    inverse_matrix = _matrix_inverse(value.matrix, f"{path}.matrix")
    return _AffineResidue(
        inverse_matrix,
        tuple(
            _reduce_mod_one(-item)
            for item in _matrix_vector(inverse_matrix, value.translation)
        ),
    )


def _residue_affine(
    value: _AffineResidue, lattice: Matrix, lattice_inverse: Matrix
) -> _Affine:
    return _Affine(
        _matrix_multiply(_matrix_multiply(lattice, value.matrix), lattice_inverse),
        _matrix_vector(lattice, value.translation),
    )


def _authenticate_primitive_lattice(
    *,
    lattice_inverse: Matrix,
    quotient: frozenset[_AffineResidue],
    representatives: Mapping[_AffineResidue, _Affine],
    steps: tuple[tuple[_AffineResidue, _Affine], ...],
) -> None:
    """Prove that the declared lattice is exactly the generated translation kernel."""

    identity_matrix = _identity_matrix(3)
    identity_residue = _AffineResidue(identity_matrix, (Fraction(0),) * 3)
    identity_linear_residues = {
        residue for residue in quotient if residue.matrix == identity_matrix
    }
    if identity_linear_residues != {identity_residue}:
        raise ValueError(
            "$export.space_group_action.translation_basis: generated translations "
            "are finer than the declared primitive lattice (nonidentity "
            "identity-linear affine residue)"
        )

    representative_inverses = {
        residue: _affine_inverse(representative, "$ambient_group.word_representative")
        for residue, representative in representatives.items()
    }
    kernel_coordinates: set[tuple[int, int, int]] = set()
    for residue, representative in representatives.items():
        for step_residue, step_affine in steps:
            target_residue = _residue_compose(step_residue, residue)
            edge_word = _affine_compose(step_affine, representative)
            schreier_word = _affine_compose(
                representative_inverses[target_residue], edge_word
            )
            if schreier_word.matrix != identity_matrix:
                raise ValueError(
                    "$export.space_group_action: Schreier kernel word is not a "
                    "pure translation"
                )
            coordinates = _matrix_vector(
                lattice_inverse, schreier_word.translation
            )
            if any(value.denominator != 1 for value in coordinates):
                raise ValueError(
                    "$export.space_group_action.translation_basis: Schreier "
                    "kernel translation is outside the declared lattice"
                )
            kernel_coordinates.add(tuple(int(value) for value in coordinates))

    ordered_coordinates = tuple(sorted(kernel_coordinates))
    integer_generators = tuple(
        tuple(column[row] for column in ordered_coordinates) for row in range(3)
    )
    smith_index = _column_lattice_index(integer_generators)
    if smith_index == 0:
        raise ValueError(
            "$export.space_group_action.translation_basis: primitive translation "
            "Schreier kernel lattice rank is smaller than 3"
        )
    if smith_index != 1:
        raise ValueError(
            "$export.space_group_action.translation_basis: primitive translation "
            f"Schreier kernel has Smith/Hermite index {smith_index}, expected 1"
        )


def _finite_affine_quotient(
    lattice: Matrix,
    generators: tuple[_Affine, ...],
) -> frozenset[_AffineResidue]:
    """Construct and verify the exact generator closure modulo primitive translations."""

    lattice_inverse = _matrix_inverse(lattice, "$ambient_lattice")
    generator_residues = tuple(
        _affine_residue(generator, lattice, lattice_inverse)
        for generator in generators
    )
    inverse_generators = tuple(
        _affine_inverse(generator, "$ambient_group.generator")
        for generator in generators
    )
    step_affines = generators + inverse_generators
    step_residues = generator_residues + tuple(
        _affine_residue(generator, lattice, lattice_inverse)
        for generator in inverse_generators
    )
    steps = tuple(zip(step_residues, step_affines, strict=True))
    identity_residue = _AffineResidue(_identity_matrix(3), (Fraction(0),) * 3)
    identity_affine = _Affine(_identity_matrix(3), (Fraction(0),) * 3)
    representatives = {identity_residue: identity_affine}
    pending = deque([identity_residue])
    while pending:
        current = pending.popleft()
        for step_residue, step_affine in steps:
            product = _residue_compose(step_residue, current)
            if product in representatives:
                continue
            product_representative = _affine_compose(
                step_affine, representatives[current]
            )
            if (
                _affine_residue(product_representative, lattice, lattice_inverse)
                != product
            ):
                raise ValueError(
                    "$export.space_group_action: affine quotient word/residue "
                    "composition mismatch"
                )
            representatives[product] = product_representative
            pending.append(product)
            # A finite three-dimensional crystallographic point group has order
            # at most 48. More cosets prove that the supplied lattice/action is
            # not the claimed primitive crystallographic quotient.
            if len(representatives) > 48:
                raise ValueError(
                    "$export.space_group_action: generated affine quotient G/L "
                    "is not a finite primitive crystallographic quotient"
                )
    quotient = frozenset(representatives)
    for left in quotient:
        for right in quotient:
            if _residue_compose(left, right) not in quotient:
                raise ValueError(
                    "$export.space_group_action: affine quotient closure failure"
                )
        if _residue_inverse(left, "$ambient_group") not in quotient:
            raise ValueError(
                "$export.space_group_action: affine quotient inverse failure"
            )
    _authenticate_primitive_lattice(
        lattice_inverse=lattice_inverse,
        quotient=quotient,
        representatives=representatives,
        steps=steps,
    )
    return quotient


def _verify_ambient_group_candidate(
    branches: Mapping[str, _Branch],
    dimension: int,
    lattice: Matrix,
    quotient: frozenset[_AffineResidue],
    transports: Mapping[str, _Transport],
    stabilizer: tuple[_Affine, ...],
    primitive_size: int,
    stabilizer_order: int,
) -> None:
    lattice_inverse = _matrix_inverse(lattice, "$ambient_lattice")

    branch_items = tuple(branches.items())
    for (left_digest, left), (right_digest, right) in combinations(branch_items, 2):
        if _same_parameter_family_modulo_lattice(left, right, lattice_inverse):
            raise ValueError(
                "$export.candidates[].orbit.branches: pairwise branches "
                f"{left_digest} and {right_digest} are equivalent modulo the "
                "primitive lattice"
            )

    for target_digest, transport in transports.items():
        if _affine_residue(
            transport.ambient, lattice, lattice_inverse
        ) not in quotient:
            raise ValueError(
                "$export.candidates[].orbit.branch_transports: "
                f"ambient-group-membership failure for target {target_digest}"
            )
    for index, element in enumerate(stabilizer):
        if _affine_residue(element, lattice, lattice_inverse) not in quotient:
            raise ValueError(
                "$export.candidates[].stabilizer.embedded_elements"
                f"[{index}]: ambient-group-membership failure"
            )

    stabilizer_residues = frozenset(
        _affine_residue(element, lattice, lattice_inverse)
        for element in stabilizer
    )
    if len(stabilizer_residues) != stabilizer_order:
        raise ValueError(
            "$export.candidates[].stabilizer: distinct literal elements collapse "
            "in the primitive affine quotient"
        )
    branch_digests = tuple(branches)
    transport_residues = tuple(
        _affine_residue(transports[digest].ambient, lattice, lattice_inverse)
        for digest in branch_digests
    )
    branch_cosets = tuple(
        frozenset(
            _residue_compose(transport, element)
            for element in stabilizer_residues
        )
        for transport in transport_residues
    )
    if any(len(coset) != stabilizer_order for coset in branch_cosets):
        raise ValueError(
            "$export.candidates[].orbit.branch_transports: stabilizer coset "
            "cardinality failure"
        )
    if len(set(branch_cosets)) != len(branch_cosets) or any(
        left & right for left, right in combinations(branch_cosets, 2)
    ):
        raise ValueError(
            "$export.candidates[].orbit.branch_transports: transports do not "
            "represent distinct stabilizer cosets"
        )
    if len(quotient) != primitive_size * stabilizer_order:
        raise ValueError(
            "$export.candidates[]: affine quotient orbit-stabilizer failure: "
            f"|G/L|={len(quotient)} but primitive_orbit_size*stabilizer_order="
            f"{primitive_size * stabilizer_order}"
        )

    induced_permutations: list[tuple[int, ...]] = []
    for residue in quotient:
        affine = _residue_affine(residue, lattice, lattice_inverse)
        targets: list[int] = []
        for source_index, branch_digest in enumerate(branch_digests):
            branch = branches[branch_digest]
            moved = _Branch(
                _vector_add(
                    _matrix_vector(affine.matrix, branch.offset),
                    affine.translation,
                ),
                _matrix_multiply(affine.matrix, branch.basis),
            )
            matches = tuple(
                index
                for index, coset in enumerate(branch_cosets)
                if _residue_compose(
                    residue, transport_residues[source_index]
                ) in coset
            )
            if len(matches) != 1:
                raise ValueError(
                    "$export.candidates[].orbit.branches: generated affine quotient "
                    f"has {len(matches)} targets for branch {branch_digest}; expected "
                    "a unique branch map"
                )
            target_index = matches[0]
            if not _families_match_modulo_lattice(
                moved, branches[branch_digests[target_index]], dimension, lattice
            ):
                raise ValueError(
                    "$export.candidates[].orbit.branches: authenticated affine "
                    f"action has a geometric orbit-closure failure for {branch_digest}"
                )
            targets.append(target_index)
        permutation = tuple(targets)
        if len(set(permutation)) != len(branch_digests):
            raise ValueError(
                "$export.candidates[].orbit.branches: generated affine quotient "
                "does not induce a branch bijection"
            )
        induced_permutations.append(permutation)

    reachable = {permutation[0] for permutation in induced_permutations}
    if reachable != set(range(len(branch_digests))):
        raise ValueError(
            "$export.candidates[].orbit.branches: generated affine quotient action "
            "is not transitive"
        )


def action_provenance_digest(export: Mapping[str, Any]) -> str:
    """Hash exact source-action provenance without geometric candidates."""

    value = _require_mapping(export, "$export")
    action = _require_mapping(value.get("space_group_action"), "$export.space_group_action")
    generators = _require_list(
        action.get("source_generators"), "$export.space_group_action.source_generators"
    )
    right_witnesses = action.get("source_right_homogeneous_matrices")
    if right_witnesses is None:
        paired_generators = [{"column_affine": generator} for generator in generators]
    else:
        witnesses = _require_list(
            right_witnesses,
            "$export.space_group_action.source_right_homogeneous_matrices",
        )
        if len(witnesses) != len(generators):
            raise ValueError(
                "$export.space_group_action: source generator/witness lengths disagree"
            )
        paired_generators = [
            {
                "column_affine": generator,
                "source_right_homogeneous_matrix": witness,
            }
            for generator, witness in zip(generators, witnesses, strict=True)
        ]
    paired_generators.sort(key=canonical_json)
    payload = {
        "domain": "mathpsg-action-provenance-v1",
        "protocol_version": value.get("protocol_version"),
        "source": value.get("source"),
        "coordinate_convention": value.get("coordinate_convention"),
        "environment": value.get("environment"),
        "paired_source_generators": paired_generators,
        "presentation_conjugation": value.get("presentation_conjugation"),
    }
    return embedding_digest(payload)


def _verify_action_protocol(export: Mapping[str, Any]) -> None:
    expected_convention = {
        "affine_action": "x_column -> matrix*x_column + translation",
        "composition_law": "C(g*h)=C(h)*C(g)",
        "rational_encoding": "q(n,d)",
        "source_action": "Cryst right-row homogeneous matrices",
        "translation_policy": "full-unreduced",
    }
    convention = _require_mapping(
        export.get("coordinate_convention"), "$export.coordinate_convention"
    )
    if dict(convention) != expected_convention:
        raise ValueError("$export.coordinate_convention: unsupported column-action convention")
    action = _require_mapping(
        export.get("space_group_action"), "$export.space_group_action"
    )
    generators = _require_list(
        action.get("source_generators"), "$export.space_group_action.source_generators"
    )
    source_right = action.get("source_right_homogeneous_matrices")
    if source_right is None:
        return
    witnesses = _require_list(
        source_right,
        "$export.space_group_action.source_right_homogeneous_matrices",
    )
    if len(witnesses) != len(generators):
        raise ValueError("$export.space_group_action: column-action witness count mismatch")
    for index, (generator_value, witness_value) in enumerate(
        zip(generators, witnesses, strict=True)
    ):
        generator = _parse_affine(
            generator_value,
            f"$export.space_group_action.source_generators[{index}]",
        )
        witness = _matrix(
            witness_value,
            4,
            4,
            f"$export.space_group_action.source_right_homogeneous_matrices[{index}]",
        )
        transposed = tuple(
            tuple(witness[column][row] for column in range(4))
            for row in range(4)
        )
        homogeneous = tuple(
            tuple(generator.matrix[row]) + (generator.translation[row],)
            for row in range(3)
        ) + ((Fraction(0), Fraction(0), Fraction(0), Fraction(1)),)
        if transposed != homogeneous:
            raise ValueError(
                "$export.space_group_action: column-action conversion witness mismatch"
            )


def _verify_pinned_versions(export: Mapping[str, Any]) -> None:
    expected = {"gap": "4.15.1", "cryst": "4.1.30"}
    source = _require_mapping(export.get("source"), "$export.source")
    environment = _require_mapping(export.get("environment"), "$export.environment")
    versions = _require_mapping(
        environment.get("versions"), "$export.environment.versions"
    )
    for component, pinned_version in expected.items():
        if source.get(component) != pinned_version:
            raise ValueError(
                f"$export.source.{component}: version differs from pinned {pinned_version}"
            )
        if versions.get(component) != pinned_version:
            raise ValueError(
                "$export.environment.versions."
                f"{component}: version differs from pinned {pinned_version}"
            )


def _verify_lattice_action(
    action: Mapping[str, Any],
) -> tuple[Matrix, tuple[_Affine, ...]]:
    lattice = _matrix(
        action.get("translation_basis"),
        3,
        3,
        "$export.space_group_action.translation_basis",
    )
    try:
        lattice_inverse = _matrix_inverse(
            lattice, "$export.space_group_action.translation_basis"
        )
    except ValueError as error:
        raise ValueError(
            "$export.space_group_action.translation_basis: lattice rank must be 3"
        ) from error
    generator_values = _require_list(
        action.get("source_generators"), "$export.space_group_action.source_generators"
    )
    if not generator_values:
        raise ValueError("$export.space_group_action.source_generators: must not be empty")
    generators = tuple(
        _parse_affine(
            value, f"$export.space_group_action.source_generators[{index}]"
        )
        for index, value in enumerate(generator_values)
    )
    for index, generator in enumerate(generators):
        _affine_inverse(
            generator, f"$export.space_group_action.source_generators[{index}]"
        )
        lattice_coordinates = _matrix_multiply(
            _matrix_multiply(lattice_inverse, generator.matrix), lattice
        )
        try:
            inverse_coordinates = _matrix_inverse(
                lattice_coordinates,
                f"$export.space_group_action.source_generators[{index}]",
            )
        except ValueError as error:
            raise ValueError(
                f"$export.space_group_action.source_generators[{index}]: "
                "linear part is not a lattice automorphism"
            ) from error
        if any(
            value.denominator != 1
            for matrix in (lattice_coordinates, inverse_coordinates)
            for row in matrix
            for value in row
        ):
            raise ValueError(
                f"$export.space_group_action.source_generators[{index}]: "
                "linear part is not a lattice automorphism"
            )
    return lattice, generators


def _paired_tuple_for_reference(
    reference_digest: str,
    branches: Mapping[str, _Branch],
    transports: Mapping[str, _Transport],
    original_stabilizer: tuple[_Affine, ...],
    dimension: int,
    primitive_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_transport = transports[reference_digest]
    reference_transport_inverse = _affine_inverse(
        reference_transport.ambient, "$export.reference_transport"
    )
    reference_parameter_inverse = _parameter_inverse(
        reference_transport.parameter, "$export.reference_parameter_action"
    )
    transported_stabilizer = tuple(
        _affine_compose(
            _affine_compose(reference_transport.ambient, element),
            reference_transport_inverse,
        )
        for element in original_stabilizer
    )
    _verify_stabilizer(
        transported_stabilizer,
        len(transported_stabilizer),
        branches[reference_digest],
        dimension,
        "$canonical.stabilizer",
    )

    relative_transports: dict[str, _Transport] = {}
    for target_digest, target_transport in transports.items():
        ambient = _affine_compose(
            target_transport.ambient, reference_transport_inverse
        )
        parameter = _parameter_compose(
            target_transport.parameter, reference_parameter_inverse
        )
        ambient = _canonical_coset_representative(ambient, transported_stabilizer)
        _verify_transport(
            ambient,
            branches[reference_digest],
            branches[target_digest],
            parameter.matrix,
            parameter.translation,
            "$canonical.branch_transport",
        )
        relative_transports[target_digest] = _Transport(ambient, parameter)

    gauge_linear, gauge_shift = _canonical_parameter_gauge(
        branches[reference_digest], dimension, "$canonical.reference_branch"
    )
    transformed_branches = {
        digest: _transform_branch(branch, gauge_linear, gauge_shift)
        for digest, branch in branches.items()
    }
    branch_mappings = {
        digest: _branch_mapping(branch, dimension)
        for digest, branch in transformed_branches.items()
    }
    normalized_digests = [mapping["branch_digest"] for mapping in branch_mappings.values()]
    if len(set(normalized_digests)) != len(normalized_digests):
        raise ValueError("$canonical.branches: duplicate canonical branch")
    ordered_source_digests = sorted(
        branches,
        key=lambda digest: canonical_json(branch_mappings[digest]),
    )
    transport_mappings: dict[str, dict[str, Any]] = {}
    for target_digest, transport in relative_transports.items():
        parameter_matrix, parameter_shift = _transform_parameter_action(
            gauge_linear,
            gauge_shift,
            transport.parameter.matrix,
            transport.parameter.translation,
        )
        transport_mappings[target_digest] = {
            "target_branch_digest": branch_mappings[target_digest]["branch_digest"],
            "parameter_dimension": dimension,
            "ambient_element": _affine_mapping(transport.ambient),
            "parameter_action": _parameter_mapping(parameter_matrix, parameter_shift),
        }
    normalized_reference_digest = branch_mappings[reference_digest]["branch_digest"]
    orbit = {
        "primitive_orbit_size": primitive_size,
        "parameter_dimension": dimension,
        "parameter_names": [f"lambda{index}" for index in range(1, dimension + 1)],
        "branches": [branch_mappings[digest] for digest in ordered_source_digests],
        "reference_branch_digest": normalized_reference_digest,
        "branch_transports": [
            transport_mappings[digest] for digest in ordered_source_digests
        ],
    }
    stabilizer = {
        "reference_branch_digest": normalized_reference_digest,
        "order": len(transported_stabilizer),
        "embedded_elements": sorted(
            (_affine_mapping(element) for element in transported_stabilizer),
            key=canonical_json,
        ),
    }
    return orbit, stabilizer


def _normalized_candidate(
    candidate: Any,
    *,
    lattice: Matrix | None = None,
    ambient_quotient: frozenset[_AffineResidue] | None = None,
    presentation_conjugation: tuple[_Affine, _Affine] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _require_mapping(candidate, "$export.candidates[]")
    source_orbit = _require_mapping(value.get("orbit"), "$export.candidates[].orbit")
    dimension = source_orbit.get("parameter_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 0 <= dimension <= 3:
        raise ValueError("$export.candidates[].orbit.parameter_dimension: expected integer in 0..3")
    source_names = _require_list(
        source_orbit.get("parameter_names"),
        "$export.candidates[].orbit.parameter_names",
    )
    if len(source_names) != dimension or any(not isinstance(name, str) for name in source_names):
        raise ValueError(
            "$export.candidates[].orbit.parameter_names: must match parameter_dimension"
        )
    names = tuple(source_names)
    source_branches = _require_list(
        source_orbit.get("branches"), "$export.candidates[].orbit.branches"
    )
    primitive_size = source_orbit.get("primitive_orbit_size")
    if (
        isinstance(primitive_size, bool)
        or not isinstance(primitive_size, int)
        or primitive_size < 1
        or primitive_size != len(source_branches)
    ):
        raise ValueError(
            "$export.candidates[].orbit.primitive_orbit_size: must equal branch count"
        )
    parsed_branches: dict[str, _Branch] = {}
    for index, item in enumerate(source_branches):
        digest, branch = _parse_branch(
            item,
            dimension,
            names,
            f"$export.candidates[].orbit.branches[{index}]",
        )
        if digest in parsed_branches:
            raise ValueError("$export.candidates[].orbit.branches: duplicate branch digest")
        if presentation_conjugation is not None:
            forward, _ = presentation_conjugation
            branch = _Branch(
                _vector_add(
                    _matrix_vector(forward.matrix, branch.offset),
                    forward.translation,
                ),
                _matrix_multiply(forward.matrix, branch.basis),
            )
        parsed_branches[digest] = branch
    source_reference_digest = source_orbit.get("reference_branch_digest")
    if source_reference_digest not in parsed_branches:
        raise ValueError(
            "$export.candidates[].orbit.reference_branch_digest: does not name a branch"
        )

    source_transports = _require_list(
        source_orbit.get("branch_transports"),
        "$export.candidates[].orbit.branch_transports",
    )
    if len(source_transports) != len(parsed_branches):
        raise ValueError(
            "$export.candidates[].orbit.branch_transports: length must match branches"
        )
    parsed_transports: dict[str, _Transport] = {}
    for index, item in enumerate(source_transports):
        path = f"$export.candidates[].orbit.branch_transports[{index}]"
        transport = _require_mapping(item, path)
        target = transport.get("target_branch_digest")
        if not isinstance(target, str) or target not in parsed_branches:
            raise ValueError(f"{path}.target_branch_digest: does not name a branch")
        if target in parsed_transports:
            raise ValueError("$export.candidates[].orbit.branch_transports: duplicate target")
        if transport.get("parameter_dimension") != dimension:
            raise ValueError(f"{path}.parameter_dimension: must match orbit")
        if transport.get("exact_transport_verified") is False:
            raise ValueError(f"{path}: exporter did not verify exact transport")
        ambient = _parse_affine(transport.get("ambient_element"), f"{path}.ambient_element")
        if presentation_conjugation is not None:
            ambient = _conjugate_affine(
                ambient,
                presentation_conjugation[0],
                presentation_conjugation[1],
            )
        parameter = _require_mapping(transport.get("parameter_action"), f"{path}.parameter_action")
        parameter_affine = _ParameterAffine(
            _matrix(
                parameter.get("matrix"),
                dimension,
                dimension,
                f"{path}.parameter_action.matrix",
            ),
            _vector(
                parameter.get("translation"),
                dimension,
                f"{path}.parameter_action.translation",
            ),
        )
        _parameter_inverse(parameter_affine, f"{path}.parameter_action")
        _verify_transport(
            ambient,
            parsed_branches[source_reference_digest],
            parsed_branches[target],
            parameter_affine.matrix,
            parameter_affine.translation,
            path,
        )
        parsed_transports[target] = _Transport(ambient, parameter_affine)
    if set(parsed_transports) != set(parsed_branches):
        raise ValueError("$export.candidates[].orbit.branch_transports: missing branch target")

    source_stabilizer = _require_mapping(
        value.get("stabilizer"), "$export.candidates[].stabilizer"
    )
    if source_stabilizer.get("reference_branch_digest") != source_reference_digest:
        raise ValueError("orbit and stabilizer reference_branch_digest values must match")
    if source_stabilizer.get("fixation_verified") is False:
        raise ValueError("$export.candidates[].stabilizer: exporter did not verify fixation")
    stabilizer_values = _require_list(
        source_stabilizer.get("embedded_elements"),
        "$export.candidates[].stabilizer.embedded_elements",
    )
    parsed_stabilizer = tuple(
        _parse_affine(item, f"$export.candidates[].stabilizer.embedded_elements[{index}]")
        for index, item in enumerate(stabilizer_values)
    )
    if presentation_conjugation is not None:
        parsed_stabilizer = tuple(
            _conjugate_affine(
                element,
                presentation_conjugation[0],
                presentation_conjugation[1],
            )
            for element in parsed_stabilizer
        )
    _verify_stabilizer(
        parsed_stabilizer,
        source_stabilizer.get("order"),
        parsed_branches[source_reference_digest],
        dimension,
        "$export.candidates[].stabilizer",
    )
    if lattice is not None and ambient_quotient is not None:
        _verify_ambient_group_candidate(
            parsed_branches,
            dimension,
            lattice,
            ambient_quotient,
            parsed_transports,
            parsed_stabilizer,
            primitive_size,
            source_stabilizer.get("order"),
        )

    paired_candidates = [
        _paired_tuple_for_reference(
            digest,
            parsed_branches,
            parsed_transports,
            parsed_stabilizer,
            dimension,
            primitive_size,
        )
        for digest in parsed_branches
    ]
    return min(
        paired_candidates,
        key=lambda pair: canonical_json({"orbit": pair[0], "stabilizer": pair[1]}),
    )


def _identity_payload(
    space_group: Mapping[str, Any],
    translation_basis: Any,
    orbit: Mapping[str, Any],
    stabilizer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "space_group": {
            "international_number": space_group.get("international_number"),
            "setting": space_group.get("setting"),
        },
        "primitive_lattice": {"translation_basis": translation_basis},
        "paired_family": {"orbit": orbit, "stabilizer": stabilizer},
    }


def normalize_gap_export(export: Mapping[str, Any]) -> tuple[CatalogueRecord, ...]:
    """Normalize one successful GAP catalogue export into immutable v1 rows."""

    value = _require_mapping(export, "$export")
    if value.get("protocol_version") != 1:
        raise ValueError("$export.protocol_version: expected 1")
    if value.get("record_type") != "catalogue-gap-export":
        raise ValueError("$export.record_type: expected catalogue-gap-export")
    if value.get("status") != "success":
        raise ValueError("$export.status: expected success")
    canonical_json(value)
    _verify_pinned_versions(value)
    _verify_action_protocol(value)
    source = _require_mapping(value.get("source"), "$export.source")
    space_group = _require_mapping(value.get("space_group"), "$export.space_group")
    action = _require_mapping(value.get("space_group_action"), "$export.space_group_action")
    presentation_conjugation = _parse_presentation_conjugation(
        value.get("presentation_conjugation")
    )
    raw_lattice, raw_generators = _verify_lattice_action(action)
    if presentation_conjugation is None:
        parsed_lattice = raw_lattice
        parsed_generators = raw_generators
    else:
        forward, inverse = presentation_conjugation
        parsed_lattice = _matrix_multiply(forward.matrix, raw_lattice)
        parsed_generators = tuple(
            _conjugate_affine(generator, forward, inverse)
            for generator in raw_generators
        )
    affine_quotient = _finite_affine_quotient(parsed_lattice, parsed_generators)
    translation_basis = [
        [_rational(item) for item in row] for row in parsed_lattice
    ]
    source_generators = sorted(
        (_affine_mapping(generator) for generator in parsed_generators),
        key=canonical_json,
    )
    provenance_digest = action_provenance_digest(value)

    normalized: list[tuple[dict[str, Any], dict[str, Any], str, bytes]] = []
    digest_payloads: dict[str, bytes] = {}
    for candidate in _require_list(value.get("candidates"), "$export.candidates"):
        orbit, stabilizer = _normalized_candidate(
            candidate,
            lattice=parsed_lattice,
            ambient_quotient=affine_quotient,
            presentation_conjugation=presentation_conjugation,
        )
        identity = _identity_payload(space_group, translation_basis, orbit, stabilizer)
        identity_bytes = canonical_json(identity)
        digest = embedding_digest(identity)
        previous = digest_payloads.get(digest)
        if previous is not None:
            if previous != identity_bytes:
                raise ValueError("embedding digest collision for unequal identity payloads")
            raise ValueError("duplicate canonical Wyckoff identity")
        digest_payloads[digest] = identity_bytes
        normalized.append((orbit, stabilizer, digest, identity_bytes))
    normalized.sort(key=lambda item: item[3])
    generator_input_digest = embedding_digest(
        {
            "domain": "mathpsg-generator-input-v1",
            "action_provenance_digest": provenance_digest,
            "identity_payloads": [item[3].decode("utf-8") for item in normalized],
        }
    )

    records: list[CatalogueRecord] = []
    for orbit, stabilizer, digest, _ in normalized:
        mapping: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "wyckoff-position",
            "space_group": {
                "international_number": space_group.get("international_number"),
                "setting": space_group.get("setting"),
                "source": {"gap": source.get("gap"), "cryst": source.get("cryst")},
            },
            "wyckoff_id": "",
            "embedding_digest": digest,
            "action_provenance_digest": provenance_digest,
            "orbit": orbit,
            "stabilizer": stabilizer,
            "space_group_action": {
                "translation_basis": translation_basis,
                "source_generators": source_generators,
            },
            "provenance": {
                "generator_input_digest": generator_input_digest,
                "normalization_version": _NORMALIZATION_VERSION,
            },
        }
        if presentation_conjugation is not None:
            mapping["presentation_conjugation"] = {
                "forward": _affine_mapping(presentation_conjugation[0]),
                "inverse": _affine_mapping(presentation_conjugation[1]),
            }
        mapping["wyckoff_id"] = catalogue_id(mapping)
        records.append(parse_catalogue_record(mapping))
    return tuple(sorted(records, key=catalogue_record_order_key))


def validate_catalogue_record_identity(
    record: CatalogueRecord | Mapping[str, Any],
) -> CatalogueRecord:
    """Recheck normalized semantic identity and return one immutable v1 row."""

    mapping_value = record.to_mapping() if isinstance(record, CatalogueRecord) else record
    parsed = parse_catalogue_record(mapping_value)
    mapping = parsed.to_mapping()
    _parse_presentation_conjugation(
        mapping.get("presentation_conjugation"),
        "$catalogue.presentation_conjugation",
    )
    record_lattice, record_generators = _verify_lattice_action(
        mapping["space_group_action"]
    )
    affine_quotient = _finite_affine_quotient(record_lattice, record_generators)
    orbit, stabilizer = _normalized_candidate(
        {"orbit": mapping["orbit"], "stabilizer": mapping["stabilizer"]},
        lattice=record_lattice,
        ambient_quotient=affine_quotient,
    )
    if canonical_json(orbit) != canonical_json(mapping["orbit"]):
        raise ValueError("catalogue record orbit is not in canonical semantic form")
    if canonical_json(stabilizer) != canonical_json(mapping["stabilizer"]):
        raise ValueError("catalogue record stabilizer is not in canonical paired form")
    identity = _identity_payload(
        mapping["space_group"],
        mapping["space_group_action"]["translation_basis"],
        orbit,
        stabilizer,
    )
    expected_embedding = embedding_digest(identity)
    if parsed.embedding_digest != expected_embedding:
        raise ValueError(
            "catalogue record embedding_digest disagrees with normalized identity payload"
        )
    expected_id = catalogue_id(parsed)
    if parsed.wyckoff_id != expected_id:
        raise ValueError("catalogue record wyckoff_id disagrees with its v1 identity contract")
    return parsed


__all__ = [
    "action_provenance_digest",
    "catalogue_id",
    "catalogue_record_order_key",
    "embedding_digest",
    "normalize_gap_export",
    "validate_catalogue_record_identity",
]
