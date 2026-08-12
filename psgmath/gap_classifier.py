"""Strict, fail-closed protocol boundary for the pinned GAP classifier."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from ._resources import asset_bytes
from .classification_schema import (
    FrozenJSONArray,
    FrozenJSONObject,
    StructuredFailure,
    canonical_classification_json,
)


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RATIONAL_RE = re.compile(r"q\((-?(?:0|[1-9][0-9]*)),([1-9][0-9]*)\)\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_PROTOCOL_VERSION = 1
_CLASSIFIER_SOURCE_PREFIX = "gap/classifier/lib/"


Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]


def _fraction(value: Any, path: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{path}: exact rational must be a q(n,d) string")
    match = _RATIONAL_RE.fullmatch(value)
    if match is None or match.group(1) == "-0":
        raise ValueError(f"{path}: invalid exact rational")
    result = Fraction(int(match.group(1)), int(match.group(2)))
    if _rational(result) != value:
        raise ValueError(f"{path}: rational must use reduced canonical spelling")
    return result


def _rational(value: Fraction) -> str:
    return f"q({value.numerator},{value.denominator})"


def _parse_matrix(value: Any, rows: int, columns: int, path: str) -> Matrix:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{path}: expected {rows} rows")
    parsed: list[tuple[Fraction, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError(f"{path}[{row_index}]: expected {columns} columns")
        parsed.append(
            tuple(
                _fraction(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    return tuple(parsed)


def _parse_vector(value: Any, length: int, path: str) -> Vector:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{path}: expected length {length}")
    return tuple(_fraction(item, f"{path}[{index}]") for index, item in enumerate(value))


def _identity_matrix(dimension: int) -> Matrix:
    return tuple(
        tuple(Fraction(row == column) for column in range(dimension))
        for row in range(dimension)
    )


def _matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(matrix))
    )


def _matrix_inverse(matrix: Matrix, path: str) -> Matrix:
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError(f"{path}: matrix must be square")
    work = [
        list(row) + list(identity_row)
        for row, identity_row in zip(matrix, _identity_matrix(dimension), strict=True)
    ]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if work[row][column]), None
        )
        if pivot is None:
            raise ValueError(f"{path}: matrix must be invertible")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [item / scale for item in work[column]]
        for row in range(dimension):
            if row == column:
                continue
            multiple = work[row][column]
            if multiple:
                work[row] = [
                    item - multiple * pivot_item
                    for item, pivot_item in zip(work[row], work[column], strict=True)
                ]
    return tuple(tuple(row[dimension:]) for row in work)


@dataclass(frozen=True, slots=True)
class AffineTransformation:
    matrix: tuple[tuple[str, ...], ...]
    translation: tuple[str, ...]

    def __post_init__(self) -> None:
        parsed_matrix = _parse_matrix(self.matrix, 3, 3, "$AffineTransformation.matrix")
        _matrix_inverse(parsed_matrix, "$AffineTransformation.matrix")
        _parse_vector(self.translation, 3, "$AffineTransformation.translation")
        object.__setattr__(self, "matrix", tuple(tuple(row) for row in self.matrix))
        object.__setattr__(self, "translation", tuple(self.translation))


@dataclass(frozen=True, slots=True)
class AffineWord:
    steps: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        normalized = tuple(tuple(step) for step in self.steps)
        for index, step in enumerate(normalized):
            if len(step) != 2:
                raise ValueError(f"$AffineWord.steps[{index}]: expected index/exponent pair")
            generator, exponent = step
            if isinstance(generator, bool) or not isinstance(generator, int) or generator < 0:
                raise ValueError(f"$AffineWord.steps[{index}].generator: expected nonnegative integer")
            if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent == 0:
                raise ValueError(f"$AffineWord.steps[{index}].exponent: expected nonzero integer")
        object.__setattr__(self, "steps", normalized)


@dataclass(frozen=True, slots=True)
class LiteralStabilizerInclusion:
    inclusion_id: str
    literal_stabilizer_digest: str
    literal_element_digest: str
    literal_elements: tuple[AffineTransformation, ...]

    def __post_init__(self) -> None:
        _identifier(self.inclusion_id, "$LiteralStabilizerInclusion.inclusion_id")
        _digest(self.literal_stabilizer_digest, "$LiteralStabilizerInclusion.literal_stabilizer_digest")
        _digest(self.literal_element_digest, "$LiteralStabilizerInclusion.literal_element_digest")
        elements = tuple(self.literal_elements)
        if not elements or any(not isinstance(item, AffineTransformation) for item in elements):
            raise ValueError("$LiteralStabilizerInclusion.literal_elements: expected nonempty affine tuple")
        if self.literal_element_digest != literal_element_authority_digest(elements):
            raise ValueError(
                "$LiteralStabilizerInclusion.literal_element_digest: "
                "literal element digest does not bind exact affine elements"
            )
        object.__setattr__(self, "literal_elements", elements)


@dataclass(frozen=True, slots=True)
class CertifiedSpaceGroupAction:
    affine_generators: tuple[AffineTransformation, ...]
    translation_basis: tuple[tuple[str, ...], ...]
    action_digest: str

    def __post_init__(self) -> None:
        generators = tuple(self.affine_generators)
        if not generators or any(not isinstance(item, AffineTransformation) for item in generators):
            raise ValueError("$CertifiedSpaceGroupAction.affine_generators: expected nonempty affine tuple")
        lattice = _parse_matrix(self.translation_basis, 3, 3, "$CertifiedSpaceGroupAction.translation_basis")
        _matrix_inverse(lattice, "$CertifiedSpaceGroupAction.translation_basis")
        _digest(self.action_digest, "$CertifiedSpaceGroupAction.action_digest")
        object.__setattr__(self, "affine_generators", generators)
        object.__setattr__(self, "translation_basis", tuple(tuple(row) for row in self.translation_basis))


@dataclass(frozen=True, slots=True)
class GAPClassifierRequest:
    request_digest: str
    action: CertifiedSpaceGroupAction
    inclusions: tuple[LiteralStabilizerInclusion, ...]
    time_reversal: bool
    max_degree: int = 4

    def __post_init__(self) -> None:
        _digest(self.request_digest, "$GAPClassifierRequest.request_digest")
        if not isinstance(self.action, CertifiedSpaceGroupAction):
            raise TypeError("$GAPClassifierRequest.action: expected CertifiedSpaceGroupAction")
        inclusions = tuple(self.inclusions)
        if not inclusions or any(not isinstance(item, LiteralStabilizerInclusion) for item in inclusions):
            raise ValueError("$GAPClassifierRequest.inclusions: expected nonempty inclusion tuple")
        if not isinstance(self.time_reversal, bool):
            raise TypeError("$GAPClassifierRequest.time_reversal: expected boolean")
        if isinstance(self.max_degree, bool) or not isinstance(self.max_degree, int) or self.max_degree != 4:
            raise ValueError("$GAPClassifierRequest.max_degree: v1 requires degree 4")
        object.__setattr__(self, "inclusions", inclusions)


@dataclass(frozen=True, slots=True)
class TransportedInclusion:
    inclusion_id: str
    literal_stabilizer_digest: str
    literal_element_digest: str
    literal_elements: tuple[AffineTransformation, ...]
    pcp_images: tuple[str, ...]
    multiplication_table: tuple[tuple[int, ...], ...]
    inverse_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.inclusion_id, "$TransportedInclusion.inclusion_id")
        _digest(
            self.literal_stabilizer_digest,
            "$TransportedInclusion.literal_stabilizer_digest",
        )
        _digest(
            self.literal_element_digest,
            "$TransportedInclusion.literal_element_digest",
        )
        elements = tuple(self.literal_elements)
        images = tuple(self.pcp_images)
        table = tuple(tuple(row) for row in self.multiplication_table)
        inverses = tuple(self.inverse_indices)
        order = len(elements)
        if order == 0 or any(not isinstance(item, AffineTransformation) for item in elements):
            raise ValueError("$TransportedInclusion.literal_elements: expected nonempty affine tuple")
        if self.literal_element_digest != literal_element_authority_digest(elements):
            raise ValueError(
                "$TransportedInclusion.literal_element_digest: "
                "literal element digest does not bind exact affine elements"
            )
        if len(images) != order:
            raise ValueError("$TransportedInclusion.pcp_images: expected one image per literal element")
        for index, word in enumerate(images):
            _parse_pcp_word(word, f"$TransportedInclusion.pcp_images[{index}]")
        if len(table) != order or any(len(row) != order for row in table):
            raise ValueError("$TransportedInclusion.multiplication_table: expected square full table")
        if len(inverses) != order:
            raise ValueError("$TransportedInclusion.inverse_indices: expected one inverse per element")
        for row_index, row in enumerate(table):
            for column_index, item in enumerate(row):
                _index(item, order, f"$TransportedInclusion.multiplication_table[{row_index}][{column_index}]")
        for index, item in enumerate(inverses):
            _index(item, order, f"$TransportedInclusion.inverse_indices[{index}]")
        object.__setattr__(self, "literal_elements", elements)
        object.__setattr__(self, "pcp_images", images)
        object.__setattr__(self, "multiplication_table", table)
        object.__setattr__(self, "inverse_indices", inverses)


@dataclass(frozen=True, slots=True)
class PCPNormalFormAuthority:
    relative_orders: tuple[int, ...]
    generator_affines: tuple[AffineTransformation, ...]

    def __post_init__(self) -> None:
        orders = tuple(self.relative_orders)
        generators = tuple(self.generator_affines)
        if not orders or len(orders) != len(generators):
            raise ValueError(
                "$PCPNormalFormAuthority: expected one relative order per generator"
            )
        for index, order in enumerate(orders):
            if (
                isinstance(order, bool)
                or not isinstance(order, int)
                or order == 1
                or order < 0
            ):
                raise ValueError(
                    f"$PCPNormalFormAuthority.relative_orders[{index}]: "
                    "expected zero or an integer at least two"
                )
        if any(not isinstance(item, AffineTransformation) for item in generators):
            raise TypeError(
                "$PCPNormalFormAuthority.generator_affines: expected affine tuple"
            )
        infinite = tuple(index for index, order in enumerate(orders) if order == 0)
        if infinite != tuple(range(len(orders) - 3, len(orders))):
            raise ValueError(
                "$PCPNormalFormAuthority.relative_orders: "
                "v1 requires exactly three infinite translation generators as a suffix"
            )
        finite_order = 1
        for order in orders[:-3]:
            finite_order *= order
            if finite_order > 4096:
                raise ValueError(
                    "$PCPNormalFormAuthority.relative_orders: finite transversal exceeds v1 bound"
                )
        object.__setattr__(self, "relative_orders", orders)
        object.__setattr__(self, "generator_affines", generators)


@dataclass(frozen=True, slots=True)
class AffinePCPIsomorphismCertificate:
    catalogue_action_digest: str
    conversion_algorithm_digest: str
    pcp_normal_form: PCPNormalFormAuthority
    affine_generator_images: tuple[str, ...]
    pcp_generator_preimages: tuple[AffineWord, ...]
    translation_basis_images: tuple[str, ...]
    transported_stabilizers: tuple[TransportedInclusion, ...]
    roundtrip_words: tuple[AffineWord, ...]
    certificate_digest: str

    def __post_init__(self) -> None:
        _digest(self.catalogue_action_digest, "$AffinePCPIsomorphismCertificate.catalogue_action_digest")
        _digest(
            self.conversion_algorithm_digest,
            "$AffinePCPIsomorphismCertificate.conversion_algorithm_digest",
        )
        if not isinstance(self.pcp_normal_form, PCPNormalFormAuthority):
            raise TypeError(
                "$AffinePCPIsomorphismCertificate.pcp_normal_form: "
                "expected PCPNormalFormAuthority"
            )
        images = tuple(self.affine_generator_images)
        preimages = tuple(self.pcp_generator_preimages)
        translations = tuple(self.translation_basis_images)
        inclusions = tuple(self.transported_stabilizers)
        roundtrips = tuple(self.roundtrip_words)
        if not images:
            raise ValueError("$AffinePCPIsomorphismCertificate.affine_generator_images: expected nonempty tuple")
        for index, word in enumerate(images):
            _parse_pcp_word(word, f"$AffinePCPIsomorphismCertificate.affine_generator_images[{index}]")
        if not preimages or any(not isinstance(word, AffineWord) for word in preimages):
            raise ValueError("$AffinePCPIsomorphismCertificate.pcp_generator_preimages: expected nonempty affine-word tuple")
        if len(translations) != 3:
            raise ValueError("$AffinePCPIsomorphismCertificate.translation_basis_images: expected three words")
        for index, word in enumerate(translations):
            _parse_pcp_word(word, f"$AffinePCPIsomorphismCertificate.translation_basis_images[{index}]")
        if any(not isinstance(item, TransportedInclusion) for item in inclusions):
            raise TypeError("$AffinePCPIsomorphismCertificate.transported_stabilizers: expected TransportedInclusion records")
        if any(not isinstance(word, AffineWord) for word in roundtrips):
            raise TypeError("$AffinePCPIsomorphismCertificate.roundtrip_words: expected AffineWord records")
        _digest(self.certificate_digest, "$AffinePCPIsomorphismCertificate.certificate_digest")
        object.__setattr__(self, "affine_generator_images", images)
        object.__setattr__(self, "pcp_generator_preimages", preimages)
        object.__setattr__(self, "translation_basis_images", translations)
        object.__setattr__(self, "transported_stabilizers", inclusions)
        object.__setattr__(self, "roundtrip_words", roundtrips)


@dataclass(frozen=True, slots=True)
class GAPClassifierResponse:
    request_digest: str
    status: str
    environment: FrozenJSONObject | None
    affine_pcp_certificate: AffinePCPIsomorphismCertificate | None
    problem: FrozenJSONObject | None
    failures: tuple[StructuredFailure, ...]
    protocol_version: int = _PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _digest(self.request_digest, "$GAPClassifierResponse.request_digest")
        if self.status == "success":
            raise ValueError(
                "$GAPClassifierResponse.status: success requires an authoritative Task 5 problem"
            )
        if self.status not in ("conversion_only", "error"):
            raise ValueError(
                "$GAPClassifierResponse.status: expected conversion_only or error"
            )
        if isinstance(self.protocol_version, bool) or self.protocol_version != _PROTOCOL_VERSION:
            raise ValueError("$GAPClassifierResponse.protocol_version: unsupported version")
        if self.environment is not None and not isinstance(self.environment, FrozenJSONObject):
            object.__setattr__(self, "environment", _freeze_object(self.environment))
        if self.problem is not None and not isinstance(self.problem, FrozenJSONObject):
            object.__setattr__(self, "problem", _freeze_object(self.problem))
        failures = tuple(self.failures)
        if any(not isinstance(item, StructuredFailure) for item in failures):
            raise TypeError("$GAPClassifierResponse.failures: expected StructuredFailure records")
        object.__setattr__(self, "failures", failures)
        if self.status == "conversion_only":
            if self.environment is None or self.affine_pcp_certificate is None:
                raise ValueError(
                    "$GAPClassifierResponse: conversion-only response requires environment and certificate"
                )
            if self.problem is not None:
                raise ValueError(
                    "$GAPClassifierResponse: conversion-only response cannot contain a Task 5 problem"
                )
            if failures:
                raise ValueError(
                    "$GAPClassifierResponse: conversion-only response cannot contain failures"
                )
        else:
            if self.affine_pcp_certificate is not None or self.problem is not None:
                raise ValueError("$GAPClassifierResponse: error response cannot expose partial certificates")
            if len(failures) != 1:
                raise ValueError("$GAPClassifierResponse: error response requires exactly one failure")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: invalid identifier")
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256:<64 lowercase hex digits>")
    return value


def _index(value: Any, bound: int, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected integer index")
    if not 0 <= value < bound:
        raise ValueError(f"{path}: index outside 0..{bound - 1}")
    return value


_PCP_FACTOR_RE = re.compile(r"p([1-9][0-9]*)(?:\^(-?[1-9][0-9]*))?\Z")


def _parse_pcp_word(value: Any, path: str) -> tuple[tuple[int, int], ...]:
    if value == "1":
        return ()
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path}: expected canonical PCP word string")
    result: list[tuple[int, int]] = []
    previous_generator = -1
    for factor in value.split("*"):
        match = _PCP_FACTOR_RE.fullmatch(factor)
        if match is None:
            raise ValueError(f"{path}: invalid canonical PCP word")
        exponent = 1 if match.group(2) is None else int(match.group(2))
        if exponent == 1 and match.group(2) is not None:
            raise ValueError(f"{path}: exponent one must be omitted")
        generator = int(match.group(1)) - 1
        if generator <= previous_generator:
            raise ValueError(
                f"{path}: canonical PCP word factors must have strictly increasing indices"
            )
        result.append((generator, exponent))
        previous_generator = generator
    return tuple(result)


def _freeze_object(value: Any) -> FrozenJSONObject:
    if isinstance(value, FrozenJSONObject):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("expected JSON object")
    return FrozenJSONObject(tuple(value.items()))


def _thaw_json(value: Any) -> Any:
    if isinstance(value, FrozenJSONObject):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, FrozenJSONArray):
        return [_thaw_json(item) for item in value.items]
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _domain_digest(domain: str, payload: Any) -> str:
    encoded = canonical_classification_json(payload)
    prefix = f"mathpsg-gap-classifier-v1|{domain}|".encode("ascii")
    return f"sha256:{hashlib.sha256(prefix + encoded).hexdigest()}"


def _classifier_source_bytes(name: str) -> bytes:
    return asset_bytes(_CLASSIFIER_SOURCE_PREFIX + name)


def tracked_affine_pcp_conversion_digest() -> str:
    try:
        protocol = _classifier_source_bytes("protocol.g")
        affine_pcp = _classifier_source_bytes("affine_pcp.g")
    except OSError as error:
        raise ValueError("tracked affine-PCP sources are unavailable") from error
    return "sha256:" + hashlib.sha256(
        b"mathpsg-affine-pcp-conversion-v1|" + protocol + b"|" + affine_pcp
    ).hexdigest()


def _affine_mapping(value: AffineTransformation) -> dict[str, Any]:
    return {"matrix": [list(row) for row in value.matrix], "translation": list(value.translation)}


def _word_mapping(value: AffineWord) -> list[list[int]]:
    return [[generator, exponent] for generator, exponent in value.steps]


def _inclusion_mapping(value: LiteralStabilizerInclusion) -> dict[str, Any]:
    return {
        "inclusion_id": value.inclusion_id,
        "literal_element_digest": value.literal_element_digest,
        "literal_elements": [_affine_mapping(item) for item in value.literal_elements],
        "literal_stabilizer_digest": value.literal_stabilizer_digest,
    }


def _action_core_mapping(value: CertifiedSpaceGroupAction) -> dict[str, Any]:
    return {
        "affine_generators": [_affine_mapping(item) for item in value.affine_generators],
        "translation_basis": [list(row) for row in value.translation_basis],
    }


def _action_mapping(value: CertifiedSpaceGroupAction) -> dict[str, Any]:
    return {**_action_core_mapping(value), "action_digest": value.action_digest}


def make_certified_space_group_action(
    affine_generators: Sequence[AffineTransformation],
    translation_basis: Sequence[Sequence[str]],
) -> CertifiedSpaceGroupAction:
    core = {
        "affine_generators": [_affine_mapping(item) for item in affine_generators],
        "translation_basis": [list(row) for row in translation_basis],
    }
    return CertifiedSpaceGroupAction(
        tuple(affine_generators),
        tuple(tuple(row) for row in translation_basis),
        _domain_digest("certified-space-group-action-v1", core),
    )


def _affine_exact(value: AffineTransformation) -> tuple[Matrix, Vector]:
    return (
        _parse_matrix(value.matrix, 3, 3, "$affine.matrix"),
        _parse_vector(value.translation, 3, "$affine.translation"),
    )


def _affine_from_exact(matrix: Matrix, translation: Vector) -> AffineTransformation:
    return AffineTransformation(
        tuple(tuple(_rational(item) for item in row) for row in matrix),
        tuple(_rational(item) for item in translation),
    )


def _affine_identity() -> AffineTransformation:
    return _affine_from_exact(_identity_matrix(3), (Fraction(0),) * 3)


def _compose_affine(
    left: AffineTransformation, right: AffineTransformation
) -> AffineTransformation:
    """Return ``left o right`` in the exact column-vector convention."""

    left_matrix, left_translation = _affine_exact(left)
    right_matrix, right_translation = _affine_exact(right)
    matrix = _matrix_multiply(left_matrix, right_matrix)
    moved = _matrix_vector(left_matrix, right_translation)
    translation = tuple(
        moved[index] + left_translation[index] for index in range(3)
    )
    return _affine_from_exact(matrix, translation)


def _inverse_affine(value: AffineTransformation) -> AffineTransformation:
    matrix, translation = _affine_exact(value)
    inverse = _matrix_inverse(matrix, "$affine.matrix")
    moved = _matrix_vector(inverse, translation)
    return _affine_from_exact(inverse, tuple(-item for item in moved))


def _power_affine(value: AffineTransformation, exponent: int) -> AffineTransformation:
    if exponent < 0:
        value = _inverse_affine(value)
        exponent = -exponent
    result = _affine_identity()
    factor = value
    while exponent:
        if exponent & 1:
            result = _compose_affine(factor, result)
        factor = _compose_affine(factor, factor)
        exponent >>= 1
    return result


def evaluate_affine_word(
    generators: Sequence[AffineTransformation], word: AffineWord
) -> AffineTransformation:
    """Evaluate a Cryst source word using ``C(x*y)=C(y) o C(x)``."""

    source = tuple(generators)
    if any(not isinstance(item, AffineTransformation) for item in source):
        raise TypeError("generators must be affine transformations")
    if not isinstance(word, AffineWord):
        raise TypeError("word must be an AffineWord")
    result = _affine_identity()
    for step_index, (generator_index, exponent) in enumerate(word.steps):
        if generator_index >= len(source):
            raise ValueError(
                f"$AffineWord.steps[{step_index}].generator: index outside generator tuple"
            )
        result = _compose_affine(
            _power_affine(source[generator_index], exponent), result
        )
    return result


def _evaluate_pcp_word(
    word: str,
    authority: PCPNormalFormAuthority,
) -> AffineTransformation:
    coordinates = _pcp_word_coordinates(word, authority)
    result = _affine_identity()
    for generator_index, exponent in enumerate(coordinates):
        if exponent == 0:
            continue
        result = _compose_affine(
            _power_affine(authority.generator_affines[generator_index], exponent),
            result,
        )
    return result


def _pcp_word_coordinates(
    word: str, authority: PCPNormalFormAuthority
) -> tuple[int, ...]:
    coordinates = [0] * len(authority.relative_orders)
    for generator_index, exponent in _parse_pcp_word(word, "$pcp_word"):
        if generator_index >= len(coordinates):
            raise ValueError("$pcp_word: generator index outside PCP presentation")
        relative_order = authority.relative_orders[generator_index]
        if relative_order and not 0 < exponent < relative_order:
            raise ValueError(
                "$pcp_word: finite PCP exponent is outside canonical range"
            )
        coordinates[generator_index] = exponent
    return tuple(coordinates)


def _normal_form_decoder(
    action: CertifiedSpaceGroupAction,
    authority: PCPNormalFormAuthority,
):
    finite_orders = authority.relative_orders[:-3]
    infinite_generators = authority.generator_affines[-3:]
    identity = _identity_matrix(3)
    infinite_vectors: list[Vector] = []
    for index, generator in enumerate(infinite_generators):
        matrix, translation = _affine_exact(generator)
        if matrix != identity:
            raise ValueError(
                f"PCP infinite generator {len(finite_orders) + index} is not a translation"
            )
        infinite_vectors.append(translation)
    infinite_basis = tuple(
        tuple(infinite_vectors[column][row] for column in range(3))
        for row in range(3)
    )
    infinite_basis_inverse = _matrix_inverse(
        infinite_basis, "$certificate.pcp_normal_form.infinite_translation_basis"
    )
    declared_basis = _parse_matrix(
        action.translation_basis, 3, 3, "$action.translation_basis"
    )
    declared_basis_inverse = _matrix_inverse(
        declared_basis, "$action.translation_basis"
    )
    for source, target_inverse in (
        (declared_basis, infinite_basis_inverse),
        (infinite_basis, declared_basis_inverse),
    ):
        for column in range(3):
            coordinates = _matrix_vector(
                target_inverse, tuple(source[row][column] for row in range(3))
            )
            if any(item.denominator != 1 for item in coordinates):
                raise ValueError(
                    "PCP infinite generators do not generate the declared primitive lattice"
                )

    representatives: list[tuple[tuple[int, ...], AffineTransformation]] = []
    exponent_ranges = tuple(range(order) for order in finite_orders)
    for finite_coordinates in itertools.product(*exponent_ranges):
        result = _affine_identity()
        for generator_index, exponent in enumerate(finite_coordinates):
            if exponent:
                result = _compose_affine(
                    _power_affine(
                        authority.generator_affines[generator_index], exponent
                    ),
                    result,
                )
        representatives.append((tuple(finite_coordinates), result))

    def decode(element: AffineTransformation) -> tuple[int, ...]:
        candidates: list[tuple[int, ...]] = []
        for finite_coordinates, representative in representatives:
            residual = _compose_affine(element, _inverse_affine(representative))
            matrix, translation = _affine_exact(residual)
            if matrix != identity:
                continue
            infinite_coordinates = _matrix_vector(
                infinite_basis_inverse, translation
            )
            if all(item.denominator == 1 for item in infinite_coordinates):
                candidates.append(
                    finite_coordinates
                    + tuple(int(item) for item in infinite_coordinates)
                )
        if len(candidates) != 1:
            raise ValueError(
                "PCP affine realization does not have unique canonical normal forms"
            )
        return candidates[0]

    for finite_coordinates, representative in representatives:
        expected = finite_coordinates + (0, 0, 0)
        if decode(representative) != expected:
            raise ValueError("PCP finite transversal is not canonical")
    for generator_index, generator in enumerate(authority.generator_affines):
        expected = tuple(
            1 if index == generator_index else 0
            for index in range(len(authority.generator_affines))
        )
        if decode(generator) != expected:
            raise ValueError(
                f"PCP generator {generator_index} lacks its declared canonical normal form"
            )
    for _, representative in representatives:
        for generator in authority.generator_affines:
            for factor in (generator, _inverse_affine(generator)):
                decode(_compose_affine(factor, representative))
                decode(_compose_affine(representative, factor))
    return decode


def _transported_inclusion_mapping(value: TransportedInclusion) -> dict[str, Any]:
    return {
        "inclusion_id": value.inclusion_id,
        "inverse_indices": list(value.inverse_indices),
        "literal_element_digest": value.literal_element_digest,
        "literal_elements": [_affine_mapping(item) for item in value.literal_elements],
        "literal_stabilizer_digest": value.literal_stabilizer_digest,
        "multiplication_table": [list(row) for row in value.multiplication_table],
        "pcp_images": list(value.pcp_images),
    }


def literal_element_authority_digest(
    literal_elements: Sequence[AffineTransformation],
) -> str:
    elements = tuple(literal_elements)
    if not elements or any(not isinstance(item, AffineTransformation) for item in elements):
        raise ValueError("literal stabilizer authority requires exact affine elements")
    return _domain_digest(
        "literal-stabilizer-authority-v1",
        [_affine_mapping(item) for item in elements],
    )


def literal_stabilizer_authority_digest(
    literal_elements: Sequence[AffineTransformation],
) -> str:
    """Compatibility alias for the pre-Task-5 exact-element digest name."""

    return literal_element_authority_digest(literal_elements)


def transported_inclusion_authority_digest(value: TransportedInclusion) -> str:
    if not isinstance(value, TransportedInclusion):
        raise TypeError("transported inclusion authority requires a TransportedInclusion")
    return _domain_digest(
        "transported-inclusion-authority-v1",
        _transported_inclusion_mapping(value),
    )


def catalogue_record_authority_digest(
    *,
    group_id: str,
    catalogue_action_digest: str,
    inclusions: Sequence[TransportedInclusion],
) -> str:
    _identifier(group_id, "$catalogue_record_authority.group_id")
    _digest(catalogue_action_digest, "$catalogue_record_authority.catalogue_action_digest")
    normalized = tuple(inclusions)
    if not normalized or any(not isinstance(item, TransportedInclusion) for item in normalized):
        raise ValueError("catalogue record authority requires transported inclusions")
    if tuple(item.inclusion_id for item in normalized) != tuple(
        sorted(item.inclusion_id for item in normalized)
    ):
        raise ValueError("catalogue record authority inclusions must be canonical")
    return _domain_digest(
        "catalogue-record-authority-v1",
        {
            "catalogue_action_digest": catalogue_action_digest,
            "group_id": group_id,
            "transported_inclusions": [
                _transported_inclusion_mapping(item) for item in normalized
            ],
        },
    )


def _pcp_normal_form_mapping(value: PCPNormalFormAuthority) -> dict[str, Any]:
    return {
        "generator_affines": [
            _affine_mapping(item) for item in value.generator_affines
        ],
        "relative_orders": list(value.relative_orders),
    }


def _certificate_core_mapping(
    value: AffinePCPIsomorphismCertificate,
) -> dict[str, Any]:
    return {
        "affine_generator_images": list(value.affine_generator_images),
        "catalogue_action_digest": value.catalogue_action_digest,
        "conversion_algorithm_digest": value.conversion_algorithm_digest,
        "pcp_normal_form": _pcp_normal_form_mapping(value.pcp_normal_form),
        "pcp_generator_preimages": [_word_mapping(word) for word in value.pcp_generator_preimages],
        "roundtrip_words": [_word_mapping(word) for word in value.roundtrip_words],
        "translation_basis_images": list(value.translation_basis_images),
        "transported_stabilizers": [
            _transported_inclusion_mapping(item)
            for item in value.transported_stabilizers
        ],
    }


def _certificate_mapping(
    value: AffinePCPIsomorphismCertificate,
) -> dict[str, Any]:
    return {
        **_certificate_core_mapping(value),
        "certificate_digest": value.certificate_digest,
    }


def affine_pcp_certificate_digest(
    value: AffinePCPIsomorphismCertificate,
) -> str:
    if not isinstance(value, AffinePCPIsomorphismCertificate):
        raise TypeError("value must be an AffinePCPIsomorphismCertificate")
    return _domain_digest("affine-pcp-isomorphism-certificate-v1", _certificate_core_mapping(value))


def make_affine_pcp_certificate(
    *,
    catalogue_action_digest: str,
    conversion_algorithm_digest: str,
    pcp_normal_form: PCPNormalFormAuthority,
    affine_generator_images: Sequence[str],
    pcp_generator_preimages: Sequence[AffineWord],
    translation_basis_images: Sequence[str],
    transported_stabilizers: Sequence[TransportedInclusion],
    roundtrip_words: Sequence[AffineWord] = (),
) -> AffinePCPIsomorphismCertificate:
    provisional = AffinePCPIsomorphismCertificate(
        catalogue_action_digest,
        conversion_algorithm_digest,
        pcp_normal_form,
        tuple(affine_generator_images),
        tuple(pcp_generator_preimages),
        tuple(translation_basis_images),
        tuple(transported_stabilizers),
        tuple(roundtrip_words),
        "sha256:" + "0" * 64,
    )
    return AffinePCPIsomorphismCertificate(
        provisional.catalogue_action_digest,
        provisional.conversion_algorithm_digest,
        provisional.pcp_normal_form,
        provisional.affine_generator_images,
        provisional.pcp_generator_preimages,
        provisional.translation_basis_images,
        provisional.transported_stabilizers,
        provisional.roundtrip_words,
        affine_pcp_certificate_digest(provisional),
    )


def verify_affine_pcp_certificate(
    action: CertifiedSpaceGroupAction,
    certificate: AffinePCPIsomorphismCertificate,
) -> bool:
    if not isinstance(action, CertifiedSpaceGroupAction):
        raise TypeError("action must be a CertifiedSpaceGroupAction")
    if not isinstance(certificate, AffinePCPIsomorphismCertificate):
        raise TypeError("certificate must be an AffinePCPIsomorphismCertificate")
    if certificate.catalogue_action_digest != action.action_digest:
        raise ValueError("certificate catalogue action digest mismatch")
    if (
        certificate.conversion_algorithm_digest
        != tracked_affine_pcp_conversion_digest()
    ):
        raise ValueError("certificate does not bind the tracked affine-PCP sources")
    if affine_pcp_certificate_digest(certificate) != certificate.certificate_digest:
        raise ValueError("affine-PCP certificate digest mismatch")
    authority = certificate.pcp_normal_form
    decode = _normal_form_decoder(action, authority)
    if len(certificate.affine_generator_images) != len(action.affine_generators):
        raise ValueError("certificate does not map every affine generator")
    if len(certificate.pcp_generator_preimages) != len(
        authority.generator_affines
    ):
        raise ValueError("certificate does not invert every PCP generator")
    for index, (generator, image) in enumerate(
        zip(action.affine_generators, certificate.affine_generator_images, strict=True)
    ):
        actual = _evaluate_pcp_word(image, authority)
        if actual != generator:
            raise ValueError(f"affine generator {index} fails the PCP round trip")
        if decode(actual) != _pcp_word_coordinates(image, authority):
            raise ValueError(f"affine generator {index} has a noncanonical PCP image")
    for index, (generator, preimage) in enumerate(
        zip(
            authority.generator_affines,
            certificate.pcp_generator_preimages,
            strict=True,
        )
    ):
        actual = evaluate_affine_word(action.affine_generators, preimage)
        if actual != generator:
            raise ValueError(f"PCP generator {index} fails the affine round trip")
        expected = tuple(
            1 if candidate == index else 0
            for candidate in range(len(authority.generator_affines))
        )
        if decode(actual) != expected:
            raise ValueError(f"PCP generator {index} does not reduce exactly to itself")
    basis = _parse_matrix(action.translation_basis, 3, 3, "$action.translation_basis")
    identity = _identity_matrix(3)
    for column, word in enumerate(certificate.translation_basis_images):
        actual = _evaluate_pcp_word(word, authority)
        matrix, translation = _affine_exact(actual)
        expected = tuple(basis[row][column] for row in range(3))
        if matrix != identity or translation != expected:
            raise ValueError(f"translation basis image {column} does not round trip")
    affine_identity = _affine_identity()
    for inclusion_index, inclusion in enumerate(certificate.transported_stabilizers):
        actual_elements = tuple(
            _evaluate_pcp_word(word, authority)
            for word in inclusion.pcp_images
        )
        if actual_elements != inclusion.literal_elements:
            raise ValueError(f"transported inclusion {inclusion_index} does not commute")
        identity_indices = tuple(
            index for index, element in enumerate(inclusion.literal_elements)
            if element == affine_identity
        )
        if len(identity_indices) != 1:
            raise ValueError(f"transported inclusion {inclusion_index} lacks a unique identity")
        identity_index = identity_indices[0]
        for left_index, left in enumerate(inclusion.literal_elements):
            for right_index, right in enumerate(inclusion.literal_elements):
                product = _compose_affine(right, left)
                target = inclusion.multiplication_table[left_index][right_index]
                if inclusion.literal_elements[target] != product:
                    raise ValueError(f"transported inclusion {inclusion_index} multiplication table fails")
            inverse_index = inclusion.inverse_indices[left_index]
            if (
                inclusion.multiplication_table[left_index][inverse_index] != identity_index
                or inclusion.multiplication_table[inverse_index][left_index] != identity_index
            ):
                raise ValueError(f"transported inclusion {inclusion_index} inverse table fails")
    mandatory_noncommuting: set[AffineWord] = set()
    for left in range(len(action.affine_generators)):
        for right in range(left + 1, len(action.affine_generators)):
            forward = AffineWord(((left, 1), (right, 1)))
            reverse = AffineWord(((right, 1), (left, 1)))
            if evaluate_affine_word(
                action.affine_generators, forward
            ) != evaluate_affine_word(action.affine_generators, reverse):
                mandatory_noncommuting.update((forward, reverse))
    if not mandatory_noncommuting.issubset(set(certificate.roundtrip_words)):
        raise ValueError("certificate omits a mandatory noncommuting challenge")
    if len(set(certificate.roundtrip_words)) != len(certificate.roundtrip_words):
        raise ValueError("certificate contains duplicate round-trip challenges")
    for word_index, word in enumerate(certificate.roundtrip_words):
        expected = evaluate_affine_word(action.affine_generators, word)
        actual = _affine_identity()
        for generator_index, exponent in word.steps:
            if generator_index >= len(certificate.affine_generator_images):
                raise ValueError(f"roundtrip word {word_index} references an absent generator")
            generator_image = _evaluate_pcp_word(
                certificate.affine_generator_images[generator_index], authority
            )
            actual = _compose_affine(_power_affine(generator_image, exponent), actual)
        if actual != expected:
            raise ValueError(f"roundtrip word {word_index} fails")
        decode(actual)
    return True


def _request_core_mapping(value: GAPClassifierRequest) -> dict[str, Any]:
    return {
        "action": _action_mapping(value.action),
        "inclusions": [_inclusion_mapping(item) for item in value.inclusions],
        "max_degree": value.max_degree,
        "time_reversal": value.time_reversal,
    }


def _request_mapping(value: GAPClassifierRequest) -> dict[str, Any]:
    return {
        **_request_core_mapping(value),
        "protocol_version": _PROTOCOL_VERSION,
        "record_type": "gap-classifier-request",
        "request_digest": value.request_digest,
    }


def make_gap_classifier_request(
    action: CertifiedSpaceGroupAction,
    inclusions: Sequence[LiteralStabilizerInclusion],
    *,
    time_reversal: bool,
    max_degree: int = 4,
) -> GAPClassifierRequest:
    provisional = GAPClassifierRequest(
        "sha256:" + "0" * 64,
        action,
        tuple(inclusions),
        time_reversal,
        max_degree,
    )
    return GAPClassifierRequest(
        _domain_digest("gap-classifier-request-v1", _request_core_mapping(provisional)),
        provisional.action,
        provisional.inclusions,
        provisional.time_reversal,
        provisional.max_degree,
    )


def _strict_json_loads(data: bytes) -> Mapping[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError("GAP classifier loaders require bytes")

    def pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_without_duplicates,
            parse_float=lambda token: (_ for _ in ()).throw(ValueError("floating-point JSON is forbidden")),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError("non-finite JSON is forbidden")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict GAP classifier JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("GAP classifier record must be an object")
    if canonical_classification_json(value) != data:
        raise ValueError("GAP classifier JSON bytes are not canonical")
    return value


def _require_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{path}: missing field {missing[0]}")
    if unexpected:
        raise ValueError(f"{path}: unexpected field {unexpected[0]}")


def _parse_affine(value: Any, path: str) -> AffineTransformation:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(value, {"matrix", "translation"}, path)
    return AffineTransformation(
        tuple(tuple(row) for row in value["matrix"]), tuple(value["translation"])
    )


def _parse_word(value: Any, path: str) -> AffineWord:
    if not isinstance(value, list):
        raise TypeError(f"{path}: expected array")
    return AffineWord(tuple(tuple(step) for step in value))


def _parse_transported_inclusion(value: Any, path: str) -> TransportedInclusion:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(
        value,
        {
            "inclusion_id",
            "inverse_indices",
            "literal_element_digest",
            "literal_elements",
            "literal_stabilizer_digest",
            "multiplication_table",
            "pcp_images",
        },
        path,
    )
    return TransportedInclusion(
        value["inclusion_id"],
        value["literal_stabilizer_digest"],
        value["literal_element_digest"],
        tuple(
            _parse_affine(item, f"{path}.literal_elements[{index}]")
            for index, item in enumerate(value["literal_elements"])
        ),
        tuple(value["pcp_images"]),
        tuple(tuple(row) for row in value["multiplication_table"]),
        tuple(value["inverse_indices"]),
    )


def _parse_pcp_normal_form(value: Any, path: str) -> PCPNormalFormAuthority:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(value, {"generator_affines", "relative_orders"}, path)
    if not isinstance(value["generator_affines"], list) or not isinstance(
        value["relative_orders"], list
    ):
        raise TypeError(f"{path}: expected generator and relative-order arrays")
    return PCPNormalFormAuthority(
        tuple(value["relative_orders"]),
        tuple(
            _parse_affine(item, f"{path}.generator_affines[{index}]")
            for index, item in enumerate(value["generator_affines"])
        ),
    )


def _parse_certificate(value: Any, path: str) -> AffinePCPIsomorphismCertificate:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(
        value,
        {
            "affine_generator_images",
            "catalogue_action_digest",
            "certificate_digest",
            "conversion_algorithm_digest",
            "pcp_normal_form",
            "pcp_generator_preimages",
            "roundtrip_words",
            "translation_basis_images",
            "transported_stabilizers",
        },
        path,
    )
    certificate = AffinePCPIsomorphismCertificate(
        value["catalogue_action_digest"],
        value["conversion_algorithm_digest"],
        _parse_pcp_normal_form(value["pcp_normal_form"], f"{path}.pcp_normal_form"),
        tuple(value["affine_generator_images"]),
        tuple(
            _parse_word(item, f"{path}.pcp_generator_preimages[{index}]")
            for index, item in enumerate(value["pcp_generator_preimages"])
        ),
        tuple(value["translation_basis_images"]),
        tuple(
            _parse_transported_inclusion(
                item, f"{path}.transported_stabilizers[{index}]"
            )
            for index, item in enumerate(value["transported_stabilizers"])
        ),
        tuple(
            _parse_word(item, f"{path}.roundtrip_words[{index}]")
            for index, item in enumerate(value["roundtrip_words"])
        ),
        value["certificate_digest"],
    )
    if affine_pcp_certificate_digest(certificate) != certificate.certificate_digest:
        raise ValueError(f"{path}.certificate_digest: does not bind certificate payload")
    if (
        certificate.conversion_algorithm_digest
        != tracked_affine_pcp_conversion_digest()
    ):
        raise ValueError(f"{path}: does not bind the tracked affine-PCP sources")
    return certificate


_LOCKED_PACKAGE_VERSIONS = {
    "Cryst": "4.1.30",
    "GAP": "4.15.1",
    "HAP": "1.70",
    "HAPcryst": "0.1.15",
}

def _locked_environment_core() -> dict[str, Any]:
    """Return the reviewed protocol constants without an external lock asset.

    The standalone launcher independently probes the host executable and exact
    package versions.  These constants only replay the copied Task4/Task5 wire
    format and never select or launch a runtime.
    """

    packages = [
        {
            "archive_sha256": "sha256:90aae4bf7eabdb94bceebef0d984c8d6ea9e9c60d8268913498526565b693a7f",
            "license_sha256": "sha256:e9c68e5cf6425d8749ca7112dcd96049a25bfdf055c39ddf800456dc12353c01",
            "name": "Cryst",
            "version": "4.1.30",
        },
        {
            "archive_sha256": "sha256:2a81d008e1638f638a035b1cd981ca39436bdabbef8c29b15b24fceb2af678e4",
            "license_sha256": "sha256:8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643",
            "name": "GAP",
            "version": "4.15.1",
        },
        {
            "archive_sha256": "sha256:300e776141be73f807a2fbdfc0ce45d871c8d4a765dc2ca3b49ba38db9d51861",
            "license_sha256": "sha256:edaef632cbb643e4e7a221717a6c441a4c1a7c918e6e4d56debc3d8739b233f6",
            "name": "HAP",
            "version": "1.70",
        },
        {
            "archive_sha256": "sha256:dda392457ecc9fcffd7d86b3633da455e9fe65118d7bcf4039cc5d4d05edfc94",
            "license_sha256": "sha256:ab15fd526bd8dd18a9e77ebc139656bf4d33e97fc7238cd11bf60e2b9b8666c6",
            "name": "HAPcryst",
            "version": "0.1.15",
        },
    ]
    expected_apis = [
        "BarResolutionEquivalence",
        "EquivariantChainMap",
        "ResolutionAlmostCrystalGroup",
        "ResolutionDirectProduct",
        "ResolutionFiniteGroup",
    ]
    expected_task5_sources = []
    for name in (
        "bar_equivalence.g",
        "characters.g",
        "resolutions.g",
        "restrictions.g",
        "u1_relative.g",
    ):
        try:
            digest = hashlib.sha256(_classifier_source_bytes(name)).hexdigest()
        except OSError as error:
            raise ValueError(
                "tracked classifier Task 5 source closure is unavailable"
            ) from error
        expected_task5_sources.append({"path": name, "sha256": digest})
    return {
        "api_closure": expected_apis,
        "lock_digest": "sha256:c92c0cef1c72a061a642ccdbb297adafd52ffad2779a84755d9e626363edb25d",
        "oci_image_digest": "sha256:726b772a1aae0cfa22fd3cdba89bb424c65eed01744a265a0f55078649a2b95d",
        "packages": packages,
        "task5_source_closure": expected_task5_sources,
    }


def _environment_core_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_mode": value["execution_mode"],
        "lock_digest": value["lock_digest"],
        "oci_image_digest": value["oci_image_digest"],
        "packages": value["packages"],
        "release_certified": value["release_certified"],
        "runtime_provenance_digest": value["runtime_provenance_digest"],
    }


def _validate_environment(value: Any, path: str) -> FrozenJSONObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(
        value,
        {
            "environment_id",
            "execution_mode",
            "lock_digest",
            "oci_image_digest",
            "packages",
            "release_certified",
            "runtime_provenance_digest",
        },
        path,
    )
    _digest(value["environment_id"], f"{path}.environment_id")
    _digest(value["lock_digest"], f"{path}.lock_digest")
    _digest(value["oci_image_digest"], f"{path}.oci_image_digest")
    execution_mode = value["execution_mode"]
    if execution_mode not in ("diagnostic_local", "locked_image"):
        raise ValueError(f"{path}.execution_mode: unsupported classifier runtime")
    if value["release_certified"] is not False:
        raise ValueError(
            f"{path}.release_certified: Task 4 responses cannot be release-certified"
        )
    runtime_provenance_digest = value["runtime_provenance_digest"]
    if execution_mode == "diagnostic_local":
        if runtime_provenance_digest is not None:
            raise ValueError(
                f"{path}.runtime_provenance_digest: diagnostic runtime cannot claim provenance"
            )
    else:
        _digest(
            runtime_provenance_digest,
            f"{path}.runtime_provenance_digest",
        )
    packages = value["packages"]
    if not isinstance(packages, list) or len(packages) != len(_LOCKED_PACKAGE_VERSIONS):
        raise ValueError(f"{path}.packages: expected all locked components")
    seen: set[str] = set()
    for index, package in enumerate(packages):
        item_path = f"{path}.packages[{index}]"
        if not isinstance(package, Mapping):
            raise TypeError(f"{item_path}: expected object")
        _require_fields(
            package,
            {"archive_sha256", "license_sha256", "name", "version"},
            item_path,
        )
        name = package["name"]
        if name not in _LOCKED_PACKAGE_VERSIONS or name in seen:
            raise ValueError(f"{item_path}.name: unexpected or duplicate package")
        seen.add(name)
        if package["version"] != _LOCKED_PACKAGE_VERSIONS[name]:
            raise ValueError(f"{item_path}.version: locked version mismatch")
        _digest(package["archive_sha256"], f"{item_path}.archive_sha256")
        _digest(package["license_sha256"], f"{item_path}.license_sha256")
    if tuple(package["name"] for package in packages) != tuple(sorted(seen)):
        raise ValueError(f"{path}.packages: packages must be sorted by name")
    expected = _domain_digest("classifier-environment-v1", _environment_core_mapping(value))
    if value["environment_id"] != expected:
        raise ValueError(f"{path}.environment_id: does not bind environment payload")
    declared_environment = {
        key: value[key]
        for key in ("lock_digest", "oci_image_digest", "packages")
    }
    locked_environment = _locked_environment_core()
    if declared_environment != {
        key: locked_environment[key]
        for key in ("lock_digest", "oci_image_digest", "packages")
    }:
        raise ValueError(f"{path}: does not match the locked classifier environment")
    return _freeze_object(value)


def _parse_failure(value: Any, path: str) -> StructuredFailure:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected object")
    _require_fields(value, {"code", "context", "message", "stage"}, path)
    if not isinstance(value["context"], Mapping):
        raise TypeError(f"{path}.context: expected object")
    return StructuredFailure(
        value["code"], value["stage"], value["message"], value["context"]
    )


def loads_gap_classifier_request(data: bytes) -> GAPClassifierRequest:
    value = _strict_json_loads(data)
    _require_fields(
        value,
        {
            "action",
            "inclusions",
            "max_degree",
            "protocol_version",
            "record_type",
            "request_digest",
            "time_reversal",
        },
        "$request",
    )
    if (
        isinstance(value["protocol_version"], bool)
        or value["protocol_version"] != _PROTOCOL_VERSION
    ):
        raise ValueError("$request.protocol_version: unsupported version")
    if value["record_type"] != "gap-classifier-request":
        raise ValueError("$request.record_type: unexpected record type")
    action_row = value["action"]
    if not isinstance(action_row, Mapping):
        raise TypeError("$request.action: expected object")
    _require_fields(
        action_row,
        {"action_digest", "affine_generators", "translation_basis"},
        "$request.action",
    )
    action = make_certified_space_group_action(
        tuple(
            _parse_affine(item, f"$request.action.affine_generators[{index}]")
            for index, item in enumerate(action_row["affine_generators"])
        ),
        action_row["translation_basis"],
    )
    if action.action_digest != _digest(action_row["action_digest"], "$request.action.action_digest"):
        raise ValueError("$request.action.action_digest: does not bind exact affine action")
    inclusions: list[LiteralStabilizerInclusion] = []
    for index, item in enumerate(value["inclusions"]):
        path = f"$request.inclusions[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{path}: expected object")
        _require_fields(
            item,
            {
                "inclusion_id",
                "literal_element_digest",
                "literal_elements",
                "literal_stabilizer_digest",
            },
            path,
        )
        inclusions.append(
            LiteralStabilizerInclusion(
                item["inclusion_id"],
                item["literal_stabilizer_digest"],
                item["literal_element_digest"],
                tuple(
                    _parse_affine(element, f"{path}.literal_elements[{element_index}]")
                    for element_index, element in enumerate(item["literal_elements"])
                ),
            )
        )
    request = GAPClassifierRequest(
        _digest(value["request_digest"], "$request.request_digest"),
        action,
        tuple(inclusions),
        value["time_reversal"],
        value["max_degree"],
    )
    expected = _domain_digest("gap-classifier-request-v1", _request_core_mapping(request))
    if request.request_digest != expected:
        raise ValueError("$request.request_digest: does not bind request payload")
    return request


def _failure_mapping(value: StructuredFailure) -> dict[str, Any]:
    return {
        "code": value.code,
        "context": _thaw_json(value.context),
        "message": value.message,
        "stage": value.stage,
    }


def _response_mapping(value: GAPClassifierResponse) -> dict[str, Any]:
    return {
        "affine_pcp_certificate": (
            None
            if value.affine_pcp_certificate is None
            else _certificate_mapping(value.affine_pcp_certificate)
        ),
        "environment": None if value.environment is None else _thaw_json(value.environment),
        "failures": [_failure_mapping(item) for item in value.failures],
        "problem": None if value.problem is None else _thaw_json(value.problem),
        "protocol_version": value.protocol_version,
        "record_type": "gap-classifier-response",
        "request_digest": value.request_digest,
        "status": value.status,
    }


def loads_gap_classifier_response(data: bytes) -> GAPClassifierResponse:
    value = _strict_json_loads(data)
    _require_fields(
        value,
        {
            "affine_pcp_certificate",
            "environment",
            "failures",
            "problem",
            "protocol_version",
            "record_type",
            "request_digest",
            "status",
        },
        "$response",
    )
    version = value["protocol_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != _PROTOCOL_VERSION:
        raise ValueError("$response.protocol_version: unsupported version")
    if value["record_type"] != "gap-classifier-response":
        raise ValueError("$response.record_type: unexpected record type")
    status = value["status"]
    if status == "success":
        raise ValueError(
            "$response.status: success requires an authoritative Task 5 problem"
        )
    certificate = (
        None
        if value["affine_pcp_certificate"] is None
        else _parse_certificate(value["affine_pcp_certificate"], "$response.affine_pcp_certificate")
    )
    environment = (
        None
        if value["environment"] is None
        else _validate_environment(value["environment"], "$response.environment")
    )
    if value["problem"] is not None:
        raise ValueError(
            "$response.problem: Task 4 responses cannot contain an authoritative Task 5 problem"
        )
    problem = None
    failures_value = value["failures"]
    if not isinstance(failures_value, list):
        raise TypeError("$response.failures: expected array")
    return GAPClassifierResponse(
        _digest(value["request_digest"], "$response.request_digest"),
        status,
        environment,
        certificate,
        problem,
        tuple(
            _parse_failure(item, f"$response.failures[{index}]")
            for index, item in enumerate(failures_value)
        ),
        version,
    )


def default_gap_classifier_command() -> tuple[str, ...]:
    exporter = Path(__file__).resolve().parents[1] / "gap/classifier/export_problem.g"
    return (
        "gap",
        "-q",
        os.fspath(exporter),
        "--",
    )


def _error_response(
    request: GAPClassifierRequest,
    code: str,
    stage: str,
    message: str,
    *,
    context: FrozenJSONObject | None = None,
) -> GAPClassifierResponse:
    return GAPClassifierResponse(
        request.request_digest,
        "error",
        None,
        None,
        None,
        (
            StructuredFailure(
                code,
                stage,
                message,
                FrozenJSONObject(()) if context is None else context,
            ),
        ),
    )


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a sequence of argument strings")
    result = tuple(command)
    if not result:
        raise ValueError("command must not be empty")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("command arguments must be nonempty strings")
    return result


def _validate_response_for_request(
    request: GAPClassifierRequest, response: GAPClassifierResponse
) -> None:
    if response.request_digest != request.request_digest:
        raise ValueError("response request digest mismatch")
    if response.status == "error":
        return
    certificate = response.affine_pcp_certificate
    assert certificate is not None
    verify_affine_pcp_certificate(request.action, certificate)
    expected_inclusions = tuple(
        (
            item.inclusion_id,
            item.literal_stabilizer_digest,
            item.literal_element_digest,
            item.literal_elements,
        )
        for item in request.inclusions
    )
    actual_inclusions = tuple(
        (
            item.inclusion_id,
            item.literal_stabilizer_digest,
            item.literal_element_digest,
            item.literal_elements,
        )
        for item in certificate.transported_stabilizers
    )
    if actual_inclusions != expected_inclusions:
        raise ValueError("response does not transport every requested literal stabilizer")


class _DiagnosticPipeDrain:
    """Drain one backend pipe while retaining at most its declared byte cap."""

    __slots__ = (
        "_condition",
        "_limit",
        "_state",
        "_stream",
        "read_error",
        "retained",
        "total_bytes",
    )

    def __init__(
        self,
        stream: Any,
        limit: int,
        condition: threading.Condition,
        state: dict[str, Any],
    ) -> None:
        self._stream = stream
        self._limit = limit
        self._condition = condition
        self._state = state
        self.retained = bytearray()
        self.total_bytes = 0
        self.read_error = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(min(64 * 1024, self._limit + 1))
                if not chunk:
                    break
                with self._condition:
                    self.total_bytes = min(
                        self._limit + 1, self.total_bytes + len(chunk)
                    )
                    remaining = self._limit - len(self.retained)
                    if remaining > 0:
                        self.retained.extend(chunk[:remaining])
                    if self.total_bytes > self._limit:
                        if self._state["failure"] is None:
                            self._state["failure"] = "diagnostic_limit"
                        self._condition.notify_all()
                        return
        except (OSError, ValueError):
            with self._condition:
                self.read_error = True
                if self._state["failure"] is None:
                    self._state["failure"] = "diagnostic_read"
                self._condition.notify_all()
        finally:
            try:
                self._stream.close()
            except OSError:
                pass
            with self._condition:
                self._state["drains_done"] += 1
                self._condition.notify_all()


def _kill_backend_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_backend(
    process: subprocess.Popen[bytes],
    condition: threading.Condition,
    state: dict[str, Any],
) -> None:
    try:
        return_code = process.wait()
    except OSError:
        with condition:
            if state["failure"] is None:
                state["failure"] = "backend_wait"
            condition.notify_all()
        return
    with condition:
        state["return_code"] = return_code
        condition.notify_all()


def run_gap_classifier(
    request: GAPClassifierRequest,
    *,
    timeout_seconds: int | float = 300,
    max_response_bytes: int = 16 * 1024 * 1024,
    max_diagnostic_bytes: int = 1024 * 1024,
    command: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> GAPClassifierResponse:
    if not isinstance(request, GAPClassifierRequest):
        raise TypeError("request must be a GAPClassifierRequest")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    if (
        isinstance(max_diagnostic_bytes, bool)
        or not isinstance(max_diagnostic_bytes, int)
        or max_diagnostic_bytes <= 0
    ):
        raise ValueError("max_diagnostic_bytes must be a positive integer")
    argv = _validate_command(
        default_gap_classifier_command() if command is None else command
    )
    if environment is not None:
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a string mapping")
        if any(
            type(key) is not str
            or not key
            or type(value) is not str
            or "\x00" in key
            or "\x00" in value
            or "=" in key
            for key, value in environment.items()
        ):
            raise ValueError("environment must contain valid string entries")
        process_environment = dict(environment)
    else:
        process_environment = None
    with tempfile.TemporaryDirectory(prefix="mathpsg-classifier-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        response_path = root / "response.json"
        request_path.write_bytes(canonical_gap_classifier_json(request))
        try:
            process = subprocess.Popen(
                (*argv, "--request", str(request_path), "--response", str(response_path)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
                env=process_environment,
            )
        except OSError:
            return _error_response(
                request,
                "backend_failed",
                "backend",
                "optional locked GAP classifier backend is unavailable",
                context=FrozenJSONObject((("reason", "unavailable"),)),
            )
        assert process.stdout is not None and process.stderr is not None
        condition = threading.Condition()
        state: dict[str, Any] = {
            "drains_done": 0,
            "failure": None,
            "return_code": None,
        }
        drains = (
            _DiagnosticPipeDrain(
                process.stdout, max_diagnostic_bytes, condition, state
            ),
            _DiagnosticPipeDrain(
                process.stderr, max_diagnostic_bytes, condition, state
            ),
        )
        drain_threads = tuple(
            threading.Thread(target=drain.run, name=f"classifier-diagnostic-{index}")
            for index, drain in enumerate(drains)
        )
        waiter = threading.Thread(
            target=_wait_for_backend,
            args=(process, condition, state),
            name="classifier-backend-waiter",
        )
        for thread in drain_threads:
            thread.start()
        waiter.start()
        deadline = time.monotonic() + timeout_seconds
        with condition:
            while state["failure"] is None and not (
                state["return_code"] is not None and state["drains_done"] == 2
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state["failure"] = "timeout"
                    break
                condition.wait(remaining)
            failure = state["failure"]
        if failure is not None:
            _kill_backend_process_group(process)
        waiter.join(timeout=2)
        if waiter.is_alive():
            _kill_backend_process_group(process)
            waiter.join(timeout=2)
        for thread in drain_threads:
            thread.join(timeout=2)
        if any(thread.is_alive() for thread in drain_threads):
            _kill_backend_process_group(process)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            for thread in drain_threads:
                thread.join(timeout=2)
        if any(thread.is_alive() for thread in (*drain_threads, waiter)):
            return _error_response(
                request,
                "backend_failed",
                "backend",
                "classifier backend cleanup failed",
            )
        if failure == "timeout":
            return _error_response(
                request,
                "backend_timeout",
                "backend",
                "classifier backend timed out",
            )
        if failure == "diagnostic_limit":
            return _error_response(
                request,
                "backend_failed",
                "protocol",
                "classifier backend diagnostics exceed size limit",
            )
        if failure is not None:
            return _error_response(
                request,
                "backend_failed",
                "backend",
                "classifier backend failed",
            )
        return_code = state["return_code"]
        if return_code != 0:
            return _error_response(
                request, "backend_failed", "backend", "classifier backend failed"
            )
        try:
            size = response_path.stat().st_size
        except OSError:
            return _error_response(
                request, "backend_failed", "protocol", "classifier backend produced no response"
            )
        if size > max_response_bytes:
            return _error_response(
                request, "backend_failed", "protocol", "classifier response exceeds size limit"
            )
        try:
            with response_path.open("rb") as response_file:
                encoded = response_file.read(max_response_bytes + 1)
            if len(encoded) > max_response_bytes:
                return _error_response(
                    request,
                    "backend_failed",
                    "protocol",
                    "classifier response exceeds size limit",
                )
            response = loads_gap_classifier_response(encoded)
        except (OSError, TypeError, ValueError) as error:
            message = str(error)
            if "unsupported version" in message:
                code = "unsupported_schema"
            elif "absolute path" in message:
                code = "certificate_invalid"
            else:
                code = "backend_failed"
            return _error_response(request, code, "protocol", "classifier response is invalid")
        try:
            _validate_response_for_request(request, response)
        except (TypeError, ValueError):
            return _error_response(
                request,
                "certificate_invalid",
                "certificate",
                "classifier certificate failed verification",
            )
        return response


def canonical_gap_classifier_json(value: object) -> bytes:
    if isinstance(value, GAPClassifierRequest):
        return canonical_classification_json(_request_mapping(value))
    if isinstance(value, GAPClassifierResponse):
        return canonical_classification_json(_response_mapping(value))
    if isinstance(value, AffinePCPIsomorphismCertificate):
        return canonical_classification_json(_certificate_mapping(value))
    return canonical_classification_json(value)


__all__ = [
    "AffineTransformation",
    "AffineWord",
    "AffinePCPIsomorphismCertificate",
    "CertifiedSpaceGroupAction",
    "GAPClassifierRequest",
    "GAPClassifierResponse",
    "LiteralStabilizerInclusion",
    "PCPNormalFormAuthority",
    "TransportedInclusion",
    "affine_pcp_certificate_digest",
    "catalogue_record_authority_digest",
    "canonical_gap_classifier_json",
    "default_gap_classifier_command",
    "evaluate_affine_word",
    "loads_gap_classifier_request",
    "loads_gap_classifier_response",
    "literal_element_authority_digest",
    "literal_stabilizer_authority_digest",
    "make_affine_pcp_certificate",
    "make_certified_space_group_action",
    "make_gap_classifier_request",
    "run_gap_classifier",
    "tracked_affine_pcp_conversion_digest",
    "transported_inclusion_authority_digest",
    "verify_affine_pcp_certificate",
]
