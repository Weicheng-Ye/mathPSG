"""Small immutable algebra objects decoded from fresh GAP output.

This module is intentionally a *projection* of the Task 4/Task 5 JSON, not a
certificate format.  It retains only the tables, matrices and comparison maps
used by the physical PSG calculation.  In particular, it has no hashes,
provenance, cache identity, version fields, replay routines or exhaustive
validators.

The parsers accept ordinary JSON-like mappings and ignore fields that the
calculation does not consume.  They still convert the required shapes to
immutable Python values so malformed process output fails near the GAP
boundary instead of much later in linear algebra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import itertools
import re
from typing import Any

from .gf2 import GF2Character, MatrixGF2, kernel_basis
from .integer_linalg import MatrixZ


@dataclass(frozen=True, slots=True)
class FiniteGroupTable:
    """A finite group in the element order emitted by GAP."""

    group_id: str
    element_order: tuple[str, ...]
    identity_index: int
    multiplication_table: tuple[tuple[int, ...], ...]

    def index(self, element: str) -> int:
        return self.element_order.index(element)


@dataclass(frozen=True, slots=True, order=True)
class SparseGroupRingTerm:
    element: str
    coefficient: int


@dataclass(frozen=True, slots=True)
class SparseGroupRingEntry:
    row: int
    column: int
    terms: tuple[SparseGroupRingTerm, ...]


@dataclass(frozen=True, slots=True)
class SparseGroupRingMatrix:
    row_count: int
    column_count: int
    entries: tuple[SparseGroupRingEntry, ...]


@dataclass(frozen=True, slots=True)
class PCPNormalForm:
    """The Task-4 PCP information needed to evaluate coefficient signs."""

    relative_orders: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FreeResolution:
    """The part of a GAP/HAP free resolution used by ``compute``."""

    group_id: str
    basis: tuple[tuple[str, ...], ...]
    boundaries: tuple[SparseGroupRingMatrix, ...]
    finite_group: FiniteGroupTable | None = None
    pcp_normal_form: PCPNormalForm | None = None


@dataclass(frozen=True, slots=True, order=True)
class SparseBarTerm:
    left_element: str
    group_tuple: tuple[str, ...]
    coefficient: int


@dataclass(frozen=True, slots=True)
class SparseBarChain:
    terms: tuple[SparseBarTerm, ...]


@dataclass(frozen=True, slots=True, order=True)
class SparseResolutionTerm:
    basis_id: str
    element: str
    coefficient: int


@dataclass(frozen=True, slots=True)
class SparseResolutionChain:
    terms: tuple[SparseResolutionTerm, ...]


@dataclass(frozen=True, slots=True)
class ResolutionBasisImage:
    degree: int
    basis_id: str
    image: SparseBarChain


@dataclass(frozen=True, slots=True)
class BarPhiValue:
    group_tuple: tuple[str, ...]
    image: SparseResolutionChain


@dataclass(frozen=True, slots=True)
class BarResolutionEquivalence:
    """Only the comparison maps pulled back by the physical solver."""

    resolution: FreeResolution
    finite_group: FiniteGroupTable
    psi_on_basis: tuple[ResolutionBasisImage, ...]
    phi_on_queries: tuple[BarPhiValue, ...]

    def normalized_tuples(self, degree: int) -> tuple[tuple[str, ...], ...]:
        return tuple(
            itertools.product(self.finite_group.element_order[1:], repeat=degree)
        )


@dataclass(frozen=True, slots=True)
class InclusionAlgebra:
    """The Task-5 data required for one local-to-ambient inclusion."""

    inclusion_id: str
    source_resolution: FreeResolution
    target_resolution: FreeResolution
    source_element_images: tuple[str, ...]
    # Indexed by cochain degree.  GAP exports degrees one and two; the parser
    # supplies the canonical degree-zero augmentation map at index zero so the
    # solver can retain its compact degree-indexed representation.
    maps: tuple[SparseGroupRingMatrix, ...]
    bar_equivalence: BarResolutionEquivalence


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected an object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected an array")
    return value


def parse_finite_group(value: Any, path: str = "$finite_group") -> FiniteGroupTable:
    raw = _object(value, path)
    return FiniteGroupTable(
        str(raw["group_id"]),
        tuple(str(item) for item in _array(raw["element_order"], path + ".element_order")),
        int(raw["identity_index"]),
        tuple(
            tuple(int(item) for item in _array(row, path + ".multiplication_table[]"))
            for row in _array(raw["multiplication_table"], path + ".multiplication_table")
        ),
    )


def parse_group_ring_matrix(
    value: Any, path: str = "$matrix"
) -> SparseGroupRingMatrix:
    raw = _object(value, path)
    parsed_entries: list[SparseGroupRingEntry] = []
    for index, value_entry in enumerate(_array(raw["entries"], path + ".entries")):
        entry = _object(value_entry, f"{path}.entries[{index}]")
        terms: list[SparseGroupRingTerm] = []
        for term_index, value_term in enumerate(
            _array(entry["terms"], f"{path}.entries[{index}].terms")
        ):
            term = _array(
                value_term, f"{path}.entries[{index}].terms[{term_index}]"
            )
            if len(term) != 2:
                raise ValueError(
                    f"{path}.entries[{index}].terms[{term_index}]: "
                    "expected [coefficient, element]"
                )
            terms.append(SparseGroupRingTerm(str(term[1]), int(term[0])))
        parsed_entries.append(
            SparseGroupRingEntry(
                int(entry["row"]), int(entry["column"]), tuple(terms)
            )
        )
    return SparseGroupRingMatrix(
        int(raw["row_count"]), int(raw["column_count"]), tuple(parsed_entries)
    )


def parse_resolution(
    value: Any,
    *,
    group_id: str,
    finite_group: FiniteGroupTable | None = None,
    pcp_normal_form: PCPNormalForm | None = None,
    path: str = "$resolution",
) -> FreeResolution:
    raw = _object(value, path)
    basis = tuple(
        tuple(str(item) for item in _array(degree, path + ".basis[]"))
        for degree in _array(raw["basis"], path + ".basis")
    )
    boundaries = tuple(
        parse_group_ring_matrix(item, f"{path}.boundaries[{index}]")
        for index, item in enumerate(_array(raw["boundaries"], path + ".boundaries"))
    )
    return FreeResolution(
        str(group_id), basis, boundaries, finite_group, pcp_normal_form
    )


def parse_bar_chain(value: Any, *, degree: int, path: str) -> SparseBarChain:
    terms: list[SparseBarTerm] = []
    for index, value_term in enumerate(_array(value, path)):
        term = _object(value_term, f"{path}[{index}]")
        terms.append(
            SparseBarTerm(
                str(term["left_element"]),
                tuple(
                    str(item)
                    for item in _array(term["group_tuple"], f"{path}[{index}].group_tuple")
                ),
                int(term["coefficient"]),
            )
        )
    return SparseBarChain(tuple(terms))


def parse_resolution_chain(value: Any, path: str) -> SparseResolutionChain:
    raw = _object(value, path)
    terms: list[SparseResolutionTerm] = []
    for index, value_term in enumerate(_array(raw["terms"], path + ".terms")):
        term = _object(value_term, f"{path}.terms[{index}]")
        terms.append(
            SparseResolutionTerm(
                str(term["basis_id"]),
                str(term["element"]),
                int(term["coefficient"]),
            )
        )
    return SparseResolutionChain(tuple(terms))


def parse_bar_equivalence(
    value: Any,
    *,
    resolution: FreeResolution,
    path: str = "$bar_equivalence",
) -> BarResolutionEquivalence:
    raw = _object(value, path)
    table = resolution.finite_group
    if table is None:
        table = parse_finite_group(raw["finite_group"], path + ".finite_group")

    psi: list[ResolutionBasisImage] = []
    for degree, value_degree in enumerate(
        _array(raw["psi_on_basis"], path + ".psi_on_basis")
    ):
        for index, value_item in enumerate(
            _array(value_degree, f"{path}.psi_on_basis[{degree}]")
        ):
            item = _object(value_item, f"{path}.psi_on_basis[{degree}][{index}]")
            psi.append(
                ResolutionBasisImage(
                    degree,
                    str(item["basis_id"]),
                    parse_bar_chain(
                        item["image"],
                        degree=degree,
                        path=f"{path}.psi_on_basis[{degree}][{index}].image",
                    ),
                )
            )

    phi: list[BarPhiValue] = []
    for index, value_item in enumerate(
        _array(raw["phi_on_queries"], path + ".phi_on_queries")
    ):
        item = _object(value_item, f"{path}.phi_on_queries[{index}]")
        phi.append(
            BarPhiValue(
                tuple(
                    str(element)
                    for element in _array(
                        item["group_tuple"],
                        f"{path}.phi_on_queries[{index}].group_tuple",
                    )
                ),
                parse_resolution_chain(
                    item["image"], f"{path}.phi_on_queries[{index}].image"
                ),
            )
        )
    return BarResolutionEquivalence(resolution, table, tuple(psi), tuple(phi))


def parse_inclusion(
    value: Any,
    *,
    inclusion_id: str,
    target_group_id: str,
    pcp_normal_form: PCPNormalForm,
    path: str = "$inclusion",
) -> InclusionAlgebra:
    """Project one raw Task-5 member into the data used by ``compute``."""

    raw = _object(value, path)
    table = parse_finite_group(raw["finite_group"], path + ".finite_group")
    source = parse_resolution(
        raw["source"],
        group_id=table.group_id,
        finite_group=table,
        path=path + ".source",
    )
    target = parse_resolution(
        raw["target"],
        group_id=target_group_id,
        pcp_normal_form=pcp_normal_form,
        path=path + ".target",
    )
    equivalence = parse_bar_equivalence(
        raw["bar_equivalence"], resolution=source, path=path + ".bar_equivalence"
    )
    raw_maps = _array(raw["restriction_maps"], path + ".restriction_maps")
    restriction_maps = tuple(
        parse_group_ring_matrix(item, f"{path}.restriction_maps[{index}]")
        for index, item in enumerate(raw_maps, start=1)
    )
    degree_zero = SparseGroupRingMatrix(
        1,
        1,
        (
            SparseGroupRingEntry(
                0,
                0,
                (SparseGroupRingTerm("1", 1),),
            ),
        ),
    )
    maps = (degree_zero,) + restriction_maps
    return InclusionAlgebra(
        str(inclusion_id),
        source,
        target,
        tuple(
            str(item)
            for item in _array(
                raw["source_element_images"], path + ".source_element_images"
            )
        ),
        maps,
        equivalence,
    )


_PCP_FACTOR = re.compile(r"p([1-9][0-9]*)(?:\^(-?[0-9]+))?")


def pcp_word_coordinates(word: str, generator_count: int) -> tuple[int, ...]:
    """Decode the simple GAP PCP normal words used in Task-5 matrices.

    Whitespace, explicit exponent one, and repeated factors are accepted; the
    exponents are accumulated.  This is intentionally less restrictive than
    the former canonical-spelling parser.
    """

    text = word.strip()
    if text in {"", "1"}:
        return (0,) * generator_count
    coordinates = [0] * generator_count
    for raw_factor in text.split("*"):
        factor = raw_factor.strip()
        match = _PCP_FACTOR.fullmatch(factor)
        if match is None:
            raise ValueError(f"unsupported PCP factor {factor!r}")
        index = int(match.group(1)) - 1
        if not 0 <= index < generator_count:
            raise ValueError(f"PCP generator p{index + 1} is outside the presentation")
        exponent = 1 if match.group(2) is None else int(match.group(2))
        coordinates[index] += exponent
    return tuple(coordinates)


def word_character(
    resolution: FreeResolution, character: Sequence[int], element: str
) -> int:
    """Evaluate a previously selected Z2 character on a group element."""

    bits = tuple(int(bit) & 1 for bit in character)
    if resolution.finite_group is not None:
        return bits[resolution.finite_group.index(element)]
    normal_form = resolution.pcp_normal_form
    if normal_form is None:
        raise ValueError("ambient resolution lacks PCP relative-order data")
    graded = resolution.group_id.endswith("+onsite-T")
    text = element.strip()
    time_bit = 0
    if graded:
        if text == "T":
            text, time_bit = "1", 1
        elif text.endswith("+T"):
            text, time_bit = text[:-2], 1
    coordinates = pcp_word_coordinates(text, len(normal_form.relative_orders))
    value = sum(bit * exponent for bit, exponent in zip(bits, coordinates))
    if graded and time_bit:
        value += bits[len(coordinates)]
    return value & 1


def twist_group_ring_matrix(
    matrix: SparseGroupRingMatrix,
    resolution: FreeResolution,
    character: Sequence[int],
) -> MatrixZ:
    """Evaluate a group-ring matrix in the one-dimensional sign module."""

    dense = [[0] * matrix.column_count for _ in range(matrix.row_count)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] += sum(
            term.coefficient
            * (-1 if word_character(resolution, character, term.element) else 1)
            for term in entry.terms
        )
    return MatrixZ(tuple(tuple(row) for row in dense), column_count=matrix.column_count)


def group_ring_matrix_mod2_transpose(matrix: SparseGroupRingMatrix) -> MatrixGF2:
    """Apply the trivial augmentation modulo two and transpose for cochains."""

    dense = [[0] * matrix.row_count for _ in range(matrix.column_count)]
    for entry in matrix.entries:
        dense[entry.column][entry.row] ^= sum(
            term.coefficient for term in entry.terms
        ) & 1
    return MatrixGF2(
        tuple(tuple(row) for row in dense), column_count=matrix.row_count
    )


def enumerate_characters(resolution: FreeResolution) -> tuple[GF2Character, ...]:
    """Enumerate ``Hom(G, Z2)`` from the degree-two resolution boundary.

    A sign character is a degree-one cocycle with trivial Z2 coefficients.
    The degree-one HAP basis is in the same PCP-generator order used by the
    emitted normal words, so the resulting bit vectors can be passed directly
    to :func:`word_character`.
    """

    if len(resolution.boundaries) < 2:
        raise ValueError("resolution does not contain its degree-two boundary")
    basis = kernel_basis(group_ring_matrix_mod2_transpose(resolution.boundaries[1]))
    characters: list[GF2Character] = []
    for coefficients in itertools.product((0, 1), repeat=len(basis)):
        characters.append(
            GF2Character(
                tuple(
                    sum(
                        coefficient * vector[index]
                        for coefficient, vector in zip(coefficients, basis)
                    )
                    & 1
                    for index in range(len(resolution.basis[1]))
                )
            )
        )
    return tuple(characters)


def bar_chain_cochain_value(
    equivalence: BarResolutionEquivalence,
    cocycle: Mapping[tuple[str, ...], Fraction],
    chain: SparseBarChain,
    character: Sequence[int],
) -> Fraction:
    """Evaluate a normalized bar cochain on a sparse comparison image."""

    return sum(
        (
            Fraction(term.coefficient)
            * (-1 if word_character(equivalence.resolution, character, term.left_element) else 1)
            * cocycle.get(term.group_tuple, Fraction(0))
            for term in chain.terms
        ),
        Fraction(0),
    )


__all__ = [
    "BarPhiValue",
    "BarResolutionEquivalence",
    "FiniteGroupTable",
    "FreeResolution",
    "InclusionAlgebra",
    "PCPNormalForm",
    "ResolutionBasisImage",
    "SparseBarChain",
    "SparseBarTerm",
    "SparseGroupRingEntry",
    "SparseGroupRingMatrix",
    "SparseGroupRingTerm",
    "SparseResolutionChain",
    "SparseResolutionTerm",
    "bar_chain_cochain_value",
    "enumerate_characters",
    "group_ring_matrix_mod2_transpose",
    "parse_bar_equivalence",
    "parse_finite_group",
    "parse_group_ring_matrix",
    "parse_inclusion",
    "parse_resolution",
    "pcp_word_coordinates",
    "twist_group_ring_matrix",
    "word_character",
]
