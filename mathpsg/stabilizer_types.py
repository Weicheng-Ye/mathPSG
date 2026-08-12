"""Canonical abstract stabilizer types for the three-dimensional Wyckoff atlas.

The persisted v1 inventory is intentionally usable before the release catalogue
exists, but it is not release-certified in that state.  Every occurrence keeps
the complete literal multiplication table and an explicit isomorphism from a
stable abstract table; a later catalogue binding must replay byte-for-byte.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Hashable

from ._resources import asset_bytes
from .catalogue import normalize_gap_export
from .catalogue_schema import CatalogueRecord, canonical_json
from .cochains import FiniteGroupTable


STABILIZER_TYPE_IDS = (
    "C1",
    "C2",
    "C3",
    "C4",
    "C6",
    "C2xC2",
    "C2xC2xC2",
    "C4xC2",
    "C6xC2",
    "D3",
    "D4",
    "D6",
    "D4xC2",
    "D6xC2",
    "A4",
    "A4xC2",
    "S4",
    "S4xC2",
)

_LIBRARY_ASSET_PREFIX = "data/stabilizers/v1/"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RATIONAL_RE = re.compile(r"q\((-?(?:0|[1-9][0-9]*)),([1-9][0-9]*)\)\Z")
_PRESENTATIONS = {
    "C1": "< | >",
    "C2": "<a | a^2=1>",
    "C3": "<a | a^3=1>",
    "C4": "<a | a^4=1>",
    "C6": "<a | a^6=1>",
    "C2xC2": "<a,b | a^2=b^2=[a,b]=1>",
    "C2xC2xC2": "<a,b,c | a^2=b^2=c^2=[a,b]=[a,c]=[b,c]=1>",
    "C4xC2": "<a,b | a^4=b^2=[a,b]=1>",
    "C6xC2": "<a,b | a^6=b^2=[a,b]=1>",
    "D3": "<r,s | r^3=s^2=1,srs=r^-1>",
    "D4": "<r,s | r^4=s^2=1,srs=r^-1>",
    "D6": "<r,s | r^6=s^2=1,srs=r^-1>",
    "D4xC2": "<r,s,z | r^4=s^2=z^2=1,srs=r^-1,[z,r]=[z,s]=1>",
    "D6xC2": "<r,s,z | r^6=s^2=z^2=1,srs=r^-1,[z,r]=[z,s]=1>",
    "A4": "<(123),(12)(34)>",
    "A4xC2": "<(123),(12)(34),z | z^2=1,z central>",
    "S4": "<(1234),(12)>",
    "S4xC2": "<(1234),(12),z | z^2=1,z central>",
}

_MANIFEST_FIELDS = {
    "catalogue_manifest_digest",
    "coverage",
    "diagnostic_source",
    "files",
    "inventory",
    "library_version",
    "release_certified",
    "schema_version",
}
_MANIFEST_FILE_ORDER = (
    "types.ndjson",
    "z2-spatial-skeletons.ndjson",
    "z2-graded-skeletons.ndjson",
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _strict_json(data: bytes) -> Any:
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
            parse_float=lambda token: (_ for _ in ()).throw(ValueError("floating JSON is forbidden")),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError("non-finite JSON is forbidden")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict JSON") from error
    return value


def _canonical_line(value: object) -> bytes:
    return canonical_json(value) + b"\n"


def _fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{context}: missing field {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"{context}: unexpected field {sorted(extra)[0]}")


def _safe_read_regular(path: Path, context: str) -> bytes:
    """Read one regular file without following a final-component symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(f"{context}: file is unavailable or unsafe") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{context}: expected a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _permutation_compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _permutation_parity(value: tuple[int, ...]) -> int:
    return sum(
        value[left] > value[right]
        for left in range(len(value))
        for right in range(left + 1, len(value))
    ) & 1


def _permutations_four() -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(4), repeat=4))  # type: ignore[return-value]


def _all_permutations_four() -> tuple[tuple[int, ...], ...]:
    from itertools import permutations

    return tuple(permutations(range(4)))


def _element_names(order: int) -> tuple[str, ...]:
    return ("1",) + tuple(f"g{index}" for index in range(1, order))


def _table_from_elements(
    group_id: str,
    elements: Sequence[Hashable],
    multiply: Callable[[Hashable, Hashable], Hashable],
) -> FiniteGroupTable:
    values = tuple(elements)
    if not values:
        raise ValueError("finite group requires an identity")
    index = {value: position for position, value in enumerate(values)}
    if len(index) != len(values):
        raise ValueError("finite group enumeration contains duplicates")
    try:
        table = tuple(tuple(index[multiply(left, right)] for right in values) for left in values)
    except KeyError as error:
        raise ValueError("finite group enumeration is not closed") from error
    identity = next(
        position
        for position in range(len(values))
        if all(table[position][other] == other and table[other][position] == other for other in range(len(values)))
    )
    if identity != 0:
        raise ValueError("canonical group enumerations must put identity first")
    inverses = tuple(
        next(
            right
            for right in range(len(values))
            if table[left][right] == identity and table[right][left] == identity
        )
        for left in range(len(values))
    )
    return FiniteGroupTable(group_id, _element_names(len(values)), identity, table, inverses)


def _cyclic(n: int, type_id: str) -> tuple[FiniteGroupTable, tuple[int, ...]]:
    values = tuple(range(n))
    table = _table_from_elements(type_id, values, lambda left, right: (int(left) + int(right)) % n)
    return table, (() if n == 1 else (1,))


def _abelian(factors: tuple[int, ...], type_id: str) -> tuple[FiniteGroupTable, tuple[int, ...]]:
    values = tuple(product(*(range(factor) for factor in factors)))

    def multiply(left: Hashable, right: Hashable) -> Hashable:
        a = left  # type: ignore[assignment]
        b = right  # type: ignore[assignment]
        return tuple((a[index] + b[index]) % factors[index] for index in range(len(factors)))

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    generators = tuple(index[tuple(int(axis == component) for axis in range(len(factors)))] for component in range(len(factors)))
    return table, generators


def _dihedral(n: int, type_id: str, *, central_c2: bool = False) -> tuple[FiniteGroupTable, tuple[int, ...]]:
    values = tuple(
        (rotation, reflection, central)
        for central in range(2 if central_c2 else 1)
        for reflection in range(2)
        for rotation in range(n)
    )

    def multiply(left: Hashable, right: Hashable) -> Hashable:
        a, b, z = left  # type: ignore[misc]
        c, d, w = right  # type: ignore[misc]
        return ((a + (-1 if b else 1) * c) % n, (b + d) % 2, (z + w) % 2)

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    generators = [index[(1, 0, 0)], index[(0, 1, 0)]]
    if central_c2:
        generators.append(index[(0, 0, 1)])
    return table, tuple(generators)


def _permutation_group(type_id: str, *, alternating: bool, central_c2: bool) -> tuple[FiniteGroupTable, tuple[int, ...]]:
    identity = (0, 1, 2, 3)
    permutations = tuple(
        value
        for value in _all_permutations_four()
        if not alternating or _permutation_parity(value) == 0
    )
    permutations = (identity,) + tuple(value for value in permutations if value != identity)
    values = tuple(
        (permutation, central)
        for central in range(2 if central_c2 else 1)
        for permutation in permutations
    )

    def multiply(left: Hashable, right: Hashable) -> Hashable:
        a, z = left  # type: ignore[misc]
        b, w = right  # type: ignore[misc]
        return (_permutation_compose(a, b), (z + w) % 2)

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    if alternating:
        first = (1, 2, 0, 3)
        second = (1, 0, 3, 2)
    else:
        first = (1, 2, 3, 0)
        second = (1, 0, 2, 3)
    generators = [index[(first, 0)], index[(second, 0)]]
    if central_c2:
        generators.append(index[(identity, 1)])
    return table, tuple(generators)


@lru_cache(maxsize=1)
def _canonical_specs() -> tuple[tuple[str, FiniteGroupTable, tuple[int, ...]], ...]:
    builders = {
        "C1": lambda: _cyclic(1, "C1"),
        "C2": lambda: _cyclic(2, "C2"),
        "C3": lambda: _cyclic(3, "C3"),
        "C4": lambda: _cyclic(4, "C4"),
        "C6": lambda: _cyclic(6, "C6"),
        "C2xC2": lambda: _abelian((2, 2), "C2xC2"),
        "C2xC2xC2": lambda: _abelian((2, 2, 2), "C2xC2xC2"),
        "C4xC2": lambda: _abelian((4, 2), "C4xC2"),
        "C6xC2": lambda: _abelian((6, 2), "C6xC2"),
        "D3": lambda: _dihedral(3, "D3"),
        "D4": lambda: _dihedral(4, "D4"),
        "D6": lambda: _dihedral(6, "D6"),
        "D4xC2": lambda: _dihedral(4, "D4xC2", central_c2=True),
        "D6xC2": lambda: _dihedral(6, "D6xC2", central_c2=True),
        "A4": lambda: _permutation_group("A4", alternating=True, central_c2=False),
        "A4xC2": lambda: _permutation_group("A4xC2", alternating=True, central_c2=True),
        "S4": lambda: _permutation_group("S4", alternating=False, central_c2=False),
        "S4xC2": lambda: _permutation_group("S4xC2", alternating=False, central_c2=True),
    }
    return tuple((type_id, *builders[type_id]()) for type_id in STABILIZER_TYPE_IDS)


def canonical_stabilizer_table(type_id: str) -> FiniteGroupTable:
    for candidate, table, _ in _canonical_specs():
        if candidate == type_id:
            return table
    raise ValueError("unknown stabilizer type")


def canonical_generators(type_id: str) -> tuple[int, ...]:
    for candidate, _, generators in _canonical_specs():
        if candidate == type_id:
            return generators
    raise ValueError("unknown stabilizer type")


def _element_orders(table: Sequence[Sequence[int]], identity: int) -> tuple[int, ...]:
    order = len(table)
    result: list[int] = []
    for element in range(order):
        value = identity
        for exponent in range(1, order + 1):
            value = table[value][element]
            if value == identity:
                result.append(exponent)
                break
        else:
            raise ValueError("multiplication table element has no finite order")
    return tuple(result)


def _validate_literal_table(value: Sequence[Sequence[int]]) -> tuple[tuple[tuple[int, ...], ...], int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError("literal multiplication table must be a nonempty sequence")
    table = tuple(tuple(row) for row in value)
    order = len(table)
    if any(len(row) != order for row in table):
        raise ValueError("literal multiplication table must be square")
    if any(type(item) is not int or not 0 <= item < order for row in table for item in row):
        raise ValueError("literal multiplication table contains an invalid index")
    identities = tuple(
        candidate
        for candidate in range(order)
        if all(table[candidate][other] == other and table[other][candidate] == other for other in range(order))
    )
    if len(identities) != 1:
        raise ValueError("literal multiplication table has no unique identity")
    identity = identities[0]
    for left in range(order):
        if not any(table[left][right] == identity and table[right][left] == identity for right in range(order)):
            raise ValueError("literal multiplication table lacks an inverse")
        for middle in range(order):
            for right in range(order):
                if table[table[left][middle]][right] != table[left][table[middle][right]]:
                    raise ValueError("literal multiplication table is not associative")
    return table, identity


def _group_signature(table: tuple[tuple[int, ...], ...], identity: int) -> tuple[object, ...]:
    element_orders = _element_orders(table, identity)
    abelian = all(table[left][right] == table[right][left] for left in range(len(table)) for right in range(len(table)))
    center = sum(
        all(table[element][other] == table[other][element] for other in range(len(table)))
        for element in range(len(table))
    )
    return len(table), abelian, center, tuple(sorted(Counter(element_orders).items()))


def _generator_words(table: FiniteGroupTable, generators: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    words: list[tuple[int, ...] | None] = [None] * len(table.element_order)
    words[table.identity_index] = ()
    queue = deque((table.identity_index,))
    while queue:
        element = queue.popleft()
        assert words[element] is not None
        for generator_position, generator in enumerate(generators):
            target = table.multiplication_table[element][generator]
            if words[target] is None:
                words[target] = words[element] + (generator_position,)
                queue.append(target)
    if any(word is None for word in words):
        raise ArithmeticError("declared canonical generators do not generate the type")
    return tuple(word for word in words if word is not None)


def _isomorphism_witness(
    canonical: FiniteGroupTable,
    generators: tuple[int, ...],
    literal: tuple[tuple[int, ...], ...],
    literal_identity: int,
) -> tuple[int, ...] | None:
    canonical_orders = _element_orders(canonical.multiplication_table, canonical.identity_index)
    literal_orders = _element_orders(literal, literal_identity)
    canonical_central = tuple(
        all(canonical.multiplication_table[element][other] == canonical.multiplication_table[other][element] for other in range(len(literal)))
        for element in range(len(literal))
    )
    literal_central = tuple(
        all(literal[element][other] == literal[other][element] for other in range(len(literal)))
        for element in range(len(literal))
    )
    words = _generator_words(canonical, generators)
    candidate_sets = tuple(
        tuple(
            element
            for element in range(len(literal))
            if literal_orders[element] == canonical_orders[generator]
            and literal_central[element] == canonical_central[generator]
        )
        for generator in generators
    )
    if not generators:
        return (literal_identity,) if len(literal) == 1 else None
    valid: list[tuple[int, ...]] = []
    for images in product(*candidate_sets):
        mapping: list[int] = []
        for word in words:
            image = literal_identity
            for generator_position in word:
                image = literal[image][images[generator_position]]
            mapping.append(image)
        witness = tuple(mapping)
        if len(set(witness)) != len(literal):
            continue
        if all(
            witness[canonical.multiplication_table[left][right]] == literal[witness[left]][witness[right]]
            for left in range(len(literal))
            for right in range(len(literal))
        ):
            valid.append(witness)
    return min(valid) if valid else None


@dataclass(frozen=True, slots=True)
class StabilizerTypeIdentification:
    type_id: str
    canonical_to_literal: tuple[int, ...]


def identify_stabilizer_type(value: Sequence[Sequence[int]]) -> StabilizerTypeIdentification:
    table, identity = _validate_literal_table(value)
    signature = _group_signature(table, identity)
    candidates = tuple(
        (type_id, canonical, generators)
        for type_id, canonical, generators in _canonical_specs()
        if _group_signature(canonical.multiplication_table, canonical.identity_index) == signature
    )
    matches: list[StabilizerTypeIdentification] = []
    for type_id, canonical, generators in candidates:
        witness = _isomorphism_witness(canonical, generators, table, identity)
        if witness is not None:
            matches.append(StabilizerTypeIdentification(type_id, witness))
    if len(matches) != 1:
        raise ValueError("local_library_incomplete: stabilizer table has no unique v1 abstract type")
    return matches[0]


@dataclass(frozen=True, slots=True)
class StabilizerOccurrenceWitness:
    wyckoff_id: str
    international_number: int
    setting: str
    literal_table_digest: str
    literal_multiplication_table: tuple[tuple[int, ...], ...]
    canonical_to_literal: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.wyckoff_id) is not str or not self.wyckoff_id:
            raise ValueError("occurrence requires a stable Wyckoff ID")
        if type(self.international_number) is not int or not 1 <= self.international_number <= 230:
            raise ValueError("occurrence International number must be in 1..230")
        if type(self.setting) is not str or not self.setting:
            raise ValueError("occurrence setting must be nonempty")
        table, _ = _validate_literal_table(self.literal_multiplication_table)
        expected = _sha256(canonical_json([list(row) for row in table]))
        if self.literal_table_digest != expected:
            raise ValueError("occurrence literal table digest does not replay")
        if isinstance(self.canonical_to_literal, (str, bytes)) or not isinstance(
            self.canonical_to_literal, Sequence
        ):
            raise TypeError("occurrence isomorphism witness must be an exact-int sequence")
        witness = tuple(self.canonical_to_literal)
        if any(type(item) is not int for item in witness):
            raise TypeError("occurrence isomorphism witness requires exact int entries")
        if sorted(witness) != list(range(len(table))):
            raise ValueError("occurrence isomorphism witness is not a permutation")
        object.__setattr__(self, "literal_multiplication_table", table)
        object.__setattr__(self, "canonical_to_literal", witness)


@dataclass(frozen=True, slots=True)
class StabilizerTypeRecord:
    type_id: str
    presentation: str
    table: FiniteGroupTable
    occurrence_count: int
    occurrences: tuple[StabilizerOccurrenceWitness, ...]

    def __post_init__(self) -> None:
        if self.type_id not in STABILIZER_TYPE_IDS or self.table.group_id != self.type_id:
            raise ValueError("stabilizer type record does not bind its canonical table")
        canonical_table = canonical_stabilizer_table(self.type_id)
        if _table_mapping(self.table) != _table_mapping(canonical_table):
            raise ValueError("stabilizer type record table differs from the built-in canonical table")
        if self.presentation != _PRESENTATIONS[self.type_id]:
            raise ValueError("stabilizer presentation is not canonical")
        occurrences = tuple(self.occurrences)
        if not occurrences:
            raise ValueError("stabilizer type record requires nonempty occurrences")
        if any(not isinstance(item, StabilizerOccurrenceWitness) for item in occurrences):
            raise TypeError("stabilizer occurrences must be occurrence witnesses")
        if type(self.occurrence_count) is not int or self.occurrence_count != len(occurrences):
            raise ValueError("stabilizer occurrence count does not replay")
        canonical = self.table.multiplication_table
        for occurrence in occurrences:
            witness = occurrence.canonical_to_literal
            literal = occurrence.literal_multiplication_table
            if len(witness) != len(canonical) or any(
                witness[canonical[left][right]] != literal[witness[left]][witness[right]]
                for left in range(len(canonical))
                for right in range(len(canonical))
            ):
                raise ValueError("stabilizer occurrence table isomorphism does not replay")
        object.__setattr__(self, "occurrences", occurrences)


def _table_mapping(table: FiniteGroupTable) -> dict[str, Any]:
    return {
        "element_order": list(table.element_order),
        "group_id": table.group_id,
        "identity_index": table.identity_index,
        "inverse_indices": list(table.inverse_indices),
        "multiplication_table": [list(row) for row in table.multiplication_table],
        "table_digest": table.table_digest,
    }


def _occurrence_mapping(value: StabilizerOccurrenceWitness) -> dict[str, Any]:
    return {
        "canonical_to_literal": list(value.canonical_to_literal),
        "international_number": value.international_number,
        "literal_multiplication_table": [list(row) for row in value.literal_multiplication_table],
        "literal_table_digest": value.literal_table_digest,
        "setting": value.setting,
        "wyckoff_id": value.wyckoff_id,
    }


def _record_mapping(value: StabilizerTypeRecord) -> dict[str, Any]:
    return {
        "occurrence_count": value.occurrence_count,
        "occurrences": [_occurrence_mapping(item) for item in value.occurrences],
        "presentation": value.presentation,
        "record_type": "stabilizer-type-v1",
        "schema_version": 1,
        "table": _table_mapping(value.table),
        "type_id": value.type_id,
    }


def _parse_record(value: Mapping[str, Any]) -> StabilizerTypeRecord:
    _fields(
        value,
        {
            "occurrence_count",
            "occurrences",
            "presentation",
            "record_type",
            "schema_version",
            "table",
            "type_id",
        },
        "stabilizer type record",
    )
    if value.get("schema_version") != 1 or value.get("record_type") != "stabilizer-type-v1":
        raise ValueError("unsupported stabilizer type record")
    type_id = value.get("type_id")
    if type(type_id) is not str or type_id not in STABILIZER_TYPE_IDS:
        raise ValueError("invalid stabilizer type ID")
    table_value = value.get("table")
    if not isinstance(table_value, Mapping):
        raise TypeError("stabilizer type table must be an object")
    _fields(
        table_value,
        {
            "element_order",
            "group_id",
            "identity_index",
            "inverse_indices",
            "multiplication_table",
            "table_digest",
        },
        "stabilizer type table",
    )
    table = FiniteGroupTable(
        group_id=table_value.get("group_id"),
        element_order=tuple(table_value.get("element_order", ())),
        identity_index=table_value.get("identity_index"),
        multiplication_table=tuple(tuple(row) for row in table_value.get("multiplication_table", ())),
        inverse_indices=tuple(table_value.get("inverse_indices", ())),
        table_digest=table_value.get("table_digest"),
    )
    occurrences_value = value.get("occurrences")
    if not isinstance(occurrences_value, list):
        raise TypeError("stabilizer occurrences must be an array")
    occurrences_list: list[StabilizerOccurrenceWitness] = []
    for item in occurrences_value:
        if not isinstance(item, Mapping):
            raise TypeError("stabilizer occurrence must be an object")
        _fields(
            item,
            {
                "canonical_to_literal",
                "international_number",
                "literal_multiplication_table",
                "literal_table_digest",
                "setting",
                "wyckoff_id",
            },
            "stabilizer occurrence",
        )
        occurrences_list.append(
            StabilizerOccurrenceWitness(
                wyckoff_id=item.get("wyckoff_id"),
                international_number=item.get("international_number"),
                setting=item.get("setting"),
                literal_table_digest=item.get("literal_table_digest"),
                literal_multiplication_table=tuple(
                    tuple(row) for row in item.get("literal_multiplication_table", ())
                ),
                canonical_to_literal=tuple(item.get("canonical_to_literal", ())),
            )
        )
    occurrences = tuple(occurrences_list)
    return StabilizerTypeRecord(
        type_id=type_id,
        presentation=value.get("presentation"),
        table=table,
        occurrence_count=value.get("occurrence_count"),
        occurrences=occurrences,
    )


def _validate_inventory_records(
    values: Iterable[StabilizerTypeRecord],
) -> tuple[StabilizerTypeRecord, ...]:
    records = tuple(values)
    if tuple(record.type_id for record in records) != STABILIZER_TYPE_IDS:
        raise ValueError("local_library_incomplete: ordered 18-type inventory differs")
    replayed: list[StabilizerTypeRecord] = []
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, StabilizerTypeRecord):
            raise TypeError("stabilizer inventory requires StabilizerTypeRecord values")
        occurrences_list: list[StabilizerOccurrenceWitness] = []
        for occurrence in record.occurrences:
            if not isinstance(occurrence, StabilizerOccurrenceWitness):
                raise TypeError(
                    "stabilizer inventory occurrences must be occurrence witnesses"
                )
            occurrences_list.append(
                StabilizerOccurrenceWitness(
                    occurrence.wyckoff_id,
                    occurrence.international_number,
                    occurrence.setting,
                    occurrence.literal_table_digest,
                    occurrence.literal_multiplication_table,
                    occurrence.canonical_to_literal,
                )
            )
        occurrences = tuple(occurrences_list)
        checked = StabilizerTypeRecord(
            record.type_id,
            record.presentation,
            record.table,
            record.occurrence_count,
            occurrences,
        )
        replayed.append(checked)
        for occurrence in checked.occurrences:
            if occurrence.wyckoff_id in seen_ids:
                raise ValueError(
                    f"duplicate Wyckoff ID violates global uniqueness: {occurrence.wyckoff_id}"
                )
            seen_ids.add(occurrence.wyckoff_id)
    if len(seen_ids) != 1731:
        raise ValueError(
            "local_library_incomplete: inventory must contain exactly 1,731 unique Wyckoff occurrences"
        )
    return tuple(replayed)


def _library_asset_bytes(name: str) -> bytes:
    return asset_bytes(_LIBRARY_ASSET_PREFIX + name)


def _read_manifest(library: Path | None) -> tuple[Mapping[str, Any], dict[str, bytes]]:
    manifest_data = (
        _library_asset_bytes("manifest.json")
        if library is None
        else _safe_read_regular(library / "manifest.json", "stabilizer manifest")
    )
    manifest_value = _strict_json(manifest_data.rstrip(b"\n"))
    if not isinstance(manifest_value, Mapping):
        raise TypeError("stabilizer manifest must be an object")
    if _canonical_line(manifest_value) != manifest_data:
        raise ValueError("stabilizer manifest is not canonical JSON")
    legacy_fields = _MANIFEST_FIELDS - {"coverage"}
    manifest_keys = set(manifest_value)
    legacy_task10 = manifest_keys == legacy_fields
    if not legacy_task10:
        _fields(manifest_value, _MANIFEST_FIELDS, "stabilizer manifest")
    if manifest_value.get("schema_version") != 1 or manifest_value.get("library_version") != 1:
        raise ValueError("unsupported stabilizer manifest version")
    release_certified = manifest_value.get("release_certified")
    digest = manifest_value.get("catalogue_manifest_digest")
    if type(release_certified) is not bool:
        raise TypeError("stabilizer manifest release_certified must be boolean")
    if release_certified:
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError("release-certified stabilizer manifest has an invalid catalogue digest")
    elif digest is not None:
        raise ValueError("uncertified stabilizer manifest must have a null catalogue digest")
    source = manifest_value.get("diagnostic_source")
    if not isinstance(source, Mapping):
        raise TypeError("stabilizer diagnostic source must be an object")
    _fields(
        source,
        {"certification_status", "cryst", "gap", "raw_cryst_indices_persisted"},
        "stabilizer diagnostic source",
    )
    if (
        source.get("certification_status") != "uncertified-direct"
        or source.get("cryst") != "4.1.30"
        or source.get("gap") != "4.15.1"
        or source.get("raw_cryst_indices_persisted") is not False
    ):
        raise ValueError("stabilizer diagnostic source differs from the v1 contract")
    inventory = manifest_value.get("inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("stabilizer manifest inventory must be an object")
    _fields(
        inventory,
        {"stabilizer_type_count", "wyckoff_occurrence_count"},
        "stabilizer manifest inventory",
    )
    if any(type(inventory.get(key)) is not int for key in inventory):
        raise TypeError("stabilizer manifest inventory counts must be integers")
    if legacy_task10:
        if release_certified is not False or digest is not None:
            raise ValueError("legacy Task10 manifest must retain the diagnostic release gate")
        coverage: Mapping[str, Any] = {"z2_spatial": True, "z2_graded": False}
    else:
        coverage_value = manifest_value.get("coverage")
        if not isinstance(coverage_value, Mapping):
            raise TypeError("stabilizer manifest coverage must be an object")
        _fields(coverage_value, {"z2_spatial", "z2_graded"}, "stabilizer manifest coverage")
        if any(type(coverage_value.get(key)) is not bool for key in coverage_value):
            raise TypeError("stabilizer manifest coverage flags must be booleans")
        if coverage_value["z2_graded"] and not coverage_value["z2_spatial"]:
            raise ValueError("stabilizer graded coverage requires spatial coverage")
        coverage = coverage_value
    files = manifest_value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("stabilizer manifest requires artifact files")
    names: list[str] = []
    artifact_data: dict[str, bytes] = {}
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise TypeError("stabilizer manifest artifact must be an object")
        _fields(item, {"path", "rows", "sha256"}, f"stabilizer manifest files[{index}]")
        name = item.get("path")
        rows = item.get("rows")
        file_digest = item.get("sha256")
        if name not in _MANIFEST_FILE_ORDER or name in names:
            raise ValueError("stabilizer manifest artifact paths differ from the v1 contract")
        if type(rows) is not int or rows <= 0:
            raise ValueError("stabilizer manifest artifact row count is invalid")
        if not isinstance(file_digest, str) or _DIGEST_RE.fullmatch(file_digest) is None:
            raise ValueError("stabilizer manifest artifact digest is invalid")
        data = (
            _library_asset_bytes(name)
            if library is None
            else _safe_read_regular(library / name, f"stabilizer artifact {name}")
        )
        if not data.endswith(b"\n") or b"\r" in data:
            raise ValueError(f"stabilizer artifact {name} is not canonical NDJSON")
        if len(data.splitlines()) != rows:
            raise ValueError(f"stabilizer artifact {name} row count disagrees with manifest")
        if _sha256(data) != file_digest:
            raise ValueError(f"stabilizer artifact {name} digest/hash disagrees with manifest")
        names.append(name)
        artifact_data[name] = data
    expected_names = ["types.ndjson"]
    if coverage["z2_spatial"]:
        expected_names.append("z2-spatial-skeletons.ndjson")
    if coverage["z2_graded"]:
        expected_names.append("z2-graded-skeletons.ndjson")
    if names != expected_names:
        raise ValueError(
            "stabilizer artifacts are not bound by manifest coverage in canonical order"
        )
    normalized_manifest = dict(manifest_value)
    normalized_manifest["coverage"] = dict(coverage)
    return normalized_manifest, artifact_data


def _load_type_rows(data: bytes) -> tuple[StabilizerTypeRecord, ...]:
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("stabilizer type library is not canonical NDJSON")
    records: list[StabilizerTypeRecord] = []
    for line in data.splitlines(keepends=True):
        value = _strict_json(line[:-1])
        if not isinstance(value, Mapping) or _canonical_line(value) != line:
            raise ValueError("stabilizer type library row is not canonical JSON")
        records.append(_parse_record(value))
    return _validate_inventory_records(records)


def load_stabilizer_type_library(
    library: Path | None = None,
    *,
    catalogue_atlas: Path | None = None,
) -> tuple[StabilizerTypeRecord, ...]:
    source = None if library is None else Path(library)
    manifest, artifacts = _read_manifest(source)
    records = _load_type_rows(artifacts["types.ndjson"])
    inventory = manifest["inventory"]
    if inventory["stabilizer_type_count"] != len(records):
        raise ValueError("stabilizer manifest type count does not derive from artifact rows")
    occurrence_count = sum(record.occurrence_count for record in records)
    if inventory["wyckoff_occurrence_count"] != occurrence_count:
        raise ValueError("stabilizer manifest occurrence count does not derive from artifact rows")
    if manifest["release_certified"]:
        if catalogue_atlas is None:
            raise ValueError(
                "release-certified library requires a verified catalogue atlas to replay its digest"
            )
        material = _verified_release_material(Path(catalogue_atlas), test_execution=None)
        if manifest["catalogue_manifest_digest"] != material[0]:
            raise ValueError("release catalogue manifest digest differs from verified atlas bytes")
        if _types_bytes(build_stabilizer_inventory(material[1])) != artifacts["types.ndjson"]:
            raise ValueError("release catalogue inventory is not byte-identical")
    return records


Affine = tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]


def _fraction(value: object) -> Fraction:
    if type(value) is not str:
        raise TypeError("affine coefficient must use q(n,d)")
    match = _RATIONAL_RE.fullmatch(value)
    if match is None:
        raise ValueError("invalid exact rational")
    result = Fraction(int(match.group(1)), int(match.group(2)))
    if f"q({result.numerator},{result.denominator})" != value:
        raise ValueError("noncanonical exact rational")
    return result


def _parse_affine(value: Mapping[str, Any]) -> Affine:
    matrix_value = value.get("matrix")
    translation_value = value.get("translation")
    if not isinstance(matrix_value, (list, tuple)) or len(matrix_value) != 3:
        raise ValueError("affine matrix must be 3x3")
    matrix = tuple(
        tuple(_fraction(item) for item in row)
        for row in matrix_value
        if isinstance(row, (list, tuple)) and len(row) == 3
    )
    if len(matrix) != 3 or not isinstance(translation_value, (list, tuple)) or len(translation_value) != 3:
        raise ValueError("affine transformation has wrong dimension")
    return matrix, tuple(_fraction(item) for item in translation_value)


def _affine_compose(left: Affine, right: Affine) -> Affine:
    matrix = tuple(
        tuple(sum(left[0][row][inner] * right[0][inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )
    translation = tuple(
        sum(left[0][row][inner] * right[1][inner] for inner in range(3)) + left[1][row]
        for row in range(3)
    )
    return matrix, translation


def _literal_table(record: CatalogueRecord) -> tuple[tuple[int, ...], ...]:
    elements_value = record.stabilizer["embedded_elements"]
    if not isinstance(elements_value, tuple):
        raise TypeError("canonical stabilizer elements must be a tuple")
    elements = tuple(_parse_affine(item) for item in elements_value if isinstance(item, Mapping))
    if len(elements) != len(elements_value):
        raise TypeError("canonical stabilizer element must be an object")
    index = {element: position for position, element in enumerate(elements)}
    if len(index) != len(elements):
        raise ValueError("canonical stabilizer contains duplicate elements")
    try:
        return tuple(tuple(index[_affine_compose(left, right)] for right in elements) for left in elements)
    except KeyError as error:
        raise ValueError("canonical stabilizer is not exactly closed") from error


def build_stabilizer_inventory(records: Iterable[CatalogueRecord]) -> tuple[StabilizerTypeRecord, ...]:
    grouped: dict[str, list[StabilizerOccurrenceWitness]] = {type_id: [] for type_id in STABILIZER_TYPE_IDS}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, CatalogueRecord):
            raise TypeError("stabilizer inventory requires validated CatalogueRecord values")
        if record.wyckoff_id in seen:
            raise ValueError("catalogue inventory contains a duplicate Wyckoff ID")
        seen.add(record.wyckoff_id)
        table = _literal_table(record)
        identified = identify_stabilizer_type(table)
        table_digest = _sha256(canonical_json([list(row) for row in table]))
        grouped[identified.type_id].append(
            StabilizerOccurrenceWitness(
                wyckoff_id=record.wyckoff_id,
                international_number=int(record.space_group["international_number"]),
                setting=str(record.space_group["setting"]),
                literal_table_digest=table_digest,
                literal_multiplication_table=table,
                canonical_to_literal=identified.canonical_to_literal,
            )
        )
    if len(seen) != 1731:
        raise ValueError("local_library_incomplete: catalogue must contain exactly 1,731 unique Wyckoff types")
    result: list[StabilizerTypeRecord] = []
    for type_id, table, _ in _canonical_specs():
        occurrences = tuple(sorted(grouped[type_id], key=lambda item: (item.international_number, item.setting, item.wyckoff_id)))
        if not occurrences:
            raise ValueError(f"local_library_incomplete: missing stabilizer type {type_id}")
        result.append(StabilizerTypeRecord(type_id, _PRESENTATIONS[type_id], table, len(occurrences), occurrences))
    return _validate_inventory_records(result)


def _types_bytes(records: Iterable[StabilizerTypeRecord]) -> bytes:
    return b"".join(_canonical_line(_record_mapping(record)) for record in records)


def _manifest_mapping(
    types_bytes: bytes,
    records: Sequence[StabilizerTypeRecord],
    *,
    skeleton_bytes: bytes | None = None,
    graded_skeleton_bytes: bytes | None = None,
) -> dict[str, Any]:
    if graded_skeleton_bytes is not None and skeleton_bytes is None:
        raise ValueError("graded stabilizer artifact requires the spatial artifact")
    files: list[dict[str, Any]] = [
        {"path": "types.ndjson", "rows": len(records), "sha256": _sha256(types_bytes)}
    ]
    if skeleton_bytes is not None:
        files.append(
            {
                "path": "z2-spatial-skeletons.ndjson",
                "rows": len(skeleton_bytes.splitlines()),
                "sha256": _sha256(skeleton_bytes),
            }
        )
    if graded_skeleton_bytes is not None:
        files.append(
            {
                "path": "z2-graded-skeletons.ndjson",
                "rows": len(graded_skeleton_bytes.splitlines()),
                "sha256": _sha256(graded_skeleton_bytes),
            }
        )
    return {
        "catalogue_manifest_digest": None,
        "coverage": {
            "z2_graded": graded_skeleton_bytes is not None,
            "z2_spatial": skeleton_bytes is not None,
        },
        "diagnostic_source": {
            "certification_status": "uncertified-direct",
            "cryst": "4.1.30",
            "gap": "4.15.1",
            "raw_cryst_indices_persisted": False,
        },
        "files": files,
        "inventory": {
            "stabilizer_type_count": len(records),
            "wyckoff_occurrence_count": sum(record.occurrence_count for record in records),
        },
        "library_version": 1,
        "release_certified": False,
        "schema_version": 1,
    }


def _write_atomic(path: Path, data: bytes) -> None:
    """Durably replace ``path`` through an exclusive random same-dir file."""

    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError("stabilizer output directory is unavailable") from None
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("stabilizer output directory is not a safe directory")
    descriptor = -1
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        raise ValueError("atomic stabilizer artifact write failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def write_stabilizer_inventory(records: Iterable[StabilizerTypeRecord], output: Path) -> None:
    values = _validate_inventory_records(records)
    types_bytes = _types_bytes(values)
    existing_skeletons = output / "z2-spatial-skeletons.ndjson"
    skeleton_bytes = (
        _safe_read_regular(existing_skeletons, "spatial skeleton artifact")
        if existing_skeletons.exists() or existing_skeletons.is_symlink()
        else None
    )
    existing_graded = output / "z2-graded-skeletons.ndjson"
    graded_skeleton_bytes = (
        _safe_read_regular(existing_graded, "graded skeleton artifact")
        if existing_graded.exists() or existing_graded.is_symlink()
        else None
    )
    manifest = _manifest_mapping(
        types_bytes,
        values,
        skeleton_bytes=skeleton_bytes,
        graded_skeleton_bytes=graded_skeleton_bytes,
    )
    _write_atomic(output / "types.ndjson", types_bytes)
    _write_atomic(output / "manifest.json", _canonical_line(manifest))


def diagnostic_records_from_direct_exports(source: Path) -> tuple[CatalogueRecord, ...]:
    records: list[CatalogueRecord] = []
    for international_number in range(1, 231):
        path = source / f"sg{international_number}.json"
        try:
            value = _strict_json(path.read_bytes().rstrip(b"\n"))
        except OSError as error:
            raise ValueError("direct diagnostic export set is incomplete") from error
        if not isinstance(value, Mapping):
            raise TypeError("direct diagnostic export must be an object")
        normalized = normalize_gap_export(value)
        if any(int(record.space_group["international_number"]) != international_number for record in normalized):
            raise ValueError("direct diagnostic export group binding differs")
        records.extend(normalized)
    return tuple(records)


def generate_diagnostic_stabilizer_library(source: Path, output: Path) -> tuple[StabilizerTypeRecord, ...]:
    records = build_stabilizer_inventory(diagnostic_records_from_direct_exports(source))
    write_stabilizer_inventory(records, output)
    return records


def regenerate_stabilizer_library_from_embedded_inventory(output: Path) -> None:
    records = load_stabilizer_type_library()
    types_bytes = _types_bytes(records)
    _, artifacts = _read_manifest(None)
    skeleton_bytes = artifacts["z2-spatial-skeletons.ndjson"]
    graded_skeleton_bytes = artifacts.get("z2-graded-skeletons.ndjson")
    _write_atomic(output / "types.ndjson", types_bytes)
    _write_atomic(output / "z2-spatial-skeletons.ndjson", skeleton_bytes)
    if graded_skeleton_bytes is not None:
        _write_atomic(output / "z2-graded-skeletons.ndjson", graded_skeleton_bytes)
    _write_atomic(
        output / "manifest.json",
        _canonical_line(
            _manifest_mapping(
                types_bytes,
                records,
                skeleton_bytes=skeleton_bytes,
                graded_skeleton_bytes=graded_skeleton_bytes,
            )
        ),
    )


def _verified_release_material(
    catalogue_atlas: Path,
    *,
    test_execution: object | None,
) -> tuple[str, tuple[CatalogueRecord, ...]]:
    """Return bytes-derived release material only after the atlas verifier passes."""

    from .catalogue_coverage import resolve_current_generation
    from .catalogue_loader import load_ndjson
    from .catalogue_runner import _test_only_verify_catalogue, verify_catalogue

    if test_execution is None:
        manifest = verify_catalogue(catalogue_atlas)
    else:
        manifest = _test_only_verify_catalogue(catalogue_atlas, execution=test_execution)
    if manifest.status.get("release_complete") is not True:
        raise ValueError("catalogue manifest is not globally release-complete")
    generation = resolve_current_generation(catalogue_atlas)
    manifest_data = _safe_read_regular(
        generation / "manifest.json", "verified catalogue manifest"
    )
    if canonical_json(manifest.to_mapping()) + b"\n" != manifest_data:
        raise ValueError("verified catalogue manifest bytes are not canonical")
    geometry_rows = [item for item in manifest.files if item.get("kind") == "geometry"]
    if len(geometry_rows) != 1:
        raise ValueError("release-complete catalogue has no unique geometry artifact")
    geometry_name = Path(str(geometry_rows[0]["path"])).name
    geometry_path = generation / geometry_name
    geometry_data = _safe_read_regular(geometry_path, "verified catalogue geometry")
    if _sha256(geometry_data) != geometry_rows[0].get("sha256"):
        raise ValueError("verified catalogue geometry hash disagrees with manifest")
    records = tuple(load_ndjson(geometry_path))
    return _sha256(manifest_data), records


def _bind_release_catalogue(
    library: Path,
    catalogue_atlas: Path,
    *,
    test_execution: object | None,
) -> Mapping[str, Any]:
    manifest_digest, catalogue_records = _verified_release_material(
        catalogue_atlas, test_execution=test_execution
    )
    records = load_stabilizer_type_library(library)
    rebuilt = build_stabilizer_inventory(catalogue_records)
    if _types_bytes(rebuilt) != _types_bytes(records):
        raise ValueError("release catalogue binding inventory is not byte-identical")
    manifest_value, _ = _read_manifest(library)
    result = dict(manifest_value)
    result["catalogue_manifest_digest"] = manifest_digest
    result["release_certified"] = True
    return result


def bind_release_catalogue(library: Path, catalogue_atlas: Path) -> Mapping[str, Any]:
    """Bind a library to a production-verified, release-complete catalogue atlas."""

    return _bind_release_catalogue(
        Path(library), Path(catalogue_atlas), test_execution=None
    )


def _test_only_bind_release_catalogue(
    library: Path,
    catalogue_atlas: Path,
    *,
    execution: object,
) -> Mapping[str, Any]:
    """Exercise the same boundary with an authenticated synthetic runner context."""

    return _bind_release_catalogue(
        Path(library), Path(catalogue_atlas), test_execution=execution
    )


__all__ = [
    "STABILIZER_TYPE_IDS",
    "StabilizerOccurrenceWitness",
    "StabilizerTypeIdentification",
    "StabilizerTypeRecord",
    "bind_release_catalogue",
    "build_stabilizer_inventory",
    "canonical_generators",
    "canonical_stabilizer_table",
    "diagnostic_records_from_direct_exports",
    "generate_diagnostic_stabilizer_library",
    "identify_stabilizer_type",
    "load_stabilizer_type_library",
    "regenerate_stabilizer_library_from_embedded_inventory",
    "write_stabilizer_inventory",
]
