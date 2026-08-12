"""The 18 crystallographic finite stabilizer tables used by Z2 enumeration."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

from .direct_algebra import FiniteGroupTable


STABILIZER_TYPE_IDS = (
    "C1", "C2", "C3", "C4", "C6", "C2xC2", "C2xC2xC2",
    "C4xC2", "C6xC2", "D3", "D4", "D6", "D4xC2", "D6xC2",
    "A4", "A4xC2", "S4", "S4xC2",
)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _parity(value: tuple[int, ...]) -> int:
    return sum(
        value[left] > value[right]
        for left in range(len(value))
        for right in range(left + 1, len(value))
    ) & 1


def _table_from_elements(
    group_id: str,
    elements: Sequence[Hashable],
    multiply: Callable[[Hashable, Hashable], Hashable],
) -> FiniteGroupTable:
    values = tuple(elements)
    index = {value: position for position, value in enumerate(values)}
    table = tuple(
        tuple(index[multiply(left, right)] for right in values) for left in values
    )
    identity = next(
        candidate
        for candidate in range(len(values))
        if all(
            table[candidate][other] == other
            and table[other][candidate] == other
            for other in range(len(values))
        )
    )
    names = ("1",) + tuple(f"g{index}" for index in range(1, len(values)))
    return FiniteGroupTable(group_id, names, identity, table)


def _cyclic(n: int, type_id: str):
    values = tuple(range(n))
    table = _table_from_elements(
        type_id, values, lambda left, right: (int(left) + int(right)) % n
    )
    return table, (() if n == 1 else (1,))


def _abelian(factors: tuple[int, ...], type_id: str):
    values = tuple(product(*(range(factor) for factor in factors)))

    def multiply(left, right):
        return tuple(
            (left[index] + right[index]) % factors[index]
            for index in range(len(factors))
        )

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    generators = tuple(
        index[tuple(int(axis == component) for axis in range(len(factors)))]
        for component in range(len(factors))
    )
    return table, generators


def _dihedral(n: int, type_id: str, *, central_c2: bool = False):
    values = tuple(
        (rotation, reflection, central)
        for central in range(2 if central_c2 else 1)
        for reflection in range(2)
        for rotation in range(n)
    )

    def multiply(left, right):
        a, b, z = left
        c, d, w = right
        return ((a + (-1 if b else 1) * c) % n, (b + d) % 2, (z + w) % 2)

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    generators = [index[(1, 0, 0)], index[(0, 1, 0)]]
    if central_c2:
        generators.append(index[(0, 0, 1)])
    return table, tuple(generators)


def _permutation_group(type_id: str, *, alternating: bool, central_c2: bool):
    identity = (0, 1, 2, 3)
    group = tuple(
        value
        for value in permutations(range(4))
        if not alternating or _parity(value) == 0
    )
    group = (identity,) + tuple(value for value in group if value != identity)
    values = tuple(
        (permutation, central)
        for central in range(2 if central_c2 else 1)
        for permutation in group
    )

    def multiply(left, right):
        a, z = left
        b, w = right
        return (_compose(a, b), (z + w) % 2)

    table = _table_from_elements(type_id, values, multiply)
    index = {value: position for position, value in enumerate(values)}
    if alternating:
        first, second = (1, 2, 0, 3), (1, 0, 3, 2)
    else:
        first, second = (1, 2, 3, 0), (1, 0, 2, 3)
    generators = [index[(first, 0)], index[(second, 0)]]
    if central_c2:
        generators.append(index[(identity, 1)])
    return table, tuple(generators)


@lru_cache(maxsize=1)
def _canonical_specs():
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
    return next(table for name, table, _ in _canonical_specs() if name == type_id)


def canonical_generators(type_id: str) -> tuple[int, ...]:
    return next(generators for name, _, generators in _canonical_specs() if name == type_id)


def _element_orders(table, identity: int) -> tuple[int, ...]:
    result: list[int] = []
    for element in range(len(table)):
        value = identity
        for exponent in range(1, len(table) + 1):
            value = table[value][element]
            if value == identity:
                result.append(exponent)
                break
    return tuple(result)


def _signature(table, identity: int):
    orders = _element_orders(table, identity)
    abelian = all(
        table[left][right] == table[right][left]
        for left in range(len(table))
        for right in range(len(table))
    )
    center = sum(
        all(table[element][other] == table[other][element] for other in range(len(table)))
        for element in range(len(table))
    )
    return len(table), abelian, center, tuple(sorted(Counter(orders).items()))


def _generator_words(table: FiniteGroupTable, generators: tuple[int, ...]):
    words: list[tuple[int, ...] | None] = [None] * len(table.element_order)
    words[table.identity_index] = ()
    queue = deque((table.identity_index,))
    while queue:
        element = queue.popleft()
        word = words[element]
        if word is None:
            continue
        for position, generator in enumerate(generators):
            target = table.multiplication_table[element][generator]
            if words[target] is None:
                words[target] = word + (position,)
                queue.append(target)
    return tuple(word for word in words if word is not None)


def _isomorphism(canonical, generators, literal, literal_identity):
    canonical_orders = _element_orders(
        canonical.multiplication_table, canonical.identity_index
    )
    literal_orders = _element_orders(literal, literal_identity)
    canonical_central = tuple(
        all(
            canonical.multiplication_table[element][other]
            == canonical.multiplication_table[other][element]
            for other in range(len(literal))
        )
        for element in range(len(literal))
    )
    literal_central = tuple(
        all(literal[element][other] == literal[other][element] for other in range(len(literal)))
        for element in range(len(literal))
    )
    if not generators:
        return (literal_identity,) if len(literal) == 1 else None
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
    valid: list[tuple[int, ...]] = []
    for images in product(*candidate_sets):
        mapping = []
        for word in words:
            image = literal_identity
            for position in word:
                image = literal[image][images[position]]
            mapping.append(image)
        witness = tuple(mapping)
        if len(set(witness)) == len(literal) and all(
            witness[canonical.multiplication_table[left][right]]
            == literal[witness[left]][witness[right]]
            for left in range(len(literal))
            for right in range(len(literal))
        ):
            valid.append(witness)
    return min(valid) if valid else None


@dataclass(frozen=True, slots=True)
class StabilizerTypeIdentification:
    type_id: str
    canonical_to_literal: tuple[int, ...]


def identify_stabilizer_type(value) -> StabilizerTypeIdentification:
    table = tuple(tuple(int(item) for item in row) for row in value)
    identity = next(
        candidate
        for candidate in range(len(table))
        if all(
            table[candidate][other] == other and table[other][candidate] == other
            for other in range(len(table))
        )
    )
    signature = _signature(table, identity)
    for type_id, canonical, generators in _canonical_specs():
        if _signature(canonical.multiplication_table, canonical.identity_index) != signature:
            continue
        witness = _isomorphism(canonical, generators, table, identity)
        if witness is not None:
            return StabilizerTypeIdentification(type_id, witness)
    raise ValueError("finite stabilizer is outside the 18 crystallographic types")


__all__ = [
    "canonical_generators",
    "canonical_stabilizer_table",
    "identify_stabilizer_type",
]
