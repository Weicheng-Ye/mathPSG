r"""Direct physical :math:`\mathbb Z_2` local-branch enumeration.

This module contains only the arithmetic needed by ``classify``.  In
particular, it does not construct IDs, hashes, witnesses, exhaustiveness
certificates, or replay objects.  Spatial homomorphisms are enumerated in the
octahedral and hexagonal-dihedral rotation hosts, quotiented by exact marked
SO(3) conjugacy, and lifted canonically to SU(2).  Graded branches are formed
analytically from the involutions in each spatial-image centralizer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
import math
from typing import Literal

from .algebraic import (
    ONE_Q23,
    ONE_QUATERNION,
    SQRT2,
    SQRT3,
    SQRT6,
    ZERO_Q23,
    ExactQuaternion,
    ExactSO3,
    Q23,
    identity_so3,
)
from .gf2 import MatrixGF2
from .direct_stabilizers import (
    canonical_generators,
    canonical_stabilizer_table,
    identify_stabilizer_type,
)


HostID = Literal["O", "D6"]
Vector = tuple[Q23, Q23, Q23]


@dataclass(frozen=True, slots=True)
class FiniteRotationHost:
    """One finite exact rotation host and its canonical SU(2) lifts."""

    host_id: HostID
    matrices: tuple[ExactSO3, ...]
    lifts: tuple[ExactQuaternion, ...]
    multiplication_table: tuple[tuple[int, ...], ...]
    inverse_indices: tuple[int, ...]
    element_orders: tuple[int, ...]
    identity_index: int


@dataclass(frozen=True, slots=True)
class Z2LocalSkeleton:
    """Physical local branch, with no certification metadata.

    ``spatial_multiplication_table`` records the original site stabilizer.
    ``so3_images``, ``su2_lifts``, and ``defect_bits`` use the spatial domain
    or its direct product with onsite time reversal. ``marking_shifts``
    contains the disconnected residual-centralizer actions on lift signs.
    """

    spatial_multiplication_table: tuple[tuple[int, ...], ...]
    so3_images: tuple[ExactSO3, ...]
    su2_lifts: tuple[ExactQuaternion, ...]
    defect_bits: MatrixGF2
    projected_image_type_id: str
    kernel_elements: tuple[int, ...]
    graded: bool
    marking_shifts: tuple[tuple[int, ...], ...]


def _q(value: int | Fraction) -> Q23:
    return Q23.from_rational(value)


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    return sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    ) % 2


def _matrix_from_integer_rows(rows: tuple[tuple[int, int, int], ...]) -> ExactSO3:
    return ExactSO3(
        tuple(tuple(_q(entry) for entry in row) for row in rows)  # type: ignore[arg-type]
    )


def _octahedral_matrices() -> tuple[ExactSO3, ...]:
    matrices: set[ExactSO3] = set()
    for permutation in permutations(range(3)):
        parity = _permutation_parity(permutation)
        for signs in product((-1, 1), repeat=3):
            if (-1 if parity else 1) * signs[0] * signs[1] * signs[2] != 1:
                continue
            rows = tuple(
                tuple(
                    signs[row] if column == permutation[row] else 0
                    for column in range(3)
                )
                for row in range(3)
            )
            matrices.add(_matrix_from_integer_rows(rows))
    return tuple(sorted(matrices, key=lambda matrix: matrix.canonical_key))


def _hexagonal_cos_sin(index: int) -> tuple[Q23, Q23]:
    half = _q(Fraction(1, 2))
    root3_half = SQRT3 * half
    return (
        (ONE_Q23, ZERO_Q23),
        (half, root3_half),
        (-half, root3_half),
        (-ONE_Q23, ZERO_Q23),
        (-half, -root3_half),
        (half, -root3_half),
    )[index % 6]


def _z_rotation_sixth(index: int) -> ExactSO3:
    cosine, sine = _hexagonal_cos_sin(index)
    return ExactSO3(
        (
            (cosine, -sine, ZERO_Q23),
            (sine, cosine, ZERO_Q23),
            (ZERO_Q23, ZERO_Q23, ONE_Q23),
        )
    )


def _dihedral_six_matrices() -> tuple[ExactSO3, ...]:
    flip = ExactSO3.diagonal((1, -1, -1))
    rotations = tuple(_z_rotation_sixth(index) for index in range(6))
    return tuple(
        sorted(
            set(rotations + tuple(rotation @ flip for rotation in rotations)),
            key=lambda matrix: matrix.canonical_key,
        )
    )


def _binary_octahedral_candidates() -> tuple[ExactQuaternion, ...]:
    candidates: set[ExactQuaternion] = set()
    basis = (
        (ONE_Q23, ZERO_Q23, ZERO_Q23, ZERO_Q23),
        (ZERO_Q23, ONE_Q23, ZERO_Q23, ZERO_Q23),
        (ZERO_Q23, ZERO_Q23, ONE_Q23, ZERO_Q23),
        (ZERO_Q23, ZERO_Q23, ZERO_Q23, ONE_Q23),
    )
    for vector in basis:
        quaternion = ExactQuaternion(*vector)
        candidates.update((quaternion, -quaternion))
    half = _q(Fraction(1, 2))
    for signs in product((-1, 1), repeat=4):
        candidates.add(ExactQuaternion(*(_q(sign) * half for sign in signs)))
    root2_half = SQRT2 * half
    for first, second in combinations(range(4), 2):
        for first_sign, second_sign in product((-1, 1), repeat=2):
            entries = [ZERO_Q23] * 4
            entries[first] = _q(first_sign) * root2_half
            entries[second] = _q(second_sign) * root2_half
            candidates.add(ExactQuaternion(*entries))
    return tuple(candidates)


def _half_angle_cos_sin(index: int) -> tuple[Q23, Q23]:
    half = _q(Fraction(1, 2))
    root3_half = SQRT3 * half
    return (
        (ONE_Q23, ZERO_Q23),
        (root3_half, half),
        (half, root3_half),
        (ZERO_Q23, ONE_Q23),
        (-half, root3_half),
        (-root3_half, half),
        (-ONE_Q23, ZERO_Q23),
        (-root3_half, -half),
        (-half, -root3_half),
        (ZERO_Q23, -ONE_Q23),
        (half, -root3_half),
        (root3_half, -half),
    )[index % 12]


def _binary_dihedral_candidates() -> tuple[ExactQuaternion, ...]:
    candidates: set[ExactQuaternion] = set()
    for index in range(12):
        cosine, sine = _half_angle_cos_sin(index)
        candidates.add(ExactQuaternion(cosine, ZERO_Q23, ZERO_Q23, sine))
        candidates.add(ExactQuaternion(ZERO_Q23, cosine, sine, ZERO_Q23))
    return tuple(candidates)


def _build_host(
    host_id: HostID,
    matrices: tuple[ExactSO3, ...],
    lift_candidates: tuple[ExactQuaternion, ...],
) -> FiniteRotationHost:
    matrices = tuple(sorted(set(matrices), key=lambda matrix: matrix.canonical_key))
    index = {matrix: position for position, matrix in enumerate(matrices)}
    try:
        table = tuple(tuple(index[left @ right] for right in matrices) for left in matrices)
    except KeyError as error:
        raise ArithmeticError(f"{host_id} rotation host is not closed") from error
    identity_index = index[identity_so3()]
    inverses = tuple(
        next(
            right
            for right in range(len(matrices))
            if table[left][right] == identity_index
            and table[right][left] == identity_index
        )
        for left in range(len(matrices))
    )
    orders: list[int] = []
    for element in range(len(matrices)):
        value = identity_index
        for exponent in range(1, len(matrices) + 1):
            value = table[value][element]
            if value == identity_index:
                orders.append(exponent)
                break
        else:
            raise ArithmeticError(f"{host_id} element has no finite order")
    by_matrix: dict[ExactSO3, ExactQuaternion] = {}
    for candidate in lift_candidates:
        by_matrix[candidate.to_so3()] = candidate.canonicalized()
    try:
        lifts = tuple(by_matrix[matrix] for matrix in matrices)
    except KeyError as error:
        raise ArithmeticError(f"binary {host_id} does not cover its rotation host") from error
    return FiniteRotationHost(
        host_id,
        matrices,
        lifts,
        table,
        inverses,
        tuple(orders),
        identity_index,
    )


@lru_cache(maxsize=2)
def finite_rotation_host(host_id: HostID) -> FiniteRotationHost:
    """Return the exact octahedral or hexagonal-dihedral host."""

    if host_id == "O":
        return _build_host(host_id, _octahedral_matrices(), _binary_octahedral_candidates())
    if host_id == "D6":
        return _build_host(host_id, _dihedral_six_matrices(), _binary_dihedral_candidates())
    raise ValueError(f"unknown finite rotation host {host_id!r}")


def _element_orders(table) -> tuple[int, ...]:
    result: list[int] = []
    for element in range(len(table.element_order)):
        value = table.identity_index
        for exponent in range(1, len(table.element_order) + 1):
            value = table.multiplication_table[value][element]
            if value == table.identity_index:
                result.append(exponent)
                break
        else:
            raise ArithmeticError("source element has no finite order")
    return tuple(result)


def _generator_words(table, generators: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    words: list[tuple[int, ...] | None] = [None] * len(table.element_order)
    words[table.identity_index] = ()
    queue = deque((table.identity_index,))
    while queue:
        element = queue.popleft()
        word = words[element]
        assert word is not None
        for position, generator in enumerate(generators):
            target = table.multiplication_table[element][generator]
            if words[target] is None:
                words[target] = word + (position,)
                queue.append(target)
    if any(word is None for word in words):
        raise ArithmeticError("source generators do not generate the finite table")
    return tuple(word for word in words if word is not None)


def _homomorphisms(
    table, generators: tuple[int, ...], host: FiniteRotationHost
) -> tuple[tuple[int, ...], ...]:
    if not generators:
        return ((host.identity_index,),)
    source_orders = _element_orders(table)
    words = _generator_words(table, generators)
    candidate_sets = tuple(
        tuple(
            element
            for element in range(len(host.matrices))
            if source_orders[generator] % host.element_orders[element] == 0
        )
        for generator in generators
    )
    result: set[tuple[int, ...]] = set()
    for generator_images in product(*candidate_sets):
        mapping: list[int] = []
        for word in words:
            image = host.identity_index
            for generator_position in word:
                image = host.multiplication_table[image][
                    generator_images[generator_position]
                ]
            mapping.append(image)
        candidate = tuple(mapping)
        if all(
            candidate[table.multiplication_table[left][right]]
            == host.multiplication_table[candidate[left]][candidate[right]]
            for left in range(len(table.element_order))
            for right in range(len(table.element_order))
        ):
            result.add(candidate)
    return tuple(sorted(result))


def _conjugate_mapping(
    mapping: tuple[int, ...], conjugator: int, host: FiniteRotationHost
) -> tuple[int, ...]:
    inverse = host.inverse_indices[conjugator]
    return tuple(
        host.multiplication_table[
            host.multiplication_table[conjugator][image]
        ][inverse]
        for image in mapping
    )


def _host_orbits(
    homomorphisms: tuple[tuple[int, ...], ...], host: FiniteRotationHost
) -> tuple[tuple[int, ...], ...]:
    remaining = set(homomorphisms)
    representatives: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        orbit = {
            _conjugate_mapping(seed, conjugator, host)
            for conjugator in range(len(host.matrices))
        }
        representative = min(orbit)
        representatives.append(representative)
        remaining.difference_update(orbit)
    return tuple(sorted(representatives))


def _trace(matrix: ExactSO3) -> Q23:
    return matrix.rows[0][0] + matrix.rows[1][1] + matrix.rows[2][2]


def _character_key(matrices: tuple[ExactSO3, ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(_trace(matrix).coefficients for matrix in matrices)


def _marked_matrices(
    host: FiniteRotationHost, mapping: tuple[int, ...]
) -> tuple[ExactSO3, ...]:
    return tuple(host.matrices[index] for index in mapping)


def _marked_lifts(
    host: FiniteRotationHost, mapping: tuple[int, ...]
) -> tuple[ExactQuaternion, ...]:
    return tuple(host.lifts[index] for index in mapping)


def _sqrt_fraction_field(value: Q23) -> Q23:
    rational = value.to_fraction()
    if rational < 0:
        raise ArithmeticError("negative squared norm")
    if rational == 0:
        return ZERO_Q23
    for square_free, radical in (
        (1, ONE_Q23),
        (2, SQRT2),
        (3, SQRT3),
        (6, SQRT6),
    ):
        reduced = rational / square_free
        numerator = math.isqrt(reduced.numerator)
        denominator = math.isqrt(reduced.denominator)
        if (
            numerator * numerator == reduced.numerator
            and denominator * denominator == reduced.denominator
        ):
            return _q(Fraction(numerator, denominator)) * radical
    raise ArithmeticError("required exact square root lies outside Q(sqrt2,sqrt3)")


def _dot(left: Vector, right: Vector) -> Q23:
    return sum((a * b for a, b in zip(left, right)), ZERO_Q23)


def _scale(value: Q23, vector: Vector) -> Vector:
    return tuple(value * entry for entry in vector)  # type: ignore[return-value]


def _subtract(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _cross(left: Vector, right: Vector) -> Vector:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Vector) -> Vector:
    norm = _sqrt_fraction_field(_dot(vector, vector))
    if not norm:
        raise ValueError("cannot normalize zero vector")
    return _scale(norm.inverse(), vector)


_COORDINATE_AXES: tuple[Vector, ...] = (
    (ONE_Q23, ZERO_Q23, ZERO_Q23),
    (ZERO_Q23, ONE_Q23, ZERO_Q23),
    (ZERO_Q23, ZERO_Q23, ONE_Q23),
)


def _axis_options(host: FiniteRotationHost, element: int) -> tuple[Vector, ...]:
    if element == host.identity_index:
        return ()
    lift = host.lifts[element]
    axis = _normalize((lift.x, lift.y, lift.z))
    return (
        (axis, _scale(_q(-1), axis))
        if host.element_orders[element] == 2
        else (axis,)
    )


def _frame(primary: Vector, secondary: Vector) -> ExactSO3 | None:
    perpendicular = _subtract(secondary, _scale(_dot(primary, secondary), primary))
    if not _dot(perpendicular, perpendicular):
        return None
    second = _normalize(perpendicular)
    third = _cross(primary, second)
    return ExactSO3(
        tuple(
            tuple((primary, second, third)[column][row] for column in range(3))
            for row in range(3)
        )  # type: ignore[arg-type]
    )


def _marked_conjugator(
    source_host: FiniteRotationHost,
    source_mapping: tuple[int, ...],
    target_host: FiniteRotationHost,
    target_mapping: tuple[int, ...],
) -> ExactSO3 | None:
    source_images = _marked_matrices(source_host, source_mapping)
    target_images = _marked_matrices(target_host, target_mapping)

    def accepts(matrix: ExactSO3) -> bool:
        transpose = matrix.transpose()
        return all(
            matrix @ source @ transpose == target
            for source, target in zip(source_images, target_images)
        )

    identity = identity_so3()
    if accepts(identity):
        return identity
    nonidentity = tuple(
        index
        for index, (source, target) in enumerate(
            zip(source_mapping, target_mapping)
        )
        if source != source_host.identity_index and target != target_host.identity_index
    )
    for primary_index in nonidentity:
        for source_primary in _axis_options(source_host, source_mapping[primary_index]):
            for target_primary in _axis_options(target_host, target_mapping[primary_index]):
                secondaries = [
                    (source_secondary, target_secondary)
                    for secondary_index in nonidentity
                    for source_secondary in _axis_options(
                        source_host, source_mapping[secondary_index]
                    )
                    for target_secondary in _axis_options(
                        target_host, target_mapping[secondary_index]
                    )
                ]
                secondaries.extend(product(_COORDINATE_AXES, _COORDINATE_AXES))
                for source_secondary, target_secondary in secondaries:
                    source_frame = _frame(source_primary, source_secondary)
                    target_frame = _frame(target_primary, target_secondary)
                    if source_frame is None or target_frame is None:
                        continue
                    candidate = target_frame @ source_frame.transpose()
                    if accepts(candidate):
                        return candidate
    return None


def _canonical_rotation_image(images: Sequence[ExactSO3]) -> tuple[ExactSO3, ...]:
    return tuple(sorted(set(images), key=lambda matrix: matrix.canonical_key))


def _image_type_from_matrices(images: Sequence[ExactSO3]) -> str:
    image = _canonical_rotation_image(images)
    index = {matrix: position for position, matrix in enumerate(image)}
    table = tuple(tuple(index[left @ right] for right in image) for left in image)
    return identify_stabilizer_type(table).type_id


def _defect(
    multiplication: tuple[tuple[int, ...], ...],
    lifts: tuple[ExactQuaternion, ...],
) -> MatrixGF2:
    rows: list[tuple[int, ...]] = []
    for left, row in enumerate(multiplication):
        bits: list[int] = []
        for right, target in enumerate(row):
            value = lifts[left] * lifts[right]
            if value == lifts[target]:
                bits.append(0)
            elif value == -lifts[target]:
                bits.append(1)
            else:
                raise ArithmeticError("SU(2) lifts have a noncentral multiplication defect")
        rows.append(tuple(bits))
    return MatrixGF2(tuple(rows), column_count=len(multiplication))


@dataclass(frozen=True, slots=True)
class _SpatialPrototype:
    host: FiniteRotationHost
    mapping: tuple[int, ...]
    character: tuple[tuple[Fraction, ...], ...]


@lru_cache(maxsize=18)
def _enumerate_canonical_type(type_id: str) -> tuple[Z2LocalSkeleton, ...]:
    table = canonical_stabilizer_table(type_id)
    generators = canonical_generators(type_id)
    orbits: list[_SpatialPrototype] = []
    for host_id in ("O", "D6"):
        host = finite_rotation_host(host_id)
        for mapping in _host_orbits(_homomorphisms(table, generators, host), host):
            matrices = _marked_matrices(host, mapping)
            orbits.append(_SpatialPrototype(host, mapping, _character_key(matrices)))

    # Character equality is the cheap filter.  Actual merging is gated by an
    # exact marked SO(3) conjugator, so no numerical tolerance is involved.
    clusters: list[list[_SpatialPrototype]] = []
    for candidate in sorted(
        orbits,
        key=lambda item: (
            item.character,
            0 if item.host.host_id == "O" else 1,
            item.mapping,
        ),
    ):
        for cluster in clusters:
            representative = cluster[0]
            if candidate.character != representative.character:
                continue
            if (
                _marked_conjugator(
                    candidate.host,
                    candidate.mapping,
                    representative.host,
                    representative.mapping,
                )
                is not None
            ):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])

    skeletons: list[Z2LocalSkeleton] = []
    for cluster in clusters:
        selected = min(
            cluster,
            key=lambda item: (
                0 if item.host.host_id == "O" else 1,
                item.mapping,
            ),
        )
        images = _marked_matrices(selected.host, selected.mapping)
        lifts = _marked_lifts(selected.host, selected.mapping)
        skeletons.append(
            Z2LocalSkeleton(
                spatial_multiplication_table=table.multiplication_table,
                so3_images=images,
                su2_lifts=lifts,
                defect_bits=_defect(table.multiplication_table, lifts),
                projected_image_type_id=_image_type_from_matrices(images),
                kernel_elements=tuple(
                    index for index, image in enumerate(images) if image == identity_so3()
                ),
                graded=False,
                marking_shifts=(),
            )
        )
    return tuple(
        sorted(
            skeletons,
            key=lambda item: (
                len(set(item.so3_images)),
                item.kernel_elements,
                _character_key(item.so3_images),
            ),
        )
    )


def _transport_spatial(
    skeleton: Z2LocalSkeleton, canonical_to_literal: tuple[int, ...]
) -> Z2LocalSkeleton:
    if canonical_to_literal == tuple(range(len(canonical_to_literal))):
        return skeleton
    literal_to_canonical = tuple(
        next(
            canonical
            for canonical, literal in enumerate(canonical_to_literal)
            if literal == index
        )
        for index in range(len(canonical_to_literal))
    )
    images = tuple(
        skeleton.so3_images[literal_to_canonical[index]]
        for index in range(len(literal_to_canonical))
    )
    lifts = tuple(
        skeleton.su2_lifts[literal_to_canonical[index]]
        for index in range(len(literal_to_canonical))
    )
    table = tuple(
        tuple(
            canonical_to_literal[
                skeleton.spatial_multiplication_table[
                    literal_to_canonical[left]
                ][literal_to_canonical[right]]
            ]
            for right in range(len(literal_to_canonical))
        )
        for left in range(len(literal_to_canonical))
    )
    return Z2LocalSkeleton(
        spatial_multiplication_table=table,
        so3_images=images,
        su2_lifts=lifts,
        defect_bits=_defect(table, lifts),
        projected_image_type_id=skeleton.projected_image_type_id,
        kernel_elements=tuple(
            index for index, image in enumerate(images) if image == identity_so3()
        ),
        graded=False,
        marking_shifts=(),
    )


def enumerate_spatial_z2_skeletons(group: object) -> tuple[Z2LocalSkeleton, ...]:
    """Enumerate physical spatial Z2 branches for a finite group table."""

    try:
        raw_table = tuple(tuple(row) for row in group.multiplication_table)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError("Z2 branch enumeration requires a finite multiplication table") from error
    identified = identify_stabilizer_type(raw_table)
    return tuple(
        _transport_spatial(skeleton, identified.canonical_to_literal)
        for skeleton in _enumerate_canonical_type(identified.type_id)
    )


def _direct_product_table(
    table: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    order = len(table)
    return tuple(
        tuple(
            ((left // order) ^ (right // order)) * order
            + table[left % order][right % order]
            for right in range(2 * order)
        )
        for left in range(2 * order)
    )


def _matrix_order(matrix: ExactSO3, bound: int = 24) -> int:
    value = identity_so3()
    for exponent in range(1, bound + 1):
        value = value @ matrix
        if value == identity_so3():
            return exponent
    raise ArithmeticError("rotation order exceeds crystallographic bound")


def _axis_from_matrix(matrix: ExactSO3) -> Vector:
    skew = (
        matrix.rows[2][1] - matrix.rows[1][2],
        matrix.rows[0][2] - matrix.rows[2][0],
        matrix.rows[1][0] - matrix.rows[0][1],
    )
    if _dot(skew, skew):
        return _normalize(skew)
    rows = tuple(
        tuple(
            matrix.rows[row][column]
            - (ONE_Q23 if row == column else ZERO_Q23)
            for column in range(3)
        )
        for row in range(3)
    )
    for first in range(3):
        for second in range(first + 1, 3):
            candidate = _cross(rows[first], rows[second])  # type: ignore[arg-type]
            if _dot(candidate, candidate):
                return _normalize(candidate)
    raise ArithmeticError("nonidentity rotation has no exact fixed axis")


def _axis_and_half_turn(
    images: Sequence[ExactSO3],
) -> tuple[Vector, ExactSO3, ExactQuaternion]:
    nonidentity = tuple(matrix for matrix in set(images) if matrix != identity_so3())
    axis = (
        _COORDINATE_AXES[2]
        if not nonidentity
        else _axis_from_matrix(
            max(
                nonidentity,
                key=lambda matrix: (_matrix_order(matrix), matrix.canonical_key),
            )
        )
    )
    lift = ExactQuaternion(ZERO_Q23, *axis).canonicalized()
    return axis, lift.to_so3(), lift


def _perpendicular_half_turn(axis: Vector) -> tuple[ExactSO3, ExactQuaternion]:
    for candidate in _COORDINATE_AXES:
        perpendicular = _subtract(candidate, _scale(_dot(candidate, axis), axis))
        if _dot(perpendicular, perpendicular):
            lift = ExactQuaternion(
                ZERO_Q23, *_normalize(perpendicular)
            ).canonicalized()
            return lift.to_so3(), lift
    raise ArithmeticError("could not construct an exact perpendicular axis")


def _known_lift(matrix: ExactSO3) -> ExactQuaternion:
    for host_id in ("O", "D6"):
        host = finite_rotation_host(host_id)
        try:
            return host.lifts[host.matrices.index(matrix)]
        except ValueError:
            pass
    raise ArithmeticError("rotation is outside the two exact lift hosts")


def _time_choices(
    image_type: str, images: tuple[ExactSO3, ...]
) -> tuple[tuple[ExactSO3, ExactQuaternion], ...]:
    identity = (identity_so3(), ONE_QUATERNION)
    axis, parallel, parallel_lift = _axis_and_half_turn(images)
    if image_type == "C1":
        return (identity, (parallel, parallel_lift))
    if image_type == "C2":
        perpendicular, perpendicular_lift = _perpendicular_half_turn(axis)
        return (identity, (parallel, parallel_lift), (perpendicular, perpendicular_lift))
    if image_type in {"C3", "C4", "C6", "D3", "D4", "D6"}:
        return (identity, (parallel, parallel_lift))
    if image_type == "C2xC2":
        return tuple((matrix, _known_lift(matrix)) for matrix in _canonical_rotation_image(images))
    if image_type in {"A4", "S4"}:
        return (identity,)
    raise ArithmeticError(f"unsupported projected image type {image_type}")


def _quaternion_inverse(value: ExactQuaternion) -> ExactQuaternion:
    return ExactQuaternion(value.scalar, -value.x, -value.y, -value.z)


def _centralizer_component_representatives(
    image_type: str, images: tuple[ExactSO3, ...]
) -> tuple[tuple[ExactSO3, ExactQuaternion], ...]:
    identity = (identity_so3(), ONE_QUATERNION)
    if image_type in {"C1", "C3", "C4", "C6", "A4", "S4"}:
        return (identity,)
    axis, parallel, parallel_lift = _axis_and_half_turn(images)
    if image_type == "C2":
        perpendicular, perpendicular_lift = _perpendicular_half_turn(axis)
        return (identity, (perpendicular, perpendicular_lift))
    if image_type == "C2xC2":
        return tuple((matrix, _known_lift(matrix)) for matrix in _canonical_rotation_image(images))
    if image_type in {"D3", "D4", "D6"}:
        return (identity, (parallel, parallel_lift))
    raise ArithmeticError(f"unsupported full image type {image_type}")


def _marking_shift(
    representative_lift: ExactQuaternion,
    lifts: tuple[ExactQuaternion, ...],
) -> tuple[int, ...]:
    inverse = _quaternion_inverse(representative_lift)
    result: list[int] = []
    for lift in lifts:
        conjugated = representative_lift * lift * inverse
        if conjugated == lift:
            result.append(0)
        elif conjugated == -lift:
            result.append(1)
        else:
            raise ArithmeticError("centralizer component acts noncentrally on SU(2) lifts")
    return tuple(result)


@lru_cache(maxsize=None)
def enumerate_graded_z2_skeletons(
    spatial: Z2LocalSkeleton,
) -> tuple[Z2LocalSkeleton, ...]:
    """Add onsite time reversal to one spatial physical branch."""

    if spatial.graded:
        raise TypeError("graded enumeration requires a spatial branch")
    source_table = spatial.spatial_multiplication_table
    multiplication = _direct_product_table(source_table)
    source_order = len(source_table)
    result: list[Z2LocalSkeleton] = []
    for time_matrix, time_lift in _time_choices(
        spatial.projected_image_type_id, spatial.so3_images
    ):
        images = spatial.so3_images + tuple(
            image @ time_matrix for image in spatial.so3_images
        )
        lifts = spatial.su2_lifts + tuple(
            lift * time_lift for lift in spatial.su2_lifts
        )
        image_type = _image_type_from_matrices(images)
        marking_shifts = tuple(
            _marking_shift(representative_lift, lifts)
            for _, representative_lift in _centralizer_component_representatives(
                image_type, images
            )
        )
        result.append(
            Z2LocalSkeleton(
                spatial_multiplication_table=source_table,
                so3_images=images,
                su2_lifts=lifts,
                defect_bits=_defect(multiplication, lifts),
                projected_image_type_id=image_type,
                kernel_elements=tuple(
                    index for index, image in enumerate(images) if image == identity_so3()
                ),
                graded=True,
                marking_shifts=marking_shifts,
            )
        )
    return tuple(result)
