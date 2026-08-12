"""Witness-gated exact crystallographic ``SO(3)`` targets and lifts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import json
from typing import Iterable, Literal

from .algebraic import (
    ONE_Q23,
    ONE_QUATERNION,
    SQRT2,
    SQRT3,
    ZERO_Q23,
    ExactQuaternion,
    ExactSO3,
    Q23,
    identity_so3,
)


HostID = Literal["O", "D6"]


def _q(value: int | Fraction) -> Q23:
    return Q23.from_rational(value)


def _digest(payload: object, domain: str) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    return sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    ) % 2


def _rotation_table_digest(
    host_id: HostID,
    matrices: tuple[ExactSO3, ...],
    table: tuple[tuple[int, ...], ...],
) -> str:
    return _digest(
        {
            "host_id": host_id,
            "matrices": [matrix.to_json() for matrix in matrices],
            "multiplication_table": table,
        },
        "mathpsg-finite-rotation-table-v1",
    )


@dataclass(frozen=True)
class CertifiedHostElement:
    host_id: HostID
    element_index: int
    matrix: ExactSO3
    table_witness_digest: str

    def __post_init__(self) -> None:
        if self.host_id not in {"O", "D6"}:
            raise ValueError("unknown exact rotation host")
        if type(self.element_index) is not int or self.element_index < 0:
            raise TypeError("element index must be a nonnegative exact int")
        if type(self.matrix) is not ExactSO3:
            raise TypeError("host matrix must be an ExactSO3")
        if (
            type(self.table_witness_digest) is not str
            or len(self.table_witness_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.table_witness_digest)
        ):
            raise ValueError("table witness digest must be lowercase SHA-256")

    @property
    def coordinate_permutation_parity(self) -> int:
        permutation: list[int] = []
        for row in self.matrix.rows:
            nonzero = [column for column, entry in enumerate(row) if entry]
            if len(nonzero) != 1 or row[nonzero[0]] not in {_q(-1), ONE_Q23}:
                raise ValueError("host element is not a signed coordinate permutation")
            permutation.append(nonzero[0])
        if len(set(permutation)) != 3:
            raise ValueError("host element is not a coordinate permutation")
        return _permutation_parity(tuple(permutation))


@dataclass(frozen=True)
class FiniteRotationGroup:
    host_id: HostID
    elements: tuple[CertifiedHostElement, ...]
    multiplication_table: tuple[tuple[int, ...], ...]
    inverse_indices: tuple[int, ...]
    element_orders: tuple[int, ...]
    table_witness_digest: str
    identity_index: int

    def __post_init__(self) -> None:
        if self.host_id not in {"O", "D6"}:
            raise ValueError("unknown finite rotation host")
        if type(self.elements) is not tuple or not self.elements:
            raise ValueError("finite rotation group must contain elements")
        if any(type(element) is not CertifiedHostElement for element in self.elements):
            raise TypeError("host elements must be certified")
        order = len(self.elements)
        expected_matrices = _canonical_host_matrices(self.host_id)
        actual_matrices = tuple(element.matrix for element in self.elements)
        if actual_matrices != expected_matrices:
            raise ValueError(
                f"{self.host_id} certificate does not contain its exact canonical host universe"
            )
        if type(self.multiplication_table) is not tuple or len(self.multiplication_table) != order:
            raise ValueError("multiplication table has the wrong order")
        if any(type(row) is not tuple or len(row) != order for row in self.multiplication_table):
            raise ValueError("multiplication table must be square")
        if any(type(value) is not int or not 0 <= value < order for row in self.multiplication_table for value in row):
            raise ValueError("multiplication table contains an invalid index")
        if type(self.inverse_indices) is not tuple or len(self.inverse_indices) != order:
            raise ValueError("inverse table has the wrong order")
        if type(self.element_orders) is not tuple or len(self.element_orders) != order:
            raise ValueError("element-order table has the wrong order")
        if type(self.identity_index) is not int or not 0 <= self.identity_index < order:
            raise ValueError("identity index is invalid")
        if (
            type(self.table_witness_digest) is not str
            or len(self.table_witness_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.table_witness_digest)
        ):
            raise ValueError("table witness digest must be lowercase SHA-256")
        if any(type(index) is not int or not 0 <= index < order for index in self.inverse_indices):
            raise TypeError("inverse indices must be exact in-range ints")
        if any(type(value) is not int or value < 1 for value in self.element_orders):
            raise TypeError("element orders must be positive exact ints")
        for index, element in enumerate(self.elements):
            if (
                element.host_id != self.host_id
                or element.element_index != index
                or element.table_witness_digest != self.table_witness_digest
            ):
                raise ValueError("host element certificate does not bind this table")
        expected_digest = _rotation_table_digest(
            self.host_id,
            tuple(element.matrix for element in self.elements),
            self.multiplication_table,
        )
        if self.table_witness_digest != expected_digest:
            raise ValueError("table witness digest does not bind the exact host table")
        if self.elements[self.identity_index].matrix != identity_so3():
            raise ValueError("identity index does not name the exact identity")
        for left in range(order):
            if self.multiplication_table[self.identity_index][left] != left or self.multiplication_table[left][self.identity_index] != left:
                raise ValueError("multiplication table has an invalid identity")
            inverse = self.inverse_indices[left]
            if self.multiplication_table[left][inverse] != self.identity_index or self.multiplication_table[inverse][left] != self.identity_index:
                raise ValueError("inverse witness does not match multiplication table")
            power = self.identity_index
            witnessed_order = None
            for exponent in range(1, order + 1):
                power = self.multiplication_table[power][left]
                if power == self.identity_index:
                    witnessed_order = exponent
                    break
            if witnessed_order != self.element_orders[left]:
                raise ValueError("element-order witness does not match multiplication table")
            for right in range(order):
                product_index = self.multiplication_table[left][right]
                if self.elements[left].matrix @ self.elements[right].matrix != self.elements[product_index].matrix:
                    raise ValueError("multiplication table does not match exact matrices")

    def multiply_indices(self, left: int, right: int) -> int:
        self._validate_index(left)
        self._validate_index(right)
        return self.multiplication_table[left][right]

    def inverse_index(self, index: int) -> int:
        self._validate_index(index)
        return self.inverse_indices[index]

    def element_order(self, index: int) -> int:
        self._validate_index(index)
        return self.element_orders[index]

    def _validate_index(self, index: int) -> None:
        if type(index) is not int:
            raise TypeError("group index must be an exact int")
        if not 0 <= index < len(self.elements):
            raise ValueError("group index is outside the certified host universe")

    def subgroup_generated(self, generators: Iterable[int]) -> tuple[int, ...]:
        seed = tuple(generators)
        if any(type(index) is not int or not 0 <= index < len(self.elements) for index in seed):
            raise ValueError("subgroup generator index is invalid")
        subgroup = {self.identity_index, *seed}
        changed = True
        while changed:
            changed = False
            current = tuple(subgroup)
            for left in current:
                for right in current:
                    product_index = self.multiply_indices(left, right)
                    if product_index not in subgroup:
                        subgroup.add(product_index)
                        changed = True
        return tuple(sorted(subgroup))


def _matrix_from_integer_rows(rows: tuple[tuple[int, int, int], ...]) -> ExactSO3:
    return ExactSO3(
        tuple(tuple(_q(entry) for entry in row) for row in rows)  # type: ignore[arg-type]
    )


def _build_group(host_id: HostID, matrices: Iterable[ExactSO3]) -> FiniteRotationGroup:
    ordered = tuple(sorted(set(matrices), key=lambda matrix: matrix.canonical_key))
    if not ordered or any(not matrix.is_rotation() for matrix in ordered):
        raise ArithmeticError("host contains a nonrotation")
    index = {matrix: position for position, matrix in enumerate(ordered)}
    try:
        table = tuple(
            tuple(index[left @ right] for right in ordered)
            for left in ordered
        )
    except KeyError as error:
        raise ArithmeticError("host matrices are not closed") from error
    identity_index = index[identity_so3()]
    inverse_indices = tuple(
        next(
            right
            for right in range(len(ordered))
            if table[left][right] == identity_index and table[right][left] == identity_index
        )
        for left in range(len(ordered))
    )
    element_orders: list[int] = []
    for value in range(len(ordered)):
        product_index = identity_index
        for exponent in range(1, len(ordered) + 1):
            product_index = table[product_index][value]
            if product_index == identity_index:
                element_orders.append(exponent)
                break
        else:
            raise ArithmeticError("finite host element has no certified order")
    digest = _rotation_table_digest(host_id, ordered, table)
    elements = tuple(
        CertifiedHostElement(host_id, position, matrix, digest)
        for position, matrix in enumerate(ordered)
    )
    return FiniteRotationGroup(
        host_id,
        elements,
        table,
        inverse_indices,
        tuple(element_orders),
        digest,
        identity_index,
    )


@lru_cache(maxsize=1)
def octahedral_rotation_group() -> FiniteRotationGroup:
    group = _build_group("O", _octahedral_matrices())
    if len(group.elements) != 24:
        raise ArithmeticError("octahedral host must have order 24")
    return group


@lru_cache(maxsize=1)
def _octahedral_matrices() -> tuple[ExactSO3, ...]:
    matrices: list[ExactSO3] = []
    for permutation in permutations(range(3)):
        parity = _permutation_parity(permutation)
        for signs in product((-1, 1), repeat=3):
            sign_product = signs[0] * signs[1] * signs[2]
            determinant = (-1 if parity else 1) * sign_product
            if determinant != 1:
                continue
            rows = tuple(
                tuple(signs[row] if column == permutation[row] else 0 for column in range(3))
                for row in range(3)
            )
            matrices.append(_matrix_from_integer_rows(rows))
    result = tuple(sorted(set(matrices), key=lambda matrix: matrix.canonical_key))
    if len(result) != 24:
        raise ArithmeticError("octahedral host must have order 24")
    return result


def _hexagonal_cos_sin(index: int) -> tuple[Q23, Q23]:
    half = _q(Fraction(1, 2))
    root3_half = SQRT3 * half
    values = (
        (ONE_Q23, ZERO_Q23),
        (half, root3_half),
        (-half, root3_half),
        (-ONE_Q23, ZERO_Q23),
        (-half, -root3_half),
        (half, -root3_half),
    )
    return values[index % 6]


def _z_rotation_sixth(index: int) -> ExactSO3:
    cosine, sine = _hexagonal_cos_sin(index)
    return ExactSO3(
        (
            (cosine, -sine, ZERO_Q23),
            (sine, cosine, ZERO_Q23),
            (ZERO_Q23, ZERO_Q23, ONE_Q23),
        )
    )


@lru_cache(maxsize=1)
def dihedral_six_rotation_group() -> FiniteRotationGroup:
    group = _build_group("D6", _dihedral_six_matrices())
    if len(group.elements) != 12:
        raise ArithmeticError("dihedral-six host must have order 12")
    return group


@lru_cache(maxsize=1)
def _dihedral_six_matrices() -> tuple[ExactSO3, ...]:
    flip = ExactSO3.diagonal((1, -1, -1))
    rotations = tuple(_z_rotation_sixth(index) for index in range(6))
    matrices = rotations + tuple(rotation @ flip for rotation in rotations)
    result = tuple(sorted(set(matrices), key=lambda matrix: matrix.canonical_key))
    if len(result) != 12:
        raise ArithmeticError("dihedral-six host must have order 12")
    return result


def _canonical_host_matrices(host_id: HostID) -> tuple[ExactSO3, ...]:
    if host_id == "O":
        return _octahedral_matrices()
    if host_id == "D6":
        return _dihedral_six_matrices()
    raise ValueError("unknown exact rotation host")


def _host(host_id: HostID) -> FiniteRotationGroup:
    if host_id == "O":
        return octahedral_rotation_group()
    if host_id == "D6":
        return dihedral_six_rotation_group()
    raise ValueError("unknown exact rotation host")


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
    if len(candidates) != 48:
        raise ArithmeticError("binary octahedral candidate set must have order 48")
    return tuple(candidates)


def _half_angle_cos_sin(index: int) -> tuple[Q23, Q23]:
    half = _q(Fraction(1, 2))
    root3_half = SQRT3 * half
    values = (
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
    )
    return values[index % 12]


def _binary_dihedral_candidates() -> tuple[ExactQuaternion, ...]:
    candidates: set[ExactQuaternion] = set()
    for index in range(12):
        cosine, sine = _half_angle_cos_sin(index)
        candidates.add(ExactQuaternion(cosine, ZERO_Q23, ZERO_Q23, sine))
        candidates.add(ExactQuaternion(ZERO_Q23, cosine, sine, ZERO_Q23))
    if len(candidates) != 24:
        raise ArithmeticError("binary dihedral-six candidate set must have order 24")
    return tuple(candidates)


@lru_cache(maxsize=2)
def _canonical_lifts(host_id: HostID) -> tuple[ExactQuaternion, ...]:
    group = _host(host_id)
    candidates = (
        _binary_octahedral_candidates()
        if host_id == "O"
        else _binary_dihedral_candidates()
    )
    by_matrix: dict[ExactSO3, ExactQuaternion] = {}
    for candidate in candidates:
        matrix = candidate.to_so3()
        canonical = candidate.canonicalized()
        previous = by_matrix.get(matrix)
        if previous is not None and previous != canonical:
            raise ArithmeticError("rotation has inconsistent canonical lifts")
        by_matrix[matrix] = canonical
    try:
        result = tuple(by_matrix[element.matrix] for element in group.elements)
    except KeyError as error:
        raise ArithmeticError("binary host does not cover its rotation host") from error
    if len(by_matrix) != len(group.elements):
        raise ArithmeticError("binary host projects outside its rotation host")
    return result


def _image_digest(image: tuple[ExactSO3, ...]) -> str:
    return _digest(
        [matrix.to_json() for matrix in image],
        "mathpsg-exact-rotation-image-v1",
    )


def _canonical_rotation_image(
    image: tuple[ExactSO3, ...], context: str
) -> tuple[ExactSO3, ...]:
    if type(image) is not tuple:
        raise TypeError(f"{context} image must be an exact tuple")
    if len(image) < 2:
        raise ValueError(f"{context} image must contain a nonidentity rotation")
    if any(type(matrix) is not ExactSO3 for matrix in image):
        raise TypeError(f"{context} image must contain ExactSO3 values")
    if any(not matrix.is_rotation() for matrix in image):
        raise ValueError(f"{context} image contains a non-SO(3) matrix")
    canonical = tuple(sorted(set(image), key=lambda matrix: matrix.canonical_key))
    if len(canonical) != len(image):
        raise ValueError(f"{context} image contains duplicate rotations")
    universe = set(canonical)
    if identity_so3() not in universe:
        raise ValueError(f"{context} image does not contain the identity")
    if any(left @ right not in universe for left in canonical for right in canonical):
        raise ValueError(f"{context} image is not closed under multiplication")
    return canonical


@dataclass(frozen=True)
class CertifiedConjugator:
    matrix: ExactSO3
    source_image_digest: str
    target_image_digest: str
    conjugacy_witness: tuple[int, ...]
    source_image: tuple[ExactSO3, ...]
    target_image: tuple[ExactSO3, ...]
    lift_witness: ExactQuaternion

    def __post_init__(self) -> None:
        if type(self.matrix) is not ExactSO3 or type(self.lift_witness) is not ExactQuaternion:
            raise TypeError("conjugator and lift must be exact values")
        if type(self.source_image) is not tuple or type(self.target_image) is not tuple:
            raise TypeError("conjugator images must be exact tuples")
        if any(type(matrix) is not ExactSO3 for matrix in self.source_image + self.target_image):
            raise TypeError("conjugator images must contain ExactSO3 values")
        if type(self.conjugacy_witness) is not tuple or any(type(index) is not int for index in self.conjugacy_witness):
            raise TypeError("conjugacy witness must be an exact tuple of indices")
        for label, digest in (
            ("source", self.source_image_digest),
            ("target", self.target_image_digest),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} image digest must be lowercase SHA-256")
        if len(self.source_image) != len(self.target_image) or len(self.conjugacy_witness) != len(self.source_image):
            raise ValueError("conjugator certificate dimensions do not agree")


def _known_lift(matrix: ExactSO3) -> ExactQuaternion | None:
    for group in (octahedral_rotation_group(), dihedral_six_rotation_group()):
        for element, lift in zip(group.elements, _canonical_lifts(group.host_id), strict=True):
            if element.matrix == matrix:
                return lift
    return None


def certify_conjugator(
    matrix: ExactSO3,
    source_image: tuple[ExactSO3, ...],
    target_image: tuple[ExactSO3, ...],
    *,
    lift_witness: ExactQuaternion | None = None,
) -> CertifiedConjugator:
    if type(matrix) is not ExactSO3 or not matrix.is_rotation():
        raise ValueError("conjugator must be an exact proper rotation")
    source_image = _canonical_rotation_image(source_image, "source")
    target_image = _canonical_rotation_image(target_image, "target")
    if len(source_image) != len(target_image):
        raise ValueError("source and target images must have equal unique cardinality")
    target_index = {value: index for index, value in enumerate(target_image)}
    witness: list[int] = []
    for source in source_image:
        if type(source) is not ExactSO3:
            raise TypeError("source image contains a non-exact rotation")
        transformed = matrix @ source @ matrix.transpose()
        if transformed not in target_index:
            raise ValueError("matrix does not conjugate source image to target image")
        witness.append(target_index[transformed])
    if sorted(witness) != list(range(len(target_image))):
        raise ValueError("conjugacy witness is not a bijection")
    chosen_lift = lift_witness if lift_witness is not None else _known_lift(matrix)
    if type(chosen_lift) is not ExactQuaternion or chosen_lift.to_so3() != matrix:
        raise ValueError("conjugator requires an exact quaternion lift witness in Q23")
    chosen_lift = chosen_lift.canonicalized()
    return CertifiedConjugator(
        matrix,
        _image_digest(source_image),
        _image_digest(target_image),
        tuple(witness),
        source_image,
        target_image,
        chosen_lift,
    )


def _verify_conjugator(certificate: CertifiedConjugator) -> ExactQuaternion:
    expected = certify_conjugator(
        certificate.matrix,
        certificate.source_image,
        certificate.target_image,
        lift_witness=certificate.lift_witness,
    )
    if certificate != expected:
        raise ValueError("conjugator certificate does not replay exactly")
    return expected.lift_witness


def lift_certified_rotation(
    rotation: CertifiedHostElement | CertifiedConjugator,
) -> tuple[ExactQuaternion, ExactQuaternion]:
    if type(rotation) is CertifiedHostElement:
        group = _host(rotation.host_id)
        if rotation.element_index >= len(group.elements) or rotation != group.elements[rotation.element_index]:
            raise ValueError("host element certificate does not replay exactly")
        positive = _canonical_lifts(rotation.host_id)[rotation.element_index]
    elif type(rotation) is CertifiedConjugator:
        positive = _verify_conjugator(rotation)
    else:
        raise TypeError("rotation lift requires a certified host element or conjugator")
    if positive.to_so3() != rotation.matrix or not positive.has_canonical_sign():
        raise ArithmeticError("invalid canonical quaternion lift")
    return positive, -positive


class UnsupportedExactRotationField(ValueError):
    def __init__(self, *, required: str) -> None:
        self.required = required
        super().__init__(f"rotation requires unsupported exact field {required}")


class UnsupportedCrystallographicRotation(ValueError):
    def __init__(self, *, order: int) -> None:
        if type(order) is not int or order < 1:
            raise TypeError("rotation order must be a positive exact int")
        self.order = order
        super().__init__(f"rotation has noncrystallographic order {order}")


def certified_axis_rotation(angle_over_pi: Fraction) -> CertifiedHostElement:
    if type(angle_over_pi) is not Fraction:
        raise TypeError("angle_over_pi must be an exact Fraction")
    normalized = angle_over_pi % 2
    candidates: list[CertifiedHostElement] = []
    for group in (octahedral_rotation_group(), dihedral_six_rotation_group()):
        for element in group.elements:
            matrix = element.matrix
            if (
                matrix.rows[2] == (ZERO_Q23, ZERO_Q23, ONE_Q23)
                and matrix.rows[0][2] == ZERO_Q23
                and matrix.rows[1][2] == ZERO_Q23
            ):
                candidates.append(element)
    # Compare the requested rational multiple against the two certified
    # crystallographic angle grids without evaluating trigonometric floats.
    allowed: dict[Fraction, list[CertifiedHostElement]] = {}
    for index in range(4):
        target = _z_rotation_quarter(index)
        allowed.setdefault(Fraction(index, 2) % 2, []).extend(
            element for element in candidates if element.matrix == target
        )
    for index in range(6):
        target = _z_rotation_sixth(index)
        allowed.setdefault(Fraction(index, 3) % 2, []).extend(
            element for element in candidates if element.matrix == target
        )
    matches = allowed.get(normalized, [])
    if matches:
        return sorted(matches, key=lambda element: (element.host_id != "O", element.element_index))[0]
    if normalized.denominator == 6:
        raise UnsupportedCrystallographicRotation(order=12)
    required = "sqrt5" if normalized.denominator % 5 == 0 else "outside-Q23"
    raise UnsupportedExactRotationField(required=required)


def _z_rotation_quarter(index: int) -> ExactSO3:
    values = (
        (ONE_Q23, ZERO_Q23),
        (ZERO_Q23, ONE_Q23),
        (-ONE_Q23, ZERO_Q23),
        (ZERO_Q23, -ONE_Q23),
    )
    cosine, sine = values[index % 4]
    return ExactSO3(
        (
            (cosine, -sine, ZERO_Q23),
            (sine, cosine, ZERO_Q23),
            (ZERO_Q23, ZERO_Q23, ONE_Q23),
        )
    )
