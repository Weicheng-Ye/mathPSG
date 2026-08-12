r"""Exact local :math:`\operatorname{Pin}^{-}(2)` data for compact-U(1) PSGs.

Write a normalizer section as ``w**q K**a``, where ``w`` is the odd
unitary normalizer element, ``w**2 = -1``, and ``a`` is the antiunitary
grade.  Its action on the U(1) coefficient is

``rho = a + q (mod 2)``,

while its normalized section defect is

``beta((q,a), (q',a')) = (q*q' + a*q') / 2 (mod 1)``.

The distinction between raw ``q``, antiunitary grade ``a``, and effective
coefficient character ``rho`` is deliberately retained in every cache
identity.  This module consumes the structural normalized-bar boundary of
Task 5 without defining a competing ``FiniteGroupTable`` type.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

from .cochains import FiniteGroupTable
from .gf2 import GF2Character
from .torus import Phase


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROTOCOL_PREFIX = b"mathpsg-u1-local-v1|"
_ZERO = Phase(Fraction(0))


def _bit(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path}: expected an exact bit")
    if value not in (0, 1):
        raise ValueError(f"{path}: expected a bit")
    return value


def _digest_value(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256:<64 lowercase hex digits>")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_digest(domain: str, payload: object) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL_PREFIX + domain.encode("ascii") + b"|" + _canonical_json(payload)
    ).hexdigest()


def _phase_text(phase: Phase) -> str:
    value = phase.value
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _vector_digest(
    domain: str,
    table_dependency_digest: str,
    element_order: tuple[str, ...],
    values: tuple[int, ...],
) -> str:
    return _domain_digest(
        domain,
        {
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_dependency_digest,
            "values": list(values),
        },
    )


def _bar_defect_digest(
    table_dependency_digest: str,
    element_order: tuple[str, ...],
    restricted_rho_digest: str,
    defect: tuple[tuple[Phase, ...], ...],
) -> str:
    return _domain_digest(
        "normalized-bar-defect",
        {
            "algorithm": "pin-minus-normalized-bar-pullback-v1",
            "coefficient_twist_digest": restricted_rho_digest,
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_dependency_digest,
            "twisted_action": "phase -> (-1)^rho(left) phase",
            "values": [[_phase_text(phase) for phase in row] for row in defect],
        },
    )


def _skeleton_digest(
    table_dependency_digest: str,
    element_order: tuple[str, ...],
    restricted_grade_digest: str,
    restricted_rho_digest: str,
    q_assignment_digest: str,
    bar_cocycle_digest: str,
) -> str:
    return _domain_digest(
        "local-skeleton",
        {
            "bar_cocycle_digest": bar_cocycle_digest,
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_dependency_digest,
            "q_assignment_digest": q_assignment_digest,
            "restricted_grade_digest": restricted_grade_digest,
            "restricted_rho_digest": restricted_rho_digest,
        },
    )


@dataclass(frozen=True, slots=True, order=True)
class NormalizerCoset:
    """One of the four raw normalizer/antiunitary section cosets."""

    q: int
    a: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _bit(self.q, "$NormalizerCoset.q"))
        object.__setattr__(self, "a", _bit(self.a, "$NormalizerCoset.a"))


@dataclass(frozen=True, slots=True)
class U1LocalSkeleton:
    """Immutable but untrusted-until-verified local pullback certificate."""

    skeleton_id: str
    table_dependency_digest: str
    element_order: tuple[str, ...]
    grade_values: tuple[int, ...]
    rho_values: tuple[int, ...]
    q_values: tuple[int, ...]
    restricted_grade_digest: str
    restricted_rho_digest: str
    q_assignment_digest: str
    normalized_bar_defect: tuple[tuple[Phase, ...], ...]
    bar_cocycle_digest: str

    def __post_init__(self) -> None:
        _validate_skeleton_content_hashes(self)


def _validate_bit_tuple(
    values: object,
    path: str,
    *,
    length: int | None = None,
) -> tuple[int, ...]:
    if type(values) is not tuple or (length is None and not values):
        raise ValueError(f"{path}: expected a nonempty tuple")
    if length is not None and len(values) != length:
        raise ValueError(f"{path}: expected length {length}")
    for index, value in enumerate(values):
        _bit(value, f"{path}[{index}]")
    return values


def _validate_skeleton_content_hashes(skeleton: U1LocalSkeleton) -> None:
    """Validate shape/content hashes only; this is not mathematical authority."""

    if type(skeleton) is not U1LocalSkeleton:
        raise TypeError("$U1LocalSkeleton: expected exact U1LocalSkeleton")
    _digest_value(skeleton.skeleton_id, "$U1LocalSkeleton.skeleton_id")
    _digest_value(
        skeleton.table_dependency_digest,
        "$U1LocalSkeleton.table_dependency_digest",
    )
    if type(skeleton.element_order) is not tuple or not skeleton.element_order:
        raise ValueError("$U1LocalSkeleton.element_order: expected a nonempty tuple")
    for index, element in enumerate(skeleton.element_order):
        if type(element) is not str or not element or element.strip() != element:
            raise ValueError(
                f"$U1LocalSkeleton.element_order[{index}]: expected a nonempty trimmed string"
            )
    if len(set(skeleton.element_order)) != len(skeleton.element_order):
        raise ValueError("$U1LocalSkeleton.element_order: duplicate canonical element")
    order = len(skeleton.element_order)
    grade_values = _validate_bit_tuple(
        skeleton.grade_values,
        "$U1LocalSkeleton.grade_values",
        length=order,
    )
    rho_values = _validate_bit_tuple(
        skeleton.rho_values,
        "$U1LocalSkeleton.rho_values",
        length=order,
    )
    q_values = _validate_bit_tuple(
        skeleton.q_values,
        "$U1LocalSkeleton.q_values",
        length=order,
    )

    _digest_value(
        skeleton.restricted_grade_digest,
        "$U1LocalSkeleton.restricted_grade_digest",
    )
    _digest_value(
        skeleton.restricted_rho_digest,
        "$U1LocalSkeleton.restricted_rho_digest",
    )
    _digest_value(
        skeleton.q_assignment_digest,
        "$U1LocalSkeleton.q_assignment_digest",
    )
    _digest_value(
        skeleton.bar_cocycle_digest,
        "$U1LocalSkeleton.bar_cocycle_digest",
    )
    expected_grade_digest = _vector_digest(
        "restricted-grade",
        skeleton.table_dependency_digest,
        skeleton.element_order,
        grade_values,
    )
    if skeleton.restricted_grade_digest != expected_grade_digest:
        raise ValueError(
            "$U1LocalSkeleton.restricted_grade_digest: digest does not bind grade_values"
        )
    expected_rho_digest = _vector_digest(
        "restricted-rho",
        skeleton.table_dependency_digest,
        skeleton.element_order,
        rho_values,
    )
    if skeleton.restricted_rho_digest != expected_rho_digest:
        raise ValueError(
            "$U1LocalSkeleton.restricted_rho_digest: digest does not bind rho_values"
        )
    expected_q_digest = _vector_digest(
        "q-assignment",
        skeleton.table_dependency_digest,
        skeleton.element_order,
        q_values,
    )
    if skeleton.q_assignment_digest != expected_q_digest:
        raise ValueError(
            "$U1LocalSkeleton.q_assignment_digest: digest does not bind q_values and table dependency"
        )

    defect = skeleton.normalized_bar_defect
    if type(defect) is not tuple or len(defect) != order:
        raise ValueError(
            "$U1LocalSkeleton.normalized_bar_defect: expected one tuple row per group element"
        )
    for row_index, row in enumerate(defect):
        if type(row) is not tuple or len(row) != order:
            raise ValueError(
                f"$U1LocalSkeleton.normalized_bar_defect[{row_index}]: expected length {order}"
            )
        for column_index, phase in enumerate(row):
            if type(phase) is not Phase:
                raise TypeError(
                    "$U1LocalSkeleton.normalized_bar_defect"
                    f"[{row_index}][{column_index}]: expected Phase"
                )
    expected_bar_digest = _bar_defect_digest(
        skeleton.table_dependency_digest,
        skeleton.element_order,
        skeleton.restricted_rho_digest,
        defect,
    )
    if skeleton.bar_cocycle_digest != expected_bar_digest:
        raise ValueError(
            "$U1LocalSkeleton.bar_cocycle_digest: digest does not bind table, twist, and normalized defect"
        )
    expected_skeleton_id = _skeleton_digest(
        skeleton.table_dependency_digest,
        skeleton.element_order,
        skeleton.restricted_grade_digest,
        skeleton.restricted_rho_digest,
        skeleton.q_assignment_digest,
        skeleton.bar_cocycle_digest,
    )
    if skeleton.skeleton_id != expected_skeleton_id:
        raise ValueError(
            "$U1LocalSkeleton.skeleton_id: identity does not bind table, grade, rho, q, and defect digests"
        )


@dataclass(frozen=True, slots=True)
class _NormalizedTable:
    group_id: str
    element_order: tuple[str, ...]
    identity_index: int
    multiplication_table: tuple[tuple[int, ...], ...]
    table_digest: str


def effective_character(coset: NormalizerCoset) -> int:
    """Return the U(1)-coefficient action ``rho = a xor q``."""

    if type(coset) is not NormalizerCoset:
        raise TypeError("$effective_character.coset: expected NormalizerCoset")
    return coset.a ^ coset.q


def pin_minus_defect(left: NormalizerCoset, right: NormalizerCoset) -> Phase:
    """Return the exact normalized Pin-minus section defect in ``R/Z``."""

    if type(left) is not NormalizerCoset:
        raise TypeError("$pin_minus_defect.left: expected NormalizerCoset")
    if type(right) is not NormalizerCoset:
        raise TypeError("$pin_minus_defect.right: expected NormalizerCoset")
    return Phase(Fraction(left.q * right.q + left.a * right.q, 2))


def weyl_comparison_cochain(coset: NormalizerCoset) -> Phase:
    """Return the global-Weyl comparison cochain ``mu(q,a) = a/2``."""

    if type(coset) is not NormalizerCoset:
        raise TypeError("$weyl_comparison_cochain.coset: expected NormalizerCoset")
    return Phase(Fraction(coset.a, 2))


def _twisted_one_coboundary(
    value_left: Phase,
    value_right: Phase,
    value_product: Phase,
    rho_left: int,
) -> Phase:
    """Evaluate ``mu(g) + (-1)^rho(g) mu(h) - mu(gh)`` exactly."""

    for label, phase in (
        ("left", value_left),
        ("right", value_right),
        ("product", value_product),
    ):
        if type(phase) is not Phase:
            raise TypeError(f"$twisted_one_coboundary.{label}: expected Phase")
    rho_bit = _bit(rho_left, "$twisted_one_coboundary.rho_left")
    action_sign = -1 if rho_bit else 1
    return Phase(
        value_left.value
        + action_sign * value_right.value
        - value_product.value
    )


def _normalize_table(value: FiniteGroupTable) -> _NormalizedTable:
    """Revalidate and bind the concrete normalized-bar table owned by Task 5."""

    if type(value) is not FiniteGroupTable:
        raise TypeError("$FiniteGroupTable: expected exact Task-5 FiniteGroupTable")
    # Reconstructing closes object.__setattr__-style mutation attacks and makes
    # the Task-5 content digest, including canonical inverses, authoritative.
    certified = FiniteGroupTable(
        group_id=value.group_id,
        element_order=value.element_order,
        identity_index=value.identity_index,
        multiplication_table=value.multiplication_table,
        inverse_indices=value.inverse_indices,
        table_digest=value.table_digest,
    )
    if certified.table_digest is None:  # pragma: no cover - Task 5 always fills it
        raise ValueError("$FiniteGroupTable.table_digest: Task-5 digest is absent")
    return _NormalizedTable(
        group_id=certified.group_id,
        element_order=certified.element_order,
        identity_index=certified.identity_index,
        multiplication_table=certified.multiplication_table,
        table_digest=certified.table_digest,
    )


def _verify_character_values(
    values: tuple[int, ...],
    table: _NormalizedTable,
    label: str,
) -> tuple[int, ...]:
    order = len(table.element_order)
    _validate_bit_tuple(values, f"${label}", length=order)
    for left in range(order):
        for right in range(order):
            product = table.multiplication_table[left][right]
            if values[product] != (values[left] ^ values[right]):
                raise ValueError(
                    f"{label} is not a homomorphism at multiplication_table"
                    f"[{left}][{right}] = {product}: "
                    f"{values[left]} xor {values[right]} != {values[product]}"
                )
    return values


def _character_values(
    character: GF2Character,
    table: _NormalizedTable,
    label: str,
) -> tuple[int, ...]:
    if type(character) is not GF2Character:
        raise TypeError(f"${label}: expected GF2Character")
    return _verify_character_values(character.bits, table, label)


def _character_digest(
    domain: str,
    table: _NormalizedTable,
    values: tuple[int, ...],
) -> str:
    return _vector_digest(
        domain,
        table.table_digest,
        table.element_order,
        values,
    )


def _verify_twisted_cocycle(
    table: _NormalizedTable,
    rho_values: tuple[int, ...],
    defect: tuple[tuple[Phase, ...], ...],
) -> None:
    order = len(table.element_order)
    identity = table.identity_index
    if any(defect[identity][right] != _ZERO for right in range(order)):
        raise ValueError("Pin-minus defect is not left-normalized")
    if any(defect[left][identity] != _ZERO for left in range(order)):
        raise ValueError("Pin-minus defect is not right-normalized")
    for left in range(order):
        action_sign = -1 if rho_values[left] else 1
        for middle in range(order):
            left_middle = table.multiplication_table[left][middle]
            for right in range(order):
                middle_right = table.multiplication_table[middle][right]
                residual = (
                    action_sign * defect[middle][right].value
                    - defect[left_middle][right].value
                    + defect[left][middle_right].value
                    - defect[left][middle].value
                ) % 1
                if residual:
                    raise ValueError(
                        "Pin-minus twisted cocycle identity failed at multiplication_table"
                        f"[{left}][{middle}] with third index {right}"
                    )


def _verify_weyl_affine_closure(
    table: _NormalizedTable,
    grade_values: tuple[int, ...],
    rho_values: tuple[int, ...],
    q_values: tuple[int, ...],
    defect: tuple[tuple[Phase, ...], ...],
) -> None:
    cosets = tuple(
        NormalizerCoset(q_value, grade_value)
        for q_value, grade_value in zip(q_values, grade_values, strict=True)
    )
    comparison = tuple(weyl_comparison_cochain(coset) for coset in cosets)
    order = len(cosets)
    for left in range(order):
        for right in range(order):
            product = table.multiplication_table[left][right]
            coboundary = _twisted_one_coboundary(
                comparison[left],
                comparison[right],
                comparison[product],
                rho_values[left],
            )
            weyl_transformed = Phase(-defect[left][right].value)
            affine_image = Phase(defect[left][right].value + coboundary.value)
            if weyl_transformed != affine_image:
                raise ValueError(
                    "global-Weyl affine closure failed at multiplication_table"
                    f"[{left}][{right}]"
                )


def verify_u1_local_skeleton(
    skeleton: U1LocalSkeleton,
    table: FiniteGroupTable,
) -> U1LocalSkeleton:
    """Replay every mathematical and dependency claim in a local skeleton.

    Construction plus self-consistent hashes is intentionally insufficient:
    callers cross the authority boundary only by replaying against the exact
    normalized Task-5 finite-group table supplied here.
    """

    if type(skeleton) is not U1LocalSkeleton:
        raise TypeError("$verify_u1_local_skeleton.skeleton: expected U1LocalSkeleton")
    _validate_skeleton_content_hashes(skeleton)
    normalized_table = _normalize_table(table)
    if skeleton.table_dependency_digest != normalized_table.table_digest:
        raise ValueError(
            "$U1LocalSkeleton.table_dependency_digest: does not bind the supplied normalized table"
        )
    if skeleton.element_order != normalized_table.element_order:
        raise ValueError(
            "$U1LocalSkeleton.element_order: differs from the supplied normalized table"
        )

    grade_values = _verify_character_values(
        skeleton.grade_values,
        normalized_table,
        "grade",
    )
    rho_values = _verify_character_values(
        skeleton.rho_values,
        normalized_table,
        "rho",
    )
    expected_q = tuple(
        grade_value ^ rho_value
        for grade_value, rho_value in zip(grade_values, rho_values, strict=True)
    )
    if skeleton.q_values != expected_q:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(skeleton.q_values, expected_q, strict=True)
            )
            if actual != expected
        )
        raise ValueError(
            f"$U1LocalSkeleton.q_values[{mismatch}]: expected grade xor rho"
        )

    expected_grade_digest = _character_digest(
        "restricted-grade",
        normalized_table,
        grade_values,
    )
    expected_rho_digest = _character_digest(
        "restricted-rho",
        normalized_table,
        rho_values,
    )
    expected_q_digest = _character_digest(
        "q-assignment",
        normalized_table,
        expected_q,
    )
    for label, actual, expected in (
        ("restricted_grade_digest", skeleton.restricted_grade_digest, expected_grade_digest),
        ("restricted_rho_digest", skeleton.restricted_rho_digest, expected_rho_digest),
        ("q_assignment_digest", skeleton.q_assignment_digest, expected_q_digest),
    ):
        if actual != expected:
            raise ValueError(f"$U1LocalSkeleton.{label}: dependency replay failed")

    cosets = tuple(
        NormalizerCoset(q_value, grade_value)
        for q_value, grade_value in zip(expected_q, grade_values, strict=True)
    )
    expected_defect = tuple(
        tuple(pin_minus_defect(left, right) for right in cosets)
        for left in cosets
    )
    for left in range(len(cosets)):
        for right in range(len(cosets)):
            if skeleton.normalized_bar_defect[left][right] != expected_defect[left][right]:
                raise ValueError(
                    "Pin-minus defect formula mismatch at normalized_bar_defect"
                    f"[{left}][{right}]"
                )

    _verify_twisted_cocycle(normalized_table, rho_values, skeleton.normalized_bar_defect)
    _verify_weyl_affine_closure(
        normalized_table,
        grade_values,
        rho_values,
        expected_q,
        skeleton.normalized_bar_defect,
    )
    expected_bar_digest = _bar_defect_digest(
        normalized_table.table_digest,
        normalized_table.element_order,
        expected_rho_digest,
        skeleton.normalized_bar_defect,
    )
    if skeleton.bar_cocycle_digest != expected_bar_digest:
        raise ValueError(
            "$U1LocalSkeleton.bar_cocycle_digest: table/twist replay failed"
        )
    expected_skeleton_id = _skeleton_digest(
        normalized_table.table_digest,
        normalized_table.element_order,
        expected_grade_digest,
        expected_rho_digest,
        expected_q_digest,
        expected_bar_digest,
    )
    if skeleton.skeleton_id != expected_skeleton_id:
        raise ValueError("$U1LocalSkeleton.skeleton_id: full replay failed")
    return skeleton


def u1_local_skeleton(
    table: FiniteGroupTable,
    grade: GF2Character,
    rho: GF2Character,
) -> U1LocalSkeleton:
    """Pull the exact Pin-minus defect back to a finite local group.

    ``grade.bits`` and ``rho.bits`` are values in the canonical finite-group
    element order.  Both are replayed against every multiplication-table row.
    The derived raw normalizer parity is ``q = grade xor rho`` elementwise.
    """

    normalized_table = _normalize_table(table)
    grade_values = _character_values(grade, normalized_table, "grade")
    rho_values = _character_values(rho, normalized_table, "rho")
    q_values = tuple(
        grade_value ^ rho_value
        for grade_value, rho_value in zip(grade_values, rho_values, strict=True)
    )

    restricted_grade_digest = _character_digest(
        "restricted-grade", normalized_table, grade_values
    )
    restricted_rho_digest = _character_digest(
        "restricted-rho", normalized_table, rho_values
    )
    q_assignment_digest = _character_digest(
        "q-assignment", normalized_table, q_values
    )

    cosets = tuple(
        NormalizerCoset(q_value, grade_value)
        for q_value, grade_value in zip(q_values, grade_values, strict=True)
    )
    defect = tuple(
        tuple(pin_minus_defect(left, right) for right in cosets)
        for left in cosets
    )
    _verify_twisted_cocycle(normalized_table, rho_values, defect)

    bar_cocycle_digest = _bar_defect_digest(
        normalized_table.table_digest,
        normalized_table.element_order,
        restricted_rho_digest,
        defect,
    )
    skeleton_id = _skeleton_digest(
        normalized_table.table_digest,
        normalized_table.element_order,
        restricted_grade_digest,
        restricted_rho_digest,
        q_assignment_digest,
        bar_cocycle_digest,
    )
    skeleton = U1LocalSkeleton(
        skeleton_id=skeleton_id,
        table_dependency_digest=normalized_table.table_digest,
        element_order=normalized_table.element_order,
        grade_values=grade_values,
        rho_values=rho_values,
        q_values=q_values,
        restricted_grade_digest=restricted_grade_digest,
        restricted_rho_digest=restricted_rho_digest,
        q_assignment_digest=q_assignment_digest,
        normalized_bar_defect=defect,
        bar_cocycle_digest=bar_cocycle_digest,
    )
    return verify_u1_local_skeleton(skeleton, table)


__all__ = [
    "NormalizerCoset",
    "U1LocalSkeleton",
    "effective_character",
    "pin_minus_defect",
    "u1_local_skeleton",
    "verify_u1_local_skeleton",
    "weyl_comparison_cochain",
]
