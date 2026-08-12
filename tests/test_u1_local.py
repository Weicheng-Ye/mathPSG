from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
from typing import get_type_hints
import unittest

from mathpsg.cochains import FiniteGroupTable
from mathpsg.gf2 import GF2Character
from mathpsg.torus import Phase


_MODULE_AVAILABLE = importlib.util.find_spec("mathpsg.u1_local") is not None


_D4_ELEMENTS = ("1", "r", "r2", "r3", "s", "rs", "r2s", "r3s")
_D4_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 2, 3, 0, 5, 6, 7, 4),
    (2, 3, 0, 1, 6, 7, 4, 5),
    (3, 0, 1, 2, 7, 4, 5, 6),
    (4, 7, 6, 5, 0, 3, 2, 1),
    (5, 4, 7, 6, 1, 0, 3, 2),
    (6, 5, 4, 7, 2, 1, 0, 3),
    (7, 6, 5, 4, 3, 2, 1, 0),
)


def _inverse_indices(table: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(
        next(
            candidate
            for candidate in range(len(table))
            if table[index][candidate] == 0 and table[candidate][index] == 0
        )
        for index in range(len(table))
    )


def _finite_table(
    group_id: str,
    elements: tuple[str, ...],
    table: tuple[tuple[int, ...], ...],
) -> FiniteGroupTable:
    return FiniteGroupTable(
        group_id=group_id,
        element_order=elements,
        identity_index=0,
        multiplication_table=table,
        inverse_indices=_inverse_indices(table),
    )


def _d4_table() -> FiniteGroupTable:
    return _finite_table("D4-spatial-v1", _D4_ELEMENTS, _D4_TABLE)


def _d4_character(r_bit: int, s_bit: int) -> tuple[int, ...]:
    rotations = tuple((power * r_bit) % 2 for power in range(4))
    reflections = tuple(((power * r_bit) + s_bit) % 2 for power in range(4))
    return rotations + reflections


def _d4_time_table() -> FiniteGroupTable:
    elements = _D4_ELEMENTS + tuple(f"{name}.T" for name in _D4_ELEMENTS)
    table = tuple(
        tuple(_D4_TABLE[left % 8][right % 8] + 8 * ((left // 8) ^ (right // 8)) for right in range(16))
        for left in range(16)
    )
    return _finite_table("D4xT-graded-v1", elements, table)


def _d4_time_character(r_bit: int, s_bit: int, t_bit: int) -> tuple[int, ...]:
    spatial = _d4_character(r_bit, s_bit)
    return spatial + tuple(value ^ t_bit for value in spatial)


def _retag_table(table: FiniteGroupTable, group_id: str) -> FiniteGroupTable:
    return FiniteGroupTable(
        group_id=group_id,
        element_order=table.element_order,
        identity_index=table.identity_index,
        multiplication_table=table.multiplication_table,
        inverse_indices=table.inverse_indices,
    )


def _domain_digest(domain: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"mathpsg-u1-local-v1|" + domain.encode("ascii") + b"|" + encoded
    ).hexdigest()


def _vector_digest(
    domain: str,
    table_digest: str,
    element_order: tuple[str, ...],
    values: tuple[int, ...],
) -> str:
    return _domain_digest(
        domain,
        {
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_digest,
            "values": list(values),
        },
    )


def _phase_text(phase: Phase) -> str:
    value = phase.value
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _bar_digest(
    table_digest: str,
    element_order: tuple[str, ...],
    rho_digest: str,
    defect: tuple[tuple[Phase, ...], ...],
) -> str:
    return _domain_digest(
        "normalized-bar-defect",
        {
            "algorithm": "pin-minus-normalized-bar-pullback-v1",
            "coefficient_twist_digest": rho_digest,
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_digest,
            "twisted_action": "phase -> (-1)^rho(left) phase",
            "values": [[_phase_text(phase) for phase in row] for row in defect],
        },
    )


def _skeleton_digest(
    table_digest: str,
    element_order: tuple[str, ...],
    grade_digest: str,
    rho_digest: str,
    q_digest: str,
    bar_digest: str,
) -> str:
    return _domain_digest(
        "local-skeleton",
        {
            "bar_cocycle_digest": bar_digest,
            "element_order": list(element_order),
            "normalized_bar_table_digest": table_digest,
            "q_assignment_digest": q_digest,
            "restricted_grade_digest": grade_digest,
            "restricted_rho_digest": rho_digest,
        },
    )


def _self_consistent_forgery(api: object, skeleton: object, **updates: object) -> object:
    payload = {field.name: getattr(skeleton, field.name) for field in fields(skeleton)}
    payload.update(updates)
    table_digest = payload["table_dependency_digest"]
    element_order = payload["element_order"]
    grade_values = payload["grade_values"]
    rho_values = payload["rho_values"]
    q_values = payload["q_values"]
    defect = payload["normalized_bar_defect"]
    grade_digest = _vector_digest("restricted-grade", table_digest, element_order, grade_values)
    rho_digest = _vector_digest("restricted-rho", table_digest, element_order, rho_values)
    q_digest = _vector_digest("q-assignment", table_digest, element_order, q_values)
    bar_digest = _bar_digest(table_digest, element_order, rho_digest, defect)
    payload.update(
        restricted_grade_digest=grade_digest,
        restricted_rho_digest=rho_digest,
        q_assignment_digest=q_digest,
        bar_cocycle_digest=bar_digest,
        skeleton_id=_skeleton_digest(
            table_digest,
            element_order,
            grade_digest,
            rho_digest,
            q_digest,
            bar_digest,
        ),
    )
    return api.U1LocalSkeleton(**payload)


class U1LocalModuleContractTests(unittest.TestCase):
    def test_exact_u1_local_module_exists(self) -> None:
        """Catch the Task-7 target being absent from the public Python package."""

        self.assertTrue(_MODULE_AVAILABLE)

    @unittest.skipUnless(_MODULE_AVAILABLE, "Task-7 production module is the RED boundary")
    def test_verifier_api_and_runtime_annotations_are_resolvable(self) -> None:
        """Catch an absent verifier or a TYPE_CHECKING-only annotation NameError."""

        api = importlib.import_module("mathpsg.u1_local")
        self.assertTrue(hasattr(api, "verify_u1_local_skeleton"))
        factory_hints = get_type_hints(api.u1_local_skeleton)
        verifier_hints = get_type_hints(api.verify_u1_local_skeleton)
        self.assertIs(factory_hints["return"], api.U1LocalSkeleton)
        self.assertIs(verifier_hints["return"], api.U1LocalSkeleton)

    @unittest.skipUnless(_MODULE_AVAILABLE, "Task-7 production module is the RED boundary")
    def test_real_task5_table_digest_is_the_local_dependency_anchor(self) -> None:
        """Catch the temporary structural-table digest drifting from Task 5."""

        api = importlib.import_module("mathpsg.u1_local")
        table = FiniteGroupTable(
            group_id="C2-task5-v1",
            element_order=("1", "g"),
            identity_index=0,
            multiplication_table=((0, 1), (1, 0)),
            inverse_indices=(0, 1),
        )
        skeleton = api.u1_local_skeleton(
            table,
            GF2Character((0, 0)),
            GF2Character((0, 1)),
        )
        self.assertEqual(skeleton.table_dependency_digest, table.table_digest)
        self.assertIs(api.verify_u1_local_skeleton(skeleton, table), skeleton)


@unittest.skipUnless(_MODULE_AVAILABLE, "Task-7 production module is the RED boundary")
class FourCosetPinMinusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = importlib.import_module("mathpsg.u1_local")
        cls.cosets = (
            cls.api.NormalizerCoset(0, 0),
            cls.api.NormalizerCoset(1, 0),
            cls.api.NormalizerCoset(0, 1),
            cls.api.NormalizerCoset(1, 1),
        )

    def test_all_sixteen_section_products_have_the_frozen_pin_minus_defect(self) -> None:
        """Catch either q*q' or a*q' being dropped from the Pin-minus section."""

        zero = Phase(Fraction(0))
        half = Phase(Fraction(1, 2))
        expected = (
            (zero, zero, zero, zero),
            (zero, half, zero, half),
            (zero, half, zero, half),
            (zero, zero, zero, zero),
        )
        actual = tuple(
            tuple(self.api.pin_minus_defect(left, right) for right in self.cosets)
            for left in self.cosets
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            self.api.pin_minus_defect(
                self.api.NormalizerCoset(1, 0),
                self.api.NormalizerCoset(1, 0),
            ),
            half,
        )

    def test_effective_coefficient_character_is_a_xor_q(self) -> None:
        """Catch antiunitary grade being confused with the U(1) module action."""

        self.assertEqual(
            tuple(self.api.effective_character(coset) for coset in self.cosets),
            (0, 1, 1, 0),
        )

    def test_defect_obeys_all_sixty_four_twisted_cocycle_identities(self) -> None:
        """Catch an untwisted or wrong-sided coefficient action in the defect."""

        for left in self.cosets:
            for middle in self.cosets:
                for right in self.cosets:
                    left_middle = self.api.NormalizerCoset(
                        left.q ^ middle.q, left.a ^ middle.a
                    )
                    middle_right = self.api.NormalizerCoset(
                        middle.q ^ right.q, middle.a ^ right.a
                    )
                    action_sign = -1 if self.api.effective_character(left) else 1
                    residual = (
                        action_sign * self.api.pin_minus_defect(middle, right).value
                        - self.api.pin_minus_defect(left_middle, right).value
                        + self.api.pin_minus_defect(left, middle_right).value
                        - self.api.pin_minus_defect(left, middle).value
                    )
                    self.assertEqual(
                        residual % 1,
                        0,
                        msg=f"twisted cocycle failed on {left}, {middle}, {right}",
                    )

    def test_weyl_comparison_is_a_over_two_and_closes_the_affine_relation(self) -> None:
        """Catch a q/2 Weyl cochain or the wrong twisted coboundary convention."""

        expected = (
            Phase(Fraction(0)),
            Phase(Fraction(0)),
            Phase(Fraction(1, 2)),
            Phase(Fraction(1, 2)),
        )
        self.assertEqual(
            tuple(self.api.weyl_comparison_cochain(coset) for coset in self.cosets),
            expected,
        )
        for left in self.cosets:
            for right in self.cosets:
                product = self.api.NormalizerCoset(left.q ^ right.q, left.a ^ right.a)
                sign = -1 if self.api.effective_character(left) else 1
                coboundary = (
                    self.api.weyl_comparison_cochain(left).value
                    + sign * self.api.weyl_comparison_cochain(right).value
                    - self.api.weyl_comparison_cochain(product).value
                )
                defect = self.api.pin_minus_defect(left, right).value
                self.assertEqual((-defect - defect - coboundary) % 1, 0)

    def test_twisted_coboundary_sign_is_visible_away_from_two_torsion(self) -> None:
        """Catch replacing the left rho action by an untwisted plus sign."""

        left = Phase(Fraction(1, 7))
        right = Phase(Fraction(1, 5))
        product = Phase(Fraction(1, 3))
        actual = self.api._twisted_one_coboundary(left, right, product, 1)
        expected = Phase(Fraction(1, 7) - Fraction(1, 5) - Fraction(1, 3))
        untwisted = Phase(Fraction(1, 7) + Fraction(1, 5) - Fraction(1, 3))
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, untwisted)

    def test_cosets_reject_non_bits_including_bool(self) -> None:
        """Catch truthy integers silently entering exact parity arithmetic."""

        for q, a in ((True, 0), (0, False), (-1, 0), (2, 0), (0, "1")):
            with self.subTest(q=q, a=a):
                with self.assertRaises((TypeError, ValueError)):
                    self.api.NormalizerCoset(q, a)


@unittest.skipUnless(_MODULE_AVAILABLE, "Task-7 production module is the RED boundary")
class U1LocalSkeletonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = importlib.import_module("mathpsg.u1_local")

    def test_spatial_d4_pullback_keeps_raw_rho_and_exact_half_phase_support(self) -> None:
        """Catch q being used in place of separately retained rho in the pullback."""

        table = _d4_table()
        grade = GF2Character((0,) * 8)
        rho_values = _d4_character(1, 0)
        skeleton = self.api.u1_local_skeleton(table, grade, GF2Character(rho_values))

        self.assertEqual(skeleton.table_dependency_digest, table.table_digest)
        self.assertEqual(skeleton.element_order, table.element_order)
        self.assertEqual(skeleton.grade_values, grade.bits)
        self.assertEqual(skeleton.rho_values, rho_values)
        self.assertEqual(skeleton.q_values, rho_values)
        half_left = {"r", "r3", "rs", "r3s"}
        half_right = {"r", "r3", "rs", "r3s"}
        for left, left_name in enumerate(table.element_order):
            for right, right_name in enumerate(table.element_order):
                expected = Fraction(1, 2) if left_name in half_left and right_name in half_right else Fraction(0)
                self.assertEqual(
                    skeleton.normalized_bar_defect[left][right].value,
                    expected,
                    msg=f"wrong D4 defect on [{left_name}|{right_name}]",
                )
        self.assertTrue(all(value == Phase(Fraction(0)) for value in skeleton.normalized_bar_defect[0]))
        self.assertTrue(all(row[0] == Phase(Fraction(0)) for row in skeleton.normalized_bar_defect))

    def test_graded_d4_pullback_uses_q_equal_grade_xor_rho(self) -> None:
        """Catch raw q and the effective character rho being interchanged in a graded group."""

        table = _d4_time_table()
        grade_values = _d4_time_character(0, 0, 1)
        rho_values = _d4_time_character(0, 1, 0)
        expected_q = tuple(a ^ rho for a, rho in zip(grade_values, rho_values, strict=True))
        skeleton = self.api.u1_local_skeleton(
            table,
            GF2Character(grade_values),
            GF2Character(rho_values),
        )

        self.assertEqual(skeleton.q_values, expected_q)
        self.assertEqual(skeleton.grade_values, grade_values)
        self.assertEqual(skeleton.rho_values, rho_values)
        half_left = {"s", "rs", "r2s", "r3s", "s.T", "rs.T", "r2s.T", "r3s.T"}
        half_right = {"s", "rs", "r2s", "r3s", "1.T", "r.T", "r2.T", "r3.T"}
        actual_support = {
            (table.element_order[left], table.element_order[right])
            for left in range(16)
            for right in range(16)
            if skeleton.normalized_bar_defect[left][right] == Phase(Fraction(1, 2))
        }
        self.assertEqual(actual_support, {(left, right) for left in half_left for right in half_right})

    def test_each_independent_rho_flip_changes_rho_q_and_skeleton_id(self) -> None:
        """Catch a cache key that omits restricted rho or the derived q assignment."""

        table = _d4_time_table()
        grade = GF2Character(_d4_time_character(0, 0, 1))
        skeletons = tuple(
            self.api.u1_local_skeleton(
                table,
                grade,
                GF2Character(_d4_time_character(r_bit, s_bit, t_bit)),
            )
            for r_bit, s_bit, t_bit in (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            )
        )
        baseline = skeletons[0]
        for flipped in skeletons[1:]:
            self.assertEqual(flipped.restricted_grade_digest, baseline.restricted_grade_digest)
            self.assertNotEqual(flipped.restricted_rho_digest, baseline.restricted_rho_digest)
            self.assertNotEqual(flipped.q_assignment_digest, baseline.q_assignment_digest)
            self.assertNotEqual(flipped.skeleton_id, baseline.skeleton_id)

    def test_identity_binds_grade_rho_and_q_even_when_q_and_defect_coincide(self) -> None:
        """Catch cache aliasing between distinct (grade,rho) sectors with identical q."""

        table = _d4_table()
        zero = GF2Character((0,) * 8)
        reflection = GF2Character(_d4_character(0, 1))
        spatial = self.api.u1_local_skeleton(table, zero, zero)
        paired = self.api.u1_local_skeleton(table, reflection, reflection)

        self.assertEqual(spatial.q_values, paired.q_values)
        self.assertEqual(spatial.normalized_bar_defect, paired.normalized_bar_defect)
        self.assertNotEqual(spatial.restricted_grade_digest, paired.restricted_grade_digest)
        self.assertNotEqual(spatial.restricted_rho_digest, paired.restricted_rho_digest)
        self.assertNotEqual(spatial.skeleton_id, paired.skeleton_id)

    def test_certificate_is_deterministic_deeply_immutable_and_sha256_bound(self) -> None:
        """Catch mutable nested defects or process-dependent cache identities."""

        table = _d4_table()
        args = (table, GF2Character((0,) * 8), GF2Character(_d4_character(1, 1)))
        left = self.api.u1_local_skeleton(*args)
        right = self.api.u1_local_skeleton(*args)

        self.assertEqual(left, right)
        for digest in (
            left.skeleton_id,
            left.restricted_grade_digest,
            left.restricted_rho_digest,
            left.q_assignment_digest,
            left.bar_cocycle_digest,
        ):
            self.assertRegex(digest, r"\Asha256:[0-9a-f]{64}\Z")
        self.assertIs(type(left.q_values), tuple)
        self.assertIs(type(left.normalized_bar_defect), tuple)
        self.assertTrue(all(type(row) is tuple for row in left.normalized_bar_defect))
        self.assertIs(type(left.element_order), tuple)
        self.assertIs(type(left.grade_values), tuple)
        self.assertIs(type(left.rho_values), tuple)
        self.assertIs(self.api.verify_u1_local_skeleton(left, table), left)
        with self.assertRaises(FrozenInstanceError):
            left.skeleton_id = "sha256:" + "0" * 64
        with self.assertRaises(TypeError):
            left.normalized_bar_defect[0][0] = Phase(Fraction(1, 2))

    def test_direct_constructor_rejects_stale_identity_and_defect_digests(self) -> None:
        """Catch a syntactically valid digest being detached from certificate content."""

        skeleton = self.api.u1_local_skeleton(
            _d4_table(),
            GF2Character((0,) * 8),
            GF2Character(_d4_character(1, 0)),
        )
        other_digest = "sha256:" + "0" * 64
        for field in (
            "restricted_grade_digest",
            "restricted_rho_digest",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    replace(skeleton, **{field: other_digest})

        with self.assertRaisesRegex(ValueError, "q_assignment_digest"):
            replace(skeleton, q_assignment_digest=other_digest)
        with self.assertRaisesRegex(ValueError, "bar_cocycle_digest"):
            replace(skeleton, bar_cocycle_digest=other_digest)

        rows = [list(row) for row in skeleton.normalized_bar_defect]
        rows[1][1] = Phase(Fraction(0))
        mutated_defect = tuple(tuple(row) for row in rows)
        with self.assertRaisesRegex(ValueError, "bar_cocycle_digest"):
            replace(skeleton, normalized_bar_defect=mutated_defect)

        mutated_q = list(skeleton.q_values)
        mutated_q[1] ^= 1
        with self.assertRaisesRegex(ValueError, "q_assignment_digest"):
            replace(skeleton, q_values=tuple(mutated_q))

    def test_self_consistent_hash_forgery_is_not_an_authoritative_character(self) -> None:
        """Catch self-consistent hashes bypassing grade homomorphism verification."""

        table = _d4_table()
        valid = self.api.u1_local_skeleton(
            table,
            GF2Character((0,) * 8),
            GF2Character(_d4_character(1, 0)),
        )
        forged_grade = (0, 0, 1, 0, 0, 0, 0, 0)
        forged_q = tuple(
            grade ^ rho
            for grade, rho in zip(forged_grade, valid.rho_values, strict=True)
        )
        cosets = tuple(
            self.api.NormalizerCoset(q, grade)
            for q, grade in zip(forged_q, forged_grade, strict=True)
        )
        forged_defect = tuple(
            tuple(self.api.pin_minus_defect(left, right) for right in cosets)
            for left in cosets
        )
        forged = _self_consistent_forgery(
            self.api,
            valid,
            grade_values=forged_grade,
            q_values=forged_q,
            normalized_bar_defect=forged_defect,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"grade.*multiplication_table\[1\]\[1\]",
        ):
            self.api.verify_u1_local_skeleton(forged, table)

    def test_self_consistent_hash_forgery_cannot_replace_the_pin_defect(self) -> None:
        """Catch a rehashed arbitrary cocycle being accepted as the Pin-minus pullback."""

        table = _d4_table()
        valid = self.api.u1_local_skeleton(
            table,
            GF2Character((0,) * 8),
            GF2Character(_d4_character(1, 0)),
        )
        rows = [list(row) for row in valid.normalized_bar_defect]
        rows[1][1] = Phase(Fraction(0))
        forged = _self_consistent_forgery(
            self.api,
            valid,
            normalized_bar_defect=tuple(tuple(row) for row in rows),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Pin-minus defect.*\[1\]\[1\]",
        ):
            self.api.verify_u1_local_skeleton(forged, table)

    def test_verifier_rebinds_table_dependency_and_canonical_order(self) -> None:
        """Catch a skeleton being replayed against a different Task-5 table identity."""

        table = _d4_table()
        skeleton = self.api.u1_local_skeleton(
            table,
            GF2Character((0,) * 8),
            GF2Character(_d4_character(0, 1)),
        )
        other_group_id = _retag_table(table, "D4-other-embedding-v1")
        with self.assertRaisesRegex(ValueError, "table_dependency_digest"):
            self.api.verify_u1_local_skeleton(skeleton, other_group_id)

    def test_q_and_bar_digests_bind_table_order_and_coefficient_twist(self) -> None:
        """Catch portable-looking digests that omit their normalized-bar dependency."""

        table = _d4_table()
        grade = GF2Character((0,) * 8)
        rho = GF2Character(_d4_character(1, 0))
        first = self.api.u1_local_skeleton(table, grade, rho)
        other = self.api.u1_local_skeleton(
            _retag_table(table, "D4-other-embedding-v1"), grade, rho
        )
        self.assertEqual(first.q_values, other.q_values)
        self.assertEqual(first.normalized_bar_defect, other.normalized_bar_defect)
        self.assertNotEqual(first.q_assignment_digest, other.q_assignment_digest)
        self.assertNotEqual(first.bar_cocycle_digest, other.bar_cocycle_digest)

        zero_twist = self.api.u1_local_skeleton(table, grade, grade)
        changed_twist = self.api.u1_local_skeleton(
            table,
            GF2Character(_d4_character(0, 1)),
            GF2Character(_d4_character(0, 1)),
        )
        self.assertEqual(zero_twist.normalized_bar_defect, changed_twist.normalized_bar_defect)
        self.assertNotEqual(zero_twist.restricted_rho_digest, changed_twist.restricted_rho_digest)
        self.assertNotEqual(zero_twist.bar_cocycle_digest, changed_twist.bar_cocycle_digest)

    def test_valid_supplied_table_digest_is_bound_and_a_mismatch_is_rejected(self) -> None:
        """Catch a Task-5 table certificate being accepted without exact replay."""

        table = _d4_table()
        grade = GF2Character((0,) * 8)
        rho = GF2Character(_d4_character(1, 0))
        explicit = self.api.u1_local_skeleton(table, grade, rho)
        self.assertEqual(explicit.table_dependency_digest, table.table_digest)

        with self.assertRaisesRegex(ValueError, "table_digest"):
            FiniteGroupTable(
                group_id=table.group_id,
                element_order=table.element_order,
                identity_index=table.identity_index,
                multiplication_table=table.multiplication_table,
                inverse_indices=table.inverse_indices,
                table_digest="sha256:" + "0" * 64,
            )

    def test_nonhomomorphic_grade_and_rho_name_the_offending_table_row(self) -> None:
        """Catch invalid coefficient data being rejected without a replayable witness."""

        table = _d4_table()
        zero = GF2Character((0,) * 8)
        invalid = GF2Character((0, 0, 1, 0, 0, 0, 0, 0))
        for label, grade, rho in (
            ("grade", invalid, zero),
            ("rho", zero, invalid),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{label}.*multiplication_table\[1\]\[1\]",
                ):
                    self.api.u1_local_skeleton(table, grade, rho)

    def test_character_lengths_and_malformed_group_tables_are_rejected(self) -> None:
        """Catch truncated characters and non-group normalized-bar inputs."""

        table = _d4_table()
        zero = GF2Character((0,) * 8)
        with self.assertRaisesRegex(ValueError, "grade.*length"):
            self.api.u1_local_skeleton(table, GF2Character((0,) * 7), zero)
        with self.assertRaisesRegex(ValueError, "rho.*length"):
            self.api.u1_local_skeleton(table, zero, GF2Character((0,) * 7))

        class StructuralForgery:
            group_id = table.group_id
            element_order = table.element_order
            identity_index = table.identity_index
            multiplication_table = table.multiplication_table
            inverse_indices = table.inverse_indices
            table_digest = table.table_digest

        with self.assertRaisesRegex(TypeError, "FiniteGroupTable"):
            self.api.u1_local_skeleton(StructuralForgery(), zero, zero)


if __name__ == "__main__":
    unittest.main()
