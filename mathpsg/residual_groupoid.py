"""Certified finite residual affine groupoids.

Residual arrows act either on the finite quotient coordinates emitted by the
Z2 classifier or on the raw relative coordinates of a compact-U1 torsor.  In
the latter case every arrow carries and replays equation and boundary
transfer matrices, so a self-consistent affine-map digest cannot hide a map
that leaves the certified torsor.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import re
from typing import Literal, Sequence, TypeAlias

from .certificates import (
    BarCoordinateTrace,
    BarEvaluatorCertificate,
    ContinuousOrbitPresentation,
    FiniteOrbitMembershipCertificate,
    FiniteOrbitRepresentative,
    UnframedQuotientCertificate,
    _digest as _certificate_digest,
    make_finite_orbit_membership,
    make_finite_orbit_path,
    make_finite_orbit_representative,
)
from .cochains import FiniteGroupTable
from .gf2 import GF2AffineArrow, MatrixGF2
from .integer_linalg import MatrixZ, identity_matrix, matmul, zero_matrix
from .torus import Phase
from .u1_classifier import TorsorStratum, symbolic_torsor_point
from .u1_local import U1LocalSkeleton, verify_u1_local_skeleton
from .z2_classifier import FiniteAffineStratum, _replay_finite_affine_stratum


NonemptyStratum: TypeAlias = FiniteAffineStratum | TorsorStratum

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROTOCOL = b"mathpsg-residual-groupoid-v1|"
_LOCAL_CONJUGACY_CONSTRUCTION_SEAL = object()
_GLOBAL_WEYL_CONSTRUCTION_SEAL = object()
_RESIDUAL_GROUPOID_CONSTRUCTION_SEAL = object()


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


def _phase_text(value: Phase) -> str:
    return str(value)


def _matrix_mapping(value: MatrixZ | MatrixGF2) -> dict[str, object]:
    return {
        "column_count": value.column_count,
        "ring": "z" if type(value) is MatrixZ else "gf2",
        "rows": [list(row) for row in value],
    }


def _negative_identity(size: int) -> MatrixZ:
    return MatrixZ(
        tuple(
            tuple(-int(row == column) for column in range(size))
            for row in range(size)
        ),
        column_count=size,
    )


def _phase_matvec(matrix: MatrixZ, vector: Sequence[Phase]) -> tuple[Phase, ...]:
    values = tuple(vector)
    if len(values) != matrix.column_count or any(type(value) is not Phase for value in values):
        raise ValueError("phase vector dimension differs from matrix")
    return tuple(
        Phase(
            sum(
                (
                    matrix[row][column] * values[column].value
                    for column in range(matrix.column_count)
                ),
                Fraction(0),
            )
        )
        for row in range(matrix.row_count)
    )


def _phase_add(left: Sequence[Phase], right: Sequence[Phase]) -> tuple[Phase, ...]:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("phase vector dimensions differ")
    return tuple(
        Phase(a.value + b.value)
        for a, b in zip(left_values, right_values, strict=True)
    )


def _phase_subtract(left: Sequence[Phase], right: Sequence[Phase]) -> tuple[Phase, ...]:
    return _phase_add(left, tuple(Phase(-value.value) for value in right))


def _integer_inverse(matrix: MatrixZ) -> MatrixZ:
    if matrix.row_count != matrix.column_count:
        raise ValueError("integral affine inverse requires a square linear part")
    size = matrix.row_count
    augmented = [
        [Fraction(value) for value in matrix[row]]
        + [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError("integral affine linear part is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if coefficient:
                augmented[row] = [
                    left - coefficient * right
                    for left, right in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    inverse_rows = tuple(
        tuple(value for value in augmented[row][size:]) for row in range(size)
    )
    if any(value.denominator != 1 for row in inverse_rows for value in row):
        raise ValueError("integral affine linear part is not unimodular")
    return MatrixZ(
        tuple(tuple(value.numerator for value in row) for row in inverse_rows),
        column_count=size,
    )


@dataclass(frozen=True, slots=True)
class IntegralAffineArrow:
    arrow_id: str
    source_stratum_id: str
    target_stratum_id: str
    linear: MatrixZ
    shift: tuple[Phase, ...]

    def __post_init__(self) -> None:
        _require_digest(self.arrow_id, "$IntegralAffineArrow.arrow_id")
        _require_digest(
            self.source_stratum_id, "$IntegralAffineArrow.source_stratum_id"
        )
        _require_digest(
            self.target_stratum_id, "$IntegralAffineArrow.target_stratum_id"
        )
        linear = MatrixZ(self.linear)
        shift = tuple(self.shift)
        if len(shift) != linear.row_count or any(type(value) is not Phase for value in shift):
            raise ValueError("$IntegralAffineArrow.shift: target dimension differs")
        core = {
            "linear": _matrix_mapping(linear),
            "shift": [_phase_text(value) for value in shift],
            "source_stratum_id": self.source_stratum_id,
            "target_stratum_id": self.target_stratum_id,
        }
        if self.arrow_id != _digest("integral-affine-arrow", core):
            raise ValueError("$IntegralAffineArrow.arrow_id: payload digest differs")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "shift", shift)

    @property
    def source_dimension(self) -> int:
        return self.linear.column_count

    @property
    def target_dimension(self) -> int:
        return self.linear.row_count

    def apply(self, point: Sequence[Phase]) -> tuple[Phase, ...]:
        return _phase_add(_phase_matvec(self.linear, tuple(point)), self.shift)

    def compose(self, other: "IntegralAffineArrow") -> "IntegralAffineArrow":
        """Return ``self`` after ``other``."""

        if type(other) is not IntegralAffineArrow:
            raise TypeError("integral affine composition requires another arrow")
        if other.target_stratum_id != self.source_stratum_id:
            raise ValueError("integral affine arrows are not composable")
        linear = matmul(self.linear, other.linear)
        shift = _phase_add(_phase_matvec(self.linear, other.shift), self.shift)
        return make_integral_affine_arrow(
            source_stratum_id=other.source_stratum_id,
            target_stratum_id=self.target_stratum_id,
            linear=linear,
            shift=shift,
        )

    def inverse(self) -> "IntegralAffineArrow":
        inverse_linear = _integer_inverse(self.linear)
        inverse_shift = tuple(
            Phase(-value.value)
            for value in _phase_matvec(inverse_linear, self.shift)
        )
        return make_integral_affine_arrow(
            source_stratum_id=self.target_stratum_id,
            target_stratum_id=self.source_stratum_id,
            linear=inverse_linear,
            shift=inverse_shift,
        )


def make_integral_affine_arrow(
    *,
    source_stratum_id: str,
    target_stratum_id: str,
    linear: MatrixZ,
    shift: Sequence[Phase],
) -> IntegralAffineArrow:
    matrix = MatrixZ(linear)
    phases = tuple(shift)
    core = {
        "linear": _matrix_mapping(matrix),
        "shift": [_phase_text(value) for value in phases],
        "source_stratum_id": source_stratum_id,
        "target_stratum_id": target_stratum_id,
    }
    return IntegralAffineArrow(
        _digest("integral-affine-arrow", core),
        source_stratum_id,
        target_stratum_id,
        matrix,
        phases,
    )


@dataclass(frozen=True, slots=True)
class WeylOrbitData:
    instance_id: str
    skeleton: U1LocalSkeleton
    evaluator: BarEvaluatorCertificate

    def __post_init__(self) -> None:
        if type(self.instance_id) is not str or not self.instance_id:
            raise ValueError("$WeylOrbitData.instance_id: invalid identifier")
        if type(self.skeleton) is not U1LocalSkeleton:
            raise TypeError("$WeylOrbitData.skeleton: expected U1LocalSkeleton")
        if type(self.evaluator) is not BarEvaluatorCertificate:
            raise TypeError("$WeylOrbitData.evaluator: invalid evaluator")
        verify_u1_local_skeleton(self.skeleton, self.evaluator.finite_group)
        if self.skeleton.element_order != self.evaluator.finite_group.element_order:
            raise ValueError("Weyl skeleton and evaluator element orders differ")


def _replay_weyl_orbit_data(value: WeylOrbitData) -> WeylOrbitData:
    if type(value) is not WeylOrbitData:
        raise TypeError("global Weyl binding must be WeylOrbitData")
    if type(value.skeleton) is not U1LocalSkeleton:
        raise TypeError("global Weyl binding has an invalid local skeleton")
    if type(value.evaluator) is not BarEvaluatorCertificate:
        raise TypeError("global Weyl binding has an invalid bar evaluator")
    skeleton = U1LocalSkeleton(
        **{
            name: getattr(value.skeleton, name)
            for name in value.skeleton.__dataclass_fields__
        }
    )
    stored_table = value.evaluator.finite_group
    table = FiniteGroupTable(
        stored_table.group_id,
        stored_table.element_order,
        stored_table.identity_index,
        stored_table.multiplication_table,
        stored_table.inverse_indices,
        stored_table.table_digest,
    )
    traces = tuple(
        BarCoordinateTrace(
            trace.trace_id,
            trace.resolution_id,
            trace.degree,
            trace.group_tuple,
            trace.coordinate_weights,
        )
        for trace in value.evaluator.traces
    )
    evaluator = BarEvaluatorCertificate(
        value.evaluator.evaluator_id,
        value.evaluator.resolution_id,
        table,
        value.evaluator.coordinate_dimensions,
        traces,
        value.evaluator.equivalence,
        value.evaluator.authority,
        value.evaluator.diagnostic,
    )
    replayed = WeylOrbitData(value.instance_id, skeleton, evaluator)
    if replayed != value:
        raise ValueError("global Weyl binding differs after canonical replay")
    return replayed


@dataclass(frozen=True, slots=True)
class LocalConjugacy:
    conjugacy_id: str
    source_stratum_id: str
    target_stratum_id: str
    kind: Literal["identity", "local", "global_weyl", "inverse", "composite"]
    gf2_arrow: GF2AffineArrow | None
    integral_arrow: IntegralAffineArrow | None
    equation_transfer: MatrixZ | None
    boundary_transfer: MatrixZ | None
    witness_digest: str
    orbit_instance_ids: tuple[str, ...]
    acted_instance_ids: tuple[str, ...]
    diagnostic: bool
    _construction_seal: InitVar[object | None] = None
    _global_weyl_construction_seal: InitVar[object | None] = None

    def __post_init__(
        self,
        _construction_seal: object | None,
        _global_weyl_construction_seal: object | None,
    ) -> None:
        _require_digest(self.conjugacy_id, "$LocalConjugacy.conjugacy_id")
        _require_digest(self.source_stratum_id, "$LocalConjugacy.source_stratum_id")
        _require_digest(self.target_stratum_id, "$LocalConjugacy.target_stratum_id")
        _require_digest(self.witness_digest, "$LocalConjugacy.witness_digest")
        if self.kind not in ("identity", "local", "global_weyl", "inverse", "composite"):
            raise ValueError("$LocalConjugacy.kind: invalid kind")
        if _construction_seal is not _LOCAL_CONJUGACY_CONSTRUCTION_SEAL:
            raise ValueError("LocalConjugacy construction is reserved to verified factories")
        if self.kind == "global_weyl":
            if _global_weyl_construction_seal is not _GLOBAL_WEYL_CONSTRUCTION_SEAL:
                raise ValueError("global Weyl construction is reserved to the common factory")
        elif _global_weyl_construction_seal is not None:
            raise ValueError("global Weyl construction seal used for another arrow kind")
        if (self.gf2_arrow is None) == (self.integral_arrow is None):
            raise ValueError("$LocalConjugacy: expected exactly one affine arrow")
        if self.integral_arrow is not None and (
            self.integral_arrow.source_stratum_id != self.source_stratum_id
            or self.integral_arrow.target_stratum_id != self.target_stratum_id
        ):
            raise ValueError("integral arrow source/target differs from conjugacy")
        equation = None if self.equation_transfer is None else MatrixZ(self.equation_transfer)
        boundary = None if self.boundary_transfer is None else MatrixZ(self.boundary_transfer)
        instances = tuple(self.orbit_instance_ids)
        acted = tuple(self.acted_instance_ids)
        if instances != tuple(sorted(set(instances))) or acted != tuple(sorted(set(acted))):
            raise ValueError("residual orbit-instance IDs must be canonical")
        if any(value not in instances for value in acted):
            raise ValueError("acted residual instance is outside its orbit tuple")
        if self.kind == "global_weyl" and acted != instances:
            raise ValueError("global Weyl must act simultaneously on the diagonal IGG")
        if type(self.diagnostic) is not bool:
            raise TypeError("$LocalConjugacy.diagnostic: expected boolean")
        core = _conjugacy_core(
            self.source_stratum_id,
            self.target_stratum_id,
            self.kind,
            self.gf2_arrow,
            self.integral_arrow,
            equation,
            boundary,
            self.witness_digest,
            instances,
            acted,
            self.diagnostic,
        )
        if self.conjugacy_id != _digest("local-conjugacy", core):
            raise ValueError("$LocalConjugacy.conjugacy_id: payload digest differs")
        object.__setattr__(self, "equation_transfer", equation)
        object.__setattr__(self, "boundary_transfer", boundary)
        object.__setattr__(self, "orbit_instance_ids", instances)
        object.__setattr__(self, "acted_instance_ids", acted)

    @property
    def ring(self) -> Literal["gf2", "torus"]:
        return "gf2" if self.gf2_arrow is not None else "torus"


def _gf2_arrow_mapping(value: GF2AffineArrow) -> dict[str, object]:
    return {
        "linear": _matrix_mapping(value.linear),
        "shift": list(value.shift),
    }


def _integral_arrow_mapping(value: IntegralAffineArrow) -> dict[str, object]:
    return {
        "arrow_id": value.arrow_id,
        "linear": _matrix_mapping(value.linear),
        "shift": [_phase_text(item) for item in value.shift],
    }


def _conjugacy_core(
    source: str,
    target: str,
    kind: str,
    gf2_arrow: GF2AffineArrow | None,
    integral_arrow: IntegralAffineArrow | None,
    equation_transfer: MatrixZ | None,
    boundary_transfer: MatrixZ | None,
    witness_digest: str,
    orbit_instance_ids: Sequence[str],
    acted_instance_ids: Sequence[str],
    diagnostic: bool,
) -> dict[str, object]:
    return {
        "acted_instance_ids": list(acted_instance_ids),
        "affine_arrow": (
            _gf2_arrow_mapping(gf2_arrow)
            if gf2_arrow is not None
            else _integral_arrow_mapping(integral_arrow)  # type: ignore[arg-type]
        ),
        "boundary_transfer": (
            None if boundary_transfer is None else _matrix_mapping(boundary_transfer)
        ),
        "diagnostic": diagnostic,
        "equation_transfer": (
            None if equation_transfer is None else _matrix_mapping(equation_transfer)
        ),
        "kind": kind,
        "orbit_instance_ids": list(orbit_instance_ids),
        "source_stratum_id": source,
        "target_stratum_id": target,
        "witness_digest": witness_digest,
    }


def _matvec_gf2(matrix: MatrixGF2, vector: Sequence[int]) -> tuple[int, ...]:
    point = tuple(vector)
    if len(point) != matrix.column_count:
        raise ValueError("GF2 vector dimension differs")
    return tuple(
        sum(entry * value for entry, value in zip(row, point, strict=True)) & 1
        for row in matrix
    )


def _matmul_gf2(left: MatrixGF2, right: MatrixGF2) -> MatrixGF2:
    if left.column_count != right.row_count:
        raise ValueError("GF2 matrix dimensions differ")
    return MatrixGF2(
        tuple(
            tuple(
                sum(
                    left[row][middle] * right[middle][column]
                    for middle in range(left.column_count)
                )
                & 1
                for column in range(right.column_count)
            )
            for row in range(left.row_count)
        ),
        column_count=right.column_count,
    )


def _gf2_identity(size: int) -> MatrixGF2:
    return MatrixGF2(
        tuple(
            tuple(int(row == column) for column in range(size))
            for row in range(size)
        ),
        column_count=size,
    )


def _gf2_inverse(matrix: MatrixGF2) -> MatrixGF2:
    if matrix.row_count != matrix.column_count:
        raise ValueError("GF2 affine inverse requires a square linear part")
    size = matrix.row_count
    rows = [
        list(matrix[row]) + [int(row == column) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot is None:
            raise ValueError("GF2 affine linear part is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(size):
            if row != column and rows[row][column]:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[column], strict=True)]
    return MatrixGF2(
        tuple(tuple(row[size:]) for row in rows),
        column_count=size,
    )


def _compose_gf2(left: GF2AffineArrow, right: GF2AffineArrow) -> GF2AffineArrow:
    linear = _matmul_gf2(left.linear, right.linear)
    shifted = _matvec_gf2(left.linear, right.shift)
    return GF2AffineArrow(
        linear,
        tuple(a ^ b for a, b in zip(shifted, left.shift, strict=True)),
    )


def _inverse_gf2(value: GF2AffineArrow) -> GF2AffineArrow:
    linear = _gf2_inverse(value.linear)
    return GF2AffineArrow(linear, _matvec_gf2(linear, value.shift))


def _verify_torus_conjugacy(
    source: TorsorStratum,
    target: TorsorStratum,
    arrow: IntegralAffineArrow,
    equation_transfer: MatrixZ,
    boundary_transfer: MatrixZ,
) -> None:
    if arrow.source_stratum_id != source.stratum_id or arrow.target_stratum_id != target.stratum_id:
        raise ValueError("torus affine arrow names different strata")
    if arrow.linear.shape != (
        target.matrices.D.column_count,
        source.matrices.D.column_count,
    ):
        raise ValueError("torus affine arrow has wrong source/target dimensions")
    if equation_transfer.shape != (
        target.matrices.D.row_count,
        source.matrices.D.row_count,
    ):
        raise ValueError("equation transfer has wrong dimension")
    if boundary_transfer.shape != (
        target.matrices.B.column_count,
        source.matrices.B.column_count,
    ):
        raise ValueError("boundary transfer has wrong dimension")
    if matmul(target.matrices.D, arrow.linear) != matmul(
        equation_transfer, source.matrices.D
    ):
        raise ValueError("affine map does not preserve the relative equations")
    left_offset = _phase_matvec(target.matrices.D, arrow.shift)
    transferred = _phase_matvec(equation_transfer, source.matrices.offset)
    required = _phase_subtract(target.matrices.offset, transferred)
    if left_offset != required:
        raise ValueError("affine shift does not preserve the target torsor equation")
    if matmul(arrow.linear, source.matrices.B) != matmul(
        target.matrices.B, boundary_transfer
    ):
        raise ValueError("affine map does not preserve relative boundaries")
    if target.matrices.D.row_count and _phase_matvec(
        target.matrices.D, arrow.apply(source.basepoint)
    ) != target.matrices.offset:
        raise ValueError("affine map sends the certified basepoint outside target torsor")


def make_local_conjugacy(
    source: NonemptyStratum,
    target: NonemptyStratum,
    arrow: GF2AffineArrow | IntegralAffineArrow,
    *,
    equation_transfer: MatrixZ | None = None,
    boundary_transfer: MatrixZ | None = None,
    kind: Literal["identity", "local", "global_weyl", "inverse", "composite"] = "local",
    witness_digest: str,
    orbit_instance_ids: Sequence[str] = (),
    acted_instance_ids: Sequence[str] = (),
    diagnostic: bool = False,
    _global_weyl_construction_seal: object | None = None,
) -> LocalConjugacy:
    if not isinstance(source, (FiniteAffineStratum, TorsorStratum)) or not isinstance(
        target, (FiniteAffineStratum, TorsorStratum)
    ):
        raise TypeError("residual conjugacy requires certified nonempty strata")
    _require_digest(witness_digest, "witness_digest")
    if (
        kind == "global_weyl"
        and _global_weyl_construction_seal is not _GLOBAL_WEYL_CONSTRUCTION_SEAL
    ):
        raise ValueError("global Weyl kind is reserved to make_global_weyl_conjugacy")
    instances = tuple(sorted(tuple(orbit_instance_ids)))
    acted = tuple(sorted(tuple(acted_instance_ids)))
    gf2_arrow = None
    integral_arrow = None
    equation = None
    boundary = None
    if type(source) is FiniteAffineStratum and type(target) is FiniteAffineStratum:
        if type(arrow) is not GF2AffineArrow:
            raise TypeError("finite affine strata require a GF2 affine arrow")
        if (
            arrow.source_dimension != source.quotient_dimension
            or arrow.target_dimension != target.quotient_dimension
        ):
            raise ValueError("GF2 residual arrow dimension differs from its strata")
        if equation_transfer is not None or boundary_transfer is not None:
            raise ValueError("GF2 quotient arrows do not accept raw transfer matrices")
        gf2_arrow = arrow
    elif type(source) is TorsorStratum and type(target) is TorsorStratum:
        if type(arrow) is not IntegralAffineArrow:
            raise TypeError("compact torsors require an integral affine arrow")
        if equation_transfer is None or boundary_transfer is None:
            raise ValueError("compact torsor arrows require equation and boundary transfers")
        equation = MatrixZ(equation_transfer)
        boundary = MatrixZ(boundary_transfer)
        _verify_torus_conjugacy(source, target, arrow, equation, boundary)
        integral_arrow = arrow
    else:
        raise TypeError("residual arrows cannot change coefficient rings")
    core = _conjugacy_core(
        source.stratum_id,
        target.stratum_id,
        kind,
        gf2_arrow,
        integral_arrow,
        equation,
        boundary,
        witness_digest,
        instances,
        acted,
        diagnostic,
    )
    return LocalConjugacy(
        _digest("local-conjugacy", core),
        source.stratum_id,
        target.stratum_id,
        kind,
        gf2_arrow,
        integral_arrow,
        equation,
        boundary,
        witness_digest,
        instances,
        acted,
        diagnostic,
        _LOCAL_CONJUGACY_CONSTRUCTION_SEAL,
        _global_weyl_construction_seal,
    )


def _replay_nonempty_stratum(value: NonemptyStratum) -> NonemptyStratum:
    """Reconstruct every typed stratum so frozen-object mutation cannot pass."""

    if type(value) is FiniteAffineStratum:
        replayed: NonemptyStratum = _replay_finite_affine_stratum(value)
    elif type(value) is TorsorStratum:
        replayed = TorsorStratum(
            value.stratum_id,
            value.rho_bits,
            value.skeleton_ids,
            value.matrices,
            value.basepoint,
            value.homogeneous_group,
            value.primal_chart,
            value.free_parameters,
            value.certificate,
        )
    else:
        raise TypeError("residual groupoid requires an exact typed stratum")
    if replayed != value:
        raise ValueError("residual stratum differs after canonical replay")
    return replayed


def _replay_local_conjugacy(
    value: LocalConjugacy,
    strata: dict[str, NonemptyStratum],
) -> LocalConjugacy:
    """Rehash one supplied generator against replayed source/target strata."""

    if type(value) is not LocalConjugacy:
        raise TypeError("residual generator must be LocalConjugacy")
    try:
        source = strata[value.source_stratum_id]
        target = strata[value.target_stratum_id]
    except KeyError as error:
        raise ValueError("local residual arrow names an unknown stratum") from error
    if value.gf2_arrow is not None:
        arrow: GF2AffineArrow | IntegralAffineArrow = GF2AffineArrow(
            MatrixGF2(
                value.gf2_arrow.linear.rows,
                column_count=value.gf2_arrow.linear.column_count,
            ),
            value.gf2_arrow.shift,
        )
        equation = None
        boundary = None
    else:
        if value.integral_arrow is None:
            raise ValueError("residual conjugacy has no affine payload")
        arrow = make_integral_affine_arrow(
            source_stratum_id=value.integral_arrow.source_stratum_id,
            target_stratum_id=value.integral_arrow.target_stratum_id,
            linear=MatrixZ(
                value.integral_arrow.linear.rows,
                column_count=value.integral_arrow.linear.column_count,
            ),
            shift=tuple(Phase(item.value) for item in value.integral_arrow.shift),
        )
        if arrow != value.integral_arrow:
            raise ValueError("integral residual arrow differs after canonical replay")
        equation = (
            None
            if value.equation_transfer is None
            else MatrixZ(
                value.equation_transfer.rows,
                column_count=value.equation_transfer.column_count,
            )
        )
        boundary = (
            None
            if value.boundary_transfer is None
            else MatrixZ(
                value.boundary_transfer.rows,
                column_count=value.boundary_transfer.column_count,
            )
        )
    replayed = make_local_conjugacy(
        source,
        target,
        arrow,
        equation_transfer=equation,
        boundary_transfer=boundary,
        kind=value.kind,
        witness_digest=value.witness_digest,
        orbit_instance_ids=value.orbit_instance_ids,
        acted_instance_ids=value.acted_instance_ids,
        diagnostic=value.diagnostic,
        _global_weyl_construction_seal=(
            _GLOBAL_WEYL_CONSTRUCTION_SEAL
            if value.kind == "global_weyl"
            else None
        ),
    )
    if replayed != value:
        raise ValueError(
            "residual conjugacy orbit tuple or payload differs after canonical replay"
        )
    return replayed


def _rebuild_local_conjugacy_from_canonical_evidence(
    *,
    conjugacy_id: str,
    source: NonemptyStratum,
    target: NonemptyStratum,
    kind: Literal["identity", "local", "global_weyl", "inverse", "composite"],
    gf2_arrow: GF2AffineArrow | None,
    integral_arrow: IntegralAffineArrow | None,
    equation_transfer: MatrixZ | None,
    boundary_transfer: MatrixZ | None,
    witness_digest: str,
    orbit_instance_ids: Sequence[str],
    acted_instance_ids: Sequence[str],
    diagnostic: bool,
) -> LocalConjugacy:
    """Rebuild one byte-decoded arrow through the ordinary verified factory."""

    if (gf2_arrow is None) == (integral_arrow is None):
        raise ValueError("canonical residual evidence has no unique affine payload")
    replayed = make_local_conjugacy(
        source,
        target,
        gf2_arrow if gf2_arrow is not None else integral_arrow,  # type: ignore[arg-type]
        equation_transfer=equation_transfer,
        boundary_transfer=boundary_transfer,
        kind=kind,
        witness_digest=witness_digest,
        orbit_instance_ids=orbit_instance_ids,
        acted_instance_ids=acted_instance_ids,
        diagnostic=diagnostic,
        _global_weyl_construction_seal=(
            _GLOBAL_WEYL_CONSTRUCTION_SEAL if kind == "global_weyl" else None
        ),
    )
    if replayed.conjugacy_id != conjugacy_id:
        raise ValueError("canonical residual evidence changed its conjugacy ID")
    return replayed


def _solve_rational_system(
    rows: Sequence[Sequence[int]], right: Sequence[Fraction], columns: int
) -> tuple[Fraction, ...]:
    matrix = [
        [Fraction(value) for value in row] + [Fraction(target)]
        for row, target in zip(rows, right, strict=True)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            coefficient = matrix[row][column]
            if coefficient:
                matrix[row] = [
                    left - coefficient * value
                    for left, value in zip(matrix[row], matrix[pivot_row], strict=True)
                ]
        pivot_columns.append(column)
        pivot_row += 1
    if any(not any(row[:columns]) and row[columns] for row in matrix):
        raise ValueError("Weyl comparison cochain is outside evaluator coordinates")
    solution = [Fraction(0)] * columns
    for row, column in enumerate(pivot_columns):
        solution[column] = matrix[row][columns]
    if any(
        sum(Fraction(value) * solution[column] for column, value in enumerate(row))
        != target
        for row, target in zip(rows, right, strict=True)
    ):
        raise ValueError("Weyl coordinate solution does not replay")
    return tuple(solution)


def _weyl_coordinates(binding: WeylOrbitData) -> tuple[Phase, ...]:
    table = binding.evaluator.finite_group
    identity = table.element_order[table.identity_index]
    rows: list[tuple[int, ...]] = []
    values: list[Fraction] = []
    index = {element: position for position, element in enumerate(table.element_order)}
    for element in table.element_order:
        if element == identity:
            continue
        weights = binding.evaluator.coordinate_weights(
            (element,),
            binding.skeleton.rho_values,
        )
        assert weights is not None
        rows.append(weights)
        values.append(Fraction(binding.skeleton.grade_values[index[element]], 2))
    coordinates = _solve_rational_system(
        rows,
        values,
        binding.evaluator.coordinate_dimensions[1],
    )
    return tuple(Phase(value) for value in coordinates)


def make_global_weyl_conjugacy(
    stratum: TorsorStratum,
    orbit_data: Sequence[WeylOrbitData],
    *,
    acted_instance_ids: Sequence[str] | None = None,
) -> LocalConjugacy:
    if type(stratum) is not TorsorStratum:
        raise TypeError("global Weyl requires a compact-U1 torsor")
    replayed_stratum = _replay_nonempty_stratum(stratum)
    if type(replayed_stratum) is not TorsorStratum:
        raise TypeError("global Weyl replay did not return a compact-U1 torsor")
    stratum = replayed_stratum
    bindings = tuple(_replay_weyl_orbit_data(item) for item in orbit_data)
    instances = stratum.matrices.coordinate_blocks.instance_ids
    if (
        not bindings
        or tuple(binding.instance_id for binding in bindings) != instances
        or len(set(instances)) != len(instances)
    ):
        raise ValueError("global Weyl data must cover the canonical orbit tuple")
    if tuple(binding.skeleton.skeleton_id for binding in bindings) != stratum.skeleton_ids:
        raise ValueError("global Weyl skeleton tuple differs from its stratum")
    acted = instances if acted_instance_ids is None else tuple(acted_instance_ids)
    if tuple(sorted(acted)) != instances:
        raise ValueError(
            "global Weyl must be one simultaneous action on the diagonal IGG"
        )
    dimension = stratum.matrices.D.column_count
    shift = [Phase(Fraction(0)) for _ in range(dimension)]
    for index, binding in enumerate(bindings):
        start, stop = stratum.matrices.coordinate_blocks.local_slices[1][index]
        coordinates = _weyl_coordinates(binding)
        if len(coordinates) != stop - start:
            raise ValueError("Weyl comparison coordinates differ from relative local block")
        shift[start:stop] = coordinates
    linear = _negative_identity(dimension)
    arrow = make_integral_affine_arrow(
        source_stratum_id=stratum.stratum_id,
        target_stratum_id=stratum.stratum_id,
        linear=linear,
        shift=tuple(shift),
    )
    witness = _digest(
        "global-weyl-coordinate-witness",
        {
            "evaluator_ids": [binding.evaluator.evaluator_id for binding in bindings],
            "instance_ids": list(instances),
            "skeleton_ids": [binding.skeleton.skeleton_id for binding in bindings],
            "shift": [_phase_text(value) for value in shift],
        },
    )
    return make_local_conjugacy(
        stratum,
        stratum,
        arrow,
        equation_transfer=_negative_identity(stratum.matrices.D.row_count),
        boundary_transfer=_negative_identity(stratum.matrices.B.column_count),
        kind="global_weyl",
        witness_digest=witness,
        orbit_instance_ids=instances,
        acted_instance_ids=instances,
        diagnostic=any(binding.evaluator.diagnostic for binding in bindings),
        _global_weyl_construction_seal=_GLOBAL_WEYL_CONSTRUCTION_SEAL,
    )


def _arrow_key(value: LocalConjugacy) -> tuple[object, ...]:
    if value.gf2_arrow is not None:
        return (
            value.source_stratum_id,
            value.target_stratum_id,
            "gf2",
            value.gf2_arrow.linear.rows,
            value.gf2_arrow.linear.column_count,
            value.gf2_arrow.shift,
        )
    assert value.integral_arrow is not None
    return (
        value.source_stratum_id,
        value.target_stratum_id,
        "torus",
        value.integral_arrow.linear.rows,
        value.integral_arrow.linear.column_count,
        tuple(phase.value for phase in value.integral_arrow.shift),
    )


def _identity_conjugacy(stratum: NonemptyStratum) -> LocalConjugacy:
    witness = _digest("residual-identity-witness", {"stratum_id": stratum.stratum_id})
    if type(stratum) is FiniteAffineStratum:
        arrow: GF2AffineArrow | IntegralAffineArrow = GF2AffineArrow(
            _gf2_identity(stratum.quotient_dimension),
            (0,) * stratum.quotient_dimension,
        )
        return make_local_conjugacy(
            stratum,
            stratum,
            arrow,
            kind="identity",
            witness_digest=witness,
        )
    dimension = stratum.matrices.D.column_count
    arrow = make_integral_affine_arrow(
        source_stratum_id=stratum.stratum_id,
        target_stratum_id=stratum.stratum_id,
        linear=identity_matrix(dimension),
        shift=tuple(Phase(Fraction(0)) for _ in range(dimension)),
    )
    return make_local_conjugacy(
        stratum,
        stratum,
        arrow,
        equation_transfer=identity_matrix(stratum.matrices.D.row_count),
        boundary_transfer=identity_matrix(stratum.matrices.B.column_count),
        kind="identity",
        witness_digest=witness,
    )


def _inverse_conjugacy(
    value: LocalConjugacy,
    strata: dict[str, NonemptyStratum],
) -> LocalConjugacy:
    source = strata[value.target_stratum_id]
    target = strata[value.source_stratum_id]
    witness = _digest("residual-inverse-witness", {"arrow_id": value.conjugacy_id})
    if value.gf2_arrow is not None:
        return make_local_conjugacy(
            source,
            target,
            _inverse_gf2(value.gf2_arrow),
            kind="inverse",
            witness_digest=witness,
            orbit_instance_ids=value.orbit_instance_ids,
            acted_instance_ids=value.acted_instance_ids,
            diagnostic=value.diagnostic,
        )
    assert value.integral_arrow is not None
    assert value.equation_transfer is not None and value.boundary_transfer is not None
    return make_local_conjugacy(
        source,
        target,
        value.integral_arrow.inverse(),
        equation_transfer=_integer_inverse(value.equation_transfer),
        boundary_transfer=_integer_inverse(value.boundary_transfer),
        kind="inverse",
        witness_digest=witness,
        orbit_instance_ids=value.orbit_instance_ids,
        acted_instance_ids=value.acted_instance_ids,
        diagnostic=value.diagnostic,
    )


def _compose_conjugacy(
    left: LocalConjugacy,
    right: LocalConjugacy,
    strata: dict[str, NonemptyStratum],
) -> LocalConjugacy:
    if right.target_stratum_id != left.source_stratum_id:
        raise ValueError("residual arrows are not composable")
    source = strata[right.source_stratum_id]
    target = strata[left.target_stratum_id]
    witness = _digest(
        "residual-composition-witness",
        {"left": left.conjugacy_id, "right": right.conjugacy_id},
    )
    if left.gf2_arrow is not None and right.gf2_arrow is not None:
        return make_local_conjugacy(
            source,
            target,
            _compose_gf2(left.gf2_arrow, right.gf2_arrow),
            kind="composite",
            witness_digest=witness,
            diagnostic=left.diagnostic or right.diagnostic,
        )
    if left.integral_arrow is None or right.integral_arrow is None:
        raise TypeError("residual composition changes coefficient rings")
    assert left.equation_transfer is not None and right.equation_transfer is not None
    assert left.boundary_transfer is not None and right.boundary_transfer is not None
    return make_local_conjugacy(
        source,
        target,
        left.integral_arrow.compose(right.integral_arrow),
        equation_transfer=matmul(left.equation_transfer, right.equation_transfer),
        boundary_transfer=matmul(left.boundary_transfer, right.boundary_transfer),
        kind="composite",
        witness_digest=witness,
        diagnostic=left.diagnostic or right.diagnostic,
    )


def _is_identity_affine_payload(value: LocalConjugacy) -> bool:
    """Replay an asserted identity without trusting its kind or witness ID."""

    if value.gf2_arrow is not None:
        dimension = value.gf2_arrow.source_dimension
        return (
            value.gf2_arrow.target_dimension == dimension
            and value.gf2_arrow.linear == _gf2_identity(dimension)
            and value.gf2_arrow.shift == (0,) * dimension
            and value.equation_transfer is None
            and value.boundary_transfer is None
        )
    if value.integral_arrow is None:
        return False
    dimension = value.integral_arrow.source_dimension
    if (
        value.integral_arrow.target_dimension != dimension
        or value.integral_arrow.linear != identity_matrix(dimension)
        or any(phase.value for phase in value.integral_arrow.shift)
        or value.equation_transfer is None
        or value.boundary_transfer is None
    ):
        return False
    return (
        value.equation_transfer
        == identity_matrix(value.equation_transfer.row_count)
        and value.boundary_transfer
        == identity_matrix(value.boundary_transfer.row_count)
    )


def _composition_payload_replays(
    left: LocalConjugacy,
    right: LocalConjugacy,
    result: LocalConjugacy,
) -> bool:
    """Compare the actual affine/transfer maps behind a table entry."""

    if (
        right.target_stratum_id != left.source_stratum_id
        or result.source_stratum_id != right.source_stratum_id
        or result.target_stratum_id != left.target_stratum_id
    ):
        return False
    if left.gf2_arrow is not None and right.gf2_arrow is not None:
        return (
            result.gf2_arrow == _compose_gf2(left.gf2_arrow, right.gf2_arrow)
            and result.integral_arrow is None
        )
    if (
        left.integral_arrow is None
        or right.integral_arrow is None
        or result.integral_arrow is None
        or left.equation_transfer is None
        or right.equation_transfer is None
        or result.equation_transfer is None
        or left.boundary_transfer is None
        or right.boundary_transfer is None
        or result.boundary_transfer is None
    ):
        return False
    expected_arrow = left.integral_arrow.compose(right.integral_arrow)
    return (
        result.integral_arrow.linear == expected_arrow.linear
        and result.integral_arrow.shift == expected_arrow.shift
        and result.equation_transfer
        == matmul(left.equation_transfer, right.equation_transfer)
        and result.boundary_transfer
        == matmul(left.boundary_transfer, right.boundary_transfer)
    )


@dataclass(frozen=True, slots=True)
class ResidualGroupoid:
    groupoid_digest: str
    object_ids: tuple[str, ...]
    arrows: tuple[LocalConjugacy, ...]
    identity_arrow_ids: tuple[tuple[str, str], ...]
    inverse_pairs: tuple[tuple[str, str], ...]
    composition_table: tuple[tuple[str, str, str], ...]
    generator_ids: tuple[str, ...]
    _construction_seal: InitVar[object | None] = None

    def __post_init__(self, _construction_seal: object | None) -> None:
        _require_digest(self.groupoid_digest, "$ResidualGroupoid.groupoid_digest")
        objects = tuple(self.object_ids)
        arrows = tuple(self.arrows)
        identities = tuple(tuple(item) for item in self.identity_arrow_ids)
        inverses = tuple(tuple(item) for item in self.inverse_pairs)
        compositions = tuple(tuple(item) for item in self.composition_table)
        generators = tuple(self.generator_ids)
        if objects != tuple(sorted(set(objects))):
            raise ValueError("residual groupoid objects must be canonical")
        if any(type(arrow) is not LocalConjugacy for arrow in arrows):
            raise TypeError("residual groupoid arrows are invalid")
        if arrows != tuple(sorted(arrows, key=lambda arrow: arrow.conjugacy_id)):
            raise ValueError("residual groupoid arrows must use canonical ID order")
        by_id = {arrow.conjugacy_id: arrow for arrow in arrows}
        if (
            len(by_id) != len(arrows)
            or generators != tuple(sorted(set(generators)))
            or set(generators) - set(by_id)
        ):
            raise ValueError("residual groupoid arrow/generator IDs differ")
        if set((arrow.source_stratum_id for arrow in arrows)) | set(
            arrow.target_stratum_id for arrow in arrows
        ) != set(objects):
            raise ValueError("residual groupoid arrows do not cover its objects")
        if identities != tuple(sorted(set(identities))):
            raise ValueError("residual groupoid identity certificates are not canonical")
        identity_map = dict(identities)
        if set(identity_map) != set(objects) or any(
            arrow_id not in by_id for arrow_id in identity_map.values()
        ):
            raise ValueError("residual groupoid identity certificates are incomplete")
        for object_id, arrow_id in identities:
            arrow = by_id[arrow_id]
            if (
                arrow.kind != "identity"
                or arrow.source_stratum_id != object_id
                or arrow.target_stratum_id != object_id
                or not _is_identity_affine_payload(arrow)
            ):
                raise ValueError("residual groupoid identity does not replay algebraically")
        if inverses != tuple(sorted(set(inverses))):
            raise ValueError("residual groupoid inverse certificates are not canonical")
        inverse_map = dict(inverses)
        if set(inverse_map) != set(by_id) or any(
            inverse not in by_id or inverse_map.get(inverse) != arrow
            for arrow, inverse in inverse_map.items()
        ):
            raise ValueError("residual groupoid inverse certificates are incomplete")
        composable = {
            (left.conjugacy_id, right.conjugacy_id)
            for left in arrows
            for right in arrows
            if right.target_stratum_id == left.source_stratum_id
        }
        if compositions != tuple(sorted(set(compositions))):
            raise ValueError("residual groupoid composition certificates are not canonical")
        composition_map = {(left, right): result for left, right, result in compositions}
        if set(composition_map) != composable or any(
            result not in by_id for result in composition_map.values()
        ):
            raise ValueError("residual groupoid composition closure is incomplete")
        for (left_id, right_id), result_id in composition_map.items():
            if not _composition_payload_replays(
                by_id[left_id],
                by_id[right_id],
                by_id[result_id],
            ):
                raise ValueError("residual groupoid composition does not replay algebraically")
        for arrow_id, inverse_id in inverse_map.items():
            arrow = by_id[arrow_id]
            inverse = by_id[inverse_id]
            if (
                inverse.source_stratum_id != arrow.target_stratum_id
                or inverse.target_stratum_id != arrow.source_stratum_id
                or composition_map[(inverse_id, arrow_id)]
                != identity_map[arrow.source_stratum_id]
                or composition_map[(arrow_id, inverse_id)]
                != identity_map[arrow.target_stratum_id]
            ):
                raise ValueError("residual groupoid inverse does not replay algebraically")
        core = {
            "arrow_ids": [arrow.conjugacy_id for arrow in arrows],
            "composition_table": [list(item) for item in compositions],
            "generator_ids": list(generators),
            "identity_arrow_ids": [list(item) for item in identities],
            "inverse_pairs": [list(item) for item in inverses],
            "object_ids": list(objects),
        }
        if self.groupoid_digest != _digest("residual-groupoid", core):
            raise ValueError("$ResidualGroupoid.groupoid_digest: payload differs")
        if _construction_seal is not _RESIDUAL_GROUPOID_CONSTRUCTION_SEAL:
            raise ValueError("ResidualGroupoid construction is reserved to the verified builder")
        object.__setattr__(self, "object_ids", objects)
        object.__setattr__(self, "arrows", arrows)
        object.__setattr__(self, "identity_arrow_ids", identities)
        object.__setattr__(self, "inverse_pairs", inverses)
        object.__setattr__(self, "composition_table", compositions)
        object.__setattr__(self, "generator_ids", generators)


def build_residual_groupoid(
    strata: Sequence[NonemptyStratum],
    local_arrows: Sequence[LocalConjugacy],
) -> ResidualGroupoid:
    values = tuple(_replay_nonempty_stratum(value) for value in strata)
    by_stratum = {value.stratum_id: value for value in values}
    if len(by_stratum) != len(values):
        raise ValueError("residual groupoid contains duplicate stratum IDs")
    supplied_generators = tuple(
        _replay_local_conjugacy(value, by_stratum) for value in local_arrows
    )
    action_keys = tuple(_arrow_key(value) for value in supplied_generators)
    if len(set(action_keys)) != len(action_keys):
        raise ValueError("conflicting semantic duplicate residual generators")
    generators = tuple(
        sorted(supplied_generators, key=lambda value: value.conjugacy_id)
    )
    if any(
        value.source_stratum_id not in by_stratum
        or value.target_stratum_id not in by_stratum
        for value in generators
    ):
        raise ValueError("local residual arrow names an unknown stratum")
    for arrow in generators:
        source = by_stratum[arrow.source_stratum_id]
        target = by_stratum[arrow.target_stratum_id]
        if type(source) is FiniteAffineStratum and type(target) is FiniteAffineStratum:
            if (
                arrow.gf2_arrow is None
                or arrow.gf2_arrow.source_dimension != source.quotient_dimension
                or arrow.gf2_arrow.target_dimension != target.quotient_dimension
            ):
                raise ValueError("GF2 residual arrow differs from classified object dimensions")
        elif type(source) is TorsorStratum and type(target) is TorsorStratum:
            if (
                arrow.integral_arrow is None
                or arrow.equation_transfer is None
                or arrow.boundary_transfer is None
            ):
                raise ValueError("torus residual arrow lacks constraint-transfer payload")
            _verify_torus_conjugacy(
                source,
                target,
                arrow.integral_arrow,
                arrow.equation_transfer,
                arrow.boundary_transfer,
            )
        else:
            raise ValueError("residual arrow changes the classified coefficient ring")
    for stratum in values:
        if type(stratum) is not TorsorStratum:
            continue
        common_weyl = tuple(
            arrow
            for arrow in generators
            if arrow.kind == "global_weyl"
            and arrow.source_stratum_id == stratum.stratum_id
            and arrow.target_stratum_id == stratum.stratum_id
        )
        if len(common_weyl) != 1:
            raise ValueError(
                "each U1 stratum requires exactly one common global Weyl generator"
            )
        actual_instances = stratum.matrices.coordinate_blocks.instance_ids
        if (
            common_weyl[0].orbit_instance_ids != actual_instances
            or common_weyl[0].acted_instance_ids != actual_instances
        ):
            raise ValueError(
                "global Weyl orbit tuple differs from coordinate-block instances"
            )

    arrows_by_key: dict[tuple[object, ...], LocalConjugacy] = {}
    for stratum in values:
        identity = _identity_conjugacy(stratum)
        arrows_by_key[_arrow_key(identity)] = identity
    generator_ids: list[str] = []
    for generator in generators:
        key = _arrow_key(generator)
        if key not in arrows_by_key:
            arrows_by_key[key] = generator
        canonical = arrows_by_key[key]
        identity_id = _identity_conjugacy(by_stratum[canonical.source_stratum_id]).conjugacy_id
        if canonical.conjugacy_id != identity_id:
            generator_ids.append(canonical.conjugacy_id)

    changed = True
    while changed:
        changed = False
        current = tuple(arrows_by_key.values())
        for arrow in current:
            inverse = _inverse_conjugacy(arrow, by_stratum)
            key = _arrow_key(inverse)
            if key not in arrows_by_key:
                arrows_by_key[key] = inverse
                changed = True
        current = tuple(arrows_by_key.values())
        for left in current:
            for right in current:
                if right.target_stratum_id != left.source_stratum_id:
                    continue
                composed = _compose_conjugacy(left, right, by_stratum)
                key = _arrow_key(composed)
                if key not in arrows_by_key:
                    arrows_by_key[key] = composed
                    changed = True
                    if len(arrows_by_key) > 4096:
                        raise ValueError("residual affine generators do not close finitely")

    arrows = tuple(sorted(arrows_by_key.values(), key=lambda value: value.conjugacy_id))
    key_to_id = {_arrow_key(value): value.conjugacy_id for value in arrows}
    identities = tuple(
        sorted(
            (
                stratum.stratum_id,
                key_to_id[_arrow_key(_identity_conjugacy(stratum))],
            )
            for stratum in values
        )
    )
    inverses = tuple(
        sorted(
            (
                arrow.conjugacy_id,
                key_to_id[_arrow_key(_inverse_conjugacy(arrow, by_stratum))],
            )
            for arrow in arrows
        )
    )
    compositions = tuple(
        sorted(
            (
                left.conjugacy_id,
                right.conjugacy_id,
                key_to_id[
                    _arrow_key(_compose_conjugacy(left, right, by_stratum))
                ],
            )
            for left in arrows
            for right in arrows
            if right.target_stratum_id == left.source_stratum_id
        )
    )
    canonical_generators = tuple(sorted(set(generator_ids)))
    objects = tuple(sorted(by_stratum))
    core = {
        "arrow_ids": [arrow.conjugacy_id for arrow in arrows],
        "composition_table": [list(item) for item in compositions],
        "generator_ids": list(canonical_generators),
        "identity_arrow_ids": [list(item) for item in identities],
        "inverse_pairs": [list(item) for item in inverses],
        "object_ids": list(objects),
    }
    return ResidualGroupoid(
        _digest("residual-groupoid", core),
        objects,
        arrows,
        identities,
        inverses,
        compositions,
        canonical_generators,
        _RESIDUAL_GROUPOID_CONSTRUCTION_SEAL,
    )


def _finite_points(stratum: NonemptyStratum) -> tuple[tuple[object, ...], ...] | None:
    if type(stratum) is FiniteAffineStratum:
        return tuple(itertools.product((0, 1), repeat=stratum.quotient_dimension))
    if stratum.homogeneous_group.free_rank:
        return None
    return tuple(
        tuple(coordinates)
        for coordinates in itertools.product(
            *(range(order) for order in stratum.homogeneous_group.torsion_orders)
        )
    )


def _finite_arrow_apply(
    arrow: LocalConjugacy,
    source: NonemptyStratum,
    target: NonemptyStratum,
    coordinates: tuple[object, ...],
) -> tuple[object, ...]:
    if arrow.gf2_arrow is not None:
        return tuple(arrow.gf2_arrow.apply(tuple(int(value) for value in coordinates)))
    assert arrow.integral_arrow is not None
    if type(source) is not TorsorStratum or type(target) is not TorsorStratum:
        raise TypeError("integral residual arrow does not join compact torsors")
    point = symbolic_torsor_point(source, (), tuple(int(value) for value in coordinates))
    raw = arrow.integral_arrow.apply(point.constant)
    target_coordinates = target.coordinates(raw)
    if target_coordinates.free:
        raise ValueError("finite compact stratum unexpectedly acquired free coordinates")
    return tuple(target_coordinates.torsion)


def certify_unframed_quotient(
    strata: Sequence[NonemptyStratum],
    groupoid: ResidualGroupoid,
) -> UnframedQuotientCertificate:
    values = tuple(strata)
    if type(groupoid) is not ResidualGroupoid:
        raise TypeError("unframed quotient requires a ResidualGroupoid")
    by_id = {value.stratum_id: value for value in values}
    if tuple(sorted(by_id)) != groupoid.object_ids or len(by_id) != len(values):
        raise ValueError("unframed quotient must preserve the complete framed stratum list")
    finite = {identifier: _finite_points(value) for identifier, value in by_id.items()}
    presentations: list[ContinuousOrbitPresentation] = []
    representatives: tuple[FiniteOrbitRepresentative, ...]
    memberships: tuple[FiniteOrbitMembershipCertificate, ...]
    framed_count: int | None
    unframed_count: int | None
    if any(points is None for points in finite.values()):
        framed_count = None
        unframed_count = None
        # Connected components of stratum objects are finite even when their
        # point sets are continuous.  Store the finite groupoid presentation,
        # never a sampled point count.
        remaining = set(by_id)
        while remaining:
            seed = min(remaining)
            component = {seed}
            changed = True
            while changed:
                changed = False
                for arrow in groupoid.arrows:
                    if arrow.source_stratum_id in component or arrow.target_stratum_id in component:
                        before = len(component)
                        component.add(arrow.source_stratum_id)
                        component.add(arrow.target_stratum_id)
                        changed |= len(component) != before
            remaining -= component
            representative = min(component)
            component_strata = tuple(by_id[item] for item in sorted(component))
            free_rank = max(
                item.homogeneous_group.free_rank
                for item in component_strata
                if type(item) is TorsorStratum
            )
            torsion = tuple(
                item.homogeneous_group.torsion_orders
                for item in component_strata
                if type(item) is TorsorStratum
            )
            component_arrows = tuple(
                arrow.conjugacy_id
                for arrow in groupoid.arrows
                if arrow.source_stratum_id in component
                and arrow.target_stratum_id in component
            )
            presentations.append(
                ContinuousOrbitPresentation(
                    representative,
                    tuple(sorted(component)),
                    component_arrows,
                    free_rank,
                    torsion,
                    sum(
                        groupoid_arrow.kind == "global_weyl"
                        for groupoid_arrow in groupoid.arrows
                        if groupoid_arrow.conjugacy_id in groupoid.generator_ids
                        and groupoid_arrow.source_stratum_id in component
                    ),
                )
            )
        representatives = ()
        memberships = ()
    else:
        all_points = {
            (identifier, tuple(point))
            for identifier, points in finite.items()
            for point in points or ()
        }
        framed_count = len(all_points)
        remaining = set(all_points)
        orbit_representatives: list[FiniteOrbitRepresentative] = []
        orbit_memberships: list[FiniteOrbitMembershipCertificate] = []
        while remaining:
            seed = min(remaining)
            orbit = {seed}
            frontier = [seed]
            while frontier:
                source_id, point = frontier.pop()
                for arrow in groupoid.arrows:
                    if arrow.source_stratum_id != source_id:
                        continue
                    target_point = _finite_arrow_apply(
                        arrow,
                        by_id[source_id],
                        by_id[arrow.target_stratum_id],
                        point,
                    )
                    target = (arrow.target_stratum_id, target_point)
                    if target not in all_points:
                        raise ValueError("residual arrow leaves the framed finite point set")
                    if target not in orbit:
                        orbit.add(target)
                        frontier.append(target)
            remaining -= orbit
            representative = min(orbit)
            typed_representative = make_finite_orbit_representative(
                representative[0],
                tuple(int(item) for item in representative[1]),
            )
            typed_members = tuple(
                make_finite_orbit_representative(
                    stratum_id,
                    tuple(int(item) for item in coordinates),
                )
                for stratum_id, coordinates in sorted(orbit)
            )
            paths_by_state: dict[
                tuple[str, tuple[object, ...]], tuple[str, ...]
            ] = {representative: ()}
            path_frontier = [representative]
            while path_frontier:
                source_id, point = path_frontier.pop(0)
                source_path = paths_by_state[(source_id, point)]
                for arrow in groupoid.arrows:
                    if arrow.source_stratum_id != source_id:
                        continue
                    target_point = _finite_arrow_apply(
                        arrow,
                        by_id[source_id],
                        by_id[arrow.target_stratum_id],
                        point,
                    )
                    target = (arrow.target_stratum_id, target_point)
                    if target in orbit and target not in paths_by_state:
                        paths_by_state[target] = source_path + (arrow.conjugacy_id,)
                        path_frontier.append(target)
            if set(paths_by_state) != orbit:
                raise ValueError("finite orbit lacks a residual-arrow path certificate")
            paths = tuple(
                make_finite_orbit_path(
                    typed_representative,
                    member,
                    paths_by_state[(member.stratum_id, member.coordinates)],
                )
                for member in typed_members
            )
            component_stratum_ids = {
                member.stratum_id for member in typed_members
            }
            orbit_representatives.append(typed_representative)
            orbit_memberships.append(
                make_finite_orbit_membership(
                    groupoid.groupoid_digest,
                    typed_representative,
                    typed_members,
                    paths,
                    tuple(
                        arrow.conjugacy_id
                        for arrow in groupoid.arrows
                        if arrow.source_stratum_id
                        in component_stratum_ids
                        and arrow.target_stratum_id
                        in component_stratum_ids
                    ),
                )
            )
        representatives = tuple(orbit_representatives)
        memberships = tuple(orbit_memberships)
        unframed_count = len(representatives)
    core = {
        "continuous_orbit_presentations": [
            presentation.mapping() for presentation in presentations
        ],
        "framed_finite_cardinality": framed_count,
        "finite_orbit_memberships": [
            membership.mapping() for membership in memberships
        ],
        "framed_stratum_ids": sorted(by_id),
        "groupoid_digest": groupoid.groupoid_digest,
        "orbit_representatives": [
            representative.mapping() for representative in representatives
        ],
        "unframed_finite_cardinality": unframed_count,
    }
    certificate_id = _certificate_digest("unframed-quotient-certificate", core)
    return UnframedQuotientCertificate(
        certificate_id,
        tuple(sorted(by_id)),
        groupoid.groupoid_digest,
        representatives,
        framed_count,
        unframed_count,
        tuple(presentations),
        memberships,
    )


__all__ = [
    "IntegralAffineArrow",
    "LocalConjugacy",
    "NonemptyStratum",
    "ResidualGroupoid",
    "WeylOrbitData",
    "build_residual_groupoid",
    "certify_unframed_quotient",
    "make_global_weyl_conjugacy",
    "make_integral_affine_arrow",
    "make_local_conjugacy",
]
