"""Certified compact-:math:`U(1)` relative classifier.

The low-level solver in this module never accepts a matrix without also
receiving, and replaying, the :class:`~psgmath.relative_complex.RelativeProblem`
from which it was assembled.  The public sector classifier adds the local
normalized-bar coordinate authority below; keeping these layers distinct
prevents a self-consistent matrix rehash from becoming a classification
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import re
from typing import Iterator, Sequence
from weakref import WeakKeyDictionary

from .classification_schema import FrozenJSONArray, FrozenJSONObject, ObstructedBranch
from .bar_evaluator import (
    BarResolutionEquivalence,
    CochainCoordinateCertificate,
    coordinate_bar_cocycle,
    bar_equivalence_mapping,
    verify_bar_resolution_equivalence,
    verify_cochain_coordinate_certificate,
)
from .cochains import (
    CertifiedCochainProblem,
    CharacterBasisCertificate,
    FreeResolutionCertificate,
    InclusionChainMapCertificate,
    LauncherExecutionAttestation,
    Task5VerificationAuthority,
    VerificationIssue,
    VerificationReport,
    character_certificate_digest,
    free_resolution_mapping,
    inclusion_chain_map_mapping,
    twist_inclusion_cochain_map,
    verify_character_basis,
)
from .gf2 import GF2Character
from .integer_linalg import (
    IntegerKernel,
    MatrixZ,
    SmithForm,
    integer_kernel,
    matmul,
    smith_form,
    transpose,
    zero_matrix,
)
from .relative_complex import (
    RelativeMatrices,
    RelativeProblem,
    assemble_relative_problem,
    verify_relative_certificate,
)
from .torus import (
    CompactGroupPresentation,
    Phase,
    PrimalTorsorChart,
    TorusObstruction,
    TorusSolution,
    TorusSolvabilityWitness,
    solve_torus_quotient,
    torsor_coordinates,
)
from .u1_local import U1LocalSkeleton, verify_u1_local_skeleton


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_PARAMETER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_PROTOCOL = b"mathpsg-u1-classifier-v1|"


class ContinuousStratumError(TypeError):
    """Raised when a positive-dimensional torsor is treated as finite."""


_U1_SECTOR_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class LocalU1Data:
    """One occupied-orbit input before normalized-bar coordination."""

    instance_id: str
    inclusion: InclusionChainMapCertificate
    bar_equivalence: BarResolutionEquivalence
    skeleton: U1LocalSkeleton

    def __post_init__(self) -> None:
        if (
            type(self.instance_id) is not str
            or _IDENTIFIER_RE.fullmatch(self.instance_id) is None
        ):
            raise ValueError("$LocalU1Data.instance_id: invalid identifier")
        if type(self.inclusion) is not InclusionChainMapCertificate:
            raise TypeError("$LocalU1Data.inclusion: expected InclusionChainMapCertificate")
        if type(self.bar_equivalence) is not BarResolutionEquivalence:
            raise TypeError("$LocalU1Data.bar_equivalence: expected BarResolutionEquivalence")
        if type(self.skeleton) is not U1LocalSkeleton:
            raise TypeError("$LocalU1Data.skeleton: expected U1LocalSkeleton")


@dataclass(frozen=True, slots=True)
class U1DefectCoordinateBinding:
    """An exact local normalized-bar-to-resolution coordinate round trip."""

    instance_id: str
    skeleton: U1LocalSkeleton
    bar_equivalence: BarResolutionEquivalence
    coordinate_certificate: CochainCoordinateCertificate
    relative_defect_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.instance_id) is not str
            or _IDENTIFIER_RE.fullmatch(self.instance_id) is None
        ):
            raise ValueError(
                "$U1DefectCoordinateBinding.instance_id: invalid identifier"
            )
        if type(self.skeleton) is not U1LocalSkeleton:
            raise TypeError(
                "$U1DefectCoordinateBinding.skeleton: expected U1LocalSkeleton"
            )
        if type(self.bar_equivalence) is not BarResolutionEquivalence:
            raise TypeError(
                "$U1DefectCoordinateBinding.bar_equivalence: "
                "expected BarResolutionEquivalence"
            )
        if type(self.coordinate_certificate) is not CochainCoordinateCertificate:
            raise TypeError(
                "$U1DefectCoordinateBinding.coordinate_certificate: "
                "expected CochainCoordinateCertificate"
            )
        _require_digest(
            self.relative_defect_digest,
            "$U1DefectCoordinateBinding.relative_defect_digest",
        )


@dataclass(frozen=True, eq=False)
class _U1SectorProblemSnapshot:
    sector_id: str
    source_snapshot_digest: str
    source: CertifiedCochainProblem
    rho: GF2Character
    grade: GF2Character
    authority: Task5VerificationAuthority
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    local_data: tuple[LocalU1Data, ...]
    bindings: tuple[U1DefectCoordinateBinding, ...]
    relative_problem: RelativeProblem
    matrices: RelativeMatrices
    diagnostic_only: bool
    trusted_release_attestations: tuple[LauncherExecutionAttestation, ...]
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _U1_SECTOR_FACTORY_TOKEN:
            raise TypeError(
                "U1 sector verification snapshot requires the Task-8 factory"
            )
        _require_digest(self.sector_id, "$_U1SectorProblemSnapshot.sector_id")
        _require_digest(
            self.source_snapshot_digest,
            "$_U1SectorProblemSnapshot.source_snapshot_digest",
        )


@dataclass(frozen=True, slots=True)
class _U1SectorFactoryRecord:
    sector_id: str
    source_snapshot_digest: str
    source: CertifiedCochainProblem
    rho: GF2Character
    grade: GF2Character
    authority: Task5VerificationAuthority
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    local_data: tuple[LocalU1Data, ...]
    bindings: tuple[U1DefectCoordinateBinding, ...]
    relative_problem: RelativeProblem
    matrices: RelativeMatrices
    diagnostic_only: bool
    trusted_release_attestations: tuple[LauncherExecutionAttestation, ...]


_U1_SECTOR_FACTORY_RECORDS: WeakKeyDictionary[
    _U1SectorProblemSnapshot,
    _U1SectorFactoryRecord,
] = WeakKeyDictionary()


def _factory_snapshot_record(
    snapshot: _U1SectorProblemSnapshot,
) -> _U1SectorFactoryRecord | None:
    record = _U1_SECTOR_FACTORY_RECORDS.get(snapshot)
    if record is None:
        return None
    if (
        snapshot._factory_token is not _U1_SECTOR_FACTORY_TOKEN
        or snapshot.sector_id != record.sector_id
        or snapshot.source_snapshot_digest != record.source_snapshot_digest
        or snapshot.source is not record.source
        or snapshot.rho is not record.rho
        or snapshot.grade is not record.grade
        or snapshot.authority is not record.authority
        or snapshot.spatial_character_basis is not record.spatial_character_basis
        or snapshot.spatial_resolution is not record.spatial_resolution
        or snapshot.local_data is not record.local_data
        or snapshot.bindings is not record.bindings
        or snapshot.relative_problem is not record.relative_problem
        or snapshot.matrices is not record.matrices
        or snapshot.diagnostic_only != record.diagnostic_only
        or snapshot.trusted_release_attestations
        is not record.trusted_release_attestations
    ):
        return None
    return record


@dataclass(frozen=True, slots=True)
class U1SectorProblem:
    """Nonserializable authority envelope for one joint compact-U1 sector."""

    sector_id: str
    source_snapshot_digest: str
    source: CertifiedCochainProblem
    rho: GF2Character
    grade: GF2Character
    authority: Task5VerificationAuthority
    spatial_character_basis: CharacterBasisCertificate | None
    spatial_resolution: FreeResolutionCertificate | None
    local_data: tuple[LocalU1Data, ...]
    bindings: tuple[U1DefectCoordinateBinding, ...]
    relative_problem: RelativeProblem
    matrices: RelativeMatrices
    diagnostic_h2_invariants: tuple[int, ...]
    diagnostic_h3_invariants: tuple[int, ...]
    diagnostic_only: bool
    _verification_seal: _U1SectorProblemSnapshot = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        snapshot = self._verification_seal
        if type(snapshot) is not _U1SectorProblemSnapshot:
            raise TypeError(
                "U1SectorProblem requires a factory-certified verification snapshot"
            )
        record = _factory_snapshot_record(snapshot)
        if (
            record is None
            or self.sector_id != record.sector_id
            or self.source_snapshot_digest != record.source_snapshot_digest
            or self.source is not record.source
            or self.rho is not record.rho
            or self.grade is not record.grade
            or self.authority is not record.authority
            or self.spatial_character_basis is not record.spatial_character_basis
            or self.spatial_resolution is not record.spatial_resolution
            or self.local_data is not record.local_data
            or self.bindings is not record.bindings
            or self.relative_problem is not record.relative_problem
            or self.matrices is not record.matrices
            or self.diagnostic_only != record.diagnostic_only
        ):
            raise TypeError(
                "U1SectorProblem differs from its factory-certified verification snapshot"
            )
        _require_digest(self.sector_id, "$U1SectorProblem.sector_id")
        _require_digest(
            self.source_snapshot_digest,
            "$U1SectorProblem.source_snapshot_digest",
        )
        if type(self.source) is not CertifiedCochainProblem:
            raise TypeError("$U1SectorProblem.source: expected CertifiedCochainProblem")
        if type(self.rho) is not GF2Character or type(self.grade) is not GF2Character:
            raise TypeError("$U1SectorProblem: rho and grade must be GF2Character values")
        if type(self.authority) is not Task5VerificationAuthority:
            raise TypeError(
                "$U1SectorProblem.authority: expected external Task5VerificationAuthority"
            )
        if self.spatial_character_basis is not None and type(
            self.spatial_character_basis
        ) is not CharacterBasisCertificate:
            raise TypeError(
                "$U1SectorProblem.spatial_character_basis: invalid certificate"
            )
        if self.spatial_resolution is not None and type(
            self.spatial_resolution
        ) is not FreeResolutionCertificate:
            raise TypeError(
                "$U1SectorProblem.spatial_resolution: invalid resolution"
            )
        local_data = tuple(self.local_data)
        bindings = tuple(self.bindings)
        if not local_data or any(type(item) is not LocalU1Data for item in local_data):
            raise ValueError("$U1SectorProblem.local_data: expected occupied-orbit tuple")
        if any(type(item) is not U1DefectCoordinateBinding for item in bindings):
            raise TypeError("$U1SectorProblem.bindings: invalid binding tuple")
        data_ids = tuple(item.instance_id for item in local_data)
        binding_ids = tuple(item.instance_id for item in bindings)
        if (
            len(set(data_ids)) != len(data_ids)
            or binding_ids != data_ids
        ):
            raise ValueError(
                "$U1SectorProblem: local inputs/bindings require one ordered instance tuple"
            )
        if type(self.relative_problem) is not RelativeProblem or type(
            self.matrices
        ) is not RelativeMatrices:
            raise TypeError("$U1SectorProblem: invalid relative source or matrices")
        h2 = _diagnostic_invariants(
            self.diagnostic_h2_invariants,
            "$U1SectorProblem.diagnostic_h2_invariants",
        )
        h3 = _diagnostic_invariants(
            self.diagnostic_h3_invariants,
            "$U1SectorProblem.diagnostic_h3_invariants",
        )
        if type(self.diagnostic_only) is not bool:
            raise TypeError("$U1SectorProblem.diagnostic_only: expected boolean")
        object.__setattr__(self, "local_data", local_data)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "diagnostic_h2_invariants", h2)
        object.__setattr__(self, "diagnostic_h3_invariants", h3)

    def with_diagnostics(
        self,
        *,
        h2: Sequence[int],
        h3: Sequence[int],
    ) -> "U1SectorProblem":
        """Replace diagnostic summaries without changing algebraic authority."""

        return U1SectorProblem(
            self.sector_id,
            self.source_snapshot_digest,
            self.source,
            self.rho,
            self.grade,
            self.authority,
            self.spatial_character_basis,
            self.spatial_resolution,
            self.local_data,
            self.bindings,
            self.relative_problem,
            self.matrices,
            tuple(h2),
            tuple(h3),
            self.diagnostic_only,
            self._verification_seal,
        )


def _diagnostic_invariants(value: object, path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected integer tuple")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{path}: expected nonnegative diagnostic integers")
    return result


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
    fraction = value.value
    return (
        str(fraction.numerator)
        if fraction.denominator == 1
        else f"{fraction.numerator}/{fraction.denominator}"
    )


def _matrix_mapping(value: MatrixZ) -> dict[str, object]:
    return {"column_count": value.column_count, "rows": [list(row) for row in value]}


def _smith_mapping(value: SmithForm) -> dict[str, object]:
    return {
        "diagonal": _matrix_mapping(value.diagonal),
        "invariant_factors": list(value.invariant_factors),
        "left": _matrix_mapping(value.left),
        "right": _matrix_mapping(value.right),
    }


def _kernel_mapping(value: IntegerKernel) -> dict[str, object]:
    return {
        "basis": _matrix_mapping(value.basis),
        "completion": _matrix_mapping(value.completion),
        "completion_inverse": _matrix_mapping(value.completion_inverse),
        "coordinate_projection": _matrix_mapping(value.coordinate_projection),
        "rank": value.rank,
        "source": _matrix_mapping(value.source),
    }


def _solvability_mapping(value: TorusSolvabilityWitness) -> dict[str, object]:
    return {
        "smith": _smith_mapping(value.smith),
        "smith_coordinates": [_phase_text(item) for item in value.smith_coordinates],
        "transformed_offset": [_phase_text(item) for item in value.transformed_offset],
        "zero_row_characters": [list(row) for row in value.zero_row_characters],
    }


def _group_mapping(value: CompactGroupPresentation) -> dict[str, object]:
    return {
        "dual_generators": _matrix_mapping(value.dual_generators),
        "dual_relations": _matrix_mapping(value.dual_relations),
        "free_rank": value.free_rank,
        "torsion_orders": list(value.torsion_orders),
    }


def _chart_mapping(value: PrimalTorsorChart) -> dict[str, object]:
    return {
        "free_character_pairing": _matrix_mapping(value.free_character_pairing),
        "free_lifts": _matrix_mapping(value.free_lifts),
        "quotient_witnesses": [_matrix_mapping(item) for item in value.quotient_witnesses],
        "raw_dimension": value.raw_dimension,
        "torsion_lifts": [
            [_phase_text(item) for item in row] for row in value.torsion_lifts
        ],
        "torsion_pairing": [
            [_phase_text(item) for item in row] for row in value.torsion_pairing
        ],
    }


def _revalidated_kernel(value: IntegerKernel) -> IntegerKernel:
    if type(value) is not IntegerKernel:
        raise TypeError("$U1StratumCertificate.dual_kernel: expected IntegerKernel")
    return IntegerKernel(
        value.source,
        value.basis,
        value.completion,
        value.completion_inverse,
        value.coordinate_projection,
        value.rank,
    )


def _revalidated_solvability(value: TorusSolvabilityWitness) -> TorusSolvabilityWitness:
    if type(value) is not TorusSolvabilityWitness:
        raise TypeError(
            "$U1StratumCertificate.solvability_witness: expected TorusSolvabilityWitness"
        )
    smith = SmithForm(
        value.smith.diagonal,
        value.smith.left,
        value.smith.right,
        value.smith.invariant_factors,
    )
    return TorusSolvabilityWitness(
        smith,
        value.transformed_offset,
        value.smith_coordinates,
        value.zero_row_characters,
    )


@dataclass(frozen=True, slots=True)
class U1StratumCertificate:
    """Replay data for the dual reduction and affine basepoint."""

    certificate_id: str
    stratum_id: str
    sector_id: str
    relative_certificate_id: str
    rho_bits: tuple[int, ...]
    skeleton_ids: tuple[str, ...]
    coordinate_certificate_ids: tuple[str, ...]
    dual_kernel: IntegerKernel
    relation_coordinates: MatrixZ
    relation_smith: SmithForm
    solvability_witness: TorusSolvabilityWitness
    basepoint_digest: str
    homogeneous_group_digest: str
    primal_chart_digest: str

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "stratum_id",
            "sector_id",
            "relative_certificate_id",
            "basepoint_digest",
            "homogeneous_group_digest",
            "primal_chart_digest",
        ):
            _require_digest(getattr(self, name), f"$U1StratumCertificate.{name}")
        rho = tuple(self.rho_bits)
        if any(type(bit) is not int or bit not in (0, 1) for bit in rho):
            raise ValueError("$U1StratumCertificate.rho_bits: expected exact bits")
        skeletons = tuple(self.skeleton_ids)
        coordinates = tuple(self.coordinate_certificate_ids)
        for index, value in enumerate(skeletons):
            _require_digest(value, f"$U1StratumCertificate.skeleton_ids[{index}]")
        for index, value in enumerate(coordinates):
            _require_digest(
                value,
                f"$U1StratumCertificate.coordinate_certificate_ids[{index}]",
            )
        dual_kernel = _revalidated_kernel(self.dual_kernel)
        relations = MatrixZ(self.relation_coordinates)
        relation_smith = SmithForm(
            self.relation_smith.diagonal,
            self.relation_smith.left,
            self.relation_smith.right,
            self.relation_smith.invariant_factors,
        )
        if matmul(
            matmul(relation_smith.left, relations), relation_smith.right
        ) != relation_smith.diagonal:
            raise ValueError(
                "$U1StratumCertificate.relation_smith: witness does not replay"
            )
        solvability = _revalidated_solvability(self.solvability_witness)
        object.__setattr__(self, "rho_bits", rho)
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "coordinate_certificate_ids", coordinates)
        object.__setattr__(self, "dual_kernel", dual_kernel)
        object.__setattr__(self, "relation_coordinates", relations)
        object.__setattr__(self, "relation_smith", relation_smith)
        object.__setattr__(self, "solvability_witness", solvability)
        if self.certificate_id != _digest(
            "u1-stratum-certificate", _certificate_core(self)
        ):
            raise ValueError(
                "$U1StratumCertificate.certificate_id: does not bind certificate"
            )


def _certificate_core(value: U1StratumCertificate) -> dict[str, object]:
    return {
        "basepoint_digest": value.basepoint_digest,
        "coordinate_certificate_ids": list(value.coordinate_certificate_ids),
        "dual_kernel": _kernel_mapping(value.dual_kernel),
        "homogeneous_group_digest": value.homogeneous_group_digest,
        "primal_chart_digest": value.primal_chart_digest,
        "relation_coordinates": _matrix_mapping(value.relation_coordinates),
        "relation_smith": _smith_mapping(value.relation_smith),
        "relative_certificate_id": value.relative_certificate_id,
        "rho_bits": list(value.rho_bits),
        "sector_id": value.sector_id,
        "skeleton_ids": list(value.skeleton_ids),
        "solvability_witness": _solvability_mapping(value.solvability_witness),
        "stratum_id": value.stratum_id,
    }


@dataclass(frozen=True, slots=True)
class SymbolicPoint:
    """A formal raw relative cochain with named free parameters."""

    point_id: str
    stratum_id: str
    parameter_names: tuple[str, ...]
    torsion_coordinates: tuple[int, ...]
    constant: tuple[Phase, ...]
    free_coefficients: MatrixZ
    relative_certificate_id: str

    def __post_init__(self) -> None:
        _require_digest(self.point_id, "$SymbolicPoint.point_id")
        _require_digest(self.stratum_id, "$SymbolicPoint.stratum_id")
        _require_digest(
            self.relative_certificate_id, "$SymbolicPoint.relative_certificate_id"
        )
        names = tuple(self.parameter_names)
        if (
            any(type(name) is not str or _PARAMETER_RE.fullmatch(name) is None for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "$SymbolicPoint.parameter_names: expected unique symbolic identifiers"
            )
        torsion = tuple(self.torsion_coordinates)
        if any(type(value) is not int or value < 0 for value in torsion):
            raise ValueError("$SymbolicPoint.torsion_coordinates: expected residues")
        constant = tuple(self.constant)
        if any(type(value) is not Phase for value in constant):
            raise TypeError("$SymbolicPoint.constant: expected exact Phase values")
        coefficients = MatrixZ(self.free_coefficients)
        if coefficients.shape != (len(constant), len(names)):
            raise ValueError("$SymbolicPoint.free_coefficients: incompatible shape")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "torsion_coordinates", torsion)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "free_coefficients", coefficients)
        if self.point_id != _digest("symbolic-point", _symbolic_point_core(self)):
            raise ValueError("$SymbolicPoint.point_id: does not bind symbolic point")

    def evaluate(self, free_values: Sequence[Phase]) -> tuple[Phase, ...]:
        values = tuple(free_values)
        if len(values) != len(self.parameter_names) or any(
            type(value) is not Phase for value in values
        ):
            raise ValueError("$SymbolicPoint.evaluate: expected one exact phase per parameter")
        return tuple(
            Phase(
                self.constant[row].value
                + sum(
                    (
                        self.free_coefficients[row][column] * values[column].value
                        for column in range(len(values))
                    ),
                    Fraction(0),
                )
            )
            for row in range(len(self.constant))
        )


def _symbolic_point_core(value: SymbolicPoint) -> dict[str, object]:
    return {
        "constant": [_phase_text(item) for item in value.constant],
        "free_coefficients": _matrix_mapping(value.free_coefficients),
        "parameter_names": list(value.parameter_names),
        "relative_certificate_id": value.relative_certificate_id,
        "stratum_id": value.stratum_id,
        "torsion_coordinates": list(value.torsion_coordinates),
    }


@dataclass(frozen=True, slots=True)
class TorsorStratum:
    stratum_id: str
    rho_bits: tuple[int, ...]
    skeleton_ids: tuple[str, ...]
    matrices: RelativeMatrices
    basepoint: tuple[Phase, ...]
    homogeneous_group: CompactGroupPresentation
    primal_chart: PrimalTorsorChart
    free_parameters: tuple[str, ...]
    certificate: U1StratumCertificate

    def __post_init__(self) -> None:
        _require_digest(self.stratum_id, "$TorsorStratum.stratum_id")
        rho = tuple(self.rho_bits)
        if any(type(bit) is not int or bit not in (0, 1) for bit in rho):
            raise ValueError("$TorsorStratum.rho_bits: expected exact bits")
        skeletons = tuple(self.skeleton_ids)
        for index, value in enumerate(skeletons):
            _require_digest(value, f"$TorsorStratum.skeleton_ids[{index}]")
        if type(self.matrices) is not RelativeMatrices:
            raise TypeError("$TorsorStratum.matrices: expected RelativeMatrices")
        matrices = RelativeMatrices(
            self.matrices.B,
            self.matrices.D,
            self.matrices.E,
            self.matrices.offset,
            self.matrices.coordinate_blocks,
            self.matrices.certificate,
        )
        if not isinstance(matrices.D, MatrixZ):
            raise TypeError("$TorsorStratum.matrices: expected torus matrices")
        basepoint = tuple(self.basepoint)
        if any(type(item) is not Phase for item in basepoint):
            raise TypeError("$TorsorStratum.basepoint: expected exact phases")
        if type(self.homogeneous_group) is not CompactGroupPresentation:
            raise TypeError(
                "$TorsorStratum.homogeneous_group: expected CompactGroupPresentation"
            )
        if type(self.primal_chart) is not PrimalTorsorChart:
            raise TypeError("$TorsorStratum.primal_chart: expected PrimalTorsorChart")
        parameters = tuple(self.free_parameters)
        expected_parameters = tuple(
            f"phi{index}" for index in range(self.homogeneous_group.free_rank)
        )
        if parameters != expected_parameters:
            raise ValueError(
                "$TorsorStratum.free_parameters: expected canonical phi0, phi1, ..."
            )
        if type(self.certificate) is not U1StratumCertificate:
            raise TypeError("$TorsorStratum.certificate: expected U1StratumCertificate")
        certificate = U1StratumCertificate(
            **{
                name: getattr(self.certificate, name)
                for name in self.certificate.__dataclass_fields__
            }
        )
        object.__setattr__(self, "rho_bits", rho)
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "basepoint", basepoint)
        object.__setattr__(self, "free_parameters", parameters)
        object.__setattr__(self, "certificate", certificate)
        _replay_stratum(self)

    @property
    def continuous(self) -> bool:
        _replay_stratum(self)
        return self.homogeneous_group.free_rank > 0

    @property
    def dual_invariants(self) -> tuple[int, ...]:
        _replay_stratum(self)
        return self.homogeneous_group.serialized_invariants

    def __len__(self) -> int:
        _replay_stratum(self)
        if self.homogeneous_group.free_rank > 0:
            raise ContinuousStratumError(
                "a positive-rank compact torsor has no finite class count"
            )
        cardinality = self.homogeneous_group.finite_cardinality
        assert cardinality is not None
        return cardinality

    def __iter__(self) -> Iterator[SymbolicPoint]:
        _replay_stratum(self)
        if self.homogeneous_group.free_rank > 0:
            raise ContinuousStratumError(
                "a positive-rank compact torsor cannot be enumerated"
            )
        residues = itertools.product(
            *(range(order) for order in self.homogeneous_group.torsion_orders)
        )
        return (
            symbolic_torsor_point(self, (), tuple(coordinates))
            for coordinates in residues
        )

    def as_finite_points(self) -> tuple[SymbolicPoint, ...]:
        _replay_stratum(self)
        if self.homogeneous_group.free_rank > 0:
            raise ContinuousStratumError(
                "a positive-rank compact torsor has no finite adapter"
            )
        return tuple(iter(self))

    def coordinates(self, raw: Sequence[Phase]):
        _replay_stratum(self)
        return torsor_coordinates(_solution(self), raw)


def _solution(value: TorsorStratum) -> TorusSolution:
    return TorusSolution(
        value.basepoint,
        value.homogeneous_group,
        value.primal_chart,
        value.certificate.solvability_witness,
        value.matrices.D,
        value.matrices.B,
        value.matrices.offset,
    )


def _basepoint_digest(value: Sequence[Phase]) -> str:
    return _digest("u1-basepoint", [_phase_text(item) for item in value])


def _stratum_core(
    *,
    sector_id: str,
    relative_certificate_id: str,
    rho_bits: Sequence[int],
    skeleton_ids: Sequence[str],
    coordinate_certificate_ids: Sequence[str],
    solution: TorusSolution,
) -> dict[str, object]:
    return {
        "basepoint_digest": _basepoint_digest(solution.basepoint),
        "coordinate_certificate_ids": list(coordinate_certificate_ids),
        "homogeneous_group": _group_mapping(solution.group),
        "primal_chart": _chart_mapping(solution.primal_chart),
        "relative_certificate_id": relative_certificate_id,
        "rho_bits": list(rho_bits),
        "sector_id": sector_id,
        "skeleton_ids": list(skeleton_ids),
    }


def _replay_stratum(value: TorsorStratum) -> None:
    certificate = value.certificate
    if (
        certificate.stratum_id != value.stratum_id
        or certificate.relative_certificate_id
        != value.matrices.certificate.certificate_id
        or certificate.rho_bits != value.rho_bits
        or certificate.skeleton_ids != value.skeleton_ids
    ):
        raise ValueError("U1 stratum certificate binding mismatch")
    solution = _solution(value)
    expected_kernel = integer_kernel(transpose(value.matrices.B))
    relation_coordinates = matmul(
        expected_kernel.coordinate_projection, transpose(value.matrices.D)
    )
    if matmul(expected_kernel.basis, relation_coordinates) != transpose(
        value.matrices.D
    ):
        raise ValueError("U1 dual relation coordinates do not lift D^T")
    if certificate.dual_kernel != expected_kernel:
        raise ValueError("U1 certificate dual kernel differs from ker(B^T)")
    if certificate.relation_coordinates != relation_coordinates:
        raise ValueError("U1 certificate relation coordinates differ from im(D^T)")
    if certificate.relation_smith != smith_form(relation_coordinates):
        raise ValueError("U1 certificate relation Smith witness differs")
    if certificate.basepoint_digest != _basepoint_digest(value.basepoint):
        raise ValueError("U1 certificate basepoint digest differs")
    if certificate.homogeneous_group_digest != _digest(
        "compact-group", _group_mapping(value.homogeneous_group)
    ):
        raise ValueError("U1 certificate homogeneous-group digest differs")
    if certificate.primal_chart_digest != _digest(
        "primal-chart", _chart_mapping(value.primal_chart)
    ):
        raise ValueError("U1 certificate primal-chart digest differs")
    expected_stratum_id = _digest(
        "u1-torsor-stratum",
        _stratum_core(
            sector_id=certificate.sector_id,
            relative_certificate_id=certificate.relative_certificate_id,
            rho_bits=value.rho_bits,
            skeleton_ids=value.skeleton_ids,
            coordinate_certificate_ids=certificate.coordinate_certificate_ids,
            solution=solution,
        ),
    )
    if value.stratum_id != expected_stratum_id:
        raise ValueError("U1 stratum ID does not bind the compact torsor")


def _make_certificate(
    *,
    stratum_id: str,
    sector_id: str,
    matrices: RelativeMatrices,
    rho_bits: tuple[int, ...],
    skeleton_ids: tuple[str, ...],
    coordinate_certificate_ids: tuple[str, ...],
    solution: TorusSolution,
) -> U1StratumCertificate:
    dual_kernel = integer_kernel(transpose(matrices.B))
    relation_coordinates = matmul(
        dual_kernel.coordinate_projection, transpose(matrices.D)
    )
    if matmul(dual_kernel.basis, relation_coordinates) != transpose(matrices.D):
        raise ValueError("im(D^T) is not contained in ker(B^T)")
    relation_smith = smith_form(relation_coordinates)
    basepoint_digest = _basepoint_digest(solution.basepoint)
    homogeneous_group_digest = _digest(
        "compact-group", _group_mapping(solution.group)
    )
    primal_chart_digest = _digest(
        "primal-chart", _chart_mapping(solution.primal_chart)
    )
    core = {
        "basepoint_digest": basepoint_digest,
        "coordinate_certificate_ids": list(coordinate_certificate_ids),
        "dual_kernel": _kernel_mapping(dual_kernel),
        "homogeneous_group_digest": homogeneous_group_digest,
        "primal_chart_digest": primal_chart_digest,
        "relation_coordinates": _matrix_mapping(relation_coordinates),
        "relation_smith": _smith_mapping(relation_smith),
        "relative_certificate_id": matrices.certificate.certificate_id,
        "rho_bits": list(rho_bits),
        "sector_id": sector_id,
        "skeleton_ids": list(skeleton_ids),
        "solvability_witness": _solvability_mapping(solution.solvability_witness),
        "stratum_id": stratum_id,
    }
    return U1StratumCertificate(
        _digest("u1-stratum-certificate", core),
        stratum_id,
        sector_id,
        matrices.certificate.certificate_id,
        rho_bits,
        skeleton_ids,
        coordinate_certificate_ids,
        dual_kernel,
        relation_coordinates,
        relation_smith,
        solution.solvability_witness,
        basepoint_digest,
        homogeneous_group_digest,
        primal_chart_digest,
    )


def _obstruction_branch(
    *,
    sector_id: str,
    skeleton_ids: tuple[str, ...],
    obstruction: TorusObstruction,
) -> ObstructedBranch:
    branch_id = _digest(
        "u1-obstructed-branch",
        {
            "character": list(obstruction.character),
            "phase": _phase_text(obstruction.phase),
            "sector_id": sector_id,
            "skeleton_ids": list(skeleton_ids),
        },
    )
    return ObstructedBranch(
        branch_id,
        skeleton_ids,
        FrozenJSONObject(
            (
                ("character", FrozenJSONArray(tuple(obstruction.character))),
                ("phase", _phase_text(obstruction.phase)),
                ("smith_invariants", FrozenJSONArray(obstruction.smith.invariant_factors)),
            )
        ),
    )


def _normalized_defect_cocycle(
    skeleton: U1LocalSkeleton,
    equivalence: BarResolutionEquivalence,
) -> dict[tuple[str, ...], Phase]:
    if skeleton.element_order != equivalence.finite_group.element_order:
        raise ValueError("local skeleton and bar equivalence element orders differ")
    index = {
        element: position
        for position, element in enumerate(skeleton.element_order)
    }
    return {
        pair: skeleton.normalized_bar_defect[index[pair[0]]][index[pair[1]]]
        for pair in equivalence.normalized_tuples(2)
    }


def _authority_mapping(value: Task5VerificationAuthority) -> dict[str, object]:
    return {
        "affine_pcp_certificate_digest": value.affine_pcp_certificate_digest,
        "backend_environment_id": value.backend_environment_id,
        "backend_lock_digest": value.backend_lock_digest,
        "catalogue_action_digest": value.catalogue_action_digest,
        "catalogue_record_digest": value.catalogue_record_digest,
        "inclusion_authority": [
            {
                "diagnostic_backend": item.diagnostic_backend,
                "diagnostic_failure_degrees": list(
                    item.diagnostic_failure_degrees
                ),
                "diagnostic_outcome": item.diagnostic_outcome,
                "diagnostic_residue_digests": list(
                    item.diagnostic_residue_digests
                ),
                "gap_inclusion_projection_digest": (
                    item.gap_inclusion_projection_digest
                ),
                "inclusion_id": item.inclusion_id,
                "launcher_attestation_id": item.launcher_attestation_id,
                "literal_element_digest": item.literal_element_digest,
                "literal_stabilizer_digest": item.literal_stabilizer_digest,
                "source_bar_equivalence_id": item.source_bar_equivalence_id,
                "target_bar_equivalence_id": item.target_bar_equivalence_id,
                "transported_inclusion_digest": item.transported_inclusion_digest,
            }
            for item in value.inclusions
        ],
        "runtime_provenance_digest": value.runtime_provenance_digest,
    }


def _skeleton_snapshot(value: U1LocalSkeleton) -> dict[str, object]:
    return {
        "bar_cocycle_digest": value.bar_cocycle_digest,
        "element_order": list(value.element_order),
        "grade_values": list(value.grade_values),
        "normalized_bar_defect": [
            [_phase_text(item) for item in row]
            for row in value.normalized_bar_defect
        ],
        "q_assignment_digest": value.q_assignment_digest,
        "q_values": list(value.q_values),
        "restricted_grade_digest": value.restricted_grade_digest,
        "restricted_rho_digest": value.restricted_rho_digest,
        "rho_values": list(value.rho_values),
        "skeleton_id": value.skeleton_id,
        "table_dependency_digest": value.table_dependency_digest,
    }


def _coordinate_snapshot(
    value: CochainCoordinateCertificate,
) -> dict[str, object]:
    return {
        "certificate_id": value.certificate_id,
        "coboundary_1cochain": [str(item) for item in value.coboundary_1cochain],
        "coefficient_character": list(value.coefficient_character.bits),
        "coordinates": [str(item) for item in value.coordinates],
        "degree": value.degree,
        "mod_one": value.mod_one,
        "resolution_id": value.resolution_id,
        "source_cocycle_digest": value.source_cocycle_digest,
    }


def _complex_snapshot(value) -> dict[str, object]:
    return {
        "authority_id": value.authority_id,
        "coefficient_character": list(value.coefficient_character.bits),
        "complex_id": value.complex_id,
        "differentials": [_matrix_mapping(item) for item in value.differentials],
        "dimensions": list(value.dimensions),
    }


def _cochain_map_snapshot(value) -> dict[str, object]:
    return {
        "instance_id": value.instance_id,
        "maps": [_matrix_mapping(item) for item in value.maps],
        "source_id": value.source_id,
        "target_id": value.target_id,
    }


def _relative_snapshot(
    problem: RelativeProblem,
    matrices: RelativeMatrices,
) -> dict[str, object]:
    certificate = matrices.certificate
    return {
        "matrices": {
            "B": _matrix_mapping(matrices.B),
            "D": _matrix_mapping(matrices.D),
            "E": _matrix_mapping(matrices.E),
            "certificate": {
                "ambient_complex_id": certificate.ambient_complex_id,
                "certificate_id": certificate.certificate_id,
                "coordinate_blocks_digest": certificate.coordinate_blocks_digest,
                "db_zero_witness_digest": certificate.db_zero_witness_digest,
                "defect_digests": list(certificate.defect_digests),
                "eb_zero_witness_digest": certificate.eb_zero_witness_digest,
                "ed_zero_witness_digest": certificate.ed_zero_witness_digest,
                "instance_ids": list(certificate.instance_ids),
                "local_complex_ids": list(certificate.local_complex_ids),
                "matrix_digest": certificate.matrix_digest,
                "offset_digest": certificate.offset_digest,
                "problem_digest": certificate.problem_digest,
                "restriction_digests": list(certificate.restriction_digests),
                "ring": certificate.ring,
            },
            "coordinate_blocks": {
                "ambient_slices": [list(item) for item in matrices.coordinate_blocks.ambient_slices],
                "instance_ids": list(matrices.coordinate_blocks.instance_ids),
                "local_slices": [
                    [list(item) for item in degree]
                    for degree in matrices.coordinate_blocks.local_slices
                ],
            },
            "offset": [_phase_text(item) for item in matrices.offset],
        },
        "problem": {
            "ambient": _complex_snapshot(problem.ambient),
            "local_defects": [
                [_phase_text(item) for item in defect]
                for defect in problem.local_defects
            ],
            "locals": [_complex_snapshot(item) for item in problem.locals],
            "restrictions": [
                _cochain_map_snapshot(item) for item in problem.restrictions
            ],
            "ring": problem.ring,
        },
    }


def _sector_source_snapshot_digest(
    *,
    source: CertifiedCochainProblem,
    rho: GF2Character,
    grade: GF2Character,
    authority: Task5VerificationAuthority,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
    local_data: Sequence[LocalU1Data],
    bindings: Sequence[U1DefectCoordinateBinding],
    relative_problem: RelativeProblem,
    matrices: RelativeMatrices,
    diagnostic_only: bool,
) -> str:
    return _digest(
        "u1-sector-source-snapshot",
        {
            "ambient_resolution": free_resolution_mapping(source.ambient),
            "authority": _authority_mapping(authority),
            "character_certificate_digest": character_certificate_digest(
                source.character_basis
            ),
            "character_certificate_id": source.character_basis.certificate_id,
            "diagnostic_only": diagnostic_only,
            "grade_bits": list(grade.bits),
            "local_data": [
                {
                    "bar_equivalence": bar_equivalence_mapping(item.bar_equivalence),
                    "inclusion": inclusion_chain_map_mapping(item.inclusion),
                    "instance_id": item.instance_id,
                    "skeleton": _skeleton_snapshot(item.skeleton),
                }
                for item in local_data
            ],
            "bindings": [
                {
                    "bar_equivalence": bar_equivalence_mapping(
                        item.bar_equivalence
                    ),
                    "coordinate_certificate": _coordinate_snapshot(
                        item.coordinate_certificate
                    ),
                    "instance_id": item.instance_id,
                    "relative_defect_digest": item.relative_defect_digest,
                    "skeleton": _skeleton_snapshot(item.skeleton),
                }
                for item in bindings
            ],
            "relative": _relative_snapshot(relative_problem, matrices),
            "rho_bits": list(rho.bits),
            "source_inclusions": [
                inclusion_chain_map_mapping(item) for item in source.inclusions
            ],
            "spatial_character_certificate": (
                None
                if spatial_character_basis is None
                else {
                    "certificate_id": spatial_character_basis.certificate_id,
                    "recomputed_digest": character_certificate_digest(
                        spatial_character_basis
                    ),
                }
            ),
            "spatial_resolution_id": (
                None
                if spatial_resolution is None
                else spatial_resolution.resolution_id
            ),
        },
    )


def _sector_core(
    *,
    source: CertifiedCochainProblem,
    rho: GF2Character,
    grade: GF2Character,
    authority: Task5VerificationAuthority,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
    bindings: Sequence[U1DefectCoordinateBinding],
    matrices: RelativeMatrices,
    source_snapshot_digest: str,
    diagnostic_only: bool,
) -> dict[str, object]:
    return {
        "ambient_resolution_id": source.ambient.resolution_id,
        "authority": _authority_mapping(authority),
        "bindings": [
            {
                "bar_equivalence_id": item.bar_equivalence.equivalence_id,
                "coordinate_certificate_id": item.coordinate_certificate.certificate_id,
                "instance_id": item.instance_id,
                "relative_defect_digest": item.relative_defect_digest,
                "skeleton_id": item.skeleton.skeleton_id,
            }
            for item in bindings
        ],
        "character_certificate_id": source.character_basis.certificate_id,
        "diagnostic_only": diagnostic_only,
        "grade_bits": list(grade.bits),
        "relative_certificate_id": matrices.certificate.certificate_id,
        "rho_bits": list(rho.bits),
        "source_snapshot_digest": source_snapshot_digest,
        "spatial_character_certificate_id": (
            None
            if spatial_character_basis is None
            else spatial_character_basis.certificate_id
        ),
        "spatial_resolution_id": (
            None
            if spatial_resolution is None
            else spatial_resolution.resolution_id
        ),
    }


def _canonical_release_attestations(
    source: CertifiedCochainProblem,
    trusted_release_attestations: Sequence[LauncherExecutionAttestation],
    *,
    allow_diagnostic: bool,
) -> tuple[LauncherExecutionAttestation, ...]:
    if type(source) is not CertifiedCochainProblem:
        raise TypeError("source must be a CertifiedCochainProblem")
    try:
        supplied = tuple(trusted_release_attestations)
    except TypeError as error:
        raise TypeError(
            "trusted release attestations must be typed Task-5 launcher "
            "attestation objects"
        ) from error
    if any(type(item) is not LauncherExecutionAttestation for item in supplied):
        raise TypeError(
            "trusted release attestations must be typed Task-5 launcher "
            "attestation objects"
        )
    if allow_diagnostic:
        if supplied:
            raise ValueError(
                "diagnostic U1 sectors cannot retain release attestation authority"
            )
        return ()
    by_id = {item.attestation_id: item for item in supplied}
    if len(by_id) != len(supplied):
        raise ValueError("duplicate trusted Task-5 release attestation")
    expected_ids = tuple(
        inclusion.launcher_attestation.attestation_id
        for inclusion in source.inclusions
    )
    if (
        len(set(expected_ids)) != len(expected_ids)
        or len(supplied) != len(expected_ids)
        or set(by_id) != set(expected_ids)
    ):
        raise ValueError(
            "trusted release attestations require exact one-to-one coverage "
            "of the Task-5 inclusions"
        )
    normalized = tuple(
        by_id[inclusion.launcher_attestation.attestation_id]
        for inclusion in source.inclusions
    )
    if any(
        trusted is not inclusion.launcher_attestation
        for inclusion, trusted in zip(
            source.inclusions,
            normalized,
            strict=True,
        )
    ):
        raise ValueError(
            "each trusted release attestation must be the exact nonserialized "
            "Task-5 object associated with its inclusion"
        )
    return normalized


def _require_spatial_character_parent(
    character_basis: CharacterBasisCertificate,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
) -> None:
    graded = (
        character_basis.presentation_kind
        == "graded-direct-product-presentation"
    )
    if graded:
        if type(spatial_character_basis) is not CharacterBasisCertificate:
            raise ValueError(
                "graded U1 character basis lacks its exact spatial certificate"
            )
        if type(spatial_resolution) is not FreeResolutionCertificate:
            raise ValueError(
                "graded U1 character basis lacks its exact spatial resolution"
            )
    elif spatial_character_basis is not None or spatial_resolution is not None:
        raise ValueError(
            "spatial parent authority is reserved to graded U1 characters"
        )


def _derive_sector_material(
    source: CertifiedCochainProblem,
    rho: GF2Character,
    grade: GF2Character,
    authority: Task5VerificationAuthority,
    spatial_character_basis: CharacterBasisCertificate | None,
    spatial_resolution: FreeResolutionCertificate | None,
    local_data: Sequence[LocalU1Data],
    *,
    trusted_release_attestations: Sequence[LauncherExecutionAttestation],
    allow_diagnostic: bool,
) -> tuple[
    tuple[LocalU1Data, ...],
    tuple[U1DefectCoordinateBinding, ...],
    RelativeProblem,
    RelativeMatrices,
]:
    if type(source) is not CertifiedCochainProblem:
        raise TypeError("source must be a CertifiedCochainProblem")
    if type(authority) is not Task5VerificationAuthority:
        raise TypeError("authority must be an external Task5VerificationAuthority")
    _require_spatial_character_parent(
        source.character_basis,
        spatial_character_basis,
        spatial_resolution,
    )
    attestations = _canonical_release_attestations(
        source,
        trusted_release_attestations,
        allow_diagnostic=allow_diagnostic,
    )
    attestations_by_id = {
        item.attestation_id: item for item in attestations
    }
    character_report = verify_character_basis(
        source.character_basis,
        source.ambient,
        authority,
        spatial_certificate=spatial_character_basis,
        spatial_resolution=spatial_resolution,
    )
    if not character_report.valid:
        raise ValueError("source coefficient-character basis is not certified")
    if type(rho) is not GF2Character or rho not in source.character_basis.characters:
        raise ValueError("rho is not one of the certified coefficient characters")
    if type(grade) is not GF2Character or grade not in source.character_basis.characters:
        raise ValueError("grade is not one of the certified coefficient characters")
    data = tuple(local_data)
    if not data or any(type(item) is not LocalU1Data for item in data):
        raise ValueError("local_data must contain occupied-orbit inputs")
    instance_ids = tuple(item.instance_id for item in data)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("local_data contains duplicate instance_id")
    if not allow_diagnostic:
        local_inclusions = tuple(item.inclusion for item in data)
        source_inclusion_identities = {id(item) for item in source.inclusions}
        if (
            any(id(item) not in source_inclusion_identities for item in local_inclusions)
            or {id(item) for item in local_inclusions}
            != source_inclusion_identities
        ):
            raise ValueError(
                "release local_data requires exact coverage of the unique "
                "Task-5 source capabilities"
            )

    ambient_complex = None
    local_complexes = []
    restrictions = []
    local_defects = []
    provisional_bindings = []
    for item in data:
        if item.inclusion not in source.inclusions:
            raise ValueError(
                f"{item.instance_id}: inclusion is absent from CertifiedCochainProblem"
            )
        if item.inclusion.target_resolution != source.ambient:
            raise ValueError(
                f"{item.instance_id}: inclusion target differs from ambient resolution"
            )
        if (
            item.bar_equivalence.resolution != item.inclusion.source_resolution
            or item.bar_equivalence.equivalence_id
            != item.inclusion.source_bar_equivalence_id
        ):
            raise ValueError(
                f"{item.instance_id}: bar equivalence is not source-bound to inclusion"
            )
        equivalence_report = verify_bar_resolution_equivalence(
            item.bar_equivalence,
            authority,
        )
        if not equivalence_report.valid:
            raise ValueError(
                f"{item.instance_id}: bar equivalence certificate is invalid"
            )
        table = item.inclusion.source_resolution.finite_group
        if table is None:
            raise ValueError(f"{item.instance_id}: local resolution has no finite table")
        verify_u1_local_skeleton(item.skeleton, table)
        local_rho = GF2Character(item.skeleton.rho_values)
        local_grade = GF2Character(item.skeleton.grade_values)
        trusted_release_attestation = (
            None
            if allow_diagnostic
            else attestations_by_id[
                item.inclusion.launcher_attestation.attestation_id
            ]
        )
        ambient, local, restriction = twist_inclusion_cochain_map(
            item.inclusion,
            rho,
            local_rho,
            authority,
            instance_id=item.instance_id,
            allow_diagnostic=allow_diagnostic,
            trusted_release_attestation=trusted_release_attestation,
        )
        if grade != rho or local_grade != local_rho:
            # This result is discarded: the call exists to certify that the
            # skeleton's normalized grade is the literal restriction of the
            # independently certified ambient grade.
            twist_inclusion_cochain_map(
                item.inclusion,
                grade,
                local_grade,
                authority,
                instance_id=item.instance_id,
                allow_diagnostic=allow_diagnostic,
                trusted_release_attestation=trusted_release_attestation,
            )
        if ambient_complex is None:
            ambient_complex = ambient
        elif ambient_complex != ambient:
            raise ValueError("occupied-orbit restrictions yield different ambient complexes")
        cocycle = _normalized_defect_cocycle(item.skeleton, item.bar_equivalence)
        coordinate = coordinate_bar_cocycle(
            item.bar_equivalence,
            cocycle,
            coefficient_character=local_rho,
        )
        coordinate_report = verify_cochain_coordinate_certificate(
            item.bar_equivalence,
            cocycle,
            coordinate,
        )
        if not coordinate_report.valid or not coordinate.mod_one:
            raise ValueError(
                f"{item.instance_id}: local defect coordinate round trip failed"
            )
        defect = tuple(Phase(value) for value in coordinate.coordinates)
        if len(defect) != local.dimensions[2]:
            raise ValueError(
                f"{item.instance_id}: coordinate dimension differs from local C2"
            )
        local_complexes.append(local)
        restrictions.append(restriction)
        local_defects.append(defect)
        provisional_bindings.append((item, coordinate))
    assert ambient_complex is not None
    relative_problem = RelativeProblem(
        "torus",
        ambient_complex,
        tuple(local_complexes),
        tuple(restrictions),
        tuple(local_defects),
    )
    matrices = assemble_relative_problem(relative_problem)
    defect_digest_by_instance = dict(
        zip(
            matrices.certificate.instance_ids,
            matrices.certificate.defect_digests,
            strict=True,
        )
    )
    bindings = tuple(
        U1DefectCoordinateBinding(
            item.instance_id,
            item.skeleton,
            item.bar_equivalence,
            coordinate,
            defect_digest_by_instance[item.instance_id],
        )
        for item, coordinate in provisional_bindings
    )
    return data, bindings, relative_problem, matrices


def make_u1_sector_problem(
    source: CertifiedCochainProblem,
    rho: GF2Character,
    *,
    grade: GF2Character,
    authority: Task5VerificationAuthority,
    local_data: Sequence[LocalU1Data],
    trusted_release_attestations: Sequence[
        LauncherExecutionAttestation
    ] = (),
    spatial_character_basis: CharacterBasisCertificate | None = None,
    spatial_resolution: FreeResolutionCertificate | None = None,
    diagnostic_h2_invariants: Sequence[int] = (),
    diagnostic_h3_invariants: Sequence[int] = (),
    allow_diagnostic: bool = False,
) -> U1SectorProblem:
    """Build the sole public authority envelope from Task-5 certificates."""

    if type(allow_diagnostic) is not bool:
        raise TypeError("allow_diagnostic must be boolean")
    attestations = _canonical_release_attestations(
        source,
        trusted_release_attestations,
        allow_diagnostic=allow_diagnostic,
    )
    data, bindings, relative_problem, matrices = _derive_sector_material(
        source,
        rho,
        grade,
        authority,
        spatial_character_basis,
        spatial_resolution,
        local_data,
        trusted_release_attestations=attestations,
        allow_diagnostic=allow_diagnostic,
    )
    source_snapshot_digest = _sector_source_snapshot_digest(
        source=source,
        rho=rho,
        grade=grade,
        authority=authority,
        spatial_character_basis=spatial_character_basis,
        spatial_resolution=spatial_resolution,
        local_data=data,
        bindings=bindings,
        relative_problem=relative_problem,
        matrices=matrices,
        diagnostic_only=allow_diagnostic,
    )
    sector_id = _digest(
        "u1-sector-problem",
        _sector_core(
            source=source,
            rho=rho,
            grade=grade,
            authority=authority,
            spatial_character_basis=spatial_character_basis,
            spatial_resolution=spatial_resolution,
            bindings=bindings,
            matrices=matrices,
            source_snapshot_digest=source_snapshot_digest,
            diagnostic_only=allow_diagnostic,
        ),
    )
    verification_snapshot = _U1SectorProblemSnapshot(
        sector_id,
        source_snapshot_digest,
        source,
        rho,
        grade,
        authority,
        spatial_character_basis,
        spatial_resolution,
        data,
        bindings,
        relative_problem,
        matrices,
        allow_diagnostic,
        attestations,
        _U1_SECTOR_FACTORY_TOKEN,
    )
    factory_record = _U1SectorFactoryRecord(
        sector_id,
        source_snapshot_digest,
        source,
        rho,
        grade,
        authority,
        spatial_character_basis,
        spatial_resolution,
        data,
        bindings,
        relative_problem,
        matrices,
        allow_diagnostic,
        attestations,
    )
    _U1_SECTOR_FACTORY_RECORDS[verification_snapshot] = factory_record
    try:
        return U1SectorProblem(
            sector_id,
            source_snapshot_digest,
            source,
            rho,
            grade,
            authority,
            spatial_character_basis,
            spatial_resolution,
            data,
            bindings,
            relative_problem,
            matrices,
            tuple(diagnostic_h2_invariants),
            tuple(diagnostic_h3_invariants),
            allow_diagnostic,
            verification_snapshot,
        )
    except BaseException:
        _U1_SECTOR_FACTORY_RECORDS.pop(verification_snapshot, None)
        raise


def _validate_sector_problem(
    problem: U1SectorProblem,
    *,
    allow_diagnostic: bool,
) -> None:
    if type(problem) is not U1SectorProblem:
        raise TypeError(
            "classify_u1_sector requires a Task-8 U1SectorProblem envelope"
        )
    snapshot = problem._verification_seal
    if type(snapshot) is not _U1SectorProblemSnapshot:
        raise ValueError("U1 sector lacks its nonserialized verification snapshot")
    factory_record = _factory_snapshot_record(snapshot)
    if factory_record is None:
        raise ValueError("U1 sector verification snapshot differs from its factory authority")
    try:
        U1SectorProblem(
            problem.sector_id,
            problem.source_snapshot_digest,
            problem.source,
            problem.rho,
            problem.grade,
            problem.authority,
            problem.spatial_character_basis,
            problem.spatial_resolution,
            problem.local_data,
            problem.bindings,
            problem.relative_problem,
            problem.matrices,
            problem.diagnostic_h2_invariants,
            problem.diagnostic_h3_invariants,
            problem.diagnostic_only,
            snapshot,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "U1 sector envelope differs from its factory verification snapshot"
        ) from error
    if problem.diagnostic_only and not allow_diagnostic:
        raise ValueError(
            "diagnostic U1 sector requires explicit diagnostic opt-in"
        )
    current_snapshot = _sector_source_snapshot_digest(
        source=problem.source,
        rho=problem.rho,
        grade=problem.grade,
        authority=problem.authority,
        spatial_character_basis=problem.spatial_character_basis,
        spatial_resolution=problem.spatial_resolution,
        local_data=problem.local_data,
        bindings=problem.bindings,
        relative_problem=problem.relative_problem,
        matrices=problem.matrices,
        diagnostic_only=problem.diagnostic_only,
    )
    if current_snapshot != problem.source_snapshot_digest:
        raise ValueError("U1 sector envelope differs from its certified source snapshot")
    replayed = _derive_sector_material(
        problem.source,
        problem.rho,
        problem.grade,
        problem.authority,
        problem.spatial_character_basis,
        problem.spatial_resolution,
        problem.local_data,
        trusted_release_attestations=(
            factory_record.trusted_release_attestations
        ),
        allow_diagnostic=problem.diagnostic_only,
    )
    if replayed != (
        problem.local_data,
        problem.bindings,
        problem.relative_problem,
        problem.matrices,
    ):
        raise ValueError(
            "U1 sector material differs from its independent Task-5 source replay"
        )
    verify_relative_certificate(problem.matrices, problem.relative_problem)
    defect_digest_by_instance = dict(
        zip(
            problem.matrices.certificate.instance_ids,
            problem.matrices.certificate.defect_digests,
            strict=True,
        )
    )
    for data, binding in zip(
        problem.local_data,
        problem.bindings,
        strict=True,
    ):
        if (
            binding.skeleton != data.skeleton
            or binding.bar_equivalence != data.bar_equivalence
        ):
            raise ValueError(
                f"{binding.instance_id}: local skeleton/bar equivalence binding differs"
            )
        equivalence_report = verify_bar_resolution_equivalence(
            binding.bar_equivalence,
            problem.authority,
        )
        if not equivalence_report.valid:
            raise ValueError(
                f"{binding.instance_id}: bar equivalence certificate replay failed"
            )
        table = data.inclusion.source_resolution.finite_group
        if table is None:
            raise ValueError(f"{data.instance_id}: local finite table is absent")
        verify_u1_local_skeleton(binding.skeleton, table)
        cocycle = _normalized_defect_cocycle(
            binding.skeleton,
            binding.bar_equivalence,
        )
        report = verify_cochain_coordinate_certificate(
            binding.bar_equivalence,
            cocycle,
            binding.coordinate_certificate,
        )
        if not report.valid or not binding.coordinate_certificate.mod_one:
            raise ValueError(
                f"{binding.instance_id}: local defect coordinate replay failed"
            )
        if (
            binding.relative_defect_digest
            != defect_digest_by_instance[binding.instance_id]
        ):
            raise ValueError(
                f"{binding.instance_id}: relative defect digest differs"
            )
    expected_id = _digest(
        "u1-sector-problem",
        _sector_core(
            source=problem.source,
            rho=problem.rho,
            grade=problem.grade,
            authority=problem.authority,
            spatial_character_basis=problem.spatial_character_basis,
            spatial_resolution=problem.spatial_resolution,
            bindings=problem.bindings,
            matrices=problem.matrices,
            source_snapshot_digest=problem.source_snapshot_digest,
            diagnostic_only=problem.diagnostic_only,
        ),
    )
    if problem.sector_id != expected_id:
        raise ValueError("U1 sector envelope differs from its certified source replay")


def verify_u1_sector_problem(
    problem: U1SectorProblem,
    *,
    allow_diagnostic: bool = False,
) -> VerificationReport:
    issues: list[VerificationIssue] = []
    try:
        _validate_sector_problem(problem, allow_diagnostic=allow_diagnostic)
    except (ArithmeticError, TypeError, ValueError) as error:
        issues.append(VerificationIssue("u1_sector_problem_invalid", str(error)))
    return VerificationReport(not issues, tuple(issues), 7 if not issues else 0)


def classify_u1_sector(
    problem: U1SectorProblem,
    rho: GF2Character,
    *,
    allow_diagnostic: bool = False,
) -> tuple[TorsorStratum | ObstructedBranch, ...]:
    """Classify one certified coefficient sector as a joint compact torsor."""

    _validate_sector_problem(problem, allow_diagnostic=allow_diagnostic)
    if type(rho) is not GF2Character or rho != problem.rho:
        raise ValueError("rho differs from the certified U1 sector envelope")
    result = _solve_source_bound_relative(
        problem.relative_problem,
        problem.matrices,
        rho,
        skeleton_ids=tuple(item.skeleton.skeleton_id for item in problem.bindings),
        coordinate_certificate_ids=tuple(
            item.coordinate_certificate.certificate_id for item in problem.bindings
        ),
        sector_id=problem.sector_id,
    )
    return (result,)


def _solve_source_bound_relative(
    problem: RelativeProblem,
    matrices: RelativeMatrices,
    rho: GF2Character,
    *,
    skeleton_ids: Sequence[str],
    coordinate_certificate_ids: Sequence[str],
    sector_id: str,
) -> TorsorStratum | ObstructedBranch:
    """Internal exact solve after a source problem has crossed its authority gate."""

    if type(problem) is not RelativeProblem or problem.ring != "torus":
        raise TypeError("compact-U1 solve requires a torus RelativeProblem")
    if type(matrices) is not RelativeMatrices:
        raise TypeError("compact-U1 solve requires RelativeMatrices")
    verify_relative_certificate(matrices, problem)
    if type(rho) is not GF2Character or rho != problem.ambient.coefficient_character:
        raise ValueError("rho differs from the source ambient cochain complex")
    _require_digest(sector_id, "$sector_id")
    skeleton_tuple = tuple(skeleton_ids)
    coordinate_tuple = tuple(coordinate_certificate_ids)
    for index, value in enumerate(skeleton_tuple):
        _require_digest(value, f"$skeleton_ids[{index}]")
    for index, value in enumerate(coordinate_tuple):
        _require_digest(value, f"$coordinate_certificate_ids[{index}]")
    if len(skeleton_tuple) != len(matrices.coordinate_blocks.instance_ids):
        raise ValueError("one U1 skeleton is required per occupied orbit")
    if not isinstance(matrices.D, MatrixZ) or not isinstance(matrices.B, MatrixZ):
        raise TypeError("compact-U1 solve requires integer cochain matrices")
    solution = solve_torus_quotient(matrices.D, matrices.B, matrices.offset)
    if isinstance(solution, TorusObstruction):
        return _obstruction_branch(
            sector_id=sector_id,
            skeleton_ids=skeleton_tuple,
            obstruction=solution,
        )
    stratum_id = _digest(
        "u1-torsor-stratum",
        _stratum_core(
            sector_id=sector_id,
            relative_certificate_id=matrices.certificate.certificate_id,
            rho_bits=rho.bits,
            skeleton_ids=skeleton_tuple,
            coordinate_certificate_ids=coordinate_tuple,
            solution=solution,
        ),
    )
    certificate = _make_certificate(
        stratum_id=stratum_id,
        sector_id=sector_id,
        matrices=matrices,
        rho_bits=rho.bits,
        skeleton_ids=skeleton_tuple,
        coordinate_certificate_ids=coordinate_tuple,
        solution=solution,
    )
    return TorsorStratum(
        stratum_id,
        rho.bits,
        skeleton_tuple,
        matrices,
        solution.basepoint,
        solution.group,
        solution.primal_chart,
        tuple(f"phi{index}" for index in range(solution.group.free_rank)),
        certificate,
    )


def verify_u1_stratum_certificate(
    stratum: TorsorStratum,
    source_problem: RelativeProblem,
) -> VerificationReport:
    """Replay a stratum against the source cone, matrices, and dual witnesses."""

    issues: list[VerificationIssue] = []
    try:
        if type(stratum) is not TorsorStratum:
            raise TypeError("stratum must be a TorsorStratum")
        if type(source_problem) is not RelativeProblem:
            raise TypeError("source_problem must be a RelativeProblem")
        verify_relative_certificate(stratum.matrices, source_problem)
        if stratum.rho_bits != source_problem.ambient.coefficient_character.bits:
            raise ValueError("rho differs from the source ambient complex")
        # Reconstructing closes direct object mutation before the semantic
        # replay against B, D, the Smith witnesses, and the primal chart.
        checked = TorsorStratum(
            stratum.stratum_id,
            stratum.rho_bits,
            stratum.skeleton_ids,
            stratum.matrices,
            stratum.basepoint,
            stratum.homogeneous_group,
            stratum.primal_chart,
            stratum.free_parameters,
            stratum.certificate,
        )
        _replay_stratum(checked)
    except (ArithmeticError, TypeError, ValueError) as error:
        issues.append(
            VerificationIssue("u1_stratum_certificate_invalid", str(error))
        )
    return VerificationReport(not issues, tuple(issues), 8 if not issues else 0)


def symbolic_torsor_point(
    stratum: TorsorStratum,
    free: tuple[str, ...],
    torsion: tuple[int, ...],
) -> SymbolicPoint:
    if type(stratum) is not TorsorStratum:
        raise TypeError("symbolic_torsor_point requires TorsorStratum")
    _replay_stratum(stratum)
    names = tuple(free)
    if len(names) != stratum.homogeneous_group.free_rank:
        raise ValueError("symbolic free-parameter count differs from free rank")
    if any(type(name) is not str or _PARAMETER_RE.fullmatch(name) is None for name in names):
        raise ValueError("symbolic free parameters must be identifier strings")
    if len(set(names)) != len(names):
        raise ValueError("symbolic free parameters must be unique")
    residues = tuple(torsion)
    if len(residues) != len(stratum.homogeneous_group.torsion_orders):
        raise ValueError("torsion-coordinate count differs from torsion rank")
    for index, (value, order) in enumerate(
        zip(residues, stratum.homogeneous_group.torsion_orders, strict=True)
    ):
        if type(value) is not int or not 0 <= value < order:
            raise ValueError(
                f"torsion coordinate {index} is not canonical modulo {order}"
            )
    constant = tuple(
        Phase(
            stratum.basepoint[row].value
            + sum(
                (
                    residues[column]
                    * stratum.primal_chart.torsion_lifts[row][column].value
                    for column in range(len(residues))
                ),
                Fraction(0),
            )
        )
        for row in range(stratum.primal_chart.raw_dimension)
    )
    if tuple(
        Phase(
            sum(
                (entry * value.value for entry, value in zip(row, constant, strict=True)),
                Fraction(0),
            )
        )
        for row in stratum.matrices.D
    ) != stratum.matrices.offset:
        raise ArithmeticError("symbolic torsion lift does not satisfy D z = offset")
    if matmul(stratum.matrices.D, stratum.primal_chart.free_lifts) != zero_matrix(
        stratum.matrices.D.row_count,
        stratum.homogeneous_group.free_rank,
    ):
        raise ArithmeticError("symbolic free lifts do not solve the homogeneous equation")
    core = {
        "constant": [_phase_text(item) for item in constant],
        "free_coefficients": _matrix_mapping(stratum.primal_chart.free_lifts),
        "parameter_names": list(names),
        "relative_certificate_id": stratum.matrices.certificate.certificate_id,
        "stratum_id": stratum.stratum_id,
        "torsion_coordinates": list(residues),
    }
    return SymbolicPoint(
        _digest("symbolic-point", core),
        stratum.stratum_id,
        names,
        residues,
        constant,
        stratum.primal_chart.free_lifts,
        stratum.matrices.certificate.certificate_id,
    )


__all__ = [
    "ContinuousStratumError",
    "LocalU1Data",
    "SymbolicPoint",
    "TorsorStratum",
    "U1DefectCoordinateBinding",
    "U1SectorProblem",
    "U1StratumCertificate",
    "classify_u1_sector",
    "make_u1_sector_problem",
    "symbolic_torsor_point",
    "verify_u1_sector_problem",
    "verify_u1_stratum_certificate",
]
