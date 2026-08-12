"""Exact symbolic reconstruction on a finite certified site domain.

The public evaluator is intentionally fail-closed.  A stratum is not enough
to reconstruct a representative: callers must supply a :class:`ReconstructionProblem`
that binds the finite presentation, exact normalized-bar queries, catalogue
site transports, generator action, and one verified local skeleton per orbit.
Only sites and bar queries present in that envelope may be evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Sequence, TypeAlias

from .algebraic import ExactQuaternion, ONE_QUATERNION
from .certificates import (
    BarEvaluatorCertificate,
    ExactGaugeElement,
    FormalPhase,
    GeneratorAction,
    OrbitReconstructionData,
    PSGEvaluatorCertificate,
    PeriodicSite,
    ReconstructionProblem,
    RelationCertificate,
    RelationEvaluation,
    Relator,
    SiteTransport,
    _digest,
    _phase_text,
)
from .integer_linalg import MatrixZ
from .relative_complex import ExactCoefficient, RelativeMatrices, RelativeProblem
from .torus import Phase
from .u1_classifier import SymbolicPoint, TorsorStratum
from .u1_local import U1LocalSkeleton
from .z2_classifier import CertifiedZ2Point, FiniteAffineStratum
from .z2_local import Z2LocalSkeleton


Stratum: TypeAlias = TorsorStratum | FiniteAffineStratum
Point: TypeAlias = SymbolicPoint | CertifiedZ2Point


class ReconstructionDomainError(ValueError):
    """A requested site, stabilizer element, or bar tuple is uncertified."""

    code = "certificate_invalid"

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        super().__init__(f"{self.code}: {stage}: {detail}")


def _matrix_mapping(matrix: MatrixZ) -> dict[str, object]:
    return {
        "column_count": matrix.column_count,
        "rows": [list(row) for row in matrix],
    }


def _formal_mapping(value: FormalPhase) -> dict[str, object]:
    return value.mapping()


def make_site_transport(
    site: PeriodicSite,
    *,
    ambient_element: str,
    catalogue_transport_digest: str,
    z2_transport_lift: ExactQuaternion | None = None,
) -> SiteTransport:
    if type(site) is not PeriodicSite:
        raise TypeError("make_site_transport requires PeriodicSite")
    core = {
        "ambient_element": ambient_element,
        "catalogue_transport_digest": catalogue_transport_digest,
        "site": site.mapping(),
        "z2_transport_lift": (
            None if z2_transport_lift is None else z2_transport_lift.to_json()
        ),
    }
    return SiteTransport(
        _digest("site-transport", core),
        site,
        ambient_element,
        catalogue_transport_digest,
        z2_transport_lift,
    )


def make_generator_action(
    *,
    name: str,
    ambient_element: str,
    antiunitary_grade: int,
    inverse_site_images: Sequence[tuple[PeriodicSite, PeriodicSite]],
    action_provenance_digest: str,
) -> GeneratorAction:
    images = tuple(sorted(tuple(tuple(pair) for pair in inverse_site_images)))
    core = {
        "action_provenance_digest": action_provenance_digest,
        "ambient_element": ambient_element,
        "antiunitary_grade": antiunitary_grade,
        "inverse_site_images": [
            [source.mapping(), target.mapping()] for source, target in images
        ],
        "name": name,
    }
    return GeneratorAction(
        _digest("generator-action", core),
        name,
        ambient_element,
        antiunitary_grade,
        images,
        action_provenance_digest,
    )


def make_relator(
    name: str,
    kind: Literal["spatial", "time_square", "mixed_time_space"],
    word: Sequence[str],
) -> Relator:
    tokens = tuple(word)
    core = {"kind": kind, "name": name, "word": list(tokens)}
    return Relator(_digest("relator", core), name, kind, tokens)


def _z2_local_grades(skeleton: Z2LocalSkeleton, order: int) -> tuple[int, ...]:
    if skeleton.full_graded_su2_lifts:
        spatial_order = len(skeleton.su2_lifts)
        if order != 2 * spatial_order:
            raise ValueError("graded Z2 skeleton and local finite table have different orders")
        return (0,) * spatial_order + (1,) * spatial_order
    if order != len(skeleton.su2_lifts):
        raise ValueError("spatial Z2 skeleton and local finite table have different orders")
    return (0,) * order


def make_orbit_reconstruction_data(
    *,
    instance_id: str,
    local_evaluator: BarEvaluatorCertificate,
    skeleton: U1LocalSkeleton | Z2LocalSkeleton,
    stabilizer_element_map: Sequence[tuple[str, str]],
    catalogue_record_digest: str,
    literal_stabilizer_digest: str,
    local_grade_values: Sequence[int] | None = None,
) -> OrbitReconstructionData:
    if type(local_evaluator) is not BarEvaluatorCertificate:
        raise TypeError("local evaluator must be a BarEvaluatorCertificate")
    mapping = tuple(sorted(tuple(tuple(pair) for pair in stabilizer_element_map)))
    order = len(local_evaluator.finite_group.element_order)
    if type(skeleton) is U1LocalSkeleton:
        inferred_grades = skeleton.grade_values
        rhos = skeleton.rho_values
    elif type(skeleton) is Z2LocalSkeleton:
        inferred_grades = _z2_local_grades(skeleton, order)
        rhos = (0,) * order
    else:
        raise TypeError("orbit reconstruction requires a verified local skeleton")
    grades = (
        inferred_grades
        if local_grade_values is None
        else tuple(local_grade_values)
    )
    core = {
        "catalogue_record_digest": catalogue_record_digest,
        "instance_id": instance_id,
        "literal_stabilizer_digest": literal_stabilizer_digest,
        "local_evaluator_id": local_evaluator.evaluator_id,
        "local_grade_values": list(grades),
        "local_rho_values": list(rhos),
        "skeleton_id": skeleton.skeleton_id,
        "stabilizer_element_map": [list(pair) for pair in mapping],
    }
    return OrbitReconstructionData(
        _digest("orbit-reconstruction-data", core),
        instance_id,
        local_evaluator,
        skeleton,
        mapping,
        tuple(grades),
        tuple(rhos),
        catalogue_record_digest,
        literal_stabilizer_digest,
    )


def make_diagnostic_reconstruction_problem(
    *,
    relative_problem: RelativeProblem,
    relative_matrices: RelativeMatrices,
    relative_certificate_id: str,
    igg: Literal["Z2", "U1"],
    ambient_evaluator: BarEvaluatorCertificate,
    ambient_grade_values: Sequence[int],
    ambient_rho_values: Sequence[int],
    generators: Sequence[GeneratorAction],
    site_transports: Sequence[SiteTransport],
    orbits: Sequence[OrbitReconstructionData],
    relators: Sequence[Relator],
) -> ReconstructionProblem:
    if type(ambient_evaluator) is not BarEvaluatorCertificate:
        raise TypeError(
            "diagnostic ReconstructionProblem requires an ambient evaluator certificate"
        )
    generator_tuple = tuple(generators)
    transport_tuple = tuple(sorted(tuple(site_transports), key=lambda item: item.site))
    orbit_tuple = tuple(sorted(tuple(orbits), key=lambda item: item.instance_id))
    relator_tuple = tuple(relators)
    grades = tuple(ambient_grade_values)
    rhos = tuple(ambient_rho_values)
    diagnostic = ambient_evaluator.diagnostic or any(
        item.local_evaluator.diagnostic for item in orbit_tuple
    )
    if not diagnostic:
        raise ValueError("diagnostic problem factory requires a diagnostic evaluator")
    core = {
        "ambient_evaluator_id": ambient_evaluator.evaluator_id,
        "ambient_grade_values": list(grades),
        "ambient_rho_values": list(rhos),
        "diagnostic": diagnostic,
        "generator_action_ids": [item.action_id for item in generator_tuple],
        "igg": igg,
        "orbit_ids": [item.orbit_id for item in orbit_tuple],
        "relative_certificate_id": relative_certificate_id,
        "relator_ids": [item.relator_id for item in relator_tuple],
        "site_transport_ids": [item.transport_id for item in transport_tuple],
    }
    return ReconstructionProblem(
        _digest("reconstruction-problem", core),
        relative_certificate_id,
        relative_problem,
        relative_matrices,
        igg,
        ambient_evaluator,
        grades,
        rhos,
        generator_tuple,
        transport_tuple,
        orbit_tuple,
        relator_tuple,
        diagnostic,
    )


@dataclass(frozen=True, slots=True)
class SymbolicPSG:
    """One exact reconstructed representative with unsampled parameters."""

    stratum_id: str
    basepoint_coordinates: tuple[ExactCoefficient, ...]
    free_parameters: tuple[str, ...]
    evaluator: PSGEvaluatorCertificate

    def __post_init__(self) -> None:
        if type(self.evaluator) is not PSGEvaluatorCertificate:
            raise TypeError("$SymbolicPSG.evaluator: expected PSGEvaluatorCertificate")
        if self.stratum_id != self.evaluator.stratum_id:
            raise ValueError("symbolic PSG and evaluator bind different strata")
        coordinates = tuple(self.basepoint_coordinates)
        if any(type(value) not in (int, Phase) for value in coordinates):
            raise TypeError("symbolic PSG basepoint contains an inexact coordinate")
        parameters = tuple(self.free_parameters)
        if parameters != self.evaluator.parameter_names:
            raise ValueError("symbolic PSG free parameters differ from evaluator")
        object.__setattr__(self, "basepoint_coordinates", coordinates)
        object.__setattr__(self, "free_parameters", parameters)


def _matrices_for(stratum: Stratum) -> RelativeMatrices:
    if type(stratum) is TorsorStratum:
        return stratum.matrices
    return stratum.certificate.matrices


def _coordinate_core(
    stratum: Stratum,
    point: Point,
    raw_constant: tuple[Phase, ...],
    raw_free: MatrixZ,
    parameters: tuple[str, ...],
) -> dict[str, object]:
    return {
        "parameter_names": list(parameters),
        "point_id": point.point_id,
        "raw_constant": [_phase_text(value) for value in raw_constant],
        "raw_free_coefficients": _matrix_mapping(raw_free),
        "stratum_id": stratum.stratum_id,
    }


def _validate_reconstruction_binding(
    problem: ReconstructionProblem,
    stratum: Stratum,
    point: Point,
    raw_constant: tuple[Phase, ...],
    raw_free: MatrixZ,
) -> None:
    matrices = _matrices_for(stratum)
    if matrices.certificate.certificate_id != problem.relative_certificate_id:
        raise ValueError("reconstruction envelope binds another relative certificate")
    if matrices != problem.relative_matrices:
        raise ValueError("classified stratum matrices differ from reconstruction envelope")
    if point.stratum_id != stratum.stratum_id:
        raise ValueError("reconstruction point binds another stratum")
    if len(raw_constant) != matrices.D.column_count:
        raise ValueError("reconstruction point has the wrong raw relative dimension")
    if raw_free.row_count != len(raw_constant):
        raise ValueError("reconstruction free lifts have the wrong raw dimension")
    blocks = matrices.coordinate_blocks
    orbit_ids = tuple(item.instance_id for item in problem.orbits)
    if blocks.instance_ids != orbit_ids:
        raise ValueError("reconstruction orbits differ from relative coordinate blocks")
    if stratum.skeleton_ids != tuple(item.skeleton.skeleton_id for item in problem.orbits):
        raise ValueError("reconstruction local skeletons differ from classified stratum")
    ambient_start, ambient_stop = blocks.ambient_slices[1]
    ambient_dimension = ambient_stop - ambient_start
    if problem.ambient_evaluator.coordinate_dimensions[2] != ambient_dimension:
        raise ValueError("ambient evaluator has the wrong degree-two coordinate domain")
    for index, orbit in enumerate(problem.orbits):
        local_start, local_stop = blocks.local_slices[1][index]
        if orbit.local_evaluator.coordinate_dimensions[1] != local_stop - local_start:
            raise ValueError(
                f"local evaluator has the wrong degree-one domain for {orbit.instance_id}"
            )


def reconstruct_psg(
    problem: ReconstructionProblem,
    stratum: Stratum,
    point: Point,
) -> SymbolicPSG:
    """Reconstruct a symbolic representative from a complete authority envelope."""

    if type(problem) is not ReconstructionProblem:
        raise TypeError(
            "reconstruct_psg requires a ReconstructionProblem envelope, not a bare stratum"
        )
    if problem.igg == "U1":
        if type(stratum) is not TorsorStratum or type(point) is not SymbolicPoint:
            raise TypeError("U1 reconstruction requires TorsorStratum and SymbolicPoint")
        if point.relative_certificate_id != problem.relative_certificate_id:
            raise ValueError("symbolic point binds another relative certificate")
        raw_constant = point.constant
        raw_free = point.free_coefficients
        parameters = point.parameter_names
        basepoint: tuple[ExactCoefficient, ...] = point.constant
    else:
        if type(stratum) is not FiniteAffineStratum or type(point) is not CertifiedZ2Point:
            raise TypeError("Z2 reconstruction requires FiniteAffineStratum and CertifiedZ2Point")
        raw_constant = tuple(Phase(Fraction(bit, 2)) for bit in point.representative)
        raw_free = MatrixZ(
            tuple(() for _ in raw_constant),
            column_count=0,
        )
        parameters = ()
        basepoint = point.representative
    _validate_reconstruction_binding(
        problem,
        stratum,
        point,
        raw_constant,
        raw_free,
    )
    coordinate_core = _coordinate_core(
        stratum,
        point,
        raw_constant,
        raw_free,
        parameters,
    )
    coordinate_digest = _digest("symbolic-reconstruction-coordinates", coordinate_core)
    evaluator_core = {
        "ambient_degree_two_slice": list(
            _matrices_for(stratum).coordinate_blocks.ambient_slices[1]
        ),
        "coordinate_digest": coordinate_digest,
        "diagnostic": problem.diagnostic,
        "parameter_names": list(parameters),
        "point_id": point.point_id,
        "problem_id": problem.problem_id,
        "raw_constant": [_phase_text(value) for value in raw_constant],
        "raw_free_coefficients": _matrix_mapping(raw_free),
        "local_degree_one_slices": [
            [instance_id, list(coordinate_slice)]
            for instance_id, coordinate_slice in zip(
                _matrices_for(stratum).coordinate_blocks.instance_ids,
                _matrices_for(stratum).coordinate_blocks.local_slices[1],
                strict=True,
            )
        ],
        "relative_certificate_id": problem.relative_certificate_id,
        "stratum_id": stratum.stratum_id,
    }
    evaluator = PSGEvaluatorCertificate(
        _digest("psg-evaluator-certificate", evaluator_core),
        problem,
        stratum.stratum_id,
        point.point_id,
        problem.relative_certificate_id,
        parameters,
        raw_constant,
        raw_free,
        _matrices_for(stratum).coordinate_blocks.ambient_slices[1],
        tuple(
            zip(
                _matrices_for(stratum).coordinate_blocks.instance_ids,
                _matrices_for(stratum).coordinate_blocks.local_slices[1],
                strict=True,
            )
        ),
        coordinate_digest,
        problem.diagnostic,
    )
    return SymbolicPSG(stratum.stratum_id, basepoint, parameters, evaluator)


def _raw_formal_coordinates(psg: SymbolicPSG) -> tuple[FormalPhase, ...]:
    certificate = psg.evaluator
    return tuple(
        FormalPhase(
            certificate.parameter_names,
            tuple(certificate.raw_free_coefficients[row]),
            certificate.raw_constant[row],
        )
        for row in range(len(certificate.raw_constant))
    )


def _bar_value(
    evaluator: BarEvaluatorCertificate,
    coordinates: Sequence[FormalPhase],
    group_tuple: Sequence[str],
    *,
    stage: str,
    coefficient_character: Sequence[int],
) -> FormalPhase:
    values = tuple(coordinates)
    names = values[0].parameter_names if values else ()
    if any(value.parameter_names != names for value in values):
        raise ValueError("bar-coordinate phases use different formal bases")
    degree = len(tuple(group_tuple))
    if degree >= len(evaluator.coordinate_dimensions) or len(values) != evaluator.coordinate_dimensions[degree]:
        raise ValueError("bar evaluator received the wrong coordinate dimension")
    try:
        weights = evaluator.coordinate_weights(
            tuple(group_tuple),
            coefficient_character,
        )
    except KeyError as error:
        raise ReconstructionDomainError(stage, str(error)) from error
    if weights is None:
        return FormalPhase.zero(names)
    result = FormalPhase.zero(names)
    for weight, value in zip(weights, values, strict=True):
        result = result + value.scale(weight)
    return result


def _element_index(table, element: str, *, stage: str) -> int:
    try:
        return table.element_order.index(element)
    except ValueError as error:
        raise ReconstructionDomainError(
            stage,
            f"finite presentation has no element {element!r}",
        ) from error


def _multiply_element(table, left: str, right: str) -> str:
    left_index = _element_index(table, left, stage="reconstruction.group_domain")
    right_index = _element_index(table, right, stage="reconstruction.group_domain")
    return table.element_order[table.multiplication_table[left_index][right_index]]


def _inverse_element(table, element: str) -> str:
    index = _element_index(table, element, stage="reconstruction.group_domain")
    return table.element_order[table.inverse_indices[index]]


@dataclass(frozen=True, slots=True)
class _ExtensionElement:
    phase: FormalPhase
    element: str


def _ambient_coordinates(psg: SymbolicPSG) -> tuple[FormalPhase, ...]:
    start, stop = psg.evaluator.ambient_degree_two_slice
    return _raw_formal_coordinates(psg)[start:stop]


def _extension_multiply(
    psg: SymbolicPSG,
    left: _ExtensionElement,
    right: _ExtensionElement,
) -> _ExtensionElement:
    problem = psg.evaluator.problem
    table = problem.ambient_evaluator.finite_group
    left_index = _element_index(table, left.element, stage="reconstruction.group_domain")
    sign = -1 if problem.ambient_rho_values[left_index] else 1
    omega = _bar_value(
        problem.ambient_evaluator,
        _ambient_coordinates(psg),
        (left.element, right.element),
        stage="reconstruction.ambient_bar_domain",
        coefficient_character=problem.ambient_rho_values,
    )
    return _ExtensionElement(
        left.phase + right.phase.scale(sign) + omega,
        _multiply_element(table, left.element, right.element),
    )


def _extension_inverse(psg: SymbolicPSG, value: _ExtensionElement) -> _ExtensionElement:
    problem = psg.evaluator.problem
    table = problem.ambient_evaluator.finite_group
    inverse = _inverse_element(table, value.element)
    index = _element_index(table, value.element, stage="reconstruction.group_domain")
    sign = -1 if problem.ambient_rho_values[index] else 1
    omega = _bar_value(
        problem.ambient_evaluator,
        _ambient_coordinates(psg),
        (value.element, inverse),
        stage="reconstruction.ambient_bar_domain",
        coefficient_character=problem.ambient_rho_values,
    )
    return _ExtensionElement((-(value.phase + omega)).scale(sign), inverse)


def _canonical_extension(psg: SymbolicPSG, element: str) -> _ExtensionElement:
    return _ExtensionElement(FormalPhase.zero(psg.free_parameters), element)


def _generator_and_direction(
    problem: ReconstructionProblem,
    token: str,
) -> tuple[GeneratorAction, bool]:
    inverse = token.endswith("^-1")
    name = token[:-3] if inverse else token
    for generator in problem.generators:
        if generator.name == name:
            return generator, inverse
    raise ReconstructionDomainError(
        "reconstruction.generator_domain",
        f"unknown generator token {token!r}",
    )


def _source_site(
    problem: ReconstructionProblem,
    token: str,
    site: PeriodicSite,
) -> PeriodicSite:
    generator, inverse = _generator_and_direction(problem, token)
    mapping = dict(generator.inverse_site_images)
    if inverse:
        mapping = {target: source for source, target in generator.inverse_site_images}
    try:
        return mapping[site]
    except KeyError as error:
        raise ReconstructionDomainError(
            "reconstruction.site_domain",
            f"site {site!r} is outside the certified generator action",
        ) from error


def _site_transport(problem: ReconstructionProblem, site: PeriodicSite) -> SiteTransport:
    for transport in problem.site_transports:
        if transport.site == site:
            return transport
    raise ReconstructionDomainError(
        "reconstruction.site_domain",
        f"site {site!r} has no certified catalogue transport",
    )


def _orbit(problem: ReconstructionProblem, instance_id: str) -> OrbitReconstructionData:
    for orbit in problem.orbits:
        if orbit.instance_id == instance_id:
            return orbit
    raise ReconstructionDomainError(
        "reconstruction.orbit_domain",
        f"orbit instance {instance_id!r} is outside the reconstruction envelope",
    )


def _local_coordinates(
    psg: SymbolicPSG,
    instance_id: str,
) -> tuple[FormalPhase, ...]:
    try:
        coordinate_slices = dict(psg.evaluator.local_degree_one_slices)
        start, stop = coordinate_slices[instance_id]
    except KeyError as error:
        raise ReconstructionDomainError(
            "reconstruction.orbit_domain",
            f"orbit instance {instance_id!r} has no relative coordinate slice",
        ) from error
    return _raw_formal_coordinates(psg)[start:stop]


def _local_element(orbit: OrbitReconstructionData, ambient_element: str) -> str:
    mapping = dict(orbit.stabilizer_element_map)
    try:
        return mapping[ambient_element]
    except KeyError as error:
        raise ReconstructionDomainError(
            "reconstruction.stabilizer_domain",
            f"ambient element {ambient_element!r} is outside orbit {orbit.instance_id!r}",
        ) from error


def _theta(
    psg: SymbolicPSG,
    orbit: OrbitReconstructionData,
    value: _ExtensionElement,
) -> ExactGaugeElement:
    local_element = _local_element(orbit, value.element)
    local_table = orbit.local_evaluator.finite_group
    local_index = _element_index(
        local_table,
        local_element,
        stage="reconstruction.local_group_domain",
    )
    cochain = _bar_value(
        orbit.local_evaluator,
        _local_coordinates(psg, orbit.instance_id),
        (local_element,),
        stage="reconstruction.local_bar_domain",
        coefficient_character=orbit.local_rho_values,
    )
    phase = value.phase + cochain
    if type(orbit.skeleton) is U1LocalSkeleton:
        return ExactGaugeElement(
            "U1",
            phase,
            orbit.skeleton.q_values[local_index],
            orbit.skeleton.grade_values[local_index],
            None,
        )
    skeleton = orbit.skeleton
    lifts = (
        skeleton.full_graded_su2_lifts
        if skeleton.full_graded_su2_lifts
        else skeleton.su2_lifts
    )
    return ExactGaugeElement(
        "Z2",
        phase,
        0,
        orbit.local_grade_values[local_index],
        lifts[local_index],
    )


def _transport_gauge(psg: SymbolicPSG, transport: SiteTransport) -> ExactGaugeElement:
    problem = psg.evaluator.problem
    table = problem.ambient_evaluator.finite_group
    ambient_element = transport.ambient_element
    index = _element_index(table, ambient_element, stage="reconstruction.transport_domain")
    grade = problem.ambient_grade_values[index]
    if problem.igg == "U1":
        rho = problem.ambient_rho_values[index]
        return ExactGaugeElement(
            "U1",
            FormalPhase.zero(psg.free_parameters),
            grade ^ rho,
            grade,
            None,
        )
    return ExactGaugeElement(
        "Z2",
        FormalPhase.zero(psg.free_parameters),
        0,
        0,
        (
            ONE_QUATERNION
            if transport.z2_transport_lift is None
            else transport.z2_transport_lift
        ),
    )


def evaluate_generator(
    psg: SymbolicPSG,
    generator: str,
    site: PeriodicSite,
) -> ExactGaugeElement:
    if type(psg) is not SymbolicPSG:
        raise TypeError("evaluate_generator requires SymbolicPSG")
    if type(site) is not PeriodicSite:
        raise TypeError("evaluate_generator requires PeriodicSite")
    problem = psg.evaluator.problem
    action, inverse = _generator_and_direction(problem, generator)
    source = _source_site(problem, generator, site)
    target_transport = _site_transport(problem, site)
    source_transport = _site_transport(problem, source)
    table = problem.ambient_evaluator.finite_group
    ambient_generator = (
        _inverse_element(table, action.ambient_element)
        if inverse
        else action.ambient_element
    )
    stabilizer = _extension_multiply(
        psg,
        _extension_multiply(
            psg,
            _extension_inverse(
                psg,
                _canonical_extension(psg, target_transport.ambient_element),
            ),
            _canonical_extension(psg, ambient_generator),
        ),
        _canonical_extension(psg, source_transport.ambient_element),
    )
    orbit = _orbit(problem, site.instance_id)
    if source.instance_id != orbit.instance_id:
        raise ReconstructionDomainError(
            "reconstruction.orbit_domain",
            "generator action crosses certified orbit instances",
        )
    result = (
        _transport_gauge(psg, target_transport)
        * _theta(psg, orbit, stabilizer)
        * _transport_gauge(psg, source_transport).inverse()
    )
    expected_grade = problem.ambient_grade_values[
        _element_index(table, ambient_generator, stage="reconstruction.group_domain")
    ]
    if result.antiunitary_grade != expected_grade:
        raise ValueError("reconstructed generator has the wrong antiunitary grade")
    return result


def evaluate_word(
    psg: SymbolicPSG,
    word: Sequence[str],
    site: PeriodicSite,
) -> ExactGaugeElement:
    if type(psg) is not SymbolicPSG:
        raise TypeError("evaluate_word requires SymbolicPSG")
    tokens = tuple(word)
    result = ExactGaugeElement.identity(psg.evaluator.problem.igg, psg.free_parameters)
    cursor = site
    for token in tokens:
        result = result * evaluate_generator(psg, token, cursor)
        cursor = _source_site(psg.evaluator.problem, token, cursor)
    return result


def _expected_word_extension(
    psg: SymbolicPSG,
    word: Sequence[str],
) -> _ExtensionElement:
    problem = psg.evaluator.problem
    table = problem.ambient_evaluator.finite_group
    identity = table.element_order[table.identity_index]
    result = _canonical_extension(psg, identity)
    for token in word:
        generator, inverse = _generator_and_direction(problem, token)
        element = (
            _inverse_element(table, generator.ambient_element)
            if inverse
            else generator.ambient_element
        )
        result = _extension_multiply(psg, result, _canonical_extension(psg, element))
    if result.element != identity:
        raise ValueError("defining relator word does not close in the ambient presentation")
    return result


def _relation_evaluation(
    relator: Relator,
    site: PeriodicSite,
    actual: FormalPhase,
    expected: FormalPhase,
) -> RelationEvaluation:
    residual = actual - expected
    core = {
        "actual_phase": _formal_mapping(actual),
        "expected_phase": _formal_mapping(expected),
        "kind": relator.kind,
        "relation_name": relator.name,
        "relator_id": relator.relator_id,
        "residual": _formal_mapping(residual),
        "site": site.mapping(),
    }
    return RelationEvaluation(
        _digest("relation-evaluation", core),
        relator.relator_id,
        relator.name,
        relator.kind,
        site,
        actual,
        expected,
        residual,
    )


def verify_relations(
    psg: SymbolicPSG,
    sites: Sequence[PeriodicSite],
) -> RelationCertificate:
    if type(psg) is not SymbolicPSG:
        raise TypeError("verify_relations requires SymbolicPSG")
    problem = psg.evaluator.problem
    supplied_sites = tuple(sites)
    if any(type(site) is not PeriodicSite for site in supplied_sites):
        raise TypeError("relation replay site domain contains a non-PeriodicSite")
    certified_sites = tuple(sorted(transport.site for transport in problem.site_transports))
    if (
        len(set(supplied_sites)) != len(supplied_sites)
        or set(supplied_sites) != set(certified_sites)
    ):
        raise ValueError("relation replay requires the complete certified site domain")
    site_tuple = certified_sites
    results: list[RelationEvaluation] = []
    for relator in problem.relators:
        expected = _expected_word_extension(psg, relator.word).phase
        for site in site_tuple:
            cursor = site
            for token in relator.word:
                cursor = _source_site(problem, token, cursor)
            if cursor != site:
                raise ValueError(
                    f"relator {relator.name!r} does not close on certified site {site!r}"
                )
            actual_element = evaluate_word(psg, relator.word, site)
            if not actual_element.is_central:
                raise ValueError(
                    f"relator {relator.name!r} does not land in the diagonal IGG"
                )
            result = _relation_evaluation(
                relator,
                site,
                actual_element.central_phase(),
                expected,
            )
            if not result.residual.is_zero:
                raise ValueError(
                    f"exact relator replay failed for {relator.name!r} at {site!r}"
                )
            results.append(result)
    result_tuple = tuple(results)
    core = {
        "diagnostic": psg.evaluator.diagnostic,
        "evaluator_certificate_id": psg.evaluator.certificate_id,
        "query_sites": [site.mapping() for site in site_tuple],
        "relation_ids": [item.relation_id for item in result_tuple],
        "verified": True,
    }
    return RelationCertificate(
        _digest("relation-certificate", core),
        psg.evaluator,
        psg.evaluator.certificate_id,
        site_tuple,
        psg.evaluator.diagnostic,
        result_tuple,
        len(result_tuple),
        True,
    )


__all__ = [
    "PeriodicSite",
    "ReconstructionDomainError",
    "SymbolicPSG",
    "evaluate_generator",
    "evaluate_word",
    "make_diagnostic_reconstruction_problem",
    "make_generator_action",
    "make_orbit_reconstruction_data",
    "make_relator",
    "make_site_transport",
    "reconstruct_psg",
    "verify_relations",
]
