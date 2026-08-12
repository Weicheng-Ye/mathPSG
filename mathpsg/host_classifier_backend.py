"""Fresh GAP algebra for the physical PSG solvers.

The GAP programs return only the finite tables, resolutions, restriction maps,
and bar-comparison coordinates consumed by :mod:`mathpsg.compute`.  This layer
does not create or replay certificates, bind source hashes, enforce package
versions, or read/write a result cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile

from .direct_algebra import (
    InclusionAlgebra,
    enumerate_characters,
    parse_inclusion,
    PCPNormalForm,
    word_character,
)
from .gf2 import GF2Character
from .local_gap import GapRuntime
from .torus import Phase
from .direct_z2 import (
    Z2LocalSkeleton,
    enumerate_graded_z2_skeletons,
    enumerate_spatial_z2_skeletons,
)


DirectInclusion = InclusionAlgebra


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DirectHostAmbient:
    resolution: object
    inclusions: tuple[DirectInclusion, ...]
    time_reversal: bool
    spatial: DirectHostAmbient | None = None

    def inclusion_for(self, inclusion_id: str) -> DirectInclusion:
        return next(
            item for item in self.inclusions if item.inclusion_id == inclusion_id
        )


@dataclass(frozen=True, slots=True)
class DirectU1LocalSkeleton:
    element_order: tuple[str, ...]
    grade_values: tuple[int, ...]
    rho_values: tuple[int, ...]
    normalized_bar_defect: tuple[tuple[Phase, ...], ...]


def _gap_sources(root: Path) -> str:
    names = (
        "protocol.g",
        "affine_pcp.g",
        "resolutions.g",
        "restrictions.g",
        "bar_equivalence.g",
    )
    try:
        return "\n".join(
            (root / "gap" / "classifier" / "lib" / name).read_text(
                encoding="utf-8"
            )
            for name in names
        )
    except (OSError, UnicodeError) as error:
        raise RuntimeError("local GAP computation sources are unavailable") from error


def _run_gap_program(
    program: str,
    *,
    runtime: GapRuntime,
    root: Path,
    timeout_seconds: int,
) -> Mapping[str, object]:
    with tempfile.TemporaryDirectory(prefix="mathpsg-gap-") as raw:
        execution_root = Path(raw)
        for item in root.iterdir():
            (execution_root / item.name).symlink_to(
                item, target_is_directory=item.is_dir()
            )
        output_name = "mathpsg-result.json"
        try:
            completed = subprocess.run(
                (runtime.executable, "-q"),
                cwd=execution_root,
                input=program.replace(
                    "{output_path}", json.dumps(output_name)
                ).encode("utf-8"),
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("GAP computation failed") from error
        if completed.returncode != 0:
            diagnostic = completed.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(
                f"GAP computation exited with status {completed.returncode}: {diagnostic}"
            )
        try:
            value = json.loads((execution_root / output_name).read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GAP output is not valid JSON") from error
    return value  # type: ignore[return-value]


def _identity_first(elements: Sequence[object]) -> tuple[object, ...]:
    identity_matrix = (
        ("q(1,1)", "q(0,1)", "q(0,1)"),
        ("q(0,1)", "q(1,1)", "q(0,1)"),
        ("q(0,1)", "q(0,1)", "q(1,1)"),
    )
    zero = ("q(0,1)", "q(0,1)", "q(0,1)")
    values = tuple(elements)
    position = next(
        (
            index
            for index, value in enumerate(values)
            if isinstance(value, Mapping)
            and tuple(tuple(row) for row in value.get("matrix", ()))
            == identity_matrix
            and tuple(value.get("translation", ())) == zero
        ),
        None,
    )
    if position is None:
        raise RuntimeError("GAP catalogue stabilizer lacks the identity")
    return (values[position],) + values[:position] + values[position + 1 :]


def _task5_payload(records: Sequence[object], time_reversal: bool) -> dict[str, object]:
    unique: dict[str, object] = {}
    for record in records:
        unique.setdefault(record.wyckoff_id, record)
    ordered = tuple(unique.values())
    first = ordered[0]
    action = {
        "affine_generators": _plain(first.space_group_action["source_generators"]),
        "translation_basis": _plain(first.space_group_action["translation_basis"]),
    }
    members: list[dict[str, object]] = []
    for record in ordered:
        elements = _identity_first(record.stabilizer["embedded_elements"])
        members.append(
            {
                "element_labels": ["1"]
                + [f"g{index}" for index in range(1, len(elements))],
                "finite_group_id": record.wyckoff_id,
                "inclusion": {
                    "inclusion_id": record.wyckoff_id,
                    "literal_elements": _plain(elements),
                },
            }
        )
    return {"action": action, "members": members, "time_reversal": time_reversal}


def _run_direct_task5(
    records: Sequence[object],
    *,
    runtime: GapRuntime,
    repository_root: Path,
    time_reversal: bool,
    timeout_seconds: int,
) -> tuple[PCPNormalForm, tuple[tuple[str, Mapping[str, object]], ...]]:
    root = Path(repository_root).resolve(strict=True)
    payload = _task5_payload(records, time_reversal)
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    program = (
        'if LoadPackage("json", false : OnlyNeeded) <> true then QUIT_GAP(2); fi;\n'
        'if LoadPackage("cryst", false : OnlyNeeded) <> true then QUIT_GAP(2); fi;\n'
        'if LoadPackage("hap", false : OnlyNeeded) <> true then QUIT_GAP(2); fi;\n'
        'if LoadPackage("hapcryst", false : OnlyNeeded) <> true then QUIT_GAP(2); fi;\n'
        + _gap_sources(root)
        + "\n"
        + f"MathPSGDirectInput := JsonStringToGap({json.dumps(encoded)});\n"
        + "MathPSGDirectContext := MathPSGClassifierTask5PrepareLiteralInclusionBatch(MathPSGDirectInput);\n"
        + "MathPSGDirectMembers := [];\n"
        + "for MathPSGDirectMember in MathPSGDirectInput.members do\n"
        + "  MathPSGDirectMemberInput := rec(action := MathPSGDirectInput.action, element_labels := MathPSGDirectMember.element_labels, finite_group_id := MathPSGDirectMember.finite_group_id, inclusion := MathPSGDirectMember.inclusion, time_reversal := MathPSGDirectInput.time_reversal);\n"
        + "  MathPSGDirectRaw := MathPSGClassifierTask5LiteralInclusionMemberRaw(MathPSGDirectMemberInput, MathPSGDirectContext);\n"
        + "  Add(MathPSGDirectMembers, rec(inclusion_id := MathPSGDirectMember.inclusion.inclusion_id, raw_output := MathPSGDirectRaw));\n"
        + "od;\n"
        + "MathPSGDirectOutput := rec(relative_orders := RelativeOrdersOfPcp(MathPSGDirectContext.conversion.pcp), members := MathPSGDirectMembers);\n"
        + "if FileString({output_path}, MathPSGClassifierJson(MathPSGDirectOutput)) = fail then QUIT_GAP(2); fi;\n"
        + "QUIT_GAP(0);\n"
    )
    envelope = _run_gap_program(
        program, runtime=runtime, root=root, timeout_seconds=timeout_seconds
    )
    try:
        normal_form = PCPNormalForm(tuple(int(item) for item in envelope["relative_orders"]))
        members = tuple(
            (str(item["inclusion_id"]), item["raw_output"])
            for item in envelope["members"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("GAP output lacks required computation data") from error
    return normal_form, members  # type: ignore[return-value]


def _build_one_direct_ambient(
    records: Sequence[object],
    *,
    runtime: GapRuntime,
    repository_root: Path,
    time_reversal: bool,
    timeout_seconds: int,
    spatial: DirectHostAmbient | None,
) -> DirectHostAmbient:
    normal_form, raw_members = _run_direct_task5(
        records,
        runtime=runtime,
        repository_root=repository_root,
        time_reversal=time_reversal,
        timeout_seconds=timeout_seconds,
    )
    target_group_id = "ambient+onsite-T" if time_reversal else "ambient"
    inclusions = tuple(
        parse_inclusion(
            raw,
            inclusion_id=inclusion_id,
            target_group_id=target_group_id,
            pcp_normal_form=normal_form,
            path=f"$gap.members[{index}]",
        )
        for index, (inclusion_id, raw) in enumerate(raw_members)
    )
    return DirectHostAmbient(
        inclusions[0].target_resolution,
        inclusions,
        time_reversal,
        spatial,
    )


def build_direct_host_ambient(
    records: Sequence[object],
    *,
    runtime: GapRuntime,
    time_reversal: bool,
    timeout_seconds: int,
    repository_root: Path,
) -> DirectHostAmbient:
    spatial = None
    if time_reversal:
        spatial = _build_one_direct_ambient(
            records,
            runtime=runtime,
            repository_root=repository_root,
            time_reversal=False,
            timeout_seconds=timeout_seconds,
            spatial=None,
        )
    return _build_one_direct_ambient(
        records,
        runtime=runtime,
        repository_root=repository_root,
        time_reversal=time_reversal,
        timeout_seconds=timeout_seconds,
        spatial=spatial,
    )


def direct_character_context(
    ambient: DirectHostAmbient,
) -> tuple[tuple[GF2Character, ...], GF2Character]:
    spatial = ambient.spatial if ambient.time_reversal else ambient
    if spatial is None:
        raise RuntimeError("time-reversal computation lacks its spatial group")
    spatial_characters = enumerate_characters(spatial.resolution)
    if ambient.time_reversal:
        characters = tuple(
            GF2Character(character.bits + (time_bit,))
            for character in spatial_characters
            for time_bit in (0, 1)
        )
        grade = GF2Character((0,) * len(spatial_characters[0].bits) + (1,))
    else:
        characters = spatial_characters
        grade = GF2Character((0,) * len(spatial_characters[0].bits))
    return characters, grade


def _spatial_source_table(table):
    if not table.group_id.endswith("+onsite-T"):
        return table
    size = len(table.element_order) // 2
    return type(table)(
        table.group_id.removesuffix("+onsite-T"),
        table.element_order[:size],
        0,
        tuple(tuple(row[:size]) for row in table.multiplication_table[:size]),
    )


def _u1_skeleton(table, grade: GF2Character, rho: GF2Character) -> DirectU1LocalSkeleton:
    grade_values = tuple(grade.bits)
    rho_values = tuple(rho.bits)
    q_values = tuple(left ^ right for left, right in zip(grade_values, rho_values))
    defect = tuple(
        tuple(
            Phase(Fraction(q_values[left] * q_values[right] + grade_values[left] * q_values[right], 2))
            for right in range(len(table.element_order))
        )
        for left in range(len(table.element_order))
    )
    return DirectU1LocalSkeleton(
        table.element_order, grade_values, rho_values, defect
    )


def enumerate_direct_local_branches(
    request,
    resolved_orbit,
    ambient: DirectHostAmbient,
) -> tuple[Z2LocalSkeleton | DirectU1LocalSkeleton, ...]:
    row = ambient.inclusion_for(resolved_orbit.record.wyckoff_id)
    table = row.source_resolution.finite_group
    if table is None:
        raise RuntimeError("GAP local computation lacks a finite group")
    if request.igg == "Z2":
        spatial_skeletons = enumerate_spatial_z2_skeletons(
            _spatial_source_table(table)
        )
        skeletons = (
            tuple(
                child
                for skeleton in spatial_skeletons
                for child in enumerate_graded_z2_skeletons(skeleton)
            )
            if request.time_reversal
            else spatial_skeletons
        )
        return skeletons
    basis, grade = direct_character_context(ambient)
    skeletons: list[DirectU1LocalSkeleton] = []
    for rho in basis:
        local_grade = GF2Character(
            tuple(
                word_character(ambient.resolution, grade.bits, image)
                for image in row.source_element_images
            )
        )
        local_rho = GF2Character(
            tuple(
                word_character(ambient.resolution, rho.bits, image)
                for image in row.source_element_images
            )
        )
        skeletons.append(_u1_skeleton(table, local_grade, local_rho))
    return tuple(skeletons)


def resolve_direct_inclusions(
    resolved_orbits: Sequence[object],
    ambient: DirectHostAmbient,
) -> tuple[DirectInclusion, ...]:
    return tuple(
        ambient.inclusion_for(item.record.wyckoff_id)
        for item in resolved_orbits
    )


__all__ = [
    "DirectHostAmbient",
    "DirectInclusion",
    "build_direct_host_ambient",
    "direct_character_context",
    "enumerate_direct_local_branches",
    "resolve_direct_inclusions",
]
