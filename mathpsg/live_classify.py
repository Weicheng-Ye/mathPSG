"""The public, cache-free PSG classification function."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .compute import (
    PhysicalClassification,
    U1PhysicalStratum,
    Z2PhysicalStratum,
    compute_classification,
)
from .live_catalogue import CatalogueError, LiveCatalogue
from .local_gap import GapRuntimeError, probe_gap


_RUNTIME_ROOT = Path(__file__).resolve().parent / "_assets"


class ClassificationError(RuntimeError):
    """The requested physical PSG classification could not be completed."""


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Immutable physical result returned by :func:`classify`."""

    request: Mapping[str, object]
    class_count: int | None
    continuous: bool
    summaries: tuple[Mapping[str, object], ...]
    details: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _Request:
    igg: str
    time_reversal: bool


@dataclass(frozen=True, slots=True)
class _ResolvedOrbit:
    record: object


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _labels(wps: object) -> tuple[str, ...]:
    if isinstance(wps, str):
        values = (wps,)
    else:
        try:
            values = tuple(wps)  # type: ignore[arg-type]
        except TypeError:
            values = (wps,)
    labels = tuple(str(value).strip().lower() for value in values)
    if not labels or any(not label for label in labels):
        raise ClassificationError("at least one Wyckoff position is required")
    return labels


def _resolve(
    it_number: object,
    labels: tuple[str, ...],
    *,
    setting: object | None,
    catalogue: LiveCatalogue,
) -> tuple[int, str, tuple[_ResolvedOrbit, ...]]:
    try:
        number = int(it_number)  # lightweight normalization, not an exact-type gate
    except (TypeError, ValueError) as error:
        raise ClassificationError("space-group number is not an integer") from error
    requested_setting = None if setting is None else str(setting).strip().lower()
    records = catalogue.records(number)
    candidate_settings = (
        (requested_setting,)
        if requested_setting is not None
        else tuple(
            sorted({str(record.space_group["setting"]).lower() for record in records})
        )
    )
    matches: list[tuple[str, tuple[object, ...]]] = []
    for candidate in candidate_settings:
        resolved: list[object] = []
        for label in labels:
            try:
                resolved.append(catalogue.resolve(number, label, candidate))
            except CatalogueError:
                break
        if len(resolved) == len(labels):
            matches.append((candidate, tuple(resolved)))
    if not matches:
        raise ClassificationError(
            f"the occupied Wyckoff positions do not match space group {number}"
        )
    if len(matches) != 1:
        choices = ", ".join(item[0] for item in matches)
        raise ClassificationError(f"space-group setting is ambiguous: {choices}")
    selected_setting, selected_records = matches[0]
    return (
        number,
        selected_setting,
        tuple(
            _ResolvedOrbit(record)
            for record in selected_records
        ),
    )


def _z2_summary(stratum: Z2PhysicalStratum) -> dict[str, object]:
    count = stratum.framed_class_count
    return {
        "kind": "finite-affine-z2",
        "dimension": stratum.quotient_dimension,
        "framed_finite_cardinality": count,
        "unframed_finite_cardinality": stratum.unframed_class_count,
    }


def _u1_summary(stratum: U1PhysicalStratum) -> dict[str, object]:
    group = stratum.solution.group
    result: dict[str, object] = {
        "kind": "compact-u1-torsor",
        "free_rank": group.free_rank,
        "torsion_orders": tuple(group.torsion_orders),
    }
    if group.free_rank == 0:
        result["finite_class_count"] = stratum.framed_class_count
    return result


def _details(
    value: PhysicalClassification,
    summaries: tuple[dict[str, object], ...],
) -> dict[str, object]:
    strata: list[dict[str, object]] = []
    for stratum, summary in zip(value.framed_strata, summaries):
        if isinstance(stratum, Z2PhysicalStratum):
            strata.append(
                {
                    **summary,
                    "basepoint": tuple(stratum.basepoint),
                    "quotient_basis": tuple(
                        tuple(vector) for vector in stratum.quotient_basis
                    ),
                }
            )
        else:
            strata.append(
                {
                    **summary,
                    "rho_bits": tuple(stratum.rho_bits),
                    "basepoint_phases": tuple(
                        str(phase) for phase in stratum.solution.basepoint
                    ),
                    "formal_parameters": tuple(
                        f"phi{index}"
                        for index in range(stratum.solution.group.free_rank)
                    ),
                }
            )
    quotient = value.quotient
    return {
        "strata": tuple(strata),
        "quotient": {
            "framed_finite_cardinality": quotient.framed_class_count,
            "unframed_finite_cardinality": quotient.class_count,
            "continuous_family_count": sum(
                isinstance(stratum, U1PhysicalStratum) and stratum.continuous
                for stratum in value.framed_strata
            ),
        },
    }


def classify(
    it_number,
    wps,
    *,
    igg="Z2",
    time_reversal=False,
    setting=None,
    details=False,
    gap="gap",
    timeout=300,
) -> ClassificationResult:
    """Compute one joint Z2 or U1 PSG classification from fresh GAP output."""

    labels = _labels(wps)
    normalized_igg = str(igg).strip().upper()
    if normalized_igg == "Z2":
        normalized_igg = "Z2"
    elif normalized_igg == "U1":
        normalized_igg = "U1"
    else:
        raise ClassificationError("igg must be Z2 or U1")
    try:
        timeout_seconds = int(timeout)
        if timeout_seconds <= 0:
            raise ValueError
        runtime = probe_gap(str(gap), timeout_seconds=min(timeout_seconds, 30))
        catalogue = LiveCatalogue(
            runtime,
            repository_root=_RUNTIME_ROOT,
            timeout_seconds=min(timeout_seconds, 120),
        )
        number, selected_setting, resolved = _resolve(
            it_number, labels, setting=setting, catalogue=catalogue
        )
        request = _Request(normalized_igg, bool(time_reversal))
        physical = compute_classification(
            request,
            resolved,
            runtime=runtime,
            repository_root=_RUNTIME_ROOT,
            timeout_seconds=timeout_seconds,
        )
    except ClassificationError:
        raise
    except (CatalogueError, GapRuntimeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ClassificationError(str(error)) from error

    summaries = tuple(
        _z2_summary(stratum)
        if isinstance(stratum, Z2PhysicalStratum)
        else _u1_summary(stratum)
        for stratum in physical.framed_strata
    )
    request_mapping = {
        "space_group": number,
        "setting": selected_setting,
        "igg": normalized_igg,
        "time_reversal": bool(time_reversal),
        "wps": labels,
    }
    detail_value = _details(physical, summaries) if bool(details) else None
    return ClassificationResult(
        request=_freeze(request_mapping),  # type: ignore[arg-type]
        class_count=physical.class_count,
        continuous=physical.continuous,
        summaries=tuple(_freeze(summary) for summary in summaries),  # type: ignore[arg-type]
        details=None if detail_value is None else _freeze(detail_value),  # type: ignore[arg-type]
    )


__all__ = ["ClassificationError", "ClassificationResult", "classify"]
