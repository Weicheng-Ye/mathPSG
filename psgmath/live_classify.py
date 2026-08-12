"""Public request construction for host-native joint PSG classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

from .catalogue_loader import CatalogueIndex
from .classification_schema import (
    FrozenJSONObject,
    canonical_classification_json,
    loads_classification_record,
)
from .classification_schema import (
    SCHEMA_VERSION,
    ClassificationRequest,
    OrbitInstance,
)
from .certified_classifier import classify_request
from .classifier_cache import ClassifierCache
from .host_classifier_backend import HostNativeClassifierBackend
from .live_catalogue import CatalogueError, LiveCatalogue
from .local_gap import host_provenance, probe_gap
from .query import make_diagnostic_verified_catalogue


_WP_RE = re.compile(r"(?:[1-9][0-9]*)?[A-Za-z]\Z")
_SETTING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_RUNTIME_ROOT = Path(__file__).resolve().parent / "_assets"


class ClassificationError(RuntimeError):
    """A complete host-native PSG classification could not be produced."""


@dataclass(frozen=True, slots=True)
class HostRuntimeProvenance:
    certification_status: str
    gap_version: str
    gap_packages: tuple[tuple[str, str], ...]
    gap_executable_sha256: str
    python_version: str
    package_version: str
    source_inventory_digest: str


@dataclass(frozen=True, slots=True)
class HostNativeClassificationResult:
    """Capability-free immutable result of one joint host-native calculation."""

    request: FrozenJSONObject
    class_count: int | None
    continuous: bool
    summaries: tuple[FrozenJSONObject, ...]
    details: FrozenJSONObject | None
    certification_status: str
    runtime: HostRuntimeProvenance


def _default_cache_root() -> Path:
    override = os.environ.get("MATHPSG_CACHE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mathpsg-standalone"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (
        Path(xdg).expanduser() / "mathpsg-standalone"
        if xdg
        else Path.home() / ".cache" / "mathpsg-standalone"
    )


def _frozen_mapping(value: Mapping[str, object]) -> FrozenJSONObject:
    return FrozenJSONObject(tuple(value.items()))


def _runtime_provenance(runtime) -> HostRuntimeProvenance:
    value = host_provenance(runtime)
    gap = value["gap"]
    package = value["package"]
    python = value["python"]
    return HostRuntimeProvenance(
        certification_status="host-native",
        gap_version=str(gap["version"]),
        gap_packages=tuple(sorted(dict(gap["packages"]).items())),
        gap_executable_sha256=str(gap["executable_sha256"]),
        python_version=str(python["version"]),
        package_version=str(package["version"]),
        source_inventory_digest=str(value["source_inventory_digest"]),
    )


def _public_result(value, request, runtime, *, details: bool):
    if value.record.layer.status != "complete" or value.unframed_quotient is None:
        failures = value.record.layer.failures
        reason = (
            "; ".join(f"{item.stage}: {item.message}" for item in failures)
            or "classification did not produce a complete quotient"
        )
        raise ClassificationError(reason)
    quotient = value.unframed_quotient
    continuous = bool(quotient.continuous_orbit_presentations)
    count = quotient.unframed_finite_cardinality
    if continuous != (count is None):
        raise ClassificationError("finite/continuous quotient claims are inconsistent")
    request_mapping = json.loads(canonical_classification_json(request))
    summaries = tuple(
        _frozen_mapping(
            {
                "kind": stratum["kind"],
                "skeleton_ids": list(stratum["skeleton_ids"]),
                "stratum_id": stratum["stratum_id"],
            }
        )
        for stratum in value.record.layer.framed_strata
    )
    detail_value = None
    if details:
        replayed = loads_classification_record(
            canonical_classification_json(value.record)
        )
        detail_value = _frozen_mapping(
            json.loads(canonical_classification_json(replayed))
        )
    return HostNativeClassificationResult(
        request=_frozen_mapping(request_mapping),
        class_count=count,
        continuous=continuous,
        summaries=summaries,
        details=detail_value,
        certification_status="host-native",
        runtime=_runtime_provenance(runtime),
    )


def _occupied_wps(wps: str | Sequence[str]) -> tuple[str, ...]:
    if type(wps) is str:
        values = (wps,)
    elif isinstance(wps, Sequence) and not isinstance(wps, (str, bytes)):
        values = tuple(wps)
    else:
        raise TypeError("wps must be a Wyckoff label or a finite sequence of labels")
    if not values:
        raise ValueError("wps must contain at least one occupied Wyckoff position")
    if any(type(value) is not str for value in values):
        raise TypeError("each occupied Wyckoff position must be a string")
    if any(_WP_RE.fullmatch(value) is None for value in values):
        raise ValueError("each occupied Wyckoff position must be a conventional label")
    return values


def resolve_occupancy_request(
    it_number: int,
    wps: str | Sequence[str],
    *,
    igg: str,
    time_reversal: bool,
    setting: str | None,
    catalogue: LiveCatalogue,
) -> ClassificationRequest:
    """Resolve one simultaneous occupied-Wyckoff configuration.

    Each item in ``wps`` becomes one ordered orbit instance. Repeated labels
    intentionally remain distinct instances in the joint request.
    """

    if type(it_number) is not int:
        raise TypeError("IT number must be an integer")
    if not 1 <= it_number <= 230:
        raise ValueError("IT number must be in 1..230")
    labels = _occupied_wps(wps)
    if type(igg) is not str:
        raise TypeError("igg must be a string")
    if igg not in ("Z2", "U1"):
        raise ValueError("igg must be Z2 or U1")
    if type(time_reversal) is not bool:
        raise TypeError("time_reversal must be a boolean")
    if setting is not None:
        if type(setting) is not str:
            raise TypeError("setting must be a string or None")
        if _SETTING_RE.fullmatch(setting) is None:
            raise ValueError("setting has invalid syntax")
    if type(catalogue) is not LiveCatalogue:
        raise TypeError("catalogue must be a LiveCatalogue")

    records = catalogue.records(it_number)
    candidate_settings = (
        (setting,)
        if setting is not None
        else tuple(sorted({str(record.space_group["setting"]) for record in records}))
    )
    matches: list[tuple[str, tuple[object, ...]]] = []
    for candidate_setting in candidate_settings:
        resolved = []
        for label in labels:
            try:
                resolved.append(catalogue.resolve(it_number, label, candidate_setting))
            except CatalogueError:
                break
        if len(resolved) == len(labels):
            matches.append((candidate_setting, tuple(resolved)))

    if not matches:
        raise CatalogueError(
            f"occupied Wyckoff configuration has no match in space group {it_number}"
        )
    if len(matches) != 1:
        settings = ", ".join(value[0] for value in matches)
        raise CatalogueError(
            f"setting is ambiguous for space group {it_number}: {settings}"
        )

    selected_setting, selected_records = matches[0]
    return ClassificationRequest(
        SCHEMA_VERSION,
        it_number,
        selected_setting,
        igg,
        time_reversal,
        tuple(
            OrbitInstance(
                f"atom-{index:04d}",
                record.wyckoff_id,
                "family",
            )
            for index, record in enumerate(selected_records)
        ),
    )


def classify(
    it_number: int,
    wps: str | Sequence[str],
    *,
    igg: str = "Z2",
    time_reversal: bool = False,
    setting: str | None = None,
    details: bool = False,
    gap: str = "gap",
    cache: str | os.PathLike[str] | None = None,
    timeout: int = 300,
) -> HostNativeClassificationResult:
    """Calculate one joint PSG classification with the exact local GAP runtime."""

    if type(it_number) is not int:
        raise TypeError("IT number must be an integer")
    if not 1 <= it_number <= 230:
        raise ValueError("IT number must be in 1..230")
    _occupied_wps(wps)
    if type(igg) is not str:
        raise TypeError("igg must be a string")
    if igg not in ("Z2", "U1"):
        raise ValueError("igg must be Z2 or U1")
    if type(time_reversal) is not bool:
        raise TypeError("time_reversal must be a boolean")
    if setting is not None:
        if type(setting) is not str:
            raise TypeError("setting must be a string or None")
        if _SETTING_RE.fullmatch(setting) is None:
            raise ValueError("setting has invalid syntax")
    if type(details) is not bool:
        raise TypeError("details must be a boolean")
    if type(gap) is not str or not gap:
        raise TypeError("gap must be a nonempty executable name or path")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    cache_root = _default_cache_root() if cache is None else Path(cache).expanduser()
    runtime = probe_gap(gap, timeout_seconds=min(timeout, 30))
    catalogue = LiveCatalogue(
        runtime,
        cache_root=cache_root / "catalogue",
        repository_root=_RUNTIME_ROOT,
        timeout_seconds=min(timeout, 120),
    )
    request = resolve_occupancy_request(
        it_number,
        wps,
        igg=igg,
        time_reversal=time_reversal,
        setting=setting,
        catalogue=catalogue,
    )
    backend = HostNativeClassifierBackend(
        runtime=runtime,
        repository_root=_RUNTIME_ROOT,
    )
    verified = make_diagnostic_verified_catalogue(
        CatalogueIndex(catalogue.records(it_number)),
        backend=backend,
    )
    value = classify_request(
        request,
        verified,
        cache=ClassifierCache(cache_root / "classifier"),
        timeout_seconds=timeout,
    )
    return _public_result(value, request, runtime, details=details)


__all__ = [
    "ClassificationError",
    "HostNativeClassificationResult",
    "HostRuntimeProvenance",
    "classify",
    "resolve_occupancy_request",
]
