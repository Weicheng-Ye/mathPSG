"""Public request construction for host-native joint PSG classification."""

from __future__ import annotations

from collections.abc import Sequence
import re

from .classification_schema import (
    SCHEMA_VERSION,
    ClassificationRequest,
    OrbitInstance,
)
from .live_catalogue import CatalogueError, LiveCatalogue


_WP_RE = re.compile(r"(?:[1-9][0-9]*)?[A-Za-z]\Z")
_SETTING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


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


__all__ = ["resolve_occupancy_request"]
