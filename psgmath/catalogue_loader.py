"""Strict dependency-free loading and lookup for canonical catalogue NDJSON."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .catalogue import catalogue_record_order_key, validate_catalogue_record_identity
from .catalogue_schema import (
    CatalogueRecord,
    DisplayRecord,
    canonical_json,
    parse_catalogue_record,
    parse_display_record,
    strict_json_loads,
    validate_manifest,
)


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RELEASE_FILE_ORDER = ("geometry", "display", "coverage", "index", "provenance")


def _safe_is_file(path: Path, context: str) -> bool:
    try:
        return path.is_file()
    except OSError:
        raise ValueError(f"{context}: unable to inspect catalogue file") from None


def _safe_read_bytes(path: Path, context: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise ValueError(f"{context}: unable to read catalogue file") from None


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _verify_release_semantic_index(
    index_data: bytes,
    *,
    geometry_data: bytes,
    display_data: bytes,
    geometry_records: Sequence[CatalogueRecord],
    display_records: Sequence[DisplayRecord],
    expected_source_generation_digest: str | None,
) -> dict[str, Any]:
    """Strictly recompute every v1 semantic-index byte from release rows."""

    try:
        value = strict_json_loads(index_data)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("catalogue semantic index is not strict JSON") from None
    if not isinstance(value, dict):
        raise TypeError("catalogue semantic index must be a JSON object")
    if canonical_json(value) + b"\n" != index_data:
        raise ValueError("catalogue semantic index is not canonical JSON")

    geometry: dict[str, CatalogueRecord] = {}
    for record in geometry_records:
        if not isinstance(record, CatalogueRecord):
            raise TypeError("catalogue semantic index geometry records are malformed")
        if record.wyckoff_id in geometry:
            raise ValueError("catalogue semantic index geometry IDs are not unique")
        geometry[record.wyckoff_id] = record
    display: dict[str, DisplayRecord] = {}
    for record in display_records:
        if not isinstance(record, DisplayRecord):
            raise TypeError("catalogue semantic index display records are malformed")
        if record.wyckoff_id in display:
            raise ValueError("catalogue semantic index display IDs are not unique")
        display[record.wyckoff_id] = record
    if len(geometry) != 1731 or set(geometry) != set(display):
        raise ValueError("catalogue semantic index requires all 1,731 geometry/display pairs")

    source_digest = expected_source_generation_digest
    if source_digest is None:
        bindings = value.get("bindings")
        if isinstance(bindings, dict):
            candidate = bindings.get("source_generation_digest")
            if isinstance(candidate, str):
                source_digest = candidate
    if not isinstance(source_digest, str) or _DIGEST_RE.fullmatch(source_digest) is None:
        raise ValueError("catalogue semantic index source generation binding is invalid")

    raise ValueError(
        "standalone release catalogue loading is unavailable; "
        "use LiveCatalogue for host-native records"
    )
    pinned_display_source_digest = ""
    if any(
        record.independent_verification["status"] != "verified"
        or record.independent_verification["source_digest"]
        != pinned_display_source_digest
        for record in display.values()
    ):
        raise ValueError(
            "catalogue semantic index display source differs from pinned authority"
        )
    expected = {
        "bindings": {
            "display_sha256": _sha256(display_data),
            "geometry_sha256": _sha256(geometry_data),
            "pinned_display_source_digest": pinned_display_source_digest,
            "source_generation_digest": source_digest,
        },
        "entries": [
            {
                "conventional_multiplicity": display[wyckoff_id].conventional_multiplicity,
                "display_letter": display[wyckoff_id].wyckoff_letter,
                "international_number": geometry[wyckoff_id].space_group[
                    "international_number"
                ],
                "setting": geometry[wyckoff_id].space_group["setting"],
                "source_embedding_digest": geometry[wyckoff_id].embedding_digest,
                "wyckoff_id": wyckoff_id,
            }
            for wyckoff_id in sorted(geometry)
        ],
        "record_type": "catalogue-semantic-index",
        "schema_version": 1,
    }
    if canonical_json(value) != canonical_json(expected):
        raise ValueError("catalogue semantic index differs from exact release semantics")
    return value


def _validate_release_crosswalk(
    source: Path,
    geometry_data: bytes,
    geometry_records: Sequence[CatalogueRecord],
    *,
    expected_source_generation_digest: str | None,
) -> None:
    manifest_path = source.with_name("manifest.json")
    if not _safe_is_file(manifest_path, "catalogue manifest"):
        return
    raise ValueError(
        "standalone release catalogue loading is unavailable; "
        "use LiveCatalogue for host-native records"
    )
    manifest_value = strict_json_loads(
        _safe_read_bytes(manifest_path, "catalogue manifest")
    )
    if not isinstance(manifest_value, dict):
        raise TypeError("catalogue manifest must be a JSON object")
    manifest = validate_manifest(manifest_value)
    geometry_ids = {record.wyckoff_id for record in geometry_records}
    geometry_rows = [row for row in manifest.files if row["kind"] == "geometry"]
    if len(geometry_rows) == 1 and geometry_rows[0]["sha256"] != _sha256(geometry_data):
        raise ValueError("catalogue geometry SHA-256 disagrees with manifest")
    display_rows = [row for row in manifest.files if row["kind"] == "display"]
    if not display_rows:
        if not manifest.status["release_complete"]:
            return
        raise ValueError("release-complete manifest requires one display metadata file")
    if len(display_rows) != 1:
        raise ValueError("catalogue manifest requires at most one display metadata file")
    display_path = source.with_name(Path(str(display_rows[0]["path"])).name)
    if not _safe_is_file(display_path, "display metadata"):
        raise ValueError("catalogue display metadata file is missing")
    data = _safe_read_bytes(display_path, "display metadata")
    if display_rows[0]["sha256"] != _sha256(data):
        raise ValueError("catalogue display SHA-256 disagrees with manifest")
    if not manifest.status["release_complete"]:
        return
    if tuple(row["kind"] for row in manifest.files) != _RELEASE_FILE_ORDER:
        raise ValueError("release-complete manifest file order is not canonical")
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("display metadata: requires LF-only canonical NDJSON")
    display_ids: set[str] = set()
    display_records: list[DisplayRecord] = []
    previous_id: str | None = None
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise ValueError(f"display metadata:{line_number}: blank row is forbidden")
        decoded = strict_json_loads(line)
        if not isinstance(decoded, dict):
            raise TypeError(f"display metadata:{line_number}: row must be an object")
        display = parse_display_record(decoded)
        if canonical_json(display) != line:
            raise ValueError(f"display metadata:{line_number}: row is not canonical JSON")
        if display.wyckoff_id in display_ids:
            raise ValueError(
                f"display metadata:{line_number}: duplicate wyckoff_id {display.wyckoff_id}"
            )
        if previous_id is not None and display.wyckoff_id <= previous_id:
            raise ValueError(
                f"display metadata:{line_number}: noncanonical display ordering"
            )
        if display.independent_verification["status"] != "verified":
            raise ValueError(
                f"display metadata:{line_number}: release row is not independently verified"
            )
        display_ids.add(display.wyckoff_id)
        display_records.append(display)
        previous_id = display.wyckoff_id
    if display_ids != geometry_ids:
        raise ValueError("release-complete display metadata does not cover geometry IDs exactly")
    expected_geometry_rows = manifest.counts["geometry_rows"]
    expected_display_rows = manifest.counts["display_rows"]
    if len(geometry_ids) != expected_geometry_rows or len(display_ids) != expected_display_rows:
        raise ValueError("release-complete manifest row counts disagree with loaded catalogue")
    index_rows = [row for row in manifest.files if row["kind"] == "index"]
    if len(index_rows) != 1:
        raise ValueError("release-complete manifest requires one semantic index")
    index_path = source.with_name(Path(str(index_rows[0]["path"])).name)
    if not _safe_is_file(index_path, "catalogue semantic index"):
        raise ValueError("catalogue semantic index file is missing")
    index_data = _safe_read_bytes(index_path, "catalogue semantic index")
    if index_rows[0]["sha256"] != _sha256(index_data):
        raise ValueError("catalogue semantic index SHA-256 disagrees with manifest")
    _verify_release_semantic_index(
        index_data,
        geometry_data=geometry_data,
        display_data=data,
        geometry_records=geometry_records,
        display_records=display_records,
        expected_source_generation_digest=expected_source_generation_digest,
    )


def load_ndjson(
    path: str | Path,
    *,
    expected_source_generation_digest: str | None = None,
) -> Iterator[CatalogueRecord]:
    """Yield immutable catalogue records from canonical newline-delimited JSON."""

    source = Path(path)
    data = _safe_read_bytes(source, "catalogue geometry")
    records: list[CatalogueRecord] = []
    if not data:
        _validate_release_crosswalk(
            source,
            data,
            (),
            expected_source_generation_digest=expected_source_generation_digest,
        )
        return iter(())
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("catalogue: trailing bytes or missing final NDJSON newline")
    previous_key: tuple[int, bytes] | None = None
    seen_ids: set[str] = set()
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise ValueError(f"catalogue:{line_number}: blank NDJSON row is forbidden")
        decoded = strict_json_loads(line)
        if not isinstance(decoded, dict):
            raise TypeError(f"catalogue:{line_number}: catalogue row must be a JSON object")
        record = validate_catalogue_record_identity(parse_catalogue_record(decoded))
        encoded = canonical_json(record)
        if encoded != line:
            raise ValueError(f"catalogue:{line_number}: row is not canonical JSON")
        ordering_key = catalogue_record_order_key(record)
        if record.wyckoff_id in seen_ids:
            raise ValueError(f"catalogue:{line_number}: duplicate wyckoff_id {record.wyckoff_id}")
        if previous_key is not None and ordering_key <= previous_key:
            raise ValueError(f"catalogue:{line_number}: noncanonical catalogue ordering")
        previous_key = ordering_key
        seen_ids.add(record.wyckoff_id)
        records.append(record)
    _validate_release_crosswalk(
        source,
        data,
        records,
        expected_source_generation_digest=expected_source_generation_digest,
    )
    return iter(tuple(records))


class CatalogueIndex:
    """Immutable exact lookup index over canonical catalogue records."""

    __slots__ = ("_records", "_by_key")

    def __init__(self, records: Iterable[CatalogueRecord]) -> None:
        materialized = tuple(records)
        by_key: dict[tuple[int, str], CatalogueRecord] = {}
        ids: dict[str, CatalogueRecord] = {}
        for index, record in enumerate(materialized):
            if not isinstance(record, CatalogueRecord):
                raise TypeError(f"records[{index}]: expected CatalogueRecord")
            validated = validate_catalogue_record_identity(record)
            group = validated.space_group["international_number"]
            if isinstance(group, bool) or not isinstance(group, int):
                raise TypeError(f"records[{index}]: invalid international_number")
            previous = ids.get(record.wyckoff_id)
            if previous is not None:
                raise ValueError(f"duplicate wyckoff_id {record.wyckoff_id}")
            key = (group, record.wyckoff_id)
            by_key[key] = record
            ids[record.wyckoff_id] = record
        self._records = materialized
        self._by_key = MappingProxyType(by_key)

    def find(self, space_group: int, wyckoff_id: str) -> CatalogueRecord:
        """Return one exact row or raise ``KeyError`` when group/ID do not pair."""

        if isinstance(space_group, bool) or not isinstance(space_group, int):
            raise TypeError("space_group must be an integer")
        if not 1 <= space_group <= 230:
            raise ValueError("space_group must be in 1..230")
        if not isinstance(wyckoff_id, str):
            raise TypeError("wyckoff_id must be a string")
        return self._by_key[(space_group, wyckoff_id)]

    def __iter__(self) -> Iterator[CatalogueRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["CatalogueIndex", "load_ndjson"]
