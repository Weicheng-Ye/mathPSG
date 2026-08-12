"""On-demand exact crystallographic records from a local GAP/Cryst runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping

from .catalogue import catalogue_record_order_key, normalize_gap_export
from .catalogue_schema import (
    CatalogueRecord,
    DisplayRecord,
    canonical_json,
    parse_catalogue_record,
    parse_display_record,
    strict_json_loads,
)
from .local_gap import GapRuntime, host_provenance, source_inventory_digest


_LABEL_RE = re.compile(r"(?:[1-9][0-9]*)?[A-Za-z]\Z")
_MAX_CAPTURE_BYTES = 16 * 1024 * 1024


class CatalogueError(RuntimeError):
    """Live geometry, display metadata, or cache validation failed."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return canonical_json(value) + b"\n"


@dataclass(frozen=True, slots=True)
class _DisplayIndex:
    by_id: Mapping[str, DisplayRecord]
    source_sha256: str


class LiveCatalogue:
    """Generate one IT-number catalogue at a time and cache exact records."""

    def __init__(
        self,
        runtime: GapRuntime,
        *,
        cache_root: Path,
        repository_root: Path,
        timeout_seconds: int = 120,
    ) -> None:
        if type(runtime) is not GapRuntime:
            raise TypeError("LiveCatalogue requires GapRuntime")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("catalogue timeout must be a positive integer")
        root = Path(repository_root).resolve(strict=True)
        cache = Path(cache_root).resolve()
        exporter = root / "gap" / "catalogue" / "export_one.g"
        normalizer = root / "gap" / "catalogue" / "lib" / "normalize_affine.g"
        crosswalk = root / "resources" / "display-crosswalk.ndjson"
        for path, label in (
            (exporter, "catalogue exporter"),
            (normalizer, "catalogue normalizer"),
            (crosswalk, "display crosswalk"),
        ):
            if not path.is_file():
                raise CatalogueError(f"{label} is unavailable")
        cache.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.cache_root = cache
        self.repository_root = root
        self.timeout_seconds = timeout_seconds
        self.exporter = exporter
        self.normalizer = normalizer
        self.crosswalk = crosswalk
        self._display = self._load_display_index(crosswalk)
        self._memory: dict[int, tuple[CatalogueRecord, ...]] = {}

    @staticmethod
    def _load_display_index(path: Path) -> _DisplayIndex:
        data = path.read_bytes()
        if not data.endswith(b"\n") or b"\r" in data:
            raise CatalogueError("display crosswalk is not canonical NDJSON")
        by_id: dict[str, DisplayRecord] = {}
        for line_number, line in enumerate(data.splitlines(), start=1):
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise CatalogueError(
                    f"display crosswalk row {line_number} is not an object"
                )
            display = parse_display_record(value)
            if canonical_json(display) != line or display.wyckoff_id in by_id:
                raise CatalogueError(
                    f"display crosswalk row {line_number} is not canonical and unique"
                )
            by_id[display.wyckoff_id] = display
        return _DisplayIndex(MappingProxyType(by_id), _sha256(data))

    def _cache_key(self, it_number: int) -> str:
        value = {
            "crosswalk_sha256": self._display.source_sha256,
            "exporter_sha256": _sha256(self.exporter.read_bytes()),
            "it_number": it_number,
            "normalizer_sha256": _sha256(self.normalizer.read_bytes()),
            "runtime": host_provenance(self.runtime),
            "source_inventory_digest": source_inventory_digest(),
        }
        return _sha256(canonical_json(value)).removeprefix("sha256:")

    def _cache_directory(self, it_number: int) -> Path:
        return self.cache_root / "catalogue" / f"sg{it_number}-{self._cache_key(it_number)}"

    def _read_cached(self, directory: Path, it_number: int) -> tuple[CatalogueRecord, ...]:
        metadata_path = directory / "record.json"
        geometry_path = directory / "wyckoff.ndjson"
        try:
            metadata_data = metadata_path.read_bytes()
            geometry_data = geometry_path.read_bytes()
            metadata = strict_json_loads(metadata_data.rstrip(b"\n"))
        except (OSError, TypeError, ValueError, UnicodeError) as error:
            raise CatalogueError("cached catalogue is malformed") from error
        if (
            not isinstance(metadata, dict)
            or metadata_data != _canonical_bytes(metadata)
            or metadata.get("certification_status") != "host-native"
            or metadata.get("it_number") != it_number
            or metadata.get("geometry_sha256") != _sha256(geometry_data)
            or not geometry_data.endswith(b"\n")
        ):
            raise CatalogueError("cached catalogue bindings differ")
        records: list[CatalogueRecord] = []
        for line in geometry_data.splitlines():
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise CatalogueError("cached catalogue row is not an object")
            record = parse_catalogue_record(value)
            if canonical_json(record) != line:
                raise CatalogueError("cached catalogue row is not canonical")
            records.append(record)
        result = tuple(sorted(records, key=catalogue_record_order_key))
        self._require_display_coverage(result)
        return result

    def _require_display_coverage(
        self, records: tuple[CatalogueRecord, ...]
    ) -> tuple[DisplayRecord, ...]:
        try:
            displays = tuple(self._display.by_id[record.wyckoff_id] for record in records)
        except KeyError as error:
            raise CatalogueError("display crosswalk omits live Wyckoff geometry") from error
        if len({item.wyckoff_id for item in displays}) != len(records):
            raise CatalogueError("display crosswalk is not one-to-one")
        return displays

    def _generate(self, it_number: int, directory: Path) -> tuple[CatalogueRecord, ...]:
        parent = directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mathpsg-catalogue-", dir=parent) as raw:
            temporary = Path(raw)
            export_path = temporary / "gap-export.json"
            try:
                completed = subprocess.run(
                    (
                        self.runtime.executable,
                        "-q",
                        os.fspath(self.exporter),
                        "--",
                        "--international-number",
                        str(it_number),
                        "--json-output",
                        os.fspath(export_path),
                    ),
                    cwd=self.repository_root,
                    check=False,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise CatalogueError("local GAP catalogue export failed") from error
            if (
                completed.returncode != 0
                or len(completed.stdout) > _MAX_CAPTURE_BYTES
                or len(completed.stderr) > _MAX_CAPTURE_BYTES
                or not export_path.is_file()
            ):
                raise CatalogueError("local GAP catalogue export failed")
            try:
                exported = strict_json_loads(export_path.read_bytes())
            except (OSError, TypeError, ValueError, UnicodeError) as error:
                raise CatalogueError("local GAP catalogue export is malformed") from error
            if not isinstance(exported, dict):
                raise CatalogueError("local GAP catalogue export is not an object")
            records = normalize_gap_export(exported)
            if not records or any(
                record.space_group["international_number"] != it_number
                for record in records
            ):
                raise CatalogueError("normalized catalogue group coverage differs")
            self._require_display_coverage(records)
            geometry = b"".join(canonical_json(record) + b"\n" for record in records)
            metadata = {
                "certification_status": "host-native",
                "geometry_sha256": _sha256(geometry),
                "it_number": it_number,
                "provenance": host_provenance(self.runtime),
                "record_count": len(records),
                "record_type": "mathpsg-live-catalogue-cache",
                "schema_version": 1,
            }
            (temporary / "wyckoff.ndjson").write_bytes(geometry)
            (temporary / "record.json").write_bytes(_canonical_bytes(metadata))
            if directory.exists():
                return self._read_cached(directory, it_number)
            temporary.rename(directory)
        return self._read_cached(directory, it_number)

    def records(self, it_number: int) -> tuple[CatalogueRecord, ...]:
        if type(it_number) is not int:
            raise TypeError("IT number must be an integer")
        if not 1 <= it_number <= 230:
            raise ValueError("IT number must be in 1..230")
        existing = self._memory.get(it_number)
        if existing is not None:
            return existing
        directory = self._cache_directory(it_number)
        result = (
            self._read_cached(directory, it_number)
            if directory.is_dir()
            else self._generate(it_number, directory)
        )
        self._memory[it_number] = result
        return result

    def resolve(
        self,
        it_number: int,
        label: str,
        setting: str | None = None,
    ) -> CatalogueRecord:
        if type(label) is not str or _LABEL_RE.fullmatch(label) is None:
            raise CatalogueError("Wyckoff label is invalid")
        records = self.records(it_number)
        candidates: list[CatalogueRecord] = []
        for record in records:
            display = self._display.by_id[record.wyckoff_id]
            full = f"{display.conventional_multiplicity}{display.wyckoff_letter}"
            if label not in {display.wyckoff_letter, full}:
                continue
            if setting is not None and str(record.space_group["setting"]) != setting:
                continue
            candidates.append(record)
        if len(candidates) != 1:
            raise CatalogueError("Wyckoff label is missing or ambiguous")
        return candidates[0]


__all__ = ["CatalogueError", "LiveCatalogue"]
