"""On-demand exact crystallographic records from a local GAP/Cryst runtime."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Mapping

from .catalogue import (
    catalogue_record_order_key,
    normalize_gap_export,
    validate_catalogue_record_identity,
)
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
_MAX_EXPORT_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024


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


@dataclass(frozen=True, slots=True)
class _ActionBindings:
    groups: Mapping[str, Mapping[str, str]]
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
        if (
            cache == root
            or cache.is_relative_to(root)
            or root.is_relative_to(cache)
        ):
            raise CatalogueError("cache must be outside the runtime tree")
        exporter = root / "gap" / "catalogue" / "export_one.g"
        normalizer = root / "gap" / "catalogue" / "lib" / "normalize_affine.g"
        crosswalk = root / "resources" / "display-crosswalk.ndjson"
        action_bindings = root / "resources" / "action-bindings.json"
        for path, label in (
            (exporter, "catalogue exporter"),
            (normalizer, "catalogue normalizer"),
            (crosswalk, "display crosswalk"),
            (action_bindings, "action bindings"),
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
        self._actions = self._load_action_bindings(action_bindings)
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

    @staticmethod
    def _load_action_bindings(path: Path) -> _ActionBindings:
        data = path.read_bytes()
        value = strict_json_loads(data.rstrip(b"\n"))
        if (
            not isinstance(value, dict)
            or data != _canonical_bytes(value)
            or set(value)
            != {
                "groups",
                "record_type",
                "schema_version",
                "source_sha256",
            }
            or value["record_type"] != "mathpsg-standalone-action-bindings"
            or value["schema_version"] != 1
            or not isinstance(value["groups"], dict)
            or set(value["groups"]) != {str(number) for number in range(1, 231)}
        ):
            raise CatalogueError("action bindings are malformed")
        groups: dict[str, Mapping[str, str]] = {}
        expected_fields = {
            "space_group_action_sha256",
        }
        for number, binding in value["groups"].items():
            if (
                not isinstance(binding, dict)
                or set(binding) != expected_fields
                or any(
                    not isinstance(digest, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                    for digest in binding.values()
                )
            ):
                raise CatalogueError(f"action binding for SG {number} is malformed")
            groups[number] = MappingProxyType(dict(binding))
        return _ActionBindings(MappingProxyType(groups), _sha256(data))

    def _cache_key(self, it_number: int) -> str:
        value = {
            "crosswalk_sha256": self._display.source_sha256,
            "action_bindings_sha256": self._actions.source_sha256,
            "exporter_sha256": _sha256(self.exporter.read_bytes()),
            "it_number": it_number,
            "normalizer_sha256": _sha256(self.normalizer.read_bytes()),
            "runtime": host_provenance(self.runtime),
            "source_inventory_digest": source_inventory_digest(),
        }
        return _sha256(canonical_json(value)).removeprefix("sha256:")

    def _cache_directory(self, it_number: int) -> Path:
        return self.cache_root / "catalogue" / f"sg{it_number}-{self._cache_key(it_number)}"

    @contextmanager
    def _cache_lock(self, directory: Path):
        lock_path = directory.parent / f".{directory.name}.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise CatalogueError("catalogue cache lock is unavailable") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CatalogueError("catalogue cache lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _expected_display_ids(self, it_number: int) -> set[str]:
        prefix = f"sg{it_number}:"
        return {
            wyckoff_id
            for wyckoff_id in self._display.by_id
            if wyckoff_id.startswith(prefix)
        }

    def _validate_record_set(
        self, records: tuple[CatalogueRecord, ...], it_number: int
    ) -> tuple[CatalogueRecord, ...]:
        if not records:
            raise CatalogueError("catalogue group coverage is empty")
        if any(
            record.space_group["international_number"] != it_number
            for record in records
        ):
            raise CatalogueError("catalogue cache contains another space group")
        ids = tuple(record.wyckoff_id for record in records)
        if len(set(ids)) != len(ids):
            raise CatalogueError("catalogue cache contains duplicate rows")
        if set(ids) != self._expected_display_ids(it_number):
            raise CatalogueError("catalogue cache crosswalk coverage differs")
        setting_actions = {
            (
                str(record.space_group["setting"]),
                canonical_json(record.space_group_action),
            )
            for record in records
        }
        if len(setting_actions) != 1:
            raise CatalogueError("catalogue cache mixes settings or actions")
        expected_action = self._actions.groups[str(it_number)]
        action_digest = _sha256(canonical_json(records[0].space_group_action))
        if (
            action_digest != expected_action["space_group_action_sha256"]
        ):
            raise CatalogueError("catalogue action provenance binding differs")
        self._require_display_coverage(records)
        return records

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def _wait_for_export(
        self,
        process: subprocess.Popen[bytes],
        *,
        stdout_path: Path,
        stderr_path: Path,
        export_path: Path,
    ) -> int:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            return_code = process.poll()
            if return_code is not None:
                return return_code
            if (
                time.monotonic() >= deadline
                or self._file_size(stdout_path) > _MAX_CAPTURE_BYTES
                or self._file_size(stderr_path) > _MAX_CAPTURE_BYTES
                or self._file_size(export_path) > _MAX_EXPORT_BYTES
            ):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise CatalogueError("local GAP catalogue export exceeded its bounds")
            time.sleep(0.02)

    def _read_cached(self, directory: Path, it_number: int) -> tuple[CatalogueRecord, ...]:
        if directory.is_symlink() or not directory.is_dir():
            raise CatalogueError("cached catalogue path is not a regular directory")
        metadata_path = directory / "record.json"
        geometry_path = directory / "wyckoff.ndjson"
        try:
            if metadata_path.is_symlink() or geometry_path.is_symlink():
                raise CatalogueError("cached catalogue files cannot be symlinks")
            if (
                metadata_path.stat().st_size > _MAX_METADATA_BYTES
                or geometry_path.stat().st_size > _MAX_EXPORT_BYTES
            ):
                raise CatalogueError("cached catalogue exceeds size limits")
            metadata_data = metadata_path.read_bytes()
            geometry_data = geometry_path.read_bytes()
            metadata = strict_json_loads(metadata_data.rstrip(b"\n"))
        except (OSError, TypeError, ValueError, UnicodeError) as error:
            raise CatalogueError("cached catalogue is malformed") from error
        expected_fields = {
            "cache_key",
            "certification_status",
            "geometry_sha256",
            "it_number",
            "provenance",
            "record_count",
            "record_type",
            "schema_version",
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != expected_fields
            or metadata_data != _canonical_bytes(metadata)
            or metadata.get("cache_key") != self._cache_key(it_number)
            or metadata.get("certification_status") != "host-native"
            or metadata.get("it_number") != it_number
            or metadata.get("provenance") != host_provenance(self.runtime)
            or metadata.get("record_type") != "mathpsg-live-catalogue-cache"
            or metadata.get("schema_version") != 1
            or metadata.get("geometry_sha256") != _sha256(geometry_data)
            or not geometry_data.endswith(b"\n")
        ):
            raise CatalogueError("cached catalogue bindings differ")
        records: list[CatalogueRecord] = []
        for line in geometry_data.splitlines():
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise CatalogueError("cached catalogue row is not an object")
            record = validate_catalogue_record_identity(parse_catalogue_record(value))
            if canonical_json(record) != line:
                raise CatalogueError("cached catalogue row is not canonical")
            records.append(record)
        result = tuple(sorted(records, key=catalogue_record_order_key))
        if tuple(records) != result or metadata.get("record_count") != len(result):
            raise CatalogueError("cached catalogue order or count differs")
        return self._validate_record_set(result, it_number)

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
            stdout_path = temporary / "stdout.bin"
            stderr_path = temporary / "stderr.bin"
            try:
                with stdout_path.open("wb") as stdout_file, stderr_path.open(
                    "wb"
                ) as stderr_file:
                    process = subprocess.Popen(
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
                        stdout=stdout_file,
                        stderr=stderr_file,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    return_code = self._wait_for_export(
                        process,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        export_path=export_path,
                    )
            except OSError as error:
                raise CatalogueError("local GAP catalogue export failed") from error
            if (
                return_code != 0
                or stdout_path.stat().st_size > _MAX_CAPTURE_BYTES
                or stderr_path.stat().st_size > _MAX_CAPTURE_BYTES
                or not export_path.is_file()
                or export_path.is_symlink()
                or export_path.stat().st_size > _MAX_EXPORT_BYTES
            ):
                raise CatalogueError("local GAP catalogue export failed")
            try:
                exported = strict_json_loads(export_path.read_bytes())
            except (OSError, TypeError, ValueError, UnicodeError) as error:
                raise CatalogueError("local GAP catalogue export is malformed") from error
            if not isinstance(exported, dict):
                raise CatalogueError("local GAP catalogue export is not an object")
            records = normalize_gap_export(exported)
            records = self._validate_record_set(records, it_number)
            geometry = b"".join(canonical_json(record) + b"\n" for record in records)
            metadata = {
                "cache_key": self._cache_key(it_number),
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
        directory.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(directory):
            if directory.is_symlink():
                raise CatalogueError("cached catalogue path cannot be a symlink")
            if directory.exists() and not directory.is_dir():
                raise CatalogueError("cached catalogue path is not a directory")
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
