from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any


RESULT_FIELDS = (
    "request",
    "class_count",
    "continuous",
    "summaries",
    "details",
)


def load_physical_cases(path: Path) -> tuple[dict[str, Any], ...]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    return tuple(fixture["cases"])


def result_field_names(result: object) -> tuple[str, ...]:
    if not is_dataclass(result):
        raise AssertionError("classification result must be a frozen dataclass")
    return tuple(field.name for field in fields(result))


def thaw(value: Any) -> Any:
    """Convert immutable public containers to plain JSON-shaped values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: thaw(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [thaw(item) for item in value]
    return value


def canonical_result(result: object) -> dict[str, Any]:
    names = result_field_names(result)
    if names != RESULT_FIELDS:
        raise AssertionError(f"unexpected result fields: {names!r}")
    value = {name: thaw(getattr(result, name)) for name in names}
    value["summaries"] = sorted(value["summaries"], key=_record_key)
    if value["details"] is not None:
        value["details"]["strata"] = sorted(
            value["details"]["strata"], key=_record_key
        )
    return value


def forbidden_public_paths(value: Any, path: str = "result") -> tuple[str, ...]:
    """Locate certificate, replay, hash, cache, or runtime data in a result."""

    forbidden_names = {
        "authority",
        "backend",
        "cache",
        "certification_status",
        "certificate",
        "digest",
        "failure",
        "failures",
        "layer_id",
        "provenance",
        "record_type",
        "replay",
        "request_digest",
        "route",
        "routes",
        "runtime",
        "schema_version",
        "skeleton_ids",
        "source",
        "status",
        "stratum_id",
        "validation",
        "witness",
        "wyckoff_id",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = f"{path}.{key}"
            normalized = str(key).lower()
            if (
                normalized in forbidden_names
                or normalized.endswith("_digest")
                or normalized.endswith("_certificate")
                or normalized.endswith("_id")
                or normalized.endswith("_ids")
            ):
                found.append(key_path)
            found.extend(forbidden_public_paths(item, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_public_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and value.startswith("sha256:"):
        found.append(path)
    return tuple(found)


def _record_key(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
