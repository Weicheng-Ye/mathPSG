#!/usr/bin/env python3
"""Build the canonical inventory for the standalone source extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/Users/victor/Downloads/mathPSG/mathPSG")
OUTPUT = ROOT / "EXTRACTED_SOURCES.json"
EXCLUDED_PARTS = {".git", "__pycache__"}
EXCLUDED_FILES = {"EXTRACTED_SOURCES.json"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name not in EXCLUDED_FILES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def main() -> int:
    files: dict[str, dict[str, str | None]] = {}
    for path in inventory_files():
        relative = path.relative_to(ROOT).as_posix()
        standalone_data = path.read_bytes()
        source = SOURCE_ROOT / relative
        source_data = source.read_bytes() if source.is_file() else None
        files[relative] = {
            "source_path": relative if source_data is not None else None,
            "source_sha256": digest(source_data) if source_data is not None else None,
            "standalone_sha256": digest(standalone_data),
        }
    document = {
        "files": files,
        "record_type": "mathpsg-standalone-source-inventory",
        "schema_version": 1,
    }
    OUTPUT.write_bytes(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
