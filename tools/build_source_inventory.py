#!/usr/bin/env python3
"""Build the canonical inventory for the standalone source extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "EXTRACTED_SOURCES.json"
EXCLUDED_PARTS = {".git", "__pycache__"}
EXCLUDED_FILES = {"EXTRACTED_SOURCES.json"}
COPIED_SOURCES = {
    "psgmath/catalogue_loader.py": (
        "psgmath/catalogue_loader.py",
        "a1b22f9a5248a01e5de1a0b6a53aef0287c8b4c198b073178642796f7022be2e",
    ),
    "psgmath/query.py": (
        "psgmath/query.py",
        "5f88e8e0580f0288d523836be8e1cb855390abfbffcc5461d5995961b70cf9d7",
    ),
    "psgmath/certified_classifier.py": (
        "psgmath/certified_classifier.py",
        "88ca499fe668085049b9be8ae5544ff27c380c8a6f0a3d530eaa59f88276efc1",
    ),
}


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
    try:
        prior_files = json.loads(OUTPUT.read_text(encoding="utf-8"))["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        prior_files = {}
    files: dict[str, dict[str, str | None]] = {}
    for path in inventory_files():
        relative = path.relative_to(ROOT).as_posix()
        standalone_data = path.read_bytes()
        prior = prior_files.get(relative, {})
        copied_source = COPIED_SOURCES.get(relative)
        source_path = (
            copied_source[0]
            if copied_source is not None
            else prior.get("source_path") if isinstance(prior, dict) else None
        )
        source_sha256 = (
            copied_source[1]
            if copied_source is not None
            else prior.get("source_sha256") if isinstance(prior, dict) else None
        )
        files[relative] = {
            "source_path": source_path,
            "source_sha256": source_sha256,
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
