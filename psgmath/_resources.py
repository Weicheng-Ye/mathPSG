"""Read the fixed set of public resources shipped with :mod:`psgmath`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
import json
from pathlib import Path
from typing import Any


ASSET_PATHS = (
    "data/stabilizers/v1/manifest.json",
    "data/stabilizers/v1/types.ndjson",
    "gap/classifier/lib/affine_pcp.g",
    "gap/classifier/lib/bar_equivalence.g",
    "gap/classifier/lib/characters.g",
    "gap/classifier/lib/protocol.g",
    "gap/classifier/lib/resolutions.g",
    "gap/classifier/lib/restrictions.g",
    "gap/classifier/lib/u1_relative.g",
)

_ASSET_PATH_SET = frozenset(ASSET_PATHS)
_RESOURCE_DIRECTORY = "_assets"
_MANIFEST_NAME = "manifest.json"


def _resource(relative_path: str):
    if type(relative_path) is not str or relative_path not in _ASSET_PATH_SET:
        raise ValueError(f"resource is not in the psgmath asset allowlist: {relative_path!r}")
    item = resources.files("psgmath").joinpath(_RESOURCE_DIRECTORY)
    for component in relative_path.split("/"):
        item = item.joinpath(component)
    return item


def asset_bytes(relative_path: str) -> bytes:
    """Return one allowlisted packaged asset as bytes."""

    return _resource(relative_path).read_bytes()


def asset_text(relative_path: str) -> str:
    """Return one allowlisted UTF-8 packaged asset as text."""

    return _resource(relative_path).read_text(encoding="utf-8")


def _release_resource(relative_path: str):
    if type(relative_path) is not str:
        raise ValueError("release resource path must be an exact string")
    components = relative_path.split("/")
    if (
        len(components) < 3
        or components[:2] != ["release", "task5"]
        or any(component in {"", ".", ".."} for component in components)
        or any("\\" in component or "\x00" in component for component in components)
    ):
        raise ValueError(f"invalid Task 5 release resource path: {relative_path!r}")
    item = resources.files("psgmath").joinpath(_RESOURCE_DIRECTORY)
    for component in components:
        item = item.joinpath(component)
        is_symlink = getattr(item, "is_symlink", None)
        if is_symlink is not None and is_symlink():
            raise ValueError("Task 5 release resources cannot contain symlinks")
    return item


def release_asset_bytes(relative_path: str) -> bytes:
    """Read one canonical path inside the separately signed release subtree."""

    item = _release_resource(relative_path)
    try:
        if not item.is_file():
            raise ValueError("Task 5 release resource is not a regular file")
        return item.read_bytes()
    except OSError as error:
        raise ValueError("Task 5 release resource is unavailable") from error


def release_asset_paths(relative_directory: str) -> tuple[str, ...]:
    """Return a deterministic recursive inventory below one release directory."""

    root = _release_resource(relative_directory)
    try:
        if not root.is_dir():
            raise ValueError("Task 5 release resource inventory is not a directory")
        rows: list[str] = []

        def visit(item, prefix: str) -> None:
            for child in sorted(item.iterdir(), key=lambda value: value.name):
                name = child.name
                if (
                    type(name) is not str
                    or name in {"", ".", ".."}
                    or "/" in name
                    or "\\" in name
                    or "\x00" in name
                ):
                    raise ValueError(
                        "Task 5 release resource has a noncanonical name"
                    )
                path = f"{prefix}/{name}"
                is_symlink = getattr(child, "is_symlink", None)
                if is_symlink is not None and is_symlink():
                    raise ValueError(
                        "Task 5 release resources cannot contain symlinks"
                    )
                if child.is_dir():
                    visit(child, path)
                elif child.is_file():
                    rows.append(path)
                else:
                    raise ValueError(
                        "Task 5 release resource is not a regular file"
                    )

        visit(root, relative_directory)
        return tuple(rows)
    except OSError as error:
        raise ValueError("Task 5 release resource inventory is unavailable") from error


@contextmanager
def as_asset_file(relative_path: str) -> Iterator[Path]:
    """Yield a filesystem path for one allowlisted asset, extracting if needed."""

    with resources.as_file(_resource(relative_path)) as materialized:
        yield Path(materialized)


def asset_manifest() -> dict[str, Any]:
    """Return a fresh decode of the packaged asset-integrity manifest."""

    manifest = resources.files("psgmath").joinpath(
        _RESOURCE_DIRECTORY, _MANIFEST_NAME
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("packaged asset manifest must be a JSON object")
    return value
