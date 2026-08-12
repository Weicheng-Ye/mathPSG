"""Host-native GAP discovery and reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import MappingProxyType
from typing import Literal, Mapping


_ROOT = Path(__file__).resolve().parents[1]
_INVENTORY = _ROOT / "EXTRACTED_SOURCES.json"
_REQUIRED_PACKAGES = ("cryst", "hap", "hapcryst", "json", "io")
_DISPLAY_NAMES = {
    "Cryst": "cryst",
    "HAP": "hap",
    "HAPcryst": "hapcryst",
    "json": "json",
    "io": "io",
}
_PROBE = r'''
Print("GAP=", GAPInfo.Version, "\n");
for item in [["cryst","Cryst"],["hap","HAP"],["hapcryst","HAPcryst"],["json","json"],["io","io"]] do
  if LoadPackage(item[1]) <> true then Error(Concatenation(item[1], " unavailable")); fi;
  Print(item[2], "=", PackageInfo(item[1])[1].Version, "\n");
od;
QUIT;
'''


class GapRuntimeError(RuntimeError):
    """The requested local GAP runtime cannot satisfy the standalone contract."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class GapRuntime:
    executable: str
    executable_sha256: str
    gap_version: str
    packages: Mapping[str, str]
    execution_mode: Literal["host-native"] = "host-native"


def parse_gap_probe(stdout: str, *, executable: str) -> GapRuntime:
    """Parse the small line protocol emitted by the local GAP probe."""

    if type(stdout) is not str or type(executable) is not str or not executable:
        raise TypeError("GAP probe output and executable must be strings")
    values: dict[str, str] = {}
    gap_version: str | None = None
    for raw_line in stdout.replace("\r", "").splitlines():
        if "=" not in raw_line:
            continue
        name, value = (part.strip() for part in raw_line.split("=", 1))
        if name == "GAP" and value:
            gap_version = value
        elif name in _DISPLAY_NAMES and value:
            values[_DISPLAY_NAMES[name]] = value
    missing = [name for name in _REQUIRED_PACKAGES if name not in values]
    if gap_version is None or missing:
        absent = (["GAP"] if gap_version is None else []) + missing
        raise GapRuntimeError("missing GAP probe fields: " + ", ".join(absent))
    resolved = Path(executable).resolve(strict=True)
    if not resolved.is_file():
        raise GapRuntimeError("GAP executable is not a regular file")
    return GapRuntime(
        executable=os.fspath(resolved),
        executable_sha256=_sha256(resolved.read_bytes()),
        gap_version=gap_version,
        packages=MappingProxyType(dict(sorted(values.items()))),
    )


def probe_gap(executable: str = "gap", timeout_seconds: int = 30) -> GapRuntime:
    """Run one bounded local GAP process and return exact observed versions."""

    if type(executable) is not str or not executable:
        raise TypeError("GAP executable must be a nonempty string")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("GAP probe timeout must be a positive integer")
    resolved = shutil.which(executable) if os.sep not in executable else executable
    if resolved is None:
        raise GapRuntimeError("GAP executable was not found")
    try:
        completed = subprocess.run(
            (resolved, "-q", "-c", _PROBE),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GapRuntimeError("GAP runtime probe failed") from error
    if completed.returncode != 0:
        raise GapRuntimeError("GAP runtime probe exited unsuccessfully")
    return parse_gap_probe(completed.stdout, executable=resolved)


def source_inventory_digest() -> str:
    """Verify listed standalone bytes and hash the canonical inventory."""

    try:
        encoded = _INVENTORY.read_bytes()
        value = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("standalone source inventory is unavailable") from error
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise RuntimeError("standalone source inventory is malformed")
    for relative, record in value["files"].items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise RuntimeError("standalone source inventory entry is malformed")
        path = _ROOT.joinpath(*relative.split("/"))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get(
            "standalone_sha256"
        ):
            raise RuntimeError(f"standalone source differs from inventory: {relative}")
    return _sha256(encoded)


def host_provenance(runtime: GapRuntime) -> dict[str, object]:
    """Return fresh, JSON-safe host-native calculation provenance."""

    if type(runtime) is not GapRuntime:
        raise TypeError("host provenance requires GapRuntime")
    from . import __version__

    return {
        "certification_status": "host-native",
        "execution_mode": runtime.execution_mode,
        "gap": {
            "executable": runtime.executable,
            "executable_sha256": runtime.executable_sha256,
            "packages": dict(runtime.packages),
            "version": runtime.gap_version,
        },
        "package": {"name": "mathpsg-standalone", "version": __version__},
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "source_inventory_digest": source_inventory_digest(),
    }


__all__ = [
    "GapRuntime",
    "GapRuntimeError",
    "host_provenance",
    "parse_gap_probe",
    "probe_gap",
    "source_inventory_digest",
]
