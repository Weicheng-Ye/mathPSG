"""Minimal host-native GAP discovery.

The public calculator only needs to know that a GAP executable can be
resolved, started, and made to return machine-readable output.  Source-tree
inventories, executable hashes, and exact package versions are development
metadata; they must not gate a fresh user's calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess


_PROBE = r'''
Print("{\"ok\":true}\n");
QUIT_GAP(0);
'''


class GapRuntimeError(RuntimeError):
    """The requested local GAP process could not be used."""


@dataclass(frozen=True, slots=True)
class GapRuntime:
    executable: str


def parse_gap_probe(stdout: str, *, executable: str) -> GapRuntime:
    """Parse the JSON success marker emitted by the local GAP probe."""

    try:
        json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise GapRuntimeError("GAP probe did not return valid JSON") from error
    resolved = Path(executable).resolve(strict=True)
    return GapRuntime(executable=os.fspath(resolved))


def probe_gap(executable: str = "gap", timeout_seconds: int = 30) -> GapRuntime:
    """Resolve GAP and require one bounded, successful JSON-producing run."""

    executable = os.fspath(executable)
    timeout_seconds = int(timeout_seconds)
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


__all__ = [
    "GapRuntime",
    "GapRuntimeError",
    "parse_gap_probe",
    "probe_gap",
]
