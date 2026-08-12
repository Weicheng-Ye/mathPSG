"""Structured diagnostics for the MathPSG runtime environment."""

from __future__ import annotations

import platform
import subprocess
import sys


_GAP_PROBE = r'''
Print("GAP=", GAPInfo.Version, "\n");
if LoadPackage("cryst") <> true then Error("Cryst unavailable"); fi;
if LoadPackage("hap") <> true then Error("HAP unavailable"); fi;
if LoadPackage("hapcryst") <> true then Error("HAPcryst unavailable"); fi;
Print("Cryst=", PackageInfo("cryst")[1].Version, "\n");
Print("HAP=", PackageInfo("hap")[1].Version, "\n");
Print("HAPcryst=", PackageInfo("hapcryst")[1].Version, "\n");
QUIT;
'''
_GAP_PROBE_TIMEOUT_SECONDS = 30


def python_environment_record() -> dict[str, object]:
    """Return the stable Python portion of an environment diagnostic."""
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable": sys.executable,
    }


def parse_gap_probe(stdout: str) -> dict[str, str]:
    """Extract the required version fields from a GAP probe transcript."""
    expected = {"GAP": "gap", "Cryst": "cryst", "HAP": "hap", "HAPcryst": "hapcryst"}
    values: dict[str, str] = {}
    for raw_line in stdout.replace("\r", "").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        key = key.strip()
        if key in expected:
            values[expected[key]] = value.strip()
    missing = sorted(set(expected.values()) - set(values))
    if missing:
        raise ValueError("missing GAP probe fields: " + ", ".join(missing))
    return values


def gap_environment_record(executable: str = "gap") -> dict[str, object]:
    """Return structured GAP-package versions or an actionable failure record."""
    try:
        result = subprocess.run(
            [executable, "-q", "-c", _GAP_PROBE],
            check=False,
            capture_output=True,
            text=True,
            timeout=_GAP_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "executable": executable,
            "error": "probe-timeout",
        }
    except FileNotFoundError:
        return {
            "available": False,
            "executable": executable,
            "error": "executable-not-found",
        }
    except OSError:
        return {
            "available": False,
            "executable": executable,
            "error": "executable-error",
        }
    record: dict[str, object] = {
        "available": result.returncode == 0,
        "executable": executable,
        "exit_code": result.returncode,
    }
    if result.returncode != 0:
        record["error"] = "probe-failed"
        record["stderr"] = result.stderr.strip()
        return record
    record.update(parse_gap_probe(result.stdout))
    return record
