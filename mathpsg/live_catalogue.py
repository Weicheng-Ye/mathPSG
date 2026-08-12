"""Fresh Wyckoff data from the local GAP/Cryst installation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .local_gap import GapRuntime


class CatalogueError(RuntimeError):
    """GAP could not provide the requested crystallographic data."""


@dataclass(frozen=True, slots=True)
class CatalogueRecord:
    """The geometry and subgroup data consumed by the PSG equations."""

    space_group: Mapping[str, object]
    wyckoff_id: str
    letter: str
    multiplicity: int
    stabilizer: Mapping[str, object]
    space_group_action: Mapping[str, object]


def _geometry_key(candidate: Mapping[str, object]) -> str:
    """Return the unhashed geometry key used only to attach Wyckoff letters."""

    orbit = candidate["orbit"]
    stabilizer = candidate["stabilizer"]
    branch = orbit["branches"][0]  # type: ignore[index]
    return json.dumps(
        {
            "basis": branch["basis"],  # type: ignore[index]
            "offset": branch["offset"],  # type: ignore[index]
            "orbit_size": orbit["primitive_orbit_size"],  # type: ignore[index]
            "stabilizer_order": stabilizer["order"],  # type: ignore[index]
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class LiveCatalogue:
    """Run GAP on first use and retain rows for this one classification call."""

    def __init__(
        self,
        runtime: GapRuntime,
        *,
        repository_root: Path,
        timeout_seconds: int = 120,
    ) -> None:
        root = Path(repository_root)
        self.runtime = runtime
        self.timeout_seconds = int(timeout_seconds)
        self.repository_root = root
        self.exporter = root / "gap" / "catalogue" / "export_one.g"
        label_path = root / "resources" / "wyckoff-labels.json"
        try:
            self._labels = json.loads(label_path.read_text(encoding="utf-8"))["groups"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
            raise CatalogueError("Wyckoff labels are unavailable") from error
        self._memory: dict[int, tuple[CatalogueRecord, ...]] = {}

    def _generate(self, it_number: int) -> tuple[CatalogueRecord, ...]:
        with tempfile.TemporaryDirectory(prefix="mathpsg-catalogue-") as raw:
            output = Path(raw) / "catalogue.json"
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
                        os.fspath(output),
                    ),
                    cwd=self.repository_root,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise CatalogueError("local GAP catalogue export failed") from error
            if completed.returncode != 0:
                raise CatalogueError(
                    f"local GAP catalogue export exited with status {completed.returncode}"
                )
            try:
                exported = json.loads(output.read_text(encoding="utf-8"))
                space_group = exported["space_group"]
                setting = str(space_group["setting"]).strip().lower()
                action = exported["space_group_action"]
                candidates = exported["candidates"]
                labels = self._labels[f"{it_number}:{setting}"]
                records = []
                for candidate in candidates:
                    label = labels[_geometry_key(candidate)]
                    letter = str(label["letter"]).lower()
                    records.append(
                        CatalogueRecord(
                            space_group=space_group,
                            wyckoff_id=f"sg{it_number}:{setting}:{letter}",
                            letter=letter,
                            multiplicity=int(label["multiplicity"]),
                            stabilizer=candidate["stabilizer"],
                            space_group_action=action,
                        )
                    )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise CatalogueError("GAP catalogue output lacks computation data") from error
        return tuple(records)

    def records(self, it_number: object) -> tuple[CatalogueRecord, ...]:
        number = int(it_number)
        if number not in self._memory:
            self._memory[number] = self._generate(number)
        return self._memory[number]

    def resolve(
        self,
        it_number: object,
        label: object,
        setting: object | None = None,
    ) -> CatalogueRecord:
        normalized = str(label).strip().lower()
        normalized_setting = None if setting is None else str(setting).strip().lower()
        matches = tuple(
            record
            for record in self.records(it_number)
            if normalized in {record.letter, f"{record.multiplicity}{record.letter}"}
            and (
                normalized_setting is None
                or str(record.space_group["setting"]).strip().lower()
                == normalized_setting
            )
        )
        if len(matches) != 1:
            raise CatalogueError("Wyckoff label is missing or ambiguous")
        return matches[0]


__all__ = ["CatalogueError", "CatalogueRecord", "LiveCatalogue"]
