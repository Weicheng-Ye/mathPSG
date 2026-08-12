"""Small command-line interface for the implemented standalone boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .catalogue_schema import canonical_json
from .live_catalogue import LiveCatalogue
from .live_evidence import build_evidence
from .local_gap import host_provenance, probe_gap
from .solver_status import solver_capabilities


_ROOT = Path(__file__).resolve().parents[1]


def _default_cache() -> Path:
    override = os.environ.get("MATHPSG_CACHE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mathpsg-standalone"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (
        Path(xdg).expanduser() / "mathpsg-standalone"
        if xdg
        else Path.home() / ".cache" / "mathpsg-standalone"
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, cache: bool) -> None:
    parser.add_argument("--gap", default="gap", help="local GAP executable")
    if cache:
        parser.add_argument(
            "--cache",
            type=Path,
            default=_default_cache(),
            help="external content-addressed cache directory",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mathpsg",
        description="Host-native GAP catalogue and verified conversion evidence",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="report exact local runtime versions")
    _add_runtime_arguments(doctor, cache=False)

    catalogue = commands.add_parser(
        "catalogue", help="generate exact Wyckoff geometry for one IT number"
    )
    _add_runtime_arguments(catalogue, cache=True)
    catalogue.add_argument("--it-number", type=int, required=True, choices=range(1, 231))

    evidence = commands.add_parser(
        "evidence", help="generate replay-verified GAP affine/PCP evidence"
    )
    _add_runtime_arguments(evidence, cache=True)
    evidence.add_argument("--it-number", type=int, required=True, choices=range(1, 231))
    evidence.add_argument("--mode", choices=("spatial", "onsite-time"), required=True)
    evidence.add_argument("--timeout", type=int, default=300)

    commands.add_parser(
        "capabilities", help="report exact solver and public-API boundaries"
    )
    return parser


def _write(value: object) -> None:
    print(canonical_json(value).decode("utf-8"))


def _catalogue(runtime, cache: Path) -> LiveCatalogue:
    return LiveCatalogue(
        runtime,
        cache_root=cache,
        repository_root=_ROOT,
    )


def _catalogue_summary(catalogue: LiveCatalogue, it_number: int) -> dict[str, object]:
    records = catalogue.records(it_number)
    positions = []
    for record in records:
        display = catalogue._display.by_id[record.wyckoff_id]
        positions.append(
            {
                "label": f"{display.conventional_multiplicity}{display.wyckoff_letter}",
                "setting": str(record.space_group["setting"]),
                "site_symmetry": display.site_symmetry_symbol,
                "wyckoff_id": record.wyckoff_id,
            }
        )
    return {
        "certification_status": "host-native",
        "it_number": it_number,
        "record_count": len(records),
        "record_type": "mathpsg-live-catalogue-summary",
        "wyckoff_positions": positions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "capabilities":
            _write(dict(solver_capabilities()))
            return 0
        runtime = probe_gap(arguments.gap)
        if arguments.command == "doctor":
            _write(host_provenance(runtime))
            return 0
        catalogue = _catalogue(runtime, arguments.cache)
        if arguments.command == "catalogue":
            _write(_catalogue_summary(catalogue, arguments.it_number))
            return 0
        records = catalogue.records(arguments.it_number)
        batch = build_evidence(
            records,
            runtime=runtime,
            repository_root=_ROOT,
            time_reversal=arguments.mode == "onsite-time",
            timeout_seconds=arguments.timeout,
        )
        _write(
            {
                "affine_certificate_digest": batch.affine_certificate.certificate_digest,
                "canonical_evidence_bytes": len(batch.canonical_data),
                "certification_status": batch.certification_status,
                "it_number": arguments.it_number,
                "member_count": len(batch.member_ids),
                "member_ids": list(batch.member_ids),
                "mode": arguments.mode,
                "record_type": "mathpsg-host-native-evidence-summary",
                "release_certified": False,
                "request_digest": batch.request.request_digest,
            }
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"mathpsg: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
