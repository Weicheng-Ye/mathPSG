#!/usr/bin/env python3
"""Extract the compact per-group action bindings from a reviewed catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ACTION_CONTEXT = {
    "coordinate_convention": {
        "affine_action": "x_column -> matrix*x_column + translation",
        "composition_law": "C(g*h)=C(h)*C(g)",
        "rational_encoding": "q(n,d)",
        "source_action": "Cryst right-row homogeneous matrices",
        "translation_policy": "full-unreduced",
    },
    "environment": {
        "certification_status": "uncertified-direct",
        "load_policy": "exact-version-only-needed",
        "versions": {
            "alnuth": "3.2.1",
            "autpgrp": "1.11.1",
            "cryst": "4.1.30",
            "gap": "4.15.1",
            "polenta": "1.3.11",
            "polycyclic": "2.17",
            "radiroot": "2.9",
        },
    },
    "protocol_version": 1,
    "source": {"cryst": "4.1.30", "gap": "4.15.1"},
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def action_provenance(action: dict[str, object]) -> str:
    paired = []
    for generator in action["source_generators"]:
        homogeneous = [
            [*row, generator["translation"][index]]
            for index, row in enumerate(generator["matrix"])
        ] + [["q(0,1)", "q(0,1)", "q(0,1)", "q(1,1)"]]
        right_witness = [list(row) for row in zip(*homogeneous, strict=True)]
        paired.append(
            {
                "column_affine": generator,
                "source_right_homogeneous_matrix": right_witness,
            }
        )
    paired.sort(key=canonical)
    return digest(
        {
            "domain": "mathpsg-action-provenance-v1",
            **ACTION_CONTEXT,
            "paired_source_generators": paired,
            "presentation_conjugation": None,
        }
    )


def identity_payload(row: dict[str, object]) -> dict[str, object]:
    return {
        "space_group": {
            "international_number": row["space_group"]["international_number"],
            "setting": row["space_group"]["setting"],
        },
        "primitive_lattice": {
            "translation_basis": row["space_group_action"]["translation_basis"]
        },
        "paired_family": {"orbit": row["orbit"], "stabilizer": row["stabilizer"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    source_data = arguments.source.read_bytes()
    rows_by_group: dict[str, list[dict[str, object]]] = {}
    for line in source_data.splitlines():
        row = json.loads(line)
        number = str(row["space_group"]["international_number"])
        rows_by_group.setdefault(number, []).append(row)
    groups: dict[str, dict[str, str]] = {}
    for number, rows in rows_by_group.items():
        action = rows[0]["space_group_action"]
        if any(row["space_group_action"] != action for row in rows):
            raise ValueError(f"space group {number} has inconsistent actions")
        action_digest = action_provenance(action)
        identity_payloads = sorted(
            (canonical(identity_payload(row)) for row in rows)
        )
        binding = {
            "action_provenance_digest": action_digest,
            "generator_input_digest": digest(
                {
                    "domain": "mathpsg-generator-input-v1",
                    "action_provenance_digest": action_digest,
                    "identity_payloads": [item.decode("utf-8") for item in identity_payloads],
                }
            ),
            "space_group_action_sha256": digest(action),
        }
        groups[number] = binding
    if set(groups) != {str(number) for number in range(1, 231)}:
        raise ValueError("catalogue does not cover IT numbers 1..230 exactly")
    document = {
        "groups": groups,
        "record_type": "mathpsg-standalone-action-bindings",
        "schema_version": 1,
        "source_sha256": "sha256:" + hashlib.sha256(source_data).hexdigest(),
    }
    arguments.output.write_bytes(canonical(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
