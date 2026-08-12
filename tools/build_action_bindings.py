#!/usr/bin/env python3
"""Extract the compact per-group action bindings from a reviewed catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    source_data = arguments.source.read_bytes()
    groups: dict[str, dict[str, str]] = {}
    for line in source_data.splitlines():
        row = json.loads(line)
        number = str(row["space_group"]["international_number"])
        binding = {
            "space_group_action_sha256": "sha256:"
            + hashlib.sha256(canonical(row["space_group_action"])).hexdigest(),
        }
        prior = groups.setdefault(number, binding)
        if prior != binding:
            raise ValueError(f"space group {number} has inconsistent action bindings")
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
