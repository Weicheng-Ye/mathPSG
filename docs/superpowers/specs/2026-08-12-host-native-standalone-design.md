# Host-Native Standalone MathPSG Design

**Date:** 2026-08-12
**Status:** Design approved; written specification awaiting user review

## Objective

Create a compact standalone Python repository at
`~/Downloads/mathpsg-standalone` for live, uniform PSG calculations over the
230 three-dimensional space-group types. The repository uses a locally
installed GAP runtime and never uses Docker. It treats former benchmark groups
through the same generic path as every other group.

The implementation is an extraction of existing generic MathPSG code. New code
is limited to host-native process configuration, on-demand crystallographic
normalization/caching, and a thin library/CLI facade. It does not introduce new
mathematical algorithms.

## Supported Scope

The repository supports the existing generic classifier's implemented modes:

1. spatial Z2;
2. spatial U1;
3. Z2 with onsite time reversal;
4. U1 with onsite time reversal.

Queries accept one or more ordered Wyckoff-position instances. Repeated
positions remain distinct inputs, and multi-position requests use the existing
joint relative solver rather than products of single-position counts.

If an existing generic path cannot produce a requested result after extraction,
the command fails with an explicit unsupported/unavailable error. It must not
substitute benchmark-specific code, fabricate an approximation, or advertise a
roadmap-only feature.

## Non-Goals

- Docker, OCI images, sealed runtimes, or container attestations.
- Release signing, asset promotion, or release-authority claims.
- Precomputed classification atlases or the unfinished packaged public atlas.
- Benchmark-specific enumerators, paper comparisons, audits, or result tables.
- PyXtal or another Python crystallography dependency.
- Physical ansatz construction, energetic filtering, or material prediction.
- Reproducing historical caches, production reports, planning ledgers, or AI
  artifacts.

## Trust and Provenance Model

Every result is labeled `host-native`, not `release-certified`. Each cache entry
and result records:

- Python implementation and version;
- standalone package version;
- GAP executable path digest and `GAPInfo.Version`;
- loaded GAP package names and exact versions;
- source digests for the Python and GAP files used by the calculation;
- normalized query digest and catalogue-record digest.

At minimum the runtime check covers Cryst, HAP, HAPcryst, json, and io. Missing
or incompatible packages are hard errors. Cache keys bind all recorded version
and source identities, so a changed runtime cannot reuse stale evidence.

This provenance is reproducibility metadata, not a claim that the host is
hermetic or independently certified.

## Repository Layout

```text
mathpsg-standalone/
├── LICENSE
├── README.md
├── pyproject.toml
├── psgmath/
│   ├── generic exact-algebra and classifier modules
│   ├── local_gap.py
│   ├── live_catalogue.py
│   ├── live_classifier.py
│   └── cli.py
├── gap/
│   ├── catalogue/
│   └── classifier/
├── resources/
│   └── display-crosswalk.ndjson
├── tests/
│   ├── exact-algebra and generic-classifier tests
│   ├── local process/provenance tests
│   └── optional real-GAP smoke tests
└── docs/
    └── architecture.md
```

Existing `psgmath` module names are retained wherever practical so copied code
requires minimal import changes. Generated geometry, classification artifacts,
and caches live outside the repository in a user-configurable cache directory.

## Crystallographic Data Flow

The repository retains only the reviewed display crosswalk (approximately
2.2 MB) for conventional settings, multiplicities, and Wyckoff letters. It
does not bundle the approximately 9.6 MB normalized geometry catalogue.

For a queried IT number:

1. the local launcher verifies GAP and required package versions;
2. the copied GAP/Cryst exporter emits exact affine geometry for that group;
3. copied normalization logic validates exact rational data;
4. the display crosswalk attaches the conventional setting and Wyckoff label;
5. canonical normalized records are written atomically to the external cache;
6. subsequent queries reuse them only when the runtime/source binding matches.

Ambiguous settings, missing crosswalk rows, inconsistent multiplicity, or
geometry/crosswalk disagreement are hard failures. No label is guessed.

## Classification Data Flow

1. The API validates the IT number, setting, ordered Wyckoff list, IGG, and
   onsite-time flag.
2. The live catalogue resolves each requested label to an exact embedded
   stabilizer record.
3. Existing GAP request construction produces one shared ambient group request
   and the required literal inclusions.
4. The local GAP adapter runs the copied classifier sources using the detected
   host GAP installation and captures canonical JSON.
5. Existing Python certificate/replay code parses and independently checks the
   returned resolutions, inclusion maps, local skeleton data, and relative
   matrices.
6. Existing Z2 or U1 joint solvers produce finite affine strata, torsor strata,
   obstructions, and residual/Weyl quotients as implemented.
7. A thin serializer emits a stable summary plus optional typed details and the
   host-native provenance record.

The adapter changes launcher and authority plumbing only. Mathematical matrix,
local-lift, relative-complex, quotient, and reconstruction code is copied rather
than rewritten.

## Public Interfaces

Python:

```python
from psgmath import classify

result = classify(
    227,
    ["8a"],
    igg="Z2",
    time_reversal=False,
    details=False,
)
```

CLI:

```text
mathpsg doctor
mathpsg catalogue --it-number 227
mathpsg classify --it-number 227 --wps 8a --igg Z2
mathpsg classify --it-number 99 --wps 1a 1b --igg U1 --time-reversal
```

`doctor` reports exact Python/GAP/package versions and whether the required
local runtime is usable. `catalogue` generates or verifies the on-demand group
cache. `classify` prints canonical JSON. Details are included only when the
copied generic serializer supports the resulting typed evidence without the
unfinished atlas layer.

## Cache and Failure Handling

- Default to the platform user cache directory; allow an explicit cache path.
- Use content-addressed immutable entries and atomic same-directory rename.
- Never store generated data inside the repository.
- Enforce timeouts and bounded stdout/stderr capture for GAP subprocesses.
- Reject malformed, noncanonical, duplicate-key, or oversized JSON.
- Preserve a failed diagnostic envelope without treating it as a reusable
  successful artifact.
- Never fall back to a benchmark calculator or stale entry.

## Extraction Rules

Include a file only when it is in the transitive dependency closure of:

- live catalogue export/normalization;
- host-native GAP request execution and replay;
- generic Z2/U1 local and joint classification;
- public serialization, CLI, provenance, or focused verification.

Exclude `benchmarks/`, `audits/`, release-generation and promotion modules,
Docker/container files, manuscript material, historical diagnostics unrelated
to the selected host-native path, bulky benchmark data, packaged SG1 fixtures,
and generated caches. Remove `__pycache__`, bytecode, temporary outputs, and
source-tree path literals.

## Testing and Acceptance

Testing proceeds in increasing cost:

1. import and syntax checks for every retained Python module;
2. copied exact-algebra and generic Z2/U1 unit tests;
3. local launcher tests using controlled fake GAP output;
4. catalogue normalization and crosswalk tests for representative groups;
5. real local-GAP smoke calculations when the required runtime is available;
6. tests proving spatial/onsite-time routing and ordered repeated/multi-WP
   preservation through existing joint-solver boundaries;
7. a repository scan proving absence of Docker references, PyXtal imports,
   benchmark-package imports, release signing, caches, and generated artifacts.

The final handoff reports exactly which real GAP calculations ran in the local
environment and which were skipped because of missing runtime components. It
does not claim all-230 computational completion unless an actual full sweep was
run successfully.

## Expected Outcome

The result is a substantially smaller research repository with one uniform
host-native code path. It can calculate using local GAP, records exact runtime
versions, contains no Docker or PyXtal dependency, and makes no signed-release
or completed-atlas claim.
