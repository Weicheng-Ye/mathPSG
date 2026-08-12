# Host-Native PSG Classification Design

## Objective

Add a public Python and command-line interface that calculates PSG
classifications from an IT number and an occupied Wyckoff-position list using
the recorded local GAP runtime. The implementation must work through one
uniform path for all 230 space-group types, must not use Docker, and must not
substitute precomputed class-count data or synthetic results.

The results are host-native and replay-checked. They are not release-signed and
must never be labelled release-certified.

## Occupancy semantics

The `wps` argument describes one physical occupancy configuration.

```python
classify(99, ["a", "b"], igg="Z2")
```

means that atoms occupy both Wyckoff positions `a` and `b` simultaneously. It
produces one joint PSG classification. It does not run two independent
classifications and does not multiply or combine separately computed counts.

Order is retained in the request and in per-orbit detail records. A repeated
entry such as `["a", "a"]` represents two distinct occupied atom-orbit
instances at the same crystallographic Wyckoff position. GAP work that depends
only on the unique stabilizer inclusion may be shared, but the mathematical
classification must retain both instances.

## Public interface

The Python entry point is:

```python
from mathpsg import classify

result = classify(
    99,
    ["a", "b"],
    igg="Z2",
    time_reversal=False,
    setting=None,
    details=False,
)
```

The optional local-execution controls are keyword-only:

- `gap`: executable name or absolute path, defaulting to `gap`;
- `cache`: cache directory, defaulting to the platform MathPSG cache;
- `timeout`: positive GAP timeout in seconds.

The command-line equivalent is:

```bash
mathpsg classify --it-number 99 --wps a b --igg Z2
```

It also accepts `--time-reversal`, `--setting`, `--details`, `--gap`, `--cache`,
and `--timeout`.

The result is an immutable public value. It reports the normalized request,
host-native status, and:

- `class_count: int` for a finite classification;
- `class_count: None` when a U1 result has continuous families;
- lightweight family/stratum summaries in normal mode;
- exact replayable strata, presentations, orbit members, arrows, and
  obstructions when `details=True`.

No live internal capability objects, mutable solver graphs, subprocess handles,
cache paths, or runtime-provenance records are returned.

## Architecture

### Query layer

The query layer validates IT numbers, IGG, settings, flags, and Wyckoff
spellings. It uses `LiveCatalogue` to generate or load the exact catalogue and
resolves each requested spelling to a catalogue record. Ambiguous or absent
spellings fail before solver execution.

The layer constructs a `ClassificationRequest` with one `OrbitInstance` per
input list element. Duplicate Wyckoff identifiers may share crystallographic
source evidence, but instance identifiers remain unique and preserve input
order.

### Host-native classifier backend

A new `HostNativeClassifierBackend` implements the existing
`ClassifierBackendAuthority` protocol:

1. **Ambient resolution:** run the grouped affine-to-PCP conversion and local
   GAP low-degree resolution export for the selected space-group setting and
   time-reversal mode.
2. **Local skeletons:** identify the finite stabilizer type from its exact
   multiplication table and enumerate the required Z2 or U1 local branches on
   demand. Do not package the large precomputed Z2 skeleton tables.
3. **Inclusions:** assemble and replay each stabilizer-to-space-group chain map
   from the same grouped GAP execution.
4. **Joint relative layer:** construct one relative cochain problem for the
   complete ordered occupied-orbit tuple, solve all coefficient sectors, and
   apply local-conjugacy and global-unmarking actions.

The backend reuses the established `certified_classifier.classify_request`
orchestration and existing exact Z2/U1 solvers. Missing orchestration modules
may be copied from the source package when they are generic and directly
required. Release-store and offline-atlas paths are not part of this backend.

### GAP execution

All GAP work uses the executable resolved by `probe_gap`. Before computation,
the package verifies the exact required versions:

- GAP 4.15.1
- Cryst 4.1.30
- HAP 1.70
- HAPcryst 0.1.15
- json 2.2.3
- io 4.9.3

The existing grouped batch exporter is used with the selected local executable.
Its conversion, resolution, finite-group, bar-equivalence, and inclusion
certificates are parsed and independently replayed in Python. Ordinary local
execution is recorded as host-native rather than release-certified.

### Cache

Expensive artifacts are cached outside the repository. Keys bind the complete
classification request, ordered orbit instances, catalogue records, local GAP
executable bytes and versions, source inventory, algorithm versions, and parent
artifact digests.

Unique inclusion evidence may be reused for repeated Wyckoff positions. The
final joint layer key always includes the full ordered instance tuple, so cache
reuse cannot collapse simultaneous or repeated occupancies.

Cached bytes are parsed and mathematically replayed before reuse. Corruption,
identity drift, or version drift causes an explicit failure rather than a
fallback calculation.

## Data flow

1. Validate the public arguments and probe local GAP.
2. Generate/load the catalogue for the requested IT number.
3. Resolve every requested Wyckoff spelling and construct ordered orbit
   instances.
4. Deduplicate only the GAP inclusion work for identical Wyckoff identifiers.
5. Run one grouped GAP conversion and low-degree certificate export.
6. Replay the GAP output and derive ambient, local, and inclusion artifacts.
7. Construct one joint Z2 or U1 relative problem for all occupied instances.
8. Solve and quotient the complete class space.
9. Convert the certified internal result into an immutable host-native public
   result.
10. Emit the same result schema through Python and canonical JSON through the
    CLI.

## Errors and limits

The public API raises typed errors for invalid requests, unavailable or
wrong-version GAP installations, ambiguous Wyckoff labels, backend timeouts,
malformed certificates, incomplete local branches, and cache corruption.

The implementation never returns a partial class count. If any required
coefficient sector, local branch, inclusion, or quotient action fails, the
classification fails with a stage-specific diagnostic.

Subprocess output, artifact size, and elapsed time remain bounded. A user may
increase the positive timeout but cannot disable mathematical replay.

## Testing strategy

Implementation follows failing-test-first development.

1. Public argument and occupancy tests establish that `["a", "b"]` makes one
   ordered joint request and that `["a", "a"]` retains two instances.
2. Backend contract tests cover local GAP selection, exact version rejection,
   grouped unique-inclusion execution, and artifact replay.
3. Z2 integration begins with SG 1 single occupancy, then simultaneous and
   repeated occupancy, followed by representative nontrivial space groups.
4. U1 integration covers finite and continuous outcomes and exact presentation
   details.
5. Cache tests cover clean reuse, ordered-tuple separation, mutation rejection,
   runtime drift, and source drift.
6. CLI tests verify canonical JSON and parity with the Python result.
7. A catalogue-wide smoke test verifies that every IT number reaches the same
   live pipeline and either returns a fully replayed result or a typed genuine
   mathematical/runtime failure; no group-specific fallback is allowed.

Focused tests must pass before broader standalone and local-GAP suites. The
finished change receives an independent code review before the README advertises
the calculator as available.

## Out of scope

- Docker or container execution;
- release signing and release-certified status;
- a bundled precomputed classification atlas;
- copied 19 MiB Z2 skeleton tables;
- special benchmark-group implementations;
- guessed counts, incomplete sectors, or products of single-Wyckoff results.
