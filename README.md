# MathPSG

MathPSG calculates Z2 and U1 projective symmetry group (PSG)
classifications for occupied Wyckoff positions in all 230 three-dimensional
space-group types. It uses one general Python/GAP pipeline, runs GAP locally,
and returns exact class counts or continuous-family presentations with
replay-checked evidence.

This repository does not use Docker or PyXtal. It does not ship a precomputed
classification atlas or special-case benchmark groups. Every query is computed
from the selected local GAP installation and cached outside the package.

## Features

- IT numbers 1 through 230 through one uniform calculation path.
- Z2 and compact-U1 IGGs, with spatial and onsite-time-reversal modes.
- Simultaneous ordered occupancy of multiple Wyckoff positions.
- Repeated labels retained as distinct occupied atom-orbit instances.
- Exact finite PSG class counts and typed evidence for each class.
- Exact continuous U1 presentations when a finite class count does not exist.
- Content-addressed cache entries that are replayed before reuse.
- Immutable, capability-free Python results and canonical JSON from the CLI.
- Internal cache bindings to the Python source and exact GAP environment.

## Requirements

- Python 3.11 or newer.
- GAP available on `PATH`, or supplied explicitly with `gap=...` or `--gap`.
- These exact GAP and package versions:

| Component | Version |
|---|---:|
| GAP | 4.15.1 |
| Cryst | 4.1.30 |
| HAP | 1.70 |
| HAPcryst | 0.1.15 |
| json | 2.2.3 |
| io | 4.9.3 |

Check the active environment after installation:

```bash
mathpsg doctor
```

The calculator rejects a different GAP/package version instead of silently
changing the computational environment.

## Installation

Clone the repository and install it into a virtual environment:

```bash
git clone https://github.com/Weicheng-Ye/mathPSG.git
cd mathPSG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The editable installation makes both `import mathpsg` and the `mathpsg`
command available from any directory while the environment is active. In a
new terminal, activate it with the absolute path to the environment:

```bash
source /path/to/mathPSG/.venv/bin/activate
```

For a regular non-editable installation, use `python -m pip install .` instead.
No third-party Python runtime package is required.

## Python API

The primary entry point is:

```python
mathpsg.classify(
    it_number,
    wps,
    *,
    igg="Z2",
    time_reversal=False,
    setting=None,
    details=False,
    gap="gap",
    cache=None,
    timeout=300,
)
```

### Arguments

| Argument | Meaning |
|---|---|
| `it_number` | International Tables space-group number, from `1` through `230`. |
| `wps` | One Wyckoff label or a nonempty ordered sequence, for example `"a"`, `["a", "b"]`, or `["a", "a"]`. |
| `igg` | Invariant gauge group: `"Z2"` or `"U1"`. The default is `"Z2"`. |
| `time_reversal` | Include onsite time reversal when `True`. The default is spatial symmetry only. |
| `setting` | Explicit catalogue setting, or `None` to accept the unique setting matching all requested labels. |
| `details` | Include the complete immutable classification record when `True`; otherwise `result.details` is `None`. |
| `gap` | GAP executable name or path. The default is `"gap"`. |
| `cache` | Cache directory. `None` selects the platform user-cache directory. |
| `timeout` | Positive per-classification timeout in seconds. The default is `300`. |

The input list is one joint physical occupancy configuration. For example,
`["a", "b"]` performs one simultaneous relative classification; it does not
run two unrelated classifications or multiply two independent counts.
Likewise, `["a", "a"]` represents two distinct occupied atom-orbit instances
at the same Wyckoff position. Input order and repeated instances are retained
in the normalized request and detailed evidence.

### Returned object

`classify()` returns an immutable `HostNativeClassificationResult` with these
attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `request` | `FrozenJSONObject` | Canonical resolved request, including the selected setting and one normalized orbit record per `wps` entry. |
| `class_count` | `int \| None` | Exact number of unframed PSG classes when the quotient is finite; `None` when accepted U1 strata contain continuous families. |
| `continuous` | `bool` | `True` exactly when continuous orbit presentations are present. Equivalently, `continuous == (class_count is None)`. |
| `summaries` | `tuple[FrozenJSONObject, ...]` | One lightweight entry per nonempty solution stratum. A summary is a family of PSG solutions, not necessarily one PSG class. |
| `details` | `FrozenJSONObject \| None` | Complete replayable record when `details=True`; otherwise `None`. |
| `certification_status` | `str` | Currently `"host-native"`, indicating the exact local host calculation path. |

Runtime provenance is deliberately not stored on or serialized with the
classification result. GAP is still probed and bound internally for execution
and cache safety.

`FrozenJSONObject` and `FrozenJSONArray` are read-only mapping-like and
sequence-like containers. Use normal indexing:

```python
result.request["space_group"]
result.summaries[0]["kind"]
result.details["layer"]["framed_strata"][0]["dimension"]
```

The objects cannot be modified in place. Opaque `sha256:...` values identify
content and certificates; they are stable cross-references, not additional
class counts or human-readable physical labels.

### Basic finite Z2 example

```python
import mathpsg

result = mathpsg.classify(1, ["a"], igg="Z2")

print(result.class_count)           # 8
print(result.continuous)            # False
print(result.certification_status)  # host-native
print(result.details)               # None

first = result.summaries[0]
print(first["kind"])                # finite-affine-z2
print(first["skeleton_ids"])        # local-branch identifiers
print(first["stratum_id"])          # identifier for this solution family
```

Here, `len(result.summaries) == 1` means the calculation has one solution
stratum. It does **not** mean there is only one PSG class: that stratum contains
all eight classes reported by `class_count`.

Each summary contains only:

- `kind`: `"finite-affine-z2"` or `"compact-u1-torsor"`;
- `skeleton_ids`: one selected local PSG-branch identifier per occupied orbit;
- `stratum_id`: the identifier for the complete joint solution family.

### Time-reversal Z2 details

```python
result = mathpsg.classify(
    1,
    ["a"],
    igg="Z2",
    time_reversal=True,
    details=True,
)

print(result.class_count)  # 128

layer = result.details["layer"]
strata = layer["framed_strata"]

print(layer["status"])                         # complete
print(len(strata))                              # 2
print([item["dimension"] for item in strata])   # [6, 6]
print([item["unframed_finite_cardinality"]
       for item in strata])                     # [64, 64]
print(layer["unframed_quotient"]
      ["unframed_finite_cardinality"])          # 128
```

Each `finite-affine-z2` stratum describes an affine space over `GF(2)`:

- `basepoint` is one solution in raw binary coordinates;
- `quotient_basis` lists the independent binary directions;
- `dimension` is the number of those directions;
- `framed_finite_cardinality` is therefore `2**dimension`;
- `unframed_finite_cardinality` is the number remaining after residual
  equivalences are applied.

The two strata in this example correspond to two inequivalent local
time-reversal branches. Their opaque `skeleton_ids` distinguish those branches;
the summary ordering should not be used as a physical label.

### U1 example

```python
result = mathpsg.classify(
    99,
    ["a", "b"],
    igg="U1",
    details=True,
)

if result.continuous:
    assert result.class_count is None
    presentations = result.details["layer"]["unframed_quotient"][
        "continuous_orbit_presentations"
    ]
    print(presentations)
else:
    assert isinstance(result.class_count, int)
    print(result.class_count)
```

U1 strata have `kind == "compact-u1-torsor"`. Important detailed fields are
`free_rank`, `torsion_orders`, `basepoint_phases`, and `formal_parameters`.
A positive `free_rank` produces a continuous family. A rank-zero U1 stratum is
finite and includes `finite_class_count`.

### Structure of `details`

When requested, `result.details` contains:

| Key | Meaning |
|---|---|
| `schema_version` | Classification-record schema version. |
| `request_digest` | Content identifier for the canonical normalized request. |
| `catalogue_manifest_digest` | Identifier for the exact catalogue material used. |
| `layer` | Joint classification layer containing strata, obstructions, failures, and the final quotient. |
| `point_routes` | Parameter-specialization routes; empty for ordinary family requests. |
| `routing_verification_digest` | Routing certificate identifier, or `None` when no point routing was required. |

The nested `layer` contains:

| Key | Meaning |
|---|---|
| `status` | `"complete"` for every result returned by the public API. Incomplete calculations raise `ClassificationError` instead of returning a partial result. |
| `failures` | Structured backend failures; empty in a complete returned result. |
| `framed_strata` | Complete Z2 or U1 solution-family records. |
| `obstructed_branches` | Local branches proved not to extend to global solutions. |
| `unframed_quotient` | Aggregate finite counts or continuous presentations after residual identifications. |
| `layer_id` | Content identifier for the complete joint layer. |

Within `unframed_quotient`, `framed_finite_cardinality` counts points before
the final residual identifications and `unframed_finite_cardinality` is the
physical PSG class count returned as `result.class_count`. For a continuous U1
quotient, both finite cardinalities are `None` and
`continuous_orbit_presentations` records the continuous components.

### Explicit execution controls

All options after `wps` are keyword-only:

```python
result = mathpsg.classify(
    70,
    ["a", "b"],
    igg="Z2",
    time_reversal=True,
    setting="2",
    details=False,
    gap="/absolute/path/to/gap",
    cache="/absolute/path/to/cache",
    timeout=300,
)
```

## Command-line calculator

```bash
mathpsg classify --it-number 1 --wps a --igg Z2
mathpsg classify --it-number 99 --wps a b --igg U1 --details
mathpsg classify --it-number 70 --wps a b --igg Z2 \
  --time-reversal --setting 2
```

Optional execution controls are `--gap PATH`, `--cache PATH`, and
`--timeout SECONDS`. Output is canonical JSON containing the normalized
request, class count/continuity, summaries, optional details, and certification
status. It does not contain a runtime block.

## Other commands

Inspect the active Python and GAP runtime:

```bash
mathpsg doctor
```

Generate or load one space-group catalogue:

```bash
mathpsg catalogue --it-number 227
```

Generate the grouped affine/PCP evidence directly:

```bash
mathpsg evidence --it-number 1 --mode spatial
mathpsg evidence --it-number 1 --mode onsite-time
```

Inspect the installed capability boundary:

```bash
mathpsg capabilities
```

## Cache and reproducibility

The default cache is the platform user cache directory. Override it with the
`MATHPSG_CACHE` environment variable, the Python `cache=` argument, or the CLI
`--cache` option.

Cache keys bind the complete ordered request, catalogue records, local GAP
executable bytes and observed versions, source inventory, algorithm versions,
and parent artifacts. Repeated Wyckoff positions may share only their common
GAP inclusion calculation; they remain separate instances in the joint layer.
Every cache hit is parsed and mathematically replayed. Corrupt or mismatched
evidence fails closed.

Results are labelled `host-native`. The calculation and caches internally bind
the local Python/GAP environment, but runtime provenance is not retained in the
returned classification object. Host-native results do not claim release
signing or release-certified authority.

## Development

Run the suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 -W error::ResourceWarning -m unittest discover -s tests -v
```

See [VERIFICATION.md](VERIFICATION.md) for the recorded runtime and completed
real-GAP regressions.

## Repository layout

```text
mathpsg/                 Python package and exact algebra
mathpsg/_assets/         Installed GAP scripts and crystallographic bindings
gap/catalogue/           GAP crystallographic catalogue exporter
gap/classifier/          GAP affine/PCP and low-degree resolution backend
resources/               Display crosswalk and action bindings
tests/                   Unit and real local-GAP integration tests
docs/                    Architecture and host-native classifier design
```

## License

See [LICENSE](LICENSE).
