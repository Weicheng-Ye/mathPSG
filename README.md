# MathPSG Standalone

MathPSG Standalone calculates Z2 and U1 projective symmetry group (PSG)
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
- Exact Python, GAP, GAP-package, executable, and source provenance.

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
git clone <repository-url> mathpsg-standalone
cd mathpsg-standalone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The editable installation makes both `import psgmath` and the `mathpsg`
command available from any directory while the environment is active. In a
new terminal, activate it with the absolute path to the environment:

```bash
source /path/to/mathpsg-standalone/.venv/bin/activate
```

For a regular non-editable installation, use `python -m pip install .` instead.
No third-party Python runtime package is required.

## Python calculator

```python
from psgmath import classify

result = classify(1, ["a"], igg="Z2")
print(result.class_count)
print(result.certification_status)  # host-native
```

Request exact details with `details=True`:

```python
result = classify(99, ["a", "b"], igg="U1", details=True)

print(result.class_count)  # integer, or None for continuous U1 families
print(result.continuous)
print(result.summaries)
print(result.details)
```

The list `['a', 'b']` is one physical occupancy configuration in which both
Wyckoff positions contain atoms. It triggers one joint relative classification;
it does not run two independent calculations or multiply their counts.

Likewise, `['a', 'a']` represents two distinct occupied atom-orbit instances at
the same Wyckoff position. Input order and repeated instances are preserved in
the request and detailed evidence.

All options are keyword-only after `wps`:

```python
result = classify(
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

`class_count` is an exact integer when the unframed quotient is finite. It is
`None` only when accepted U1 strata contain continuous families; their exact
presentations are included when details are requested.

## Command-line calculator

```bash
mathpsg classify --it-number 1 --wps a --igg Z2
mathpsg classify --it-number 99 --wps a b --igg U1 --details
mathpsg classify --it-number 70 --wps a b --igg Z2 \
  --time-reversal --setting 2
```

Optional execution controls are `--gap PATH`, `--cache PATH`, and
`--timeout SECONDS`. Output is canonical JSON containing the normalized request,
class count/continuity, summaries or details, and exact runtime provenance.

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

Results are labelled `host-native`: they report and bind the local Python/GAP
environment but do not claim release signing or release-certified authority.

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
psgmath/                 Python package and exact algebra
psgmath/_assets/         Installed GAP scripts and crystallographic bindings
gap/catalogue/           GAP crystallographic catalogue exporter
gap/classifier/          GAP affine/PCP and low-degree resolution backend
resources/               Display crosswalk and action bindings
tests/                   Unit and real local-GAP integration tests
docs/                    Architecture and host-native classifier design
```

## License

See [LICENSE](LICENSE).
