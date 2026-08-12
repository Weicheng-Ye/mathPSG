# MathPSG Standalone

MathPSG Standalone is a Python and GAP toolkit for working with all 230
three-dimensional space-group types. It builds exact Wyckoff-position
catalogues and generates machine-checkable evidence for spatial and
onsite-time conversions.

> **Scope:** this package produces Wyckoff-position catalogues and conversion
> evidence. It does not calculate final PSG class counts and does not provide a
> `classify()` API.

## Features

- One uniform calculation path for IT numbers 1 through 230.
- Exact space-group and Wyckoff-position geometry from GAP and Cryst.
- Spatial and onsite-time conversion modes.
- Canonical JSON output checked by Python before it is returned.
- Content-addressed catalogue caching.
- Exact runtime provenance, including the GAP executable digest and package
  versions.
- No third-party Python runtime dependencies.

## Requirements

- Python 3.11 or newer.
- GAP available through `PATH`, or selected with the `--gap` option on
  `doctor`, `catalogue`, and `evidence`.
- The GAP packages listed below.

The calculation code requires these exact GAP and package versions:

| Component | Version |
|---|---:|
| GAP | 4.15.1 |
| Cryst | 4.1.30 |
| HAP | 1.70 |
| HAPcryst | 0.1.15 |
| json | 2.2.3 |
| io | 4.9.3 |

Check the active runtime with:

```bash
gap -q -c 'Print(GAPInfo.Version, "\n"); QUIT;'
```

## Installation

Clone this repository, enter its directory, and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The `mathpsg` command is then available from any directory while the virtual
environment is active:

```bash
mathpsg doctor
```

Activate the environment again in a new terminal session with:

```bash
source /path/to/mathpsg-standalone/.venv/bin/activate
```

An installation is optional. From the repository directory, every command can
also be run as `python3 -m psgmath`.

## Quick start

Inspect the Python and GAP runtime:

```bash
mathpsg doctor
```

Generate the Wyckoff catalogue for space group 227:

```bash
mathpsg catalogue --it-number 227
```

Generate spatial conversion evidence for space group 1:

```bash
mathpsg evidence --it-number 1 --mode spatial
```

Generate onsite-time conversion evidence:

```bash
mathpsg evidence --it-number 1 --mode onsite-time
```

Display the available modules and public interfaces:

```bash
mathpsg capabilities
```

All commands emit canonical JSON.

## Commands

### `doctor`

Reports:

- Python implementation, executable, and version;
- MathPSG Standalone version;
- resolved GAP executable and its SHA-256 digest;
- GAP and GAP-package versions;
- source digest used to identify the installed code.

```bash
mathpsg doctor
mathpsg doctor --gap /absolute/path/to/gap
```

### `catalogue`

Generates or loads the exact catalogue for one IT number:

```bash
mathpsg catalogue --it-number 70
mathpsg catalogue --it-number 227 --cache /path/to/cache
```

Valid IT numbers are 1 through 230. The output contains the setting, Wyckoff
labels, site-symmetry symbols, and stable identifiers for the Wyckoff
positions.

### `evidence`

Runs the local GAP conversion calculation and checks the returned certificate
in Python:

```bash
mathpsg evidence --it-number 99 --mode spatial
mathpsg evidence --it-number 99 --mode onsite-time
```

The evidence command processes every Wyckoff position in the selected setting.
Its summary includes the request digest, certificate digest, member IDs, and
mode.

### `capabilities`

Reports which mathematical modules and public interfaces are available:

```bash
mathpsg capabilities
```

## Python usage

Runtime discovery is available as a Python API:

```python
from psgmath import probe_gap

runtime = probe_gap()
print(runtime.gap_version)
print(dict(runtime.packages))
```

The exact-algebra modules are importable under `psgmath`. The public high-level
interface covers catalogue generation and conversion evidence, but not final
PSG classification results.

## Cache and reproducibility

Generated catalogues are stored outside the repository. The default location
is the platform user cache directory; use `--cache PATH` to choose another
location.

Each cache entry records:

- the requested IT number;
- GAP executable bytes and observed package versions;
- catalogue exporter and normalizer sources;
- the packaged display and action data;
- a digest of the installed source files.

Cached records are parsed and checked again before reuse. Space-group coverage,
record counts, actions, provenance fields, and identifiers must match the
recorded calculation inputs.

Outputs use the status `host-native`, meaning that they record the local Python
and GAP runtime used for the calculation.

## Development

Run the test suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 -W error::ResourceWarning -m unittest discover -s tests -v
```

The verification record includes real local-GAP checks for representative
space groups and both evidence modes. See [VERIFICATION.md](VERIFICATION.md).

## Repository layout

```text
psgmath/                 Python package
psgmath/_assets/         Packaged GAP scripts and reference bindings
gap/catalogue/           GAP crystallographic catalogue exporter
gap/classifier/          GAP affine/PCP conversion backend
resources/               Display crosswalk and action bindings
tests/                   Unit and local-GAP integration tests
docs/architecture.md     Data flow and validation design
```

## License

See [LICENSE](LICENSE).
