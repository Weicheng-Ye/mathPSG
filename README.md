# MathPSG Standalone

This is a compact Python/GAP extraction from the larger MathPSG research
repository. It treats all IT numbers 1 through 230 through one on-demand GAP
path. Former benchmark groups have no special cases.

The repository implements:

- strict discovery and provenance for a local GAP installation;
- exact GAP/Cryst Wyckoff geometry for any IT number, normalized and cached on
  demand;
- replay-verified affine-to-PCP conversion evidence for spatial and onsite-time
  modes;
- the copied generic exact-algebra, Z2, and U1 solver source modules.

It deliberately does **not** expose `classify()`. The source snapshot has no
reviewed host-native bridge from live GAP evidence into the final Z2/U1 solver
inputs. That unfinished feature is omitted instead of being simulated. The
standalone commands calculate crystallographic and conversion evidence, not
final PSG class counts.

## Requirements

- Python 3.11 or newer; no third-party Python runtime packages.
- GAP available locally.
- GAP packages Cryst, HAP, HAPcryst, json, and io.

The development host used these exact observed versions:

| Component | Version |
|---|---:|
| Python | 3.14.6 |
| GAP | 4.15.1 |
| Cryst | 4.1.30 |
| HAP | 1.70 |
| HAPcryst | 0.1.15 |
| json | 2.2.3 |
| io | 4.9.3 |

Every command probes and records the versions actually used. Evidence is
labeled `host-native`, never release-certified.

## Install for use from any directory

The recommended setup is an editable installation in a dedicated virtual
environment:

```bash
python3 -m venv ~/.venvs/mathpsg
source ~/.venvs/mathpsg/bin/activate
python -m pip install -e ~/Downloads/mathpsg-standalone
```

After activation, the package and command are available from any directory:

```bash
mathpsg doctor
mathpsg catalogue --it-number 227
python -c 'from psgmath import probe_gap; print(probe_gap())'
```

Activate the environment again in each new terminal session:

```bash
source ~/.venvs/mathpsg/bin/activate
```

To use the package without installing it, set `PYTHONPATH` for the command:

```bash
PYTHONPATH=~/Downloads/mathpsg-standalone \
  python3 -m psgmath catalogue --it-number 227
```

The local `gap` executable must be available through `PATH`. To use a specific
executable, pass its absolute path, for example:

```bash
mathpsg doctor --gap /absolute/path/to/gap
```

## Run

From the repository directory, you can run the package directly without
installing it:

```bash
python3 -m psgmath doctor
```

If you installed and activated the virtual environment described above, you
can use the shorter `mathpsg` command in place of `python3 -m psgmath` in all
the examples below.

Generate or replay the exact catalogue for one space group:

```bash
python3 -m psgmath catalogue --it-number 227
```

Generate the implemented GAP conversion evidence for every Wyckoff position in
one setting:

```bash
python3 -m psgmath evidence --it-number 1 --mode spatial
python3 -m psgmath evidence --it-number 1 --mode onsite-time
```

Inspect the exact implementation boundary:

```bash
python3 -m psgmath capabilities
```

Use `--gap /absolute/path/to/gap` to select a GAP executable and `--cache PATH`
to place generated data somewhere other than the platform user cache.

## Cache and provenance

Generated catalogues live outside the repository. Catalogue cache keys bind the
IT number, GAP executable bytes, observed GAP/package versions, copied exporter
and normalizer bytes, display-crosswalk bytes, and the complete standalone
source inventory. A compact 230-row action-binding table prevents a
self-rehashed cache from changing the ambient space-group generators. Cached
canonical JSON is re-parsed and semantically replayed before reuse.

`EXTRACTED_SOURCES.json` records every retained file and, where it was copied,
its original source digest. `mathpsg doctor` verifies that inventory and reports
the Python executable/version, package version, GAP executable/digest, GAP
version, and all required GAP-package versions.

## Excluded material

The extraction contains no container setup, PyXtal dependency, benchmark
calculator, signed release store, precomputed all-group geometry, generated
atlas, model artifacts, manuscript, or historical production cache. It also
omits the large precomputed Z2 stabilizer-skeleton tables; the corresponding
generic source remains present for inspection and future wiring, but it is not
advertised as a runnable final calculator.

See [docs/architecture.md](docs/architecture.md) for the data flow and trust
boundary, and [VERIFICATION.md](VERIFICATION.md) for checks run on this copy.
