# Host-Native Standalone MathPSG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `~/Downloads/mathpsg-standalone`, a compact Python/GAP repository that generates and verifies uniform host-native space-group/Wyckoff evidence without Docker, PyXtal, benchmark paths, or release artifacts.

**Architecture:** Copy the reviewed generic exact-algebra, catalogue, GAP protocol, and diagnostic certificate code with minimal changes. Add a small local-runtime probe, on-demand Cryst catalogue cache, and CLI around the already implemented host-native boundaries. Do not create the missing live-artifact-to-final-classifier authority bridge; retain the generic Z2/U1 solvers as library code and document that final all-group class queries remain unavailable without that bridge.

**Tech Stack:** Python 3.10+ standard library, GAP 4.15.1, Cryst 4.1.30, HAP 1.70, HAPcryst 0.1.15, json 2.2.3, io 4.9.3, unittest.

## Global Constraints

- Never invoke, install, or mention Docker as an execution option.
- Do not add PyXtal or any third-party Python runtime dependency.
- Treat every space group through the same catalogue and GAP code path.
- Record exact Python, package, GAP, and GAP-package versions with source digests.
- Label results `host-native`; never claim release certification.
- Copy mathematical code; new code is limited to local process, cache, and CLI plumbing.
- Omit roadmap-only functionality instead of simulating or weakening it.
- Keep generated geometry and evidence outside the repository.
- Use canonical JSON, bounded subprocess captures, atomic writes, and fail-closed validation.

---

### Task 1: Create the clean repository and copy the generic dependency closure

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `psgmath/*.py` from the approved generic source list
- Create: `psgmath/_assets/gap/classifier/lib/*.g`
- Create: `psgmath/_assets/data/stabilizers/v1/{manifest.json,types.ndjson}`
- Create: `gap/catalogue/{export_one.g,lib/normalize_affine.g}`
- Create: `gap/classifier/{export_problem.g,lib/*.g}`
- Create: `resources/display-crosswalk.ndjson`
- Create: `tests/test_extraction.py`
- Create: `EXTRACTED_SOURCES.json`

**Interfaces:**
- Consumes: the current handoff source tree at `/Users/victor/Downloads/mathPSG/mathPSG`.
- Produces: importable `psgmath` package and a canonical source inventory used by all provenance records.

- [ ] **Step 1: Write the extraction-boundary test**

```python
class ExtractionBoundaryTests(unittest.TestCase):
    def test_forbidden_trees_and_dependencies_are_absent(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "containers").exists())
        self.assertFalse((root / "release").exists())
        self.assertFalse((root / "psgmath" / "benchmarks").exists())
        self.assertFalse((root / "psgmath" / "audits").exists())
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("pyxtal", pyproject.lower())

    def test_source_inventory_replays(self):
        root = Path(__file__).resolve().parents[1]
        inventory = json.loads((root / "EXTRACTED_SOURCES.json").read_text())
        for relative, expected in inventory["files"].items():
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected["standalone_sha256"])
```

- [ ] **Step 2: Run the test and verify the empty staging tree fails**

Run: `python3 -m unittest tests.test_extraction -v`

Expected: FAIL because the package scaffold and inventory are absent.

- [ ] **Step 3: Copy only the reviewed generic files**

Use a deterministic copy manifest. Include the top-level modules required by
`catalogue`, `catalogue_schema`, `gap_classifier`, `bar_evaluator`, `cochains`,
`certificates`, `gf2`, `integer_linalg`, `torus`, `relative_complex`,
`stabilizer_types`, `z2_targets`, `z2_local`, `z2_classifier`, `u1_local`,
`u1_classifier`, `residual_groupoid`, `reconstruction`, `classification_schema`,
`query`, `classifier_cache`, `backend_artifacts`, and `certified_classifier`.
Copy their transitive pure-Python dependencies (`algebraic`, `affine`, `lattice`,
`periodic`, `presentation`, `su2`, `antiunitary`, and `_resources`). Do not copy
`public_api`, `classification_atlas`, `task5_release*`, `catalogue_release`,
`catalogue_runner`, `production_backend`, benchmark modules, or audit modules.

Generate `EXTRACTED_SOURCES.json` from the actual bytes with this exact shape:

```python
entry = {
    "source_path": "psgmath/gf2.py",
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "standalone_sha256": hashlib.sha256(standalone_bytes).hexdigest(),
}
document = {
    "files": {"psgmath/gf2.py": entry},
    "record_type": "mathpsg-standalone-source-inventory",
    "schema_version": 1,
}
```

The actual document contains one sorted entry per retained source/resource and
uses canonical compact JSON plus one final newline.

- [ ] **Step 4: Make resource loading standalone-specific**

Replace `_resources.ASSET_PATHS` with only the retained packaged GAP libraries
and stabilizer inventory files. Add `resource_bytes(relative_path: str) -> bytes`
for `resources/display-crosswalk.ndjson`; reject absolute paths and components
in `{"", ".", ".."}`.

- [ ] **Step 5: Run import and extraction tests**

Run: `python3 -m compileall -q psgmath`

Run: `python3 -m unittest tests.test_extraction -v`

Expected: PASS with no bytecode retained after verification cleanup.

- [ ] **Step 6: Commit the extraction checkpoint**

```bash
git add .
git commit -m "chore: extract generic MathPSG core"
```

---

### Task 2: Add exact host-runtime provenance

**Files:**
- Create: `psgmath/local_gap.py`
- Modify: `psgmath/environment.py`
- Test: `tests/test_local_gap.py`

**Interfaces:**
- Consumes: executable name/path and `EXTRACTED_SOURCES.json`.
- Produces: `GapRuntime`, `probe_gap(executable: str = "gap") -> GapRuntime`, `source_inventory_digest() -> str`, and `host_provenance(runtime: GapRuntime) -> dict[str, object]`.

- [ ] **Step 1: Write failing parser and provenance tests**

```python
def test_parse_probe_requires_exact_package_set(self):
    transcript = "\n".join([
        "GAP=4.15.1", "Cryst=4.1.30", "HAP=1.70",
        "HAPcryst=0.1.15", "json=2.2.3", "io=4.9.3",
    ])
    runtime = parse_gap_probe(transcript, executable="/opt/gap/bin/gap")
    self.assertEqual(runtime.packages["json"], "2.2.3")
    self.assertEqual(runtime.execution_mode, "host-native")

def test_provenance_binds_runtime_and_source_inventory(self):
    record = host_provenance(self.runtime)
    self.assertEqual(record["execution_mode"], "host-native")
    self.assertRegex(record["source_inventory_digest"], r"^sha256:[0-9a-f]{64}$")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m unittest tests.test_local_gap -v`

Expected: FAIL because `GapRuntime` and the strict six-version parser do not exist.

- [ ] **Step 3: Implement immutable runtime records and the probe**

```python
@dataclass(frozen=True, slots=True)
class GapRuntime:
    executable: str
    executable_sha256: str
    gap_version: str
    packages: Mapping[str, str]
    execution_mode: Literal["host-native"] = "host-native"

def probe_gap(executable: str = "gap", timeout_seconds: int = 30) -> GapRuntime:
    resolved = shutil.which(executable) if os.sep not in executable else executable
    if resolved is None:
        raise GapRuntimeError("GAP executable was not found")
    # Run one bounded GAP process, parse exactly GAP/Cryst/HAP/HAPcryst/json/io,
    # hash the resolved regular executable, and reject missing fields.
```

The GAP probe loads exact package names and prints only `name=version` records.
It accepts installed versions as observations rather than pin enforcement.

- [ ] **Step 4: Implement canonical source/runtime provenance**

`source_inventory_digest()` recomputes every retained file hash before hashing
the canonical inventory payload. `host_provenance()` includes Python
implementation/version, `psgmath.__version__`, resolved GAP path and digest,
GAP/package versions, source inventory digest, and `certification_status` equal
to `host-native`.

- [ ] **Step 5: Run focused and real local probe tests**

Run: `python3 -m unittest tests.test_local_gap -v`

Run: `python3 -c 'from psgmath.local_gap import probe_gap; print(probe_gap())'`

Expected on this host: GAP 4.15.1, Cryst 4.1.30, HAP 1.70, HAPcryst 0.1.15,
json 2.2.3, and io 4.9.3.

- [ ] **Step 6: Commit the runtime checkpoint**

```bash
git add psgmath/local_gap.py psgmath/environment.py tests/test_local_gap.py
git commit -m "feat: record host GAP provenance"
```

---

### Task 3: Generate and cache one-group crystallographic catalogues

**Files:**
- Create: `psgmath/live_catalogue.py`
- Modify: `psgmath/catalogue.py`
- Modify: `psgmath/catalogue_schema.py`
- Test: `tests/test_live_catalogue.py`

**Interfaces:**
- Consumes: `GapRuntime`, IT number, copied GAP exporter, and reviewed display crosswalk.
- Produces: `LiveCatalogue.records(it_number: int) -> tuple[CatalogueRecord, ...]` and `LiveCatalogue.resolve(it_number: int, label: str, setting: str | None) -> CatalogueRecord`.

- [ ] **Step 1: Write cache and label-resolution tests**

```python
def test_cached_group_is_bound_to_runtime_and_source(self):
    first = self.catalogue.records(1)
    second = self.catalogue.records(1)
    self.assertEqual(first, second)
    document = json.loads(next(self.cache.rglob("record.json")).read_text())
    self.assertEqual(document["certification_status"], "host-native")

def test_missing_crosswalk_match_fails_without_guessing(self):
    with self.assertRaisesRegex(CatalogueError, "crosswalk"):
        self.catalogue._attach_display_rows(self.geometry, ())
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python3 -m unittest tests.test_live_catalogue -v`

Expected: FAIL because `LiveCatalogue` is absent.

- [ ] **Step 3: Implement bounded local Cryst export**

```python
class LiveCatalogue:
    def __init__(self, runtime: GapRuntime, cache_root: Path, repository_root: Path): ...

    def records(self, it_number: int) -> tuple[CatalogueRecord, ...]:
        if type(it_number) is not int or not 1 <= it_number <= 230:
            raise ValueError("IT number must be in 1..230")
        # Resolve a content-addressed cache key; otherwise run export_one.g,
        # normalize the canonical JSON, attach exact display rows, and publish atomically.
```

Use `gap -q gap/catalogue/export_one.g -- --international-number N
--json-output PATH`, a fresh temporary directory, a 120-second timeout, and
16 MiB stdout/stderr limits. Never pass shell text or `shell=True`.

- [ ] **Step 4: Attach display labels without PyXtal**

Load the copied canonical crosswalk once. Match by IT number, setting, exact
embedding/display identity fields already used by the source catalogue code,
and require a one-to-one complete match. Reuse `normalize_gap_export()` and the
existing crosswalk validation helpers; do not derive letters from ordering.

- [ ] **Step 5: Implement atomic immutable cache publication**

The key is SHA-256 over IT number, runtime record, exporter/normalizer hashes,
and crosswalk hash. Write `record.json` and `wyckoff.ndjson` in a sibling
temporary directory, `fsync` files, then rename into an absent final directory.
An existing entry must replay byte-for-byte and semantically before reuse.

- [ ] **Step 6: Run focused and real SG1/SG70/SG227 checks**

Run: `python3 -m unittest tests.test_live_catalogue -v`

Run: `python3 -m psgmath catalogue --it-number 1 --cache /private/tmp/mathpsg-catalogue-smoke`

Repeat for IT numbers 70 and 227. Expected: canonical labelled records and
`certification_status=host-native`; no PyXtal import.

- [ ] **Step 7: Commit the catalogue checkpoint**

```bash
git add psgmath/live_catalogue.py psgmath/catalogue.py psgmath/catalogue_schema.py tests/test_live_catalogue.py
git commit -m "feat: generate live GAP catalogues"
```

---

### Task 4: Run and verify host-native GAP resolution/inclusion evidence

**Files:**
- Create: `psgmath/live_evidence.py`
- Copy: `psgmath/direct_certificate_v2.py`
- Copy: `psgmath/direct_certificate_v2_gap_adapter.py`
- Copy: `psgmath/stabilizer_adapted_witness_v2.py`
- Modify: `psgmath/gap_classifier.py`
- Modify: `psgmath/bar_evaluator.py`
- Test: `tests/test_live_evidence.py`

**Interfaces:**
- Consumes: one setting's `CatalogueRecord` tuple, `GapRuntime`, and `time_reversal` boolean.
- Produces: `HostNativeEvidenceBatch` containing the verified affine-PCP certificate, verified direct target/source resolutions and inclusion maps, canonical artifact bytes, and host provenance.

- [ ] **Step 1: Write command-injection and mode tests**

```python
def test_evidence_uses_argument_vector_and_host_native_status(self):
    batch = build_evidence(self.records, runtime=self.runtime, time_reversal=False)
    self.assertEqual(batch.certification_status, "host-native")
    self.assertFalse(batch.time_reversal)
    self.assertEqual(tuple(m.inclusion_id for m in batch.members), self.ids)

def test_onsite_time_preserves_same_member_universe(self):
    spatial = build_evidence(self.records, runtime=self.runtime, time_reversal=False)
    onsite = build_evidence(self.records, runtime=self.runtime, time_reversal=True)
    self.assertEqual(spatial.member_ids, onsite.member_ids)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python3 -m unittest tests.test_live_evidence -v`

Expected: FAIL because the host-native evidence facade is absent.

- [ ] **Step 3: Configure the existing Task4 diagnostic mode**

Call `run_gap_classifier()` with the explicit argument vector
`(runtime.executable, "-q", absolute_export_problem_path, "--")` and an
environment containing `MATHPSG_CLASSIFIER_DIAGNOSTIC=1`. Modify the runner to
accept an explicit environment mapping while retaining its bounded process
handling and canonical response replay.

- [ ] **Step 4: Configure the existing Task5 batch exporter**

Call `export_gap_inclusion_batch_raw(..., command=(runtime.executable, "-q"))`.
Retain its `release_certified=False` diagnostic attestation and verify using the
existing diagnostic opt-in path. Copy the direct-certificate and GAP-adapter
modules from `tools/diagnostics/` into the package, changing imports only.

- [ ] **Step 5: Assemble and cache `HostNativeEvidenceBatch`**

```python
@dataclass(frozen=True, slots=True)
class HostNativeEvidenceBatch:
    member_ids: tuple[str, ...]
    time_reversal: bool
    affine_certificate: AffinePCPIsomorphismCertificate
    direct_batch: VerifiedDirectBatchV2
    canonical_data: bytes
    provenance: Mapping[str, object]
    certification_status: Literal["host-native"] = "host-native"
```

Require exact member order/coverage and bind the cache key to catalogue record
digests, mode, GAP/runtime provenance, and retained source inventory.

- [ ] **Step 6: Run the focused tests and one real SG1 batch per mode**

Run: `python3 -m unittest tests.test_live_evidence -v`

Run: `python3 -m psgmath evidence --it-number 1 --mode spatial --cache /private/tmp/mathpsg-evidence-smoke`

Run: `python3 -m psgmath evidence --it-number 1 --mode onsite-time --cache /private/tmp/mathpsg-evidence-smoke`

Expected: verified canonical evidence with `release_certified=false` and exact
host version metadata.

- [ ] **Step 7: Commit the evidence checkpoint**

```bash
git add psgmath/live_evidence.py psgmath/direct_certificate_v2.py psgmath/direct_certificate_v2_gap_adapter.py psgmath/stabilizer_adapted_witness_v2.py psgmath/gap_classifier.py psgmath/bar_evaluator.py tests/test_live_evidence.py
git commit -m "feat: verify host GAP evidence"
```

---

### Task 5: Preserve generic Z2/U1 solvers without inventing the missing authority bridge

**Files:**
- Create: `psgmath/solver_status.py`
- Modify: `psgmath/__init__.py`
- Test: `tests/test_solver_status.py`
- Copy focused tests: `tests/test_{z2_local_spatial,z2_local_graded,z2_classifier,u1_local,u1_classifier,multi_orbit_classifier}.py`

**Interfaces:**
- Consumes: copied generic solver modules and `HostNativeEvidenceBatch`.
- Produces: `solver_capabilities() -> Mapping[str, object]`, with no misleading live-classification entry point.

- [ ] **Step 1: Write the boundary test**

```python
def test_live_final_classification_is_not_falsely_advertised(self):
    status = solver_capabilities()
    self.assertTrue(status["generic_z2_solver_present"])
    self.assertTrue(status["generic_u1_solver_present"])
    self.assertFalse(status["live_evidence_bridge_present"])
    self.assertNotIn("classify", psgmath.__all__)
```

- [ ] **Step 2: Run the test and verify it fails against the copied package**

Run: `python3 -m unittest tests.test_solver_status -v`

Expected: FAIL because the original `__init__` still advertises the unfinished
packaged-atlas `classify` facade.

- [ ] **Step 3: Remove the misleading public classifier facade**

Replace `psgmath.__init__.classify` with exports for `LiveCatalogue`,
`build_evidence`, `probe_gap`, and `solver_capabilities`. Do not copy
`public_api.py` or its packaged-atlas loader.

- [ ] **Step 4: Implement explicit capability reporting**

```python
def solver_capabilities() -> Mapping[str, object]:
    return MappingProxyType({
        "generic_z2_solver_present": True,
        "generic_u1_solver_present": True,
        "ordered_multi_orbit_algorithms_present": True,
        "live_evidence_bridge_present": False,
        "reason": "source snapshot has no reviewed host-native authority bridge",
    })
```

The CLI must not define a `classify` subcommand. This prevents an intermediate
evidence generator from being misrepresented as a complete PSG query engine.

- [ ] **Step 5: Run copied generic solver tests**

Run the selected unittest modules individually. Retain tests that exercise
diagnostic constructors, exact matrix algebra, all four mode semantics, and
ordered repeated/multi-orbit composition. Remove fixture cases tied to signed
release bundles rather than editing their assertions.

Expected: all retained generic tests pass with no benchmark-module import.

- [ ] **Step 6: Commit the solver-boundary checkpoint**

```bash
git add psgmath/__init__.py psgmath/solver_status.py tests/test_solver_status.py tests/test_z2_local_spatial.py tests/test_z2_local_graded.py tests/test_z2_classifier.py tests/test_u1_local.py tests/test_u1_classifier.py tests/test_multi_orbit_classifier.py
git commit -m "docs: expose exact solver boundary"
```

---

### Task 6: Add the concise CLI and documentation

**Files:**
- Create: `psgmath/cli.py`
- Create: `psgmath/__main__.py`
- Create: `README.md`
- Create: `docs/architecture.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `probe_gap`, `LiveCatalogue`, `build_evidence`, and `solver_capabilities`.
- Produces: `mathpsg doctor`, `mathpsg catalogue`, `mathpsg evidence`, and `mathpsg capabilities`.

- [ ] **Step 1: Write CLI contract tests**

```python
def test_cli_has_only_implemented_commands(self):
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("doctor", "catalogue", "evidence", "capabilities"):
        self.assertIn(command, help_text)
    self.assertNotIn("classify", help_text)

def test_doctor_json_records_all_versions(self):
    completed = run_cli("doctor", "--json")
    record = json.loads(completed.stdout)
    self.assertEqual(record["certification_status"], "host-native")
    self.assertEqual(set(record["gap"]["packages"]), {"cryst", "hap", "hapcryst", "json", "io"})
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python3 -m unittest tests.test_cli -v`

Expected: FAIL because the standalone parser is absent.

- [ ] **Step 3: Implement the four commands**

All commands support `--gap PATH` and `--cache PATH`. `catalogue` requires
`--it-number 1..230`. `evidence` requires `--it-number` and
`--mode {spatial,onsite-time}`, computes the complete selected setting member
batch, and prints a canonical summary rather than the potentially huge artifact.
Invalid input returns exit code 2; backend failure returns exit code 1.

- [ ] **Step 4: Write accurate standalone documentation**

README sections: installation, exact GAP package versions observed on the
development host, doctor, on-demand catalogue, evidence commands, cache model,
provenance, implemented solver modules, missing final host-native authority
bridge, and excluded material. State plainly that the repository does not yet
offer an all-group `classify()` convenience API.

- [ ] **Step 5: Run CLI tests and smoke commands**

Run: `python3 -m unittest tests.test_cli -v`

Run: `python3 -m psgmath doctor --json`

Run: `python3 -m psgmath capabilities`

Expected: canonical JSON and truthful host-native status.

- [ ] **Step 6: Commit the interface checkpoint**

```bash
git add psgmath/cli.py psgmath/__main__.py README.md docs/architecture.md tests/test_cli.py pyproject.toml
git commit -m "feat: add host-native CLI"
```

---

### Task 7: Verify, prune, and publish the standalone repository

**Files:**
- Modify: `EXTRACTED_SOURCES.json`
- Modify: `README.md`
- Create: `VERIFICATION.md`
- Test: all retained tests

**Interfaces:**
- Consumes: completed staging repository.
- Produces: verified `~/Downloads/mathpsg-standalone` repository and a report of executed/skipped checks.

- [ ] **Step 1: Run the complete retained suite**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -W error::ResourceWarning -m unittest discover -s tests -v`

Record test count, duration, failures, and skips in `VERIFICATION.md`. A skip is
allowed only for a clearly unavailable expensive real-GAP smoke, not for exact
algebra or serialization tests.

- [ ] **Step 2: Run real local GAP smoke gates**

Run `doctor`, catalogue generation for IT 1/70/227, and spatial plus onsite-time
evidence for IT 1. Record exact commands, versions, durations, and output SHA-256
digests in `VERIFICATION.md`.

- [ ] **Step 3: Scan for excluded content**

Run repository-wide searches for case-insensitive `docker`, `pyxtal`,
`psgmath.benchmarks`, `psgmath.audits`, signing keys, release fixtures,
`__pycache__`, `.pyc`, and absolute source-worktree paths. Documentation may say
Docker and PyXtal are excluded; executable/configuration references are forbidden.

- [ ] **Step 4: Measure the final repository**

Record total bytes, file count, Python source bytes, GAP source bytes, and
resource bytes. Verify no generated cache or evidence file is inside the tree.

- [ ] **Step 5: Run an independent code review**

Request review of all copied/modified code for import closure, subprocess safety,
cache confinement, provenance accuracy, mathematical-code preservation, and
truthful capability claims. Resolve every high/medium finding and rerun affected
tests.

- [ ] **Step 6: Commit the verified state**

```bash
git add EXTRACTED_SOURCES.json README.md VERIFICATION.md
git commit -m "test: verify standalone host-native package"
```

- [ ] **Step 7: Copy into the requested Downloads location**

First require that `~/Downloads/mathpsg-standalone` is absent. Copy the verified
staging repository byte-for-byte, excluding no tracked files. Recompute the
inventory and rerun `python3 -m unittest tests.test_extraction -v` from the final
location. Do not overwrite an existing target.
