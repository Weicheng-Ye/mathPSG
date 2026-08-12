# Host-Native PSG Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public `classify()` function and CLI command that perform one
joint host-native Z2 or U1 PSG calculation for an ordered occupied-Wyckoff
configuration using local GAP.

**Architecture:** Restore the generic verified query/classifier orchestration,
then implement a `HostNativeClassifierBackend` that obtains grouped Task4/Task5
evidence from local GAP and computes local branches on demand. A public adapter
resolves Wyckoff spellings, constructs one joint request, runs the backend, and
returns immutable host-native values.

**Tech Stack:** Python 3.11+, standard library only, GAP 4.15.1, Cryst 4.1.30,
HAP 1.70, HAPcryst 0.1.15, json 2.2.3, io 4.9.3, `unittest`.

## Global Constraints

- Use local GAP only; never invoke Docker or `/opt/mathpsg`.
- Treat all IT numbers 1 through 230 through the same code path.
- `wps=["a", "b"]` is one simultaneous occupancy and one joint solve.
- Preserve input order and repeated positions as distinct instances.
- Deduplicate only GAP work for identical crystallographic inclusions.
- Return host-native, replay-checked results; never claim release certification.
- Do not copy the large spatial or graded Z2 skeleton tables.
- Do not invent counts, skip sectors, or multiply single-WP counts.
- Observe a relevant failing test before every production change.

## File Structure

- `mathpsg/catalogue_loader.py`: immutable `CatalogueIndex`.
- `mathpsg/query.py`: request resolution and verified catalogue authority.
- `mathpsg/certified_classifier.py`: staged joint classifier orchestration.
- `mathpsg/host_classifier_backend.py`: local GAP production and four backend stages.
- `mathpsg/live_classify.py`: public validation, orchestration, and result values.
- `tests/test_classify_request_orbits.py`: occupancy semantics.
- `tests/test_host_classifier_backend.py`: backend contract and replay.
- `tests/test_live_classify.py`: API and real-GAP integration.

---

### Task 1: Restore Generic Joint-Classifier Orchestration

**Files:**
- Create: `mathpsg/catalogue_loader.py`
- Create: `mathpsg/query.py`
- Create: `mathpsg/certified_classifier.py`
- Create: `tests/test_classify_request_orbits.py`

**Interfaces:**
- Produces `CatalogueIndex`, `make_diagnostic_verified_catalogue`,
  `ClassifierBackendAuthority`, and `classify_request`.
- Consumes the existing catalogue, schema, cache, cochain, residual-groupoid,
  Z2, and U1 modules.

- [ ] **Step 1: Write the failing import test**

```python
from mathpsg.catalogue_loader import CatalogueIndex
from mathpsg.query import make_diagnostic_verified_catalogue
from mathpsg.certified_classifier import ClassifierBackendAuthority

def test_live_records_form_diagnostic_catalogue(self):
    index = CatalogueIndex(self.records)
    result = make_diagnostic_verified_catalogue(index, backend=None)
    self.assertFalse(result.release_complete)
    self.assertEqual(tuple(result.index), self.records)
```

- [ ] **Step 2: Verify RED**

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_classify_request_orbits -v`.
Expected: import failure for `mathpsg.catalogue_loader`.

- [ ] **Step 3: Copy the three modules directly from the source tree**

Copy only these generic files. Do not copy offline atlas, release signing, or
container code. Make a standalone import adjustment only when the import test
demonstrates it is necessary.

- [ ] **Step 4: Verify GREEN**

Run the focused test plus `tests.test_extraction`; require every shipped Python
module to import and the source inventory to replay.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/catalogue_loader.py mathpsg/query.py mathpsg/certified_classifier.py
git add tests/test_classify_request_orbits.py EXTRACTED_SOURCES.json
git commit -m "feat: restore joint classifier orchestration"
```

### Task 2: Construct One Ordered Occupancy Request

**Files:**
- Create: `mathpsg/live_classify.py`
- Modify: `tests/test_classify_request_orbits.py`

**Interfaces:**
- Produces `resolve_occupancy_request(it_number, wps, *, igg,
  time_reversal, setting, catalogue) -> ClassificationRequest`.
- One `OrbitInstance` is created per input item.

- [ ] **Step 1: Write failing simultaneous and repeated occupancy tests**

```python
def test_a_and_b_are_one_joint_request(self):
    request = resolve_occupancy_request(
        99, ["a", "b"], igg="Z2", time_reversal=False,
        setting=None, catalogue=self.catalogue,
    )
    self.assertEqual(tuple(x.wyckoff_id for x in request.orbits),
                     (self.a_id, self.b_id))

def test_repeated_a_keeps_two_instances(self):
    request = resolve_occupancy_request(
        1, ["a", "a"], igg="Z2", time_reversal=False,
        setting=None, catalogue=self.catalogue,
    )
    self.assertNotEqual(request.orbits[0].instance_id,
                        request.orbits[1].instance_id)
    self.assertEqual(request.orbits[0].wyckoff_id,
                     request.orbits[1].wyckoff_id)
```

- [ ] **Step 2: Verify RED**

Expected: `resolve_occupancy_request` is unavailable.

- [ ] **Step 3: Implement validation and exact label resolution**

Accept one WP string or a nonempty finite sequence. Resolve `a` and `1a`
spellings against the selected setting, assign deterministic index-bearing
instance IDs, and construct exactly one `ClassificationRequest`.

- [ ] **Step 4: Verify invalid-input behavior**

Test invalid IT numbers, empty occupancy, unknown or ambiguous WPs, invalid IGG,
non-boolean time reversal, and mismatched settings.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/live_classify.py tests/test_classify_request_orbits.py
git add EXTRACTED_SOURCES.json
git commit -m "feat: construct joint occupancy requests"
```

### Task 3: Produce Grouped Host-Native GAP Source Evidence

**Files:**
- Create: `mathpsg/host_classifier_backend.py`
- Create: `tests/test_host_classifier_backend.py`

**Interfaces:**
- Produces `HostNativeSourceEvidence` and
  `build_host_source_evidence(records, *, runtime, time_reversal, timeout,
  repository_root)`.
- Consumes existing Task4 conversion and Task5 grouped batch exporters.

- [ ] **Step 1: Write the failing grouped-evidence test**

```python
def test_grouped_source_deduplicates_only_identical_inclusions(self):
    result = build_host_source_evidence(
        (self.a, self.a, self.b), runtime=self.runtime,
        time_reversal=False, timeout=300, repository_root=self.root,
    )
    self.assertEqual(result.instance_wyckoff_ids,
                     (self.a.wyckoff_id, self.a.wyckoff_id, self.b.wyckoff_id))
    self.assertEqual(result.unique_inclusion_ids,
                     (self.a.wyckoff_id, self.b.wyckoff_id))
    self.assertEqual(result.certification_status, "host-native")
```

- [ ] **Step 2: Verify RED**

Expected: import failure for `mathpsg.host_classifier_backend`.

- [ ] **Step 3: Implement grouped local GAP execution**

Run Task4 with `runtime.executable`, replay its affine/PCP certificate, then call
`export_gap_inclusion_batch_raw` with
`command=(runtime.executable, "-q")`. Replay the batch and every child, retain
the original instance-to-unique-inclusion map, and record host provenance.

- [ ] **Step 4: Test identity and corruption rejection**

Mutate an inclusion ID, Task4 digest, Task5 child, executable digest, or package
version and require failure before returning evidence.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/host_classifier_backend.py tests/test_host_classifier_backend.py
git add EXTRACTED_SOURCES.json
git commit -m "feat: build host-native classifier evidence"
```

### Task 4: Generate Local Z2 and U1 Branches On Demand

**Files:**
- Modify: `mathpsg/host_classifier_backend.py`
- Modify: `tests/test_host_classifier_backend.py`

**Interfaces:**
- Produces `HostNativeClassifierBackend.local_skeleton_plans(...)` returning
  exact `LocalSkeletonPlan` objects.
- Consumes replayed finite-group tables and existing Z2/U1 local enumerators.

- [ ] **Step 1: Write failing SG1 local-coverage tests**

```python
def test_z2_spatial_local_plans_are_exhaustive(self):
    plans = self.backend.local_skeleton_plans(
        self.z2_request, self.resolved_a, self.ambient, 300)
    self.assertGreater(len(plans), 0)
    self.assertEqual(len({p.plan.plan_digest for p in plans}), len(plans))
```

Add equivalent graded-Z2 and every-ambient-rho U1 assertions.

- [ ] **Step 2: Verify RED**

Expected: `local_skeleton_plans` raises `NotImplementedError`.

- [ ] **Step 3: Implement exhaustive on-demand enumeration**

Normalize the finite-group table, enumerate every requested Z2 branch, derive
one U1 skeleton per ambient coefficient sector, and wrap the values in existing
diagnostic evidence factories without weakening replay.

- [ ] **Step 4: Test incomplete/reordered/foreign-table failures**

Removing, duplicating, reordering, or rebinding branches must fail.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/host_classifier_backend.py tests/test_host_classifier_backend.py
git add EXTRACTED_SOURCES.json
git commit -m "feat: enumerate live local PSG branches"
```

### Task 5: Assemble One Joint Relative Z2 Layer

**Files:**
- Modify: `mathpsg/host_classifier_backend.py`
- Create: `tests/test_live_classify.py`

**Interfaces:**
- Produces working ambient, inclusion, and relative-layer plans for `igg="Z2"`.
- Returns one `JointLayerMaterial` covering the complete ordered occupancy.

- [ ] **Step 1: Write the failing real-GAP SG1 test**

```python
def test_sg1_a_z2_returns_complete_host_result(self):
    result = classify(1, ["a"], igg="Z2", cache=self.cache)
    self.assertEqual(result.certification_status, "host-native")
    self.assertIsInstance(result.class_count, int)
    self.assertGreater(result.class_count, 0)
```

Add a backend-spy test proving `["a", "a"]` invokes one relative solve with two
orbit instances.

- [ ] **Step 2: Verify RED**

Expected: the backend relative layer is unavailable.

- [ ] **Step 3: Implement Z2 ambient, inclusion, and joint relative plans**

Assemble and replay the ambient cochain complex and all inclusions from grouped
Task5 evidence. Build every diagnostic Z2 branch, call
`classify_z2_diagnostic`, construct quotient actions, and return one canonical
`JointLayerMaterial`.

- [ ] **Step 4: Verify simultaneous/repeated occupancy**

Assert one result for `["a", "b"]`, ordered instance details, and no invocation
of separate single-WP classification or count multiplication.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/host_classifier_backend.py tests/test_live_classify.py
git add EXTRACTED_SOURCES.json
git commit -m "feat: solve joint host-native Z2 classifications"
```

### Task 6: Assemble Every U1 Sector and Continuous Result

**Files:**
- Modify: `mathpsg/host_classifier_backend.py`
- Modify: `tests/test_live_classify.py`

**Interfaces:**
- Extends the relative layer for `igg="U1"`.
- Produces exhaustive `U1SectorCoverage` and exact finite or continuous strata.

- [ ] **Step 1: Write failing finite/continuous U1 tests**

```python
def test_continuous_u1_has_no_finite_count(self):
    result = classify(self.continuous_group, [self.wp], igg="U1", details=True)
    self.assertIsNone(result.class_count)
    self.assertTrue(result.continuous)
    self.assertTrue(result.details.presentations)
```

Also assert that every ambient coefficient character appears exactly once.

- [ ] **Step 2: Verify RED**

Expected: U1 relative-layer support is unavailable.

- [ ] **Step 3: Implement exhaustive joint U1 solving**

Enumerate ambient characters, restrict each to every occupied stabilizer,
assemble one joint relative problem per sector, call
`classify_u1_sector(..., allow_diagnostic=True)`, and build exact sector
coverage. Reject missing, duplicated, or failed sectors.

- [ ] **Step 4: Verify multi-WP U1 semantics**

Require the same simultaneous and repeated occupancy behavior as Z2 and exact
presentation details for continuous families.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/host_classifier_backend.py tests/test_live_classify.py
git add EXTRACTED_SOURCES.json
git commit -m "feat: solve joint host-native U1 classifications"
```

### Task 7: Publish Immutable Python Results and Cache Replay

**Files:**
- Modify: `mathpsg/live_classify.py`
- Modify: `mathpsg/__init__.py`
- Modify: `mathpsg/solver_status.py`
- Modify: `tests/test_live_classify.py`

**Interfaces:**
- Produces `classify(it_number, wps, *, igg="Z2", time_reversal=False,
  setting=None, details=False, gap="gap", cache=None, timeout=300) ->
  HostNativeClassificationResult`.
- The immutable result exposes request, count/continuity, summaries or details,
  and host status. Runtime provenance remains internal to execution and cache
  validation.

- [ ] **Step 1: Write failing export and isolation tests**

```python
def test_public_classify_is_exported_and_returns_fresh_values(self):
    from mathpsg import classify
    first = classify(1, ["a"], igg="Z2", cache=self.cache)
    second = classify(1, ["a"], igg="Z2", cache=self.cache)
    self.assertEqual(first, second)
    self.assertIsNot(first, second)
    self.assertFalse(hasattr(first, "backend"))
```

- [ ] **Step 2: Verify RED**

Expected: `mathpsg.classify` is absent.

- [ ] **Step 3: Implement lazy orchestration and public conversion**

Probe GAP, build the live/verified catalogue and backend, construct one request,
run `classify_request`, require a complete layer, and copy it into frozen public
dataclasses. Count finite quotient members exactly; use `None` when an accepted
U1 stratum is continuous.

- [ ] **Step 4: Implement and verify cache replay**

Bind ordered instances, source inventory, runtime provenance, algorithm
versions, and parent digests. Test byte-identical reuse plus failures for cache
mutation and runtime/source drift.

- [ ] **Step 5: Commit**

```bash
git add mathpsg/live_classify.py mathpsg/__init__.py mathpsg/solver_status.py
git add tests/test_live_classify.py EXTRACTED_SOURCES.json
git commit -m "feat: expose host-native PSG classify API"
```

### Task 8: Add CLI, Documentation, and Full Verification

**Files:**
- Modify: `mathpsg/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `VERIFICATION.md`

**Interfaces:**
- Produces `mathpsg classify --it-number N --wps ... --igg Z2|U1` with canonical
  JSON matching the Python result.

- [ ] **Step 1: Write the failing CLI test**

```python
def test_classify_cli_matches_python_result(self):
    code, stdout, stderr = self.run_cli(
        "classify", "--it-number", "1", "--wps", "a", "--igg", "Z2")
    self.assertEqual((code, stderr), (0, ""))
    self.assertEqual(json.loads(stdout)["certification_status"], "host-native")
```

- [ ] **Step 2: Verify RED**

Expected: argparse rejects `classify`.

- [ ] **Step 3: Implement CLI and documentation**

Add all designed flags, serialize only public values, replace the README scope
warning with truthful calculator examples, and record exact runtime versions
and commands in `VERIFICATION.md`.

- [ ] **Step 4: Run focused and complete verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error::ResourceWarning -m unittest \
  tests.test_classify_request_orbits tests.test_host_classifier_backend \
  tests.test_live_classify tests.test_cli -v
PYTHONDONTWRITEBYTECODE=1 python3 -W error::ResourceWarning \
  -m unittest discover -s tests -v
```

Run representative real-GAP Z2/U1, spatial/onsite-time, and
single/simultaneous/repeated queries. Run the catalogue-wide uniform-path smoke;
typed failures are diagnostics, not successful classifications.

- [ ] **Step 5: Independent review and final commit**

Request review of mathematical completeness, occupancy semantics, cache
authority, subprocess bounds, and public claims. Fix each important finding
through a new failing test, rerun verification, regenerate the inventory, and
commit.

```bash
git add mathpsg tests README.md VERIFICATION.md EXTRACTED_SOURCES.json
git commit -m "docs: publish host-native PSG calculator"
```
