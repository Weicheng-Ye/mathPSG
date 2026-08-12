# Verification record

Verified on 2026-08-12 in the standalone build tree before the final Downloads
copy.

## Runtime observed

- CPython 3.14.6
- GAP 4.15.1
- Cryst 4.1.30
- HAP 1.70
- HAPcryst 0.1.15
- GAP json 2.2.3
- GAP io 4.9.3
- GAP executable SHA-256:
  `6651f91b7054facb69fb7df968bed8116035bab2e2f3bb5e8fe8f9d1ff2725a6`

## Reviewed calculator boundaries

The implementation and focused regressions establish:

- exact live catalogue resolution before solver execution;
- one grouped local-GAP ambient/inclusion calculation;
- exhaustive spatial and onsite-time Z2 local branches;
- exhaustive spatial and onsite-time U1 coefficient-character sectors;
- one joint relative solve for ordered simultaneous/repeated occupancy;
- exact finite unframed class counts and continuous-U1 presentations;
- diagnostic host-native Weyl evidence that cannot cross a release-only gate;
- pure replay of content-addressed cache evidence without rerunning the GAP
  catalogue, Task4, or Task5 solver jobs; the selected GAP runtime is freshly
  reprobed to verify its executable and package identity;
- immutable public results with no retained backend, Task5 authority, or
  runtime-provenance record;
- internal runtime, package, executable, and installed-source provenance for
  cache identity and replay.

## Real local-GAP regressions completed

- SG 1 Z2, spatial, one occupied orbit.
- SG 1 Z2, spatial, repeated `['a', 'a']` occupancy.
- SG 1 Z2, onsite-time.
- SG 2 Z2, distinct `['a', 'b']` and `['b', 'a']` occupancy.
- SG 1 U1, spatial, exhaustive sector coverage.
- SG 1 U1, spatial, repeated `['a', 'a']` occupancy.
- SG 1 U1, onsite-time, exhaustive sector coverage.
- SG 2 U1, spatial, distinct `['a', 'b']` joint occupancy.
- public Python `classify()` finite/count, details, cache replay, and U1
  continuity behavior.

An installed-layout simulation copied only the `mathpsg` package into a fresh
site directory, changed the working directory outside the repository, and ran:

```bash
python3 -c 'from mathpsg import classify'
python3 -m mathpsg capabilities
python3 -m mathpsg classify --it-number 1 --wps a --igg Z2
```

The real CLI query completed with exact unframed class count `8`,
`continuous: false`, host-native status, and the runtime versions/digests
recorded above. This verifies that runtime assets are loaded relative to the
installed package rather than the current directory.

The SG 2 U1 joint integration completed in 636.609 seconds. It inspected the
typed relative artifact and proved exactly one relative plan call, two ordered
local inputs/bindings/restrictions in every coefficient sector, and two ordered
diagnostic Weyl evaluator rows.

## Focused suite results

Recorded green runs during implementation include:

- 35 orchestration, host-evidence, and extraction tests in 78.884 seconds;
- 25 request/public-result/extraction tests in 104.189 seconds;
- 4 public Python calculator tests in 85.160 seconds;
- targeted live U1 single/repeated/order tests, 3 tests in 60.050 seconds.

All completed reviews reported no remaining Critical or Important findings at
their approved boundaries.

## Authority statement

These are host-native, replay-checked calculations. They are not signed and do
not claim release-certified authority. No Docker image, precomputed
classification atlas, PyXtal dependency, or group-specific benchmark path is
used.

## Coverage statement

The code exposes the same live path for IT numbers 1 through 230, but this
record does not claim that every possible IT-number/setting/occupancy query has
been exhaustively swept. A query is successful only when every required local
branch or coefficient sector, inclusion, relative solve, residual action, and
quotient replay completes; otherwise the API fails without returning a partial
count.
