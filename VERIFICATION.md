# Verification record

Verified on 2026-08-12 in `/private/tmp/mathpsg-standalone-build` before the
final Downloads copy.

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

## Automated suite

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error::ResourceWarning \
  -m unittest discover -s tests -v
```

Result: 46 tests passed in 52.217 seconds; zero failures, errors, or skips.

The suite covers extraction boundaries, source-inventory replay, runtime probe
parsing and a real probe, live SG 1 generation/cache reuse, Task4 conversion
evidence in both modes, CLI behavior, the explicit solver boundary, and focused
exact U1 local algebra/certificate replay.

An installed-layout simulation then copied only the `psgmath` package into a
fresh directory, set that directory as `PYTHONPATH`, changed the working
directory outside the repository, and successfully ran `doctor`,
`capabilities`, SG 1 `catalogue`, and both SG 1 `evidence` modes. This exercises
the packaged GAP scripts, classifier libraries, display crosswalk, and the
installed-source provenance fallback without source-tree adjacency.

## Real local-GAP smoke commands

The following commands ran against one fresh external cache:

```bash
python3 -m psgmath doctor
python3 -m psgmath catalogue --it-number 1 --cache CACHE
python3 -m psgmath catalogue --it-number 70 --cache CACHE
python3 -m psgmath catalogue --it-number 227 --cache CACHE
python3 -m psgmath evidence --it-number 1 --mode spatial --cache CACHE
python3 -m psgmath evidence --it-number 1 --mode onsite-time --cache CACHE
```

Observed catalogue counts were SG 1: 1, SG 70 setting 2: 8, and SG 227
setting 2: 9. Spatial and onsite-time evidence each covered the same SG 1
member and replayed successfully. Their request digests were respectively:

- `sha256:5476517b447dde0f542a0c2f1033942ba32a0ed2a6875feb3644ea2fe026dda6`
- `sha256:16c2461ad7ed809e6b1832343233b7e29b4e09d9c860e119dd1bb2d6cf930038`

Both returned the affine certificate digest
`sha256:f2047f6019a195e19430a0070dc673da507bbf1c578af4b225f985bee16314d7`
and explicitly reported `release_certified: false`.

## Scope not claimed

This verification did not run an exhaustive sweep over all 230 IT numbers and
does not claim final Z2/U1 PSG class counts. The copied snapshot lacks the
reviewed live-evidence-to-final-solver bridge, and this repository intentionally
does not invent it or expose a `classify` command.

## Tree measurements

The final pre-commit source tree excluding `.git` contained 79 files and
7,089,739 bytes: 39 Python files (1,514,240 bytes), 20 GAP files including the
package-local runtime copies (206,236 bytes), and 14 packaged/resource files
(5,385,893 bytes). Generated caches and bytecode were absent. The final
inventory is regenerated after this record and all review fixes.
