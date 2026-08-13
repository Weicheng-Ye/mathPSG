# MathPSG

MathPSG computes Z2 and compact-U1 projective symmetry group (PSG)
classifications for occupied Wyckoff positions in three-dimensional space
groups. It is a Python library with one public calculation function:

```python
import mathpsg

result = mathpsg.classify(1, ["a"], igg="Z2")
```

Every result is calculated locally from the requested space group and
occupancy. The package does not contain a precomputed classification atlas and
does not provide a command-line interface.

## What `classify` computes

For one joint occupancy request, `classify`:

1. obtains the space-group and stabilizer data from local GAP;
2. enumerates every local PSG branch at every occupied Wyckoff position;
3. solves the joint Z2 or U1 extension equations;
4. quotients the solution spaces by gauge and residual symmetries; and
5. counts the finite orbits or identifies continuous U1 families.

GAP must exist, terminate successfully, and return valid JSON. MathPSG does
not require an exact GAP/package version, replay certificates, validate a
source-tree hash inventory, or keep a persistent result cache.

## Requirements

- Python 3.11 or newer.
- GAP on `PATH`, or an executable supplied with `gap=...`.
- GAP packages used by the calculation: Cryst, HAP, HAPcryst, and json.

Package versions are not pinned. If GAP or a required package cannot perform
the calculation, `classify` raises `ClassificationError`.

## Installation

```bash
git clone https://github.com/Weicheng-Ye/mathPSG.git
cd mathPSG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The package has no third-party Python runtime dependency.

## Python API

```python
mathpsg.classify(
    it_number,
    wps,
    *,
    igg="Z2",
    time_reversal=False,
    setting=None,
    details=False,
    cochain=False,
    gap="gap",
    timeout=300,
)
```

### Arguments

| Argument | Meaning |
|---|---|
| `it_number` | International Tables space-group number, normally `1` through `230`. |
| `wps` | One Wyckoff label or an ordered iterable of labels, such as `"a"`, `["a", "b"]`, or `["a", "a"]`. |
| `igg` | Invariant gauge group, `"Z2"` or `"U1"`. Letter case and surrounding whitespace are normalized. |
| `time_reversal` | Whether to include onsite time reversal. |
| `setting` | A specific space-group setting, or `None` when the requested labels select a unique setting. |
| `details` | Include the physical presentation of every nonempty solution stratum when truthy. |
| `cochain` | Add an explicit affine basepoint and one representative per mathematical solution-basis generator to every detail stratum. This also enables `details`. |
| `gap` | GAP executable name or path. |
| `timeout` | Maximum time in seconds for each GAP invocation, not an overall calculation deadline. |

The Wyckoff list is one simultaneous physical occupancy. For example,
`["a", "b"]` performs one joint relative classification; it does not multiply
two independent answers. Repeated labels remain distinct occupied orbits, so
`["a", "a"]` also denotes one joint problem with two instances.

There is no `cache` argument and no `validate` argument.

## Result object

`classify` returns an immutable `ClassificationResult` with exactly five
attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `request` | immutable mapping | The effective physical request: `space_group`, `setting`, `igg`, `time_reversal`, and `wps`. |
| `class_count` | `int \| None` | Number of unframed PSG classes when the quotient is finite; `None` if continuous families occur. |
| `continuous` | `bool` | Whether the final quotient contains at least one continuous family. |
| `summaries` | tuple of immutable mappings | One compact physical summary for each nonempty solution stratum. |
| `details` | immutable mapping or `None` | Full physical solution presentations when `details=True` or `cochain=True`; otherwise `None`. |

The result does not contain runtime measurements, certification status,
certificate hashes, source hashes, cache metadata, backend objects, replay
records, or opaque stratum/skeleton identifiers.

Mappings support normal indexing:

```python
result.request["space_group"]
result.summaries[0]["kind"]
```

### `summaries`

A summary describes one solution family (a stratum), not necessarily one PSG
class. Therefore `len(result.summaries)` is generally not the same as
`result.class_count`.

For a Z2 stratum:

```python
{
    "kind": "finite-affine-z2",
    "dimension": 3,
    "framed_finite_cardinality": 8,
    "unframed_finite_cardinality": 8,
}
```

- `dimension` is the dimension of the affine solution space over GF(2) after
  gauge quotienting.
- `framed_finite_cardinality` is the number of points before residual
  identifications.
- `unframed_finite_cardinality` is the number remaining after residual
  symmetries act within that stratum.

For a U1 stratum:

```python
{
    "kind": "compact-u1-torsor",
    "free_rank": 3,
    "torsion_orders": (),
}
```

- `free_rank` counts independent continuous U1 parameters.
- `torsion_orders` gives the cyclic finite factors.
- A rank-zero stratum also has `finite_class_count`, the product of its torsion
  orders.

### `details`

When requested, details have two keys:

```python
{
    "strata": (...),
    "quotient": {...},
}
```

`strata` contains one physical presentation per nonempty branch. `quotient`
contains:

- `framed_finite_cardinality`;
- `unframed_finite_cardinality`; and
- `continuous_family_count`.

For a continuous result the two finite aggregate cardinalities are `None`.

Each finite affine Z2 detail contains:

- `kind`;
- `basepoint`, one binary solution;
- `quotient_basis`, the independent binary solution directions;
- `dimension`;
- `framed_finite_cardinality`; and
- `unframed_finite_cardinality`.

Each compact-U1 detail contains:

- `kind`;
- `rho_bits`, the coefficient-character sector;
- `basepoint_phases`, exact phases written as reduced strings;
- `free_rank`;
- `torsion_orders`;
- `formal_parameters`, named `phi0`, `phi1`, ...; and
- `finite_class_count` when `free_rank == 0`.

### Cochain basis representatives

With `cochain=True`, every detail stratum also contains a `cochain` mapping.
Its coordinate blocks use the relative-complex order

```text
ambient C^2, local C^1 for orbit 0, local C^1 for orbit 1, ...
```

The occupied-orbit index is explicit, so repeated Wyckoff labels remain
distinct. For Z2, `basepoint` is one affine solution and each order-two basis
entry contains a homogeneous `direction` and the corresponding affine
`representative`. Binary combinations of these directions give one
representative of every unframed residual-equivalence class. The sibling
`quotient_basis` remains the framed basis before residual translations.

For U1, `basepoint_phases` and `weyl_shift_phases` are exact reduced phase
strings. A free basis entry supplies the integer coefficient vector of a named
parameter, while a torsion entry supplies its cyclic order, exact phase
direction, and affine representative. Thus every framed cochain has the form

```text
z(phi,t) = z0 + sum_i phi_i f_i + sum_j t_j tau_j  (mod 1).
```

The U1 Weyl quotient is generally not an abelian group, so the reported basis
is the framed compact-torsor basis together with the shift in the Weyl action
`z -> -z + weyl_shift`; it is not described as a basis of the final unframed
orbit space. The output contains no
certificates, hashes, replay records, or cached data.

## Examples

### Finite Z2 classification

```python
import mathpsg

result = mathpsg.classify(1, ["a"], igg="Z2")

print(result.class_count)       # 8
print(result.continuous)        # False
print(len(result.summaries))    # 1
print(result.summaries[0]["kind"])       # finite-affine-z2
print(result.summaries[0]["dimension"])  # 3
print(result.details)           # None
```

One summary here contains all eight unframed classes.

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
print(len(result.details["strata"]))  # 2
print([s["dimension"] for s in result.details["strata"]])
# [6, 6]
print(result.details["quotient"]["unframed_finite_cardinality"])
# 128
```

The two strata are inequivalent local time-reversal branches; each has 64
unframed points.

### Continuous U1 classification

```python
result = mathpsg.classify(1, ["a"], igg="U1", details=True)

print(result.class_count)  # None
print(result.continuous)   # True

continuous_strata = [
    stratum
    for stratum in result.details["strata"]
    if stratum["free_rank"] > 0
]
print(continuous_strata[0]["free_rank"])         # 3
print(continuous_strata[0]["formal_parameters"]) # ("phi0", "phi1", "phi2")
```

Rank-zero U1 strata may coexist with a continuous stratum. Once any continuous
family is present, the aggregate `class_count` is `None`.

### Multiple occupied Wyckoff positions

```python
result = mathpsg.classify(2, ["a", "b"], igg="Z2")
print(result.class_count)  # 128
```

This is one coupled classification with both occupied orbits.

### Explicit GAP and setting

```python
result = mathpsg.classify(
    3,
    ["a"],
    setting="b",
    igg="Z2",
    gap="/absolute/path/to/gap",
    timeout=900,
)
```

## Errors

`ClassificationError` reports calculation failures such as:

- GAP cannot be found or started;
- GAP times out or exits unsuccessfully;
- GAP output is not valid JSON;
- required GAP computation data are missing; or
- the requested Wyckoff labels do not select a unique setting.

These are operational errors, not partial classification results.

## License

See [LICENSE](LICENSE).
