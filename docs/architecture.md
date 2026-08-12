# Architecture and scope

The public path is intentionally short:

```text
IT number
  -> local GAP + Cryst exporter
  -> copied exact affine normalizer
  -> reviewed display crosswalk
  -> canonical external catalogue cache
  -> copied affine/PCP GAP exporter
  -> independent Python certificate replay
  -> host-native evidence summary
```

`mathpsg.local_gap` resolves and hashes the executable and probes GAP, Cryst,
HAP, HAPcryst, json, and io. `mathpsg.live_catalogue` generates one group only
when requested, matches exact geometry to display metadata by persistent
Wyckoff ID, and never guesses letters. `mathpsg.live_evidence` is a thin copy of
the generic catalogue-to-GAP request adapter plus local process wiring; the
mathematical affine/PCP conversion and replay logic stays in the copied
`gap_classifier` module.

All groups and both modes use this same path. There is no benchmark dispatch.
The repository includes generic Z2/U1 algebraic source modules but lacks the
reviewed adapter that would turn this live evidence into their final typed
inputs. Consequently the public API stops at verified conversion evidence and
does not claim PSG class counts.

Generated data is never written into the source tree. Host provenance binds
the full source inventory and exact runtime observations, but it is
reproducibility metadata rather than a hermetic certification claim.
