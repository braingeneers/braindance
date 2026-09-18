# validate_tutorial_data.py

**Path:** `braindance/examples/validate_tutorial_data.py`
**Module:** `braindance.examples.validate_tutorial_data`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Downloads or opens a tutorial bundle through the public tutorial API and validates it through normal catalog/Recording access. It checks frozen counts, dimensions, ordering, ranges, mapping/stimulation data, and byte-level array hashes.

## Connections
- **Used by:** `braindance.examples.get_tutorial_data` — import consumer hint; not a proven runtime call.
- **Uses:** `get_test_data` from `braindance.tutorial` — imports (static evidence).
- **Uses:** `bin_spike_data_vectorized` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_catalog` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** Calls `tutorial.get_test_data`, `load_catalog`, and `bin_spike_data_vectorized`; then exercises Recording spikes, stimulation log, and mapping properties.

## Dependencies
- `braindance.tutorial.get_test_data` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.bin_spike_data_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_catalog` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `main(name='closed-loop-small', version=None, cache_dir=None, offline=False, update=False, manifest_path=None)`
> Validate a named tutorial dataset against its frozen reference through the same APIs used by users and return per-recording reports.
> **Called by:** braindance/examples/get_tutorial_data.py:13 (named-call hint); braindance/examples/validate_tutorial_data.py:100 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/validate_tutorial_data.py:10`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI flags: `--name`, `--version`, `--cache-dir`, `--manifest-path`, `--offline`, and `--update`. |

## Data Shapes
- Binned spike counts must be time bins x neurons and sum to the per-neuron spike count inside `[0, duration_ms)`.
- Reference JSON contains a `recordings` list with identifiers, expected scalar counts, `bin_ms`, and SHA-256 digests for canonical little-endian arrays.

## Notes
- Every Recording cache is cleared in a `finally` block.
- The module is retained for compatibility; its docstring points users to `braindance.examples.get_tutorial_data --validate`.
