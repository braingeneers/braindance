# native_catalog.py

**Path:** `braindance/examples/streaming_workshop/native_catalog.py`
**Module:** `braindance.examples.streaming_workshop.native_catalog`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Statically discovers V3 scientific phases and constructor contracts without importing optional scientific dependencies. The default browser catalog exposes seven supported phases while legacy classes remain available for validation and execution.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Uses:** `data_metadata` from `braindance.examples.streaming_workshop.data_contracts` — imports (static evidence).
- **Uses:** `source_url` from `braindance.examples.streaming_workshop.source_links` — imports (static evidence).
- **Shared data:** native_runner uses the complete catalog to verify and instantiate saved native phase specifications.
- **Shared data:** data_contracts.data_metadata supplies browser-facing input/output shapes; source_links.source_url provides source links.

## Dependencies
- `braindance.examples.streaming_workshop.data_contracts.data_metadata` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.source_links.source_url` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `_catalog()`
> Discover and cache inherited phase contracts, constructor schemas and known scientific caveats using AST only.
> **Called by:** braindance/examples/streaming_workshop/native_catalog.py:132 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:150 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:152 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:200 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_catalog.py:18`
### `native_catalog(include_legacy=False)`
> Return a detached browser metadata catalog, optionally including legacy research classes.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:186 (named-call hint); braindance/examples/streaming_workshop/main.py:123 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:159 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:20 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_catalog.py:129`
### `validate_native_parameters(spec)`
> Reject unknown or missing constructor arguments, nonfinite JSON and simple type mismatches without importing phase implementations.
> **Called by:** braindance/examples/streaming_workshop/native_catalog.py:197 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:47 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_catalog.py:144`
### `instantiate_native(spec)`
> Validate a saved specification, dynamically import its phase class, supply the phase ID as name when supported and invoke its constructor.
> **Called by:** braindance/examples/streaming_workshop/native_runner.py:163 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_catalog.py:195`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `native_catalog(include_legacy=False)` exposes RecordPhaseV3, FrequencyStimPhaseV3, NeuralSweepPhaseV3, RTSortPhaseV3, CartPolePhase, FoodLandPhaseV3 and AntPhaseV3; true returns the complete discovered catalog. |
| Discovery scans core/phases_v3/phases*.py, excluding phases_binned.py which has a separately registered acquisition runtime. |

## Data Shapes
- Catalog maps native:<module>.<class> to constructor params/defaults/type annotations, required_params, canonical inputs/outputs, category, source_url, mapping_contract, caveats and data metadata.
- Mapping contracts specify sensory counts or n_features parameter and minimum motor/training neuron counts.

## Notes
- An lru_cache retains AST discovery for the process; callers receive deep copies.
- Legacy requires/provides assignments normalize to inputs/outputs. Computed contracts and nonliteral constructor defaults carry caveats rather than execution.
- Validation checks finite JSON, supported arguments and simple types, not dependency availability or actual array dimensions. Fractional duration/max_time values remain valid despite old int annotations.
- instantiate_native imports the selected class and calls its constructor; resource allocation occurs only then.
