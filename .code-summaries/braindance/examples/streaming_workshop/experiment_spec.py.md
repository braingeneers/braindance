# experiment_spec.py

**Path:** `braindance/examples/streaming_workshop/experiment_spec.py`
**Module:** `braindance.examples.streaming_workshop.experiment_spec`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Defines browser metadata and preflight validation for the reusable binned V3 recording, response-probe and mapped-environment phases plus custom analysis. Scientific native sequences are delegated to native validation while workshop phase type IDs remain stable for saved plans.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.code_export` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `DataContext` from `braindance.core.phases_v3.data_context` — imports (static evidence).
- **Uses:** `PhaseValidator` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `CustomAnalysisPhase` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `analysis_definition` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `resolve_analysis_config` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `data_metadata` from `braindance.examples.streaming_workshop.data_contracts` — imports (static evidence).
- **Uses:** `verify_native` from `braindance.examples.streaming_workshop.native_runner` — imports (static evidence).
- **Uses:** `LEGACY_OUTPUTS` from `braindance.examples.streaming_workshop.session` — imports (static evidence).
- **Uses:** `build_phase` from `braindance.examples.streaming_workshop.session` — imports (static evidence).
- **Uses:** `source_url` from `braindance.examples.streaming_workshop.source_links` — imports (static evidence).
- **Shared data:** session.build_phase instantiates reusable core phases and supplies the legacy output mapping for compatibility.
- **Shared data:** PhaseValidator checks projected phase input availability one phase at a time; DataContext holds initial and projected values.
- **Shared data:** native_runner.verify_native handles any native-prefixed sequence or a nonempty all-custom-analysis sequence.
- **Shared data:** custom_analysis owns participant contracts; data_contracts adds type/shape metadata and source_links references core phase definitions.

## Dependencies
- `braindance.core.phases_v3.data_context.DataContext` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseValidator` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.CustomAnalysisPhase` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.analysis_definition` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.resolve_analysis_config` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.data_contracts.data_metadata` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_runner.verify_native` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.session.LEGACY_OUTPUTS` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.session.build_phase` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.source_links.source_url` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `phase_catalog()`
> Describe supported binned phases and custom analysis using canonical V3 data contracts and UI defaults.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:121 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:54 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:61 (named-call hint); braindance/examples/streaming_workshop/main.py:123 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/experiment_spec.py:10`
### `normalize_phases(config, skip=False)`
> Choose the default/skip sequence or validate saved interactive phase identifiers, types and parameter names.
> **Called by:** braindance/examples/streaming_workshop/code_export.py:22 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:91 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/experiment_spec.py:44`
### `uses_native_runner(config)`
> Select native execution for native-prefixed or exclusively custom-analysis sequences.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:83 (named-call hint); braindance/examples/streaming_workshop/main.py:292 (named-call hint); braindance/examples/streaming_workshop/main.py:70 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/experiment_spec.py:70`
### `verify_spec(config, skip=False, code=None)`
> Resolve code contracts and initial baselines, validate mappings/timing/dependencies and return UI-ready phase reports.
> **Called by:** braindance/examples/streaming_workshop/code_export.py:138 (named-call hint); braindance/examples/streaming_workshop/main.py:232 (named-call hint); braindance/examples/streaming_workshop/session.py:284 (named-call hint); braindance/examples/streaming_workshop/session.py:335 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/experiment_spec.py:75`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default sequence: recording, causal, environment; skip returns environment only. Explicit sequences require 1–32 uniquely identified phases. |
| Recording default 3 seconds; response probes default 4 repeats and stimulation electrodes [0,1]. Environment defaults 120 seconds, 10-second episodes, CartPole, encoder_gain=100 and decoder_gain=.025. |
| baseline_hz, calibration, channels, detection, source/live_config and grid_shape control initial-data and mapping checks; duration values require positive 20 ms increments. |

## Data Shapes
- Catalog exposes canonical inputs/outputs and metadata plus compatibility requires/provides aliases.
- recording produces recording_baseline_hz; probes consume baseline and produce response_probe_hz/response_probe_trials; environment produces environment_episodes/environment_reward.
- Normalized phase specs contain id/type/params. Validation rows include resolved params, input origins and output names, with compatibility requirements/provides fields.

## Notes
- There is no artificial workshop configuration input dependency. Legacy output names are resolved through session.LEGACY_OUTPUTS for existing participant code.
- Baseline values must be finite nonnegative rates; synthetic nonsorted sources additionally require one value per channel.
- Sensory observation index defaults to 2 for CartPole and 0 for other games. Decoder channel pools default to channel halves and must not overlap.
- Synthetic spatial electrode validation uses physical row spacing of 220. Stimulating binned phases require exactly two unique electrode IDs.
- Preflight projects output presence, not computed scientific values; calibration contents are checked during source preflight.
- Binned phase construction is skipped once preflight errors have been collected, so invalid settings are returned as validation messages instead of triggering constructor exceptions.
