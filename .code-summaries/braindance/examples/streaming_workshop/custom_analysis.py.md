# custom_analysis.py

**Path:** `braindance/examples/streaming_workshop/custom_analysis.py`
**Module:** `braindance.examples.streaming_workshop.custom_analysis`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Defines participant analysis decorators, static contract extraction and the AnalysisPhaseV3 execution adapter. Canonical inputs/outputs declarations drive builder metadata and runtime data checks while legacy requires/provides spellings remain accepted.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.code_export` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.profiles` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** experiment_spec and native_runner resolve analysis contracts before dependency validation; both may execute CustomAnalysisPhase.
- **Shared data:** CustomAnalysisPhase inherits AnalysisPhaseV3 and needs no acquisition environment.

## Dependencies
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.

## Classes
### CustomAnalysisPhase(AnalysisPhaseV3)
> Run a one-shot participant function in the V3 data/lifecycle contract.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:129`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/experiment_spec.py:172 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:162 (named-call hint); braindance/examples/streaming_workshop/session.py:493 (named-call hint)
**Constructor:** `__init__(self, spec, config, module=None, code=None)`
**Key attributes:**
None
**Methods:**
#### `run(self, experiment)`
> Load the active function, enforce inputs, overlay params temporarily and validate returned outputs.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:144`

## Functions
### `analysis_phase(*, inputs=None, outputs=None, requires=(), provides=())`
> Attach canonical input/output metadata to a synchronous experiment callback while accepting old keyword names.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:12`
### `analysis_definition(params)`
> Normalize old parameter names and validate custom-analysis names and distinct string-list contracts.
> **Called by:** braindance/examples/streaming_workshop/custom_analysis.py:100 (named-call hint); braindance/examples/streaming_workshop/custom_analysis.py:140 (named-call hint); braindance/examples/streaming_workshop/custom_analysis.py:83 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:124 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:65 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:37 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:30`
### `analysis_contracts(code)`
> Parse literal decorators and supported function signatures without importing participant code.
> **Called by:** braindance/examples/streaming_workshop/custom_analysis.py:120 (named-call hint); braindance/examples/streaming_workshop/custom_analysis.py:94 (named-call hint); braindance/examples/streaming_workshop/main.py:222 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:51`
### `verify_analysis_code(code, phases)`
> Ensure every selected custom-analysis phase has a valid declared function.
> **Called by:** braindance/examples/streaming_workshop/code_export.py:121 (named-call hint); braindance/examples/streaming_workshop/custom_analysis.py:117 (named-call hint); braindance/examples/streaming_workshop/main.py:235 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:153 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:92`
### `resolve_analysis_config(config, code=None)`
> Replace persisted contract caches in copied phases/phase_plan with the current code declarations.
> **Called by:** braindance/examples/streaming_workshop/code_export.py:137 (named-call hint); braindance/examples/streaming_workshop/code_export.py:15 (named-call hint); braindance/examples/streaming_workshop/custom_analysis.py:139 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:80 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:17 (named-call hint); braindance/examples/streaming_workshop/profiles.py:64 (named-call hint); braindance/examples/streaming_workshop/profiles.py:80 (named-call hint); braindance/examples/streaming_workshop/session.py:281 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/custom_analysis.py:108`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| @analysis_phase(inputs=..., outputs=...) marks synchronous function(exp)->dict callbacks; requires/provides are legacy aliases. |
| Analysis params include function_name (default custom_analysis), inputs, outputs (default analysis_result) and legacy parameter_keys. |
| Static parsing accepts at most 100000 code characters and literal list/tuple contracts only. |

## Data Shapes
- analysis_contracts returns {contracts: {function_name: {inputs: [...], outputs: [...]}}, errors: [...]}.
- Resolved configuration copies phases and phase_plan, replacing cached contract params with declarations read from participant code.
- Runtime reads named input keys from experiment.data and requires a returned dictionary containing every declared output.

## Notes
- Static inspection uses AST without executing participant code; runtime compiles/executes functions_file in a fresh module when no module/provider is supplied.
- Constructor and each run use the active decorated function contract, so hot reload supersedes stale builder declarations.
- Callback params temporarily merge config, native_settings, scoped settings and original experiment params (highest priority); original params are restored even on error.
- The static parser rejects simultaneous canonical and legacy spelling for the same contract; runtime decorator canonical values take precedence when supplied.
