# phase_base_v3.py

**Path:** `braindance/core/phases_v3/phase_base_v3.py`
**Module:** `braindance.core.phases_v3.phase_base_v3`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Defines the V3 phase lifecycle and canonical inputs/outputs data contracts, with synchronized requires/provides compatibility aliases. Includes analysis-only phases, shared-environment groups, ordered dependency validation and callable phase adapters.

## Connections
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_selection` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_binned` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.2_rapid_pairing` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.analysis_checkpoint_smoke` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.simple_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.custom_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.labyrinth` — import consumer hint; not a proven runtime call.
- **Uses:** `DataContext` from `braindance.core.phases_v3.data_context` — imports (static evidence).
- **Shared data:** Experiment validates pipelines through PhaseValidator, configures phase inputs from DataContext and calls lifecycle hooks.
- **Shared data:** AnalysisPhaseV3 suppresses environment creation; PhaseGroup exposes children to the runner and validator.

## Dependencies
- `braindance.core.phases_v3.data_context.DataContext` — intra-repo import; source import evidence.

## Classes
### _PhaseContractMeta(ABCMeta)
> Synchronize canonical and historical contract names across class definitions and later class assignment.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:15`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `__new__(mcls, name, bases, namespace, **kwargs)`
> Normalize subclass contracts and reject conflicting duplicate declarations.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:18`
#### `__setattr__(cls, name, value)`
> Keep aliases synchronized when a class contract is reassigned.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:28`
### PhaseV3(ABC)
> Enhanced base phase class with dependency management and lifecycle hooks.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:37`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name: str=None, suffix: str='', recording_tag: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_env_time_origin` | inferred at runtime | `None` |
| `env` | inferred at runtime | `None` |
| `experiment` | inferred at runtime | `None` |
| `inputs` | inferred at runtime | `self.inputs` |
| `metadata` | inferred at runtime | `{}` |
| `name` | inferred at runtime | `name or self.__class__.__name__` |
| `outputs` | inferred at runtime | `self.outputs` |
| `recording_tag` | inferred at runtime | `recording_tag` |
| `start_time` | inferred at runtime | `None` |
| `suffix` | inferred at runtime | `suffix` |
**Methods:**
#### `__setattr__(self, name, value)`
> Copy assigned contract lists and synchronize their canonical and historical instance attributes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:65`
#### `needs_environment(self) -> bool`
> Override to specify if this phase needs a Maxwell environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:73`
#### `close_environment_after(self) -> bool`
> Override to specify if environment should be closed after this phase.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:77`
#### `set_experiment(self, experiment)`
> Set experiment reference.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:81`
#### `set_env(self, env)`
> Set environment reference.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:85`
#### `validate_requirements(self, data_context) -> bool`
> Reject missing input keys before scientific execution.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:92`
#### `configure_from_experiment(self, experiment)`
> Expose declared input values on the phase instance from shared experiment data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:103`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Customize Maxwell environment parameters. Override to modify environment creation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:113`
#### `run(self, experiment) -> Dict[str, Any]`
> Execute the phase logic.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:121`
#### `cleanup(self)`
> Cleanup after phase execution. Override if needed.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:133`
#### `time_elapsed(self) -> float`
> Get elapsed time since phase started.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:137`
#### `info(self) -> Dict[str, Any]`
> Describe canonical contracts alongside compatibility aliases and metadata.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:145`
### AnalysisPhaseV3(PhaseV3)
> Base class for analysis phases that don't need Maxwell environments.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:157`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `needs_environment(self) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:162`
#### `close_environment_after(self) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:165`
### PhaseGroup()
> Container for phases that should share the same environment/save file.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:169`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/experiment_v3.py:538 (named-call hint); braindance/examples/streaming_workshop/session.py:493 (named-call hint)
**Constructor:** `__init__(self, phases: List[PhaseV3], name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_is_group` | inferred at runtime | `True` |
| `name` | inferred at runtime | `name or 'phase_group'` |
| `phases` | inferred at runtime | `phases` |
**Methods:**
#### `__iter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:179`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:182`
#### `needs_environment(self) -> bool`
> Group needs environment if any phase needs it.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:185`
#### `close_environment_after(self) -> bool`
> Close environment after group completes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:189`
### ValidationError(Exception)
> Raised when phase validation fails.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:194`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phase_base_v3.py:237 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### PhaseValidator()
> Validates phase dependencies and data flow.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:199`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `validate_pipeline(phases: List[PhaseV3], existing_data: Optional['DataContext']=None) -> bool`
> Walk grouped and individual phases in order, rejecting inputs absent from initial data and preceding outputs.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:205`
### QuickPhaseV3(AnalysisPhaseV3)
> A phase that executes a lambda/function inline with automatic dependency inference.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:252`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phase_base_v3.py:418 (named-call hint); braindance/core/phases_v3/phase_base_v3.py:437 (named-call hint)
**Constructor:** `__init__(self, func: Callable[[Any], Dict[str, Any]], requires: Optional[List[str]]=None, provides: Optional[List[str]]=None, name: str=None, *, inputs: Optional[List[str]]=None, outputs: Optional[List[str]]=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `func` | inferred at runtime | `func` |
| `func_source` | inferred at runtime | `self._get_function_source(func)` |
| `inputs` | inferred at runtime | `inferred_req if requires is None else requires` |
| `outputs` | inferred at runtime | `inferred_prov if provides is None else provides` |
**Methods:**
#### `_infer_dependencies(self, func: Callable) -> tuple[List[str], List[str]]`
> Use limited Python-source patterns to infer data keys for simple inline analyses.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:332`
#### `_get_function_source(self, func: Callable) -> str`
> Get function source for debugging.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:371`
#### `run(self, experiment) -> Dict[str, Any]`
> Execute the lambda function.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:378`

## Functions
### `quick_phase(func: Callable[[Any], Dict[str, Any]], requires: Optional[List[str]]=None, provides: Optional[List[str]]=None, name: str=None, *, inputs: Optional[List[str]]=None, outputs: Optional[List[str]]=None) -> QuickPhaseV3`
> Wrap a callable as an analysis-only V3 phase with optional explicit contracts.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:397`
### `phase(name: str=None, requires: Optional[List[str]]=None, provides: Optional[List[str]]=None, *, inputs: Optional[List[str]]=None, outputs: Optional[List[str]]=None)`
> Decorate a callable into a QuickPhaseV3 instance.
> **Called by:** braindance/examples/2_rapid_pairing.py:81 (named-call hint); braindance/examples/closed_loop.py:24 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phase_base_v3.py:421`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Phase constructors accept name, suffix and recording_tag; inputs/outputs name Experiment.data keys. |
| QuickPhaseV3, quick_phase and phase accept canonical keyword-only inputs/outputs plus historical positional requires/provides; omitted contracts are inferred from source. |

## Data Shapes
- run(experiment) returns dict[str, value]; contracts are lists of string data keys.
- info() exposes name, inputs, outputs, requires, provides and metadata; canonical and historical lists refer to the same contract.

## Notes
- Conflicting canonical and historical declarations raise ValueError. Class and instance assignments synchronize aliases; instance initialization copies contract lists.
- Source inference recognizes exp.data.X and string dictionary keys, so explicit contracts are more reliable than inference for complex functions.
- Replay phase time uses the acquisition environment clock; other phases use perf_counter.
