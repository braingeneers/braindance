# phase_flow.js

**Path:** `braindance/examples/streaming_workshop/phase_flow.js`
**Module:** `braindance.examples.streaming_workshop.phase_flow.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Implements the visual phase board and pure sequence/dependency helpers. Supports phase insertion, ordered data lineage, loop expansion, guarded reordering, parameter inspection, undo/redo and mouse/touch navigation.

## Connections
- **Shared data:** Uses builder.js phaseDefinition, contractLabel, parameterEditor, sourceLink, renderPhases and scheduleValidation to keep board and forms consistent.
- **Shared data:** Calls code_workspace.js openAnalysisPhase for custom-analysis nodes; launch forwards to the same launchExperiment button as the forms.
- **Shared data:** phaseFlowTrace evaluates canonical and legacy declaration names locally; backend verifyExperiment remains authoritative for runtime preflight.

## Dependencies
None

## Classes
None

## Functions
### `function expandPhasePlan(plan)`
> Expand contiguous loop sections into an ordered flat sequence with unique repeated IDs and bounded size.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:2`
### `function phaseDataKey(key)`
> Translate historical workshop data names to canonical operation/data keys.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:19`
### `function phaseFlowTrace(plan, definitions, initial)`
> Project ordered output availability and report missing inputs or incompatible native/binned modes.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:22`
### `function phaseFlowLineage(key, definitions, available, native, seen=new Set())`
> Explain possible same-mode producers recursively, identifying cycles and missing initial inputs.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:35`
### `function movePhasePlan(plan, source, target, section=false)`
> Move a phase or complete loop section between original-plan gaps while preserving valid loop boundaries.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:45`
### `function flowInputs(native)`
> Read explicit initial data/settings or baseline/calibration availability from setup controls.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:75`
### `function flowProblem(next)`
> Return the first dependency, execution-mode or expansion error for a proposed sequence.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:86`
### `function flowCommit(next,remember=true)`
> Validate a proposed plan before replacing shared state, remembering undo history and refreshing views.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:94`
### `function flowNewPhase(type)`
> Construct catalog defaults with a unique legal phase ID.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:100`
### `function flowPropose(payload,target)`
> Create a candidate insertion or reorder, inheriting a surrounding loop for inserted phases.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:105`
### `function flowDrop(payload,target)`
> Apply a drag or library insertion through validation and update selection feedback.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:113`
### `function flowEndDrag()`
> Clear all drag indicators and transient target state.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:118`
### `function flowStartDrag(payload,node,event)`
> Initialize a drag payload and mark valid or blocked drop connections.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:119`
### `function flowDraggable(node,payload,handle=node)`
> Attach desktop drag and pointer-based touch dragging to a node/handle.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:124`
### `function flowSetZoom(value,clientX=null,clientY=null)`
> Scale the board around the pointer or viewport center while preserving its content anchor.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:150`
### `function flowTarget(x,y)`
> Resolve a screen point to a sequence gap, including node halves and nearby board space.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:201`
### `function flowPreviewTarget(x,y)`
> Display the proposed drop location and dependency error without changing the sequence.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:214`
### `function flowSelectionUI()`
> Update loop selection controls, selected node indicators and guidance.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:226`
### `function flowSelectRange(index)`
> Select a contiguous nonloop range using an anchor and reject crossing existing loops.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:234`
### `function flowButton(text,label,action)`
> Create labeled accessible buttons for board actions.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:242`
### `function flowLabel(definition)`
> Turn catalog class labels into short human-readable phase names.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:243`
### `function flowKind(definition)`
> Classify phases for analysis or experiment board styling.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:244`
### `function flowInspector()`
> Render and apply native constructor/settings or binned phase forms using guarded plan commits.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:245`
### `function renderPhaseFlow()`
> Render searchable availability, sequence/loops, input/output chips, selected connection data and inspector.
**Source:** `braindance/examples/streaming_workshop/phase_flow.js:271`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Loops repeat 1–32 times; expanded sequences are limited to 32 phases; nested loop sections are rejected. |
| Undo history retains at most 50 plans; board zoom is clamped to 55–140%; run availability is polled every 250 ms. |
| Flow initial data comes from native initial_data/native_settings or an explicit binned baseline/calibration. |

## Data Shapes
- Plans contain phase id/type/params, optional settings and contiguous loop objects {id,count}; expanded repeated IDs are collision-checked.
- Dependency trace returns {rows,available,ok}; rows contain before, missing and mixed; available maps keys to origin/value/projected metadata.
- Historical workshop_baseline_hz, workshop_causal_hz/trials and workshop_episodes/reward are normalized to recording_baseline_hz, response_probe_hz/trials and environment_episodes/reward.

## Notes
- Pure plan helpers export through CommonJS without requiring a DOM; board setup only runs when document exists.
- Dependency-invalid edits and mixed native/binned sequences are rejected before committing; trace shows projected outputs, not measured values.
- Shared phaseSpecs/catalog come from builder.js; page-level state tracks drag, selection, loop grouping and undo histories.
- Touch pinches cancel node drags; Ctrl-wheel and Safari gesture events implement anchored zoom; ordinary wheel scrolling remains native.
