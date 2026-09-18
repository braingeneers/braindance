# builder.js

**Path:** `braindance/examples/streaming_workshop/builder.js`
**Module:** `braindance.examples.streaming_workshop.builder.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds the experiment identity, ordered phase cards, native parameter forms and input/output dependency reports in the browser. Loads the shared server catalog and synchronizes custom-analysis decorators before structure validation or preflight.

## Connections
- **Shared data:** GET /api/phase-catalog supplies Python catalog declarations; POST /api/analysis-contracts parses participant decorators; POST /api/verify-experiment validates setup or runs preflight.
- **Shared data:** Calls phase_flow.js expandPhasePlan, flowCommit and renderPhaseFlow; parameterEditor and contractLabel are reused by board/code workspace.
- **Shared data:** Uses request, collect, settings, start and running supplied by other browser scripts; precise ownership is outside this file.

## Dependencies
None

## Classes
None

## Functions
### `function builderConfig()`
> Serialize experiment identity, native settings, initial data and both grouped/expanded phase plans.
**Source:** `braindance/examples/streaming_workshop/builder.js:44`
### `function loadBuilder(values)`
> Restore saved settings or construct the default recording/probe/environment sequence and validate it.
**Source:** `braindance/examples/streaming_workshop/builder.js:45`
### `function parameterEditor(parent, values=`
> Maintain one parameter object behind typed form controls and advanced JSON, enforcing finite numeric values and JSON-object shape.
**Source:** `braindance/examples/streaming_workshop/builder.js:53`
### `async function syncAnalysisContracts()`
> Resolve editor decorators through the server and update custom-analysis phase declarations without trusting stale responses.
**Source:** `braindance/examples/streaming_workshop/builder.js:95`
### `function phaseDefinition(phase, definitions=catalog)`
> Overlay custom-analysis inputs/outputs onto catalog metadata, accepting historical contract spellings.
**Source:** `braindance/examples/streaming_workshop/builder.js:120`
### `function contractLabel(key, definition=`
> Format a data key with declared type and optional shape from catalog contracts.
**Source:** `braindance/examples/streaming_workshop/builder.js:124`
### `function sourceLink(url)`
> Create a safe external link to the committed Python definition.
**Source:** `braindance/examples/streaming_workshop/builder.js:128`
### `function renderPhases()`
> Render ordered phase cards, constructors, mappings, source links and contract placeholders.
**Source:** `braindance/examples/streaming_workshop/builder.js:134`
### `function displayValidation(report,preflight=false)`
> Render backend validation status and input origins/output descriptions for matching phase IDs.
**Source:** `braindance/examples/streaming_workshop/builder.js:178`
### `function scheduleValidation()`
> Invalidate current verification and debounce structure checks while updating the board.
**Source:** `braindance/examples/streaming_workshop/builder.js:189`
### `async function verifyExperiment(preflight=false,skip=false)`
> Synchronize analysis code, collect setup, request validation/preflight and suppress obsolete results.
**Source:** `braindance/examples/streaming_workshop/builder.js:196`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default experiment_name is streaming_workshop; native_settings and initial_data are editable JSON objects. |
| Validation debounce is 350 ms; launch availability is refreshed every 250 ms. |
| Binned environment choices are cartpole, foodland and ant; explicit zero baseline has one entry per configured channel. |

## Data Shapes
- phaseSpecs entries have id, type, params and optional settings/loop; builderConfig returns both original phase_plan and expanded phases.
- Catalog definitions carry params, native, parameters, inputs, outputs, data_contracts, mapping_contract, caveats and source_url.
- Verification reports carry ok, errors, caveats and per-phase input origins/output names; old requirements/provides fields are accepted.

## Notes
- Mutates shared browser globals catalog, phaseSpecs and validation caches; depends on the existing page DOM and global helpers.
- Version counters discard outdated asynchronous validation results; analysis contracts are cached against exact editor text.
- Native and binned scientific controllers remain separate execution modes; metadata forms do not execute Python.
