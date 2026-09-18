# code_workspace.js

**Path:** `braindance/examples/streaming_workshop/code_workspace.js`
**Module:** `braindance.examples.streaming_workshop.code_workspace.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Adds one-way builder-to-Python previews/export, participant function navigation, train-hook scaffolding and custom-analysis editing. Displays execution source/timing and derives analysis data references from phase declarations.

## Connections
- **Shared data:** Calls /api/code-preview and /api/code-export, backed by code_export.py.
- **Shared data:** Uses builder.js phaseDefinition/contractLabel and phase_flow.js flowCommit for canonical custom-analysis contracts and ordered data references.
- **Shared data:** Generated analysis functions import analysis_phase from streaming_workshop.custom_analysis.

## Dependencies
None

## Classes
None

## Functions
### `function showCodeFile(name)`
> Switch between participant editor and generated read-only Python previews.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:17`
### `function jumpToFunction(name)`
> Locate a top-level participant function and focus its editor position.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:23`
### `function refreshAnalysisReference()`
> Show earlier phase outputs and experiment parameter values/types for the selected analysis position.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:53`
### `function createAnalysis()`
> Add a unique custom-analysis node and matching decorated Python template with canonical inputs/outputs.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:73`
### `function renderPreview()`
> Syntax-highlight the active generated file without interpreting code as HTML.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:89`
### `function exportRequest()`
> Collect current shared setup and exact participant code for preview/export.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:95`
### `function schedulePreview()`
> Debounce code preview requests.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:96`
### `async function refreshPreview()`
> Fetch and cache a generated bundle, rejecting obsolete responses and reporting code errors.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:97`
### `window.openAnalysisPhase=(id=null,position=phaseSpecs.length)`
> Open or create a custom-analysis phase at an ordered position and focus its decorator/function with current data references.
**Source:** `braindance/examples/streaming_workshop/code_workspace.js:38`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Code preview debounce is 300 ms; configuration/execution polling runs every 400 ms. |
| Python function names are limited to 64 characters and reject Python keywords plus encode/decode/train/analysis_phase. |
| Generated preview tabs are environment_setup.py and experiment.py; participant functions retain their editable panel. |

## Data Shapes
- Export requests contain {config,code}; preview responses contain files and errors, while exports return a directory.
- Custom-analysis phases use {id,type:"custom_analysis",params:{function_name,inputs,outputs}} and an @analysis_phase decorator.
- The train template accepts transition, dt_s, params, state and returns a diagnostics dictionary; shared encode/decode state resets per episode.
- Execution badges read state.execution source/timing/bin_ms; streaming p95_ms above 20 is highlighted.

## Notes
- Script runs as an IIFE and exposes window.openAnalysisPhase; depends on shared builder/page globals.
- Preview is read-only and displays Python as text through paintPython; export does not round-trip arbitrary Python into forms.
- Preview responses use version counters and exact serialized-config caching; generated files are separate drafts.
