# watcher.html

**Path:** `braindance/examples/streaming_workshop/watcher.html`
**Module:** `braindance.examples.streaming_workshop.watcher.html`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Base document for the local BrainDance workshop, containing source/setup forms, participant editor, simulator connectivity matrix, run controls, live plots, and script load order. Later JavaScript moves these elements into SPA views and adds builder/catalog/analysis/settings interfaces.

## Connections
- **Shared data:** Loads watcher, UI, settings, playground, simulator, builder, phase-flow, code, analysis, and catalog scripts in dependency order.

## Dependencies
None

## Classes
None

## Functions
### `<details id="setup">`
> Acquisition source, mappings, gains, and duration form.
**Source:** `braindance/examples/streaming_workshop/watcher.html:25`
### `<details id="functionsPanel">`
> Participant profile and Python editor shell.
**Source:** `braindance/examples/streaming_workshop/watcher.html:51`
### `<details id="connectivityPanel">`
> Editable simulator adjacency matrix shell.
**Source:** `braindance/examples/streaming_workshop/watcher.html:60`
### `<div class="toolbar">`
> Start/skip/pause/step/stop controls and status.
**Source:** `braindance/examples/streaming_workshop/watcher.html:67`
### `<div class="grid">`
> Environment, mappings, raw, spikes, causal response, and run details.
**Source:** `braindance/examples/streaming_workshop/watcher.html:70`
### `<script>…</script>`
> Inject token and load workshop modules in order.
**Source:** `braindance/examples/streaming_workshop/watcher.html:79`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Defaults include CartPole, simulation, events detection, 400 channels, seed 7, 20 ms-compatible durations, two stim inputs, and mapping gains. |

## Data Shapes
- Live canvases show scene, encoding, decoding, raw voltage, raster, and causal response matrices; connectivity rows are targets and columns sources.

## Notes
- Server replaces `__TOKEN__` with a random local control token.
- Inline CSS supplies the initial responsive/light/dark layout before external theme/workspace sheets.
