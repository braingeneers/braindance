# phase_flow.css

**Path:** `braindance/examples/streaming_workshop/phase_flow.css`
**Module:** `braindance.examples.streaming_workshop.phase_flow.css`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Styles the interactive phase-board library, connected phase cards, loops, dependency chips, inspector, drag/drop states, zoom workspace, and responsive layout.

## Connections
- **Shared data:** Loaded dynamically by `phase_flow.js`; uses shared workshop variables and builder-created elements.

## Dependencies
None

## Classes
None

## Functions
### `.flow-layout`
> Two-column phase library and board workspace.
**Source:** `braindance/examples/streaming_workshop/phase_flow.css:6`
### `#flowViewport`
> Scrollable, pannable and zoomable dotted phase canvas.
**Source:** `braindance/examples/streaming_workshop/phase_flow.css:10`
### `.flow-piece`
> Visual phase node with type/input connectors and validity states.
**Source:** `braindance/examples/streaming_workshop/phase_flow.css:13`
### `.flow-loop`
> Grouped repeated section with controls and loop cue.
**Source:** `braindance/examples/streaming_workshop/phase_flow.css:15`
### `.flow-inspector`
> Selected-phase parameter editor.
**Source:** `braindance/examples/streaming_workshop/phase_flow.css:17`

## Config / CLI
None

## Data Shapes
None

## Notes
- Uses distinct experiment, analysis, and loop colors with dark-theme variants; relies on modern `color-mix`, `:has`, and touch-action behavior.
