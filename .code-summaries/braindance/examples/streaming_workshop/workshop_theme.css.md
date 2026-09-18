# workshop_theme.css

**Path:** `braindance/examples/streaming_workshop/workshop_theme.css`
**Module:** `braindance.examples.streaming_workshop.workshop_theme.css`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Adds consistent category colors, navigation and setup cues, accessible focus rings, simulation/catalog/settings styling, dark-theme variants, and responsive refinements to the workshop shell.

## Connections
- **Shared data:** Loaded by `watcher.html`; complements inline base CSS and phase-board-specific CSS.

## Dependencies
None

## Classes
None

## Functions
### `:root`
> Define shared category palette and washes.
**Source:** `braindance/examples/streaming_workshop/workshop_theme.css:3`
### `.app-nav button`
> Apply category-aware navigation states.
**Source:** `braindance/examples/streaming_workshop/workshop_theme.css:34`
### `#view-experiment .setup-grid`
> Color-code setup fieldsets by concern.
**Source:** `braindance/examples/streaming_workshop/workshop_theme.css:70`
### `.global-settings`
> Lay out global settings and feature cards.
**Source:** `braindance/examples/streaming_workshop/workshop_theme.css:156`
### `#saveRunDialog`
> Style save-before-run modal and backdrop.
**Source:** `braindance/examples/streaming_workshop/workshop_theme.css:168`

## Config / CLI
None

## Data Shapes
None

## Notes
- Category palette maps experiment/run/function/playground contexts to green, blue, violet, and amber without encoding meaning solely through color.
