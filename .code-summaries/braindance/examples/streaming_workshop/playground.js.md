# playground.js

**Path:** `braindance/examples/streaming_workshop/playground.js`
**Module:** `braindance.examples.streaming_workshop.playground.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Browser-side JavaScript for the streaming workshop playground interface.

## Connections
None

## Dependencies
None

## Classes
None

## Functions
### `pause()`
> Stops manual play and clears held input before redrawing.
**Source:** `braindance/examples/streaming_workshop/playground.js:15`
### `zero()`
> Clears keyboard/pointer input and refreshes feedback.
**Source:** `braindance/examples/streaming_workshop/playground.js:16`
### `held(direction)`
> Reports whether a directional key or pointer is held.
**Source:** `braindance/examples/streaming_workshop/playground.js:17`
### `action()`
> Combines current controls into the selected game action.
**Source:** `braindance/examples/streaming_workshop/playground.js:21`
### `feedback()`
> Displays the action and current playground values.
**Source:** `braindance/examples/streaming_workshop/playground.js:29`
### `playOnInput()`
> Starts the manual game loop from user input.
**Source:** `braindance/examples/streaming_workshop/playground.js:41`
### `draw()`
> Renders the playground state and controls.
**Source:** `braindance/examples/streaming_workshop/playground.js:45`
### `send(command)`
> Sends a playground command to the workshop backend.
**Source:** `braindance/examples/streaming_workshop/playground.js:56`
### `toggle()`
> Toggles manual play/pause.
**Source:** `braindance/examples/streaming_workshop/playground.js:100`

## Config / CLI
None

## Data Shapes
None

## Notes
- Runs in the browser and mutates the workshop DOM/canvas state.
