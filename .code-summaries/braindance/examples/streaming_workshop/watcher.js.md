# watcher.js

**Path:** `braindance/examples/streaming_workshop/watcher.js`
**Module:** `braindance.examples.streaming_workshop.watcher.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Core browser state/controller for profiles, launch controls, setup collection, connectivity editing, live telemetry rendering, and the continuous session-state poll.

## Connections
- **Shared data:** Calls profiles/control/state APIs and shares globals with UI, builder, simulator, phase flow, and code workspace.

## Dependencies
None

## Classes
None

## Functions
### `request(path, body)`
> Authenticated JSON request helper.
**Source:** `braindance/examples/streaming_workshop/watcher.js:19`
### `control(kind, extra={})`
> Send run control and surface errors.
**Source:** `braindance/examples/streaming_workshop/watcher.js:28`
### `collect()`
> Serialize and validate form, builder, and simulator settings.
**Source:** `braindance/examples/streaming_workshop/watcher.js:38`
### `hydrate(values)`
> Populate forms/matrix from defaults and saved settings.
**Source:** `braindance/examples/streaming_workshop/watcher.js:59`
### `start(skip)`
> Ensure code is saved and experiment preflight passes before launch.
**Source:** `braindance/examples/streaming_workshop/watcher.js:102`
### `saveProfile(name)`
> Persist current code and merged settings.
**Source:** `braindance/examples/streaming_workshop/watcher.js:157`
### `renderMatrix()`
> Render a windowed editable adjacency matrix.
**Source:** `braindance/examples/streaming_workshop/watcher.js:175`
### `applyMatrix()`
> Schedule validation and push live adjacency changes.
**Source:** `braindance/examples/streaming_workshop/watcher.js:207`
### `scene(s,target='scene')`
> Draw CartPole, FoodLand, or projected Ant telemetry.
**Source:** `braindance/examples/streaming_workshop/watcher.js:242`
### `draw()`
> Render controls, performance, plots, raw, raster, and causal telemetry.
**Source:** `braindance/examples/streaming_workshop/watcher.js:266`
### `poll()`
> Continuously fetch state and redraw at visibility-dependent cadence.
**Source:** `braindance/examples/streaming_workshop/watcher.js:308`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Defaults define environment/detection/channels/seed, gains, durations, pools, two stimulation electrodes, and optional baseline. |

## Data Shapes
- Adjacency is target × source; state history contains counts, observations, actions, rates, and delivered inputs; raw telemetry is channel × envelope samples.

## Notes
- Unsaved participant code triggers a modal save-before-run flow.
- Matrix DOM is windowed to 32×32 cells even for 256 neurons; polling is ~30 Hz visible and 2 Hz hidden.
