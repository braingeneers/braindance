# simulator.js

**Path:** `braindance/examples/streaming_workshop/simulator.js`
**Module:** `braindance.examples.streaming_workshop.simulator.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Provides the browser editor and visualization for neural simulation geometry, Gaussian electrode footprints, weighted connectivity, test recordings, and live spatial voltage.

## Connections
- **Shared data:** Shares connectivity matrix and controls with `watcher.js`; starts a recording-only phase through the main control API; renders session spatial telemetry.

## Dependencies
None

## Classes
None

## Functions
### `simulatorExtent()`
> Derive grid dimensions and physical extent.
**Source:** `braindance/examples/streaming_workshop/simulator.js:31`
### `simulatorElectrodes()`
> Generate channel electrode coordinates.
**Source:** `braindance/examples/streaming_workshop/simulator.js:32`
### `simulatorConfig()`
> Serialize simulator state into launch settings.
**Source:** `braindance/examples/streaming_workshop/simulator.js:33`
### `loadSimulator(values)`
> Hydrate geometry and waveform settings.
**Source:** `braindance/examples/streaming_workshop/simulator.js:34`
### `resizeNeurons(n,changed=true)`
> Resize positions and square adjacency while retaining values.
**Source:** `braindance/examples/streaming_workshop/simulator.js:37`
### `setSimulatorConnections(targets,remove=false)`
> Set/remove weighted edges from the selected neuron.
**Source:** `braindance/examples/streaming_workshop/simulator.js:54`
### `simProjection(w,h,positions)`
> Fit physical coordinates into canvas space.
**Source:** `braindance/examples/streaming_workshop/simulator.js:69`
### `drawSimulator()`
> Render culture footprint/connectivity and live electrode activity.
**Source:** `braindance/examples/streaming_workshop/simulator.js:77`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| 1–256 neurons; electrode pitch, spatial sigma, waveform amplitude, threshold, sorter path, raw-save flag, and optional 20×20 grid. |

## Data Shapes
- Neuron positions are `[x_um,y_um]`; adjacency is target × source; spatial telemetry carries channel positions, electrode IDs, peak and RMS µV arrays.

## Notes
- Connections are neuron-to-neuron and independent of electrode channels; disabled for file/hardware sources.
- Canvas preview is a model, not recorded data.
