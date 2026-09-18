# simulation.py

**Path:** `braindance/core/simulation.py`
**Module:** `braindance.core.simulation`
**Feature Area:** `Replay`
**Entry point:** no — library or imported component

## Overview
Implements a deterministic, stimulation-responsive NumPy recurrent leaky integrate-and-fire source compatible with Maxwell replay. It produces 20 kHz raw ADC samples, ground-truth spike events, spatial extracellular templates, configurable recurrent coupling, and serialized geometry for editor UIs.

## Connections
- **Used by:** `braindance.examples.labyrinth` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `MAPPING_DTYPE` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `SpikeEvent` from `braindance.core.replay` — imports (static evidence).
- **Shared data:** Imports `MAPPING_DTYPE` and `SpikeEvent` from replay and can be passed directly as `MaxwellEnv(replay={'source': simulation})`; Maxwell stimulation calls its `on_stimulation`.

## Dependencies
- `braindance.core.replay.MAPPING_DTYPE` — intra-repo import; source import evidence.
- `braindance.core.replay.SpikeEvent` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### NeuralSimulationSource()
> Provide an editable synthetic neural acquisition source implementing the direct replay contract.
**Source:** `braindance/core/simulation.py:35`
**Kind:** class. **Instantiated by:** braindance/examples/labyrinth.py:16 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:215 (named-call hint); braindance/examples/streaming_workshop/session.py:362 (named-call hint)
**Constructor:** `__init__(self, num_channels=8, seed=7, adjacency=None, speed=1.0, sampling_hz=20000, background=1.25, noise_uv=4.0, duration_s=3600, clock=time.perf_counter, sleep=time.sleep, num_neurons=None, grid_shape=None, electrode_pitch_um=17.5, neuron_positions=None, spatial_sigma_um=20.0, waveform_amplitude_uv=85.0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_closed` | inferred at runtime | `False` |
| `_drive` | inferred at runtime | `np.zeros(self.num_neurons)` |
| `_frame` | inferred at runtime | `0` |
| `_last_spikes` | inferred at runtime | `np.zeros(self.num_neurons)` |
| `_pulse_id` | inferred at runtime | `0` |
| `_pulses` | inferred at runtime | `[]` |
| `_refractory` | inferred at runtime | `np.zeros(self.num_neurons, dtype=int)` |
| `_stim_weights` | inferred at runtime | `np.exp(-squared_distances / (2 * (self.electrode_pitch_um / 2) ** 2))` |
| `_stop` | inferred at runtime | `int(duration_s * self.sampling_hz)` |
| `_tail` | inferred at runtime | `np.zeros((len(self._wave), self.num_channels))` |
| `_voltage` | inferred at runtime | `self._rng.uniform(0, 0.6, self.num_neurons)` |
| `_wall_start` | inferred at runtime | `None` |
| `_wave` | inferred at runtime | `-np.exp(-((x - 0.0004) / 0.00018) ** 2)` |
| `adjacency` | inferred at runtime | `matrix.copy()` |
| `background` | inferred at runtime | `_number(background, 'background')` |
| `electrode_pitch_um` | inferred at runtime | `_number(electrode_pitch_um, 'electrode_pitch_um')` |
| `finished` | inferred at runtime | `False` |
| `grid_shape` | inferred at runtime | `tuple((_number(v, 'grid_shape', integer=True) for v in grid_shape))` |
| `mapping` | inferred at runtime | `np.zeros(self.num_channels, dtype=MAPPING_DTYPE)` |
| `neuron_positions` | inferred at runtime | `positions.copy()` |
| `noise_uv` | inferred at runtime | `_number(noise_uv, 'noise_uv')` |
| `num_channels` | inferred at runtime | `_number(num_channels, 'num_channels', integer=True)` |
| `num_neurons` | inferred at runtime | `self.num_channels if num_neurons is None else _number(num_neurons, 'num_neurons', integer=True)` |
| `primary_channels` | inferred at runtime | `squared_distances.argmin(axis=1)` |
| `sampling_hz` | inferred at runtime | `_number(sampling_hz, 'sampling_hz')` |
| `seed` | inferred at runtime | `None if seed is None else _number(seed, 'seed', integer=True)` |
| `spatial_sigma_um` | inferred at runtime | `_number(spatial_sigma_um, 'spatial_sigma_um')` |
| `spatial_weights` | inferred at runtime | `np.exp(-squared_distances / (2 * self.spatial_sigma_um ** 2))` |
| `speed` | inferred at runtime | `0.0 if str(speed) == 'max' else _number(speed, 'speed')` |
| `templates` | inferred at runtime | `np.tile(self._wave, (self.num_neurons, 1))` |
| `waveform_amplitude_uv` | inferred at runtime | `_number(waveform_amplitude_uv, 'waveform_amplitude_uv')` |
**Methods:**
#### `_initialize_rng(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/simulation.py:118`
#### `reset(self, seed=None)`
> Rewind frame, random streams, neural state, pulse queue, and waveform tail while preserving geometry and connections.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:122`
#### `set_neuron_positions(self, positions)`
> Validate neuron geometry and recompute recording/stimulation spatial footprints plus primary channels.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:142`
#### `snapshot(self)`
> Return JSON-serializable model, geometry, dynamics, adjacency, mapping, and noise settings without mutable aliases.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/simulation.py:156`
#### `elapsed_s(self)`
> Report simulated source time from the current frame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/simulation.py:169`
#### `set_adjacency(self, adjacency)`
> Validate and copy the recurrent target-by-source weight matrix.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:172`
#### `on_stimulation(self, action, frame, stim_electrodes)`
> Validate explicit Maxwell pulse commands and enqueue their channel-local artifacts and neural drive at future unread frames.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:180`
#### `read(self, count=1, convert_raw=True)`
> Advance dynamics, render waveforms/artifacts/noise, pace simulation time, and return a replay-compatible batch.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:222`
#### `close(self)`
> Mark the source closed so further reads and resets fail.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/simulation.py:282`

## Functions
### `_number(value, name, integer=False)`
> Validate public scalar configuration without silently truncating indices.
> **Called by:** braindance/core/simulation.py:127 (named-call hint); braindance/core/simulation.py:197 (named-call hint); braindance/core/simulation.py:205 (named-call hint); braindance/core/simulation.py:206 (named-call hint); braindance/core/simulation.py:210 (named-call hint); braindance/core/simulation.py:227 (named-call hint); braindance/core/simulation.py:41 (named-call hint); braindance/core/simulation.py:42 (named-call hint); braindance/core/simulation.py:43 (named-call hint); braindance/core/simulation.py:48 (named-call hint); braindance/core/simulation.py:51 (named-call hint); braindance/core/simulation.py:52 (named-call hint); braindance/core/simulation.py:53 (named-call hint); braindance/core/simulation.py:62 (named-call hint); braindance/core/simulation.py:79 (named-call hint); braindance/core/simulation.py:84 (named-call hint); braindance/core/simulation.py:85 (named-call hint); braindance/core/simulation.py:90 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/simulation.py:22`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Requires at least 2 channels and exactly 20 kHz sampling; defaults to 8 channels/neurons, seed 7, real-time speed 1, one-hour duration, background 1.25, and noise 4 microvolts. |
| Geometry defaults to a compact grid with 17.5 micrometer pitch, spatial sigma 20 micrometers, and waveform amplitude 85 microvolts; custom grid width is capped at 220. |
| `speed='max'` disables wall-clock pacing; adjacency weights must be finite in [-2,2]. |

## Data Shapes
- Mapping is a structured `MAPPING_DTYPE` array of length channels; neuron positions are neurons x 2 and adjacency is target-neuron x source-neuron.
- Templates are neurons x 60 samples; spatial/stimulation weights are neurons x channels; raw output is samples x channels.
- `read` returns frame vectors, uint16 ADC, optional float32 millivolts, and a list of per-frame `SpikeEvent` lists.

## Notes
- Neural dynamics update every 20 raw samples (1 ms), with three-update refractory state and scheduled pulses delivered at the next unread frame.
- Separate spawned RNG streams keep neural dynamics and raw noise reproducible; neuron placement uses a third independent stream.
- Legacy auto-layout uses channel IDs as electrode IDs, while explicit grids encode Maxwell's 220-column electrode numbering.
