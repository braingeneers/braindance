# phases_binned.py

**Path:** `braindance/core/phases_v3/phases_binned.py`
**Module:** `braindance.core.phases_v3.phases_binned`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Implements reusable V3 baseline recording, randomized response probes and callback-controlled game phases over a runner-supplied binned runtime. Scientific loops emit events while pacing, persistence, controls and rendering remain external.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** session.build_phase constructs concrete classes; WorkshopExperiment supplies WorkshopRuntime through experiment.phase_runtime.
- **Shared data:** All concrete phases extend PhaseV3 and use canonical data contracts consumed by Experiment.
- **Shared data:** The normalized runtime game supports CartPole, FoodLand or Ant without phase-owned visualization.

## Dependencies
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### BinnedRuntime(Protocol)
> Define the runner-provided acquisition, game, mapping and observer interface.
**Source:** `braindance/core/phases_v3/phases_binned.py:16`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `read_counts(self, action=None, tag=None, pace=True)`
> Acquire one neural bin and return its counts and exact frame count.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:28`
#### `map(self, name, value, dt)`
> Invoke a named scientific mapping callback through the runtime.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:29`
#### `stimulate(self, action, tag)`
> Deliver a local-input stimulation action through the acquisition runner.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:30`
#### `reset_mapping(self)`
> Clear controller state at phase or episode boundaries.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:31`
#### `emit(self, event, **values)`
> Notify external observers of scientific state changes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:32`
#### `finish_step(self)`
> Complete runtime bookkeeping and pacing after a full game/control cycle.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:33`
### BinnedPhaseV3(PhaseV3)
> Base for binned phases sharing a continuously running acquisition environment.
**Source:** `braindance/core/phases_v3/phases_binned.py:36`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `_bin_count(duration, dt)`
> Validate positive finite duration/bin width and require an exact whole acquisition-bin count before any work.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:40`
#### `close_environment_after(self)`
> Keep the environment open between binned phases.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:50`
#### `run(self, experiment)`
> Reject execution of the non-concrete binned base.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:53`
### BinnedRecordingPhaseV3(BinnedPhaseV3)
> Measure baseline rates from exact acquisition frame counts.
**Source:** `braindance/core/phases_v3/phases_binned.py:57`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:41 (named-call hint)
**Constructor:** `__init__(self, duration=3.0, name=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
**Methods:**
#### `run(self, experiment)`
> Measure baseline rates from accumulated spike counts and actual acquired duration.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:66`
### ResponseProbePhaseV3(BinnedPhaseV3)
> Randomized stimulus/sham trials with matched pre/post windows.
**Source:** `braindance/core/phases_v3/phases_binned.py:81`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:43 (named-call hint)
**Constructor:** `__init__(self, repeats=4, seed=7, amplitude_mv=100.0, phase_width_us=100, name=None)`
**Key attributes:**
None
**Methods:**
#### `run(self, experiment)`
> Measure randomized stimulated-minus-sham changes using matched pre/post windows.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:93`
### MappedEnvironmentPhaseV3(BinnedPhaseV3)
> Closed-loop game controlled by decode, train and encode callbacks.
**Source:** `braindance/core/phases_v3/phases_binned.py:123`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:47 (named-call hint)
**Constructor:** `__init__(self, duration=120.0, amplitude_mv=100.0, phase_width_us=100, name=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
**Methods:**
#### `run(self, experiment)`
> Decode neural counts, advance a game, train mappings and encode sensory stimulation while emitting coherent per-step state.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_binned.py:138`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| BinnedRecordingPhaseV3 duration defaults to 3 seconds; MappedEnvironmentPhaseV3 duration defaults to 120 seconds. |
| ResponseProbePhaseV3 defaults to 4 repeats, seed 7, amplitude_mv=100 and phase_width_us=100; game sensory pulses use the same amplitude/width defaults. |
| experiment.phase_runtime supplies dt, sampling_hz, n, game and acquisition/mapping/event methods. |
| Recording/game durations must be finite positive whole multiples of runtime.dt; probe repeats must be a positive integer. |

## Data Shapes
- read_counts returns (counts[n], acquired_frames); baseline is recording_baseline_hz[n] in Hz.
- Response outputs are response_probe_hz[2][n] and response_probe_trials[2][2] with stimulated/sham trial counts.
- Game callback decode returns an action vector; encode returns two stimulation rates; train receives observation, next_observation, action, spike_counts, reward and done.
- Game outputs are environment_episodes (count) and environment_reward (cumulative reward within the phase); emitted game_step events include matching observation/action/rates and delivered input indices.

## Notes
- No web, transport or workshop modules are imported; the runtime owns those responsibilities.
- Response probes use two local stimulation inputs, five-bin pre/post windows and five-bin separation; trial order is seeded and randomized.
- Game pulse credit accumulates rates multiplied by acquired-bin duration and resets with mapping state on episode completion.
- BinnedPhaseV3 keeps the acquisition environment open for subsequent phases.
