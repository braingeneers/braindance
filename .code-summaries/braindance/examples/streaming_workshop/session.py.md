# session.py

**Path:** `braindance/examples/streaming_workshop/session.py`
**Module:** `braindance.examples.streaming_workshop.session`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Hosts reusable binned V3 phases in a worker-owned Experiment runner and exposes bounded telemetry to the workshop UI. Runner/runtime layers own retries, controller hot reload, source selection, stimulation routing, game scenes, playback logs and pacing rather than embedding them in phase classes.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseGroup` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `BinnedRecordingPhaseV3` from `braindance.core.phases_v3.phases_binned` — imports (static evidence).
- **Uses:** `MappedEnvironmentPhaseV3` from `braindance.core.phases_v3.phases_binned` — imports (static evidence).
- **Uses:** `ResponseProbePhaseV3` from `braindance.core.phases_v3.phases_binned` — imports (static evidence).
- **Uses:** `H5ReplaySource` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `SpikeEvent` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `NeuralSimulationSource` from `braindance.core.simulation` — imports (static evidence).
- **Uses:** `RTSort` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `CustomAnalysisPhase` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `resolve_analysis_config` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `WorkshopGame` from `braindance.examples.streaming_workshop.environments` — imports (static evidence).
- **Uses:** `verify_spec` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** build_phase constructs BinnedRecordingPhaseV3, ResponseProbePhaseV3 or MappedEnvironmentPhaseV3 from builder rows.
- **Shared data:** WorkshopExperiment overrides Experiment._execute_phase to inject WorkshopRuntime and own retries/artifacts.
- **Shared data:** WorkshopRuntime implements binned acquisition/mapping/game services and translates scientific events into browser telemetry.
- **Shared data:** WorkshopSession uses H5ReplaySource or NeuralSimulationSource, WorkshopGame and CustomAnalysisPhase; verify_spec validates the selected sequence before run.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseGroup` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_binned.BinnedRecordingPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_binned.MappedEnvironmentPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_binned.ResponseProbePhaseV3` — intra-repo import; source import evidence.
- `braindance.core.replay.H5ReplaySource` — intra-repo import; source import evidence.
- `braindance.core.replay.SpikeEvent` — intra-repo import; source import evidence.
- `braindance.core.simulation.NeuralSimulationSource` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.RTSort` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.CustomAnalysisPhase` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.resolve_analysis_config` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.environments.WorkshopGame` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.verify_spec` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### RestartPhase(Exception)
> A participant requested a new attempt at the next bin boundary.
**Source:** `braindance/examples/streaming_workshop/session.py:21`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:532 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### WorkshopRuntime()
> Adapt session acquisition and mapping services to the binned phase protocol and observe scientific events externally.
**Source:** `braindance/examples/streaming_workshop/session.py:56`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:146 (named-call hint)
**Constructor:** `__init__(self, session)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `session` | inferred at runtime | `session` |
**Methods:**
#### `dt(self)`
> Expose the acquisition bin duration in seconds.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:63`
#### `sampling_hz(self)`
> Expose the acquisition sampling rate.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:67`
#### `n(self)`
> Expose the number of counted channels or sorted units.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:71`
#### `game(self)`
> Expose the selected normalized game adapter.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:75`
#### `read_counts(self, action=None, tag=None, pace=True)`
> Delegate acquisition of one bin to the session worker.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:78`
#### `map(self, name, value, dt)`
> Run a mapping callback and retain its diagnostics for telemetry.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:81`
#### `stimulate(self, action, tag)`
> Route local input indices into the shared acquisition stimulation union.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:86`
#### `reset_mapping(self)`
> Clear mapping state and pending stimulation at episode boundaries.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:90`
#### `emit(self, event, **values)`
> Translate baseline, probe and game events into cumulative session metrics, matching scenes and history.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:98`
#### `finish_step(self)`
> Complete session timing, pacing and playback logging after the phase finishes a game step.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:122`
### WorkshopExperiment(Experiment)
> Own interactive attempts and acquisition configuration around reusable scientific phases.
**Source:** `braindance/examples/streaming_workshop/session.py:126`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:486 (named-call hint)
**Constructor:** `__init__(self, session, *args, **kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `phase_runtime` | inferred at runtime | `WorkshopRuntime(s)` |
| `session` | inferred at runtime | `session` |
**Methods:**
#### `_create_environment_for_phase(self, phase)`
> Configure live or replay acquisition and union stimulation routing in the external runner.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/session.py:132`
#### `_execute_phase(self, phase)`
> Snapshot attempt inputs, execute a core phase, expose compatibility outputs and retry on participant request.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/session.py:142`
#### `_prepare_phase(self, phase)`
> Apply phase mapping settings, choose the correct game and publish phase-start state.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:194`
### WorkshopSession()
> Manage a thread-owned acquisition experiment, interactive commands and immutable telemetry snapshots.
**Source:** `braindance/examples/streaming_workshop/session.py:221`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, output_dir, config)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_below` | inferred at runtime | `np.zeros(self.raw_channels, dtype=bool)` |
| `_deadline_misses` | inferred at runtime | `0` |
| `_last_detected` | inferred at runtime | `np.full(self.raw_channels, -100000, dtype=np.int64)` |
| `_last_publish` | inferred at runtime | `0` |
| `_pacing_debt` | inferred at runtime | `0.0` |
| `_performance_ticks` | inferred at runtime | `deque(maxlen=100)` |
| `_tick_start` | inferred at runtime | `time.perf_counter()` |
| `_total_bins` | inferred at runtime | `0` |
| `_total_frames` | inferred at runtime | `0` |
| `acquisition_electrodes` | inferred at runtime | `list(dict.fromkeys((e for p in checked['phases'] for e in p['params'].get('stim_electrodes', []))))` |
| `bin_frames` | inferred at runtime | `round(self.dt * self.sampling_hz)` |
| `commands` | inferred at runtime | `queue.Queue()` |
| `dt` | inferred at runtime | `0.02` |
| `env` | inferred at runtime | `None` |
| `error` | inferred at runtime | `''` |
| `events` | inferred at runtime | `[[(e.frame - int(frames[0])) / self.sampling_hz, e.channel] for e in events]` |
| `game` | inferred at runtime | `None` |
| `last_episode_end` | inferred at runtime | `''` |
| `module` | inferred at runtime | `module` |
| `n` | inferred at runtime | `self.source.num_channels if self.source is not None else c['channels']` |
| `observation` | inferred at runtime | `self.game.reset()` |
| `params` | inferred at runtime | `dict(environment=c['environment'], action_size=len(self.game.action_names), left_channels=c.get('left_channels', list(range(self.…` |
| `paused` | inferred at runtime | `True` |
| `pending_action` | inferred at runtime | `None` |
| `phase_plan` | inferred at runtime | `[]` |
| `playback_log` | inferred at runtime | `None` |
| `pulse_credit` | inferred at runtime | `np.zeros(2)` |
| `raw` | inferred at runtime | `uv[:, :16].round(3).T.tolist()` |
| `raw_channels` | inferred at runtime | `self.n` |
| `run_dir` | inferred at runtime | `run_dir` |
| `sampling_hz` | inferred at runtime | `20000` |
| `signature` | inferred at runtime | `dict(source=str(c.get('source') or c.get('live_config') or 'simulation'), channels=self.n, raw_channels=self.raw_channels, detect…` |
| `snapshot` | inferred at runtime | `{'status': 'ready', 'phase': 'setup', 'error': '', 'config': self.config}` |
| `sorter` | inferred at runtime | `None` |
| `source` | inferred at runtime | `None` |
| `spatial` | inferred at runtime | `None` |
| `status` | inferred at runtime | `'starting'` |
| `stop_event` | inferred at runtime | `threading.Event()` |
| `thread` | inferred at runtime | `None` |
**Methods:**
#### `route_stimulus(self, action)`
> Map phase-local encoder outputs to the acquisition's routed union.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:222`
#### `start(self, overrides=None, skip=False)`
> Launch a fresh worker after rejecting concurrent runs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/session.py:240`
#### `validate_action(self, value)`
> Check game-specific action shape, finiteness and bounds.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:252`
#### `validate_rates(self, value)`
> Require two finite nonnegative stimulation rates within the configured maximum.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:260`
#### `reload_functions(self)`
> Compile participant hooks in a fresh namespace, test sample outputs and reset mapping state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/session.py:266`
#### `call_mapping(self, name, values, dt)`
> Execute and validate a callback, pausing for repair on interactive errors or raising headlessly.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/session.py:296`
#### `run(self, skip=False, verify_only=False)`
> Validate sources/controllers, construct a V3 group, run it and save calibration/performance while closing resources.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/session.py:324`
#### `controls(self)`
> Apply queued pause, step, reload, restart and adjacency commands at acquisition boundaries.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/session.py:522`
#### `finish_tick(self)`
> Measure compute time, recover pacing debt and write coherent per-bin playback records.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/session.py:561`
#### `tick(self, action=None, tag=None, pace=True)`
> Acquire an exact bin, route stimulation, detect spikes and retain bounded raw/event/count telemetry.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/session.py:585`
#### `performance(self)`
> Rolling active-worker rates; pauses and setup are excluded.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:666`
#### `publish(self, force=False)`
> Build a finite immutable JSON snapshot for readers without exposing live mutable worker state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/session.py:688`

## Functions
### `build_phase(kind, identifier=None, params=None)`
> Translate saved workshop phase kinds and parameters into reusable core V3 phase instances.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:174 (named-call hint); braindance/examples/streaming_workshop/session.py:494 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/session.py:34`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Inputs include source H5 path or live_config or simulator settings; detection selects events, threshold or rt-sort, with sorter_path required for RT-Sort. |
| Uses exact 20 ms bins, simulation seed/channel/spatial settings, speed pacing, threshold_uv, stim electrode routing, recording/game durations and two-output encoder settings. |
| functions_file supplies decode, encode and optional train; baseline_hz or a signature-matched calibration pickle may replace baseline recording. |
| Supports CartPole, FoodLand and Ant through WorkshopGame, with episode_seconds and per-phase mapping overrides. |

## Data Shapes
- Raw bins are frames × raw_channels; preview raw is up to 16 channels × frames, converted to microvolts.
- Detected counts are length n where n is channels or RT-Sort sequences; baseline is n Hz values and response matrices are 2 × n.
- Snapshot contains bounded history (250 bins), raw/events/scene/spatial arrays, game action/observation names, mapping diagnostics and performance; JSON round-trip enforces finite serializability.
- Attempt files store phase ID, attempt number, start/end frames, phase parameters, baseline, status and cumulative reward/episodes; each attempt snapshots functions.py.
- playback.jsonl stores one original bin/scene per row; calibration.pkl stores source/detector signature, baseline and response rates.

## Notes
- WorkshopPhase is replaced by build_phase plus WorkshopExperiment/WorkshopRuntime; reusable scientific phase classes live under core/phases_v3/phases_binned.py.
- LEGACY_OUTPUTS aliases old workshop_* data names at the runner boundary while canonical outputs describe recording, response probes and environment results.
- Only the worker thread mutates acquisition/game state; commands apply at bin boundaries and snapshot publication is limited to 30 Hz unless forced.
- Preflight exercises local source/game and mapping behavior without creating MaxwellEnv; live operation requires maxlab and threshold or RT-Sort detection.
- RT-Sort uses a prebuilt compatible sorter; threshold detection tracks crossings and a 40-frame refractory across bins.
- Restart restores phase-entry baseline/probe/reward/controller state but creates a new attempt on the ongoing acquisition timeline.
