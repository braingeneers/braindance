# maxwell_env.py

**Path:** `braindance/core/maxwell_env.py`
**Module:** `braindance.core.maxwell_env`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Implements the BrainDance environment for live Maxwell MaxOne acquisition/stimulation and direct local replay through the same step contract. It manages hardware setup, ZeroMQ packet parsing, stimulation sequences/logs, raw recording writers, replay sources, and optional worker processes.

## Connections
- **Used by:** `braindance.analysis.intermediate_rt_sorting.rt_sort` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.envs.maxwell_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.rt_sort` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.3_busybee` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.game` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.amplitude_sweep` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_continuous` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_fast` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_phase` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_schedule` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_freq` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_tetanus_phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.multi_freq_stim` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.neural_config_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.pong` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.sample_experiment` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.sample_manual_sequence` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.stim_pattern` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.stim_responses` — import consumer hint; not a proven runtime call.
- **Uses:** `BaseEnv` from `braindance.core.base_env` — imports (static evidence).
- **Uses:** `dummy_maxlab` from `braindance.core` — imports (static evidence).
- **Uses:** `dummy_zmq_np` from `braindance.core` — imports (static evidence).
- **Uses:** `H5ReplaySource` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `H5ReplayWriter` from `braindance.core.replay` — imports (static evidence).
- **Shared data:** Extends `BaseEnv`; live setup uses maxlab plus ZeroMQ, while replay imports `H5ReplaySource`/`H5ReplayWriter` and can notify a source's `on_stimulation` callback.
- **Shared data:** Closed-loop V3 phases call `step`, inspect `stim_dts`/`latest_frame`, and rely on CSV stimulation logs and the saved raw file path.

## Dependencies
- `braindance.core.base_env.BaseEnv` — intra-repo import; source import evidence.
- `braindance.core.dummy_maxlab` — intra-repo import; source import evidence.
- `braindance.core.dummy_zmq_np` — intra-repo import; source import evidence.
- `braindance.core.replay.H5ReplaySource` — intra-repo import; source import evidence.
- `braindance.core.replay.H5ReplayWriter` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.animation` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `maxlab` — external or unresolved local import; source import evidence.
- `maxlab.chip` — external or unresolved local import; source import evidence.
- `maxlab.saving` — external or unresolved local import; source import evidence.
- `maxlab.system` — external or unresolved local import; source import evidence.
- `maxlab.util` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `psutil` — external or unresolved local import; source import evidence.
- `tqdm.trange` — external or unresolved local import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
### MaxwellEnv(BaseEnv)
> Unify live Maxwell acquisition/stimulation and deterministic replay behind the experiment environment step API.
**Source:** `braindance/core/maxwell_env.py:87`
**Kind:** class. **Instantiated by:** braindance/cli/replay.py:129 (named-call hint); braindance/core/phases_v3/experiment_v3.py:814 (named-call hint); braindance/examples/3_busybee.py:33 (named-call hint); braindance/examples/paper_cartpole/causal.py:45 (named-call hint); braindance/examples/paper_cartpole/game.py:67 (named-call hint); braindance/examples/paper_cartpole/recording.py:40 (named-call hint); braindance/experiments/amplitude_sweep.py:14 (named-call hint); braindance/experiments/cartpole_continuous.py:73 (named-call hint); braindance/experiments/cartpole_fast.py:49 (named-call hint); braindance/experiments/cartpole_phase.py:20 (named-call hint); braindance/experiments/cartpole_schedule.py:47 (named-call hint); braindance/experiments/causal_connectivity.py:15 (named-call hint); braindance/experiments/causal_freq.py:53 (named-call hint); braindance/experiments/causal_tetanus_phases.py:27 (named-call hint); braindance/experiments/multi_freq_stim.py:17 (named-call hint); braindance/experiments/neural_config_analysis.py:22 (named-call hint); braindance/experiments/pong.py:59 (named-call hint); braindance/experiments/real_time_sorting.py:49 (named-call hint); braindance/experiments/sample_experiment.py:22 (named-call hint); braindance/experiments/sample_manual_sequence.py:87 (named-call hint); braindance/experiments/stim_pattern.py:20 (named-call hint); braindance/experiments/stim_responses.py:43 (named-call hint)
**Constructor:** `__init__(self, config=None, name='', stim_electrodes=None, max_time_sec=60, save_dir='data', multiprocess=False, render=False, filt=False, observation_type='spikes', verbose=1, smoke_test=False, dummy=None, start=True, skip_offset=False, replay=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_closed` | inferred at runtime | `False` |
| `_maxlab` | inferred at runtime | `maxlab` |
| `_replay_done` | inferred at runtime | `False` |
| `active_units` | inferred at runtime | `[]` |
| `array` | inferred at runtime | `None` |
| `base_name` | inferred at runtime | `name` |
| `config` | inferred at runtime | `config` |
| `config_data` | inferred at runtime | `None if self.is_replay else Config(config)` |
| `cur_time` | inferred at runtime | `0.0` |
| `data_queue` | inferred at runtime | `Queue()` |
| `dummy` | inferred at runtime | `dummy` |
| `event_queue` | inferred at runtime | `Queue()` |
| `is_replay` | inferred at runtime | `replay is not None` |
| `is_stimulation` | inferred at runtime | `False` |
| `last_stim_time` | inferred at runtime | `0` |
| `last_stim_times` | inferred at runtime | `np.zeros(len(stim_electrodes))` |
| `latest_batch` | inferred at runtime | `None` |
| `latest_frame` | inferred at runtime | `None` |
| `multiprocess` | inferred at runtime | `multiprocess` |
| `name` | inferred at runtime | `name` |
| `num_channels` | inferred at runtime | `self.replay_source.num_channels` |
| `observation_type` | inferred at runtime | `observation_type` |
| `plot_worker` | inferred at runtime | `None` |
| `replay` | inferred at runtime | `dict(replay or {})` |
| `replay_source` | inferred at runtime | `None` |
| `replay_write_output` | inferred at runtime | `bool(self.replay.pop('write_output', True))` |
| `replay_writer` | inferred at runtime | `None` |
| `save_dir` | inferred at runtime | `str(Path(save_dir).resolve())` |
| `save_file` | inferred at runtime | `os.path.join(self.save_dir, f'{self.name}')` |
| `saver` | inferred at runtime | `self._maxlab.saving.Saving()` |
| `seq` | inferred at runtime | `self._maxlab.Sequence()` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_electrodes` | inferred at runtime | `[int(e) for e in stim_electrodes]` |
| `stim_electrodes_dict` | inferred at runtime | `{unit: electrode for unit, electrode in zip(self.stim_units, self.stim_electrodes)}` |
| `stim_log_file` | inferred at runtime | `None` |
| `stim_log_writer` | inferred at runtime | `writer(self.stim_log_file)` |
| `stim_num` | inferred at runtime | `0` |
| `stim_units` | inferred at runtime | `[self._maxlab.chip.StimulationUnit(i) for i in range(len(stim_electrodes))]` |
| `subscriber` | inferred at runtime | `None` |
| `worker` | inferred at runtime | `None` |
**Methods:**
#### `start(self)`
> Restart timing, recording, and stimulation logging for an initialized environment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:312`
#### `reset(self)`
> Clean up and reopen a live-hardware recording under a collision-free name; replay cannot reset.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:330`
#### `clear_buffer(self, num_successive_waits=10, min_wait_f=0.5, buffer_size=10, samp_freq_hz=20000, timeout_s=30)`
> Drain live acquisition until receive timing indicates the caller has caught up, bounded by an optional timeout.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:369`
#### `get_observation(self, buffer_size=None)`
> Read and format one live packet or replay batch as spike events or raw frames.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:429`
#### `step(self, action=None, tag=None, buffer_size=None)`
> Acquire observations, optionally stimulate, and return `(observation, done)` for phase loops.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:502`
#### `stimulate(self, action, tag=None)`
> Validate/build a pulse command, dispatch it to hardware or replay callback, log it, and update per-electrode stimulation times.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:543`
#### `_clock_now(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:581`
#### `time_elapsed(self)`
> Return replay source time or live wall-clock experiment time.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:586`
#### `wall_time_elapsed(self)`
> Wall time since environment initialization, for diagnostics.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:592`
#### `dt(self)`
> Return elapsed source/wall time since the environment's most recent step timestamp.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:597`
#### `stim_dt(self)`
> Returns time since last stimulation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:601`
#### `stim_dts(self)`
> Returns time since last stimulation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:606`
#### `close(self)`
> Idempotently stop recording/replay resources and close the stimulation log.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:610`
#### `_validate_name(self)`
> Validate the name of the environment. If the name already exists, increment the name until it doesn't exist.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:615`
#### `_create_stim_pulse(self, stim_command)`
> Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief period. This should be combined with code that connects electrodes to the DAC in order to actually generate stimulus behavior.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:635`
#### `_create_stim_pulse_sequence(self, stim_commands)`
> Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief period. This should be combined with code that connects electrodes to the DAC in order to actually generate stimulus behavior.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:673`
#### `_insert_square_wave(self, amplitude_mV=150, phase_length_us=100)`
> Adds a square wave to the sequence with a set amplitude and duty cycle.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:743`
#### `_insert_sine_wave(self, amplitude_mV=50, frequency_Hz=1)`
> Adds a sine wave to the sequence with a set amplitude and frequency.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:771`
#### `_init_save(self)`
> Initialize the save file for the environment. Saved in self.save_dir with name self.name.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:800`
#### `_init_log_stimulation(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:829`
#### `_log_stimulation(self, stim_command, tag=None)`
> Log the stimulation command to a csv file. Stim command is a tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:841`
#### `_cleanup(self)`
> Shuts down the environment and saves the data.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:867`
#### `disconnect_all(self)`
> Power down and disconnect every configured stimulation unit.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:892`
#### `connect_units(self, units=None, inds=None)`
> Connect supplied stimulation units or selected configured indices.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:900`
### MaxwellStim()
> Build and send Maxwell biphasic stimulation sequences from a separate process.
**Source:** `braindance/core/maxwell_env.py:913`
**Kind:** class. **Instantiated by:** braindance/core/maxwell_env.py:1006 (named-call hint)
**Constructor:** `__init__(self, stim_units)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `active_units` | inferred at runtime | `[]` |
| `seq` | inferred at runtime | `maxlab.Sequence()` |
| `stim_num` | inferred at runtime | `0` |
| `stim_units` | inferred at runtime | `stim_units` |
**Methods:**
#### `_create_stim_pulse(self, stim_command)`
> Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief period. This should be combined with code that connects electrodes to the DAC in order to actually generate stimulus behavior.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/maxwell_env.py:921`
#### `_insert_square_wave(self, amplitude_mV=150, phase_length_us=100)`
> Adds a square wave to the sequence with a set amplitude and duty cycle.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:959`
### Config()
> Parse a Maxwell array configuration string into typed channel/electrode/position mappings.
**Source:** `braindance/core/maxwell_env.py:1509`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1150 (named-call hint); braindance/core/maxwell_env.py:204 (named-call hint); braindance/core/spikesorter/rt_sort.py:1376 (named-call hint)
**Constructor:** `__init__(self, filename)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `config` | inferred at runtime | `[]` |
| `mappings` | inferred at runtime | `[]` |
**Methods:**
#### `get_channels(self)`
> Return routed channel IDs from parsed mappings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1525`
#### `get_electrodes(self)`
> Return physical electrode IDs from parsed mappings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1528`
#### `get_channels_for_electrodes(self, electrodes)`
> Return routed channels whose parsed electrode IDs are in the requested collection.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1531`
#### `get_num_channels(self)`
> Return the number of parsed routed channels.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1534`

## Functions
### `sleep_with_progress(seconds, desc, verbose=1)`
> Sleep for a requested number of seconds, optionally presenting per-second tqdm progress or a plain status line.
> **Called by:** braindance/core/maxwell_env.py:1187 (named-call hint); braindance/core/maxwell_env.py:286 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:33`
### `find_process_by_port(port)`
> Return the first process with a connection bound to the requested local port, ignoring inaccessible processes.
> **Called by:** braindance/core/maxwell_env.py:57 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:47`
### `stop_process_using_port(port)`
> Locate, terminate, and wait for the process using a port, or report that none was found.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:56`
### `stim_process(stim_process_ready, stim_units, stim_ready, stim_shm_name, env_done, stim_amp=400, stim_length=100, stim_tag='stim')`
> Poll a shared boolean electrode-selection array and deliver stimulation until an external done event is set.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:987`
### `init_maxone(config, stim_electrodes, filt=True, verbose=1, gain=512, cutoff='1Hz', spike_thresh=5, dummy=False, skip_offset=False)`
> Initialize hardware settings, subscriber, routed array, stimulation units, and packet alignment.
> **Called by:** braindance/core/maxwell_env.py:247 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1042`
### `init_maxone_settings(gain=512, amp_gain=512, cutoff='1Hz', spike_thresh=5, verbose=1)`
> Initialize MaxOne and set gain and high-pass filter
> **Called by:** braindance/core/maxwell_env.py:1073 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1086`
### `setup_subscribers(filt, verbose=1)`
> Create and connect a configured ZeroMQ subscriber to the Maxwell data stream.
> **Called by:** braindance/core/maxwell_env.py:1074 (named-call hint); braindance/core/maxwell_env.py:1395 (named-call hint); braindance/core/maxwell_env.py:1496 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1121`
### `select_electrodes(config, stim_electrodes, verbose=1, dummy=False, skip_offset=False)`
> Load a Maxwell array config, route recording/stimulation electrodes, optionally settle and recompute offsets, and return units plus electrode mapping.
> **Called by:** braindance/core/maxwell_env.py:1075 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1140`
### `power_cycle_stim_electrodes(stim_units)`
> "Power up and down again all the stimulation units. It appears this is needed to equilibrate the electrodes" - from maxwell code
> **Called by:** braindance/core/maxwell_env.py:1184 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1199`
### `disconnect_stim_electrodes(stim_units)`
> Disconnect each stimulation unit in the list.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1221`
### `connect_stim_electrodes(stim_units)`
> Connect each stimulation unit in the list.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1228`
### `_unpack_spike_event(name, fmt, event_data)`
> unclear — see source
> **Called by:** braindance/core/maxwell_env.py:1282 (named-call hint); braindance/core/maxwell_env.py:1320 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1236`
### `_score_spike_events(events)`
> unclear — see source
> **Called by:** braindance/core/maxwell_env.py:1288 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1246`
### `_detect_spike_struct_format(events_data)`
> unclear — see source
> **Called by:** braindance/core/maxwell_env.py:1298 (named-call hint); braindance/core/maxwell_env.py:1309 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1258`
### `get_spike_struct_size(events_data=None)`
> Return the detected event-record size for a buffer or the legacy default when no buffer is supplied.
> **Called by:** braindance/core/maxwell_env.py:469 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1297`
### `parse_events_list(events_data)`
> Decode a raw Maxwell event buffer into `SpikeEvent` records after detecting its binary layout.
> **Called by:** braindance/core/maxwell_env.py:1400 (named-call hint); braindance/core/maxwell_env.py:471 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1302`
### `parse_frame(frame_data)`
> Expose a raw binary voltage frame as a float array without copying through NumPy.
> **Called by:** braindance/core/maxwell_env.py:1402 (named-call hint); braindance/core/maxwell_env.py:486 (named-call hint); braindance/core/maxwell_env.py:496 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1325`
### `receive_packet(subscriber, buffer_size=None)`
> Receive one or a requested number of multipart Maxwell frames and event buffers, tolerating receive timeouts.
> **Called by:** braindance/core/maxwell_env.py:1397 (named-call hint); braindance/core/maxwell_env.py:463 (named-call hint); braindance/core/maxwell_env.py:481 (named-call hint); braindance/core/maxwell_env.py:495 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1332`
### `socket_worker(data_queue, event_queue, subscriber_args)`
> Worker function that reads from the ZeroMQ socket.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1393`
### `plot_worker(queue)`
> Continuously consume raw frames from a multiprocessing queue and animate the first channel in a Tk matplotlib window.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1405`
### `ignore_first_packet(subscriber, verbose=1)`
> Discard multipart fragments until a packet boundary is reached, failing after three seconds without server data.
> **Called by:** braindance/core/maxwell_env.py:1078 (named-call hint); braindance/core/maxwell_env.py:1499 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1439`
### `ignore_remaining_packets(subscriber, verbose=1)`
> Non-blockingly drain all currently buffered subscriber parts.
> **Called by:** braindance/core/maxwell_env.py:1079 (named-call hint); braindance/core/maxwell_env.py:287 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1462`
### `launch_dummy_server(dummy)`
> Start the legacy dummy ZMQ publisher in a process and wait for its first packet.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell_env.py:1480`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `MaxwellEnv` chooses live hardware with `config` or replay with `replay={'source': ..., 'transport': 'direct', 'write_output': bool}`; deprecated `dummy` is converted to replay. |
| Observation type is `spikes` or `raw`; Maxwell subscribers use localhost ports 7204 unfiltered and 7205 filtered with 100 ms receive timeout. |
| Stimulation pulses express amplitude in mV and phase width in microseconds; delays in command sequences are milliseconds and convert with the 20 kHz `fs_ms=20` constant. |

## Data Shapes
- `SpikeEvent` contains `(frame, channel, amplitude)`; binary event packets support padded and well-ID layouts selected heuristically and cached globally.
- Replay batches carry frame vectors, uint16/float32 raw matrices shaped samples x channels, and per-frame event lists; raw hardware frames are `array('f')` buffers.
- Stimulation CSV rows contain time, amplitude, duty time, physical electrodes, tag, and optional replay frame.

## Notes
- Import falls back to `dummy_maxlab` when vendor maxlab is unavailable and prints that choice.
- Replay rejects multiprocess/render and manual sequences, uses source-frame time, and may write a new HDF5 via `H5ReplayWriter`.
- Spike bursts on at least 15% of channels are discarded as artifacts in spike-observation mode.
- `select_electrodes` constructs `RuntimeError` objects for missing/conflicting units without raising them, so those checks currently do not stop configuration.
