# replay.py

**Path:** `braindance/core/replay.py`
**Module:** `braindance.core.replay`
**Feature Area:** `Replay`
**Entry point:** no — library or imported component

## Overview
Reads Maxwell H5 recordings lazily and writes compatible recordings incrementally without maxlab. Also resolves input recordings and serializes the three-part replay transport format.

## Connections
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.dummy_zmq_np` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.maxwell_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.simulation` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.playback` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Shared data:** MaxwellEnv consumes H5ReplaySource and H5ReplayWriter; dummy_zmq_np uses packet encoding.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### H5ReplaySource()
> Sequential, bounded, lazy reader for Maxwell raw H5 recordings.
**Source:** `braindance/core/replay.py:142`
**Kind:** class. **Instantiated by:** braindance/core/dummy_zmq_np.py:37 (named-call hint); braindance/core/maxwell_env.py:238 (named-call hint); braindance/examples/streaming_workshop/playback.py:177 (named-call hint); braindance/examples/streaming_workshop/session.py:360 (named-call hint)
**Constructor:** `__init__(self, source, start_frame=0, stop_frame=None, channels=None, speed=1.0, loop=False, chunk_frames=2000, clock=time.perf_counter, sleep=time.sleep)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_channel_to_output` | inferred at runtime | `{int(channel): index for index, channel in enumerate(self.mapping['channel'])}` |
| `_clock` | inferred at runtime | `clock` |
| `_closed` | inferred at runtime | `False` |
| `_emitted_frames` | inferred at runtime | `0` |
| `_frame_nos` | inferred at runtime | `None` |
| `_npy` | inferred at runtime | `not self._synthetic and str(source).lower().endswith('.npy')` |
| `_npy_data` | inferred at runtime | `None` |
| `_position` | inferred at runtime | `self.start_frame` |
| `_raw_cache` | inferred at runtime | `None` |
| `_raw_cache_start` | inferred at runtime | `None` |
| `_raw_cache_stop` | inferred at runtime | `None` |
| `_sleep` | inferred at runtime | `sleep` |
| `_spike_cache` | inferred at runtime | `None` |
| `_spike_cache_frames` | inferred at runtime | `None` |
| `_spike_cache_position` | inferred at runtime | `0` |
| `_spike_cursor` | inferred at runtime | `0` |
| `_spikes` | inferred at runtime | `None` |
| `_start_clock` | inferred at runtime | `None` |
| `_synthetic` | inferred at runtime | `str(source) in {'sine', 'manual'}` |
| `_virtual_next` | inferred at runtime | `None` |
| `channel_indices` | inferred at runtime | `np.arange(len(mapping), dtype=np.int64)` |
| `chunk_frames` | inferred at runtime | `max(1, int(chunk_frames))` |
| `finished` | inferred at runtime | `False` |
| `gain` | inferred at runtime | `np.float32(512.0)` |
| `h5file` | inferred at runtime | `None` |
| `hpf` | inferred at runtime | `np.float32(1.0)` |
| `loop` | inferred at runtime | `bool(loop)` |
| `lsb` | inferred at runtime | `np.float32(1.0 / 512.0 / 1000.0)` |
| `mapping` | inferred at runtime | `mapping[self.channel_indices].copy()` |
| `num_channels` | inferred at runtime | `len(self.mapping)` |
| `path` | inferred at runtime | `None` |
| `raw` | inferred at runtime | `None` |
| `sampling_hz` | inferred at runtime | `20000.0` |
| `source` | inferred at runtime | `source` |
| `speed` | inferred at runtime | `float(speed)` |
| `start_frame` | inferred at runtime | `int(start_frame)` |
| `stop_frame` | inferred at runtime | `None if stop_frame is None else int(stop_frame)` |
| `total_frames` | inferred at runtime | `20000 * 60` |
**Methods:**
#### `_source_frame_numbers(self, start, stop)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:279`
#### `elapsed_s(self)`
> Biological replay time emitted so far, independent of wall pacing.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:285`
#### `wall_elapsed_s(self)`
> Wall time since first read; useful for replay throughput diagnostics.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:290`
#### `_read_raw(self, start, stop)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:296`
#### `_spike_lower_bound(self, target_frame)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:337`
#### `_load_spike_cache(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:348`
#### `_reset_spike_cache(self, target_frame)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:360`
#### `_read_events(self, source_frames, virtual_frames)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:366`
#### `read(self, count=1, convert_raw=True)`
> Fetch bounded frames/events, optionally convert ADC to float32, advance biological time, and pace against wall time.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:409`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:470`
### H5ReplayWriter()
> Incremental writer for a minimal modern Maxwell-compatible H5 file.
**Source:** `braindance/core/replay.py:480`
**Kind:** class. **Instantiated by:** braindance/core/maxwell_env.py:810 (named-call hint)
**Constructor:** `__init__(self, path, mapping, sampling_hz, lsb, gain, hpf=1.0, chunk_frames=2000)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_chunk_frames` | inferred at runtime | `max(1, int(chunk_frames))` |
| `_closed` | inferred at runtime | `False` |
| `_frames_written` | inferred at runtime | `0` |
| `_pending_count` | inferred at runtime | `0` |
| `_pending_frames` | inferred at runtime | `[]` |
| `_pending_raw` | inferred at runtime | `[]` |
| `_pending_spikes` | inferred at runtime | `[]` |
| `_start_time_ms` | inferred at runtime | `int(time.time() * 1000)` |
| `frame_nos` | inferred at runtime | `routed.create_dataset('frame_nos', shape=(0,), maxshape=(None,), chunks=(chunk_frames,), dtype=np.uint64)` |
| `gain` | inferred at runtime | `float(gain)` |
| `h5file` | inferred at runtime | `h5py.File(self.path, 'w')` |
| `hpf` | inferred at runtime | `float(hpf)` |
| `lsb` | inferred at runtime | `float(lsb)` |
| `mapping` | inferred at runtime | `np.asarray(mapping, dtype=MAPPING_DTYPE)` |
| `path` | inferred at runtime | `Path(path).expanduser().resolve()` |
| `raw` | inferred at runtime | `routed.create_dataset('raw', shape=(len(self.mapping), 0), maxshape=(len(self.mapping), None), chunks=raw_chunks, dtype=np.uint16)` |
| `sampling_hz` | inferred at runtime | `float(sampling_hz)` |
| `spikes` | inferred at runtime | `rec.create_dataset('spikes', shape=(0,), maxshape=(None,), chunks=(max(256, chunk_frames),), dtype=SPIKE_DTYPE)` |
| `stop_time` | inferred at runtime | `rec.create_dataset('stop_time', data=[self._start_time_ms])` |
**Methods:**
#### `append(self, batch)`
> Buffer frame-major ADC batches and remap event channels before appending Maxwell H5 datasets.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:543`
#### `_flush_pending(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:567`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/replay.py:589`

## Functions
### `_decode_scalar(dataset, default='')`
> unclear — see source
> **Called by:** braindance/core/replay.py:82 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:36`
### `_find_layout(h5file)`
> Return raw/settings/recording groups for supported Maxwell layouts.
> **Called by:** braindance/core/replay.py:228 (named-call hint); braindance/core/replay.py:69 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:45`
### `inspect_recording(path)`
> Read recording metadata without loading trace data.
> **Called by:** braindance/cli/replay.py:116 (named-call hint); braindance/cli/replay.py:194 (named-call hint); braindance/core/replay.py:98 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:65`
### `find_recordings(data_dir)`
> Return compatible ``*.raw.h5`` recordings ordered by file size.
> **Called by:** braindance/cli/replay.py:47 (named-call hint); braindance/core/replay.py:132 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:87`
### `resolve_replay_source(h5_path=None, recording=None, sample=False, data_dir=None, doctor=False)`
> Resolve explicit, environment, fixture, or data-root recording paths without changing global path configuration.
> **Called by:** braindance/cli/replay.py:109 (named-call hint); braindance/cli/replay.py:188 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:104`
### `pack_replay_packet(frame_number, raw_float32, events=())`
> Encode one replay frame using Maxwell dummy-server wire format.
> **Called by:** braindance/cli/replay.py:68 (named-call hint); braindance/core/dummy_zmq_np.py:55 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:599`
### `unpack_replay_packet(parts)`
> Decode packet produced by :func:`pack_replay_packet`.
> **Called by:** braindance/cli/replay.py:77 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/replay.py:611`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| BRAINDANCE_REPLAY_H5; configured get_data_dir; explicit path, recording, sample, and doctor resolution |
| Frame bounds, channel indices, speed, loop, and chunk_frames control H5ReplaySource. |

## Data Shapes
- Raw source datasets are (channels, frames); read batches are (frames, selected_channels).
- Mapping fields: channel, electrode, x, y; spikes: frameno, channel, amplitude; events expose frame/channel/amplitude.

## Notes
- Supports legacy sig, data_store/data0000, and first well/recording layouts; NPY and sine/manual sources also supported.
- Looping keeps virtual frame numbers monotonic; channels refer to row indices and event channels are remapped to output positions.
