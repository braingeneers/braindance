# LinearArtifactRemoval3.py

**Path:** `braindance/utils/LinearArtifactRemoval3.py`
**Module:** `braindance.utils.LinearArtifactRemoval3`
**Feature Area:** `Artifact Removal`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Applies a high-pass recurrence and adaptive baseline subtraction with saturation recovery and threshold spike flags. Optionally synchronizes artifact blanking across channels and records processing timings.

## Connections
None

## Dependencies
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numba` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### LinearArtifactRemoval3()
> Optimized artifact removal with fast saturation synchronization.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:221`
**Kind:** class. **Instantiated by:** braindance/utils/LinearArtifactRemoval3.py:445 (named-call hint); braindance/utils/LinearArtifactRemoval3.py:460 (named-call hint)
**Constructor:** `__init__(self, n_channels, sat_threshold=100, spike_thresh=[-3.5, 20], recovery_time_ms=3.0, sampling_rate=20000, hp_cutoff_hz=300, baseline_tau_ms=100, moving_baseline_tau_ms=10, sync_saturation=False, saturation_ratio=None, pre_artifact_ms=0.5, n_threads=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_artifacts_buffer` | inferred at runtime | `None` |
| `_channel_saturation_count` | inferred at runtime | `None` |
| `_clean_buffer` | inferred at runtime | `None` |
| `_saturation_events` | inferred at runtime | `None` |
| `_saturation_frames` | inferred at runtime | `None` |
| `_spikes_buffer` | inferred at runtime | `None` |
| `baseline_alpha` | inferred at runtime | `np.exp(-1.0 / baseline_samples)` |
| `baselines` | inferred at runtime | `np.zeros(n_channels, dtype=np.float32)` |
| `filter_states` | inferred at runtime | `np.zeros(n_channels, dtype=np.float32)` |
| `hp_a1` | inferred at runtime | `-alpha` |
| `hp_b0` | inferred at runtime | `(1 + alpha) / 2` |
| `hp_b1` | inferred at runtime | `-(1 + alpha) / 2` |
| `last_samples` | inferred at runtime | `np.zeros(n_channels, dtype=np.float32)` |
| `moving_alpha` | inferred at runtime | `np.exp(-1.0 / moving_baseline_samples)` |
| `moving_baselines` | inferred at runtime | `np.zeros(n_channels, dtype=np.float32)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `pre_artifact_samples` | inferred at runtime | `int(pre_artifact_ms * sampling_rate / 1000)` |
| `recovery_counters` | inferred at runtime | `np.zeros(n_channels, dtype=np.int32)` |
| `recovery_samples` | inferred at runtime | `int(recovery_time_ms * sampling_rate / 1000)` |
| `sat_threshold` | inferred at runtime | `float(sat_threshold)` |
| `saturation_ratio` | inferred at runtime | `float(saturation_ratio) if saturation_ratio is not None else -1.0` |
| `spike_high` | inferred at runtime | `float(spike_thresh[1])` |
| `spike_low` | inferred at runtime | `float(spike_thresh[0])` |
| `spike_states` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `states` | inferred at runtime | `np.zeros(n_channels, dtype=np.int8)` |
| `sync_saturation` | inferred at runtime | `sync_saturation` |
| `timing` | inferred at runtime | `{'total': 0, 'processing': 0, 'sync': 0}` |
**Methods:**
#### `fit_step(self, frames_batch)`
> Process multiple frames with optimized saturation synchronization.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:313`
#### `warmup(self, n_samples=1000, minibatch_size=200)`
> Compile with synthetic batches, then clear filter states and timings.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:374`
#### `reset_channel(self, channel_idx)`
> Clear filter and recovery state for one channel.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:395`
#### `get_stats(self)`
> Get processing statistics.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:404`

## Functions
### `apply_iir_filter(x, y_prev, b0, b1, a1)`
> Single-pole IIR high-pass filter step.
> **Called by:** braindance/utils/LinearArtifactRemoval3.py:112 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:6`
### `process_channels_simplified_with_tracking(data, clean_out, artifacts_out, spikes_out, states, recovery_counters, baselines, filter_states, moving_baselines, sat_threshold, spike_low, spike_high, recovery_samples, hp_b0, hp_b1, hp_a1, baseline_alpha, spike_states, moving_alpha, saturation_events, sync_saturation, saturation_ratio, channel_saturation_count, pre_artifact_samples)`
> Processing with efficient saturation tracking for sync mode.
> **Called by:** braindance/utils/LinearArtifactRemoval3.py:336 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:12`
### `fast_sync_saturation(clean_out, states, recovery_counters, saturation_events, recovery_samples, n_frames, pre_artifact_samples)`
> Optimized synchronization using pre-computed saturation events.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:156`
### `collect_saturation_frames(saturation_events, saturation_frames, n_frames)`
> Collect indices of frames with saturation events.
> **Called by:** braindance/utils/LinearArtifactRemoval3.py:354 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:183`
### `ultra_fast_sync_saturation(clean_out, states, recovery_counters, saturation_frames, n_saturation_frames, recovery_samples, n_frames, pre_artifact_samples)`
> Ultra-fast synchronization processing only saturation frames.
> **Called by:** braindance/utils/LinearArtifactRemoval3.py:359 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/LinearArtifactRemoval3.py:193`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| sat_threshold=100,spike_thresh=[-3.5,20],recovery_time_ms=3,sampling_rate=20000,hp_cutoff_hz=300,baseline_tau_ms=100,moving_baseline_tau_ms=10 |
| sync_saturation,saturation_ratio,pre_artifact_ms,n_threads. |

## Data Shapes
- Input (channels,frames) converted to float32; output clean/artifact float32 and spike bool copies with matching shape.

## Notes
- n_threads changes Numba global thread count; persistent state needs reset between recordings.
- Synchronization post-pass changes clean outputs/state but does not recompute artifacts/spike flags.
- Script guard runs synthetic synchronization benchmark; filter recurrence uses previous input rather than separately retained previous output.
