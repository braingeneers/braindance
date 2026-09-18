# pong.py

**Path:** `braindance/experiments/pong.py`
**Module:** `braindance.experiments.pong`
**Feature Area:** `Experiment Phases`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Controls a Pong paddle from differences in two recording-channel spike counts. Encodes paddle-to-ball distance as stimulation frequency and applies episode-end training patterns.

## Connections
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `PongEnv` from `braindance.games.pong_env` — imports (static evidence).

## Dependencies
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.games.pong_env.PongEnv` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main(config=None, stim_electrodes=None, up_channels=None, down_channels=None, save_dir=None, name='pong', episodes=5, render=True, seed=0, max_time_sec=300, read_period_ms=50, sampling_hz=20000, amplitude_mv=200, phase_width_us=100)`
> Connect neural activity to Pong and Pong state to stimulation. ``up_channels`` and ``down_channels`` are Maxwell recording channel IDs. ``stim_electrodes`` contains two physical electrode IDs; index 0 signals that the ball is above the paddle and index 1 signals that it is below.
> **Called by:** braindance/experiments/pong.py:136 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/experiments/pong.py:17`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Uses BrainDance configured output directory for generated results. |
| config.cfg; fixed stimulation electrodes; read channels 481 and 455 |
| Sensory rates clipped to 1–30 Hz; tetanus_Hz=30 |

## Data Shapes
- Spike events (frame,channel,amplitude); CSV time,dist_to_ball,reward,action,spike_count_l,spike_count_r

## Notes
- Executes at import; mutates shared maxwell_params; imports games.pong_env through unqualified games namespace.
