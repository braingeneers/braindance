# 3_busybee.py

**Path:** `braindance/examples/3_busybee.py`
**Module:** `braindance.examples.3_busybee`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
BusyBee continuous recording/frequency sweeps, from proj/busy_bee/continuous.py.

## Connections
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `RecordPhase` from `braindance.core.phases` — imports (static evidence).
- **Shared data:** Acquisition path loads JSON, copies `maxwell_params`, creates MaxwellEnv and PhaseManager, and repeats RecordPhase then NeuralSweepPhase for .5/1/2/4/8 Hz sweeps.

## Dependencies
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.phases.RecordPhase` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main(json_file=None, cycles=40, record_seconds=600, dry_run=False)`
> Prints a deterministic BusyBee schedule in dry-run mode or executes repeated recording and causal frequency sweeps on Maxwell hardware.
> **Called by:** braindance/examples/3_busybee.py:57 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/3_busybee.py:7`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `main(json_file=None, cycles=40, record_seconds=600, dry_run=False)`; CLI `--json/-j`, `--cycles` (40), `--record-seconds` (600), and `--dry-run`. |

## Data Shapes
- Dry-run returns one schedule dict per cycle/frequency with record_seconds, frequency_hz, replicates, amplitude_mv=400, phase_width_us=200, order=rna, and single_connect=True.

## Notes
- Dry-run performs no hardware access; acquisition requires JSON keys name, config, and stim_electrodes and closes MaxwellEnv in finally.
