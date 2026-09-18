# params.py

**Path:** `braindance/core/params.py`
**Module:** `braindance.core.params`
**Feature Area:** `Configuration`
**Entry point:** no — library or imported component

## Overview
Provides a small default Maxwell experiment parameter dictionary. Defaults select config.cfg, no stimulation electrodes, five minutes, and a data output directory.

## Connections
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
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
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.sample_experiment` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.sample_manual_sequence` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.stim_pattern` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.stim_responses` — import consumer hint; not a proven runtime call.

## Dependencies
None

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| maxwell_params: config, stim_electrodes, max_time_sec, save_dir |

## Data Shapes
None

## Notes
None
