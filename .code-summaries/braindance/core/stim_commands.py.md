# stim_commands.py

**Path:** `braindance/core/stim_commands.py`
**Module:** `braindance.core.stim_commands`
**Feature Area:** `Stimulation`
**Entry point:** no — library or imported component

## Overview
Builds pulse command tuples from individual or grouped electrode indices. Distinguishes single pulses from tagged command sequences by inspecting the nested first element.

## Connections
- **Used by:** `braindance.core.phases2_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.multi_freq_stim` — import consumer hint; not a proven runtime call.

## Dependencies
None

## Classes
None

## Functions
### `is_single_command(command)`
> Check if the given command is a single command or sequence of commands
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/stim_commands.py:3`
### `generate_stimulations(electrode_inds, amp=400, phase_width=200)`
> Creates a list of stimulation commands for the given electrodes with the given amplitude and phase width
> **Called by:** braindance/core/phases2_loop.py:127 (named-call hint); braindance/experiments/multi_freq_stim.py:24 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/stim_commands.py:14`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| amp=400 mV; phase_width=200 microseconds |

## Data Shapes
- Output list of (electrode_indices, amplitude, phase_width) tuples.

## Notes
- Command classification requires nonempty nested inputs.
