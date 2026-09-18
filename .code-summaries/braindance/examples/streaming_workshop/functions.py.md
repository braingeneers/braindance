# functions.py

**Path:** `braindance/examples/streaming_workshop/functions.py`
**Module:** `braindance.examples.streaming_workshop.functions`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Supplies editable encode, decode and optional train hooks for streaming workshop control. Decodes baseline-centered smoothed rates into actions and encodes one observation component into two bounded stimulation rates.

## Connections
- **Shared data:** WorkshopSession loads these functions; WorkshopGame applies decoded actions.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `decode(spike_counts, dt_s, params, state)`
> Return action vector and intermediate diagnostics; actions use [-1, 1].
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/functions.py:10`
### `encode(observation, dt_s, params, state)`
> Map a selected sensory variable to two stimulation rates in Hz.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/functions.py:32`
### `train(transition, dt_s, params, state)`
> Optional learning hook, called after game.step and before encode.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/functions.py:41`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| baseline_hz,left_channels,right_channels,decoder_gain,action_size,environment |
| sensory_index,encoder_gain,max_stim_hz |

## Data Shapes
- decode(counts,dt_s,params,state) -> action vector,diagnostics
- encode(observation,dt_s,params,state) -> two rates,diagnostics; train(transition,dt_s,params,state) -> dict

## Notes
- State resets on reload, phase restart and episode reset; train currently returns empty dict.
- Ant output is illustrative channel cycling, not trained walking control.
