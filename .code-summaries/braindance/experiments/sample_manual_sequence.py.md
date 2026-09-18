# sample_manual_sequence.py

**Path:** `braindance/experiments/sample_manual_sequence.py`
**Module:** `braindance.experiments.sample_manual_sequence`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds a square-wave maxlab DAC sequence and submits it as a manual stimulation action. The main loop repeatedly drives two configured stimulation electrodes.

## Connections
- **Uses:** `dummy_maxlab` from `braindance.core` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Shared data:** Sequence appends chip.DAC and system.DelaySamples; MaxwellEnv.step sends manual action.

## Dependencies
- `braindance.core.dummy_maxlab` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `maxlab` — external or unresolved local import; source import evidence.
- `maxlab.chip` — external or unresolved local import; source import evidence.
- `maxlab.saving` — external or unresolved local import; source import evidence.
- `maxlab.system` — external or unresolved local import; source import evidence.
- `maxlab.util` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `my_waveform(amplitude_mV=150, freq=1000)`
> Adds a square wave to the sequence with a set amplitude and frequency.
> **Called by:** braindance/experiments/sample_manual_sequence.py:90 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/experiments/sample_manual_sequence.py:31`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| my_waveform defaults amplitude_mV=150,freq=1000; sampling_rate=20000; DAC baseline=512, scale=2.9 mV/LSB |

## Data Shapes
- my_waveform -> maxlab.Sequence; action ('manual',sequence,[0,1])

## Notes
- Falls back to dummy_maxlab if maxlab import fails; no validation for invalid/high frequencies.
