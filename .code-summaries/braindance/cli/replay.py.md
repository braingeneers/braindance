# replay.py

**Path:** `braindance/cli/replay.py`
**Module:** `braindance.cli.replay`
**Feature Area:** `Replay`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Provides list, doctor and run commands for hardware-free Maxwell H5 replay. Doctor checks imports, direct replay output compatibility and ZMQ transport, while run constructs a V3 recording with optional simulated stimulation.

## Connections
- **Uses:** `load_data_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `FrequencyStimPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `SpikeEvent` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `find_recordings` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `inspect_recording` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `pack_replay_packet` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `resolve_replay_source` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `unpack_replay_packet` from `braindance.core.replay` — imports (static evidence).
- **Shared data:** core.replay resolves/inspects sources and packs packets.
- **Shared data:** run_command creates phases_v3.Experiment with RecordPhaseV3 and optional FrequencyStimPhaseV3.
- **Shared data:** doctor checks load_data_maxwell and SpikeInterface can read replay-written H5.

## Dependencies
- `braindance.analysis.data_loader.load_data_maxwell` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.FrequencyStimPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.replay.SpikeEvent` — intra-repo import; source import evidence.
- `braindance.core.replay.find_recordings` — intra-repo import; source import evidence.
- `braindance.core.replay.inspect_recording` — intra-repo import; source import evidence.
- `braindance.core.replay.pack_replay_packet` — intra-repo import; source import evidence.
- `braindance.core.replay.resolve_replay_source` — intra-repo import; source import evidence.
- `braindance.core.replay.unpack_replay_packet` — intra-repo import; source import evidence.
- `maxlab` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `_print_recording(info, root=None)`
> Formats recording metadata for the list and doctor commands.
> **Called by:** braindance/cli/replay.py:118 (named-call hint); braindance/cli/replay.py:51 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:29`
### `list_command(args)`
> Lists compatible replay recordings under the selected data root.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:42`
### `_zmq_loopback()`
> Verifies replay packet transport through a local ZMQ PUB/SUB loopback.
> **Called by:** braindance/cli/replay.py:161 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:55`
### `doctor_command(args)`
> Checks replay dependencies, input compatibility, direct MaxwellEnv output, and ZMQ transport.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:86`
### `_parse_speed(value)`
> Parses a nonnegative replay speed or the max sentinel.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:175`
### `run_command(args)`
> Resolves a replay source, runs a V3 recording phase with optional stimulation, and writes output unless disabled.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:184`
### `build_parser()`
> Builds the list, doctor, and run command-line parser.
> **Called by:** braindance/cli/replay.py:290 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:238`
### `main(argv=None)`
> Dispatches the replay CLI subcommand and returns its shell status.
> **Called by:** braindance/cli/replay.py:295 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/cli/replay.py:285`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Subcommands: list, doctor, run |
| Source flags --h5, --recording, --sample; --data-dir, --output-dir, --name |
| --record-seconds, --speed RATE\\|max, --stim-seconds, --stim-electrode, --stim-hz, --amplitude-mv, --phase-width-us, --no-write-output, --verbose |
| doctor --require-env |

## Data Shapes
- Recording metadata dictionaries contain path, size_bytes, num_channels, sampling_hz, duration_s and version
- Returns shell status 0 on success and 1 on diagnostic/run failure

## Notes
- EXAMPLE_H5_URL and EXAMPLE_H5_SHA256 are unset placeholders.
- Doctor creates temporary output and opens loopback PUB/SUB sockets.
