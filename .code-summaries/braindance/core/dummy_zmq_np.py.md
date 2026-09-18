# dummy_zmq_np.py

**Path:** `braindance/core/dummy_zmq_np.py`
**Module:** `braindance.core.dummy_zmq_np`
**Feature Area:** `Replay`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Publishes lazy replay frames through the legacy Maxwell ZMQ transport. Explicit stop and readiness events, frame limits, and final cleanup support bounded transport compatibility tests.

## Connections
- **Used by:** `braindance.core.maxwell_env` — import consumer hint; not a proven runtime call.
- **Uses:** `H5ReplaySource` from `braindance.core.replay` — imports (static evidence).
- **Uses:** `pack_replay_packet` from `braindance.core.replay` — imports (static evidence).
- **Shared data:** H5ReplaySource.read -> pack_replay_packet -> filtered and unfiltered PUB sockets; new acquisition code should use MaxwellEnv(replay=...).

## Dependencies
- `braindance.core.replay.H5ReplaySource` — intra-repo import; source import evidence.
- `braindance.core.replay.pack_replay_packet` — intra-repo import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `run(data_path, speed=1.0, loop=True, start_frame=0, stop_frame=None, channels=None, unfiltered_endpoint='tcp://*:7204', filtered_endpoint='tcp://*:7205', stop_event=None, ready_event=None, max_frames=None)`
> Publish replay frames using Maxwell's three-part dummy packet format.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_zmq_np.py:14`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| speed, loop, frame bounds, channels, endpoint overrides, max_frames |

## Data Shapes
- Three-part packets encoded by pack_replay_packet.

## Notes
- Default loop=True; waits 0.1 seconds for PUB/SUB handshake.
