# dummy_zmq.py

**Path:** `braindance/core/dummy_zmq.py`
**Module:** `braindance.core.dummy_zmq`
**Feature Area:** `Replay`
**Entry point:** no — library or imported component

## Overview
Runs a synthetic Maxwell-like ZMQ publisher with random frame values and spike events. Both filtered and unfiltered sockets publish the same three-part packets in an infinite loop.

## Connections
None

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| tcp://*:7204 and tcp://*:7205; n_channels=1024 |

## Data Shapes
- Packet parts: uint64 frame number, bytes(frame_data), packed 8xLif spike records.

## Notes
- Import immediately binds sockets and starts an infinite unpaced loop.
- Raw bytes are one byte per channel, unlike the float32 contract of current replay transport.
