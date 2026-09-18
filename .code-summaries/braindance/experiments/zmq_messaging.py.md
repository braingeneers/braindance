# zmq_messaging.py

**Path:** `braindance/experiments/zmq_messaging.py`
**Module:** `braindance.experiments.zmq_messaging`
**Feature Area:** `Acquisition`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Publishes microscope command strings and records elapsed command windows to CSV. A Windows receiver forwards commands to PrairieLink until a stop sentinel arrives.

## Connections
- **Shared data:** CommandSender binds PUB; command_receiver connects SUB and calls PrairieLink.SendScriptCommands.

## Dependencies
- `pandas.DataFrame` — external or unresolved local import; source import evidence.
- `win32com.client` — external or unresolved local import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
### BaseEnv()
> unclear — see source
**Source:** `braindance/experiments/zmq_messaging.py:8`
**Kind:** class. **Instantiated by:** braindance/experiments/zmq_messaging.py:131 (named-call hint)
**Constructor:** `__init__(self) -> None`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `save_dir` | inferred at runtime | `'test'` |
**Methods:**
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/zmq_messaging.py:12`
### CommandSender()
> Class to send commands to another process with ZMQ and wait for it to finish
**Source:** `braindance/experiments/zmq_messaging.py:24`
**Kind:** class. **Instantiated by:** braindance/experiments/zmq_messaging.py:132 (named-call hint)
**Constructor:** `__init__(self, env: BaseEnv, log_save_name='2p_command_log.csv', ip_address=maxone_ip, port=1150, setup_time_sec=2, verbose=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `command_log` | inferred at runtime | `DataFrame(columns=['command', 'start_second', 'end_second'])` |
| `env` | inferred at runtime | `env` |
| `log_save_path` | inferred at runtime | `Path(env.save_dir) / log_save_name` |
| `start_socket_pub` | inferred at runtime | `start_socket_pub` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `send(self, command, duration_sec)`
> Send command
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/experiments/zmq_messaging.py:50`
#### `close(self)`
> Stops the CommandSender object by Sending stop command to command_receiver Closing sockets Saving command log
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/experiments/zmq_messaging.py:66`

## Functions
### `command_receiver(ip_address=maxone_ip, start_port=1150, verbose=True)`
> Receive commands from another process and then start the command, sending a signal when done
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/zmq_messaging.py:80`
### `main()`
> unclear — see source
> **Called by:** braindance/experiments/zmq_messaging.py:144 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/experiments/zmq_messaging.py:130`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default maxone_ip=128.111.208.217, port=1150; setup_time_sec=2 |
| STOP_COMMAND='STOP COMMAND_RECEIVER'; log_save_name=2p_command_log.csv |

## Data Shapes
- CSV columns command,start_second,end_second; ZMQ string messages

## Notes
- Sender sleeps for requested duration without receiving completion acknowledgment.
- Receiver requires win32com and PrairieLink.Application; main also sends -ZSeries after each entered command.
