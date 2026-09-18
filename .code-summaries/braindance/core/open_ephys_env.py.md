# open_ephys_env.py

**Path:** `braindance/core/open_ephys_env.py`
**Module:** `braindance.core.open_ephys_env`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Wraps Open Ephys HTTP recording control and Falcon Output ZMQ acquisition. Embedded FlatBuffers accessors decode continuous channel-major samples into frame-major observations.

## Connections
- **Uses:** `BaseEnv` from `braindance.core.base_env` — imports (static evidence).
- **Shared data:** OpenEphysHTTPServer controls record/acquire/idle; ContinuousData reads received FlatBuffers.

## Dependencies
- `braindance.core.base_env.BaseEnv` — intra-repo import; source import evidence.
- `flatbuffers` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `open_ephys.control.OpenEphysHTTPServer` — external or unresolved local import; source import evidence.
- `zmq` — external or unresolved local import; source import evidence.

## Classes
### OpenEphysEnv(BaseEnv)
> The OpenEphysEnv class extends from the BaseEnv class and implements a specific environment for interacting with Open Ephys
**Source:** `braindance/core/open_ephys_env.py:9`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, max_time_sec=60, verbose=1, ip_address='127.0.0.1', port=3335, start=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cur_time` | inferred at runtime | `time.perf_counter()` |
| `gui` | inferred at runtime | `OpenEphysHTTPServer(ip_address)` |
| `socket` | inferred at runtime | `socket` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
**Methods:**
#### `step(self, action=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:40`
#### `_cleanup(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:73`
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:77`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:80`
### ContinuousData(object)
> unclear — see source
**Source:** `braindance/core/open_ephys_env.py:92`
**Kind:** class. **Instantiated by:** braindance/core/open_ephys_env.py:98 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_tab` | inferred at runtime | `flatbuffers.table.Table(buf, pos)` |
**Methods:**
#### `GetRootAs(cls, buf, offset=0)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:96`
#### `GetRootAsContinuousData(cls, buf, offset=0)`
> This method is deprecated. Please switch to GetRootAs.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:103`
#### `Init(self, buf, pos)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/open_ephys_env.py:108`
#### `Samples(self, j)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:112`
#### `SamplesAsNumpy(self) -> np.ndarray`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:120`
#### `SamplesLength(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:127`
#### `SamplesIsNone(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:134`
#### `Stream(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:139`
#### `NChannels(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:146`
#### `NSamples(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:154`
#### `SampleNum(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:162`
#### `Timestamp(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:170`
#### `MessageId(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:178`
#### `SampleRate(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:186`

## Functions
### `Start(builder)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:199 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:194`
### `ContinuousDataStart(builder)`
> This method is deprecated. Please switch to Start.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:197`
### `AddSamples(builder, samples)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:208 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:202`
### `ContinuousDataAddSamples(builder, samples)`
> This method is deprecated. Please switch to AddSamples.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:206`
### `StartSamplesVector(builder, numElems)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:217 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:211`
### `ContinuousDataStartSamplesVector(builder, numElems)`
> This method is deprecated. Please switch to Start.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:215`
### `AddStream(builder, stream)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:226 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:220`
### `ContinuousDataAddStream(builder, stream)`
> This method is deprecated. Please switch to AddStream.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:224`
### `AddNChannels(builder, nChannels)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:235 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:229`
### `ContinuousDataAddNChannels(builder, nChannels)`
> This method is deprecated. Please switch to AddNChannels.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:233`
### `AddNSamples(builder, nSamples)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:243 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:238`
### `ContinuousDataAddNSamples(builder, nSamples)`
> This method is deprecated. Please switch to AddNSamples.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:241`
### `AddSampleNum(builder, sampleNum)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:252 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:246`
### `ContinuousDataAddSampleNum(builder, sampleNum)`
> This method is deprecated. Please switch to AddSampleNum.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:250`
### `AddTimestamp(builder, timestamp)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:261 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:255`
### `ContinuousDataAddTimestamp(builder, timestamp)`
> This method is deprecated. Please switch to AddTimestamp.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:259`
### `AddMessageId(builder, messageId)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:270 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:264`
### `ContinuousDataAddMessageId(builder, messageId)`
> This method is deprecated. Please switch to AddMessageId.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:268`
### `AddSampleRate(builder, sampleRate)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:279 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:273`
### `ContinuousDataAddSampleRate(builder, sampleRate)`
> This method is deprecated. Please switch to AddSampleRate.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:277`
### `End(builder)`
> unclear — see source
> **Called by:** braindance/core/open_ephys_env.py:287 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:282`
### `ContinuousDataEnd(builder)`
> This method is deprecated. Please switch to End.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/open_ephys_env.py:285`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ip_address=127.0.0.1; port=3335; start=True |

## Data Shapes
- step returns (samples shaped (num_samples, num_channels), done); parse/size failures return empty (1,1).

## Notes
- Actions are unimplemented; reset is a no-op; start=False leaves timing uninitialized.
