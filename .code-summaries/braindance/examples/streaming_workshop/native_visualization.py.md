# native_visualization.py

**Path:** `braindance/examples/streaming_workshop/native_visualization.py`
**Module:** `braindance.examples.streaming_workshop.native_visualization`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Adapts native game state to browser scene geometry outside scientific phase code. A transparent game proxy samples reset/step results and atomically updates the native runner progress file.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Shared data:** native_runner installs ObservedGame around supported phase.game_env values during execution.
- **Shared data:** Scene fields match the existing browser watcher canvas renderers.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### ObservedGame()
> Observe game lifecycle and write progress without changing native phase control logic.
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:33`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/native_runner.py:201 (named-call hint)
**Constructor:** `__init__(self, environment, kind, progress, phase_id)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `episodes` | inferred at runtime | `0` |
| `last_publish` | inferred at runtime | `0.0` |
| `reward` | inferred at runtime | `0.0` |
**Methods:**
#### `__getattr__(self, name)`
> Delegate attributes and methods not implemented by the observer.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:42`
#### `publish(self, observation, force=False)`
> Throttle geometry sampling and atomically replace the browser progress snapshot.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:45`
#### `reset(self, *args, **kwargs)`
> Reset the wrapped environment and immediately publish its initial observation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:57`
#### `step(self, *args, **kwargs)`
> Advance the wrapped game, accumulate reward/completed episodes and publish current geometry.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:62`

## Functions
### `game_scene(kind, environment, observation)`
> Extract JSON-safe CartPole observations, FoodLand positions or live MuJoCo Ant geometry from nested game wrappers.
> **Called by:** braindance/examples/streaming_workshop/native_visualization.py:50 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_visualization.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ObservedGame receives an environment, kind (cartpole/foodland/ant), progress Path and phase ID. |
| Normal publications are limited to one every 0.05 seconds; resets and episode endings force publication. |

## Data Shapes
- All scenes include kind and an observation list; CartPole uses that observation directly.
- FoodLand adds agent position, direction, food and hazard coordinate lists.
- Ant adds root xyz and sphere/capsule geometry entries with endpoint vectors a/b and radius from current MuJoCo model/data.
- Progress JSON contains phase, phase_kind=environment, scene, cumulative reward and completed episodes.

## Notes
- Nested env wrappers are traversed to access the physical game; Ant floor and other unsupported geometry types are skipped.
- No desktop renderer or video encoding is required. publish writes a sibling .tmp file then replaces the progress JSON atomically.
- The proxy returns original reset/step values and delegates other attributes. Episodes increment for termination or Gymnasium truncation.
