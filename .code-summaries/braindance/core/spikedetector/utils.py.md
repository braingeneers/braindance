# utils.py

**Path:** `braindance/core/spikedetector/utils.py`
**Module:** `braindance.core.spikedetector.utils`
**Feature Area:** `Spike Detection`
**Entry point:** no — library or imported component

## Overview
Provides binary confusion counts and percentage metrics plus reproducibility and tensor conversion helpers. Also supplies timestamp names and source-copy support for saved detector artifacts.

## Connections
- **Used by:** `braindance.core.spikedetector.model2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.plot` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.train` — import consumer hint; not a proven runtime call.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `confusion_matrix(preds, labels)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:9`
### `confusion_stats(confusion_matrix)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:20`
### `random_seed(seed, silent=False)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:29`
### `get_time()`
> Gets the time in the string format of yymmdd_HHMMSS_ffffff (https://www.programiz.com/python-programming/datetime/strftime) Ex: If run at 9/26/22 at 3:53pm and 30 seconds and 1 microsecond, yymmdd_HHMMSS_ffffff = 220926_155330_000001
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:37`
### `copy_file(src_path, dest_folder)`
> Copies file at src_path and stores in dest_folder
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:49`
### `round(n)`
> Rounds a float (n) to the nearest integer Uses standard math rounding, i.e. 0.5 rounds up to 1
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:65`
### `torch_to_np(tensor)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/utils.py:79`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| random_seed seeds Python, NumPy, and PyTorch globally |

## Data Shapes
- Confusion order is FN,TN,FP,TP; stats returns accuracy/recall/precision percentages.

## Notes
- Custom round array branch truncates fractional differences before doubling.
