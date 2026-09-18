# validators.py

**Path:** `braindance/utils/data_manager/utils/catalogging/validators.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.validators`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Defines experiment-name baseline rules and extracts project/experiment identifiers and ordering keys. Estimates stimulation frequency from short timestamp differences or count heuristics and snaps estimates to supported frequency categories.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.experiment` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.generator` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.postprocessing` — import consumer hint; not a proven runtime call.
- **Shared data:** experiment derives stimulus metadata; postprocessing derives baseline flags and earlier experiment ordering.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `is_baseline_with_reason(experiment, freq=None)`
> Determine if an experiment is a baseline and return the reason.
> **Called by:** braindance/utils/data_manager/utils/catalogging/validators.py:110 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:16`
### `is_baseline(experiment, freq=None)`
> Determine if an experiment is a baseline.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:80`
### `calculate_stim_frequency(log)`
> Calculate stimulation frequency from a stimulus log DataFrame.
> **Called by:** braindance/utils/data_manager/utils/catalogging/experiment.py:193 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:114`
### `categorize_frequency(freq)`
> Categorize a frequency into predefined stimulation paradigms.
> **Called by:** braindance/utils/data_manager/utils/catalogging/experiment.py:195 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:157`
### `calculate_freq_from_stim_count(n_stims)`
> Estimate frequency based on stimulation count.
> **Called by:** braindance/utils/data_manager/utils/catalogging/experiment.py:199 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:187`
### `extract_proj_from_s3_path(s3_path)`
> Extract project identifier from an S3 path.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:222`
### `extract_exp_from_experiment_path(experiment)`
> Extract experiment type from experiment path.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:258`
### `count_digits(string)`
> Count the number of digits in a string.
> **Called by:** braindance/utils/data_manager/utils/catalogging/generator.py:88 (named-call hint); braindance/utils/data_manager/utils/catalogging/generator.py:99 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:280`
### `extract_number(file_string, regex='[_.](\\d+)[_.]')`
> Extract a number from a file string.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:285`
### `parse_exp_number(exp_name)`
> Extract numeric value from exp name for chronological ordering.
> **Called by:** braindance/utils/data_manager/utils/catalogging/validators.py:362 (named-call hint); braindance/utils/data_manager/utils/catalogging/validators.py:363 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:302`
### `compare_exp_names(exp_a, exp_b)`
> Return True if exp_a is "earlier" than exp_b.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:401 (named-call hint); braindance/utils/data_manager/utils/catalogging/postprocessing.py:410 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/validators.py:337`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Frequency categories 0.5/1/2/4/8 Hz; timestamp gaps >=10 seconds excluded. |

## Data Shapes
- is_baseline_with_reason→(bool,reason); stimulation DataFrame requires time seconds; path helpers return strings/None.

## Notes
- Baseline logic uses names rather than supplied freq argument; BL1 exact variants handled separately from other BL names.
- Frequency interval calculation does not reject zero/negative intervals.
