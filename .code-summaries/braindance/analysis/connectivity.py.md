# connectivity.py

**Path:** `braindance/analysis/connectivity.py`
**Module:** `braindance.analysis.connectivity`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Calculates a cross-correlogram using FFT convolution of spike-count vectors. Returns rounded coincidence counts alongside the requested inclusive lag range.

## Connections
- **Shared data:** Uses scipy.signal.fftconvolve on reversed bt1 and padded bt2.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `ccg(bt1, bt2, ccg_win=[-10, 10], t_lags_shift=0)`
> Cross-correlogram of two non-sparse binary spike trains. Values must be counts of spikes, so NO floats Parameters ---------- bt1 : array_like First binary spike train. bt2 : array_like Second binary spike train. ccg_win : array_like Window for cross-correlogram. t_lags_shift : array_like Shift in time lags. Returns ------- cross_corr : array_like Cross-correlogram. lags : array_like Time lags.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/connectivity.py:6`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ccg_win defaults to [-10, 10]; t_lags_shift defaults to 0 |

## Data Shapes
- Two one-dimensional spike-count trains -> (cross_corr array, lags array)

## Notes
- Integer-valued input assertion checks bt1 only; asymmetric windows should be checked against returned convolution length.
