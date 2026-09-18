# utils.py

**Path:** `braindance/analysis/utils.py`
**Module:** `braindance.analysis.utils`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Assigns electrode colors from normalized mapping coordinates. Converts coordinate-derived HSV triples to RGB values for spatial plotting.

## Connections
- **Shared data:** Uses matplotlib.colors.hsv_to_rgb.

## Dependencies
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `colors_from_mapping(mapping)`
> Takes in mapping, and returns colors based off of the x and y positions. Hue is determined by x, and saturation is determined by y.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/utils.py:6`

## Config / CLI
None

## Data Shapes
- DataFrame with x and y columns -> list of RGB arrays

## Notes
- Zero coordinate range produces division by zero during normalization.
