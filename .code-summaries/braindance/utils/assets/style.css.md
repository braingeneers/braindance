# style.css

**Path:** `braindance/utils/assets/style.css`
**Module:** `braindance.utils.assets.style.css`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Sets a black page background and white body text for browser-based views that load this stylesheet. It also declares body margin and padding as unitless nonzero values, which standard CSS engines ignore.

## Connections
- **Shared data:** Applies globally to the document body when included by a browser view; the stylesheet itself defines no loader or application wiring.

## Dependencies
None

## Classes
None

## Functions
### `body`
> Apply global background and foreground colors.
**Source:** `braindance/utils/assets/style.css:1`

## Config / CLI
None

## Data Shapes
- CSS body selector declarations: background-color, color, margin and padding.

## Notes
- The black background uses !important; margin: 4 and padding: 2 omit required units.
