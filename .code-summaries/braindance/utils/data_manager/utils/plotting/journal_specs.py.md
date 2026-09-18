# journal_specs.py

**Path:** `braindance/utils/data_manager/utils/plotting/journal_specs.py`
**Module:** `braindance.utils.data_manager.utils.plotting.journal_specs`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Defines frozen journal page and typography specifications used by Styler for publication geometry, text-size roles, DPI floors, and compliance behavior.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.plotting.plot_styler` — import consumer hint; not a proven runtime call.
- **Shared data:** `Styler` resolves `journal` through `get_journal_spec()` and uses the resulting geometry and text-role fields for figure sizing and compliance.

## Dependencies
None

## Classes
### TextSpec()
> Defines frozen typography role sizes and panel-letter rules shared by paper and draft plots.
**Source:** `braindance/utils/data_manager/utils/plotting/journal_specs.py:43`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/plotting/journal_specs.py:98 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### JournalPageSpec()
> Defines a journal profile’s page geometry, DPI floors, typography, page presence, and strict compliance behavior.
**Source:** `braindance/utils/data_manager/utils/plotting/journal_specs.py:102`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/plotting/journal_specs.py:141 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:151 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:176 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:218 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None

## Functions
### `_mm_to_pt(mm: float) -> float`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/plotting/journal_specs.py:143 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:144 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:145 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:153 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:154 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:155 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:220 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:221 (named-call hint); braindance/utils/data_manager/utils/plotting/journal_specs.py:222 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/journal_specs.py:38`
### `get_journal_spec(journal) -> JournalPageSpec`
> Resolve a journal name (str) or an already-built `JournalPageSpec` to a spec. Passing a `JournalPageSpec` directly through lets a caller define a one-off/local profile without adding it to the shared registry.
> **Called by:** braindance/utils/data_manager/utils/plotting/plot_styler.py:112 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/journal_specs.py:242`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `JOURNAL_SPECS` registers `nature_neuro`, `nature`, `cell_reports_methods`, and `draft`; `get_journal_spec()` accepts a registered name or JournalPageSpec instance. |
| Page dimensions are stored in points after millimeter conversion; TextSpec/JournalPageSpec carry text role sizes, panel-letter settings, DPI floors, and strictness. |
| Supported journal keys are `nature_neuro`, `nature`, `cell_reports_methods`, and `draft`; `get_journal_spec()` also accepts a JournalPageSpec instance. |

## Data Shapes
- JournalPageSpec and TextSpec are frozen dataclasses containing page geometry, typography role sizes, DPI thresholds, has_page, and strict fields.

## Notes
- `draft` has no page geometry and warns rather than raising; journal profiles enforce different text bands and panel-letter case.
- Cell Reports Methods values include source uncertainty documented in the module comments.
