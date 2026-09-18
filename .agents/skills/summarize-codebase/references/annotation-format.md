# Batch review format

Read each assigned source file fully. Return a JSON object keyed by repo-relative
POSIX source path. Compute `source_sha256` from the bytes actually reviewed (raw bytes
or bytes with CRLF normalized to LF). Do not hash a later version without reviewing it.

```json
{
  "braindance/example.py": {
    "source_sha256": "<sha256>",
    "overview": "What this file does and its role. When and how users invoke it.",
    "feature_area": "Neural Analysis",
    "config": ["`bin_ms=20`: width of spike-count bins in milliseconds."],
    "data_shapes": ["Input spike rows contain time and unit identifiers; output is units × time bins."],
    "notes": ["Explain source-backed global state, hardware needs, hardcoded paths, or subtle cache behavior."],
    "connections": ["Which data passes between named modules, and through which entry points."],
    "purposes": {
      "ClassName": "Purpose and system role.",
      "ClassName.method": "What and why, not line-by-line implementation.",
      "function_name": "Purpose; side effects and actual callers when verified."
    }
  }
}
```

All four lists are required; empty means `None`, not unreviewed. For JavaScript,
HTML and CSS, add `symbols`: a list of `{name, signature, purpose, line}` for useful
functions/handlers. Line numbers are one-based. Python symbols come from AST.
Supply purposes for undocumented public APIs and key nontrivial methods; documented
symbols can use their docstrings. Record unresolved behavior explicitly rather than
filling with plausible guesses. Include CLI arguments, environment reads, units,
array orientation, cache format and side effects where present.

Each generated summary has these sections, even when `None`:

```markdown
# filename
**Path:** `repo/path.py`
**Module:** `pkg.module`
**Feature Area:** `canonical tag`
**Entry point:** yes/no — explanation

## Overview
Two precise sentences.

## Connections
- **Uses:** `Symbol` from `module` — source-backed relation.
- **Used by:** `consumer` — import hint, not a proven runtime call.
- **Shared data:** concrete flow.

## Dependencies
- `package` — role.

## Classes
### ClassName(Base)
> Purpose.
**Source:** `path:line`
**Kind:** class. **Instantiated by:** known caller or unclear — see source
**Constructor:** `__init__(...)`
**Key attributes:** name/type/assignment table
**Methods:**
#### `method(...) -> R`
> Purpose.
> **Called by:** verified caller or unresolved. **Side effects:** known effects or unclear.

## Functions
### `function(...) -> R`
> Purpose. **Called by:** evidence. **Side effects:** evidence.
**Source:** `path:line`

## Config / CLI
Arguments, defaults, environment variables and configuration keys.

## Data Shapes
Inputs, outputs, schemas, units and cache formats.

## Notes
Subtle behavior and assumptions.
```

Connections intentionally precede long API details so short bounded reads are useful.
Do not use imports as proof of exact call sites. Full method signatures may be long;
navigation should grep a symbol and read its neighborhood rather than entire files.
