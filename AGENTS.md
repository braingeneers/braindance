# Codex Coding Preferences for BrainDance

## Codebase navigation — READ THIS FIRST
For questions about `braindance/`, use `.code-summaries/` before broad source reads.
Start with [navigate](.agents/skills/navigate/SKILL.md); any agent can follow this file.
`_INDEX.md` maps files/symbols; `_FEATURES.md` maps behavior; `_GRAPH.md` maps import edges.
Each `braindance/<path>.<ext>.md` describes its source; `_META.json` tracks freshness.
```sh
rg --hidden -n -i 'KEYWORD' .code-summaries/_FEATURES.md
rg --hidden -n 'SYMBOL' .code-summaries/_INDEX.md
rg --hidden -n 'MODULE' .code-summaries/_GRAPH.md
rg --hidden -n -F '**Uses:** `SYMBOL`' .code-summaries/braindance
```
Read only matching sections. Verify source locations before citing `path:line`.
Once per session: `python .agents/skills/summarize-codebase/scripts/summaries.py check`.
Stale relevant file → read its source; commit-only checks miss dirty/partial updates.
Refresh: [summarize-codebase](.agents/skills/summarize-codebase/SKILL.md), incremental by
default; `--all` rebuilds all, `--since REF` or a quoted path/glob narrows selection.
Codex: `$navigate` / `$summarize-codebase`; other agents: read the linked instructions.
Holes: `proj/`, tests by default, notebooks/assets, archived/manuscript/internal-usage
scripts; exact exclusions appear in `_INDEX.md`. Exact source paths bypass navigation.
These files work from a repo checkout; a wheel-only install does not include them.

## Code Organization
- Prefer inline logic in `main()` over small helper functions
- Only extract truly reusable/generic operations
- Put configuration as `main()` default parameters (not top-level constants):
```python
def main(
    proj="24-01-07_data",
    chip="20217",
    experiments=None,
    bin_ms=20,
    n_subsample=500,
    recompute=False,
):
    if experiments is None:
        experiments = ["exp2/exp2_cartpole_long_6"]
    ...
```

## Paths & Configuration
### Data Input (handled automatically)
- Use `load_catalog()` - data paths configured in data_manager
- Never hardcode data paths or use `get_output_dir()` for input

### S3 Access
- Use the endpoint "aws --endpoint https://s3-west.nrp-nautilus.io"

### Output Paths
- Use `get_output_dir()` for plots/results only
- Create script subdirectories: `get_output_dir() / "my_analysis"`
- **Never call `set_output_dir()`** in scripts

## Data Manager APIs
**Use standardized tooling:**
- `load_catalog()` + `Recording` objects
- `calculate_latencies()` for evoked responses
- `bin_spike_data_vectorized()` for binning
- `Styler` for plots
- Avoid external helpers if data_manager has equivalent

## Data Structures
- Prefer dicts, numpy arrays, pandas DataFrames over rigid formats (AnnData)
- Use pickle (`.pkl`) for caching flexibility

## Caching
### File-level caching
```python
if not recompute and cache_file.exists():
    # Load cached
else:
    # Compute and save
```

### Per-recording caching
Use `rec.results` for persistent per-recording cache:
```python
cache_key = 'my_analysis'
if not recompute and cache_key in rec.results:
    data = rec.results[cache_key]['data']
else:
    data = compute_data(...)
    rec.results[cache_key] = {'data': data, 'params': PARAMS}
```

## Code Style
- Call `rec.clear_cache()` after processing each recording
- Print progress for long operations

## Standard Imports
```python
from braindance.utils.data_manager import (
    load_catalog, Styler, bin_spike_data_vectorized, calculate_latencies
)
from braindance.config import (
    get_output_dir,      # For output/plots directory
    get_data_dir,        # For data input directory (rarely needed)
    get_catalog_path,    # For catalog CSV path (rarely needed)
    # Never use: set_output_dir, set_data_dir, set_catalog_path
) 
```

## Workflow Rules
Before exploring the full codebase, ask the user which specific files or directories are relevant. Do not start broad codebase exploration unless explicitly asked.

## Delegate bulk reading to subagents (prefer cheaper models)
When a task requires reading several large files for context (multi-file refactors, cross-file consistency checks, "go look at how X works"), dispatch a subagent to do the reading and return a focused summary instead of pulling every file into the main (Opus) context. Reserve direct `Read` in the main thread for the small set of files actually being edited. Rule of thumb: if the answer needs >2 files or any file >500 lines just for context, delegate.

Pick the cheapest model that can do the job:
- **Haiku** — default for mechanical reads: locating files, grepping for symbols, summarizing one or two files, listing call sites, extracting a function signature. Use the `Explore` agent or `general-purpose` agent with `model: "haiku"`.
- **Sonnet** — when judgment is required: multi-file design analysis, cross-file consistency reasoning, planning a refactor, evaluating tradeoffs across components. Pass `model: "sonnet"`.
- **Opus (main thread)** — only synthesizes the subagent results and does the actual edits. Don't use Opus for raw reading.

When in doubt, start with Haiku and escalate to Sonnet only if the Haiku output is too shallow.

Disagree if you think the user is wrong, but always explain the underlying reasononing and assumptions as there may be other context you have not received.

## Debugging
When the user reports a bug or unexpected behavior, ask what changed recently before proposing fixes. Do not immediately jump to modifying settings or code without understanding the triggering change.

## Environment
This project primarily uses Python with conda env 'brain'. Use unless otherwise specified.

## Testing & Verification
When verifying a fix, test exactly what the user asked to verify — not related but different metrics. For example, if user says 'verify routing is correct,' check routing, not R² performance.
