# source_links.py

**Path:** `braindance/examples/streaming_workshop/source_links.py`
**Module:** `braindance.examples.streaming_workshop.source_links`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds commit-pinned GitHub source links for workshop analyses without importing their scientific implementations. It reads definitions from the repository's committed HEAD so URL line fragments remain consistent with the linked revision.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_catalog` — import consumer hint; not a proven runtime call.
- **Shared data:** Uses Git CLI `rev-parse`, `remote get-url`, and `show` plus Python AST parsing to resolve committed definitions without importing workshop analysis modules.
- **Shared data:** Maps manual analysis kinds to `AnalysisWorkspace._analyze`, `analysis_evoked.analyze_evoked`, and `analysis_sorting.run_sorting`.

## Dependencies
None

## Classes
None

## Functions
### `_git(root, *args)`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/source_links.py:19 (named-call hint); braindance/examples/streaming_workshop/source_links.py:20 (named-call hint); braindance/examples/streaming_workshop/source_links.py:23 (named-call hint); braindance/examples/streaming_workshop/source_links.py:32 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/source_links.py:10`
### `_repository()`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/source_links.py:54 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/source_links.py:17`
### `_definitions(root, revision, path)`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/source_links.py:59 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/source_links.py:30`
### `source_url(module, symbol)`
> Resolve a committed module symbol to a stable GitHub URL pinned to the current repository HEAD, returning no link when the origin or definition is unavailable.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:27 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:47 (named-call hint); braindance/examples/streaming_workshop/source_links.py:66 (named-call hint); braindance/examples/streaming_workshop/source_links.py:68 (named-call hint); braindance/examples/streaming_workshop/source_links.py:69 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/source_links.py:48`
### `analysis_source_urls()`
> Return source links for each browser analysis kind, reusing the workspace dispatcher link for its four built-in spike/raw analyses.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:117 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/source_links.py:63`

## Config / CLI
None

## Data Shapes
- `source_url` accepts a dotted Python module and qualified symbol name, returning a GitHub blob URL with commit SHA and one-based line fragment or `None`.
- `analysis_source_urls` returns keys `raw`, `spikes`, `sttc`, `latency`, `overlap`, and `sorting`; the first four share the `AnalysisWorkspace._analyze` definition URL.

## Notes
- Only GitHub HTTPS, SCP-style SSH, and `ssh://git@github.com/` origin URLs are recognized; missing Git metadata, unpublished files/symbols, command failures, or syntax errors disable links.
- Git subprocesses have a five-second timeout; repository metadata is cached once and per-file definition maps are cached up to 128 entries.
