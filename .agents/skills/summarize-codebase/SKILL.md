---
name: summarize-codebase
description: Build or refresh BrainDance code summaries when asked to "summarize codebase", "refresh summaries", "update code map", or "summarize-codebase". Covers braindance/ only; incremental by default.
---

# Summarize BrainDance

Run from a repository checkout. Python 3.10+ and Git suffice; no BrainDance imports,
API keys, vendor SDK, or personal configuration required. Any coding agent can follow
this file; use parallel subagents when available, otherwise process batches sequentially.
Codex can invoke `$summarize-codebase`; other agents can read this file directly.
Arguments: `[--all | --since <git-ref> | <path-or-glob> ...]`.

## Scope and feature areas

Source root: `braindance/`. Extensions: `.py`, `.js`, `.html`, `.css`.
Skip tests unless `--include-tests`, notebooks/assets/weights/data, caches, generated,
vendor/third_party, build/dist/node_modules, `archive`, `internal_usage`, `stdp_pairs`,
`manuscript_code`, `user_manual`, and `TODO_*` placeholders. List actual holes in index.
Trivial `__init__.py` files get index export listings, not individual summaries.
Nontrivial initializers, tutorials, examples, and legacy modules remain in scope.
Never expand to `proj/` or unrelated root tooling without a user request.

Canonical feature areas (one primary tag per file):
Configuration; Acquisition; Experiment Phases; Stimulation; Replay; Spike Detection;
Spike Sorting; Artifact Removal; Data Catalog; Neural Analysis; Visualization;
Games and Reinforcement; Examples and Workshop.

## Plan, review, build

1. Run `python .agents/skills/summarize-codebase/scripts/summaries.py plan --json`
   with the requested mode arguments. Print mode, since commit, count, and file list
   before reading. Default: `_META.json` commit, then last commit touching summaries,
   then full scope. Per-file content hashes also catch dirty/untracked changes and
   missing summaries. `--since REF` adds changes from REF to the current working tree.
   Paths are repo-relative and quoted globs; path mode selects matching files regardless
   of Git status. Public-symbol removals also select known import consumers.
2. Read [references/annotation-format.md](references/annotation-format.md). Split planned
   files into directory batches of roughly 10–15; reduce for large files. Delegate up
   to the available agent limit, scheduling remaining batches as slots open. Each
   agent receives the full file list, canonical areas, annotation schema, and quality
   rules below. Each reads **all source** for its files, including chunked reads of
   large modules; writes a separate temporary annotation JSON; returns file count and
   concise feature/data-flow findings. Do not import/run source to inspect it.
3. Merge no source files. Run the helper with identical mode arguments:
   `python .agents/skills/summarize-codebase/scripts/summaries.py build --annotations <batch1.json> <batch2.json>`.
   The helper validates all selected annotations and source hashes before writing.
   It extracts exact Python signatures, attributes, import edges, and source lines;
   renders mirrored `<source.ext>.md`; removes deleted summaries; rebuilds all three
   indexes from **all summaries on disk**; then writes `_META.json` last.
   Review generated summaries for semantic accuracy and useful method purposes.
   Correct annotations and rebuild when necessary. Delete temporary batch JSON after
   successful review; committed Markdown and metadata are the durable artifacts.
4. Run `python .agents/skills/summarize-codebase/scripts/summaries.py check`.
   Exit 0 means full selected coverage is current; exit 1 lists stale/missing/deleted
   files. Targeted runs can legitimately leave unrelated files stale; report them.
   Finish with one navigation example using the [navigate skill](../navigate/SKILL.md).

All modes rebuild indexes, including no-change runs (which print `up to date`).
Content hashes, not the aggregate commit, determine freshness after partial runs.
Hashes normalize CRLF to LF for cross-platform clones. The commit records HEAD at
build time, **not** a claim that every source file matches HEAD. If Git is unavailable,
hash checks still work; explicit `--since` fails instead of silently ignoring the ref.
`CODE_SUMMARIES_GIT` can choose a Git executable. Never alter global Git trust settings.
After changing the helper/template, use `--all` to regenerate existing formatting.
Incoming import/call hints are derived facts: the helper updates them in unchanged
summaries too, without claiming their semantic descriptions were re-reviewed.

## Quality rules

- Specific > generic. "Streams parquet shards windowed by session" beats "loads data".
- Methods: what & why, never line-by-line how.
- Always record shapes/schemas and CLI args/config keys — highest-value info.
- Flag global state, singletons, env reads, hardcoded paths/IDs.
- `Uses` = intra-repo + critical externals only.
- Trivial files: one-liner, skip empty sections.
- Never invent. If unsure write `unclear — see source`.

Import edges prove imports, not runtime calls. Mark unresolved receivers, re-exports,
wildcards, and consumers outside coverage as incomplete. Do not promote guesses to
`calls` or `instantiates` edges. Include precise runtime connections in annotations
only when source supports them. Verify source lines before citing navigation answers.

## Artifacts

- `_INDEX.md`: directory tree, linked one-liners, symbols, package exports, known holes.
- `_FEATURES.md`: product behavior → entry points → data flow → supporting files.
- `_GRAPH.md`: feature-grouped source-backed import edges; dynamic calls remain hints.
- `_META.json`: commit/date/mode/count, per-file hashes/symbols/imports, stale-file list.
- `braindance/<path>.<ext>.md`: rigid sections from the annotation reference.

Check out the repository to use these artifacts. A wheel-only `pip install braindance`
does not include repository instructions or summaries. Agents without skill discovery
can start at `AGENTS.md`; no installation or symlinks are needed in a checkout.
