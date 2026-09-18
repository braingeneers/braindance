---
name: navigate
description: Use BrainDance summaries before source exploration for "how does X work", "where is X", "what calls X", or "which files implement X" under braindance/. Skip for exact source paths or documented coverage holes.
---

# Navigate BrainDance cheaply

Use `.code-summaries/` from the repository root. No vendor-specific tools required.
Target: 2–4 small bounded reads. Stop once answered. More than six files or a full
summary read usually means the query needs narrowing.

## Check freshness once per session

Run `python .agents/skills/summarize-codebase/scripts/summaries.py check --json`.
This reads `_META.json` and hashes scoped source without importing it. It detects
committed, staged, unstaged, untracked, deleted, and missing-summary changes.
Hashes normalize line endings. The aggregate commit alone is insufficient after
partial refreshes. If the relevant file is stale, say
`summaries stale for X — refresh with summarize-codebase`, then inspect that source.
For broad changes, refresh before trusting the indexes. Missing artifacts → read
targeted source or follow [summarize-codebase](../summarize-codebase/SKILL.md).

## Walk

1. Grep **one** index, with line numbers and content output. `_FEATURES.md` answers
   behavior; `_INDEX.md` finds locations/symbols; `_GRAPH.md` finds dependencies.
2. Read only the matching section, typically 15–40 lines. Never dump a whole index.
3. Read the first ~40 lines of one linked summary for overview/connections, or grep
   its symbol then read 10–30 nearby lines. Long files need smaller sections.
4. For consumers, grep `Uses` or graph edges for the qualified module/symbol. These
   are import hints. Runtime dispatch and external consumers require source checks.
   If the summary already names relevant call sites, verify those directly instead
   of reading another index. This keeps combined definition/caller queries to four files.
5. Open the targeted source only when necessary for implementation, stale facts,
   uncertain relationships, or verifying citations. Use the summary's `Path` and
   `Source` locations; verify them before citing `source/path:line` in the answer.

Query layers, coarse first: `^## ` sections; `^### ` symbols; `^> ` purposes;
`^- ` edges; labelled `**Path:**`, `**Feature Area:**`, `**Uses:**`, `**Used by:**`,
`**Called by:**`; `\|` table rows. Use `rg --hidden` for this hidden directory.
If ripgrep is absent, use `grep` or PowerShell `Select-String` with equivalent scope.

```sh
rg --hidden -n -i 'replay|checkpoint' .code-summaries/_FEATURES.md
rg --hidden -n 'calculate_latencies' .code-summaries/_INDEX.md
rg --hidden -n 'data_loading.recording' .code-summaries/_GRAPH.md
rg --hidden -n '^### .*Recording' .code-summaries/braindance/utils/data_manager/utils/data_loading/recording.py.md
rg --hidden -n -F '**Uses:** `Recording`' .code-summaries/braindance
rg --hidden -n '^## (Config / CLI|Data Shapes)' .code-summaries/braindance/cli/replay.py.md
```

Bounded read equivalents: `sed -n '20,50p' FILE` on POSIX;
`Get-Content FILE | Select-Object -Skip 19 -First 31` in PowerShell;
or a tool's offset/limit. No unbounded reads before source fallback.

## Worked patterns

- **Where is latency calculation defined?** Search `calculate_latencies` in `_INDEX`;
  read its row; inspect function heading in linked summary; verify its source location.
- **How does replay work?** Search `Replay` in `_FEATURES`; read feature flow and entry
  point; inspect replay summary overview and config. Source only for exact semantics.
- **What calls Recording?** Search `Recording` in `_GRAPH`/`Uses` lines; distinguish
  import consumers from constructors. Inspect narrow call sites before claiming callers.
- **Which env var sets outputs?** Search `Configuration` in `_FEATURES`; open
  `config.py.md`'s `Config / CLI` section; verify named variable/default in source.
- **User names an exact file / asks about proj/**: skip this index; inspect requested
  path directly. Covered symbol absent in source means stale; report and fall back.
