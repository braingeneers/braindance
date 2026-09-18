---
name: results-numbers-doc
description: How to write and maintain the numbers-LEDGER markdown doc under proj/predictor/figures/ — the short, citable "numbers + their exact source CSV/column + caveats" document that lives in the ANALYSIS repo and is the trustworthy source a results section is later written FROM. This is NOT about manuscript prose or voice (that is `paper-writing-voice`); it is the provenance ledger that feeds it — every number traces to a file+column, superseded numbers are marked not deleted. Covers the three real genres in this repo (per-topic hand-written draft, whole-figure narrative-of-record, fully code-generated numbers-of-record), a canonical section skeleton with a copy-paste template, table/number formatting rules, how to mark superseded numbers, and when a doc must be code-generated. Invoke when asked to assemble/update a PAPER_RESULTS_DRAFT.md / NUMBERS_OF_RECORD.md / *_NARRATIVE.md-style ledger, or "make a table of numbers for the paper." NOT for drafting the actual manuscript .tex prose — that is `paper-writing-voice`. Companion to `figure-pipeline-organization` (owns folder/file layout; this owns the ledger doc's internal structure), `neural-predictor-figures` (owns this project's statistical gotchas this doc's caveats should cite, not restate), and `paper-writing-voice` (the downstream consumer that renders these numbers into manuscript prose).
argument-hint: [topic directory, or path to an existing numbers-ledger doc to extend]
---

# paper-results-draft — the citable results doc

This is about **what goes inside a paper-facing results markdown file and how it's kept
honest over time** — not where it lives in the folder tree (`figure-pipeline-organization`
owns that: it's the `## For the paper` pointer at the bottom of a topic README) and not the
project's specific statistical rules (`neural-predictor-figures` / project memory own those —
cite them, don't re-derive them here).

The existing real examples in this repo (`NUMBERS_OF_RECORD.md`, `FIG2_NARRATIVE.md`,
`shank3_discrim/PAPER_RESULTS_DRAFT.md`) were each written by a different agent session under
different pressure, not designed together as a system. Treat them as raw material, not as
gospel — where one of their conventions is genuinely good, this skill keeps it explicitly;
where it's just accretion, this skill improves on it instead of preserving it out of
inertia. If you find a better pattern while using this skill, update it here rather than
copying whatever the nearest existing doc happens to do.

## Three genres — pick the right one, don't force a topic into the wrong shape

| Genre | Real example | Use when |
|---|---|---|
| **Per-topic draft** (default) | `shank3_discrim/PAPER_RESULTS_DRAFT.md` | One topic dir, 1-4 sibling analyses, numbers already sit in a handful of committed CSVs. **This is what most new topics want.** |
| **Narrative-of-record** | `general_metrics/FIG2_NARRATIVE.md` | A whole *figure* (spanning multiple topic dirs / many panels) needs one prose story building to a single headline sentence, and that story will be quoted/paraphrased directly into paper text. |
| **Generated numbers-of-record** | `general_metrics/NUMBERS_OF_RECORD.md` | The doc aggregates numbers **algorithmically** across many pipelines/axes (sweeps, cross-pipeline joins, anything where hand-transcription has actually caused a wrong number before). |

Don't reach for the generated genre by default — it needs a real generator script
(`gen_numbers_of_record.py`-style) and is worth the setup cost only when the aggregation is
wide enough that hand-transcription risk is real. A 2-analysis topic dir doesn't need it; a
7-axis sweep across 3 pipelines does. Don't reach for narrative-of-record either unless you're
actually writing the story for the abstract/headline figure — most topics just need the
per-topic draft.

The rest of this skill is the **per-topic draft** skeleton in detail, since that's the common
case; the "Escalating" section at the bottom covers when and how to graduate to the other two.

## Canonical skeleton (per-topic draft)

```markdown
# <Topic> — results-section draft

**Status: <DRAFT|FINAL>, <what's populated> as of <date>.** Re-derive with `<script path>`
(<runtime estimate>, <S3/pod needed or not>).

---

## `<sibling-analysis-1>` — <one-line question this analysis answers>

> Full writeup: `<subdir>/README.md`. Numbers below are the citable ones — this block is the
> single source of truth if the two ever drift (re-derive, don't hand-copy).

**<Headline sentence in prose, bold, one or two sentences, states the claim and its
direction.>**

| <metric/level> | <col> | <col> | ... |
|---|---|---|---|
| ...            |       |       |     |

n = <n>, <cohort one-liner>. `<col>` = <definition, including sign convention if a
difference>; `<col>` = <definition>.

**Caveats (see `<subdir>/README.md` for the full versions):** <bullet-dense paragraph or
list — power, known artifacts, what NOT to conclude from this alone>.

---

## `<sibling-analysis-2>` — ...

<same shape>

---

## Combined narrative

<Only write this section once ≥2 sibling analyses actually exist — don't stub it in advance
with a "TODO, other agent fill in" placeholder unless there genuinely is another agent
actively working the other half; a stub that sits unfilled for a long time is worse than no
section. Written from BOTH sides once both exist — do not write it having only read one.>
```

## Rules, in priority order

1. **Every number traces to a specific file + column, not a plot or a memory.** State the
   source inline or in the doc's status line (`re-derive with <script>`). If a number can't be
   traced to a committed CSV or a named generator script, it doesn't go in this doc yet —
   go compute it into a CSV first, then cite it.
2. **Re-derive, don't hand-copy, if numbers drift.** Say this explicitly wherever a number
   might later change (a re-run, a cohort fix, a new seed). This is the single line that
   prevents the doc from becoming a stale copy nobody trusts. All three real examples say some
   version of this; keep saying it.
3. **Caveats get a visually distinct marker — always, never folded silently into results
   prose.** For a short per-topic draft, a bold lead-in phrase is enough:
   `**Caveats (see X/README.md for the full versions):**` followed by a bullet list. For a
   longer multi-section doc (narrative-of-record scale), use the heavier severity-emoji
   convention below instead — don't mix the two conventions within one doc.
4. **Cite this project's existing statistical gotchas by name, don't restate them.** If a
   caveat is really "don't average across organoids" or "state counts not p-values at this n"
   or "don't call this cohort matched," that's a project-wide rule that already lives in
   project memory / `neural-predictor-figures` — link to it (`[[memory-name]]` or a path), one
   sentence of why it applies here, and move on. Don't re-derive the general rule inline every
   time; that's how the same caveat ends up worded three different ways in three docs.
5. **Never silently delete a superseded number — mark it and say why.** When a number in this
   doc turns out wrong (cohort bug, stale cache, a corrected join), don't just overwrite it
   with no trace. Either: (a) keep a short "was X, corrected to Y because Z, see
   `<derive_data>/README.md`" note in the section itself, or (b) for a doc large enough to
   accumulate several of these, a dedicated closing section listing every retracted number and
   where it might still appear uncorrected (`FIG2_NARRATIVE.md`'s `## §9 — Numbers that are
   VOID` is the real example of pattern (b) — reach for it once pattern (a)'s inline notes
   start cluttering the results prose, not before). Either way: **update the memory file that
   originally cited the old number**, not just this doc — a stale number in project memory is
   worse than one in a markdown file nobody re-reads, since memory gets auto-loaded.
6. **The doc is shorter than the topic README, always.** If you're pasting in a paragraph of
   methods detail that already lives in the README, stop — link to it
   (`> Full writeup: <path>/README.md`) instead. The test: could someone write the paper's
   results paragraph from this doc alone, without opening any subdir? If yes, it's doing its
   job; if it's also trying to be the methods appendix, it's grown into the wrong genre.

## Table formatting

- **p-values**: 3 significant figures, scientific notation preserved when small
  (`2.59e-10`, not `0.0000000003` or `<0.001`) — a reader comparing across rows needs the
  actual magnitude, not a threshold bucket.
- **Effect sizes** (Cohen's d, correlations): 2 decimal places. State the sign convention in
  prose immediately below the table (`d = A − B, so negative = A lower`) — never make a reader
  infer which direction is positive from context.
- **n**: always present, either as its own column or in the prose line below the table. Never
  a table with effect sizes and no n visible in the same view.
- **Bold the headline number(s)** inside the table itself (the specific cell the prose claim
  rests on), not just in the surrounding prose — a reader scanning the table should be able to
  find the load-bearing number without re-reading the paragraph.
- Column headers name the actual statistic, not a vague label — `p_perm` /
  `auc_loo_cv` / `cohens_d`, matching the CSV's own column names where practical, so a reader
  can grep the source CSV for the same string.

## Escalating to the other two genres

**→ Narrative-of-record**, when a doc like this needs to become the story for a whole figure:
keep the per-topic drafts as-is (they're still the citable number source), and write the
narrative as prose "beats" that each open with the claim and then cite `<per-topic-draft>` /
`<generated-doc>` by section number rather than restating numbers inline where avoidable.
Close with a void/retracted-numbers table once the story has been revised enough times that
inline corrections would clutter it.

**→ Generated numbers-of-record**, when the aggregation crosses enough pipelines that
hand-transcription is a real risk: write a generator script that reads the same committed CSVs
this skeleton would have hand-cited, and emits the markdown with an unmissable header —
```markdown
<!-- GENERATED — do not hand-edit. Regenerate with `<path/to/generator.py>`. -->
- source: <csv/pkl paths>, md5 <hash>
- git: <short sha>
- generated: <ISO8601 UTC>
```
Every warning/caveat sentence in a generated doc should also be an f-string in the generator,
not hand-typed into the output — otherwise the next regeneration silently drops it. Once a
doc is generated, it stays generated; don't hand-patch a generated file even for a "quick fix"
— fix the generator.
