---
name: final-figure-assembly
description: How proj/predictor/figures/final_figures/ assembles paper figures (fig1-fig6) by importing draw_*(ax, ...) panel functions from existing topic dirs (busy_drift, general_metrics, electrode_discr, cartpole_threeway_selection, shank3_discrim, pan_cohort_dynamics, human_to_mouse) onto one shared composite canvas per figure — as distinct from a single topic dir's own internal pipeline layout. Covers the main-vs-supplemental build-script naming convention, the journal-page-size mechanism and this project's Nature Neuroscience target, panel-mapping bookkeeping, and the cross-topic refactor a panel needs before it can join a composite. Invoke when asked to build, update, or plan a fig1-fig6 composite, work under proj/predictor/figures/final_figures/, or refactor a panel script so it can join one. Companion to figure-pipeline-organization (owns a topic dir's own internal derive_data/plots skeleton — the panels THIS skill assembles should already follow it) and figure-style (owns the Styler API itself, including the journal-profile/composite-canvas mechanism this skill's project-level convention is built on).
argument-hint: [figN name or panel]
---

# final_figures — cross-topic paper-figure assembly

`final_figures/` is a different layer from a topic dir's own pipeline: `figure-pipeline-
organization` covers how ONE topic (`busy_drift/`, `shank3_discrim/`, ...) organizes its
own `derive_data/`+`plots/`; this skill covers how SEVERAL topics' already-built panels get
stitched into one of the paper's six numbered figures. Nothing about a source topic's own
layout changes because of this — `final_figures/` imports in place, it never moves or
duplicates a topic's plotting code.

**`final_figures/README.md` is the numbering authority** — the live table of all six
figures, their status, and which topic dir(s) source each. Read it first; don't re-derive
the fig1–6 mapping from `../FIGURE_PLAN.md`, which predates this reorg and used different
numbers (see that file's own superseded-by banner). Each `figN_*/README.md` carries that
figure's specific panel→source-script mapping table — keep the mapping there, not in this
skill (it changes per figure; this skill only owns the *mechanism* all six share).

## Lock a panel layout before writing any code

The panel→script mapping for a figure starts as a conversation with the PI, not a script.
When the PI hands over a rough layout (a hand-drawn sketch, an Illustrator export with
plots already dropped in, or just a description), the sequence is:

1. **Read the sketch/description. If a panel's content is genuinely ambiguous, say so and
   ask** — don't silently guess a plausible-sounding mapping into a confident-looking
   table. A wrong guess that *looks* confident is worse than a flagged unknown, because
   nobody double-checks a table that already looks done.
2. **Identify a real candidate source script per panel by actually checking the topic
   dir's `plots/`** (open the script, or its README/`derive_data` doc if the filename
   alone is ambiguous) — don't pattern-match a filename to a one-line sketch description
   and call that confirmed. **Also check the topic dir's root for a working-notes/
   decision-backlog doc** (e.g. `figure_notes.md`, `SHORE_UP.md`) before drafting a layout
   — `README.md`/`derive_data` docs describe what a pipeline computes, but a separate notes
   file is often where the PI already made binding calls about panel *placement and
   figure boundaries* (which panels are in/out, whether a story is one figure or two).
   A layout drafted from headline docs alone can look complete while silently
   contradicting a call the PI already made — a dry-run test surfaced exactly this for
   `fig5_cartpole_drift/` (`cartpole_threeway_selection/figure_notes.md`'s D9 already
   decided that figure should split into two, which the scaffold didn't reflect). `ls` the
   topic dir root, not just `plots/` and `derive_data/`, before calling sourcing confirmed.
3. **Write the locked layout at the TOP of that `figN_*/README.md`**, immediately after
   the title and question line, as two short lists — main figure and candidate
   supplemental, same format:

   ```markdown
   ## Panel layout

   **Main figure**:
   - A) <one-line content> — `<topic>/plots/<script>.py`
   - B) <one-line content> — `<topic>/plots/<script>.py`

   **Supplemental (candidate)**:
   - A) <one-line content> — `<topic>/plots/<script>.py`
   ```

   Non-code panels (photos, hand-drawn diagrams) get `(non-code — reserved axes)` in the
   script slot rather than being omitted from the list — see "reserved empty axes" below.
4. **Only after this section exists** (and, while the PI is actively reviewing a figure,
   only after they've confirmed it) does the composite-build step start: the `draw_*`
   refactor per panel (with its before/after render check), then `build_main_figure.py`
   (or a content-named supplemental script) actually assembling and rendering the png/svg
   for Illustrator finishing.

This section is a living plan, not a one-time write — update it in place if a panel's
real source script turns out different once actually opened, rather than leaving a stale
mapping sitting next to a build script that quietly does something else. `fig6_human_asd/
README.md` is the reference example of this layout format once a real sketch has been
worked through.

## The mechanism, in one line

One `Styler.create_composite_figure` canvas + nested `GridSpec`s; each panel is drawn by
calling the SAME `draw_*(ax, ...)` function its topic's own standalone script uses, at
this composite's own gridspec-carved axes — zero duplicated plotting logic, zero risk to
any panel's own canonical standalone render. `busy_drift/build_supp_figure.py`'s module
docstring is the reference implementation; read it in full before writing a new composite
script (it covers the `tight_layout()`-vs-`sharey` gotcha, `label_panel` for A/B/C
letters, `add_group_title` for a sub-group heading, reserved-empty-axes for non-code
panels).

**Refactor gate, before a panel can join any composite**: split its plotting into a
`draw_*(ax, ...)` function taking an externally-created `ax`, separate from data
loading/computation and its own `save_panel` call — see any of the five scripts
`build_supp_figure.py` imports for the pattern. **Verify the panel's own standalone render
is byte-/pixel-identical before and after this refactor** (before/after PNG diff) — this
is the acceptance test, not optional, especially when several panels for one figure are
being refactored in parallel by different agents.

**Non-code panels** (photos, hand-drawn diagrams) get a reserved empty axes — dashed
border, right size/position carved from the same grid — not a gap in the script or an
omission from it. Matches `general_metrics/fig2_main/build_figure.py`'s existing
placeholder-diagram-slot trick.

**Sequence the hardest/most fragile panel first, not as one of several equal parallel
refactors** — if one panel's claim is contested or its own topic dir's memory carries open
retractions (check that topic's README/NUMBERS_OF_RECORD/memory index), settle what it's
actually claiming and verify that claim before it becomes load-bearing in a composite
layout; don't let 4 easy panels' parallelism sweep the 1 hard one along at the same pace.
`fig6_human_asd/README.md`'s panel-C caution is the concrete example.

## Journal / page size: this project targets Nature Neuroscience

Every `final_figures/` composite is built with `Styler(journal="nature_neuro")`, sized via
`styler.composite_canvas_size(columns=2, depth=...)`. This is a `Styler` construction-time
profile, not a number to copy into a script — see `figure-style`'s "Composite pages &
journal page sizes" section for the full mechanism (`journal_specs.py`, `JournalPageSpec`, why it's a profile-as-data
registry and not a `Styler` subclass per journal) and `neural-predictor-figures`'s note
confirming this project's venue decision (2026-08-13). Don't hardcode `518.7`/`700.2` (or
any other mm/pt conversion) in a `final_figures` script.

**What binds** (all enforced by `Styler(journal="nature_neuro")`):

- page **180 mm** wide max, **170 mm** deep, 88 mm single-column — no US Letter pages;
- all text **5–7 pt**; **panel letters 8 pt bold LOWERCASE** — `label_panel(ax, "a",
  fontsize=styler.composite_letter_fontsize)`, not `"A"` at 12;
- text and line art stay vector and editable — never outlined or rasterized;
- `finish_plot()` runs `Styler.check_journal_compliance()` on every save. On a strict profile it **writes the files and then raises** — you can open the offending render, but the script exits non-zero. `'draft'` only warns. ⚠️ It can't check scale
  bars, error bars, editable text or Symbol-font Greek — a clean run is not a certificate.

**Before the first render, set these four — they are what usually costs the iteration:** (a)
`depth` sized to the row count, not `"max"` reflexively (trap 3); (b) each row-gap a comfortable
default via independent spacer rows, not one shared `hspace` (trap 4); (c) `set_anchor("W")` on
every `set_aspect("equal")` panel that sits in a left column, and place all panel letters after a
single `fig.canvas.draw()` (trap 4); (d) each panel's width chosen as a **target page-fraction**
(e.g. "b and c each ~1/3 of the page"), not a guessed ratio — then read the render and nudge.

🚨 **Four traps when moving a page onto this canvas.** (1) `hspace`/`wspace` are *fractions
of row size*, so a smaller canvas shrinks every gap and titles land on the next row's
x-labels — re-tune spacing rather than carrying over a bigger page's values. (2) Never
shrink an oversized page in Illustrator: it scales the text too (6.5 pt × 0.83 = 5.4 pt),
so rebuild at the target size, where 7 pt stays 7 pt.

(3) 🚨 **`depth="max"` is not the default for every composite — pick it deliberately, scaled
to how many rows the composite actually has, especially when saving with
`finish_plot(..., exact_size=True)`.** `exact_size=True` is what every real `build_*_figure.py`
in this repo uses, and it means there is no "crop in Illustrator afterward" step coming —
the rendered canvas ships as-is. A 1-row, 2-panel supplemental built at `depth="max"`
doesn't render with blank space to crop; `fig.subplots_adjust(**margins)` fills the
canvas, so the single row of axes is stretched to fill the full 170 mm page depth, and the
panels come out looking inflated/empty even though nothing is technically out of spec
(confirmed on `fig5_cartpole_state_separation`'s `build_supp_figure_alignment_and_
stimmask_controls.py`: identical 180×170 mm canvas rendered for a 2-panel, 1-row composite
as for a 5-panel, 2-row one). Size `depth` to the row count instead — e.g. roughly
`max_depth * nrows / <rows a full page holds>`, or an explicit smaller pt value — and
reserve `depth="max"` for composites with enough rows to actually use the page. Only rely
on a later manual Illustrator crop if you are not calling `exact_size=True`.

(4) 🚨 **A `set_aspect("equal")` panel does NOT fill its cell — so `width_ratios`/
`height_ratios` stop being its size+position knob.** This is the "why is b overlapping into
row 2 / why is panel a's letter indented" foot-gun. Two modes:

- **`adjustable="box"`** (matplotlib's default): the box shrinks to a square and **centers** in
  the cell — letterboxing, sitting inboard of the left margin, and *detaching the letter*
  (`label_panel`/`_letter_loc` read `get_position()`).
- **`adjustable="datalim"`** (e.g. `fig5` panel b, `02_plot_centroid_projection.py`, where equal
  aspect is load-bearing): the box fills the cell both ways and the *data limits* stretch — so its
  height is set by the **row**, and narrowing/widening its column changes neither its height nor,
  usefully, much else. Its bottom stays at the row edge, abutting the row below.

Fixes (both from this session's `fig5` pass):

- **Crowds the row below** → widen the inter-row **gap**, never resize the panel (you can't
  shorten a `datalim`-equal panel by thinning its column). Use **independent per-gap spacer rows**,
  not one `hspace`: `create_composite_figure(nrows=2N-1, height_ratios=[row,gap,row,gap,row],
  hspace=0)`, each gap tuned alone (one `hspace` is a single fraction for every gap, but each
  title-pad eats it differently, so equal `hspace` reads *unequal*). Comfortable gap default ~0.25;
  to truly shorten such a panel, cut its **row's** `height_ratios`, never its column.
- **Centers horizontally, breaking a left column's alignment** (panels a/c/e not flush) → `ax.set_
  anchor("W")` on each equal-aspect left-column panel so its square hugs the cell's left edge, then
  place **all letters after one `fig.canvas.draw()`** (the aspect box is only final at draw time; a
  letter set before it lands on the pre-anchor center).

Corollary: a uniform grid (`create_figure(nrows, ncols)`, equal cells) still won't *look* uniform
with mixed aspects — an equal-aspect panel letterboxes inside an equal cell, so equal cells ≠
equal-looking panels. Compose ratios deliberately; target page-fractions.

Detection: like the other traps, **invisible to `check_no_fontsize.py` and
`check_journal_compliance()`** — read the PNG for crowded seams and letter alignment.

## 🚨 "There is too much white space" — diagnose it, don't eyeball it

The PI says the page is over-spaced and the panels should be bigger. **They are almost always
right, and you will almost always fail to see it by looking.** Whitespace on a composite is
distributed in small amounts across many seams, so it reads as "a bit loose" while adding up
to a third of the page. ⛔ Do not respond by nudging one `width_ratios` and re-rendering —
that is how a five-round tuning loop starts, and this session burned exactly that.

**Measure first, always. Two commands, before changing anything:**

1. **The area budget.** Monkeypatch `finish_plot` (or add a debug flag) to dump
   `ax.get_position()` for every axes, then sum the panel rectangles:

   ```python
   for ax in fig.axes:
       p = ax.get_position()
       print(f"x {p.x0:.3f}-{p.x1:.3f}  y {p.y0:.3f}-{p.y1:.3f}  aspect {p.width*W_pt/(p.height*H_pt):.2f}")
   ```

   Panels should be **≥ 60-65% of page area**. A real measured page came in at **54.5%** —
   22.6% in the two row gaps, 7.5% left margin, 10% top+bottom, 5.2% column gap. That is the
   number to quote back, and it converts "it looks empty" into a target.
2. **An ink census** of any raster artwork (and of the page), on a coarse grid, to find bands
   that carry no content:

   ```python
   ink = np.array(Image.open(png).convert("L")) < 245
   # per-row / per-column profiles -> contiguous all-empty bands, in DATA coords
   ```

**The four things it will find, in the order they usually bite:**

**(a) Uniform margins and gaps sized for the WORST case, applied to everything.** One
`hspace`, one `wspace`, one `subplots_adjust(left=...)`. The row whose tick labels are
rotated 45° sets the gap for every row; the one panel with a two-line y-label sets the left
margin for the whole page, so panels with no y-label are pushed inboard by space they do not
use — **this is the "why aren't a, c and e flush to the left edge" complaint**, and it is not
fixable with `width_ratios`. Use independent spacer rows (trap 4) and per-row `wspace`
tuples; accept that `subplots_adjust` margins are genuinely page-wide and therefore set by
the hungriest panel, so shortening THAT panel's y-label is what buys the margin back.
⚠️ Spacer-row `height_ratios` entries are in **RATIO units, on the same scale as the rows** —
⛔ not page fractions. A row at ratio 1.00 rendering ~0.23 of the page means a 0.10
page-fraction gap is a ratio of ~0.44. Setting them as fractions collapses every gap to ~3%
and the rows overlap.

**(b) An `imshow` artwork letterboxed and CENTRED in its cell.** `imshow` sets
`aspect='equal', adjustable='box'`, so the axes shrinks to the image's aspect and centres —
leftover height becomes white **above and below the art**, framing it, which is precisely
what makes a reader say "the schematic is small and surrounded by white". ✅ Size the cell to
the artwork's real aspect, and `ax.set_anchor("NW")` so any residual slack falls to one edge
and merges with the row gap instead of boxing the art. ⛔ Never `aspect='auto'` — that
restretches art drawn at a deliberate aspect. (Same root cause as trap 4; this is its raster
form.) ⚠️ Its letter must also be placed after `fig.canvas.draw()`.

**(c) Dead space INSIDE the artwork costs PAGE WIDTH, not just its own area.** A raster panel
is drawn at `width = aspect × row_height`, so a tall empty band at the bottom of the diagram
lowers its aspect and therefore shrinks the width it can occupy. A real case: a 4-row
cell-line legend column that, with the crop floor, was **28% of the artwork's height at 0-3%
ink**. Re-flowing it to one row took the aspect 1.94 → 2.24 and bought the panel real page
width for free. ⛔ Always ink-census the artwork before concluding the composite is at fault.

**(d) A rasterized artwork has a HARD MINIMUM SIZE, and it is usually the binding
constraint.** Its text is baked in at N pt relative to its own design width, so on the page

    effective text pt = design_pt × (drawn width / design width)

A diagram designed at 339 pt with 6 pt type is **4.9 pt at 277 pt wide and 3.9 pt at 219 pt** —
under the 5 pt floor, invisible to `check_journal_compliance()` because it is pixels, not text
artists. ⛔ **Re-rendering the artwork at a smaller design width does NOT fix this**: the text
is absolute points while the boxes are data units, so every label overflows its box (verified).
The only fix is a **re-layout in data units** — fewer units across, or bigger boxes. **Compute
this number before promising a panel can shrink**; it converts "make it a bit smaller" into a
hard yes/no, and it can put a layout the PI asked for in genuine conflict with legibility, in
which case say so with the number rather than shipping illegible art.

**Corollary — when two demands are geometrically incompatible, say so with numbers.** "Panels
b and c must be the same size" plus "the schematic must be bigger" can be mutually exclusive:
if the metric strips sit in the right column of one row and the left column of another, equal
panels force equal COLUMNS, which fixes the artwork at half the page. Show the table of the
trade rather than oscillating between the two.

## Output location — one dir per figure, matching the repo's `figN_*` name

Every `final_figures/` composite render (`build_main_figure.py`, any
`build_supp_figure_<content>.py`) writes to:

```
/Volumes/hunter_ssd/neural_predictor_figures/<figN_dirname>/<output-name>.{png,svg}
```

where `<figN_dirname>` is **exactly** the repo directory name under `final_figures/`
(`fig1_architecture_data`, `fig2_metrics_ablations`, `fig3_electrode_discrimination`,
`fig4_busy_drift`, `fig5_cartpole_drift`, `fig6_human_asd`) — not a topic name, not a
narrative-beat name. This is a **separate output tree from each source topic's own
`<TOPIC>_PUB_DIR`/`FIG_DIR`** (`figure-pipeline-organization`'s convention, still correct
for that topic's own standalone panel renders) — a composite script writes its assembled
page here, in addition to (not instead of) whatever its constituent panels' own standalone
scripts still write to their own topic's `FIG_DIR`. Don't route a composite's output
through `get_output_dir()` or a topic `PUB_DIR` — hardcode this path, the way every
`build_*_figure.py` under `final_figures/` already does.

**No exceptions as of 2026-08-13.** `fig4_busy_drift/build_supp_figure.py` used to be
grandfathered onto `<busy_drift FIG_DIR>/composite/`, from when it still lived in the
topic dir; it moved into `final_figures/` and now writes `busy_drift_supp.{png,svg}`
here with the rest. If you find a `composite/` copy of that page under the topic tree,
it is a stale pre-2026-08-13 render, not a second live output.

## Build-script naming — content discriminator, not arbitrary numbering

Each `figN_*/` dir is flat — no `main/`/`supp/` subfolder split (that bakes an editorial
main-vs-supplement classification into a path, which `figure-pipeline-organization`
already flags as a defect this project has made once and doesn't want to repeat).
Discriminate by filename instead:

- **Main figure**: `build_main_figure.py`.
- **Supplemental(s)**: `build_supp_figure_<content>.py`, where `<content>` names what that
  supplement actually shows (e.g. `build_supp_figure_stimmask_controls.py`,
  `build_supp_figure_bin_axis_robustness.py`) — **not** a positional/arbitrary discriminator
  (`build_supp_figure2.py`, `build_supp_figureb.py`). The name should tell a reader what's
  in the file without opening it, the same reason `drugs/plots/`'s sub-story files
  (`shank3_01_nbqx_age_scatter.py`, not `01b_...`) get a content-based prefix instead of a
  positional one. If a figure ends up with several supplements, this also means their names
  don't depend on the order they were built in.
- **Existing exception, grandfathered**: `busy_drift/build_supp_figure.py` predates this
  convention (it's `fig4`'s sole supplemental so far) and does not need a rename to add a
  discriminator it doesn't yet need. If/when `fig4` gets a second supplemental, reconcile
  naming for both at that point (most likely: give the existing one a retroactive
  content-suffix to match) rather than leaving one bare `build_supp_figure.py` beside a
  content-named second one — but don't preemptively rename it now for a supplement that
  doesn't exist yet.

## See also

- `figure-pipeline-organization` — a topic dir's own internal layout (`derive_data/`,
  `plots/`, `stepN_*.py`/`NN_plot_*.py` naming, `<TOPIC>_PUB_DIR`/`FIG_DIR`). The panels
  this skill assembles should already follow it; this skill doesn't restate it.
- `figure-style` — the `Styler` API itself: composite-canvas mechanics
  (`create_composite_figure`, `label_panel`, `add_group_title`), the journal-profile
  registry (`journal_specs.py`) this skill's Nature Neuro convention is built on, and all
  the per-panel aesthetic rules (titles, color, violins, significance brackets) that still
  apply to every panel pulled into a composite.
- `neural-predictor-figures` — this project's specific file:line pointers and the venue
  decision (Nature Neuroscience) this skill's journal convention cites.
