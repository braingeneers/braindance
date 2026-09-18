---
name: figure-style
description: House style for matplotlib figures using the BrainDance package's canonical Styler class (braindance.utils.data_manager.Styler) — how to use it correctly and avoid the "generic AI-generated matplotlib" look (default gridlines, claim-as-title headlines, faint/washed-out violins, mixed marker shapes, clipped data). Use whenever creating a new plot script, reviewing/critiquing an existing figure's look, or asked to "clean up" or "make publication-ready" a panel, in ANY BrainDance project. Project-specific file pointers and case studies live in companion skills — see below.
argument-hint: [plot script path or figure topic]
---

# Figure style — work with `Styler`, don't fight it or fork it

`Styler` is package-level infrastructure (`braindance/utils/data_manager/utils/plotting/plot_styler.py`),
not scoped to any one project — this skill applies wherever a BrainDance project makes a
matplotlib figure. Project-specific detail (which files/scripts use it, project case studies,
project semantic palettes) belongs in a companion skill, not here.

## Project-specific companions

🚨 **Assembling a multi-panel PAPER PAGE (`final_figures/figN_*/build_*_figure.py`)? LOAD
`final-figure-assembly` BEFORE YOU TOUCH THE LAYOUT — it is not optional and not a
"see also".** This skill owns per-panel style; that one owns the page: the four
composite traps (equal-aspect panels not filling their cell, letters detaching, `depth`
sizing, per-gap spacer rows) and the "there is too much white space" diagnostic. Every
one of those is INVISIBLE to `check_no_fontsize.py` and `check_journal_compliance()`, so
nothing will tell you that you skipped it — a 2026-09-03 session tuned a page through
five rounds of `width_ratios` guesses because that skill had not been loaded, and every
answer it needed was already written down there.


- Working in BrainDance's `proj/predictor/figures/` (the "neural_predictor" project)? Also load
  `neural-predictor-figures` for its file:line pointers and case studies. Load both.
- Other projects (`proj/busy_bee_analysis`, `proj/figs`, ...) have their own, more-widely-used
  `Styler` forks (`cell_styler.py`, `styler.py`) that predate the canonical one — this skill's
  rules still apply conceptually, but check which styling class that project actually uses before
  assuming it's this one.

## The one rule that matters most

**Everything routes through `braindance.utils.data_manager.Styler`
(`braindance/utils/data_manager/utils/plotting/plot_styler.py`).** Never write a parallel
rcParams block, a new `.mplstyle` file, or a bespoke color dict that isn't registered through it.
If `Styler` is missing a capability you need (see "Gaps worth extending" below), extend the class
itself — don't route around it in a script.

🚨 **This applies per-call, not just per-script — an explicit `fontsize=` on a single
`set_title`/`set_xlabel`/`legend`/`text` call is the same violation**, and it is the one
agents actually commit. It bites because of how composites work: a panel's `draw_*(ax, ...)`
is called *unchanged* on the paper page, so a `fontsize=9` written for a comfortable
standalone render arrives on the composite as a literal 9 pt and breaks the 7 pt ceiling.
**The journal profile sets every text size by role**, so a title, label, tick or legend
needs no `fontsize=` at all — see "Composite pages & journal page sizes" below for the
table. A sweep in 2026-08 found 210 out-of-spec literals across 68 live files from exactly
this habit. Full statement of the rule in the Don'ts list below.

⚠️ **Historical note that explains why the habit formed, and why "just delete them" is only
half a fix.** `set_style()` used to hardcode *every* role to 7 pt — the top of the 5–7 pt
band. Legal, but degenerate: with no legal *downward* move, a script had no way to make a
legend key quieter than an axis label except writing `fontsize=5.5` itself, and ~26 scripts
in one topic did. **Those overrides were the missing house style, not a deviation from it**
— deleting them dropped legend boxes onto the data in four panels. The role table replaced
them centrally (2026-08-13). If you ever face this again: fix the defaults FIRST, then
strip, then render-verify once.

**It is now enforced twice, because the two checks catch different failures**:
1. `finish_plot()` runs `Styler.check_journal_compliance()` on every save. On a strict
   profile (`nature_neuro`/`nature`) it **writes the files and then RAISES** on text outside
   5–7 pt or an oversized canvas; `"draft"` only warns. A per-topic `save_panel()` that
   calls `plt.savefig` directly bypasses this and must call the check itself
   (`elec_helper.save_panel` does).
2. `python proj/predictor/figures/check_no_fontsize.py` fails on any literal `fontsize=` in
   a live `plots/`/`final_figures/` script. 🚨 **This is the one that enforces the actual
   rule** — the runtime check measures *sizes* and is blind to an in-spec override, so
   `fontsize=6` sails past it forever.

🚨 Neither check can see a **collision**. They enumerate text artists and measure point
size; a legend box sitting on a data point is invisible to both, and exactly that shipped
in four panels under a silent check. **Read the rendered PNG.**

Know that legacy forks of this idea exist elsewhere in the repo (`proj/busy_bee_analysis/utils/cell_styler.py`,
`proj/figs/styler.py`, plus a dead `proj/predictor/nrp/scripts/utils/ephys_manager/utils/plotting/styler.py`)
— each predates or diverged from the canonical class. Don't import one of these into a project
that already uses the canonical `Styler`; check which one a project actually uses before assuming.

## The ordering foot-gun

`Styler()` mutates **global** rcParams as a side effect of construction. Anything plotted with a
bare `plt.subplots()` *before* `get_styler()`/`Styler()` runs in that process is unstyled — wrong
font, wrong color cycle, wrong spines/ticks. Call `get_styler()` at the top of the script, before
any `plt.subplots()` call, every time — not lazily inside a helper function that might not run
first.

## Font

`Styler` sets `font.family: 'Arial'`, but on machines where Arial isn't resolving the render
silently falls back to DejaVu Sans. This is an environment/font-resolution issue, not a bug in
`Styler` — don't "fix" it by hardcoding a different font family unless a project explicitly asks;
confirm with whoever owns the figure before changing it.

## Reality check on the Illustrator-finished target

`Styler`'s SVG/PDF settings (`svg.fonttype: none`, `pdf.fonttype: 42`) exist specifically so
exported text stays editable — the intended pipeline is matplotlib → clean vector skeleton →
manual Illustrator finishing for hero figures. Don't chase pixel-identical Illustrator polish in
matplotlib itself. Do make the skeleton clean enough that (a) supplementary panels that never get
an Illustrator pass still look professional on their own, and (b) an Illustrator pass starts from
something good instead of fighting bad defaults. (Concrete example of this pattern in practice:
`neural-predictor-figures`.)

## Composite (multi-panel) pages & journal page sizes

A **composite figure** — several panels laid out on one page for a paper (main or
supplemental) — is a different thing from a single `create_figure()` panel, and gets its
own canvas call: `styler.create_composite_figure(width_pt, height_pt, nrows=..., ...)`
returns `(fig, outer_gridspec)`; carve cells with `outer[i, j].subgridspec(...)` and draw
onto them with each source panel's own `draw_*(ax, ...)` function — see
`proj/predictor/figures/busy_drift/build_supp_figure.py`'s module docstring for the full
reference pattern (why one shared canvas, the panel→script→draw-function map, the
`tight_layout()` gotcha with `sharey` axes, `label_panel` for A/B/C letters,
`add_group_title` for a sub-group's own heading). `figure-pipeline-organization` covers
the file/folder side of this (where `build_figure.py`/`build_supp_figure.py` live); this
section covers sizing the canvas itself.

**Journal page sizes are a `Styler` construction-time profile, not a number you copy into
each script.** `Styler(journal="nature_neuro")` resolves a registered `JournalPageSpec`
(single/double-column width, max page depth, dpi floors —
`braindance/utils/data_manager/utils/plotting/journal_specs.py`) onto `self.journal`.
`styler.composite_canvas_size(columns=1|2, depth="max"|<pt>)` then does the width/height
pt lookup for you — hand the result straight to `create_composite_figure`:

```python
styler = get_styler(journal="nature_neuro")  # or Styler(journal="nature_neuro") directly
fig, outer = styler.create_composite_figure(*styler.composite_canvas_size(columns=2), nrows=3, ...)
```

🚨 **`journal` is REQUIRED — `Styler()` with no argument is a `TypeError`** (2026-08-13).
It used to be optional, and 165 of the repo's 172 call sites passed nothing, so the house
style actually in force was written down nowhere. Registered profiles: `"nature_neuro"`,
`"nature"`, `"draft"`.

- **`"nature_neuro"`** for anything feeding a paper panel — **standalone panels included**.
  (The old "a panel is not a page, so pass `None`" reasoning was wrong twice: the geometry
  check only ever flags an *oversize* canvas, so it passes trivially below page size; and
  the profile is what sets the TEXT SIZES, so a `None` default silently gave a standalone
  panel different typography from the composite calling the same `draw_*`.)
- **`"draft"`** for genuinely non-paper output — scratch analysis, QC, the `braindance`
  library's own plotting helpers, which have callers outside this repo. It warns instead of
  raising, has no page (`composite_canvas_size()` raises on it), and uses larger on-screen
  sizes. ⛔ It is not "no journal" wearing a name; it exists so the choice is always written
  down. ⛔ Don't unify its sizes with Nature's — 6 pt is sized for a 180 mm printed page.

Per-project convention: pick `depth` deliberately, scaled to how many rows the composite
actually has — see `final-figure-assembly`'s "Three traps when moving a page onto this
canvas" for why `depth="max"` on a 1-2 row composite saved with `exact_size=True` (the
norm for every real `build_*_figure.py` here) renders stretched/inflated panels rather
than a page with blank space to crop later. Reserve `depth="max"` for composites with
enough rows to actually use the page.

**A journal profile carries text rules, not just a page size.** It sets `min_text_pt`/
`max_text_pt` (5–7 pt), `panel_letter_pt`/`panel_letter_case` (**8 pt bold, lowercase
`a, b, c`** for Nature), and — pushed into `rcParams` at construction — the per-role sizes:

| role | nature_neuro | draft | rcParam |
|---|---|---|---|
| axes title | 7 | 10 | `axes.titlesize` |
| axis label | 7 | 9 | `axes.labelsize` |
| tick label | 6 | 8 | `x/ytick.labelsize` |
| **legend** | **5** | 8 | `legend.fontsize` |
| annotation (`ax.text`, stars, `n=`) | 6 | 9 | `font.size` |
| panel letter | 8 | 12 | `composite_letter_fontsize` |

✅ **The only sanctioned way to change a size** is at construction:
`Styler(journal="nature_neuro", tick_pt=5)` — validated against the profile's band, visible
and greppable in one place. ⛔ Never a `fontsize=` on an artist. 🚨 `axes.titlesize` is set
explicitly rather than left at matplotlib's `'large'`, which is 1.2 × `font.size` and would
float above the ceiling as the base size changes. Don't restore a relative title size.

**Adding a new journal**: add one `JournalPageSpec` entry to `JOURNAL_SPECS` in
`journal_specs.py` — this is data, not behavior, so it's never a reason to subclass
`Styler`. **Only add a profile once you have the publisher's actual verified column-width
+ max-depth spec.** Do not infer one from a project's own working-canvas habit (e.g. a
612×792 pt page someone chose to drop Illustrator panels into by hand) — that's a
convenience size, not a publisher mandate, and canonizing a guess into shared package code
is worse than leaving that journal out of the registry until the real numbers are known.

## Do

- **Sizing**: use `styler.create_figure(width_pt=..., height_pt=..., size_preset=...)` (points —
  1 pt = 1/72 inch, **not** DPI, which is print resolution and unrelated to figure size) so panels
  land on consistent journal-column widths. Bare `plt.subplots(figsize=(a, b))` inherits an
  un-overridden default figure size — that's how figure sizes drift per-script across a project.
- **Don't combine `create_figure(width_pt=..., height_pt=...)` sizing with
  `ax.set_aspect("equal", adjustable="box")` unless the data genuinely needs 1:1 physical/spatial
  scaling** (a literal spatial/anatomical map, an electrode layout — cases where a unit of x must
  equal a unit of y on the page). It's tempting to reach for `set_aspect("equal")` on any y=x
  parity/reference-line scatter (e.g. a masked-vs-unmasked decode comparison) just to make it
  *look* square, but that's the wrong tool: `adjustable="box"` shrinks the axes box to fit
  whichever of width/height is smaller, and once one dimension is pinned by the other, the default
  `bbox_inches="tight"` save (see next bullet) crops away the excess — so you can keep editing the
  now-irrelevant dimension and the saved PNG/SVG won't change at all. Confirmed empirically on
  `electrode_discr/plots/19_plot_mask_retention.py`: with `height=250`, `width=250`, `300`, and
  `350` all rendered to the identical 973×1016 px PNG. Left at matplotlib's default
  (`aspect='auto'`, i.e. just don't call `set_aspect` at all), the axes box stretches to fill
  exactly the rectangle `width_pt`/`height_pt` requested, so whichever ratio you pick is what
  renders — no fighting between the sizing call and the aspect call. (Fixed on that panel by
  dropping the `set_aspect("equal", ...)` call entirely.) The same aspect-vs-cell fight has a
  **composite** consequence when the panel keeps `set_aspect("equal")` for a genuine reason and
  is dropped onto a paper page: it won't shrink to fit its gridspec cell, so it crowds the row
  below and can't be shortened by narrowing its `width_ratios` share — see `final-figure-assembly`'s
  trap (4) "A `set_aspect('equal')` panel does NOT fill its cell" for the gap-not-panel fix and the
  `set_anchor("W")` + letters-after-`draw()` trick for left-justifying such panels in a column.
- **Same `width_pt`/`height_pt` across two scripts does not guarantee the same final pixel
  dimensions, by default.** `finish_plot()` saves with `bbox_inches='tight'`, which crops to each
  panel's own rendered content — so two panels requesting the identical canvas can still land at
  different sizes if one has a longer title, a two-line wrapped axis label, or an annotation that
  overflows the axes box (`clip_on=False` text sitting at the plot edge, common for reference-line
  labels). Confirmed on `electrode_discr` panels 16 vs 19 at identical `width_pt=200,
  height_pt=250`: 16 → 825×963 px, 19 → 856×1016 px, purely from label/title footprint differences.
  For panels meant to sit side-by-side in a composed figure (Illustrator, a multi-panel paper
  figure), pass `finish_plot(..., exact_size=True)` instead — it saves at exactly the
  `width_pt`/`height_pt` canvas (via `tight_layout()` + an explicit full-figure `Bbox`), so the
  same declared size always produces the same pixel size regardless of content. Content that
  doesn't fit the requested canvas gets **clipped**, not auto-expanded (this bit a title on panel
  19 — "Electrode decode: full vs. identity-masked model" ran off the right edge at `width=200`;
  fixed by shortening the title, not by widening the panel, since the panel had to stay 16's size).
  When you touch this, know the underlying trap: matplotlib's `bbox_inches=None` does **not** mean
  "no tight crop" — it means "use `rcParams['savefig.bbox']`", and `Styler.configure_svg_settings()`
  sets that rcParam to `'tight'` globally, so a naive `bbox_inches=None` silently still tight-crops.
  `exact_size=True` works around this by passing an explicit `Bbox` spanning the whole figure
  instead of relying on `None`.
- **Declare `width`/`height` as `main()` default parameters**, per the root `CLAUDE.md` convention
  of putting configuration as `main()` defaults rather than top-level constants (or, worse, a bare
  tuple literal buried inside `create_figure(width_pt=..., height_pt=...)` / `figsize=(...)`). The
  point is discoverability: to resize a panel, you should be able to open its script and see every
  panel's size sitting right in `main()`'s signature — not hunt through a tuple embedded in a
  function-call argument list several stack frames down, and not have to know which nested helper
  a given panel's numbers live in.
  - **One panel per script**: `width`/`height` are just two of `main()`'s parameters, threaded
    straight to that one `create_figure` call:
    ```python
    def main(width=250, height=250):
        styler = get_styler()
        fig, ax = styler.create_figure(width_pt=width, height_pt=height)
        ...
    ```
  - **Several independently-sized panels** (a `NN_plot_*.py` rendering multiple named sub-panels —
    see `figure-pipeline-organization`'s note on multi-panel files): every panel gets its own
    `width`/`height` pair in `main()`'s signature — not a single constant forced onto every panel —
    each tagged with an inline comment naming the panel(s) it controls (by output/save name, so the
    comment is greppable against what actually lands on disk), then passed down to the
    panel-producing function as parameters (with defaults matching what they replace, so the
    function stays independently callable/testable):
    ```python
    def main(
        rescue_width=317,     # p00_rescue_{bits,corr,resid}
        rescue_height=360,
        recovery_width=250,   # p01_recovery_bits, p02_recovery_resid
        recovery_height=250,
        benefit_width=360,    # p03_benefit_gradient
        benefit_height=302,
    ):
        ...
        _rescue_panels(d, width=rescue_width, height=rescue_height)
    ```
    Real example: `general_metrics/fine_tuning/plots/01_main_panels.py`. Once `width`/`height` are
    parameters, don't also re-declare a local `width = ...` inside the panel function body — the
    parameter *is* the declaration; just pass it straight to `create_figure`.
  - If a shared helper only accepts an inches-based `figsize=` kwarg (e.g. a cross-pipeline
    card-plotting helper), still take `width`/`height` in points as the calling function's own
    parameters and convert only at the call site (`figsize=(width / 72, height / 72)`) — don't
    convert to inches earlier just to dodge the division; points stay the unit a reader reasons in.
  - Older scripts (e.g. `human_to_mouse/plots/01_plot_pseudotime_scatter.py`) still declare
    `width`/`height` as a local variable inside `main()` rather than a parameter — that predates
    this convention. Don't treat it as the pattern to copy; convert opportunistically when you're
    already touching a script's sizing, not as a drive-by change.
- **Color**: register every category through `styler.register_color_map(name, {...})` and read it
  back with `get_category_color`. Don't write a new module-level hex dict — that's how a "shared"
  palette silently drifts out of sync across files. Pull a base palette from `styler.colors` /
  `styler.named_colors` rather than retyping hex.
- **Never reuse a color for two different meanings, within a figure or across the figures a reader
  will see together.** A color is a semantic label — once a hex means "SHANK3" (or "ventral," or
  "cartpole condition"), it can't also mean a brain region, a different genotype, or "this is the
  group median" in another panel of the same story, even if the two meanings never appear in the
  same axes. Before picking a color for a new category, check what that hex/`register_color_map`
  entry already means elsewhere in the project (companion skills like `neural-predictor-figures`
  document standing project-level semantic palettes — check there first) rather than picking the
  next unused-looking color and hoping it doesn't collide. The one sanctioned exception: reusing a
  color for the *same* category shown a second way (e.g. genotype's color also marking that
  genotype's median line) — same referent, still one meaning.
- **Titles**: sentence case, always. Short **bold** descriptive title (what's plotted — "bits/spike,"
  not a finding) + an optional smaller, regular-weight subtitle line only if truly needed. Bold is
  `Styler`'s enforced default (`axes.titleweight: 'bold'` in `set_style()`, `plot_styler.py`) — call
  `ax.set_title(...)` plain, don't pass `fontweight="bold"` yourself; a script that hand-sets
  `fontweight` is either fighting the default for no reason or (more commonly, when it omits
  `fontweight` entirely expecting *regular* weight) about to get overridden by it. This was added
  2026-08 after `electrode_discr` panels 16-19 silently drifted apart — some scripts remembered to
  pass `fontweight="bold"`, some didn't — precisely the incoherence a shared default exists to
  prevent; don't re-litigate it per-script. Never ALL-CAPS a word for emphasis — use bold/italic
  elsewhere in the text. Never let the title carry the figure's conclusion, a sample size, or a
  provenance tag (`n=42`, a model slot name) — that belongs in the external caption or stdout
  (`print(...)`), not baked into the image; if you're removing one, check the number is still
  surfaced somewhere before deleting it, not just off the panel. This recurs even on panels that
  otherwise follow house style — `cartpole_threeway_selection` panels 03/06 both titled
  `f"Location: centroid shift  (n={len(piv)} organoids)"` — so treat `n=` (or any other stat) in an
  `f-string` passed to `set_title` as a hard stop, not just a style nit; the fix is a bare string
  literal (`"Location: centroid shift"`), with the count staying in the `print_ratio_panel_stats`
  stdout block that already existed. A title that
  reads like a sentence-length claim ("X because Y, not Z") is the single most common way an
  otherwise well-styled figure still looks unpublishable. Titles also accrete cruft across edits —
  a qualifier like "by split" or "novel tissue" that duplicates what the x-tick labels or axis
  already say tends to survive from an earlier draft. When touching a title, drop anything the
  rest of the panel already states, and keep it short — a long title is the first thing that
  clips off-canvas if the panel is later saved with `exact_size=True` (see Sizing, below).
- **⛔No underscores in anything the reader sees.** User preference, stated explicitly
  (2026-07-31): `snake_case` is a code identifier, not English, and it reads as unfinished in a
  figure. This covers every string that lands in the image — tick labels, axis labels, titles,
  legend entries, annotations — and it applies to *rendered* text, not to the variables behind
  it: keep the column named `pop_bits_per_spike`, but label the axis "bits / spike". Write the
  human phrase (`d_model` → "model width", `eff_dim` → "effective dimensionality",
  `auc_lin_matched` → "AUC, rate-matched", `organoid_id` → "organoid", `resid_corr` →
  "residual correlation", `p_perm` → "permutation p"). Two things make this recur, so check
  both: (a) a bare column name passed straight through — `ax.set_xlabel(metric)`,
  `set_yticklabels(df["organoid_id"])`, `sns.*plot(x="abs_d")` letting seaborn auto-label from
  the column; (b) a display dict that keeps the code vocabulary because it started as a copy of
  the keys. Keep a `key → display` mapping next to the config it labels and put the English in
  there once. (Genuine math subscripts in mathtext — `$\lambda_i$`, `gain$_i$` — are not this
  and are fine.)
- **Label the thing the axis measures, not the variant that produced it.** On a *paired-delta*
  panel (y = `with − without`), a column labelled with the ablated variant's name ("source
  embedding off") inverts how a positive value reads — the reader sees "off" over a bar pointing
  up and concludes turning it off helped, when positive means the opposite. Name the feature
  ("source embedding") and say `(with − without)` on the value axis. The same panel family
  drawn as *absolute* values per model (each column IS a variant, beside a baseline column)
  should keep the variant names. Two vocabularies, one per question — don't force one shared
  label dict across both. Where a column's sign convention is inverted relative to its
  neighbours (an *added* feature among *removed* ones), flag it in the tick label itself, e.g.
  "input dropout 0.15 (added)".
- **Violins**: use `seaborn.violinplot` (not matplotlib's raw `ax.violinplot`) for the KDE
  rendering. Aim for a bold, confident violin — solid saturated fill (alpha ~0.5–0.6, not a faint
  tint), a defined dark edge line, `cut=0` so the shape doesn't extend past the data range. Two
  regimes:
  - **Population comparison, no pairing to preserve** (no per-subject line connecting points
    across columns): the violin IS the primary encoding. Prefer the built-in `inner="box"` (thick
    IQR bar + white median dot) over hand-rolling a separate median/CI element on top — that's
    what the box-inner is for. No scatter overlay needed; a swarm on a real box-inner violin is
    usually redundant.
  - **Paired / small-n data where individual points carry meaning** (per-subject connector lines
    between columns, n low enough that any single point matters): keep points + median-line +
    CI-whisker + connector-lines as the evidentiary layer, and add the bold violin as a backdrop
    underneath (low zorder). Don't drop the points for a cleaner pure-violin look — the pairing is
    usually the actual evidence, not decoration. If n is small (rule of thumb: under ~20), caption
    the violin as informal/KDE-only — read the points, not the outline.
  - If a panel uses custom, non-uniform x positions (grouped columns with gaps between blocks),
    seaborn's default categorical placement won't match them — draw at the default position and
    translate the resulting collection's path vertices to the real x, rather than fighting the
    axis.
- **Significance brackets**: use `styler.add_significance_bracket(ax, x1, x2, y, label)` — caliper
  shape (short vertical end-ticks, horizontal bar, label centered above), `***`/`**`/`*`/`n.s.` as
  the label convention. Stack multiple comparisons on one axes by calling it again with increasing
  `y`.
  - **Color is fixed by the default, never a per-call kwarg.** `add_significance_bracket`'s
    default is `color='black'` — it's a structural line element, so it stays black.
    `add_significance_label`'s default is `color='0.3'`, a dark gray — it sits closer to the
    data than a bracket does, and a slightly softer gray reads better than pure black at 5-7
    pt. **Both defaults ARE the house style**, chosen deliberately, not placeholders. Never
    pass an explicit `color=` to either call — not even to make a significance mark match a
    category/genotype/region color scheme the same panel is already using elsewhere. This is
    the same violation as an explicit `fontsize=` override (see **Don't override a value
    `Styler` already governs on a per-call basis**, below) — a value `Styler` governs by
    default, overridden per-call instead of per-class. If a panel genuinely needs a different
    significance-mark color for a new reason, that's a `Styler`/`journal_specs.py`
    construction-time option, not a per-call kwarg.
  - **When a panel compares ≥3 groups pairwise, bracket every pairwise contrast, not just the
    ones touching one "primary" group.** A design that only tests primary-vs-each-other (skipping
    the one contrast between the two non-primary groups) silently omits a comparison the panel is
    visually inviting the reader to make just by putting all 3 groups on the same axis — and that
    omitted contrast can itself be the interesting result (`cartpole_threeway_selection` panel 06:
    the un-tested third pair, spont-vs-causal vs. causal-vs-cartpole, came back significant, `**`,
    once added). For 3 groups that's `C(3,2)=3` brackets total; stack the narrowest-span contrasts
    lowest and the full-span one highest so they nest visually instead of crossing. If only some of
    the contrasts have a pre-registered direction (e.g. "is X the largest?"), keep those one-sided
    and run the rest two-sided — don't silently invent a direction for a comparison that was never
    pre-specified. When you add a bracket row, re-check any single-entry legend (e.g. a null-line
    key) placed near a plot corner — it can end up sitting directly on top of the new bracket and
    needs to move (`lower center`, anchored on the reference line itself, is usually clear).
  - **Single-cell significance (no bracket)**: when a panel tests each column/cell against an
    implicit reference (zero, a reference-model row) rather than against another visible column,
    there's nothing to draw a caliper between — use `styler.add_significance_label(ax, x, label)`
    instead (added 2026-08-12). Defaults to the active profile's annotation size, positioned via a blended transform
    (x in data coords, y in axes-fraction) so the label sits at a consistent height across cells
    regardless of each cell's own y-range. This replaces the ad hoc `ax.text(x, 0.985, label,
    fontsize=6, color='black')` pattern that had drifted across several `fig2_main/plots/` forest
    panels (`01_plot_axis_ablation_forest.py`, `04_plot_forest_matrix.py` still use the old inline
    form; new panels should call the Styler method instead of copying theirs).
- **One chart per saved PNG/SVG, by default.** User preference (stated explicitly 2026-07-28,
  `shank3_discrim/predictability_discrim`): don't combine several distinct charts into one
  multi-panel gridspec composite (`fig.add_gridspec(2, 3, ...)` + `label_panel(ax, "A")`/`"B"`)
  unless the panels genuinely have to be read together as one evidentiary unit (e.g. a scatter
  and its marginal histograms, or two axes sharing one x-axis that would be meaningless split
  apart). The default is: one `styler.create_figure()` + one `save_panel()` call per chart, even
  when a script previously produced "Panel A: 3 metric subplots, Panel B: summary bar chart" as
  a single image — split each of those into its own file instead
  (`01_bits_zeroshot_vs_ft.png`, `01_corr_zeroshot_vs_ft.png`, `01_cohens_d_summary.png`, ...).
  A single `NN_plot_*.py` script producing several separately-saved named outputs (not one
  shared canvas) is still the normal, encouraged pattern — see `figure-pipeline-organization`'s
  note on `01_main_panels.py` rendering "several named sub-panels." Don't call
  `styler.label_panel(ax, "A")` on a lone-chart file — that convention is only for the rare case
  where multiple panels genuinely share one canvas.
- **Grouped bar + error + value label**: for win-rate/proportion-style panels, the clean
  convention is: bars grouped and colored by category, error bars (Wilson or bootstrap CI), the
  numeric value printed just above each bar, a significance annotation below the value, and a
  clean legend below or beside the panel rather than floating inside the plot area.
- **Legends**: `Styler` sets a boxed legend as the house default (`legend.frameon: True`,
  `legend.fancybox: False` for square corners, `legend.edgecolor: '#231f20'`,
  `legend.facecolor: 'none'` so the box never occludes data underneath it, `legend.framealpha: 1`)
  — leave it, call `ax.legend()` plain. Reproduces the look of a hand-drawn Illustrator legend box
  (transparent fill, thin near-black square-cornered outline) without chasing an exact SVG match.
  Flipped from frameless 2026-08 — if you're reading an old panel/PR from before that date with a
  floating unboxed legend, that's not a regression to "fix" back to frameless. If a legend renders
  with an unexpected fill or rounded corners, something explicitly passed `facecolor=`/`fancybox=`;
  remove that override rather than patching around it. Prefer inline text labels next to the last
  data point/bar over a legend entirely when there are ≤3 categories AND the label can sit right
  next to what it names. For a shared reference/baseline line that isn't anchored to one specific
  point (an EWMA control, a null value), a small legend entry (line swatch + label) reads cleaner
  than italic inline text floated next to the line, especially once the panel already has other
  inline annotations competing for space.
- **Markers: circles/dots only, never mix marker shapes.** Every scatter/strip/swarm point in a
  panel (and across panels of the same figure family) uses the same round marker — don't reach for
  `'s'`, `'^'`, `'D'`, etc. to distinguish a category; use color (via `register_color_map`) or
  position/faceting instead. Don't add a marker edge/outline (`edgecolor=`) by default — plain
  filled dots (`edgecolor='none'`, the effective matplotlib default) unless a specific panel
  explicitly needs outlined points to stay legible (e.g. pale fill colors that vanish on a white
  background) and that need is called out at the point it's added, not applied blanket.
- **Never clip data — restructure with a broken axis instead.** If a panel's y-limits (or x-limits)
  would cut off part of the data (an outlier point, a bar that overshoots the rest), don't set
  `ax.set_ylim(...)` to crop it out of view and don't silently let matplotlib clip it. Use a broken
  axis (two stacked/adjacent axes sharing the data with a gap and break marks between them) so the
  full value stays visibly represented. This is a design decision to make at the point you notice
  clipping, not a `Styler` method to reach for automatically — ask before building out a shared
  broken-axis helper if a specific panel needs one.
- **Reference/null lines**: a named reference or null value gets one continuous labeled line
  (dashed), not a short segment repeated once per column/group — this holds even when the value is
  drawn per-column (e.g. one baseline measured separately for each column but landing on the same
  number): if the values agree, draw one line spanning the columns, not disconnected dashes under
  each. An unlabeled reference line (a bare dotted rule at zero, with no legend entry and no
  caption reference) should be labeled or removed — don't leave a mark on the panel whose meaning
  isn't stated anywhere.
- **Don't repeat an unchanging n across columns**: when paired/shared-cohort columns share one
  sample size, state "(n=...)" once (on the first column, or in the caption) rather than
  suffixing every column and legend entry with the same number — repetition reads as if the counts
  might differ.
- **Whitespace over dividers**: prefer real subplot gaps (`wspace`/separate axes) over drawn
  dashed rules to fake panel separation.
- **Panel scope — don't let one figure carry a tangential comparison**: if a subgroup needs its
  own exclusion criteria, its own marker/style treatment, or a paragraph of caveats to justify
  sitting next to the main columns, that complexity is a signal it doesn't belong in the main
  panel at all. Order columns/groups to read as the narrative progresses, with nothing extra
  wedged in the middle — give a tangential group its own small supplementary panel instead, or
  drop it.
- **Verify a subgroup is real before designing plotting logic around it**: before adding
  exclude/split/aside handling for a subgroup, check its label and membership against the actual
  source data, not just the comment/docstring that introduced it — a plotting workaround for a
  mislabeled or artifact-driven category just launders the bug into a published figure.

## Don't

- Don't call `ax.grid(True)` or apply a seaborn/whitegrid style after `Styler()` runs — grid is
  off by default (nothing in `Styler` ever turns it on); if gridlines show up in a figure,
  something downstream explicitly enabled them. Delete that call. If a value-reading aid is
  genuinely needed, use at most one light line at a meaningful reference value (`color='gray',
  alpha=0.3, ls=':'`), never a full grid.
- Don't bake explanatory prose (marker-semantics legends, methodology footnotes, multi-line
  captions) into the image — that belongs in the external figure caption, not the PNG/SVG.
- **Never auto-add a small gray italic line below the title** (e.g. `fig.text(..., color='gray',
  style='italic', fontsize=6)` or an `ax.text(..., style='italic')` block/cohort subtitle) —
  whether it restates numbers already visible in the plot/console, or just adds cohort/context
  prose ("held-out organoids (novel tissue)") that reads as a caveat. This is a recurring model
  tic — newer agents reach for it reflexively on almost every figure — not a house-style element;
  nothing in `Styler` produces one, and it wasn't asked for. Print that information to the
  console / write it in the script docstring instead. Only put a caption line in the image if the
  user explicitly asks for one on that panel.
- Don't instantiate `Styler()` or call `get_styler()` after the first plot call in a script.
- Don't let a floating annotation (a Δ value, an effect size) sit directly on top of the data it
  summarizes without a leader line or bracket connecting it — move it clear or connect it.
- Don't hand-roll rcParams or import scienceplots/seaborn styles directly in a plot script —
  that's `Styler`'s job.
- **Don't override a value `Styler` already governs on a per-call basis** — an explicit
  `fontsize=`/`fontweight=`/`fontfamily=` on a title, label, or legend call; a one-off
  `linewidth=`/`color=` that departs from what `Styler`'s rcParams would already produce. This is
  the same rule as the one above applied per-element instead of per-script: the title-bold example
  already documented under **Titles** (a script passing `fontweight="bold"` to fight or duplicate
  `axes.titleweight`) is one instance of it, not a special case — so is a `color=` passed to
  `add_significance_bracket`/`add_significance_label` to match a panel's own category/genotype
  palette instead of the house `black`/`0.3` defaults (see **Significance brackets** above). If a `Styler` default is actually
  wrong for a panel, that's a signal to fix the default in `journal_specs.py`'s role table (or, if
  it's genuinely figure-specific, `Styler(journal=..., tick_pt=5)` at construction) — not to patch
  around it locally, which is exactly how panels drift out of sync one override at a time.
  🚨 **A `fontsize=` is never the fix for crowded text — LAYOUT is.** Shorten the label string,
  wrap it, widen the panel, move the legend, open y-headroom. The PI's standing directive is that
  an override comes out *even if it breaks the figure*, and breakage is repaired structurally.
  `check_no_fontsize.py` enforces this and will fail the moment one reappears.

## Gaps in `Styler` itself worth extending (don't route around them, fix the class)

- ~~`finish_plot()` has no shared `tight_layout` toggle~~ — fixed 2026-08: `finish_plot(...,
  exact_size=True)` now saves at exactly `width_pt`/`height_pt` instead of `bbox_inches='tight'`'s
  content-dependent crop (see the Sizing bullets above). Per-topic `save_panel()` wrappers
  (`elec_helper.save_panel`, etc.) still call `plt.savefig` directly rather than through
  `finish_plot`, so they don't get `exact_size` — route them through `finish_plot` (or add the same
  `Bbox` trick locally) if a topic needs pixel-consistent composed panels via `save_panel`.
- `Styler.colors` (the Python list) and the `axes.prop_cycle` rcParam are two hand-synced copies
  of the same 12-hex list (`plot_styler.py:101-114` and `:141-152`) — a future edit to one that
  misses the other silently reintroduces drift. Derive one from the other.

## Where `Styler` lives

- Canonical class: `braindance/utils/data_manager/utils/plotting/plot_styler.py`
- API used above: `create_figure`, `register_color_map` / `get_category_color`, `add_scalebar`,
  `add_significance_bracket`, `label_panel`, `finish_plot()` / project-level `save_panel()`
  wrappers.
