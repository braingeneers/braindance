"""Journal page-layout specs for `Styler`'s composite (multi-panel) figures.

A journal's column widths / max page depth / dpi floor are DATA, not behavior -- nothing
about how a composite figure gets drawn changes between journals, only the numbers that
size its canvas. So this is a small frozen-dataclass registry, not a `Styler` subclass per
journal: `Styler(journal="nature_neuro")` resolves one of these onto `self.journal` at
construction (see `plot_styler.py`), rather than each journal getting its own class.

Add a new journal by adding one `JournalPageSpec` entry to `JOURNAL_SPECS` below -- do not
subclass `Styler` to do it.

Only add a profile once you have the publisher's actual column-width + max-depth spec --
never a working canvas someone chose by hand in Illustrator.

🚨 **`"nature_neuro"` is NOT flagship `"nature"`.** Nature Neuroscience is a Nature-*branded*
research journal and gets a different page: **88 / 180 / 170 mm**, where the flagship's own
guide says 89 / 183 / 247. Both are registered below; don't reconcile them.

🚨 **`journal` is REQUIRED at `Styler` construction** (2026-08-13 PI directive) -- there is
no default, so a script cannot inherit a house style by omission. Non-paper work passes
`"draft"`, which is a registered profile like any other: it warns instead of raising and
has no page geometry. A profile carries three things, not one: the page, the text SIZE
BAND used for checking, and the per-role SIZES applied to `rcParams` at construction. That
third one is new -- see `TextSpec`'s role-size block for why it exists and what it fixes.

Text rules, from Nature's artwork guides (verified 2026-08-13), enforced via
`Styler.check_journal_compliance()`:
  * sans-serif (Helvetica/Arial), **5-7 pt**; Symbol font for Greek;
  * panel letters **8 pt bold, upright, lowercase `a, b, c`**;
  * text and line art stay vector and editable -- never outlined or rasterized;
  * scale bars, not magnification factors; bitmaps >=300 dpi at final size.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


def _mm_to_pt(mm: float) -> float:
    return mm * 72 / 25.4


@dataclass(frozen=True)
class TextSpec:
    """Text-size house style, with no page geometry attached.

    Text sizes are canvas-independent -- a 10 pt label is over the ceiling whether it is
    rendered standalone or carved onto a composite page. That is why this exists apart
    from `JournalPageSpec`: a paper panel is still built with its real journal (the
    geometry check only ever flags an OVERSIZE canvas, so it passes trivially on something
    smaller than a page), while non-paper work uses `"draft"`, which has no page at all.
    Either way the text rules are the same object, because a `draw_*` function is
    exactly what a composite later calls, so a text override that goes unflagged there
    reappears on the paper page. `Styler.check_journal_compliance()` checks text against
    this on EVERY figure, and geometry only when a page profile is set.

    🚨 **The ROLE SIZES below are the reason a plot script should never write `fontsize=`.**
    Before 2026-08-13 `Styler.set_style()` hardcoded every one of these to 7 -- the TOP of
    the 5-7 pt band. Nothing about that violated the spec, but it collapsed a range to a
    point: there was no legal DOWNWARD move that wasn't a call-site override, so every
    dense panel hand-rolled its own hierarchy (`legend fontsize=5.5` appeared in ~26
    cartpole scripts alone). Those overrides were not deviations from house style, they
    WERE the missing house style -- which is why stripping them (PI directive, same day)
    dropped legend boxes onto the data in panels 03/06/06b/08c. The hierarchy belongs
    here, once, so that the strip costs nothing.

    ⛔ Do not "fix" a crowded panel by raising/lowering one of these at the call site. A
    genuine per-figure need goes through `Styler(journal=..., tick_pt=5)` -- a
    CONSTRUCTION-time override of the resolved spec, visible in one place and greppable --
    never a `fontsize=` on an artist. See `Styler.__init__`.
    """
    min_text_pt: float = 5.0
    max_text_pt: float = 7.0
    panel_letter_pt: float = 8.0
    panel_letter_case: str = "lower"
    # --- role sizes, all inside [min_text_pt, max_text_pt] -----------------------------
    # Three tiers: PRIMARY text (what the reader must read to parse the panel -- the title
    # and the axis names) sits at the 7 pt ceiling; SECONDARY text (tick labels, in-panel
    # annotation) at 6; the LEGEND at the 5 pt floor (PI, 2026-08-13). `axes.titleweight`
    # is already bold house style, so the title is separated by weight as well as size.
    # The legend is the quietest role on purpose: in this project's panels the key is
    # usually glossing something the axis already names (a reference line, a pair label
    # that is also an x tick), so it should cost the least space of anything on the panel.
    title_pt: float = 7.0
    label_pt: float = 7.0
    tick_pt: float = 6.0
    legend_pt: float = 5.0
    # `annotation_pt` backs matplotlib's `font.size`, i.e. it is the FALLBACK any artist
    # gets when nothing more specific applies -- bare `ax.text()`, significance-bracket
    # stars, `n=` counts, colorbar labels. 6 for the same reason as legend/tick: an
    # annotation is a gloss on the data, not a heading.
    annotation_pt: float = 6.0


#: Names of the role fields a caller may override at `Styler` construction time.
ROLE_FIELDS: Tuple[str, ...] = (
    "title_pt", "label_pt", "tick_pt", "legend_pt", "annotation_pt")

DEFAULT_TEXT_SPEC = TextSpec()


@dataclass(frozen=True)
class JournalPageSpec:
    name: str
    single_col_pt: float
    double_col_pt: float
    max_depth_pt: float
    min_dpi_color: int
    min_dpi_line_art: int
    # --- text/label house style (see module docstring + TextSpec) --------------------
    min_text_pt: float = 5.0
    max_text_pt: float = 7.0
    panel_letter_pt: float = 8.0
    panel_letter_case: str = "lower"
    title_pt: float = 7.0
    label_pt: float = 7.0
    tick_pt: float = 6.0
    legend_pt: float = 5.0
    annotation_pt: float = 6.0
    # --- enforcement ------------------------------------------------------------------
    # `has_page`: False for DRAFT, which is a text house style with no publisher page
    # behind it. Guards both the geometry half of `check_journal_compliance()` and
    # `Styler.composite_canvas_size()`, which raises rather than inventing a page.
    # `strict`: True makes an out-of-band text size FATAL at save time (the file is still
    # written first -- see `Styler.finish_plot` -- so the offender can be looked at). A
    # printed warning was ignorable and got ignored: 220 out-of-spec labels accumulated
    # under one. ⛔ There is deliberately NO per-call escape hatch on a strict profile;
    # a figure that genuinely needs a different size changes the SPEC (`Styler(...,
    # tick_pt=5)`), which is one greppable place, not a `fontsize=` buried in a panel.
    has_page: bool = True
    strict: bool = True


# Nature-branded research journals (Nature Neuroscience, Nature Methods, ...):
#   1-column  88 mm -> 249.4 pt      2-column 180 mm -> 510.2 pt (a stated MAXIMUM)
#   depth    170 mm -> 481.9 pt      (leaves the page room for the legend)
# The depth is a build target, not a hard ceiling: the branded-journal artwork guide allows
# a 2-column figure 185 mm deep at a <300-word caption, 225 mm at <50 words. Extended Data
# is 180 x 170 mm, so this profile sizes those too.
# `min_dpi_line_art=600` is a local convention -- no Nature guide states one, because line
# art is expected to stay vector; it is the floor for when a raster is unavoidable.
NATURE_NEURO = JournalPageSpec(
    name="nature_neuro",
    single_col_pt=_mm_to_pt(88),
    double_col_pt=_mm_to_pt(180),
    max_depth_pt=_mm_to_pt(170),
    min_dpi_color=300,
    min_dpi_line_art=600,
)

# Flagship *Nature*: 89 / 183 mm, full page depth 247 mm (1.5-column: 120 or 136 mm).
NATURE = JournalPageSpec(
    name="nature",
    single_col_pt=_mm_to_pt(89),
    double_col_pt=_mm_to_pt(183),
    max_depth_pt=_mm_to_pt(247),
    min_dpi_color=300,
    min_dpi_line_art=600,
)

# Exploratory / non-paper work: analysis scratch, QC, GUI, and the `braindance` library's
# own plotting helpers, which have external callers who are not writing a paper.
#
# 🚨 THIS IS NOT "no journal" WEARING A NAME. It exists so that `journal` can be a REQUIRED
# argument (2026-08-13 PI directive: an agent must not be able to get a spec by omission).
# Before this, `Styler()` silently produced figures under an unstated house style and 165
# of the repo's 172 call sites took that path. Now the choice is always written down.
#
# Differences from a real journal profile, both deliberate:
#   * `strict=False` -- warns instead of raising. A scratch plot is not a compliance
#     surface, and hard-failing `braindance/utils/.../plotting/*.py` would break shipped
#     library functions for figures nobody is publishing.
#   * ROLE SIZES ARE LARGER (10/9/8/8/9, band 4-16 pt). Nature's 6 pt tick label is sized
#     for a 180 mm printed page, and is genuinely hard to read on screen at a scratch
#     figure's size. ⛔ Do not "unify" these with the Nature numbers -- the point of a
#     draft profile is that it is NOT the paper's typography.
DRAFT = JournalPageSpec(
    name="draft",
    single_col_pt=0.0, double_col_pt=0.0, max_depth_pt=0.0,   # unused; see has_page
    min_dpi_color=150, min_dpi_line_art=150,
    min_text_pt=4.0, max_text_pt=16.0, panel_letter_pt=12.0,
    title_pt=10.0, label_pt=9.0, tick_pt=8.0, legend_pt=8.0, annotation_pt=9.0,
    has_page=False, strict=False,
)

# Cell Reports Methods (Cell Press, life-science family). Added 2026-08-14 for a
# colleague's submission. Researched via live web search -- cell.com 403's every
# automated fetch (including from an agentic web-research delegate), so every
# number below comes from Google's index of the real page or a search-grounded
# summary of it, not a page fetched and read directly, except where flagged.
#
# 🚨 COLUMN WIDTHS: two candidate grids surfaced and only one applies here.
#   * sibling Cell Press *physical-science* journals (Chem, Matter, Joule, Cell
#     Reports Physical Science) publish a 3-column grid: 53 / 112 / 172 mm.
#   * the flagship *Cell* page -- same life-science family as Cell Reports
#     Methods -- gives 85 / 114 / 174 mm, plus a separately stated recommended
#     max full-figure envelope of 6.5 x 8 in = 165 x 203 mm.
# This lab is life science (confirmed 2026-08-14), so the 85/114/174 mm grid is
# the right family, not the 53/112/172 one. The 165 mm envelope width reads as a
# conservative "leave room for margins/caption" recommendation sitting inside
# the 174 mm column ceiling, not a competing hard limit -- so `double_col_pt`
# uses 174, and `max_depth_pt` uses the envelope's 203 mm (8 in), matching how
# NATURE_NEURO already treats its own depth figure as "a build target, not a
# hard ceiling." ⚠️ None of this was confirmed by directly reading the live
# page or a downloaded submission template (cell.com 403'd every fetch
# attempt) -- before final submission, open
# cell.com/cell-reports-methods/information-for-authors in a real browser and
# diff against these three numbers.
#
# Text band IS corroborated across independent sources: 6-8 pt at final print
# size; Arial as the accepted fallback for Cell's house font Avenir -- already
# the hardcoded `font.family` at plot_styler.py:189, so no code change needed
# there. Panel letters are UPPERCASE bold at Cell Press, unlike Nature's
# lowercase -- 🚨 but `panel_letter_case` is DEAD DATA: nothing in
# plot_styler.py reads it, `label_panel(ax, letter, ...)` takes a literal string
# from every call site. Set here for documentation only; making it real means
# auditing every `label_panel(ax, "a", ...)` call in the topic dirs and changing
# "a" -> "A", which is a separate, out-of-scope task.
CELL_REPORTS_METHODS = JournalPageSpec(
    name="cell_reports_methods",
    single_col_pt=_mm_to_pt(85),
    double_col_pt=_mm_to_pt(174),
    max_depth_pt=_mm_to_pt(203),
    min_dpi_color=300,
    min_dpi_line_art=1000,  # one source says 500 for non-line grayscale; 1000 is the
                             # stricter (vector line-art) figure -- the safer floor to hold
                             # code to, since overshooting a real 500 dpi minimum can't fail
                             # a submission the way undershooting a real 1000 dpi one could.
    min_text_pt=6.0, max_text_pt=8.0,
    panel_letter_pt=8.0, panel_letter_case="upper",
    title_pt=8.0, label_pt=8.0, tick_pt=7.0, legend_pt=6.0, annotation_pt=7.0,
    has_page=True, strict=True,
)

JOURNAL_SPECS: Dict[str, JournalPageSpec] = {
    "nature_neuro": NATURE_NEURO,
    "nature": NATURE,
    "cell_reports_methods": CELL_REPORTS_METHODS,
    "draft": DRAFT,
}


def get_journal_spec(journal) -> JournalPageSpec:
    """Resolve a journal name (str) or an already-built `JournalPageSpec` to a spec.

    Passing a `JournalPageSpec` directly through lets a caller define a one-off/local
    profile without adding it to the shared registry.
    """
    if isinstance(journal, JournalPageSpec):
        return journal
    try:
        return JOURNAL_SPECS[journal]
    except KeyError:
        raise ValueError(
            f"Unknown journal {journal!r}. Available: {sorted(JOURNAL_SPECS)}. "
            "Add a new JournalPageSpec to journal_specs.py, or pass a JournalPageSpec "
            "instance directly for a one-off profile."
        )
