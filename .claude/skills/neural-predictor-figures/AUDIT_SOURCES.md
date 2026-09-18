# Source material for figure-style/SKILL.md

Raw output of the two subagents run 2026-07-27 to derive the skill. `SKILL.md` is the
distilled, actionable version — read this only if you need the full reasoning, exact
numbers, or a citation the summary compressed away.

Reference figure used: `electrode_disc.png` (Illustrator-finished, from
`/Volumes/hunter_ssd/neural_predictor/electrode_disc/`). Current-output comparators:
`general_metrics/fine_tuning/plots/01_main_panels.py`'s `p00_card_bits.png`,
`p00_card_corr.png`, `p03_benefit_gradient.png` (rendered 2026-07-27 to `$VAL_FIG_DIR`).

---

## Report 1 — Opus, visual/stylistic teardown (reference vs current output)

### Images examined
- **Reference (target style):** `electrode_disc.png` — 5-panel (A–E) composite figure, Illustrator-finished
- **Current output (style to move away from):** `p00_card_bits.png`, `p00_card_corr.png`, `p03_benefit_gradient.png` — pure matplotlib

### 1. Typography

**Reference:**
- Font reads as a clean grotesque sans-serif (Helvetica/Arial-family — not visible DejaVu Sans artifacts like the double-story "g" or exaggerated x-height). Uniform stroke width, tight, professional.
- Clear four-level hierarchy: **panel letter** (A/B/C/D/E) is the single largest/boldest text in the whole figure, bold, ~16–18pt-equivalent, positioned flush top-left of each panel, outside the axes box. **Panel title** (e.g. "Electrode-stim separability," "Stim feature masking (frozen model latents)") is bold, ~12–13pt, centered over the panel. **Panel subtitle/annotation** (e.g. "One point per stim response, colored by electrode; X marks centroids (n = 10)") is regular weight, smaller, sometimes gray, occasionally italic (the "evoked population response" caption inside panel A is italic). **Axis/tick labels** are the smallest, regular weight.
- Capitalization is **strictly sentence case** everywhere — "Representative electrode discrimination by each method," "FT increase extends to pretraining-holdout chips." Abbreviations (FT, LD1, LD2, LDA) are the only capitalized tokens. Never Title Case, never ALL CAPS for emphasis.
- Titles are **neutral/descriptive**, not claims — they name what's being measured ("Electrode-stim separability"), not the conclusion.
- Text sits outside/above the plot area, never overlapping data.

**Current output:**
- Font in all three files visibly reads as matplotlib's default DejaVu Sans — rounder terminals, wider numerals, a slightly "software default" feel rather than a typeset one. This alone is a strong "AI-generated" tell because it's the one thing every unstyled matplotlib figure shares.
- Titles do double duty as **headline + caption**: `"bits/spike (scale-sensitive): the cost is novel TISSUE, not a held-out recording"` and `"Fine-tuning supplies organoid identity (large OOD gain; on a known organoid, a small loss)"`. These are long, wrap to 1–2 lines, and dominate the panel visually because there's no smaller/lighter subtitle tier to defer to — the title carries all the weight matplotlib's single `title()` call gives it.
- Capitalization is **inconsistent**: mostly sentence case, but "TISSUE" is capitalized mid-sentence purely for emphasis. That's a caps-as-shouting move a journal figure never makes — emphasis should be italics or bold, not case-shift.
- `p00_card_*` uses a secondary italic-gray annotation layer (section headers like *"acute mouse slice (held-out block ≈ novel)"*, and two lines of italic caption text below the x-axis explaining marker semantics). This is effectively a figure caption baked into the raster image — over-explanation that a print caption should carry instead.
- No panel-letter convention at all in the current output (these are single-panel figures, so it doesn't directly apply, but there's also no analogous "eyebrow" text tier).

### 2. Color

**Reference:**
- Palette is **categorical but restrained**: the schematic in panel A defines exactly three colors — **medium gray** (~#7F7F7F, "raw FR"), **muted steel blue** (~#3D6FB4, "frozen FM"), **brick/brand red** (~#C0392B–#B33227, "LoRA+MLP FT"). These three colors are then **reused verbatim** in panels C and D (boxplots) and again in panel E — the legend is defined once, visually, in the schematic, and never repeated as a boxed legend afterward.
- Panel B's scatter uses ~8–10 pastel colors at low alpha (~0.3) for the point cloud, with the **same hue at full saturation** for the "X" centroid marker — a consistent "light cloud + saturated summary mark" convention.
- Panel E introduces one new orthogonal encoding (train = dusty blue, holdout = tan/wheat), and *only there* does a small legend box appear — because it's a genuinely new dimension, not a redundant repeat.
- Overall saturation is **low-to-medium** — nothing reads as "default matplotlib blue/orange/green." Colors feel chosen (flat-UI / Tableau-muted register), not defaulted.
- Background is pure white; no fills except plot elements themselves.

**Current output:**
- `p00_card_bits/corr` colors are already reasonably muted (blue/orange open circles for the acute-slice pair, blue/gold filled circles for in-dist, teal-green circles+diamonds for holdout) — this part is close to reference quality.
- `p03_benefit_gradient` is the outlier: the green bar reads as a **saturated teal-green**, notably brighter/more synthetic than anything in the reference palette, sitting next to a flat gray — this is a two-color-only chart but the green pulls toward "default-looking" because it's more saturated than its gray partner rather than tonally matched to it (reference pairs always feel like they belong to the same "muted" family).
- No cross-panel color-key reuse strategy is visible in the current set (each figure re-derives its own palette rather than inheriting one fixed key), though this may just be a sampling artifact since only one figure per "family" was shown.

### 3. Layout & Composition

**Reference:**
- Multi-panel grid (2 rows: A/B top band, C/D/E bottom band) with generous whitespace between panels — panels don't touch, don't share a border, and are visually independent blocks.
- **No gridlines anywhere.** Not even subtle ones.
- **Spines: left + bottom only**, thin, likely ~0.75–1pt, medium gray or black. Top and right always removed.
- Tick density is sparse — panel B's LD1/LD2 axes show ~4–5 ticks max, panel C/D/E y-axes show ~5–6 ticks. No overflow, no crowding.
- Legends are avoided in favor of (a) a schematic that defines the color key once, (b) inline subtitle text ("X marks centroids"), and (c) a single small boxed legend only in panel E, where genuinely necessary — and even that legend has minimal/no visible border weight, sitting flush in a corner rather than looking like a UI widget dropped on the chart.
- One deliberate boxed element: panel A's raster plot has a subtle rectangular frame — used because it's literally a schematic diagram element, not decoration on a normal axes.

**Current output:**
- `p00_card_*`: panels are internally divided into 3 sections using **vertical dashed gray lines** — functional, but this is doing with chart-junk (extra line elements) what the reference would do with actual subplot/whitespace separation.
- Horizontal dotted zero-line plus **multiple short dashed reference-line segments** (one per column-pair, labeled "causal EWMA (per column)") — precise, but visually busy: six separate short dashed strokes instead of one continuous line or a single annotation.
- `p03_benefit_gradient`: has classic **matplotlib/seaborn "whitegrid" horizontal gridlines** at every major y-tick, spanning full plot width. This is the single most recognizable "unstyled/default" signature in the whole set — the reference has zero gridlines, period.
- `p03` legend is a **fully boxed rectangle** (visible border, top-right corner) — exactly the generic matplotlib legend widget the reference systematically avoids.
- Tick density and margins otherwise reasonable; no severe crowding.

### 4. Chart-junk / Minimalism — what to explicitly REMOVE

1. **Horizontal gridlines** in `p03` (seaborn/whitegrid default) — delete entirely, or if orientation cues are truly needed, use a single very-light (`alpha≈0.15`, gray) line only at y=0.
2. **Boxed legend with visible border** in `p03` — replace with inline text labels near the data, or a borderless/loose legend (`frameon=False`) placed to not overlap points.
3. **ALL-CAPS mid-sentence emphasis** ("TISSUE") — remove; use italics or bold instead if emphasis is needed at all.
4. **Multi-line, claim-as-title headlines** that also carry the figure's "finding" — split into a short neutral bold title + a smaller regular-weight descriptive subtitle (mirrors reference's title/subtitle split).
5. **Bottom-of-figure italic caption block** (2 lines of prose explaining marker semantics) in `p00_card_*` — this belongs in the LaTeX caption, not baked into the PNG; if it must stay for now, shrink it and set it clearly apart (e.g., footnote-style, very small, gray, single line).
6. **Vertical dashed section-divider lines** — acceptable as a last resort, but prefer actual whitespace gaps or genuine `plt.subplots` panel breaks over drawn dashed rules where possible.
7. **Saturated single-color bars next to muted grays** in `p03` (the bright green) — pull down saturation/lightness to match the muted register used elsewhere (dusty teal, not bright teal).
8. Floating **Δ annotation text with no leader line/bracket** sitting directly over jittered scatter points in `p03` — either move it clear of the point cloud or connect it to what it's summarizing with a thin bracket, matching the reference's significance-bracket convention.

### 5. Statistical / Annotation Conventions

**Reference:**
- Significance is shown with a **thin black horizontal bracket** connecting two box/group centers, small vertical tick-ends, with `***` or `ns` centered above the bracket — classic GraphPad/ggpubr `stat_compare_means` convention. Brackets nest/stack cleanly when there are multiple comparisons (panel C shows three stacked brackets).
- Boxplots: standard box (25/50/75%), black median line, whiskers with caps, **individual points overlaid** as small circles, colored by group, black outline, semi-transparent connecting lines between paired points (thin, low-alpha gray "spaghetti" lines) showing within-subject pairing.
- Sample size stated once in the subtitle text ("n = 10"), not repeated per axis tick.
- No shaded confidence bands are shown in this particular figure — CI/spread communicated through the boxplot itself plus raw point jitter.

**Current output:**
- `p00_card_*` uses its **own coherent, arguably more information-dense** convention: thick black horizontal bar = median, thin gray vertical bar = bootstrap 95% CI, connector lines = paired organoids, open vs. filled markers = distinct cohort semantics, dashed line = a named reference model (EWMA) — this is actually good scientific-annotation design and close in *spirit* to the reference (real error semantics, not decorative error bars). The weakness is presentation density (see §4), not the underlying encoding logic.
- `p03_benefit_gradient` mixes a bar chart with jittered points on top and a bold floating Δ value — a legitimate combination, but lacks the reference's bracket/tick convention for tying the annotation visually to what it summarizes.
- Sample sizes (`n=47`, `n=11`) are placed under x-tick labels in both current-output figures — this actually matches reference convention well.

### 6. Data-ink Ratio

The reference earns nearly every stroke: the schematic (panel A) doubles as a methods diagram *and* the color legend for every subsequent panel; the "X" centroid marker directly encodes the summary statistic being compared; significance brackets appear only where a comparison is made, stacked compactly. By contrast, in the current output some elements earn their place (CI bars, connector lines, dashed reference line) but others are decorative overhead: full-width gridlines in `p03` don't help read any value not already on the y-ticks; the boxed legend border is pure chrome; caps-emphasis and prose captions duplicate what the trimmed axis/title text already implies.

### 7. Overall Gestalt

The reference reads as **professionally typeset science** because every choice reduces to "define once, reuse everywhere, remove anything not load-bearing": one three-color key set once in a schematic and inherited by three later panels, one consistent centroid-marker convention across three scatterplots, one bracket-and-stars idiom for every significance claim, sentence-case neutral titles that never compete with the data for attention, and — critically — **zero gridlines and zero boxed legends anywhere in the figure**. The Illustrator finishing pass likely tightened kerning, unified the font to a proper Helvetica-class sans, and cleaned up spine/tick weights, but the underlying compositional decisions (panel-letter hierarchy, color-key reuse, restrained tick density, bracket-style stats) are all things a matplotlib script can do on its own.

The current output reads as **"AI-generated matplotlib"** for three concrete, fixable reasons: (1) DejaVu Sans is still the rendered font, instantly recognizable as unstyled; (2) `p03` still carries matplotlib/seaborn's default whitegrid gridlines and a bordered legend box, the two most common "I didn't touch `rcParams`" tells; (3) titles do the job of both headline and caption simultaneously, producing long wrapped multi-line titles with an inconsistent capitalization convention (caps-for-emphasis) a copyedited journal figure would never contain. `p00_card_*` is actually the closer of the three to reference quality already — its statistical encoding is sound and information-dense — its remaining gap is presentation restraint, not a fundamentally different visual vocabulary.

### 8. Prioritized Checklist (highest visual impact first)

1. Set the font once, globally, at the top of every plotting script: font family fallback list (Helvetica/Arial/Nimbus Sans/DejaVu Sans, first available wins) — eliminating visible DejaVu Sans is the single highest-leverage fix.
2. Kill all gridlines by default (`ax.grid(False)`); at most one light reference line at a meaningful value.
3. Remove top/right spines always; keep left+bottom thin, gray-black not pure black.
4. Never use a bordered/boxed legend — `frameon=False` or inline text labels.
5. Define a categorical color key once and reuse those exact hex values across every panel in a figure family — never let matplotlib's default cycle leak through.
6. Cap saturation — muted/flat palette, not matplotlib tab10 defaults.
7. Titles: sentence case only, ever, no ALL-CAPS emphasis; split any finding-as-title into a short bold descriptive title + optional smaller subtitle.
8. Strict font-size hierarchy: panel label > title > subtitle/axis label > tick label.
9. Significance annotations: bracket + stars/`ns`, not floating text; stack multiple comparisons with small offsets.
10. Reference/null lines: one continuous dashed line per condition, labeled inline at its right end, not many short per-column segments.
11. Move explanatory prose out of the image into the external caption; if something must stay, one line, small, gray, italic, bottom-left.
12. Replace dashed vertical section-divider rules with real whitespace/subplot gaps where feasible.
13. Thin tick density — 4–6 major ticks per axis, no minor ticks by default.
14. Floating annotations (Δ values, effect sizes) should sit clear of the data or connect via a thin leader line/bracket.
15. Boxplots/violin+scatter combos: box + black median line + individually-colored outlined points + low-alpha thin gray connector lines for paired data — this part of the current style is already good, keep it and don't let chrome undercut it.

---

## Report 2 — Sonnet, infrastructure survey

### 1. The canonical `Styler` class

**File:** `braindance/utils/data_manager/utils/plotting/plot_styler.py`. `__init__.py` re-exports `Styler`, enabling `from braindance.utils.data_manager import Styler`.

**Import chain / hard dependency:** `Styler.__init__` → `set_style()` → `plt.style.use(['science'])`, inside a `try` that hard-raises `ImportError` if `scienceplots` isn't installed. Undeclared load-bearing dependency, not mentioned in CLAUDE.md's standard-imports block.

**What `plt.style.use(['science'])` contributes** (read from `scienceplots/styles/science.mplstyle` in the `brain` conda env):
```
axes.prop_cycle : 7-color cycle                — overridden by Styler
figure.figsize  : 3.5, 2.625                    — NOT overridden anywhere
xtick/ytick.direction : in                      — overridden to 'out'
xtick.top / ytick.right : True                  — overridden to False
xtick/ytick.minor.visible : True                — overridden to False
xtick/ytick.major.size : 3, major.width : 0.5   — NOT overridden
axes.linewidth : 0.5                            — re-set to same value (redundant)
legend.frameon : False                          — NOT overridden (shapes every legend)
font.family : serif, mathtext.fontset : dejavuserif — overridden to Arial / dejavusans
text.usetex : True                              — overridden to False
savefig.bbox : tight, pad_inches : 0.05         — re-set to same values (redundant)
```
Practical takeaway: house style is "scienceplots minus LaTeX minus serif minus inward/framed ticks." `legend.frameon: False` and tick major size/width survive untouched. `figure.figsize` (3.5×2.625in) is the implicit default for any bare `plt.subplots()` that doesn't go through `Styler.create_figure()`.

**`Styler.set_style()` override block:** single `plt.rcParams.update({...})` setting `font.family: 'Arial'`, `font.size: 7`, `axes.linewidth: 0.5`, tick/label/legend fontsize 7, `lines.linewidth: 0.5`, minor ticks off, ticks `out`, spines top/right hidden via `mpl.rc('axes.spines', ...)`, and a 12-color `axes.prop_cycle` (Wong-derived colorblind-safe cycle). Then `text.usetex: False`, then a second `mpl.rcParams.update(svg_settings)` from `configure_svg_settings()`: `svg.fonttype: 'none'`, `pdf.fonttype: 42`, `ps.fonttype: 42`, `figure.dpi`/`savefig.dpi: 300`, `savefig.bbox: 'tight'`.

**Design intent, stated in code:** `svg.fonttype:'none'` + `pdf/ps.fonttype:42` exist specifically so exported SVG/PDF text stays editable in Illustrator rather than becoming vector paths — corroborated by `finish_plot`'s default `formats=['png','svg']` with SVG metadata `{'Creator': 'Matplotlib/BrainDance'}`, and `electrode_discr`'s own docstrings describing scripts as producing "skeletons for Illustrator." **The house style is matplotlib-to-editable-vector, not matplotlib-as-final-output.**

`Styler.colors` (the 12-hex list) is defined separately from, and duplicated verbatim into, the `axes.prop_cycle` set — two copies of the same palette kept in sync by hand.

**Public methods:** `register_color_map(name, mapping)`; `create_figure(nrows, ncols, width_pt, height_pt, size_preset, **kwargs)` (points-based sizing, presets `'single'`≈277.5×223.8pt / `'1.5'` / `'double'` — called by almost nobody; most scripts pass explicit `width_pt`/`height_pt` or bypass entirely with bare `plt.subplots(figsize=...)`); `finish_plot(save_plots, save_dir, name, dpi=300, formats=['png','svg'], show=False)` (tight_layout, defensive spine hiding, saves each format, SVG metadata); `get_heatmap_cmap(key)` (used by `example_plots`, not `electrode_discr`). Spine removal (top/right) enforced both globally in `set_style()` and defensively again in `finish_plot()`. Gridlines are never turned on anywhere in the class — off by default project-wide, no code path re-enables it.

**The ordering foot-gun (stated as a rule, not just observed):** `Styler.__init__` mutates global rcParams as a side effect the moment it's constructed. Any figure created via `plt.subplots()` *before* `get_styler()`/`Styler()` runs in that process is unstyled. Because `get_styler()` is sometimes called deep inside a helper function rather than at module top (e.g. `general_metrics/.../01_main_panels.py` inside `_pub_card_panels`, not at import time), this is a live risk.

### 2. `general_metrics/` usage

**`validation_metrics/plots/_common.py`:** `get_styler()` instantiates the real `Styler` and registers project color maps; `save_panel()` wraps `finish_plot`/direct savefig (PNG+SVG, dpi=300); `FIG_DIR` resolved via `get_output_dir()`. Palette constants (`_common.py:57-78`, commented "Split ordering and colors (Wong colour-blind-safe palette)"): `SPLIT_ORDER = ["train","val","holdout"]`; `SPLIT_COLORS = {"train":"#0072B2","val":"#E69F00","holdout":"#009E73"}`; `TYPE_MARKERS = {"spontaneous":"o","stim":"^"}`; `RUNG_COLORS = {"raw":"#999999","gain":"#56B4E9","ewma":"#000000","rung1_gainbias":"#E69F00","rung2_linhead":"#009E73","rung3_mlphead":"#CC79A7"}`. A recurring de facto swarm-plot idiom (`edgecolor="white", linewidth=0.3`, median hline height 2.4, IQR bar 1.0 `ls=":"`) appears near-identically at three call sites instead of one shared function — call the existing `_swarm_column` helper rather than re-deriving.

**`fine_tuning/plots/_common.py`:** re-exports/wraps the validation_metrics helpers, structurally a parallel companion, not an independent stack.

**`_sweep_common.py`** (third, hand-synced palette): `SPLIT_STYLE = {"train_val": dict(color="#0072B2", ls="--", marker="o", ...), "holdout": dict(color="#009E73", ls="-", marker="s", ...)}` — duplicates `SPLIT_COLORS`' hex by hand instead of importing.

**`_strat_common.py`:** `_STRAT_PALETTE = ["#0072B2","#D55E00","#009E73","#CC79A7","#F0E442","#56B4E9","#E69F00","#000000"]` — the literal canonical 8-color Wong (2011) set, labeled "8-class Wong-extended palette for stratifier values."

**Missing-feature trap:** `save_cross_panel` has no `tight_layout` parameter, so scripts hand-roll `plt.tight_layout()` themselves (`11_organoid_distribution.py:276,304,319`; `19_plot_sweeps.py:283,335`).

**Ad-hoc literals bypassing shared constants (sampled):** `01_main_panels.py:276`; `11_organoid_distribution.py:156-159`; `19_plot_sweeps.py:262,318`; per-panel one-offs (`10_plot_seed_variance.py:34-40` `BIN_COLORS`/`BIN_ORDER`/`BIN_MARKERS`, a 3-way variant of `15_plot_bin_advantage_vs_scale.py:40`'s 2-way `BIN_COLORS`; `22_plot_shank3_separability.py:43` `GENO_COLOR`); `deprecated/14_plot_bin_paired_delta.py:36` and two `derive_data/scratch/` files hand-copy `SPLIT_COLORS` verbatim instead of importing.

### 3. `electrode_discr/` and `example_plots/`

**Bottom line: electrode_discr uses the exact same canonical Styler as everything else — the visual quality difference is NOT due to different styling machinery.**

Every current and deprecated panel imports `get_styler` from `elec_helper.py`, which imports the real class directly (`from braindance.utils.data_manager import Styler`). `elec_helper.py`'s `get_styler()` registers two color maps via the real `register_color_map` API (`"method"`, `"pair_d"`, e.g. `raw_FR_LDA="#888888"`, `ft_FM_MLP="#d62728"`). If the real import fails, a `_FallbackStyler` kicks in whose `colors` list is byte-identical to `cell_styler.py:86-99` — confirming the canonical palette's lineage traces back to the older `busy_bee_analysis` Styler. No import from `cell_styler.py`, `proj/figs/styler.py`, or the ephys_manager styler anywhere in `electrode_discr/`.

Sizing consistently goes through `styler.create_figure(width_pt=..., height_pt=...)`; dims are near-identical between old/deprecated and new panels. A consistent box-plot recipe recurs old and new: `patch.set_alpha(0.35)`, `edgecolor="black"`, `linewidth=0.6`, median `lw=0.8` black, whiskers/caps `lw=0.5` black.

**Illustrator handoff is explicit and structural, not incidental:** `05_plot_schematic.py` docstring — "pipeline schematic (skeleton for Illustrator)... the finished art replaces these with vector graphics in Illustrator"; `06_plot_centroid_example.py` — "Saves one figure per (run, rep) = 6 figures total so each is its own Illustrator block." `README.md` confirms panels A/C are Illustrator-composited. No automated post-processing touches saved SVG/PDF, and no `.ai` companion files exist *in the repo* (they live only on the external SSD, untracked). So: matplotlib produces dimensionally-correct, editable-text PNG+SVG skeletons at dpi=300; a human finishes them in Illustrator outside the repo.

`example_plots/plots/_common.py` also imports the canonical class directly — one shared backbone across the whole `figures/` tree. Its `get_styler()` is a bare `Styler()` with no custom color-map registration (docstring: "A bare journal-style Styler (7pt, editable-SVG text, PNG+SVG)") — needs `styler.get_heatmap_cmap()` (inferno/RdBu_r) for rasters/PSTHs, not categorical palettes.

**The three older duplicate `Styler` classes — lineage and status:**

| File | Relationship to canonical | Import count | Status |
|---|---|---|---|
| `proj/figs/styler.py` | Most primitive ancestor: `plt.style.use(['science','nature'])`, `sns.set_palette('colorblind')`, no `register_color_map`/points API | ~31 files, outside `proj/predictor/figures/` | Alive, separate older figure project |
| `proj/busy_bee_analysis/utils/cell_styler.py` | Structurally near-identical rcParams block, identical 12-color palette/order to electrode_discr's fallback, `create_figure`/points API surface — looks like a direct fork/ancestor, but bespoke dict+getter per category instead of generic `register_color_map` | ~51 files — the single most-imported styler module repo-wide | Alive, dominant outside `figures/` |
| `proj/predictor/nrp/scripts/utils/ephys_manager/utils/plotting/styler.py` | Near-clone of `cell_styler.py` (byte-identical rcParams/palette), retargeted for human SHANK3 cell lines | 0 external importers | Dead code |

**Net effect:** the canonical Styler (45 importers) is *not* the numerically dominant styling module repo-wide — `cell_styler.py` alone (51) outnumbers it, and combined with `proj/figs/styler.py` (31) the two non-canonical duplicates (82) outnumber canonical Styler's 45 nearly 2:1. It *is* exclusively used inside `proj/predictor/figures/` — every `get_styler()` wrapper found there (9 total, one per topic) instantiates the real canonical class.

### 4. Global style files / bypasses

Exactly one `.mplstyle` file repo-wide: `proj/figs/plotting/plot_style.mplstyle` — its own independent house style (figsize 8.27×6.20in, dpi 300, Arial-family sans, 8pt font, its own colorblind prop_cycle). Entirely separate from and never touched by canonical `Styler`. 18 `plt.style.use(...)` calls repo-wide, only one relevant to `proj/predictor/figures/`: `plot_styler.py`'s `['science']`. ~35 rcParams touch-points repo-wide, only 2 are the canonical Styler itself; the rest are full parallel reimplementations (`cell_styler.py`, ephys_manager `styler.py`) or bespoke per-script blocks unrelated to the figures pipeline (GUI dark themes, `interact_cartpole`, etc). Within `example_plots/diagrams/`: `bin_stride_schematic.py` sets `font.family` directly after `Styler()` presumably already ran — a potential silent override worth checking if touched.

### 5. Color palette inventory

The repo explicitly names its palette "Wong colour-blind-safe palette" (Bang Wong, *Nature Methods* 2011) in 6+ separate comments across `plot_styler.py`, `_common.py`, `_sweep_common.py`, `_strat_common.py`, `22_plot_shank3_separability.py`, `15_plot_bin_advantage_vs_scale.py`. Canonical 8-tuple: `#000000` black, `#E69F00` orange, `#56B4E9` sky blue, `#009E73` bluish green, `#F0E442` yellow, `#0072B2` blue, `#D55E00` vermilion, `#CC79A7` reddish purple.

Named project-defined dicts found (non-exhaustive, ~15+): `SPLIT_COLORS`, `SPLIT_ORDER`, `TYPE_MARKERS`, `RUNG_COLORS` (`validation_metrics/_common.py`); `SPLIT_STYLE` (`_sweep_common.py`, hand-synced duplicate); `_STRAT_PALETTE` (`_strat_common.py`); `BIN_COLORS` 2-way and 3-way variants (`15_plot_bin_advantage_vs_scale.py`, `10_plot_seed_variance.py`); `GENO_COLOR` (`22_plot_shank3_separability.py`); `GENOTYPE_COLORS`/`REGION_COLORS`/`SPLIT_COLORS`/`SPECIES_COLORS` (`human_to_mouse/plots/_common.py`); `REGION_COLORS`/`GENOTYPE_COLORS`/`CMAP`/`CMAP_GT` (`drugs/plots/_common.py`, explicit comment "IDENTICAL to the sibling discrimination figures so the whole paper stays consistent"); `OBS_COLOR`/`NULL_COLOR` (`drugs/plots/shank3_04_perm_null.py`); `CMAP="viridis"` reused across "pseudotime" panels; `GROUP_COLOR`/`AXIS_DARK`/`AXIS_LIGHT`/`GREY`/`INK` (`old/shank3_wt_discrimination/plots/gene_region_common.py`, not Wong-derived); `DIM_COLORS` (`example_plots/plots/07_plot_dim_metrics_battery.py`); `METHODS`/`PAIR_D_KEYS` (`electrode_discr/plots/elec_helper.py` — gray family raw-FR, blue family frozen-FM, `#d62728` red FT/LoRA — its own scheme, not Wong-derived).

A raw repo-wide sweep for hex literals hits 44 distinct files and 170+ individual occurrences outside `deprecated/`/`scratch/` alone — confirming the named-constant tables are the tip of the iceberg; a large fraction of color choices are inline per-script literals. Not all "wrong" (schematics often need bespoke one-offs), but a repo-wide grep for `#[0-9A-F]{6}` is the only way to get a complete inventory.

### 6. If you want to change the house style — ordered by leverage

1. `braindance/utils/data_manager/utils/plotting/plot_styler.py` — the two global `rcParams.update()` calls plus the `Styler.colors`/`axes.prop_cycle` pair. Propagates to every `get_styler()` caller in `proj/predictor/figures/` and 45 other importers repo-wide. The only place genuinely central to `figures/`.
2. `figure.figsize`/sizing has no central home — scienceplots' implicit 3.5×2.625in default is never overridden by `Styler`; `create_figure()`'s points-based presets are used consistently only in `electrode_discr/`; most `general_metrics`/`example_plots` scripts hardcode their own `figsize=`/`width_pt=`/`height_pt=` per call. Standardizing size means touching every script's call site, not one function.
3. `validation_metrics/plots/_common.py:57-78` — `SPLIT_COLORS`/`RUNG_COLORS`/`TYPE_MARKERS`, the real project-level semantic palette most `general_metrics` scripts key off.
4. `_sweep_common.py:74-79` (`SPLIT_STYLE`) and `_strat_common.py:347-357` (`_STRAT_PALETTE`) — hand-synced duplicates of #3, will drift silently if #3 changes without lockstep updates.
5. Per-topic `_common.py`/`elec_helper.py` files — each topic's local semantic-color registration layered on top of #1; change here only affects that one figure family.
6. The ~44 files with inline hex literals — the long tail; any global palette change has to be grepped-and-verified against these individually.
7. `proj/busy_bee_analysis/utils/cell_styler.py` and `proj/figs/styler.py` — out of scope for `proj/predictor/figures/`, but share DNA (identical 12-color list) with canonical Styler's fallback path and are dominant elsewhere; a house-style decision made only in `plot_styler.py` will not propagate to those.
8. Outside the repo, in Adobe Illustrator — final polish for `electrode_discr` (and likely other hero figures) happens manually post-export; not fixable programmatically, only documentable as the true last pipeline stage.
