---
name: figure-pipeline-organization
description: Folder/file organization convention for proj/predictor/figures/<topic>/ analysis pipelines — the derive_data/plots/scratch skeleton, stepN_*.py and NN_plot_*.py naming, derive_data/README.md and top-level README.md skeletons, the <TOPIC>_PUB_DIR/FIG_DIR env-var convention, and when a new pipeline should re-export a sibling's plots/_common.py instead of writing its own. Invoke when scaffolding a new figures/<topic>/ pipeline, asked to "set up a new derive_data pipeline," "where does this analysis go," or auditing an existing topic dir against convention. Companion to `figure-style` (Styler API/aesthetics) and `neural-predictor-figures` (file:line pointers, palettes) — those own what a panel looks like; this owns where files live and how they're wired together.
argument-hint: [new or existing topic name under proj/predictor/figures/]
---

# figure pipeline organization — `proj/predictor/figures/<topic>/`

This is about **folder/file layout and wiring**, not plot aesthetics — for the actual `Styler`
API, titles, violins, significance brackets, etc. see `figure-style`; for this project's file
pointers, palettes and case studies see `neural-predictor-figures`. Load those alongside this one
when actually drawing a panel; this skill only tells you where the file goes and what it's named.

## The old memory-file reference is dead — use these three dirs instead

Project memory (`feedback_figure_dir_layout.md`) says to model new topics on
`proj/predictor/figures/dorsal_vent_discrimination/README.md`. **That path no longer exists.**
It (and `shank3_wt_discrimination/`) were retired wholesale to
`proj/predictor/figures/old/dorsal_vent_discrimination/` and `.../old/shank3_wt_discrimination/`
(verified 2026-07-28) — don't chase that reference, and don't hand-derive the convention from
`memory/` prose alone; use these three **live** examples instead, by role:

- **`general_metrics/fine_tuning/`** — fullest worked example (9-step `derive_data/`, 2-file
  `plots/`, a cross-directory `_common.py` re-export). Read this first if the new pipeline is
  nontrivial.
- **`drugs/`** — mature, independently reproducible without S3 (`README.md`'s
  `## Reproduce immediately (new agent, no S3)` header — the single most useful section absent
  from the memory file, copy it). Also shows the `derive_data/old/` (superseded-but-was-real)
  pattern and per-script `get_pub_dir(script_file)`.
- **`shank3_discrim/predictability_discrim/`** — smallest correct instance (two `stepN_*.py`
  files, no plots yet, `main(recompute=True)` + `if __name__ == "__main__"`). Use this as the
  floor for "is this even worth the full skeleton" — yes, always use `derive_data/stepN_*.py`
  naming even for a 2-script pipeline.

There is also a **fourth tier**: `proj/predictor/figures/old/<topic>/` holds entire retired
topic pipelines (not just retired steps within a live one — see "three places dead code goes"
below).

## The canonical skeleton

```
<topic>/
├── README.md                        # headline result + reproduce cmd + layout tree + cohort
│                                     # + quick start + cache/artifacts table + environment
├── derive_data/
│   ├── README.md                    # | Step | Where | Device | Reads | Writes | table + order
│   ├── stepN_<verb>_<noun>.py       # sequential int, imperative verb_noun, one artifact per file
│   ├── <topic>_lib.py               # shared domain code, NO step prefix, not run directly
│   ├── old/                         # superseded steps that WERE the real pipeline once
│   │   └── README.md                # table: file -> what it was -> why/what replaced it
│   └── scratch/                     # exploratory probes that were NEVER load-bearing
│       └── README.md                # (optional) says what's in here and that none is the result
└── plots/
    ├── _common.py                   # PUB_DIR/FIG_DIR, get_styler(), save_panel(subfolder=...)
    ├── NN_plot_<noun>.py             # zero-padded 2-digit prefix, controls render order + subfolder
    └── scratch/                     # abandoned plot experiments only
```

Small joined/summary CSVs that a step writes (e.g. `shank3wt_ladder_joined.csv`,
`discrimination_results.csv`, `dv_dynamics_features.csv`) are **committed beside the script that
writes them in `derive_data/`** in all three live examples — this is what makes drugs'
"reproduce with no S3" header possible. Reserve `<TOPIC>_PUB_DIR` / S3 (see below) for the large
or expensive artifacts (pkl caches, latents, per-recording batteries) per the root CLAUDE.md
"always persist expensive computations" rule — don't over-read the old memory file's "every file
artifact lands there, never in the repo" as a rule against ever committing a derived CSV; all
three real pipelines violate it on purpose.

## Naming — verified against real files, not just the memory file

**derive_data/**: `stepN_<verb>_<noun>.py`, sequential integers, **no zero-padding**. Real
examples: `step1_build_cohort.py` … `step9_analyze.py` (fine_tuning);
`step1_join_ladder_genotype.py`, `step2_discrimination_test.py` (predictability_discrim).
- **Letter suffixes are the sanctioned way to insert a step later** without renumbering
  everything downstream: fine_tuning has `step5_build_val_cohort.py` →
  `step5b_build_val_cohort_perseed.py` → `step5c_build_train_cohort_perseed.py`. Use this, don't
  renumber.
- **A bare numeric collision is tolerated, not a bug to "fix" by renumbering**: fine_tuning
  actually has two `step8_*.py` files (`step8_prep_labels.py` and `step8_run_lora_ceiling.py`) —
  order between them doesn't matter because neither depends on the other. If you hit this, check
  whether the two scripts are actually independent before renumbering either one.
- Shared library code (no `__main__`, imported by steps and by plots) gets **no step prefix**:
  `ft_lib.py`, `build_ei_manifest_v2.py`.
- Cross-step chaining idiom (mirrors CLAUDE.md's `main()`-with-defaults convention): a later step
  imports the earlier step's `main` directly rather than shelling out or re-reading its output
  file redundantly —
  ```python
  from step1_join_ladder_genotype import main as build_joined
  ...
  def main(recompute=True):
      joined = build_joined(recompute=recompute)
  ```
  (`shank3_discrim/predictability_discrim/derive_data/step2_discrimination_test.py:20,77`). Every
  step's `main()` should accept `recompute: bool` and early-return a cached read when
  `not recompute and OUT.exists()` — same file-level caching pattern as CLAUDE.md, applied
  per-step.

**plots/**: `NN_plot_<noun>.py`, zero-padded 2-digit prefix. Real examples: `01_plot_drug_ruler.py`
… `09_plot_ei_geometry_overlay.py` (drugs); `01_main_panels.py`/`02_supp_panels.py`
(fine_tuning, one file rendering several named sub-panels, not one file per panel — both are
legitimate depending on how many panels share loaders/layout).
- **Letter suffixes for late insertions here too**: `03_plot_age_scatter.py` →
  `03b_plot_age_scatter_nbqx.py` → `03d_plot_age_scatter_dim5.py` (drugs) — there is no `03c`
  (verified: no deleted `03c*` in git history for this dir); gaps in the letter sequence are
  fine, don't backfill or renumber to close them.
- A **sub-story within a topic** gets its own name prefix instead of continuing the main numeric
  sequence: drugs' SHANK3 story is `shank3_01_nbqx_age_scatter.py` …
  `shank3_04_perm_null.py`, sitting in the same `plots/` dir as `01`–`09` without colliding.
- Local plot-only helpers (color maps beyond what `_common.py` registers, one-off geometry) can
  get a `<topic>_helper.py` (`electrode_discr/plots/elec_helper.py`) — see `neural-predictor-figures`
  for the details of that specific file.

**Never bake a main-vs-supplemental judgment into the folder path** (a `main/`, `supp/`, or
`other/` grouping directory that a panel's `subfolder` nests under). Which panels are main-text
vs. supplement is an editorial call that moves during paper writing — a panel drafted as
supplement routinely gets promoted, and vice versa — and if that call lives in the directory
name, every reclassification means renaming/moving a folder, which goes stale the moment someone
updates the doc but not the path (or the reverse). Keep every panel's output folder flat and
named ONLY after its own script (`Path(__file__).stem`, or nested one level under a real
STRUCTURAL grouping the topic-nesting section below covers — never under an editorial one).
Track main-vs-supplement classification in the narrative/figure-plan doc (`FIG2_NARRATIVE.md`'s
panel inventory, `FIGURE_PLAN.md`), where reclassifying a panel is a one-line edit, not a
filesystem move. This is a real thing this project already got wrong once (`validation_metrics/
plots/` grouped panels under `supp/` and `other/` folders that don't correspond to any script
number) — don't reproduce it in a new topic.

**Three places dead code goes — don't conflate them**:
1. `derive_data/scratch/` / `plots/scratch/` — exploration that was **never** load-bearing.
   `drugs/derive_data/scratch/README.md`: "Everything here supports the … story but is **not**
   part of the committed headline pipeline."
2. `derive_data/old/` — steps that **were** the real pipeline and got superseded by a better
   method within the same topic. Needs its own README explaining *why* it was replaced, not just
   that it was: `drugs/derive_data/old/README.md` gives a file table (what each old step did) and
   names the replacement/what makes it wrong (a centroid axis that's "null for the cross-organoid
   drug contrast," replaced by a dynamics axis).
3. `proj/predictor/figures/old/<topic>/` — a **whole topic** retired (e.g.
   `old/dorsal_vent_discrimination/`, `old/shank3_wt_discrimination/`). Use this tier when the
   entire pipeline, not just one step inside it, is superseded — e.g. by a rename/merge into a
   different topic dir.

## `derive_data/README.md` — copy-paste table template

Reproduce the real column headers exactly (`fine_tuning/derive_data/README.md:38`):

```markdown
## Execution order

| Step | Where | Device | Reads | Writes |
|---|---|---|---|---|
| `step1_<verb>_<noun>.py` | pod/local | CPU/GPU/S3 | <inputs> | `<artifact>.pkl` |
| `step2_<verb>_<noun>.py` | pod | GPU | `step1` output | `<artifact>.pkl` |

Real dependency order: 1 → 2 → 3; steps sharing a number run independently.
```

Follow the table with one `## Step N — stepN_name.py (WHERE, DEVICE)` subsection per step (e.g.
`## Step 4 — step4_run_ladder.py (POD, GPU)`), a couple sentences of what it does and what it
writes, and call out any landmine inline where it happened (fine_tuning's is the s43-only cache
gate, documented right under Step 5 rather than in a separate gotchas file).

## Top-level `README.md` — which sections are load-bearing

Cross-checked against `fine_tuning/README.md` and `drugs/README.md` (drugs has the more complete
set; fine_tuning's numbered-headers are the memory file's original source). Sections both
pipelines actually have:

- **Headline paragraph** — the claim, in prose, up top. Drugs uses an explicit `## Headline`
  header; fine_tuning folds it into the opening paragraph. Either is fine, but it must be the
  first thing after the title.
- **`## Reproduce immediately (new agent, no S3)`** (drugs) — a literal runnable bash block that
  gets a cold agent to the headline number from committed data alone, no cluster/S3 needed if the
  derived CSVs are committed. **Not in the old memory file's list — add it**; it's the single
  highest-leverage section for exactly the token-burn problem this skill exists to fix.
- **`## Cohort & definitions`** / **`## Cohort`** — n, inclusion rule, what's excluded and why.
- **`## Layout`** / **`## Pipeline & panels`** — the annotated directory tree (fine_tuning) or a
  `derive_data`-table + `plots`-table pair (drugs), each row saying what it writes and its role.
- **`## Quick start`** — literal pod/cluster invocation, in run order.
- **`## Cache / artifacts`** table, or a `## Model slot & environment` section — S3 paths,
  checkpoint slot, pod name, `PYTHONPATH`/image notes.
- **`## Superseded / process trail`** (drugs) or a **`## Future directions`** close (fine_tuning)
  — where the abandoned threads and open follow-ups live, so they don't get silently re-attempted.

The old memory file's "three-claim narrative" block does not appear verbatim in either real
example — treat it as optional, not a required section.

A topic dir that shares a paper-facing "for the paper" doc with siblings (e.g.
`shank3_discrim/PAPER_RESULTS_DRAFT.md`, one level above the individual topic dirs) should have
a `## For the paper` closing section pointing to it — see `results-numbers-doc` for that doc's
own internal structure (headline/table/caveats skeleton, table formatting, how to mark
superseded numbers); that's a different skill from this one, this one only owns getting the
pointer into the README.

For the doc-*tree* level above individual topics (when a family like `general_metrics/` has many
sub-pipelines and free-floating `.md` files), see `general_metrics/README.md` for a different,
complementary pattern: an index that sorts docs into canonical / active / historical / pipeline-
specific, so a reader doesn't have to guess which `.md` is current. That's a doc-index README, not
a topic README — don't conflate the two genres.

## `<TOPIC>_PUB_DIR` / `FIG_DIR` — disambiguate this from CLAUDE.md's `get_output_dir()`

**This is where agents get confused.** The root `CLAUDE.md` says use `get_output_dir()` for
plots/results, and that's still the right call for a standalone script or a topic outside this
family. But `general_metrics/`'s pipelines (`validation_metrics`, `fine_tuning`) — and `drugs/` —
layer a **different, more granular convention on top**: a per-topic env var pair, not
`get_output_dir()`, for their published derive_data caches and rendered figures. Concretely
(`validation_metrics/plots/_common.py:37-56`):

```python
PUB_DIR = Path(os.environ.get(
    "VAL_PUB_DIR",
    "/Volumes/hunter_ssd/busy_bee/plots/validation_metrics_out"
    if not Path("/app/workspace").exists()
    else "/app/workspace/validation_metrics_out",
))
DERIV = PUB_DIR / "derived_data"
FIG_DIR = Path(os.environ.get(
    "VAL_FIG_DIR",
    "/Volumes/hunter_ssd/busy_bee/plots/updated_plots/validation_metrics"
    if not Path("/app/workspace").exists()
    else str(PUB_DIR / "plots"),
))
```

Rule of thumb for a new topic under `general_metrics/`: one env var for **data**
(`<TOPIC>_PUB_DIR`, default an SSD path locally / `/app/workspace/<topic>_out` on a pod — the
`Path("/app/workspace").exists()` pod-detection idiom is the standard test, reuse it verbatim, do
not hardcode a different pod check), plus figures routed through `save_panel(styler, name,
subfolder=...)` into a `FIG_DIR` — which may be **your own** topic's `FIG_DIR`, or, if you're
downstream of an existing hub (see re-export section below), the hub's `FIG_DIR` so all panels
land in one consolidated image tree (this is what `fine_tuning/plots/_common.py:35` does: its
figures render into `VAL_FIG_DIR`, not a separate `FT_FIG_DIR`, even though its *data* stays under
its own `FT_PUB_DIR`).

Not every topic uses this scheme identically — `drugs/` (outside `general_metrics/`) uses one
combined `DRUGS_PUB_DIR` for both data and figures, plus a per-script
`get_pub_dir(script_file) -> DRUGS_PUB_DIR / Path(script_file).stem` helper
(`drugs/plots/_common.py:20-30`), and several of its `derive_data/step*.py` files call
**`get_output_dir()` directly** for large intermediate batteries synced from S3
(`drugs/derive_data/step1_ei_axis_probe.py:273`, `step3c_drug_latent_dynamics.py:143`,
`step6_dv_disinhibition_axis.py:124`, `step14_raw_latent_ei_axis.py:264-265`) — i.e.
`get_output_dir()` and a topic `PUB_DIR` can legitimately coexist in one pipeline, at different
stages. **When in doubt, grep the topic's own existing scripts for which one they already use
before picking** — don't assume `general_metrics/`'s convention applies unmodified outside it, and
don't assume `get_output_dir()` is banned everywhere just because `validation_metrics/` doesn't
use it.

### The default case: one topic, one tree, folder number always = script number

`human_to_mouse/` is the cleanest real example of a topic that ISN'T sharing its tree with
anyone: one `H2M_PUB_DIR`, `FIG_DIR = PUB_DIR / "plots"`, and `save_panel(styler, subfolder,
name="main")` where every caller passes `subfolder` derived from its own filename
(`01_plot_pseudotime_scatter.py` → `plots/01_plot_pseudotime_scatter/`). Standing in an output
folder, the number IS the script — no lookup, no doc table, no ambiguity. **This is the bar every
topic should meet.** When scaffolding a new topic, default to this, not to the hub-consolidation
pattern below — that pattern exists only because `general_metrics/`'s two pipelines render into
one combined image tree for a single paper figure, which is a real but narrow need.

### The hub-consolidation exception — and the collision it creates if done carelessly

When two+ pipelines must render into ONE shared `FIG_DIR` (as `fine_tuning` does into
`validation_metrics`'s tree, so all of Fig 2's panels sit in one browsable folder), a bare
`subfolder=Path(__file__).stem` is no longer enough to guarantee uniqueness: nothing stops
`fine_tuning/plots/01_main_panels.py` and `validation_metrics/plots/23_plot_bin_axis_pearson.py`
(grouped under `01_axes_indist/`) from both presenting a folder that starts with `01` at the
SAME tree level — indistinguishable from outside once both are just subfolders of one `FIG_DIR`.
This actually happened and had to be fixed (2026-07-29): every non-hub pipeline nests one level
deeper, under its own topic name, via a `topic` kwarg on `save_panel`/`save_cross_panel`
(`validation_metrics/plots/_common.py`):

```python
def save_panel(styler, name, ..., subfolder=None, topic=None):
    parts = [p for p in (topic, subfolder) if p]
    target_dir = FIG_DIR.joinpath(*parts) if parts else FIG_DIR
    ...
```

The hub itself (`validation_metrics`) passes `topic=None` (its default) — its own `FIG_DIR` env
var (`VAL_FIG_DIR`) already names it, so its scripts render at the tree root exactly as before,
un-nested. The consolidated-in pipeline's re-export partial-binds its own name instead of
redefining the function (`fine_tuning/plots/_common.py`):

```python
save_panel = functools.partial(_val_common.save_panel, tight_layout=True, topic="fine_tuning")
```

Net result: `<VAL_FIG_DIR>/01_axes_indist/23_plot_bin_axis_pearson/` (validation_metrics, at
root) vs. `<VAL_FIG_DIR>/fine_tuning/01_main_panels/` (fine_tuning, nested) — mirrors the actual
`general_metrics/{validation_metrics,fine_tuning}/` source layout, and two pipelines' numbers can
never collide again. **If you add a third pipeline to an existing hub's tree, give it the same
treatment** — partial-bind its own `topic=` name in its `_common.py` re-export, don't just reuse
the hub's bare `save_panel` and hope its own numbering happens not to collide.

A narrative-beat name (e.g. "the OOD-collapse beat", "beat 4") is a legitimate thing to document
in a paper-facing narrative doc (`FIG2_NARRATIVE.md`'s panel inventory table) — but it must never
be what a folder is NAMED. The folder name is always `<script-stem>` (or `<topic>/<script-stem>`
under a hub); the beat-to-folder mapping is a separate, doc-side lookup, not the routing itself.
Folder names that encode a beat/narrative concept instead of the rendering script's own identity
are the exact defect this section exists to prevent — don't reintroduce it when adding a panel.

## Cross-directory `plots/_common.py` re-export — when to write one vs. reuse a hub's

**Decision rule**: if the new topic reads data that's already served by an existing sibling
dir's loaders — same Styler config, same color maps, same `<TOPIC>_PUB_DIR` family, e.g. the new
pipeline just joins/re-analyzes an existing seedavg or perorg CSV (this is exactly
`predictability_discrim`'s relationship to `fine_tuning`+`validation_metrics` — it reads their
already-built `ladder_seedavg.csv`/`perorg_seedavg.csv` directly) — **prefer re-exporting the
existing hub's `_common.py` over writing a new one from scratch.** This is what
`fine_tuning/plots/_common.py` does with `validation_metrics/plots/_common.py`, and the rationale
is recorded in `general_metrics/CLEANUP_PLAN.md` §7 ("Genuinely duplicated infra... safe to
dedupe": `SPLIT_COLORS` was byte-identical, `get_styler` and most of `save_panel` were too).
**Caveat, also from §7**: the "line-for-line duplicate" claim in that doc turned out to be wrong
about one keyword arg (`save_panel`'s `tight_layout` default) — verify behavioral identity
yourself before deduping, don't just trust a cleanup doc's claim; where behavior genuinely
diverges, `functools.partial` is how you re-export while preserving the difference (see below),
not a fork.

Template (genericize `_TOPIC_` / `hub_topic` for your case;
`fine_tuning/plots/_common.py` is the real file this is lifted from):

```python
"""Thin re-export of <hub_topic>/plots/_common.py -- the shared infra (colors, save_panel,
get_styler, loaders) lives there; this module re-points at it under the names <this_topic>'s
plot scripts already import.
"""
import functools
import importlib.util
import sys
from pathlib import Path

# Load <hub_topic>/plots/_common.py by file path (sibling dir, NOT a package -- both dirs use
# leading-digit-free `_common.py` so a normal sys.path import would collide between the two).
# Walk is repo-structure-relative to the shared PARENT both topics live under -- recompute the
# parents[N] index for THIS file's actual depth, don't copy a number from another example.
# Real values differ: fine_tuning/plots/_common.py:23 (plots/ -> fine_tuning/ -> general_metrics/)
# uses parents[2]; predictability_discrim's step1 (one level deeper, derive_data/ not plots/,
# under shank3_discrim/ not general_metrics/) uses parents[3] to reach general_metrics/
# (shank3_discrim/predictability_discrim/derive_data/step1_join_ladder_genotype.py:19).
_HUB_COMMON_PATH = (Path(__file__).resolve().parents[N] / "<hub_topic>"
                    / "plots" / "_common.py")
_spec = importlib.util.spec_from_file_location("_hub_plots_common", _HUB_COMMON_PATH)
_hub_common = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_hub_plots_common", _hub_common)
_spec.loader.exec_module(_hub_common)

# ---- re-exports this topic's plot scripts already import from `_common` ----
SPLIT_COLORS = _hub_common.SPLIT_COLORS
# The HUB module defines this topic's own data-dir env var (not the other way round): compare
# validation_metrics/plots/_common.py:300 defining FT_PUB_DIR, which fine_tuning/plots/_common.py
# just re-points at below. Add your topic's <THIS_TOPIC>_PUB_DIR to the hub's _common.py, then
# re-export it here -- don't define it locally, that produces a second source of truth.
PUB_DIR = _hub_common.THIS_TOPIC_PUB_DIR
FIG_DIR = _hub_common.FIG_DIR                # figures consolidate into the hub's image tree
load_something = _hub_common.load_something

# Where behavior must diverge slightly, partial-bind rather than redefining or forking:
get_styler = functools.partial(_hub_common.get_styler, rung=True)
save_panel = functools.partial(_hub_common.save_panel, tight_layout=True)
```

If the new topic's data model is genuinely different (new color axis, new file formats, nothing
already loaded by a sibling), write its own `_common.py` from scratch following the skeleton
above — don't force a re-export where there's no real shared infra, that's how you'd get a
`_common.py` littered with unrelated hub internals.

## See also

- `figure-style` — the `Styler` API itself (sizing, color registration, titles, violins,
  significance brackets, panel labels). This skill assumes you already know that; it only says
  which file the plot script lives in and what loads its data.
- `neural-predictor-figures` — this project's specific file:line pointers, `general_metrics`'
  `SPLIT_COLORS`/`RUNG_COLORS` palette, and case studies (acute-mouse-slice cohort bug, the
  electrode_discr Illustrator reference).
