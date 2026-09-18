---
name: neural-predictor-figures
description: BrainDance neural_predictor project-specific figure detail — file:line pointers into proj/predictor/figures/, its semantic color palettes, and case studies (acute-mouse-slice mislabel, the electrode_discr Illustrator reference). Invoke when the task mentions "neural_predictor", "general_metrics", a "fine_tuning card", or a path under proj/predictor/figures/. Companion to the general "figure-style" skill (which owns the actual Styler API/rules) — load both when working on these figures.
argument-hint: [plot script path under proj/predictor/figures/]
---

# neural_predictor figures — project-specific detail

Companion to the general `figure-style` skill — load both when touching a figure under
`proj/predictor/figures/`. That skill owns the actual rules (the `Styler` API, titles, violins,
significance brackets, panel scope, etc.) since `Styler` is BrainDance package-level
infrastructure, not specific to this project. This skill only holds what genuinely doesn't
generalize: which files in `proj/predictor/figures/` do what, this project's specific palettes,
and case studies particular to this data.

## Target venue: Nature Neuroscience — use `journal="nature_neuro"`

This project's paper-facing composite figures (`proj/predictor/figures/final_figures/fig*/`)
target **Nature Neuroscience** (decided 2026-08-13, superseding the Cell Press page size
`busy_drift/build_supp_figure.py` was originally hand-laid-out for). Construct the
`Styler` for any `final_figures/fig*/build_figure.py` (or `build_supp_figure.py`) with
`Styler(journal="nature_neuro")`, and size the page via
`styler.composite_canvas_size(columns=2, depth="max")` — see `figure-style`'s "Composite
pages & journal page sizes" section for the general mechanism
(`braindance/utils/data_manager/utils/plotting/journal_specs.py`). Don't hardcode any mm/pt
conversion directly in a `final_figures` script — that's exactly the per-script
magic-number drift the journal-profile registry exists to prevent.

🚨 Nature Neuroscience is a Nature-**branded research journal**, so its page is **180 mm
max width, 170 mm depth, 88 mm single column** (510.2 × 481.9 pt) — *not* flagship
*Nature*'s 89/183/247. Don't carry 518.7/700.2 pt over from an older doc.

`fig4`'s supplemental lives at `final_figures/fig4_busy_drift/build_supp_figure.py`, on
that page since 2026-08-13; its panels are still imported in place from `busy_drift/plots/`.

## File pointers

- `general_metrics` semantic palette: `general_metrics/validation_metrics/plots/_common.py:57-78`
  (`SPLIT_COLORS`, `RUNG_COLORS`, `TYPE_MARKERS`) — the canonical source for this project.
  `_strat_common.py`'s `_STRAT_PALETTE` and `22_plot_shank3_separability.py`'s `GENO_COLOR` are
  separate, deliberately *not* merged into it (different semantic axes — genotype and per-stratum
  palettes aren't the split palette). Only fix duplication at the source for things that really
  are the same palette.
- Violin-with-custom-x-position pattern (a `seaborn.violinplot` translated onto non-uniform x
  positions, for grouped-column cards with gaps between blocks): see `_card_column` in
  `validation_metrics/plots/_common.py`.
- `electrode_discr` local wrapper: `electrode_discr/plots/elec_helper.py` — same canonical
  `Styler`, its own registered color maps (`"method"`, `"pair_d"`).

## The electrode_discr reference, concretely

`electrode_discr/` (`electrode_disc.png`) is this project's example of the general
matplotlib-skeleton → manual-Illustrator-finishing pattern described in `figure-style`: it uses
the *exact same* `Styler` as every other panel here — its extra polish is a human finishing pass
outside the repo (`elec_helper.py`'s docstring literally says "skeleton for Illustrator"), not
different code or a different styling class.

## Case study: verify a subgroup is real before building plotting logic around it

`fine_tuning/plots/01_main_panels.py` carried an "acute mouse slice" aside block for a while
before anyone checked the catalog against it. It turned out to be neither acute nor mouse-slice by
the catalog's own `sample_type`/`acute_any` fields, and the "train X / val Y, same organoid"
pairing driving it was an `organoid_id` collision between two unrelated recordings, not a real
held-out session. Fix was to drop the 11 organoid_ids outright, not to keep showing them more
carefully. See `spike_sorting_per_recording_no_unit_identity` and
`acute_mouse_slice_id_collision_bug` in project memory for the full writeup — the general lesson
("check source data before trusting a subgroup label") lives in the `figure-style` skill.

## Where the `tight_layout`/gap in `Styler` shows up here

`figure-style` flags that `Styler`'s save wrapper has no shared `tight_layout` toggle. In this
project that shows up as inconsistent hand-called `plt.tight_layout()` in
`general_metrics/validation_metrics/plots/11_organoid_distribution.py:276,304,319` and
`19_plot_sweeps.py:283,335`.

Raw teardown of both style-reference images used to derive these conventions:
`AUDIT_SOURCES.md` in this skill folder.
