# plot_styler.py

**Path:** `braindance/utils/data_manager/utils/plotting/plot_styler.py`
**Module:** `braindance.utils.data_manager.utils.plotting.plot_styler`
**Feature Area:** `Visualization`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Centralizes figure sizing, journal typography, palettes, axis styling, compliance checks, significance annotations, and PNG/SVG export for neural-data plots.

## Connections
- **Used by:** `braindance.utils.data_manager.advanced_tutorials.04_s3_loading_tutorial` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.evoked_response` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.population` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.raster` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.spatial_latency` — import consumer hint; not a proven runtime call.
- **Uses:** `DEFAULT_TEXT_SPEC` from `braindance.utils.data_manager.utils.plotting.journal_specs` — imports (static evidence).
- **Uses:** `JOURNAL_SPECS` from `braindance.utils.data_manager.utils.plotting.journal_specs` — imports (static evidence).
- **Uses:** `JournalPageSpec` from `braindance.utils.data_manager.utils.plotting.journal_specs` — imports (static evidence).
- **Uses:** `ROLE_FIELDS` from `braindance.utils.data_manager.utils.plotting.journal_specs` — imports (static evidence).
- **Uses:** `get_journal_spec` from `braindance.utils.data_manager.utils.plotting.journal_specs` — imports (static evidence).
- **Shared data:** Used by PlotAccessor and plotting helpers; public data_manager reexports Styler.

## Dependencies
- `braindance.utils.data_manager.utils.plotting.journal_specs.DEFAULT_TEXT_SPEC` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.journal_specs.JOURNAL_SPECS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.journal_specs.JournalPageSpec` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.journal_specs.ROLE_FIELDS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.journal_specs.get_journal_spec` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.font_manager` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `matplotlib.transforms.blended_transform_factory` — external or unresolved local import; source import evidence.
- `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scienceplots` — external or unresolved local import; source import evidence.

## Classes
### Styler()
> Applies journal-aware figure sizing, typography, palettes, styling, compliance checks, annotation helpers, and export.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:42`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/advanced_tutorials/04_s3_loading_tutorial.py:93 (named-call hint); braindance/utils/data_manager/utils/plotting/accessor.py:46 (named-call hint); braindance/utils/data_manager/utils/plotting/evoked_response.py:246 (named-call hint); braindance/utils/data_manager/utils/plotting/evoked_response.py:79 (named-call hint); braindance/utils/data_manager/utils/plotting/plot_styler.py:1183 (named-call hint); braindance/utils/data_manager/utils/plotting/population.py:177 (named-call hint); braindance/utils/data_manager/utils/plotting/population.py:84 (named-call hint); braindance/utils/data_manager/utils/plotting/raster.py:134 (named-call hint); braindance/utils/data_manager/utils/plotting/spatial_latency.py:167 (named-call hint)
**Constructor:** `__init__(self, journal: Union[str, JournalPageSpec], **role_overrides: float)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_custom_color_maps` | Dict[str, Dict[str, str]] | `{}` |
| `colors` | inferred at runtime | `['#AD5E99', '#009E73', '#F0E442', '#56B4E9', '#E69F00', '#0072B2', '#CC79A7', '#D55E00', '#000000', '#999999', '#00743C', '#9B238…` |
| `composite_hspace` | inferred at runtime | `0.35` |
| `composite_letter_fontsize` | inferred at runtime | `self.journal.panel_letter_pt` |
| `composite_wspace` | inferred at runtime | `0.3` |
| `default_sizes` | inferred at runtime | `{'single': {'width': 277.5191, 'height': 223.7852}, '1.5': {'width': 448.8, 'height': 336.6}, 'double': {'width': 685.0, 'height'…` |
| `heatmap_cmaps` | inferred at runtime | `{'sequential': 'viridis', 'diverging': 'RdBu_r', 'temporal': 'Blues', 'intensity': 'inferno'}` |
| `journal` | JournalPageSpec | `spec` |
| `named_colors` | inferred at runtime | `{'purple': self.colors[0], 'green': self.colors[1], 'yellow': self.colors[2], 'light_blue': self.colors[3], 'orange': self.colors…` |
**Methods:**
#### `set_style(self)`
> Apply the scientific plot style.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:162`
#### `configure_svg_settings(self)`
> Configure settings for SVG/PDF export with editable text.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:247`
#### `set_color_palette(self)`
> Set up the default colorblind-friendly color palette.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:264`
#### `get_color(self, index: int) -> str`
> Get color by index from the palette.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:305`
#### `get_named_color(self, name: str) -> str`
> Get color by name.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:323`
#### `register_color_map(self, name: str, color_dict: Dict[str, str])`
> Register a custom color mapping for specific categories.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:339`
#### `get_category_color(self, map_name: str, category: str) -> str`
> Get color for a category from a registered color map.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:360`
#### `get_heatmap_cmap(self, style: str='sequential') -> str`
> Get a colormap name for heatmaps.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:379`
#### `points_to_inches(self, points: float) -> float`
> Convert points to inches (72 points = 1 inch).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:395`
#### `get_figure_size(self, width_pt: Optional[float]=None, height_pt: Optional[float]=None, size_preset: Optional[str]=None) -> Tuple[float, float]`
> Get figure size in inches.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:399`
#### `create_figure(self, nrows: int=1, ncols: int=1, width_pt: Optional[float]=None, height_pt: Optional[float]=None, size_preset: Optional[str]=None, **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]`
> Create a figure with standardized dimensions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:434`
#### `composite_canvas_size(self, columns: int=2, depth: Union[str, float]='max') -> Tuple[float, float]`
> Resolve a (width_pt, height_pt) pair from this Styler's `journal` profile, for handing straight to `create_composite_figure(width_pt, height_pt, ...)`. Requires `journal=...` to have been passed to `Styler(...)` at construction -- this is a lookup into that profile, not a new size convention of its own.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:477`
#### `create_composite_figure(self, width_pt: float, height_pt: float, nrows: int=1, ncols: int=1, height_ratios: Optional[List[float]]=None, width_ratios: Optional[List[float]]=None, **gridspec_kwargs) -> Tuple[plt.Figure, 'mpl.gridspec.GridSpec']`
> Creates a full-page multi-panel figure using one outer GridSpec and the resolved journal geometry.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:525`
#### `check_journal_compliance(self, fig=None, verbose: bool=True) -> List[str]`
> Checks all text sizes against the resolved profile and checks page geometry when the profile has a page; panel-letter exemptions are identity-marked.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:616`
#### `finish_plot(self, save_plots: bool=False, save_dir: Optional[str]=None, name: str='figure', dpi: int=300, formats: List[str]=['png', 'svg'], show: bool=True, exact_size: bool=False, tight_layout: bool=True)`
> Finalize and optionally save the current plot.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:703`
#### `plot_color_palette(self) -> plt.Figure`
> Display all available colors in the palette. Returns ------- fig : matplotlib.figure.Figure Figure showing the color palette.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:828`
#### `style_raster(self, ax: plt.Axes, spine_color: str='#333333')`
> Apply styling suitable for raster plots.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:855`
#### `style_heatmap(self, ax: plt.Axes)`
> Apply styling suitable for heatmaps.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:861`
#### `add_scalebar(self, ax: plt.Axes, length: float, label: str, loc: str='lower right', color: str='black', fontsize: Optional[float]=None)`
> Add a scale bar to an axes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:867`
#### `add_significance_bracket(self, ax: plt.Axes, x1: float, x2: float, y: float, label: str, *, height: float=None, color: str='black', linewidth: float=0.8, linestyle: str='-', fontsize: Optional[float]=None)`
> Draw a pairwise significance bracket (caliper shape: short end-ticks, label centered above the horizontal bar) between two x positions. For multiple comparisons on one axes, call this once per pair with increasing `y` so the brackets stack rather than overlap.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:917`
#### `add_significance_label(self, ax: plt.Axes, x: float, label: str, *, y: float=0.985, transform=None, fontsize: Optional[float]=None, fontweight: str='bold', color: str='0.3', ha: str='center', va: str='top')`
> Adds a profile-sized bold significance annotation at an x position.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:977`
#### `label_panel(self, ax: plt.Axes, letter: str, *, loc: tuple=(-0.12, 1.05), fontsize: Optional[float]=None, fontweight: str='bold')`
> Add a bold panel letter (A, B, C, ...) at the top-left of an axes, in the standard multi-panel-figure convention.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:1043`
#### `mark_panel_letter(self, artist)`
> Marks a hand-placed text artist as a panel letter for identity-based compliance exemption.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:1083`
#### `add_group_title(self, fig: plt.Figure, axes, text: str, *, gap: float=0.028, fontsize: Optional[float]=None, fontweight: str='bold', zorder: int=10)`
> Adds a profile-sized title spanning a group of axes after layout placement.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:1104`

## Functions
### `main()`
> Test the Styler with example plots.
> **Called by:** braindance/utils/data_manager/utils/plotting/plot_styler.py:1209 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/plot_styler.py:1181`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `Styler(journal=...)` requires a journal profile; `draft` is the non-paper profile. Role sizes can be overridden at construction. |
| Strict journal profiles write files then raise ValueError on compliance violations; draft warns. |
| The module test entry point uses `Styler(journal="draft")`; library callers must choose a journal profile at construction. |

## Data Shapes
- create_figure→(Figure,Axes or array of Axes); get_figure_size→inches tuple.

## Notes
- Requires scienceplots; initialization mutates global Matplotlib rcParams.
- finish_plot saves and closes current pyplot figure; saving path does not honor show=True afterward.
- Panel letters are marked by artist identity for compliance exemptions; significance labels, brackets, and group titles use profile annotation/title sizes.
- Composite canvas sizing and compliance use JournalPageSpec geometry; scale bars and annotation helpers centralize publication styling.
