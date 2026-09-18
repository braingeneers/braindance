"""
Standardized plotting utilities for BrainDance.

Provides consistent, publication-ready matplotlib styling with:
- Scientific plot aesthetics (requires scienceplots package)
- Colorblind-friendly color palette
- Configurable figure sizes (single/double column)
- SVG/PDF export with editable text

Usage:
    from braindance.utils.data_manager import Styler

    # `journal` is REQUIRED -- it sets the text sizes, not just the page. Use "draft"
    # for exploratory/QC work, a real profile for anything headed to a paper.
    styler = Styler(journal="draft")
    fig, ax = styler.create_figure(size_preset='single')
    ax.plot(x, y)
    styler.finish_plot(save_plots=True, save_dir='./figs', name='my_plot')

Requirements:
    pip install scienceplots
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import blended_transform_factory
import pathlib
import numpy as np
from typing import Optional, Tuple, Dict, List, Union

from dataclasses import replace

#: Marker set on the text artist `label_panel` creates, so the compliance check can
#: exempt panel letters by identity rather than by their point size. See `label_panel`.
PANEL_LETTER_GID = "braindance-panel-letter"

from braindance.utils.data_manager.utils.plotting.journal_specs import (
    DEFAULT_TEXT_SPEC, JOURNAL_SPECS, ROLE_FIELDS, JournalPageSpec, get_journal_spec,
)


class Styler:
    """
    Standardized plotting style manager for publication-ready figures.
    
    Provides consistent styling across all plots with:
    - Science-style formatting (requires scienceplots)
    - Colorblind-friendly palette
    - Configurable figure dimensions
    - SVG/PDF export with editable text
    
    Example:
        >>> styler = Styler(journal="draft")   # or "nature_neuro" for a paper figure
        >>> fig, ax = styler.create_figure(size_preset='single')
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> styler.finish_plot(save_plots=True, save_dir='./figs', name='example')
    """
    
    def __init__(self, journal: Union[str, JournalPageSpec], **role_overrides: float):
        """
        Initialize the Styler against a required house-style profile.

        Applies science-style formatting using the scienceplots package.

        Parameters
        ----------
        journal : str or JournalPageSpec
            **Required.** Name of a registered profile (`journal_specs.JOURNAL_SPECS`:
            `"nature_neuro"`, `"nature"`, `"draft"`), or a `JournalPageSpec` instance
            for a one-off. Resolves onto `self.journal` and drives three things: the
            page size returned by `composite_canvas_size()`, the per-role text sizes
            pushed into `rcParams` by `set_style()`, and what
            `check_journal_compliance()` measures against.

            🚨 There is deliberately NO DEFAULT (2026-08-13 PI directive). Until then
            the argument was optional and 165 of this repo's 172 call sites passed
            nothing, so the house style in force was never written down anywhere a
            reader (or an agent) could see it. Non-paper work -- scratch analysis, QC,
            the GUI, `braindance`'s own plotting helpers -- passes `journal="draft"`,
            which warns instead of raising and has no page geometry. Passing `None`
            raises rather than silently reverting to the old behavior.
        **role_overrides : float
            Optional per-role text-size overrides applied to the resolved profile
            (`title_pt`, `label_pt`, `tick_pt`, `legend_pt`, `annotation_pt`) -- e.g.
            `Styler(journal="nature_neuro", tick_pt=5)` for a panel whose tick labels
            genuinely do not fit at 6.

            ⛔ This is the ONLY sanctioned way to change a text size. It exists so that
            "the panel needs smaller ticks" is expressed once, at construction, where it
            is visible and greppable -- instead of as a `fontsize=` on an artist, which
            is what the house style is trying to eliminate. Overrides are validated
            against the profile's `[min_text_pt, max_text_pt]` band, so this cannot be
            used to smuggle an out-of-spec size past the compliance check.

        Raises
        ------
        TypeError
            If `journal` is omitted, or `None`, or an override names a field that is not
            a role size.
        ValueError
            If an override falls outside the profile's legal text band.
        """
        self._custom_color_maps: Dict[str, Dict[str, str]] = {}

        if journal is None:
            raise TypeError(
                "Styler(journal=...) is required and cannot be None. Pass a registered "
                f"profile ({', '.join(sorted(JOURNAL_SPECS))}) -- 'draft' is the right "
                "one for exploratory/non-paper figures. A journal profile sets the text "
                "sizes, not just the page, so there is no meaningful 'no profile' mode."
            )
        spec = get_journal_spec(journal)

        if role_overrides:
            unknown = set(role_overrides) - set(ROLE_FIELDS)
            if unknown:
                raise TypeError(
                    f"Styler() got unexpected keyword argument(s) {sorted(unknown)}. "
                    f"Only role sizes may be overridden: {list(ROLE_FIELDS)}."
                )
            for field, value in role_overrides.items():
                if not spec.min_text_pt <= float(value) <= spec.max_text_pt:
                    raise ValueError(
                        f"{field}={value} is outside {spec.name}'s legal text band "
                        f"{spec.min_text_pt}-{spec.max_text_pt} pt. An override may "
                        "retune the hierarchy inside the spec; it may not leave it."
                    )
            spec = replace(spec, **{k: float(v) for k, v in role_overrides.items()})

        #: The resolved profile. Always a spec now -- never `None` -- so downstream code
        #: reads sizes from it unconditionally; only PAGE questions consult `has_page`.
        self.journal: JournalPageSpec = spec

        self.set_style()
        self.set_color_palette()
        self.configure_svg_settings()
        
        # Default figure sizes in points (72 points = 1 inch)
        # These correspond to common journal column widths
        self.default_sizes = {
            'single': {'width': 277.5191, 'height': 223.7852},  # ~3.85" x 3.1"
            '1.5': {'width': 448.8, 'height': 336.6},           # ~6.2" x 4.7"
            'double': {'width': 685.0, 'height': 514.0}         # ~9.5" x 7.1"
        }

        # Default spacing for `create_composite_figure` -- so a NEW composite script
        # gets a reasonable row/column gap for free, instead of every script guessing
        # its own `hspace`/`wspace` by eye. Tuned 2026-08-12 against
        # `proj/predictor/figures/busy_drift/build_supp_figure.py`'s 3-row page: the
        # original ad hoc guess (hspace=0.7) left roughly 40% of the page as dead
        # vertical whitespace between rows. Still fully overridable per call --
        # `create_composite_figure(..., hspace=..., wspace=...)` wins if passed.
        self.composite_hspace = 0.35
        self.composite_wspace = 0.3
        # `label_panel` fontsize convention for a composite page's a/b/c... letters --
        # its own default (14) is tuned for a single-panel figure's outer label, too
        # large for a dense multi-panel grid (per 2026-08-12 PI direction).
        # Comes from the profile (8 pt lowercase for Nature -- see
        # `journal.panel_letter_case`; 12 for "draft"), never a literal here.
        self.composite_letter_fontsize = self.journal.panel_letter_pt

    def set_style(self):
        """Apply the scientific plot style."""
        try:
            import scienceplots  # Must import to register styles with matplotlib
            plt.style.use(['science'])
        except ImportError:
            raise ImportError(
                "scienceplots package is required but not installed.\n"
                "Install with: pip install scienceplots"
            )
        except OSError:
            raise ImportError(
                "scienceplots package is installed but 'science' style not found.\n"
                "Try reinstalling: pip install --upgrade scienceplots"
            )
        
        # 🚨 EVERY text size below comes from the resolved profile -- none is a literal.
        # Until 2026-08-13 all seven were hardcoded to 7, the top of Nature's 5-7 pt band,
        # which made the band a single point: a script had no legal way to make a legend
        # key quieter than an axis label except a call-site `fontsize=`. That is where the
        # ~26 hand-written `fontsize=5.5` legends came from, and why deleting them (PI
        # directive, same day) dropped legend boxes onto the data. The hierarchy lives in
        # `journal_specs.TextSpec` now; see its role-size block. ⛔ Do not reintroduce a
        # literal here -- a figure that needs a different size passes the size to
        # `Styler(journal=..., tick_pt=...)`, which is validated against the band.
        spec = self.journal
        plt.rcParams.update({
            'font.family': 'Arial',
            # Backs every artist with no more specific rcParam: bare `ax.text`,
            # significance-bracket stars, `n=` counts, colorbar tick labels.
            'font.size': spec.annotation_pt,
            'axes.linewidth': 0.5,
            'axes.labelsize': spec.label_pt,
            # House style: titles are always bold (figure-style skill) -- set
            # once here so scripts don't each need fontweight='bold'.
            'axes.titleweight': 'bold',
            # Set explicitly (not left at matplotlib's `titlesize='large'`, which is
            # 1.2 x font.size and would float above the ceiling as font.size changes).
            # Same for `figure.titlesize` (suptitle), also 'large' by default.
            'axes.titlesize': spec.title_pt,
            'figure.titlesize': spec.title_pt,
            'xtick.labelsize': spec.tick_pt,
            'ytick.labelsize': spec.tick_pt,
            'legend.fontsize': spec.legend_pt,
            # House style: legend box is boxed, not floating -- square corners,
            # transparent fill (so it never occludes data), thin near-black
            # edge matching the axes spine weight.
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.edgecolor': '#231f20',
            'legend.facecolor': 'none',
            'legend.framealpha': 1,
            'patch.linewidth': 0.5,
            'lines.linewidth': 0.5,
            'lines.markersize': 1,
            # Hide ticks on top and right
            'xtick.top': False,
            'ytick.right': False,
            # Make ticks point outward
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            # Disable minor ticks
            'xtick.minor.visible': False,
            'ytick.minor.visible': False,
            'axes.prop_cycle': plt.cycler('color', [
                '#AD5E99',  # Purple
                '#009E73',  # Green
                '#F0E442',  # Yellow
                '#56B4E9',  # Light Blue
                '#E69F00',  # Orange
                '#0072B2',  # Dark Blue
                '#CC79A7',  # Pink
                '#D55E00',  # Dark Orange
                '#000000',  # Black
                '#999999',  # Gray
                '#00743C',  # Forest Green
                '#9B2385'   # Magenta
            ])
        })
        
        # Set spine visibility
        mpl.rc('axes.spines', right=False, top=False)
        mpl.rc('axes', linewidth=0.5)

    def configure_svg_settings(self):
        """Configure settings for SVG/PDF export with editable text."""
        svg_settings = {
            'svg.fonttype': 'none',      # Text as text, not paths
            'svg.image_inline': True,
            'svg.hashsalt': None,
            'text.usetex': False,
            'mathtext.fontset': 'dejavusans',
            'pdf.fonttype': 42,          # TrueType fonts in PDF
            'ps.fonttype': 42,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05
        }
        mpl.rcParams.update(svg_settings)

    def set_color_palette(self):
        """Set up the default colorblind-friendly color palette."""
        self.colors = [
            '#AD5E99',  # Purple
            '#009E73',  # Green (colorblind-safe)
            '#F0E442',  # Yellow
            '#56B4E9',  # Light Blue (colorblind-safe)
            '#E69F00',  # Orange (colorblind-safe)
            '#0072B2',  # Dark Blue (colorblind-safe)
            '#CC79A7',  # Pink
            '#D55E00',  # Dark Orange/Red (colorblind-safe)
            '#000000',  # Black
            '#999999',  # Gray
            '#00743C',  # Forest Green
            '#9B2385'   # Magenta
        ]
        
        # Named color shortcuts
        self.named_colors = {
            'purple': self.colors[0],
            'green': self.colors[1],
            'yellow': self.colors[2],
            'light_blue': self.colors[3],
            'orange': self.colors[4],
            'dark_blue': self.colors[5],
            'pink': self.colors[6],
            'red': self.colors[7],
            'black': self.colors[8],
            'gray': self.colors[9],
            'forest_green': self.colors[10],
            'magenta': self.colors[11]
        }
        
        # Default heatmap colormaps
        self.heatmap_cmaps = {
            'sequential': 'viridis',
            'diverging': 'RdBu_r',
            'temporal': 'Blues',
            'intensity': 'inferno'
        }

    def get_color(self, index: int) -> str:
        """
        Get color by index from the palette.
        
        Parameters
        ----------
        index : int
            Index into the color palette (0-11)
            
        Returns
        -------
        str
            Hex color code, or black if index out of range
        """
        if 0 <= index < len(self.colors):
            return self.colors[index]
        return '#000000'

    def get_named_color(self, name: str) -> str:
        """
        Get color by name.
        
        Parameters
        ----------
        name : str
            Color name (e.g., 'purple', 'green', 'light_blue')
            
        Returns
        -------
        str
            Hex color code, or black if name not found
        """
        return self.named_colors.get(name, '#000000')

    def register_color_map(self, name: str, color_dict: Dict[str, str]):
        """
        Register a custom color mapping for specific categories.
        
        Parameters
        ----------
        name : str
            Name for this color map (e.g., 'conditions', 'cell_types')
        color_dict : dict
            Mapping of category names to hex color codes
            
        Example
        -------
        >>> styler.register_color_map('conditions', {
        ...     'control': '#000000',
        ...     'treatment': '#E69F00'
        ... })
        >>> color = styler.get_category_color('conditions', 'treatment')
        """
        self._custom_color_maps[name] = color_dict

    def get_category_color(self, map_name: str, category: str) -> str:
        """
        Get color for a category from a registered color map.
        
        Parameters
        ----------
        map_name : str
            Name of the registered color map
        category : str
            Category name within the map
            
        Returns
        -------
        str
            Hex color code, or black if not found
        """
        color_map = self._custom_color_maps.get(map_name, {})
        return color_map.get(category, '#000000')

    def get_heatmap_cmap(self, style: str = 'sequential') -> str:
        """
        Get a colormap name for heatmaps.
        
        Parameters
        ----------
        style : str
            One of 'sequential', 'diverging', 'temporal', 'intensity'
            
        Returns
        -------
        str
            Matplotlib colormap name
        """
        return self.heatmap_cmaps.get(style, 'viridis')

    def points_to_inches(self, points: float) -> float:
        """Convert points to inches (72 points = 1 inch)."""
        return points / 72.0

    def get_figure_size(
        self, 
        width_pt: Optional[float] = None, 
        height_pt: Optional[float] = None, 
        size_preset: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Get figure size in inches.
        
        Parameters
        ----------
        width_pt : float, optional
            Width in points
        height_pt : float, optional
            Height in points
        size_preset : str, optional
            One of 'single', '1.5', 'double' for preset journal sizes
            
        Returns
        -------
        tuple
            (width in inches, height in inches)
        """
        if width_pt is not None and height_pt is not None:
            return (self.points_to_inches(width_pt), self.points_to_inches(height_pt))
        elif size_preset is not None:
            preset = self.default_sizes.get(size_preset)
            if preset is None:
                raise ValueError(f"Unknown size preset: {size_preset}. Use 'single', '1.5', or 'double'")
            return (self.points_to_inches(preset['width']), self.points_to_inches(preset['height']))
        else:
            # Default to single column size
            preset = self.default_sizes['single']
            return (self.points_to_inches(preset['width']), self.points_to_inches(preset['height']))

    def create_figure(
        self, 
        nrows: int = 1, 
        ncols: int = 1, 
        width_pt: Optional[float] = None, 
        height_pt: Optional[float] = None, 
        size_preset: Optional[str] = None, 
        **kwargs
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a figure with standardized dimensions.
        
        Parameters
        ----------
        nrows : int, default=1
            Number of subplot rows
        ncols : int, default=1
            Number of subplot columns
        width_pt : float, optional
            Width in points (overrides preset)
        height_pt : float, optional
            Height in points (overrides preset)
        size_preset : str, optional
            One of 'single', '1.5', 'double' for preset sizes
        **kwargs
            Additional arguments passed to plt.subplots
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes or ndarray of Axes
            The axes object(s)
            
        Example
        -------
        >>> fig, ax = styler.create_figure(size_preset='single')
        >>> fig, axes = styler.create_figure(nrows=2, ncols=2, size_preset='double')
        """
        figsize = self.get_figure_size(width_pt, height_pt, size_preset)
        fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, **kwargs)
        return fig, ax

    def composite_canvas_size(
        self,
        columns: int = 2,
        depth: Union[str, float] = "max",
    ) -> Tuple[float, float]:
        """
        Resolve a (width_pt, height_pt) pair from this Styler's `journal` profile, for
        handing straight to `create_composite_figure(width_pt, height_pt, ...)`.

        Requires `journal=...` to have been passed to `Styler(...)` at construction --
        this is a lookup into that profile, not a new size convention of its own.

        Parameters
        ----------
        columns : {1, 2}, default=2
            1 -> `journal.single_col_pt`, 2 -> `journal.double_col_pt`.
        depth : "max" or float, default="max"
            "max" -> `journal.max_depth_pt` (the publisher's page-depth ceiling).
            Pass an explicit point value instead when the composite has few enough
            rows that the max depth would stretch it -- especially when saving with
            `finish_plot(..., exact_size=True)`, which most composites in this repo
            do: that flag ships the canvas exactly as requested, so there is no
            later "crop in Illustrator" step to rely on, and a 1-2 row composite
            built at max depth renders with its axes inflated to fill the full page
            rather than with blank space to trim. Reserve "max" for composites with
            enough rows to actually use the page.
            🚨 For the Nature profiles this ceiling is a real editorial limit, not a
            page size: 170 mm exists "to allow space for the figure legend to fit
            underneath", so a page built taller cannot be fixed by cropping margins.

        Returns
        -------
        (width_pt, height_pt)
        """
        if not self.journal.has_page:
            raise ValueError(
                f"composite_canvas_size() needs a profile with a real publisher page; "
                f"{self.journal.name!r} has none. Construct with "
                "Styler(journal=\"nature_neuro\") (or another registered name / a "
                "JournalPageSpec instance), or call create_composite_figure(...) "
                "directly with explicit width_pt/height_pt."
            )
        if columns not in (1, 2):
            raise ValueError(f"columns must be 1 or 2, got {columns!r}")
        width_pt = self.journal.single_col_pt if columns == 1 else self.journal.double_col_pt
        height_pt = self.journal.max_depth_pt if depth == "max" else float(depth)
        return width_pt, height_pt

    def create_composite_figure(
        self,
        width_pt: float,
        height_pt: float,
        nrows: int = 1,
        ncols: int = 1,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        **gridspec_kwargs
    ) -> Tuple[plt.Figure, "mpl.gridspec.GridSpec"]:
        """
        Create a full-page composite figure -- a multi-panel figure (a paper's
        main or supplemental page) drawn as ONE matplotlib canvas with one
        outer GridSpec, rather than several independently-rendered panels
        spliced together afterward.

        This is the composite-figure analogue of `create_figure`: use
        `create_figure` for a single panel meant to stand alone. Use this one
        for the composite page itself, including pages with reserved
        diagram/placeholder slots that have nothing to draw yet -- a
        placeholder is just an empty axes (or an axes-fraction `Rectangle` +
        centered `text`) carved from this same grid, not a reason to reach
        for a separate SVG-splicing tool (`FigureComposer`, now legacy --
        2026-08-12, see its module docstring). Every panel should have a
        `draw_*(ax, ...)` function to call directly onto axes carved from
        this grid -- see `proj/predictor/figures/busy_drift/build_supp_figure.py`
        for the reference pattern this generalizes. Refactor each source
        panel script so its plotting logic lives in a `draw_*(ax, ...)`
        function separate from data loading and its own `save_panel` call,
        then call that same function here on an axes from this grid -- one
        drawing implementation serves both the standalone panel and the
        composite. For a panel that needs whole-figure-scoped layout calls
        of its own (`fig.subplots_adjust(...)`, `fig.tight_layout()`) when
        rendered standalone, give it a reserved fixed-size row/column in this
        grid instead (e.g. a dedicated legend row sized in the subgridspec)
        rather than letting it call figure-global layout methods against a
        `fig` that other panels also share -- those calls affect the entire
        canvas, not just the caller's own cell.

        Give each row/column its own further nesting with
        `outer[i, j].subgridspec(...)` (repeatable -- a cell holding several
        sub-axes, e.g. two `sharey` panels, nests one level deeper again) --
        this method only owns the outer grid.

        Parameters
        ----------
        width_pt, height_pt : float
            Canvas size in points (1 pt = 1/72 inch), e.g. 612x792 for a US
            Letter portrait page. Save with
            `finish_plot(..., exact_size=True, tight_layout=False)` once the
            layout is pinned, so the saved canvas matches this exactly rather
            than tight-cropping to content.
        nrows, ncols : int, default=1
            Outer grid shape.
        height_ratios, width_ratios : list of float, optional
            Forwarded to `fig.add_gridspec`.
        **gridspec_kwargs
            Additional arguments forwarded to `fig.add_gridspec`, e.g.
            `hspace`, `wspace`. If omitted, both default to
            `self.composite_hspace`/`self.composite_wspace` (0.35/0.3) --
            override here for the OUTER grid only if this page's rows/columns
            genuinely need more or less room than that; a nested
            `.subgridspec(...)` call still needs its own `hspace`/`wspace`
            passed explicitly (it does not inherit these).

        Returns
        -------
        fig : matplotlib.figure.Figure
        outer : matplotlib.gridspec.GridSpec
            Index it (`outer[i, j]`) and call `.subgridspec(...)` on the
            result for nested layouts, or `fig.add_subplot(outer[i, j])`
            directly for a plain cell.

        Example
        -------
        >>> fig, outer = styler.create_composite_figure(
        ...     612, 792, nrows=3, height_ratios=[0.93, 1.0, 0.89])  # hspace defaults to 0.35
        >>> row1 = outer[0].subgridspec(1, 2, width_ratios=[1.17, 1.0], wspace=0.32)
        >>> ax_a = fig.add_subplot(row1[0, 0])
        >>> some_panel_module.draw(ax_a, styler, data)
        >>> styler.label_panel(ax_a, 'A', fontsize=12)
        """
        gridspec_kwargs.setdefault('hspace', self.composite_hspace)
        gridspec_kwargs.setdefault('wspace', self.composite_wspace)
        figsize = (width_pt / 72, height_pt / 72)
        fig = plt.figure(figsize=figsize)
        outer = fig.add_gridspec(nrows=nrows, ncols=ncols,
                                  height_ratios=height_ratios,
                                  width_ratios=width_ratios, **gridspec_kwargs)
        return fig, outer

    def check_journal_compliance(self, fig=None, verbose: bool = True) -> List[str]:
        """Check a figure against the text house style and this Styler's journal profile.

        Two checks with deliberately different scopes:

        * **Text sizes -- ALWAYS checked**, against `[min_text_pt, max_text_pt]` of the
          resolved profile. Text size does not depend on the canvas, and a standalone
          panel's `draw_*` function is exactly what a composite later calls at page size,
          so an oversized label caught here is caught before it reaches the page.
        * **Page geometry -- only when the profile HAS a page** (`has_page`). A panel is
          not a page, and `"draft"` has no publisher behind it at all.

        Everything else in a journal's guide (scale bars rather than magnification
        factors, error bars present and defined, editable rather than outlined text,
        Symbol font for Greek) is a judgement call this cannot make, so a clean result
        here is NOT a certificate of compliance. 🚨 Neither is it evidence of a readable
        page: this enumerates `Text` artists and measures their POINT SIZE. It cannot see
        a legend box sitting on a data point, and exactly that shipped in four cartpole
        panels under a silent check. Read the render.

        Text at exactly `panel_letter_pt` is treated as a panel letter and exempted from
        the text-size ceiling, since the guides carve those out explicitly ("8 pt bold,
        upright (not italic) a, b, c's" vs "maximum text size for all other text: 7 pt").

        This function itself only ever RETURNS violations -- `finish_plot()` owns the
        consequence, and on a `strict` profile that consequence is an exception AFTER the
        file is written (2026-08-13 PI directive). A printed warning was ignorable and got
        ignored; 220 out-of-spec labels accumulated under one. Writing first means the
        offending render is still on disk to look at, which is the only reason the old
        warn-don't-raise behavior was defensible.

        ⛔ A size that is *in* band is invisible here -- `fontsize=6` passes. The rule that
        a plot script should name NO size at all is a static one; see
        `proj/predictor/figures/check_no_fontsize.py`.
        """
        fig = fig if fig is not None else plt.gcf()
        spec = self.journal
        label = spec.name
        bad_sizes: Dict[float, str] = {}
        for t in fig.findobj(mpl.text.Text):
            if not t.get_text().strip():
                continue
            # 🚨 EXEMPT BY IDENTITY, NOT BY SIZE. A panel letter is exempt because it IS
            # one -- `label_panel` stamps this gid on the artist it creates. Until
            # 2026-08-13 the test was `size == spec.panel_letter_pt`, i.e. "anything that
            # happens to be 8.0 pt is a panel letter", and that is a hole you can drive a
            # figure through: `add_group_title`'s old `fontsize=8` default made every
            # group heading inherit the carve-out by coincidence of number, so
            # out-of-spec body text shipped under a green check. Nature's wording is
            # "maximum text size for all other text: 7 pt" -- a group heading is other
            # text, and it earns its prominence from weight and spacing instead.
            # ⛔ Do NOT restore a size-based fallback for hand-drawn letters. A letter
            # drawn with a bare `ax.text(..., fontsize=8)` SHOULD fail this check: it is
            # also invisible to `check_no_fontsize.py`'s static sweep only because that
            # sweep exempts `label_panel` call sites, so the runtime check is the only
            # thing standing between a hand-rolled letter and the page. Route it through
            # `label_panel`, which is the one place that knows the journal's number.
            if t.get_gid() == PANEL_LETTER_GID:
                continue
            size = round(float(t.get_fontsize()), 2)
            if spec.min_text_pt <= size <= spec.max_text_pt:
                continue
            bad_sizes.setdefault(size, t.get_text()[:40])
        violations = [
            f"text at {size} pt (e.g. {sample!r}) -- outside "
            f"{spec.min_text_pt}-{spec.max_text_pt} pt"
            for size, sample in sorted(bad_sizes.items())
        ]
        # Geometry only when a page was actually declared -- see the docstring.
        if spec.has_page:
            w_pt, h_pt = (v * 72 for v in fig.get_size_inches())
            if w_pt > spec.double_col_pt + 0.5:
                violations.append(
                    f"canvas {w_pt:.0f} pt ({w_pt * 25.4 / 72:.0f} mm) wider than "
                    f"{spec.name}'s {spec.double_col_pt:.0f} pt "
                    f"({spec.double_col_pt * 25.4 / 72:.0f} mm) maximum")
            if h_pt > spec.max_depth_pt + 0.5:
                violations.append(
                    f"canvas {h_pt:.0f} pt ({h_pt * 25.4 / 72:.0f} mm) deeper than "
                    f"{spec.name}'s {spec.max_depth_pt:.0f} pt "
                    f"({spec.max_depth_pt * 25.4 / 72:.0f} mm) maximum")
        if violations and verbose:
            print(f"[Styler] ⚠️ {label} compliance: {len(violations)} issue(s)")
            for v in violations:
                print(f"[Styler]    - {v}")
        return violations

    def finish_plot(
        self,
        save_plots: bool = False,
        save_dir: Optional[str] = None,
        name: str = 'figure',
        dpi: int = 300,
        formats: List[str] = ['png', 'svg'],
        show: bool = True,
        exact_size: bool = False,
        tight_layout: bool = True
    ):
        """
        Finalize and optionally save the current plot.

        Parameters
        ----------
        save_plots : bool, default=False
            Whether to save the plot to files
        save_dir : str, optional
            Directory to save plots (created if doesn't exist)
        name : str, default='figure'
            Base filename (without extension)
        dpi : int, default=300
            Resolution for raster formats
        formats : list of str, default=['png', 'svg']
            File formats to save (e.g., ['png', 'svg', 'pdf'])
        show : bool, default=True
            Whether to show the plot if not saving
        exact_size : bool, default=False
            When False (default), saves with `bbox_inches='tight'`, which crops
            to each panel's own content bounding box -- two panels created with
            the identical `width_pt`/`height_pt` can still land at different
            final pixel dimensions if their titles/labels/annotations have
            different footprints. Set True to save at exactly the
            `width_pt`/`height_pt` canvas instead (via `tight_layout()` +
            `bbox_inches=None`), so panels meant to sit side-by-side in a
            multi-panel composition (Illustrator, a paper figure) come out
            pixel-identical for a given size. Content that doesn't fit the
            requested canvas gets clipped rather than expanding it -- widen
            `width_pt`/`height_pt` if that happens, don't fall back to 'tight'.
        tight_layout : bool, default=True
            Only consulted when `exact_size=True`. `exact_size` pins the CANVAS,
            but `tight_layout()` still places the axes box from each panel's own
            tick labels -- so a set of panels can come out pixel-identical in
            canvas size while their plot areas sit at different offsets, and they
            will not align when stacked. Pass False when the caller has already
            pinned the layout itself (`fig.subplots_adjust(...)` with fixed
            margins), which is the only way to guarantee alignment across panels
            whose tick labels differ in width.

        Example
        -------
        >>> styler.finish_plot(save_plots=True, save_dir='./figs', name='my_plot')
        """
        if save_plots and save_dir:
            save_path = pathlib.Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Text sizes are checked on EVERY save, page geometry only when the profile
            # has a page. Runs BEFORE the save so the report sits next to the
            # "[saved] ..." line in the job log rather than scrolling away above it --
            # but the CONSEQUENCE is deferred to after the files are written, see below.
            violations = self.check_journal_compliance()

            if exact_size:
                if tight_layout:
                    plt.tight_layout()
                # `bbox_inches=None` means "use rcParams['savefig.bbox']", and
                # configure_svg_settings() sets that rcParam to 'tight' globally
                # -- so None here would silently still tight-crop. Pass an
                # explicit Bbox spanning the whole figure to actually get the
                # untouched width_pt/height_pt canvas.
                fig = plt.gcf()
                w_in, h_in = fig.get_size_inches()
                full_bbox = mpl.transforms.Bbox.from_bounds(0, 0, w_in, h_in)

            for fmt in formats:
                filepath = save_path / f'{name}.{fmt}'

                if fmt == 'svg':
                    plt.savefig(
                        filepath,
                        format='svg',
                        bbox_inches=full_bbox if exact_size else 'tight',
                        pad_inches=0 if exact_size else 0.1,
                        transparent=False,
                        dpi=dpi,
                        metadata={'Creator': 'Matplotlib/BrainDance'}
                    )
                else:
                    plt.savefig(
                        filepath, dpi=dpi,
                        bbox_inches=full_bbox if exact_size else 'tight'
                    )

            plt.close()

            # 🚨 SAVE-THEN-RAISE, and in that order deliberately (2026-08-13 PI
            # directive). On a `strict` profile an out-of-band text size is FATAL: a
            # printed warning is ignorable and was ignored -- 220 out-of-spec labels
            # accumulated under one, and a "silent check" was repeatedly mistaken for
            # proof a page was clean. Raising after the files are on disk keeps the one
            # genuine merit of the old behavior (you can open the offending render and
            # see what broke) while making the failure impossible to walk past: the
            # script exits non-zero and any pipeline built on it stops.
            # ⛔ There is no per-call `allow_violations` escape, on purpose. A figure
            # that genuinely needs a different size changes the SPEC at construction --
            # `Styler(journal=..., tick_pt=5)` -- which is one greppable place. `"draft"`
            # is `strict=False` because scratch/QC/library figures are not a compliance
            # surface and hard-failing them would break callers outside this repo.
            if violations and self.journal.strict:
                raise ValueError(
                    f"[Styler] {self.journal.name} compliance: {len(violations)} "
                    f"issue(s) in {name!r} -- files were still written to {save_dir} so "
                    "you can look at them, but this is a hard failure.\n  - "
                    + "\n  - ".join(violations)
                    + f"\nFix by LAYOUT, or retune the spec at construction "
                      f"(Styler(journal=..., {ROLE_FIELDS[2]}=5)). Do not add a "
                      "`fontsize=` to a plot script."
                )
        else:
            plt.tight_layout()
            if show:
                plt.show()

    def plot_color_palette(self) -> plt.Figure:
        """
        Display all available colors in the palette.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure showing the color palette
        """
        fig, ax = plt.subplots(figsize=(12, 2))
        
        n_colors = len(self.colors)
        for i, color in enumerate(self.colors):
            ax.bar(i, 1, color=color, width=0.8, edgecolor='black', linewidth=0.5)
            ax.text(i, -0.15, f'{i}', ha='center', va='top', fontsize=8)
            ax.text(i, -0.35, color, ha='center', va='top', fontsize=6, rotation=45)
        
        ax.set_title('Styler Color Palette', fontsize=10)
        ax.set_xlim(-0.5, n_colors - 0.5)
        ax.set_ylim(-0.6, 1.1)
        ax.axis('off')
        
        plt.tight_layout()
        return fig

    # Convenience methods for common plot elements
    
    def style_raster(self, ax: plt.Axes, spine_color: str = '#333333'):
        """Apply styling suitable for raster plots."""
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(spine_color)
        ax.tick_params(left=False, labelleft=False)

    def style_heatmap(self, ax: plt.Axes):
        """Apply styling suitable for heatmaps."""
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)

    def add_scalebar(
        self, 
        ax: plt.Axes, 
        length: float, 
        label: str, 
        loc: str = 'lower right',
        color: str = 'black',
        fontsize: Optional[float] = None,
    ):
        """
        Add a scale bar to an axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add scale bar to
        length : float
            Length of scale bar in data units
        label : str
            Label for the scale bar (e.g., '100 ms')
        loc : str, default='lower right'
            Location: 'lower right', 'lower left', 'upper right', 'upper left'
        color : str, default='black'
            Color of the scale bar
        fontsize : float, optional
            Defaults to the profile's `annotation_pt`. Was a hardcoded 7 -- in spec, but
            still a number typed into the library rather than read from it, so it would
            not follow a profile change. ⚠️ A scale bar's label is also one of the things
            `check_journal_compliance` CANNOT vet for you: it lives inside an
            `AnchoredSizeBar` artist, and the journal's real requirement here is
            editorial ("scale bars, not magnification factors"), not typographic.
        """
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        if fontsize is None:
            fontsize = self.journal.annotation_pt
        fontprops = fm.FontProperties(size=fontsize)
        scalebar = AnchoredSizeBar(
            ax.transData,
            length, label,
            loc=loc,
            pad=0.1,
            color=color,
            frameon=False,
            size_vertical=0,
            fontproperties=fontprops
        )
        ax.add_artist(scalebar)

    def add_significance_bracket(
        self,
        ax: plt.Axes,
        x1: float,
        x2: float,
        y: float,
        label: str,
        *,
        height: float = None,
        color: str = 'black',
        linewidth: float = 0.8,
        linestyle: str = '-',
        fontsize: Optional[float] = None,
    ):
        """
        Draw a pairwise significance bracket (caliper shape: short end-ticks,
        label centered above the horizontal bar) between two x positions.

        For multiple comparisons on one axes, call this once per pair with
        increasing `y` so the brackets stack rather than overlap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        x1, x2 : float
            The two x positions being compared.
        y : float
            Y position of the bracket's horizontal bar, in data coordinates.
        label : str
            Text centered above the bracket, e.g. '***', '**', '*', 'n.s.'
        height : float, optional
            Height of the end-ticks, in data units. Defaults to 1.5% of the
            current y-axis range.
        color : str, default='black'
        linewidth : float, default=0.8
        linestyle : str, default='-'
            Added so a panel can distinguish two overlapping bracket FAMILIES (e.g. a
            pooled-control comparison vs. a within-preparation comparison) by dashing one
            of them, without a per-script `ax.plot` hack duplicating this method's caliper
            geometry. Backward compatible: every existing caller is solid, unchanged.
        fontsize : float, optional
            Defaults to the profile's `annotation_pt` (6 pt for Nature). 🚨 This was a
            hardcoded **8**, which is `panel_letter_pt` -- so every '***'/'ns' label
            inherited the compliance check's PANEL-LETTER carve-out and shipped as
            out-of-spec body text under a green check. Found 2026-08-13 the moment that
            carve-out was made identity-based: fig3's page had three 8 pt 'ns' labels
            nobody had ever seen flagged. A significance star is annotation -- a gloss
            on the data -- so it takes the annotation tier, not a size of its own.
        """
        if fontsize is None:
            fontsize = self.journal.annotation_pt
        if height is None:
            ylo, yhi = ax.get_ylim()
            height = 0.015 * (yhi - ylo)
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=color,
                linewidth=linewidth, linestyle=linestyle, clip_on=False,
                solid_capstyle='butt')
        ax.text((x1 + x2) / 2, y + height, label, ha='center', va='bottom',
                fontsize=fontsize, color=color)

    def add_significance_label(
        self,
        ax: plt.Axes,
        x: float,
        label: str,
        *,
        y: float = 0.985,
        transform=None,
        fontsize: Optional[float] = None,
        fontweight: str = 'bold',
        color: str = '0.3',
        ha: str = 'center',
        va: str = 'top',
    ):
        """
        Draw a single significance annotation ('***' / '**' / '*' / 'n.s.') above
        one x position, with no bracket -- for panels where each column/cell is
        tested against an implicit reference (zero, a reference-model row, a null)
        rather than against another visible column. Use `add_significance_bracket`
        instead when the panel is comparing two specific x positions to each
        other and the caliper shape is meaningful.

        Bold by default (added 2026-08-12: ad hoc `ax.text(..., fontsize=6,
        color='black')` calls for this exact pattern had drifted across several panels
        in `fig2_main/plots/` -- centralizing it here so the asterisk row reads
        consistently without every script re-deriving the same fontsize/weight).

        🚨 It used to also be "slightly larger than body text", via a hardcoded
        `fontsize=8`. That number is `panel_letter_pt`, so every asterisk row inherited
        the compliance check's PANEL-LETTER carve-out and shipped out-of-spec under a
        green check -- the same latent bug as `add_group_title` and
        `add_significance_bracket`, all three found on 2026-08-13 when the carve-out
        was made identity-based. The size now comes from the profile's `annotation_pt`
        (6 pt for Nature); ⛔ prominence comes from `fontweight='bold'`, not from
        exceeding the journal's ceiling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        x : float
            X position (data coordinates) of the cell being annotated.
        label : str
            '***', '**', '*', or an explicit 'n.s.'
        y : float, default=0.985
            Y position. Interpreted in AXES-fraction coordinates by default (via
            the blended transform below), so the label sits at a consistent
            height across cells regardless of each cell's own y-range -- pass a
            custom `transform` (e.g. `ax.transData` twice) to use data
            coordinates on both axes instead.
        transform : matplotlib transform, optional
            Defaults to a blended transform (x in data coords, y in axes-fraction
            coords) -- the standard "one asterisk row across a shared x, each row
            its own y-scale" pattern used by this project's forest-matrix panels
            (e.g. `proj/predictor/figures/general_metrics/fig2_main/plots/`).
        fontsize : float, optional -- defaults to the profile's `annotation_pt`.
        fontweight : str, default='bold'
        color : str, default='0.3' -- a dark neutral gray, not stark black; keeps
            the label legible without being the darkest ink on the panel.
        """
        if fontsize is None:
            fontsize = self.journal.annotation_pt
        if transform is None:
            transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(x, y, label, transform=transform, ha=ha, va=va,
                fontsize=fontsize, fontweight=fontweight, color=color)

    def label_panel(
        self,
        ax: plt.Axes,
        letter: str,
        *,
        loc: tuple = (-0.12, 1.05),
        fontsize: Optional[float] = None,
        fontweight: str = 'bold',
    ):
        """
        Add a bold panel letter (A, B, C, ...) at the top-left of an axes, in
        the standard multi-panel-figure convention.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        letter : str
            The panel label, e.g. 'A', 'B'.
        loc : tuple, default=(-0.12, 1.05)
            Position in axes-fraction coordinates.
        fontsize : int, optional
            Defaults to the profile's `panel_letter_pt` (8 pt for Nature). Pass a
            value only for a non-composite one-off; `composite_letter_fontsize`
            already resolves this for composite pages.
        fontweight : str, default='bold'
        """
        if fontsize is None:
            fontsize = self.journal.panel_letter_pt
        t = ax.text(*loc, letter, transform=ax.transAxes, fontsize=fontsize,
                    fontweight=fontweight, va='bottom', ha='right')
        # 🚨 MARK IT. `check_journal_compliance` exempts panel letters from the 5-7 pt
        # ceiling, and it used to do that by SIZE -- "text at exactly panel_letter_pt is a
        # panel letter". That is a hole: any body text that happens to be 8.0 pt inherits
        # the carve-out by coincidence of number, which is exactly how `add_group_title`'s
        # old `fontsize=8` default shipped out-of-spec group headings past a green check.
        # Nature's wording is "maximum text size for all other text: 7 pt" -- a group
        # heading is other text. Exempting by IDENTITY closes that permanently.
        t.set_gid(PANEL_LETTER_GID)
        return t

    def mark_panel_letter(self, artist):
        """Claim a hand-placed `Text` as a panel letter, exempting it from the 5-7 pt
        ceiling in `check_journal_compliance`.

        `label_panel` marks its own artist, so this is ONLY for a page that places its
        letters some other way. That is not an exotic case: `label_panel`'s `loc` is a
        fraction of the AXES, so a page whose panels differ in width gets ragged letters,
        and the fix is to measure real ink and place them in FIGURE coordinates via
        `fig.text` -- see `fig2_metrics_ablations/build_main_figure.py::_place_letters`.
        Such a letter is a panel letter in every sense except that the Styler did not
        draw it, and before this method existed the only way to keep it legal was for the
        checker to exempt by SIZE, which is the hole identity-marking closed.

        ⛔ Do not reach for this to silence a violation on text that is not a panel
        letter. The exemption exists because Nature specifies letters at 8 pt while
        capping all *other* text at 7; a group heading, an `n = 12`, or a significance
        star is other text, and the fix there is the profile's role table.
        """
        artist.set_gid(PANEL_LETTER_GID)
        return artist

    def add_group_title(
        self,
        fig: plt.Figure,
        axes,
        text: str,
        *,
        gap: float = 0.028,
        fontsize: Optional[float] = None,
        fontweight: str = 'bold',
        zorder: int = 10,
    ):
        """
        Add a title centered above a GROUP of axes that together form one
        conceptual panel inside a composite figure (e.g. two `sharey`
        sub-axes drawn by one `draw_*((ax0, ax1), ...)` function) -- the
        multi-axes analogue of `ax.set_title()` for a panel that is really N
        axes, not one.

        A panel's own standalone render typically captions this with
        `fig.suptitle(...)`, which is meaningless once that panel is 2 of N
        axes sharing one composite figure with its own unrelated titles.
        Call this instead, from the composite script, after the panel's axes
        have been placed by the outer `GridSpec` -- it reads their REAL
        post-layout position via `get_position()` rather than a hand-guessed
        offset, so it tracks wherever the surrounding grid actually puts them.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The composite figure (not any one panel's own figure).
        axes : sequence of matplotlib.axes.Axes
            The axes this title spans. Centered between the leftmost axes'
            left edge and the rightmost axes' right edge; placed above the
            highest axes' top edge.
        text : str
        gap : float, default=0.028
            Vertical gap above the axes, in figure-fraction. Deliberately
            generous: at a small gap (~0.012) this text can render BEHIND the
            axes' own opaque background and be invisible despite being
            present in the saved SVG -- confirmed empirically on
            `busy_drift/build_supp_figure.py`'s substrate panel (2026-08-12).
            If the title still clips or overlaps a neighboring row, widen
            this or that row's own `hspace` first -- not this method.
        fontsize : float, optional
            Defaults to the profile's `title_pt` (7 pt for Nature) -- a group
            heading is body text and takes the body ceiling. 🚨 This default was
            a hardcoded **8**, which is `panel_letter_pt`, so every group title
            inherited the panel-letter carve-out in `check_journal_compliance`
            and shipped out-of-spec while the check stayed green. A group title
            must read as superior to the panel titles beneath it, but it earns
            that from **weight and vertical space** (`gap`), not size -- Nature
            caps all non-panel-letter text at 7 pt with no exception for
            headings. ⛔ Do not pass 8 here to restore the old look.
        fontweight : str, default='bold'
        zorder : int, default=10
            Kept high for the same occlusion reason as `gap`'s default.

        Call this AFTER the figure's layout is finalized (e.g. after
        `fig.tight_layout()`), and save with `finish_plot(...,
        tight_layout=False)` afterward -- a second internal `tight_layout()`
        pass would shift the axes out from under a position already read.

        Example
        -------
        >>> styler.add_group_title(fig, [ax_c0, ax_c1],
        ...     "Drift exponent gap across substrates")
        """
        if fontsize is None:
            fontsize = self.journal.title_pt
        positions = [ax.get_position() for ax in axes]
        x0 = min(p.x0 for p in positions)
        x1 = max(p.x1 for p in positions)
        y1 = max(p.y1 for p in positions)
        fig.text((x0 + x1) / 2, y1 + gap, text, ha='center', va='bottom',
                 fontsize=fontsize, fontweight=fontweight, zorder=zorder)


def main():
    """Test the Styler with example plots."""
    styler = Styler(journal="draft")
    
    # Create test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = -np.sin(x)
    
    # Test single column figure
    fig, ax = styler.create_figure(size_preset='single')
    ax.plot(x, y1, label='Sin(x)')
    ax.plot(x, y2, label='Cos(x)')
    ax.plot(x, y3, label='-Sin(x)')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Test Plot')
    ax.legend()
    
    styler.finish_plot(save_plots=False, show=True)
    
    # Show color palette
    styler.plot_color_palette()
    plt.show()


if __name__ == '__main__':
    main()
