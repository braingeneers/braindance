"""
Plotting utilities for BrainDance.
"""

from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
from braindance.utils.data_manager.utils.plotting.journal_specs import (
    JournalPageSpec,
    JOURNAL_SPECS,
    get_journal_spec,
)
from braindance.utils.data_manager.utils.plotting.raster import plot_raster_with_pop
from braindance.utils.data_manager.utils.plotting.evoked_response import (
    plot_evoked_raster,
    plot_evoked_psth
)
from braindance.utils.data_manager.utils.plotting.population import (
    plot_sttc_matrix,
    plot_firing_rate_histogram
)
from braindance.utils.data_manager.utils.plotting.accessor import PlotAccessor
from braindance.utils.data_manager.utils.plotting.spatial_latency import (
    spatial_latency,
    spatial_response_map,
    spatial_animation
)

__all__ = [
    'Styler',
    'JournalPageSpec',
    'JOURNAL_SPECS',
    'get_journal_spec',
    'PlotAccessor',
    'plot_raster_with_pop',
    'plot_evoked_raster',
    'plot_evoked_psth',
    'plot_sttc_matrix',
    'plot_firing_rate_histogram',
    'spatial_latency',
    'spatial_response_map',
    'spatial_animation',
]
