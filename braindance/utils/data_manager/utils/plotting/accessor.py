"""
Plotting accessor for BrainDance Recording objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

from braindance.utils.data_manager.utils.plotting.raster import plot_raster_with_pop
from braindance.utils.data_manager.utils.plotting.evoked_response import plot_evoked_raster, plot_evoked_psth
from braindance.utils.data_manager.utils.plotting.population import plot_sttc_matrix, plot_firing_rate_histogram
from braindance.utils.data_manager.utils.plotting.spatial_latency import (
    spatial_latency, 
    spatial_response_map, 
    spatial_animation
)

if TYPE_CHECKING:
    from braindance.utils.data_manager.utils.data_loading.recording import Recording
    from braindance.utils.data_manager.utils.plotting.plot_styler import Styler


class PlotAccessor:
    """
    Plotting accessor for Recording objects.
    
    Provides convenient access to plotting functions via rec.pl.method()
    
    Example
    -------
    >>> rec = load_recording(...)
    >>> rec.pl.raster_with_pop(time_window=(0, 60))
    >>> rec.pl.sttc_matrix()
    >>> rec.pl.firing_rate_hist()
    """
    
    def __init__(self, recording: 'Recording'):
        self._recording = recording
        self._styler = None
    
    @property
    def styler(self) -> 'Styler':
        """Get or create the default styler."""
        if self._styler is None:
            from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
            self._styler = Styler(journal="draft")
        return self._styler
    
    @styler.setter
    def styler(self, value: 'Styler'):
        """Set a custom styler."""
        self._styler = value
    
    def raster_with_pop(
        self,
        title: Optional[str] = None,
        time_window: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        size_preset: str = 'single',
        smoothing_window: float = 100,
        sort_by_fr: bool = False,
        time_unit: str = 'seconds',
        pop_rate_unit: str = 'Hz',
        stim_log = None,
        show_pop_rate: bool = True,
        show_stim_markers: bool = False,
        stim_marker_style: str = 'line',
        stim_color: Optional[str] = None,
        stim_alpha: float = 0.3,
        stim_time_col: str = 'auto',
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]:
        """
        Plot raster with population rate for this recording.
        """
        if title is None:
            title = getattr(self._recording, 'identifier', None)

        # Auto-fetch stim_log if not provided
        if stim_log is None and hasattr(self._recording, 'stim_log'):
            stim_log = self._recording.stim_log

        return plot_raster_with_pop(
            spike_data=self._recording.spikes,
            title=title,
            time_window=time_window,
            y_lim=y_lim,
            styler=self.styler,
            size_preset=size_preset,
            smoothing_window=smoothing_window,
            sort_by_fr=sort_by_fr,
            time_unit=time_unit,
            pop_rate_unit=pop_rate_unit,
            stim_log=stim_log,
            show_pop_rate=show_pop_rate,
            show_stim_markers=show_stim_markers,
            stim_marker_style=stim_marker_style,
            stim_color=stim_color,
            stim_alpha=stim_alpha,
            stim_time_col=stim_time_col,
            save_path=save_path,
            filename=filename,
            show=show,
            **kwargs
        )
    
    def sttc_matrix(
        self,
        sttc_matrix: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        size_preset: str = 'single',
        cmap: str = 'RdBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        delt: float = 20.0,
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot STTC connectivity matrix for this recording.
        """
        if sttc_matrix is None:
            # Try to get from results
            if hasattr(self._recording, 'results') and hasattr(self._recording.results, 'sttc'):
                sttc_matrix = self._recording.results.sttc
            elif hasattr(self._recording, 'spikes') and self._recording.spikes is not None:
                # Auto-compute STTC
                print(f"Computing STTC matrix (delt={delt}ms)...")
                sttc_matrix = self._recording.spikes.spike_time_tilings(delt=delt)
            else:
                raise ValueError("No STTC matrix provided and cannot auto-compute.")
        
        sttc_matrix = getattr(sttc_matrix, 'matrix', sttc_matrix)

        if title is None:
            title = f'STTC - {getattr(self._recording, "identifier", "Recording")}'
        
        return plot_sttc_matrix(
            sttc_matrix=sttc_matrix,
            title=title,
            styler=self.styler,
            size_preset=size_preset,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            filename=filename,
            show=show,
            **kwargs
        )
    
    def firing_rate_hist(
        self,
        title: Optional[str] = None,
        size_preset: str = 'single',
        bins: int = 30,
        log_scale: bool = True,
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot firing rate histogram for this recording.
        """
        return plot_firing_rate_histogram(
            spike_data=self._recording.spikes,
            title=title,
            styler=self.styler,
            size_preset=size_preset,
            bins=bins,
            log_scale=log_scale,
            save_path=save_path,
            filename=filename,
            show=show,
            **kwargs
        )

    def evoked_raster(
        self,
        electrode_id: int,
        neuron_idx: int,
        plot_window: Tuple[float, float] = (-100, 100),
        max_trials: int = 400,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot evoked response raster for a neuron-electrode pair.
        """
        if 'latency_data' not in kwargs:
            latency_data = None
            if hasattr(self._recording, 'results') and hasattr(self._recording.results, 'evoked_pairs'):
                pair_key = f"electrode_{electrode_id}_neuron_{neuron_idx}"
                latency_data = self._recording.results.evoked_pairs.get(pair_key)
            kwargs['latency_data'] = latency_data

        return plot_evoked_raster(
            spike_data=self._recording.spikes,
            stim_log=self._recording.stim_log,
            electrode_id=electrode_id,
            neuron_idx=neuron_idx,
            plot_window=plot_window,
            max_trials=max_trials,
            styler=self.styler,
            save_path=save_path,
            show=show,
            **kwargs
        )

    def evoked_psth(
        self,
        electrode_id: int,
        neuron_idx: int,
        plot_window: Tuple[float, float] = (-100, 100),
        max_trials: int = 400,
        psth_bins: int = 75,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot evoked response PSTH for a neuron-electrode pair.
        """
        if 'latency_data' not in kwargs:
            latency_data = None
            if hasattr(self._recording, 'results') and hasattr(self._recording.results, 'evoked_pairs'):
                pair_key = f"electrode_{electrode_id}_neuron_{neuron_idx}"
                latency_data = self._recording.results.evoked_pairs.get(pair_key)
            kwargs['latency_data'] = latency_data

        return plot_evoked_psth(
            spike_data=self._recording.spikes,
            stim_log=self._recording.stim_log,
            electrode_id=electrode_id,
            neuron_idx=neuron_idx,
            plot_window=plot_window,
            max_trials=max_trials,
            psth_bins=psth_bins,
            styler=self.styler,
            save_path=save_path,
            show=show,
            **kwargs
        )
    
    def spatial_latency(
        self,
        electrode_id: Optional[int] = None,
        latency_results: Optional[Dict] = None,
        time_window: Tuple[float, float] = (0, 100),
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = False
    ) -> Any:
        """
        Plot spatial distribution of onset latencies.
        """
        return spatial_latency(
            recording=self._recording,
            electrode_id=electrode_id,
            latency_results=latency_results,
            time_window=time_window,
            styler=self.styler,
            save_path=save_path,
            filename=filename,
            show=show
        )
    
    def spatial_response_map(
        self,
        electrode_id: int,
        metric: str = 'latency',
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = False
    ) -> Any:
        """
        Plot spatial map of neural responses to stimulation.
        """
        return spatial_response_map(
            recording=self._recording,
            electrode_id=electrode_id,
            metric=metric,
            styler=self.styler,
            save_path=save_path,
            filename=filename,
            show=show
        )
    
    def spatial_animation(
        self,
        electrode_id: int,
        time_window: Tuple[float, float] = (-50, 100),
        smoothing_ms: float = 20,
        save_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Any:
        """
        Create animation of neural response propagation.
        """
        return spatial_animation(
            recording=self._recording,
            electrode_id=electrode_id,
            time_window=time_window,
            smoothing_ms=smoothing_ms,
            styler=self.styler,
            save_path=save_path,
            filename=filename
        )
