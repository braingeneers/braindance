import numpy as np
from braindance.core.phases_analysis import FootprintPhase


def get_footprints(analysis, selected_electrodes=None, file_path=None):
    """Return the footprint channels and waveforms for the given electrodes."""
    if file_path is None:
        file_path = analysis.file_path
    
    if selected_electrodes is None:
        selected_electrodes = analysis.selected_electrodes

    assert file_path is not None, "Must provide file_path or assign to analysis"
    assert selected_electrodes is not None, "Must provide selected_electrodes or assign to analysis"
    
    footprint_phase = FootprintPhase(verbose=True, rms_mult=1, wind=60, load_whole_recording=False, 
                                num_channel_thresh=120, remove_bad=True, remove_redundant=True,
                                similarity_thresh=.65)
    
    analysis.select_electrodes(selected_electrodes)
    analysis = footprint_phase.run(analysis)
    footprint_chs = analysis.selected_footprint_chans
    footprint_waves = analysis.selected_footprint_waves
    mapping = analysis.mapping
    return footprint_chs, footprint_waves, mapping
    
    