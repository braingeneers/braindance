"""
After causal frequency experiment, sort spikes in intrinsic windows and save as sorted.mat/sorted.npz files 
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import savemat
from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.preprocessing.filter import bandpass_filter
from spikeinterface.preprocessing.scale import scale_to_uV
from tqdm import tqdm

from braindance.analysis import data_loader


def _get_unit_dict(task):
    unit_idx, spike_frames = task
    
    template = np.zeros((n_before+n_after+1, len(chan_locs)))
    if len(spike_frames) > num_waveforms:
        np.random.seed(231)
        template_spike_frames = np.random.choice(spike_frames, num_waveforms, replace=False)
    else:
        template_spike_frames = spike_frames
        
    for frame in template_spike_frames:
        assert frame-n_before >= 0, "If using RT-Sort with default 50ms pre-medians, this should be True. This makes next line easier"
        waveform = recording.get_traces(start_frame=frame-n_before, end_frame=frame+n_after+1)
        template[-waveform.shape[0]:] += waveform  # In case frame+n_after+1 is after filtered_traces ends
    template /= len(template_spike_frames) if len(template_spike_frames) > 0 else 1  # Prevent divide by 0
    
    max_chan = np.argmax(np.max(np.abs(template), axis=0))
    x, y = chan_locs[max_chan]
    return unit_idx, {
        "unit_id": unit_idx,
        "spike_train": spike_frames,
        "x_max": x,
        "y_max": y,
        "template": template,
        "electrode": electrodes[max_chan]
    }

def sort_intrinsic(recording_path, save_root_path, rt_sort, phases,
                   ms_before=3, ms_after=4, num_waveforms_for_template=300):
    """
    phases should be a list where each element is a list of
        [phase_name, num_cycles, intrinsic_start_min, cycle_duration_min] 
        min refers to minutes
        intrinsic_start_min is when the intrinsic period starts in each cycle
        
        NOTE: phases assumes that in each cycle, the beginning period has stimulation and the remaining period is just intrinsic activity
        
    phases example:
            ["pre", 3, 2, 6],
            ["seq_stim", 14, 3, 6],
            ["post1", 3, 2, 6],
            ["intrinsic", 7, 0, 6],
            ["post2", 3, 2, 6],
    """
    global cycle_traces, n_before, n_after, num_waveforms, chan_locs, electrodes, intrinsic_start_frame, recording
    
    save_root_path = Path(save_root_path)
    save_root_path.mkdir(exist_ok=True, parents=True)
    n_before = round(ms_before * rt_sort.samp_freq)
    n_after = round(ms_after * rt_sort.samp_freq)
    num_waveforms = num_waveforms_for_template
    
    samp_freq_hz = rt_sort.samp_freq * 1000
    
    recording = MaxwellRecordingExtractor(recording_path)
    recording = scale_to_uV(recording)
    recording = bandpass_filter(recording, 300, 6000)
    chan_locs = recording.get_channel_locations()
    electrodes = recording.get_property('electrode')
    if electrodes is None:
        electrodes = electrodes.recording.get_channel_ids()

    cycle_start_frame = 0
    for phase_name, num_cycles, intrinsic_start_min, cycle_duration_min in phases:
        print(f"Processing {phase_name} phase, {num_cycles} cycles")
        for cycle_idx in tqdm(range(num_cycles)):
            # tasks.append([phase_name, cycle_idx, recording_time+intrinsic_start_min, recording_time+cycle_duration_min])
            # Get cycle data
            intrinsic_start_frame = cycle_start_frame + intrinsic_start_min * 60 * samp_freq_hz
            cycle_traces = data_loader.load_data_maxwell(
                str(recording_path), 
                start=intrinsic_start_frame,
                length=(cycle_duration_min - intrinsic_start_min) * 60 * samp_freq_hz
            )  # (num_chans, num_frames)

            # Spike sort
            sequence_detections = rt_sort.sort_offline(cycle_traces)
            del cycle_traces
            
            # Get unit data
            all_unit_dicts = [None] * len(sequence_detections)
            with Pool(processes=8) as pool:
                all_spike_frames = [[unit_idx, np.round(spike_times_ms * rt_sort.samp_freq + intrinsic_start_frame).astype(int)]
                                    for unit_idx, spike_times_ms in enumerate(sequence_detections)]
                for unit_idx, unit_dict in pool.imap_unordered(_get_unit_dict, all_spike_frames):
                    all_unit_dicts[unit_idx] = unit_dict
            
            # Save file
            save_path = str(save_root_path / f"{phase_name}_cycle{cycle_idx}")
            compile_dict = {
                "units": all_unit_dicts,
                "locations": chan_locs,
                "fs": rt_sort.samp_freq * 1000,
                "cycle_start_frame": cycle_start_frame,
            }
            savemat(save_path + '.mat', compile_dict)
            np.savez(save_path + ".npz", **compile_dict)
            
            cycle_start_frame += cycle_duration_min*60*samp_freq_hz
            
if __name__ == "__main__":   
    sort_intrinsic(
        "/data/MEAprojects/primary_mouse/e_stim_seq/231122/22469/causal_freq_22469_C_231122.raw.h5", None, None,
        [
            ["pre", 3, 2, 6],
            ["seq_stim", 14, 3, 6],
            ["post1", 3, 2, 6],
            ["intrinsic", 7, 0, 6],
            ["post2", 3, 2, 6]
        ]
    )