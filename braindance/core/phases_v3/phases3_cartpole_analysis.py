"""Native V3 analysis for the paper CartPole protocol (20 kHz Maxwell data)."""
from pathlib import Path

import numpy as np

from .phase_base_v3 import AnalysisPhaseV3


def _read_windows(loader, filename, starts, width, channels=None):
    """Load complete windows, propagating I/O errors instead of substituting zeros."""
    windows = []
    for start in starts:
        data = loader.load_data_maxwell(
            filename, start=int(start), length=width, channels=channels)
        if data.ndim != 2 or data.shape[1] != width:
            raise ValueError(f'Incomplete {width}-frame window at frame {start}')
        windows.append(data)
    return np.stack(windows)


class CartPoleFootprintPhaseV3(AnalysisPhaseV3):
    """Activity maxima, spike-triggered footprints, then overlap rejection."""
    inputs = ['recording_file', 'recording_duration', 'stim_electrodes']
    outputs = ['selected_electrodes', 'selected_channels', 'mapping_file_path',
               'stim_electrodes', 'footprint_channels', 'footprint_waves']

    def __init__(self, wind=100, rms_mult=1, spikes_per_channel=300,
                 num_channel_thresh=120, num_spikes_min=20,
                 similarity_thresh=.65, pk_to_pk_thresh=9, seed=None):
        super().__init__()
        self.wind = wind
        self.rms_mult = rms_mult
        self.spikes_per_channel = spikes_per_channel
        self.num_channel_thresh = num_channel_thresh
        self.num_spikes_min = num_spikes_min
        self.similarity_thresh = similarity_thresh
        self.pk_to_pk_thresh = pk_to_pk_thresh
        self.seed = seed

    def run(self, experiment):
        from scipy.ndimage import maximum_filter, minimum_filter
        from braindance.analysis import data_loader

        if experiment.data.recording_duration < 60:
            raise ValueError('Paper footprint analysis needs at least 60 seconds of baseline')
        filename = str(experiment.data.recording_file)
        mapping = data_loader.load_mapping_maxwell(filename).copy()
        spikes = data_loader.load_data_maxwell(filename, spikes=True)
        shape = data_loader.load_info_maxwell(filename)['shape']
        if shape[1] < 60 * 20000:
            raise ValueError('Baseline data must contain the full 50–60 second RMS window')
        if spikes.empty:
            raise ValueError('Baseline contains no spikes for footprint selection')

        # Historical heatmap score: normalized log spike count and mean amplitude.
        counts = np.log1p(spikes.groupby('channel').size())
        span = counts.max() - counts.min()
        counts = (counts - counts.min()) / span if span > 0 else counts * 0
        means = spikes.groupby('channel')['amplitude'].mean()
        mapping['spike_count'] = mapping['channel'].map(counts)
        mapping['mean_amp'] = -.1 * mapping['channel'].map(means)
        mapping['spike_amp_product'] = (1 + mapping.spike_count) * (1 + mapping.mean_amp)
        values = np.zeros(26400)
        values[mapping.electrode.to_numpy(dtype=int)] = mapping.spike_amp_product.fillna(0)
        values = values.reshape(120, 220)
        rows, cols = np.any(values, axis=1), np.any(values, axis=0)
        if not rows.any() or not cols.any():
            raise ValueError('Baseline has no finite activity maxima')
        ymin, ymax = np.flatnonzero(rows)[[0, -1]]
        xmin, xmax = np.flatnonzero(cols)[[0, -1]]
        cropped = values[ymin:ymax + 1, xmin:xmax + 1]
        electrode_grid = np.arange(26400).reshape(120, 220)[ymin:ymax + 1, xmin:xmax + 1]
        maxima = ((maximum_filter(cropped, size=5) == cropped)
                  ^ minimum_filter(cropped == 0, size=5))
        candidates = electrode_grid[maxima][np.argsort(cropped[maxima])[::-1]]
        channel_by_electrode = mapping.set_index('electrode')['channel']
        rms_data = data_loader.load_data_maxwell(filename, start=50 * 20000,
                                                  length=10 * 20000)
        if rms_data.shape != (shape[0], 10 * 20000):
            raise ValueError('Incomplete baseline RMS window')
        rms = np.std(rms_data, axis=1)
        rng = np.random.default_rng(self.seed)
        footprints = []
        for electrode in candidates:
            channel = int(channel_by_electrode.loc[electrode])
            frames = spikes.loc[spikes.channel == channel, 'frame'].to_numpy(dtype=int)
            frames = frames[(frames >= self.wind) & (frames + self.wind <= shape[1])]
            if len(frames) < self.num_spikes_min:
                continue
            if len(frames) > self.spikes_per_channel:
                frames = rng.choice(frames, self.spikes_per_channel, replace=False)
            print(f'   Footprint electrode {electrode}: {len(frames)} spikes')
            windows = _read_windows(data_loader, filename, frames - self.wind, 2 * self.wind)
            averages = windows.mean(axis=0)
            peaks = np.ptp(averages, axis=1)
            channels = np.flatnonzero((peaks > self.pk_to_pk_thresh)
                                     & (peaks >= self.rms_mult * rms[channel]))
            if len(channels) > self.num_channel_thresh:
                continue
            # Keep the amplitude with its own footprint even when earlier units fail.
            footprints.append(dict(channel=channel, electrode=int(electrode),
                                   channels=channels, waves=averages[channels],
                                   amplitude=peaks[channel]))

        removed = set()
        for i, first in enumerate(footprints):
            for j, second in enumerate(footprints):
                if i == j or i in removed or j in removed:
                    continue
                overlap = len(set(first['channels']) & set(second['channels']))
                if (overlap > self.similarity_thresh * len(first['channels'])
                        and overlap > self.similarity_thresh * len(second['channels'])):
                    removed.add(i if first['amplitude'] < second['amplitude'] else j)
        selected = [item for i, item in enumerate(footprints) if i not in removed]
        selected_electrodes = [item['electrode'] for item in selected]
        stimulation = [int(e) for e in experiment.data.stim_electrodes
                       if e in selected_electrodes]
        if len(stimulation) < 6 or len(set(stimulation)) != len(stimulation):
            raise ValueError('Footprint selection must retain at least six unique configured '
                             'stimulation electrodes for sensory, motor, and training roles')
        mapping_path = filename + '_mapping.csv'
        mapping.to_csv(mapping_path)
        return dict(selected_electrodes=selected_electrodes,
                    selected_channels=[item['channel'] for item in selected],
                    mapping_file_path=mapping_path, stim_electrodes=stimulation,
                    footprint_channels=[item['channels'] for item in selected],
                    footprint_waves=[item['waves'] for item in selected])


class CartPoleCausalAnalysisPhaseV3(AnalysisPhaseV3):
    """Align pulses, remove artifacts, detect evoked spikes, and count connections."""
    inputs = ['sweep_file']
    outputs = ['valid_stim_electrodes', 'derived_dir', 'causal_reactivity',
               'causal_reactivity_times', 'causal_channels']

    def __init__(self, wind_ms=300, tag='causal', remove_start_frames=40,
                 art_rem_N=60, first_order_ms=30, multi_order_ms=100):
        super().__init__()
        if wind_ms < max(first_order_ms, multi_order_ms):
            raise ValueError('Response window must cover both connectivity windows')
        self.wind_ms = wind_ms
        self.tag = tag
        self.remove_start_frames = remove_start_frames
        self.art_rem_N = art_rem_N
        self.first_order_ms = first_order_ms
        self.multi_order_ms = multi_order_ms

    def run(self, experiment):
        from scipy.signal import find_peaks
        from braindance.analysis import data_loader
        from braindance.core.artifact_removal import cubic_fit5

        filename = str(experiment.data.sweep_file)
        mapping = data_loader.load_mapping_maxwell(filename)
        stim_log = data_loader.adjust_stim_times2(filename, stim_offset_ms=10, tag=self.tag).copy()
        electrodes = []
        for value in stim_log.stim_electrodes:
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != 1:
                    raise ValueError('Causal screening requires one electrode per pulse')
                value = value[0]
            if value is None or not np.isfinite(value):
                raise ValueError('Causal stimulation log contains an invalid electrode')
            electrodes.append(int(value))
        stim_log['stim_electrodes'] = electrodes
        electrodes = list(dict.fromkeys(electrodes))
        if not electrodes:
            raise ValueError('No causal stimuli found in recording')
        channels = [int(mapping.loc[mapping.electrode == e, 'channel'].item()) for e in electrodes]
        responses = np.empty((len(electrodes), len(electrodes)), dtype=object)
        reactivity = np.zeros((*responses.shape, self.wind_ms))
        width = self.remove_start_frames + self.wind_ms * 20 + self.art_rem_N
        total_frames = data_loader.load_info_maxwell(filename)['shape'][1]
        for i, electrode in enumerate(electrodes):
            times = stim_log.loc[stim_log.stim_electrodes == electrode, 'time_mod'].to_numpy()
            if not np.isfinite(times).all():
                raise ValueError('Causal stimulation alignment contains invalid timestamps')
            starts = (times * 20000).astype(int)
            if np.any(starts < 0) or np.any(starts + width > total_frames):
                raise ValueError('Recording does not contain a full causal response window')
            print(f'   Causal electrode {electrode}: {len(starts)} repetitions')
            windows = _read_windows(data_loader, filename, starts, width, channels=channels)
            windows[:, :, :self.remove_start_frames] = 0
            for j in range(len(channels)):
                stds = []
                for repeat in range(len(starts)):
                    trace = windows[repeat, j, self.remove_start_frames:]
                    trace[:] = cubic_fit5(trace, self.art_rem_N, self.art_rem_N)[0]
                    trace[-self.art_rem_N:] = 0
                    if np.max(trace) > 200 or np.min(trace) < -200:
                        trace[:] = 0
                    stds.append(np.std(trace))
                threshold = np.median(stds) * 5
                peaks_per_repeat = []
                for repeat in range(len(starts)):
                    trace = windows[repeat, j, self.remove_start_frames:-self.art_rem_N]
                    peaks = find_peaks(-trace, height=threshold, distance=3 * 20)[0]
                    peaks_per_repeat.append(peaks)
                responses[i, j] = peaks_per_repeat
                counts = np.bincount(np.concatenate(peaks_per_repeat) // 20,
                                     minlength=self.wind_ms)
                reactivity[i, j] = counts
        derived = Path(filename).parent / 'derived'
        derived.mkdir(parents=True, exist_ok=True)
        for order, cutoff in [('first', self.first_order_ms * 20),
                              ('multi', self.multi_order_ms * 20)]:
            matrix = np.zeros(responses.shape)
            for i, j in np.ndindex(responses.shape):
                if i != j:
                    matrix[i, j] = sum(np.count_nonzero(peaks <= cutoff)
                                       for peaks in responses[i, j])
            means, stds = matrix.mean(axis=0), matrix.std(axis=0)
            stds[stds == 0] = 1
            for suffix, values in [('', matrix), ('_mean', means), ('_std', stds),
                                   ('_norm', (matrix - means) / stds)]:
                np.save(derived / f'causal_connectivity_{order}{suffix}.npy', values)
        return dict(valid_stim_electrodes=electrodes, derived_dir=str(derived.resolve()),
                    causal_reactivity=reactivity, causal_reactivity_times=responses,
                    causal_channels=channels)
