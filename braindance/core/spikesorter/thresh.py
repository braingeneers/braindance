import numpy as np

from braindance.core.spikesorter.sort import Sort

class ThreshSort(Sort):
    """
    Use threshold crossings (such as 5RMS) for spike sorting
        Instead of sequences' spikes, get electrodes' spikes
    """
    def __init__(self, num_elecs, thresh,
                 samp_freq,
                 noise_ms=50, 
                 dtype="float32"):
        """
        Params
            thresh:
                causal_connectivity.py uses 3.5
                5RMS is common
            noise_ms:
                To determine background noise for thresholding
        """
        super().__init__(num_units=num_elecs, samp_freq=samp_freq)
        
        self.dtype = dtype
        
        self.x_thresh = thresh
        self.thresh = None  # Actual threshold number used (varies depending on noise)
        
        self.total_noise_frames = round(noise_ms * samp_freq)
        self.cache_frame = 2
        self.traces_cache = np.zeros((self.total_noise_frames + 2, num_elecs), dtype=dtype)  # Faster to have num_elecs last since always slicing all elecs at once
            # +2 to include the last two frames of previous iteration so the last frame in previous iteration can be sorted
                
    def reset(self):
        super().reset()
        self.cache_frame = 2
        self.thresh = None
                
    def running_sort(self, obs, env=None):
        """
        Params
            obs:
                Has shape (num_frames, num_chans)
            env: None or MaxwellEnv
                If not None, use env to keep self.latest_frame on track instead of internal counter
        """
        
        obs = obs - np.median(np.asarray(obs), axis=0, keepdims=True)
        new_frames = obs.shape[0]
        
        if env is None:
            self.latest_frame += new_frames
        else:
            self.latest_frame = env.latest_frame + 1
        
        self.traces_cache[self.cache_frame:self.cache_frame + new_frames, :] = obs
        sorting_chunk = self.traces_cache[self.cache_frame-2:self.cache_frame + new_frames, :]  # Get last two frames of previous iteration so the last frame in previous iteration can be sorted
        
        self.cache_frame += new_frames
        if self.cache_frame - 2 == self.total_noise_frames:
            self.thresh = -np.std(self.traces_cache[2:], axis=0, keepdims=True) * self.x_thresh
            self.cache_frame = 2
            self.traces_cache[:2, :] = self.traces_cache[-2: :]

        if self.thresh is None:
            return []
        
        return self.sort_chunk(sorting_chunk, self.latest_frame - (new_frames + 1))
    
    def sort_chunk(self, chunk, spike_times_frame_offset=0):
        main = chunk[1:-1, :]
        greater_than_left = main > chunk[:-2, :]
        greater_than_right = main > chunk[2:, :]
        peaks = greater_than_left & greater_than_right
        crosses = main <= self.thresh
        
        thresh_crossings = np.argwhere((peaks & crosses).T).astype(float)
        thresh_crossings[:, 1] = (thresh_crossings[:, 1] + spike_times_frame_offset) / self.samp_freq
        return thresh_crossings   


