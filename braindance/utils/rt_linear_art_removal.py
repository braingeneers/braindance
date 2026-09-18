import numpy as np
from numba import njit, prange
import time

# Pre-computed factorial lookup table (simplified, only need up to 1!)
LOOKUP_TABLE = np.array([1, 1], dtype=np.int64)

@njit
def fast_factorial(n):
    """Fast factorial calculation using lookup table."""
    if n > 1:
        raise ValueError("Factorial too large")
    return LOOKUP_TABLE[n]

@njit
def fast_mmean(q, frame):
    """Exponential moving mean update."""
    q = q * 0.8 + 0.2 * frame
    return q

# Numba-optimized core functions - simplified for linear fit
@njit(fastmath=True)
def compute_T(nc, N):
    """Compute the T vector for linear fit."""
    T = np.zeros(3, dtype=np.float64)  # Only need first 3 terms for linear fit
    for k in range(3):
        for n in range(nc - N, nc + N + 1):
            T[k] += (n - nc) ** k
    return T

@njit(fastmath=True)
def compute_S(T):
    """Compute and invert the S matrix for linear fit."""
    S = np.zeros((2, 2), dtype=np.float64)  # 2x2 matrix for linear fit
    for k in range(2):
        for j in range(2):
            S[k, j] = T[k + j]
    return np.linalg.inv(S)

@njit(fastmath=True)
def compute_W(v, nc, N):
    """Compute initial W vector for linear fit."""
    W = np.zeros(2, dtype=np.float64)  # Only 2 elements for linear fit
    range_start = 0
    range_end = nc + N + 1
    
    for k in range(2):
        total = 0.0
        for n in range(range_start, range_end):
            total += (n - nc) ** k * v[n]
        W[k] = total
    
    return W

@njit(fastmath=True)
def compute_W_rec(Wp, N, nc, v, outgoing):
    """Compute W recursively using previous W for linear fit."""
    W = np.zeros(2, dtype=np.float64)
    for k in range(2):
        cur_sum = 0.0
        for j in range(k+1):
            # Calculate binomial coefficient using factorials
            num = (-1)**(k-j)*fast_factorial(k)
            den = fast_factorial(j)*fast_factorial(k-j)
            cur_sum += (num/den)*Wp[j] 
        
        cur_sum += (N**k) * v[nc+N] - ((-N-1)**k) * outgoing
        W[k] = cur_sum
    return W

@njit(fastmath=True)
def compute_a(S, W, a):
    """Compute linear fit coefficients."""
    for k in range(2):  # Only compute 2 coefficients for linear fit
        total = 0.0
        for j in range(2):
            total += S[k, j] * W[j]
        a[k] = total
    return a

@njit
def shift_buffer(buffer, new_val):
    """Shift values in buffer left and add new value at the end."""
    buffer[:-1] = buffer[1:]
    buffer[-1] = new_val
    return buffer

@njit(parallel=True)
def process_channel_minibatch_numba(channels, frames_batch, clean_values, artifacts, spikes, 
                                  artifact_width, remove_frames_before, 
                                  states, init_inds, moving_means, need_resets, spike_flags,
                                  buffers, W_vectors, Wp_vectors, S_matrices,
                                  min_val, max_val, spike_thresh_min, spike_thresh_max):
    """
    Process a minibatch of frames for multiple channels in parallel using Numba.
    Simplified for linear fit.
    """
    n_channels = len(channels)
    n_frames = frames_batch.shape[1]
    state_count = 0
    
    for i in prange(n_channels):  # Parallelize over channels
        ch = channels[i]
        N = buffers.shape[1] // 2 # Half-window size
        
        # Process each frame sequentially for this channel to maintain state
        for f in range(n_frames):
            frame_val = frames_batch[ch, f]
            
            # State machine implementation (same as original)
            if states[ch] == 0:  # 'init'
                buffers[ch, init_inds[ch]] = frame_val
                
                if init_inds[ch] >= buffers.shape[1]-1:
                    # Complete initialization and transition to fit state
                    moving_means[ch] = np.mean(buffers[ch])
                    # Compute W from scratch when entering the fit state
                    W_vectors[ch] = compute_W(buffers[ch], N, N)
                    states[ch] = 1  # Change to 'fit' state
                    clean_values[ch, f] = 0.0
                    artifacts[ch, f] = 0.0
                    spikes[ch, f] = False
                    if (ch == 0):
                        print("Count of zeros", np.count_nonzero(buffers[ch]==0))
                    continue
                else:
                    init_inds[ch] += 1
                    artifacts[ch, f] = 0.0
                    spikes[ch, f] = False
                    clean_values[ch, f] = 0
                    continue
            
            elif states[ch] == 1:  # 'fit'
                # Shift buffer and add new frame
                outgoing = buffers[ch, 0]
                buffers[ch] = shift_buffer(buffers[ch], frame_val)
                
                # Check for artifact
                if buffers[ch][N] - moving_means[ch] > max_val or buffers[ch][N] - moving_means[ch] < min_val:
                    states[ch] = 2  # Change to 'depeg' state
                    need_resets[ch] = 2*N + 1 + artifact_width
                    
                    # Mark removed frames
                    clean_values[ch, f] = 0.0
                    artifacts[ch, f] = buffers[ch][N]
                    spikes[ch, f] = False
                    continue
                
                # Update moving mean
                moving_means[ch] = fast_mmean(moving_means[ch], frame_val)
                
                # Update W recursively
                Wp_vectors[ch,:] = W_vectors[ch,:].copy()
                W_vectors[ch,:] = compute_W_rec(Wp_vectors[ch,:], N, N, buffers[ch,:], outgoing)
                
                # Compute linear fit coefficients
                a = np.zeros(2, dtype=np.float64)  # Only 2 coefficients for linear fit
                a = compute_a(S_matrices[ch], W_vectors[ch], a)
                
                # Calculate clean value and artifact (a[0] is the constant term - the artifact)
                clean_values[ch, f] = buffers[ch, N] - a[0]
                artifacts[ch, f] = a[0]
                
                # Check for spike
                if clean_values[ch, f] < spike_thresh_min and clean_values[ch, f] > spike_thresh_max and not spike_flags[ch]:
                    spike_flags[ch] = True
                    spikes[ch, f] = True
                elif clean_values[ch, f] > spike_thresh_min and spike_flags[ch]:
                    spike_flags[ch] = False
                    spikes[ch, f] = False
                else:
                    spikes[ch, f] = False
                continue
            
            elif states[ch] == 2:  # 'depeg' (saturation period)
                # During artifact period
                if need_resets[ch] > 0:
                    # During artifact period, shift buffer with actual frame
                    buffers[ch] = shift_buffer(buffers[ch], frame_val)
                    need_resets[ch] -= 1
                    
                    # Mark as artifact
                    clean_values[ch, f] = 0
                    artifacts[ch, f] = buffers[ch,N]
                    spikes[ch, f] = False
                    
                    # Critical: When need_reset becomes 0, recompute W from scratch
                    if need_resets[ch] == 0:
                        states[ch] = 1
                        moving_means[ch] = np.mean(buffers[ch])
                        # Compute W from scratch when entering the fit state
                        W_vectors[ch] = compute_W(buffers[ch], N, N)
                    continue

class LinearArtifactRemoval:
    """Optimized multi-channel artifact removal with linear fitting."""
    
    def __init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, 
                 spike_thresh=[-3.5, 20], batch_size=None):
        """Initialize optimized multi-channel processor with linear fit.
        
        Parameters:
        -----------
        n_channels : int
            Number of channels to process
        N : int
            Half-window size for linear fitting
        nc_start : int
            Starting index for processing
        min_val, max_val : float
            Thresholds for artifact detection
        spike_thresh : list of float
            Thresholds for spike detection, spike is detected when 
            the value is less than spike_thresh[0] and greater than spike_thresh[1]
        batch_size : int or None
            Size of channel batches to process in parallel
        """
        self.n_channels = n_channels
        self.N = N
        self.nc_start = nc_start
        self.min_val = min_val
        self.max_val = max_val
        self.spike_thresh_min = spike_thresh[0]
        self.spike_thresh_max = spike_thresh[1]
        
        # Determine channel batch size - optimize for AVX-512
        if batch_size is None:
            self.batch_size = 8 * ((n_channels + 7) // 8)  # Round up to multiple of 8
        else:
            self.batch_size = batch_size
        
        # Pre-allocate all arrays for channels
        buffer_size = 2*N + 1
        
        # Arrays for state
        self.states = np.zeros(n_channels, dtype=np.int64)  # 0=init, 1=fit, 2=depeg
        self.init_inds = np.zeros(n_channels, dtype=np.int64)
        self.moving_means = np.zeros(n_channels, dtype=np.float64)
        self.need_resets = np.zeros(n_channels, dtype=np.int64)
        self.spike_flags = np.zeros(n_channels, dtype=np.bool_)
        
        # Arrays for buffers and calculations - simplified for linear fit
        self.buffers = np.zeros((n_channels, buffer_size), dtype=np.float64)
        self.W_vectors = np.zeros((n_channels, 2), dtype=np.float64)  # Only 2 elements for linear fit
        self.Wp_vectors = np.zeros((n_channels, 2), dtype=np.float64)  # Only 2 elements for linear fit
        
        # Pre-compute S matrices for each channel - simplified for linear fit
        self.S_matrices = np.zeros((n_channels, 2, 2), dtype=np.float64)  # 2x2 matrix for linear fit
        T = compute_T(nc_start, N)
        S_inv = compute_S(T)  # Note: compute_S returns the INVERSE of S
        for i in range(n_channels):
            self.S_matrices[i] = S_inv
        
        # Create channel batches
        self.channel_batches = []
        for i in range(0, n_channels, self.batch_size):
            end = min(i + self.batch_size, n_channels)
            self.channel_batches.append(np.array(list(range(i, end)), dtype=np.int64))
        
        # Performance tracking
        self.timing = {'total': 0, 'processing': 0}
    
    def fit_step(self, frames_batch, artifact_width=60, remove_frames_before=8):
        """Process multiple frames across all channels using batch parallelism.
        
        Parameters:
        -----------
        frames_batch : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Batch of frames for all channels
        artifact_width, remove_frames_before :
            Parameters for artifact handling
            
        Returns:
        --------
        clean_values : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Cleaned values for all channels
        artifacts : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Artifact estimates for all channels
        spikes : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Spike detection flags for all channels
        """
        start_time = time.perf_counter()
        
        n_frames = frames_batch.shape[1]
        
        # Initialize output arrays
        clean_values = np.zeros((self.n_channels, n_frames), dtype=np.float64)
        artifacts = np.zeros((self.n_channels, n_frames), dtype=np.float64)
        spikes = np.zeros((self.n_channels, n_frames), dtype=np.bool_)
        
        # Convert frames to float64
        frames_f64 = frames_batch.astype(np.float64)
        
        # Process all channel batches
        for batch in self.channel_batches:
            process_time = time.perf_counter()
            process_channel_minibatch_numba(
                batch, frames_f64, clean_values, artifacts, spikes,
                artifact_width, remove_frames_before,
                self.states, self.init_inds, self.moving_means, 
                self.need_resets, self.spike_flags,
                self.buffers, self.W_vectors, self.Wp_vectors, self.S_matrices,
                self.min_val, self.max_val, self.spike_thresh_min, self.spike_thresh_max
            )
            self.timing['processing'] += time.perf_counter() - process_time
        
        self.timing['total'] += time.perf_counter() - start_time
        return clean_values, artifacts, spikes
    

    def warmup(self, n_samples=1000, minibatch_size = 40):
        """Warm up the artifact removal by processing dummy data.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to process for warmup. Default is 100.
        """
        # Generate random data for warmup
        dummy_data = np.random.randn(self.n_channels, n_samples)
        
        # Process the dummy data
        for i in range(0,n_samples, minibatch_size):
            self.fit_step(dummy_data[:, i:i+minibatch_size])


class Timer:
    def __init__(self, name='Timer'):
        self.name = name
        self.start_time = None
        self.end_time = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time() - self.start_time
        if exc_val:
            raise exc_val
        else:
            print(f'{self.name} End time: {self.end_time:.3f}s')


# Optimized version with better memory layout and vectorization
@njit(fastmath=True, cache=True)
def compute_W_vectorized(v, nc, N, W_out):
    """Vectorized W computation with pre-allocated output."""
    # Compute W[k] = sum from n=0 to nc+N of (n-nc)^k * v[n]
    # For k=0: sum of all values from 0 to nc+N
    W_out[0] = np.sum(v[:nc+N+1])
    
    # For k=1: weighted sum
    total = 0.0
    for n in range(nc+N+1):
        total += (n - nc) * v[n]
    W_out[1] = total

@njit(fastmath=True, cache=True)
def compute_W_rec_optimized(Wp, N, nc, v, W_out, outgoing):
    """Optimized recursive W computation."""
    # Based on the recursive formula from compute_W_rec
    # For k=0: W[0] = Wp[0] + v[nc+N] - v[nc-N]
    W_out[0] = Wp[0] + v[nc+N] - outgoing
    
    # For k=1: Using binomial expansion
    # W[1] = -Wp[0] + Wp[1] + N*v[nc+N] - (-N-1)*v[nc-N]
    W_out[1] = -Wp[0] + Wp[1] + N*v[nc+N] + (N+1)*outgoing

@njit(fastmath=True, cache=True)
def compute_linear_fit_inline(S_inv_00, S_inv_01, S_inv_11, W0, W1):
    """Inline computation of linear fit coefficients using pre-computed S inverse."""
    # We already have S_inv, so just do the matrix multiplication
    # a = S_inv @ W
    a0 = S_inv_00 * W0 + S_inv_01 * W1
    a1 = S_inv_01 * W0 + S_inv_11 * W1
    
    return a0, a1

@njit(parallel=True, fastmath=True, cache=True)
def process_channels_optimized(channels_data, clean_out, artifacts_out, spikes_out,
                              states, init_inds, moving_means, need_resets, spike_flags,
                              buffers, W_current, W_prev, S_inv_params,
                              min_val, max_val, spike_thresh_min, spike_thresh_max,
                              artifact_width, N):
    """Optimized processing with better memory access patterns."""
    n_channels, n_frames = channels_data.shape
    buffer_size = 2 * N + 1
    
    # Process channels in parallel
    for ch in prange(n_channels):
        # Extract S inverse matrix parameters for this channel
        S_inv_00 = S_inv_params[ch, 0]
        S_inv_01 = S_inv_params[ch, 1]
        S_inv_11 = S_inv_params[ch, 2]
        
        # Local copies for better cache locality
        state = states[ch]
        init_ind = init_inds[ch]
        moving_mean = moving_means[ch]
        need_reset = need_resets[ch]
        spike_flag = spike_flags[ch]
        
        # Process frames sequentially for this channel
        for f in range(n_frames):
            frame_val = channels_data[ch, f]
            
            if state == 0:  # Init state
                buffers[ch, init_ind] = frame_val
                
                if init_ind >= buffer_size - 1:
                    # Transition to fit state
                    moving_mean = np.mean(buffers[ch])
                    compute_W_vectorized(buffers[ch], N, N, W_current[ch])
                    state = 1
                    clean_out[ch, f] = 0.0
                    artifacts_out[ch, f] = 0.0
                    spikes_out[ch, f] = False
                else:
                    init_ind += 1
                    clean_out[ch, f] = 0.0
                    artifacts_out[ch, f] = 0.0
                    spikes_out[ch, f] = False
                    
            elif state == 1:  # Fit state
                # Shift buffer inline for better performance
                outgoing = buffers[ch, 0]
                for i in range(buffer_size - 1):
                    buffers[ch, i] = buffers[ch, i + 1]
                buffers[ch, buffer_size - 1] = frame_val
                
                center_val = buffers[ch, N]
                diff = center_val - moving_mean
                
                # Check for artifact
                if diff > max_val or diff < min_val:
                    state = 2
                    need_reset = 2 * N + 1 + artifact_width
                    clean_out[ch, f] = 0.0
                    artifacts_out[ch, f] = center_val
                    spikes_out[ch, f] = False
                else:
                    # Update moving mean
                    moving_mean = 0.8 * moving_mean + 0.2 * frame_val
                    
                    # Update W recursively
                    W_prev[ch, 0] = W_current[ch, 0]
                    W_prev[ch, 1] = W_current[ch, 1]
                    compute_W_rec_optimized(W_prev[ch], N, N, buffers[ch], W_current[ch], outgoing)
                    
                    # Compute linear fit
                    a0, a1 = compute_linear_fit_inline(S_inv_00, S_inv_01, S_inv_11, 
                                                      W_current[ch, 0], W_current[ch, 1])
                    
                    # Calculate clean value
                    clean_val = center_val - a0
                    clean_out[ch, f] = clean_val
                    artifacts_out[ch, f] = a0
                    
                    # Spike detection
                    if clean_val < spike_thresh_min and clean_val > spike_thresh_max and not spike_flag:
                        spike_flag = True
                        spikes_out[ch, f] = True
                    elif clean_val > spike_thresh_min and spike_flag:
                        spike_flag = False
                        spikes_out[ch, f] = False
                    else:
                        spikes_out[ch, f] = False
                        
            else:  # Depeg state
                # Shift buffer
                for i in range(buffer_size - 1):
                    buffers[ch, i] = buffers[ch, i + 1]
                buffers[ch, buffer_size - 1] = frame_val
                
                need_reset -= 1
                clean_out[ch, f] = 0.0
                artifacts_out[ch, f] = buffers[ch, N]
                spikes_out[ch, f] = False
                
                if need_reset == 0:
                    state = 1
                    moving_mean = np.mean(buffers[ch])
                    compute_W_vectorized(buffers[ch], N, N, W_current[ch])
        
        # Write back state variables
        states[ch] = state
        init_inds[ch] = init_ind
        moving_means[ch] = moving_mean
        need_resets[ch] = need_reset
        spike_flags[ch] = spike_flag


@njit(parallel=True, cache=True)
def process_saturation_linear(frames, clean, artifact, spikes, states, init_inds,
                             rail_counts, spike_flags, buffers, W_current, W_prev,
                             S, rail_min, rail_max, spike_min, spike_max, N):
    """Use the original polynomial algebra, excluding rail-contaminated windows.

    Outputs retain the original N-sample delay. Startup and any window containing
    a rail or nonfinite value return NaN, never fabricated recovered zeros.
    """
    width = 2 * N + 1
    for ch in prange(frames.shape[0]):
        count = rail_counts[ch]
        for f in range(frames.shape[1]):
            value = frames[ch, f]
            bad = not np.isfinite(value) or value <= rail_min or value >= rail_max
            clean[ch, f] = np.nan
            artifact[ch, f] = np.nan
            spikes[ch, f] = False
            if states[ch] == 0:
                buffers[ch, init_inds[ch]] = value
                count += int(bad)
                init_inds[ch] += 1
                if init_inds[ch] == width:
                    states[ch] = 2
                continue
            outgoing = buffers[ch, 0]
            old_bad = not np.isfinite(outgoing) or outgoing <= rail_min or outgoing >= rail_max
            for i in range(width - 1):
                buffers[ch, i] = buffers[ch, i + 1]
            buffers[ch, width - 1] = value
            count += int(bad) - int(old_bad)
            if count:
                states[ch] = 2
                spike_flags[ch] = False
                continue
            if states[ch] == 2:
                compute_W_vectorized(buffers[ch], N, N, W_current[ch])
                states[ch] = 1
            else:
                W_prev[ch, :] = W_current[ch, :]
                compute_W_rec_optimized(W_prev[ch], N, N, buffers[ch], W_current[ch], outgoing)
            a0, a1 = compute_linear_fit_inline(S[ch, 0], S[ch, 1], S[ch, 2], W_current[ch, 0], W_current[ch, 1])
            residual = buffers[ch, N] - a0
            clean[ch, f] = residual
            artifact[ch, f] = a0
            if residual < spike_min and residual > spike_max and not spike_flags[ch]:
                spike_flags[ch] = True
                spikes[ch, f] = True
            elif residual > spike_min and spike_flags[ch]:
                spike_flags[ch] = False
        rail_counts[ch] = count


class LinearArtifactRemoval2:
    """Highly optimized multi-channel artifact removal with linear fitting.

    ``blanking_mode='saturation'`` requires finite ``rail_min``/``rail_max``
    in input units. It returns NaN during startup and while any sample in the
    centered fitting window reaches a rail or is nonfinite. This excludes N
    samples on either side of clipping, with no additional recovery timeout;
    artifact_width and remove_frames_before do not apply in this mode.
    The default ``'threshold'`` mode retains the original behavior.
    
    Optimizations:
    - Better memory layout for cache efficiency
    - Pre-computed S matrix inverse parameters
    - Vectorized operations where possible
    - Reduced function call overhead
    - Optimized buffer shifting
    - Better parallel work distribution
    """
    
    def __init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, 
                 spike_thresh=[-3.5, 20], n_threads=None, *,
                 blanking_mode="threshold", rail_min=None, rail_max=None):
        """Initialize optimized processor.
        
        Parameters:
        -----------
        n_channels : int
            Number of channels to process
        N : int
            Half-window size for linear fitting
        nc_start : int
            Starting index for processing
        min_val, max_val : float
            Thresholds for artifact detection
        spike_thresh : list of float
            Thresholds for spike detection
        n_threads : int or None
            Number of threads to use (None for auto)
        """
        if blanking_mode not in ("threshold", "saturation"):
            raise ValueError("blanking_mode must be 'threshold' or 'saturation'")
        if blanking_mode == "saturation":
            if rail_min is None or rail_max is None or not np.isfinite(rail_min) or not np.isfinite(rail_max) or rail_min >= rail_max:
                raise ValueError("Saturation mode requires finite rail_min < rail_max")
        self.blanking_mode = blanking_mode
        self.rail_min, self.rail_max = rail_min, rail_max
        self.rail_counts = np.zeros(n_channels, dtype=np.int64)
        self.n_channels = n_channels
        self.N = N
        self.nc_start = nc_start
        self.min_val = min_val
        self.max_val = max_val
        self.spike_thresh_min = spike_thresh[0]
        self.spike_thresh_max = spike_thresh[1]
        self.artifact_width = 60  # Default value
        
        # Set number of threads
        if n_threads is not None:
            import numba
            numba.set_num_threads(n_threads)
        
        buffer_size = 2 * N + 1
        
        # State arrays - aligned for better cache performance
        self.states = np.zeros(n_channels, dtype=np.int64)
        self.init_inds = np.zeros(n_channels, dtype=np.int64)
        self.moving_means = np.zeros(n_channels, dtype=np.float64)
        self.need_resets = np.zeros(n_channels, dtype=np.int64)
        self.spike_flags = np.zeros(n_channels, dtype=np.bool_)
        
        # Buffers with better memory layout
        self.buffers = np.zeros((n_channels, buffer_size), dtype=np.float64, order='C')
        self.W_current = np.zeros((n_channels, 2), dtype=np.float64, order='C')
        self.W_prev = np.zeros((n_channels, 2), dtype=np.float64, order='C')
        
        # Pre-compute S matrix inverse parameters
        T = compute_T(nc_start, N)
        S_inv = compute_S(T)  # Note: compute_S returns the INVERSE of S
        
        # Store the S_inv values directly since compute_S already returns the inverse
        self.S_inv_params = np.zeros((n_channels, 3), dtype=np.float64)
        for i in range(n_channels):
            self.S_inv_params[i, 0] = S_inv[0, 0]
            self.S_inv_params[i, 1] = S_inv[0, 1]
            self.S_inv_params[i, 2] = S_inv[1, 1]
        
        # Pre-allocate output arrays
        self._clean_buffer = None
        self._artifacts_buffer = None
        self._spikes_buffer = None
        
        # Timing
        self.timing = {'total': 0, 'processing': 0}
    
    def fit_step(self, frames_batch, artifact_width=60, remove_frames_before=8):
        """Process multiple frames across all channels.
        
        Parameters:
        -----------
        frames_batch : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Batch of frames for all channels
        artifact_width : int
            Width of artifact window
        remove_frames_before : int
            Frames to remove before artifact (unused in current implementation)
            
        Returns:
        --------
        clean_values : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Cleaned values for all channels
        artifacts : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Artifact estimates for all channels
        spikes : numpy.ndarray, shape (n_channels, n_frames_in_batch)
            Spike detection flags for all channels
        """
        start_time = time.perf_counter()
        
        n_channels, n_frames = frames_batch.shape
        self.artifact_width = artifact_width
        
        # Ensure input is float64 and C-contiguous
        if not frames_batch.flags['C_CONTIGUOUS'] or frames_batch.dtype != np.float64:
            frames_f64 = np.ascontiguousarray(frames_batch, dtype=np.float64)
        else:
            frames_f64 = frames_batch
        
        # Allocate or reuse output buffers
        if self._clean_buffer is None or self._clean_buffer.shape != (n_channels, n_frames):
            self._clean_buffer = np.zeros((n_channels, n_frames), dtype=np.float64, order='C')
            self._artifacts_buffer = np.zeros((n_channels, n_frames), dtype=np.float64, order='C')
            self._spikes_buffer = np.zeros((n_channels, n_frames), dtype=np.bool_, order='C')
        
        # Process all channels
        process_time = time.perf_counter()
        if self.blanking_mode == "saturation":
            process_saturation_linear(
                frames_f64, self._clean_buffer, self._artifacts_buffer, self._spikes_buffer,
                self.states, self.init_inds, self.rail_counts, self.spike_flags,
                self.buffers, self.W_current, self.W_prev, self.S_inv_params,
                self.rail_min, self.rail_max, self.spike_thresh_min, self.spike_thresh_max, self.N)
            self.timing['processing'] += time.perf_counter() - process_time
            self.timing['total'] += time.perf_counter() - start_time
            return self._clean_buffer.copy(), self._artifacts_buffer.copy(), self._spikes_buffer.copy()
        process_channels_optimized(
            frames_f64, self._clean_buffer, self._artifacts_buffer, self._spikes_buffer,
            self.states, self.init_inds, self.moving_means, self.need_resets, self.spike_flags,
            self.buffers, self.W_current, self.W_prev, self.S_inv_params,
            self.min_val, self.max_val, self.spike_thresh_min, self.spike_thresh_max,
            self.artifact_width, self.N
        )
        self.timing['processing'] += time.perf_counter() - process_time
        
        self.timing['total'] += time.perf_counter() - start_time
        
        # Return copies to prevent external modification
        return self._clean_buffer.copy(), self._artifacts_buffer.copy(), self._spikes_buffer.copy()
    
    def warmup(self, n_samples=1000, minibatch_size=200):
        """Warm up the processor with dummy data.
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples for warmup
        minibatch_size : int
            Size of each minibatch
        """
        # Generate dummy data
        dummy_data = np.random.randn(self.n_channels, n_samples).astype(np.float64)
        
        # Process in minibatches
        for i in range(0, n_samples, minibatch_size):
            end = min(i + minibatch_size, n_samples)
            self.fit_step(dummy_data[:, i:end])
        
        # Reset timing after warmup
        self.timing = {'total': 0, 'processing': 0}













# Example usage
if __name__ == "__main__":
    # Performance test parameters
    n_channels = 1000
    n_frames = 200
    n_iterations = 5000
    
    print(f"Performance Benchmark: {n_channels} channels x {n_frames} frames x {n_iterations} iterations")
    print("=" * 80)
    
    # Generate test data
    print("Generating test data...")
    frames = np.random.randn(n_channels, n_frames).astype(np.float64)
    
    # Test LinearArtifactRemoval (original)
    print(f"\n1. Testing LinearArtifactRemoval (original) - {n_iterations} iterations...")
    processor1 = LinearArtifactRemoval(n_channels, N=60)
    
    # Warmup
    print("   Warming up...")
    processor1.warmup(n_samples=500, minibatch_size=200)
    processor1.timing = {'total': 0, 'processing': 0}  # Reset timing after warmup
    
    # Benchmark
    print("   Running benchmark...")
    start_time = time.time()
    for i in range(n_iterations):
        clean_values1, artifacts1, spikes1 = processor1.fit_step(frames.copy())
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"   Iteration {i+1:4d}: {rate:.1f} iterations/sec")
    
    total_time1 = time.time() - start_time
    rate1 = n_iterations / total_time1
    
    print(f"\nOriginal Results:")
    print(f"   Total time: {total_time1:.3f}s")
    print(f"   Rate: {rate1:.2f} iterations/sec")
    print(f"   Processing time: {processor1.timing['processing']:.3f}s")
    print(f"   Processing rate: {n_iterations / processor1.timing['processing']:.2f} iterations/sec")
    
    # Test LinearArtifactRemoval2 (optimized)
    print(f"\n2. Testing LinearArtifactRemoval2 (optimized) - {n_iterations} iterations...")
    processor2 = LinearArtifactRemoval2(n_channels, N=60)
    
    # Warmup
    print("   Warming up...")
    processor2.warmup(n_samples=500, minibatch_size=200)
    processor2.timing = {'total': 0, 'processing': 0}  # Reset timing after warmup
    
    # Benchmark
    print("   Running benchmark...")
    start_time = time.time()
    for i in range(n_iterations):
        clean_values2, artifacts2, spikes2 = processor2.fit_step(frames.copy())
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"   Iteration {i+1:4d}: {rate:.1f} iterations/sec")
    
    total_time2 = time.time() - start_time
    rate2 = n_iterations / total_time2
    
    print(f"\nOptimized Results:")
    print(f"   Total time: {total_time2:.3f}s")
    print(f"   Rate: {rate2:.2f} iterations/sec")
    print(f"   Processing time: {processor2.timing['processing']:.3f}s")
    print(f"   Processing rate: {n_iterations / processor2.timing['processing']:.2f} iterations/sec")
    
    # Performance comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"Total time speedup:      {total_time1/total_time2:.2f}x")
    print(f"Processing time speedup: {processor1.timing['processing']/processor2.timing['processing']:.2f}x")
    print(f"Rate improvement:        {rate2/rate1:.2f}x")
    
    # Data throughput analysis
    total_samples = n_channels * n_frames * n_iterations
    throughput1 = total_samples / total_time1 / 1e6  # Million samples per second
    throughput2 = total_samples / total_time2 / 1e6  # Million samples per second
    
    print(f"\nData Throughput:")
    print(f"   Original:  {throughput1:.2f} million samples/sec")
    print(f"   Optimized: {throughput2:.2f} million samples/sec")
    print(f"   Improvement: {throughput2/throughput1:.2f}x")
    
    # Correctness check
    print(f"\n{'='*80}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*80}")
    clean_close = np.allclose(clean_values1, clean_values2, rtol=1e-10, atol=1e-10)
    artifacts_close = np.allclose(artifacts1, artifacts2, rtol=1e-10, atol=1e-10)
    spikes_equal = np.array_equal(spikes1, spikes2)
    
    print(f"Clean values match:  {clean_close}")
    print(f"Artifacts match:     {artifacts_close}")
    print(f"Spikes match:        {spikes_equal}")
    
    if not (clean_close and artifacts_close and spikes_equal):
        print("\nWARNING: Results differ between implementations!")
        if not clean_close:
            max_diff = np.max(np.abs(clean_values1 - clean_values2))
            print(f"   Max clean value difference: {max_diff}")
        if not artifacts_close:
            max_diff = np.max(np.abs(artifacts1 - artifacts2))
            print(f"   Max artifact difference: {max_diff}")
    else:
        print("✓ All outputs match - optimization is correct!")
