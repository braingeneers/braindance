import numpy as np
from numba import njit, prange
from functools import wraps

def docstring_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@docstring_wrapper
@njit
def cubic_fit5(v, N=60, nc_start=60, mean_arr=None, mean_std=None, artifact_width=50, remove_frames_before=8):
    """
    Perform cubic fitting on the input data.

    Args:
        v (numpy.ndarray): Input data array.
        N (int): Window size for fitting. Default is 60.
        nc_start (int): Starting point for fitting. Default is 60.
        mean_arr (numpy.ndarray, optional): Array of mean values. Default is None.
        mean_std (float, optional): Standard deviation of mean values. Default is None.
        artifact_width (int): Width of artifact to remove. Default is 50.
        remove_frames_before (int): Number of frames to remove before artifact. Default is 8.

    Returns:
        tuple: A tuple containing:
            - v_cleaned (numpy.ndarray): Cleaned data array.
            - art (numpy.ndarray): Artifact array.
    """
    # ... (rest of the function code remains the same)

@docstring_wrapper
@njit(parallel=True)
def mean_numba(a):
    """
    Compute mean of 2D array along axis 0 using Numba.

    Args:
        a (numpy.ndarray): Input 2D array.

    Returns:
        numpy.ndarray: Array of mean values.
    """
    res = []
    for i in prange(a.shape[1]):
        res.append(a[:, i].mean())

    return np.array(res)

@docstring_wrapper
@njit(parallel=True)
def cubic_fit2d(data, N=60, nc_start=60, return_artifacts=False, artifact_width=50,
                remove_frames_before=8,n_stds=2):
    """
    Perform 2D cubic fitting on the input data.

    Args:
        data (numpy.ndarray): Input 2D data array.
        N (int): Window size for fitting. Default is 60.
        nc_start (int): Starting point for fitting. Default is 60.
        return_artifacts (bool): Whether to return artifacts. Default is False.
        artifact_width (int): Width of artifact to remove. Default is 50.
        remove_frames_before (int): Number of frames to remove before artifact. Default is 8.
        n_stds (int): Number of standard deviations for threshold. Default is 2.

    Returns:
        tuple: A tuple containing:
            - v_cleaned (numpy.ndarray): Cleaned 2D data array.
            - art (numpy.ndarray or None): Artifact 2D array if return_artifacts is True, else None.
    """
    # ... (rest of the function code remains the same)

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@docstring_wrapper
@njit
def fast_factorial(n):
    """
    Compute factorial using a lookup table for faster computation.

    Args:
        n (int): Input number.

    Returns:
        int: Factorial of n.

    Raises:
        ValueError: If n is greater than 20.
    """
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

@docstring_wrapper
@njit
def fast_median(a):
    """
    Compute median of an array using Numba for faster computation.

    Args:
        a (numpy.ndarray): Input array.

    Returns:
        float: Median of the input array.
    """
    return np.median(a)

@docstring_wrapper
@njit
def fast_mmean(q, frame):
    """
    Compute moving mean.

    Args:
        q (float): Current moving mean.
        frame (float): New value.

    Returns:
        float: Updated moving mean.
    """
    q *= .8
    q += .2*frame
    return q




class ArtifactRemoval:

    def __init__(self, N, nc_start=0, min_val=-100, max_val=100, spike_thresh = [-3.5, -20]):
        self.N = N
        self.nc_start = nc_start
        self.prev_a = None  # to store the previous coefficients
        self.prev_nc = nc_start
        self.v = np.zeros(2*N + 2)
        # self.v = deque(maxlen=2*N + 2)
        self.Wp = np.zeros(4)
        self.W = np.zeros(4)

        # Precompute
        self.T = self._compute_T(nc_start)
        self.S = self._compute_S(self.T)
        self.moving_mean = 0

        self.spike = False
        self.spike_thresh_min = spike_thresh[0]
        self.spike_thresh_max = spike_thresh[1]
        
        self.depeg_count = 0

        self.min_val = min_val
        self.max_val = max_val
        # self.S = np.zeros((4, 4)) # Make sure this is inited so it doesn't error
        # States: init, depeg, fit
        self.state = 'init'
        self.init_ind = 0
        # self.state_timers = {'init': 0, 'depeg': 0, 'fit': 0, 'init-fit': 0, 'fit-depeg': 0,
        #                       'depeg-init': 0, 'fit-shifting': 0, 'fit-median':0}
    

    def _compute_T(self, nc):
        T = np.zeros((7))
        for k in range(7):
            for n in range(nc - self.N, nc + self.N + 1):
                T[k] += (n - nc) ** k
        return T

    def _compute_S(self, T):
        S = np.zeros((4, 4))
        for k in range(4):
            for l in range(4):
                S[k, l] = T[k + l]
        return np.linalg.inv(S)

    def _compute_W(self, v, nc):
        n_points = len(v)
        W = np.zeros(4)
        range_start = max(0, nc - self.N)
        range_end = min(n_points, nc + self.N + 1)
        for k in range(4):
            total = 0.0
            for n in range(range_start, range_end):
                total += (n - nc) ** k * v[n]
            W[k] = total
        return W
    
    def _compute_W_step(self, v):
        # n_points = len(v)
        # W = np.zeros(4)
        # for k in range(4):
        #     total = 0.0
        #     for n in range(max(0, nc - self.N),self.N + 1):
        #         total += (n - nc) ** k * v[n]
        #     W[k] = total
        # return W
        pass

    @staticmethod
    @njit
    def _compute_W_rec(W, Wp, v, nc, N):
        # W = np.zeros(4)
        for k in range(4):
            cur_sum = 0
            for l in range(k+1):
                num = (-1)**(k-l)*fast_factorial(k)
                den = fast_factorial(l)*fast_factorial(k-l)
                cur_sum += num/den*Wp[l] 
            # print(v.shape)
            # print("nc + N + 1", nc + self.N + 1)
            # print("nc - N:", nc - self.N) # Was vvv + 1
            cur_sum += (N**k) * v[nc+N + 1] - ((-N - 1)**k) * v[nc - N]
            W[k] = cur_sum
        return W

    @staticmethod
    @njit
    def _compute_a(S, W):
        a = np.zeros(4)
        for k in range(4):
            total = 0.0
            for l in range(4):
                total += S[k, l] * W[l]
            a[k] = total
        return a
    

    def fit(self, v, T, S):
        # Length of points to fit
        n_points = len(v) - int(self.N+1)
        v_cleaned = np.zeros(n_points)
        nc = self.nc_start
        art = np.zeros(n_points)

        
        W = self._compute_W(v, nc)
        a = self._compute_a(self.S, W)

        A = np.zeros(self.nc_start + 1)
        for n in range(-1, self.nc_start):
            A[n] = a[0] + a[1] * (n - nc) + a[2] * (n - nc) ** 2 + a[3] * (n - nc) ** 3
        v_slice = list(itertools.islice(v, 0, self.nc_start))
        v_cleaned[:self.nc_start] = v_slice[:self.nc_start] - A[:self.nc_start]
        art[:self.nc_start] = A[:self.nc_start]

        Wp = W.copy()
        while nc < n_points:
            # W = self._compute_W(v, nc)
            Wp = W.copy()
            W = self._compute_W_rec(W,Wp, v, nc, self.N)
            a = self._compute_a(self.S, W)

            # A = np.zeros(2*self.N + 1)
            # for n in range(nc - self.N, nc + self.N):
            #     A[n - nc + self.N] = a[0] + a[1] * (n - nc) + a[2] * (n - nc) ** 2 + a[3] * (n - nc) ** 3
            v_cleaned[nc] = v[nc] - a[0]
            art[nc] = a[0]
            nc += 1

        self.W = W
        return v_cleaned, art
    
    
    def fit_step(self, frame):
        """ Recursively computes the new W
        then solves for the artifact of the center frame when the new frame is added
        """
        # t_start = time.perf_counter()
        if self.state == 'init':
            # Fill up v, when it is size 2N+1, run fit
            self.v[self.init_ind] = frame
            
            if self.init_ind >= 2*self.N + 1:
                self.moving_mean = np.mean(self.v)
                self.fit(self.v, self.T, self.S)
                self.state = 'fit'
                # self.state_timers['init-fit'] += time.perf_counter() - t_start
                return 0,0, False
            else:
                self.init_ind += 1
                # self.state_timers['init'] += time.perf_counter() - t_start
                return 0,0, False
        
        if self.state == 'fit':
            # Move v
            # median = fast_median(self.v)
            # t_start = time.perf_counter()
            if frame - self.moving_mean > self.max_val or frame - self.moving_mean < self.min_val:
                self.state = 'depeg'
                self.spike = False
                # self.state_timers['fit-depeg'] += time.perf_counter() - t_start
                return 0,0, False
            

            # t_start = time.perf_counter()
            self.moving_mean = fast_mmean(self.moving_mean, frame)
            # self.state_timers['fit-median'] += time.perf_counter() - t_start

            # t_start = time.perf_counter()
            self.shift_ind(self.v, frame)
            # self.state_timers['fit-shifting'] += time.perf_counter() - t_start

            # t_start = time.perf_counter()
            self.Wp = self.W.copy()
            self.W = self._compute_W_rec(self.W, self.Wp, self.v, self.N, self.N) # Could be N + 1
            a = self._compute_a(self.S, self.W)
            # self.state_timers['fit'] += time.perf_counter() - t_start

            new_spike = False
            
            cur_val = self.v[self.N + 1] - a[0]
            # Check if spike
            if (cur_val < self.spike_thresh_min and cur_val > self.spike_thresh_max
                        and self.spike == False):
                self.spike = True
                new_spike = True
            elif cur_val > self.spike_thresh_min and self.spike == True:
                self.spike = False

            return self.v[self.N+1] - a[0], a[0], new_spike
        
        # If we are saturating
        if self.state == 'depeg':
            # median = fast_median(self.v)
            # self.moving_mean*= .8
            # self.moving_mean += .2*frame
            self.depeg_count += 1
            # t_start = time.perf_counter()
            if (frame - self.moving_mean < self.max_val and frame - self.moving_mean > self.min_val) or (
                        self.depeg_count > 20):
                
                self.state = 'init'
                self.init_ind = 0
                # self.state_timers['depeg-init'] += time.perf_counter() - t_start
                self.depeg_count = 0
                return 0,0,False
            # Insert the median
            
            self.shift_ind(self.v, self.moving_mean)
            # self.state_timers['depeg'] += time.perf_counter() - t_start
            return 0,0,False

        
        return self.v[self.N] - a[0], a[0], False
    
    @staticmethod
    @njit
    def shift_ind(arr, val):
        arr[:-1] = arr[1:]
        arr[-1] = val
        return arr
    
    def run(self, data, return_artifacts=False, return_spikes=False, progress_bar=False, n_workers=1):
        '''Runs the process for the whole data input.
        If data is a 1D array, it will return the cleaned data
        If data is a 2D array, it will run for each row'''
        output_data = np.zeros(data.shape)
        output_art = np.zeros(data.shape)

        if progress_bar:
            from tqdm import tqdm
        else:
            def tqdm(x, *args, **kwargs):
                return x
        
        if len(data.shape) == 2:
            all_spikes = [] 
            for ch, full_trace in tqdm(enumerate(data)):
                output_data[ch], output_art[ch], spikes = self.run(full_trace, return_artifacts=True, return_spikes=True)
                all_spikes.append(spikes)
            if return_artifacts and return_spikes:
                return output_data, output_art, all_spikes
            elif return_artifacts:
                return output_data, output_art
            elif return_spikes:
                return output_data, all_spikes
            else:
                return output_data
            
        spike_times = []

        # Fit the first N points
        output_data[:self.nc_start], output_art[:self.nc_start] = self.fit(data[:2*self.N+1], self.T, self.S)
        
        for i, frame in tqdm(enumerate(data), total=len(data), desc="Processing data", leave=False):
            if i < self.nc_start:
                continue
            output_data[i-self.nc_start], output_art[i-self.nc_start], spike = self.fit_step(frame)
            if spike:
                spike_times.append(i-self.nc_start)

        if return_artifacts and return_spikes:
            return output_data, output_art, spike_times
        elif return_artifacts:
            return output_data, output_art
        elif return_spikes:
            return output_data, spike_times
        else:
            return output_data

    