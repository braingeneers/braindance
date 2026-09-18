import numpy as np
import matplotlib.pyplot as plt
import glob
from braindance.core.artifact_removal import ArtifactRemoval, ArtifactRemoval2
from braindance.analysis.data_loader import load_data_maxwell
import time


use_real_data = True
# Load a piece of test data
file_dir = '/media/danser-lab/hippo1/ephys/2023-10-13-e-cP001223-dryrun/original/'
data_filepath = sorted(glob.glob(file_dir + 'data/*.raw.h5'))[0]

full_length = 20000*10
if use_real_data:
    data_loaded = load_data_maxwell(data_filepath, channels=[0,100],start=120, length=full_length)
    data = data_loaded[0]
    data2 = data_loaded[1]
    fake_art = np.zeros_like(data, dtype=np.float64)
else:
    # Test data
    data = np.arange(full_length)
    fake_art = np.sin(data/10) + np.sin(data/20) + np.sin(data/30)
    data = fake_art.copy()
    # Add some noise
    data += np.random.randn(len(data))

N = 20
nc_start = 20

ar = ArtifactRemoval(N=N, nc_start=nc_start)
# ar2 = ArtifactRemoval(N=N, nc_start=nc_start)


from numba import njit


print(data.dtype)

a = np.random.random((2*N + 2,))


def time_median(a, new_data, iters=20000*10):
    for i in range(iters):
        # a[i%(2*N + 2)] = new_data[i]
        median = np.median(a)
    
@njit
def time_median_njit(a, new_data, iters=20000*10):
    for i in range(iters):
        # a[i%(2*N + 2)] = new_data[i]
        median = np.median(a)

def time_mean(a, new_data, iters=20000*10):
    for i in range(iters):
        a[i%(2*N + 2)] = new_data[i]
        mean = np.mean(a)

@njit
def time_mean_njit(a, new_data, iters=20000*10):
    for i in range(iters):
        a[i%(2*N + 2)] = new_data[i]
        mean = np.mean(a)


def time_moving_mean(a, new_data, iters=20000*10):
    moving_mean = np.mean(a)
    for i in range(iters):
        a[-1] = new_data[i]
        moving_mean*= .8
        moving_mean += .2*a[-1]

@njit
def time_moving_mean_njit(a, new_data, iters=20000*10):
    moving_mean = np.mean(a)
    for i in range(iters):
        a[-1] = new_data[i]
        moving_mean*= .8
        moving_mean += .2*a[-1]


import numpy as np
from numba import int64, float64, float32
from numba.experimental import jitclass

spec = [
    ('lower_half', float32[:]),
    ('upper_half', float32[:]),
]

@jitclass(spec)
class MovingMedian:
    def __init__(self, initial_data):
        sorted_data = np.sort(initial_data.astype(np.float32))
        mid = len(initial_data) // 2
        self.lower_half = np.array([-x for x in sorted_data[:mid]], dtype=np.float32)
        self.upper_half = sorted_data[mid:]
    
    def _push_heap(self, heap, val):
        heap[0], heap[-1] = heap[-1], heap[0]
        heap[-1] = val
        self._sift_down(heap, 0, len(heap) - 1)
    
    def _pop_heap(self, heap, negate=False):
        last_idx = len(heap) - 1
        last = heap[last_idx]
        heap = heap[:last_idx]  # Truncate the array
        if len(heap) > 0:
            return_item = heap[0]
            heap[0] = last
            self._sift_down(heap, 0, len(heap) - 1)
        else:
            return_item = last
        if negate:
            return -return_item, heap
        else:
            return return_item, heap
    
    def _sift_down(self, heap, start, end):
        root = start
        while True:
            child = root * 2 + 1
            if child > end:
                break
            if child + 1 <= end and heap[child] < heap[child + 1]:
                child += 1
            if heap[root] < heap[child]:
                heap[root], heap[child] = heap[child], heap[root]
                root = child
            else:
                break
    
    def add(self, num):
        if num <= -self.lower_half[0]:
            self._push_heap(self.lower_half, -num)
            if len(self.lower_half) - len(self.upper_half) > 1:
                moved_item, self.lower_half = self._pop_heap(self.lower_half, negate=True)
                self._push_heap(self.upper_half, moved_item)
        else:
            self._push_heap(self.upper_half, num)
            if len(self.upper_half) > len(self.lower_half):
                moved_item, self.upper_half = self._pop_heap(self.upper_half)
                self._push_heap(self.lower_half, -moved_item)


# Lets time how fast MovingMedian is 
def time_moving_median(a, new_data, iters=20000*10):
    moving_median = MovingMedian(a)
    for i in range(iters):
        a[-1] = new_data[i]
        moving_median.add(a[-1])


# ========== OPTIMIZED VERSION ==========

LOOKUP_TABLE_FACT = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

# @njit
# def fast_factorial(n):
#     result = 1
#     for i in range(1, n + 1):
#         result *= i
#     return result

def _compute_T( nc, N=20):
    T = np.zeros((7))
    for k in range(7):
        for n in range(nc - N, nc + N + 1):
            T[k] += (n - nc) ** k
    return T

def _compute_S(T):
    S = np.zeros((4, 4))
    for k in range(4):
        for l in range(4):
            S[k, l] = T[k + l]
    return np.linalg.inv(S)

@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE_FACT[n]

@njit
def _compute_W_rec(Wp, v, nc, N):
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
        Wp[k] = cur_sum
    return Wp

def _compute_W_rec2(Wp, v, nc, N):
    for k in range(4):
        print(k, '==========')
        cur_sum = 0
        for l in range(k+1):
            print(l, '------------')
            num = (-1)**(k-l)*fast_factorial(k)
            den = fast_factorial(l)*fast_factorial(k-l)
            print(num, den, ':', num/den)
            cur_sum += num/den*Wp[l] 
        cur_sum += (N**k) * v[nc+N + 1] - ((-N - 1)**k) * v[nc - N]
        Wp[k] = cur_sum
    return Wp

def precompute_num_den():
    arr = np.zeros((4,4))
    for k in range(4):
        cur_sum = 0
        for l in range(k+1):
            num = (-1)**(k-l)*fast_factorial(k)
            den = fast_factorial(l)*fast_factorial(k-l)
            # cur_sum += num/den
            arr[k,l] = num/den
    return arr

def precompute_n_to_k(N=20):
    arr = np.zeros((4))
    for k in range(4):
        arr[k] = N**k
    return arr

def precompute_np_to_k(N=20):
    arr = np.zeros((4))
    for k in range(4):
        arr[k] = (-N-1)**k
    return arr


LOOKUP_TABLE_NUM_DEN = precompute_num_den()
LOOKUP_TABLE_N_TO_K = precompute_n_to_k()
LOOKUP_TABLE_NP_TO_K = precompute_np_to_k()

# @njit
def _compute_W_rec_precomp(Wp, v, nc, N):
    for k in range(4):
        cur_sum = 0
        for l in range(k+1):
            cur_sum += LOOKUP_TABLE_NUM_DEN[k,l]*Wp[l] 
        cur_sum += (LOOKUP_TABLE_N_TO_K[k]) * v[nc+N + 1] - (LOOKUP_TABLE_NP_TO_K[k]) * v[nc - N]
        Wp[k] = cur_sum
    return Wp

@njit
def _compute_a(S, W, a):
    # a = np.zeros(4)
    for k in range(4):
        total = 0.0
        for l in range(4):
            total += S[k, l] * W[l]
        a[k] = total
    return a

# Vectorized
@njit
def _compute_a_vec(S, W, a):
    return np.dot(S, W)
    

# ~~~~~~~~~~~~~~~~ A ~~~~~~~~~~~~~~~~




# Lets time how fast a few options are
import time
from braindance.core.artifact_removal import Timer
test_time = False
test_opt = True

if test_opt:
    W = np.arange(4, dtype=np.float64)
    nc = 20
    N = 20
    v = np.random.random((2*N+2))

    a = np.zeros(4)
    
    S = _compute_S(_compute_T(nc, N))
    # Make S a float64
    S = S.astype(np.float64)

    # _compute_W_rec2(W, v, nc, N)

    with Timer('W rec'):
        for i in range(100000):
            w1 = _compute_W_rec(W.copy(), v, nc, N)
    print(w1)

    with Timer('W rec Precomp'):
        for i in range(100000):
            w2 = _compute_W_rec_precomp(W.copy(), v, nc, N)
    print(w2)

    with Timer('A normal'):
        for i in range(1000000):
            a1 = _compute_a(S, w1, a.copy())
    print(a1)

    with Timer('A vec'):
        for i in range(1000000):
            a2 = _compute_a_vec(S, w2, a.copy())
    print(a2)
    # print(w1.dtype, w2.dtype, a1.dtype, a2.dtype)
    

if test_time:

    time_start = time.perf_counter()
    time_median(a, data)
    print(f'Time elapsed for median: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_median_njit(a, data)
    print(f'Time elapsed for median njit: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_mean(a, data)
    print(f'Time elapsed for mean: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_mean_njit(a, data)
    print(f'Time elapsed for mean njit: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_moving_mean(a, data)
    print(f'Time elapsed for moving mean: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_moving_mean_njit(a, data)
    print(f'Time elapsed for moving mean njit: {time.perf_counter() - time_start}')

    time_start = time.perf_counter()
    time_moving_median(a, data)
    print(f'Time elapsed for moving median: {time.perf_counter() - time_start}')

