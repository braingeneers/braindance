import numpy as np
from numba import jit, njit

@njit
def cubic_fit5(v, N=60, nc_start=60, mean_arr=None, mean_std=None, artifact_width=50, remove_frames_before=8):
    n_points = len(v)
    v_cleaned = np.zeros(n_points, dtype=np.float32)
    nc = nc_start
    
    def compute_T(nc):
        T = np.zeros((7))
        for k in range(7):
            for n in range(nc - N, nc + N + 1):
                T[k] += (n - nc) ** k
        return T
    

    def compute_S(T):
        S = np.zeros((4, 4))
        for k in range(4):
            for l in range(4):
                S[k, l] = T[k + l]
        return np.linalg.inv(S)
    

    def compute_W(v, nc, N):
        W = np.zeros(4)
        range_start = max(0, nc - N)
        range_end = min(n_points, nc + N + 1)
        for k in range(4):
            total = 0.0
            for n in range(range_start, range_end):
                total += (n - nc) ** k * v[n]
            W[k] = total
        return W
    

    def compute_W_rec(Wp, N, nc, v):
        W = np.zeros(4)
        for k in range(4):
            cur_sum = 0
            for l in range(k+1):
                num = (-1)**(k-l)*fast_factorial(k)
                den = fast_factorial(l)*fast_factorial(k-l)
                cur_sum += num/den*Wp[l] 
            cur_sum += (N**k) * v[nc+N + 1] - ((-N - 1)**k) * v[nc - N]
            W[k] = cur_sum
        return W

    
    def compute_a(S, W):
        a = np.zeros(4)
        for k in range(4):
            total = 0.0
            for l in range(4):
                total += S[k, l] * W[l]
            a[k] = total
        return a

    def deviation(v, A, nc, N, d):
        total = 0.0
        A_start = max(0, nc - N)
        A_end = min(n_points, nc + N)
        for n in range(A_start, A_start + d - 1):
            total += v[n] - A[n - A_start]
        return total
    
    def initialize(v,art, A, nc, nc_start, a):
        for n in range(-1, nc_start):
            A[n] = a[0] + a[1] * (n - nc) + a[2] * (n - nc) ** 2 + a[3] * (n - nc) ** 3
        v_cleaned[:nc_start] = v[:nc_start] - A[:nc_start]
        art[:nc_start] = A[:nc_start]

        return v_cleaned, art
    
    def reinitialize(v,art, A, nc, nc_start, a):
        for n in range(-1, nc_start):
            A[n] = a[0] + a[1] * (n - nc) + a[2] * (n - nc) ** 2 + a[3] * (n - nc) ** 3
        return A[:nc_start]
    
    art = np.zeros(n_points)
    W = compute_W(v, nc, N)

    T = compute_T(nc)
    S = compute_S(T)
    a = compute_a(S, W)

    A = np.zeros(nc_start + 1)

    v_cleaned, art = initialize(v,art,A,nc,nc_start, a)

    need_reset = 0

    if mean_arr is None:
        moving_mean = np.mean(v[:nc_start])

    while nc < (n_points - N - 1):        

        # Check for an artifact
        if mean_arr is not None:
            if mean_arr[nc] - mean_arr[nc-1] > mean_std or mean_arr[nc] - mean_arr[nc-1] < -mean_std:
                v_cleaned[nc-remove_frames_before:nc] = 0
                art[nc-remove_frames_before:nc] = v[nc-remove_frames_before:nc]
                nc += 1
                # reset w
                need_reset = artifact_width
                continue
        else:
            if v[nc] - moving_mean > 50 or v[nc] - moving_mean < -50:
                v_cleaned[nc-remove_frames_before:nc] = 0
                art[nc-remove_frames_before:nc] = v[nc-remove_frames_before:nc]
                nc += 1
                # reset w
                need_reset = artifact_width
                continue
        
        # Period after saturation
        if need_reset > 0:
            v_cleaned[nc] = 0
            art[nc] = v[nc]

            need_reset -= 1
            nc += 1
            if need_reset == 0:
                W = compute_W(v, nc, N)
                a = compute_a(S, W)

            continue


        # Normal iterative removal
        if mean_arr is None:
            moving_mean = fast_mmean(moving_mean, v[nc])

        W = compute_W_rec(W, N, nc, v)
        a = compute_a(S, W)

        v_cleaned[nc] = v[nc] - a[0]
        art[nc] = a[0]
        nc += 1

    return v_cleaned, art


from numba import prange

@njit(parallel=True)
def mean_numba(a):

    res = []
    for i in prange(a.shape[1]):
        res.append(a[:, i].mean())

    return np.array(res)


@njit(parallel=True)
def cubic_fit2d(data, N=60, nc_start=60, return_artifacts=False, artifact_width=50,
                remove_frames_before=8,n_stds=2):
    n_channels = data.shape[0]
    v_cleaned = np.zeros(data.shape, dtype=np.float32)
    mean_arr = mean_numba(data)
    mean_std = np.std(mean_arr)*n_stds

    art = np.zeros(data.shape)
    for i in prange(n_channels):
        if return_artifacts:
            v_cleaned[i], art[i] = cubic_fit5(data[i], N, nc_start, mean_arr, mean_std,
                                               artifact_width=artifact_width,remove_frames_before=remove_frames_before)
        else:
            v_cleaned[i], _ = cubic_fit5(data[i], N, nc_start, mean_arr, mean_std,
                                         artifact_width=artifact_width,remove_frames_before=remove_frames_before)

    if return_artifacts:
        return v_cleaned, art
    else:
        return v_cleaned, None



LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

@njit
def fast_median(a):
    return np.median(a)

@njit
def fast_mmean(q, frame):
    q*= .8
    q += .2*frame
    return q