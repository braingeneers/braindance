import numpy as np
import matplotlib.pyplot as plt
import glob
from braindance.core.artifact_removal import ArtifactRemoval, ArtifactRemoval2
from braindance.analysis.data_loader import load_data_maxwell
import time
from braindance.core.artifact_removal import Timer

use_real_data = True
# Load a piece of test data
file_dir = '/media/danser-lab/hippo1/ephys/2023-10-13-e-cP001223-dryrun/original/'
data_filepath = sorted(glob.glob(file_dir + 'data/*.raw.h5'))[0]

full_length = 20000*10
if use_real_data:
    data_loaded = load_data_maxwell(data_filepath, channels=[50,100],start=130*20000, length=full_length)
    data = data_loaded[0]
    data2 = data_loaded[1]
    fake_art = np.zeros_like(data, dtype=np.float64)
else:
    # Test data
    data = np.arange(full_length)
    fake_art = np.sin(data/10) + np.sin(data/20) + np.sin(data/30)
    # fake_art = np.zeros_like(data, dtype=np.float64)
    # fake_art[150:180] = np.sin(data[150:180]/2)*200
    # fake_art[180:200] = np.cos(data[180:200]/2)*200
    # fake_art[200:250] = 0
    data = fake_art.copy()
    # Add some noise
    data += np.random.randn(len(data))

N = 60
nc_start = N

ar = ArtifactRemoval(N=N, nc_start=nc_start, min_val=-200, max_val=200)
# ar2 = ArtifactRemoval(N=N, nc_start=nc_start)

data_lag = N + 1

first_chunk = 100


clean_data = np.zeros_like(data)
# clean_data2 = np.zeros_like(data)
art = np.zeros_like(data)
# art2 = np.zeros_like(data)

print('Now fitting')
fig, ax = plt.subplots(2,1)
# Call functions for numba
# ar.fit(data[:first_chunk], ar.T, ar.S)
for i in range(1000):
    ar.fit_step(data[first_chunk])
# ar2.fit(data2[:first_chunk], ar.T, ar.S)
# ar2.fit_step(data2[first_chunk])
print("Beginning")

time_start = time.perf_counter()

skip_step = 0
art_step = 0
time_start = time.time()

tags = []
with Timer('steps'):
    for i in range(0, len(data)):
        # print(ar.state)
        # tags.append(ar.state)
        if skip_step != 0 and i % skip_step != 0:
            if ar.state == 'depeg':
                clean_data_step = 0
            else:
                clean_data_step = data[i] - art_step
        else:

            clean_data_step, art_step, spike = ar.fit_step(data[i])

        

        art[i-N] = art_step
        # art2[i-N] = art_step2
        clean_data[i-N] = clean_data_step

        if spike and False:
            ax[0].scatter(i-N, clean_data[i-N], c='r', marker='x')

        # clean_data2[i-N] = clean_data_step2
        if i > 20000*10:
            break

print(f'Time elapsed: {time.time() - time_start}')
print("Time sum",np.sum(list(ar.state_timers.values())))

print("State times")
print(ar.state_timers)


print(f"For {len(data)} samples")

print(ar.state_timers)
ax[0].plot(data)[:20000]
ax[0].plot(clean_data)[:20000]

# Plot the list of tag strings
# ax[0].plot(tags)


# ax[1].plot(data2)[:20000]
# ax[1].plot(clean_data2)[:20000]
ax[1].plot(art)[:1000]
plt.show()
raise Exception('Done')



# Fit the data
# clean_data[:first_chunk-data_lag], art[:first_chunk-data_lag] = ar.fit(data[:first_chunk])
# print('Now fitting')
# ar.v = data[:first_chunk][:2*N + 2]
# for i in range(first_chunk, len(data)):
#     print(ar.state)
#     clean_data_step, art_step = ar.fit_step(data[i])
#     # print(ar.v)
#     # print(clean_data_step)
#     art[i-N] = art_step
#     clean_data[i-N] = clean_data_step
#     if i > 1000:
#         break
# Now try step by step after

# Plot the data
fig, ax = plt.subplots(2,1)
ax[0].plot(data)
ax[0].plot(clean_data)
ax[1].plot(fake_art)
ax[1].plot(art)

plt.show()

# print("nc + N + 1", nc + self.N + 1)
# print("nc - N", nc - self.N)


# #2012 info processing capacity?!
# Unknown, it was a paper on how to do this
# Also 2004 paper on using median to calculate std 
# Set a beginning tester script to send, then receive data from japan
# Test realtime sorter latency
