import multiprocessing
from multiprocessing import shared_memory

import time
import numpy as np

from braindance.core.maxwell_env import MaxwellEnv

BUFFER_SIZE = 100
CHANS = 942
STIM_ELECS = np.array(range(10))
NUM_STIM_ELECS = len(STIM_ELECS)

maxwell_params = {
    "max_time_sec": 0.1,
    "save_dir": "/data/MEAprojects/BrainDance/delete_me",
    "name": "stim",
    "config": None,  # "/data/MEAprojects/maxwell_recordings/configs/240725/11h45m58s.cfg",
    "multiprocess": False,
    "render": False,
    "observation_type": "raw",
    "dummy": "manual",
    
    'stim_electrodes': []
}

def read_obs(obs_shm_name, obs_ready, obs_processed, stim_shm_name, env_done):
    env = MaxwellEnv(**maxwell_params)
    
    # Create shared memory block
    obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
    obs_array = np.ndarray((BUFFER_SIZE, CHANS), dtype=np.float16, buffer=obs_shm.buf)
    
    stim_shm = shared_memory.SharedMemory(name=stim_shm_name)
    stim_array = np.ndarray((NUM_STIM_ELECS,), dtype=bool, buffer=stim_shm.buf)
    
    obs_processed.set()
    env.clear_buffer()    
    done = False
    while not done:
        obs, done = env.step(buffer_size=BUFFER_SIZE)
        
        if np.any(stim_array):
            stim_elecs = STIM_ELECS[stim_array]
            stim_array[:] = False
            
        # # Wait until previous obs has been processed
        # obs_processed.wait()

        # Write data to shared memory
        start_time = time.perf_counter()
        obs_array[:] = obs
        obs_ready.set()  # Signal that obs is ready
        elapsed_time = time.perf_counter() - start_time
        print(f"Write time: {elapsed_time * 1000} ms")
        # obs_processed.wait()  # Wait until the reading process is done
    
    # obs_processed.wait()
    env_done.set()
    
    # Clean up
    # start_time = time.perf_counter()
    obs_shm.close()
    stim_shm.close()
    # elapsed_time = time.perf_counter() - start_time
    # print(f"Writing time: {elapsed_time * 1000} ms")

def process_obs(obs_shm_name, obs_ready, obs_processed, stim_shm_name, env_done):    
    # Create shared memory block
    obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
    obs_array = np.ndarray((BUFFER_SIZE, CHANS), dtype=np.float16, buffer=obs_shm.buf)
    
    stim_shm = shared_memory.SharedMemory(name=stim_shm_name)
    
    while not env_done.is_set():
        obs_ready.wait()    
        
        start_time = time.perf_counter()
        data = obs_array * 5
        elapsed_time = time.perf_counter() - start_time
        # print(f"Read time: {elapsed_time * 1000} ms")
        
        # idx = np.random.randint(0, NUM_STIM_ELECS)
        # stim_shm.buf[idx] = True
        
        # start_time = time.perf_counter()
        obs_ready.clear()
        # obs_processed.set() 
        # elapsed_time = time.perf_counter() - start_time
        # print(f"Clear time: {elapsed_time * 1000} ms")
    
    # Clean up
    # start_time = time.perf_counter()
    obs_shm.close()
    stim_shm.close()
    # elapsed_time = time.perf_counter() - start_time

if __name__ == "__main__":           
    obs_shm = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE*CHANS * np.dtype(np.float16).itemsize)    
    obs_ready = multiprocessing.Event()
    obs_processed = multiprocessing.Event()
    
    stim_shm = shared_memory.SharedMemory(create=True, size=max(16, NUM_STIM_ELECS))
    
    env_done = multiprocessing.Event()
    
    p1 = multiprocessing.Process(target=read_obs, args=(obs_shm.name, obs_ready, obs_processed, stim_shm.name, env_done))
    p2 = multiprocessing.Process(target=process_obs, args=(obs_shm.name, obs_ready, obs_processed, stim_shm.name, env_done))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    # Clean up shared memory
    obs_shm.unlink()
    stim_shm.unlink()