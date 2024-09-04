import zmq
import struct
import time
import random
import numpy as np

from braindance.analysis import data_loader

from collections import namedtuple

def get_sine_wave(n_channels=1024, fs=20000, total_time_steps=20000*30):
    time_array = np.arange(total_time_steps)/fs  # Time array in seconds

    # Pre-generate the frame data as a 2D array of sin waves for each channel
    frame_data = np.sin(2*np.pi*1*time_array)  # 1 Hz sin wave
    frame_data = np.tile(frame_data, (n_channels, 1))
    frame_data = frame_data.astype(np.float32)
    return frame_data




def run(data_path, random_events=True):
    context = zmq.Context()

    # Define the ZMQ publishers
    publisher_unfiltered = context.socket(zmq.PUB)
    publisher_unfiltered.bind("tcp://*:7204")

    publisher_filtered = context.socket(zmq.PUB)
    publisher_filtered.bind("tcp://*:7205")

    # SpikeEvent is a named tuple that represents a single spike event
    SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')

    _spike_struct = '8xLif'
    _spike_struct_size = struct.calcsize(_spike_struct)

    n_channels=1024
    fs=20000
    total_time_steps=20000*60
    
    if data_path == 'sine':
        frame_data = get_sine_wave(n_channels=n_channels, fs=fs, total_time_steps=total_time_steps)
    elif data_path == "manual":
        # # Set traces as diagonal line (helpful when checking if all frames are retrieved)
        total_time_steps=1000
        n_channels=942
        frame_data = np.arange(total_time_steps)
        frame_data = np.tile(frame_data, (n_channels, 1))
        frame_data = frame_data.astype(np.float32)
        
        # Traces stored as (num_channels, num_frames) in .npy file
        # frame_data = np.load("/data/MEAprojects/primary_mouse/patch_ground_truth/200724/2602/rt_sort/dl_model/230101/cell2/scaled_traces.npy", mmap_mode="r")
        # frame_data = np.load("/data/MEAprojects/BrainDance/braindance/core/data/test_rt_sort2/dl_model/scaled_traces.npy", mmap_mode="r")
        # n_channels = frame_data.shape[0]
        # frame_data = frame_data[:, :total_time_steps].astype(np.float32)
    else:
        if str(data_path).endswith(".h5"):
            total_time_steps = 20000*20
            frame_data = data_loader.load_data_maxwell(data_path, start=0, length=total_time_steps)
            n_channels = frame_data.shape[0]
            random_events=True
        elif str(data_path).endswith(".npy"):
            frame_data = np.load(data_path, mmap_mode="r")
            n_channels, total_time_steps = frame_data.shape
            random_events=False
        else:
            raise NotImplementedError(f"Cannot load recording {data_path} because recording file type is not implemented")


    # Pre-generate random events for each time step
    events_data = []
    for t in range(total_time_steps):
        events_at_t = []
        n_events = random.randint(1, 5)  # Random number of events at this time step
        for i in range(n_events):
            event = SpikeEvent(frame=t, channel=random.randint(0, n_channels), amplitude=random.uniform(-50, -10))
            packed_event = struct.pack(_spike_struct, *(list(event)))
            events_at_t.append(packed_event)
        events_data.append(events_at_t)
        if not random_events:
            break


    print("Pre-generated data complete, beginning transmission...")
    # Main loop
    time_prev = time.perf_counter()
    time_curr = time.perf_counter()
    frame_number = 0
    while True:
        # Get the data for this frame
        time_curr = time.perf_counter()
        if time_curr - time_prev < 1/fs:
            continue
        time_prev = time_curr
        frame = frame_data[:, frame_number]
        
        if random_events:
            events = events_data[frame_number]
        else:
            events = events_data[0]

        # Pack the frame number as a long long
        packed_frame_number = struct.pack('Q', frame_number)

        # Send the data
        publisher_unfiltered.send_multipart([packed_frame_number, frame.tobytes(), b''.join(events)])
        publisher_filtered.send_multipart([packed_frame_number, frame.tobytes(), b''.join(events)])

        # Increment the frame number, looping back to 0 when we reach the end
        frame_number = (frame_number + 1) % total_time_steps

if __name__ == '__main__':
    run()