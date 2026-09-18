import zmq
import struct
import time
import numpy as np

from collections import namedtuple

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

frame_number = 0

n_channels = 1024
fs = 20000
start_time = time.time()

# Preallocate buffer for event data
events_data = [b'']*5

while True:
    # Create fake data using numpy
    frame_data = np.random.randint(0, 100, size=n_channels).tolist()
    
    # Generate some random SpikeEvent objects and pack them into binary format
    for i in range(1):
        event = SpikeEvent(frame=frame_number, channel=i, amplitude=np.random.uniform(-50, -10))
        packed_event = struct.pack(_spike_struct, *(list(event)))
        events_data[i] = packed_event

    # Concatenate the binary data into a single byte string
    packed_events_data = b''.join(events_data)

    # Pack the frame number as a long long
    packed_frame_number = struct.pack('Q', frame_number)

    publisher_unfiltered.send_multipart([packed_frame_number, bytes(frame_data), packed_events_data])
    publisher_filtered.send_multipart([packed_frame_number, bytes(frame_data), packed_events_data])

    # Increment the frame number
    frame_number += 1

    # Sleep for a bit to control the rate of data generation
    # time.sleep(1/fs)
