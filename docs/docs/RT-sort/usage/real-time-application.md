---
sidebar_position: 3
---

# Real-time Application (Online Use)

Once you have detected sequences, you can use RT-Sort for real-time applications. Here's how to do it:

## Reset the RTSort Object

Before online use, reset the RTSort object:

```python
rt_sort.reset()
```

**Note**: For optimal performance, call `rt_sort.reset()` if more than 50ms have passed since the last `rt_sort.running_sort(obs)` method call.

## Continuous Sorting

During online use, continuously call the following method to sort a stream of data:

```python
sequence_detections = rt_sort.running_sort(obs)
```

### Parameters

- `obs`: NumPy array with shape (num_frames, num_electrodes). This should contain only the most recent frames of data.

### Returns

A list where each element is a tuple of length 2 containing the data for a sorted spike in the `obs` recording chunk:
- 0th element: ID number of the sequence that the spike was assigned to
- 1st element: Time the spike occurred (in milliseconds)

### Notes

- The first 50ms of data passed as `obs` to `running_sort` is needed to initialize the RTSort object after the last `rt_sort.reset()`, so no spikes will be sorted during this time.
- `num_frames` must be at least 1 and can be different for each `running_sort` call.
- To minimize sorting latency, `num_frames` should ideally change and be equal to the amount of time that has passed since the last `running_sort` call.
- To align with the RT-Sort Methods, use 100 frames for 20kHz MEAs and 150 frames for 30kHz Neuropixels.

## Example with Maxwell MEA

Here's an example of how to use RT-Sort with a Maxwell MEA:

```python
from braindance.core.maxwell_env import MaxwellEnv

env = MaxwellEnv(**params)
done = False
while not done:
    obs, done = env.step(buffer_size=100)
    sequence_detections = rt_sort.running_sort(obs)
    # Process sequence_detections
```

## Example with Open Ephys GUI

Here's an example of how to use RT-Sort with a Maxwell MEA:

First, start the Open Ephys GUI. Then, add a [Record Node](https://open-ephys.github.io/gui-docs/User-Manual/Building-a-signal-chain.html) and a [Falcon Output](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Falcon-Output.html) node. (See [this guide](https://open-ephys.github.io/gui-docs/User-Manual/Exploring-the-user-interface.html) for help with understanding the user interface). Optionally, [change the buffer size in the GUI](https://open-ephys.github.io/gui-docs/Tutorials/Closed-Loop-Latency.html#:~:text=The%20second%2C%20and,in%20most%20cases) (not through Python code) to achieve optimal performance. Finally, start sorting:

```python
from braindance.core.open_ephys_env import OpenEphysEnv

env = OpenEphysEnv(**params)
done = False
while not done:
    obs, done = env.step()
    sequence_detections = rt_sort.running_sort(obs)
    # Process sequence_detections
```


## Next Steps

If you want to customize RT-Sort further, you might be interested in [training your own models](training-models).