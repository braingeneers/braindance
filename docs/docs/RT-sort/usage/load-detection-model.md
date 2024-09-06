---
sidebar_position: 1
---

# Loading Detection Models

To use RT-Sort, you first need to load a detection model. Here's how to do it:

```python
from braindance.core.spikedetector.model2 import ModelSpikeSorter

detection_model = ModelSpikeSorter.load(detection_model_path)
```

## Parameters

- `detection_model_path`: Path to a folder containing a ModelSpikeSorter object's `init_dict.json` and `state_dict.pt`.

## Returns

- `ModelSpikeSorter`: Represents the loaded detection model.

## Pre-trained Models

Pre-trained ModelSpikeSorter objects for Maxwell MEAs and Neuropixels used in the RT-Sort manuscript can be found in the BrainDance GitHub repository:

`BrainDance/braindance/core/spikedetector/detection_models`

## Next Steps

Once you've loaded a detection model, you can proceed to [sequence detection](sequence-detection) or [real-time application](real-time-application).