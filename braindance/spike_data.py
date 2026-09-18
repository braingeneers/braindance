"""SpikeLab objects and compatibility loading for trusted BrainDance pickles.

Historical SpikeData pickles name the old class's module. Resolve that class
locally and initialize SpikeLab's state, including its new time origin, without
requiring the retired package. This is ordinary pickle loading, not a security
sandbox: only load trusted files.
"""

import pickle

from spikelab import SpikeData


def as_spike_data(data):
    """Normalize supported recording formats while preserving acquisition metadata."""
    if isinstance(data, SpikeData):
        return data
    if isinstance(data, dict):
        train_key = next((key for key in ("train", "spike_trains") if key in data), None)
        if train_key is None:
            return data
        return SpikeData(
            data[train_key], N=data.get("N"), length=data.get("length"),
            start_time=data.get("start_time", 0.0),
            metadata=data.get("metadata", {}),
            neuron_attributes=data.get("neuron_attributes"),
            raw_data=data.get("raw_data"), raw_time=data.get("raw_time"),
        )
    if isinstance(data, list):
        return SpikeData(data)
    return data


class _LegacySpikeData:
    """Temporary pickle reconstruction target; instances become native SpikeLab."""

    def __setstate__(self, state):
        converted = as_spike_data(state)
        # Preserve extra user-attached fields as well as the validated core state.
        self.__dict__.update(state)
        self.__dict__.update(vars(converted))
        self.__class__ = SpikeData


class _SpikeDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SpikeData" and module in {
            "spikedata", "spikedata.spikedata",
            "braingeneers.analysis.analysis", "braingeneers.analysis.spikedata",
        }:
            return _LegacySpikeData
        return super().find_class(module, name)


def load_spike_pickle(file):
    """Load a trusted pickle, converting legacy SpikeData even inside containers."""
    return _SpikeDataUnpickler(file).load()
