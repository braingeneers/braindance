import numpy as np
import pytest

from braindance.core.replay import H5ReplayWriter, MAPPING_DTYPE, SpikeEvent


@pytest.fixture
def maxwell_h5(tmp_path):
    path = tmp_path / "fixture.raw.h5"
    mapping = np.zeros(4, dtype=MAPPING_DTYPE)
    mapping["channel"] = [10, 20, 30, 40]
    mapping["electrode"] = [101, 102, 103, 104]
    mapping["x"] = [0.0, 17.5, 35.0, 52.5]
    mapping["y"] = [0.0, 0.0, 0.0, 0.0]
    raw = (512 + np.arange(800, dtype=np.uint16).reshape(200, 4) % 100).astype(np.uint16)
    frames = np.arange(1000, 1200, dtype=np.uint64)
    events = [[] for _ in range(len(frames))]
    events[2] = [SpikeEvent(1002, 1, -12.5)]
    events[50] = [SpikeEvent(1050, 3, -8.0)]
    writer = H5ReplayWriter(
        path,
        mapping=mapping,
        sampling_hz=20000.0,
        lsb=3.147125e-6,
        gain=1024.0,
        hpf=1.0,
        chunk_frames=32,
    )
    writer.append({
        "raw_uint16": raw,
        "frame_numbers": frames,
        "events": events,
    })
    writer.close()
    return path
