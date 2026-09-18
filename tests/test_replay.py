import os

import numpy as np
import pytest

from braindance.core.replay import (
    H5ReplaySource,
    SpikeEvent,
    inspect_recording,
    pack_replay_packet,
    resolve_replay_source,
    unpack_replay_packet,
)


def test_inspect_and_lazy_slice(maxwell_h5):
    info = inspect_recording(maxwell_h5)
    assert info["num_channels"] == 4
    assert info["num_frames"] == 200
    assert info["sampling_hz"] == 20000.0

    source = H5ReplaySource(maxwell_h5, channels=[2, 0], start_frame=1, stop_frame=4, speed=0)
    batch = source.read(3)
    assert batch["raw_uint16"].shape == (3, 2)
    assert batch["raw_uint16"][0].tolist() == [518, 516]
    expected = (batch["raw_uint16"].astype(np.float32) - 512) * source.lsb * source.gain * 1000
    np.testing.assert_allclose(batch["raw_float32"], expected)
    assert source.finished
    assert source.read() is None
    source.close()


def test_spikes_and_loop_frames_are_deterministic(maxwell_h5):
    source = H5ReplaySource(maxwell_h5, start_frame=0, stop_frame=3, speed=0, loop=True)
    first = source.read(3)
    second = source.read(3)
    assert first["frame_numbers"].tolist() == [1000, 1001, 1002]
    assert second["frame_numbers"].tolist() == [1003, 1004, 1005]
    assert first["events"][2] == [SpikeEvent(1002, 1, pytest.approx(-12.5))]
    assert second["events"][2][0].frame == 1005
    assert source.elapsed_s == pytest.approx(6 / 20000)
    source.close()


def test_wall_pacing_does_not_change_replay_time(maxwell_h5):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    source = H5ReplaySource(
        maxwell_h5,
        stop_frame=20,
        speed=2,
        clock=clock,
        sleep=clock.sleep,
    )
    source.read(20)
    assert source.elapsed_s == pytest.approx(0.001)
    assert source.wall_elapsed_s == pytest.approx(0.0005)
    source.close()

    max_clock = FakeClock()
    source = H5ReplaySource(
        maxwell_h5,
        stop_frame=20,
        speed="max",
        clock=max_clock,
        sleep=max_clock.sleep,
    )
    source.read(20)
    assert source.speed == 0
    assert source.elapsed_s == pytest.approx(0.001)
    assert max_clock.now == 0
    source.close()


def test_packet_round_trip():
    parts = pack_replay_packet(42, [1.5, -2.0], [SpikeEvent(42, 1, -9.5)])
    frame, raw, events = unpack_replay_packet(parts)
    assert frame == 42
    np.testing.assert_array_equal(raw, [1.5, -2.0])
    assert events[0].frame == 42
    assert events[0].channel == 1
    assert events[0].amplitude == pytest.approx(-9.5)


def test_legacy_npy_source(tmp_path):
    path = tmp_path / "legacy.npy"
    expected = np.arange(30, dtype=np.float32).reshape(3, 10)
    np.save(path, expected)
    source = H5ReplaySource(path, speed=0, stop_frame=4)
    batch = source.read(4)
    np.testing.assert_array_equal(batch["raw_float32"], expected[:, :4].T)
    source.close()


def test_invalid_h5_releases_windows_handle(tmp_path):
    import h5py

    path = tmp_path / "invalid.raw.h5"
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("unrelated", data=[1])
    with pytest.raises(ValueError, match="raw dataset not found"):
        H5ReplaySource(path)
    path.unlink()
    assert not path.exists()


def test_source_resolution_precedence(maxwell_h5, monkeypatch, tmp_path):
    root = tmp_path / "data"
    sample_dir = root / "replay_fixtures"
    sample_dir.mkdir(parents=True)
    sample = sample_dir / "maxwell_replay_smoke.raw.h5"
    sample.write_bytes(maxwell_h5.read_bytes())
    monkeypatch.setenv("BRAINDANCE_REPLAY_H5", str(maxwell_h5))
    assert resolve_replay_source(sample=True, data_dir=root) == maxwell_h5.resolve()
    monkeypatch.delenv("BRAINDANCE_REPLAY_H5")
    assert resolve_replay_source(sample=True, data_dir=root) == sample.resolve()


@pytest.mark.integration
def test_optional_external_recording():
    path = os.getenv("BRAINDANCE_REPLAY_H5")
    if not path:
        pytest.skip("BRAINDANCE_REPLAY_H5 not set")
    info = inspect_recording(path)
    source = H5ReplaySource(path, speed=0, stop_frame=min(100, info["num_frames"]))
    assert source.read(100)["raw_uint16"].shape[1] == info["num_channels"]
    source.close()
