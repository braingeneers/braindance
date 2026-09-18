"""Lazy Maxwell H5 replay and recording utilities.

This module deliberately has no dependency on ``maxlab``.  It can therefore be
used to exercise BrainDance experiment code on machines without Maxwell
hardware while preserving the raw-frame and spike-event contracts used by
``MaxwellEnv``.
"""

from __future__ import annotations

import os
import struct
import time
from collections import namedtuple
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


SpikeEvent = namedtuple("SpikeEvent", "frame channel amplitude")
MAPPING_DTYPE = np.dtype([
    ("channel", "<i4"),
    ("electrode", "<i4"),
    ("x", "<f8"),
    ("y", "<f8"),
])
SPIKE_DTYPE = np.dtype([
    ("frameno", "<i8"),
    ("channel", "<i4"),
    ("amplitude", "<f4"),
])


def _decode_scalar(dataset, default=""):
    if dataset is None:
        return default
    value = dataset[0]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _find_layout(h5file):
    """Return raw/settings/recording groups for supported Maxwell layouts."""
    if "sig" in h5file:
        return h5file["sig"], h5file.get("settings"), h5file

    data_path = "/data_store/data0000"
    if data_path in h5file:
        rec = h5file[data_path]
        return rec["groups/routed/raw"], rec["settings"], rec

    if "wells" in h5file:
        well_name = sorted(h5file["wells"].keys())[0]
        well = h5file["wells"][well_name]
        rec_name = sorted(well.keys())[0]
        rec = well[rec_name]
        return rec["groups/routed/raw"], rec["settings"], rec

    raise ValueError("Not a supported Maxwell H5 file: raw dataset not found")


def inspect_recording(path):
    """Read recording metadata without loading trace data."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as h5file:
        raw, settings, rec = _find_layout(h5file)
        sampling = float(settings["sampling"][0]) if settings is not None and "sampling" in settings else 20000.0
        mapping = settings.get("mapping") if settings is not None else h5file.get("mapping")
        channels = raw.shape[0] if mapping is None else len(mapping)
        spikes = len(rec["spikes"]) if "spikes" in rec else 0
        return {
            "path": path,
            "size_bytes": path.stat().st_size,
            "num_channels": int(channels),
            "num_frames": int(raw.shape[1]),
            "sampling_hz": sampling,
            "duration_s": float(raw.shape[1] / sampling),
            "dtype": str(raw.dtype),
            "version": _decode_scalar(h5file.get("version"), "unknown"),
            "num_spikes": int(spikes),
        }


def find_recordings(data_dir):
    """Return compatible ``*.raw.h5`` recordings ordered by file size."""
    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        return []
    recordings = []
    for path in data_dir.rglob("*.raw.h5"):
        relative_parts = path.relative_to(data_dir).parts
        if relative_parts and relative_parts[0].lower() == "outputs":
            continue
        try:
            recordings.append(inspect_recording(path))
        except (OSError, KeyError, ValueError):
            continue
    return sorted(recordings, key=lambda item: (item["size_bytes"], str(item["path"])))


def resolve_replay_source(h5_path=None, recording=None, sample=False, data_dir=None, doctor=False):
    """Resolve replay input without mutating BrainDance path configuration."""
    from braindance.config import get_data_dir

    root = Path(data_dir).expanduser() if data_dir is not None else get_data_dir()
    candidates = []
    if h5_path:
        candidates.append(Path(h5_path).expanduser())
    if os.getenv("BRAINDANCE_REPLAY_H5"):
        candidates.append(Path(os.environ["BRAINDANCE_REPLAY_H5"]).expanduser())
    if sample:
        candidates.append(root / "replay_fixtures" / "maxwell_replay_smoke.raw.h5")
    if recording:
        candidates.append(root / recording)
    if doctor and not candidates:
        sample_path = root / "replay_fixtures" / "maxwell_replay_smoke.raw.h5"
        if sample_path.is_file():
            return sample_path.resolve()

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if candidates:
        attempted = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Replay H5 not found. Attempted:\n  {attempted}")

    if doctor:
        recordings = find_recordings(root)
        if recordings:
            return recordings[0]["path"]

    raise FileNotFoundError(
        f"No replay H5 selected under data root {root}. "
        "Use --h5, --recording, --sample, or BRAINDANCE_REPLAY_H5."
    )


class H5ReplaySource:
    """Sequential, bounded, lazy reader for Maxwell raw H5 recordings."""

    def __init__(
        self,
        source,
        start_frame=0,
        stop_frame=None,
        channels=None,
        speed=1.0,
        loop=False,
        chunk_frames=2000,
        clock=time.perf_counter,
        sleep=time.sleep,
    ):
        self.source = source
        self.start_frame = int(start_frame)
        self.stop_frame = None if stop_frame is None else int(stop_frame)
        if isinstance(speed, str) and speed.lower() in {"max", "unthrottled"}:
            speed = 0.0
        self.speed = float(speed)
        self.loop = bool(loop)
        self.chunk_frames = max(1, int(chunk_frames))
        self._clock = clock
        self._sleep = sleep
        self._closed = False
        self.finished = False
        self._emitted_frames = 0
        self._start_clock = None
        self._virtual_next = None
        self._synthetic = str(source) in {"sine", "manual"}
        self._npy = not self._synthetic and str(source).lower().endswith(".npy")
        self._npy_data = None
        self._spike_cursor = 0
        self._spike_cache = None
        self._spike_cache_frames = None
        self._spike_cache_position = 0
        self._raw_cache_start = None
        self._raw_cache_stop = None
        self._raw_cache = None

        if self.speed < 0:
            raise ValueError("Replay speed must be >= 0")

        if self._synthetic:
            self.path = None
            self.h5file = None
            self.raw = None
            self.sampling_hz = 20000.0
            self.total_frames = 20000 * 60
            self.lsb = np.float32(1.0 / 512.0 / 1000.0)
            self.gain = np.float32(512.0)
            self.hpf = np.float32(1.0)
            all_channels = np.arange(1024 if str(source) == "sine" else 942, dtype=np.int64)
            mapping = np.zeros(len(all_channels), dtype=MAPPING_DTYPE)
            mapping["channel"] = all_channels
            mapping["electrode"] = all_channels
            mapping["x"] = all_channels % 32
            mapping["y"] = all_channels // 32
            self._frame_nos = None
            self._spikes = None
        elif self._npy:
            self.path = Path(source).expanduser().resolve()
            if not self.path.is_file():
                raise FileNotFoundError(f"Replay source does not exist: {self.path}")
            self.h5file = None
            self._npy_data = np.load(self.path, mmap_mode="r")
            if self._npy_data.ndim != 2:
                raise ValueError("Legacy replay NPY must have shape (channels, frames)")
            self.raw = None
            self.sampling_hz = 20000.0
            self.total_frames = int(self._npy_data.shape[1])
            self.lsb = np.float32(1.0 / 512.0 / 1000.0)
            self.gain = np.float32(512.0)
            self.hpf = np.float32(1.0)
            mapping = np.zeros(self._npy_data.shape[0], dtype=MAPPING_DTYPE)
            mapping["channel"] = np.arange(len(mapping))
            mapping["electrode"] = np.arange(len(mapping))
            self._frame_nos = None
            self._spikes = None
        else:
            self.path = Path(source).expanduser().resolve()
            if not self.path.is_file():
                raise FileNotFoundError(f"Replay source does not exist: {self.path}")
            self.h5file = h5py.File(self.path, "r")
            try:
                self.raw, settings, rec = _find_layout(self.h5file)
                self.total_frames = int(self.raw.shape[1])
                self.sampling_hz = float(settings["sampling"][0]) if settings is not None and "sampling" in settings else 20000.0
                self.lsb = np.float32(settings["lsb"][0]) if settings is not None and "lsb" in settings else np.float32(6.294e-6)
                self.gain = np.float32(settings["gain"][0]) if settings is not None and "gain" in settings else np.float32(512.0)
                self.hpf = np.float32(settings["hpf"][0]) if settings is not None and "hpf" in settings else np.float32(1.0)
                if settings is not None and "mapping" in settings:
                    mapping = np.asarray(settings["mapping"], dtype=MAPPING_DTYPE)
                elif "mapping" in self.h5file:
                    mapping = np.asarray(self.h5file["mapping"], dtype=MAPPING_DTYPE)
                else:
                    mapping = np.zeros(self.raw.shape[0], dtype=MAPPING_DTYPE)
                    mapping["channel"] = np.arange(self.raw.shape[0])
                    mapping["electrode"] = np.arange(self.raw.shape[0])
                self._frame_nos = rec.get("groups/routed/frame_nos")
                self._spikes = rec.get("spikes")
            except Exception:
                self.h5file.close()
                self.h5file = None
                raise

        if self.start_frame < 0 or self.start_frame >= self.total_frames:
            self.close()
            raise ValueError(f"start_frame {self.start_frame} outside recording with {self.total_frames} frames")
        self.stop_frame = self.total_frames if self.stop_frame is None else min(self.stop_frame, self.total_frames)
        if self.stop_frame <= self.start_frame:
            self.close()
            raise ValueError("stop_frame must be greater than start_frame")

        if channels is None:
            self.channel_indices = np.arange(len(mapping), dtype=np.int64)
        else:
            self.channel_indices = np.asarray(channels, dtype=np.int64)
            if self.channel_indices.ndim != 1 or len(np.unique(self.channel_indices)) != len(self.channel_indices):
                self.close()
                raise ValueError("Replay channels must be a unique one-dimensional sequence")
            if np.any(self.channel_indices < 0) or np.any(self.channel_indices >= len(mapping)):
                self.close()
                raise ValueError("Replay channel index outside recording mapping")

        self.mapping = mapping[self.channel_indices].copy()
        self.num_channels = len(self.mapping)
        self._channel_to_output = {
            int(channel): index for index, channel in enumerate(self.mapping["channel"])
        }
        self._position = self.start_frame
        first_source_frame = self._source_frame_numbers(self.start_frame, self.start_frame + 1)[0]
        self._virtual_next = int(first_source_frame)
        if self._spikes is not None:
            self._spike_cursor = self._spike_lower_bound(int(first_source_frame))

    def _source_frame_numbers(self, start, stop):
        if self._frame_nos is None:
            return np.arange(start, stop, dtype=np.uint64)
        return np.asarray(self._frame_nos[start:stop], dtype=np.uint64)

    @property
    def elapsed_s(self):
        """Biological replay time emitted so far, independent of wall pacing."""
        return self._emitted_frames / self.sampling_hz

    @property
    def wall_elapsed_s(self):
        """Wall time since first read; useful for replay throughput diagnostics."""
        if self._start_clock is None:
            return 0.0
        return self._clock() - self._start_clock

    def _read_raw(self, start, stop):
        if self._synthetic:
            frame_indices = np.arange(start, stop, dtype=np.float64)
            if str(self.source) == "manual":
                values = np.broadcast_to(frame_indices[:, None], (len(frame_indices), self.num_channels))
                return np.asarray(values % 1024, dtype=np.uint16)
            phase = np.sin(2 * np.pi * frame_indices / self.sampling_hz)
            values = 512.0 + 100.0 * phase[:, None]
            return np.broadcast_to(values, (len(frame_indices), self.num_channels)).astype(np.uint16)

        if self._npy:
            order = np.argsort(self.channel_indices)
            sorted_indices = self.channel_indices[order]
            values = np.asarray(self._npy_data[sorted_indices, start:stop], dtype=np.float32).T
            return values[:, np.argsort(order)]

        if (
            self._raw_cache is not None
            and self._raw_cache_start <= start
            and stop <= self._raw_cache_stop
        ):
            offset = start - self._raw_cache_start
            return self._raw_cache[offset:offset + (stop - start)]

        cache_stop = min(self.stop_frame, max(stop, start + self.chunk_frames))
        order = np.argsort(self.channel_indices)
        sorted_indices = self.channel_indices[order]
        if len(sorted_indices) and np.all(np.diff(sorted_indices) == 1):
            channel_slice = slice(int(sorted_indices[0]), int(sorted_indices[-1]) + 1)
            raw = np.asarray(
                self.raw[channel_slice, start:cache_stop], dtype=np.uint16
            ).T
        else:
            raw = np.asarray(
                self.raw[sorted_indices, start:cache_stop], dtype=np.uint16
            ).T
        self._raw_cache = raw[:, np.argsort(order)]
        self._raw_cache_start = start
        self._raw_cache_stop = cache_stop
        return self._raw_cache[:stop - start]

    def _spike_lower_bound(self, target_frame):
        left = 0
        right = len(self._spikes)
        while left < right:
            middle = (left + right) // 2
            if int(self._spikes[middle]["frameno"]) < target_frame:
                left = middle + 1
            else:
                right = middle
        return left

    def _load_spike_cache(self):
        if self._spikes is None or self._spike_cursor >= len(self._spikes):
            self._spike_cache = None
            self._spike_cache_frames = None
            return False
        stop = min(self._spike_cursor + 4096, len(self._spikes))
        self._spike_cache = np.asarray(self._spikes[self._spike_cursor:stop])
        self._spike_cache_frames = np.asarray(self._spike_cache["frameno"], dtype=np.int64)
        self._spike_cache_position = 0
        self._spike_cursor = stop
        return True

    def _reset_spike_cache(self, target_frame):
        self._spike_cursor = self._spike_lower_bound(target_frame)
        self._spike_cache = None
        self._spike_cache_frames = None
        self._spike_cache_position = 0

    def _read_events(self, source_frames, virtual_frames):
        events = [[] for _ in range(len(source_frames))]
        if self._spikes is None or len(source_frames) == 0:
            return events

        first = int(source_frames[0])
        last = int(source_frames[-1])
        batches = []
        while True:
            if self._spike_cache is None or self._spike_cache_position >= len(self._spike_cache):
                if not self._load_spike_cache():
                    break
            position = self._spike_cache_position
            if self._spike_cache_frames[position] > last:
                break
            start = position + int(np.searchsorted(
                self._spike_cache_frames[position:], first, side="left"
            ))
            stop = position + int(np.searchsorted(
                self._spike_cache_frames[position:], last, side="right"
            ))
            if stop > start:
                batches.append(self._spike_cache[start:stop])
            self._spike_cache_position = stop
            if stop < len(self._spike_cache):
                break
        if not batches:
            return events

        frame_to_index = {int(frame): index for index, frame in enumerate(source_frames)}
        for batch in batches:
            for spike in batch:
                output_channel = self._channel_to_output.get(int(spike["channel"]))
                event_index = frame_to_index.get(int(spike["frameno"]))
                if output_channel is None or event_index is None:
                    continue
                events[event_index].append(SpikeEvent(
                    int(virtual_frames[event_index]),
                    output_channel,
                    float(spike["amplitude"]),
                ))
        return events

    def read(self, count=1, convert_raw=True):
        """Read up to ``count`` sequential frames and associated events."""
        if self._closed:
            raise RuntimeError("Replay source is closed")
        count = max(1, int(count))
        if self.finished and not self.loop:
            return None
        if self._position >= self.stop_frame:
            if not self.loop:
                self.finished = True
                return None
            self._position = self.start_frame
            if self._spikes is not None:
                first_source_frame = self._source_frame_numbers(self.start_frame, self.start_frame + 1)[0]
                self._reset_spike_cache(int(first_source_frame))

        stop = min(self._position + count, self.stop_frame)
        source_frames = self._source_frame_numbers(self._position, stop)
        raw_data = self._read_raw(self._position, stop)
        virtual_frames = np.arange(
            self._virtual_next,
            self._virtual_next + len(source_frames),
            dtype=np.uint64,
        )
        events = self._read_events(source_frames, virtual_frames)
        if self._npy:
            raw_float32 = np.asarray(raw_data, dtype=np.float32)
            raw_uint16 = np.clip(np.rint(raw_float32) + 512.0, 0, 65535).astype(np.uint16)
        else:
            raw_uint16 = raw_data
            raw_float32 = None
            if convert_raw:
                raw_float32 = (
                    (raw_uint16.astype(np.float32) - np.float32(512.0))
                    * self.lsb
                    * self.gain
                    * np.float32(1000.0)
                )

        if self._start_clock is None:
            self._start_clock = self._clock()
        if self.speed > 0:
            target = (self._emitted_frames + len(source_frames)) / self.sampling_hz / self.speed
            remaining = target - (self._clock() - self._start_clock)
            if remaining > 0:
                self._sleep(remaining)

        self._position = stop
        self._virtual_next += len(source_frames)
        self._emitted_frames += len(source_frames)
        if self._position >= self.stop_frame and not self.loop:
            self.finished = True

        return {
            "frame_numbers": virtual_frames,
            "source_frame_numbers": source_frames,
            "raw_uint16": raw_uint16,
            "raw_float32": raw_float32,
            "events": events,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.h5file is not None:
            self.h5file.close()
        if isinstance(self._npy_data, np.memmap) and getattr(self._npy_data, "_mmap", None) is not None:
            self._npy_data._mmap.close()


class H5ReplayWriter:
    """Incremental writer for a minimal modern Maxwell-compatible H5 file."""

    def __init__(self, path, mapping, sampling_hz, lsb, gain, hpf=1.0, chunk_frames=2000):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping = np.asarray(mapping, dtype=MAPPING_DTYPE)
        self.sampling_hz = float(sampling_hz)
        self.lsb = float(lsb)
        self.gain = float(gain)
        self.hpf = float(hpf)
        self._closed = False
        self._frames_written = 0
        self._chunk_frames = max(1, int(chunk_frames))
        self._pending_raw = []
        self._pending_frames = []
        self._pending_spikes = []
        self._pending_count = 0
        self._start_time_ms = int(time.time() * 1000)

        self.h5file = h5py.File(self.path, "w")
        self.h5file.create_dataset("version", data=np.asarray([b"20190530"], dtype="S8"))
        self.h5file.create_dataset("hdf_version", data=np.asarray([b"1.0"], dtype="S6"))
        self.h5file.create_dataset("mxw_version", data=np.asarray([b"replay"], dtype="S8"))

        wells = self.h5file.create_group("wells")
        rec = wells.create_group("well000").create_group("rec0000")
        settings = rec.create_group("settings")
        settings.create_dataset("sampling", data=[self.sampling_hz])
        settings.create_dataset("lsb", data=[self.lsb])
        settings.create_dataset("gain", data=[self.gain])
        settings.create_dataset("hpf", data=[self.hpf])
        settings.create_dataset("spike_threshold", data=[5.0])
        settings.create_dataset("mapping", data=self.mapping)

        routed = rec.create_group("groups").create_group("routed")
        chunk_frames = self._chunk_frames
        raw_chunks = (max(1, min(len(self.mapping), 64)), chunk_frames)
        self.raw = routed.create_dataset(
            "raw",
            shape=(len(self.mapping), 0),
            maxshape=(len(self.mapping), None),
            chunks=raw_chunks,
            dtype=np.uint16,
        )
        self.frame_nos = routed.create_dataset(
            "frame_nos", shape=(0,), maxshape=(None,), chunks=(chunk_frames,), dtype=np.uint64
        )
        routed.create_dataset("channels", data=self.mapping["channel"].astype(np.uint16))
        routed.create_dataset("triggered", data=np.asarray([0], dtype=np.int32))
        self.spikes = rec.create_dataset(
            "spikes", shape=(0,), maxshape=(None,), chunks=(max(256, chunk_frames),), dtype=SPIKE_DTYPE
        )
        rec.create_dataset("start_time", data=[self._start_time_ms])
        self.stop_time = rec.create_dataset("stop_time", data=[self._start_time_ms])
        rec.create_dataset("recording_id", data=np.asarray([0], dtype=np.int32))
        rec.create_dataset("well_id", data=np.asarray([0], dtype=np.int32))

        data_store = self.h5file.create_group("data_store")
        data_store["data0000"] = rec
        recordings = self.h5file.create_group("recordings").create_group("rec0000")
        recordings["well000"] = rec

    def append(self, batch):
        if self._closed:
            raise RuntimeError("Replay writer is closed")
        raw = np.asarray(batch["raw_uint16"], dtype=np.uint16)
        frames = np.asarray(batch["frame_numbers"], dtype=np.uint64)
        if raw.ndim != 2 or raw.shape != (len(frames), len(self.mapping)):
            raise ValueError("Replay writer received inconsistent frame shape")

        rows = []
        for events in batch.get("events", []):
            for event in events:
                rows.append((
                    int(event.frame),
                    int(self.mapping[int(event.channel)]["channel"]),
                    float(event.amplitude),
                ))
        self._pending_raw.append(raw)
        self._pending_frames.append(frames)
        if rows:
            self._pending_spikes.append(np.asarray(rows, dtype=SPIKE_DTYPE))
        self._pending_count += len(frames)
        if self._pending_count >= self._chunk_frames:
            self._flush_pending()

    def _flush_pending(self):
        if not self._pending_count:
            return
        raw = np.concatenate(self._pending_raw, axis=0)
        frames = np.concatenate(self._pending_frames)
        old = self._frames_written
        new = old + len(frames)
        self.raw.resize((len(self.mapping), new))
        self.raw[:, old:new] = raw.T
        self.frame_nos.resize((new,))
        self.frame_nos[old:new] = frames
        self._frames_written = new
        if self._pending_spikes:
            spike_rows = np.concatenate(self._pending_spikes)
            spike_old = len(self.spikes)
            self.spikes.resize((spike_old + len(spike_rows),))
            self.spikes[spike_old:] = spike_rows
        self._pending_raw.clear()
        self._pending_frames.clear()
        self._pending_spikes.clear()
        self._pending_count = 0

    def close(self):
        if self._closed:
            return
        self._flush_pending()
        self._closed = True
        self.stop_time[0] = int(time.time() * 1000)
        self.h5file.flush()
        self.h5file.close()


def pack_replay_packet(frame_number, raw_float32, events=()):
    """Encode one replay frame using Maxwell dummy-server wire format."""
    event_bytes = []
    for event in events:
        event_bytes.append(struct.pack("8xLif", int(event.frame), int(event.channel), float(event.amplitude)))
    return [
        struct.pack("Q", int(frame_number)),
        np.asarray(raw_float32, dtype=np.float32).tobytes(),
        b"".join(event_bytes),
    ]


def unpack_replay_packet(parts):
    """Decode packet produced by :func:`pack_replay_packet`."""
    if len(parts) != 3:
        raise ValueError(f"Expected three packet parts, received {len(parts)}")
    frame_number = struct.unpack("Q", parts[0])[0]
    raw = np.frombuffer(parts[1], dtype=np.float32).copy()
    event_size = struct.calcsize("8xLif")
    if len(parts[2]) % event_size:
        raise ValueError("Malformed replay event payload")
    events = []
    for offset in range(0, len(parts[2]), event_size):
        frame, channel, amplitude = struct.unpack("8xLif", parts[2][offset:offset + event_size])
        events.append(SpikeEvent(int(frame), int(channel), float(amplitude)))
    return frame_number, raw, events
