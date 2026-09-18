"""Small stimulation-responsive neural source for the Maxwell replay contract.

The recurrent leaky integrate-and-fire (LIF) network is implemented here with
NumPy; no external SNN package or prerecorded spikes are used. Neurons update
at 1 ms, while their extracellular templates are sampled at 20 kHz. This is a
teaching model, not a fitted biological culture. Connections use
adjacency[target, source]. Positions are micrometers; raw ADC counts represent
1 microvolt per count and converted raw/event amplitudes are millivolts.

SpikeEvent reports each neuron's firing on its nearest recording channel; these
are simulator ground truth, not threshold detections or sorted units. Raw voltage
contains Gaussian spatial footprints and can use the normal processing pipeline.
"""
import heapq
import time

import numpy as np

from braindance.core.replay import MAPPING_DTYPE, SpikeEvent


def _number(value, name, integer=False):
    """Validate public scalar configuration without silently truncating indices."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(number) or (integer and not number.is_integer()):
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"}')
    return int(number) if integer else number


class NeuralSimulationSource:
    def __init__(self, num_channels=8, seed=7, adjacency=None, speed=1.0,
                 sampling_hz=20000, background=1.25, noise_uv=4.0,
                 duration_s=3600, clock=time.perf_counter, sleep=time.sleep,
                 num_neurons=None, grid_shape=None, electrode_pitch_um=17.5,
                 neuron_positions=None, spatial_sigma_um=20., waveform_amplitude_uv=85.):
        self.num_channels = _number(num_channels, 'num_channels', integer=True)
        self.sampling_hz = _number(sampling_hz, 'sampling_hz')
        self.speed = 0.0 if str(speed) == 'max' else _number(speed, 'speed')
        if self.num_channels < 2 or self.sampling_hz != 20000:
            raise ValueError('Use at least two channels and 20000 Hz sampling')
        if not np.isfinite(self.speed) or self.speed < 0:
            raise ValueError('speed must be finite and nonnegative')
        self.num_neurons = self.num_channels if num_neurons is None else _number(num_neurons, 'num_neurons', integer=True)
        if self.num_neurons < 1:
            raise ValueError('num_neurons must be positive')
        self.electrode_pitch_um = _number(electrode_pitch_um, 'electrode_pitch_um')
        self.spatial_sigma_um = _number(spatial_sigma_um, 'spatial_sigma_um')
        self.waveform_amplitude_uv = _number(waveform_amplitude_uv, 'waveform_amplitude_uv')
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.electrode_pitch_um, self.spatial_sigma_um, self.waveform_amplitude_uv)):
            raise ValueError('Pitch, spatial sigma and waveform amplitude must be finite and positive')
        legacy_layout = grid_shape is None
        if grid_shape is None:
            cols = min(4, self.num_channels)
            grid_shape = ((self.num_channels + cols - 1) // cols, cols)
        try:
            self.grid_shape = tuple(_number(v, 'grid_shape', integer=True) for v in grid_shape)
        except TypeError as exc:
            raise ValueError('grid_shape must be a pair of positive integers') from exc
        if (len(self.grid_shape) != 2 or min(self.grid_shape) < 1
                or np.prod(self.grid_shape) < self.num_channels or self.grid_shape[1] > 220):
            raise ValueError('grid_shape must contain all channels and at most 220 columns')
        self.mapping = np.zeros(self.num_channels, dtype=MAPPING_DTYPE)
        channels = np.arange(self.num_channels)
        rows, cols = channels // self.grid_shape[1], channels % self.grid_shape[1]
        self.mapping['channel'] = channels
        # Maxwell electrode IDs use a 220-column physical array, not channel IDs.
        self.mapping['electrode'] = channels if legacy_layout else rows * 220 + cols
        self.mapping['x'] = cols * self.electrode_pitch_um
        self.mapping['y'] = rows * self.electrode_pitch_um
        self.lsb, self.gain, self.hpf = 1e-6, 1.0, 1.0
        self.finished = self._closed = False
        self._frame = 0
        duration_s = _number(duration_s, 'duration_s')
        if duration_s <= 0 or not np.isfinite(duration_s * self.sampling_hz):
            raise ValueError('duration_s must be positive and finite')
        self._stop = int(duration_s * self.sampling_hz)
        self._clock, self._sleep, self._wall_start = clock, sleep, None
        self.background = _number(background, 'background')
        self.noise_uv = _number(noise_uv, 'noise_uv')
        if self._stop <= 0 or not 0 <= self.background <= 5:
            raise ValueError('Invalid duration or background drive (0..5)')
        if not 0 <= self.noise_uv <= 100:
            raise ValueError('noise_uv must be in 0..100')
        self.seed = None if seed is None else _number(seed, 'seed', integer=True)
        if self.seed is not None and self.seed < 0:
            raise ValueError('seed must be nonnegative')
        self._initialize_rng()
        coordinates = np.column_stack((self.mapping['x'], self.mapping['y']))
        if neuron_positions is None:
            if self.num_neurons == self.num_channels:
                neuron_positions = coordinates.copy()
            else:
                # A separate stream keeps placement independent of neural/noise draws.
                placement_rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(3)[2])
                neuron_positions = placement_rng.uniform(coordinates.min(axis=0),
                                                          coordinates.max(axis=0),
                                                          (self.num_neurons, 2))
        self.set_neuron_positions(neuron_positions)
        if adjacency is None:
            adjacency = np.zeros((self.num_neurons, self.num_neurons))
            if self.num_neurons > 1:
                for neuron in range(self.num_neurons):
                    adjacency[(neuron + 1) % self.num_neurons, neuron] = .35
        self.set_adjacency(adjacency)
        x = np.arange(60) / self.sampling_hz
        self._wave = -np.exp(-((x - .0004) / .00018)**2)
        self._wave += (25 / 85) * np.exp(-((x - .0011) / .0004)**2)
        self._wave *= self.waveform_amplitude_uv
        self.templates = np.tile(self._wave, (self.num_neurons, 1))
        self.reset()

    def _initialize_rng(self):
        seeds = np.random.SeedSequence(self.seed).spawn(2)
        self._rng, self._raw_rng = [np.random.default_rng(s) for s in seeds]

    def reset(self, seed=None):
        """Rewind dynamics/noise/clock; preserve the edited geometry and connections."""
        if self._closed:
            raise RuntimeError('Simulation source is closed')
        if seed is not None:
            seed = _number(seed, 'seed', integer=True)
            if seed < 0:
                raise ValueError('seed must be nonnegative')
            self.seed = seed
        self._initialize_rng()
        self._voltage = self._rng.uniform(0, .6, self.num_neurons)
        self._refractory = np.zeros(self.num_neurons, dtype=int)
        self._last_spikes = np.zeros(self.num_neurons)
        self._drive = np.zeros(self.num_neurons)
        self._pulses = []
        self._pulse_id = self._frame = 0
        self.finished = False
        self._wall_start = None
        self._tail = np.zeros((len(self._wave), self.num_channels))

    def set_neuron_positions(self, positions):
        """Update extracellular and stimulation footprints without changing IDs."""
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.num_neurons, 2) or not np.all(np.isfinite(positions)):
            raise ValueError('neuron_positions must be finite with shape (neurons, 2)')
        coordinates = np.column_stack((self.mapping['x'], self.mapping['y']))
        if np.any(positions < coordinates.min(axis=0)) or np.any(positions > coordinates.max(axis=0)):
            raise ValueError('Neuron positions must lie within the electrode workspace')
        self.neuron_positions = positions.copy()
        squared_distances = ((positions[:, None, :] - coordinates[None, :, :])**2).sum(axis=2)
        self.spatial_weights = np.exp(-squared_distances / (2 * self.spatial_sigma_um**2))
        self._stim_weights = np.exp(-squared_distances / (2 * (self.electrode_pitch_um / 2)**2))
        self.primary_channels = squared_distances.argmin(axis=1)

    def snapshot(self):
        """Serializable geometry for a simulator editor, without mutable state aliases."""
        return dict(model='NumPy recurrent leaky integrate-and-fire',
                    num_neurons=self.num_neurons, num_channels=self.num_channels,
                    sampling_hz=self.sampling_hz, neural_update_hz=1000,
                    grid_shape=list(self.grid_shape), electrode_pitch_um=self.electrode_pitch_um,
                    spatial_sigma_um=self.spatial_sigma_um, waveform_amplitude_uv=self.waveform_amplitude_uv,
                    seed=self.seed, background=self.background, noise_uv=self.noise_uv,
                    neuron_positions=self.neuron_positions.tolist(), adjacency=self.adjacency.tolist(),
                    channel_positions=np.column_stack((self.mapping['x'], self.mapping['y'])).tolist(),
                    electrodes=self.mapping['electrode'].tolist())

    @property
    def elapsed_s(self):
        return self._frame / self.sampling_hz

    def set_adjacency(self, adjacency):
        matrix = np.asarray(adjacency, dtype=float)
        if matrix.shape != (self.num_neurons, self.num_neurons):
            raise ValueError('Adjacency must have shape (neurons, neurons)')
        if not np.all(np.isfinite(matrix)) or np.any(np.abs(matrix) > 2):
            raise ValueError('Connection weights must be finite and in [-2, 2]')
        self.adjacency = matrix.copy()

    def on_stimulation(self, action, frame, stim_electrodes):
        """Queue pulses at the next unread frame; sequence delays are milliseconds.

        ``frame`` is the caller's last acquired frame for logging, not a request
        to rewrite acquired data. The electrical artifact starts at the scheduled
        sample; neurons integrate its drive on their next 1 ms update.
        """
        if isinstance(action[0], str):
            raise ValueError('Simulator needs explicit pulse commands, not manual sequences')
        commands = action if isinstance(action[0][0], str) else [('stim', *action)]
        scheduled = []
        offset = 0
        electrode_to_channel = {int(e): i for i, e in enumerate(self.mapping['electrode'])}
        for command in commands:
            if command[0] == 'next':
                break
            if command[0] == 'delay':
                delay = _number(command[1], 'delay')
                if not np.isfinite(delay) or delay < 0:
                    raise ValueError('Delay must be nonnegative')
                offset += int(round(delay * self.sampling_hz / 1000))
                continue
            if command[0] != 'stim':
                raise ValueError(f'Unknown stimulation command: {command[0]}')
            _, indices, amplitude, width = command
            amplitude = _number(amplitude, 'amplitude')
            width = _number(width, 'width')
            if not np.isfinite(amplitude) or not np.isfinite(width) or amplitude < 0 or width <= 0:
                raise ValueError('Pulse amplitude/width must be finite and positive')
            for index in indices:
                index = _number(index, 'Stimulation index', integer=True)
                if not 0 <= index < len(stim_electrodes):
                    raise ValueError('Stimulation index outside configured electrodes')
                electrode = stim_electrodes[int(index)]
                if electrode not in electrode_to_channel:
                    raise ValueError(f'Electrode {electrode} has no simulator channel')
                strength = min(4., float(amplitude) / 100 * float(width) / 100)
                scheduled.append((self._frame + offset, electrode_to_channel[electrode], strength))
        for at, channel, strength in scheduled:
            self._pulse_id += 1
            heapq.heappush(self._pulses, (at, self._pulse_id, channel, strength))

    def read(self, count=1, convert_raw=True):
        if self._closed:
            raise RuntimeError('Simulation source is closed')
        if self.finished:
            return None
        count = min(_number(count, 'count', integer=True), self._stop - self._frame)
        if count <= 0:
            raise ValueError('count must be positive')
        if self._wall_start is None:
            self._wall_start = self._clock()
        frames = np.arange(self._frame, self._frame + count, dtype=np.uint64)
        events = [[] for _ in range(count)]
        signal = np.zeros((count + len(self._wave), self.num_channels))
        signal[:len(self._tail)] += self._tail
        # Only neural updates and scheduled pulses need Python work. Raw samples
        # between them are generated in the vectorized waveform/noise path below.
        stop = self._frame + count
        next_update = ((self._frame + 19) // 20) * 20
        while True:
            frame = min(next_update, self._pulses[0][0] if self._pulses else stop)
            if frame >= stop:
                break
            i = frame - self._frame
            while self._pulses and self._pulses[0][0] <= frame:
                _, _, channel, strength = heapq.heappop(self._pulses)
                self._drive += strength * self._stim_weights[:, channel]
                signal[i:i + 8, channel] += 120 * strength * np.exp(-np.arange(8) / 2)
            if frame == next_update:
                next_update += 20
                active = self._refractory <= 0
                self._voltage += ((self.background - self._voltage) / 20
                                  + self.adjacency @ self._last_spikes
                                  + self._drive + self._rng.normal(0, .025, self.num_neurons)) * active
                fired = active & (self._voltage >= 1)
                self._voltage[fired] = 0
                self._refractory -= 1
                self._refractory[fired] = 3
                self._last_spikes = fired.astype(float)
                self._drive *= .35
                for neuron in np.flatnonzero(fired):
                    channel = int(self.primary_channels[neuron])
                    amplitude = float(self.templates[neuron].min() * self.spatial_weights[neuron, channel] / 1000)
                    events[i].append(SpikeEvent(frame, channel, amplitude))
                    signal[i:i + len(self._wave)] += (self.templates[neuron, :, None]
                                                      * self.spatial_weights[neuron])
        self._tail = signal[count:].copy()
        raw_uv = signal[:count] + self._raw_rng.normal(0, self.noise_uv, (count, self.num_channels))
        raw_uv += 3 * np.sin(2 * np.pi * frames[:, None] / self.sampling_hz * .7 + np.arange(self.num_channels))
        adc = np.clip(np.rint(raw_uv + 512), 0, 1023).astype(np.uint16)
        self._frame += count
        self.finished = self._frame >= self._stop
        if self.speed:
            remaining = self.elapsed_s / self.speed - (self._clock() - self._wall_start)
            if remaining > 0:
                self._sleep(remaining)
        return dict(frame_numbers=frames, source_frame_numbers=frames.copy(),
                    raw_uint16=adc,
                    raw_float32=(adc.astype(np.float32) - 512) * .001 if convert_raw else None,
                    events=events)

    def close(self):
        self._closed = True
