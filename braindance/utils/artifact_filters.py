"""Stateful centered polynomial baseline subtraction with explicit exclusions.

This implementation does not replace historical filter classes. The cubic
variant is SALPA-style, not a reproduction of the original SALPA state machine.
Inputs and outputs are channel-by-sample arrays in consistent voltage units.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numba import njit, prange


@dataclass
class FilterResult:
    clean: np.ndarray
    artifact: np.ndarray
    valid: np.ndarray


@njit(cache=True)
def _process_channel(ch, x, buffer, bad, means, holds, anchors, moments,
                     bad_counts, position, seen, threshold, recovery_samples,
                     rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid):
    """Update one channel; the cubic intercept needs only moments 0, 1, 2."""
    samples = x.shape[1]
    width = buffer.shape[1]
    half = width // 2
    mean, hold, anchor = means[ch], holds[ch], anchors[ch]
    s0, s1, s2 = moments[ch, 0], moments[ch, 1], moments[ch, 2]
    bad_count = bad_counts[ch]
    for t in range(samples):
        value = x[ch, t]
        if seen == 0:
            mean = anchor = value
        excursion = abs(value - mean) > threshold
        rail = value <= rail_min or value >= rail_max
        if excursion or rail:
            hold = recovery_samples + 1
        flagged = hold > 0
        bad_count += int(flagged) - int(bad[ch, position])
        bad[ch, position] = flagged
        if hold > 0:
            hold -= 1
        mean = 0.8 * mean + 0.2 * value

        # A fixed channel offset reduces cancellation for large ADC baselines.
        incoming = value - anchor
        outgoing = buffer[ch, position]
        if c2 != 0.0:
            s2 = s2 - 2.0 * s1 + s0 - (half + 1)**2 * outgoing + half**2 * incoming
            s1 = s1 - s0 + (half + 1) * outgoing + half * incoming
        s0 += incoming - outgoing
        buffer[ch, position] = incoming
        first = position + 1
        if first == width:
            first = 0
        # Rebase on absolute sample number, independent of input chunking.
        # This bounds floating-point drift without rescanning every window.
        if (seen + 1) % rebase_interval == 0:
            s0 = s1 = s2 = 0.0
            index = first
            for j in range(width):
                v = buffer[ch, index]
                s0 += v
                if c2 != 0.0:
                    s1 += (j - half) * v
                    s2 += (j - half)**2 * v
                index += 1
                if index == width:
                    index = 0
        ok = seen >= width - 1 and bad_count == 0
        valid[ch, t] = ok
        if ok:
            baseline = c0 * s0 + c2 * s2
            center = first + half
            if center >= width:
                center -= width
            artifact[ch, t] = baseline + anchor
            clean[ch, t] = buffer[ch, center] - baseline
        else:
            clean[ch, t] = np.nan
            artifact[ch, t] = np.nan
        position = first
        seen += 1
    means[ch], holds[ch], anchors[ch] = mean, hold, anchor
    moments[ch, 0], moments[ch, 1], moments[ch, 2] = s0, s1, s2
    bad_counts[ch] = bad_count


@njit(cache=True, parallel=True)
def _process_parallel(x, buffer, bad, means, holds, anchors, moments,
                      bad_counts, position, seen, threshold, recovery_samples,
                      rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid):
    for ch in prange(x.shape[0]):
        _process_channel(ch, x, buffer, bad, means, holds, anchors, moments,
                         bad_counts, position, seen, threshold, recovery_samples,
                         rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid)


@njit(cache=True)
def _process_serial(x, buffer, bad, means, holds, anchors, moments,
                    bad_counts, position, seen, threshold, recovery_samples,
                    rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid):
    for ch in range(x.shape[0]):
        _process_channel(ch, x, buffer, bad, means, holds, anchors, moments,
                         bad_counts, position, seen, threshold, recovery_samples,
                         rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid)


class ArtifactRemover(ABC):
    """Common API; output column t describes input column t-delay_samples.

Startup and excluded samples are NaN with valid=False. No end padding or
fabricated recovery is supplied. Retain the object across acquisition blocks.
"""
    @property
    @abstractmethod
    def degree(self):
        pass

    def __init__(self, n_channels, half_window=60, excursion_threshold=100.0,
                 recovery_samples=60, rail_min=-np.inf, rail_max=np.inf, *,
                 parallel=True, rebase_interval=256):
        if not isinstance(n_channels, int) or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        if not isinstance(half_window, int) or half_window < 2:
            raise ValueError("half_window must be an integer >= 2")
        if not isinstance(recovery_samples, int) or recovery_samples < 0:
            raise ValueError("recovery_samples must be a nonnegative integer")
        if not excursion_threshold > 0 or not rail_min < rail_max:
            raise ValueError("Invalid excursion threshold or rail bounds")
        if self.degree not in (1, 3):
            raise ValueError("Optimized ArtifactRemover supports degree 1 or 3")
        if not isinstance(rebase_interval, int) or rebase_interval < 1:
            raise ValueError("rebase_interval must be a positive integer")
        self.n_channels = n_channels
        self.delay_samples = half_window
        self.excursion_threshold = float(excursion_threshold)
        self.recovery_samples = recovery_samples
        self.rail_min, self.rail_max = float(rail_min), float(rail_max)
        self.parallel = bool(parallel)
        self.rebase_interval = rebase_interval
        positions = np.linspace(-1, 1, 2 * half_window + 1)
        design = np.vander(positions, self.degree + 1, increasing=True)
        self.weights = np.ascontiguousarray(np.linalg.pinv(design)[0])
        if self.degree == 1:
            self.c0, self.c2 = 1.0 / len(positions), 0.0
        else:
            j = np.arange(-half_window, half_window + 1, dtype=np.float64)
            sum2, sum4 = np.sum(j*j), np.sum(j**4)
            denominator = len(j) * sum4 - sum2 * sum2
            self.c0, self.c2 = sum4 / denominator, -sum2 / denominator
        self.reset()

    def reset(self):
        """Discard recording state without changing parameters."""
        self.buffer = np.zeros((self.n_channels, len(self.weights)))
        self.bad = np.zeros(self.buffer.shape, dtype=bool)
        self.means = np.zeros(self.n_channels)
        self.holds = np.zeros(self.n_channels, dtype=np.int64)
        self.anchors = np.zeros(self.n_channels)
        self.moments = np.zeros((self.n_channels, 3))
        self.bad_counts = np.zeros(self.n_channels, dtype=np.int64)
        self.position = self.seen = 0
        return self

    def process(self, frames):
        """Process a block; preserve state and return clean/artifact/valid."""
        frames = np.ascontiguousarray(frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[0] != self.n_channels:
            raise ValueError("Expected (n_channels, n_samples)")
        if not np.isfinite(frames).all():
            raise ValueError("Input must be finite; pass rail bounds for clipping")
        c, a = np.empty(frames.shape), np.empty(frames.shape)
        v = np.empty(frames.shape, dtype=bool)
        kernel = _process_parallel if self.parallel and self.n_channels >= 32 else _process_serial
        kernel(
            frames, self.buffer, self.bad, self.means, self.holds,
            self.anchors, self.moments, self.bad_counts,
            self.position, self.seen, self.excursion_threshold,
            self.recovery_samples, self.rail_min, self.rail_max,
            self.c0, self.c2, self.rebase_interval, c, a, v)
        self.position = (self.position + frames.shape[1]) % len(self.weights)
        self.seen += frames.shape[1]
        return FilterResult(c, a, v)

    def fit_step(self, frames):
        """Alias for process; returns FilterResult, not legacy spike flags."""
        return self.process(frames)

    def warmup(self):
        """Compile a disposable instance, leaving this instance unchanged."""
        other = type(self)(self.n_channels, self.delay_samples,
                           self.excursion_threshold, self.recovery_samples,
                           self.rail_min, self.rail_max, parallel=self.parallel,
                           rebase_interval=self.rebase_interval)
        other.process(np.zeros((self.n_channels, 2 * self.delay_samples + 2)))
        return self


class SalpaArtifactRemover(ArtifactRemover):
    """SALPA-style cubic fit with the shared conservative exclusion policy."""
    degree = 3


class LinearFitArtifactRemover(ArtifactRemover):
    """Local degree-one least-squares fit with the shared exclusion policy."""
    degree = 1
