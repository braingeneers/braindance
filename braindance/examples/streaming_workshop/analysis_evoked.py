"""Stimulation-wrapped responses following BrainDance's causal_reactions workflow.

Uses its actual cubic artifact-removal kernel. Descriptive response matrices use
milliseconds directly, without the legacy metric function's fixed 20 kHz units.
"""
from __future__ import annotations

import numpy as np


def analyze_evoked(events_ms, patterns, *, spike_times_ms=None, unit_ids=None,
                   selected_units=None, raw_reader=None, sampling_frequency=20000,
                   params=None, progress=None):
    params = dict(params or {})
    report = progress or (lambda fraction, message: None)
    def number(name, default, low, high):
        value = float(params.get(name, default))
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        return value
    pre = number('pre_ms', 20, 0, 1000)
    post = number('post_ms', 200, .1, 5000)
    blank = number('blank_ms', 3, 0, post)
    first = number('first_order_ms', 10, .01, post)
    width = number('bin_ms', 2, .01, 1000)
    sigma = number('threshold_sigma', 5, .1, 100)
    if first <= blank:
        raise ValueError('first_order_ms must exceed blank_ms')
    if (pre + post) / width > 5000:
        raise ValueError('Choose at most 5000 response histogram bins')
    cap = int(number('max_events', 100, 1, 200))
    max_units = int(number('max_units', 64, 1, 64))
    events = np.asarray(events_ms, dtype=float)
    labels = np.asarray([str(p) for p in patterns])
    if events.ndim != 1 or labels.shape != events.shape or not np.all(np.isfinite(events)):
        raise ValueError('Each finite stimulation time must have one stimulation pattern')
    if not len(events):
        raise ValueError('No complete stimulation windows available')
    events, labels = events[:cap], labels[:cap]
    groups = list(dict.fromkeys(labels.tolist()))
    raw = raw_reader is not None
    if raw:
        from braindance.core.artifact_removal import cubic_fit5
        from scipy.signal import find_peaks
        units = np.asarray(params.get('channels', [params.get('channel', 0)]), dtype=int)[:max_units]
        fs = float(sampling_frequency)
        if not np.isfinite(fs) or fs < 1000 or not len(units) or np.any(units < 0):
            raise ValueError('Raw response analysis requires valid channels and sample rate >=1000 Hz')
        if pre < 5:
            raise ValueError('Raw threshold detection requires at least 5 ms of pre-stimulus baseline')
        pad = max(2, round(.003 * fs))
        npre, npost = round(pre * fs / 1000), round(post * fs / 1000)
    else:
        times = np.asarray(spike_times_ms, dtype=float)
        identifiers = np.asarray(unit_ids, dtype=int)
        if times.ndim != 1 or times.shape != identifiers.shape or not np.all(np.isfinite(times)):
            raise ValueError('Supply matching finite spike times and unit IDs')
        units = np.asarray(selected_units if selected_units is not None else np.unique(identifiers), dtype=int)[:max_units]
        trains = [np.sort(times[identifiers == unit]) for unit in units]
    if not len(units):
        raise ValueError('No units or channels selected')
    if len(events) * len(units) * (pre + post) * (sampling_frequency / 1000 if raw else 1) > 20000000:
        raise ValueError('Choose fewer events/channels or a shorter response window')
    responses = [[None for _ in units] for _ in events]
    raw_series, clean_series = [], []
    report(.05, 'Wrapping responses around each stimulation pattern')
    for trial, event in enumerate(events):
        for column, unit in enumerate(units):
            if raw:
                center = round(event * fs / 1000)
                start = center - npre - pad
                stop = center + npost + pad + 2
                if start < 0:
                    raise ValueError('Raw response needs 3 ms extra recording context around each window')
                trace = np.asarray(raw_reader(int(unit), start, stop), dtype=float)
                if len(trace) != stop - start:
                    raise ValueError('Incomplete raw response window (include 3 ms extra recording context)')
                cleaned, _ = cubic_fit5(np.ascontiguousarray(trace), pad, pad)
                waveform = cleaned[pad:pad+npre+npost]
                baseline = waveform[:npre]
                noise = float(np.median(np.abs(baseline - np.median(baseline))) / .6745)
                threshold = max(sigma * noise, np.finfo(float).eps)
                peaks, _ = find_peaks(-waveform, height=threshold, distance=max(1, round(fs / 1000)))
                relative = peaks * 1000 / fs - npre * 1000 / fs
                if column == 0 and labels[trial] == groups[0] and len(raw_series) < 50:
                    step = max(1, int(np.ceil(len(waveform) / 1500)))
                    x = ((np.arange(len(waveform))[::step] - npre) * 1000 / fs).tolist()
                    raw_series.append(dict(name=f'Trial {trial+1}', x=x, y=trace[pad:pad+len(waveform):step].tolist()))
                    clean_series.append(dict(name=f'Trial {trial+1}', x=x, y=waveform[::step].tolist()))
            else:
                train = trains[column]
                relative = train[np.searchsorted(train, event-pre):np.searchsorted(train, event+post)] - event
            # Blank only post-stimulation artifact interval; retain baseline for visualization.
            responses[trial][column] = relative[(relative < 0) | (relative >= blank)]
        report(.1 + .75 * (trial+1)/len(events), f'Analyzing response {trial+1}/{len(events)}')
    short, late, total = [], [], []
    for group in groups:
        indices = np.flatnonzero(labels == group)
        short.append([float(np.mean([np.any((responses[i][j] >= blank) & (responses[i][j] < first)) for i in indices])) for j in range(len(units))])
        late.append([float(np.mean([np.count_nonzero((responses[i][j] >= first) & (responses[i][j] < post)) for i in indices])) for j in range(len(units))])
        total.append([float(np.mean([np.any((responses[i][j] >= blank) & (responses[i][j] < post)) for i in indices])) for j in range(len(units))])
    plots = []
    if raw:
        for title, series in [('Raw stimulation overlap', raw_series), ('Cubic artifact-removed overlap', clean_series)]:
            plots.append(dict(type='line', title=f'{title} · {groups[0]} · channel {units[0]}', series=series,
                              x_label='Time from stimulation (ms)', y_label='ADC counts'))
    trial_x, trial_y, aligned = [], [], []
    indices = np.flatnonzero(labels == groups[0])
    for trial_number, index in enumerate(indices):
        values = responses[index][0]
        aligned.extend(values.tolist())
        trial_x.extend(values.tolist()); trial_y.extend([trial_number+1] * len(values))
    display_step = max(1, int(np.ceil(len(trial_x) / 20000)))
    plots.append(dict(type='scatter', title=f'Wrapped spike responses · {groups[0]} · unit/channel {units[0]}',
                      x=trial_x[::display_step], y=trial_y[::display_step],
                      x_label='Time from stimulation (ms)', y_label='Trial'))
    bins = np.unique(np.concatenate([np.arange(-pre, post, width), [0, blank, post]]))
    counts, _ = np.histogram(aligned, bins)
    rates = counts / len(indices) / (np.diff(bins)/1000)
    # Excluded bins are gaps, never presented as evidence of zero activity.
    rates = [None if left >= 0 and right <= blank else float(rate)
             for left, right, rate in zip(bins[:-1], bins[1:], rates)]
    plots.append(dict(type='line', title='Peristimulus firing rate (first displayed pattern and unit)',
                      x=((bins[:-1]+bins[1:])/2).tolist(), y=rates,
                      x_label='Time from stimulation (ms)', y_label='Spikes / second'))
    for title, values in [(f'Short response probability ({blank:g}–{first:g} ms)', short),
                          (f'Late mean spikes / trial ({first:g}–{post:g} ms)', late),
                          (f'Total response probability ({blank:g}–{post:g} ms)', total)]:
        plots.append(dict(type='heatmap', title=title, x=units.tolist(), y=groups, z=values,
                          x_label='Unit / channel', y_label='Stimulation pattern'))
    report(1, 'Evoked response overlaps and response matrices complete')
    return dict(summary=[f'{len(events)} trials; {len(groups)} stimulation patterns; {len(units)} units/channels.',
                         f'Post-stimulation interval 0–{blank:g} ms excluded; short/late windows split at {first:g} ms.',
                         'Raw channel threshold detections after cubic artifact removal; these are not sorted neurons.' if raw else 'Responses use supplied spike detections; no raw artifact correction is applied.',
                         'Response matrices are descriptive associations, not proof of direct or causal connectivity. Unequal trial counts are normalized per pattern; overlapping stimulation windows may confound responses.'],
                plots=plots, unit_ids=units.tolist(), pattern_counts={g:int(np.sum(labels==g)) for g in groups},
                response_metrics=dict(short_probability=short, late_mean_spikes=late, total_probability=total))
