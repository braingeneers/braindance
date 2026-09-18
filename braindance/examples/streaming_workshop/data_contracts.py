"""Documented data contracts; dimensions describe runtime values, not validation."""


def data_metadata(keys):
    known = {
        'recording_baseline_hz': ('list[float]', '(n_channels,)', 'Mean firing rate in Hz; n_channels is detected channels or sorted units.'),
        'response_probe_hz': ('list[list[float]]', '(2, n_channels)', 'Stimulus minus sham rate change in Hz, per stimulation input.'),
        'response_probe_trials': ('list[list[int]]', '(2, 2)', 'Completed stimulated and sham trial counts per input.'),
        'environment_episodes': ('int', 'scalar', 'Completed episode count.'),
        'environment_reward': ('float', 'scalar', 'Accumulated game reward.'),
        'recording_file': ('str', '', 'Acquisition recording path.'),
        'recording_duration': ('float', 'scalar', 'Recorded duration in seconds.'),
    }
    return {key: dict(zip(('type', 'shape', 'description'), known.get(key,
        ('Any', '', 'Type and shape depend on the producing phase; inspect exp.data at runtime.')))) for key in keys}
