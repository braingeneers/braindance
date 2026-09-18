"""Run the legacy online sorting example with an explicitly selected recording.

Relative recording paths resolve under BRAINDANCE_DATA_DIR. Intermediate sorting
files are isolated beneath the configured output directory and cleaned on exit.
This example requires the rtsort extra and a Maxwell-compatible environment.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from braindance.config import get_data_dir, get_output_dir


def main(recording_path, max_time_sec=60, stim_electrodes=None,
         recording_window_ms=(120_000, 140_000), buffer_size=120):
    path = Path(recording_path).expanduser()
    if not path.is_absolute():
        path = get_data_dir() / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if stim_electrodes is None:
        stim_electrodes = [10254, 14130]

    import numpy as np
    from spikeinterface.extractors import MaxwellRecordingExtractor
    from braindance.core.maxwell_env import MaxwellEnv
    from braindance.core.params import maxwell_params
    from braindance.core.spikedetector.model2 import ModelSpikeSorter
    from braindance.core.spikesorter.rt_sort import detect_sequences
    from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval

    params = dict(maxwell_params)
    params.update(name='test', stim_electrodes=stim_electrodes,
                  max_time_sec=max_time_sec, config=None,
                  observation_type='raw', dummy=str(path))
    recording = MaxwellRecordingExtractor(str(path))
    n_channels = recording.get_num_channels()
    output_dir = get_output_dir() / 'real_time_sorting'
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='sort-', dir=output_dir) as work_dir:
        print('Pre-sorting with RT-sort')
        rt_sort = detect_sequences(
            recording, Path(work_dir), ModelSpikeSorter.load_mea(),
            return_spikes=False, delete_inter=True,
            recording_window_ms=recording_window_ms,
        )
        print('Warming up artifact removal')
        art_remover = LinearArtifactRemoval(n_channels=n_channels, batch_size=n_channels)
        art_remover.warmup()
        env = MaxwellEnv(**params)
        try:
            rt_sort.reset()
            done = False
            while not done:
                action = ([0], 150, 100) if env.stim_dt > 0.3 else None
                obs, done = env.step(action=action, buffer_size=buffer_size)
                if obs is None or len(obs) == 0:
                    continue
                cleaned, _, _ = art_remover.fit_step(np.asarray(obs).T)
                detections = rt_sort.running_sort(cleaned.T)
                if detections is not None and len(detections):
                    print(detections)
        finally:
            env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording_path', help='Absolute path or path relative to configured data directory')
    parser.add_argument('--max-time-sec', type=float, default=60)
    args = parser.parse_args()
    main(args.recording_path, max_time_sec=args.max_time_sec)
