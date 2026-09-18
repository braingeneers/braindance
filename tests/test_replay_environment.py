import csv
from pathlib import Path

import numpy as np
import pytest

from braindance.analysis.data_loader import load_data_maxwell
from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import (
    FrequencyStimPhaseV3,
    NeuralSweepPhaseV3,
    RecordPhaseV3,
)
from braindance.cli.replay import main as replay_main


def test_environment_raw_stimulation_and_cleanup(maxwell_h5, tmp_path):
    env = MaxwellEnv(
        config=None,
        name="env",
        save_dir=tmp_path,
        stim_electrodes=[101],
        observation_type="raw",
        verbose=0,
        replay={
            "source": maxwell_h5,
            "speed": 0,
            "stop_frame": 5,
            "write_output": True,
        },
    )
    observation, done = env.step(action=([0], 250, 100), tag="test")
    assert len(observation) == 4
    assert not done
    assert env.time_elapsed() == pytest.approx(1 / 20000)
    assert env.stim_dt == pytest.approx(0)
    buffered, done = env.step(buffer_size=4)
    assert len(buffered) == 4
    assert done
    assert env.stim_dt == pytest.approx(4 / 20000)
    env.close()
    env.close()

    output = tmp_path / "env.raw.h5"
    assert output.exists()
    assert load_data_maxwell(output, start=0, length=5).shape == (4, 5)
    with (tmp_path / "env_log.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["stim_electrodes"] == "[101]"
    assert rows[0]["tag"] == "test"
    assert rows[0]["replay_frame"] == "1000"


def test_verbose_replay_stimulation(maxwell_h5, tmp_path):
    env = MaxwellEnv(
        config=None,
        name="verbose",
        save_dir=tmp_path,
        stim_electrodes=[101],
        observation_type="raw",
        verbose=2,
        replay={"source": maxwell_h5, "speed": 0, "stop_frame": 2, "write_output": False},
    )
    env.step(action=([0], 250, 100))
    env.close()


def test_environment_spike_observation(maxwell_h5, tmp_path):
    env = MaxwellEnv(
        config=None,
        name="spikes",
        save_dir=tmp_path,
        observation_type="spikes",
        verbose=0,
        replay={"source": maxwell_h5, "speed": 0, "stop_frame": 3, "write_output": False},
    )
    assert env.step()[0] == []
    assert env.step()[0] == []
    events, done = env.step()
    assert done
    assert len(events) == 1
    assert events[0].frame == 1002
    assert events[0].channel == 1
    env.close()


def test_output_opens_with_spikeinterface(maxwell_h5, tmp_path):
    spikeinterface = pytest.importorskip("spikeinterface.extractors")
    env = MaxwellEnv(
        config=None,
        name="compatible",
        save_dir=tmp_path,
        observation_type="raw",
        verbose=0,
        replay={"source": maxwell_h5, "speed": 0, "stop_frame": 10},
    )
    env.step(buffer_size=10)
    env.close()
    recording = spikeinterface.MaxwellRecordingExtractor(tmp_path / "compatible.raw.h5")
    assert recording.get_num_channels() == 4
    assert recording.get_num_samples() == 10
    assert recording.get_sampling_frequency() == 20000.0


def test_v3_record_and_stimulation(maxwell_h5, tmp_path):
    params = {
        "record_duration": 0.005,
        "stim_electrodes": [101],
        "verbose": False,
        "maxwell_env": {
            "replay": {
                "source": maxwell_h5,
                "speed": 1,
                "stop_frame": 100,
                "write_output": True,
            }
        },
    }
    experiment = Experiment("e2e", params=params, save_dir=tmp_path / "e2e")
    experiment.add_phase(RecordPhaseV3(duration=0.005))
    experiment.add_phase(FrequencyStimPhaseV3(
        stim_command=([0], 250, 100),
        stim_freq=1000,
        duration=0.005,
    ))
    assert experiment.run(validate=True)
    assert (tmp_path / "e2e" / "001_rec" / "001.raw.h5").exists()
    assert (tmp_path / "e2e" / "002_freq_stim" / "002.raw.h5").exists()
    assert experiment.data.stim_count > 0


def test_cli_speed_uses_source_duration(maxwell_h5, tmp_path):
    assert replay_main([
        "run",
        "--h5", str(maxwell_h5),
        "--record-seconds", "0.005",
        "--speed", "2",
        "--output-dir", str(tmp_path),
        "--name", "speed",
    ]) == 0
    output = tmp_path / "speed" / "001_rec" / "001.raw.h5"
    assert load_data_maxwell(output, start=0, length=100).shape == (4, 100)


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0, "max"])
def test_frequency_count_uses_replay_time_at_every_speed(maxwell_h5, tmp_path, speed):
    env = MaxwellEnv(
        config=None,
        name=f"frequency_{speed}",
        save_dir=tmp_path,
        stim_electrodes=[101],
        observation_type="spikes",
        verbose=0,
        replay={
            "source": maxwell_h5,
            "speed": speed,
            "stop_frame": 100,
            "write_output": False,
        },
    )
    phase = FrequencyStimPhaseV3(
        stim_command=([0], 250, 100),
        stim_freq=1000,
        duration=0.005,
    )
    phase.set_env(env)
    result = phase.run(None)
    assert result["stim_count"] == 5
    assert phase.time_elapsed() == pytest.approx(0.005)
    env.close()


def test_reused_replay_environment_resets_phase_time_origin(maxwell_h5, tmp_path):
    env = MaxwellEnv(
        config=None,
        name="shared",
        save_dir=tmp_path,
        observation_type="spikes",
        verbose=0,
        replay={
            "source": maxwell_h5,
            "speed": "max",
            "stop_frame": 80,
            "write_output": False,
        },
    )
    first = RecordPhaseV3(duration=0.002)
    first.set_env(env)
    assert first.run(None)["recording_duration"] == pytest.approx(0.002)

    second = RecordPhaseV3(duration=0.002)
    second.set_env(env)
    assert second.run(None)["recording_duration"] == pytest.approx(0.002)
    assert env.time_elapsed() == pytest.approx(0.004)
    env.close()


def test_neural_sweep_uses_replay_deadlines(maxwell_h5, tmp_path):
    env = MaxwellEnv(
        config=None,
        name="sweep",
        save_dir=tmp_path,
        stim_electrodes=[101],
        observation_type="spikes",
        verbose=0,
        replay={
            "source": maxwell_h5,
            "speed": "max",
            "stop_frame": 100,
            "write_output": False,
        },
    )
    phase = NeuralSweepPhaseV3(
        neuron_list=[0],
        amp_bounds=250,
        stim_freq=1000,
        replicates=3,
    )
    phase.set_env(env)
    result = phase.run(None)
    assert len(result["sweep_results"]) == 3
    assert [row["time"] for row in result["sweep_results"]] == pytest.approx([
        0.00005,
        0.00105,
        0.00205,
    ])
    env.close()


def test_cli_accepts_max_speed(maxwell_h5, tmp_path):
    assert replay_main([
        "run",
        "--h5", str(maxwell_h5),
        "--record-seconds", "0.002",
        "--speed", "max",
        "--output-dir", str(tmp_path),
        "--name", "max_speed",
    ]) == 0
    output = tmp_path / "max_speed" / "001_rec" / "001.raw.h5"
    assert load_data_maxwell(output, start=0, length=40).shape == (4, 40)

    assert replay_main([
        "run",
        "--h5", str(maxwell_h5),
        "--record-seconds", "0.002",
        "--speed", "max",
        "--no-write-output",
        "--output-dir", str(tmp_path),
        "--name", "max_stream_only",
    ]) == 0
    assert not (tmp_path / "max_stream_only" / "001_rec" / "001.raw.h5").exists()
