"""Native paper protocol: routing, data handoff, and V3 pipeline construction."""
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from braindance.core.phases_v3.phase_base_v3 import PhaseValidator
from braindance.core.phases_v3.phases3_cartpole import (
    CartPoleFootprintPhaseV3, CartPoleCausalAnalysisPhaseV3,
    CartPoleRankPairsPhaseV3, PaperCartPolePhaseV3,
)


def fake(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


def test_default_pipeline_and_resume(tmp_path, monkeypatch):
    phases = []
    exp = Mock(current_phase_idx=2)
    exp.add_phase.side_effect = phases.append
    exp.run.return_value = True
    constructor = Mock(return_value=exp)
    fake(monkeypatch, 'braindance.core.phases_v3.experiment_v3', Experiment=constructor)
    config = dict(config='routing.cfg', stim_electrodes=list(range(6)), type='C1')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['1_cartpole', '--json', str(path), '--resume'])
    importlib.import_module('braindance.examples.1_cartpole').main()
    assert [type(p).__name__ for p in phases] == [
        'RecordPhaseV3', 'CartPoleFootprintPhaseV3', 'CartPoleCausalSweepPhaseV3',
        'CartPoleCausalAnalysisPhaseV3', 'CartPoleRankPairsPhaseV3', 'PaperCartPolePhaseV3']
    assert PhaseValidator.validate_pipeline(phases, config)
    exp.run.assert_called_once_with(resume=True)
    assert constructor.call_args.kwargs['overwrite_existing'] is True
    assert constructor.call_args.kwargs['auto_load_data'] is False
    assert json.loads(path.read_text()) == config
    phases[2].configure_from_experiment(SimpleNamespace(
        data=SimpleNamespace(stim_electrodes=[9, 3, 8, 1, 5, 7]),
        get_param=lambda key, default: default))
    commands = phases[2].generate_stim_commands()
    assert len(commands) == 300
    assert [command[0][0] for command in commands[:6]] == list(range(6))
    assert all(command[1:] == (400.0, 200) for command in commands)


def test_rank_matches_historical_electrode_mapping(tmp_path):
    from braindance.examples.paper_cartpole.ranking import find_connectivity_patterns
    matrix = np.random.default_rng(5).normal(size=(6, 6))
    np.save(tmp_path / 'causal_connectivity_multi.npy', matrix)
    electrodes = [20, 30, 70, 80, 90, 100]
    exp = SimpleNamespace(data=SimpleNamespace(derived_dir=str(tmp_path),
                                             valid_stim_electrodes=electrodes))
    result = CartPoleRankPairsPhaseV3(rank=2).run(exp)
    a, b, c, d, *_ = find_connectivity_patterns(matrix)[1]
    assert result['sensory_electrodes'] == [electrodes[a], electrodes[c]]
    assert result['motor_electrodes'] == [electrodes[b], electrodes[d]]
    assert len(result['pair_selection']['matrix_sha256']) == 64
    np.save(tmp_path / 'causal_connectivity_multi.npy', np.ones((6, 6)))
    with pytest.raises(ValueError, match='zero-variance'):
        CartPoleRankPairsPhaseV3().run(exp)


@pytest.mark.parametrize('mode,index,expected', [('C1', 0, 2), ('C1', 1, 0), ('C2', 1, 1), ('C7', 0, 0)])
def test_game_routes_and_uses_v3_environment(tmp_path, monkeypatch, mode, index, expected):
    trainers = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]
    fake(monkeypatch, 'braindance.core.trainer',
         TetanusTrainer=Mock(side_effect=trainers), generate_permutations=Mock(return_value=[]))
    mapping = Mock()
    mapping.get_orig_channels.return_value = [802, 41]
    fake(monkeypatch, 'braindance.analysis.mapping',
         Mapping=SimpleNamespace(from_csv=Mock(return_value=mapping)))
    np.save(tmp_path / 'causal_connectivity_multi_mean.npy', [10., 20., 30., 40.])
    np.save(tmp_path / 'causal_connectivity_multi_std.npy', [1., 2., 3., 4.])
    data = dict(sensory_electrodes=[100, 300], motor_electrodes=[200, 400],
                stim_electrodes=[600, 200, 300, 700, 100, 400],
                valid_stim_electrodes=[100, 200, 300, 400], derived_dir=str(tmp_path),
                mapping_file_path='mapping.csv')
    exp = SimpleNamespace(params=dict(type=mode, config='routing.cfg'), name='game',
                          data=SimpleNamespace(**data), save_dir=tmp_path,
                          current_recording_dir=tmp_path)
    phase = PaperCartPolePhaseV3(run_index=index)
    phase.configure_from_experiment(exp)
    env_params = phase.customize_environment_params(dict(save_dir='v3-owned', name='003'))
    assert env_params['stim_electrodes'] == [600, 300, 700, 100]
    assert env_params['save_dir'] == 'v3-owned'
    np.testing.assert_equal(phase.sensory_neurons, [3, 1])
    np.testing.assert_equal(phase.motor_neurons, [802, 41])
    np.testing.assert_equal(phase.training_neurons, [0, 2])
    assert phase.trainer is trainers[expected]
    assert phase.trainer.phase is phase
    assert [phase.read_period_ms, phase.train_period_ms, phase.wait_period_ms] == [200, 400, 3000]
    if mode in ('C1', 'C2'):
        assert phase.normalization == 1
    else:
        np.testing.assert_equal(phase.normalization, [20, 40, 2, 4])


def test_causal_sweep_records_final_response(monkeypatch):
    from braindance.core.phases_v3.phases3 import NeuralSweepPhaseV3
    from braindance.core.phases_v3.phases3_cartpole import CartPoleCausalSweepPhaseV3

    phase = CartPoleCausalSweepPhaseV3()
    phase.neuron_list = [0]
    clock = [0.]
    monkeypatch.setattr(phase, 'time_elapsed', lambda: clock[0])
    def step():
        clock[0] += .1
        return None, False
    phase.env = SimpleNamespace(step=step)
    result = dict(sweep_file='causal', sweep_results=[{}] * 50)
    monkeypatch.setattr(NeuralSweepPhaseV3, 'run', lambda *args: result)
    assert phase.run(SimpleNamespace()) is result
    assert clock[0] >= .5
    result['sweep_results'] = [{}] * 49
    with pytest.raises(RuntimeError, match='before all stimuli'):
        phase.run(SimpleNamespace())


@pytest.mark.parametrize('training_type,expected_training', [
    ('punishment', True), ('reward', False), ('always', True),
])
def test_native_loop_reads_spikes_steps_game_and_stimulates(
        tmp_path, monkeypatch, training_type, expected_training):
    """Execute the V3 state machine, with only external hardware/game replaced."""
    import csv

    class Acquisition:
        observation_type = 'raw'
        save_file = str(tmp_path / 'game')
        def __init__(self):
            self.now = 0.
            self.stim_dts = np.zeros(4)
            self.events = []
        def step(self, action=None, tag=None):
            self.now += .01
            self.stim_dts += .01
            self.events.append((self.now, action, tag))
            if tag == 'sensory':
                self.stim_dts[action[0]] = 0
            # Distinct acquisition channels, not physical IDs or stim indices.
            return np.array([0., -10., 0., 0., 0.]), False

    class Game:
        def __init__(self, **kwargs):
            self.actions, self.resets, self.closed = [], 0, False
            self.steps = 0
        def reset(self):
            self.resets += 1
            self.steps = 0
            return np.array([0., 0., .2, 0.]), {}
        def step(self, action):
            self.actions.append(action)
            self.steps += 1
            return np.array([0., 0., .2, 0.]), 1., self.steps == 2, False, {}
        def close(self):
            self.closed = True

    class Detector:
        def __init__(self, **kwargs):
            pass
        def fit_step(self, value):
            return value, 0., value < -5

    game = Game()
    fake(monkeypatch, 'braindance.games.cartpole_continuous',
         CartPoleContinuousEnv=lambda **kwargs: game)
    fake(monkeypatch, 'braindance.core.artifact_removal', ArtifactRemoval=Detector)
    # Any accidental delegation must fail, rather than silently passing a mock.
    fake(monkeypatch, 'braindance.core.phases2')
    fake(monkeypatch, 'braindance.examples.paper_cartpole.game')
    phase = PaperCartPolePhaseV3(n_episodes=3, render_mode=None)
    env = Acquisition()
    phase.set_env(env)
    monkeypatch.setattr(phase, 'time_elapsed', lambda: env.now)
    phase.sensory_neurons, phase.motor_neurons = np.array([0, 2]), np.array([1, 4])
    phase.training_neurons = np.array([1, 3])
    pulse = [('stim', [1], 400, 100), ('delay', 10), ('stim', [3], 400, 100)]
    phase.trainer = Mock()
    phase.trainer.get_action.return_value = (pulse, 10, 0)
    phase.trainer.get_probs.return_value = [1.]
    phase.trainer.get_values.return_value = [1.]
    phase.normalization, phase.training_type, phase.verbose = 1, training_type, False
    phase.config = {}
    exp = SimpleNamespace(current_recording_dir=tmp_path, save_dir=tmp_path)
    result = phase.run(exp)
    assert result['final_rewards'] == [2., 2., 2.]
    assert result['total_episodes'] == 3
    assert game.actions == [-1.] * 6
    assert game.resets == 3
    sensory = [(t, action) for t, action, tag in env.events if tag == 'sensory']
    assert sensory
    assert all(set(action[0]) <= {0, 2} for _, action in sensory)
    training = [(t, action) for t, action, tag in env.events if tag == 'train']
    assert bool(training) is expected_training
    assert all(action == pulse for _, action in training)
    # Three-second rest is included between episodes, not skipped by the port.
    assert env.now >= 6 + 6 * .2
    with open(result['game_log_file']) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert all(float(row['spike_count_l']) > 0 and float(row['spike_count_r']) == 0 for row in rows)
    with open(result['reward_log_file']) as stream:
        assert len(list(csv.DictReader(stream))) == 3
    phase.cleanup()
    assert game.closed


def test_native_loop_runs_real_cartpole_and_spike_detector(tmp_path, monkeypatch):
    """Real game physics and artifact removal, driven by a hardware test source."""
    from braindance.games.cartpole_continuous import CartPoleContinuousEnv

    class Acquisition:
        observation_type = 'raw'
        save_file = str(tmp_path / 'real_game')
        def __init__(self):
            self.now = 0.
            self.stim_dts = np.zeros(4)
        def step(self, action=None, tag=None):
            self.now += .01
            self.stim_dts += .01
            if tag == 'sensory':
                self.stim_dts[action[0]] = 0
            return np.zeros(4), False

    phase = PaperCartPolePhaseV3(n_episodes=1, render_mode=None)
    env = Acquisition()
    phase.set_env(env)
    monkeypatch.setattr(phase, 'time_elapsed', lambda: env.now)
    phase.sensory_neurons, phase.motor_neurons = np.array([0, 2]), np.array([1, 3])
    phase.training_neurons = np.array([1, 3])
    phase.trainer = Mock()  # One episode ends before the first training period.
    phase.normalization, phase.training_type, phase.verbose = 1, 'punishment', False
    phase.config = {}
    result = phase.run(SimpleNamespace(current_recording_dir=tmp_path, save_dir=tmp_path))
    assert isinstance(phase.game_env, CartPoleContinuousEnv)
    assert result['total_episodes'] == 1
    assert result['final_rewards'][0] > 0
    phase.cleanup()


def test_v3_cartpole_has_no_legacy_phase_or_example_imports():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'braindance/core/phases_v3'
    for name in ('phases3_cartpole.py', 'phases3_cartpole_analysis.py'):
        for node in ast.walk(ast.parse((root / name).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith('braindance.examples')
                assert node.module not in ('braindance.core.phases', 'braindance.core.phases2',
                                           'braindance.core.phases_analysis', 'braindance.core.phases_analysis_2')
