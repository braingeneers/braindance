"""Protocol-preservation checks for the numbered paper entrypoints (no hardware)."""
import ast
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def test_ranked_metric_matches_historical_source():
    from braindance.examples.paper_cartpole.ranking import find_connectivity_patterns
    source = Path(__file__).resolve().parents[1] / 'proj/cartpole_v2/ranked_pairs.py'
    if not source.exists():
        pytest.skip('Historical source absent from installed distribution')
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'find_connectivity_patterns')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    matrix = np.random.default_rng(71).normal(size=(6, 6))
    expected = namespace['find_connectivity_patterns'](matrix)
    np.testing.assert_array_equal(find_connectivity_patterns(matrix), expected)


def test_rank_cli_maps_electrodes_and_records_provenance(tmp_path, monkeypatch):
    module = importlib.import_module('braindance.examples.1_cartpole')
    from braindance.examples.paper_cartpole.ranking import find_connectivity_patterns
    matrix = np.random.default_rng(5).normal(size=(4, 4))
    np.save(tmp_path / 'causal_connectivity_multi.npy', matrix)
    electrodes = [103, 57, 2048, 99]
    config = tmp_path / 'experiment.json'
    config.write_text(json.dumps({'valid_stim_electrodes': electrodes}))
    monkeypatch.setattr(sys, 'argv', ['1_cartpole', 'rank', '--json', str(config),
                                    '--derived-dir', str(tmp_path), '--select-rank', '1'])
    module.main()
    result = json.loads(config.read_text())
    a, b, c, d, *_ = find_connectivity_patterns(matrix)[0]
    assert result['sensory_electrodes'] == [electrodes[a], electrodes[c]]
    assert result['motor_electrodes'] == [electrodes[b], electrodes[d]]
    assert len(result['pair_selection']['matrix_sha256']) == 64
    np.save(tmp_path / 'causal_connectivity_multi.npy', np.ones((4, 4)))
    with pytest.raises(ValueError, match='zero-variance'):
        module.main()


def install_fake(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("success", [True, False])
def test_busybee_constructs_original_400_phases_with_v3(tmp_path, monkeypatch, capsys, success):
    module = importlib.import_module('braindance.examples.3_busybee')
    exp = Mock(current_phase_idx=3)
    exp.run.return_value = success
    constructor = Mock(return_value=exp)
    record = Mock(side_effect=lambda **kw: ('record', kw))
    sweep = Mock(side_effect=lambda **kw: ('sweep', kw))
    install_fake(monkeypatch, 'braindance.core.phases_v3.experiment_v3',
                 Experiment=constructor)
    install_fake(monkeypatch, 'braindance.core.phases_v3.phases3',
                 RecordPhaseV3=record, NeuralSweepPhaseV3=sweep)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'name': 'busy', 'config': 'routing.cfg',
                               'stim_electrodes': [900, 17], 'save_dir': str(tmp_path)}))
    module.main(path, project_id='paper', chip_id='chip', resume=True)
    assert exp.add_phase.call_count == 400
    assert record.call_count == sweep.call_count == 200
    for i, call in enumerate(sweep.call_args_list):
        assert call.kwargs == dict(amp_bounds=400, stim_freq=(.5, 1, 2, 4, 8)[i % 5],
                                   tag='causal', replicates=(50, 100, 200, 400, 800)[i % 5],
                                   order='ran', single_connect=True, phase_length=200)
    assert all(call.kwargs == {'duration': 600} for call in record.call_args_list)
    assert [call.args[0][0] for call in exp.add_phase.call_args_list] == ['record', 'sweep'] * 200
    constructor.assert_called_once_with(
        'busy_cont', params={'config': 'routing.cfg', 'stim_electrodes': [900, 17],
                             'verbose': True},
        save_dir=str(tmp_path), project_id='paper', chip_id='chip')
    exp.run.assert_called_once_with(resume=True)
    output = capsys.readouterr().out
    assert ('All phases completed!' if success else 'stopped at phase 3') in output


def test_busybee_dry_run_preserves_schedule():
    module = importlib.import_module('braindance.examples.3_busybee')
    schedule = module.main(dry_run=True)
    assert len(schedule) == 200
    assert [item['frequency_hz'] for item in schedule] == [.5, 1, 2, 4, 8] * 40
    assert [item['replicates'] for item in schedule] == [50, 100, 200, 400, 800] * 40
    assert all(item['record_seconds'] == 600 for item in schedule)


@pytest.mark.parametrize('mode,index,trainer_index,trigger', [
    ('C1', 0, 2, 'punishment'), ('C1', 1, 0, 'punishment'),
    ('C2', 1, 1, 'punishment'), ('C7', 0, 0, 'punishment'),
    ('reward', 0, 0, 'reward'), ('always', 0, 0, 'always')])
def test_cartpole_preserves_routing_timing_and_training_modes(
        tmp_path, monkeypatch, mode, index, trainer_index, trigger):
    from braindance.examples.paper_cartpole.game import main
    env = Mock(name='env')
    env.name = 'run_cartpole_F'
    phase = Mock()
    manager = Mock()
    trainers = [object(), object(), object()]
    trainer = Mock(side_effect=trainers)
    permutations = Mock(return_value=['fixture pattern'])
    install_fake(monkeypatch, 'braindance.core.maxwell_env', MaxwellEnv=Mock(return_value=env))
    install_fake(monkeypatch, 'braindance.core.phases2', CartPolePhase=phase,
                 PhaseManager=Mock(return_value=manager))
    install_fake(monkeypatch, 'braindance.core.trainer', TetanusTrainer=trainer,
                 generate_permutations=permutations)
    mapping = Mock()
    mapping.get_orig_channels.return_value = [802, 41]
    install_fake(monkeypatch, 'braindance.analysis.mapping',
                 Mapping=SimpleNamespace(from_csv=Mock(return_value=mapping)))
    np.save(tmp_path / 'causal_connectivity_multi_mean.npy', [10., 20., 30., 40.])
    np.save(tmp_path / 'causal_connectivity_multi_std.npy', [1., 2., 3., 4.])
    config = dict(type=mode, name='run', config='routing.cfg', save_dir=str(tmp_path),
                  derived_dir=str(tmp_path), mapping_file_path='explicit_mapping.csv',
                  valid_stim_electrodes=[100, 200, 300, 400], sensory_electrodes=[100, 300],
                  motor_electrodes=[200, 400], stim_electrodes=[600, 200, 300, 700, 100, 400])
    main(config, run_index=index)
    args = phase.call_args.kwargs
    assert args['sensory_neurons'] == [3, 1]
    assert args['motor_neurons'] == [802, 41]
    assert args['training_neurons'] == [0, 2]
    assert [args[k] for k in ('read_period_ms', 'train_period_ms', 'wait_period_ms')] == [200, 400, 3000]
    assert args['trainer'] is trainers[trainer_index]
    assert args['training_type'] == trigger
    if mode in ('C1', 'C2'):
        assert args['normalization'] == 1
    else:
        np.testing.assert_array_equal(args['normalization'], [20, 40, 2, 4])
    permutations.assert_called_once_with([0, 2], stim_count=2, delay_ms=10)
    assert config['stim_electrodes'] == [600, 200, 300, 700, 100, 400]
    env.close.assert_called_once()


def test_rapid_pairing_pipeline_matches_original():
    root = Path(__file__).resolve().parents[1] / 'braindance/examples'
    old = ast.parse((root / 'closed_loop.py').read_text())
    new = ast.parse((root / '2_rapid_pairing.py').read_text())
    # The same acquisition/analysis/controller constructor settings are retained.
    names = {'RecordPhaseV3', 'RTSortPhaseV3', 'ConnectivityPhaseV3', 'ClosedLoopPhaseV3'}
    def calls(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id in names]
    assert calls(old) == calls(new)
