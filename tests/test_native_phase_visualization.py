"""Native environment observation uses browser geometry outside phase code."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from braindance.examples.streaming_workshop.native_visualization import ObservedGame, game_scene
from braindance.examples.streaming_workshop.native_catalog import native_catalog
from braindance.examples.streaming_workshop.custom_analysis import analysis_contracts


def test_supported_catalog_and_legacy_lookup():
    catalog = native_catalog()
    assert {entry['class_name'] for entry in catalog.values()} == {
        'RecordPhaseV3', 'FrequencyStimPhaseV3', 'NeuralSweepPhaseV3',
        'RTSortPhaseV3', 'CartPolePhase', 'FoodLandPhaseV3', 'AntPhaseV3'}
    assert all('inputs' in entry and 'outputs' in entry for entry in catalog.values())
    assert len(native_catalog(include_legacy=True)) > len(catalog)


@pytest.mark.parametrize('input_name,output_name', [('inputs', 'outputs'), ('requires', 'provides')])
def test_analysis_decorator_spelling(input_name, output_name):
    code = f'@analysis_phase({input_name}=["recording_baseline_hz"], {output_name}=["mean_rate"])\ndef rates(exp):\n    return {{}}\n'
    report = analysis_contracts(code)
    assert not report['errors']
    assert report['contracts']['rates'] == dict(inputs=['recording_baseline_hz'], outputs=['mean_rate'])


def test_cartpole_native_observer(tmp_path):
    gym = pytest.importorskip('gymnasium')
    game = gym.make('CartPole-v1', render_mode=None)
    progress = tmp_path / 'progress.json'
    observed = ObservedGame(game, 'cartpole', progress, 'balance')
    try:
        observation, _ = observed.reset(seed=2)
        state = json.loads(progress.read_text())
        assert state['phase_kind'] == 'environment'
        assert state['scene']['observation'] == observation.tolist()
        result = observed.step(0)
        observed.publish(result[0], force=True)
        state = json.loads(progress.read_text())
        assert state['reward'] == result[1]
        assert state['scene']['observation'] == result[0].tolist()
    finally:
        observed.close()


def test_ant_scene_uses_current_mujoco_geometry():
    gym = pytest.importorskip('gymnasium')
    pytest.importorskip('mujoco')
    game = gym.make('Ant-v5', render_mode=None)
    try:
        observation, _ = game.reset(seed=1)
        scene = game_scene('ant', SimpleNamespace(env=game), observation)
        assert scene['geometry']
        assert scene['root'] == game.unwrapped.data.xpos[1].tolist()
        assert all(np.isfinite(item['a']).all() and np.isfinite(item['b']).all() for item in scene['geometry'])
        assert all(item['radius'] > 0 for item in scene['geometry'])
    finally:
        game.close()


def test_foodland_scene_uses_live_positions():
    env = SimpleNamespace(agent_pos=np.array([50, 60]), agent_dir=.5,
                          food_positions=np.array([[2, 3]]), spike_positions=np.array([[8, 9]]))
    scene = game_scene('foodland', SimpleNamespace(env=env), [1., 2.])
    assert scene['agent'] == [50, 60]
    assert scene['food'] == [[2, 3]]
    assert scene['hazards'] == [[8, 9]]
