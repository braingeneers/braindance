from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from braindance.games.labyrinth import LabyrinthEnv
from braindance.experiments.labyrinth import LabyrinthPhase, encode_position, decode_counts


def test_gym_contract():
    env = LabyrinthEnv()
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        env.step([np.nan, 0])
    for _ in range(100):
        obs, _, ended, timeout, _ = env.step([0, 1])
        assert env.observation_space.contains(obs)
        assert obs[1] <= 0.27 - env.radius + 1e-6
        assert not ended and not timeout


@pytest.mark.parametrize('outcome,position,reward', [
    ('goal', [0.9, 0.88], 1), ('hole', [0.48, 0.16], -1)])
def test_terminal_paths(outcome, position, reward):
    env = LabyrinthEnv()
    env.reset()
    env.position = np.array(position)
    _, actual, terminated, truncated, info = env.step([0, 0])
    assert (actual, terminated, truncated, info['outcome']) == (reward, True, False, outcome)
    with pytest.raises(RuntimeError):
        env.step([0, 0])


def test_timeout_and_reset_required():
    env = LabyrinthEnv(max_episode_steps=1)
    with pytest.raises(RuntimeError):
        env.step([0, 0])
    env.reset()
    assert env.step([0, 0])[2:4] == (False, True)
    with pytest.raises(RuntimeError):
        env.step([0, 0])
    env.reset()
    env.step([0, 0])


def test_all_ten_place_fields_and_direction_mapping():
    env = LabyrinthEnv()
    for i, center in enumerate(env.place_centers):
        activation, index = encode_position(center, env.place_centers)
        assert index == i and activation[i] == 1
    for counts, expected in [([1, 0, 0, 0], [-1, 0]), ([0, 1, 0, 0], [1, 0]),
                             ([0, 0, 1, 0], [0, -1]), ([0, 0, 0, 1], [0, 1]),
                             ([0, 0, 0, 0], [0, 0]), ([2, 2, 4, 4], [0, 0])]:
        np.testing.assert_array_equal(decode_counts(counts, [1, 1, 1, 1]), expected)
    np.testing.assert_array_equal(decode_counts([4, 2, 0, 0], [2, 1, 1, 1]), [0, 0])


def phase(**kwargs):
    return LabyrinthPhase(list(range(100, 110)),
                          dict(left=[0], right=[1], up=[2], down=[3]),
                          amplitude_mv=100, phase_width_us=100, **kwargs)


class FakeAcquisition:
    stim_electrodes = list(reversed(range(100, 110)))
    num_channels = 16
    latest_frame = None

    def __init__(self, done_at=4, fail=False):
        self.reads, self.stimuli = 0, []
        self.done_at, self.fail = done_at, fail

    def step(self):
        self.reads += 1
        if self.fail and self.reads == 2:
            raise RuntimeError('acquisition failed')
        self.latest_frame = self.reads * 500
        return [SimpleNamespace(channel=1)], self.reads >= self.done_at

    def stimulate(self, action):
        self.stimuli.append(action)


def test_phase_routes_and_accumulates_complete_window():
    controller = phase(max_decisions=2)
    neural = FakeAcquisition()
    controller.set_env(neural)
    result = controller.run(None)['labyrinth_results']
    assert neural.reads == 4
    # Starting place field 0 -> physical electrode 100 -> configured local index 9.
    assert neural.stimuli == [[('stim', [9], 100, 100)]]
    assert result['windows'][0]['counts'] == [0, 3, 0, 0]
    assert result['windows'][0]['tilt'] == [1, 0]
    assert result['windows'][0]['last_frame'] - result['windows'][0]['first_frame'] == 1000
    assert result['windows'][0]['next_observation'][0] > 0.1
    assert result['stop_reason'] == 'acquisition_done'
    assert controller.game.closed
    controller.cleanup()


@pytest.mark.parametrize('done_at', [1, 2])
def test_stream_completion_prevents_extra_stimulation_or_game_step(done_at):
    controller = phase()
    neural = FakeAcquisition(done_at=done_at)
    controller.set_env(neural)
    result = controller.run(None)['labyrinth_results']
    assert len(neural.stimuli) == done_at - 1
    assert result['windows'] == []
    assert controller.game.closed


def test_cleanup_on_acquisition_error():
    controller = phase()
    controller.set_env(FakeAcquisition(fail=True))
    with pytest.raises(RuntimeError, match='acquisition failed'):
        controller.run(None)
    assert controller.game.closed


def test_phase_episode_timeout_stops_without_extra_step():
    controller = phase(episodes=1, episode_seconds=0.05)
    controller.set_env(FakeAcquisition())
    result = controller.run(None)['labyrinth_results']
    assert len(result['windows']) == 1
    assert result['episodes'][0]['truncated']
    assert not result['episodes'][0]['terminated']
    assert result['stop_reason'] == 'episodes_complete'


def test_render_headless():
    pytest.importorskip('pygame')
    env = LabyrinthEnv(render_mode='rgb_array')
    env.reset()
    image = env.render()
    assert image.shape == (720, 1040, 3) and image.dtype == np.uint8
    env.close()


def test_environment_configuration_and_invalid_routing():
    controller = phase()
    params = controller.customize_environment_params({'observation_type': 'raw'})
    assert params['observation_type'] == 'spikes'
    assert params['stim_electrodes'] == list(range(100, 110))
    neural = FakeAcquisition()
    neural.stim_electrodes = [999]
    controller.set_env(neural)
    with pytest.raises(ValueError, match='exactly once'):
        controller.run(None)


def test_real_experiment_runner_with_neural_simulation(monkeypatch, tmp_path):
    from braindance.examples.labyrinth import main
    monkeypatch.setenv('BRAINDANCE_OUTPUT_DIR', str(tmp_path))
    results = main(headless=True, steps=3, neural_simulation=True)
    assert len(results['windows']) == 3
    assert results['stop_reason'] == 'decision_limit'
    assert all(w['last_frame'] - w['first_frame'] >= 1000 for w in results['windows'])


def test_stalled_acquisition_raises_and_cleans_up():
    class Stalled(FakeAcquisition):
        def step(self):
            self.latest_frame = 10
            return [], False
    controller = phase()
    controller.set_env(Stalled())
    with pytest.raises(RuntimeError, match='did not advance'):
        controller.run(None)
    assert controller.game.closed
