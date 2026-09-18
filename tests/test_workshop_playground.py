"""Manual controls must reach real dynamics without acquisition or decoding."""
import numpy as np
import pytest

from braindance.examples.streaming_workshop.environments import WorkshopGame
from braindance.examples.streaming_workshop.playground import Playground


@pytest.mark.parametrize('environment,action', [('cartpole', [.75]), ('foodland', [.5, 1.]), ('ant', [0.2] * 8)])
def test_manual_actions_match_adapter_trajectory(environment, action):
    if environment == 'foodland':
        pytest.importorskip('gym')
        pytest.importorskip('pygame')
    if environment == 'ant':
        pytest.importorskip('mujoco')
    playground = Playground()
    reference = WorkshopGame(environment, seed=42, episode_seconds=120.)
    try:
        initial = playground.control(dict(kind='reset', environment=environment, seed=42))
        np.testing.assert_allclose(initial['observation'], reference.reset())
        for _ in range(4):
            result = playground.control(dict(kind='step', action=action, steps=2))
            reference.step(action)
            obs, _, done = reference.step(action)
            np.testing.assert_allclose(result['observation'], obs)
            assert result['done'] == done
            assert result['action'] == action
            assert len(result['samples']) == 2
        assert result['time'] == pytest.approx(.16)
        assert result['scene']['kind'] == environment
    finally:
        reference.close()
        playground.close()


def test_manual_validation_and_replacement():
    pg = Playground()
    with pytest.raises(ValueError, match='Load'):
        pg.control(dict(kind='step', action=[0.]))
    try:
        pg.control(dict(kind='reset'))
        original = pg.game
        for action in ([2.], [float('nan')], [], [0., 0.]):
            with pytest.raises(ValueError, match='Action'):
                pg.control(dict(kind='step', action=action))
        for steps in (0, 6, 1.5, True):
            with pytest.raises(ValueError, match='Steps'):
                pg.control(dict(kind='step', action=[0.], steps=steps))
        for seed in (-1, 2**32, True, 1.5):
            with pytest.raises(ValueError, match='Seed'):
                pg.control(dict(kind='reset', seed=seed))
        assert pg.game is original
        assert pg.time == 0
        pg.game.max_steps = 1
        result = pg.control(dict(kind='step', action=[0.], steps=5))
        assert result['done'] and len(result['samples']) == 1
        with pytest.raises(ValueError, match='ended'):
            pg.control(dict(kind='step', action=[0.]))
        result = pg.control(dict(kind='reset'))
        assert not result['done'] and result['time'] == 0
        assert pg.game is not original
    finally:
        pg.close()
    pg.close()


def test_foodland_manual_action_order():
    # The public controls are turn/speed; the legacy game expects speed/turn.
    class Food:
        def step(self, action):
            self.received = action
            return np.zeros(9), 0., False, {}
    game = WorkshopGame.__new__(WorkshopGame)
    game.name, game.env = 'foodland', Food()
    game.steps, game.max_steps = 0, 10
    game._food_rng = np.random.get_state()
    game.step([-.5, .75])
    np.testing.assert_array_equal(game.env.received, [.75, -.5])
    assert game.action_bounds == [[-1., 1.], [0., 1.]]
