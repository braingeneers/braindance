import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from braindance.games.pong_env import PaddleBallEnv, PongEnv


def test_pong_env_follows_gymnasium_contract():
    env = PongEnv()
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert observation.shape == (6,)
    assert env.observation_space.contains(observation)
    assert info == {}
    assert PaddleBallEnv is PongEnv


def test_seed_reproduces_initial_ball_velocity():
    env = PongEnv()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)


def test_left_paddle_hit_rewards_and_reflects_ball():
    env = PongEnv()
    env.reset(seed=1)
    env.ball_x = 0.07
    env.ball_y = env.paddle_pos
    env.ball_velocity_x = -env.ball_speed
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == 1.0
    assert env.ball_velocity_x > 0
    assert not terminated
    assert not truncated
    assert info == {"hit": True}


def test_rgb_array_render_is_headless():
    pytest.importorskip("pygame")
    env = PongEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame.shape == (600, 800, 3)
    env.close()


def test_episode_is_truncated_at_configured_step_limit():
    env = PongEnv(max_episode_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(1)
    assert not terminated
    assert truncated


def test_ball_cannot_be_caught_after_passing_paddle():
    env = PongEnv()
    env.reset(seed=1)
    env.ball_x = 0.01
    env.ball_y = env.paddle_pos
    env.ball_velocity_x = -env.ball_speed
    _, reward, terminated, _, info = env.step(1)
    assert reward == 0.0
    assert terminated
    assert info == {"hit": False}


def test_two_player_right_paddle_reflects_ball():
    env = PongEnv(play_human=True)
    env.reset(seed=1)
    right_edge = 1 - (10 + env.paddle_width + env.ball_radius) / env.screen_width
    env.ball_x = right_edge - env.ball_speed / 2
    env.ball_y = env.human_paddle_pos
    env.ball_velocity_x = env.ball_speed
    _, _, terminated, _, _ = env.step(np.array([0, 1]))
    assert env.ball_velocity_x < 0
    assert not terminated


@pytest.mark.parametrize("env, action", [
    (PongEnv(), 2),
    (PongEnv(play_human=True), np.array([0.5, 1.0])),
])
def test_invalid_action_does_not_advance_environment(env, action):
    env.reset(seed=1)
    before = (env.elapsed_steps, env.paddle_pos, env.ball_x, env.ball_y)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(action)
    assert (env.elapsed_steps, env.paddle_pos, env.ball_x, env.ball_y) == before
