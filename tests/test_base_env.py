"""Tests for braindance.core.base_env.BaseEnv."""

import time

import pytest

from braindance.core.base_env import BaseEnv


class ConcreteEnv(BaseEnv):
    """Minimal concrete subclass of BaseEnv for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_called = False

    def _cleanup(self):
        self.cleanup_called = True

    def step(self, action):
        return action

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass


class TestBaseEnvInit:
    """Tests for BaseEnv.__init__."""

    def test_default_parameters(self):
        """Tests:
        - max_time_sec defaults to 60
        - verbose defaults to 1
        """
        env = ConcreteEnv()
        assert env.max_time_sec == 60
        assert env.verbose == 1

    def test_custom_parameters(self):
        """Tests:
        - max_time_sec and verbose accept custom values
        """
        env = ConcreteEnv(max_time_sec=120, verbose=0)
        assert env.max_time_sec == 120
        assert env.verbose == 0


class TestBaseEnvInitTimeManagement:
    """Tests for BaseEnv._init_time_management."""

    def test_sets_start_time_and_cur_time(self):
        """Tests:
        - start_time is set to a recent timestamp
        - cur_time equals start_time after initialization
        """
        env = ConcreteEnv()
        env._init_time_management()
        assert hasattr(env, "start_time")
        assert hasattr(env, "cur_time")
        assert env.start_time == env.cur_time

    def test_start_time_is_recent(self):
        """Tests:
        - start_time is close to current perf_counter value
        """
        env = ConcreteEnv()
        before = time.perf_counter()
        env._init_time_management()
        after = time.perf_counter()
        assert before <= env.start_time <= after


class TestBaseEnvTimeElapsed:
    """Tests for BaseEnv.time_elapsed."""

    def test_returns_increasing_values(self):
        """Tests:
        - time_elapsed returns a non-negative value that increases over time
        """
        env = ConcreteEnv()
        env._init_time_management()
        t1 = env.time_elapsed()
        # Burn a tiny amount of time
        for _ in range(1000):
            pass
        t2 = env.time_elapsed()
        assert t1 >= 0
        assert t2 >= t1

    def test_returns_value_close_to_actual_sleep(self):
        """Tests:
        - time_elapsed roughly reflects real wall-clock time
        """
        env = ConcreteEnv()
        env._init_time_management()
        time.sleep(0.05)
        elapsed = env.time_elapsed()
        assert elapsed >= 0.04  # allow small timing tolerance


class TestBaseEnvDt:
    """Tests for BaseEnv.dt property."""

    def test_dt_returns_time_since_cur_time(self):
        """Tests:
        - dt reflects elapsed time since cur_time was last set
        """
        env = ConcreteEnv()
        env._init_time_management()
        time.sleep(0.05)
        dt_val = env.dt
        assert dt_val >= 0.04

    def test_dt_resets_when_cur_time_updated(self):
        """Tests:
        - After manually updating cur_time, dt returns a smaller value
        """
        env = ConcreteEnv()
        env._init_time_management()
        time.sleep(0.05)
        dt_before = env.dt
        env.cur_time = time.perf_counter()
        dt_after = env.dt
        assert dt_after < dt_before


class TestBaseEnvCheckIfDone:
    """Tests for BaseEnv._check_if_done."""

    def test_returns_false_when_time_not_exceeded(self):
        """Tests:
        - Returns False when elapsed time is less than max_time_sec
        """
        env = ConcreteEnv(max_time_sec=60)
        env._init_time_management()
        assert env._check_if_done() is False

    def test_returns_true_when_time_exceeded(self):
        """Tests:
        - Returns True when elapsed time exceeds max_time_sec
        """
        env = ConcreteEnv(max_time_sec=0.01)
        env._init_time_management()
        time.sleep(0.02)
        assert env._check_if_done() is True

    def test_calls_cleanup_when_time_exceeded(self):
        """Tests:
        - _cleanup is called when max time is exceeded
        """
        env = ConcreteEnv(max_time_sec=0.01)
        env._init_time_management()
        time.sleep(0.02)
        env._check_if_done()
        assert env.cleanup_called is True

    def test_does_not_call_cleanup_when_time_not_exceeded(self):
        """Tests:
        - _cleanup is not called when time has not been exceeded
        """
        env = ConcreteEnv(max_time_sec=60)
        env._init_time_management()
        env._check_if_done()
        assert env.cleanup_called is False

    def test_prints_message_when_verbose(self, capsys):
        """Tests:
        - A message is printed when verbose >= 1 and time is exceeded
        """
        env = ConcreteEnv(max_time_sec=0.01, verbose=1)
        env._init_time_management()
        time.sleep(0.02)
        env._check_if_done()
        captured = capsys.readouterr()
        assert "Max time" in captured.out

    def test_no_message_when_not_verbose(self, capsys):
        """Tests:
        - No message is printed when verbose < 1 and time is exceeded
        """
        env = ConcreteEnv(max_time_sec=0.01, verbose=0)
        env._init_time_management()
        time.sleep(0.02)
        env._check_if_done()
        captured = capsys.readouterr()
        assert captured.out == ""


class TestBaseEnvAbstractMethods:
    """Tests for abstract methods that raise NotImplementedError."""

    def test_cleanup_raises(self):
        """Tests:
        - _cleanup raises NotImplementedError on the base class
        """
        env = BaseEnv()
        with pytest.raises(NotImplementedError):
            env._cleanup()

    def test_step_raises(self):
        """Tests:
        - step raises NotImplementedError on the base class
        """
        env = BaseEnv()
        with pytest.raises(NotImplementedError):
            env.step(None)

    def test_reset_raises(self):
        """Tests:
        - reset raises NotImplementedError on the base class
        """
        env = BaseEnv()
        with pytest.raises(NotImplementedError):
            env.reset()

    def test_render_raises(self):
        """Tests:
        - render raises NotImplementedError on the base class
        """
        env = BaseEnv()
        with pytest.raises(NotImplementedError):
            env.render()

    def test_close_raises(self):
        """Tests:
        - close raises NotImplementedError on the base class
        """
        env = BaseEnv()
        with pytest.raises(NotImplementedError):
            env.close()


class TestBaseEnvEdgeMaxTimeSec:
    """Edge-case tests for max_time_sec boundary values.

    Tests:
    - Zero max_time_sec causes immediate done
    - Negative max_time_sec causes immediate done
    - Very large max_time_sec does not trigger done
    """

    def test_zero_max_time(self):
        """Tests:
        - With max_time_sec=0, _check_if_done returns True immediately
        """
        env = ConcreteEnv(max_time_sec=0)
        env._init_time_management()
        assert env._check_if_done() is True

    def test_negative_max_time(self):
        """Tests:
        - With max_time_sec=-1, _check_if_done returns True (already elapsed)
        """
        env = ConcreteEnv(max_time_sec=-1)
        env._init_time_management()
        assert env._check_if_done() is True

    def test_very_large_max_time(self):
        """Tests:
        - With max_time_sec=999999, _check_if_done returns False
        """
        env = ConcreteEnv(max_time_sec=999999)
        env._init_time_management()
        assert env._check_if_done() is False


class TestBaseEnvEdgeTimeManagement:
    """Edge-case tests for time management initialization.

    Tests:
    - Accessing time_elapsed before _init_time_management raises AttributeError
    - Accessing dt before _init_time_management raises AttributeError
    - Calling _init_time_management twice resets start_time
    """

    def test_time_elapsed_before_init(self):
        """Tests:
        - time_elapsed raises AttributeError when called before _init_time_management
        """
        env = ConcreteEnv()
        with pytest.raises(AttributeError):
            env.time_elapsed()

    def test_dt_before_init(self):
        """Tests:
        - dt raises AttributeError when accessed before _init_time_management
        """
        env = ConcreteEnv()
        with pytest.raises(AttributeError):
            _ = env.dt

    def test_double_init_resets_times(self):
        """Tests:
        - Calling _init_time_management a second time resets start_time to a newer value
        """
        env = ConcreteEnv()
        env._init_time_management()
        first_start = env.start_time
        time.sleep(0.05)
        env._init_time_management()
        second_start = env.start_time
        assert second_start > first_start