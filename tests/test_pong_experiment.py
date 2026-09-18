from types import SimpleNamespace

import pytest

from braindance.experiments import pong


class FakeMaxwellEnv:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_channels = 16
        self.latest_frame = None
        self.calls = []
        self.stimuli = []
        self.closed = False
        self.packet = 0
        self.instances.append(self)

    def step(self, action=None):
        self.calls.append(action)
        frames = [100, 500, 900, 1_100, 1_500, 1_900]
        events = [[SimpleNamespace(channel=7)], [SimpleNamespace(channel=7)],
                  [SimpleNamespace(channel=8), SimpleNamespace(channel=8)], [], [], []]
        self.latest_frame = frames[self.packet]
        result = events[self.packet]
        self.packet += 1
        return result, False

    def stimulate(self, action):
        self.stimuli.append(action)

    def close(self):
        self.closed = True


class FakePongEnv:
    instances = []

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.actions = []
        self.closed = False
        self.instances.append(self)

    def reset(self, seed=None):
        return [0.6, 0.2], {}

    def step(self, action):
        self.actions.append(action)
        return [0.3, 0.8], 1.0, len(self.actions) == 2, False, {}

    def close(self):
        self.closed = True


def test_closed_loop_routes_spikes_stimulus_and_period(monkeypatch, tmp_path):
    FakeMaxwellEnv.instances.clear()
    FakePongEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", FakeMaxwellEnv)
    monkeypatch.setattr(pong, "PongEnv", FakePongEnv)
    config = tmp_path / "array.cfg"
    config.touch()
    rewards = pong.main(config=config, stim_electrodes=[101, 202],
                        up_channels=[7], down_channels=[8], save_dir=tmp_path,
                        episodes=1, render=False, read_period_ms=20)
    neural = FakeMaxwellEnv.instances[0]
    game = FakePongEnv.instances[0]
    assert rewards == [2.0]
    assert game.actions == [0, 1]
    assert neural.calls == [None] * 5
    assert neural.stimuli == [[("stim", [0], 200, 100)], [("stim", [1], 200, 100)]]
    assert neural.kwargs["stim_electrodes"] == [101, 202]
    assert neural.closed and game.closed


def test_empty_acquisition_done_stops_before_game_action(monkeypatch, tmp_path):
    class DoneMaxwell(FakeMaxwellEnv):
        def step(self, action=None):
            self.calls.append(action)
            self.latest_frame = 100
            return [], True

    FakePongEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", DoneMaxwell)
    monkeypatch.setattr(pong, "PongEnv", FakePongEnv)
    config = tmp_path / "array.cfg"
    config.touch()
    rewards = pong.main(config=config, stim_electrodes=[101, 202],
                        up_channels=[7], down_channels=[8], save_dir=tmp_path,
                        episodes=1, render=False)
    assert rewards == []
    assert FakePongEnv.instances[0].actions == []
    assert DoneMaxwell.instances[-1].calls == [None]
    assert DoneMaxwell.instances[-1].stimuli == []


def test_empty_spike_window_holds_previous_action(monkeypatch, tmp_path):
    class TieMaxwell(FakeMaxwellEnv):
        def step(self, action=None):
            self.calls.append(action)
            self.latest_frame = 100 if self.latest_frame is None else 1_100
            return [], False

    class OneStepPong(FakePongEnv):
        def step(self, action):
            self.actions.append(action)
            return [0.3, 0.8], 0.0, True, False, {}

    FakePongEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", TieMaxwell)
    monkeypatch.setattr(pong, "PongEnv", OneStepPong)
    config = tmp_path / "array.cfg"
    config.touch()
    pong.main(config=config, stim_electrodes=[101, 202], up_channels=[7],
              down_channels=[8], save_dir=tmp_path, episodes=1, render=False)
    assert FakePongEnv.instances[0].actions == [0]


def test_neural_environment_closes_if_game_construction_fails(monkeypatch, tmp_path):
    class BrokenPong:
        def __init__(self, render_mode=None):
            raise RuntimeError("display failed")

    FakeMaxwellEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", FakeMaxwellEnv)
    monkeypatch.setattr(pong, "PongEnv", BrokenPong)
    config = tmp_path / "array.cfg"
    config.touch()
    with pytest.raises(RuntimeError, match="display failed"):
        pong.main(config=config, stim_electrodes=[101, 202], up_channels=[7],
                  down_channels=[8], save_dir=tmp_path, render=False)
    assert FakeMaxwellEnv.instances[0].closed


def test_neural_environment_closes_if_game_cleanup_fails(monkeypatch, tmp_path):
    class BrokenClosePong(FakePongEnv):
        def close(self):
            raise RuntimeError("cleanup failed")

    class DoneMaxwell(FakeMaxwellEnv):
        def step(self, action=None):
            self.calls.append(action)
            self.latest_frame = 100
            return [], True

    FakeMaxwellEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", DoneMaxwell)
    monkeypatch.setattr(pong, "PongEnv", BrokenClosePong)
    config = tmp_path / "array.cfg"
    config.touch()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pong.main(config=config, stim_electrodes=[101, 202], up_channels=[7],
                  down_channels=[8], save_dir=tmp_path, render=False)
    assert FakeMaxwellEnv.instances[0].closed


def test_both_environments_close_if_acquisition_fails(monkeypatch, tmp_path):
    class BrokenMaxwell(FakeMaxwellEnv):
        def step(self, action=None):
            raise RuntimeError("acquisition failed")

    FakeMaxwellEnv.instances.clear()
    FakePongEnv.instances.clear()
    monkeypatch.setattr(pong, "MaxwellEnv", BrokenMaxwell)
    monkeypatch.setattr(pong, "PongEnv", FakePongEnv)
    config = tmp_path / "array.cfg"
    config.touch()
    with pytest.raises(RuntimeError, match="acquisition failed"):
        pong.main(config=config, stim_electrodes=[101, 202], up_channels=[7],
                  down_channels=[8], save_dir=tmp_path, render=False)
    assert FakeMaxwellEnv.instances[0].closed
    assert FakePongEnv.instances[0].closed


@pytest.mark.parametrize("kwargs, message", [
    ({"stim_electrodes": [1]}, "above and below"),
    ({"up_channels": []}, "both be non-empty"),
    ({"up_channels": [8]}, "must not overlap"),
    ({"stim_electrodes": [1, 1]}, "must be distinct"),
    ({"down_channels": [-1]}, "non-negative integers"),
])
def test_hardware_routing_configuration_is_required(kwargs, message, tmp_path):
    config = tmp_path / "array.cfg"
    config.touch()
    params = dict(config=config, stim_electrodes=[1, 2],
                  up_channels=[7], down_channels=[8])
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        pong.main(**params)
