"""Historical standalone launcher; shared session setup now lives in Phase V3."""
from braindance.core.phases_v3.phases3_cartpole import prepare_session


def main(config, run_index=0, n_episodes=200, max_time_sec=900):
    import json
    from pathlib import Path
    from braindance.core.maxwell_env import MaxwellEnv
    from braindance.core.phases2 import CartPolePhase, PhaseManager

    config, params, phase_params = prepare_session(
        config, run_index, n_episodes, max_time_sec)
    env = MaxwellEnv(**params)
    try:
        (Path(params['save_dir']) / f"{env.name}_example.json").write_text(
            json.dumps(config, indent=2) + "\n")
        phase = CartPolePhase(env, **phase_params)
        manager = PhaseManager(env)
        manager.add_phase(phase)
        print(manager.summary())
        manager.run()
    finally:
        env.close()
