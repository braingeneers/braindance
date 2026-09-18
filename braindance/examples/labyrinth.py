"""Try the labyrinth without hardware: python -m braindance.examples.labyrinth."""
import argparse
import numpy as np

from braindance.games.labyrinth import LabyrinthEnv
from braindance.experiments.labyrinth import encode_position, decode_counts


def main(headless=False, steps=600, seed=0, screenshot=None, neural_simulation=False):
    if neural_simulation:
        from braindance.config import get_output_dir
        from braindance.core.phases_v3.experiment_v3 import Experiment
        from braindance.core.simulation import NeuralSimulationSource
        from braindance.experiments.labyrinth import LabyrinthPhase

        source = NeuralSimulationSource(num_channels=14, seed=seed,
                                        speed=0 if headless else 1,
                                        duration_s=steps * 0.05 + 2)
        experiment = Experiment(
            "labyrinth_simulation",
            save_dir=get_output_dir() / "labyrinth_simulation",
            params={"maxwell_env": {"replay": {"source": source,
                                               "write_output": False}, "verbose": 0}},
            auto_load_data=False)
        experiment.add_phase(LabyrinthPhase(
            input_electrodes=list(range(10)),
            output_channels=dict(left=[10], right=[11], up=[12], down=[13]),
            # Synthetic stimulus values for the simulator only.
            amplitude_mv=100, phase_width_us=100, max_decisions=steps,
            render=not headless, seed=seed))
        if not experiment.run():
            raise RuntimeError("Labyrinth simulation failed")
        return experiment.data.labyrinth_results
    env = LabyrinthEnv(render_mode="rgb_array" if headless else "human")
    try:
        observation, _ = env.reset(seed=seed)
        env.input_activity, _ = encode_position(observation[:2], env.place_centers)
        env.render()
        pg = env._pygame
        clock = pg.time.Clock()
        for step in range(steps if headless else 10**9):
            if env.closed:
                break
            if headless:
                # Synthetic output activity for a reproducible visualization.
                counts = [0, 4, 0, 0] if step % 240 < 120 else [0, 0, 0, 4]
            else:
                keys = pg.key.get_pressed()
                if keys[pg.K_ESCAPE]:
                    break
                if keys[pg.K_r]:
                    observation, _ = env.reset(seed=seed)
                counts = [int(keys[key]) for key in (pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN)]
            env.output_activity = np.asarray(counts) / max(1, max(counts))
            action = decode_counts(counts, [1, 1, 1, 1])
            observation, reward, terminated, truncated, info = env.step(action)
            env.input_activity, _ = encode_position(observation[:2], env.place_centers)
            env.render()
            if terminated or truncated:
                print(f"Episode ended: {info['outcome']} (reward {reward:+g})")
                observation, _ = env.reset(seed=seed)
            if not headless:
                clock.tick(env.metadata["render_fps"])
        if screenshot:
            pg.image.save(env.screen, screenshot)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--neural-simulation", action="store_true",
                        help="Run the actual BrainDance phase with synthetic neural activity")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screenshot")
    main(**vars(parser.parse_args()))
