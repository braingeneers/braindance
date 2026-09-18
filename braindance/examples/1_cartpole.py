"""Paper CartPole using Phase V3.

Baseline → footprint analysis → causal screening → ranked pairs → game.
Supply a JSON with config (Maxwell routing), stim_electrodes (physical IDs),
and type (C1/C2/C7/punishment/reward/always). Use --resume after a failed phase.
The historical recording/causal/rank/run commands remain available.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    # Keep explicit historical stage commands available for manual preparation.
    if len(sys.argv) > 1 and sys.argv[1] in ('recording', 'causal', 'rank', 'run'):
        from braindance.examples.paper_cartpole.cli import main as run_stage
        return run_stage()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', '-j', required=True, help='Experiment configuration JSON')
    parser.add_argument('--record_duration', type=int, default=300)
    parser.add_argument('--project_id', default='cartpole')
    parser.add_argument('--chip_id', default='default_chip')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--select-rank', type=int, default=1, help='1-based ranked pair selection')
    parser.add_argument('--order', choices=('first', 'multi'), default='multi')
    parser.add_argument('--run-index', type=int, default=0)
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--max-time-sec', type=float, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.json).read_text())
    if args.record_duration < 60:
        parser.error('Footprint analysis requires --record_duration >= 60 seconds')
    if args.select_rank < 1 or args.run_index < 0 or args.n_episodes < 1 or args.max_time_sec <= 0:
        parser.error('rank, episodes, and time must be positive; run-index must be nonnegative')
    if not config.get('config'):
        parser.error('JSON must specify a Maxwell routing config')
    electrodes = config.get('stim_electrodes', [])
    if len(electrodes) < 6 or len(set(electrodes)) != len(electrodes):
        parser.error('Provide at least six unique, hardware-routable stim_electrodes '
                     '(two sensory, two motor, and at least two training electrodes)')
    if config.get('type') not in ('C1', 'C2', 'C7', 'punishment', 'reward', 'always'):
        parser.error('JSON type must be C1, C2, C7, punishment, reward, or always')

    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.core.phases_v3.phases3 import RecordPhaseV3
    from braindance.core.phases_v3.phases3_cartpole import (
        CartPoleFootprintPhaseV3, CartPoleCausalSweepPhaseV3,
        CartPoleCausalAnalysisPhaseV3, CartPoleRankPairsPhaseV3, PaperCartPolePhaseV3,
    )

    params = dict(config, verbose=True)
    params.pop('record_duration', None)  # Duration belongs to the baseline phase.
    exp = Experiment(
        args.experiment_name or config.get('name', 'cartpole'),
        params=params,
        save_dir=config.get('save_dir'),
        project_id=args.project_id,
        chip_id=args.chip_id,
        auto_load_data=False,
        overwrite_existing=True,  # Analysis replaces configured selections in DataContext.
    )

    # --- Build phase pipeline ---
    exp.add_phase(RecordPhaseV3(duration=args.record_duration))
    exp.add_phase(CartPoleFootprintPhaseV3())
    exp.add_phase(CartPoleCausalSweepPhaseV3())
    exp.add_phase(CartPoleCausalAnalysisPhaseV3())
    exp.add_phase(CartPoleRankPairsPhaseV3(rank=args.select_rank, order=args.order))
    exp.add_phase(PaperCartPolePhaseV3(
        run_index=args.run_index, n_episodes=args.n_episodes,
        max_time_sec=args.max_time_sec,
    ))

    # --- Run (with optional resume) ---
    success = exp.run(resume=args.resume)
    if success:
        print('\n All phases completed!')
    else:
        print(f'\n Experiment stopped at phase {exp.current_phase_idx}. '
              'Re-run with --resume to continue.')


if __name__ == '__main__':
    main()
