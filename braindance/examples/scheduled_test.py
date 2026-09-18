"""
Test script for Phase V3 spike sorting and data saving functionality
"""

from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import (
    RTSortPhaseV3,
    ConnectivityPhaseV3,
)
from braindance.core.phases_v3.phase_base_v3 import phase
import argparse
import numpy as np
import time
from braindance.core.utils import SmartPlug


# argparse config
parser = argparse.ArgumentParser(description="Closed-Loop Plasticity Experiment")
parser.add_argument(
    "--config", type=str, default=None, help="Maxwell configuration file"
)
parser.add_argument("--plug", type=str, default='nile_plug', required=True, help="Plug name")
parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
parser.add_argument(
    "--record_duration", type=int, default=600, help="Recording duration in seconds"
)
parser.add_argument(
    "--experiment_name", type=str, default="bee_ctrl", help="Experiment name"
)
parser.add_argument("--num_loops", type=int, default=48, help="Number of loops")
args = parser.parse_args()

num_loops = args.num_loops
print(f"Number of loops: {num_loops}, recordings will be 50m apart")
print(f"Recording duration: {args.record_duration} seconds")
print(f"Experiment name: {args.experiment_name}")

if args.plug is not None:
    plug = SmartPlug(args.plug)
else:
    raise ValueError("Plug name is required, use --plug <plug_name>")

params = {
    "config": args.config,
    "plug": args.plug,
    "plug_name": args.plug,
    "verbose": args.verbose,
    "record_duration": args.record_duration,
}

# Create experiment
exp = Experiment(
    args.experiment_name,
    params=params,
    save_dir=f"/media/mxwbio/anchor/{args.experiment_name}",
)


@phase("WaitFor50m")
def wait_for_50m(exp):
    # turn off the plug
    plug.turn_off()
    # simply
    time.sleep(50 * 60)
    plug.turn_on()
    return {"wait_for_50m": True}


# Put together the phases
for i in range(num_loops):
    exp.add_phase(RecordPhaseV3(duration=args.record_duration))  # Record for 20 seconds
    exp.add_phase(wait_for_50m)
exp.add_phase(RecordPhaseV3(duration=args.record_duration))

# Run experiment
success = exp.run()
print("\n✨ All tests completed!")
