"""Load the context from a previous experiment"""
from braindance.core.phases_v3.experiment_v3 import Experiment
import argparse
import IPython

# Parse argument of the experiment to load, assuming it is in the current directory
args = argparse.ArgumentParser()
args.add_argument("--experiment", type=str, required=True)
args = args.parse_args()

# Load the experiment
exp = Experiment(args.experiment, save_dir="./test_experiments/spike_sort_test")

# Load the context
exp.load_data()
IPython.embed()