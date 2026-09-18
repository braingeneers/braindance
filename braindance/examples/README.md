# Examples

## Load a saved experiment

[load_test_data.py](load_test_data.py) is a small, commented walkthrough using the
already-downloaded tutorial data. It shows catalog selection, spike trains (in
milliseconds), stimulation logs, electrode mapping, recording results, and the
experiment's saved phase `DataContext`. It plots a spike raster, population rate,
and saved connectivity matrix.

```sh
python -m braindance.examples.load_test_data
# If downloaded to a custom cache; save plots without opening windows:
python -m braindance.examples.load_test_data --cache-dir /path/to/cache --no-show
```

Plots go to `get_output_dir() / "load_test_data"`. For interactive exploration:

```python
from braindance.examples.load_test_data import main
rec, exp = main(show=False)
rec.spikes.train[0]             # first neuron's spike times in milliseconds
rec.stim_log.head()             # stimulation events
rec.results["sorted_spikedata"] # recording-level saved result
exp.data.connectivity_matrix   # shared context loaded from the baseline phase
```

The script uses offline mode. If needed, download first with
`python -m braindance.examples.get_tutorial_data` (same `--cache-dir`, if supplied).
To adapt it to your own experiment, use `load_catalog()` with your configured
data paths, change the recording filter, and point `Experiment` and `load_data`
at your existing experiment and phase-results directories.

## Paper base examples

The numbered entrypoints follow the paper order:

1. [CartPole](1_cartpole.py): baseline/footprint selection → stimulus-response screening → ranked sensory/motor pairs → CartPole with stimulation training.
2. [Rapid pairing](2_rapid_pairing.py): baseline → RT-Sort → STTC → pair selection → spike-triggered stimulation → post recording.
3. [BusyBee](3_busybee.py): repeated spontaneous recordings and frequency sweeps.

These are reference entrypoints being prepared for release. Live acquisition needs
Maxwell, its vendor SDK, routing configuration, and the relevant scientific dependencies.
Use Python 3.11 and install `.[analysis,rl]` for CartPole/BusyBee; rapid pairing also
needs `.[rtsort]`, CUDA and recording-specific sorting support. Manage the Maxwell
server externally. Example 1 does not launch a workstation-specific server or smart plug.

Run modules with `python -m braindance.examples.1_cartpole --help` (similarly for
2 and 3). Numeric module names work with `python -m`; use `importlib.import_module`
if importing them from Python.

## 1. CartPole: Phase V3

The default entrypoint builds a resumable `Experiment`, in the same style as
`2_rapid_pairing.py`: recording → footprint selection → causal sweep → causal
analysis → ranked-pair selection → one CartPole session. It preserves the paper's
count-based connectivity metric, direct neural force control, and tetanus trainer.

```sh
python -m braindance.examples.1_cartpole --json experiment.json
python -m braindance.examples.1_cartpole --json experiment.json --resume
```

The JSON needs `config` (Maxwell routing file), `stim_electrodes` (physical IDs of
hardware-routable candidates), and `type` (training mode from the table below).
Optional `name` and `save_dir` control the experiment name and output directory;
`--project_id`, `--chip_id`, and `--experiment_name` also configure the V3 experiment.
Footprint selection filters the configured stimulation pool; at least six electrodes
must survive to allow two sensory, two motor, and at least two training electrodes.
It never adds electrodes outside that configured pool. If too few survive, inspect
the baseline and routing before starting a new experiment with revised settings.

The baseline defaults to 300 seconds (`--record_duration`, minimum 60). Screening
uses 50 repeats at 2 Hz, 400 mV and 200 μs phase width. `--select-rank` defaults to
1 and `--order` to `multi`. The game defaults to 200 episodes with a 900-second cap
(`--n-episodes`, `--max-time-sec`). Use `--run-index` for the C1/C2 session cycle.
Selections, mappings, derived matrices, and logs are saved under the V3 recording
folders; the input JSON is not rewritten. Keep the same arguments when resuming.
The implementations are native V3 phases: `phases3_cartpole.py` owns ranking,
routing, neural encoding/decoding, the CartPole game loop, and training;
`phases3_cartpole_analysis.py` owns activity/footprint selection and evoked-response
analysis. The default pipeline does not invoke legacy phase classes or require
RT-Sort. Shared low-level recording, artifact-removal, game-physics, and trainer
utilities are still used. V3 closes game logs on completion or failure and saves
footprints and response arrays alongside the recording results.

### Historical stage commands

The explicit stage commands remain available for manual preparation and offline
ranking. Their working-JSON behavior is unchanged.


The GUI maps **Recording**, **Causal**, **Rank pairs**, and **Cartpole-Force Train**
to separate scripts. `proj/cartpole_v2/full_cartpole.py` stops before gameplay.
The remembered selection function is `find_connectivity_patterns` in
`proj/cartpole_v2/ranked_pairs.py`, not V3's automatic allocation of RT-Sort units.

Start with a **working copy** of the experiment JSON. Preparation and explicit
rank selection update this working JSON. Required initial fields are `name`,
`save_dir` (new acquisition/output location), and `config` (Maxwell routing file).
The baseline stage saves footprint selections; provide the chosen, hardware-routable
`stim_electrodes` before causal screening, as in the GUI workflow.

```sh
python -m braindance.examples.1_cartpole recording --json experiment.json
python -m braindance.examples.1_cartpole causal --json experiment.json
python -m braindance.examples.1_cartpole rank --json experiment.json --derived-dir /path/to/derived --order multi
# Save a chosen rank, or retain manually selected electrodes in the JSON:
python -m braindance.examples.1_cartpole rank --json experiment.json --derived-dir /path/to/derived --order multi --select-rank 1
python -m braindance.examples.1_cartpole run --json experiment.json --derived-dir /path/to/derived
```

`rank` is offline. It requires `valid_stim_electrodes` in the exact row/column order
of `causal_connectivity_first.npy` or `causal_connectivity_multi.npy`. Set this from
the screening output; do not substitute the original requested electrode list.
Historical code assumed `{name}_1/derived`; other runs write `plots/derived`.
The example requires an explicit directory rather than guessing from filenames.
`--order first` and `--order multi` expose the two historical rankings; multi is the
example default, not a claim about which one was selected for a particular paper run.

For distinct `(a,b,c,d)`, sensory electrodes are `(a,c)` and motors `(b,d)`.
With column mean μ, standard deviation σ, and z = (C−μ)/σ, the original score is:

```text
z[a,b]σ[b] + z[c,d]σ[d] − z[a,d]σ[d] − z[c,b]σ[b]
− 0.3z[a,c] − 0.3z[c,a] − 5|μ[b]−μ[d]|
```

The original script printed candidates without saving a choice. `--select-rank`
now stores physical electrode IDs, rank, score, matrix order/path/hash and reference
source commit. Zero-variance columns are rejected because the metric is undefined.
Ranking is exhaustive over four distinct electrodes; large matrices can be slow.
This heuristic is preserved for provenance, not presented as a newly validated metric.

The game stage additionally requires:

- `type`: one of the modes below.
- `sensory_electrodes`, `motor_electrodes`: two distinct physical electrodes each.
- `stim_electrodes`: sensory and training electrodes; motor electrodes are removed
  from the stimulation route. At least two training electrodes are required.
- `mapping_file_path`: explicit baseline mapping CSV, used to obtain motor channels.
- `valid_stim_electrodes`: normalization-array order.
- `causal_connectivity_multi_mean.npy` and `causal_connectivity_multi_std.npy` in
  the derived directory, regardless of the ranking order chosen.

| JSON type | Trainer across sessions | Normalization | Training trigger |
| --- | --- | --- | --- |
| C1 | none, adaptive, none | 1 | punishment |
| C2 | none, random, none | 1 | punishment |
| C7 / punishment | adaptive | historical motor normalization | punishment |
| reward | adaptive | historical motor normalization | reward |
| always | adaptive | historical motor normalization | always |

`--run-index` selects the session within the C1/C2 cycle, starting at zero.
One invocation runs **one session**, with 200 episodes and a 900-second acquisition
cap. The old launcher repeated hourly indefinitely; use an external scheduler for
repeated sessions. Adaptive trainer state is initialized for each invocation, so
separate invocations do not reproduce the old process's across-session trainer state.
Within-session timings remain read/train/wait = 200/400/3000 ms, phase width 200 µs,
10 Hz training patterns with 10 ms separation. The phase currently uses the motor
means from the historical four-value normalization array. Resolved routing is saved
beside the recording as `<recording_name>_example.json`.

## 2. Rapid pairing

```sh
python -m braindance.examples.2_rapid_pairing --config /path/to/routing.cfg --project_id paper --chip_id CHIP --record_duration 300
```

This preserves `closed_loop.py`, including random pair selection among at most
20 units with STTC strictly between .3 and .6. It records post-stimulation activity
but does not run a post-stimulation analysis. `--resume` uses existing V3 checkpoint
semantics; an interrupted phase restarts from its beginning. See
[checkpoint recovery](../../docs/checkpoint_recovery.md).

**Remaining phase work:** the controller currently triggers on pair element 1 and
stimulates element 0, despite its opposite printed arrow. Its delay/refractory
parameters are unused. Copying the entrypoint does not fix those issues or establish
measured electrical delivery latency. See the [release assessment](../../docs/paper-release-assessment-2026-09-14.md).

## 3. BusyBee

Prepare a working JSON such as:

```json
{
  "name": "busybee",
  "config": "/path/to/routing.cfg",
  "save_dir": "/path/to/new/output",
  "stim_electrodes": [1234, 5678]
}
```

Replace the example IDs with actual routed electrodes. If `save_dir` is omitted,
outputs use `get_output_dir() / "busybee"`.

```sh
python -m braindance.examples.3_busybee --dry-run
python -m braindance.examples.3_busybee --json busybee.json
python -m braindance.examples.3_busybee --json busybee.json --resume
```

Original defaults are preserved: 40 cycles of .5/1/2/4/8 Hz, with respectively
50/100/200/400/800 replicates per electrode. Each sweep follows 600 seconds of
spontaneous recording, uses amplitude 400 mV, phase width 200 µs, and
`single_connect=True`. V3's `order='ran'` preserves the original `rna` ordering
because there is only one amplitude. The `Experiment` runs 400 alternating
`RecordPhaseV3` and `NeuralSweepPhaseV3` phases and manages environment cleanup.
`--resume` continues from the last successful phase; `--project_id` and
`--chip_id` set experiment metadata. Numbered phase recording directories and
checkpoints are stored inside `save_dir`.

V3 currently limits each phase environment to one hour, replacing the original
three-day environment cap. A default sweep takes about 100 seconds per electrode,
so large electrode sets can exceed that limit. `--cycles` and `--record-seconds` allow shorter
runs; those runs should be labeled as modified protocols. `--dry-run` prints the
schedule without constructing an acquisition environment.

## Provenance and validation

Source snapshot: `5f733a8821d63d502042acc66e59abd303e6dae3`.

| Example | Original source |
| --- | --- |
| 1 preparation | `proj/cartpole_v1/neural_config_analysis.py`, `causal_analysis.py` |
| 1 ranking | `proj/cartpole_v2/ranked_pairs.py` |
| 1 game | `proj/cartpole_v1/cartpole_force_train.py` (`core.phases2`) |
| 2 | `braindance/examples/closed_loop.py` |
| 3 | `proj/busy_bee/continuous.py` (`core.phases`) |

Original research scripts remain in place. BusyBee source is owner-confirmed;
CartPole is reconstructed from the remembered GUI actions and rapid pairing from
the existing example. Exact historical run/configuration matches remain unconfirmed.
The old CartPole preparation analysis remains deprecated; it has not been replaced
with V3 placeholder analyses. None of these copies claims full phase consolidation.

Run `python -m pytest tests/test_paper_examples.py -q`. The checks compare the ranking
metric with the original source and verify phase construction, electrode mapping,
timing arguments, trainer modes and cleanup using hardware-free fixtures. They do
not establish live hardware equivalence or reproduce the paper's biological results.

Validation on 2026-09-14: **54 passed** across `test_paper_examples.py`,
`test_legacy_analysis_cleanup.py`, and `test_experiment_checkpoints.py`. A wheel
built and installed into a temporary target passed all three command-help checks,
support-module imports, and BusyBee schedule construction from `/tmp` under
`python -I`, without importing `proj.cartpole_v2.experiment`. That check reused
the `brain` environment's dependencies; it was not a fresh dependency-resolution
or live acquisition test.
