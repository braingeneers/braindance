# BrainDance streaming workshop

This example runs BrainDance's `MaxwellEnv` replay path and V3 phase framework.
It is not the Redis `proj/stream_demo` project. The new neural source implements
the same batch contract as `H5ReplaySource`; `dummy_maxlab` only substitutes for
hardware command construction. Local commands never send hardware sequences,
including on machines where real `maxlab` is installed.

## Start locally

Follow the [root installation guide](../../../README.md) to install from source
in a Python 3.11 conda environment. The default
package includes everything needed for the CPU CartPole workshop. Then, from any
directory:

```powershell
conda activate brain
python -m braindance.examples.streaming_workshop.main
```

The local watcher opens at `http://127.0.0.1:8765`. It uses no CDN or external
accounts. Stop the server with Ctrl+C. Use `--port 8766` if the port is occupied.
The default source is generated locally: no H5 download or lab recording needed.

Fresh CPU and GPU conda installations were tested against built distributions
outside the checkout. See the root guide for exact commands and verification
limits. The adjacent requirements file remains a source-checkout convenience;
installed users do not need to install it separately.

On Windows, `-X utf8` keeps redirected phase logs compatible with Unicode.

## Teaching flow

Start in **Experiment**: name the experiment, select simulated, recorded, or live
data, and configure the phase sequence and parameters. Simulated data exposes a
**Simulator** tab within Experiment for editing and testing the culture.
Select **Replay** to drop or choose an H5/NPY recording; choose another file to
replace it for the next run. Select **Live** to open the Maxwell `.cfg` file picker.
Simulator and replay do not require a Maxwell configuration. Existing server paths
can also be entered directly. Picked files are copied into the workshop output's
`uploads/` directory so saved profiles can reuse them.
**Task playground** lets you try task controls; **Functions** contains Python and
saved profiles. Choose **Run experiment** after the phase settings to launch and
open **Live recording** automatically. The default simulator has 40 neurons over 400 electrodes
in a 20 × 20 grid. Threshold detection and raw H5 saving are enabled by default;
use `--no-write-output` to disable raw saving.
The menu button hides the sidebar. The top-bar theme switch remembers light/dark
mode locally. Run controls are available in Experiment and Live recording. Phase progress uses the
run's phase list; skipping calibration shows only the environment phase.

### Sort uploaded recordings

Open **Spike sorting** for a click-only workflow: upload one or more raw Maxwell
`.h5`/`.hdf5` or `.nwb` recordings, pick the baseline from the uploaded files,
check the targets, choose a sorter, and click **Sort selected recordings**.
Uncheck the baseline if you do not want it included in the sorted outputs.
Uploads stream to disk, with a 64 GiB limit per file.

RT-Sort discovers units on the baseline and applies them to all selected targets;
electrode routing and sampling rate must match. Its options include CPU/CUDA and
the baseline time window. The menu also offers installed SpikeInterface sorters,
which use their default parameters and fit each recording independently: they do
not use the baseline or preserve unit identities across files.

Under **Install sorter dependencies**, choose **RT-Sort dependencies** or
**SpikeInterface Python sorters** and click **Install dependencies**. This runs
pip in the workshop's current Python environment, with a fixed package list and
visible logs. It requires internet and environment write access. Installation is
blocked during acquisition, analysis or sorting; restart the workshop afterward
before running jobs (also after a failed install, since pip may have changed some
packages). No packages are installed automatically when opening the tab.
Both installation plans require PyTorch >=2.6. Intel macOS Python is unsupported
(including an Intel Python running through Rosetta on Apple Silicon); the button
rejects it before starting pip, without requiring a restart. On Apple Silicon,
create a separate Python 3.11 environment with native ARM64 Miniforge or Miniconda,
install BrainDance there following the root guide, and launch the workshop from
that environment. Check `python -c "import platform; print(platform.machine())"`
prints `arm64` before retrying. The existing Intel environment can stay intact.
On an Intel Mac, use a supported Linux or Windows machine for these sorters.
The SpikeInterface option installs dependencies for Simple, SpyKING CIRCUS 2 and
Tridesclous 2, plus NWB support. MATLAB, GPU drivers and external sorter executables
still need manual installation.

Progress and errors appear under Results. Download each portable spike NPZ and
the batch manifest, or click **Open in Analysis**. Raw uploads stay in `uploads/`
and results in `sorting/` under the workshop output directory. The tab's list is
retained across page reloads for the current server session; files and the saved
`batch.json` remain on disk after shutdown.

### Try environments without acquisition

In **Task playground**, choose CartPole, FoodLand, or Ant and load it.
CartPole is included in the base package. Install all three from the repository
with `python -m pip install ".[foodland,ant]"`; `rl` is only for training and is
not required. On Apple Silicon Macs, follow the root README's native `brain-mac`
setup and activate that environment instead of `brain`.

Press-and-hold buttons feed the real environment actions and light up for both
mouse and keyboard input. Focus the game canvas to use
the displayed keyboard controls: arrows/A/D for CartPole, left/right/up or A/D/W for
FoodLand, and keys 1–8 to select an Ant joint with left/right to apply torque. Ant also
has buttons to select all eight named joints. A direction press starts play;
releasing it returns that input to zero. FoodLand manual turning is limited to
about 115° per simulated second. Pause, single-step, reset,
and select any observation to plot its recent history. The displayed action
values show what was actually applied. The playground uses a separate game
instance and never opens a recording or acquisition connection.

### Build a culture and record it

In **Neural simulator**, set the neuron count independently of the electrode
count. Select and drag neurons, enter coordinates, or choose Place mode and click
to add one. The shaded electrodes preview the selected neuron's Gaussian
waveform footprint. Adjust electrode spacing, spatial spread, and waveform
amplitude. Connect a source neuron to a target with a signed weight; the drawing
shows connections touching the selected neuron. In Connect mode, keep one source
selected and click multiple targets; Shift-click removes a connection. Target
buttons also support selecting a group and setting or removing its connections.
The matrix supports bulk edits.
Neuron positions and geometry apply to the next run; connectivity can change
at a bin boundary during a streaming run. Save a profile to retain the setup.

Choose a recording duration and detector, then **Start recording**. This runs
only the recording phase and opens **Live recording**, where the electrode map
shows negative peak or RMS voltage from every channel in the latest 20 ms bin.
It uses acquired raw voltage, not the footprint preview or simulator firing labels.
The raw traces and spike plots remain available below it. With **Save raw recording**
checked, a Maxwell-compatible `.raw.h5` is written in the run's output directory.
The simulator settings also flow through native V3 phases, profiles, and exported
Python experiments.

### Build an experiment

In **Experiment**, set a name and notes, then configure the shared source.
Game selection, encoder stimulation electrodes, and decoder channel pools belong
to each binned closed-loop phase. Add phases from the catalog, give each a unique ID, and use
Move up/down or Remove to edit their order. Repeated phases are supported. Duration,
repeat count, and mapping gains belong to the individual phase. Supported phase
types focus on recording, stimulation, stimulation sweep, RT-Sort and the three
environments. The workshop's binned recording, response probe and mapped
environment phases also use the V3 contract.
Use **New sequence** to start a native sequence; **Restore previous sequence**
restores the prior unsaved sequence. Native and binned V3 phases currently
use separate execution modes and cannot be mixed in one sequence.

Streaming probes and closed-loop phases require exactly two stimulation electrodes
because the editable encoder returns two rates. Recording requires none. Different
phases can use different electrode IDs: acquisition routes their union once, and
phase-local output indices are translated into that physical map. This avoids
the core PhaseGroup's incomplete per-child electrode reconfiguration. Native game
phases retain their own mapping contracts: native CartPole uses 2 sensory, at least
4 motor and 2 training sequences; Foodland 2/2/2; Ant defaults to 8/8/2. Motor
sequences are sorted units, not raw electrode IDs. Native sensory counts can depend
on constructor parameters such as Ant's `n_features`.

### Native V3 phases

The catalog scans V3 source definitions without importing optional scientific
packages. Its default native library contains seven implementations: Record,
Frequency Stim, Neural Sweep, RT-Sort, CartPole, FoodLand and Ant. Legacy research
classes remain resolvable for existing specifications but are not offered by
the default library. Native cards expose constructor JSON, per-phase experiment settings,
parameter references, inputs and outputs, mapping counts, and implementation caveats.
Nonliteral Python defaults remain in the native constructor when omitted. Python
callables/custom objects still require a Python experiment, rather than JSON.
Experiment-wide native parameters and initial JSON data are available in the
**Native V3 experiment inputs** section.

Native **Verify experiment** checks schema, imports, constructors and V3 dependencies
in a separate process. It does not run the scientific analysis or prove shapes of
future outputs. Constructor preflight stops after 45 seconds if it cannot finish.
Native Launch uses the real `Experiment.run` after per-phase V3 validation with
scoped inputs, preserving native requirements, files, and checkpoint behavior.
Local acquisition still uses MaxwellEnv replay,
never hardware. Each native environment owns a fresh simulator/source and recording;
it does not share the streaming adapter's continuous source clock. Native phases
retain their own controllers. Runner-side observers publish CartPole, FoodLand and
Ant scenes to Live, alongside progress and the retained console log; visualization
does not require a streaming or web implementation inside the native phase.
Other repository research phases can have incomplete scientific implementations;
structural validation is not scientific validation.

### One phase contract, separate execution adapters

New phases declare `inputs` and `outputs` and return a result dictionary from
`run(experiment)`. The old `requires` and `provides` spellings remain compatibility
aliases. Use operation/data names rather than workshop or experiment names:

| Binned phase | Inputs | Outputs |
|---|---|---|
| Recording | None | `recording_baseline_hz` |
| Response probe | `recording_baseline_hz` | `response_probe_hz`, `response_probe_trials` |
| Mapped environment | `recording_baseline_hz` | `environment_episodes`, `environment_reward` |

These phases receive a `BinnedRuntime` implementation through
`experiment.phase_runtime`. The implementation handles acquisition, callbacks,
interactive controls and observation of scientific events. The phase module has
no dependency on the workshop web app or streaming transport.

CartPole, FoodLand and Ant use the same mapped environment phase with different
runner-provided game adapters. Their editable binned controllers remain distinct
from native learning controllers and retain their own timing and data schemas.
The native implementations are `phases3_loop.CartPolePhase`,
`phases3_foodland.FoodLandPhaseV3` and `phases3_ant.AntPhaseV3`; do not use the
unimplemented CartPole scaffold in `phases3.py`. See the
[phase authoring guide](../../../docs/phases.md) for the supported API.

### Restart a streaming phase

**Restart phase** in Live starts a new attempt at the next bin boundary. It restores
phase-entry baseline/results, clears mapping state, resets the game when applicable,
and reruns the phase. It does not rewind acquisition or undo delivered stimulation.
Prior raw data (when enabled), stimulation logs, and history remain retained.
`attempts/<phase-id>/<attempt>/` stores code and `attempt.json`, including parameters,
baseline, start/end frames, status, and counters. Aborted outputs are not committed
as a completed phase result. Native loops have no shared cooperative restart API;
Stop ends their process and a new launch creates a separate output directory.

Each phase lists inputs with their origin (settings or an earlier phase)
and its outputs. The builder checks dependencies and parameter ranges
as you edit, using the same `PhaseValidator` as V3 execution. A later recording
cannot satisfy an earlier phase. To omit recording, enter one baseline rate per
channel, click **Use zero baseline** deliberately, or load a compatible calibration.
Blank baseline is not treated as supplied data in a built experiment. Response
probes are optional because the current environment phase does not consume their
outputs. **Skip to environment** runs only the configured environment phases and
still requires an explicit baseline or calibration.

For streaming phases, **Verify experiment** also initializes the selected local source/game, calls the
current editor functions on sample inputs, checks action/rate ranges and shapes,
and steps the local game. It executes trusted participant Python but creates no
Maxwell environment, delivers no stimulation, and writes no run output. Changes
invalidate the result. Starting repeats validation; runtime checks still apply.
Preflight cannot prove behavior for every future input or validate live hardware.

**Save profile** retains experiment name, notes, ordered phases, their parameters,
and shared settings in the profile's `SETTINGS`. Runs save `experiment.json` with
the resolved phase plan and settings. Scientific phase implementations live in
`braindance/core/phases_v3/`, including `phases_binned.py` for the binned workflow.
Add scientific behavior there, then register its settings/constructor in the
workshop catalog. The runner supplies acquisition and mapping services and owns
pause/restart, reload, persistence and display updates. Dependency declarations
are shared between validation and execution.

1. **Run experiment:** the default plan's 3 s recording estimates baseline firing rates. Randomized
   causal trials stimulate each of two selected inputs and compare post-minus-pre
   rates against interleaved sham trials. Then the game runs for 120 s.
2. **Start environment:** skip the first two phases. Choose disjoint decoder pools,
   two stimulation electrode IDs, sensory index, and optional baseline rates.
   Use **Use zero baseline** when intentionally skipping baseline estimation.
3. Open **Python functions and saved profiles**. Edit the template, give it a name,
   and **Save profile**. The displayed path is an ordinary Python file you can also
   edit in your editor. During a run, **Reload functions** reads that saved file,
   validates outputs, clears function state and pending rate credit, and pauses.
   Click **Resume** to continue. Saving alone does not change running code.
4. Use **Step 20 ms** while paused to see one acquisition/action/stimulation cycle.
   Dropdowns select displayed observation, stimulation output, spike channel/unit,
   and action component. The setup sensory index selects what the default encoder
   actually uses; changing a plot dropdown alone does not change the experiment.
5. Open **Simulator connections**, choose a weight, then click or drag across cells.
   Hold Shift to clear cells. Rows are targets, columns are sources:
   `matrix[target, source]`. Positive weights excite; negative weights inhibit.
   The default is a directed ring with weight 0.35, not a fully connected network.
   The diagonal starts at zero; enable self-connections to edit it. Changes apply
   at the next bin during a run, or at startup when stopped. Save a profile to keep
   the matrix. Changing connections does not rerun baseline or response probes.
6. Stop, adjust setup, and start again to reset the experiment and simulator.

The plots show exact numeric values plus independently scaled paired histories.
The raw panel shows the first 16 acquisition channels with vertical offsets and
min/max downsampling; it is not one voltage sample per 20 ms bin. Activity marks
represent spike counts in each 20 ms bin, not individual event timestamps.
Requested stimulation rates and actually dispatched pulse input indices are
shown separately. Default pulse-rate ceiling is 40 Hz; pulse timing is quantized
to the 20 ms control grid.

## Edit the functions

The Python editor highlights keywords, strings, comments, and numbers. Tab inserts
four spaces, Enter retains indentation, and Escape moves focus out of the editor.
**Verify Python** checks the current editor text, including unsaved changes, and
reports syntax errors with line/column positions. It also checks that `encode`
and `decode` are synchronous, undecorated functions accepting four positional
arguments. Verification does not execute code, import dependencies, validate
physical behavior, or save/reload the profile. Startup/reload separately checks
returned shapes and ranges; runtime checks remain active during the run.

Profiles live in the output directory's `profiles/` folder; the UI shows the full
file path. Each `.py` contains `encode`, `decode`, and a literal `SETTINGS`
dictionary. **Save profile** takes settings from the setup form. Previous versions
go to `profiles/revisions/<name>/`. **Load profile** restores code and setup while
stopped. **New template** starts another named experiment from the default code.

```powershell
python -m braindance.examples.streaming_workshop.main --profiles-dir ./workshop_profiles --profile my_experiment
```

Save `my_experiment` in that directory first. Alternatively, keep using
`--functions-file path/to/functions.py`. Reload changes functions only; environment,
channel selection, gains, and durations take effect on the next run.

```python
def decode(spike_counts, dt_s, params, state):
    # Return one value per named action component plus JSON-compatible diagnostics.
    return action, diagnostics

def encode(observation, dt_s, params, state):
    # Return two stimulation rates in Hz plus JSON-compatible diagnostics.
    return rates, diagnostics
```

Counts are detected spikes in the current bin. The default decoder smooths rates
over 100 ms and subtracts the recording baseline. CartPole/Foodland compare two
pools; Ant maps individual channel rates to eight separate joint torques. Actions
are in [-1, 1], except Foodland forward speed is [0, 1]. State dictionaries reset
on reload and episode reset. Functions must be short and nonblocking; Python code
is trusted local participant code, not a sandbox for arbitrary uploads.

Syntax errors preserve the last validated module. Runtime errors and invalid
outputs pause the interactive run; fix, reload, then resume. Headless runs fail
instead of waiting for an editor. Initial invalid code prevents startup.

## Environment switches

`main.py` includes adjacent commented choices, or use CLI flags:

```powershell
python -m braindance.examples.streaming_workshop.main --environment cartpole
python -m pip install gym==0.26.2 "pygame>=2.5"
python -m braindance.examples.streaming_workshop.main --environment foodland
python -m pip install "gymnasium[mujoco]>=1.1,<2"
python -m braindance.examples.streaming_workshop.main --environment ant
```

Quote requirement strings containing comparison operators in your shell.

- **CartPole:** existing `braindance.games.cartpole_continuous` physics, scalar
  force action, four named observations. Its existing track-wrap behavior remains.
- **Foodland:** existing `braindance.games.food_land.FoodLandEnv`. Workshop public
  action order is `[turn, speed]`; adapter translates the legacy internal order.
  The actual nine observations are labeled in the UI. Its movement is per step;
  at 20 ms bins the teaching game runs at 50 steps/s.
- **Ant:** Gymnasium's Ant-v5, already used by BrainDance's Ant wrapper. This
  example uses raw observations and eight torque actions, with `frame_skip=2`
  and an explicit 20 ms physics check. Canvas draws MuJoCo capsule geometry from
  an oblique view, so no OpenGL renderer is needed. Numeric observation labels
  preserve the installed model's actual observation dimension.

Adapter startup/step/reset was tested for all three environments in `brain`.
Ant uses raw observations rather than the existing wrapper's random projected
features, so choose an appropriate sensory index/gain before experimentation.
The default decoder is an editable mapping, not a trained walking controller;
Ant can still fall. Torso height outside 0.3–1.0 m ends its episode. All games
also reset at the configured episode limit (default 10 s), independently of total
environment time. The watcher shows episode count, elapsed time, and reset reason.

## What response probes measure

Each trial has 100 ms before stimulation, 100 ms for the response, and 100 ms
recovery. Two inputs each get the configured number of stimulated trials and sham
trials, in randomized order. A sham has the same timing but sends no pulse.
With four repeats, this is 16 trials × 300 ms = **4.8 seconds** of experiment time.
Recording adds 3 seconds by default. Wall time can be longer if processing overruns.

The response matrix reports `(post spikes − pre spikes) / 0.1 seconds`, averaged
over stimulated trials, minus the corresponding sham average. Rows are the two
stimulation inputs; columns are detected channels/units. It estimates evoked
responses, not simulator weights. Skip goes directly to the environment using
your chosen channels and supplied baseline. The builder requires an explicit
baseline or calibration; the legacy CLI `--skip` without a phase plan retains its
zero-baseline preset.

## Raw signals, thresholds, RT-sort, and replay

```powershell
# Synthetic raw traces with actual threshold detection:
python -m braindance.examples.streaming_workshop.main --detection threshold --threshold-uv -30

# Existing Maxwell H5; stimulation is logged but cannot change recorded spikes:
python -m braindance.examples.streaming_workshop.main --source "path/to/recording.raw.h5" --loop

# Compatible prebuilt RT-sort object and configured detection model:
python -m braindance.examples.streaming_workshop.main --source "path/to/recording.raw.h5" --detection rt-sort --sorter-path "path/to/sorter.pkl"
```

The SNN is implemented in [`braindance/core/simulation.py`](../../core/simulation.py),
using NumPy directly; it does not depend on Brian2, NEST, or another SNN package.
`NeuralSimulationSource` uses seeded leaky integrate-and-fire neurons with 1 ms
internal updates, background drive, refractoriness, recurrent connections, and
stimulation input. Neurons have independent physical x/y positions in µm.
A spike injects a biphasic waveform template into every electrode with amplitude
`A * exp(-distance² / (2 * sigma²))`. Overlapping waveforms sum before noise, drift,
stimulation artifacts, and ADC quantization are applied. Stimulation drives neurons
according to their distance from the selected electrode. Connectivity is
`adjacency[target_neuron, source_neuron]`, independent of electrode count.
The explicit 20 × 20 grid has 17.5 µm pitch; physical Maxwell-style electrode IDs
are `row * 220 + column`. Raw channels remain contiguous 0–399. Use physical
electrode IDs for stimulation and channel IDs for threshold-based decoder pools.
This is a teaching model, not a biological fit. Raw ADC resolution is 1 µV/count;
the source exposes the existing Maxwell amplified-mV stream convention. The
watcher converts back to input-referred µV using source gain. Replay uses H5
metadata; verify calibration for any recordings with nonstandard scaling.

Threshold detection uses negative threshold crossings and 2 ms refractoriness
carried across bin boundaries. Artifacts are intentionally visible; this is a
basic detector, not production artifact rejection. `events` means simulator
firings assigned to each neuron's nearest channel (multiple neurons can share
one channel), or stored H5 detector events. These labels are not threshold crossings
and are not sorted units. RT-sort counts are sorted unit IDs,
which are distinct from raw channels and physical stimulation electrodes.

RT-sort is optional and requires its existing model/dependencies plus a trusted,
compatible saved sorter. The adapter does not train a sorter. It respects the
sorter's internal buffer size (normally 100 frames / 5 ms), accumulating output
into 20 ms workshop bins. It does not convert already-scaled raw data a second
time. Expect an initial warm-up period. Its saved device and trained channel
order must match the runtime; actual model inference was not tested here. The
subchunk/frame/count contract is covered with a fake sorter.

H5 replay is explicitly labeled **open-loop**. Do not interpret its causal matrix
as a response to newly delivered stimulation. EOF stops a run; a partial final
20 ms bin reports an explanatory error. `--loop` restarts file data with monotonic
virtual frame time. Choose mapped electrode IDs rather than assuming 0 and 1
exist in a real recording. Use `load_catalog()`/`Recording` to choose project
recordings when available, then pass the selected H5 path to this existing replay
interface. The source path is never inferred from the output directory.

## Forms and Python

Open **Functions** (or **Experiment → View / export Python**) to see participant
functions, environment setup, and the ordered experiment. Generated previews
follow the builder and use Python syntax highlighting. Previewing writes no files.

Choose **Python style** to switch between **Phase objects (.add_phase)** (the default)
and **Config (PHASES)**. Object exports expose an editable `build_experiment()`:

```python
exp = PythonExperiment(settings)
exp.add_phase(RecordingPhase(duration=3., name='baseline'))
exp.add_phase(CartPolePhase(duration=120., name='game'),
              settings={'stim_electrodes': [0, 1]})
```

These are real `PhaseV3` objects; the same instances are validated and run.
Edit constructors, change object attributes, reorder calls, or use Python loops
and conditions. Give phases unique names. Scientific exports import their concrete
V3 classes and use `PythonExperiment(settings, native=True)`. Per-phase `settings`
hold streaming mappings or scoped scientific inputs. Streaming acquisition stays
with the workshop runtime; exported native objects execute in the script's Python
process with the same source setup as browser native runs. Config style retains
the `PHASES` list and existing worker-based native runner. Both styles support
custom analysis and export expanded loop iterations in execution order.

Click **Convert setup to code** to create a new folder under the output directory's
`exports/` directory. The bundle contains:

- `participant_functions.py`: the exact editor code, with encode/decode, optional
  train, and any helpers. A saved profile's SETTINGS is retained for reference;
  the exported runner takes settings from the other two files.
- `environment_setup.py`: source, acquisition, simulation, and shared settings,
  with commented replay/live switches.
- `experiment.py`: phase order, phase-specific parameters, and an executable entry point.
- `builder_snapshot.json` and `README.md`: original setup and running instructions.

Activate `brain`, then run `python PATH/experiment.py --verify`, followed by
`python PATH/experiment.py --output-dir ./results`. No browser is required.
Export checks syntax and builder constraints; `--verify` performs runtime preflight.
Each export creates a fresh folder. Hand edits are never overwritten, and arbitrary
Python is not converted back into forms. Keep the saved UI profile to continue
editing through the builder.

Use encode/decode/train buttons to jump within participant code. **Add train hook**
adds a template to older profiles. Streaming environment steps call decode, step
the game, call train, then encode. Train receives observations before/after the
action, action, spike counts, reward and done; return a JSON diagnostics dictionary.
It can update `state['decode']` and `state['encode']` for your functions to read.
Hook state resets on episode end, phase restart and code reload. Nothing learns
automatically: the default train hook does nothing.

Streaming adapters are workshop PhaseV3 implementations sharing a MaxwellEnv
source and 20 ms loop. Native V3 phases retain their original controllers,
trainers, files and timing; participant hooks are not automatically injected into
them. Both use the V3 framework, with separate execution modes.

The top bar shows the current server session's source, wall-clock pacing and run
status. Simulation/replay at 1× targets elapsed real time; unpaced mode runs as
fast as possible. Neither provides a hard real-time guarantee. Native phases
control their own step sizes. Unsaved builder changes do not change these badges.

## Analysis-selected stimulation electrodes

Custom analysis works in either runner. Its decorator declares the data keys
available to later phases; returning an undeclared key does not advertise that
key to the builder. Output names must match the top-level returned dictionary:

```python
from braindance.examples.streaming_workshop.custom_analysis import analysis_phase

@analysis_phase(inputs=[], outputs=["stim_electrodes"])
def select_stim_electrodes(exp):
    # Replace with selection from earlier data, declaring those keys in inputs.
    return {"stim_electrodes": [101, 205, 309]}
```

These are physical electrode IDs, which must match your acquisition mapping.
For this contract, use a native sequence: recording / analysis → electrode
selection → Neural Sweep or Frequency Stim. Neural Sweep with `neuron_list=None`
sweeps all selected electrodes. Explicit neuron lists and frequency stimulation
commands index the selected electrode list. Leave downstream per-phase
`settings.stim_electrodes` unset to use the earlier analysis output.

Binned probes and games currently use exactly two electrodes configured in phase
settings and route their union at startup; they do not consume `stim_electrodes`
from an analysis output. Native CartPole selects its sensory and motor units from
RT-sort data internally. Mixing native and binned acquisition phases in one
sequence is not supported.

Custom phase cards display the function name. Renaming one decorated function in
the editor updates its phase reference when the rename is unambiguous, preserving
the phase ID and loop membership. For an old profile or multiple simultaneous
renames, open the phase and set **Function name** to the existing decorated
function you intend to run, then save the profile.

## Live Maxwell

The commented live configuration in `main.py` selects the same phase/controller
logic. Provide a real acquisition config, routed-channel count, decoder pools,
stimulation electrodes, and threshold or RT-sort detection. For example:

```powershell
python -m braindance.examples.streaming_workshop.main --live-config "path/to/config.cfg" --channels 942 --stim-electrodes 123 456 --left-channels 10 11 --right-channels 20 21 --detection threshold
```

Replace all example channel/electrode values with your configured mappings.
`maxlab` and running Maxwell acquisition are required. This path sends real
stimulation; the local replay path does not. Review pulse amplitude and width
in `main.py` for your setup before using it. No live hardware test was performed.
Live display conversion assumes gain 512, consistent with the current environment
initialization; update and validate that value if the acquisition gain changes.
Pausing stops new commands, not hardware acquisition; live resumption may expose
queued packets. Validate live buffering, stop behavior, and timing before class.

## Output and reproducibility

Default output is `get_output_dir() / "streaming_workshop"`, with unique run
subdirectories. This script never changes global data or output configuration.
`--output-dir PATH` gives an explicit per-run output override.

Each run saves V3 results, setup JSON, initial/reloaded function copies, adjacency
changes, a trusted local `calibration.pkl`, and `performance.json`. The latter
reports processing time excluding intentional pacing; its window is the most
recent 10,000 bins. Raw saving is enabled by default and saves actual consumed
frames and source events using the existing Maxwell-compatible writer; use
`--no-write-output` to disable it. The in-memory UI history
is bounded to 250 bins.

Load a compatible calibration with `--calibration path/to/calibration.pkl` and
`--skip`. Source identity, mapping, sample rate, detector, and simulator seed/
initial adjacency, neuron geometry, and waveform settings are checked. Load only
your own trusted pickle files.

## Headless tests and timing

```powershell
python -m braindance.examples.streaming_workshop.main --headless --record-seconds .1 --causal-repeats 1 --environment-seconds 1 --speed max
python -m braindance.examples.streaming_workshop.main --headless --skip --environment ant --environment-seconds 10 --speed max
python -m pytest tests/test_simulation.py tests/test_streaming_workshop.py tests/test_replay.py tests/test_replay_environment.py -q
```

Choose **As fast as possible** under Acquisition pace (or use `--speed max`)
to remove deliberate sleeps. It changes wall pacing, not experiment
time. Streaming acquisition runs synchronously in a dedicated Python worker
thread, separate from HTTP requests and browser drawing; native V3 runs use a
subprocess. The simulator generates full raw voltage at 20 kHz per channel,
with neuron dynamics updated at 1 kHz and control in 20 ms bins.
The browser requests updates at up to 30 Hz and shows measured display updates/s.
Run details report achieved acquisition rate, processing capacity, real-time
factor, p95 processing time and late bins; these are separate from display FPS.
Processing is Python best-effort, not hard real time; observe timing and
performance output on your laptop.
Pause durations are not scientific processing benchmarks.

Test coverage includes continuous phase clocks, randomized causal trial counts,
exact function-to-action and pulse routing, raw/event consistency, split-read
determinism, adjacency effects, local hardware isolation, generated-H5 round trip,
threshold activity, RT-sort chunk routing, invalid setup, and reload recovery.
External H5 and live hardware validation remain separate from generated tests.

Latest verification in this workstation's `brain` environment: **50 passed,
1 skipped** (external H5 not configured), with the existing legacy Gym precision
warning. Tested dependency versions: NumPy 2.3.5, h5py 3.13.0, pyzmq 26.4.0,
psutil 7.0.0, Gymnasium 1.1.1, Gym 0.26.2, pygame 2.6.1, MuJoCo 3.3.2.
A generated 790-bin CartPole run measured 2.0 ms median, 3.8 ms p95, and 32.7 ms
maximum processing time; browser polling in a separate run reached roughly
10 ms p95. Occasional overruns remain possible.

### Visual phase builder

Open **Experiment → Phase builder** to assemble the shared experiment sequence.
The green **▶** button runs the sequence through the same verification and launch
path as **Run experiment**. It is disabled for empty sequences and during a run.
For a hardware-free Maxwell check, choose **Dummy Maxwell (sine)** in Experiment
data. This uses Maxwell’s deterministic 1024-channel dummy/replay backend and
simulated stimulation commands, rather than the neural network simulator.
Drag a phase from the searchable library onto a **+** connection, or click the
connection and choose a phase. Drag existing nodes onto another node or onto the
board to reorder them; the highlighted edge previews the new position. Green rounded
connectors identify experiment phases; violet angular connectors identify analysis.
The data contracts determine valid placement, including recording-before-analysis
requirements.
Unavailable phases are gray; hover or expand **Why unavailable?** to see their
prerequisite lineage. The context panel shows configured input values and projected
output keys at the selected gap. Scientific output values are not calculated by
this planning view. Click a node or **Parameters** to edit its settings below the
board. Pinch on a trackpad or touchscreen to zoom around the gesture position.
Pan the board by dragging its empty area, use the zoom controls, or choose
**Expand board** for more room. Arrow buttons offer a keyboard-accessible alternative
to dragging; **Undo** and **Redo** restore changes made on the board.

Choose **Select loop section**, click the first and last phases, then choose
**Create loop**. The whole range is highlighted and included. You can also use the
visible **Include in loop** controls on individual nodes. Drag its header to move the whole loop, adjust its repetition count, or
**Unwrap** it to edit the grouping. Individual nodes can also move into or out of a
loop. Loops
are non-nested and expand to at most 32 executed phases, with distinct IDs for each
iteration. Profiles preserve the editable grouping; verification, export and
execution receive the same expanded sequence. Dependency-breaking additions,
removals and reorders are rejected. **Verify experiment** still checks parameters
and runtime prerequisites beyond the catalog's declared data dependencies.

#### Real-protocol smoke example

`real_experiment_busy_bee.json` translates the BusyBee protocol from the companion
`braindance_proj/braindance_figs/busy_bee_fig/protocol_example.py` into workshop
phases. It includes all five frequency conditions (0.5, 1, 2, 4 and 8 Hz), each
with a recording followed by stimulation. For a quick hardware-free execution
check, it uses one cycle, 0.04-second recordings, one simulated neuron and one
replicate per condition. It retains the source's `phase_length=200` setting. This
is a shortened simulation check, not the full 40-cycle scientific experiment.

`tests/test_real_experiment_workshop_fixture.py` exports the example and runs the
exported Python experiment to completion. The browser regression
`test_real_busy_bee_builds_and_runs_from_board` loads all ten phases into the
board, verifies the resulting sequence, and launches it through **Run experiment**.

### Catalog and experiment selection

Open **Catalog** in the sidebar to search the configured recording catalog and
saved workshop runs. **Refresh catalog** rereads the CSV configured in Settings
and discovers new experiment folders in the workshop output directory.
**Add experiment** saves a local directory, experiment JSON, HDF5 or NPZ path in
`catalog_experiments.json` in that output directory; it does not rewrite the
configured CSV or regenerate its S3 inventory.

Use **Select for analysis** to load an entry in the Analysis workspace. Entries
without locally available recordings remain visible with an availability message.
The **Current experiment** card follows the workshop's run status, phase and output
directory, with shortcuts to live recording and analysis. The selected analysis
experiment stays independent of the running experiment.

### Manual analysis workspace

Open **Analysis** in the sidebar. The folder browser starts at the data directory
configured by `get_data_dir()`. Choose an experiment or recording from the dropdown
and click **Load**, or use **Open folder** / **Up one folder** to navigate.
**Data folder** returns to the configured location; **Browse folder** opens a typed
path without changing your global configuration. **Use current experiment** opens the current
run's output; alternatively enter an experiment directory, `experiment.json`,
a Maxwell HDF5 recording, or a portable sorted spike NPZ. Recorded phases and
attempts appear on the left. Select a phase and recording before choosing a tool.
Streaming phase selections use saved frame boundaries within their shared file.

- **Raw data** plots a channel over a chosen time window.
- **Spikes** plots channel detections or sorted units, identified in the result.
- **Connectivity (STTC)** and **Neuron latencies** use SpikeLab. STTC describes
  association; signed nearest-spike latency histograms do not establish causality.
- **Stimulus overlap** follows `braindance.analysis.causal_connectivity`: raw
  and cubic artifact-removed overlaps, wrapped spike rasters, peristimulus firing
  rates, and early/late response matrices grouped by stimulation pattern. Choose
  artifact blanking, response windows, and threshold settings. Sorted files use
  their supplied spikes; raw files use explicitly labeled channel detections.
  Results report the timing convention and do not claim causal significance.
- **Spike sorting** initializes RT-Sort on a selected baseline window, then sorts
  the target recording. Set the device, worker count, detection thresholds and
  sequence criteria. Baseline and target must have the same routed channels and
  sampling rate. Results are saved separately from inputs and become available
  to spike-based analyses.

Jobs run on a separate worker with stage progress and errors shown in the page.
Plot payloads are bounded for browser responsiveness; results report display
subsampling. Download result JSON to retain the displayed numeric values.
Unavailable tools explain missing data or optional dependencies. RT-Sort needs
the sorting dependencies and detection model; CUDA is checked when requested.
