# BrainDance
Making neural stimulation easier.

## Paper examples

The [numbered base examples](braindance/examples/README.md) cover **1. CartPole**
(including the historical pair-ranking workflow), **2. rapid pairing**, and
**3. BusyBee**. The guide includes configuration, provenance, and remaining
validation work.

## Installation

Use Python 3.11 (Python 3.11–3.12 supported). The default install includes SpikeLab analysis, plotting, the CPU simulator, and CartPole, FoodLand, and Ant. No GPU, Maxwell hardware, or recordings are needed.

```bash
conda create -n brain python=3.11 pip -y
conda activate brain
git clone https://github.com/braingeneers/BrainDance
cd BrainDance
python -m pip install .[foodland,ant]
```

For development, use `python -m pip install -e .` instead.
For the full CPU-friendly install, including the Qt experiment launcher:

```bash
python -m pip install -e '.[full]'
```

`all` is an alias for `full`. Neither installs RL training, GPU sorters, or vendor
hardware SDKs; those remain explicit extras. The default install is larger than
the former simulator-only install, and includes MuJoCo for Ant.
These packaging changes are not published yet. After release, `python -m pip install braindance` will replace the clone and local install steps.

### Platform notes

- **Windows:** CPU and NVIDIA GPU paths below. Use `python -X utf8` if redirected output has Unicode errors.
- **Linux:** Same commands; NVIDIA GPU support requires a compatible driver.
- **macOS:** See the native Apple Silicon setup below. RT-sort currently uses CUDA, so its GPU path is unavailable on macOS.

### Mac setup (Apple Silicon)

The default/full install selects Intel Mac wheels for Numba/llvmlite and MuJoCo
when Python runs under Rosetta. After pulling updates,
run `python -m pip install -e '.[full]' --only-binary=numba,llvmlite,mujoco`.
The binary flag prevents a lengthy source build if no matching wheel is available.
The dependency limits apply only to Intel macOS Python.

Use native ARM Python, including when your existing Conda installation runs through
Rosetta, for native execution. From the cloned repository, create a separate environment:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n brain-mac python=3.11 pip -y
conda activate brain-mac
conda config --env --set subdir osx-arm64
export LC_ALL=en_US.UTF-8
python -c "import platform; print(platform.machine())"  # must print arm64
python -m pip install .
```

This installs **analysis, CartPole, FoodLand, and Ant**. The `rl` extra is for training and
is not needed to play any of them. Stop any workshop started from the old `brain`
environment with Ctrl+C, then launch from the new environment:

```bash
conda activate brain-mac
python -m braindance.examples.streaming_workshop.main
```

If FoodLand reports missing pygame or Ant reports missing MuJoCo, check
`python -c "import sys; print(sys.executable)"`: it must point into `brain-mac`.
For development, use `python -m pip install -e .` (or `-e '.[full]'` for Qt).
The locale setting avoids a Conda `readline` crash seen with `C.UTF-8` on macOS.
To save it for future activations, run
`conda env config vars set LC_ALL=en_US.UTF-8` while this environment is active.

An `x86_64` Python on Apple Silicon uses Intel packages, which can fail to resolve
or try compiling dependencies. Creating the environment above leaves your old
`brain` environment intact. The new default with all games has not been verified
on an Intel Mac. Modern
[PyTorch no longer ships Intel Mac builds](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690),
so `rl` is not supported by the current dependency set on Intel Python.

### GPU / RT-sort

On Windows or Linux with an NVIDIA GPU, install CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) first, then the RT-sort extra. For example, the tested CUDA 12.4 setup:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install ".[rtsort]"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The extra includes diptest, NVML bindings, scikit-learn, and SpikeInterface. Pretrained detection models are bundled; sorting a recording still needs its own sequence templates. [Torch-TensorRT](https://pytorch.org/TensorRT/getting_started/installation.html) is optional; RT-sort works without it.

### Other features

Install extras from the checkout with `python -m pip install ".[extra]"`, or combine them: `".[analysis,gui]"`.
Analysis and all three workshop games are included by default; their named extras
remain supported for existing installation commands.

| Extra | Adds |
| --- | --- |
| `analysis` | SpikeLab analysis and plotting (also included by default) |
| `full` / `all` | Default analysis and games plus the Qt launcher; no GPU/training dependencies |
| `gui` | Legacy Qt experiment launcher (`braindance-gui` command) |
| `rl` | Stable Baselines3 and Atari support |
| `foodland` / `ant` | Workshop games (also included by default) |
| `kilosort` | Python dependencies for [Kilosort2](https://github.com/jamesjun/Kilosort2), which must be installed separately |
| `open-ephys` | `open-ephys-python-tools` for Open Ephys integration |
| `dev` | Build and test tools |

Live Maxwell acquisition requires the vendor's `maxlab` SDK and a configured workstation.

## Get the tutorial data

```bash
python -m braindance.examples.get_tutorial_data
python -m braindance.examples.get_tutorial_data --validate
```

The default download includes all three original H5 recordings and supporting
files (about 1.1 GB), excluding RT-Sort intermediate traces. Use `--version 1`
for the original small processed-only sample.
The first command downloads and verifies the cached files; `--validate` also
checks the recording, spikes, stimulation log, mapping, and binning. Use
`--cache-dir /path/to/cache` to choose a location, or `--offline` to reuse a
verified download. Data installs under `tutorials/closed-loop-small/`, with the
project/chip/experiment/recording hierarchy preserved. Existing version folders
migrate automatically; use `--update` for subsequent dataset releases. Refresh the
workshop Catalog to discover the tutorial experiment, or add its parent folder
if you downloaded elsewhere. The old `validate_tutorial_data` command still works.
See [tutorial details](docs/tutorial-data.md) and the
[SpikeLab migration notes](docs/spikelab-migration.md).

## Try the workshop

```bash
python -m pip check
braindance
```

Reinstall after pulling (`python -m pip install -e '.[full]'`) to update console
commands. `braindance --help` lists workshop options; for example,
`braindance --no-browser --port 8766`. The module command still works. The old Qt
launcher is available as `braindance-gui` with the `gui` or `full` extra.

Open <http://127.0.0.1:8765> and click **Run experiment**. Stop with Ctrl+C.
If installation reports an error, stop and fix it before launching. `pip check`
only checks packages already installed; it does not prove BrainDance was installed.
See the [workshop guide](braindance/examples/streaming_workshop/README.md) for replay, profiles, and experiment setup.

Windows CPU/CUDA and native Apple Silicon workshop installs were checked outside
the checkout. The Mac check includes all three games. See
[validation details](docs/installation-validation.md) for results and remaining platform checks.
