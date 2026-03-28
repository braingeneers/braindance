# BrainDance

## Making neural stimulation easier.

[Please check out our wiki here!](https://braingeneers.github.io/braindance)

---

## Installation

We recommend using [conda](https://docs.anaconda.com/miniconda/miniconda-install/) for setting up dependencies.

```bash
conda create -n brain python=3.11
conda activate brain
```

### Core only (no GPU features)

Installs the base library — environments, phases, data loading, artifact removal.

```bash
pip install git+https://github.com/braingeneers/braindance
```

### With spike detection / RT-Sort (GPU recommended)

Spike detection and RT-Sort require **PyTorch**. Install PyTorch first with the CUDA version that matches your GPU driver, then install BrainDance with the `rt-sort` extra.

```bash
# Step 1 — Install PyTorch (pick ONE line matching your platform)
# See https://pytorch.org/get-started/locally/ for the latest commands

# Linux / Windows — CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Linux / Windows — CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Linux / Windows — CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU acceleration — slower inference)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Install BrainDance with RT-Sort extras
pip install git+https://github.com/braingeneers/braindance#egg=braindance[rt-sort]
```

### TensorRT (optional, Linux only)

TensorRT significantly speeds up real-time spike detection inference but is **optional** and **Linux-only**. If you skip this, BrainDance falls back to standard PyTorch automatically.

```bash
# Install torch_tensorrt matching your PyTorch version
pip install torch_tensorrt
```

See the [Torch-TensorRT installation guide](https://pytorch.org/TensorRT/getting_started/installation.html) for details.

### Version compatibility

The table below lists tested combinations. Your NVIDIA driver's CUDA version (shown by `nvidia-smi`) must be **equal to or higher than** the CUDA version PyTorch was built for.

| CUDA (driver) | PyTorch   | torch_tensorrt | Notes              |
|----------------|-----------|----------------|--------------------|
| 12.6           | 2.6.x     | 2.6.x          | Latest recommended |
| 12.4           | 2.5.x     | 2.5.x          |                    |
| 12.1           | 2.4.x     | 2.4.x          |                    |
| 11.8           | 2.3.x     | 2.3.x          | Legacy CUDA        |

**Not sure what you have?** Run the built-in diagnostic:

```bash
python -m braindance.install_check
```

This detects your OS, GPU, CUDA version, and installed packages, and prints exactly what to install or fix.

---

## Additional integrations

### Kilosort2

To use Kilosort2 within BrainDance, see the public Kilosort2 [GitHub repository](https://github.com/jamesjun/Kilosort2) for installation.

### Open Ephys

If you want to read data in real time from an Open Ephys GUI, [install the Open Ephys GUI](https://open-ephys.github.io/gui-docs/User-Manual/Installing-the-GUI.html) and [Falcon Output plugin](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Falcon-Output.html).

In your Python environment, install [the Open Ephys Python package](https://github.com/open-ephys/open-ephys-python-tools):

```bash
pip install open-ephys-python-tools
```
