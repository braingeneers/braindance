# BrainDance

## Making neural stimulation easier.

[Please check out our wiki here!](https://braingeneers.github.io/braindance)

### Easy installation
We recommend using [conda]([url](https://docs.anaconda.com/miniconda/miniconda-install/)) for setting up dependencies.

```bash
conda create -n brain python=3.11
conda activate brain
```


#### Install only BrainDance:
```bash
pip install git+https://github.com/braingeneers/braindance
```

#### Install BrainDance and RT-Sort

```
pip install git+https://github.com/braingeneers/braindance#egg=braindance[rt-sort]
```
Additionally, install [PyTorch](https://pytorch.org/get-started/locally/) with any version of CUDA as the compute platform.   
If running on a Linux machine, install [Torch-TensorRT](https://pytorch.org/TensorRT/getting_started/installation.html) as well for faster RT-Sort computation speed.

#### Kilosort2 integration
To use Kilosort2 within BrainDance, see the public Kilosort2 [GitHub repository](https://github.com/jamesjun/Kilosort2) for installation 

#### Open Ephys
If you want to read data in real time from an Open Ephys GUI, [install the Open Ephys GUI](https://open-ephys.github.io/gui-docs/User-Manual/Installing-the-GUI.html) and [Falcon Output plugin](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Falcon-Output.html).

In your Python environment, install [the Open Ephys Python package](https://github.com/open-ephys/open-ephys-python-tools).
```bash
pip install open-ephys-python-tools
```
