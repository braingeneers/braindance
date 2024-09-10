# BrainDance

## Making neural stimulation easier.

[Please check out our wiki here!](https://braingeneers.github.io/braindance)

### Easy installation
We recommend using [conda]([url](https://docs.anaconda.com/miniconda/miniconda-install/)) for setting up dependencies.

```bash
conda create -n brain python=3.11
conda activate brain
```


```bash
pip install git+https://github.com/braingeneers/braindance
```

#### RT-Sort installation
To use RT-Sort, install the following additional libraries in your Python environment:
- [torch_tensorrt](https://pytorch.org/TensorRT/getting_started/installation.html)
  - If running on a Windows OS, install [torch](https://pytorch.org/get-started/locally/) instead.
  - RT-Sort should still work even if you see torch_tensorrt import warnings. But to remove them, create a new Python environment and install torch_tensorrt FIRST (even before BrainDance).
- [diptest](https://pypi.org/project/diptest/)
- [pynvml](https://pypi.org/project/pynvml/)
- [scikit-learn](https://scikit-learn.org/stable/install)
- [spikeinterface](https://spikeinterface.readthedocs.io/en/stable/get_started/installation.html)

#### Kilosort2 installation
To use Kilosort2, you must install [Kilosort2](https://github.com/jamesjun/Kilosort2) and install the following additional libraries in your Python environment:
- [natsort](https://pypi.org/project/natsort/)
- [spikeinterface](https://spikeinterface.readthedocs.io/en/stable/get_started/installation.html) 

#### Open Ephys
If you want to read data in real time from an Open Ephys GUI, [install the Open Ephys GUI](https://open-ephys.github.io/gui-docs/User-Manual/Installing-the-GUI.html) and [Falcon Output plugin](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Falcon-Output.html).

In your Python environment, install [the Open Ephys Python package](https://github.com/open-ephys/open-ephys-python-tools).
```bash
pip install open-ephys-python-tools
```
