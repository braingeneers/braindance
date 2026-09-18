# model2.py

**Path:** `braindance/core/spikedetector/model2.py`
**Module:** `braindance.core.spikedetector.model2`
**Feature Area:** `Spike Detection`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements convolutional spike-localization models and their training, evaluation, threshold tuning, persistence, and TorchScript/TensorRT compilation. Alternative architectures include a one-dimensional U-Net and an RMS-threshold baseline.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.model` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.
- **Uses:** `plot` from `braindance.core.spikedetector` — imports (static evidence).
- **Uses:** `utils` from `braindance.core.spikedetector` — imports (static evidence).
- **Shared data:** train.py calls fit; RT-Sort loads/compiles detector and interprets frame logits.

## Dependencies
- `braindance` — external or unresolved local import; source import evidence.
- `braindance.core.spikedetector.plot` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.utils` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch_tensorrt` — external or unresolved local import; source import evidence.

## Classes
### ModelSpikeSorter(nn.Module)
> DL model for spike sorting
**Source:** `braindance/core/spikedetector/model2.py:21`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:1081 (named-call hint); braindance/core/spikedetector/model2.py:1409 (named-call hint)
**Constructor:** `__init__(self, num_channels_in: int, sample_size: int, buffer_front_sample: int, buffer_end_sample: int, loc_prob_thresh: float=35, buffer_front_loc: int=0, buffer_end_loc: int=0, input_scale=0.01, samp_freq=None, device: str='cuda', dtype=torch.float16, architecture_params=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `architecture_params` | inferred at runtime | `architecture_params` |
| `buffer_end_loc` | inferred at runtime | `buffer_end_loc` |
| `buffer_end_sample` | inferred at runtime | `buffer_end_sample` |
| `buffer_front_loc` | inferred at runtime | `buffer_front_loc` |
| `buffer_front_sample` | inferred at runtime | `buffer_front_sample` |
| `device` | inferred at runtime | `device` |
| `dtype` | inferred at runtime | `dtype` |
| `input_scale` | inferred at runtime | `input_scale` |
| `loc_first_frame` | inferred at runtime | `self.buffer_front_sample - self.buffer_front_loc` |
| `loc_last_frame` | inferred at runtime | `sample_size - buffer_end_sample + buffer_end_loc - 1` |
| `loc_prob_thresh_logit` | inferred at runtime | `0` |
| `logs` | inferred at runtime | `{}` |
| `loss_localize` | inferred at runtime | `nn.BCEWithLogitsLoss(reduction='none')` |
| `model` | inferred at runtime | `model` |
| `num_channels_in` | inferred at runtime | `num_channels_in` |
| `num_output_locs` | inferred at runtime | `sample_size - buffer_end_sample + buffer_end_loc - (buffer_front_sample - buffer_front_loc)` |
| `path` | inferred at runtime | `None` |
| `samp_freq` | inferred at runtime | `samp_freq` |
| `sample_size` | inferred at runtime | `sample_size` |
**Methods:**
#### `init_weights_and_biases(self, method: str, prelu_init=0.25)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:119`
#### `init_final_bias(self, num_wfs_probs: list)`
> Initialize bias of the final layer based on the waveform probabilities of training dataset (assumes 50% of samples contain no waveform)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:131`
#### `forward(self, x)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:158`
#### `loss(self, outputs, num_wfs, wf_locs)`
> Build sparse location labels and sum BCE-with-logits loss across predicted frame locations.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:173`
#### `train_epoch(self, dataloader, optim)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:221`
#### `fit(self, dataloader_train, dataloader_val=None, optim='adam', num_epochs=100, epoch_patience=10, training_thresh=0.5, lr=0.0003, momentum=0.9, lr_patience=5, lr_factor=0.1, tune_thresh_every=10, save_best=True)`
> Fit self to dataloader_train
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/model2.py:251`
#### `set_loc_prob_thresh(self, loc_prob_thresh)`
> loc_prob_thresh is in (0, 100) internally, self.loc_prob_thresh_logit is (-inf, inf) since model's outputs are not from sigmoid
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/model2.py:464`
#### `get_loc_prob_thresh(self)`
> loc_prob_thresh is in (0, 100) internally, self.loc_prob_thresh_logit is (-inf, inf) since model's outputs are not from sigmoid
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:472`
#### `loc_to_logit(self, loc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:479`
#### `logit_to_loc(self, logit)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:483`
#### `outputs_to_preds(self, outputs, return_wf_count=False)`
> Convert raw model outputs to predictions
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:490`
#### `perf(self, dataloader, loc_buffer=8, plot_preds=(), max_plots=10, outputs_list=None)`
> Greedily match closest predicted and labeled spike frames within tolerance to compute detection and localization metrics.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:541`
#### `plot_pred(self, trace: torch.Tensor, output, pred, num_wf, wf_labels, wf_alphas, multi_rec=None)`
> Plot models prediction for a sample
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:738`
#### `plot_loc_probs(self, model_output, axis)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:827`
#### `save(self, folder, logs=(), verbose=True)`
> In folder, saves another folder with time it was created which contains all relevant info about the model in the following hierarchy: folder yymmdd_HHMMSS_ffffff (see utils.get_time for more details) state_dict.pt: model's PyTorch parameters (weights, biases, etc) init_dict.json: model's init args # src: All source code in src folder (except __init__.py) that are needed to recreate and run model # data.py # model.py # plot.py # train.py # utils.…
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikedetector/model2.py:843`
#### `tune_loc_prob_thresh(self, dataloader, start=None, stop=50, step=2.5, verbose=True, outputs_list=None)`
> Reuse outputs while sweeping thresholds and retain the best F1 score.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:917`
#### `log(self, path, save_data)`
> Save save_data to model_path/log/path
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikedetector/model2.py:977`
#### `compile(self, n_dim_0: int, model_save_path=None, input_size=None, dtype=torch.float16, device='cuda')`
> Compile model with torch tensorrt
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikedetector/model2.py:998`
#### `load_compiled(model_save_path)`
> Load saved compiled model
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1046`
#### `load(detection_model_path)`
> Loads a model from the specified folder detection_model_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1053`
#### `get_output_shape(layer, input_shape, device='cpu', dtype=torch.float32)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1079`
#### `get_same_padding(conv_kwargs)`
> Get padding layer analogous to TensorFlow's SAME padding (output size is same as input IFF stride=1)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1084`
#### `perf_report(preface, perf)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1092`
#### `load_mea()`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1107`
#### `load_neuropixels()`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1112`
### ModelTuning(nn.Module)
> Build valid Conv1d layers with optional noise features or U-Net architecture.
**Source:** `braindance/core/spikedetector/model2.py:1117`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:92 (named-call hint)
**Constructor:** `__init__(self, architecture, num_channels, relu, add_conv, bottleneck, noise, filter, sample_size=200)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `conv` | inferred at runtime | `conv` |
| `filter` | inferred at runtime | `data.BandpassFilter((300, 3000)) if filter else None` |
| `last_layer` | inferred at runtime | `list(conv.modules())[-1]` |
| `noise` | inferred at runtime | `nn.Flatten()` |
**Methods:**
#### `forward(self, x)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1193`
#### `init_final_bias(self, num_output_locs: int, num_wfs_probs: list)`
> Initialize bias of the final layer based on the waveform probabilities of training dataset (assumes 50% of samples contain no waveform)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1208`
#### `parse_architecture(architecture)`
> Convert architecture number to num_layers and kernel_size
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1232`
### UNet(nn.Module)
> unclear — see source
**Source:** `braindance/core/spikedetector/model2.py:1281`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:1180 (named-call hint)
**Constructor:** `__init__(self, depth=4, first_conv_channels=32)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `contracting` | inferred at runtime | `nn.ModuleList()` |
| `expanding` | inferred at runtime | `nn.ModuleList()` |
| `last` | inferred at runtime | `nn.Conv1d(in_channels_x, 1, 1)` |
| `pool` | inferred at runtime | `nn.MaxPool1d(2, 2)` |
**Methods:**
#### `forward(self, x)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1303`
### ConvBlock(nn.Module)
> unclear — see source
**Source:** `braindance/core/spikedetector/model2.py:1322`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:1302 (named-call hint); braindance/core/spikedetector/model2.py:1351 (named-call hint)
**Constructor:** `__init__(self, in_channels, out_channels, kernel_size=3)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `conv1` | inferred at runtime | `nn.Conv1d(in_channels, out_channels, kernel_size)` |
| `conv2` | inferred at runtime | `nn.Conv1d(out_channels, out_channels, kernel_size)` |
| `relu` | inferred at runtime | `nn.ReLU()` |
**Methods:**
#### `forward(self, x)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1329`
### ExpandBlock(nn.Module)
> unclear — see source
**Source:** `braindance/core/spikedetector/model2.py:1333`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:1310 (named-call hint)
**Constructor:** `__init__(self, in_channels_x, kernel_size=3)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `conv_block` | inferred at runtime | `ConvBlock(in_channels_x, in_channels_x // 2, kernel_size)` |
| `relu` | inferred at runtime | `nn.ReLU()` |
| `up_conv` | inferred at runtime | `nn.ConvTranspose1d(in_channels_x, in_channels_x // 2, 2, 2)` |
**Methods:**
#### `forward(self, x, cat)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1342`
#### `crop(x, size_out)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1348`
### RMSThresh(nn.Module)
> Model where spikes are classified based on RMS threshold
**Source:** `braindance/core/spikedetector/model2.py:1355`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/model2.py:94 (named-call hint)
**Constructor:** `__init__(self, thresh=5, sample_size=200, buffer_front=40, buffer_end=40)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `buffer_end` | inferred at runtime | `buffer_end` |
| `buffer_front` | inferred at runtime | `buffer_front` |
| `filter` | inferred at runtime | `data.BandpassFilter((300, 3000))` |
| `thresh` | inferred at runtime | `thresh` |
**Methods:**
#### `forward(self, x)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1369`

## Functions
### `sigmoid(x)`
> unclear — see source
> **Called by:** braindance/core/spikedetector/model2.py:678 (named-call hint); braindance/core/spikedetector/model2.py:687 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1380`
### `main()`
> unclear — see source
> **Called by:** braindance/core/spikedetector/model2.py:1423 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/model2.py:1389`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| sample_size, front/end buffers, percentage detection threshold, input_scale, samp_freq in kHz |
| architecture_params selects convolution depth/kernel, activation, bottleneck and noise options; device/dtype |

## Data Shapes
- Input (batch,channels_in,sample_size); output logits (batch,output_locations); predictions are variable-length frame-index arrays.
- Saved folder contains state_dict.pt, init_dict.json and log artifacts; compiled model filename compiled.ts.

## Notes
- compile traces model.conv directly, so callers must apply wrapper input scaling themselves.
- Optional filter and RMSThresh paths reference data without importing it.
- load uses stored device configuration; saved init_dict omits samp_freq and dtype.
