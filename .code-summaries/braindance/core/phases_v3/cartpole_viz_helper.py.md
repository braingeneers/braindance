# cartpole_viz_helper.py

**Path:** `braindance/core/phases_v3/cartpole_viz_helper.py`
**Module:** `braindance.core.phases_v3.cartpole_viz_helper`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Provides a live Matplotlib dashboard for neural RL rewards, motor spikes, actions, weight norms, and stimulation rates. Can attach visualization by wrapping a phase's motor-signal and run methods.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Shared data:** add_visualization_to_phase monkey-patches phase.get_motor_signal/run and returns NeuralRLVisualizer.

## Dependencies
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### NeuralRLVisualizer()
> Real-time visualization for neural RL training.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:10`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/cartpole_viz_helper.py:308 (named-call hint); braindance/core/phases_v3/phases3_loop.py:1400 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:943 (named-call hint)
**Constructor:** `__init__(self, n_motor_neurons, n_sensory_neurons, window_size=100, update_interval=500)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `action_history` | inferred at runtime | `deque(maxlen=100)` |
| `ax_actions` | inferred at runtime | `plt.subplot2grid((3, 3), (1, 1))` |
| `ax_current_spikes` | inferred at runtime | `plt.subplot2grid((3, 3), (1, 2))` |
| `ax_metrics` | inferred at runtime | `plt.subplot2grid((3, 3), (2, 1), colspan=2)` |
| `ax_rewards` | inferred at runtime | `plt.subplot2grid((3, 3), (0, 0), colspan=2)` |
| `ax_sensory` | inferred at runtime | `plt.subplot2grid((3, 3), (2, 0))` |
| `ax_spikes` | inferred at runtime | `plt.subplot2grid((3, 3), (1, 0))` |
| `ax_weight_change` | inferred at runtime | `plt.subplot2grid((3, 3), (0, 2))` |
| `bars_spikes` | inferred at runtime | `self.ax_current_spikes.bar(x_pos, np.zeros(self.n_motor), color='blue')` |
| `data_queue` | inferred at runtime | `queue.Queue()` |
| `episode_rewards` | inferred at runtime | `deque(maxlen=window_size)` |
| `im_sensory` | inferred at runtime | `self.ax_sensory.imshow(sensory_data, aspect='auto', cmap='viridis', vmin=0, vmax=50, interpolation='nearest')` |
| `im_spikes` | inferred at runtime | `self.ax_spikes.imshow(spike_data, aspect='auto', cmap='hot', vmin=0, vmax=10, interpolation='nearest')` |
| `n_motor` | inferred at runtime | `n_motor_neurons` |
| `n_sensory` | inferred at runtime | `n_sensory_neurons` |
| `running` | inferred at runtime | `True` |
| `sensory_stim_history` | inferred at runtime | `deque(maxlen=20)` |
| `spike_rates_history` | inferred at runtime | `deque(maxlen=20)` |
| `text_metrics` | inferred at runtime | `self.ax_metrics.text(0.05, 0.5, 'Waiting for data...', fontsize=10, family='monospace', verticalalignment='center')` |
| `update_interval` | inferred at runtime | `update_interval / 1000.0` |
| `update_thread` | inferred at runtime | `threading.Thread(target=self.update_loop)` |
| `weight_history` | inferred at runtime | `deque(maxlen=window_size)` |
| `window_size` | inferred at runtime | `window_size` |
**Methods:**
#### `setup_figure(self)`
> Setup the matplotlib figure with subplots.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:53`
#### `update_data(self, episode_data)`
> Thread-safe data update.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:141`
#### `update_loop(self)`
> Update loop running in separate thread.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:156`
#### `process_data(self, data)`
> Process incoming data and update history.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:179`
#### `update_plots(self)`
> Update all plot elements.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:203`
#### `close(self)`
> Clean shutdown of visualizer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:290`

## Functions
### `add_visualization_to_phase(phase_instance)`
> Add visualization to an existing CartPolePhase instance. Call this after configure_from_experiment.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/cartpole_viz_helper.py:299`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_motor_neurons,n_sensory_neurons,window_size=100,update_interval=500ms. |

## Data Shapes
- Queue dictionaries accept reward,spike_rates,weights,action,sensory_stim; fixed neuron-length vectors update rolling heatmaps.

## Notes
- Import forces TkAgg backend; constructor opens interactive figure and starts daemon thread.
- Background thread updates Matplotlib GUI; close stops thread; wrapper expects CartPole-specific phase attributes.
