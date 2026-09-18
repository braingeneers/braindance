"""
BrainDance Data Manager - Advanced Tutorial 05: Loading RL/CartPole Logs from S3

This tutorial demonstrates how to load reinforcement learning experiment logs
from S3 storage, cache them locally, and visualize learning dynamics.

Learning objectives:
- Load CartPole experiment data with automatic S3 fallback
- Access RL-specific logs (game_log, pattern_log, reward_log)
- Extract training metrics (episode rewards, state dynamics)
- Visualize learning curves and neural-behavioral correlations

Prerequisites:
  - Advanced Tutorial 01 (working with catalog)
  - Basic understanding of RL and CartPole environment
  - S3 credentials configured (for downloading from braingeneers bucket)

Data Structure:
    game_log.csv    - CartPole state at each timestep (pole_angle, reward, action, neural activity)
    pattern_log.csv - Stimulation patterns with policy probabilities
    reward_log.csv  - Episode-level reward summaries (optional, may not exist)

Example Experiment:
    Project: 24-04-18_butterfly
    Chip:    p001237
    Exp:     exp6_cartpole_long_1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path

from braindance.utils.data_manager import load_recording, load_catalog, Styler


# =============================================================================
# Configuration
# =============================================================================
# Load catalog to get correct S3 paths
catalog = load_catalog()

PROJ = "24-04-18_butterfly"
CHIP = "p001237"
EXPERIMENT = "exp4/exp4_cartpole_long_1"


# =============================================================================
# STEP 1: Setup and Experiment Overview
# =============================================================================
print("=" * 70)
print("STEP 1: CartPole RL Experiment Setup")
print("=" * 70)

print("\nThis tutorial uses a CartPole reinforcement learning experiment where")
print("neural organoids control a simulated inverted pendulum (CartPole).")
print()
print("Experiment details:")
print(f"  Project: {PROJ}")
print(f"  Chip:    {CHIP}")
print(f"  Exp:     {EXPERIMENT}")
print()
print("The experiment includes:")
print("  • Standard neural spike data (as in previous tutorials)")
print("  • game_log: CartPole state at each timestep")
print("  • pattern_log: Stimulation patterns with policy info")
print("  • reward_log: Episode summaries (may not exist for all experiments)")
print()


# =============================================================================
# STEP 2: Load Recording with S3 Fallback
# =============================================================================
print("=" * 70)
print("STEP 2: Load Recording from S3")
print("=" * 70)

print("\nLoading recording (will download from S3 if not cached locally)...")

# Load recording directly (it will download from S3 if not cached)
rec = load_recording(PROJ, CHIP, EXPERIMENT)

print(f"✓ Loaded recording: {rec.identifier}")
print()

# Show basic recording info
print("Recording metadata:")
print(f"  • Recording ID: {rec.identifier}")
print(f"  • Has spike data: {rec.spikes is not None}")
print(f"  • Has game log:  {rec.game_log is not None}")
print(f"  • Has stim log:  {rec.stim_log is not None}")
print()


# =============================================================================
# STEP 3: Load RL Logs (Lazy Loading Demonstration)
# =============================================================================
print("=" * 70)
print("STEP 3: Load RL-Specific Logs")
print("=" * 70)

# 3.1 Load game_log
print("\n3.1 Loading game_log (CartPole state data)...")
game_log = rec.game_log

print(f"✓ Loaded game_log: {len(game_log)} timesteps")
print(f"  Columns: {list(game_log.columns)}")
print(f"  Time range: {game_log['time'].min():.2f}s - {game_log['time'].max():.2f}s")
print()
print("First few rows:")
print(game_log.head(3))
print()

# 3.2 Load pattern_log
print("\n3.2 Loading pattern_log (stimulation patterns)...")
pattern_log = rec.pattern_log

print(f"✓ Loaded pattern_log: {len(pattern_log)} stimulation events")
print(f"  Columns: {list(pattern_log.columns)}")
print(f"  Reward range: {pattern_log['reward'].min():.1f} - {pattern_log['reward'].max():.1f}")
    print()

# 3.3 Load reward_log (optional)
print("\n3.3 Loading reward_log (episode summaries - optional)...")
reward_log = rec.reward_log

print(f"✓ Loaded reward_log: {len(reward_log)} episodes")
print(f"  Columns: {list(reward_log.columns)}")
    print()

print("\n💡 All RL logs are lazy-loaded - only downloaded from S3 when accessed")
print()


# =============================================================================
# STEP 4: Data Analysis - Extract Metrics
# =============================================================================
print("=" * 70)
print("STEP 4: Analyze CartPole Performance")
print("=" * 70)

# 4.1 Calculate cumulative reward
print("\n4.1 Computing cumulative reward over time...")
game_log['cumulative_reward'] = game_log['reward'].cumsum()

print(f"  • Total reward: {game_log['cumulative_reward'].iloc[-1]:.0f}")
print(f"  • Mean reward per step: {game_log['reward'].mean():.3f}")
print()

# 4.2 Extract state components
# The 'state' column is now automatically parsed into numpy arrays by the library!

print("4.2 Extracting CartPole state variables...")
valid_states = game_log['state'].dropna()
        
if len(valid_states) > 0:
    state_matrix = np.vstack(valid_states.values)
    game_log.loc[valid_states.index, 'cart_position'] = state_matrix[:, 0]
    game_log.loc[valid_states.index, 'cart_velocity'] = state_matrix[:, 1]
    game_log.loc[valid_states.index, 'pole_angle_state'] = state_matrix[:, 2]
    game_log.loc[valid_states.index, 'pole_angular_velocity'] = state_matrix[:, 3]

    print(f"  ✓ Processed {len(valid_states)} state vectors")
    print(f"    • Cart position range: {state_matrix[:, 0].min():.3f} to {state_matrix[:, 0].max():.3f}")
    print(f"    • Pole angle range: {state_matrix[:, 2].min():.3f} to {state_matrix[:, 2].max():.3f} rad")

print()

# 4.3 Action distribution
print("4.3 Analyzing action distribution...")
print(f"  • Action range: {game_log['action'].min():.3f} to {game_log['action'].max():.3f}")
print(f"  • Mean action: {game_log['action'].mean():.3f}")
print()

# 4.4 Calculate spike totals
# Spike counts are also automatically parsed into numpy arrays!
print("4.4 Computing neural activity totals...")

game_log['total_spikes_l'] = game_log['spike_count_l'].apply(lambda x: np.sum(x) if isinstance(x, np.ndarray) else 0)
game_log['total_spikes_r'] = game_log['spike_count_r'].apply(lambda x: np.sum(x) if isinstance(x, np.ndarray) else 0)
game_log['total_spikes'] = game_log['total_spikes_l'] + game_log['total_spikes_r']

print(f"  ✓ Computed total spike counts")
print(f"    • Mean left motor spikes: {game_log['total_spikes_l'].mean():.2f} per timestep")
print(f"    • Mean right motor spikes: {game_log['total_spikes_r'].mean():.2f} per timestep")

print()


# =============================================================================
# STEP 5: Visualization
# =============================================================================
print("=" * 70)
print("STEP 5: Visualizing Learning Dynamics")
print("=" * 70)

print("\nCreating separate visualizations...")

styler = Styler(journal="draft")

# Plot 1: Cumulative Reward Over Time
fig1, ax1 = styler.create_figure()
ax1.set_title('Learning Curve: Reward Accumulation', fontweight='bold')
ax1.plot(game_log['time'], game_log['cumulative_reward'],
             color=styler.get_named_color('dark_blue'), linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Cumulative Reward')
ax1.grid(axis='both', alpha=0.3, linewidth=0.5)

# Plot 2: Pole Angle Over Time

fig2, ax2 = styler.create_figure()
ax2.set_title('Pole Angle Dynamics (colored by reward)', fontweight='bold')
scatter = ax2.scatter(game_log['time'], game_log['pole_angle'],
                        c=game_log['cumulative_reward'], cmap='viridis',
                        s=5, alpha=0.6)
plt.colorbar(scatter, ax=ax2, label='Cumulative Reward')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pole Angle (rad)')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Upright')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(axis='both', alpha=0.3, linewidth=0.5)

# Plot 3: Action Distribution Evolution
fig3, ax3 = styler.create_figure()
ax3.set_title('Action Distribution: Early vs Late Training', fontweight='bold')
midpoint = len(game_log) // 2
early_actions = game_log['action'].iloc[:midpoint]
late_actions = game_log['action'].iloc[midpoint:]

ax3.hist(early_actions, bins=30, alpha=0.6, color=styler.get_named_color('orange'),
            label=f'Early (first {midpoint} steps)', density=True)
ax3.hist(late_actions, bins=30, alpha=0.6, color=styler.get_named_color('dark_blue'),
            label=f'Late (last {len(game_log) - midpoint} steps)', density=True)
ax3.set_xlabel('Action Value')
ax3.set_ylabel('Density')
ax3.legend(loc='upper right', framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, linewidth=0.5)

# Plot 4: Neural Activity vs Reward
fig4, ax4 = styler.create_figure()
ax4.set_title('Neural Activity vs Reward', fontweight='bold')
time_normalized = (game_log['time'] - game_log['time'].min()) / (game_log['time'].max() - game_log['time'].min())

scatter = ax4.scatter(game_log['total_spikes'], game_log['reward'],
                        c=time_normalized, cmap='plasma', s=10, alpha=0.5)
plt.colorbar(scatter, ax=ax4, label='Time (normalized)')
ax4.set_xlabel('Total Spike Count')
ax4.set_ylabel('Reward')
ax4.grid(axis='both', alpha=0.3, linewidth=0.5)

# Show all plots
plt.show()

print("✓ Visualization complete!")
print()
print("Plot interpretations:")
print("  • Top-left: Shows cumulative reward - should increase if learning occurs")
print("  • Top-right: Pole angle over time - color shows correlation with reward")
print("  • Bottom-left: Action distribution changes - shows policy evolution")
print("  • Bottom-right: Neural activity patterns - colored by time progression")
print()


# =============================================================================
# Summary and Next Steps
# =============================================================================
print("=" * 70)
print("Tutorial Complete!")
print("=" * 70)

print("\nYou've learned:")
print("  ✓ How to load RL experiment logs (game_log, pattern_log, reward_log)")
print("  ✓ How S3 fallback works for lazy-loaded optional data")
print("  ✓ How to parse array-valued columns (state, spike_counts)")
print("  ✓ How to extract training metrics from game logs")
print("  ✓ How to visualize learning dynamics and neural-behavioral correlations")

print("\nKey takeaways:")
print("  💡 RL logs are optional - gracefully return None if not present")
print("  💡 Array columns require special parsing (ast.literal_eval)")
print("  💡 S3 downloads happen automatically on first access")
print("  💡 Second run is instant due to local caching")
print("  💡 Episode detection can be improved - consult collaborators for better methods")

print("\nNext steps:")
print("  • Analyze multiple CartPole experiments using RecordingCatalog")
print("  • Compare different RL algorithms or hyperparameters")
print("  • Correlate policy evolution with neural activity patterns")
print("  • Use pattern_log to analyze stimulation strategy effectiveness")
print("  • Run this tutorial again to see instant loading from cache!")
print("=" * 70)
