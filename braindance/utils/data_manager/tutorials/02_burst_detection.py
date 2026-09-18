"""
BrainDance Data Manager - Tutorial 02: Burst Detection & Network Analysis

This tutorial covers network-level analysis: detecting population bursts and
measuring burst latencies in response to electrical stimulation.

Bursts are synchronized firing events across many neurons - a hallmark of
organoid and neural culture activity. This tutorial shows how to:
1. Detect bursts in population activity
2. Analyze burst properties (width, amplitude, frequency)
3. Measure stimulus-evoked burst latencies

Prerequisites:
  - Completed Tutorial 01
  - A recording with spike data (and optionally stimulation data)
"""

# TODO: add in that can color UMAP by different things such as pop rate and electrode and what not

import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from braindance.utils.data_manager import (
    load_recording,
    BurstDetector,
    BurstLatencyAnalyzer,
    Styler
)


# =============================================================================
# 1. Load Recording Data
# =============================================================================

# proj = '24-105-10_drug_causal'
# chip = '20247'
# experiment = 'BL1_causal'


proj = '25-02-25_busybees'
chip = '25123ic'
experiment = 'exp1_cont_1'

save_dir = '/Volumes/hunter_ssd/busy_bee/test_plots'

print(f"Loading recording: {proj}/{chip}/{experiment}")
rec = load_recording(proj, chip, experiment)
spikes = rec.spikes

print(f"✓ Loaded: {spikes.N} neurons, {spikes.length/1000:.1f}s duration")


# =============================================================================
# 2. Detect Network Bursts
# =============================================================================
# BurstDetector analyzes population activity to find synchronized firing events

print("\n" + "="*70)
print("BURST DETECTION: Finding Network Synchronization Events")
print("="*70)

# Initialize burst detector with default parameters
detector = BurstDetector(
    spikes,
    bin_size=1.0,           # 1ms bins for population activity
    smoothing_window=50,    # 50ms smoothing window
    burst_detection_params={
        'baseline_percentile': 25,      # Baseline from 25th percentile
        'peak_threshold_factor': 2.5,   # Peaks must be 2.5 std above baseline
        'peak_distance': 200,           # Minimum 200ms between burst peaks
        'peak_prominence': 0.5          # Minimum peak prominence
    },
    burst_edge_params={
        'edge_threshold_factor': 0.2,   # Edge detection threshold
        'min_burst_width': 20,          # Minimum burst width (ms)
        'max_burst_width': 500          # Maximum burst width (ms)
    }
)

# Detect bursts with automatic caching - results are cached for fast subsequent runs!
results = detector.detect_bursts(cache=rec.cache)

print(f"\n✓ Detected {results.n_bursts} network bursts")


# =============================================================================
# 3. Examine Burst Properties
# =============================================================================

print("\nBurst Properties:")
print("-" * 50)

# Get burst times (start, end for each burst)
burst_times = detector.get_burst_times()
print(f"  Burst timing array shape: {burst_times.shape}")

# Burst statistics
summary = detector.get_summary_metrics()
print(f"\n  Number of bursts: {summary['n_bursts']}")
print(f"  Burst frequency: {summary.get('burst_frequency', 0):.3f} Hz")
print(f"  Mean burst width: {summary.get('mean_burst_width', 0):.1f} ms")
print(f"  Std burst width: {summary.get('std_burst_width', 0):.1f} ms")
print(f"  Mean burst amplitude: {summary.get('mean_burst_amplitude', 0):.1f} spikes/bin")

# Burst involvement coefficient (BIC)
# Shows how reliably each neuron participates in bursts
bic = results.bic_matrix
print(f"\nBurst Involvement Coefficients:")
print(f"  Mean BIC: {np.mean(bic):.2f}")
print(f"  Neurons with BIC > 0.5: {np.sum(bic > 0.5)}/{len(bic)}")
print(f"  Neurons with BIC > 0.9 (rigid): {np.sum(bic > 0.9)}/{len(bic)}")

# Backbone classification
backbone = results.backbone_classification
print(f"\nBackbone Classification:")
print(f"  Rigid neurons (participate in >90% of bursts): {len(backbone['rigid'])}")
print(f"  Non-rigid neurons: {len(backbone['nonrigid'])}")

# Show first few bursts
print(f"\nFirst 5 bursts (start_ms, end_ms):")
for i, (start, end) in enumerate(burst_times[:5]):
    width = end - start
    print(f"  Burst {i+1}: {start:.0f} - {end:.0f} ms (width: {width:.0f} ms)")



# =============================================================================
# 4. Plot Population Activity with Detected Bursts
# =============================================================================

print("\nCreating population activity plot with burst overlays...")

# Initialize styler for consistent formatting
styler = Styler()

# Create figure with two subplots: full view and zoomed view
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Convert time bins from ms to seconds for plotting
time_s = results.time_bins / 1000.0

# ========== SUBPLOT 1: Full Recording Overview ==========
ax1.plot(time_s, results.smoothed_activity,
         color=styler.colors[5], linewidth=1.0, label='Population Activity')

# Overlay detected bursts as shaded regions
burst_times = detector.get_burst_times()
for i, (start_ms, end_ms) in enumerate(burst_times):
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    ax1.axvspan(start_s, end_s,
                color=styler.colors[0], alpha=0.3,
                label='Detected Burst' if i == 0 else '')

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Population Activity (spikes/bin)')
ax1.set_title(f'Population Activity - Full Recording ({results.n_bursts} bursts)')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# ========== SUBPLOT 2: Zoomed View (first 60 seconds) ==========
# Simple zoom: show first 60 seconds of recording
zoom_duration_s = 60
zoom_start_s = time_s[0]
zoom_end_s = min(zoom_start_s + zoom_duration_s, time_s[-1])

# Plot zoomed region
zoom_mask = (time_s >= zoom_start_s) & (time_s <= zoom_end_s)
ax2.plot(time_s[zoom_mask], results.smoothed_activity[zoom_mask],
         color=styler.colors[5], linewidth=1.2)

# Overlay bursts in zoomed region
for start_ms, end_ms in burst_times:
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    if start_s <= zoom_end_s and end_s >= zoom_start_s:
        ax2.axvspan(start_s, end_s, color=styler.colors[0], alpha=0.3)

ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Population Activity (spikes/bin)')
ax2.set_title(f'Zoomed View: First {zoom_end_s - zoom_start_s:.0f} seconds')
ax2.set_xlim(zoom_start_s, zoom_end_s)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
Path(save_dir).mkdir(parents=True, exist_ok=True)
save_path = Path(save_dir) / 'population_activity_with_bursts'
plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight')
print(f"✓ Saved population activity plot to {save_path}")
plt.close()



# =============================================================================
# 5. Stimulus-Evoked Burst Analysis (if stimulation data available)
# =============================================================================

stim_log = rec.stim_log

# Initialize burst latency analyzer
burst_analyzer = BurstLatencyAnalyzer(
    spikes,
    stim_log,
    baseline_window_ms=500.0,    # 500ms baseline before stimulus
    analysis_window_ms=500.0,    # Look for bursts within 500ms post-stim
    validation_params={
        'probability_increase_factor': 1.5,  # 50% increase in burst probability
        'max_p_value': 0.01,                 # Statistical significance
        'min_evoked_bursts': 3,              # Minimum evoked bursts
        'max_mean_latency_ms': 400.0         # Reasonable timing window for late bursts
    },
    verbose=False,
    cache=rec.cache  # Enable caching to reuse burst detection results
)
    
# Analyze burst latencies per electrode
burst_results = burst_analyzer.analyze_burst_latencies()
    
print(f"\n✓ Analyzed {len(burst_results)} stimulation electrodes")
    
# Show results - detailed for validated electrodes only
validated_count = 0
for electrode_id, electrode_data in burst_results.items():
    is_valid = electrode_data.get('validated', False)
    if is_valid:
        validated_count += 1

        n_stim = electrode_data['n_stimuli']
        n_evoked = electrode_data['n_evoked_bursts']
        prob = electrode_data['evoked_burst_probability']
        mean_lat = electrode_data['mean_burst_latency']

        print(f"\n  ✓ Electrode {electrode_id}: VALIDATED")
        print(f"    Stimuli: {n_stim}, Evoked bursts: {n_evoked}")
        print(f"    Evoked burst probability: {prob:.1%}")

        if not np.isnan(mean_lat):
            std_lat = electrode_data['std_burst_latency']
            print(f"    Mean burst latency: {mean_lat:.1f} ± {std_lat:.1f} ms")

            # Show burst type breakdown
            burst_types = electrode_data['burst_types']
            print(f"    Immediate (<25ms): {burst_types['immediate']}")
            print(f"    Late (>25ms): {burst_types['late']}")

            # Show validation details
            if 'validation' in electrode_data:
                val = electrode_data['validation']
                print(f"    P-value: {val.get('p_value', 1.0):.4f}")
                print(f"    Probability ratio: {val.get('probability_ratio', 0):.2f}x baseline")

print(f"\n✓ {validated_count}/{len(burst_results)} electrodes show significant burst evocation")

# Get summary statistics
summary_stats = burst_analyzer.get_summary_statistics()
print(f"\nSummary:")
print(f"  Total bursts in recording: {summary_stats['total_bursts_detected']}")



# =============================================================================
# 6. Peri-Stimulus Burst Probability (PSTH-style analysis)
# =============================================================================

print("\nCreating peri-stimulus burst probability plot...")

# Analysis parameters
psth_window_ms = 500  # ±500ms around stimulation
psth_bin_size_ms = 25  # 25ms bins for probability calculation

# Create time bins relative to stimulation
time_bins = np.arange(-psth_window_ms, psth_window_ms + psth_bin_size_ms, psth_bin_size_ms)
bin_centers = time_bins[:-1] + psth_bin_size_ms / 2
n_bins = len(time_bins) - 1

# Initialize burst count per bin
burst_counts = np.zeros(n_bins)

# Get stim times as numpy array (much faster than iterrows!)
stim_times_ms = stim_log['time'].values * 1000
n_total_stims = len(stim_times_ms)

# Get burst times as array (n_bursts x 2: [start, end])
burst_times_array = detector.get_burst_times()

# Vectorized computation: for each time bin, count overlapping bursts
for bin_idx in range(n_bins):
    bin_start_rel = time_bins[bin_idx]
    bin_end_rel = time_bins[bin_idx + 1]

    # Absolute time windows for all stimuli at once
    bin_starts = stim_times_ms + bin_start_rel  # shape: (n_stims,)
    bin_ends = stim_times_ms + bin_end_rel      # shape: (n_stims,)

    # Check overlap using broadcasting: (n_stims, 1) vs (1, n_bursts)
    burst_starts = burst_times_array[:, 0][np.newaxis, :]  # (1, n_bursts)
    burst_ends = burst_times_array[:, 1][np.newaxis, :]    # (1, n_bursts)

    # Overlap: burst_start <= bin_end AND burst_end >= bin_start
    overlaps = (burst_starts <= bin_ends[:, np.newaxis]) & (burst_ends >= bin_starts[:, np.newaxis])

    # Count stimuli with at least one overlapping burst
    has_burst = np.any(overlaps, axis=1)  # (n_stims,)
    burst_counts[bin_idx] = np.sum(has_burst)

# Calculate burst probability per bin
burst_probability = burst_counts / n_total_stims

# Calculate baseline burst probability (from pre-stimulus period)
baseline_bins = bin_centers < 0
baseline_probability = np.mean(burst_probability[baseline_bins]) if np.any(baseline_bins) else 0

# Create figure
fig, ax = styler.create_figure(size_preset='1.5')

# Plot burst probability
ax.bar(bin_centers, burst_probability, width=psth_bin_size_ms * 0.9,
       color=styler.colors[0], alpha=0.7, label='Burst Probability')

# Add baseline reference line
ax.axhline(y=baseline_probability, color=styler.colors[7],
           linestyle='--', alpha=0.7, linewidth=1.5,
           label=f'Baseline ({baseline_probability:.2%})')

# Add vertical line at stimulus time
ax.axvline(x=0, color=styler.colors[5], linestyle='-',
           alpha=0.8, linewidth=2, label='Stimulation')

# Formatting
ax.set_xlabel('Time relative to stimulus (ms)')
ax.set_ylabel('Burst Probability')
ax.set_title(f'Peri-Stimulus Burst Probability\n({n_total_stims} stimuli, {psth_bin_size_ms}ms bins)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Set y-axis to percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()

# Save the plot
save_path = Path(save_dir) / 'peri_stimulus_burst_probability'
plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight')
print(f"✓ Saved peri-stimulus burst probability to {save_path}")
plt.close()


# =============================================================================
# 7. Show an aligned raster of the electrodes that show significant burst evocation
# =============================================================================

print("\nCreating evoked burst heatmaps for validated electrodes...")

# Filter to only validated electrodes
validated_electrodes = [(eid, data) for eid, data in burst_results.items() 
                       if data.get('validated', False)]

print(f"Found {len(validated_electrodes)} validated electrode(s)")

# Analysis parameters
window_ms = 500  # ±500ms around stimulation
bin_size_ms = 50  # 50ms bins

for electrode_id, electrode_data in validated_electrodes:
    print(f"  Creating heatmap for electrode {electrode_id}...")
    
    # Get stimulation times for this electrode
    electrode_stims = []
    for idx, row in stim_log.iterrows():
        # Parse stimulated electrodes
        stim_electrodes = row['stim_electrodes']
        if isinstance(stim_electrodes, str):
            stim_electrodes = ast.literal_eval(stim_electrodes)
        elif not isinstance(stim_electrodes, (list, tuple)):
            stim_electrodes = [stim_electrodes]
        
        # Check if this electrode was stimulated
        if electrode_id in stim_electrodes:
            stim_time_ms = row['time'] * 1000
            electrode_stims.append(stim_time_ms)
    
    # Create time bins
    time_bins = np.arange(-window_ms, window_ms + bin_size_ms, bin_size_ms)
    n_bins = len(time_bins) - 1
    n_trials = len(electrode_stims)
    
    # Create burst presence matrix: trials x time_bins
    burst_matrix = np.zeros((n_trials, n_bins))
    
    # Get all burst times
    burst_times = detector.get_burst_times()
    
    # Fill matrix with burst activity
    for trial_idx, stim_time in enumerate(electrode_stims):
        trial_start = stim_time - window_ms
        trial_end = stim_time + window_ms
        
        # Find bursts in this trial window
        for burst_start, burst_end in burst_times:
            # Check if burst overlaps with trial window
            if burst_start <= trial_end and burst_end >= trial_start:
                # Determine which time bins this burst affects
                burst_start_rel = burst_start - trial_start
                burst_end_rel = burst_end - trial_start
                
                # Find corresponding bin indices
                start_bin = int(np.clip(burst_start_rel // bin_size_ms, 0, n_bins - 1))
                end_bin = int(np.clip(burst_end_rel // bin_size_ms, 0, n_bins - 1))
                
                # Mark bins as having burst activity
                for bin_idx in range(start_bin, end_bin + 1):
                    if 0 <= bin_idx < n_bins:
                        burst_matrix[trial_idx, bin_idx] = 1
    
    # Create heatmap visualization
    fig, ax = styler.create_figure(size_preset='single')
    
    # Create binary colormap: white background, colored bursts
    from matplotlib.colors import ListedColormap
    burst_color = styler.colors[0]  # Purple for bursts
    binary_cmap = ListedColormap(['white', burst_color])
    
    # Bin centers for x-axis
    bin_centers = time_bins[:-1] + bin_size_ms/2
    
    # Create heatmap
    im = ax.imshow(burst_matrix, aspect='auto', cmap=binary_cmap,
                  interpolation='nearest', vmin=0, vmax=1,
                  extent=[bin_centers[0] - bin_size_ms/2,
                         bin_centers[-1] + bin_size_ms/2,
                         -0.5, n_trials - 0.5])
    
    # Add vertical line at stimulus time (t=0)
    ax.axvline(x=0, color=styler.colors[5], linestyle='--', 
              alpha=0.7, linewidth=1.5, label='Stimulation')
    
    # Labels and title
    ax.set_xlabel('Time relative to stimulus (ms)')
    ax.set_ylabel('Trial number')
    ax.set_title(f'Evoked Burst Activity - Electrode {electrode_id}\n'
                f'({n_trials} trials, {electrode_data["n_evoked_bursts"]} evoked bursts)')
    
    # Set x-axis ticks
    tick_interval = 100
    x_ticks = np.arange(-window_ms, window_ms + tick_interval, tick_interval)
    ax.set_xticks(x_ticks)
    
    # Set y-axis ticks
    y_ticks = np.arange(0, n_trials + 1, max(1, n_trials // 10))
    ax.set_yticks(y_ticks)
    
    # Add analysis info
    mean_lat = electrode_data.get('mean_burst_latency', np.nan)
    prob = electrode_data.get('evoked_burst_probability', 0)
    info_text = f"Evoked probability: {prob:.1%}\nMean latency: {mean_lat:.1f} ms"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=7,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / f'evoked_bursts_electrode_{electrode_id}'
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight')
    print(f"  ✓ Saved evoked burst heatmap to {save_path}")
    plt.close()


# =============================================================================
# 8. Save Results
# =============================================================================

# Burst detection results are automatically cached by detector.detect_bursts()
# Save burst latency analysis results to DataContext
rec.results.burst_latency_results = burst_results
rec.save_results()
print("\n✓ Burst latency results saved to cache")


print("\n" + "=" * 70)
print("Tutorial 02 complete! You've learned:")
print("  • How to detect network bursts using BurstDetector")
print("  • How to interpret burst properties (width, amplitude, frequency)")
print("  • How to identify backbone neurons using BIC")
print("  • How to analyze stimulus-evoked burst latencies")
print("  • How to validate burst evocation statistically")

print("\nNext steps:")
print("  • Tutorial 03: Population vectors and dimensionality reduction")
print("=" * 70)
