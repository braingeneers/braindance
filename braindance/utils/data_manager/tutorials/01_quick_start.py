"""
BrainDance Data Manager - Tutorial 01: Loading Data & Latency Analysis

This tutorial covers the basics of loading neural recordings and performing
stimulus-evoked latency analysis to identify neurons that respond to stimulation.

Prerequisites:
  Run the setup command first (see README.md):

  python -m braindance.utils.data_manager setup \\
    --data-dir /your/path/to/data \\
    --catalog /your/path/to/catalog.csv
"""

import numpy as np
from braindance.utils.data_manager import load_recording, calculate_latencies


# =============================================================================
# 1. Load a Single Recording
# =============================================================================
# This uses the base data directory you configured in setup

proj = '24-105-10_drug_causal'
chip = '20247'
experiment = 'BL1_causal'

print(f"Loading recording: {proj}/{chip}/{experiment}")
rec = load_recording(proj, chip, experiment)

print(f"✓ Recording loaded: {rec.identifier}")


# =============================================================================
# 2. Access Spike Data
# =============================================================================
# Spike data contains the neural activity - when each neuron fired.
# Data is loaded automatically when you access rec.spikes

print("\nLoading spike data...")
spikes = rec.spikes

print(f"✓ Loaded spike data")
print(f"  Number of neurons: {spikes.N}")
print(f"  Recording duration: {spikes.length / 1000:.1f} seconds ({spikes.length} ms)")


# =============================================================================
# 3. Examine Individual Neurons
# =============================================================================
# Each neuron's spike times are stored as an array (in milliseconds)

# Look at the first neuron
neuron_0_spikes = spikes.train[0]
print(f"\nNeuron 0:")
print(f"  Total spikes: {len(neuron_0_spikes)}")
print(f"  First 5 spike times (ms): {neuron_0_spikes[:5]}")
print(f"  Firing rate: {len(neuron_0_spikes) / (spikes.length / 1000):.2f} Hz")


# =============================================================================
# 4. Compute Firing Rates for All Neurons
# =============================================================================

firing_rates = spikes.rates(unit="Hz")

print("\nFiring Rate Statistics:")
print(f"  Mean: {np.mean(firing_rates):.2f} Hz")
print(f"  Median: {np.median(firing_rates):.2f} Hz")
print(f"  Min: {np.min(firing_rates):.2f} Hz")
print(f"  Max: {np.max(firing_rates):.2f} Hz")

# How many neurons are active? (firing > 0.1 Hz)
active = np.sum(firing_rates > 0.1)
print(f"  Active neurons (>0.1 Hz): {active}/{len(firing_rates)}")


# =============================================================================
# 5. Visualize: Raster Plot with Population Rate
# =============================================================================
# The built-in plotting accessor provides publication-ready visualizations

print("\n" + "="*70)
print("VISUALIZATION: Raster Plot with Population Rate")
print("="*70)

# Use the Recording's plot accessor
rec.pl.raster_with_pop(time_window=(0, 60), 
                       time_unit='seconds', 
                       pop_rate_unit='Hz',
                       show=True)

print("✓ Raster plot displayed")
print("  - Black ticks: individual spikes")
print("  - Blue line: population firing rate")

# You can also save directly:
# rec.pl.raster_with_pop(
#     time_window=(0, 60),
#     save_path='./figures',
#     filename='raster_example'
# )


# =============================================================================
# 5.5 Visualize: Spike Time Tiling Coefficient (STTC) Matrix
# =============================================================================
# STTC measures pairwise correlation between spike trains.
# The underlying function is from SpikeLab: spikes.spike_time_tilings()

print("\n" + "="*70)
print("VISUALIZATION: STTC Correlation Matrix")
print("="*70)

# Compute STTC directly using SpikeLab, then plot
sttc_matrix = spikes.spike_time_tilings(delt=20.0).matrix  # delt = time window in ms

# The plotting function will automatically compute STTC if not provided but calculating above can give you more control
rec.pl.sttc_matrix(sttc_matrix=sttc_matrix, 
                    delt=20.0,
                    cmap='viridis',
                    vmin=0,
                    vmax=1,
                    show=True)



print("✓ STTC matrix displayed")
print("  - Each cell shows correlation between two neurons")
print("  - Red = positive correlation, Blue = negative correlation")
print("  - Diagonal = 1 (self-correlation)")



# =============================================================================
# 6. Access Stimulation Information
# =============================================================================
# For stimulated recordings, see when and where stimulation occurred

print("\nLoading stimulation log...")
stim_log = rec.stim_log

if stim_log is not None and len(stim_log) > 0:
    print(f"✓ Loaded stimulation data")
    print(f"  Total stimulations: {len(stim_log)}")
    print(f"  Columns: {list(stim_log.columns)}")
    print("\nFirst 3 stimulations:")
    print(stim_log.head(3))
else:
    print("  No stimulation data available for this recording")


# =============================================================================
# 6.5 Raster Plot with Stimulation Times
# =============================================================================
# Plot raster with stimulation times instead of population rate
    
# Use the plot accessor - no need to create a Styler
rec.pl.raster_with_pop(
                    time_window=(0, 60),           # First 60 seconds
                    time_unit='seconds',
                    stim_log=stim_log,             # Pass stimulation log
                    show_pop_rate=False,           # Hide population rate
                    show_stim_markers=True,        # Show stimulation markers
                    stim_marker_style='line',      # Vertical lines at stim times
                    stim_alpha=0.5,
                    show=True
                    )
    
print("✓ Raster plot with stimulation displayed")
print("  - Black ticks: individual spikes")
print("  - Red lines: stimulation times")


# =============================================================================
# 7. Latency Analysis: Find Stimulus-Evoked Neurons
# =============================================================================
# This is the core analysis: which neurons respond to stimulation?

if stim_log is not None and len(stim_log) > 0:
    print("\n" + "="*70)
    print("LATENCY ANALYSIS: Detecting Stimulus-Evoked Responses")
    print("="*70)
    
    # Calculate latencies with statistical validation
    # Returns only neurons with significant responses
    evoked_pairs = calculate_latencies(
        spikes, 
        stim_log,
        min_response_ratio=1.5,    # Response must be 1.5x baseline
        max_p_value=0.0001,        # Statistical significance threshold
        baseline_window=(-100, 0), # 100ms before stimulus
        response_window=(0, 100),  # 100ms after stimulus
        verbose=True)
    
    print(f"\n✓ Found {len(evoked_pairs)} stimulus-evoked neuron pairs")
    
    if len(evoked_pairs) > 0:
        # Show summary of evoked responses
        print("\nEvoked Response Summary:")
        print("-" * 50)
        
        latencies = []
        for pair_key, data in list(evoked_pairs.items())[:10]:  # Show first 10
            latency = data['onset_latency']
            ratio = data['response_ratio']
            p_val = data['p_value']
            latencies.append(latency)
            print(f"  {pair_key}:")
            print(f"    Onset latency: {latency:.1f} ms")
            print(f"    Response ratio: {ratio:.2f}x baseline")
            print(f"    P-value: {p_val:.2e}")
        
        # Overall statistics
        all_latencies = [d['onset_latency'] for d in evoked_pairs.values()]
        print(f"\nLatency Statistics (n={len(all_latencies)}):")
        print(f"  Mean: {np.mean(all_latencies):.1f} ms")
        print(f"  Median: {np.median(all_latencies):.1f} ms")
        print(f"  Range: {np.min(all_latencies):.1f} - {np.max(all_latencies):.1f} ms")
        
        # Classify response types
        immediate = sum(1 for l in all_latencies if l < 10)
        multi_order = sum(1 for l in all_latencies if l >= 10)
        print(f"\nResponse Classification:")
        print(f"  First-order (<10ms): {immediate} neurons")
        print(f"  Multi-order (≥10ms): {multi_order} neurons")
        
        # Save results
        rec.results.evoked_pairs = evoked_pairs
        rec.results.n_evoked = len(evoked_pairs)
        rec.save_results()
        print("\n✓ Latency results saved to cache")

else:
    print("\nSkipping latency analysis (no stimulation data)")


# =============================================================================
# 8. Visualize a Single Evoked Response
# =============================================================================
# Now let's visualize what an evoked response actually looks like!
# We'll pick one neuron-electrode pair from our latency results

if stim_log is not None and len(evoked_pairs) > 0:
    print("\n" + "="*70)
    print("VISUALIZATION: Single Neuron Evoked Response")
    print("="*70)

    # Pick the first evoked pair as an example
    example_pair_key = list(evoked_pairs.keys())[0]
    example_data = evoked_pairs[example_pair_key]

    # Parse electrode and neuron from pair key
    # Expected format: "electrode_X_neuron_Y"
    parts = example_pair_key.split('_')
    electrode_id = int(parts[1])
    neuron_idx = int(parts[3])

    print(f"\nPlotting example: {example_pair_key}")
    print(f"  Onset latency: {example_data['onset_latency']:.1f} ms")
    print(f"  Response ratio: {example_data['response_ratio']:.2f}x")

    # Create the raster plot using the Recording's plot accessor
    # No need to create a Styler - the accessor handles it internally
    print("\nCreating raster plot...")
    fig_raster = rec.pl.evoked_raster(
        electrode_id=electrode_id,
        neuron_idx=neuron_idx,
        plot_window=(-100, 100),   # 100ms before and after stimulus
        max_trials=400,            # Limit for performance
        latency_data=example_data, # Overlay latency analysis results
        show=True
    )

    print("\n✓ Raster plot displayed")
    print("  Each row = one stimulation trial")
    print("  • Black marks: all spikes in the time window")
    print("  • Red marks: first spike after stimulation (response)")
    print("  • Vertical blue line: stimulation onset (time = 0)")
    print("  • Vertical red line: detected onset latency")

    # Create the PSTH plot using the Recording's plot accessor
    print("\nCreating PSTH (peri-stimulus time histogram)...")
    fig_psth = rec.pl.evoked_psth(
        electrode_id=electrode_id,
        neuron_idx=neuron_idx,
        plot_window=(-100, 100),   # Same window as raster
        max_trials=400,
        psth_bins=75,              # 75 time bins
        latency_data=example_data,
        show=True
    )

    print("\n✓ PSTH plot displayed")
    print("  Shows average firing rate across all trials")
    print("  • Histogram bars: spike rate in time bins")
    print("  • Dotted lines: baseline vs response firing rates")
    print("  • Text box: statistical summary")
    print("\nThe PSTH reveals the average temporal structure of the response")
    print("that isn't obvious from individual trials in the raster plot")

    # You can also use the recording's plot accessor:
    # fig_raster = rec.pl.evoked_raster(electrode_id=electrode_id, neuron_idx=neuron_idx)
    # fig_psth = rec.pl.evoked_psth(electrode_id=electrode_id, neuron_idx=neuron_idx)

else:
    print("\nSkipping evoked response visualization (no evoked pairs found)")


# =============================================================================
# 9. Save Basic Results
# =============================================================================

rec.results.firing_rates = firing_rates
rec.results.mean_firing_rate = np.mean(firing_rates)
rec.save_results()

print("\n✓ All results saved to cache")


print("\n" + "=" * 70)
print("Tutorial 01 complete! You've learned:")
print("  • How to load a recording")
print("  • How to access spike data (spike times for each neuron)")
print("  • How to compute firing rates")
print("  • How to access stimulation information")
print("  • How to run latency analysis to find evoked neurons")
print("  • How to interpret latency results")
print("  • How to visualize single neuron evoked responses (raster + PSTH)")
print("  • How to create raster plots with population rate overlay")

print("\nNext steps:")
print("  • Tutorial 02: Burst detection and network-level analysis")
print("  • Tutorial 03: Population vectors and dimensionality reduction")
print("=" * 70)
