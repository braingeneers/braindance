"""
BrainDance Data Manager - Advanced Tutorial 01: Working with Recording Catalogs

This tutorial teaches you how to use catalogs to organize and filter recordings,
then perform batch analysis across different experimental conditions.

Learning Path:
  1. Load and explore your catalog
  2. Filter recordings by conditions
  3. Practical analysis: Compare latencies across frequencies

Prerequisites:
  - Completed Tutorial 01 (quick_start) - you should know how to load a single recording
  - Have a catalog CSV file (created during setup)
"""

from braindance.utils.data_manager import load_catalog, Styler
import numpy as np


# =============================================================================
# STEP 1: Load Your Catalog
# =============================================================================
print("="*70)
print("STEP 1: Loading Your Catalog")
print("="*70)

catalog = load_catalog()
print(f"\n✓ Loaded catalog with {len(catalog)} recordings")
print(f"  • Different chips: {catalog.df['chip'].nunique()}")
print(f"  • Different projects: {catalog.df['proj'].nunique()}")
print()


# =============================================================================
# STEP 2: Basic Filtering
# =============================================================================
print("="*70)
print("STEP 2: Filtering Recordings")
print("="*70)

# Simple filters using comparison suffixes
print("\nBasic filtering examples:")
print(f"  • Stimulated (freq > 0): {len(catalog.filter(freq__gt=0))} recordings")
print(f"  • Spontaneous (freq = 0): {len(catalog.filter(freq=0))} recordings")
print(f"  • High-freq (>= 4 Hz): {len(catalog.filter(freq__gte=4))} recordings")

# Combined filters (AND logic)
print("\nCombined filtering (multiple conditions):")
stim_non_baseline = catalog.filter(freq__gt=0, baseline=False)
print(f"  • Stimulated + non-baseline: {len(stim_non_baseline)} recordings")

# String matching
print("\nString matching (chip names):")
chip_subset = catalog.filter(chip__contains='251')
print(f"  • Chips containing '251': {len(chip_subset)} recordings")

print("\n💡 Filter suffixes: __gt (>), __gte (>=), __lt (<), __lte (<=), __in, __contains")
print()


# =============================================================================
# STEP 3: Grouping and Summary
# =============================================================================
print("="*70)
print("STEP 3: Grouping and Viewing Data")
print("="*70)

# Group by frequency
by_frequency = catalog.group_by('freq')
print(f"\nFound {len(by_frequency)} different stimulation frequencies:")
for freq in sorted(by_frequency.keys())[:6]:
    count = len(by_frequency[freq])
    print(f"  {freq:6.1f} Hz: {count:3d} recordings")

# Quick iteration example
print("\nExample: Looking at 8 Hz recordings")
freq_8hz = catalog.filter(freq=8)
if len(freq_8hz) > 0:
    print(f"  Found {len(freq_8hz)} recordings at 8 Hz")
    print("  First 2:")
    for i, rec in enumerate(freq_8hz[:2]):
        print(f"    {i+1}. {rec.chip} - {rec.experiment}")
print()


# =============================================================================
# STEP 4: Practical Analysis - Latency Across Frequencies
# =============================================================================
print("="*70)
print("STEP 4: PRACTICAL EXAMPLE - Latency Analysis Across Frequencies")
print("="*70)

print("\nScenario: Compare stimulus-evoked responses at different frequencies")
print("Goal: Find how many evoked neuron pairs we detect at 1, 2, and 4 Hz")

# Filter for chip 25123ic first, then by frequency
chip_catalog = catalog.filter(chip='25123ic', baseline=False)
freq_1hz_recs = chip_catalog.filter(freq=1)
freq_2hz_recs = chip_catalog.filter(freq=2)
freq_4hz_recs = chip_catalog.filter(freq=4)

print("Step 1: Select recordings to analyze (from chip 25123ic)")
print(f"  • Available at 1 Hz: {len(freq_1hz_recs)} recordings")
print(f"  • Available at 2 Hz: {len(freq_2hz_recs)} recordings")
print(f"  • Available at 4 Hz: {len(freq_4hz_recs)} recordings")
print()

# Analyze one recording from each frequency
results_summary = {}

for freq_hz, freq_recs in [(1, freq_1hz_recs), (2, freq_2hz_recs), (4, freq_4hz_recs)]:
    if len(freq_recs) == 0:
        print(f"⚠️  No recordings found at {freq_hz} Hz, skipping...")
        continue

    # Get the first recording
    rec = freq_recs[0]

    print(f"\nAnalyzing {freq_hz} Hz: {rec.identifier}")

    # Calculate latencies (automatically loads from cache if available)
    evoked_pairs = rec.calculate_latencies(
        min_response_ratio=1.5,
        max_p_value=0.0001,
        baseline_window=(-100, 0),
        response_window=(0, 100),
        save_results=True,
        result_key='evoked_latencies',
        verbose=True
    )

    # Load spikes and stim_log for metadata
    spikes = rec.spikes
    stim_log = rec.stim_log

    print(f"  ✓ Loaded {len(spikes.train)} neurons, {len(stim_log)} stimulations")
    print(f"  ✓ Found {len(evoked_pairs)} evoked neuron-electrode pairs")

    # Store results for plotting
    results_summary[freq_hz] = {
        'recording': rec.identifier,
        'n_neurons': len(spikes.train),
        'n_stimulations': len(stim_log),
        'n_evoked_pairs': len(evoked_pairs),
        'evoked_pairs': evoked_pairs
    }

    # Show latency statistics if we found any
    if len(evoked_pairs) > 0:
        latencies = [d['onset_latency'] for d in evoked_pairs.values()]
        print(f"    • Latency range: {np.min(latencies):.1f} - {np.max(latencies):.1f} ms")
        print(f"    • Median latency: {np.median(latencies):.1f} ms")

    # Clear cache to free memory
    rec.clear_cache()


# =============================================================================
# STEP 5: Visualize Results Across Frequencies
# =============================================================================
print("="*70)
print("STEP 5: Visualizing Evoked Responses Across Frequencies")
print("="*70)

# Extract data for plotting
frequencies = sorted(results_summary.keys())
n_evoked = [results_summary[f]['n_evoked_pairs'] for f in frequencies]
n_neurons = [results_summary[f]['n_neurons'] for f in frequencies]

# Calculate evoked fraction
evoked_fractions = [n_evoked[i] / n_neurons[i] * 100 if n_neurons[i] > 0 else 0
                    for i in range(len(frequencies))]

# Create figure using Styler
styler = Styler(journal="draft")
fig, axes = styler.create_figure(nrows=1, ncols=2, size_preset='single')

# Plot 1: Number of evoked pairs
axes[0].bar(frequencies, n_evoked, color=styler.get_named_color('dark_blue'),
            alpha=0.7, edgecolor='black', width=0.6)
axes[0].set_xlabel('Stimulation Frequency (Hz)')
axes[0].set_ylabel('Number of Evoked Pairs')
axes[0].set_title('Evoked Neuron-Electrode Pairs by Frequency', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linewidth=0.5)
axes[0].set_xlim(0.5, max(frequencies) + 0.5)

# Add value labels on bars
for freq, count in zip(frequencies, n_evoked):
    axes[0].text(freq, count, str(count), ha='center', va='bottom')

# Plot 2: Evoked fraction (percentage)
axes[1].bar(frequencies, evoked_fractions, color=styler.get_named_color('orange'),
            alpha=0.7, edgecolor='black', width=0.6)
axes[1].set_xlabel('Stimulation Frequency (Hz)')
axes[1].set_ylabel('Evoked Response Rate (%)')
axes[1].set_title('Percentage of Neurons with Evoked Responses', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linewidth=0.5)
axes[1].set_xlim(0.5, max(frequencies) + 0.5)

# Add percentage labels on bars
for freq, pct in zip(frequencies, evoked_fractions):
    axes[1].text(freq, pct, f'{pct:.1f}%', ha='center', va='bottom')

# Show the plot using Styler
styler.finish_plot(show=True)

print("\n✓ Visualization complete!")
print("  Left plot: Raw count of evoked neuron-electrode pairs")
print("  Right plot: Percentage of neurons responding to stimulation")
print("\n💡 This helps identify if stimulation frequency affects responsiveness")


# =============================================================================
# Summary and Next Steps
# =============================================================================
print("="*70)
print("Tutorial Complete! 🎉")
print("="*70)

print("\nYou've learned:")
print("  ✓ How to load and explore a catalog")
print("  ✓ Filtering recordings by experimental conditions")
print("  ✓ Using filter suffixes (__, __gt, __contains, etc.)")
print("  ✓ Grouping recordings by variables")
print("  ✓ Practical workflow: Batch latency analysis across frequencies")
print("  ✓ Visualizing evoked responses across experimental conditions")

print("\nKey takeaways:")
print("  💡 Catalogs make it easy to organize and filter many recordings")
print("  💡 Combine filtering with analysis functions for batch processing")
print("  💡 Results are automatically cached - second run is instant!")
print("  💡 Always clear cache (rec.clear_cache()) after processing to manage memory")

print("\nNext steps:")
print("  • Try filtering by other conditions (chip, project, drug, etc.)")
print("  • Analyze more recordings per frequency for statistical power")
print("  • Run this tutorial again - notice how much faster it is with cached results!")
print("  • Save results using rec.results and rec.save_results()")
print("  • Advanced Tutorial 02: Advanced batch processing patterns")
print("="*70)
