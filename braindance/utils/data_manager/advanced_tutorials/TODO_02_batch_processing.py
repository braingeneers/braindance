"""
BrainDance Data Manager - Advanced Tutorial 02: Batch Processing Multiple Recordings

This tutorial demonstrates efficient batch processing workflows for analyzing many
recordings at once. You'll learn practical patterns for memory management, caching,
and parallel processing while analyzing real experimental data.

Learning objectives:
  1. Basic batch iteration patterns
  2. Filtering and grouping recordings by experimental conditions
  3. Batch analysis workflows (latency and burst detection across conditions)
  4. Visualization of batch results
  5. Memory management and caching strategies
  6. Parallel processing for speedup

Prerequisites:
  - Completed Tutorial 01 (quick_start)
  - Completed Advanced Tutorial 01 (working_with_catalog)
  - Understand how to analyze individual recordings

Practical scenario:
  In this tutorial, we'll analyze chip 25123ic from the busybees project, which has
  ~100 recordings across different stimulation frequencies. This is a realistic
  batch processing scenario where we need to compare network properties across
  experimental conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from braindance.utils.data_manager import (
    load_catalog,
    BurstDetector,
    Styler
)


# =============================================================================
# Setup: Configure Paths and Load Catalog
# =============================================================================

# Set up directory for saving figures
save_dir = Path('/Volumes/hunter_ssd/busy_bee/test_plots/batch_tutorial')
save_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("BATCH PROCESSING TUTORIAL: Analyzing Chip 25123ic")
print("="*70)
print(f"\nFigures will be saved to: {save_dir}")
print()


# =============================================================================
# 1. Load and Explore Catalog for Our Target Chip
# =============================================================================

print("="*70)
print("STEP 1: Loading and Filtering Catalog")
print("="*70)

# Load the full catalog
catalog = load_catalog()
print(f"\n✓ Loaded catalog with {len(catalog)} total recordings")

# Filter to our target chip
chip_catalog = catalog.filter(chip='25123ic')
print(f"✓ Filtered to chip 25123ic: {len(chip_catalog)} recordings")

# Explore the metadata
print("\nExperimental conditions in this chip:")
freq_groups = chip_catalog.group_by('freq')
for freq in sorted(freq_groups.keys())[:10]:  # Show first 10 frequencies
    count = len(freq_groups[freq])
    print(f"  • {freq:5.1f} Hz: {count:3d} recordings")

if len(freq_groups) > 10:
    print(f"  ... and {len(freq_groups) - 10} more frequencies")

# Count baseline vs stimulated
baseline_count = len(chip_catalog.filter(freq=0))
stim_count = len(chip_catalog.filter(freq__gt=0))
print(f"\nRecording types:")
print(f"  • Spontaneous (freq=0): {baseline_count} recordings")
print(f"  • Stimulated (freq>0): {stim_count} recordings")
print()


# =============================================================================
# 2. Basic Batch Iteration: Computing Firing Rates
# =============================================================================

print("="*70)
print("STEP 2: Basic Batch Iteration Pattern")
print("="*70)
print("\nComputing firing rates for first 5 recordings...")
print("-"*70)

# Take a small batch for demonstration
sample_recordings = chip_catalog[:5]

# Initialize results storage
firing_rate_results = []

for i, rec in enumerate(sample_recordings):
    # Load spike data
    spikes = rec.spikes
    
    # Compute firing rates
    firing_rates = spikes.rates(unit="Hz")
    
    # Extract metadata and results
    result = {
        'experiment': rec.experiment,
        'freq': rec.freq,
        'n_neurons': len(spikes.train),
        'duration_sec': spikes.length / 1000,
        'mean_fr': np.mean(firing_rates),
        'median_fr': np.median(firing_rates),
        'active_neurons': np.sum(firing_rates > 0.1)
    }
    firing_rate_results.append(result)
    
    print(f"  {i+1}. {rec.experiment}:")
    print(f"     {result['n_neurons']} neurons, "
          f"{result['mean_fr']:.2f} Hz mean FR, "
          f"{result['active_neurons']} active")
    
    # CRITICAL: Clear cache to free memory
    rec.clear_cache()

# Convert to DataFrame for easy viewing
firing_df = pd.DataFrame(firing_rate_results)
print("\n✓ Batch processing complete!")
print("\nSummary table:")
print(firing_df.to_string(index=False))
print("\n💡 Note: rec.clear_cache() after each recording prevents memory issues")
print()


# =============================================================================
# 3. Filtering and Grouping by Experimental Condition
# =============================================================================

print("="*70)
print("STEP 3: Filtering and Grouping Recordings")
print("="*70)

# Select specific frequencies of interest
frequencies_of_interest = [0, 1, 2, 4, 8]

print(f"\nGrouping recordings by frequency: {frequencies_of_interest}")
print("-"*70)

frequency_groups = {}
for freq in frequencies_of_interest:
    filtered = chip_catalog.filter(freq=freq)
    frequency_groups[freq] = filtered
    print(f"  • {freq} Hz: {len(filtered)} recordings")

print("\n💡 Use catalog.filter() to select specific experimental conditions")
print("💡 Use catalog.group_by() to organize recordings by a metadata field")
print()


# =============================================================================
# 4. Batch Latency Analysis Across Frequencies
# =============================================================================

print("="*70)
print("STEP 4: Batch Latency Analysis Across Stimulation Frequencies")
print("="*70)
print("\nResearch question: How does stimulation frequency affect evoked responses?")
print("-"*70)

# We'll analyze a subset of recordings per frequency
recordings_per_freq = 3

latency_results = []

for freq in [1, 2, 4, 8]:  # Analyze stimulated conditions
    if freq not in frequency_groups or len(frequency_groups[freq]) == 0:
        print(f"\n⚠️  No recordings at {freq} Hz, skipping...")
        continue
    
    print(f"\nAnalyzing {freq} Hz stimulation ({len(frequency_groups[freq])} recordings available):")
    
    # Take subset of recordings
    freq_recordings = frequency_groups[freq][:recordings_per_freq]
    
    freq_evoked_counts = []
    freq_latencies = []
    
    for rec in freq_recordings:
        # Calculate latencies with caching enabled
        evoked_pairs = rec.calculate_latencies(
            min_response_ratio=1.5,
            max_p_value=0.0001,
            baseline_window=(-100, 0),
            response_window=(0, 100),
            save_results=True,
            result_key='evoked_latencies',
            verbose=False  # Suppress individual recording output
        )
        
        n_evoked = len(evoked_pairs)
        freq_evoked_counts.append(n_evoked)
        
        # Collect latencies
        if n_evoked > 0:
            latencies = [d['onset_latency'] for d in evoked_pairs.values()]
            freq_latencies.extend(latencies)
        
        print(f"  • {rec.experiment}: {n_evoked} evoked pairs")
        
        # Clear memory
        rec.clear_cache()
    
    # Store aggregate results
    latency_results.append({
        'freq': freq,
        'n_recordings': len(freq_recordings),
        'mean_evoked_pairs': np.mean(freq_evoked_counts),
        'total_evoked_pairs': sum(freq_evoked_counts),
        'latencies': freq_latencies,
        'mean_latency': np.mean(freq_latencies) if freq_latencies else np.nan,
        'median_latency': np.median(freq_latencies) if freq_latencies else np.nan
    })
    
    if freq_latencies:
        print(f"  ✓ Total evoked pairs: {sum(freq_evoked_counts)}")
        print(f"  ✓ Mean latency: {np.mean(freq_latencies):.1f} ms")
        print(f"  ✓ Latency range: {np.min(freq_latencies):.1f} - {np.max(freq_latencies):.1f} ms")

print("\n✓ Batch latency analysis complete!")
print("\n💡 calculate_latencies() automatically caches results - rerunning is instant!")
print()


# =============================================================================
# 4.5 Visualize Batch Latency Results
# =============================================================================

print("="*70)
print("VISUALIZATION: Latency Analysis Results")
print("="*70)

# Initialize Styler for publication-quality plots
styler = Styler()

# Create two-panel figure
fig, axes = styler.create_figure(nrows=1, ncols=2, size_preset='single')

# Panel 1: Number of evoked pairs per frequency
frequencies = [r['freq'] for r in latency_results]
evoked_counts = [r['total_evoked_pairs'] for r in latency_results]

axes[0].bar(frequencies, evoked_counts, 
            color=styler.get_named_color('dark_blue'),
            alpha=0.7, edgecolor='black', width=0.6)
axes[0].set_xlabel('Stimulation Frequency (Hz)')
axes[0].set_ylabel('Total Evoked Pairs')
axes[0].set_title('Evoked Responses by Frequency', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for freq, count in zip(frequencies, evoked_counts):
    axes[0].text(freq, count, str(count), ha='center', va='bottom', fontsize=8)

# Panel 2: Latency distributions using box plot
latency_data = [r['latencies'] for r in latency_results if r['latencies']]
latency_labels = [f"{r['freq']} Hz" for r in latency_results if r['latencies']]

bp = axes[1].boxplot(latency_data, tick_labels=latency_labels,
                      patch_artist=True, widths=0.6)

# Style the boxplot
for patch in bp['boxes']:
    patch.set_facecolor(styler.get_named_color('orange'))
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')

axes[1].set_xlabel('Stimulation Frequency')
axes[1].set_ylabel('Onset Latency (ms)')
axes[1].set_title('Latency Distributions', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Save figure
styler.finish_plot(save_plots=True, save_dir=str(save_dir), name='batch_latency_analysis')
print(f"\n✓ Saved latency analysis plots to {save_dir}/batch_latency_analysis.png/.svg")
plt.close()


# =============================================================================
# 5. Batch Burst Analysis on Spontaneous Recordings
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Batch Burst Analysis on Spontaneous Activity")
print("="*70)
print("\nAnalyzing burst properties across spontaneous recordings (freq=0)...")
print("-"*70)

# Get spontaneous recordings
spontaneous_recs = frequency_groups.get(0, [])

if len(spontaneous_recs) > 0:
    # Analyze subset
    n_to_analyze = min(5, len(spontaneous_recs))
    sample_spontaneous = spontaneous_recs[:n_to_analyze]
    
    burst_results = []
    
    for i, rec in enumerate(sample_spontaneous):
        print(f"\n  {i+1}. {rec.experiment}:")
        
        # Load spikes
        spikes = rec.spikes
        
        # Detect bursts with caching
        detector = BurstDetector(
            spikes,
            bin_size=1.0,
            smoothing_window=50,
            burst_detection_params={
                'baseline_percentile': 25,
                'peak_threshold_factor': 2.5,
                'peak_distance': 200,
                'peak_prominence': 0.5
            }
        )
        
        results = detector.detect_bursts(cache=rec.cache)
        summary = detector.get_summary_metrics()
        
        burst_results.append({
            'experiment': rec.experiment,
            'n_bursts': summary['n_bursts'],
            'burst_freq_hz': summary.get('burst_frequency', 0),
            'mean_width_ms': summary.get('mean_burst_width', 0),
            'mean_amplitude': summary.get('mean_burst_amplitude', 0)
        })
        
        print(f"     • Bursts detected: {summary['n_bursts']}")
        print(f"     • Burst frequency: {summary.get('burst_frequency', 0):.3f} Hz")
        print(f"     • Mean width: {summary.get('mean_burst_width', 0):.1f} ms")
        
        rec.clear_cache()
    
    # Summary statistics
    burst_df = pd.DataFrame(burst_results)
    print("\n" + "="*70)
    print("Burst Analysis Summary:")
    print("="*70)
    print(burst_df.to_string(index=False))
    
    print(f"\nAggregate statistics:")
    print(f"  • Mean bursts per recording: {burst_df['n_bursts'].mean():.1f}")
    print(f"  • Mean burst frequency: {burst_df['burst_freq_hz'].mean():.4f} Hz")
    print(f"  • Mean burst width: {burst_df['mean_width_ms'].mean():.1f} ms")
    
else:
    print("  ⚠️  No spontaneous recordings available")

print("\n💡 BurstDetector automatically caches results for fast reanalysis")
print()


# =============================================================================
# 6. Advanced Pattern: Parallel Processing
# =============================================================================

print("="*70)
print("STEP 6: Parallel Processing for Speedup")
print("="*70)
print("\nWhen analyzing many recordings, parallel processing can speed things up.")
print("The catalog.apply() method distributes work across CPU cores.")
print("-"*70)

# Define a simple analysis function
def compute_mean_firing_rate(rec):
    """Compute mean firing rate for one recording."""
    spikes = rec.spikes
    firing_rates = spikes.rates(unit="Hz")
    mean_fr = np.mean(firing_rates)
    rec.clear_cache()
    return mean_fr

# Process a batch in parallel
print("\nProcessing 10 recordings in parallel with 4 workers...")
batch_for_parallel = chip_catalog[:10]

parallel_results = batch_for_parallel.apply(
    compute_mean_firing_rate,
    max_workers=4
)

print(f"✓ Processed {len(parallel_results)} recordings")
print(f"  Mean firing rates: {[f'{x:.2f}' for x in parallel_results[:5]]}... (showing first 5)")

print("\n💡 Use catalog.apply() with max_workers > 1 for CPU-intensive analyses")
print("💡 Benefit increases with more recordings and longer computations")
print()


# =============================================================================
# 7. Caching Strategy: First Run vs Second Run
# =============================================================================

print("="*70)
print("STEP 7: Understanding Caching for Efficient Re-Analysis")
print("="*70)

print("\nCaching saves expensive computation results for reuse.")
print("This tutorial has been using caching throughout!")
print("-"*70)

# Demonstrate caching with a single recording
demo_rec = chip_catalog.filter(freq=4)[0]

print(f"\nDemo recording: {demo_rec.identifier}")
print("\nFirst run: Computing latencies...")

import time
start = time.time()
evoked = demo_rec.calculate_latencies(
    min_response_ratio=1.5,
    max_p_value=0.0001,
    save_results=True,
    result_key='demo_latencies',
    verbose=False
)
elapsed_first = time.time() - start

print(f"  ✓ Found {len(evoked)} evoked pairs in {elapsed_first:.2f} seconds")

# Clear the in-memory cache but keep saved results
demo_rec.clear_cache()

print("\nSecond run: Loading from cache...")
start = time.time()
evoked_cached = demo_rec.calculate_latencies(
    min_response_ratio=1.5,
    max_p_value=0.0001,
    save_results=True,
    result_key='demo_latencies',
    verbose=False
)
elapsed_second = time.time() - start

print(f"  ✓ Loaded {len(evoked_cached)} evoked pairs in {elapsed_second:.2f} seconds")
print(f"  ⚡ Speedup: {elapsed_first / elapsed_second:.1f}x faster!")

print("\n💡 Results are stored in: rec.results.<key>")
print("💡 Call rec.save_results() to save for future sessions")
print("💡 Set save_results=True in analysis functions to auto-save")
demo_rec.clear_cache()
print()


# =============================================================================
# 8. Memory Management Best Practices
# =============================================================================

print("="*70)
print("STEP 8: Memory Management Best Practices")
print("="*70)

print("""
When processing many recordings, memory management is critical:

1. **Always call rec.clear_cache() after processing each recording**
   - Frees spike data from RAM immediately
   - Prevents out-of-memory errors with large batches

2. **Process recordings in batches**
   - Don't try to load 100 recordings at once
   - Process 10-20 at a time, aggregate results

3. **Use caching for expensive computations**
   - Latency analysis: use rec.calculate_latencies(save_results=True)
   - Burst detection: pass cache=rec.cache to BurstDetector
   - Custom results: rec.results.my_metric = value; rec.save_results()

4. **Monitor memory usage**
   - Each recording's spike data: ~10-50 MB
   - Full burst detection results: ~5-20 MB
   - Plan batch sizes based on available RAM

5. **Use parallel processing wisely**
   - Good for: independent analyses on many recordings
   - Each worker needs its own memory copy
   - Reduce max_workers if hitting memory limits

6. **Clear results when recomputing**
   - To invalidate cache: delete specific keys from rec.results
   - Or use force recomputation parameters where available
""")


# =============================================================================
# 9. Complete End-to-End Workflow
# =============================================================================

print("="*70)
print("STEP 9: Complete Analysis Pipeline")
print("="*70)
print("\nResearch Question: How does stimulation frequency affect network excitability?")
print("-"*70)

print("""
A publication-ready workflow:

1. Define experimental conditions (frequencies)
2. For each condition:
   a. Filter recordings
   b. Compute metrics (latencies, burst stats, firing rates)
   c. Aggregate results
3. Compare conditions statistically
4. Create multi-panel publication figure
5. Export results to CSV

This tutorial has walked you through steps 1-3!
For step 4, you can combine multiple visualizations using Styler.
For step 5, use pandas to export:
""")

# Example: Export latency results to CSV
latency_export = pd.DataFrame([
    {
        'frequency_hz': r['freq'],
        'n_recordings': r['n_recordings'],
        'total_evoked_pairs': r['total_evoked_pairs'],
        'mean_latency_ms': r['mean_latency']
    }
    for r in latency_results
])

export_path = save_dir / 'batch_latency_results.csv'
latency_export.to_csv(export_path, index=False)
print(f"\n✓ Exported results to: {export_path}")
print("\nExported data:")
print(latency_export.to_string(index=False))


# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("Tutorial Complete! 🎉")
print("="*70)

print("""
You've learned:
  ✓ How to filter and iterate through recordings in batches
  ✓ Memory management with rec.clear_cache()
  ✓ Batch latency analysis across experimental conditions
  ✓ Batch burst detection on spontaneous recordings
  ✓ Visualization of batch results using Styler
  ✓ Parallel processing with catalog.apply()
  ✓ Caching strategies for efficient re-analysis
  ✓ Complete workflows from data to publication

Key patterns:
  💡 Filter → Iterate → Analyze → Aggregate → Visualize
  💡 Always clear_cache() after each recording
  💡 Use save_results=True for expensive analyses
  💡 Build DataFrames for easy aggregation and export

Next steps:
  • Try this workflow on your own data
  • Add statistical comparisons between conditions
  • Create multi-panel figures combining different analyses
  • Explore other metadata fields for grouping (chip, drug, etc.)
  • Scale up to analyze entire datasets!
""")

print("="*70)
