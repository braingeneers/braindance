# BrainDance Data Manager - Functions Cheat Sheet

This cheat sheet provides a quick reference for the core functions and classes in the BrainDance data manager.

---

## 1. Loading & Data Discovery

| Function / Method | Description | Example |
| :--- | :--- | :--- |
| `load_recording(proj, chip, exp)` | Main entry point to load a single recording. | `rec = load_recording('25-02-25_busybees', '25123ic', 'exp1')` |
| `RecordingCatalog(df)` | Create a catalog from a DataFrame for batch processing. | `cat = RecordingCatalog(df)` |
| `catalog.filter(**kwargs)` | Django-style filtering of recordings. | `cat.filter(chip='25123ic', freq__gt=0)` |
| `catalog.apply(func)` | Apply a function to every recording in a catalog. | `results = cat.apply(lambda r: r.detect_bursts())` |
| `rec.spikes` | [Lazy] Access spike data (returns `SpikeData` object). | `spikes = rec.spikes` |
| `rec.stim_log` | [Lazy] Access stimulation log (returns `pd.DataFrame`). | `log = rec.stim_log` |
| `rec.mapping` | [Lazy] Access electrode mapping (returns `Mapping` object). | `mapping = rec.mapping` |
| `rec.spike_locations` | [Lazy] Access neuron spatial coordinates (returns list of (x,y) arrays). | `locs = rec.spike_locations` |
| `rec.info()` | Get a summary of recording metadata and loaded data. | `print(rec.info())` |

---

## 2. Spontaneous Activity (Burst Detection)

| Method / Attribute | Description | Example |
| :--- | :--- | :--- |
| `rec.detect_bursts()` | Detect synchronized network-level bursts. | `results = rec.detect_bursts(bin_size=1.0, smoothing_window=50)` |
| `results.n_bursts` | Total number of detected bursts. | `print(results.n_bursts)` |
| `results.bic_matrix` | Burst Involvement Coefficient per neuron. | `bic = results.bic_matrix` |
| `results.backbone_classification` | Identify "rigid" vs "non-rigid" neurons. | `rigid = results.backbone_classification['rigid']` |
| `results.burst_frequency` | Mean bursts per second (Hz). | `freq = results.burst_frequency` |

---

## 3. Evoked Activity (Latency & PSTH)

| Method / Attribute | Description | Example |
| :--- | :--- | :--- |
| `rec.calculate_latencies()` | Detect and validate stimulus-evoked responses. | `evoked = rec.calculate_latencies(min_response_ratio=1.5, max_p_value=0.001)` |
| `rec.group_stimulations_by_electrode()` | Group stimulation times by electrode ID. | `groups = rec.group_stimulations_by_electrode()` |
| `compute_psth_vectorized(...)` | Fast Peri-Stimulus Time Histogram calculation. | `psth, t = compute_psth_vectorized(spikes, stim_times)` |

---

## 4. Spatial Analysis

| Method / Attribute | Description | Example |
| :--- | :--- | :--- |
| `rec.mapping.get_positions()` | Get (x, y) positions for electrodes or channels. | `pos = rec.mapping.get_positions(electrodes=[123])` |
| `rec.mapping.get_channels()` | Convert electrode IDs to channel numbers. | `ch = rec.mapping.get_channels(electrodes=[123])` |
| `rec.mapping.get_electrodes()` | Convert channel numbers to electrode IDs. | `el = rec.mapping.get_electrodes(channels=[5])` |
| `rec.pl.spatial_latency()` | Plot spatial distribution of onset latencies. | `rec.pl.spatial_latency(electrode_id=123, latency_results=results)` |

---

## 5. Connectivity & Network Analysis

| Method / Attribute | Description | Example |
| :--- | :--- | :--- |
| `rec.pl.sttc_matrix()` | Plot Spike Time Tiling Coefficient correlation matrix. | `rec.pl.sttc_matrix(delt=20.0)` |
| `plot_connectivity_matrix(...)` | General connectivity visualization. | `plot_connectivity_matrix(sttc_matrix)` |

---

## 6. Visualization (Plotting)

Access all plotting via the `rec.pl` accessor.

| Method | Description | Example |
| :--- | :--- | :--- |
| `rec.pl.raster_with_pop()` | Raster plot with population activity overlay. | `rec.pl.raster_with_pop(time_window=(0, 60))` |
| `rec.pl.firing_rate_hist()` | Histogram of firing rates across all neurons. | `rec.pl.firing_rate_hist(log_scale=True)` |
| `rec.pl.evoked_raster()` | Trial-by-trial raster aligned to stimulation. | `rec.pl.evoked_raster(electrode_id=1, neuron_idx=0)` |
| `rec.pl.evoked_psth()` | Evoked PSTH aligned to stimulation. | `rec.pl.evoked_psth(electrode_id=1, neuron_idx=0)` |
| `rec.pl.spatial_latency()` | Spatial distribution of onset latencies. | `rec.pl.spatial_latency(electrode_id=1, latency_results=res)` |
| `rec.pl.sttc_matrix()` | Spike Time Tiling Coefficient correlation matrix. | `rec.pl.sttc_matrix(delt=20.0)` |

> [!TIP]
> Use `rec.pl.styler = Styler(size_preset='single')` to change global styling.

---

## 7. Caching & Results Management

| Method | Description | Example |
| :--- | :--- | :--- |
| `rec.save()` | Save all results/derived data to disk. | `rec.save()` |
| `rec.cache` | Access the `ResultsCache` for manual operations. | `rec.cache.sync_all_to_s3()` |
| `rec.clear_cache()` | Clear in-memory cached properties (for memory mgmt). | `rec.clear_cache()` |
