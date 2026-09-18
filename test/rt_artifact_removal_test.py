import numpy as np
import matplotlib.pyplot as plt
import time
import sys

try:
    # Try to import original implementation
    from braindance.core.artifact_removal import ArtifactRemoval, Timer
    from braindance.analysis.data_loader import load_data_maxwell
    import glob
    BRAINDANCE_AVAILABLE = True
except ImportError:
    print("Warning: braindance module not found. Running with synthetic data only.")
    BRAINDANCE_AVAILABLE = False
    class Timer:
        def __init__(self, name='Timer'):
            self.name = name
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val:
                raise exc_val
            else:
                print(f'{self.name} End time: {time.time() - self.start_time:.3f}s')

from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval

def generate_synthetic_data(length, n_channels=1, artifact_locations=None, artifact_amplitude=200):
    """Generate synthetic data with optional artifacts for testing."""
    # Generate clean signal with some oscillations
    t = np.arange(length)
    
    if n_channels == 1:
        clean_signal = np.sin(t/10) + np.sin(t/20) + np.sin(t/30)
        # Add Gaussian noise
        noisy_signal = clean_signal + np.random.normal(0, 0.5, length)
        
        # Create artifacts at specified locations
        artifacts = np.zeros(length)
        if artifact_locations is not None:
            for loc in artifact_locations:
                start, end = loc
                # Create a decaying artifact shape
                artifact_length = end - start
                decay = np.exp(-np.arange(artifact_length)/10)
                artifacts[start:end] = artifact_amplitude * decay
        
        # Combine signal and artifacts
        signal_with_artifacts = noisy_signal + artifacts
        
        return signal_with_artifacts, artifacts, noisy_signal
    else:
        # Multi-channel case
        clean_signals = np.zeros((n_channels, length))
        noisy_signals = np.zeros((n_channels, length))
        artifacts_array = np.zeros((n_channels, length))
        signal_with_artifacts = np.zeros((n_channels, length))
        
        for i in range(n_channels):
            # Add slightly different phase for each channel
            phase_shift = 2 * np.pi * i / n_channels
            clean_signals[i] = np.sin(t/10 + phase_shift) + np.sin(t/20 + phase_shift) + np.sin(t/30 + phase_shift)
            noisy_signals[i] = clean_signals[i] + np.random.normal(0, 0.5, length)
            
            # Add artifacts with random timing variations for each channel
            if artifact_locations is not None:
                for loc in artifact_locations:
                    start, end = loc
                    # Add some jitter to artifact timing for different channels
                    jitter = np.random.randint(-5, 6)
                    start_j = max(0, start + jitter)
                    end_j = min(length, end + jitter)
                    
                    # Create a decaying artifact shape
                    artifact_length = end_j - start_j
                    if artifact_length > 0:
                        decay = np.exp(-np.arange(artifact_length)/10)
                        # Vary amplitude slightly between channels
                        amp_factor = 0.8 + 0.4 * np.random.random()
                        artifacts_array[i, start_j:end_j] = artifact_amplitude * amp_factor * decay
            
            signal_with_artifacts[i] = noisy_signals[i] + artifacts_array[i]
        
        return signal_with_artifacts, artifacts_array, noisy_signals

# def benchmark_realtime_methods(data, N=60, nc_start=60, min_val=-200, max_val=200, n_workers=None):
#     """Benchmark the real-time (online) artifact removal methods."""
#     print("\nBenchmarking real-time methods...")
#     results = {}
    
#     n_channels = 1 if len(data.shape) == 1 else data.shape[0]
#     print(f"Processing {n_channels} channel(s)...")
    
#     # Dictionary of methods to test
#     methods = {
#         "OptimizedArtifactRemoval": OptimizedArtifactRemoval(
#             N=N, nc_start=nc_start, min_val=min_val, max_val=max_val)
#     }
    
#     if n_channels > 1:
#         methods["OptimizedArtifactRemoval"] = OptimizedMultiChannelArtifactRemoval(
#             n_channels=n_channels, N=N, nc_start=nc_start, min_val=min_val, max_val=max_val,
#             n_workers=n_workers, use_threading=True)
        
#         methods["OptimizedMultiChannelArtifactRemoval"] = OptimizedMultiChannelArtifactRemoval(
#             n_channels=n_channels, N=N, nc_start=nc_start, min_val=min_val, max_val=max_val)
    
#     if BRAINDANCE_AVAILABLE and n_channels == 1:
#         methods["OriginalArtifactRemoval"] = ArtifactRemoval(
#             N=N, nc_start=nc_start, min_val=min_val, max_val=max_val)
    
#     # Test each method
#     for method_name, remover in methods.items():
#         print(f"\nRunning {method_name}...")
        
#         if n_channels == 1:
#             # Single channel case
#             # Pre-process some samples to initialize
#             for i in range(100):
#                 remover.fit_step(data[0])
            
#             clean_data = np.zeros_like(data)
#             artifacts = np.zeros_like(data)
            
#             # Time the actual processing
#             with Timer(f"{method_name} processing") as timer:
#                 for i in range(len(data)):
#                     clean_val, artifact_val, _ = remover.fit_step(data[i])
                    
#                     if i >= N:  # Account for algorithm delay
#                         clean_data[i-N] = clean_val
#                         artifacts[i-N] = artifact_val
            
#             # Calculate processing rate
#             processing_rate = len(data) / timer.end_time
#             print(f"Processing rate: {processing_rate:.2f} samples/second")
#             print(f"Realtime factor: {processing_rate / 20000:.2f}x")  # Assuming 20 kHz sampling rate
#         else:
#             # Multi-channel case
#             # Pre-process some samples to initialize
#             for i in range(100):
#                 if i < data.shape[1]:
#                     if method_name.startswith("Optimized") or method_name.startswith("Vectorized"):
#                         remover.fit_step(data[:, i])
            
#             clean_data = np.zeros_like(data)
#             artifacts = np.zeros_like(data)
            
#             # Time the actual processing
#             with Timer(f"{method_name} processing") as timer:
#                 for i in range(data.shape[1]):
#                     if method_name.startswith("Optimized") or method_name.startswith("Vectorized"):
#                         clean_vals, artifact_vals, _ = remover.fit_step(data[:, i])
                    
#                     if i >= N:  # Account for algorithm delay
#                         clean_data[:, i-N] = clean_vals
#                         artifacts[:, i-N] = artifact_vals
            
#             # Calculate processing rate
#             processing_rate = data.shape[1] / timer.end_time
#             print(f"Processing rate: {processing_rate:.2f} samples/second")
#             print(f"Realtime factor: {processing_rate / 20000:.2f}x")  # Assuming 20 kHz sampling rate
#             print(f"Channels × samples per second: {processing_rate * n_channels:.2f}")
        
#         results[method_name] = {
#             "clean_data": clean_data,
#             "artifacts": artifacts
#         }
        
#         if hasattr(remover, "state_timers"):
#             print(f"State times for {method_name}:")
#             for state, time_spent in remover.state_timers.items():
#                 print(f"  {state}: {time_spent:.6f}s")
#             total_time = sum(remover.state_timers.values())
#             print(f"  Total time in states: {total_time:.6f}s")
    
#     return results

def compare_results(data, true_artifacts, results, title, plot_length=None, plot_channels=None):
    """Compare and plot results from different methods."""
    if plot_length is None:
        plot_length = len(data) if len(data.shape) == 1 else data.shape[1]
    
    n_channels = 1 if len(data.shape) == 1 else data.shape[0]
    
    if n_channels == 1:
        # Single channel case
        fig, axes = plt.subplots(len(results) + 1, 1, figsize=(15, 3 * (len(results) + 1)), sharex=True)
        
        # Plot original data
        axes[0].plot(data[:plot_length], 'k', label='Original signal')
        if true_artifacts is not None:
            axes[0].plot(true_artifacts[:plot_length], 'r--', label='True artifacts')
        axes[0].set_title('Original Signal with Artifacts')
        axes[0].legend()
        
        # Plot results from each method
        for i, (method_name, result) in enumerate(results.items(), 1):
            clean_data = result["clean_data"]
            artifacts = result["artifacts"]
            
            axes[i].plot(data[:plot_length], 'k', alpha=0.3, label='Original')
            axes[i].plot(clean_data[:plot_length], 'g', label='Cleaned')
            axes[i].plot(artifacts[:plot_length], 'r--', label='Estimated artifacts')
            axes[i].set_title(f'{method_name} Result')
            axes[i].legend()
    else:
        # Multi-channel case - plot a subset of channels
        if plot_channels is None:
            # Plot first 4 channels or all if fewer
            plot_channels = min(4, n_channels)
        
        fig, axes = plt.subplots(plot_channels, len(results) + 1, 
                               figsize=(5 * (len(results) + 1), 3 * plot_channels))
        
        # Handle single row case
        if plot_channels == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each channel
        for ch in range(plot_channels):
            # Plot original data
            axes[ch, 0].plot(data[ch, :plot_length], 'k', label='Original')
            if true_artifacts is not None:
                axes[ch, 0].plot(true_artifacts[ch, :plot_length], 'r--', label='True artifacts')
            axes[ch, 0].set_title(f'Ch {ch} - Original')
            axes[ch, 0].legend()
            
            # Plot results from each method
            for i, (method_name, result) in enumerate(results.items(), 1):
                clean_data = result["clean_data"]
                artifacts = result["artifacts"]
                
                axes[ch, i].plot(data[ch, :plot_length], 'k', alpha=0.3, label='Original')
                axes[ch, i].plot(clean_data[ch, :plot_length], 'g', label='Cleaned')
                axes[ch, i].plot(artifacts[ch, :plot_length], 'r--', label='Artifacts')
                axes[ch, i].set_title(f'Ch {ch} - {method_name}')
                if ch == 0:  # Only show legend for first row
                    axes[ch, i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()

def test_500_channels(data_length=20000, N=60, nc_start=60, artifact_locations=None):
    """Test the implementations with 500 channels."""
    print("\nTesting with 500 channels...")
    n_channels = 800
    
    # Generate synthetic data for 500 channels
    print("Generating synthetic data...")
    if artifact_locations is None:
        artifact_locations = [
            (1000, 1020), (3000, 3050), (7000, 7025), 
            (10000, 10040), (15000, 15030)
        ]

    print("Loading real neural data...")
    try:
        use_real_data = True
        # Use your data loading code here
        data_filepath = '/media/danser-lab/hippo/cartpole/24-04-18_butterfly/23138/exp1/exp1_cartpole_long_7'
        full_length = 20000 * 10
        data_loaded = load_data_maxwell(
            data_filepath, start=130*20000, length=data_length)
        data = data_loaded[:n_channels]
        true_artifacts = None  # We don't know the true artifacts in real data
    except Exception as e:
        print(f"Error loading real data: {e}")
        use_real_data = False

    if not use_real_data:
        data, true_artifacts, clean_signal = generate_synthetic_data(
            data_length, n_channels=n_channels, artifact_locations=artifact_locations, 
            artifact_amplitude=200)
        
    
    # Set parameters
    min_val = -200
    max_val = 200
    
    # Create instances to test
    removers = {
        "Linear_minibatch": LinearArtifactRemoval(
                                        n_channels=n_channels,
                                        N=60,
                                        nc_start=60,
                                        min_val=-100,
                                        max_val=100,
                                        batch_size=n_channels
                                        )
    }
    
    # Test each implementation
    results = {}
    for name, remover in removers.items():
        print(f"\nTesting {name}...")
        minibatch_size = 40
        
        # Pre-initialize with some data
        print("Initializing...")
        if "minibatch" in name:
            for i in range(1,100*minibatch_size,minibatch_size):
                remover.fit_step(data[:, i:i+minibatch_size])
        else:
            for i in range(100):
                remover.fit_step(data[:, i])
        
        # Time the actual processing
        print(f"Processing {data_length} samples for {n_channels} channels...")
        
        clean_data = np.zeros_like(data)
        artifacts = np.zeros_like(data)

        if hasattr(remover, "timing"):
            remover.timing = {'total': 0, 'processing': 0}

        if "minibatch" in name:
            
            print("Running with minibatch...")
            with Timer(f"{name} processing") as timer:
                for i in range(N,data.shape[1] - N,minibatch_size):
                    # print(data[:, i:i+minibatch_size].shape)
                    clean_vals, artifact_vals, _ = remover.fit_step(data[:, i:i+minibatch_size])
                    
                    if i >= N:  # Account for algorithm delay
                        clean_data[:, i-N:i-N+minibatch_size] = clean_vals
                        artifacts[:, i-N:i-N+minibatch_size] = artifact_vals
        else:
            with Timer(f"{name} processing") as timer:
                for i in range(data.shape[1]):
                    clean_vals, artifact_vals, _ = remover.fit_step(data[:, i])
                    
                    if i >= N:  # Account for algorithm delay
                        clean_data[:, i-N] = clean_vals
                        artifacts[:, i-N] = artifact_vals
        
        # Calculate processing rate
        processing_rate = data.shape[1] / timer.end_time
        print(f"Processing rate: {processing_rate:.2f} samples/second")
        print(f"Realtime factor: {processing_rate / 20000:.2f}x")  # Assuming 20 kHz sampling rate
        print(f"Channels × samples per second: {processing_rate * n_channels:.2f}")

        if hasattr(remover, "state_timers"):
            print(f"State times for {name}:")
            for state, time_spent in remover.state_timers.items():
                print(f"  {state}: {time_spent:.6f}s")
            total_time = sum(remover.state_timers.values())
            print(f"  Total time in states: {total_time:.6f}s")
        if hasattr(remover, "timing"):
            print(f"Timing breakdown for {name}:")
            for timing_type, time_spent in remover.timing.items():
                print(f"  {timing_type}: {time_spent:.6f}s")
        
        results[name] = {
            "clean_data": clean_data,
            "artifacts": artifacts,
            "time": timer.end_time,
            "rate": processing_rate
        }
    
    # Create a summary table
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Method':<40} | {'Time (s)':<10} | {'Rate (sps)':<15} | {'Realtime Factor':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        time_val = result["time"]
        rate = result["rate"]
        rt_factor = rate / 20000  # Assuming 20 kHz sampling rate
        print(f"{name:<40} | {time_val:<10.3f} | {rate:<15.2f} | {rt_factor:<15.2f}x")
    
    # Plot a subset of the results
    plot_channels = 4  # Number of channels to plot
    plot_length = data_length  # Number of samples to plot
    
    # Plot comparison of results
    compare_results(
        data, true_artifacts, 
        {name: {"clean_data": results[name]["clean_data"], 
                "artifacts": results[name]["artifacts"]} 
         for name in results},
        "500 Channel Artifact Removal Comparison", 
        plot_length=plot_length, plot_channels=plot_channels
    )
    
    return results

def main():
    """Run the retained linear artifact-removal benchmark."""
    data_length = 20000 * 10 if len(sys.argv) > 1 and sys.argv[1] == "500" else 30000
    test_500_channels(data_length=data_length)


if __name__ == "__main__":
    main()
