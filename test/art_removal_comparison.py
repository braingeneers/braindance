import numpy as np
import matplotlib.pyplot as plt
import time

try:
    # Try to import original implementation
    from braindance.core.artifact_removal import ArtifactRemoval
    BRAINDANCE_AVAILABLE = True
except ImportError:
    print("Warning: original braindance module not found.")
    BRAINDANCE_AVAILABLE = False

def generate_test_data(length=20000, artifact_amplitude=200):
    """Generate synthetic data with artifacts for testing."""
    t = np.arange(length)
    
    # Generate clean signal with oscillations
    clean_signal = np.sin(t/100) * 10 + np.sin(t/50) * 5 + 400  # Base level around 400
    
    # Add Gaussian noise
    noisy_signal = clean_signal + np.random.normal(0, 2, length)
    
    # Add artifacts - large spikes
    artifacts = np.zeros(length)
    artifact_locations = [2000, 5000, 8000, 11000, 14000, 17000]
    
    for loc in artifact_locations:
        # Create spike artifacts
        artifacts[loc:loc+20] = -artifact_amplitude * np.exp(-np.arange(20)/5)
    
    # Combine signal and artifacts
    signal_with_artifacts = noisy_signal + artifacts
    
    return signal_with_artifacts, artifacts, noisy_signal

def test_implementations():
    """Test and compare different artifact removal implementations."""
    # Generate test data
    print("Generating test data...")
    data, true_artifacts, clean_signal = generate_test_data()
    
    # Parameters
    N = 60
    nc_start = 60
    min_val = -50
    max_val = 50
    
    # Create implementations to test
    implementations = {}
    
    if BRAINDANCE_AVAILABLE:
        implementations["Original"] = ArtifactRemoval(N=N, nc_start=nc_start, min_val=min_val, max_val=max_val)
    
    # Process data with each implementation
    results = {}
    for name, impl in implementations.items():
        print(f"Processing with {name} implementation...")
        
        # Initialize
        for i in range(nc_start):
            impl.fit_step(data[i])
        
        # Process data
        clean_data = np.zeros_like(data)
        artifacts = np.zeros_like(data)
        
        start_time = time.time()
        for i in range(len(data)):
            clean_val, artifact_val, _ = impl.fit_step(data[i])
            
            if i >= N:  # Account for algorithm delay
                clean_data[i-N] = clean_val
                artifacts[i-N] = artifact_val
        
        end_time = time.time()
        process_time = end_time - start_time
        
        results[name] = {
            "clean_data": clean_data,
            "artifacts": artifacts,
            "time": process_time
        }
        
        print(f"  Processing time: {process_time:.3f}s")
    
    # Plot results
    plot_results(data, true_artifacts, clean_signal, results)
    
    return results

def plot_results(data, true_artifacts, clean_signal, results):
    """Plot comparison of results from different implementations."""
    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(15, 12), sharex=True)
    
    # Plot original data
    axes[0].plot(data, 'k', label='Original signal')
    axes[0].set_title('Original Signal with Artifacts')
    axes[0].legend()
    
    # Plot results from each implementation
    for i, (name, result) in enumerate(results.items(), 1):
        clean_data = result["clean_data"]
        artifacts = result["artifacts"]
        
        axes[i].plot(data, 'k', alpha=0.3, label='Original')
        axes[i].plot(clean_data, 'g', label='Cleaned')
        axes[i].plot(artifacts, 'r--', label='Estimated artifacts')
        axes[i].set_title(f'{name} Implementation - Time: {result["time"]:.3f}s')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('artifact_removal_comparison.png')
    plt.show()

if __name__ == "__main__":
    print("Testing single-channel implementation...")
    test_implementations()
