from braindance.utils.data_manager import load_catalog
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from IPython import embed

# Configuration
proj_id = "24-04-18_butterfly"
chip_id = "p001237"
exp_filter = "exp2"

# Load catalog and filter
print(f"Loading catalog for {proj_id}...")
catalog = load_catalog()
filtered_cat = catalog.filter(
    proj=proj_id, chip=chip_id, experiment__contains=exp_filter
)

# Report data availability
# In RecordingCatalog, we can check availability by looking at the dataframe
# DataManager's Recording objects are lazy, but we can check if files exist
# However, the original script used a 'data_available' column if present in the CSV
if "data_available" in filtered_cat.columns:
    true_count = sum(filtered_cat.df["data_available"])
    false_count = len(filtered_cat) - true_count
    print(
        f"Data available (from catalog): {true_count}, Data not available: {false_count}"
    )

    if false_count > 0:
        not_available_df = filtered_cat.df[~filtered_cat.df["data_available"]]
        print("Data not available for the following experiments:")
        print(not_available_df[["proj", "chip", "experiment"]])

if len(filtered_cat) == 0:
    print(
        f"No recordings found for {proj_id}/{chip_id} with experiment containing {exp_filter}"
    )
    exit()

# Get the first recording
rec = filtered_cat[0]
print(f"Loaded recording: {rec.identifier}")

# Access data
sd = rec.spikes
log = rec.stim_log

if log is None or len(log) == 0:
    print("No stimulation log available for this recording.")
    exit()

print(log[["time"]].head())
if "time_mod" in log.columns:
    print(f"Mean offset: {(log['time'] - log['time_mod']).mean():.6f} s")

# Extract spike indices and times
idces, times = sd.idces_times()
times_sec = times / 1000

# Extract artifact times from metadata
artifact_times = []
if hasattr(sd, "metadata") and "artifact_times" in sd.metadata:
    artifact_times = sd.metadata["artifact_times"] / 1000  # Convert to seconds
    print(f"Found {len(artifact_times)} artifact times")
else:
    print("No artifact times found in metadata")

# Create a raster plot
plt.figure(figsize=(20, 8))

# Use different marker and size for better visibility
plt.scatter(times_sec, idces, s=1, color="black", alpha=0.8, marker="|")

# Add artifact times as vertical lines
if len(artifact_times) > 0:
    for i, artifact_time in enumerate(artifact_times):
        plt.axvline(
            artifact_time,
            color="orange",
            linestyle=":",
            alpha=0.75,
            linewidth=1.5,
            label="Artifact Times" if i == 0 else "",
        )

# Add stimulation times from log as vertical lines
stim_times = log["time"].values
for i, stim_time in enumerate(stim_times):
    plt.axvline(
        stim_time,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="Stim Times" if i == 0 else "",
    )

plt.xlabel("Time (s)")
plt.ylabel("Neuron")
plt.title(f"Raster: {rec.identifier} with Artifact and Stimulation Times")
plt.grid(True, alpha=0.3)
plt.legend()

# Set reasonable axis limits to see the data
plt.xlim(200, 400)
if len(idces) > 0:
    plt.ylim(0, max(idces))

plt.tight_layout()
plt.show()

# Print some stats to verify
print(f"Time range: {min(times_sec):.2f} - {max(times_sec):.2f} seconds")
print(f"Neuron indices range: {min(idces)} - {max(idces)}")
print(f"Total spikes: {len(times_sec)}")
print(f"Number of artifact times: {len(artifact_times)}")
print(f"Number of stim times: {len(stim_times)}")

# Check alignment between artifact and stim times
if len(artifact_times) > 0 and len(stim_times) > 0:
    print(f"First few artifact times: {artifact_times[:5]}")
    print(f"First few stim times: {stim_times[:5]}")

    # Find closest matches
    distances = cdist(artifact_times.reshape(-1, 1), stim_times.reshape(-1, 1))
    min_distances = np.min(distances, axis=1)

    print(
        f"Mean time difference between artifact and closest stim: {np.mean(min_distances):.4f} seconds"
    )
    print(f"Max time difference: {np.max(min_distances):.4f} seconds")

# Optional: drop into interactive shell
# embed()
