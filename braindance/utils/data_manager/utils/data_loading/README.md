# Data Loading Utilities

This directory contains the core data loading and management layer for the BrainDance framework. These utilities are designed to provide a unified, abstract interface for interacting with diverse experimental data, managing the transition between remote (S3/NRP) and local environments, and ensuring the persistence of derived analytical results.

## The Data Manager Ecosystem

The files in this directory work together to form a tiered data access and persistence system.

---

### [catalog.py]
**Role: Metadata Orchestration & Search**
The `catalog.py` module serves as the entry point for the BrainDance data ecosystem. It is responsible for organizing potentially thousands of individual recording sessions into a queryable collection.
- **Ecosystem Function**: It provides the abstraction needed to perform broad metadata-driven analysis across multiple experiments. By wrapping experimental CSV logs into a `RecordingCatalog`, it allows researchers to filter, slice, and batch-process data based on biological or technical variables (e.g., genotype, stimulation frequency, or project name) without manually managing file paths.

### [recording.py]
**Role: Individual Session Abstraction**
The `recording.py` module defines the `Recording` class, which is the primary object for interacting with a single experimental session's data.
- **Ecosystem Function**: It acts as a "smart container" that hides the complexity of where data resides. Whether a session is stored on local disk or must be fetched from S3 (NRP), the `Recording` object provides a uniform API for accessing spikes, stimulus logs, and metadata. It handles lazy loading to minimize memory footprint and ensures that standard and legacy directory structures are resolved automatically, providing a consistent interface to the rest of the package.

### [data_context.py]
**Role: Stateful Result Persistence**
The `data_context.py` module provides the `DataContext` class, which manages the lifecycle of computed results associated with a recording.
- **Ecosystem Function**: It enables a "compute once, access anywhere" workflow. Instead of requiring users to manually manage file types and storage for their analysis results (like firing rates or connectivity matrices), `DataContext` provides a memory-like interface that automatically handles serialization (choosing between .npy, .csv, or .pkl as needed) and lazy loading. This ensures that analytical state is preserved across different sessions or jobs.

### [results_cache.py]
**Role: Distributed Job Caching**
The `results_cache.py` module is a specialized caching layer designed specifically for high-performance computing environments like the Pacific Research Platform (NRP).
- **Ecosystem Function**: In environments where jobs may be distributed across many nodes, `results_cache.py` ensures that expensive computations (e.g., binning tens of thousands of spikes) are shared. It implements a local-first strategy but is backed by S3, allowing subsequent jobs to pull pre-computed results instead of recalculating them. It uses a manifest system to track parameters, ensuring that cache hits are mathematically valid for the requested analysis.

### [s3_loader.py]
**Role: Infrastructure Transport Layer**
The `s3_loader.py` module is the low-level utility that handles the actual communication with S3-compatible storage.
- **Ecosystem Function**: It serves as the foundation for the remote data capabilities of the other modules. It abstracts the `boto3` and `smart_open` calls required to transfer large files reliably, particularly optimized for the NRP S3 endpoint. It provides the essential plumbing that allows `Recording` and `ResultsCache` to function in cloud-native or hybrid-local workflows.
