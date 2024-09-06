---
sidebar_position: 1
---

# Introduction to RT-Sort

RT-Sort is a powerful tool for real-time spike detection and sorting with millisecond latencies. This user manual will guide you through the installation, usage, and advanced features of RT-Sort.

## Hardware Requirements

RT-Sort requires a GPU to be installed on the computer that runs the software. Here are the key requirements:

- Any NVIDIA GPU compatible with PyTorch and Torch-TensorRT
- In the RT-Sort manuscript, an NVIDIA RTX A5000 was used
- RT-Sort automatically adjusts its settings for different GPU specs
- Less powerful GPUs might yield larger computation times and detection latencies
- Recommended: At least 8GB of RAM (allows for online detections on ~50 sequences with 1020 electrodes)

## What's Next?

- [Installation Guide](installation)
- [Loading Detection Models](usage/load-detection-model)
- [Sequence Detection](usage/sequence-detection)
- [Real-time Application](usage/real-time-application)
- [Training Your Own Models](usage/training-models)
- [API Reference](api-reference)