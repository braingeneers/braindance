"""
BrainDance environment diagnostic tool.

Run with:
    python -m braindance.install_check

Detects OS, Python version, GPU, CUDA toolkit, and installed packages,
then prints actionable recommendations for missing or mismatched dependencies.
"""

import importlib
import platform
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _ok(msg):
    print(f"  [OK]   {msg}")


def _warn(msg):
    print(f"  [WARN] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def _info(msg):
    print(f"  [INFO] {msg}")


def _try_import(package):
    """Return the module if importable, else None."""
    try:
        return importlib.import_module(package)
    except Exception:
        return None


def _get_nvidia_smi_output():
    """Run nvidia-smi and return stdout, or None on failure."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def _parse_driver_cuda(nvidia_smi_output):
    """Extract driver version and CUDA version from nvidia-smi output."""
    driver_version = None
    cuda_version = None
    for line in nvidia_smi_output.splitlines():
        if "Driver Version:" in line:
            for part in line.split():
                try:
                    float(part)
                    if driver_version is None:
                        driver_version = part
                    else:
                        cuda_version = part
                except ValueError:
                    continue
    return driver_version, cuda_version


# ---------------------------------------------------------------------------
# Version compatibility table
# ---------------------------------------------------------------------------

# Tested combinations: (CUDA toolkit, PyTorch, torch_tensorrt)
COMPAT_TABLE = [
    ("12.6", "2.6.x", "2.6.x"),
    ("12.4", "2.5.x", "2.5.x"),
    ("12.1", "2.4.x", "2.4.x"),
    ("11.8", "2.3.x", "2.3.x"),
]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_platform():
    _section("Platform")
    _info(f"OS:      {platform.system()} {platform.release()} ({platform.machine()})")
    _info(f"Python:  {sys.version}")

    if sys.version_info < (3, 9):
        _warn("Python 3.9+ is recommended. Some dependencies may not install.")
    elif sys.version_info >= (3, 12):
        _warn("Python 3.12+ may have limited compatibility with some scientific packages.")
    else:
        _ok(f"Python {sys.version_info.major}.{sys.version_info.minor} is supported.")

    return platform.system()


def check_gpu():
    _section("GPU / CUDA")

    smi_output = _get_nvidia_smi_output()
    if smi_output is None:
        _fail("nvidia-smi not found or failed to run.")
        _info("No NVIDIA GPU detected. GPU-accelerated features (spike detection,")
        _info("RT-Sort, TensorRT) will not be available.")
        _info("BrainDance core features (environments, phases, data loading) will still work.")
        return None, None

    driver_ver, cuda_ver = _parse_driver_cuda(smi_output)
    if driver_ver:
        _ok(f"NVIDIA driver version: {driver_ver}")
    if cuda_ver:
        _ok(f"CUDA version (driver): {cuda_ver}")
    else:
        _warn("Could not parse CUDA version from nvidia-smi output.")

    return driver_ver, cuda_ver


def check_pytorch(cuda_version):
    _section("PyTorch")

    torch = _try_import("torch")
    if torch is None:
        _fail("PyTorch is not installed.")
        _info("Spike detection and RT-Sort require PyTorch.")
        _info("")
        if cuda_version:
            _info("Install PyTorch for your CUDA version:")
            _info(f"  https://pytorch.org/get-started/locally/")
            _info("")
            _info(f"Your driver supports CUDA {cuda_version}. Example install command:")
            cuda_major_minor = cuda_version.replace(".", "")[:3]
            _info(f"  pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_major_minor}")
        else:
            _info("For CPU-only (no GPU acceleration):")
            _info("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None

    _ok(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        _ok(f"CUDA available in PyTorch (version: {torch.version.cuda})")
        _ok(f"GPU: {torch.cuda.get_device_name(0)}")

        # Check CUDA version alignment
        if cuda_version:
            torch_cuda = torch.version.cuda
            driver_major = cuda_version.split(".")[0]
            torch_major = torch_cuda.split(".")[0]
            if driver_major != torch_major:
                _warn(
                    f"CUDA major version mismatch: driver has {cuda_version}, "
                    f"PyTorch built with {torch_cuda}."
                )
                _info("This may cause issues. Consider reinstalling PyTorch for your CUDA version.")
    else:
        _warn("PyTorch is installed but CUDA is NOT available.")
        if cuda_version:
            _info("You have an NVIDIA GPU but PyTorch was installed without CUDA support.")
            cuda_major_minor = cuda_version.replace(".", "")[:3]
            _info("Reinstall with CUDA support:")
            _info(f"  pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_major_minor}")
        else:
            _info("No GPU detected — PyTorch will run in CPU mode (slower inference).")

    return torch


def check_tensorrt(os_name, torch_mod):
    _section("TensorRT")

    if os_name != "Linux":
        _info(f"TensorRT is only supported on Linux (you are on {os_name}).")
        _info("This is optional — spike detection works without it, just slower.")
        return

    if torch_mod is None:
        _info("Skipping — PyTorch is not installed (required for TensorRT).")
        return

    trt = _try_import("torch_tensorrt")
    if trt is None:
        _warn("torch_tensorrt is not installed.")
        _info("TensorRT is optional but significantly speeds up real-time spike detection.")
        _info("Install: https://pytorch.org/TensorRT/getting_started/installation.html")
        _info("Make sure the torch_tensorrt version matches your PyTorch version.")
    else:
        version = getattr(trt, "__version__", "unknown")
        _ok(f"torch_tensorrt {version}")


def check_optional_deps():
    _section("Optional Dependencies")

    deps = [
        ("maxlab", "MaxWell hardware SDK", "Required for MaxOne MEA experiments"),
        ("spikeinterface", "SpikeInterface", "Required for RT-Sort and Kilosort2 integration"),
        ("numba", "Numba", "Required for accelerated artifact removal"),
        ("h5py", "h5py", "Required for MaxWell HDF5 data loading"),
        ("zmq", "ZeroMQ", "Required for real-time data streaming"),
        ("flatbuffers", "FlatBuffers", "Required for Open Ephys data parsing"),
        ("diptest", "diptest", "Required for RT-Sort (install with braindance[rt-sort])"),
        ("sklearn", "scikit-learn", "Required for RT-Sort (install with braindance[rt-sort])"),
        ("pynvml", "pynvml", "Required for RT-Sort GPU monitoring (install with braindance[rt-sort])"),
    ]

    for module_name, display_name, description in deps:
        mod = _try_import(module_name)
        if mod is not None:
            version = getattr(mod, "__version__", "installed")
            _ok(f"{display_name} ({version})")
        else:
            _info(f"{display_name} — not installed. {description}.")


def check_braindance():
    _section("BrainDance")

    bd = _try_import("braindance")
    if bd is None:
        _fail("BrainDance package not found.")
        _info("Install with: pip install git+https://github.com/braingeneers/braindance")
    else:
        # Handle editable installs where the version may be on a nested subpackage
        version = getattr(bd, "__version__", None)
        if version is None:
            inner = _try_import("braindance.braindance")
            if inner is not None:
                version = getattr(inner, "__version__", None)
        version = version or "unknown"
        _ok(f"BrainDance {version}")


def print_compat_table():
    _section("Version Compatibility Reference")
    _info("Tested CUDA / PyTorch / TensorRT combinations:")
    _info("")
    _info(f"  {'CUDA':<10} {'PyTorch':<12} {'torch_tensorrt':<16}")
    _info(f"  {'-' * 10} {'-' * 12} {'-' * 16}")
    for cuda, pytorch, trt in COMPAT_TABLE:
        _info(f"  {cuda:<10} {pytorch:<12} {trt:<16}")
    _info("")
    _info("TensorRT is optional and Linux-only. PyTorch CUDA version should")
    _info("match or be lower than your driver's CUDA version.")


def print_recommendations(os_name, cuda_version, torch_mod):
    _section("Recommendations")

    if torch_mod is None and cuda_version:
        _info("1. Install PyTorch with CUDA support:")
        cuda_major_minor = cuda_version.replace(".", "")[:3]
        _info(f"   pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_major_minor}")
        _info("")

    if torch_mod is None and cuda_version is None:
        _info("1. Install PyTorch (CPU-only):")
        _info("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        _info("")

    if os_name == "Linux" and torch_mod is not None:
        trt = _try_import("torch_tensorrt")
        if trt is None:
            _info("Consider installing torch_tensorrt for faster real-time inference:")
            _info("   https://pytorch.org/TensorRT/getting_started/installation.html")
            _info("")

    _info("For full RT-Sort support:")
    _info("   pip install git+https://github.com/braingeneers/braindance#egg=braindance[rt-sort]")
    _info("")
    _info("For more details, see the BrainDance README or run this tool again after changes.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nBrainDance Environment Diagnostic")
    print("=" * 60)

    os_name = check_platform()
    driver_ver, cuda_ver = check_gpu()
    torch_mod = check_pytorch(cuda_ver)
    check_tensorrt(os_name, torch_mod)
    check_optional_deps()
    check_braindance()
    print_compat_table()
    print_recommendations(os_name, cuda_ver, torch_mod)

    print(f"\n{'=' * 60}")
    print("  Diagnostic complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
