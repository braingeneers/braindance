"""
Results Cache - S3-backed caching for derived data (binned FR, latencies, etc.)

This module provides efficient caching of computed results to avoid expensive
recomputation across multiple jobs. Results are stored per-recording and
synced to S3 for cross-machine access.

Key Features:
- Parameter-keyed file naming (e.g., binned_fr_20ms.npz)
- Per-experiment manifest tracking
- Automatic S3 sync for container deployments
- Local-first with optional S3 upload

Architecture:
    Recording.get_binned_fr(bin_ms=20)
           ↓
    ResultsCache.get_or_compute('binned_fr', params={'bin_ms': 20}, compute_fn)
           ↓
    Check local cache → Check S3 → Compute → Save locally → Upload to S3 (if enabled)

Usage:
    cache = ResultsCache(
        local_path=Path('results/'),
        s3_path='s3://braingeneers/braindance/proj/chip/exp/results/',
        auto_upload=os.environ.get('BRAINDANCE_AUTO_UPLOAD') == '1'
    )

    # Get or compute binned firing rates
    binned_fr = cache.get_or_compute(
        'binned_fr',
        params={'bin_ms': 20},
        compute_fn=lambda: bin_spike_data_vectorized(spikes, bin_ms=20)
    )
"""

import os
import io
import json
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

try:
    import boto3
    import smart_open
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import zstandard as _zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# NRP S3 endpoint
NRP_S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"

# ---------------------------------------------------------------------------
# SHARED S3 client — do NOT go back to one client per ResultsCache.
#
# Every ResultsCache used to build its own boto3 client lazily. Callers that walk
# a catalog construct one Recording (hence one ResultsCache, hence one client) per
# row, and a botocore client owns a urllib3 connection pool whose sockets are NOT
# closed when the client is garbage-collected. Sockets therefore accumulated at
# roughly one per cache MISS until the process hit RLIMIT_NOFILE.
#
# In a container that limit is 1024. Measured on three independent step3 pods that
# had silently stopped reading: fds=1023, sockets=1020. Past that, every socket()
# fails, boto3 raises, and download_from_s3's `except Exception: return False`
# converts it into an indistinguishable "cache miss" — so a run does not crash, it
# quietly reports the rest of the catalog as uncached. Slots survived only if they
# had fewer than ~1024 misses (53 misses -> complete; ~1075 -> truncated at 1020).
#
# One client per endpoint bounds sockets by the connection pool instead of by the
# number of rows. boto3 clients are thread-safe for API calls, which is what the
# ThreadPoolExecutor callers need.
_S3_CLIENTS: Dict[str, Any] = {}
_S3_CLIENTS_LOCK = threading.Lock()


def _shared_s3_client(endpoint_url: Optional[str]):
    """Process-wide boto3 S3 client, one per endpoint."""
    key = endpoint_url or "__default__"
    client = _S3_CLIENTS.get(key)
    if client is not None:
        return client
    with _S3_CLIENTS_LOCK:
        client = _S3_CLIENTS.get(key)
        if client is None:
            try:
                from botocore.config import Config as _BotoConfig
                # Pool must cover the widest fan-out we run (step3 uses 8 threads);
                # 32 leaves headroom without approaching the fd limit.
                cfg = _BotoConfig(max_pool_connections=32,
                                  retries={"max_attempts": 4, "mode": "adaptive"})
                client = boto3.client("s3", endpoint_url=endpoint_url, config=cfg)
            except Exception:
                client = boto3.client("s3", endpoint_url=endpoint_url)
            _S3_CLIENTS[key] = client
    return client


# ---------------------------------------------------------------------------
# Two-tier zstd compression for cache files.
#
# At scale (extracting latents/predictions for ~2000 recordings × 9 models)
# the cost is dominated by (a) moving uncompressed `.npz` blobs to/from S3 and
# (b) the local `.npz` cache filling pod ephemeral storage (binned_fr peaked
# ~89 GB → forced a 150 Gi request). zstd (level 3) compresses these sparse
# spike arrays ~50–240×.
#
#   WIRE tier  (S3 object stored as `<name><key>.npz.zst`):
#     binned_fr, val_metrics_pred*, latents_*, popfr_latents_*.
#   LOCAL tier (file ALSO stored compressed on disk; decompressed only in RAM
#     at load_local) — binned_fr ONLY. binned_fr is read exclusively through
#     get_binned_fr()→load_local in every pipeline, so on-disk compression is
#     safe and collapses the ephemeral footprint. val_metrics_pred / latents
#     are deliberately NOT local-compressed because step3 / the adapter
#     direct-load them via `_local_file_path()` + `np.load`.
#
# Backward-compatible: load/exists check `.npz.zst` then legacy `.npz`;
# download prefers the `.npz.zst` S3 object and falls back to legacy `.npz`.
# BRAINDANCE_ZSTD_MIGRATE=1 rewrites a fetched legacy `.npz` to `.npz.zst`.
# BRAINDANCE_DISABLE_ZSTD=1 forces plain `.npz` everywhere.
# ---------------------------------------------------------------------------
_WIRE_ZSTD_PREFIXES  = ('binned_fr', 'val_metrics_pred', 'latents_', 'popfr_latents_')
_LOCAL_ZSTD_PREFIXES = ('binned_fr',)
_ZSTD_LEVEL = 3
_ZST_EXT = '.npz.zst'


def _zstd_disabled() -> bool:
    return os.environ.get('BRAINDANCE_DISABLE_ZSTD', '').lower() in ('1', 'true', 'yes')


def _wire_zstd_eligible(name: str) -> bool:
    """Whether this cache name's S3 object should be zstd-compressed on WRITE."""
    if not (ZSTD_AVAILABLE and BOTO3_AVAILABLE) or _zstd_disabled():
        return False
    return any(name == p or name.startswith(p) for p in _WIRE_ZSTD_PREFIXES)


def _wire_zstd_readable(name: str) -> bool:
    """Whether a `.npz.zst` S3 object should be LOOKED FOR on read.

    🚨 Deliberately NOT gated on ZSTD_AVAILABLE, unlike `_wire_zstd_eligible`.
    Every `binned_fr` / `latents_*` / `val_metrics_pred*` object in the bucket is
    stored `.npz.zst` — the plain `.npz` sibling does not exist. Reusing the WRITE
    predicate on the read path meant that in any environment without `zstandard`
    installed, `exists_s3()` silently returned False and `download_from_s3()`
    silently returned False, so `get_or_compute` fell through to `compute_fn` and
    re-derived binned FR from raw spikes (slow) or raised "no spike data". That
    read as "the cache was never extracted" and has repeatedly sent agents off
    re-extracting data that was there the whole time. Now the object is always
    found; a missing `zstandard` raises a NAMED error instead of a silent miss.
    """
    if not BOTO3_AVAILABLE or _zstd_disabled():
        return False
    return any(name == p or name.startswith(p) for p in _WIRE_ZSTD_PREFIXES)


def _require_zstd(where: str):
    if not ZSTD_AVAILABLE:
        raise ImportError(
            f"{where}: this cache object is zstd-compressed (`.npz.zst`) but the "
            f"`zstandard` package is not installed in this environment. "
            f"`pip install zstandard`. (Do NOT conclude the cache is missing and "
            f"re-extract — the compressed object exists on S3.)"
        )


def _local_zstd_eligible(name: str) -> bool:
    """Whether this cache name's LOCAL file should be stored zstd-compressed."""
    if not ZSTD_AVAILABLE or _zstd_disabled():
        return False
    return any(name == p or name.startswith(p) for p in _LOCAL_ZSTD_PREFIXES)


def _zstd_migrate_enabled() -> bool:
    return os.environ.get('BRAINDANCE_ZSTD_MIGRATE', '').lower() in ('1', 'true', 'yes')


def _zstd_compress(blob: bytes, level: int = _ZSTD_LEVEL) -> bytes:
    # `.compress()` embeds the content size in the frame header, so the
    # one-shot `.decompress()` below needs no size hint.
    return _zstd.ZstdCompressor(level=level).compress(blob)


def _zstd_decompress(blob: bytes) -> bytes:
    return _zstd.ZstdDecompressor().decompress(blob)


def _param_key(params: Dict[str, Any]) -> str:
    """
    Generate a parameter key suffix for file naming.
    
    Examples:
        {'bin_ms': 20} → '_20ms'
        {'bin_ms': 20, 'method': 'gaussian'} → '_20ms_gaussian'
        {} → ''
    
    Simple params get simple suffixes. Complex params use hash.
    """
    if not params:
        return ''
    
    # Simple case: single param like bin_ms
    if len(params) == 1:
        key, val = list(params.items())[0]
        if key == 'bin_ms':
            return f'_{int(val)}ms'
        elif isinstance(val, (int, float)):
            return f'_{val}'
        elif isinstance(val, str):
            return f'_{val}'
    
    # Multiple params: use abbreviated key format
    parts = []
    for key, val in sorted(params.items()):
        if key == 'bin_ms':
            parts.append(f'{int(val)}ms')
        elif isinstance(val, (int, float)):
            parts.append(f'{val}')
        elif isinstance(val, str):
            parts.append(val[:10])  # Truncate long strings
    
    if parts:
        return '_' + '_'.join(parts)
    
    # Fallback: hash for complex params
    param_str = json.dumps(params, sort_keys=True)
    return '_' + hashlib.md5(param_str.encode()).hexdigest()[:8]


class ResultsManifest:
    """
    Tracks cached results metadata per-experiment.
    
    Stored as manifest.json in the results directory.
    """
    
    def __init__(self, path: Path):
        self.path = path / 'manifest.json'
        self._manifest: Dict = {'version': '1.0', 'results': {}}
        self._load()
    
    def _load(self):
        """Load manifest from disk if exists."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self._manifest = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load manifest: {e}")
    
    def save(self):
        """Save manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._manifest, f, indent=2)
    
    def add_result(
        self, 
        name: str, 
        params: Dict[str, Any],
        filename: str,
        shape: Optional[Tuple] = None,
        dtype: Optional[str] = None
    ):
        """
        Add a result entry to the manifest.
        
        Args:
            name: Result type (e.g., 'binned_fr')
            params: Parameters used (e.g., {'bin_ms': 20})
            filename: File name (e.g., 'binned_fr_20ms.npz')
            shape: Shape of the data
            dtype: Data type
        """
        entry_key = f"{name}_{_param_key(params)}"
        self._manifest['results'][entry_key] = {
            'name': name,
            'params': params,
            'filename': filename,
            'shape': list(shape) if shape else None,
            'dtype': dtype,
            'created_at': datetime.now().isoformat(),
        }
        self.save()
    
    def get_result(self, name: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Get result entry from manifest."""
        entry_key = f"{name}_{_param_key(params)}"
        return self._manifest['results'].get(entry_key)
    
    def has_result(self, name: str, params: Dict[str, Any]) -> bool:
        """Check if result exists in manifest."""
        entry_key = f"{name}_{_param_key(params)}"
        return entry_key in self._manifest['results']
    
    def list_results(self) -> List[Dict]:
        """List all cached results."""
        return list(self._manifest['results'].values())


class ResultsCache:
    """
    S3-backed cache for derived data with local-first strategy.
    
    File naming convention:
        binned_fr_20ms.npz        - Binned firing rates at 20ms
        latency_stim_20ms.npz     - Latency data for 20ms bins
        connectivity.npz          - Connectivity matrices
    
    Directory structure:
        Local:  {data_dir}/{proj}/{chip}/{exp}/results/
        S3:     s3://braingeneers/braindance/{proj}/{chip}/{exp}/results/
    
    Attributes:
        local_path: Local results directory
        s3_path: S3 results path (optional)
        auto_upload: Whether to automatically upload after computing
    """
    
    def __init__(
        self,
        local_path: Path,
        s3_path: Optional[str] = None,
        auto_upload: bool = False,
        auto_download: bool = True,
        endpoint_url: str = NRP_S3_ENDPOINT
    ):
        """
        Initialize ResultsCache.
        
        Args:
            local_path: Local directory for results
            s3_path: S3 path for results (e.g., 's3://bucket/path/results/')
            auto_upload: Automatically upload after computing (for containers)
            auto_download: Try to download from S3 if not found locally
            endpoint_url: S3 endpoint URL
        """
        self.local_path = Path(local_path)
        self.s3_path = s3_path.rstrip('/') if s3_path else None
        self.auto_upload = auto_upload
        self.auto_download = auto_download
        self.endpoint_url = endpoint_url
        
        # Initialize manifest
        self._manifest = ResultsManifest(self.local_path)
        
        # S3 client (lazy init)
        self._s3_client = None
    
    def _get_s3_client(self):
        """Get or create S3 client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required. Install: pip install boto3 smart-open[s3]")

        # Shared per-endpoint client. A per-instance client leaks a socket per
        # ResultsCache and silently truncates catalog walks at RLIMIT_NOFILE —
        # see the _shared_s3_client comment above.
        if self._s3_client is None:
            self._s3_client = _shared_s3_client(self.endpoint_url)
        return self._s3_client
    
    def _filename(self, name: str, params: Dict[str, Any], ext: str = '.npz') -> str:
        """Generate filename from name and params."""
        return f"{name}{_param_key(params)}{ext}"
    
    def _local_file_path(self, name: str, params: Dict[str, Any], ext: str = '.npz') -> Path:
        """Get local file path for result."""
        return self.local_path / self._filename(name, params, ext)
    
    def _s3_file_path(self, name: str, params: Dict[str, Any], ext: str = '.npz') -> Optional[str]:
        """Get S3 path for result."""
        if not self.s3_path:
            return None
        return f"{self.s3_path}/{self._filename(name, params, ext)}"
    
    def _local_zst_path(self, name: str, params: Dict[str, Any]) -> Path:
        """Local path for the zstd-compressed form (`<name><key>.npz.zst`)."""
        return self._local_file_path(name, params, ext=_ZST_EXT)

    def _existing_local(self, name: str, params: Dict[str, Any]) -> Optional[Path]:
        """Return the on-disk cache file (compressed preferred), or None."""
        z = self._local_zst_path(name, params)
        if z.exists():
            return z
        n = self._local_file_path(name, params)
        return n if n.exists() else None

    def exists_local(self, name: str, params: Dict[str, Any]) -> bool:
        """Check if result exists locally (compressed `.npz.zst` or `.npz`)."""
        return self._existing_local(name, params) is not None

    def evict_local(self, name: str, params: Dict[str, Any]) -> bool:
        """Delete the on-disk cache file for (name, params), if present.

        For batch jobs that stream through many recordings once each (no
        reuse), the downloaded/computed file otherwise accumulates on disk
        forever — `Recording.clear_cache()` only drops the in-memory
        `DataContext`, not this file. Returns True if a file was removed.
        """
        path = self._existing_local(name, params)
        if path is not None:
            path.unlink()
            return True
        return False

    def _head_s3(self, s3_path: str) -> bool:
        """True if the given s3:// object exists."""
        try:
            parts = s3_path.replace('s3://', '').split('/', 1)
            bucket, key = parts[0], parts[1]
            self._get_s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def _put_s3(self, s3_path: str, data: bytes):
        """Upload bytes to s3:// via boto3 put_object.

        Uses boto3 directly rather than `smart_open.open(...,'wb')` — the latter
        pulls in a `backports` module absent from the container image and fails
        every write. boto3 (already used for head/download) has no such dep.
        """
        bucket, key = s3_path.replace('s3://', '', 1).partition('/')[::2]
        self._get_s3_client().put_object(Bucket=bucket, Key=key, Body=data)

    def _get_s3(self, s3_path: str) -> bytes:
        """Download raw object bytes from s3:// via boto3 get_object.

        Mirrors `_put_s3`: avoids `smart_open.open(...,'rb')`, whose
        extension-based auto-decompression imports a `backports` module
        absent from the container image and throws on every `.npz.zst`
        read. Returning raw (still-zstd-compressed) bytes lets the caller
        decompress with our own `_zstd_decompress`.
        """
        bucket, key = s3_path.replace('s3://', '', 1).partition('/')[::2]
        resp = self._get_s3_client().get_object(Bucket=bucket, Key=key)
        return resp['Body'].read()

    def exists_s3(self, name: str, params: Dict[str, Any]) -> bool:
        """Check if result exists on S3 (compressed `.npz.zst` or legacy `.npz`)."""
        if not self.s3_path or not BOTO3_AVAILABLE:
            return False

        if _wire_zstd_readable(name):
            if self._head_s3(self._s3_file_path(name, params, ext=_ZST_EXT)):
                return True
        return self._head_s3(self._s3_file_path(name, params))

    def load_local(self, name: str, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """Load result from local disk (transparently handles `.npz.zst`)."""
        path = self._existing_local(name, params)
        if path is None:
            return None

        if path.suffix == '.zst':
            # Raise, don't swallow: the except below would turn a missing
            # `zstandard` into a printed warning + None, i.e. a phantom cache miss.
            _require_zstd(f"load_local({name})")

        try:
            src = (io.BytesIO(_zstd_decompress(path.read_bytes()))
                   if path.suffix == '.zst' else path)
            with np.load(src, allow_pickle=True) as data:
                result = {}
                for key in data.keys():
                    value = data[key]
                    # Unwrap 0-dimensional arrays that contain Python objects
                    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
                        result[key] = value.item()
                    else:
                        # Materialize array before NpzFile closes the zip.
                        # Without this, returned arrays are dangling references
                        # that segfault when accessed later.
                        result[key] = np.array(value)
                return result
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")
            return None
    
    def download_from_s3(self, name: str, params: Dict[str, Any]) -> bool:
        """
        Download result from S3 to local cache.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.s3_path or not BOTO3_AVAILABLE:
            return False

        transport_params = {'client': self._get_s3_client()}
        # READ predicate, not the write one — see `_wire_zstd_readable`.
        wire = _wire_zstd_readable(name)
        local_zst = _local_zstd_eligible(name)

        def _write_local(blob_npz: bytes):
            """Persist npz bytes locally as `.npz.zst` or `.npz` per tier."""
            dst = self._local_zst_path(name, params) if local_zst \
                else self._local_file_path(name, params)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_zstd_compress(blob_npz) if local_zst else blob_npz)
            return dst

        # Preferred path: compressed `.npz.zst` object on S3. Read raw bytes
        # via boto3 (not smart_open) — smart_open auto-decompresses on the
        # `.zst` extension via a missing `backports` module and throws.
        if wire:
            zst_s3 = self._s3_file_path(name, params, ext=_ZST_EXT)
            try:
                comp = self._get_s3(zst_s3)
            except Exception:
                comp = None  # object absent -> fall back to legacy uncompressed
            if comp is not None:
                # The `.npz.zst` object EXISTS. If we cannot decompress it, that is a
                # broken environment, not a cache miss — say so loudly rather than
                # falling through to a legacy `.npz` that almost certainly isn't there.
                _require_zstd(f"download_from_s3({name})")
                # comp is zstd(npz). Keep compressed locally if local-tier; else expand.
                if local_zst:
                    dst = self._local_zst_path(name, params)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(comp)
                else:
                    _write_local(_zstd_decompress(comp))
                print(f"  [S3 ↓zst] Downloaded {self._filename(name, params)}")
                return True

        # Legacy path: plain `.npz` on S3 (boto3 raw read).
        s3_path = self._s3_file_path(name, params)
        try:
            raw = self._get_s3(s3_path)
            _write_local(raw)
            print(f"  [S3 ↓] Downloaded {self._filename(name, params)}")
            # Opportunistically migrate the legacy object to compressed form.
            if wire and _zstd_migrate_enabled():
                self.upload_to_s3(name, params)
            return True
        except Exception:
            # Silently fail - file may not exist on S3
            return False

    def upload_to_s3(self, name: str, params: Dict[str, Any]) -> bool:
        """
        Upload result from local cache to S3.

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_path or not BOTO3_AVAILABLE:
            return False

        local = self._existing_local(name, params)
        if local is None:
            print(f"⚠️  Cannot upload - local file not found: "
                  f"{self._local_file_path(name, params)}")
            return False

        try:
            if _wire_zstd_eligible(name):
                # S3 object is `.npz.zst`. Local file may already be compressed.
                blob = local.read_bytes() if local.suffix == '.zst' \
                    else _zstd_compress(local.read_bytes())
                self._put_s3(self._s3_file_path(name, params, ext=_ZST_EXT), blob)
                print(f"  [S3 ↑zst] Uploaded {self._filename(name, params)} "
                      f"({len(blob)/1024:.1f} KB)")
            else:
                raw = local.read_bytes()
                self._put_s3(self._s3_file_path(name, params), raw)
                print(f"  [S3 ↑] Uploaded {local.name} ({len(raw)/1024:.1f} KB)")
            return True
        except Exception as e:
            print(f"⚠️  Failed to upload to S3: {e}")
            return False
    
    def save_local(
        self, 
        name: str, 
        params: Dict[str, Any], 
        data: Dict[str, np.ndarray]
    ):
        """
        Save result to local disk.
        
        Args:
            name: Result type (e.g., 'binned_fr')
            params: Parameters used
            data: Dict of arrays to save (e.g., {'rates': ..., 'time_axis_ms': ...})
        """
        local_path = self._local_file_path(name, params)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if _local_zstd_eligible(name):
            buf = io.BytesIO()
            np.savez(buf, **data)
            out_path = self._local_zst_path(name, params)
            out_path.write_bytes(_zstd_compress(buf.getvalue()))
        else:
            np.savez(local_path, **data)
            out_path = local_path

        # Update manifest
        first_array = list(data.values())[0] if data else None
        self._manifest.add_result(
            name=name,
            params=params,
            filename=out_path.name,
            shape=first_array.shape if first_array is not None else None,
            dtype=str(first_array.dtype) if first_array is not None else None
        )
    
    def get_or_compute(
        self,
        name: str,
        params: Dict[str, Any],
        compute_fn: Callable[[], Dict[str, np.ndarray]],
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get cached result or compute and cache.
        
        This is the main entry point for caching. The workflow is:
        1. Check local cache
        2. If not found and auto_download, try S3
        3. If not found, compute using compute_fn
        4. Save to local cache
        5. If auto_upload, sync to S3
        
        Args:
            name: Result type (e.g., 'binned_fr')
            params: Parameters (e.g., {'bin_ms': 20})
            compute_fn: Function that computes the result
            force_recompute: Bypass cache and recompute
        
        Returns:
            Dict of arrays (e.g., {'rates': ..., 'time_axis_ms': ...})
        """
        param_str = _param_key(params)
        
        # Check local cache first
        if not force_recompute and self.exists_local(name, params):
            result = self.load_local(name, params)
            if result is not None:
                return result
        
        # Try S3 if auto_download enabled
        if not force_recompute and self.auto_download and self.s3_path:
            if self.download_from_s3(name, params):
                result = self.load_local(name, params)
                if result is not None:
                    return result
        
        # Compute
        result = compute_fn()
        
        # Save locally
        self.save_local(name, params, result)
        
        # Upload to S3 if enabled
        if self.auto_upload:
            self.upload_to_s3(name, params)
        
        return result
    
    def sync_manifest_to_s3(self) -> bool:
        """Upload manifest.json to S3."""
        if not self.s3_path or not BOTO3_AVAILABLE:
            return False
        
        manifest_s3_path = f"{self.s3_path}/manifest.json"

        try:
            self._put_s3(manifest_s3_path, Path(self._manifest.path).read_bytes())
            return True
        except Exception as e:
            print(f"⚠️  Failed to sync manifest to S3: {e}")
            return False
    
    def sync_all_to_s3(self) -> Tuple[int, int]:
        """
        Upload all cached results to S3.
        
        Returns:
            Tuple of (success_count, fail_count)
        """
        if not self.s3_path:
            print("⚠️  No S3 path configured")
            return (0, 0)
        
        success, fail = 0, 0
        
        for result_info in self._manifest.list_results():
            name = result_info['name']
            params = result_info['params']
            
            if self.upload_to_s3(name, params):
                success += 1
            else:
                fail += 1
        
        # Sync manifest
        self.sync_manifest_to_s3()
        
        print(f"\n  [S3 SYNC] Complete: {success} uploaded, {fail} failed")
        return (success, fail)


def get_auto_upload_enabled() -> bool:
    """
    Check if auto-upload is enabled via environment variable.
    
    Set BRAINDANCE_AUTO_UPLOAD=1 to enable (for container deployments).
    """
    return os.environ.get('BRAINDANCE_AUTO_UPLOAD', '').lower() in ('1', 'true', 'yes')
