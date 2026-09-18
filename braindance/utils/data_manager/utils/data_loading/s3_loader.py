"""
S3 file loader with caching support for BrainDance.

Handles custom endpoints (NRP) and AWS S3.
Extracted from BusyBeeLoader for reusability.
"""

import boto3
import pickle
from pathlib import Path
from typing import Optional, Any
from braindance.spike_data import load_spike_pickle

try:
    import smart_open
except ImportError:
    smart_open = None

# NRP S3 endpoint
NRP_S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"


class S3Loader:
    """
    S3 file loader with caching support.
    Handles custom endpoints (NRP) and AWS S3.
    """

    def __init__(
        self,
        endpoint_url: str = NRP_S3_ENDPOINT,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize S3Loader.

        Args:
            endpoint_url: S3 endpoint URL (default: NRP endpoint)
            cache_dir: Optional local cache directory
        """
        self.endpoint_url = endpoint_url
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Configure S3 client
        self._s3_client = boto3.client('s3', endpoint_url=endpoint_url)

    def download_file(self, s3_path: str, local_path: Path) -> bool:
        """
        Download a file from S3 to local storage.

        Args:
            s3_path: S3 path (e.g., 's3://bucket/key.pkl')
            local_path: Local path to save to

        Returns:
            True if successful, False otherwise
        """
        if smart_open is None:
            print(f"  [S3] ✗ smart_open not available")
            return False

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            transport_params = {'client': self._s3_client}
            print(f"  [S3] Downloading: {s3_path}")
            with smart_open.open(s3_path, 'rb', transport_params=transport_params) as s3_file:
                with open(local_path, 'wb') as local_file:
                    local_file.write(s3_file.read())

            size_kb = local_path.stat().st_size / 1024
            print(f"  [S3] ✓ Downloaded to: {local_path} ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            print(f"  [S3] ✗ Failed to download {s3_path}: {e}")
            return False

    def upload_file(self, local_path: Path, s3_path: str) -> bool:
        """
        Upload a file from local storage to S3.

        Args:
            local_path: Local path to upload from
            s3_path: S3 path (e.g., 's3://bucket/key.pkl')

        Returns:
            True if successful, False otherwise
        """
        if smart_open is None:
            print(f"  [S3] ✗ smart_open not available")
            return False

        if not local_path.exists():
            print(f"  [S3] ✗ Local file not found: {local_path}")
            return False

        try:
            transport_params = {'client': self._s3_client}
            print(f"  [S3] Uploading: {s3_path}")
            with open(local_path, 'rb') as local_file:
                with smart_open.open(s3_path, 'wb', transport_params=transport_params) as s3_file:
                    s3_file.write(local_file.read())

            size_kb = local_path.stat().st_size / 1024
            print(f"  [S3] ✓ Uploaded {local_path.name} ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            print(f"  [S3] ✗ Failed to upload {s3_path}: {e}")
            return False

    def load_pickle(self, s3_path: str, use_cache: bool = True) -> Optional[Any]:
        """
        Load pickle from S3 with optional local caching.

        Args:
            s3_path: S3 path (e.g., 's3://bucket/key.pkl')
            use_cache: Whether to use local cache

        Returns:
            Loaded data or None if failed
        """
        if smart_open is None:
            raise ImportError("smart_open required. Install: pip install smart-open[s3]")

        # Check cache first
        if use_cache and self.cache_dir:
            cache_path = self._get_cache_path(s3_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        return load_spike_pickle(f)
                except Exception as e:
                    print(f"⚠️  Cache corrupted for {cache_path.name}: {e}")
                    # Continue to load from S3

        # Load from S3
        try:
            transport_params = {'client': self._s3_client}
            with smart_open.open(s3_path, 'rb', transport_params=transport_params) as f:
                data = load_spike_pickle(f)

            # Cache locally
            if use_cache and self.cache_dir:
                self._cache_file(s3_path, data)

            return data
        except Exception as e:
            print(f"⚠️  Failed to load from S3: {s3_path} - {e}")
            return None

    def load_csv(self, s3_path: str, use_cache: bool = True):
        """
        Load CSV from S3.

        Args:
            s3_path: S3 path (e.g., 's3://bucket/key.csv')
            use_cache: Whether to use local cache

        Returns:
            pandas DataFrame or None if failed
        """
        if smart_open is None:
            raise ImportError("smart_open required")

        try:
            import pandas as pd
            transport_params = {'client': self._s3_client}
            with smart_open.open(s3_path, 'r', transport_params=transport_params) as f:
                return pd.read_csv(f)
        except Exception as e:
            print(f"⚠️  Failed to load CSV: {s3_path} - {e}")
            return None

    def _cache_file(self, s3_path: str, data: Any):
        """
        Cache data to local disk.

        Args:
            s3_path: Original S3 path
            data: Data to cache
        """
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(s3_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            pass

    def _get_cache_path(self, s3_path: str) -> Path:
        """
        Generate cache path from S3 URL.

        Args:
            s3_path: S3 path

        Returns:
            Local cache path
        """
        filename = Path(s3_path).name
        return self.cache_dir / filename
