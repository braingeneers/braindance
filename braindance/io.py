"""Standard smart_open with explicit NRP routing for BrainDance S3 paths."""

import os

import smart_open


def open_file(uri, mode="r", *, transport_params=None, **kwargs):
    """Open local/HTTP/S3 data without importing the Braingeneers SDK.

    S3 defaults to NRP, as the former Braingeneers wrapper did. ``ENDPOINT``
    overrides that default; an explicit transport client takes precedence.
    Local and HTTP opens do not construct an S3 client or look up credentials.
    """
    params = dict(transport_params or {})
    if isinstance(uri, (str, os.PathLike)) and os.fspath(uri).startswith("s3://"):
        endpoint = os.environ.get("ENDPOINT", "https://s3-west.nrp-nautilus.io")
        if "client" not in params and endpoint.startswith(("https://", "http://")):
            import boto3

            params["client"] = boto3.client("s3", endpoint_url=endpoint)
    return smart_open.open(uri, mode, transport_params=params, **kwargs)
