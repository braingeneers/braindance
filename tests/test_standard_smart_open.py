"""Preserve NRP routing while using the standard smart_open package."""

from io import BytesIO

import boto3
import pytest

from braindance.io import open_file


def test_local_mapping_roundtrip_does_not_create_s3_client(tmp_path, monkeypatch):
    import pandas as pd
    from braindance.analysis.mapping import Mapping

    def unexpected_client(*args, **kwargs):
        pytest.fail("Local file access must not request S3 credentials")

    monkeypatch.setattr(boto3, "client", unexpected_client)
    path = tmp_path / "mapping.csv"
    with open_file(path, "w") as stream:
        pd.DataFrame({"channel": [1, 2], "electrode": [10, 20]}).to_csv(stream, index=False)
    mapping = Mapping.from_csv(path)
    assert mapping.channels == [1, 2]
    assert mapping.electrodes == [10, 20]


@pytest.mark.parametrize("endpoint", [None, "https://other.example"])
def test_s3_reads_use_selected_endpoint(monkeypatch, endpoint):
    # Exercise real smart_open S3 streaming, with only the remote boto client mocked.
    from unittest.mock import Mock

    client = Mock()
    client.get_object.return_value = {
        "Body": BytesIO(b"NRP data"), "ContentLength": 8,
        "ContentRange": "bytes 0-7/8",
        "ResponseMetadata": {"HTTPStatusCode": 206, "RetryAttempts": 0},
    }
    factory = Mock(return_value=client)
    monkeypatch.setattr(boto3, "client", factory)
    monkeypatch.delenv("ENDPOINT", raising=False)
    if endpoint:
        monkeypatch.setenv("ENDPOINT", endpoint)
    with open_file("s3://braingeneersdev/test.bin", "rb") as stream:
        assert stream.read() == b"NRP data"
    factory.assert_called_once_with("s3", endpoint_url=endpoint or "https://s3-west.nrp-nautilus.io")
    assert client.get_object.call_args.kwargs["Bucket"] == "braingeneersdev"
    assert client.get_object.call_args.kwargs["Key"] == "test.bin"


def test_explicit_client_is_preserved(monkeypatch):
    from braindance import io
    from unittest.mock import Mock

    client = object()
    params = {"client": client, "buffer_size": 1024}
    factory = Mock(side_effect=AssertionError("Unexpected client construction"))
    monkeypatch.setattr(boto3, "client", factory)
    opened = Mock(return_value="stream")
    monkeypatch.setattr(io.smart_open, "open", opened)
    assert open_file("s3://custom/key", "rb", transport_params=params) == "stream"
    opened.assert_called_once_with("s3://custom/key", "rb", transport_params=params)
    assert params == {"client": client, "buffer_size": 1024}


def test_workshop_cli_forwards_arguments(monkeypatch):
    from braindance.examples.streaming_workshop import main as workshop
    from unittest.mock import Mock

    run = Mock(return_value=0)
    monkeypatch.setattr(workshop, "main", run)
    assert workshop.cli(["--no-browser", "--port", "8766", "--environment", "ant"]) == 0
    assert run.call_args.kwargs["port"] == 8766
    assert run.call_args.kwargs["open_browser"] is False
    assert run.call_args.kwargs["environment"] == "ant"
