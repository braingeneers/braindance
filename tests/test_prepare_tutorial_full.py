import copy
import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def preparation(monkeypatch):
    import braindance.tutorial as tutorial

    path = Path(__file__).resolve().parents[1] / "scripts/prepare_tutorial_full.py"
    spec = importlib.util.spec_from_file_location("prepare_tutorial_full", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = "2026-04-10-closedloop/25245hs5"
    prefix = f"braindance/{project}/"
    mapping = f"{project}/closed_loop_plasticity/mapping.csv"
    exports = {"catalog.csv": b"proj,chip,experiment\n2026-04-10-closedloop,25245hs5,closed_loop_plasticity/002_closed_loop\n", mapping: b"export mapping"}
    entries = [{"path": name, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "url": f"https://example.org/v1/{name}"}
               for name, data in exports.items()]
    registry = {"datasets": {"closed-loop-small": {
        "default_version": "1", "versions": {"1": {"files": entries}}}}}
    monkeypatch.setattr(tutorial, "_load_registry", lambda *_: registry)
    downloads = []

    def download(entry, destination):
        downloads.append(entry["path"])
        destination.write_bytes(exports[entry["path"]])

    monkeypatch.setattr(tutorial, "_download_file", download)
    return module, tutorial, registry, project, prefix, exports, downloads


def test_full_bundle_layout_hashes_and_reuse(tmp_path, monkeypatch, preparation):
    module, tutorial, registry, project, prefix, exports, export_downloads = preparation
    original_registry = copy.deepcopy(registry)
    raw_key = prefix + "closed_loop_plasticity/002_closed_loop/002.raw.h5"
    raw_path = f"{project}/closed_loop_plasticity/002_closed_loop/002.raw.h5"
    other_raw_key = prefix + "closed_loop_plasticity/001_baseline/001.raw.h5"
    mapping_key = prefix + "closed_loop_plasticity/mapping.csv"
    support_key = prefix + "closed_loop_plasticity/rt_sort_inter_metadata.json"
    payloads = {raw_key: b"middle raw", other_raw_key: b"baseline raw",
                mapping_key: b"original mapping", support_key: b"{}"}
    objects = [{"Key": key, "Size": len(data)} for key, data in payloads.items()]
    objects.extend([
        {"Key": prefix + "rt_sort_inter/traces.npy", "Size": 123},
        {"Key": prefix + "closed_loop_plasticity/rt_sort_inter/other.npy", "Size": 456},
        {"Key": prefix + "closed_loop_plasticity/", "Size": 0},
    ])
    commands = []

    def listing(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"Contents": objects[::-1]}))

    monkeypatch.setattr(module.subprocess, "run", listing)
    urls = []

    def download_source(url, timeout):
        urls.append(url)
        key = url.split("/braingeneers/", 1)[1]
        response = io.BytesIO(payloads[key])
        response.geturl = lambda: url
        return response

    monkeypatch.setattr(module, "urlopen", download_source)
    folder = tmp_path / "closed-loop-small"
    preseed = folder / raw_path
    preseed.parent.mkdir(parents=True)
    preseed.write_bytes(payloads[raw_key])
    manifest = module.main(tmp_path)
    generated = json.loads(manifest.read_text())
    _, entries = tutorial._resolve_dataset(generated, "closed-loop-small", "3")
    by_path = {entry["path"]: entry for entry in entries}
    expected_paths = set(exports) | {
        raw_path, other_raw_key.removeprefix("braindance/"),
        f"{project}/closed_loop_plasticity/original_mapping.csv",
        support_key.removeprefix("braindance/"),
    }
    assert set(by_path) == expected_paths
    assert [entry["path"] for entry in entries] == sorted(expected_paths)
    assert len(urls) == 3
    assert not any(url.endswith(raw_key) for url in urls)
    assert set(export_downloads) == set(exports)
    assert registry == original_registry
    assert generated["datasets"]["closed-loop-small"]["versions"]["1"] == registry["datasets"]["closed-loop-small"]["versions"]["1"]
    assert generated["datasets"]["closed-loop-small"]["default_version"] == "3"
    assert commands[0][:4] == ["aws", "--endpoint", "https://s3-west.nrp-nautilus.io", "s3api"]
    for entry in entries:
        assert entry["sha256"] == hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest()
    assert tutorial._verify_dataset(folder, entries)
    assert json.loads((folder / ".complete.json").read_text()) == {"files": entries}

    module.main(tmp_path)
    assert len(urls) == 3
    assert len(export_downloads) == len(exports) + 1  # Derived catalog is regenerated from the immutable source
    assert json.loads(manifest.read_text()) == generated


@pytest.mark.parametrize("payload", [b"short", b"too many bytes"])
def test_wrong_source_size_leaves_no_complete_cache(tmp_path, monkeypatch, preparation, payload):
    module, _, _, _, prefix, _, _ = preparation
    key = prefix + "support.json"
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 0, json.dumps({"Contents": [{"Key": key, "Size": 8}]})))

    def response(url, timeout):
        stream = io.BytesIO(payload)
        stream.geturl = lambda: url
        return stream

    monkeypatch.setattr(module, "urlopen", response)
    folder = tmp_path / "closed-loop-small"
    folder.mkdir(parents=True)
    (folder / ".complete.json").write_text("old marker")
    with pytest.raises(ValueError, match="listed size"):
        module.main(tmp_path)
    assert not (folder / ".complete.json").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not (folder / key.removeprefix("braindance/")).exists()
    assert not list(folder.rglob("*.part"))
