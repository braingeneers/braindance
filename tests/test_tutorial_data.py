import hashlib
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from braindance.tutorial import TutorialDataError, get_test_data


def test_default_manifest_includes_full_recordings_without_intermediate_traces():
    from braindance.tutorial import _load_registry, _resolve_dataset

    manifest = _load_registry(None, None)
    version, entries = _resolve_dataset(manifest, "closed-loop-small", None)
    assert version == "3"
    paths = {entry["path"] for entry in entries}
    raw_paths = {path for path in paths if path.endswith(".raw.h5")}
    assert len(raw_paths) == 3
    assert "2026-04-10-closedloop/25245hs5/closed_loop_plasticity/002_closed_loop/002.raw.h5" in raw_paths
    assert not any("rt_sort_inter" in path.split("/") for path in paths)
    assert any(path.endswith("chip_metadata.json") for path in paths)
    assert any(path.endswith("experiment_log.json") for path in paths)
    assert any(path.endswith("RT_sort.pkl") for path in paths)
    _, small_entries = _resolve_dataset(manifest, "closed-loop-small", "1")
    assert len(small_entries) == 5
    assert all("/25245hs5/002_closed_loop/" not in path for path in paths)


@contextmanager
def serve(files):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = files.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def registry(base_url, files, *, name="closed-loop-small", version="1"):
    entries = []
    for path, body in files.items():
        entries.append(
            {
                "path": path,
                "url": f"{base_url}/{path}",
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    return {
        "datasets": {
            name: {
                "default_version": version,
                "versions": {version: {"files": entries}},
            }
        }
    }


def test_download_then_reuse_verified_cache_without_network(tmp_path):
    payloads = {"catalog.csv": b"name,path\nexample,data.npz\n", "nested/data.npz": b"spikes"}
    with serve({f"/{path}": body for path, body in payloads.items()}) as (url, requests):
        manifest = registry(url, payloads)
        folder = get_test_data(cache_dir=tmp_path, manifest=manifest)
        assert (folder / "catalog.csv").read_bytes() == payloads["catalog.csv"]
        assert (folder / "nested/data.npz").read_bytes() == payloads["nested/data.npz"]
        assert len(requests) == 2

    assert get_test_data(cache_dir=tmp_path, manifest=manifest) == folder
    assert get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True) == folder


def test_download_command_without_analysis_dependencies(tmp_path):
    import json
    import subprocess
    import sys

    payloads = {"data.bin": b"verified tutorial"}
    manifest_path = tmp_path / "manifest.json"
    with serve({f"/{path}": body for path, body in payloads.items()}) as (url, requests):
        manifest_path.write_text(json.dumps(registry(url, payloads)))
        # Exercise the real CLI with analysis imports forbidden, even when the
        # test environment has the packages installed.
        code = """
import importlib.abc, runpy, sys
class NoAnalysis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'spikelab', 'spikedata', 'numpy', 'scipy'} or fullname.startswith('braindance.utils.data_manager'):
            raise AssertionError('Download imported analysis: ' + fullname)
sys.meta_path.insert(0, NoAnalysis())
sys.argv = ['get_tutorial_data'] + sys.argv[1:]
runpy.run_module('braindance.examples.get_tutorial_data', run_name='__main__')
"""
        command = [sys.executable, "-c", code, "--cache-dir", str(tmp_path / "cache"),
                   "--manifest-path", str(manifest_path)]
        downloaded = subprocess.run(command, check=True, capture_output=True, text=True)
        assert "Tutorial data:" in downloaded.stdout
        assert "[1/1]" in downloaded.stdout
        assert "100.0%" in downloaded.stderr and "verified" in downloaded.stderr
        assert requests == ["/data.bin"]
    cached = subprocess.run(command + ["--offline"], check=True, capture_output=True, text=True)
    assert "Using verified cached tutorial data:" in cached.stdout
    assert "100.0%" not in cached.stderr


def test_online_corruption_is_redownloaded(tmp_path):
    payload = b"known-good"
    with serve({"/data.bin": payload}) as (url, requests):
        manifest = registry(url, {"data.bin": payload})
        folder = get_test_data(cache_dir=tmp_path, manifest=manifest)
        (folder / "data.bin").write_bytes(b"corrupt!!!")
        assert get_test_data(cache_dir=tmp_path, manifest=manifest) == folder
        assert (folder / "data.bin").read_bytes() == payload
        assert requests == ["/data.bin", "/data.bin"]


@pytest.mark.parametrize("corrupt_digest", [False, True])
def test_terminal_progress_updates_and_only_verifies_valid_files(tmp_path, monkeypatch, corrupt_digest):
    import io
    from braindance import tutorial

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr(tutorial.sys, "stderr", terminal)
    payload = b"x" * (128 * 1024)
    with serve({"/data.bin": payload}) as (url, _):
        entry = registry(url, {"data.bin": payload})["datasets"]["closed-loop-small"]["versions"]["1"]["files"][0]
        if corrupt_digest:
            entry["sha256"] = "0" * 64
            with pytest.raises(TutorialDataError, match="SHA-256 mismatch"):
                tutorial._download_file(entry, tmp_path / "download.bin")
        else:
            tutorial._download_file(entry, tmp_path / "download.bin")
    output = terminal.getvalue()
    assert "50.0%" in output and "\r" in output
    assert output.endswith("\n")
    assert ("100.0%" in output and "verified" in output) == (not corrupt_digest)


def test_offline_rejects_missing_and_corrupt_cache(tmp_path):
    payload = b"payload"
    manifest = registry("http://127.0.0.1:9", {"data.bin": payload})
    with pytest.raises(TutorialDataError, match="verified cache"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True)

    with serve({"/data.bin": payload}) as (url, _):
        live_manifest = registry(url, {"data.bin": payload})
        folder = get_test_data(cache_dir=tmp_path, manifest=live_manifest)
        (folder / "data.bin").write_bytes(b"damage!")
        with pytest.raises(TutorialDataError, match="verified cache"):
            get_test_data(cache_dir=tmp_path, manifest=live_manifest, offline=True)


def test_short_download_leaves_no_complete_dataset(tmp_path):
    actual = b"partial"
    with serve({"/data.bin": actual}) as (url, _):
        manifest = registry(url, {"data.bin": actual})
        entry = manifest["datasets"]["closed-loop-small"]["versions"]["1"]["files"][0]
        entry["bytes"] += 4
        with pytest.raises(TutorialDataError, match="Wrong byte count"):
            get_test_data(cache_dir=tmp_path, manifest=manifest)

    target = tmp_path / "closed-loop-small"
    assert not target.exists()
    assert not list(target.parent.glob(".closed-loop-small.staging-*"))


def test_failed_replacement_preserves_previous_complete_cache(tmp_path):
    original = b"original"
    replacement = b"short"
    with serve({"/data.bin": original}) as (url, _):
        old_manifest = registry(url, {"data.bin": original})
        folder = get_test_data(cache_dir=tmp_path, manifest=old_manifest)

    with serve({"/data.bin": replacement}) as (url, _):
        new_manifest = registry(url, {"data.bin": replacement})
        entry = new_manifest["datasets"]["closed-loop-small"]["versions"]["1"]["files"][0]
        entry["bytes"] += 1
        with pytest.raises(TutorialDataError, match="Wrong byte count"):
            get_test_data(cache_dir=tmp_path, manifest=new_manifest, update=True)

    assert (folder / "data.bin").read_bytes() == original
    assert get_test_data(cache_dir=tmp_path, manifest=old_manifest, offline=True) == folder


def test_cache_symlink_is_replaced_without_writing_through_it(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.bin").write_bytes(b"outside")
    target = tmp_path / "cache" / "closed-loop-small"
    target.parent.mkdir(parents=True)
    target.symlink_to(outside, target_is_directory=True)
    payload = b"downloaded"
    offline_manifest = registry("http://127.0.0.1:9", {"data.bin": payload})

    with pytest.raises(TutorialDataError, match="verified cache"):
        get_test_data(cache_dir=tmp_path / "cache", manifest=offline_manifest, offline=True)
    assert target.is_symlink()
    assert (outside / "data.bin").read_bytes() == b"outside"

    with serve({"/data.bin": payload}) as (url, _):
        manifest = registry(url, {"data.bin": payload})
        assert get_test_data(cache_dir=tmp_path / "cache", manifest=manifest) == target

    assert not target.is_symlink()
    assert (target / "data.bin").read_bytes() == payload
    assert (outside / "data.bin").read_bytes() == b"outside"



def test_configured_cache_root_symlink_is_allowed(tmp_path):
    concrete = tmp_path / "concrete"
    concrete.mkdir()
    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(concrete, target_is_directory=True)
    payload = b"downloaded"

    with serve({"/data.bin": payload}) as (url, _):
        manifest = registry(url, {"data.bin": payload})
        folder = get_test_data(cache_dir=linked_cache, manifest=manifest)

    assert folder == concrete / "closed-loop-small"
    assert (folder / "data.bin").read_bytes() == payload


def test_simultaneous_callers_download_once(tmp_path):
    payload = b"one download"
    with serve({"/data.bin": payload}) as (url, requests):
        manifest = registry(url, {"data.bin": payload})
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def fetch():
            try:
                barrier.wait()
                results.append(get_test_data(cache_dir=tmp_path, manifest=manifest))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=fetch) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []
    assert results == [tmp_path / "closed-loop-small"] * 2
    assert requests == ["/data.bin"]


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../../escape", "a\\escape", "C:/x", "C:x", "a\0x"]
)
def test_unsafe_manifest_paths_are_rejected(tmp_path, path):
    payload = b"x"
    manifest = registry("http://127.0.0.1:9", {path: payload})
    with pytest.raises(TutorialDataError, match="Unsafe manifest file path"):
        get_test_data(cache_dir=tmp_path, manifest=manifest)
    assert not (tmp_path.parent / "escape").exists()


def test_unknown_dataset_and_version_are_clear(tmp_path):
    manifest = {"datasets": {}}
    with pytest.raises(TutorialDataError, match="Unknown tutorial dataset"):
        get_test_data(cache_dir=tmp_path, manifest=manifest)

    manifest = registry("http://127.0.0.1:9", {"data.bin": b"x"})
    with pytest.raises(TutorialDataError, match="Unknown version"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, version="2")


@pytest.mark.parametrize("value", ["C:dataset", "bad\0name"])
def test_ambiguous_dataset_names_are_rejected(tmp_path, value):
    with pytest.raises(TutorialDataError, match="Invalid tutorial dataset name"):
        get_test_data(name=value, cache_dir=tmp_path, manifest={"datasets": {}})


@pytest.mark.parametrize("value", ["C:1", "bad\0version"])
def test_ambiguous_versions_are_rejected(tmp_path, value):
    manifest = registry("http://127.0.0.1:9", {"data.bin": b"x"})
    with pytest.raises(TutorialDataError, match="Invalid tutorial dataset version"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, version=value)


def test_manifest_path_is_supported(tmp_path):
    payload = b"from json"
    with serve({"/data.bin": payload}) as (url, _):
        path = tmp_path / "manifest.json"
        import json

        path.write_text(json.dumps(registry(url, {"data.bin": payload})), encoding="utf-8")
        folder = get_test_data(cache_dir=tmp_path / "cache", manifest_path=path)
    assert (folder / "data.bin").read_bytes() == payload


def seed_legacy_cache(root, manifest, payloads, version="1"):
    import json
    from braindance.tutorial import _resolve_dataset
    folder = root / "closed-loop-small" / version
    for path, body in payloads.items():
        destination = folder / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    _, entries = _resolve_dataset(manifest, "closed-loop-small", version)
    (folder / ".complete.json").write_text(json.dumps({"files": entries}))
    return folder


def test_migrate_versions_offline_relocates_verified_data_and_preserves_results(tmp_path):
    old = {"chip/rec/raw.h5": b"recording", "catalog.csv": b"old catalog"}
    manifest = registry("http://127.0.0.1:9", old)
    legacy = seed_legacy_cache(tmp_path, manifest, old)
    (legacy / "user-results.pkl").write_bytes(b"my analysis")
    second = seed_legacy_cache(tmp_path, manifest, old, version="1")
    second.rename(second.with_name("2"))
    seed_legacy_cache(tmp_path, manifest, old)
    latest = registry("http://127.0.0.1:9", {"chip/experiment/rec/raw.h5": b"recording"}, version="3")
    entries = latest["datasets"]["closed-loop-small"]["versions"]["3"]["files"]
    content = "new catalog"
    entries.append(dict(path="catalog.csv", content=content, bytes=len(content),
                        sha256=hashlib.sha256(content.encode()).hexdigest()))
    folder = get_test_data(cache_dir=tmp_path, manifest=latest, offline=True)
    assert folder == tmp_path / "closed-loop-small"
    assert not (folder / "1").exists() and not (folder / "2").exists()
    assert (folder / "chip/experiment/rec/raw.h5").read_bytes() == b"recording"
    assert (folder / "user-results.pkl").read_bytes() == b"my analysis"
    assert (folder / "catalog.csv").read_text() == content
    assert get_test_data(cache_dir=tmp_path, manifest=latest, offline=True) == folder


def test_update_requires_explicit_request_and_preserves_user_files(tmp_path):
    with serve({"/data.bin": b"first"}) as (url, _):
        old = registry(url, {"data.bin": b"first"})
        folder = get_test_data(cache_dir=tmp_path, manifest=old)
    (folder / "results").mkdir()
    (folder / "results/custom.pkl").write_bytes(b"user result")
    with serve({"/data.bin": b"second"}) as (url, _):
        latest = registry(url, {"data.bin": b"second"}, version="2")
        with pytest.raises(TutorialDataError, match="--update"):
            get_test_data(cache_dir=tmp_path, manifest=latest)
        assert (folder / "data.bin").read_bytes() == b"first"
        assert get_test_data(cache_dir=tmp_path, manifest=latest, update=True) == folder
    assert (folder / "data.bin").read_bytes() == b"second"
    assert (folder / "results/custom.pkl").read_bytes() == b"user result"
    assert not (folder / "2").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_modified_downloaded_results_are_never_discarded(tmp_path, legacy):
    payloads = {"experiment/results/results.json": b"original"}
    manifest = registry("http://127.0.0.1:9", payloads)
    folder = seed_legacy_cache(tmp_path, manifest, payloads)
    if not legacy:
        moved = tmp_path / "temporary"
        folder.rename(moved)
        folder.parent.rmdir()
        moved.rename(folder.parent)
        folder = folder.parent
    (folder / "experiment/results/results.json").write_bytes(b"my new results")
    with pytest.raises(TutorialDataError, match="results have been modified"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, update=True)
    assert (folder / "experiment/results/results.json").read_bytes() == b"my new results"


def test_migration_conflicts_leave_old_versions_intact(tmp_path):
    payloads = {"data.bin": b"data"}
    manifest = registry("http://127.0.0.1:9", payloads)
    first = seed_legacy_cache(tmp_path, manifest, payloads)
    (first / "custom.pkl").write_bytes(b"first result")
    import shutil
    second = first.with_name("2")
    shutil.copytree(first, second)
    (second / "custom.pkl").write_bytes(b"second result")
    with pytest.raises(TutorialDataError, match="Conflicting user files"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True)
    assert (first / "custom.pkl").read_bytes() == b"first result"
    assert (second / "custom.pkl").read_bytes() == b"second result"


def test_migration_does_not_copy_through_user_symlink(tmp_path):
    payloads = {"data.bin": b"data"}
    manifest = registry("http://127.0.0.1:9", payloads)
    legacy = seed_legacy_cache(tmp_path, manifest, payloads)
    (legacy / "results").mkdir()
    (legacy / "results/new.pkl").write_bytes(b"result")
    outside = tmp_path / "outside"
    outside.mkdir()
    (legacy.parent / "results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TutorialDataError, match="symlink"):
        get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True)
    assert not (outside / "new.pkl").exists()
    assert (legacy / "results/new.pkl").read_bytes() == b"result"


def test_inline_metadata_hash_is_checked_before_installing(tmp_path):
    manifest = registry('http://127.0.0.1:9', {'reference.json': b'{}'})
    entry = manifest['datasets']['closed-loop-small']['versions']['1']['files'][0]
    entry.pop('url')
    entry['content'] = '{}'
    folder = get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True)
    assert (folder / 'reference.json').read_text() == '{}'
    entry['content'] = '[]'
    with pytest.raises(TutorialDataError, match='Inline tutorial file failed verification'):
        get_test_data(cache_dir=tmp_path, manifest=manifest, update=True)
    assert (folder / 'reference.json').read_text() == '{}'


def test_update_removes_empty_obsolete_recording_directories(tmp_path):
    old = {'chip/recording/raw.h5': b'raw'}
    manifest = registry('http://127.0.0.1:9', old)
    legacy = seed_legacy_cache(tmp_path, manifest, old)
    target = get_test_data(cache_dir=tmp_path, manifest=manifest, offline=True)
    latest = registry('http://127.0.0.1:9', {'chip/experiment/recording/raw.h5': b'raw'}, version='2')
    get_test_data(cache_dir=tmp_path, manifest=latest, update=True, offline=True)
    assert not (target / 'chip/recording').exists()
    assert (target / 'chip/experiment/recording/raw.h5').read_bytes() == b'raw'
