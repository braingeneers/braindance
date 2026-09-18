import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_prepare_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_tutorial_closed_loop.py"
    spec = importlib.util.spec_from_file_location("prepare_tutorial_closed_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_sources(tmp_path, times=None):
    times = np.array([1.0, 101.0, 4.0], dtype=float) if times is None else np.asarray(times)
    spikes = tmp_path / "spikes.npz"
    np.savez(spikes, during_times_ms=times, during_unit_ids=np.array([0, 0, 1]))
    stim = tmp_path / "stim.csv"
    pd.DataFrame({
        "time": [0.1, 0.2], "amplitude": [400, 400], "note": ["x" * 600, "y" * 600]
    }).to_csv(stim, index=False)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({
        "channel": [3, 4], "electrode": [30, 40],
        "x": [1.0, 2.0], "y": [3.0, 4.0],
    }).to_csv(mapping, index=False)
    return spikes, stim, mapping


def test_prepared_bundle_values_and_provenance(tmp_path, monkeypatch):
    from braindance.analysis.mapping import Mapping
    from braindance.utils.data_manager import load_catalog
    from spikelab import SpikeData

    module = _load_prepare_module()
    sources = _synthetic_sources(tmp_path)
    bundle = module.prepare_bundle(
        *sources, tmp_path / "bundle", project="p", chip="c",
        experiment="phase/r", duration_ms=200, neuron_count=2, bin_ms=100,
    )
    moved = tmp_path / "moved"
    bundle.rename(moved)
    bundle = moved
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rec = load_catalog(bundle / "catalog.csv", base_path=bundle)[0]
    assert rec._spikes_path == bundle / "p/c/phase/r/r_spike_data.pkl"
    assert rec._stim_log_path == bundle / "p/c/phase/r/r_log.csv"
    assert not (bundle / "p/c/r").exists()
    reference = json.loads((bundle / "reference.json").read_text())

    assert isinstance(rec.spikes, SpikeData)
    assert [train.tolist() for train in rec.spikes.train] == [[1.0, 101.0], [4.0]]
    assert rec.spikes.length == 200
    assert rec.stim_log["time"].tolist() == [0.1, 0.2]
    assert isinstance(rec.mapping, Mapping)
    assert rec.mapping.mapping["electrode"].tolist() == [30, 40]
    assert reference["recordings"][0]["per_neuron_spike_counts"] == [2, 1]
    assert reference["recordings"][0]["binned_shape"] == [2, 2]
    assert reference["recordings"][0]["mapping_samples"] == {}
    assert set(reference["provenance"]["source_files"]) == {
        "spikes.npz", "stim.csv", "mapping.csv"
    }


@pytest.mark.parametrize(
    "times, match",
    [([1.0, 0.5, 4.0], "sorted"), ([1.0, float("nan"), 4.0], "finite")],
)
def test_invalid_spike_times_fail_before_writing(tmp_path, times, match):
    module = _load_prepare_module()
    sources = _synthetic_sources(tmp_path, times=times)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match=match):
        module.prepare_bundle(
            *sources, output, project="p", chip="c", experiment="phase/r",
            duration_ms=200, neuron_count=2,
        )
    assert not output.exists()


def test_real_closed_loop_exports_when_available(tmp_path):
    module = _load_prepare_module()
    repo = Path(__file__).resolve().parents[1]
    source_root = repo / "proj/braindance_figs/closed_loop_fig"
    sources = (
        source_root / "narrative_2026_09_05/source_data/offline_phase_spikes.npz",
        source_root / "timing_reanalysis_2026_09_05/source_data/original_stim_log.csv",
        source_root / "timing_reanalysis_2026_09_05/source_data/raw_electrode_mapping.csv",
    )
    if not all(path.exists() for path in sources):
        pytest.skip("Frozen closed-loop exports are not present")
    bundle = module.prepare_bundle(*sources, tmp_path / "bundle")
    reference = json.loads((bundle / "reference.json").read_text())["recordings"][0]
    assert reference["spike_count"] == 30_405
    assert reference["stim_row_count"] == 137
    assert reference["mapping_rows"] == 1018


@pytest.mark.parametrize('wrong_reference', [False, True])
def test_downloader_to_recording_validator(tmp_path, wrong_reference):
    import hashlib
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from braindance.examples.validate_tutorial_data import main

    module = _load_prepare_module()
    bundle = module.prepare_bundle(
        *_synthetic_sources(tmp_path), tmp_path / 'bundle', project='p', chip='c',
        experiment='phase/r', duration_ms=200, neuron_count=2, bin_ms=100,
    )
    if wrong_reference:
        reference_path = bundle / 'reference.json'
        reference = json.loads(reference_path.read_text())
        reference['recordings'][0]['spike_count'] = 4
        reference_path.write_text(json.dumps(reference))

    class Handler(SimpleHTTPRequestHandler):
        requests = 0

        def do_GET(self):
            type(self).requests += 1
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(bundle)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    entries = []
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            relative = path.relative_to(bundle).as_posix()
            entries.append({'path': relative, 'url': f'http://127.0.0.1:{server.server_port}/{relative}',
                            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / 'registry.json'
    manifest.write_text(json.dumps({'datasets': {'fixture': {'default_version': '1',
                        'versions': {'1': {'files': entries}}}}}))
    try:
        if wrong_reference:
            with pytest.raises(AssertionError, match='spike_count differs'):
                main(name='fixture', cache_dir=tmp_path / 'cache', manifest_path=manifest)
        else:
            report = main(name='fixture', cache_dir=tmp_path / 'cache', manifest_path=manifest)
            assert report['recordings'][0]['spikes'] == 3
            assert report['recordings'][0]['bin_shape'] == [2, 2]
            downloads = Handler.requests
            assert downloads == len(entries)
            offline = main(name='fixture', cache_dir=tmp_path / 'cache', manifest_path=manifest, offline=True)
            assert offline == report
            assert Handler.requests == downloads
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
